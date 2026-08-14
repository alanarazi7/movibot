"""Catalog access: the structured movie table and the long plot texts.

Two backends, chosen by the MOVIBOT_BACKEND env var:

    local  (default)  read the prepared CSVs in data_preprocessing/data_ready/
    cloud             read the `movies` table from Supabase

Both return the same shape, so nothing above this module knows which is in
use. `local` costs nothing and needs no credentials, which is why it is the
default: the whole agent is runnable and testable before any account exists.

Everything is loaded once and cached in-process. The catalog is 238 rows, so
a single bulk read plus Python-side filtering is simpler and faster than
round-tripping a query per request.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_READY = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready")

CATALOG_CSV = os.path.join(_DATA_READY, "supabase_movies.csv")
SYNOPSES_CSV = os.path.join(_DATA_READY, "pinecone_candidates.csv")
WIKI_CACHE_CSV = os.path.join(_DATA_READY, "wikipedia_cache.csv")

# JSON-encoded list columns in the CSV that callers want as real lists.
_LIST_COLUMNS = (
    "genres",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "keywords",
)


def backend() -> str:
    """Which catalog backend is active: 'local' or 'cloud'."""
    return os.environ.get("MOVIBOT_BACKEND", "local").strip().lower()


def _parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in _LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.strip() else []
            )
    return df


def _load_from_supabase() -> pd.DataFrame:
    """Fetch the whole `movies` table. Requires SUPABASE_URL / SUPABASE_KEY."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "MOVIBOT_BACKEND=cloud requires SUPABASE_URL and SUPABASE_KEY. "
            "Unset MOVIBOT_BACKEND to use the local CSV backend instead."
        )

    client = create_client(url, key)
    # 238 rows fits comfortably in one page, but paginate anyway so this does
    # not silently truncate if the catalog is ever widened past the default
    # PostgREST limit.
    rows: list[dict[str, Any]] = []
    page_size = 1000
    start = 0
    while True:
        chunk = (
            client.from_("movies")
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        rows.extend(chunk)
        if len(chunk) < page_size:
            break
        start += page_size

    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def movies() -> pd.DataFrame:
    """The structured catalog: one row per feature film, list columns parsed.

    Shorts under 45 minutes were dropped at preparation time and
    `weighted_rating` was precomputed there, so both ranking guardrails hold
    here without this module re-deriving them.
    """
    if backend() == "cloud":
        df = _load_from_supabase()
        # Supabase returns jsonb as real lists already; only parse if the
        # driver handed back strings.
        for col in _LIST_COLUMNS:
            if col in df.columns and df[col].apply(lambda v: isinstance(v, str)).any():
                df = _parse_list_columns(df)
                break
    else:
        df = _parse_list_columns(pd.read_csv(CATALOG_CSV))

    if "weighted_rating" not in df.columns:
        raise RuntimeError(
            "Catalog is missing the `weighted_rating` column. Re-run "
            "data_preprocessing/prepare_movibot_data.py to regenerate it."
        )

    return df


@lru_cache(maxsize=1)
def _synopsis_index() -> dict[int, str]:
    """movie_id -> long plot text, best source first.

    MPST synopsis where we have one (159 of 238), else the cached Wikipedia
    Plot section. Wikipedia is read from the offline cache scraped at
    preparation time, never fetched live: live fetches are slow, rate-limited,
    and would make an agent turn's latency depend on a third party.
    """
    index: dict[int, str] = {}

    wiki = pd.read_csv(WIKI_CACHE_CSV)
    for _, row in wiki.iterrows():
        text = row.get("plot_text")
        if isinstance(text, str) and text.strip():
            index[int(row["id"])] = text.strip()

    # MPST synopses are richer, so they overwrite the Wikipedia fallback.
    syn = pd.read_csv(SYNOPSES_CSV)
    for _, row in syn.iterrows():
        text = row.get("plot_synopsis")
        if isinstance(text, str) and text.strip():
            index[int(row["movie_id"])] = text.strip()

    return index


def synopsis(movie_id: int) -> str | None:
    """Long plot text for one movie, or None if we have neither source."""
    return _synopsis_index().get(int(movie_id))


def has_synopsis(movie_id: int) -> bool:
    return int(movie_id) in _synopsis_index()


def title_of(movie_id: int) -> str | None:
    df = movies()
    row = df[df["id"] == int(movie_id)]
    return None if row.empty else str(row.iloc[0]["title"])


def stats() -> dict[str, Any]:
    """Small summary used by tests and the /api/agent_info payload."""
    df = movies()
    return {
        "backend": backend(),
        "movies": len(df),
        "with_synopsis": len(_synopsis_index()),
        "year_range": [int(df["release_year"].min()), int(df["release_year"].max())],
        "min_runtime": int(df["runtime_minutes"].min()),
    }
