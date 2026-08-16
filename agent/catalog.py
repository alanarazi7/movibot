"""Catalog access: the structured movie table and the long plot texts.

Read from the prepared CSVs in data_preprocessing/data_ready/, loaded once and
cached in-process. There is no database, for the same reason there is no vector
database: the catalog is 238 rows and 181 KB, and it loads in about 2 ms. A
bulk read plus Python-side filtering is simpler and faster than round-tripping
a query per request, and it needs no credentials -- so the whole agent is
runnable from a fresh clone.

A Supabase backend existed here behind MOVIBOT_BACKEND=cloud. It was removed
rather than finished: it never ran, it duplicated a table small enough to hold
in memory, and keeping an unexercised second path is how the two quietly stop
agreeing. rag/store.py records the same argument for vectors.
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


def _parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in _LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.strip() else []
            )
    return df


@lru_cache(maxsize=1)
def movies() -> pd.DataFrame:
    """The structured catalog: one row per feature film, list columns parsed.

    Shorts under 45 minutes were dropped at preparation time and
    `weighted_rating` was precomputed there, so both ranking guardrails hold
    here without this module re-deriving them.
    """
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


def label_of(movie_id: int) -> str | None:
    """A film's public name: "Title (Year)".

    This is how films are addressed everywhere the model can see. Titles alone
    are not unique -- the catalog holds four remake pairs, including two
    Beauty and the Beasts -- but title with year is unique across all 238, so
    it can replace the numeric id in every tool signature. Ids are plumbing;
    nothing is served by spending prompt tokens on them.
    """
    df = movies()
    row = df[df["id"] == int(movie_id)]
    if row.empty:
        return None
    r = row.iloc[0]
    return f"{r['title']} ({int(r['release_year'])})"


@lru_cache(maxsize=1)
def _labels() -> tuple[dict[str, int], dict[str, list[int]]]:
    """(exact "Title (Year)" -> id, lowercase bare title -> [ids])."""
    df = movies()
    exact, bare = {}, {}
    for r in df.itertuples():
        exact[f"{r.title} ({int(r.release_year)})".lower()] = int(r.id)
        bare.setdefault(str(r.title).lower(), []).append(int(r.id))
    return exact, bare


def resolve(label: str) -> int | None:
    """Turn a label back into an id, or None if it names nothing.

    Accepts a bare title too, but only when it is unambiguous -- "Frozen"
    resolves, "The Jungle Book" does not, because the catalog holds both the
    1967 and 2016 films and guessing between them would be worse than failing.
    """
    if not label:
        return None
    key = str(label).strip().lower()
    exact, bare = _labels()
    if key in exact:
        return exact[key]
    ids = bare.get(key)
    return ids[0] if ids and len(ids) == 1 else None


def stats() -> dict[str, Any]:
    """Small summary used by tests and the /api/agent_info payload."""
    df = movies()
    return {
        "source": "committed CSVs",
        "movies": len(df),
        "with_synopsis": len(_synopsis_index()),
        "year_range": [int(df["release_year"].min()), int(df["release_year"].max())],
        "min_runtime": int(df["runtime_minutes"].min()),
    }
