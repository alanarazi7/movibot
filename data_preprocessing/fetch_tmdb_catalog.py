"""Builds the whole Disney/Pixar movie catalog from the TMDB API.

WHY THIS EXISTS
    The catalog originally came from Kaggle's "The Movies Dataset", a
    MovieLens dump ending in July 2017. It could not answer the project's own
    flagship demo query for a "not-too-old Disney movie", and regenerating it
    needed Kaggle credentials. TMDB turned out to serve the entire history --
    1,626 Disney/Pixar entries from the 1930s to today -- so it replaces the
    dump outright rather than merely topping it up. Kaggle is no longer a
    dependency for the catalog.

WHY TMDB AND NOT ANOTHER DUMP
    The catalog schema is already TMDB-shaped: `id` is a TMDB movie id and
    every list column comes from TMDB's nested-dict fields. One live source
    means one id space, one spelling of every studio, and no cross-dataset
    reconciliation or title-based guessing anywhere in the pipeline.

USAGE
    Unlike prepare_movibot_data.py (run from inside data_preprocessing/),
    this module imports from the package, so run it from the repo root:

        python -m data_preprocessing.fetch_tmdb_catalog
        python -m data_preprocessing.fetch_tmdb_catalog --limit 10   # test run

OUTPUT
    data_preprocessing/data_ready/tmdb_catalog.csv -- same 25 columns in
    the same order as supabase_movies.csv, ready to concatenate. Gitignored
    and regenerable, like every other file under data_ready/.

CREDENTIALS
    Set TMDB_ACCESS_TOKEN (v4 bearer, preferred) or TMDB_API_KEY (v3) in the
    environment. A TMDB key is free. No LLM budget is spent by this script.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from data_preprocessing.prepare_movibot_data import DEMO_STUDIOS, json_list

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# TMDB company ids for DEMO_STUDIOS. Verified at runtime against the live
# company names before any discovery call, so a wrong id fails loudly instead
# of silently returning somebody else's catalog.
STUDIO_COMPANY_IDS = {
    2: "Walt Disney Pictures",
    3166: "Walt Disney Productions",
    6125: "Walt Disney Animation Studios",
    171656: "Walt Disney Feature Animation",
    3: "Pixar",
}

# TMDB has since renamed company 3 to plain "Pixar", but the 2017 Kaggle dump
# -- and therefore DEMO_STUDIOS and every existing catalog row -- spells it
# "Pixar Animation Studios". New rows are rewritten to the catalog spelling so
# the studio column stays internally consistent and the DEMO_STUDIOS filter
# keeps working on both halves of the data.
STUDIO_NAME_ALIASES = {
    "Pixar": "Pixar Animation Studios",
}

# The whole catalog by default. TMDB covers Disney's entire history, so there
# is no reason to start at the Kaggle dump's July 2017 cutoff -- that dump is
# no longer a source. Disney's first feature is 1937; 1920 is a safe floor.
DEFAULT_RELEASE_FLOOR = "1920-01-01"

# TMDB catalogues Pixar SparkShorts, "Forky Asks a Question" episodes, promo
# clips and making-of featurettes as standalone movies. On the Disney/Pixar
# scope they are 39% of everything released since 2017, and recommending a
# 3-minute Disney+ clip to someone asking for a movie is simply wrong. 40
# minutes is the long-standing Academy threshold for a feature film.
# Pass --min-runtime 0 to keep every entry.
MIN_FEATURE_RUNTIME_MINUTES = 40

# Run from the repo root, so this must be the package-qualified path -- it is
# the same folder prepare_movibot_data.py writes to when run from inside
# data_preprocessing/, and it is gitignored.
DEFAULT_OUT_DIR = Path("data_preprocessing") / "data_ready"

# Column order must stay identical to prepare_movibot_data.py's Supabase output.
OUTPUT_COLUMNS = [
    "id", "imdb_id", "title", "original_title", "release_year", "release_date",
    "runtime_minutes", "genres", "production_companies", "production_countries",
    "spoken_languages", "belongs_to_collection", "popularity", "vote_average",
    "vote_count", "budget", "revenue", "overview", "tagline", "status",
    "original_language", "adult", "video", "keywords", "has_mpst_synopsis",
]

_IMDB_ID = re.compile(r"^tt\d+$")
_REQUEST_PAUSE_SECONDS = 0.05


class TmdbError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------

def build_session() -> requests.Session:
    token = (os.environ.get("TMDB_ACCESS_TOKEN") or "").strip()
    api_key = (os.environ.get("TMDB_API_KEY") or "").strip()

    if not token and not api_key:
        raise TmdbError(
            "No TMDB credentials found. Set TMDB_ACCESS_TOKEN (v4 bearer) or "
            "TMDB_API_KEY (v3) in your environment/.env. Both are free."
        )

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        session.params = {"api_key": api_key}
    return session


def get_json(
    session: requests.Session, path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    url = f"{TMDB_BASE_URL}{path}"

    for attempt in range(4):
        try:
            response = session.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            if attempt == 3:
                raise TmdbError(f"Network error calling {path}: {exc}") from exc
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", "2")) + 1)
            continue
        if response.status_code == 401:
            raise TmdbError(
                f"TMDB rejected the credentials (401) on {path}. "
                "Check TMDB_ACCESS_TOKEN / TMDB_API_KEY."
            )
        if response.status_code >= 500:
            if attempt == 3:
                raise TmdbError(f"TMDB server error {response.status_code} on {path}")
            time.sleep(2 ** attempt)
            continue
        if not response.ok:
            raise TmdbError(f"TMDB returned {response.status_code} on {path}")

        time.sleep(_REQUEST_PAUSE_SECONDS)
        return response.json()

    raise TmdbError(f"Gave up calling {path} after retries")


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

def verify_company_ids(session: requests.Session) -> None:
    """Fails loudly if a hardcoded company id no longer means what we think."""
    for company_id, expected in STUDIO_COMPANY_IDS.items():
        actual = (get_json(session, f"/company/{company_id}").get("name") or "").strip()
        if actual.lower() != expected.lower():
            raise TmdbError(
                f"TMDB company id {company_id} resolves to {actual!r}, "
                f"expected {expected!r}. Refusing to fetch a wrong catalog."
            )


def discover_movie_ids(
    session: requests.Session, release_floor: str
) -> Iterator[int]:
    seen: set[int] = set()
    company_filter = "|".join(str(i) for i in STUDIO_COMPANY_IDS)
    page = 1

    while True:
        payload = get_json(
            session,
            "/discover/movie",
            {
                "with_companies": company_filter,
                "primary_release_date.gte": release_floor,
                "sort_by": "primary_release_date.asc",
                "include_adult": "false",
                "page": page,
            },
        )

        results = payload.get("results") or []
        if not results:
            return

        for item in results:
            movie_id = item.get("id")
            if isinstance(movie_id, int) and movie_id not in seen:
                seen.add(movie_id)
                yield movie_id

        if page >= min(int(payload.get("total_pages") or 1), 500):
            return
        page += 1


# ---------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------

def names_of(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = str((item or {}).get("name", "")).strip() if isinstance(item, dict) else ""
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def canonicalize_companies(names: list[str]) -> list[str]:
    """Rewrites TMDB's current studio spellings to the catalog's, order-stable."""
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = STUDIO_NAME_ALIASES.get(name, name)
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def is_in_demo_scope(companies: list[str]) -> bool:
    """Same studio-membership rule prepare_movibot_data.py applies."""
    return any(name in DEMO_STUDIOS for name in canonicalize_companies(companies))


def build_row(detail: dict[str, Any], keywords: list[str]) -> dict[str, Any] | None:
    """Applies the same usability rules as clean_movies(); None means unusable."""
    movie_id = detail.get("id")
    title = str(detail.get("title") or "").strip()
    overview = str(detail.get("overview") or "").strip()
    release_date = str(detail.get("release_date") or "").strip()
    runtime = detail.get("runtime")

    if not isinstance(movie_id, int) or not title or not overview:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_date):
        return None
    if not isinstance(runtime, (int, float)) or runtime <= 0:
        return None

    imdb_id = str(detail.get("imdb_id") or "").strip()
    collection = detail.get("belongs_to_collection") or {}

    return {
        "id": movie_id,
        "imdb_id": imdb_id if _IMDB_ID.fullmatch(imdb_id) else "",
        "title": title,
        "original_title": str(detail.get("original_title") or "").strip(),
        "release_year": int(release_date[:4]),
        "release_date": release_date,
        "runtime_minutes": int(runtime),
        "genres": json_list(names_of(detail.get("genres"))),
        "production_companies": json_list(
            canonicalize_companies(names_of(detail.get("production_companies")))
        ),
        "production_countries": json_list(names_of(detail.get("production_countries"))),
        "spoken_languages": json_list(names_of(detail.get("spoken_languages"))),
        "belongs_to_collection": str(collection.get("name", "")).strip()
        if isinstance(collection, dict) else "",
        "popularity": detail.get("popularity"),
        "vote_average": detail.get("vote_average"),
        "vote_count": detail.get("vote_count") or 0,
        "budget": detail.get("budget") or 0,
        "revenue": detail.get("revenue") or 0,
        "overview": overview,
        "tagline": str(detail.get("tagline") or "").strip(),
        "status": str(detail.get("status") or "").strip(),
        "original_language": str(detail.get("original_language") or "").strip(),
        "adult": bool(detail.get("adult")),
        "video": bool(detail.get("video")),
        "keywords": json_list(keywords),
        # MPST is a static 2018 corpus, so nothing released after the Kaggle
        # cutoff can have a synopsis in it. Transcripts are a separate track.
        "has_mpst_synopsis": False,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output folder (default: {DEFAULT_OUT_DIR}, relative to the repo root).",
    )
    parser.add_argument(
        "--since", default=DEFAULT_RELEASE_FLOOR,
        help=f"Earliest primary release date, YYYY-MM-DD (default: {DEFAULT_RELEASE_FLOOR}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only fetch the first N discovered movies (cheap test run).",
    )
    parser.add_argument(
        "--min-runtime", type=int, default=MIN_FEATURE_RUNTIME_MINUTES,
        help=(
            "Drop entries shorter than this many minutes "
            f"(default: {MIN_FEATURE_RUNTIME_MINUTES}, the feature-film threshold). "
            "Use 0 to keep shorts and featurettes."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.since):
        print(f"--since must be YYYY-MM-DD, got {args.since!r}", file=sys.stderr)
        return 2

    if args.min_runtime < 0:
        print(f"--min-runtime must be >= 0, got {args.min_runtime}", file=sys.stderr)
        return 2

    try:
        session = build_session()
        print("1/3 Verifying TMDB company ids...")
        verify_company_ids(session)

        print(f"2/3 Discovering {'/'.join(STUDIO_COMPANY_IDS.values())} releases since {args.since}...")
        movie_ids = list(discover_movie_ids(session, args.since))
        if args.limit is not None:
            movie_ids = movie_ids[: args.limit]
        print(f"    {len(movie_ids):,} candidate movies")

        print("3/3 Fetching details...")
        rows: list[dict[str, Any]] = []
        skipped_scope = 0
        skipped_unusable = 0
        skipped_short = 0

        for index, movie_id in enumerate(movie_ids, start=1):
            detail = get_json(session, f"/movie/{movie_id}")

            if not is_in_demo_scope(names_of(detail.get("production_companies"))):
                skipped_scope += 1
                continue

            keywords_payload = get_json(session, f"/movie/{movie_id}/keywords")
            row = build_row(detail, names_of(keywords_payload.get("keywords")))
            if row is None:
                skipped_unusable += 1
                continue

            # Genre is never filtered: animated and live-action features are
            # both in scope. Only sub-feature runtimes are dropped.
            if row["runtime_minutes"] < args.min_runtime:
                skipped_short += 1
                continue

            rows.append(row)
            if index % 25 == 0:
                # flush: a full-catalog run takes many minutes, and buffered
                # stdout makes it look hung when piped to a file.
                print(f"    {index}/{len(movie_ids)}...", flush=True)

    except TmdbError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "tmdb_catalog.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== TMDB CATALOG UPDATE COMPLETE ===")
    print(f"Discovered:        {len(movie_ids):,}")
    print(f"Dropped, scope:    {skipped_scope:,} (no DEMO_STUDIOS credit on the full record)")
    print(f"Dropped, unusable: {skipped_unusable:,} (missing date/runtime/overview)")
    print(f"Dropped, short:    {skipped_short:,} (under {args.min_runtime} min)")
    print(f"Written:           {len(rows):,} -> {out_path}")
    if rows:
        years = [r["release_year"] for r in rows]
        print(f"Year range:        {min(years)}-{max(years)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
