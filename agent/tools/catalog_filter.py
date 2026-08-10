"""CatalogFilter tool - structured filtering (year, studio, genre, runtime).

Mock: reads from data_preprocessing/data_ready/supabase_movies.csv.
Real (Chunk 2+4): queries the `movies` table via Supabase.
"""

import os
import json
from typing import Any

import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready", "supabase_movies.csv")

_catalog_cache = None
_genres_cache = None
_studios_cache = None


def _load_catalog():
    """Load and cache the CSV once."""
    global _catalog_cache, _genres_cache, _studios_cache
    if _catalog_cache is not None:
        return

    _catalog_cache = pd.read_csv(_CSV_PATH)

    # Parse JSON columns
    _catalog_cache["genres_list"] = _catalog_cache["genres"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    _catalog_cache["studios_list"] = _catalog_cache["production_companies"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    # Build genre and studio vocabularies
    all_genres = set()
    all_studios = set()
    for genres in _catalog_cache["genres_list"]:
        all_genres.update(genres)
    for studios in _catalog_cache["studios_list"]:
        all_studios.update(studios)

    _genres_cache = frozenset(all_genres)
    _studios_cache = frozenset(all_studios)


def known_genres() -> set[str]:
    """Return the set of unique genres in the catalog."""
    _load_catalog()
    return set(_genres_cache)


def known_studios() -> set[str]:
    """Return the set of unique studios in the catalog."""
    _load_catalog()
    return set(_studios_cache)


def run(constraints: dict[str, Any]) -> dict[str, Any]:
    """Filter catalog by year, runtime, studio, genre.

    Args:
        constraints: dict with keys year_min, year_max, studio, genre,
                     runtime_min, runtime_max (any/all can be None).

    Returns:
        {"matched_count": int, "candidates": [...], "constraints_applied": constraints}
    """
    _load_catalog()
    df = _catalog_cache.copy()

    # Apply year filter
    if constraints.get("year_min"):
        df = df[df["release_year"] >= constraints["year_min"]]
    if constraints.get("year_max"):
        df = df[df["release_year"] <= constraints["year_max"]]

    # Apply runtime filter
    if constraints.get("runtime_min"):
        df = df[df["runtime_minutes"] >= constraints["runtime_min"]]
    if constraints.get("runtime_max"):
        df = df[df["runtime_minutes"] <= constraints["runtime_max"]]

    # Apply genre filter (exact match, case-insensitive)
    if constraints.get("genre"):
        target_genre = constraints["genre"]
        df = df[df["genres_list"].apply(
            lambda genres: any(g.lower() == target_genre.lower() for g in genres)
        )]

    # Apply studio filter (substring match, case-insensitive)
    if constraints.get("studio"):
        target_studio = constraints["studio"]
        df = df[df["studios_list"].apply(
            lambda studios: any(target_studio.lower() in s.lower() for s in studios)
        )]

    # Build candidate list (minimal columns)
    candidates = []
    for _, row in df.iterrows():
        candidates.append({
            "id": int(row["id"]),
            "movie_id": int(row["id"]),
            "title": row["title"],
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "runtime_minutes": row["runtime_minutes"],
            "genres": row["genres_list"],
            "production_companies": row["studios_list"]
        })

    return {
        "matched_count": len(candidates),
        "candidates": candidates,
        "constraints_applied": constraints
    }
