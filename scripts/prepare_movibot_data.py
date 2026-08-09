#!/usr/bin/env python3
"""
MoviBot final data-preparation pipeline.

INPUTS (default: data_full/)
    movies_metadata.csv
    keywords.csv
    mpst_full_data.csv

OUTPUTS (default: data_ready/)
    supabase_movies.csv
    pinecone_candidates.csv

The script:
1. Cleans Kaggle movie metadata.
2. Cleans/merges Kaggle keywords.
3. Cleans MPST without loading its irrelevant `review` column.
4. Matches Kaggle <-> MPST by exact normalized IMDb ID.
5. Builds one Supabase-ready catalog containing ALL usable Kaggle movies.
6. Builds one Pinecone-ingestion candidate file containing ALL exact MPST matches,
   sorted by descending Kaggle popularity.

No APIs are called.
No embeddings are generated.
No source files are modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare final MoviBot Supabase and Pinecone CSVs from raw datasets."
    )
    parser.add_argument(
        "--data-full",
        type=Path,
        default=Path("data_full"),
        help="Folder containing raw CSVs (default: data_full).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_ready"),
        help="Output folder (default: data_ready).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def file_mib(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_imdb_id(series: pd.Series) -> pd.Series:
    """
    Keep only IMDb IDs of the form tt12345...
    Missing/invalid IDs become pandas NA.
    """
    s = series.astype("string").str.strip()
    return s.where(s.str.fullmatch(r"tt\d+", na=False), pd.NA)


def load_jsonish(value: Any) -> Any:
    if pd.isna(value):
        return []

    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()
    if not text:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass

    return []


def names_from_nested(value: Any) -> list[str]:
    """
    TMDB-style fields are normally lists of dictionaries:
        [{"id": 16, "name": "Animation"}, ...]

    Return only stable, non-empty names.
    """
    obj = load_jsonish(value)
    if not isinstance(obj, list):
        return []

    out: list[str] = []
    seen: set[str] = set()

    for item in obj:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()

        if name and name not in seen:
            seen.add(name)
            out.append(name)

    return out


def keyword_names(value: Any) -> list[str]:
    return names_from_nested(value)


def json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def stable_union(list_series: pd.Series) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for values in list_series:
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)

    return out


def tags_to_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    # MPST tags are comma-separated labels.
    out: list[str] = []
    seen: set[str] = set()
    for item in text.split(","):
        tag = item.strip()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# ---------------------------------------------------------------------
# Clean Kaggle movie metadata
# ---------------------------------------------------------------------

def clean_movies(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [
        "id",
        "imdb_id",
        "title",
        "release_date",
        "runtime",
        "genres",
        "production_companies",
        "popularity",
        "overview",
    ]

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"movies_metadata.csv missing columns: {missing}")

    df = pd.read_csv(path, usecols=required, low_memory=False)
    raw_rows = len(df)

    # Raw values used for diagnostics / objective filtering.
    raw_id = df["id"]
    raw_title = clean_text(df["title"])
    raw_overview = clean_text(df["overview"])

    df["id"] = pd.to_numeric(raw_id, errors="coerce")
    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["runtime_minutes"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    valid_id = df["id"].notna()
    valid_title = raw_title.ne("")
    valid_date = df["release_date_parsed"].notna()
    valid_runtime_numeric = df["runtime_minutes"].notna()
    valid_runtime_positive = df["runtime_minutes"].gt(0)
    valid_overview = raw_overview.ne("")

    usable = (
        valid_id
        & valid_title
        & valid_date
        & valid_runtime_numeric
        & valid_runtime_positive
        & valid_overview
    )

    stats = {
        "raw_rows": raw_rows,
        "invalid_id": int((~valid_id).sum()),
        "blank_title": int((~valid_title).sum()),
        "invalid_release_date": int((~valid_date).sum()),
        "invalid_runtime": int((~valid_runtime_numeric).sum()),
        "runtime_nonpositive": int((valid_runtime_numeric & ~valid_runtime_positive).sum()),
        "blank_overview": int((~valid_overview).sum()),
        "rows_after_usability_rules": int(usable.sum()),
    }

    df = df.loc[usable].copy()

    df["id"] = df["id"].astype("int64")
    df["imdb_id"] = normalize_imdb_id(df["imdb_id"])
    df["title"] = clean_text(df["title"])
    df["overview"] = clean_text(df["overview"])
    df["release_year"] = df["release_date_parsed"].dt.year.astype("int64")

    df["genres_list"] = df["genres"].apply(names_from_nested)
    df["companies_list"] = df["production_companies"].apply(names_from_nested)

    # Deterministic duplicate resolution.
    # Prefer the row with richer metadata, then higher popularity,
    # then longer overview.
    df["_completeness"] = (
        df["imdb_id"].notna().astype(int)
        + df["popularity"].notna().astype(int)
        + df["genres_list"].map(bool).astype(int)
        + df["companies_list"].map(bool).astype(int)
    )
    df["_pop_sort"] = df["popularity"].fillna(float("-inf"))
    df["_overview_len"] = df["overview"].str.len()

    duplicate_rows = int(df["id"].duplicated(keep=False).sum())

    df = (
        df.sort_values(
            ["id", "_completeness", "_pop_sort", "_overview_len"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("id", keep="first")
        .reset_index(drop=True)
    )

    stats["duplicate_rows_before_dedup"] = duplicate_rows
    stats["duplicates_removed"] = stats["rows_after_usability_rules"] - len(df)
    stats["final_movies"] = len(df)

    out = pd.DataFrame(
        {
            "id": df["id"],
            "imdb_id": df["imdb_id"],
            "title": df["title"],
            "release_year": df["release_year"],
            "runtime_minutes": df["runtime_minutes"],
            "genres": df["genres_list"].apply(json_list),
            "production_companies": df["companies_list"].apply(json_list),
            "popularity": df["popularity"],
            "overview": df["overview"],
        }
    )

    return out, stats


# ---------------------------------------------------------------------
# Clean Kaggle keywords
# ---------------------------------------------------------------------

def clean_keywords(
    path: Path,
    valid_movie_ids: set[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = ["id", "keywords"]

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"keywords.csv missing columns: {missing}")

    df = pd.read_csv(path, usecols=required, low_memory=False)
    raw_rows = len(df)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    invalid_ids = int(df["id"].isna().sum())
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype("int64")

    df["_keywords_list"] = df["keywords"].apply(keyword_names)

    duplicate_rows = int(df["id"].duplicated(keep=False).sum())
    unique_before = int(df["id"].nunique())

    merged = (
        df.groupby("id", sort=False)["_keywords_list"]
        .apply(stable_union)
        .reset_index()
    )

    unique_after = len(merged)

    before_alignment = len(merged)
    merged = merged[merged["id"].isin(valid_movie_ids)].copy()
    removed_nonmovie = before_alignment - len(merged)

    merged["keywords"] = merged["_keywords_list"].apply(json_list)
    merged = merged[["id", "keywords"]].reset_index(drop=True)

    stats = {
        "raw_rows": raw_rows,
        "invalid_id": invalid_ids,
        "duplicate_rows_before_merge": duplicate_rows,
        "unique_ids_before_merge": unique_before,
        "unique_ids_after_merge": unique_after,
        "keyword_ids_removed_not_in_clean_movies": removed_nonmovie,
        "final_keyword_rows": len(merged),
    }

    return merged, stats


# ---------------------------------------------------------------------
# Clean MPST
# ---------------------------------------------------------------------

def clean_mpst(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()

    wanted = [
        "imdb_id",
        "title",
        "plot_synopsis",
        "tags",
        "synopsis_source",
    ]
    required = {"imdb_id", "title", "plot_synopsis"}

    available = [c for c in wanted if c in header]
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(f"mpst_full_data.csv missing columns: {missing}")

    # Crucial: do NOT load the huge irrelevant `review` column.
    df = pd.read_csv(path, usecols=available, low_memory=False)
    raw_rows = len(df)

    for col in available:
        df[col] = clean_text(df[col])

    df["imdb_id"] = normalize_imdb_id(df["imdb_id"])

    valid = (
        df["imdb_id"].notna()
        & df["title"].ne("")
        & df["plot_synopsis"].ne("")
    )

    unusable = int((~valid).sum())
    df = df.loc[valid].copy()

    duplicate_rows = int(df["imdb_id"].duplicated(keep=False).sum())

    # If duplicates ever appear, prefer the richest synopsis.
    df["_synopsis_words"] = df["plot_synopsis"].str.split().str.len()

    df = (
        df.sort_values(
            ["imdb_id", "_synopsis_words"],
            ascending=[True, False],
        )
        .drop_duplicates("imdb_id", keep="first")
        .reset_index(drop=True)
    )

    df["tags_list"] = (
        df["tags"].apply(tags_to_list)
        if "tags" in df.columns
        else [[] for _ in range(len(df))]
    )

    out = pd.DataFrame(
        {
            "imdb_id": df["imdb_id"],
            "mpst_title": df["title"],
            "plot_synopsis": df["plot_synopsis"],
            "mpst_tags": df["tags_list"].apply(json_list),
            "synopsis_source": (
                df["synopsis_source"]
                if "synopsis_source" in df.columns
                else ""
            ),
        }
    )

    stats = {
        "raw_rows": raw_rows,
        "unusable_rows": unusable,
        "duplicate_rows_before_dedup": duplicate_rows,
        "final_mpst_movies": len(out),
    }

    return out, stats


# ---------------------------------------------------------------------
# Embedding text
# ---------------------------------------------------------------------

def decode_json_list_for_text(value: Any) -> str:
    obj = load_jsonish(value)
    if isinstance(obj, list):
        return ", ".join(str(x).strip() for x in obj if str(x).strip())
    return ""


def build_embedding_text(row: pd.Series) -> str:
    """
    Local text to send to text-embedding-3-small later.

    The long synopsis is NOT intended to be stored as Pinecone metadata.
    """
    parts = [
        f"Title: {row['title']}",
        f"Plot synopsis: {row['plot_synopsis']}",
    ]

    tags = decode_json_list_for_text(row["mpst_tags"])
    keywords = decode_json_list_for_text(row["keywords"])

    if tags:
        parts.append(f"Story tags: {tags}")

    if keywords:
        parts.append(f"Keywords: {keywords}")

    return "\n".join(parts)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_full = args.data_full
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    movies_path = data_full / "movies_metadata.csv"
    keywords_path = data_full / "keywords.csv"
    mpst_path = data_full / "mpst_full_data.csv"

    for path in [movies_path, keywords_path, mpst_path]:
        require(path)

    print("1/5 Cleaning Kaggle movies...")
    movies, movie_stats = clean_movies(movies_path)

    print("2/5 Cleaning Kaggle keywords...")
    keywords, keyword_stats = clean_keywords(
        keywords_path,
        set(movies["id"].astype(int)),
    )

    # Every clean movie gets a keyword array, even if empty.
    movies = movies.merge(
        keywords,
        on="id",
        how="left",
        validate="one_to_one",
    )
    movies["keywords"] = movies["keywords"].fillna("[]")

    print("3/5 Cleaning MPST...")
    mpst, mpst_stats = clean_mpst(mpst_path)

    print("4/5 Exact IMDb-ID matching...")
    matched = movies[movies["imdb_id"].notna()].merge(
        mpst,
        on="imdb_id",
        how="inner",
        validate="one_to_one",
    )

    matched_ids = set(matched["id"].astype(int))

    # -------------------------------------------------------------
    # SUPABASE OUTPUT: ALL usable movies
    # -------------------------------------------------------------
    supabase = movies.copy()
    supabase["has_mpst_synopsis"] = supabase["id"].isin(matched_ids)

    supabase = supabase[
        [
            "id",
            "imdb_id",
            "title",
            "release_year",
            "runtime_minutes",
            "genres",
            "production_companies",
            "popularity",
            "overview",
            "keywords",
            "has_mpst_synopsis",
        ]
    ].sort_values("id").reset_index(drop=True)

    supabase_out = out_dir / "supabase_movies.csv"
    supabase.to_csv(supabase_out, index=False)

    # -------------------------------------------------------------
    # PINECONE OUTPUT: ALL exact MPST matches, NO row deletion
    # -------------------------------------------------------------
    print("5/5 Building ranked Pinecone candidate file...")

    # Pure popularity priority, as chosen for the course demo.
    matched["_priority_popularity"] = pd.to_numeric(
        matched["popularity"], errors="coerce"
    ).fillna(float("-inf"))

    pinecone = (
        matched.sort_values(
            ["_priority_popularity", "release_year", "id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
        .copy()
    )

    pinecone.insert(0, "priority_rank", range(1, len(pinecone) + 1))
    pinecone["embedding_text"] = pinecone.apply(build_embedding_text, axis=1)

    # This file is an INGESTION CANDIDATE file.
    # Only movie_id/title/release_year are proposed as persistent Pinecone metadata.
    pinecone = pd.DataFrame(
        {
            "priority_rank": pinecone["priority_rank"],
            "movie_id": pinecone["id"],
            "imdb_id": pinecone["imdb_id"],
            "title": pinecone["title"],
            "release_year": pinecone["release_year"],
            "popularity": pinecone["popularity"],
            "genres": pinecone["genres"],
            "production_companies": pinecone["production_companies"],
            "synopsis_source": pinecone["synopsis_source"],
            "embedding_text": pinecone["embedding_text"],
        }
    )

    pinecone_out = out_dir / "pinecone_candidates.csv"
    pinecone.to_csv(pinecone_out, index=False)

    # -------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------
    print("\n=== MOVIBOT DATA PREPARATION COMPLETE ===")

    print("\nKaggle movies:")
    for k, v in movie_stats.items():
        print(f"  {k}: {v:,}")

    print("\nKaggle keywords:")
    for k, v in keyword_stats.items():
        print(f"  {k}: {v:,}")

    print("\nMPST:")
    for k, v in mpst_stats.items():
        print(f"  {k}: {v:,}")

    coverage = len(pinecone) / len(supabase) * 100 if len(supabase) else 0.0

    print("\nFinal outputs:")
    print(
        f"  Supabase: {len(supabase):,} movies | "
        f"{file_mib(supabase_out):.2f} MiB | {supabase_out}"
    )
    print(
        f"  Pinecone candidates: {len(pinecone):,} movies | "
        f"{file_mib(pinecone_out):.2f} MiB local ingestion file | {pinecone_out}"
    )
    print(f"  Exact MPST coverage: {coverage:.2f}%")

    if len(pinecone) >= 3000:
        print("\nFor the planned demo-sized Pinecone index:")
        print("  use priority_rank <= 3000")
        print("  (the file intentionally keeps every exact MPST match).")

    print("\nPersistent Pinecone metadata recommendation:")
    print("  movie_id, title, release_year")
    print("  Do NOT store embedding_text as Pinecone metadata.")
    print("  embedding_text is only the local input used to create each vector.")


if __name__ == "__main__":
    main()
