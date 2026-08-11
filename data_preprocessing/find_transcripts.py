#!/usr/bin/env python3
"""Find movie transcripts from HuggingFace mocboch/movie_scripts for catalog movies.

Discover which of the 303 catalog movies have a matching transcript available.
Writes transcript_matches.csv with coverage metadata only (not the transcript text itself).
"""

import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from datasets import load_dataset

_BASE_DIR = Path(__file__).parent.parent
_DATA_FULL = _BASE_DIR / "data_preprocessing" / "data_full"
_DATA_READY = _BASE_DIR / "data_preprocessing" / "data_ready"
_CATALOG_CSV = _DATA_READY / "supabase_movies.csv"
_OUTPUT_CSV = _DATA_READY / "transcript_matches.csv"


def normalize_title(title: str) -> str:
    """Normalize title for matching: lowercase, strip punctuation/year."""
    # Remove year suffix (e.g., " (2013)" -> "")
    title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
    # Lowercase and remove extra whitespace
    title = title.lower().strip()
    # Remove common punctuation
    title = re.sub(r'[\'"\-&]', '', title)
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title)
    return title


def load_catalog() -> pd.DataFrame:
    """Load catalog CSV and extract titles."""
    df = pd.read_csv(_CATALOG_CSV)
    return df[['id', 'title', 'imdb_id']].copy()


def load_transcripts_from_huggingface() -> dict[str, str]:
    """Load mocboch/movie_scripts from HuggingFace.

    Returns a mapping of normalized title -> original title.
    Attempts direct CSV loading if dataset API fails.
    """
    print("Loading mocboch/movie_scripts from HuggingFace...")

    # Try using the datasets library first (standard approach)
    try:
        dataset = load_dataset(
            "mocboch/movie_scripts",
            data_files="Table_1_Exploratory_Data_With_Scripts.csv",
            split="train"
        )
        transcripts = {}
        for row in dataset:
            # This table has 'Title' or similar field
            for title_key in ['Title', 'title', 'movie_name', 'name']:
                if title_key in row and row[title_key]:
                    norm_title = normalize_title(str(row[title_key]))
                    if norm_title:
                        transcripts[norm_title] = str(row[title_key])
                    break
        print(f"Loaded {len(transcripts)} transcripts from HuggingFace (via datasets API)")
        return transcripts
    except Exception as e:
        print(f"HuggingFace datasets API failed: {e}")

    # Fallback: try direct CSV via pandas
    print("Attempting direct CSV download from HuggingFace...")
    try:
        url = "https://huggingface.co/datasets/mocboch/movie_scripts/resolve/main/Table_1_Exploratory_Data_With_Scripts.csv"
        df = pd.read_csv(url, low_memory=False)
        transcripts = {}

        # Find the title column
        title_col = None
        for col in ['Title', 'title', 'movie_name', 'name']:
            if col in df.columns:
                title_col = col
                break

        if title_col:
            for idx, row in df.iterrows():
                title = str(row[title_col]) if pd.notna(row[title_col]) else None
                if title:
                    norm_title = normalize_title(title)
                    if norm_title:
                        transcripts[norm_title] = title

        print(f"Loaded {len(transcripts)} transcripts from direct CSV")
        return transcripts
    except Exception as e:
        print(f"Direct CSV download also failed: {e}")
        print("\nNote: HuggingFace dataset has schema issues. Returning empty.")
        return {}


def match_catalog_to_transcripts(
    catalog: pd.DataFrame,
    transcripts: dict[str, str]
) -> pd.DataFrame:
    """Match catalog movies to available transcripts."""
    results = []

    for _, row in catalog.iterrows():
        norm_cat_title = normalize_title(row['title'])
        transcript_found = norm_cat_title in transcripts
        transcript_source = transcripts[norm_cat_title] if transcript_found else None

        results.append({
            'id': row['id'],
            'title': row['title'],
            'imdb_id': row['imdb_id'],
            'transcript_found': transcript_found,
            'transcript_source_file': transcript_source
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("MoviBot: Transcript Coverage Discovery")
    print("=" * 60)

    # Load catalog
    print(f"\nLoading catalog from {_CATALOG_CSV}...")
    catalog = load_catalog()
    print(f"Loaded {len(catalog)} catalog movies")

    # Load transcripts from HuggingFace
    transcripts = load_transcripts_from_huggingface()

    if not transcripts:
        print("Warning: No transcripts loaded. Proceeding with empty results.")

    # Match
    print("\nMatching catalog to transcripts...")
    matches = match_catalog_to_transcripts(catalog, transcripts)

    # Statistics
    found_count = matches['transcript_found'].sum()
    total_count = len(matches)
    coverage_pct = 100.0 * found_count / total_count if total_count > 0 else 0

    print(f"\n{found_count} of {total_count} catalog movies have a transcript ({coverage_pct:.1f}% coverage)")

    # Write output
    _DATA_READY.mkdir(parents=True, exist_ok=True)
    matches.to_csv(_OUTPUT_CSV, index=False)
    print(f"\nWrote {_OUTPUT_CSV}")

    # Show first few matches
    if found_count > 0:
        print("\nFirst 5 matches found:")
        for _, row in matches[matches['transcript_found']].head(5).iterrows():
            print(f"  - {row['title']} (source: {row['transcript_source_file']})")
    else:
        print("\nNo matches found in this transcript corpus.")
        print("Consider escalating to a larger corpus:")
        print("  - Kaggle ismaeldwikat/movies-scripts (246MB)")
        print("  - Kaggle gufukuro/movie-scripts-corpus (2.22GB)")
        print("  - Kaggle fayaznoor10/movie-transcripts-59k (2.28GB)")


if __name__ == "__main__":
    main()
