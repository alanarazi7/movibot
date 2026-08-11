#!/usr/bin/env python3
"""Scrape Wikipedia content once for all 303 catalog movies.

Pre-cache Wikipedia pages offline to remove live API calls from agent runtime.
Saves Plot text and non-Plot text (Reception, Themes, etc.) for each movie.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Import the existing wikipedia_client from agent tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.tools import wikipedia_client

_BASE_DIR = Path(__file__).parent.parent
_DATA_READY = _BASE_DIR / "data_preprocessing" / "data_ready"
_CATALOG_CSV = _DATA_READY / "supabase_movies.csv"
_OUTPUT_CSV = _DATA_READY / "wikipedia_cache.csv"

DELAY_BETWEEN_REQUESTS = 0.5  # seconds, to be polite to Wikipedia


def scrape_wikipedia_for_catalog() -> pd.DataFrame:
    """Scrape Wikipedia for each catalog movie."""
    catalog = pd.read_csv(_CATALOG_CSV)

    results = []
    total = len(catalog)

    print(f"\nScraping Wikipedia for {total} catalog movies...")
    print("(showing progress every 50 movies)")
    print("-" * 60)

    for idx, row in catalog.iterrows():
        title = row['title']
        year = int(row['release_year']) if pd.notna(row['release_year']) else None

        # Fetch page extract (with year for better disambiguation)
        extract = wikipedia_client.fetch_page_extract(title, year=year)

        if extract:
            # Extract Plot section
            sections = wikipedia_client.split_into_sections(extract)
            plot_text = None
            non_plot_text = None

            for section_name, section_body in sections.items():
                if "plot" in section_name.lower() or "synopsis" in section_name.lower():
                    plot_text = section_body
                    break

            # Non-plot text
            non_plot_text = wikipedia_client.get_non_plot_text(title)

            page_found = True
        else:
            plot_text = None
            non_plot_text = None
            page_found = False

        results.append({
            'id': row['id'],
            'title': title,
            'imdb_id': row['imdb_id'],
            'wiki_page_found': page_found,
            'plot_text': plot_text,
            'non_plot_text': non_plot_text
        })

        # Progress report every 50
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{total}] processed {title}")

        # Be polite to Wikipedia
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"  [{total}/{total}] Done")
    print("-" * 60)

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("MoviBot: Wikipedia Cache Builder")
    print("=" * 60)

    # Scrape
    cache_df = scrape_wikipedia_for_catalog()

    # Statistics
    page_found = cache_df['wiki_page_found'].sum()
    plot_found = cache_df['plot_text'].notna().sum()
    non_plot_found = cache_df['non_plot_text'].notna().sum()
    total = len(cache_df)

    print(f"\nResults:")
    print(f"  {page_found}/{total} pages found ({100.0 * page_found / total:.1f}%)")
    print(f"  {plot_found}/{total} with Plot section ({100.0 * plot_found / total:.1f}%)")
    print(f"  {non_plot_found}/{total} with non-Plot text ({100.0 * non_plot_found / total:.1f}%)")

    # Write
    _DATA_READY.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(_OUTPUT_CSV, index=False)
    print(f"\nWrote {_OUTPUT_CSV}")

    # Show first few
    print("\nFirst 5 entries:")
    for _, row in cache_df.head(5).iterrows():
        plot_len = len(row['plot_text']) if pd.notna(row['plot_text']) else 0
        non_plot_len = len(row['non_plot_text']) if pd.notna(row['non_plot_text']) else 0
        print(f"  {row['title']:30s} | page:{row['wiki_page_found']!s:5s} | plot:{plot_len:4d} chars | non-plot:{non_plot_len:4d} chars")


if __name__ == "__main__":
    main()
