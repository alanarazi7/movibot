#!/usr/bin/env python3
"""Scrape Wikipedia once for every catalog film, into an offline cache.

The agent reads this cache and never fetches Wikipedia live -- which is why
wikipedia_client lives here in data_preprocessing rather than under agent/.

Each film is fetched exactly once and both halves are derived from that single
article: the Plot section for scene questions, everything else (production,
reception, themes) for tone questions. An earlier version fetched twice, the
second time without the release year, so the two halves could come from two
different articles.

Row count follows catalog.csv -- feature films only, 238 at current
scope.

    python data_preprocessing/scrape_wikipedia.py
    python data_preprocessing/scrape_wikipedia.py --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wikipedia_client  # noqa: E402

_DATA_READY = Path(__file__).resolve().parent / "data_ready"
_CATALOG_CSV = _DATA_READY / "catalog.csv"
_OUTPUT_CSV = _DATA_READY / "wikipedia_cache.csv"

DELAY_BETWEEN_REQUESTS = 0.5  # seconds, to be polite to Wikipedia


def scrape_catalog(limit: int | None = None) -> pd.DataFrame:
    catalog = pd.read_csv(_CATALOG_CSV)
    if limit:
        catalog = catalog.head(limit)

    total = len(catalog)
    print(f"\nScraping Wikipedia for {total} catalog films...")
    print("-" * 68)

    results = []
    for position, (_, row) in enumerate(catalog.iterrows(), start=1):
        title = row["title"]
        year = int(row["release_year"]) if pd.notna(row["release_year"]) else None

        found = wikipedia_client.fetch_page_extract(title, year=year)

        if found:
            resolved, extract = found
            sections = wikipedia_client.split_into_sections(extract)
            plot_text = wikipedia_client.get_plot(sections)
            non_plot_text = wikipedia_client.get_non_plot(sections)
        else:
            resolved, plot_text, non_plot_text = None, None, None

        results.append({
            "id": row["id"],
            "title": title,
            "imdb_id": row["imdb_id"],
            "wiki_page_found": found is not None,
            "wiki_title": resolved,
            "plot_text": plot_text,
            "non_plot_text": non_plot_text,
        })

        if position % 25 == 0 or position == total:
            print(f"  [{position}/{total}] {title}")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("-" * 68)
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Only scrape the first N films.")
    args = parser.parse_args()

    cache = scrape_catalog(args.limit)

    total = len(cache)
    found = int(cache["wiki_page_found"].sum())
    plots = int(cache["plot_text"].notna().sum())
    non_plots = int(cache["non_plot_text"].notna().sum())

    print("\nResults:")
    print(f"  {found}/{total} articles resolved ({100.0 * found / total:.1f}%)")
    print(f"  {plots}/{total} with a Plot section ({100.0 * plots / total:.1f}%)")
    print(f"  {non_plots}/{total} with non-Plot text ({100.0 * non_plots / total:.1f}%)")

    missing = cache.loc[~cache["wiki_page_found"], "title"].tolist()
    if missing:
        print(f"\n  No article found for {len(missing)}: {', '.join(missing)}")

    no_plot = cache.loc[cache["wiki_page_found"] & cache["plot_text"].isna(), "title"]
    if len(no_plot):
        print(f"\n  Article but no Plot section for {len(no_plot)}: {', '.join(no_plot)}")

    if args.limit:
        print("\n--limit was set; not overwriting the cache.")
        return

    _DATA_READY.mkdir(parents=True, exist_ok=True)
    cache.to_csv(_OUTPUT_CSV, index=False)
    print(f"\nWrote {_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
