"""Joins the TMDB catalog with the Wikipedia plot summaries into one CSV.

This is a LEFT join on the catalog: every movie stays, and the plot columns
are simply blank for films with no English Wikipedia article. Dropping those
rows instead would quietly shrink the library the agent can filter over, so
that structured queries -- "a Disney film under 90 minutes" -- would silently
never see a third of the catalog.

`has_plot` marks which rows carry usable long-form text. Those are the rows
that go to Pinecone for semantic search and that SceneSearch can reason over;
the rest can still be reached by CatalogFilter and, if needed, by a live
Wikipedia lookup at query time.

USAGE
    Run from the repo root, after fetch_tmdb_catalog.py and
    fetch_wikipedia_plots.py:

        python -m data_preprocessing.build_movie_dataset

OUTPUT
    data_preprocessing/data_ready/movies_with_plots.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

DEFAULT_DIR = Path("data_preprocessing") / "data_ready"
CATALOG_NAME = "tmdb_catalog.csv"
PLOTS_NAME = "wikipedia_plots.csv"
OUTPUT_NAME = "movies_with_plots.csv"

# Appended to every catalog column, in this order.
PLOT_COLUMNS = ["wikipedia_title", "wikipedia_url", "plot_words", "plot_text"]

# csv has no booleans; keep it explicit and lowercase so Postgres and pandas
# both read it back the same way.
TRUE = "true"
FALSE = "false"


def attach_plots(
    catalog: Sequence[dict[str, str]],
    plots: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Left-joins plots onto catalog rows. Never mutates the inputs."""
    by_movie: dict[str, dict[str, str]] = {}
    for row in plots:
        key = (row.get("movie_id") or "").strip()
        if key and key not in by_movie:
            by_movie[key] = row

    merged: list[dict[str, Any]] = []
    matched_keys: set[str] = set()

    for row in catalog:
        movie_id = (row.get("id") or "").strip()
        plot = by_movie.get(movie_id)
        text = (plot or {}).get("plot_text", "").strip()

        out = dict(row)
        if text:
            matched_keys.add(movie_id)
            for column in PLOT_COLUMNS:
                out[column] = (plot or {}).get(column, "")
        else:
            for column in PLOT_COLUMNS:
                out[column] = ""
        out["has_plot"] = TRUE if text else FALSE
        merged.append(out)

    with_plot = len(matched_keys)
    return merged, {
        "total": len(catalog),
        "with_plot": with_plot,
        "without_plot": len(catalog) - with_plot,
        # Plot rows whose movie is not in the catalog: a sign the two files
        # were generated from different catalog versions.
        "orphan_plots": len([k for k in by_movie if k not in matched_keys]),
    }


def build_rows(
    catalog: Sequence[dict[str, str]],
    plots: Sequence[dict[str, str]],
    catalog_columns: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not catalog:
        raise ValueError("catalog is empty -- run fetch_tmdb_catalog.py first")

    merged, _ = attach_plots(catalog, plots)
    columns = [*catalog_columns, *PLOT_COLUMNS, "has_plot"]

    merged.sort(key=lambda r: int(r["id"]))
    ordered = [{c: row.get(c, "") for c in columns} for row in merged]
    return ordered, columns


def read_csv(path: Path, required: bool = True) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return [], []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--min-votes", type=int, default=0,
        help="Drop catalog rows below this TMDB vote_count. Disney's 1950s-70s "
             "TV output has no Wikipedia article and few votes; 10 removes it "
             "without touching any well-known film (default: 0, keep all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        catalog, catalog_columns = read_csv(args.data_dir / CATALOG_NAME)
        plots, _ = read_csv(args.data_dir / PLOTS_NAME, required=False)
    except FileNotFoundError as exc:
        print(f"ERROR: missing input {exc}. Run fetch_tmdb_catalog.py first.",
              file=sys.stderr)
        return 1

    dropped_votes = 0
    if args.min_votes > 0:
        kept = []
        for row in catalog:
            try:
                votes = int(float(row.get("vote_count") or 0))
            except ValueError:
                votes = 0
            if votes >= args.min_votes:
                kept.append(row)
            else:
                dropped_votes += 1
        catalog = kept

    try:
        rows, columns = build_rows(catalog, plots, catalog_columns)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _, stats = attach_plots(catalog, plots)

    out_path = args.data_dir / OUTPUT_NAME
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("=== MOVIE DATASET BUILT ===")
    if dropped_votes:
        print(f"Dropped, few votes: {dropped_votes:,} (under {args.min_votes})")
    print(f"Movies:             {stats['total']:,}")
    print(f"  with a plot:      {stats['with_plot']:,} "
          f"({stats['with_plot'] / stats['total']:.1%})")
    print(f"  without:          {stats['without_plot']:,}")
    if stats["orphan_plots"]:
        print(f"Orphan plot rows:   {stats['orphan_plots']:,} "
              "(plot exists but its movie is not in the catalog -- "
              "regenerate both files from the same run)")
    print(f"Columns:            {len(columns)}")
    print(f"Written:            {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
