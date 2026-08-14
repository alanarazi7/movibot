#!/usr/bin/env python3
"""Builds the static JSON the GUI's Data Navigator browses.

Why static files rather than an API endpoint: the navigator has to work on
Vercel, where `public/` is served by static routing and never touches the Python
function. That is guaranteed to work; whether the serverless bundle ships the
CSVs under data_preprocessing/ is not. The cost is ~4.5 MB of duplicated text,
which buys an inspector that cannot break in production.

Two tiers, because inlining everything would mean a 4.5 MB page load to look at
one film:

    catalog.json      one row per film, metadata only (~270 KB), loaded when
                      the panel is first opened -- enough to search, filter,
                      sort, and render the table
    films/<id>.json   the long text for one film (~18 KB), fetched only when
                      that film's row is expanded

Regenerate after any change to data_ready/:

    python scripts/build_data_navigator.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_DATA_READY = _ROOT / "data_preprocessing" / "data_ready"
_OUT = _ROOT / "public" / "data"

# Stored as JSON-encoded strings in the CSVs; decoded back to real arrays here
# so the browser does not have to parse them twice.
LIST_COLUMNS = [
    "genres", "production_companies", "production_countries",
    "spoken_languages", "keywords",
]


def parse_list(value) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def clean(value):
    """NaN -> None, numpy scalars -> plain Python, so json.dump succeeds."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    catalog = pd.read_csv(_DATA_READY / "supabase_movies.csv")
    synopses = pd.read_csv(_DATA_READY / "pinecone_candidates.csv")
    wiki = pd.read_csv(_DATA_READY / "wikipedia_cache.csv")
    chunks = pd.read_parquet(_DATA_READY / "plot_chunks.parquet")

    syn_by_id = {int(r.movie_id): r for r in synopses.itertuples()}
    wiki_by_id = {int(r.id): r for r in wiki.itertuples()}
    chunks_by_id = {mid: grp.sort_values("chunk_index")
                    for mid, grp in chunks.groupby("movie_id")}

    if _OUT.exists():
        shutil.rmtree(_OUT)
    (_OUT / "films").mkdir(parents=True)

    index = []

    for row in catalog.itertuples():
        mid = int(row.id)
        syn = syn_by_id.get(mid)
        wk = wiki_by_id.get(mid)
        film_chunks = chunks_by_id.get(mid)

        wiki_plot = clean(wk.plot_text) if wk is not None else None
        wiki_nonplot = clean(wk.non_plot_text) if wk is not None else None

        index.append({
            "id": mid,
            "imdb_id": clean(row.imdb_id),
            "title": row.title,
            "original_title": clean(row.original_title),
            "year": int(row.release_year),
            "release_date": clean(row.release_date),
            "runtime": clean(row.runtime_minutes),
            "genres": parse_list(row.genres),
            "companies": parse_list(row.production_companies),
            "countries": parse_list(row.production_countries),
            "languages": parse_list(row.spoken_languages),
            "keywords": parse_list(row.keywords),
            "collection": clean(row.belongs_to_collection),
            "popularity": clean(row.popularity),
            "vote_average": clean(row.vote_average),
            "vote_count": clean(row.vote_count),
            "weighted_rating": clean(row.weighted_rating),
            "budget": clean(row.budget),
            "revenue": clean(row.revenue),
            "tagline": clean(row.tagline),
            "status": clean(row.status),
            "original_language": clean(row.original_language),
            "adult": bool(row.adult),
            "video": bool(row.video),
            "overview": clean(row.overview),
            # Coverage flags -- what the table shows at a glance, and what the
            # filters act on.
            "has_synopsis": syn is not None,
            "n_chunks": int(len(film_chunks)) if film_chunks is not None else 0,
            "has_wiki": wk is not None and bool(wk.wiki_page_found),
            "has_wiki_plot": wiki_plot is not None,
        })

        detail = {
            "id": mid,
            "title": row.title,
            "mpst_title": clean(syn.mpst_title) if syn is not None else None,
            "synopsis_source": clean(syn.synopsis_source) if syn is not None else None,
            "mpst_tags": parse_list(syn.mpst_tags) if syn is not None else [],
            "plot_synopsis": clean(syn.plot_synopsis) if syn is not None else None,
            "embedding_text_chars": (
                len(str(syn.embedding_text)) if syn is not None else 0
            ),
            "chunks": (
                [
                    {
                        "chunk_id": c.chunk_id,
                        "index": int(c.chunk_index),
                        "tokens": int(c.tokens),
                        "text": c.text,
                    }
                    for c in film_chunks.itertuples()
                ]
                if film_chunks is not None else []
            ),
            "wiki_title": clean(wk.wiki_title) if wk is not None else None,
            "wiki_plot": wiki_plot,
            "wiki_non_plot": wiki_nonplot,
        }

        (_OUT / "films" / f"{mid}.json").write_text(
            json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    index.sort(key=lambda f: -(f["weighted_rating"] or 0))

    summary = {
        "films": len(index),
        "with_synopsis": sum(f["has_synopsis"] for f in index),
        "passages": int(len(chunks)),
        "with_wiki_plot": sum(f["has_wiki_plot"] for f in index),
        "genres": sorted({g for f in index for g in f["genres"]}),
        "studios": sorted({c for f in index for c in f["companies"]}),
        "languages": sorted({f["original_language"] for f in index if f["original_language"]}),
        "year_min": min(f["year"] for f in index),
        "year_max": max(f["year"] for f in index),
    }

    (_OUT / "catalog.json").write_text(
        json.dumps({"summary": summary, "films": index},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    detail_bytes = sum(p.stat().st_size for p in (_OUT / "films").glob("*.json"))
    print(f"catalog.json    {(_OUT / 'catalog.json').stat().st_size / 1024:>8,.0f} KB"
          f"   {len(index)} films")
    print(f"films/*.json    {detail_bytes / 1024 / 1024:>8,.1f} MB"
          f"   {len(list((_OUT / 'films').glob('*.json')))} files")
    print(f"summary: {summary['films']} films, {summary['with_synopsis']} with synopsis, "
          f"{summary['passages']:,} passages, {summary['with_wiki_plot']} with a wiki plot")


if __name__ == "__main__":
    main()
