#!/usr/bin/env python3
"""Builds the passage index: chunk, embed, store.

This is the whole ingest path in one place. It replaces
scripts/build_chunk_index.py (which embedded locally with E5) and the unwritten
Pinecone half of scripts/ingest.py.

    python -m rag.ingest                    # chunk, embed, write the matrix
    python -m rag.ingest --limit 20         # same, first 20 films only
    python -m rag.ingest --pinecone         # also upsert to Pinecone
    python -m rag.ingest --dry-run          # chunk and report, embed nothing

Embedding costs money -- roughly $0.0075 for the full corpus, a few hundred
thousand tokens at text-embedding-3-small rates. --dry-run is free and answers
"how many passages would this produce" without spending anything.

Outputs (committed, so the repo runs without a rebuild):
    plot_chunks.parquet     one row per passage, with its text
    chunk_embeddings.npy    float32 matrix, one normalised row per passage
    chunk_index_meta.json   model, dims, chunk parameters, counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import chunking, corpora, embed  # noqa: E402
from rag.config import (  # noqa: E402
    CHUNK_TOKENS,
    DEBUG_CHUNKS_PER_CORPUS,
    CHUNKS_PARQUET,
    EMBED_MODEL,
    INDEX_META,
    MIN_CHUNK_TOKENS,
    OVERLAP_RATIO,
    PINECONE_INDEX,
    VECTORS_NPY,
)


def build_chunks(sources: list[str], limit: int | None = None,
                 debug: bool = False) -> pd.DataFrame:
    """Chunk every selected corpus, tagging each passage with where it came
    from so search can be restricted to one kind of evidence.

    `debug` caps each corpus at DEBUG_CHUNKS_PER_CORPUS passages -- enough to
    exercise chunk, embed, store and search end to end for a fraction of a
    cent, without waiting for or paying for the full corpus.
    """
    records = []
    for key in sources:
        corpus = corpora.CORPORA[key]
        df = corpus.load()
        if limit:
            df = df.head(limit)

        produced = []
        for row in df.itertuples():
            produced.extend(
                chunking.chunk_movie(
                    int(row.movie_id), str(row.title), str(row.text), source=key
                )
            )
            if debug and len(produced) >= DEBUG_CHUNKS_PER_CORPUS:
                break
        if debug:
            produced = produced[:DEBUG_CHUNKS_PER_CORPUS]

        records.extend(produced)
        print(f"  {corpus.label:24} {len(df):>4} films -> {len(produced):>5} passages"
              + ("   (debug cap)" if debug else ""))

    if not records:
        raise SystemExit("No chunks produced -- are the prepared CSVs populated?")
    return pd.DataFrame(records)


def upsert_pinecone(chunks: pd.DataFrame, vectors: np.ndarray) -> None:
    """Push the same vectors to Pinecone, so both stores agree.

    Metadata carries the passage text because search returns it as evidence;
    without it the planner would have a film id and no quotable line.
    """
    from pinecone import Pinecone
    import os

    key = os.environ.get("PINECONE_API_KEY", "")
    if not key or "your-" in key:
        raise SystemExit("--pinecone needs a real PINECONE_API_KEY.")

    index = Pinecone(api_key=key).Index(PINECONE_INDEX)
    batch = 100
    for start in range(0, len(chunks), batch):
        rows = chunks.iloc[start:start + batch]
        index.upsert([
            {
                "id": r.chunk_id,
                "values": vectors[start + i].tolist(),
                "metadata": {
                    "movie_id": int(r.movie_id),
                    "title": str(r.title),
                    "chunk_index": int(r.chunk_index),
                    "source": str(r.source),
                    "text": str(r.text),
                },
            }
            for i, r in enumerate(rows.itertuples())
        ])
        print(f"  upserted {min(start + batch, len(chunks))}/{len(chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only process the first N films")
    parser.add_argument("--sources", nargs="*", default=None,
                        help=f"corpora to index: {' '.join(corpora.CORPORA)} or all "
                             f"(default: {' '.join(corpora.DEFAULT_SOURCES)})")
    parser.add_argument("--pinecone", action="store_true",
                        help="also upsert the vectors to Pinecone")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and report only; embeds nothing, costs nothing")
    parser.add_argument("--debug", action="store_true",
                        help=f"only {DEBUG_CHUNKS_PER_CORPUS} passages per corpus, "
                             "to exercise the pipeline cheaply")
    args = parser.parse_args()

    sources = corpora.resolve(args.sources) if args.sources else corpora.DEFAULT_SOURCES
    print(f"Chunking {len(sources)} corpus/corpora"
          + (f", debug cap {DEBUG_CHUNKS_PER_CORPUS}/corpus" if args.debug else "") + " ...")
    chunks = build_chunks(sources, args.limit, debug=args.debug)
    films = chunks.movie_id.nunique()
    tokens = int(chunks.tokens.sum())

    print(
        f"\n  {len(chunks):,} passages from {films} distinct films "
        f"({chunks.tokens.median():.0f} median tokens, {tokens:,} total)"
    )

    if args.dry_run:
        print(f"\n--dry-run: embedding {tokens:,} tokens would cost roughly "
              f"${tokens / 1e6 * 0.02:.4f}. Nothing was sent.")
        return

    print(f"\nEmbedding with {EMBED_MODEL} ...")
    vectors, stats = embed.embed_texts_cached(
        chunks.embedding_text.tolist(), progress=True
    )

    if vectors.shape[0] != len(chunks):
        raise SystemExit(f"Vector/chunk mismatch: {vectors.shape[0]} vs {len(chunks)}")

    write_index(chunks, vectors, sources, debug=args.debug)

    print(f"\nWrote {Path(CHUNKS_PARQUET).name}, "
          f"{Path(VECTORS_NPY).name} {vectors.shape}, {Path(INDEX_META).name}")

    if args.pinecone:
        print(f"\nUpserting to Pinecone index {PINECONE_INDEX} ...")
        upsert_pinecone(chunks, vectors)

    # Only the freshly-embedded passages cost anything; the rest came from the
    # cache, which is the whole point of re-running this safely.
    sent_tokens = int(tokens * stats["embedded"] / max(len(chunks), 1))
    print(f"\nReused {stats['reused']:,} cached, embedded {stats['embedded']:,} new")
    print(f"Approximate cost this run: ${sent_tokens / 1e6 * 0.02:.4f}"
          f"  (full corpus would be ${tokens / 1e6 * 0.02:.4f})")


def write_index(chunks: pd.DataFrame, vectors: np.ndarray, sources: list[str],
                debug: bool = False) -> None:
    """Persist the passages, their vectors, and the metadata that lets a stale
    index be detected later."""
    tokens = int(chunks.tokens.sum())
    chunks.to_parquet(CHUNKS_PARQUET, index=False)
    np.save(VECTORS_NPY, vectors)
    Path(INDEX_META).write_text(json.dumps({
        "model": EMBED_MODEL,
        "embedding_dim": int(vectors.shape[1]),
        "num_chunks": int(len(chunks)),
        "num_movies": int(chunks.movie_id.nunique()),
        "sources": sources,
        "debug": bool(debug),
        "passages_by_source": {
            k: int(v) for k, v in chunks.source.value_counts().items()
        },
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_ratio": OVERLAP_RATIO,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "source_tokens": tokens,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
