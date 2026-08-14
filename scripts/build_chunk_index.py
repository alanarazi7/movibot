#!/usr/bin/env python3
"""Builds the local passage-level embedding index.

Chunks every plot synopsis, embeds each passage with E5-small-v2 on this
machine, and writes the vectors plus passage metadata to data_ready/. Free and
offline -- no API calls, no budget.

This replaces the earlier document-level index, which embedded only the first
~512 tokens of each synopsis and so covered ~7% of the longest films.

    python scripts/build_chunk_index.py            # full rebuild
    python scripts/build_chunk_index.py --limit 20 # quick smoke test

Outputs (committed, so the repo runs without a rebuild; all regenerable):
    plot_chunks.parquet        one row per passage, with its text
    chunk_embeddings.npy       float32 matrix, one row per passage
    chunk_index_meta.json      model, dims, chunk parameters, counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import chunking  # noqa: E402

_DATA_READY = Path(__file__).resolve().parent.parent / "data_preprocessing" / "data_ready"
SOURCE_CSV = _DATA_READY / "pinecone_candidates.csv"
CHUNKS_OUT = _DATA_READY / "plot_chunks.parquet"
VECTORS_OUT = _DATA_READY / "chunk_embeddings.npy"
META_OUT = _DATA_READY / "chunk_index_meta.json"

MODEL = "intfloat/e5-small-v2"
BATCH = 64


def build_chunks(limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    if limit:
        df = df.head(limit)

    records = []
    for _, row in df.iterrows():
        text = row.get("plot_synopsis")
        if not isinstance(text, str) or not text.strip():
            continue
        records.extend(
            chunking.chunk_movie(int(row["movie_id"]), str(row["title"]), text)
        )

    if not records:
        raise SystemExit("No chunks produced -- is pinecone_candidates.csv populated?")
    return pd.DataFrame(records)


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    # "passage:" is E5's document-side prefix; queries use "query:" at search
    # time. Mismatching the two measurably degrades recall.
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Only process the first N films.")
    args = parser.parse_args()

    print(f"Reading {SOURCE_CSV.name} ...")
    chunks = build_chunks(args.limit)

    films = chunks.movie_id.nunique()
    print(
        f"Chunked {films} films into {len(chunks):,} passages "
        f"({len(chunks)/films:.1f} per film, "
        f"{chunks.tokens.median():.0f} median tokens)"
    )

    print(f"Embedding with {MODEL} (local, free) ...")
    vectors = embed(chunks.embedding_text.tolist())

    if vectors.shape[0] != len(chunks):
        raise SystemExit(
            f"Vector/chunk mismatch: {vectors.shape[0]} vs {len(chunks)}"
        )

    chunks.to_parquet(CHUNKS_OUT, index=False)
    np.save(VECTORS_OUT, vectors)
    META_OUT.write_text(json.dumps({
        "model": MODEL,
        "embedding_dim": int(vectors.shape[1]),
        "num_chunks": int(len(chunks)),
        "num_movies": int(films),
        "chunk_tokens": chunking.CHUNK_TOKENS,
        "overlap_ratio": chunking.OVERLAP_RATIO,
        "min_chunk_tokens": chunking.MIN_CHUNK_TOKENS,
    }, indent=2))

    print(f"\nWrote {CHUNKS_OUT.name}, {VECTORS_OUT.name} {vectors.shape}, {META_OUT.name}")
    for path in (CHUNKS_OUT, VECTORS_OUT):
        print(f"  {path.name}: {path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
