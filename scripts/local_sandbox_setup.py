#!/usr/bin/env python3
"""Initialize local embedding cache for PlotSearch sandbox.

Precomputes E5-small-v2 embeddings for all 170 movies with plot synopses.
Outputs:
  - data_preprocessing/data_ready/plot_embeddings.npy (embedding matrix)
  - data_preprocessing/data_ready/plot_embeddings_mapping.json (movie_id ordering)

One-time setup; regenerable if cache is deleted. Downloads model weights once
from HuggingFace Hub (~130 MB), cached locally under ~/.cache/huggingface.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Build local E5 embedding cache for PlotSearch"
    )
    parser.add_argument(
        "--input",
        default="data_preprocessing/data_ready/pinecone_candidates.csv",
        help="Path to pinecone_candidates.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="data_preprocessing/data_ready",
        help="Output directory for cache files"
    )
    parser.add_argument(
        "--model",
        default="intfloat/e5-small-v2",
        help="Embedding model from HuggingFace"
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Found {len(df)} movies")

    print(f"\nLoading embedding model {args.model}...")
    print("  (This downloads ~130 MB on first run, cached under ~/.cache/huggingface)")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n❌ sentence-transformers not installed.")
        print("   Run: pip install -r requirements-local.txt")
        exit(1)

    model = SentenceTransformer(args.model)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Model dimension: {embedding_dim}")

    print(f"\nEmbedding {len(df)} movies (with 'passage: ' prefix for E5)...")
    # E5 requires "passage: " prefix for documents being indexed
    passages = [f"passage: {text}" for text in df["embedding_text"]]
    embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)

    # L2-normalize embeddings for cosine similarity (dot product after L2 norm = cosine)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  L2-normalized for cosine similarity")

    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "plot_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"\n✅ Saved embeddings to {embeddings_path}")

    # Save mapping: order of embeddings → movie_id
    movie_ids = df["movie_id"].tolist()
    mapping = {
        "model": args.model,
        "embedding_dim": int(embedding_dim),
        "num_movies": len(movie_ids),
        "movie_ids": movie_ids
    }
    mapping_path = output_dir / "plot_embeddings_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved mapping to {mapping_path}")

    print(f"\n✅ Done! Local sandbox ready.")
    print(f"\n   Set environment variable to enable embedding backend:")
    print(f"   export PLOT_SEARCH_BACKEND=embedding")
    print(f"\n   Then run the agent:")
    print(f"   python app.py")


if __name__ == "__main__":
    main()
