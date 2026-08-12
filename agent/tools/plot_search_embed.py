"""PlotSearch tool with local E5 embeddings backend.

Real semantic search via E5-small-v2 embeddings (384-dim) computed locally.
Matches the Pinecone production contract exactly, but runs entirely offline.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Any

import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready", "pinecone_candidates.csv")
_CACHE_DIR = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready")
_EMBEDDINGS_NPY = os.path.join(_CACHE_DIR, "plot_embeddings.npy")
_MAPPING_JSON = os.path.join(_CACHE_DIR, "plot_embeddings_mapping.json")

_embeddings_cache = None
_movie_ids_cache = None
_model_cache = None


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer("intfloat/e5-small-v2")
    return _model_cache


def _load_cache():
    """Load pre-computed embeddings and movie ID mapping."""
    global _embeddings_cache, _movie_ids_cache
    if _embeddings_cache is not None:
        return

    if not os.path.exists(_EMBEDDINGS_NPY) or not os.path.exists(_MAPPING_JSON):
        raise FileNotFoundError(
            f"Embedding cache not found. Run: python scripts/local_sandbox_setup.py\n"
            f"  Expected files:\n"
            f"    {_EMBEDDINGS_NPY}\n"
            f"    {_MAPPING_JSON}"
        )

    _embeddings_cache = np.load(_EMBEDDINGS_NPY)
    with open(_MAPPING_JSON, "r") as f:
        mapping = json.load(f)
        _movie_ids_cache = mapping["movie_ids"]


def run(
    query: str,
    top_k: int = 10,
    candidate_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Score candidates using E5 cosine similarity (production-like behavior).

    Args:
        query: search query string.
        top_k: return top K results (default 10).
        candidate_ids: if provided, restrict scoring to these movie IDs.

    Returns:
        List of {"movie_id", "title", "release_year", "score", "matched_terms"}.
    """
    if not query.strip():
        return []

    _load_cache()
    model = _get_model()

    # Load CSV to get title/year metadata
    df = pd.read_csv(_CSV_PATH)

    # E5 requires "query: " prefix for queries
    query_embedding = model.encode(f"query: {query}", convert_to_numpy=True)

    # L2-normalize query
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Compute cosine similarity: embeddings are already L2-normalized in cache
    scores = np.dot(_embeddings_cache, query_embedding)

    # Build results
    results = []
    for idx, score in enumerate(scores):
        movie_id = _movie_ids_cache[idx]

        # If candidate_ids filter is provided, skip movies not in the list
        if candidate_ids and movie_id not in candidate_ids:
            continue

        # Get movie metadata
        movie_row = df[df["movie_id"] == movie_id]
        if movie_row.empty:
            continue

        row = movie_row.iloc[0]
        results.append({
            "movie_id": int(movie_id),
            "title": row["title"],
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "score": float(score),
            "matched_terms": []  # Embeddings don't produce discrete terms
        })

    # Sort by score descending and return top-k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
