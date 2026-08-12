"""PlotSearch tool - semantic search over plot text via Pinecone."""

import os
import json
from typing import Any

import pandas as pd
from pinecone import Pinecone

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready", "pinecone_candidates.csv")

_candidates_cache = None
_pinecone_index = None


def _get_pinecone_index():
    """Get or initialize Pinecone index."""
    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME", "movibot-plots")

        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        pc = Pinecone(api_key=api_key)
        _pinecone_index = pc.Index(index_name)

    return _pinecone_index


def _load_candidates():
    """Load candidates CSV for metadata (title, year)."""
    global _candidates_cache
    if _candidates_cache is not None:
        return
    _candidates_cache = pd.read_csv(_CSV_PATH)


def run(
    query: str,
    top_k: int = 10,
    candidate_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Search plot embeddings via Pinecone cosine similarity.

    Args:
        query: search query string.
        top_k: return top K results (default 10).
        candidate_ids: if provided, restrict scoring to these movie IDs (via Pinecone filter).

    Returns:
        List of {"movie_id", "title", "release_year", "score", "matched_terms"}.
    """
    if not query.strip():
        return []

    _load_candidates()
    index = _get_pinecone_index()

    # Embed query using LLMod.ai text-embedding-3-small
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    query_embedding = client.embeddings.create(
        model="MB5R2CF-azure/text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Search Pinecone
    if candidate_ids:
        # Filter to candidate_ids only
        filter_dict = {"movie_id": {"$in": candidate_ids}}
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
    else:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

    # Transform Pinecone results to match contract
    matches = []
    for match in results.get("matches", []):
        movie_id = match["metadata"]["movie_id"]
        movie_row = _candidates_cache[_candidates_cache["movie_id"] == movie_id]

        if movie_row.empty:
            continue

        row = movie_row.iloc[0]
        matches.append({
            "movie_id": int(movie_id),
            "title": row["title"],
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "score": float(match["score"]),
            "matched_terms": []  # Pinecone doesn't provide discrete terms
        })

    return matches
