"""Where the passage vectors live, and how they are searched.

Two stores, selected by MOVIBOT_VECTOR_STORE:

    matrix     (default) a committed .npy scored with numpy. 1,254 vectors
               score in ~0.5 ms, so this is not a compromise at this scale --
               it is faster than a network round trip to a vector database.
    pinecone   the same vectors, served by Pinecone.

The distinction is only *where the vectors are kept*. Both embed the query with
the same model, so the two stores return the same ranking. That was not true of
the previous local/cloud split, where the two paths used different models and
therefore different vectors, and local testing never exercised what production
would do.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np

from rag import embed
from rag.config import (
    CHUNKS_PARQUET,
    EMBED_MODEL,
    FETCH_MULTIPLIER,
    INDEX_META,
    PINECONE_INDEX,
    TOP_K,
    VECTORS_NPY,
    vector_store,
)


@lru_cache(maxsize=1)
def _matrix():
    """(vectors, chunks dataframe), built by `python -m rag.ingest`."""
    import pandas as pd

    if not os.path.exists(VECTORS_NPY):
        raise RuntimeError(
            "No passage index found. Build it with:\n"
            "    python -m rag.ingest"
        )

    vectors = np.load(VECTORS_NPY).astype("float32")
    chunks = pd.read_parquet(CHUNKS_PARQUET)

    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"Passage index is inconsistent: {vectors.shape[0]} vectors but "
            f"{len(chunks)} chunk rows. Rebuild both with `python -m rag.ingest`."
        )

    # A stale index is the dangerous failure: vectors from a different model
    # still score, they just score meaninglessly. Refuse rather than mislead.
    meta = _meta()
    if meta.get("model") and meta["model"] != EMBED_MODEL:
        raise RuntimeError(
            f"Passage index was built with {meta['model']}, but this project "
            f"now embeds with {EMBED_MODEL}. The vectors are not comparable.\n"
            "    Rebuild with: python -m rag.ingest"
        )

    return vectors, chunks


def _meta() -> dict[str, Any]:
    if not os.path.exists(INDEX_META):
        return {}
    with open(INDEX_META, "r", encoding="utf-8") as f:
        return json.load(f)


def _passages_matrix(query: str, fetch: int, allowed: set[int] | None) -> list[dict]:
    vectors, chunks = _matrix()
    scores = vectors @ embed.embed_query(query)

    # Mask ineligible films before ranking, so a restricted search still
    # returns `fetch` usable passages rather than whatever survives filtering.
    if allowed is not None:
        eligible = chunks.movie_id.isin(allowed).to_numpy()
        if not eligible.any():
            return []
        scores = np.where(eligible, scores, -np.inf)

    take = min(fetch, int(np.isfinite(scores).sum()))
    if take <= 0:
        return []

    top = np.argpartition(-scores, take - 1)[:take]
    top = top[np.argsort(-scores[top])]

    out = []
    for i in top:
        row = chunks.iloc[int(i)]
        out.append({
            "movie_id": int(row.movie_id),
            "chunk_index": int(row.chunk_index),
            "text": str(row.text),
            "score": float(scores[int(i)]),
        })
    return out


def _passages_pinecone(query: str, fetch: int, allowed: set[int] | None) -> list[dict]:
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or "your-" in api_key:
        raise RuntimeError(
            "MOVIBOT_VECTOR_STORE=pinecone requires PINECONE_API_KEY. Unset it "
            "to use the committed matrix instead, which needs no credentials."
        )

    index = Pinecone(api_key=api_key).Index(PINECONE_INDEX)
    kwargs: dict[str, Any] = {
        "vector": embed.embed_query(query).tolist(),
        "top_k": fetch,
        "include_metadata": True,
    }
    if allowed:
        kwargs["filter"] = {"movie_id": {"$in": sorted(allowed)}}

    return [
        {
            "movie_id": int(m["metadata"]["movie_id"]),
            "chunk_index": int(m["metadata"].get("chunk_index", 0)),
            "text": str(m["metadata"].get("text", "")),
            "score": float(m["score"]),
        }
        for m in index.query(**kwargs).get("matches", [])
    ]


def search_passages(
    query: str, top_k: int = 20, candidate_ids: list[int] | None = None
) -> list[dict]:
    """Rank individual passages, best first, for callers that want evidence."""
    if not query or not query.strip():
        return []

    allowed = set(int(c) for c in candidate_ids) if candidate_ids else None
    if allowed is not None and not allowed:
        return []

    fn = _passages_pinecone if vector_store() == "pinecone" else _passages_matrix
    return fn(query.strip(), top_k, allowed)


def search(
    query: str, top_k: int = TOP_K, candidate_ids: list[int] | None = None
) -> list[dict]:
    """Rank films by their single best-matching passage, best first.

    Scoring a film by its strongest passage rather than its average is what
    makes a specific beat findable: a 28,000-character story whose betrayal
    occupies one passage should rank on that passage, not be diluted by the
    thirty others that are about something else.
    """
    fetch = max(top_k * FETCH_MULTIPLIER, top_k)
    passages = search_passages(query, top_k=fetch, candidate_ids=candidate_ids)

    best: dict[int, dict[str, Any]] = {}
    for p in passages:
        movie_id = p["movie_id"]
        existing = best.get(movie_id)
        if existing is None:
            best[movie_id] = {
                "movie_id": movie_id,
                "score": p["score"],
                "passage": p["text"],
                "chunk_index": p["chunk_index"],
                "passage_count": 1,
            }
        else:
            existing["passage_count"] += 1
            if p["score"] > existing["score"]:
                existing.update(
                    score=p["score"], passage=p["text"], chunk_index=p["chunk_index"]
                )

    return sorted(best.values(), key=lambda r: -r["score"])[:top_k]


def coverage() -> dict[str, Any]:
    """Index size and parameters, for diagnostics and /api/agent_info."""
    if vector_store() == "pinecone":
        return {"store": "pinecone", "index": PINECONE_INDEX, "model": EMBED_MODEL}

    vectors, chunks = _matrix()
    meta = _meta()
    return {
        "store": "matrix",
        "model": meta.get("model", EMBED_MODEL),
        "chunks": int(vectors.shape[0]),
        "movies": int(chunks.movie_id.nunique()),
        "dim": int(vectors.shape[1]),
        "chunk_tokens": meta.get("chunk_tokens"),
        "overlap_ratio": meta.get("overlap_ratio"),
    }
