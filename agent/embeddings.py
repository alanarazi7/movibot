"""Semantic search over plot passages.

Retrieval is passage-level, not document-level. Embedding a whole synopsis
covered only its first ~512 tokens, so on long films most of the story was
invisible to search (see agent/chunking.py for the measurements). Synopses are
split into ~300-token passages, each embedded independently, and passage hits
are aggregated back to films so callers still reason about movies.

Two backends, chosen by the MOVIBOT_EMBEDDINGS env var:

    local  (default)  E5-small-v2 run on this machine against the passage
                      matrix in data_ready/chunk_embeddings.npy
    cloud             LLMod.ai text-embedding-3-small + Pinecone

Both return the same shape. The local backend costs nothing and needs no
credentials, so the semantic path is exercisable before any budget is spent.

E5 requires asymmetric prefixes: passages were embedded as "passage: ..." at
index time, so queries must be encoded as "query: ...". Dropping the prefix
measurably degrades recall, so it is applied here rather than left to callers.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_READY = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready")

CHUNKS_PARQUET = os.path.join(_DATA_READY, "plot_chunks.parquet")
CHUNK_VECTORS = os.path.join(_DATA_READY, "chunk_embeddings.npy")
CHUNK_META = os.path.join(_DATA_READY, "chunk_index_meta.json")

LOCAL_MODEL = "intfloat/e5-small-v2"
CLOUD_MODEL = "MB5R2CF-text-embedding-3-small"

# One film can legitimately own several of the strongest passages, so fetching
# exactly top_k passages could yield far fewer than top_k films. Over-fetch,
# then aggregate. Mirrors the fetch_k = top_k x 3 pattern used in the sibling
# medium-rag project, widened slightly because passages here are denser.
FETCH_MULTIPLIER = 5


def backend() -> str:
    """Which embedding backend is active: 'local' or 'cloud'."""
    return os.environ.get("MOVIBOT_EMBEDDINGS", "local").strip().lower()


# ---------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _local_index():
    """(normalised vectors, chunks dataframe). Built by scripts/build_chunk_index.py."""
    import pandas as pd

    if not os.path.exists(CHUNK_VECTORS):
        raise RuntimeError(
            "No passage index found. Build it with:\n"
            "    python scripts/build_chunk_index.py\n"
            "It is free and runs offline."
        )

    vectors = np.load(CHUNK_VECTORS).astype("float32")
    chunks = pd.read_parquet(CHUNKS_PARQUET)

    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"Passage index is inconsistent: {vectors.shape[0]} vectors but "
            f"{len(chunks)} chunk rows. Rebuild both together."
        )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms, chunks


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(LOCAL_MODEL)


def _passages_local(query: str, fetch: int, allowed: set[int] | None) -> list[dict[str, Any]]:
    matrix, chunks = _local_index()

    vector = _local_model().encode(
        f"query: {query}", normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")

    scores = matrix @ vector

    # Mask out ineligible films before ranking, so a restricted search still
    # returns `fetch` usable passages instead of whatever survives filtering.
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


# ---------------------------------------------------------------------
# Cloud backend
# ---------------------------------------------------------------------

def _passages_cloud(query: str, fetch: int, allowed: set[int] | None) -> list[dict[str, Any]]:
    from openai import OpenAI
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOVIBOT_EMBEDDINGS=cloud requires PINECONE_API_KEY. Unset "
            "MOVIBOT_EMBEDDINGS to use the local backend instead."
        )

    openai_client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    vector = openai_client.embeddings.create(
        model=os.environ.get("MOVIBOT_EMBEDDING_MODEL", CLOUD_MODEL),
        input=query,
    ).data[0].embedding

    index = Pinecone(api_key=api_key).Index(
        os.environ.get("PINECONE_INDEX_NAME", "movibot-plots")
    )

    kwargs: dict[str, Any] = {
        "vector": vector, "top_k": fetch, "include_metadata": True
    }
    if allowed:
        kwargs["filter"] = {"movie_id": {"$in": sorted(allowed)}}

    matches = index.query(**kwargs).get("matches", [])
    return [
        {
            "movie_id": int(m["metadata"]["movie_id"]),
            "chunk_index": int(m["metadata"].get("chunk_index", 0)),
            "text": str(m["metadata"].get("text", "")),
            "score": float(m["score"]),
        }
        for m in matches
    ]


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------

def search_passages(
    query: str, top_k: int = 20, candidate_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Rank individual passages, best first. Used when the caller wants
    evidence text rather than a film ranking."""
    if not query or not query.strip():
        return []

    allowed = set(int(c) for c in candidate_ids) if candidate_ids else None
    if allowed is not None and not allowed:
        return []

    fn = _passages_cloud if backend() == "cloud" else _passages_local
    return fn(query.strip(), top_k, allowed)


def search(
    query: str, top_k: int = 10, candidate_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Rank films by their single best-matching passage, best first.

    Scoring a film by its strongest passage rather than its average is what
    makes a specific beat findable: a 28,000-character story whose betrayal
    occupies one passage should rank on that passage, not be diluted by the
    thirty others that are about something else.

    Returns [{movie_id, score, passage, chunk_index, passage_count}].
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

    ranked = sorted(best.values(), key=lambda r: -r["score"])
    return ranked[:top_k]


def coverage() -> dict[str, Any]:
    """Index size and chunk parameters. Local backend only."""
    if backend() == "cloud":
        return {"backend": "cloud", "chunks": "unknown (server-side)"}

    matrix, chunks = _local_index()
    meta = {}
    if os.path.exists(CHUNK_META):
        with open(CHUNK_META, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "backend": "local",
        "model": LOCAL_MODEL,
        "chunks": int(matrix.shape[0]),
        "movies": int(chunks.movie_id.nunique()),
        "dim": int(matrix.shape[1]),
        "chunk_tokens": meta.get("chunk_tokens"),
        "overlap_ratio": meta.get("overlap_ratio"),
    }
