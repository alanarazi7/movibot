"""Where the passage vectors live, and how they are searched.

In memory: a committed .npy, loaded once and scored with a numpy dot product.
No vector database, deliberately -- see rag/config.py for the arithmetic. The
short version is that 3,159 vectors score in about 0.5 ms, faster than the
network hop a hosted index would add, and the query has to be embedded either
way. A database would buy nothing here and cost a credential.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np

from rag import embed
from rag.config import (
    EMBED_MODEL,
    FETCH_MULTIPLIER,
    INDEX_META,
    INDEX_NPZ,
    TOP_K,
)


@lru_cache(maxsize=1)
def _matrix():
    """(vectors, passages) from the committed archive.

    `passages` is a plain list of dicts. There is no DataFrame here: the only
    operations are a membership test and an index lookup, both of which numpy
    and a list do without pulling pandas into the request path.
    """
    if not os.path.exists(INDEX_NPZ):
        raise RuntimeError(
            "No passage index found. Build it with:\n"
            "    python -m rag.ingest"
        )

    blob = np.load(INDEX_NPZ, allow_pickle=False)
    vectors = blob["vectors"].astype("float32")
    passages = json.loads(str(blob["table"].item()))

    if vectors.shape[0] != len(passages):
        raise RuntimeError(
            f"Passage index is inconsistent: {vectors.shape[0]} vectors but "
            f"{len(passages)} passages. Rebuild with `python -m rag.ingest`."
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

    return vectors, passages


def _meta() -> dict[str, Any]:
    if not os.path.exists(INDEX_META):
        return {}
    with open(INDEX_META, "r", encoding="utf-8") as f:
        return json.load(f)


def _passages_matrix(query: str, fetch: int, allowed: set[int] | None,
                     sources: list[str] | None = None) -> list[dict]:
    vectors, passages = _matrix()
    scores = vectors @ embed.embed_query(query)

    # Mask ineligible passages before ranking, so a restricted search still
    # returns `fetch` usable passages rather than whatever survives filtering.
    eligible = np.ones(len(passages), dtype=bool)
    if allowed is not None:
        eligible &= np.array([p["movie_id"] in allowed for p in passages])
    if sources is not None:
        wanted = set(sources)
        eligible &= np.array([p.get("source", "mpst") in wanted for p in passages])
    if not eligible.all():
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
        row = passages[int(i)]
        out.append({
            "movie_id": int(row["movie_id"]),
            "chunk_index": int(row["chunk_index"]),
            "text": str(row["text"]),
            "source": str(row.get("source", "mpst")),
            "title": str(row.get("title", "")),
            "score": float(scores[int(i)]),
        })
    return out


def search_passages(
    query: str,
    top_k: int = 20,
    candidate_ids: list[int] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Rank individual passages, best first, for callers that want evidence.

    `sources` restricts the search to particular corpora; None searches all of
    them. Keeping them separable matters because they answer different
    questions -- a plot says whether someone dies, a reception section says
    whether a film is thought frightening.
    """
    if not query or not query.strip():
        return []

    allowed = set(int(c) for c in candidate_ids) if candidate_ids else None
    if allowed is not None and not allowed:
        return []

    return _passages_matrix(query.strip(), top_k, allowed, sources)


def search(
    query: str,
    top_k: int = TOP_K,
    candidate_ids: list[int] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Rank films by their single best-matching passage, best first.

    Scoring a film by its strongest passage rather than its average is what
    makes a specific beat findable: a 28,000-character story whose betrayal
    occupies one passage should rank on that passage, not be diluted by the
    thirty others that are about something else.
    """
    fetch = max(top_k * FETCH_MULTIPLIER, top_k)
    passages = search_passages(query, top_k=fetch, candidate_ids=candidate_ids,
                               sources=sources)

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
                "source": p.get("source", "mpst"),
                "passage_count": 1,
            }
        else:
            existing["passage_count"] += 1
            if p["score"] > existing["score"]:
                existing.update(
                    score=p["score"], passage=p["text"],
                    chunk_index=p["chunk_index"], source=p.get("source", "mpst"),
                )

    return sorted(best.values(), key=lambda r: -r["score"])[:top_k]


def coverage() -> dict[str, Any]:
    """Index size and parameters, for diagnostics and /api/agent_info."""
    vectors, passages = _matrix()
    meta = _meta()
    return {
        "store": "in-memory matrix",
        "model": meta.get("model", EMBED_MODEL),
        "chunks": int(vectors.shape[0]),
        "movies": len({p["movie_id"] for p in passages}),
        "dim": int(vectors.shape[1]),
        "chunk_tokens": meta.get("chunk_tokens"),
        "overlap_ratio": meta.get("overlap_ratio"),
    }
