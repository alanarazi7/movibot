"""Turns text into vectors, via the one embedding model this project uses.

Both the ingest path and the query path come through here, so a passage and a
query can never be embedded by different models -- which is exactly the bug the
previous design had, where dev embedded with E5 and production with OpenAI.

Costs are small but not zero: roughly $0.0075 to embed the whole corpus once,
and ~$0.000002 for a query. That is the price of deleting a 518 MB dependency
and the local/production divergence that came with it.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from rag.config import EMBED_BATCH, EMBED_DIM, EMBED_MODEL


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    if not key or "your-" in key:
        raise RuntimeError(
            "Embedding requires OPENAI_API_KEY. Run "
            "`python scripts/check_credentials.py` to see what is missing."
        )
    return OpenAI(api_key=key, base_url=base or None)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def embed_texts(texts: list[str], progress: bool = False) -> np.ndarray:
    """Embed many texts. Returns a normalised (len(texts), EMBED_DIM) matrix."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        resp = _client().embeddings.create(model=EMBED_MODEL, input=batch)
        # The API does not guarantee response order, and silently mismatched
        # vectors would be undetectable later, so sort by the index it returns.
        for item in sorted(resp.data, key=lambda d: d.index):
            out.append(item.embedding)
        if progress:
            print(f"  embedded {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")

    matrix = np.asarray(out, dtype="float32")
    if matrix.shape[1] != EMBED_DIM:
        raise RuntimeError(
            f"{EMBED_MODEL} returned {matrix.shape[1]}-dim vectors, expected "
            f"{EMBED_DIM}. Update EMBED_DIM in rag/config.py."
        )
    return _normalise(matrix)


def embed_query(query: str) -> np.ndarray:
    """One query vector, normalised. Costs about two millionths of a dollar."""
    return embed_texts([query])[0]
