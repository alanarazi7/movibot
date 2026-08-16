"""Turns text into vectors, via the one embedding model this project uses.

Both the ingest path and the query path come through here, so a passage and a
query can never be embedded by different models -- which is exactly the bug the
previous design had, where dev embedded with E5 and production with OpenAI.

Costs are small but not zero: roughly $0.0075 to embed the whole corpus once,
and ~$0.000002 for a query. That is the price of deleting a 518 MB dependency
and the local/production divergence that came with it.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np

from rag.config import EMBED_BATCH, EMBED_DIM, EMBED_MODEL, EMBED_CACHE


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


# ---------------------------------------------------------------------
# Content-addressed cache
# ---------------------------------------------------------------------
# Embedding the same text twice is pure waste, and it happens constantly:
# re-running ingest after adding a corpus, or after a chunker change that
# altered only a handful of passages. The cache is keyed by a hash of the model
# id and the exact text, so a changed passage misses and an unchanged one hits,
# and a change of model invalidates everything automatically.
#
# It is a local accelerator, not an artifact: gitignored, and a fresh clone
# simply re-embeds. The committed index is the thing that matters.


def _key(text: str) -> str:
    return hashlib.sha1(f"{EMBED_MODEL}\x00{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, np.ndarray]:
    if not os.path.exists(EMBED_CACHE):
        return {}
    try:
        blob = np.load(EMBED_CACHE, allow_pickle=False)
        return dict(zip(blob["keys"].tolist(), blob["vectors"]))
    except Exception:
        # A corrupt cache must never be fatal: it only ever costs a re-embed.
        return {}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)
    np.savez(
        EMBED_CACHE,
        keys=np.array(list(cache.keys())),
        vectors=np.asarray(list(cache.values()), dtype="float32"),
    )


def embed_texts_cached(texts: list[str], progress: bool = False) -> tuple[np.ndarray, dict]:
    """Embed `texts`, reusing anything already embedded with this model.

    Returns (matrix, stats) where stats reports how many were reused and how
    many were actually sent -- the difference is money not spent.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32"), {"reused": 0, "embedded": 0}

    cache = _load_cache()
    keys = [_key(t) for t in texts]

    # Deduplicate within this batch too: the same passage text can legitimately
    # appear under two corpora, and there is no reason to pay for it twice.
    missing_order: list[str] = []
    missing_text: list[str] = []
    seen: set[str] = set()
    for k, t in zip(keys, texts):
        if k not in cache and k not in seen:
            seen.add(k)
            missing_order.append(k)
            missing_text.append(t)

    if missing_text:
        if progress:
            print(f"  {len(texts) - len(missing_text):,} already embedded, "
                  f"{len(missing_text):,} to send")
        fresh = embed_texts(missing_text, progress=progress)
        for k, vec in zip(missing_order, fresh):
            cache[k] = vec
        _save_cache(cache)
    elif progress:
        print(f"  all {len(texts):,} passages already embedded -- nothing sent")

    matrix = np.asarray([cache[k] for k in keys], dtype="float32")
    return matrix, {"reused": len(texts) - len(missing_text), "embedded": len(missing_text)}
