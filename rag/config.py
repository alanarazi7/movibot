"""Every knob the retrieval pipeline has, in one place.

These are the numbers a reader should be able to find without grepping, and the
ones ingest and query time must agree on -- a matrix embedded with one chunk
size and searched with another is silently wrong rather than broken.

The rationale for each value is in rag/DECISIONS.md. The short version is that
the course's default for long-form prose is 512-1024 tokens at 5-15% overlap,
and this corpus deviates from it deliberately: the retrieval unit here is a
story beat, not a section.
"""

from __future__ import annotations

import os

# --- Embedding -------------------------------------------------------------
# One model everywhere. There is no local model: an earlier design ran
# E5-small-v2 on the dev machine and text-embedding-3-small in production,
# which meant local testing never exercised the vectors production would use.
EMBED_MODEL = os.environ.get(
    "MOVIBOT_EMBEDDING_MODEL", "MB5R2CF-azure/text-embedding-3-small"
)
EMBED_DIM = 1536

# Passages per embedding request at ingest time. The limit is request size
# rather than anything semantic; 96 keeps a batch comfortably inside it.
EMBED_BATCH = 96

# --- Chunking --------------------------------------------------------------
# Sentence-boundary, not paragraph: zero of the 159 synopses contain blank-line
# paragraphs and 66 have no newline at all, so a paragraph splitter emits one
# chunk per document. Measured, not assumed -- see DECISIONS.md.
CHUNK_TOKENS = 300
OVERLAP_RATIO = 0.2
MIN_CHUNK_TOKENS = 50

# text-embedding-3-small tokenises with cl100k_base, so these counts are the
# model's own units. Under E5 they were BERT tokens counted with an OpenAI
# tokeniser -- close, but not the same thing.
TOKENIZER = "cl100k_base"

# --- Retrieval -------------------------------------------------------------
# Films returned by a search. The course suggests k = 3-5 for general text;
# this is higher because a film here is one line of evidence, not a document
# pasted into the context window.
TOP_K = 10

# One film can own several of the strongest passages, so fetching exactly
# top_k passages could yield far fewer than top_k films. Over-fetch, then
# aggregate to best-passage-per-film.
FETCH_MULTIPLIER = 5

# --- Where the vectors live ------------------------------------------------
# "matrix"   a committed .npy read from disk and scored with numpy. 1,254
#            vectors score in well under a millisecond, so this is not a
#            compromise at this scale.
# "pinecone" the same vectors, served by Pinecone.
#
# Note this selects the *store*, not the model: both paths embed with
# EMBED_MODEL. MOVIBOT_EMBEDDINGS is still read for compatibility, where its
# old values map local -> matrix and cloud -> pinecone.
_STORE_ALIASES = {"local": "matrix", "cloud": "pinecone"}


def vector_store() -> str:
    raw = (
        os.environ.get("MOVIBOT_VECTOR_STORE")
        or os.environ.get("MOVIBOT_EMBEDDINGS")
        or "matrix"
    ).strip().lower()
    return _STORE_ALIASES.get(raw, raw)


PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "movibot-plots")

# --- Artifacts -------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_READY = os.path.join(_ROOT, "data_preprocessing", "data_ready")

SOURCE_CSV = os.path.join(DATA_READY, "pinecone_candidates.csv")
CHUNKS_PARQUET = os.path.join(DATA_READY, "plot_chunks.parquet")
VECTORS_NPY = os.path.join(DATA_READY, "chunk_embeddings.npy")
INDEX_META = os.path.join(DATA_READY, "chunk_index_meta.json")


def as_dict() -> dict:
    """The parameters, for /api/agent_info and the Status page.

    Exposing them mirrors the sibling medium-rag project, which reports
    chunk_size/overlap_ratio/top_k from /api/stats.
    """
    return {
        "embedding_model": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_ratio": OVERLAP_RATIO,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "top_k": TOP_K,
        "fetch_multiplier": FETCH_MULTIPLIER,
        "vector_store": vector_store(),
    }
