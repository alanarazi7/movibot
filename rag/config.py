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

# Not a discard threshold: a trailing passage shorter than this is folded into
# the previous one rather than dropped. Nothing is ever thrown away, so the
# final passage of a synopsis may run up to this much over CHUNK_TOKENS.
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
# In memory, in a committed .npy scored with numpy. There is no vector database
# and no switch to enable one.
#
# The reasoning is just arithmetic: 3,159 vectors is not a search problem. A
# brute-force dot product over all of them takes about 0.5 ms, which is faster
# than the network round trip a hosted index would add -- and the query still
# has to be embedded either way, so a database saves nothing there. What it
# would add is a credential, an account, a quota, and a failure mode, in
# exchange for making retrieval slower and stopping a fresh clone from working.
#
# A vector database earns its place when the index no longer fits in memory or
# when brute force stops being instant. Neither is remotely true here: the
# matrix is 19 MB. If this ever grew past a demo catalog, rag/store.py is the
# only file that would need to change.

# --- Artifacts -------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_READY = os.path.join(_ROOT, "data_preprocessing", "data_ready")

# Named from an early assumption that a vector DB would ingest it. Kept as
# the filename to avoid churning the prepared data; it is just the films
# that matched an MPST synopsis.
SOURCE_CSV = os.path.join(DATA_READY, "pinecone_candidates.csv")
# Vectors and passage table in one compressed .npz. Previously a .npy plus a
# parquet, which pulled in pyarrow -- 124 MB, half of Vercel's serverless
# limit, to read a 3,159-row table. The table travels as JSON inside the
# archive, so nothing needs a columnar engine and nothing needs pickle.
INDEX_NPZ = os.path.join(DATA_READY, "chunk_index.npz")
INDEX_META = os.path.join(DATA_READY, "chunk_index_meta.json")

# Local accelerator, gitignored: text -> vector, so re-running ingest never
# pays twice for a passage that has not changed.
EMBED_CACHE = os.path.join(DATA_READY, "embedding_cache.npz")

# --debug ingests this many passages per corpus -- enough to exercise the whole
# pipeline (chunk, embed, store, search) for a fraction of a cent.
DEBUG_CHUNKS_PER_CORPUS = 10


def as_dict() -> dict:
    """The parameters, for /api/agent_info and the Status page.

    Exposed rather than buried so the retrieval parameters an answer depended
    on are inspectable from the API, not only from the source.
    """
    return {
        "embedding_model": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_ratio": OVERLAP_RATIO,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "top_k": TOP_K,
        "fetch_multiplier": FETCH_MULTIPLIER,
        "vector_store": "in-memory matrix",
    }
