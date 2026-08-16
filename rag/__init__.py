"""Retrieval: chunking, embedding, storage and search.

Everything the passage index needs lives here -- the parameters (config.py),
how synopses are cut up (chunking.py), how text becomes vectors (embed.py),
where those vectors live and how they are searched (store.py), and the one
command that builds it all (ingest.py). The reasoning behind the numbers is in
DECISIONS.md.

Callers should need only:

    from rag import search, search_passages
"""

from rag.chunking import chunk_movie, chunk_text
from rag.config import as_dict as parameters
from rag.store import coverage, search, search_passages

__all__ = [
    "chunk_movie",
    "chunk_text",
    "coverage",
    "parameters",
    "search",
    "search_passages",
]
