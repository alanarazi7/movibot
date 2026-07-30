"""PlotSearch tool - semantic search over TMDB `overview` text via Pinecone.

TODO (next pass, after ingestion is approved to run):
  1. Embed the query with EMBEDDING_MODEL (agent/llm_client.py).
  2. Query the Pinecone index populated by scripts/ingest.py.
  3. Return top matches (movie_id, title, release_year, score) for the
     Reasoner to narrow down further with SceneSearch/ExternalContext.

Not called anywhere yet - app.py's /api/execute is fully stubbed.
"""

from typing import Any


def run(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    raise NotImplementedError("PlotSearch is not wired up yet - skeleton pass only.")
