"""Downsamples Kaggle "The Movies Dataset" into Supabase (structured fields)
and Pinecone (embedded `overview` text), matching medium-rag-hw/scripts/ingest.py's
pattern of a --limit flag for cheap test runs before the full ~5K-movie ingest.

TODO (next pass, blocked on: Alan creating the Supabase project, and explicit
go-ahead to spend LLMod.ai budget on embeddings):
  1. Load movies_metadata.csv (+ optionally keywords.csv/credits.csv).
  2. Downsample to ~5K movies per the team's assignment doc.
  3. Write structured fields (title, release_year, runtime_minutes, genres,
     production_companies, popularity, overview) into the Supabase `movies` table.
  4. Embed `overview` with EMBEDDING_MODEL (agent/llm_client.py) and upsert into
     Pinecone with metadata {movie_id, title, release_year}.

Not executed yet - this is skeleton-only, no LLMod.ai/Pinecone/Supabase calls
have been made.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to movies_metadata.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N rows (for a cheap test run before the full ingest)",
    )
    parser.parse_args()

    raise NotImplementedError("Ingestion is not wired up yet - skeleton pass only.")


if __name__ == "__main__":
    main()
