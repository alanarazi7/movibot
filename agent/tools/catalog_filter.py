"""CatalogFilter tool - structured filtering (year, studio, genre, runtime) via Supabase.

TODO (next pass, blocked on Alan creating the Supabase project):
  1. Read SUPABASE_URL / SUPABASE_KEY from env (see .env.example).
  2. Use the CATALOG_FILTER_SYSTEM_PROMPT (agent/prompts.py) to translate the
     structured part of the user's request into filter args.
  3. Query the `movies` table (see scripts/ingest.py for schema) with those args.
  4. Return matching rows for the Reasoner to pass to PlotSearch/SceneSearch.

Not called anywhere yet - app.py's /api/execute is fully stubbed.
"""

from typing import Any


def run(constraints: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("CatalogFilter is not wired up yet - skeleton pass only.")
