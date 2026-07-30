"""SceneSearch tool - per-candidate deep check for risky/event-level constraints.

TODO (next pass):
  1. Fetch the movie's Wikipedia page "Plot" section live (no pre-indexing -
     see the plan's open decision on live-fetch vs pre-indexed).
  2. Use SCENE_SEARCH_SYSTEM_PROMPT (agent/prompts.py) to reason over that
     text only, per title, for the specific constraint being checked
     (e.g. "does any character die").
  3. Return {title, constraint, satisfied, evidence} for the Reasoner/Synthesizer.

Not called anywhere yet - app.py's /api/execute is fully stubbed.
"""

from typing import Any


def run(title: str, constraint: str) -> dict[str, Any]:
    raise NotImplementedError("SceneSearch is not wired up yet - skeleton pass only.")
