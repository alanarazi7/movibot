"""ExternalContext tool - subtext/tone/reception not covered by catalog or plot.

TODO (next pass):
  1. Fetch relevant Wikipedia page sections beyond "Plot" (e.g. Reception,
     Themes) live, only for candidates that still need this check.
  2. Use EXTERNAL_CONTEXT_SYSTEM_PROMPT (agent/prompts.py) to reason over
     that text for the specific constraint being checked.
  3. Return {title, constraint, satisfied, evidence} for the Reasoner/Synthesizer.

Only called when Reasoner decides SceneSearch's plot text isn't enough to
resolve a constraint. Not called anywhere yet - app.py's /api/execute is
fully stubbed.
"""

from typing import Any


def run(title: str, constraint: str) -> dict[str, Any]:
    raise NotImplementedError("ExternalContext is not wired up yet - skeleton pass only.")
