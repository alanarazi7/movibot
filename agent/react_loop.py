"""Orchestrates the ReAct loop: Reason -> Act -> Observe -> Stop? -> Synthesizer.

TODO (next pass, after skeleton review):
  1. Call the Reasoner (llm_client + REASONER_SYSTEM_PROMPT) each iteration to
     pick the next tool (agent/tools/*) or to stop.
  2. Append one steps[] entry per LLM/tool call, in order, using the exact
     required schema: {"module": ..., "prompt": {"system_prompt", "user_prompt"},
     "response": ...}. Module names must match assets/architecture.png exactly.
  3. Cap loop iterations to respect the $13 budget and Vercel's 300s limit.
  4. Call Synthesizer (SYNTHESIZER_SYSTEM_PROMPT) once Stop is decided, and
     return {status, error, response, steps} in the exact required shape.

Not called anywhere yet - app.py's /api/execute currently returns a
hardcoded stub response instead of calling this module.
"""

from typing import Any


def execute(prompt: str) -> dict[str, Any]:
    raise NotImplementedError("react_loop.execute is not wired up yet - skeleton pass only.")
