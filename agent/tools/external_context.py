"""ExternalContext tool - tone/reception/themes not covered by plot.

Mock: live Wikipedia fetch for non-Plot sections + keyword heuristic.
Real (Chunk 4): LLM reasoning over fetched text per EXTERNAL_CONTEXT_SYSTEM_PROMPT.
"""

from typing import Any

from agent import llm_client
from agent.tools import wikipedia_client


def run(title: str, constraint: str) -> dict[str, Any]:
    """Verify a per-candidate constraint against tone/reception/themes."""
    text = wikipedia_client.get_non_plot_text(title)

    if not text:
        return {
            "title": title,
            "constraint": constraint,
            "satisfied": None,
            "evidence": "No supplementary Wikipedia text (Reception, Themes, etc.) could be found for this title."
        }

    # Use mock LLM client to verify
    client = llm_client.get_mock_client()
    return client.verify_external_constraint(title, constraint, text)
