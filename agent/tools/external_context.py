"""ExternalContext tool - tone/reception/themes not covered by plot.

Mock: cached Wikipedia pages (pre-scraped offline) + keyword heuristic.
Real (Chunk 4): LLM reasoning over cached text per EXTERNAL_CONTEXT_SYSTEM_PROMPT.
"""

import os
from typing import Any
from pathlib import Path
import pandas as pd

from agent import llm_client
from agent.tools import wikipedia_client

_BASE_DIR = Path(__file__).parent.parent.parent
_WIKI_CACHE_PATH = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready", "wikipedia_cache.csv")
_wiki_cache = None


def _load_wiki_cache():
    """Load Wikipedia cache CSV once."""
    global _wiki_cache
    if _wiki_cache is not None:
        return
    if os.path.exists(_WIKI_CACHE_PATH):
        _wiki_cache = pd.read_csv(_WIKI_CACHE_PATH)
    else:
        _wiki_cache = pd.DataFrame()


def run(title: str, constraint: str) -> dict[str, Any]:
    """Verify a per-candidate constraint against tone/reception/themes."""
    # Try cached Wikipedia first
    _load_wiki_cache()
    text = None

    if not _wiki_cache.empty:
        wiki_row = _wiki_cache[
            _wiki_cache["title"].str.lower() == title.lower()
        ]
        if not wiki_row.empty and pd.notna(wiki_row.iloc[0].get("non_plot_text")):
            text = wiki_row.iloc[0]["non_plot_text"]

    # Fall back to live Wikipedia if cache unavailable
    if not text:
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
