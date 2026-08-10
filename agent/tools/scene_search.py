"""SceneSearch tool - per-candidate check for risky/event-level constraints.

Mock: looks up plot_synopsis from local CSV first, falls back to live Wikipedia.
Real (Chunk 4): LLM reasoning over fetched text per SCENE_SEARCH_SYSTEM_PROMPT.
"""

import os
import json
from typing import Any

import pandas as pd

from agent import llm_client
from agent.tools import wikipedia_client

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready", "pinecone_candidates.csv")

_candidates_cache = None


def _load_candidates():
    """Load candidates CSV (with local plot synopses) once."""
    global _candidates_cache
    if _candidates_cache is not None:
        return
    _candidates_cache = pd.read_csv(_CSV_PATH)


def _get_plot_text(title: str) -> str | None:
    """Get plot text: prefer local plot_synopsis, fall back to live Wikipedia."""
    _load_candidates()

    # Try to find in local candidates by exact title match (case-insensitive)
    row = _candidates_cache[
        _candidates_cache["title"].str.lower() == title.lower()
    ]

    if not row.empty and pd.notna(row.iloc[0].get("plot_synopsis")):
        return row.iloc[0]["plot_synopsis"]

    # Fall back to live Wikipedia
    return wikipedia_client.get_section(title, ["Plot", "Plot summary", "Synopsis"])


def run(title: str, constraint: str) -> dict[str, Any]:
    """Verify a per-candidate risky-event constraint against plot text."""
    plot_text = _get_plot_text(title)

    if not plot_text:
        return {
            "title": title,
            "constraint": constraint,
            "satisfied": None,
            "evidence": "No plot text could be found for this title (no local synopsis and Wikipedia fetch failed)."
        }

    # Use mock LLM client to verify
    client = llm_client.get_mock_client()
    return client.verify_scene_constraint(title, constraint, plot_text)
