"""LLMod.ai client wrapper - OpenAI-compatible, mirrors medium-rag-hw's app.py.

Also provides MockLLMClient for zero-cost mock execution during development.
"""

import os
import re
from typing import Any
from collections import Counter

CHAT_MODEL = "MB5R2CF-azure/gpt-5.4-mini"
EMBEDDING_MODEL = "MB5R2CF-azure/text-embedding-3-small"


def get_client():
    """Returns an OpenAI-compatible client pointed at LLMod.ai.

    Requires OPENAI_API_KEY / OPENAI_BASE_URL to be set (see .env.example).
    """
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )


class MockLLMClient:
    """Mock LLM client using deterministic logic and local CSV data.

    All methods return the exact JSON shape expected by the system prompts,
    so they can be swapped for real LLM calls with identical call sites.
    """

    def __init__(self):
        self._stopwords = frozenset([
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "have", "he", "her", "hers", "him", "his", "how", "i",
            "if", "in", "into", "is", "it", "its", "me", "my", "myself",
            "no", "nor", "not", "of", "on", "or", "other", "our", "ours",
            "out", "over", "own", "same", "she", "should", "so", "some",
            "such", "than", "that", "the", "their", "theirs", "them",
            "then", "there", "these", "they", "this", "those", "to",
            "too", "under", "until", "up", "very", "was", "we", "were",
            "what", "when", "where", "which", "while", "who", "whom",
            "why", "will", "with", "you", "your", "yours", "yourself"
        ])
        self._death_keywords = frozenset([
            "death", "die", "dies", "died", "dies", "killing", "killed",
            "murder", "killed", "dies", "dead", "perish", "perishes",
            "passed away", "passes away", "no longer", "succumb"
        ])
        self._scary_keywords = frozenset([
            "scary", "scared", "frighten", "frightened", "horror", "horrify",
            "terrify", "terrified", "terrifying", "nightmare", "nightmares",
            "creepy", "spooky", "eerie", "sinister", "ominous", "dark"
        ])

    def reason_next_action(
        self, user_prompt: str, gathered: dict[str, Any]
    ) -> dict[str, Any]:
        """Deterministic policy: CatalogFilter -> PlotSearch -> Scene/External -> Stop."""
        actions_taken = gathered.get("actions_taken", set())

        if "CatalogFilter" not in actions_taken:
            return {
                "action": "CatalogFilter",
                "reason": "Start with structured filtering to narrow the catalog by year, studio, genre, runtime.",
                "args": {}
            }
        elif "PlotSearch" not in actions_taken:
            # Check if there's thematic content beyond structured fields
            if any(word in user_prompt.lower() for word in ["animal", "theme", "character", "involving", "about", "involves", "lighthearted", "tone", "adventure", "magic", "love"]):
                return {
                    "action": "PlotSearch",
                    "reason": "Query has thematic content; search plot overviews for character/theme matches.",
                    "args": {}
                }
            else:
                return {
                    "action": "Stop",
                    "reason": "No thematic content detected; catalog filter results are sufficient.",
                    "args": {}
                }
        elif "SceneSearch" not in actions_taken or "ExternalContext" not in actions_taken:
            # Check if there are risky-event or tone constraints
            risky_keywords = ["no deaths", "death", "scary", "safe", "without", "violence", "safe for"]
            if any(keyword in user_prompt.lower() for keyword in risky_keywords):
                if "SceneSearch" not in actions_taken and "ExternalContext" not in actions_taken:
                    return {
                        "action": "SceneSearch",
                        "reason": "Query has risky-event constraints (death, scary content); verify per-candidate.",
                        "args": {}
                    }
                elif "SceneSearch" in actions_taken and "ExternalContext" not in actions_taken:
                    return {
                        "action": "ExternalContext",
                        "reason": "Risky events verified; now check external context (tone, reception) for subjective constraints.",
                        "args": {}
                    }
            return {
                "action": "Stop",
                "reason": "All relevant tools have run; hand off to synthesizer.",
                "args": {}
            }
        else:
            return {
                "action": "Stop",
                "reason": "All tools have been run; ready to synthesize final answer.",
                "args": {}
            }

    def translate_catalog_constraints(self, nl_hint: str) -> dict[str, Any]:
        """Extract year, studio, genre, runtime from natural language."""
        from agent.tools import catalog_filter

        constraints = {
            "year_min": None,
            "year_max": None,
            "studio": None,
            "genre": None,
            "runtime_min": None,
            "runtime_max": None
        }

        lower = nl_hint.lower()
        known_genres = catalog_filter.known_genres()
        known_studios = catalog_filter.known_studios()

        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', nl_hint)
        if year_match:
            year = int(year_match.group())
            if any(word in lower[:lower.find(year_match.group())] for word in ["before", "until", "up to", "no older than"]):
                constraints["year_max"] = year
            else:
                constraints["year_min"] = year

        # Extract range like "1990+"
        if "+" in nl_hint:
            match = re.search(r'(\d{4})\+', nl_hint)
            if match:
                constraints["year_min"] = int(match.group(1))

        # Extract studio: match full studio names or known single-word nicknames
        # Avoid matching random words (e.g., "space" in "Space Opera" shouldn't match NASA)
        studio_nicknames = {
            "disney": {"Walt Disney Pictures", "Walt Disney Animation Studios", "DisneyToon Studios"},
            "pixar": {"Pixar Animation Studios"},
            "dreamworks": {"DreamWorks Animation"},
            "universal": {"Universal Pictures"},
            "sony": {"Sony Pictures Animation", "Sony Pictures Entertainment"},
            "warner": {"Warner Bros. Animation", "Warner Bros. Entertainment"},
        }

        for nickname, studio_set in studio_nicknames.items():
            if nickname in lower:
                # Pick the longest match (most specific)
                studio_matches = [s for s in studio_set if s in known_studios]
                if studio_matches:
                    constraints["studio"] = max(studio_matches, key=len)
                    break

        # Extract genre (exact match against known genres)
        for genre in known_genres:
            if genre.lower() in lower:
                constraints["genre"] = genre
                break

        # Extract runtime
        runtime_match = re.search(r'(?:under|less than|<)\s*(\d+)\s*(?:min|minutes)', lower)
        if runtime_match:
            constraints["runtime_max"] = int(runtime_match.group(1))

        runtime_match = re.search(r'(?:at least|>|more than)\s*(\d+)\s*(?:min|minutes)', lower)
        if runtime_match:
            constraints["runtime_min"] = int(runtime_match.group(1))

        return constraints

    def build_plot_query(self, nl_hint: str) -> str:
        """Extract thematic/character keywords for plot search."""
        # Light cleanup: remove stopwords, keep meaningful tokens
        tokens = re.findall(r"[a-z0-9']+", nl_hint.lower())
        meaningful = [t for t in tokens if t not in self._stopwords and len(t) > 2]
        return " ".join(meaningful) if meaningful else nl_hint

    def verify_scene_constraint(
        self, title: str, constraint: str, plot_text: str
    ) -> dict[str, Any]:
        """Heuristic: keyword-based verdict for risky-event constraints."""
        return self._keyword_verdict(title, constraint, plot_text, "scene")

    def verify_external_constraint(
        self, title: str, constraint: str, page_text: str
    ) -> dict[str, Any]:
        """Heuristic: keyword-based verdict for tone/reception constraints."""
        return self._keyword_verdict(title, constraint, page_text, "external")

    def _keyword_verdict(
        self, title: str, constraint: str, text: str, category: str
    ) -> dict[str, Any]:
        """Shared heuristic verdict logic."""
        if not text:
            return {
                "title": title,
                "constraint": constraint,
                "satisfied": None,
                "evidence": f"No {category} text available to verify this constraint."
            }

        lower_text = text.lower()
        lower_constraint = constraint.lower()

        # Detect constraint category
        if any(word in lower_constraint for word in ["death", "die", "dying", "dies"]):
            detected_keywords = self._death_keywords
        elif any(word in lower_constraint for word in ["scary", "frighten", "horror", "terrif"]):
            detected_keywords = self._scary_keywords
        else:
            return {
                "title": title,
                "constraint": constraint,
                "satisfied": None,
                "evidence": "Constraint type not recognized by the mock verifier; would need real LLM reasoning."
            }

        # Detect negation in the constraint
        wants_absent = any(neg in lower_constraint for neg in ["no ", "not ", "n't", "without", "safe", "free"])

        # Check for keywords in text
        found_keywords = []
        for keyword in detected_keywords:
            if keyword in lower_text:
                found_keywords.append(keyword)

        if found_keywords:
            # Extract the first sentence containing a found keyword as evidence
            sentences = text.split(". ")
            for sent in sentences:
                if any(kw in sent.lower() for kw in found_keywords):
                    evidence = sent.strip() + "."
                    break
            else:
                evidence = f"Found keywords: {', '.join(found_keywords[:2])}"

            satisfied = not wants_absent  # If wants absent and we found it, unsatisfied
        else:
            evidence = f"No occurrence of '{constraint.lower()}' found in text."
            satisfied = wants_absent  # If wants absent and we didn't find it, satisfied

        return {
            "title": title,
            "constraint": constraint,
            "satisfied": satisfied,
            "evidence": evidence
        }

    def synthesize_answer(
        self, user_prompt: str, gathered: dict[str, Any]
    ) -> str:
        """Compose final answer from verified candidates."""
        candidates = gathered.get("candidates", [])
        verified = gathered.get("verified", [])
        actions_taken = gathered.get("actions_taken", set())

        if not candidates:
            return "No movies in the catalog matched your constraints. Try relaxing some filters (e.g., year range, studio) or refining your thematic keywords."

        # Build answer listing candidates with release year
        lines = [f"Found {len(candidates)} matching movie(s):"]
        for cand in candidates[:10]:  # Cap at top 10 for readability
            year = cand.get("release_year", "unknown")
            lines.append(f"• {cand['title']} ({year})")

        # Note how constraints were checked
        if verified:
            lines.append("\nConstraint verification:")
            for verdict in verified[:5]:  # Show top 5 verifications
                status = "✓" if verdict.get("satisfied") else ("?" if verdict.get("satisfied") is None else "✗")
                lines.append(f"  {status} {verdict.get('constraint', 'unknown')}: {verdict.get('evidence', '')[:80]}")

        if len(candidates) > 10:
            lines.append(f"\n(and {len(candidates) - 10} more...)")

        if len(candidates) > 0 and "SceneSearch" not in actions_taken and "ExternalContext" not in actions_taken:
            lines.append("\nNote: Per-title constraints (death, scary content, tone) were not verified; these results are based on catalog filters and plot themes only.")

        return "\n".join(lines)


def get_mock_client() -> MockLLMClient:
    """Returns a mock LLM client for zero-cost testing and development."""
    return MockLLMClient()
