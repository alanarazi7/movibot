"""Shared Wikipedia fetching utilities for SceneSearch and ExternalContext."""

import re
import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
TIMEOUT = 5.0


def fetch_page_extract(title: str) -> str | None:
    """Fetch plaintext extract from Wikipedia for a movie title.

    Args:
        title: movie title to look up.

    Returns:
        Plaintext extract, or None on any failure (network, timeout, missing page).
        Retries with "{title} (film)" if the initial lookup returns short/missing result.
    """
    def _fetch_attempt(t: str) -> str | None:
        try:
            params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "titles": t,
                "format": "json"
            }
            resp = requests.get(WIKIPEDIA_API_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page = next(iter(pages.values()))
            extract = page.get("extract", "").strip()
            return extract if extract else None
        except Exception:
            return None

    extract = _fetch_attempt(title)
    if not extract or len(extract) < 200:
        # Retry with "(film)" suffix
        extract = _fetch_attempt(f"{title} (film)")

    return extract


def split_into_sections(extract: str) -> dict[str, str]:
    """Heuristically split plaintext Wikipedia extract into sections.

    Assumes headings are short lines (< 80 chars) flanked by blank lines.

    Returns:
        {"section_title": "section_body", ...}
    """
    sections = {}
    current_section = "Introduction"
    current_text = []

    lines = extract.split("\n")
    for line in lines:
        stripped = line.strip()

        # Heuristic: a likely section heading is a short line with no punctuation/colons
        if (
            stripped
            and len(stripped) < 80
            and not stripped.endswith(".")
            and ":" not in stripped
            and len(stripped.split()) <= 5
        ):
            # Save previous section if it has content
            body = "\n".join(current_text).strip()
            if body:
                sections[current_section] = body
            current_section = stripped
            current_text = []
        else:
            current_text.append(line)

    # Save final section
    body = "\n".join(current_text).strip()
    if body:
        sections[current_section] = body

    return sections


def get_section(
    title: str, section_names: list[str]
) -> str | None:
    """Fetch and find a specific section by name.

    Args:
        title: movie title.
        section_names: list of section names to try, in order (e.g., ["Plot", "Plot summary"]).

    Returns:
        Section body (plaintext), or None if not found.
    """
    extract = fetch_page_extract(title)
    if not extract:
        return None

    sections = split_into_sections(extract)
    for name in section_names:
        for section_key, section_body in sections.items():
            if name.lower() in section_key.lower():
                return section_body

    return None


def get_non_plot_text(title: str, max_chars: int = 4000) -> str | None:
    """Fetch Wikipedia page and concatenate all non-Plot sections.

    Used by ExternalContext to get tone/reception/themes text.

    Args:
        title: movie title.
        max_chars: truncate concatenated text to this length.

    Returns:
        Concatenated non-Plot sections (plaintext), or None if no page found.
    """
    extract = fetch_page_extract(title)
    if not extract:
        return None

    sections = split_into_sections(extract)
    exclude_names = {"plot", "plot summary", "synopsis", "introduction"}

    text_parts = []
    for section_name, section_body in sections.items():
        if section_name.lower() not in exclude_names:
            text_parts.append(section_body)

    combined = "\n\n".join(text_parts)
    return combined[:max_chars] if combined else None
