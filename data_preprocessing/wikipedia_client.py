"""Wikipedia fetching for the offline cache builder.

Used only by scrape_wikipedia.py, which caches the result. The agent reads that
cache and never calls Wikipedia at query time.

Resolving a film to its article is the whole problem here. A bare movie title is
very often a Wikipedia *disambiguation* page -- "Frozen" lists eleven entries,
five of them other films -- and an earlier version of this module asked for the
bare title first and accepted any extract over 200 characters. Disambiguation
pages always clear that bar, so the film's real article was never requested and
the cache filled up with lists of links.

So: try the year-qualified title first, and reject disambiguation pages outright
using the API's own `pageprops.disambiguation` flag rather than guessing from
the text. If every candidate title fails, fall back to Wikipedia search.

Redirects are the subtler trap, because they resolve silently to a page nobody
asked for -- an index article, or the wrong film in a series. Every fetch
therefore checks what it actually landed on; see _fetch_article.
"""

import re
import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
TIMEOUT = 10.0
HEADERS = {
    "User-Agent": "MoviBot/1.0 (Educational; +https://github.com/alanarazi7/movibot)"
}

# Index/list articles a redirect can land on -- never a film.
_INDEX_PAGE = re.compile(r"^(list|index|outline) of ", re.IGNORECASE)

# Sections that hold plot narration, in the order we prefer them.
PLOT_SECTIONS = ("plot", "synopsis", "story")

# Excluded when concatenating the non-plot half: the plot sections themselves,
# plus the trailing apparatus, which is citations and navigation rather than
# prose about the film.
NON_PLOT_EXCLUDE = {
    "introduction", "plot", "plot summary", "synopsis", "story",
    "references", "external links", "see also", "notes", "bibliography",
    "further reading", "citations", "works cited",
}


def _query(**params) -> dict | None:
    """One API call. Returns the single page dict, or None on any failure."""
    try:
        resp = requests.get(
            WIKIPEDIA_API_URL,
            params={"action": "query", "format": "json", "redirects": "1", **params},
            timeout=TIMEOUT,
            headers=HEADERS,
        )
        resp.raise_for_status()
        return resp.json().get("query", {})
    except Exception:
        return None


def _is_right_film(extract: str, year: int | None) -> bool:
    """Does this article's lead sentence describe a film from the right year?

    A film article opens "X is a YYYY American animated musical...", so the year
    belongs in the lead. Checking only the lead matters: the 2008 "Beverly Hills
    Chihuahua" article mentions 2012 further down, when listing its sequels, so
    a whole-document search would wave the wrong film through.

    Adjacent years are accepted, since festival and wide-release dates straddle
    year boundaries and the catalog and Wikipedia need not agree on which to use.
    """
    if year is None:
        return True
    lead = extract[:400]
    return any(str(y) in lead for y in (year - 1, year, year + 1))


def _fetch_article(title: str, year: int | None = None) -> tuple[str, str] | None:
    """Fetch one exact title. Returns (resolved_title, extract), or None if the
    page is missing, empty, a disambiguation page, an index page, or a redirect
    that landed on a different film.

    The redirect cases are why the last two checks exist -- both are real:
    "The Prince and the Pauper (film)" redirects to "List of adaptations of The
    Prince and the Pauper", and "Beverly Hills Chihuahua 3" redirects to the
    2008 first film, whose plot would then be cached as the third film's.
    """
    query = _query(prop="extracts|pageprops", explaintext="1", titles=title)
    if not query:
        return None

    pages = query.get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "missing" in page:
        return None

    # The API's own flag -- far more reliable than inspecting the prose.
    if "disambiguation" in page.get("pageprops", {}):
        return None

    resolved = page.get("title", title)
    if _INDEX_PAGE.match(resolved):
        return None

    extract = (page.get("extract") or "").strip()
    if not extract:
        return None

    if not _is_right_film(extract, year):
        return None

    return resolved, extract


def _search_article(title: str, year: int | None) -> tuple[str, str] | None:
    """Last resort: ask Wikipedia's search for the film, then fetch the top hit.

    This is the only path that can return an article for a title we never asked
    for, so it is the only one that can silently return the *wrong film*. Two
    guards, both learned from real failures on this catalog:

      * "The Prince and the Pauper" matched "List of adaptations of The Prince
        and the Pauper" -- an index page, not a film.
      * "Beverly Hills Chihuahua 3" matched "Beverly Hills Chihuahua", giving
        the first film's plot for the third. A wrong plot is worse than none,
        so requiring the article to mention the release year rejects it.

    A rejected hit falls through to "not found", which is the honest answer.
    """
    terms = f"{title} {year} film" if year else f"{title} film"
    query = _query(list="search", srsearch=terms, srlimit=3)
    if not query:
        return None

    for hit in query.get("search", []):
        if _INDEX_PAGE.match(hit["title"]):
            continue

        found = _fetch_article(hit["title"], year)
        if found:
            return found

    return None


def fetch_page_extract(title: str, year: int | None = None) -> tuple[str, str] | None:
    """Resolve a film to its Wikipedia article.

    Candidates are tried most-specific first, so a disambiguation page can never
    shadow the film's own article:

        "Frozen (2013 film)" -> "Frozen (film)" -> "Frozen" -> search

    Returns (resolved_title, plaintext_extract), or None if nothing resolved.
    """
    candidates = []
    if year:
        candidates.append(f"{title} ({year} film)")
    candidates.append(f"{title} (film)")
    candidates.append(title)

    for candidate in candidates:
        found = _fetch_article(candidate, year)
        if found:
            return found

    return _search_article(title, year)


def split_into_sections(extract: str) -> dict[str, str]:
    """Split a plaintext extract into {section_title: body}.

    Plaintext extracts mark headings as "== Plot ==" / "=== Casting ===". Match
    those explicitly; the previous heuristic (any short line without a full
    stop) also caught ordinary short sentences and split mid-paragraph.
    """
    sections: dict[str, str] = {}
    current = "Introduction"
    body: list[str] = []

    for line in extract.split("\n"):
        heading = re.fullmatch(r"\s*={2,6}\s*(.+?)\s*={2,6}\s*", line)
        if heading:
            text = "\n".join(body).strip()
            if text:
                sections[current] = text
            current = heading.group(1)
            body = []
        else:
            body.append(line)

    text = "\n".join(body).strip()
    if text:
        sections[current] = text

    return sections


def get_plot(sections: dict[str, str]) -> str | None:
    """The plot narration, or None if the article has no plot section."""
    for wanted in PLOT_SECTIONS:
        for name, text in sections.items():
            if wanted in name.lower():
                return text
    return None


def get_non_plot(sections: dict[str, str], max_chars: int = 4000) -> str | None:
    """Everything that is not plot -- production, reception, themes, music.

    This is what answers tone questions ("is it lighthearted?") that the plot
    itself cannot settle.
    """
    parts = [
        text for name, text in sections.items()
        if name.lower().strip() not in NON_PLOT_EXCLUDE
        and not any(w in name.lower() for w in PLOT_SECTIONS)
    ]
    combined = "\n\n".join(parts).strip()
    return combined[:max_chars] if combined else None
