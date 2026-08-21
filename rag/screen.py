"""Exhaustive lexical screening, for conditions phrased as a negation.

"A film where nobody dies" is not a ranking problem, and treating it as one gets
it backwards: embed "nobody dies" and the top hits are the films where somebody
does. Similarity finds the films that FAIL the condition.

So negations are screened instead of ranked. Every plot passage of every
candidate is scanned for a small vocabulary, and the films with no match are the
answer. Two properties make this the right tool rather than a shortcut:

  exhaustive   a scan touches all 2,080 plot passages, so no film escapes the
               check by ranking eleventh. Top-k retrieval cannot promise this.
  one-sided    "dead heat" over-excludes Cars; it never under-excludes. For a
               negative request that is the harmless direction, which is what
               lets the result be trusted without a model reading anything.

Free, and fast enough that scope is irrelevant: 66 ms for the full corpus.

WHAT THIS DOES NOT DO

It cannot tell an attempt from an outcome. "Randall attempts to kill Sulley",
"believing Woody murdered Buzz", "Judy asks if Bellwether is going to kill her"
all match, and nobody dies in any of them. That is modality, not vocabulary, and
no word list resolves it -- which is why a match makes a film `flagged`, meaning
unresolved, and never `rejected`. Resolving a flag is a job for the reader, and
it is cheap because the matching passages come back with the flag: the model
judges two sentences instead of a 5,000-word plot.
"""

from __future__ import annotations

import re
from typing import Any

from rag import store
from rag.corpora import DEFAULT_SOURCES

# A film is only certified clear if it offered enough text for the absence of a
# match to mean something. Below this, "no death word" is absence of evidence
# rather than evidence of absence -- median plot length is 1,207 tokens for
# clear films against 2,153 for flagged ones, precisely because a short summary
# has fewer chances to contain any given word. Screening a 200-token synopsis
# and calling it verified is the one way this tool could actively mislead.
MIN_SCREEN_TOKENS = 600

# Curated vocabularies for the negations that actually get asked. The caller may
# always pass its own words; these exist so answer quality does not depend on
# whatever synonyms the model happens to produce on a given turn.


# How many matching passages come back per flagged film. One is usually enough
# to judge modality; three covers a film whose first match is incidental and
# whose second is the real death.
MAX_EVIDENCE_PER_FILM = 3
MAX_EVIDENCE_CHARS = 400

def _blacklist_re(phrases: list[str] | None = None) -> re.Pattern[str] | None:
    """Strip idioms that contain a scanned word without carrying its meaning.

    Nothing is built in. There used to be a fixed list -- "dead end",
    "deadline", "kill time" -- which only ever fitted the fixed vocabularies it
    grew up beside. Both are gone: the caller writes the words it is scanning
    for and the phrasings that would trip them, because only the caller knows
    which pairing it means. "Shot" wants "shot a photograph" excluded when the
    question is violence and not when it is photography, and no list decided in
    advance can tell those apart.

    None when there is nothing to strip, so the caller skips the substitution.
    """
    cleaned = []
    for phrase in phrases or []:
        phrase = str(phrase).strip().lower()
        if phrase and phrase not in cleaned:
            cleaned.append(phrase)
    if not cleaned:
        return None
    # Longest first, so a phrase wins over a shorter one sharing its start.
    cleaned.sort(key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in cleaned) + r")\b", re.I
    )


def resolve_words(words: list[str] | None) -> list[str]:
    """Normalise the caller's word list: lowercased, trimmed, de-duplicated."""
    out: list[str] = []
    for w in words or []:
        w = str(w).strip().lower()
        if w and w not in out:
            out.append(w)
    return out


def _pattern(words: list[str]) -> re.Pattern[str]:
    # Word boundaries, so "die" does not match "diesel" or "audience". The whole
    # value of the screen is that a match means something.
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I
    )


def screen(
    words: list[str],
    candidate_ids: list[int] | None = None,
    sources: list[str] | None = None,
    and_words: list[str] | None = None,
    exclude_phrases: list[str] | None = None,
) -> dict[str, Any]:
    """Split candidates into clear / flagged / insufficient_text.

    Returns ids, not labels: the caller decides how films are named, and the
    working set stays a set of ids in Python either way.
    """
    if not words:
        raise ValueError("screen() needs at least one word to look for.")

    pattern = _pattern(words)
    blacklist = _blacklist_re(exclude_phrases)
    passages = store.plot_passages(sources or DEFAULT_SOURCES)

    allowed = set(int(c) for c in candidate_ids) if candidate_ids is not None else None

    tokens: dict[int, int] = {}
    # A conjunction is not two scans intersected: "a cat that wears a hat" is
    # not answered by films mentioning a cat somewhere and a hat somewhere,
    # which on this catalog is 11 films and mostly nonsense. Both have to land
    # in the SAME passage -- 300 tokens of one scene -- before the pairing means
    # anything. Without this the tool returned 51 films for cat-or-hat and the
    # planner spent the answer explaining why each one did not fit.
    and_pattern = _pattern(and_words) if and_words else None

    evidence: dict[int, list[dict[str, Any]]] = {}

    for p in passages:
        movie_id = int(p["movie_id"])
        if allowed is not None and movie_id not in allowed:
            continue

        tokens[movie_id] = tokens.get(movie_id, 0) + int(p.get("tokens", 0))

        # Blank the idioms rather than the whole passage: a passage containing
        # both "dead heat" and a real death must still flag.
        text = str(p.get("text", ""))
        if blacklist is not None:
            text = blacklist.sub(" ", text)
        found = pattern.search(text)
        if found is None:
            continue

        second = and_pattern.search(text) if and_pattern is not None else None
        if and_pattern is not None and second is None:
            continue

        hits = evidence.setdefault(movie_id, [])
        if len(hits) < MAX_EVIDENCE_PER_FILM:
            # Quote the span that holds both, so the reader can judge whether
            # the two words are actually related or merely adjacent.
            lo = min(found.start(), second.start()) if second else found.start()
            hi = max(found.end(), second.end()) if second else found.end()
            start = max(0, lo - MAX_EVIDENCE_CHARS // 2)
            hit = {
                "word": found.group(0).lower(),
                "chunk_index": int(p.get("chunk_index", 0)),
                "quote": text[start:hi + MAX_EVIDENCE_CHARS // 2].strip(),
            }
            if second is not None:
                hit["with"] = second.group(0).lower()
            hits.append(hit)

    # Candidates with no indexed plot text at all never appear in the loop above,
    # so they have to be added back explicitly. Silently dropping them would be
    # the worst outcome available: they would vanish from all three buckets and
    # look like films that were considered.
    universe = set(allowed) if allowed is not None else set(tokens)
    for movie_id in universe:
        tokens.setdefault(movie_id, 0)

    clear, flagged, insufficient = [], [], []
    for movie_id in sorted(universe):
        if movie_id in evidence:
            flagged.append(movie_id)
        elif tokens[movie_id] < MIN_SCREEN_TOKENS:
            insufficient.append(movie_id)
        else:
            clear.append(movie_id)

    return {
        "words": words,
        "clear": clear,
        "flagged": flagged,
        "insufficient_text": insufficient,
        "evidence": evidence,
        "tokens_scanned": sum(tokens.values()),
        "min_screen_tokens": MIN_SCREEN_TOKENS,
    }
