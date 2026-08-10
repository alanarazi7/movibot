"""Remake-safe (title, year) matching for sources that carry no IMDb ID.

The existing Kaggle<->MPST join uses exact IMDb IDs and is validated
one-to-one, so it cannot mis-attach a synopsis. External transcript corpora
and most scraped catalogs have no such key -- only a title and, sometimes, a
year. Under a Disney/Pixar scope that is the hardest possible case, because
almost every animated classic has a live-action remake sharing its exact
title (The Lion King, The Jungle Book, Aladdin, Cinderella, Dumbo, Mulan,
Beauty and the Beast, Pinocchio, Lady and the Tramp, 101 Dalmatians).

A silent mis-attachment here is worse than a miss: the agent would cite a
real scene from the wrong film and sound completely confident doing it. So
every ambiguity is reported rather than guessed, and a candidate claimed by
two movies is rejected for both -- reproducing the one-to-one guarantee the
IMDb-ID join gets for free.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

MATCHED = "matched"
AMBIGUOUS = "ambiguous"
UNMATCHED = "unmatched"

# Articles that appear either leading ("The Jungle Book") or inverted
# ("Jungle Book, The" -- the house style of several script corpora).
ARTICLES = ("the", "a", "an")

# "I" and "X" are letters far more often than numerals in film titles
# ("Malcolm X"), so converting them would corrupt more than it fixes.
# The rest are safe and fix a real Disney case: "Frozen II" vs "Frozen 2".
ROMAN_NUMERALS = {
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
}

_TRAILING_YEAR = re.compile(r"\s*\((?:18|19|20)\d{2}\)\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class MatchCandidate:
    """One side of the join. `key` is an opaque caller-owned identifier."""

    key: str
    title: str
    year: int | None = None


@dataclass(frozen=True)
class MatchResult:
    target_key: str
    status: str
    candidate_key: str | None = None
    candidate_title: str | None = None
    candidate_year: int | None = None
    # Populated on AMBIGUOUS so collisions can be reviewed by hand.
    alternatives: tuple[str, ...] = ()


def normalize_title(title: object) -> str:
    """Folds a title down to a comparable form.

    Handles the differences actually observed between TMDB and scraped
    corpora: case, padding, diacritics, punctuation style ("WALL-E" vs
    "WALL E"), ampersands, inverted articles, trailing "(1994)" stamps and
    roman-numeral sequels.
    """
    if title is None:
        return ""

    text = str(title).strip()
    if not text:
        return ""

    text = _TRAILING_YEAR.sub("", text)

    # Fold accents: "Amelie" and "Amélie" must land on the same key.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")

    # "Wizard of Oz, The" -> "the wizard of oz"
    if "," in text:
        head, _, tail = text.rpartition(",")
        if tail.strip() in ARTICLES:
            text = f"{tail.strip()} {head.strip()}"

    text = _NON_ALNUM.sub(" ", text).strip()

    for article in ARTICLES:
        prefix = f"{article} "
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    words = text.split()
    if len(words) > 1 and words[-1] in ROMAN_NUMERALS:
        words[-1] = ROMAN_NUMERALS[words[-1]]
        text = " ".join(words)

    return text.strip()


def _unmatched(target: MatchCandidate) -> MatchResult:
    return MatchResult(target_key=target.key, status=UNMATCHED)


def _matched(target: MatchCandidate, candidate: MatchCandidate) -> MatchResult:
    return MatchResult(
        target_key=target.key,
        status=MATCHED,
        candidate_key=candidate.key,
        candidate_title=candidate.title,
        candidate_year=candidate.year,
    )


def _ambiguous(target: MatchCandidate, options: Iterable[MatchCandidate]) -> MatchResult:
    return MatchResult(
        target_key=target.key,
        status=AMBIGUOUS,
        alternatives=tuple(c.key for c in options),
    )


def _pick_unique(
    target: MatchCandidate, options: Sequence[MatchCandidate]
) -> MatchResult:
    if len(options) == 1:
        return _matched(target, options[0])
    return _ambiguous(target, options)


def _resolve(
    target: MatchCandidate,
    bucket: Sequence[MatchCandidate],
    year_tolerance: int,
) -> MatchResult:
    if not bucket:
        return _unmatched(target)

    if target.year is None:
        return _pick_unique(target, bucket)

    dated = [c for c in bucket if c.year is not None]
    undated = [c for c in bucket if c.year is None]

    if dated:
        within = [c for c in dated if abs(c.year - target.year) <= year_tolerance]
        if within:
            closest = min(abs(c.year - target.year) for c in within)
            best = [c for c in within if abs(c.year - target.year) == closest]
            return _pick_unique(target, best)
        if undated:
            # Every dated namesake is implausible, but an undated candidate
            # could still be the right film -- only accept it if unique.
            return _pick_unique(target, undated)
        return _unmatched(target)

    return _pick_unique(target, undated)


def match_titles(
    targets: Sequence[MatchCandidate],
    candidates: Sequence[MatchCandidate],
    year_tolerance: int = 1,
) -> tuple[MatchResult, ...]:
    """Joins `targets` to `candidates` on normalized title plus year proximity.

    Returns exactly one result per target, in input order. Nothing is dropped
    silently: a target is always reported as MATCHED, AMBIGUOUS or UNMATCHED.
    """
    buckets: dict[str, list[MatchCandidate]] = {}
    for candidate in candidates:
        buckets.setdefault(normalize_title(candidate.title), []).append(candidate)

    provisional = [
        _resolve(t, buckets.get(normalize_title(t.title), ()), year_tolerance)
        for t in targets
    ]

    # One transcript must never be attached to two different movies. If two
    # targets claim the same candidate, neither can be trusted.
    claims = Counter(r.candidate_key for r in provisional if r.candidate_key is not None)

    return tuple(
        MatchResult(
            target_key=r.target_key,
            status=AMBIGUOUS,
            alternatives=(r.candidate_key,),
        )
        if r.candidate_key is not None and claims[r.candidate_key] > 1
        else r
        for r in provisional
    )


def summarize(results: Sequence[MatchResult]) -> dict[str, int]:
    """Coverage counts, for the report the probe script prints."""
    counts = Counter(r.status for r in results)
    return {
        "total": len(results),
        MATCHED: counts.get(MATCHED, 0),
        AMBIGUOUS: counts.get(AMBIGUOUS, 0),
        UNMATCHED: counts.get(UNMATCHED, 0),
    }
