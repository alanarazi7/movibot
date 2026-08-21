"""Rank fusion: turn one ranked list per condition into one ordered shortlist.

The problem this exists to solve is greed. A request with three story
conditions used to be answered by searching for one of them, taking the top
hits, and reading those -- which quietly assumes the other two conditions will
happen to hold in whatever the first one returned. Nothing makes that true. A
film ranking first on "a princess" and thirty-first on "snow and ice" was read
and recommended; the film that placed tenth on both was never looked at.

So every condition gets its own search, and the lists are fused *before* a
single model call is spent. What comes out is one ordering in which a film
that satisfies everything moderately beats a film that satisfies one thing
perfectly, which is the whole point.

Fusion is two-tiered, and the tiering is the part that matters:

    1. how many conditions the film appears for   (more is better)
    2. its average rank among those               (lower is better)

Average rank alone is not enough, and the failure is worth writing down. Given
top-20 lists and a penalty of 21 for absence, a film ranked 1st, 1st and absent
averages 7.7, while a film ranked 10th, 10th and 10th averages 10 -- so the one
that misses a condition outright wins. That is exactly the greedy answer this
module exists to prevent, arriving by arithmetic instead of by tool order.
Coverage first, rank second, and the failure disappears.

Nothing here calls a model or touches the network: it is a pure function of the
lists it is handed, which is why it can be tested exhaustively for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """One film's standing across every condition."""

    movie_id: int
    ranks: dict[str, int] = field(default_factory=dict)   # condition -> 1-based rank
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def covered(self) -> int:
        return len(self.ranks)

    def average_rank(self) -> float:
        """Mean rank among the conditions this film actually placed for.

        Among those, deliberately. A film absent from a list has no rank there,
        and inventing one to average in would blur the coverage tier that the
        ordering rests on.
        """
        return sum(self.ranks.values()) / len(self.ranks) if self.ranks else float("inf")


def fuse(
    condition_lists: dict[str, list[int]],
    ratings: dict[int, float] | None = None,
) -> list[Candidate]:
    """Merge per-condition rankings into one ordering, best first.

    `condition_lists` maps a condition to the film ids it retrieved, best
    first. Order within each list is all that is read; scores are carried only
    so the trace can show them.

    Ties are broken by the catalog's vote-weighted rating, which is the only
    ordering this project ever uses when nothing else separates two films.
    """
    ratings = ratings or {}
    pool: dict[int, Candidate] = {}

    for condition, ids in condition_lists.items():
        for position, movie_id in enumerate(ids, start=1):
            cand = pool.setdefault(int(movie_id), Candidate(movie_id=int(movie_id)))
            # A film can surface twice for the same condition only if a caller
            # passes a list with duplicates; keep its best placing.
            if condition not in cand.ranks or position < cand.ranks[condition]:
                cand.ranks[condition] = position

    return sorted(
        pool.values(),
        key=lambda c: (-c.covered, c.average_rank(), -ratings.get(c.movie_id, 0.0)),
    )


def explain(cand: Candidate, conditions: list[str], label: str) -> dict[str, Any]:
    """One row of the shortlist, in the shape the trace and the model both read.

    Absence is written as None rather than omitted. A condition missing from
    the row would read as an oversight; None says the search ran and this film
    did not place, which is a finding.
    """
    return {
        "film": label,
        "conditions_matched": f"{cand.covered}/{len(conditions)}",
        "ranks": {c: cand.ranks.get(c) for c in conditions},
        "average_rank": round(cand.average_rank(), 2),
    }
