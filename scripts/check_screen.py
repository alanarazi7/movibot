#!/usr/bin/env python3
"""Checks the guardrails that can silently do nothing. Free, offline, ~2 seconds.

    python scripts/check_screen.py

The screen is only trustworthy if its error is one-sided. Over-exclusion is
harmless: a film wrongly flagged on "nobody dies" is merely not recommended.
Under-exclusion is the failure that matters -- a film where somebody plainly
dies coming back `clear` would be presented to the user as verified safe.

So the assertions are asymmetric on purpose. Every film in KNOWN_DEATHS must be
flagged, and that is a hard failure. Films in KNOWN_CLEAN are checked too, but a
regression there is reported as a warning: it costs precision, not safety.

It also checks title exclusion, for the same reason. That filter reached
production reporting itself as applied while matching nothing: the planner is
told films are named "Title (Year)", passed exactly that to `exclude_titles`,
and the filter compared it against the bare `title` column, where it could never
match. The answer then avoided the excluded films by the model's own judgement
rather than by the filter -- indistinguishable from working, until you read the
counts in the narrowing trace.

Run this after touching VOCABULARIES, BLACKLIST_PHRASES, MIN_SCREEN_TOKENS, the
chunking parameters, or filter_catalog -- rechunking changes what text each
passage holds, and therefore what the screen can see.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import catalog, tools  # noqa: E402
from rag import screen  # noqa: E402

# Films whose plot involves a death. Chosen to span the sources: MPST synopses,
# Wikipedia plots, and both long and short texts.
KNOWN_DEATHS = [
    "The Lion King (1994)",
    "Up (2009)",
    "Frozen (2013)",
    "Tarzan (1999)",
    "Big Hero 6 (2014)",
    "Finding Nemo (2003)",
    "Mulan (1998)",
    "The Hunchback of Notre Dame (1996)",
    "Ratatouille (2007)",   # Gusteau dies before the film; the screen caught
                            # this when a hand-written ground truth did not
    "Pocahontas (1995)",
    "Hercules (1997)",
    "Brave (2012)",
    "The Good Dinosaur (2015)",
]

# Films with no death, where a flag would mean lost precision. Toy Story,
# Monsters Inc. and Zootopia are deliberately absent: all three legitimately
# match on an attempted or believed killing, which is modality rather than
# vocabulary and is not fixable with a word list.
KNOWN_CLEAN = [
    "Toy Story 2 (1999)",
    "Toy Story 3 (2010)",
    "Inside Out (2015)",
    "WALL·E (2008)",
    "Monsters University (2013)",
    "Cars (2006)",          # "dead heat" -- only clear because of the blacklist
]


def check_exclusions() -> list[str]:
    """Title exclusion must actually remove rows, in both addressing forms."""
    def matched(**kwargs) -> int:
        ctx = tools.ToolContext()
        return tools.filter_catalog(ctx=ctx, **kwargs)["matched"]

    failures = []
    pixar = matched(studio="Pixar")

    # A label, which is what the planner naturally passes.
    if matched(studio="Pixar", exclude_titles=["Toy Story (1995)"]) != pixar - 1:
        failures.append("excluding by \"Title (Year)\" label removed nothing")

    # A bare title, which is what the schema's examples suggest.
    if matched(studio="Pixar", exclude_titles=["Toy Story"]) != pixar - 1:
        failures.append("excluding by bare title removed nothing")

    # A bare title spanning a remake pair drops both; a label drops one.
    if matched(exclude_titles=["The Jungle Book"]) != 236:
        failures.append("bare title did not drop both Jungle Books")
    if matched(exclude_titles=["The Jungle Book (1967)"]) != 237:
        failures.append("label did not drop exactly one Jungle Book")

    # An exclusion the catalog cannot match must be reported, never swallowed.
    ctx = tools.ToolContext()
    out = tools.filter_catalog(ctx=ctx, exclude_titles=["Toy Story 4 (2019)"])
    if not out["filters_applied"].get("exclude_titles_unmatched"):
        failures.append("an unmatchable exclusion was silently ignored")

    return failures


def check_keep_flagged() -> list[str]:
    """The forward direction must return matches, quoted, and narrow to them.

    The scan engine was always general -- it takes any word list -- but every
    affordance pointed backwards: the tool excluded, narrowed to `clear`, and
    told the caller to "recommend from the clear set" when too many films
    matched. So a request to *find* something got answered by ranking instead,
    which is the one thing a single incidental word cannot survive.
    """
    failures = []
    words = ["hat", "hats", "fez", "bonnet"]

    fwd = tools.screen_out(words=words, keep="flagged")
    back = tools.screen_out(words=words, keep="clear")

    if fwd["flagged"] != back["flagged"] or fwd["clear"] != back["clear"]:
        failures.append("the two directions disagree on the same word list")
    if not fwd.get("matching_films"):
        failures.append("keep='flagged' returned no matching films")

    # Every match must arrive with the passage that produced it: a presence
    # claim the model cannot cite is exactly the kind it should not make.
    for m in fwd.get("matching_films", []):
        if not m.get("quote"):
            failures.append(f"{m['film']} matched but came back without a quote")
            break

    # The two films where the wearer really is an animal. Both are invisible to
    # ranking on this request; both are trivially found by the scan.
    found = {m["film"] for m in fwd.get("matching_films", [])}
    for film in ["Zootopia (2016)", "Aladdin (1992)"]:
        if film not in found:
            failures.append(f"{film} names a hat in its plot text but did not match")

    if fwd.get("flagged_note"):
        failures.append("keep='flagged' told the caller to use the clear set instead")

    # Narrowing must follow the direction, or the next tool searches the films
    # that failed the condition.
    ctx = tools.ToolContext()
    tools.screen_out(words=words, keep="flagged", ctx=ctx)
    if set(ctx.candidates() or []) != set(
            screen.screen(words, candidate_ids=[int(i) for i in catalog.movies()["id"]])["flagged"]):
        failures.append("keep='flagged' did not narrow the working set to the matches")

    return failures


def main() -> int:
    result = tools.screen_out(vocabulary="death")
    full = screen.screen(
        screen.VOCABULARIES["death"],
        candidate_ids=[int(i) for i in catalog.movies()["id"]],
    )
    clear = {catalog.label_of(i) for i in full["clear"]}
    flagged = {catalog.label_of(i) for i in full["flagged"]}
    thin = {catalog.label_of(i) for i in full["insufficient_text"]}

    total = len(clear) + len(flagged) + len(thin)
    print(f"screened {total} films: {len(clear)} clear, {len(flagged)} flagged, "
          f"{len(thin)} insufficient_text")

    failures, warnings = [], []

    # Every film must land in exactly one bucket. A film that falls out of all
    # three looks considered when it was not, which is how the four films with
    # no plot text vanished before the tool passed an explicit candidate list.
    if total != len(catalog.movies()):
        failures.append(f"{len(catalog.movies())} films in the catalog but "
                        f"{total} in the three buckets -- some were dropped")

    for film in KNOWN_DEATHS:
        if film in flagged:
            continue
        where = "clear" if film in clear else ("insufficient_text" if film in thin
                                              else "not in catalog")
        failures.append(f"{film} has a death but came back {where}")

    for film in KNOWN_CLEAN:
        if film not in clear:
            where = "flagged" if film in flagged else ("insufficient_text" if film in thin
                                                       else "not in catalog")
            warnings.append(f"{film} has no death but came back {where}")

    # The tool wrapper must agree with the module, and must narrow the working
    # set to exactly the clear films.
    if result["clear"] != len(clear):
        failures.append(f"tool reports {result['clear']} clear, module says {len(clear)}")

    failures.extend(check_exclusions())
    failures.extend(check_keep_flagged())

    for line in warnings:
        print(f"  warning (precision only): {line}")
    for line in failures:
        print(f"  FAIL: {line}")

    if failures:
        print(f"\n{len(failures)} failure(s). These guardrails are not safe to trust.")
        return 1
    print(f"\nOK -- no film with a known death was reported clear, title exclusion "
          f"removes rows in both addressing forms, and the forward scan returns "
          f"quoted matches."
          f"{f' {len(warnings)} precision warning(s).' if warnings else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
