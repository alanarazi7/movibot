#!/usr/bin/env python3
"""Checks the lexical screen's safety property. Free, offline, ~2 seconds.

    python scripts/check_screen.py

The screen is only trustworthy if its error is one-sided. Over-exclusion is
harmless: a film wrongly flagged on "nobody dies" is merely not recommended.
Under-exclusion is the failure that matters -- a film where somebody plainly
dies coming back `clear` would be presented to the user as verified safe.

So the assertions are asymmetric on purpose. Every film in KNOWN_DEATHS must be
flagged, and that is a hard failure. Films in KNOWN_CLEAN are checked too, but a
regression there is reported as a warning: it costs precision, not safety.

Run this after touching VOCABULARIES, BLACKLIST_PHRASES, MIN_SCREEN_TOKENS, or
the chunking parameters -- rechunking changes what text each passage holds, and
therefore what the screen can see.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The screen never calls an API. Setting this proves it: if any code path here
# tried to embed, it would raise instead of quietly spending money.
os.environ.setdefault("MOVIBOT_OFFLINE", "1")

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

    for line in warnings:
        print(f"  warning (precision only): {line}")
    for line in failures:
        print(f"  FAIL: {line}")

    if failures:
        print(f"\n{len(failures)} failure(s). The screen is not safe to trust.")
        return 1
    print(f"\nOK -- no film with a known death was reported clear."
          f"{f' {len(warnings)} precision warning(s).' if warnings else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
