"""Verifier: one film at a time, every condition at once.

This replaces a reader that had the axes the wrong way round. It took ONE
question and up to eight films, which is efficient and answers the wrong thing:
a request with three conditions got one of them adjudicated and the other two
were left resting on whatever the search happened to return. A film could be
recommended having been checked for deaths and never checked for the princess
or the ice it was asked to have.

So the axes flip. The Verifier takes ONE film and EVERY condition, with that
film's plot text in front of it, and answers each condition separately. What
comes back is a row of a matrix -- film x condition -- and a film is accepted
only when every cell says yes. That list is built in Python, which is the
point: "I can stand behind one title" stops being a claim the model makes about
its own reasoning and becomes a count of a list.

Two properties carried over from that reader, both load-bearing:

  Quotes are verbatim and verified. Every quote is checked to be a literal
  substring of the text it claims to come from, and discarded otherwise, so a
  verdict either cites evidence or carries none.

  `unclear` is a real answer. A condition the text does not settle is not a
  condition satisfied, and an unverified film is not recommended -- which is
  the whole difference between this agent and a plausible-sounding one.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent import llm_client

# Conditions are stated as REQUIREMENTS the film must satisfy, not as questions
# about what the text shows. That single change removes the inversion the old
# design forced on the planner: asked "does anyone die", a `yes` meant the film
# FAILED a no-deaths request, and every reading of the trace had to flip it.
# Here `yes` always means satisfied, for every condition, including negative
# ones.
VERIFIER_PROMPT = """\
You are the Verifier. You are given ONE film's plot text and a list of \
REQUIREMENTS. For each requirement, decide whether this film satisfies it, \
using only the text in front of you.

For each requirement return one object:

  requirement  the requirement text, copied exactly as given
  verdict      yes      the text establishes that the film satisfies it
               no       the text establishes that the film does NOT satisfy it
               unclear  the text does not settle it either way
  quote        the exact sentence or clause that decides it, copied character
               for character from the text. Empty string when unclear.
  note         at most one short sentence, only where the quote needs it --
               who a name refers to, or that an act was attempted rather than
               completed.

Rules that decide most cases:

- **Only the text counts.** Never use anything you know about this film from \
outside the passage you were given. If the text does not settle a \
requirement, the answer is `unclear`, however certain you feel.
- **The quote must be a literal substring of the text.** Do not tidy it, \
shorten it with ellipses, or reconstruct it from memory. A quote that is not \
found in the text is discarded and its verdict loses all evidence.
- **A requirement of absence -- "no character dies", "nothing frightening" -- \
is satisfied only by substantial text in which the thing does not occur.** If \
the text is short or fragmentary, that is `unclear`, not `yes`.
- **Threats, attempts, metaphors, accusations, false beliefs and near-misses \
do not establish that a thing happened.** "He tried to kill him", "believing \
Woody had murdered Buzz" and "I'll kill you" all leave an actual death NOT \
established. Only a death the text actually states is a death. The same \
applies to every other requirement: an intention is not an event.
- **A near-miss is `no`, not `yes`.** A film where someone seizes power \
without pretending to love anyone does not satisfy a requirement about \
pretending to love someone to seize power.
- **A requirement with several parts needs ONE thing that has all of them, \
shown in the text.** "An animal that wears a hat" is not an animal in one \
sentence and a hat in another. It needs a single creature the text shows \
wearing a hat. A butler who leaves his hat in the countryside is a person \
with a hat, so the answer is `no`. A hamster the text never puts a hat on is \
`unclear`, however sure you are the film shows one. A Mad Hatter is not an \
animal. Check every part against the same subject before you answer `yes`.
- **Your quote must show the WHOLE requirement by itself**, not the half of it \
you could find. If the sentence establishes only one part, you do not have the \
evidence and the verdict is `unclear`. A real sentence that does not say the \
thing is worse than no sentence, because it looks like proof.
- **If your note would explain why the quote is not quite it, the verdict is \
not `yes`.** "No hat-wearing animal is explicitly named, however..." and \
"later scenes depict him with a hat" are both you noticing that the text does \
not support the answer you are about to give. Write `unclear` instead. A note \
identifies who a name refers to; it never argues a quote into saying more than \
it says.

**The user's request is context, never evidence.** It tells you what a \
requirement means -- whether "a hat" is one someone wears or one on a shelf -- \
and nothing more. It cannot make a film satisfy anything, and wanting to \
answer it is not a reason to read a passage generously. Judge each requirement \
from the plot text alone.

Return JSON only: {"findings": [ ... ]}. No prose around it.\
"""


# No stored phrasings anywhere in this module. An earlier version matched a
# verdict's note against a list of hedging words -- "however", "not a",
# "associated", "later scene" -- to catch a `yes` that argued against itself.
# It caught the phrasings that had been seen and missed the rest, which is what
# any stored list does. What survives are checks that need no vocabulary: a
# quote must be a literal substring of the text, a decisive verdict must have
# one, every requirement gets exactly one finding, and an absence cannot be
# evidenced by a sentence containing the very words the Decomposer wrote for
# it. Those hold whatever the request is about.


def _accepted(findings: list[dict[str, Any]], conditions: list[str]) -> bool:
    """Every condition satisfied, none missing. Nothing else counts.

    Missing is treated as unsatisfied deliberately: a model that returns four
    verdicts for five requirements has not verified the fifth, and reading its
    silence as assent is exactly the failure this module exists to stop.
    """
    verdicts = {f.get("requirement"): f.get("verdict") for f in findings}
    return all(verdicts.get(c) == "yes" for c in conditions)


def _refuted_by_its_own_quote(requirement: str, quote: str,
                              deny: dict[str, list[str]] | None) -> str | None:
    """The word in this quote that contradicts the absence it is supporting.

    The Verifier returned `yes` for "no character dies" quoting "McLeach is
    swept over the waterfall to his death." Every other guard passed: the
    quote is a literal substring, it carries no hedging note, and the verdict
    cites evidence. Quote-checking proves where a sentence came from, never
    what it shows.

    What makes this decidable is that the Decomposer wrote the vocabulary for
    the thing being denied AND said which requirement it wrote it for, so the
    pairing is stated rather than guessed. An earlier version guessed, by
    matching the requirement against a list of English negation words -- which
    fired on "no character dies" and did nothing for "everyone lives", because
    the list only ever covers the phrasings someone thought of.
    """
    if not deny or not quote:
        return None
    words = deny.get((requirement or "").strip())
    if not words:
        return None
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", quote, re.I):
            return word
    return None


def verify(film: str, conditions: list[str], text: str,
           deny: dict[str, list[str]] | None = None,
           request: str = "") -> dict[str, Any]:
    """Check one film against every condition. One model call.

    Returns the verdict row plus the step record the loop logs, so a Verifier
    call is traced exactly like a planner turn: module, prompt, response.
    """
    conditions = [c for c in (conditions or []) if str(c).strip()]
    if not conditions or not (text or "").strip():
        return {"film": film, "findings": [], "accepted": False, "usage": None,
                "skipped": "no conditions, or no text to read"}

    listed = "\n".join(f"- {c}" for c in conditions)
    asked = f"WHAT THE USER ASKED FOR\n{request.strip()}\n\n" if request else ""
    user = (f"{asked}FILM: {film}\n\nREQUIREMENTS\n{listed}\n\n"
            f"PLOT TEXT\n{text}")

    message, usage = llm_client.complete([
        {"role": "system", "content": VERIFIER_PROMPT},
        {"role": "user", "content": user},
    ])

    raw = (message.content or "").strip()
    try:
        parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        findings = parsed.get("findings") or []
    except (ValueError, AttributeError):
        # A malformed reply must not be laundered into evidence.
        return {"film": film, "findings": [], "accepted": False, "usage": usage,
                "raw": raw, "error": "the Verifier did not return usable JSON"}

    # One finding per requirement asked for, keyed by the requirement text.
    # Anything else is dropped: the model has returned stray objects carrying
    # only a `note` and no verdict, which sailed through as a finding and
    # rendered as "undefined: undefined". A requirement with no finding is
    # `unclear` -- silence is not assent.
    by_requirement: dict[str, dict[str, Any]] = {}
    for f in findings:
        if not isinstance(f, dict):
            continue
        req = f.get("requirement")
        if req in conditions and req not in by_requirement:
            by_requirement[req] = f

    checked = []
    for condition in conditions:
        f = by_requirement.get(condition)
        if f is None:
            checked.append({
                "requirement": condition,
                "verdict": "unclear",
                "quote": "",
                "note": "the Verifier returned no verdict for this requirement",
            })
            continue

        quote = (f.get("quote") or "").strip()
        if quote and quote not in text:
            f["quote"] = ""
            f["quote_rejected"] = (
                "not a literal substring of the plot text; discarded, so this "
                "verdict carries no evidence"
            )
            quote = ""

        # A decisive verdict has to cite something. Without this, "yes" with an
        # empty quote counted as satisfied -- which is the model asserting a
        # story fact, exactly what this module exists to stop. `unclear` is
        # allowed to have no quote; it is the verdict that claims nothing.
        # A `yes` for an absence, evidenced by a sentence containing the very
        # thing denied, is not a yes.
        if f.get("verdict") == "yes":
            hit = _refuted_by_its_own_quote(condition, quote, deny)
            if hit:
                f["verdict"] = "no"
                f["downgraded"] = (
                    f"the quote offered for this absence contains {hit!r}, so it "
                    f"shows the opposite of what the verdict claims"
                )

        if f.get("verdict") in ("yes", "no") and not quote:
            f["verdict"] = "unclear"
            f["downgraded"] = (
                "a decisive verdict with no surviving quote is an assertion, not "
                "a finding"
            )
        checked.append(f)

    return {
        "film": film,
        "findings": checked,
        "accepted": _accepted(checked, conditions),
        "usage": usage,
        "prompt": user,
    }
