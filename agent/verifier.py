"""Verifier: one film at a time, every condition at once.

This replaces a reader that had the axes the wrong way round. The Observer took
ONE question and up to eight films, which is efficient and answers the wrong
thing: a request with three conditions got one of them adjudicated and the
other two were left resting on whatever the search happened to return. A film
could be recommended having been checked for deaths and never checked for the
princess or the ice it was asked to have.

So the axes flip. The Verifier takes ONE film and EVERY condition, with that
film's plot text in front of it, and answers each condition separately. What
comes back is a row of a matrix -- film x condition -- and a film is accepted
only when every cell says yes. That list is built in Python, which is the
point: "I can stand behind one title" stops being a claim the model makes about
its own reasoning and becomes a count of a list.

Two properties carried over from the Observer, both load-bearing:

  Quotes are verbatim and verified. Every quote is checked to be a literal
  substring of the text it claims to come from, and discarded otherwise, so a
  verdict either cites evidence or carries none.

  `unclear` is a real answer. A condition the text does not settle is not a
  condition satisfied, and an unverified film is not recommended -- which is
  the whole difference between this agent and a plausible-sounding one.
"""

from __future__ import annotations

import json
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

Return JSON only: {"findings": [ ... ]}. No prose around it.\
"""


def _accepted(findings: list[dict[str, Any]], conditions: list[str]) -> bool:
    """Every condition satisfied, none missing. Nothing else counts.

    Missing is treated as unsatisfied deliberately: a model that returns four
    verdicts for five requirements has not verified the fifth, and reading its
    silence as assent is exactly the failure this module exists to stop.
    """
    verdicts = {f.get("requirement"): f.get("verdict") for f in findings}
    return all(verdicts.get(c) == "yes" for c in conditions)


def verify(film: str, conditions: list[str], text: str) -> dict[str, Any]:
    """Check one film against every condition. One model call.

    Returns the verdict row plus the step record the loop logs, so a Verifier
    call is traced exactly like a planner turn: module, prompt, response.
    """
    conditions = [c for c in (conditions or []) if str(c).strip()]
    if not conditions or not (text or "").strip():
        return {"film": film, "findings": [], "accepted": False, "usage": None,
                "skipped": "no conditions, or no text to read"}

    listed = "\n".join(f"- {c}" for c in conditions)
    user = f"FILM: {film}\n\nREQUIREMENTS\n{listed}\n\nPLOT TEXT\n{text}"

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

    checked = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        quote = (f.get("quote") or "").strip()
        if quote and quote not in text:
            f["quote"] = ""
            f["quote_rejected"] = (
                "not a literal substring of the plot text; discarded, so this "
                "verdict carries no evidence"
            )
            # An unsupported "yes" is the dangerous direction: it is the one
            # that puts a film in front of a user. Downgrade it.
            if f.get("verdict") == "yes":
                f["verdict"] = "unclear"
        checked.append(f)

    return {
        "film": film,
        "findings": checked,
        "accepted": _accepted(checked, conditions),
        "usage": usage,
        "prompt": user,
    }
