"""Observer: reads plot text for ONE question, when read_synopses is called.

Not the main reader any more. agent/verifier.py is: it takes one film and
every condition of the request at once, and its accepted list is what the loop
gates the final answer on. This module survives for the narrower job it was
always good at -- the planner has a single specific question about films it has
already named, asks read_synopses, and gets verdicts back without the plot text
entering its context. It cannot accept a film; only the Verifier can.

The docstring below is the original argument for splitting reading out of the
planner, and it still holds for both readers.


The planner carries 5,867 tokens of system prompt and tool schemas on every
turn, almost all of it about routing conditions to tools. None of that helps
with the only question plot text can answer: *does this passage show the thing
that was asked for?* Yet before this module existed, a full synopsis read --
5,151 tokens for four films -- landed in the planner's context and was re-sent
on every subsequent turn, so the most expensive payload in the system was read
by the most expensive reader.

So reading is split out. The Observer gets the question, the passages, and a
prompt of its own that is roughly a tenth of the planner's, because it needs no
tool schemas, no scope rules, no routing taxonomy and no output format. It
returns a verdict per film and the sentence that justifies it, and the planner
sees that instead of the raw text.

**Quotes are verbatim, and that is not a style preference.** The agent's whole
claim is that it never asserts a story fact it was not shown. If the Observer
paraphrased, the planner would be citing a summary of evidence rather than
evidence, and the chain would be broken one link before the answer. So the
prompt demands exact substrings and `observe()` verifies them against the
source text, dropping any quote it cannot find.
"""

from __future__ import annotations

import json
from typing import Any

from agent import llm_client

# Roughly a tenth of the planner's prompt. Everything the planner knows about
# scope, cost, tool order and answer shape is irrelevant here and would only
# invite this call to start recommending films, which is not its job.
OBSERVER_PROMPT = """\
You are the Observer. You are given one question and the plot text of one or \
more films. For each film, decide whether that text settles the question, and \
quote the part that decides it.

You are not recommending anything. You do not know or care which film is \
better, more popular, or more suitable. You report what the text shows.

For each film return one object:

  film      exactly the label you were given, "Title (Year)"
  verdict   yes       the text shows the thing asked about
            no        the text is substantial and shows the opposite, or shows
                      the thing plainly absent where it would have to appear
            unclear   the text does not settle it either way
  quote     the exact sentence or clause from the text that decides it, copied
            character for character. Empty string when the verdict is unclear.
  note      at most one short sentence of context, only if the quote needs it
            -- who a name refers to, or that an act was attempted rather than
            completed. Never a summary of the film.

If the question is two-sided -- "whether the ending is sad or not sad" -- \
`yes` and `no` cannot both be meaningful. Answer about the FIRST thing it \
names, say so in `note`, and let the quote carry the rest.

Rules that matter more than fluency:

- **The quote must be copied exactly from the text you were given.** Do not \
tidy it, shorten it with ellipses, fix its punctuation, or reconstruct it from \
memory. A quote that is not a literal substring is discarded and your verdict \
loses its evidence.
- Never use anything you know about a film from outside the text provided. If \
the passages do not settle it, the answer is `unclear`, however certain you \
feel.
- `unclear` is a real answer and often the right one. Plot summaries record \
events, not appearance, tone or intent, so questions about how a film feels or \
what a character wears are usually unsettleable from this text.
- A near-miss is `no`, not `yes`. A film where someone seizes power without \
pretending to love anyone does not answer a question about pretending to love \
someone to seize power.

Return JSON only: {"findings": [ ... ]}. No prose around it.\
"""


def _passages_of(entry: dict[str, Any]) -> str:
    """The readable text of one film, however read_synopses returned it."""
    if entry.get("passages"):
        return "\n\n".join(str(p.get("text", "")) for p in entry["passages"])
    return str(entry.get("synopsis") or entry.get("text") or "")


def _render(question: str, entries: list[dict[str, Any]]) -> str:
    blocks = [f"QUESTION\n{question.strip()}", ""]
    for entry in entries:
        blocks.append(f"FILM: {entry.get('film')}")
        blocks.append(_passages_of(entry))
        blocks.append("")
    return "\n".join(blocks)


def observe(question: str, synopses: list[dict[str, Any]]) -> dict[str, Any]:
    """Settle `question` against each film's plot text. One model call.

    Returns the findings plus the step record the loop logs, so the trace shows
    this call the same way it shows a planner turn: module, prompt, response.
    """
    entries = [e for e in synopses if _passages_of(e).strip()]
    if not question or not entries:
        return {"findings": [], "usage": None, "skipped": "nothing readable to observe"}

    user = _render(question, entries)
    message, usage = llm_client.complete([
        {"role": "system", "content": OBSERVER_PROMPT},
        {"role": "user", "content": user},
    ])

    raw = (message.content or "").strip()
    try:
        parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        findings = parsed.get("findings") or []
    except (ValueError, AttributeError):
        # A malformed reply must not be laundered into evidence.
        return {"findings": [], "usage": usage, "raw": raw,
                "error": "the Observer did not return usable JSON"}

    # Every quote is checked against the text it claims to come from. This is
    # the whole point of the module: an unverifiable quote is worse than none,
    # because the planner would cite it as though it had been read.
    by_film = {e.get("film"): _passages_of(e) for e in entries}
    checked = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        film = f.get("film")
        quote = (f.get("quote") or "").strip()
        source = by_film.get(film, "")
        if quote and quote not in source:
            f["quote"] = ""
            f["quote_rejected"] = (
                "not a literal substring of the plot text; discarded, so this "
                "verdict carries no evidence and may not be stated as fact"
            )
            if f.get("verdict") == "yes":
                f["verdict"] = "unclear"
        checked.append(f)

    return {"findings": checked, "usage": usage, "prompt": user}
