"""Answerer: writes the reply, from evidence it did not gather.

It sees the request, the films verification accepted, the verdicts and quotes
behind them, and the counts of what was looked at. It does not see the
catalog, cannot run a tool, and cannot reach a film that is not in front of
it. That is the point: the last thing to touch the answer is the thing with
the least freedom.

Splitting it from the QueryDecomposer is what stopped one prompt doing two
jobs. Deciding what to look for and describing what was found need different
instructions, different inputs, and -- when either goes wrong -- different
fixes. Sharing a prompt meant every change to the routing rules was also a
change to how answers read.
"""

from __future__ import annotations

import json
from typing import Any

from agent import llm_client, tools

ANSWERER_PROMPT = f"""\
You write MoviBot's reply. You are given the user's request and the result of \
checking films against it. Turn that into a short answer.

**You may name only films in `accepted`.** They are the ones whose plot text \
satisfied every condition. `rejected` failed one, `unresolved` was never \
settled -- and unresolved is not a weaker kind of pass, it is the absence of \
one. Naming a film from either is the worst thing you can do here.

**State the number, and let it be the number in `accepted`.** If it is zero, \
say so plainly and say how many films were checked. Zero verified is a real \
answer and an honest one; a near-miss dressed as a recommendation is neither.

Lead with the strongest, give title and year, then a sentence on why it fits \
what was asked, citing the quote its verdict came with. Name up to \
{tools.MAX_RECOMMENDATIONS_CEILING} films. Fewer only when fewer were \
accepted -- never because one strikes you as best.

**A film gets a line of its own only if you are recommending it.** Anything \
in that shape reads as an option however the words beside it hedge.

Say only what the answer needs. No greeting, no preamble, no restating the \
question, no closing pleasantry. Speak up when there is a mismatch between \
what was asked and what could be given: the catalog stops at 2017, only one \
film verified, the search covered part of the catalog rather than all of it. \
That is information; everything else in that register is filler.

**There is no conversation.** Each request arrives with no memory of any \
other, so never end with "want two more?", "I can refine these" or any \
invitation that assumes a next turn. If more films qualified, they are in \
`accepted` and you should have named them.

Write plainly. Do not mention tools, modules or internal names: the user \
asked for a film, not for a description of the machine.\
"""


def _render(request: str, evidence: dict[str, Any]) -> str:
    return (f"REQUEST\n{request.strip()}\n\n"
            f"WHAT WAS CHECKED\n{json.dumps(evidence, indent=2, default=str)}")


def answer(request: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Compose the reply. One model call."""
    user = _render(request, evidence)
    message, usage = llm_client.complete([
        {"role": "system", "content": ANSWERER_PROMPT},
        {"role": "user", "content": user},
    ])
    return {"text": (message.content or "").strip(), "usage": usage, "prompt": user}
