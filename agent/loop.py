"""MoviBot's agent loop: native tool calling, bounded, fully traced.

    plan -> call tools -> observe -> (repeat) -> answer

The model chooses which tools to call and when to stop. What it cannot choose
is how long to keep going.

**Two bounds, and they are different quantities.** MAX_ROUNDS bounds the
planner's turns. It does not bound spend, because the planner is not the only
thing that calls a model: every read_synopses that returns text sends the
Observer, so one round can cost two calls. A cap on rounds was therefore being
quoted as a cap on cost, which it never was -- five rounds could be eleven
calls, and nothing in the code said otherwise.

MAX_TOTAL_LLM_CALLS is the real budget. It counts planner and Observer calls
together, is enforced here rather than asked for in a prompt, and reserves its
last call for the planner so a request runs out of budget holding an answer
instead of a timeout.

Cost per request, worst case:

    <= MAX_ROUNDS rounds              (planner turns)
    <= MAX_TOTAL_LLM_CALLS model calls (paid -- planner + Observer)
    unlimited tool calls              (free -- local CSV reads)

Tool calls are free, which inverts the usual tuning problem: there is no
reason to skimp on them, only on model calls, which is exactly what this loop
bounds.
"""

from __future__ import annotations

import json
from typing import Any

from agent import catalog, llm_client, observer, prompts, tools

# Enough for filter -> search -> read -> answer, with one round spare for a
# correction. Queries needing more than this are usually a sign the model is
# thrashing, and cutting it off is cheaper than letting it continue.
MAX_ROUNDS = 5

# The one that bounds spend. Sized against the busiest request actually
# observed, which spent six calls (three planner rounds and three Observer
# reads), so there is real headroom rather than a bound that binds at the
# worst case already seen.
#
# One Observer call adjudicates a whole read, up to MAX_SYNOPSES films, so
# this is not the limit on how much text gets inspected -- at 12 the worst
# case still leaves seven reads, which is 56 films, against a catalog of 238
# and a shortlist that is never more than a few dozen. What actually bounds
# inspection is MAX_SYNOPSES, per read.
#
# The pool is shared, which means a planner that thrashes through five rounds
# leaves less for the Observer. That is a real tradeoff and it is the reason
# the number is 12 rather than 8: the slack absorbs a thrashing planner
# without starving the reader.
#
# Raising MAX_ROUNDS without raising this does nothing, which is the right way
# round -- rounds are a thinking budget, calls are the bill.
MAX_TOTAL_LLM_CALLS = 16


class _Budget:
    """Counts the two kinds of model call, so neither can hide inside the other.

    Kept as an object rather than two ints because the reserve rule -- never
    spend the last call on an Observer -- is a question about the pair, and
    written inline it would be an off-by-one waiting to happen.
    """

    def __init__(self) -> None:
        self.planner = 0
        self.observer = 0
        self.verifier = 0

    @property
    def total(self) -> int:
        return self.planner + self.observer + self.verifier

    @property
    def remaining(self) -> int:
        return MAX_TOTAL_LLM_CALLS - self.total

    def can_observe(self) -> bool:
        """Only when a planner call would still be affordable afterwards.

        An Observer call that consumes the last of the budget buys evidence
        nobody can spend: the planner never gets another turn to write it up,
        so the user pays for a read and receives a timeout.
        """
        return self.remaining >= 2

    def as_dict(self) -> dict[str, int]:
        return {"planner": self.planner, "observer": self.observer,
                "verifier": self.verifier, "total": self.total,
                "cap": MAX_TOTAL_LLM_CALLS}


def execute(prompt: str) -> dict[str, Any]:
    """Answer one user request.

    Returns the /api/execute contract, which fixes the top-level fields
    exactly -- four, no more:
        {"status": "ok"|"error", "error": str|None,
         "response": str|None, "steps": [...]}

    Everything diagnostic therefore lives inside a step, where the spec only
    requires that module, prompt and response be present. Token usage rides on
    the Planner step that spent it; the working set after each tool call rides
    on that tool's step as `scope`. The condition ledger needs no field of its
    own: it is the content of the first Planner step's response.

    `steps` traces every model turn and tool call in order, so the whole
    decision path is inspectable from the API response alone.
    """
    steps: list[dict[str, Any]] = []
    # The candidate set for this request. filter_catalog fills it; search and
    # read scope themselves to it. It never enters the prompt, and it dies with
    # the request, so there is no cross-request state to reason about.
    ctx = tools.ToolContext()
    budget = _Budget()
    corrected = False
    # What verification actually accepted, and what it looked at. Held here, in
    # Python, because the exit gate below compares the answer against it: a
    # film the model names is recommendable only if it is in this set.
    verified: dict[str, set[str] | bool] = {"accepted": set(), "seen": set(),
                                            "ran": False}

    if not prompt or not prompt.strip():
        return _error("The 'prompt' field is required.", steps)

    if not llm_client.is_configured():
        return _error(
            "MoviBot cannot compose an answer: OPENAI_API_KEY is unset or "
            "still a placeholder value. The catalog and search tools run "
            "locally and are unaffected.",
            steps,
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]

    try:
        for round_index in range(MAX_ROUNDS):
            # Two ways to be on the last turn, and the budget is the one that
            # can arrive early: an Observer-heavy request exhausts calls before
            # it exhausts rounds. Either way the tools are withheld, so the
            # model has to answer with what it already has instead of
            # requesting a call we would have to discard.
            last_affordable = budget.remaining <= 1
            final_round = round_index == MAX_ROUNDS - 1 or last_affordable

            message, usage = llm_client.complete(
                messages, tools=None if final_round else tools.TOOL_SCHEMAS
            )
            budget.planner += 1
            tool_calls = getattr(message, "tool_calls", None) or []

            steps.append({
                "module": "Planner",
                "round": round_index + 1,
                "usage": usage,
                # Both counts, on every planner step. The distinction between
                # rounds and calls is exactly what was being blurred, so the
                # trace states it rather than leaving a reader to infer it.
                "llm_calls": budget.as_dict(),
                "prompt": {
                    "system_prompt": prompts.SYSTEM_PROMPT,
                    "user_prompt": prompt.strip() if round_index == 0 else "(continued)",
                },
                "response": {
                    "content": message.content,
                    "tool_calls": [
                        {"name": c.function.name, "arguments": c.function.arguments}
                        for c in tool_calls
                    ],
                },
            })

            if not tool_calls:
                answer = (message.content or "").strip()
                if not answer:
                    return _error(
                        "The model returned an empty answer.", steps
                    )

                # The ceiling is checked, not requested. Two rounds of prompt
                # wording failed on it: one answer listed six films and then
                # quoted the "at most three" line underneath them, which is
                # what an instruction looks like when it has become boilerplate.
                # Counting is exact -- every film is named "Title (Year)" and
                # the catalog knows all 238 -- so the loop can simply refuse to
                # return an over-long answer, and pay one turn to fix it.
                named = catalog.labels_in(answer)

                # The exit gate. A film may be named only if verification
                # accepted it, and this is checked rather than requested for
                # the same reason the ceiling below is: an instruction the
                # model can read past is not a guarantee. It is what makes
                # greed self-defeating -- a shortcut that skips verification
                # produces films with no verdicts, and no answer can be built
                # out of them, so the cheap route stops being a route at all.
                #
                # Only armed once verification has run. A request that never
                # needed it -- "who are you", "what can you do", an impossible
                # premise -- has nothing to gate, and gating it would be the
                # loop refusing an answer for not citing evidence it was never
                # supposed to gather.
                unverified = ([f for f in named if f not in verified["accepted"]]
                              if verified["ran"] else [])
                if unverified and not corrected:
                    corrected = True
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You named {', '.join(unverified)}, which "
                            f"verification did not accept. "
                            + (f"Accepted: {', '.join(sorted(verified['accepted']))}. "
                               if verified["accepted"] else
                               "Nothing was accepted. ")
                            + "Only accepted films may be recommended, and the "
                            "count you state must match how many there are. "
                            "Rewrite the answer using only accepted films; if "
                            "there are none, say plainly that nothing could be "
                            "verified and how many films were checked. Do not "
                            "present an unresolved title as a near-miss."
                        ),
                    })
                    steps.append({
                        "module": "Planner",
                        "round": round_index + 1,
                        "llm_calls": budget.as_dict(),
                        "prompt": {"system_prompt": prompts.SYSTEM_PROMPT,
                                   "user_prompt": "(unverified films named, answer rejected)"},
                        "response": {"content": answer,
                                     "rejected": f"named {unverified}, "
                                                 f"accepted was "
                                                 f"{sorted(verified['accepted'])}"},
                    })
                    continue

                if len(named) > prompts.MAX_RECOMMENDATIONS and not corrected:
                    corrected = True
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You named {len(named)} films: "
                            f"{', '.join(named)}. The ceiling is "
                            f"{prompts.MAX_RECOMMENDATIONS} and it is not "
                            "negotiable. Send the same answer again keeping "
                            "only the best "
                            f"{prompts.MAX_RECOMMENDATIONS}, with the same "
                            "evidence for each, and say the list is not "
                            "complete."
                        ),
                    })
                    steps.append({
                        "module": "Planner",
                        "round": round_index + 1,
                        "prompt": {"system_prompt": prompts.SYSTEM_PROMPT,
                                   "user_prompt": "(ceiling exceeded, answer rejected)"},
                        "response": {"content": answer,
                                     "rejected": f"named {len(named)} films, "
                                                 f"ceiling is {prompts.MAX_RECOMMENDATIONS}"},
                    })
                    continue
                return {
                    "status": "ok",
                    "error": None,
                    "response": answer,
                    "steps": steps,
                }

            # The assistant message carrying the tool calls must be replayed
            # verbatim before the results, or the provider rejects the thread.
            messages.append(message)

            for call in tool_calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {}
                    result: dict[str, Any] = {
                        "error": f"Arguments were not valid JSON: {exc}"
                    }
                else:
                    ctx.calls_remaining = budget.remaining
                    result = tools.dispatch(name, arguments, ctx)

                if name == "verify_candidates" and not result.get("error"):
                    verified["ran"] = True
                    verified["accepted"] |= set(result.get("accepted") or [])
                    for bucket in ("accepted", "rejected", "unresolved"):
                        verified["seen"] |= set(result.get(bucket) or [])
                    # Each film cost a model call, and they were made inside
                    # the tool, so the budget has to learn about them here or
                    # the cap it enforces is a cap on the wrong number.
                    budget.verifier += int(result.get("verified") or 0)

                steps.append({
                    "module": tools.TRACE_NAMES.get(name, name),
                    "round": round_index + 1,
                    "prompt": {"tool": name, "arguments": arguments},
                    "scope": ctx.scope_note,
                    "llm_calls": budget.as_dict(),
                    "response": result,
                })

                # Plot text is the one payload worth a reader of its own. A
                # synopsis read is ~5,000 tokens and would otherwise sit in the
                # planner's context for the rest of the request, re-sent every
                # turn, to answer a question a 422-token prompt can answer
                # better. So the Observer reads it and the planner sees the
                # findings instead.
                #
                # The full text still goes into `steps` above, so the trace
                # stays completely auditable while the context stays small --
                # what a reviewer can inspect and what the model must carry are
                # deliberately different things.
                payload = result
                if (name == "read_synopses" and result.get("synopses")
                        and not budget.can_observe()):
                    # Out of budget for a second reader. The text still goes to
                    # the planner -- unadjudicated evidence beats none -- but it
                    # is labelled as unread, because the note the Observer path
                    # attaches ("every quote here was checked against the text")
                    # would be a lie about this payload.
                    steps.append({
                        "module": "Observer",
                        "round": round_index + 1,
                        "llm_calls": budget.as_dict(),
                        "prompt": {"system_prompt": observer.OBSERVER_PROMPT,
                                   "user_prompt": "(skipped: LLM call budget exhausted)"},
                        "response": {"findings": [],
                                     "skipped": f"{budget.total} of "
                                                f"{MAX_TOTAL_LLM_CALLS} calls spent; "
                                                f"the last is reserved for the answer"},
                    })
                    payload = {
                        **result,
                        "note": (
                            "Read but NOT adjudicated -- the call budget was spent, so "
                            "no Observer checked this text. Quote it only if you can see "
                            "the words yourself, and say it was unverified."
                        ),
                    }
                elif name == "read_synopses" and result.get("synopses"):
                    seen = observer.observe(
                        arguments.get("about") or prompt.strip(),
                        result["synopses"],
                    )
                    budget.observer += 1
                    steps.append({
                        "module": "Observer",
                        "round": round_index + 1,
                        "usage": seen.get("usage"),
                        "llm_calls": budget.as_dict(),
                        "prompt": {
                            "system_prompt": observer.OBSERVER_PROMPT,
                            "user_prompt": seen.get("prompt", ""),
                        },
                        "response": {"findings": seen["findings"],
                                     **({"error": seen["error"]} if seen.get("error") else {})},
                    })
                    # Only substitute when there is something to substitute. A
                    # failed Observer must not silently blank the evidence.
                    if seen["findings"]:
                        payload = {
                            "read": [e.get("film") for e in result["synopses"]],
                            "question": arguments.get("about"),
                            "findings": seen["findings"],
                            "note": (
                                "Read by the Observer. Every quote here is verbatim from "
                                "the plot text and was checked against it. A verdict of "
                                "`unclear` means the text does not settle the question -- "
                                "do not upgrade it. You may cite these quotes; you may not "
                                "assert anything about these films that is not in one."
                            ),
                        }

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(payload, default=str),
                })

        # Fell out of the loop without a final answer. Name the bound that
        # actually stopped it: "five rounds" is a misleading thing to tell
        # someone whose request ended because it spent eight model calls.
        return _error(
            f"Stopped after {budget.planner} planner rounds and "
            f"{budget.total} model calls "
            f"(bounds: {MAX_ROUNDS} rounds, {MAX_TOTAL_LLM_CALLS} calls) "
            "without a final answer. The query may be too open-ended for "
            "the catalog.",
            steps,
        )

    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as JSON
        return _error(f"{type(exc).__name__}: {exc}", steps)


def _error(
    message: str, steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Errors still return the steps gathered so far -- a failed run is often
    only diagnosable from how far it got."""
    return {
        "status": "error",
        "error": message,
        "response": None,
        "steps": steps,
    }
