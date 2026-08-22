"""MoviBot: decompose, narrow, verify, answer.

    request -> Decomposer -> the operations it asked for -> Verifier x N
            -> Answerer -> checked against what was verified -> reply

The Decomposer reads the request and decides what evidence it needs. The
operations it asks for narrow the catalog to candidates without a text-model
call. The Verifier reads one film at a time, against every condition at once,
until enough have passed or the candidates run out. The Answerer drafts the
reply from what was verified and cannot reach anything else.

**What varies is what the request needs.** A question answerable from columns
asks for one operation; one mixing a studio, an absence and two story
conditions asks for three and reads ten films. What does not vary is the
standard of evidence: a film is named only if its own plot text was checked
against every condition of the request.

Cost:

    1 text-model call      Decomposer
    <= MAX_VERIFICATIONS   one per film read, stopping early once enough pass
    1 text-model call      Answerer, plus at most one more if the check rejects
    <= MAX_TOTAL_LLM_CALLS in total, enforced here rather than requested

The last thing that happens is a check, not a model call. A reply may not name
a film that was not verified, omit one that was, name more films than were
asked for, or offer a follow-up this API cannot honour.
"""

from __future__ import annotations

from typing import Any

from agent import answerer, catalog, decomposer, llm_client, tools, verifier

# The whole request's budget, counting every role. Enforced here and threaded
# into the stages that spend inside a single call, so a walk cannot overrun it
# between checks.
MAX_TOTAL_LLM_CALLS = 16

# There is no list of closing-offer phrasings here any more. Matching answers
# against "if you want", "let me know", "happy to" caught those and not the
# next wording, and a rule that only holds for remembered phrasings is not a
# rule. The Answerer is told the interaction is stateless; what the code checks
# is what it can check without knowing any English in advance -- which films
# were named, against which were verified.


class _Budget:
    """Counts every model call, so no role can hide inside another."""

    def __init__(self) -> None:
        self.decomposer = 0
        self.verifier = 0
        self.answerer = 0

    @property
    def total(self) -> int:
        return self.decomposer + self.verifier + self.answerer

    @property
    def remaining(self) -> int:
        return MAX_TOTAL_LLM_CALLS - self.total

    def as_dict(self) -> dict[str, int]:
        return {"decomposer": self.decomposer, "verifier": self.verifier,
                "answerer": self.answerer, "total": self.total,
                "cap": MAX_TOTAL_LLM_CALLS}


def execute(prompt: str) -> dict[str, Any]:
    """Answer one request.

    Returns the /api/execute contract, whose top-level fields are fixed at
    four. Everything diagnostic therefore lives inside a step: every model
    call is one step carrying its own system and user prompt, and every
    deterministic stage is one step carrying its arguments and its result, so
    the whole path is inspectable from the response alone.
    """
    steps: list[dict[str, Any]] = []
    budget = _Budget()

    if not prompt or not prompt.strip():
        return _error("The 'prompt' field is required.", steps)

    if not llm_client.is_configured():
        return _error(
            "MoviBot cannot compose an answer: OPENAI_API_KEY is unset or "
            "still a placeholder value. The catalog and every local read "
            "are unaffected.",
            steps,
        )

    try:
        # ---- decompose -------------------------------------------------
        got = decomposer.decompose(prompt)
        budget.decomposer += 1
        plan = got.get("plan")
        steps.append({
            "module": "Decomposer",
            "usage": got.get("usage"),
            "llm_calls": budget.as_dict(),
            "prompt": {"system_prompt": decomposer.DECOMPOSER_PROMPT,
                       "user_prompt": prompt.strip()},
            "response": plan if plan else {"error": got.get("error"),
                                           "raw": got.get("raw", "")},
        })
        if not plan:
            return _error(got.get("error") or "The request could not be read.", steps)

        # A refusal and a question about the agent both skip every stage: the
        # decomposer already wrote what there is to say, and running a filter
        # to confirm a film does not exist would be theatre.
        if plan["outcome"] in ("refuse", "about_self") and plan["message"]:
            return {"status": "ok", "error": None,
                    "response": plan["message"], "steps": steps}

        # ---- execute ---------------------------------------------------
        ctx = tools.ToolContext()
        evidence: dict[str, Any] = {"conditions": plan["conditions"]}

        result = tools.filter_catalog(ctx=ctx, **plan["filter"])
        steps.append(_tool_step("CatalogFilter", plan["filter"], ctx, result, budget))
        evidence["filtered_to"] = result.get("matched")
        if not result.get("matched"):
            evidence["note"] = "No film matches the catalog constraints."

        # Two retrievals over two corpora: what happens, and what a film is
        # written about as being. Their rankings merge into one shortlist, so
        # a film that places on both comes first.
        for field, fn, module in (
            ("search_plots", tools.retrieve_plots, "PlotRetrieval"),
            ("search_metadata", tools.retrieve_metadata, "MetadataRetrieval"),
        ):
            if plan[field] and (ctx.working_set is None or ctx.working_set):
                result = fn(conditions=plan[field], ctx=ctx)
                steps.append(_tool_step(module, {"conditions": plan[field]},
                                        ctx, result, budget))
                evidence.setdefault("retrieval", {})[module] = {
                    k: result.get(k) for k in
                    ("conditions", "candidates", "matching_every_condition")
                }

        # ---- verify ----------------------------------------------------
        if plan["verify"]:
            ctx.calls_remaining = budget.remaining - 1   # reserve the Answerer
            walk = tools.verify_candidates(request=plan["request"],
                                           conditions=plan["verify"],
                                           max_accept=plan["max_films"], ctx=ctx)
            budget.verifier += int(walk.get("verified") or 0)
            steps.append(_tool_step("CandidateWalk",
                                    {"conditions": plan["verify"],
                                     "max_accept": plan["max_films"]},
                                    ctx, {k: v for k, v in walk.items()
                                          if k != "verdicts"}, budget))
            for row in walk.get("verdicts") or []:
                steps.append({
                    "module": "Verifier",
                    "usage": row.get("usage"),
                    "llm_calls": budget.as_dict(),
                    "prompt": {"system_prompt": verifier.VERIFIER_PROMPT,
                               "user_prompt": row.get("prompt", "")},
                    "response": {"film": row.get("film"),
                                 "accepted": row.get("accepted"),
                                 "findings": row.get("findings")},
                })
            accepted = list(walk.get("accepted") or [])
            evidence.update({k: walk.get(k) for k in
                             ("accepted", "rejected", "unresolved", "verified",
                              "unsettleable", "not_verifiable")
                             if walk.get(k)})
            evidence["verdicts"] = _thin_verdicts(walk.get("verdicts") or [])
        else:
            # Nothing for a reader to settle, so the ranking is the answer and
            # the best-rated survivors are what there is to recommend.
            # Nothing here needs plot text, so nothing was verified and the
            # word must not appear. Reported under its own key: calling this
            # `accepted` had the Answerer write "3 verified" about films
            # nothing had looked at, and quote the explanation as if it were
            # evidence.
            accepted = _best(ctx, plan["max_films"])
            evidence["ranked_not_verified"] = accepted
            evidence["basis"] = (
                "NOTHING WAS VERIFIED. No requirement was given to check, so no "
                "film was read. "
                + ("These are the top of the retrieval ranking. A ranking is not "
                   "evidence: say they are the closest matches, not that they "
                   "satisfy anything."
                   if ctx.shortlist else
                   "These are the best-rated films matching the catalog "
                   "constraints, and that is all you may say about them.")
            )

        # ---- answer, then check ----------------------------------------
        return _compose(prompt, evidence, accepted, plan, steps, budget)

    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as JSON
        return _error(f"{type(exc).__name__}: {exc}", steps)


def _compose(prompt: str, evidence: dict[str, Any], accepted: list[str],
             plan: dict[str, Any], steps: list[dict[str, Any]],
             budget: "_Budget") -> dict[str, Any]:
    """Write the reply, check it, and pay for at most one correction."""
    correction: str | None = None

    for attempt in range(2):
        payload = dict(evidence)
        if correction:
            payload["correction"] = correction

        got = answerer.answer(prompt, payload)
        budget.answerer += 1
        text = got["text"]
        steps.append({
            "module": "Answerer",
            "usage": got.get("usage"),
            "llm_calls": budget.as_dict(),
            "prompt": {"system_prompt": answerer.ANSWERER_PROMPT,
                       "user_prompt": got.get("prompt", "")},
            "response": {"content": text,
                         **({"correction_of": correction} if correction else {})},
        })

        if not text:
            return _error("The model returned an empty answer.", steps)

        problem = _check(text, accepted, plan)
        if problem is None or attempt == 1 or budget.remaining < 1:
            # Out of attempts or out of budget: return what there is rather
            # than an error. A flawed answer with its trace beats a timeout.
            return {"status": "ok", "error": None, "response": text,
                    "steps": steps,
                    **({} if problem is None else {})}

        correction = problem
        steps.append({
            "module": "AnswerCheck",
            "llm_calls": budget.as_dict(),
            "prompt": {"tool": "answer_check", "arguments": {"accepted": accepted}},
            "response": {"rejected": problem, "content": text},
        })

    return {"status": "ok", "error": None, "response": text, "steps": steps}


def _check(text: str, accepted: list[str], plan: dict[str, Any]) -> str | None:
    """What is wrong with this answer, or None.

    Checked rather than requested, for the reason the recommendation ceiling
    taught: an instruction the model can read past is not a guarantee.
    """
    named = catalog.labels_in(text)
    allowed = set(accepted)

    unverified = [f for f in named if f not in allowed]
    if unverified:
        return (f"You named {', '.join(unverified)}, which was not accepted. "
                + (f"Accepted: {', '.join(accepted)}. " if accepted
                   else "Nothing was accepted. ")
                + "Only accepted films may be named. If none were, say plainly "
                  "that nothing could be verified and how many were checked.")

    missed = [f for f in accepted if f not in named]
    if accepted and not named:
        return (f"Verification accepted {', '.join(accepted)} and your answer "
                "names none of them. Recommend the accepted films, citing the "
                "quote each verdict came with.")

    if len(named) > plan["max_films"]:
        return (f"You named {len(named)} films and the limit is "
                f"{plan['max_films']}. Keep the best {plan['max_films']}, with "
                "the same evidence for each.")

    if len(named) < plan["max_films"] and len(missed) > 0:
        return (f"You named {len(named)} of {len(accepted)} accepted films and "
                f"left out {', '.join(missed)}. Every accepted film qualified; "
                "name them all.")

    return None


def _best(ctx: "tools.ToolContext", limit: int) -> list[str]:
    """The films to fall back on when nothing was verified.

    The retrieved ranking when there is one, and only otherwise the best-rated
    survivors of the filter. Reading the working set first threw the retrieval
    away: "a strong female character, besides Frozen and Moana" retrieved
    twenty candidates and answered with the three best-rated films in the
    catalog, which is the same answer it would have given to no request at
    all.
    """
    if ctx.shortlist:
        ids = list(ctx.shortlist)[:limit]
    else:
        ids = sorted(ctx.working_set or (),
                     key=lambda i: -(catalog.rating_of(i) or 0.0))[:limit]
    return [catalog.label_of(i) for i in ids if catalog.label_of(i)]


def _thin_verdicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The verdict matrix without the plot text, which the Answerer must not
    quote from directly -- only the sentences the Verifier already vouched for."""
    return [{"film": r.get("film"), "accepted": r.get("accepted"),
             "findings": r.get("findings")} for r in rows]


def _tool_step(module: str, arguments: dict[str, Any], ctx: "tools.ToolContext",
               result: dict[str, Any], budget: "_Budget") -> dict[str, Any]:
    """A deterministic stage, traced with what it was given and what it left."""
    return {"module": module, "prompt": {"tool": module, "arguments": arguments},
            "scope": ctx.scope_note, "llm_calls": budget.as_dict(),
            "response": result}


def _error(message: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Errors keep the steps gathered so far -- a failed run is often only
    diagnosable from how far it got."""
    return {"status": "error", "error": message, "response": None, "steps": steps}
