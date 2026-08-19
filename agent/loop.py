"""MoviBot's agent loop: native tool calling, bounded, fully traced.

    plan -> call tools -> observe -> (repeat) -> answer

The model chooses which tools to call and when to stop. What it cannot choose
is how long to keep going: MAX_ROUNDS bounds the number of model turns, so
the worst case cost of a request is fixed no matter what the query asks for
or how the model behaves.

Cost per request, worst case:

    <= MAX_ROUNDS model calls   (paid)
    unlimited tool calls        (free -- local CSV reads and local embeddings)

Tools are free here because both backends default to local. That inverts the
usual tuning problem: there is no reason to skimp on tool calls, only on
model turns, which is exactly what this loop bounds.
"""

from __future__ import annotations

import json
from typing import Any

from agent import llm_client, prompts, tools

# Enough for filter -> search -> read -> answer, with one round spare for a
# correction. Queries needing more than this are usually a sign the model is
# thrashing, and cutting it off is cheaper than letting it continue.
MAX_ROUNDS = 5


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
            final_round = round_index == MAX_ROUNDS - 1

            # On the last round, withhold the tools so the model has to answer
            # with what it already has instead of requesting a call we would
            # have to discard.
            message, usage = llm_client.complete(
                messages, tools=None if final_round else tools.TOOL_SCHEMAS
            )
            tool_calls = getattr(message, "tool_calls", None) or []

            steps.append({
                "module": "Planner",
                "round": round_index + 1,
                "usage": usage,
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
                    result = tools.dispatch(name, arguments, ctx)

                steps.append({
                    "module": tools.TRACE_NAMES.get(name, name),
                    "round": round_index + 1,
                    "prompt": {"tool": name, "arguments": arguments},
                    "scope": ctx.scope_note,
                    "response": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })

        # Fell out of the loop: MAX_ROUNDS turns without a final answer.
        return _error(
            f"Stopped after {MAX_ROUNDS} rounds without a final answer. "
            "The query may be too open-ended for the catalog.",
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
