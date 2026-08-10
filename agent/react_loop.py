"""Orchestrates the ReAct loop: Reason -> Act -> Observe -> Stop? -> Synthesizer."""

from typing import Any

from agent import llm_client, prompts
from agent.tools import catalog_filter, plot_search, scene_search, external_context

MAX_ITERATIONS = 6


def execute(prompt: str) -> dict[str, Any]:
    """Run the full ReAct loop end-to-end.

    Returns: {"status": "ok"|"error", "error": str|None, "response": str, "steps": list}
    """
    steps = []
    gathered = {
        "user_prompt": prompt,
        "candidates": None,
        "actions_taken": set(),
        "verified": []
    }

    client = llm_client.get_mock_client()

    try:
        # Main ReAct loop
        for iteration in range(MAX_ITERATIONS):
            # Reasoner: decide next action
            decision = client.reason_next_action(prompt, gathered)

            reasoner_prompt_user = _build_reasoner_trace_prompt(prompt, gathered)
            steps.append({
                "module": "Reasoner",
                "prompt": {
                    "system_prompt": prompts.REASONER_SYSTEM_PROMPT,
                    "user_prompt": reasoner_prompt_user
                },
                "response": decision
            })

            action = decision.get("action")

            # Check for termination
            if action == "Stop":
                break
            if action in gathered["actions_taken"]:
                # Already ran this tool, skip
                break
            if action not in ["CatalogFilter", "PlotSearch", "SceneSearch", "ExternalContext"]:
                # Unknown action
                break

            # Dispatch to tool
            try:
                if action == "CatalogFilter":
                    constraints = client.translate_catalog_constraints(prompt)
                    result = catalog_filter.run(constraints)
                    steps.append({
                        "module": "CatalogFilter",
                        "prompt": {
                            "system_prompt": prompts.CATALOG_FILTER_SYSTEM_PROMPT,
                            "user_prompt": f"Constraints extracted from user request: {constraints}"
                        },
                        "response": result
                    })
                    gathered["candidates"] = result.get("candidates", [])

                elif action == "PlotSearch":
                    query = client.build_plot_query(prompt)
                    candidate_ids = [c["movie_id"] for c in gathered.get("candidates", [])] or None
                    result = plot_search.run(query, top_k=20, candidate_ids=candidate_ids)
                    steps.append({
                        "module": "PlotSearch",
                        "prompt": {
                            "system_prompt": prompts.PLOT_SEARCH_SYSTEM_PROMPT,
                            "user_prompt": f"Search query: {query}"
                        },
                        "response": {
                            "query": query,
                            "matches": result
                        }
                    })
                    # Re-rank candidates to plot search matches
                    if result:
                        matched_ids = {r["movie_id"] for r in result}
                        gathered["candidates"] = [c for c in gathered.get("candidates", []) if c["movie_id"] in matched_ids]
                    else:
                        gathered["candidates"] = []

                elif action == "SceneSearch":
                    # Check all current candidates for risky-event constraints
                    checked = []
                    for candidate in gathered.get("candidates", []):
                        verdict = scene_search.run(candidate["title"], prompt)
                        checked.append(verdict)
                    gathered["verified"].extend(checked)

                    steps.append({
                        "module": "SceneSearch",
                        "prompt": {
                            "system_prompt": prompts.SCENE_SEARCH_SYSTEM_PROMPT,
                            "user_prompt": f"Check these titles for risky-event constraints: {[c['title'] for c in gathered.get('candidates', [])]}"
                        },
                        "response": {
                            "checked": checked
                        }
                    })

                    # Drop candidates that failed the check (satisfied=False)
                    failing_titles = {v["title"] for v in checked if v.get("satisfied") is False}
                    gathered["candidates"] = [c for c in gathered.get("candidates", []) if c["title"] not in failing_titles]

                elif action == "ExternalContext":
                    # Check all current candidates for tone/reception constraints
                    checked = []
                    for candidate in gathered.get("candidates", []):
                        verdict = external_context.run(candidate["title"], prompt)
                        checked.append(verdict)
                    gathered["verified"].extend(checked)

                    steps.append({
                        "module": "ExternalContext",
                        "prompt": {
                            "system_prompt": prompts.EXTERNAL_CONTEXT_SYSTEM_PROMPT,
                            "user_prompt": f"Check these titles for tone/audience constraints: {[c['title'] for c in gathered.get('candidates', [])]}"
                        },
                        "response": {
                            "checked": checked
                        }
                    })

                    # Drop candidates that failed the check
                    failing_titles = {v["title"] for v in checked if v.get("satisfied") is False}
                    gathered["candidates"] = [c for c in gathered.get("candidates", []) if c["title"] not in failing_titles]

            except Exception as e:
                # Tool execution failed; log but continue with graceful degradation
                pass

            gathered["actions_taken"].add(action)

        # Synthesizer: compose final answer
        answer = client.synthesize_answer(prompt, gathered)
        steps.append({
            "module": "Synthesizer",
            "prompt": {
                "system_prompt": prompts.SYNTHESIZER_SYSTEM_PROMPT,
                "user_prompt": _build_synth_trace_prompt(gathered)
            },
            "response": {
                "answer": answer
            }
        })

        return {
            "status": "ok",
            "error": None,
            "response": answer,
            "steps": steps
        }

    except Exception as exc:
        # Top-level error; return whatever steps we've collected so far
        return {
            "status": "error",
            "error": str(exc),
            "response": None,
            "steps": steps
        }


def _build_reasoner_trace_prompt(user_prompt: str, gathered: dict[str, Any]) -> str:
    """Build the Reasoner's trace prompt for logging."""
    lines = [f"Original user request: {user_prompt}"]

    candidates = gathered.get("candidates")
    if candidates:
        lines.append(f"Current candidates: {len(candidates)} movies")
        lines.append(f"  Titles: {', '.join(c['title'] for c in candidates[:3])}")
        if len(candidates) > 3:
            lines.append(f"  ... and {len(candidates) - 3} more")

    verified = gathered.get("verified", [])
    if verified:
        lines.append(f"Verified constraints: {len(verified)} checks so far")

    actions = gathered.get("actions_taken", set())
    if actions:
        lines.append(f"Tools already run: {', '.join(sorted(actions))}")

    return "\n".join(lines)


def _build_synth_trace_prompt(gathered: dict[str, Any]) -> str:
    """Build the Synthesizer's trace prompt for logging."""
    candidates = gathered.get("candidates", [])
    verified = gathered.get("verified", [])

    lines = [f"Gathered {len(candidates)} final candidate(s)"]
    if candidates:
        for c in candidates[:5]:
            lines.append(f"  • {c['title']} ({c.get('release_year', '?')})")

    if verified:
        lines.append(f"\nVerified {len(verified)} constraint checks:")
        for v in verified[:5]:
            status = "✓" if v.get("satisfied") else ("?" if v.get("satisfied") is None else "✗")
            lines.append(f"  {status} {v.get('title', '?')}: {v.get('constraint', '')}")

    return "\n".join(lines)
