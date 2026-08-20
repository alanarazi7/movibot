#!/usr/bin/env python3
"""Checks the release gates that a reviewer can check mechanically. Free, offline.

    python scripts/check_gates.py

These are not defects, they are claims the project makes about itself that
would be embarrassing to have a grader disprove. Each one is cheap to assert
and expensive to notice by eye, which is exactly the kind that rots:

  G03  every module name is identical across the architecture diagram, the
       /api/agent_info description and the `module` field of a steps trace.
       The assignment requires this explicitly.
  G04  the configured model ids are the two the course provides.
  G09  every count shown to a user is reproducible from the shipped data.

G09 is the one that bites. Counts get typed into prose during a review, the
data is regenerated a week later, and nothing fails -- the page simply lies.
This file turns each displayed figure into an assertion against the artifacts
the app actually reads.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OK, BAD = "\033[32m✓\033[0m", "\033[31m✗\033[0m"

GUI = _ROOT / "public" / "index.html"
COURSE_TEXT_MODEL = "MB5R2CF-azure/gpt-5.4-mini"
COURSE_EMBED_MODEL = "MB5R2CF-azure/text-embedding-3-small"


def g03_module_names() -> list[str]:
    """One name per module, everywhere it appears."""
    from agent import tools

    failures = []
    info = json.loads((_ROOT / "agent_info.json").read_text())
    declared = set(info["architecture"]["tool_modules"])
    traced = set(tools.TRACE_NAMES.values())

    if declared != traced:
        failures.append(
            f"agent_info tool_modules {sorted(declared)} != TRACE_NAMES {sorted(traced)}"
        )

    # The diagram renders TRACE_NAMES[key], so the risk is a key that no longer
    # exists rather than a name that disagrees.
    src = (_ROOT / "scripts" / "generate_architecture_diagram.py").read_text()
    keys = re.findall(r'\(\s*"([a-z_]+)",\s*"[^"]*",\s*"\\U', src)
    if not keys:
        failures.append("could not read the diagram's tool list -- check the regex here")
    for key in keys:
        if key not in tools.TRACE_NAMES:
            failures.append(f"diagram names tool {key!r}, which is not in TRACE_NAMES")

    # Every schema the model is offered must have a trace name, or a step will
    # be logged under a module the diagram does not show.
    for schema in tools.TOOL_SCHEMAS:
        name = schema["function"]["name"]
        if name not in tools.TRACE_NAMES:
            failures.append(f"tool {name!r} is offered to the model but has no TRACE_NAME")

    # Planner and Observer are modules too, and they are the ones that make the
    # model calls the steps trace exists to record.
    if "Planner" not in (info["architecture"].get("planner_module") or ""):
        failures.append("agent_info does not name the Planner module")
    if "Observer" not in (info["architecture"].get("observer_module") or ""):
        failures.append("agent_info does not name the Observer module")

    # Every module the loop can log must be declared somewhere in agent_info,
    # or a trace shows a name the architecture never mentions.
    declared_all = declared | {"Planner", "Observer"}
    loop_src = (_ROOT / "agent" / "loop.py").read_text()
    for logged in set(re.findall(r'"module":\s*"(\w+)"', loop_src)):
        if logged not in declared_all:
            failures.append(f"loop.py logs module {logged!r}, which agent_info does not declare")

    return failures


def g04_models() -> list[str]:
    """The two deployments the course provides, and no others."""
    from agent import llm_client
    from rag import config as ragcfg

    failures = []
    if llm_client.model_name() != COURSE_TEXT_MODEL:
        failures.append(f"text model is {llm_client.model_name()}, expected {COURSE_TEXT_MODEL}")
    if ragcfg.EMBED_MODEL != COURSE_EMBED_MODEL:
        failures.append(f"embedding model is {ragcfg.EMBED_MODEL}, expected {COURSE_EMBED_MODEL}")
    return failures


def g09_counts() -> list[str]:
    """Every figure the GUI states, recomputed from the shipped artifacts.

    Anchored to the sentence each number appears in, not to the bare digits:
    the page is allowed to omit a count, it is not allowed to state a stale
    one. A claim that changes shape shows up here as a missing anchor, which
    is the right failure -- it means someone edited the prose and this file
    needs to follow.
    """
    from agent import catalog, tools
    from rag import store

    failures = []
    gui = GUI.read_text()

    def claim(template: str, value: object, what: str) -> None:
        """`template` must contain {} where the number goes."""
        anchor = template.format(value)
        if anchor in gui:
            return
        loose = template.format(r"[\d,]+")
        if re.search(loose, gui):
            failures.append(f"the GUI states a stale {what}; it is now {value}"
                            f"  (looking for {anchor!r})")
        else:
            failures.append(f"the claim about {what} is no longer phrased as "
                            f"{template.format('N')!r} -- update this check")

    cov = store.coverage()
    claim("{} Disney and Pixar feature films", len(catalog.movies()), "catalog size")
    claim("{} vectors is not a search problem", f"{cov['chunks']:,}", "passage count")
    claim("{}-dim", cov["dim"], "embedding dimension")

    death = tools.screen_out(vocabulary="death")
    claim("{} films clear on the death screen", death["clear"], "death-screen clear count")

    ctx = tools.ToolContext()
    animated = tools.filter_catalog(ctx=ctx, genres=["Animation"])["matched"]
    a_death = tools.screen_out(vocabulary="death", ctx=ctx)
    claim("{} animated films split", animated, "animated film count")
    claim("split {} clear", a_death["clear"], "animated clear count")
    claim("{} flagged", a_death["flagged"], "animated flagged count")

    mpst = int(catalog.movies()["has_mpst_synopsis"].sum())
    summary = json.loads((_ROOT / "public" / "data" / "catalog.json").read_text())["summary"]
    if summary.get("with_synopsis") != mpst:
        failures.append(f"catalog.json with_synopsis is {summary.get('with_synopsis')}, "
                        f"has_mpst_synopsis is {mpst}")

    # The number is right; the label around it is the trap. 159 films have an
    # MPST synopsis and 234 have plot text of some kind, so calling 159 "films
    # with a plot synopsis" implies the other 79 have none. That exact framing
    # is what told the system prompt it could not verify story claims about
    # them, which was wrong for two months.
    plot_bearing = len({p["movie_id"] for p in store.plot_passages()
                        if p["source"] in ("mpst", "wiki_plot")})
    if "with a plot synopsis" in gui:
        failures.append(
            f"the Data tab labels the MPST count 'with a plot synopsis', implying the other "
            f"{len(catalog.movies()) - mpst} films have none -- {plot_bearing} have plot text"
        )
    return failures


def main() -> int:
    total = 0
    for name, fn, what in [
        ("G03", g03_module_names, "module names agree across diagram, agent_info and steps"),
        ("G04", g04_models, "the course-provided model deployments are the ones configured"),
        ("G09", g09_counts, "every displayed count comes from the shipped data"),
    ]:
        failures = fn()
        total += len(failures)
        mark = OK if not failures else BAD
        print(f"  {mark} {name}  {what}")
        for line in failures:
            print(f"        {line}")

    if total:
        print(f"\n{total} gate failure(s).")
        return 1
    print("\nAll mechanical gates pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
