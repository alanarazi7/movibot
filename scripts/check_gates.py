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

    # Planner and the readers are modules too, and they are the ones that make
    # the model calls the steps trace exists to record. They are declared
    # apart from the tools on purpose: a reader is sent by a tool and cannot
    # be called by the planner, and collapsing the two is what let a tool step
    # be logged under the name "Verifier".
    if "Planner" not in (info["architecture"].get("planner_module") or ""):
        failures.append("agent_info does not name the Planner module")

    readers = info["architecture"].get("reader_modules") or {}
    for name in ("Observer", "Verifier"):
        if name not in readers:
            failures.append(f"agent_info does not declare the {name} reader module")
    for name in readers:
        if name in declared:
            failures.append(f"{name!r} is declared both as a tool and as a reader; a "
                            f"reader is sent by a tool, not called by the planner")

    # The worked examples in agent_info are a trace like any other, and they go
    # stale the moment a module is renamed -- which is exactly what happened
    # when PlotSearch became ShortlistFusion and the examples kept naming a
    # module that no longer exists. Nothing checked them, because this gate
    # only ever read the declarations.
    declared_all = declared | {"Planner"} | set(readers)
    for i, example in enumerate(info.get("prompt_examples") or []):
        for step in example.get("steps") or []:
            module = step.get("module")
            if module and module not in declared_all:
                failures.append(
                    f"prompt_examples[{i}] traces module {module!r}, which is not a "
                    f"module any more -- regenerate the examples with "
                    f"scripts/capture_examples.py"
                )
                break

    # Every module the loop can log must be declared somewhere in agent_info,
    # or a trace shows a name the architecture never mentions.
    loop_src = (_ROOT / "agent" / "loop.py").read_text()
    for logged in set(re.findall(r'"module":\s*"(\w+)"', loop_src)):
        if logged not in declared_all:
            failures.append(f"loop.py logs module {logged!r}, which agent_info does not declare")

    return failures


def g05_runtime_is_structured() -> list[str]:
    """Runtime is reachable as a filter argument, exact, and cited back.

    The catalog has always held a runtime for all 316 films, but no filter
    argument exposed it, so "under 110 minutes" fell through to plot search --
    which cannot establish a runtime, because no synopsis states one. The
    agent then reported a deterministic fact as unverifiable. This gate holds
    the column, the argument, the schema and the returned field together, and
    checks the bounds the prompt states in prose.
    """
    from agent import catalog, prompts, tools

    failures = []
    df = catalog.movies()

    missing = int(df["runtime_minutes"].isna().sum())
    if missing:
        failures.append(f"{missing} films have no runtime_minutes; the filter would "
                        f"silently drop them")

    # The argument exists, on the function and in the schema the model reads.
    schema = next(s for s in tools.TOOL_SCHEMAS
                  if s["function"]["name"] == "filter_catalog")
    props = schema["function"]["parameters"]["properties"]
    for arg in ("runtime_min", "runtime_max"):
        if arg not in props:
            failures.append(f"filter_catalog's schema does not offer {arg!r}, so the "
                            f"model cannot use it however well the function works")

    # Inclusive at both ends, and the schema must say so -- an off-by-one here
    # returns a 110-minute film to someone who asked for under 110.
    lo, hi = int(df["runtime_minutes"].min()), int(df["runtime_minutes"].max())
    at_hi = tools.filter_catalog(runtime_max=hi, list_all=True)
    below_hi = tools.filter_catalog(runtime_max=hi - 1, list_all=True)
    if at_hi["matched"] != len(df):
        failures.append(f"runtime_max={hi} matched {at_hi['matched']} of {len(df)}; "
                        f"the bound is meant to be inclusive")
    if below_hi["matched"] >= at_hi["matched"]:
        failures.append(f"runtime_max={hi - 1} matched {below_hi['matched']}, not fewer "
                        f"than {at_hi['matched']} -- the filter is not being applied")
    if "INCLUSIVE" not in props.get("runtime_max", {}).get("description", ""):
        failures.append("runtime_max's description no longer says it is inclusive, which "
                        "is what stops the model passing 110 for 'under 110'")

    # Below the floor is a real absence, not an empty search.
    if tools.filter_catalog(runtime_max=lo - 1)["matched"] != 0:
        failures.append(f"runtime_max={lo - 1} matched something, but the shortest film "
                        f"is {lo} minutes")

    # Every returned film carries its runtime, or the planner cannot cite one.
    sample = tools.filter_catalog(year_min=2010, list_all=True)["films"]
    if not all(isinstance(f.get("runtime"), int) for f in sample):
        failures.append("filter_catalog returns films without an integer `runtime`, so "
                        "the planner can filter on length but not show it")

    # The bounds are stated in prose in two places. Typed-in numbers rot.
    for where, text in [("the system prompt", prompts.SYSTEM_PROMPT),
                        ("filter_catalog's description",
                         schema["function"]["description"])]:
        if f"{lo} to {hi} minutes" not in text:
            failures.append(f"{where} no longer states the real runtime range "
                            f"({lo} to {hi} minutes)")
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


def g06_call_budget() -> list[str]:
    """The total-call cap holds against a model that never stops calling tools.

    Free, offline and hermetic: the model, the Observer and the tool dispatch
    are all stubbed, so this drives the real loop with a fake adversary rather
    than paying to discover the bound. Stubbing dispatch matters -- the first
    version of this gate called the real read_synopses, which needs an
    embedding, raised without credentials, and returned an error the loop
    treated as "nothing to observe". The gate passed while exercising none of
    the code it exists to check.

    The adversary is the worst case on purpose: every planner turn asks to read
    synopses, so every round tries to spend two calls. What is asserted is that
    the cap binds the SUM. A round limit alone lets this exact adversary spend
    eleven calls while honestly reporting five rounds.
    """
    from agent import llm_client, loop, observer, tools

    failures = []

    class _Call:
        def __init__(self, name, arguments):
            self.id = "stub"
            self.function = type("F", (), {"name": name, "arguments": arguments})()

    class _Msg:
        def __init__(self, calls):
            self.content = None if calls else "Final answer, no film named."
            self.tool_calls = calls

    calls = {"planner": 0, "observer": 0}

    def fake_complete(messages, tools=None):
        calls["planner"] += 1
        # Keep asking to read, forever, unless the loop withholds the schemas.
        if tools is None:
            return _Msg([]), {"total_tokens": 0}
        return _Msg([_Call("read_synopses",
                           '{"films": ["Inside Out (2015)"], "about": "anything"}')]), \
            {"total_tokens": 0}

    def fake_dispatch(name, arguments, ctx=None):
        return {"synopses": [{"film": "Inside Out (2015)",
                              "synopsis": "Riley moves to San Francisco."}]}

    def fake_observe(question, synopses):
        calls["observer"] += 1
        return {"findings": [{"film": "Inside Out (2015)", "verdict": "unclear",
                              "quote": ""}], "usage": {"total_tokens": 0},
                "prompt": "stub"}

    saved = (llm_client.complete, observer.observe, tools.dispatch,
             llm_client.is_configured)
    llm_client.complete = fake_complete
    observer.observe = fake_observe
    tools.dispatch = fake_dispatch
    llm_client.is_configured = lambda: True
    try:
        result = loop.execute("read everything you can, repeatedly")
    finally:
        (llm_client.complete, observer.observe, tools.dispatch,
         llm_client.is_configured) = saved

    total = calls["planner"] + calls["observer"]
    cap = loop.MAX_TOTAL_LLM_CALLS

    # The gate must exercise the path it claims to. An adversary that never
    # reaches the Observer proves nothing about a cap that spans both.
    if calls["observer"] == 0:
        failures.append("the adversary never triggered an Observer call, so this gate "
                        "is not testing the sum it claims to test")
    if total > cap:
        failures.append(f"the adversary spent {total} model calls "
                        f"({calls['planner']} planner + {calls['observer']} observer) "
                        f"against a cap of {cap}")
    if calls["planner"] > loop.MAX_ROUNDS:
        failures.append(f"{calls['planner']} planner calls against MAX_ROUNDS "
                        f"{loop.MAX_ROUNDS}")

    # The reserve: the budget must not run out mid-read, leaving nobody to
    # write the answer. Spending the last call on an Observer buys evidence
    # the user never receives.
    if result.get("status") != "ok" or not result.get("response"):
        failures.append(f"the budget ran out without an answer: status "
                        f"{result.get('status')!r}, error {result.get('error')!r}")

    # Every planner step states both counts, or the trace re-blurs the
    # distinction this fix exists to draw.
    planner_steps = [s for s in result.get("steps", []) if s["module"] == "Planner"]
    if not planner_steps:
        failures.append("no Planner steps in the trace")
    for s in planner_steps:
        if set(s.get("llm_calls") or {}) != {"planner", "observer", "verifier",
                                             "total", "cap"}:
            failures.append(f"a Planner step reports llm_calls as {s.get('llm_calls')!r}")
            break

    # And the published figure must be the enforced one.
    info = json.loads((_ROOT / "agent_info.json").read_text())
    published = info["architecture"].get("max_total_llm_calls_per_request")
    if published != cap:
        failures.append(f"agent_info publishes max_total_llm_calls_per_request "
                        f"{published!r}, code enforces {cap}")
    if info["architecture"].get("max_planner_rounds_per_request") != loop.MAX_ROUNDS:
        failures.append("agent_info does not publish the planner-round bound separately "
                        "-- rounds and calls are different quantities")
    return failures


def g07_fusion_beats_greed() -> list[str]:
    """Coverage outranks average rank, and the greedy answer loses. Free.

    Pure arithmetic on synthetic lists, so it costs nothing and can assert the
    exact property the mechanism exists for: a film satisfying every condition
    moderately must beat a film satisfying one condition perfectly.

    The second case is the one worth keeping. Averaging a penalty rank for a
    missing condition -- the obvious implementation -- makes a film ranked
    1st, 1st and absent (7.7 with a penalty of 21) beat one ranked 10th, 10th
    and 10th (10.0). That is the greedy answer arriving by arithmetic, and it
    is why the ordering is tiered by coverage first.
    """
    from agent import shortlist

    failures = []

    # Satisfying everything moderately beats satisfying one thing perfectly.
    fused = shortlist.fuse({"a": [7, 1], "b": [7, 2], "c": [7]})
    if not fused or fused[0].movie_id != 7:
        failures.append(f"the film matching all three conditions did not rank first: "
                        f"{[(c.movie_id, c.covered) for c in fused]}")

    # Coverage must dominate average rank, not merely contribute to it.
    fused = shortlist.fuse({"a": [1, 9], "b": [1, 9], "c": [9]})
    order = [c.movie_id for c in fused]
    if order[:1] != [9]:
        failures.append(f"film 9 covers 3/3 at rank 9 and film 1 covers 2/3 at rank 1; "
                        f"order was {order} -- coverage is not dominating")

    # Within a coverage tier, average rank decides.
    fused = shortlist.fuse({"a": [1, 2], "b": [1, 2]})
    if [c.movie_id for c in fused] != [1, 2]:
        failures.append("within one coverage tier the better average rank did not win")

    # Ties break on the catalog rating, the only ordering this project uses.
    fused = shortlist.fuse({"a": [1, 2]}, ratings={1: 5.0, 2: 9.0})
    fused = shortlist.fuse({"a": [1], "b": [2]}, ratings={1: 5.0, 2: 9.0})
    if [c.movie_id for c in fused] != [2, 1]:
        failures.append("equal coverage and equal rank did not break on rating")

    # A rank of None must be reported, not omitted: a condition missing from
    # the row reads as an oversight rather than as a finding.
    row = shortlist.explain(fused[0], ["a", "b"], "Film (2000)")
    if set(row["ranks"]) != {"a", "b"}:
        failures.append(f"explain() dropped a condition instead of reporting None: "
                        f"{row['ranks']}")

    # The tool is offered to the model and traced.
    from agent import tools
    if "build_shortlist" not in {s["function"]["name"] for s in tools.TOOL_SCHEMAS}:
        failures.append("build_shortlist is not in TOOL_SCHEMAS, so the model cannot use it")
    if "build_shortlist" not in tools.TRACE_NAMES:
        failures.append("build_shortlist has no TRACE_NAME")
    return failures


def g08_no_follow_up_offers() -> list[str]:
    """Closing offers are caught, and honest first-person reports are not. Free.

    The interaction is stateless: each request arrives with no memory of any
    other, so "want two more?" is a promise nothing can keep. It is also
    usually evidence of a second failure, because films offered for later were
    films that qualified now -- which is exactly what "The Lion King ... If you
    want, I can give you two more 1990s Disney options" was.

    The false-positive half of this gate matters as much as the other. "I can
    only stand behind one title" is an honest report the agent must be able to
    make, and a detector that rejected it would spend a correction turn
    punishing the most careful answer in the set.
    """
    from agent import loop

    offers = [
        "If you want, I can give you two more 1990s Disney options in the same vein.",
        "Want two more? Just say the word.",
        "Let me know if you want something funnier.",
        "Would you like me to narrow this further?",
        "Happy to refine these.",
    ]
    honest = [
        "I can only stand behind one title here.",
        "For more options, submit a new request with the criteria you want.",
        "Only one film verified; the catalog stops at 2017.",
        "The search covered 76 films, not the whole catalog.",
        "Nothing under 47 minutes exists in this catalog.",
    ]

    failures = []
    for text in offers:
        if loop._closing_offer(text) is None:
            failures.append(f"a follow-up offer went undetected: {text!r}")
    for text in honest:
        hit = loop._closing_offer(text)
        if hit is not None:
            failures.append(f"an honest report was flagged as an offer on {hit!r}: {text!r}")

    # The prompt has to say it too. The check is the backstop, not the policy:
    # a model corrected every time costs a turn every time.
    from agent import prompts
    if "no conversation" not in prompts.SYSTEM_PROMPT.lower():
        failures.append("the system prompt no longer tells the model the interaction "
                        "is stateless, so the gate below would fire on every answer")
    return failures


def g10_verdicts_need_evidence() -> list[str]:
    """A verdict is only as good as the quote under it. Free, offline.

    Three ways a `yes` was reaching the accepted list without support, all
    observed on "a movie with an animal that wears a hat":

      the quote was real but showed half the requirement -- a butler's hat
      for "an animal that wears a hat";
      the note argued against the verdict in the same object -- "Edgar is a
      butler, not an animal", "later scenes depict him with a hat";
      a stray object carrying only a note, no requirement and no verdict,
      counted as a finding and rendered as "undefined: undefined".

    The model is stubbed here, so this asserts the post-processing rather than
    the model's judgement, which is the half that can be guaranteed.
    """
    from agent import llm_client, verifier

    failures = []

    class _M:
        def __init__(self, content): self.content = content

    def run(payload, conditions, text):
        real = llm_client.complete
        llm_client.complete = lambda msgs, tools=None: (_M(json.dumps(payload)), {})
        try:
            return verifier.verify("Film (2000)", conditions, text)
        finally:
            llm_client.complete = real

    text = "A butler realises he left his hat in the countryside. A rabbit puts her cap on."
    cond = ["an animal that wears a hat"]

    # A yes whose note argues against it.
    r = run({"findings": [{"requirement": cond[0], "verdict": "yes",
                           "quote": "A butler realises he left his hat in the countryside.",
                           "note": "Edgar is a butler, not an animal."}]}, cond, text)
    if r["accepted"] or r["findings"][0]["verdict"] != "unclear":
        failures.append("a `yes` whose note contradicts it was accepted")

    # A yes with no quote at all is an assertion.
    r = run({"findings": [{"requirement": cond[0], "verdict": "yes", "quote": ""}]},
            cond, text)
    if r["accepted"]:
        failures.append("a `yes` with no quote was accepted")

    # A yes whose quote is not in the text loses its evidence.
    r = run({"findings": [{"requirement": cond[0], "verdict": "yes",
                           "quote": "a hamster wearing a tiny fez"}]}, cond, text)
    if r["accepted"]:
        failures.append("a `yes` quoting text that does not exist was accepted")

    # Stray objects are not findings, and every requirement gets exactly one.
    r = run({"findings": [{"note": "the Cheshire Cat is an animal"},
                          {"requirement": "something never asked about",
                           "verdict": "yes", "quote": "A rabbit puts her cap on."}]},
            cond, text)
    if len(r["findings"]) != 1:
        failures.append(f"expected one finding per requirement, got {len(r['findings'])}")
    if any(f.get("requirement") not in cond for f in r["findings"]):
        failures.append("a finding survived for a requirement that was never asked")
    if r["findings"][0]["verdict"] != "unclear":
        failures.append("a requirement the model ignored was not recorded as unclear")
    if r["accepted"]:
        failures.append("a film was accepted on a requirement that got no verdict")

    # A clean identifying note must survive, or every honest yes is punished.
    r = run({"findings": [{"requirement": cond[0], "verdict": "yes",
                           "quote": "A rabbit puts her cap on.",
                           "note": "The rabbit is Judy."}]}, cond, text)
    if not r["accepted"]:
        failures.append("a supported `yes` with a plain identifying note was rejected")
    return failures


def g11_screen_orders_not_filters() -> list[str]:
    """A scan for an absence must order the candidates, never remove any. Free.

    Deleting the flagged half turned a word scan into a verdict. 194 of 316
    films flag on death vocabulary, and Toy Story flags on "killing" and
    "murdered" for a belief that turns out to be false, Monsters, Inc. and
    Zootopia on attempts nobody dies of. All three were dropped from the
    request before the Verifier -- whose entire job is telling an attempt from
    an outcome -- could see them.

    The forward direction is different and must keep narrowing: there a match
    IS the finding, so films without one are genuinely out of scope.
    """
    from agent import catalog, tools

    failures = []
    words = ["dies", "died", "dead", "killed", "murdered", "perished", "funeral"]

    ctx = tools.ToolContext()
    tools.filter_catalog(genres=["Animation"], ctx=ctx)
    pool = set(ctx.working_set or ())
    r = tools.screen_out(words=words, keep="clear", ctx=ctx)

    if set(ctx.working_set or ()) != pool:
        failures.append(f"keep='clear' changed the working set from {len(pool)} to "
                        f"{len(ctx.working_set or ())}; it must order, not remove")
    if not ctx.preferred:
        failures.append("keep='clear' recorded no preference, so the walk gains nothing")
    if len(ctx.preferred) != r["clear"]:
        failures.append(f"{len(ctx.preferred)} films preferred but {r['clear']} are clear")

    # A flagged film stays reachable -- the whole point.
    flagged_ids = pool - set(ctx.preferred)
    if not flagged_ids:
        failures.append("no film was flagged, so this proves nothing about reachability")
    for title in ("Toy Story", "Monsters, Inc."):
        row = catalog.movies()
        row = row[row["title"] == title]
        if len(row):
            mid = int(row.iloc[0]["id"])
            if mid in pool and mid not in (ctx.working_set or ()):
                failures.append(f"{title} was removed from the request by a word scan")

    # The forward scan still narrows.
    ctx2 = tools.ToolContext()
    tools.filter_catalog(genres=["Animation"], ctx=ctx2)
    before = len(ctx2.working_set or ())
    tools.screen_out(words=["train", "trains"], keep="flagged", ctx=ctx2)
    if len(ctx2.working_set or ()) >= before:
        failures.append("keep='flagged' did not narrow; a presence match is a finding")

    # The model writes the list; the presets are not on offer.
    schema = next(x for x in tools.TOOL_SCHEMAS
                  if x["function"]["name"] == "screen_out")
    props = schema["function"]["parameters"]["properties"]
    if "vocabulary" in props:
        failures.append("screen_out still offers a fixed `vocabulary`; the word list "
                        "is written per request now")
    if "words" not in (schema["function"]["parameters"].get("required") or []):
        failures.append("screen_out does not require `words`")

    # And a short list says so, because that is this design's failure mode.
    if not tools.screen_out(words=["dies"], keep="clear").get("thin_word_list"):
        failures.append("a one-word scan for an absence did not warn that it is thin")

    # Nothing about words or phrases ships. A fixed list only ever fits the
    # requests someone thought of, and both of these grew up fitting each
    # other rather than any particular question.
    from rag import screen as rag_screen
    for gone in ("VOCABULARIES", "BLACKLIST_PHRASES"):
        if hasattr(rag_screen, gone):
            failures.append(f"rag/screen.py still ships {gone}; the planner writes "
                            f"the words and the exclusions per request")

    # The exclusions are the planner's too, and they have to actually apply.
    hit = rag_screen.screen(["dead"], candidate_ids=None)
    miss = rag_screen.screen(["dead"], candidate_ids=None,
                             exclude_phrases=["dead"])
    if len(miss["flagged"]) >= len(hit["flagged"]):
        failures.append("exclude_phrases did not remove any match, so a caller "
                        "cannot correct a false positive it can see")
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

    claim("{} Disney and Pixar feature films", len(catalog.movies()), "catalog size")
    # The passage count and embedding dimension were stated on the Retrieval
    # tab, which is gone. They are still served by /api/rag/info for anyone who
    # wants them; the page no longer repeats a number it would have to keep.

    # The screen's counts are no longer checked here, and cannot be: the word
    # list is written per request by the planner, so "films clear on the death
    # screen" is a property of a run rather than of the shipped data. The GUI
    # states it as an example of what exhaustive means instead of as a figure.

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


def g01_execute_contract() -> list[str]:
    """Every malformed input to /api/execute answers with the four fields, at 200.

    The spec fixes the top-level fields "exactly", and describes the error case
    as a response format rather than a transport failure. Asserting it here
    means the contract cannot drift the next time an early return is added.

    The type cases are the ones that bit. This gate used to test only "" and
    {}, both of which are falsy strings by the time they reach the check, so it
    passed while {"prompt": 123} raised AttributeError on .strip() and Flask
    served a 500 HTML page. An automated grader posting junk finds that in one
    request. Anything that is not a dict with a string prompt has to come back
    as JSON in the same shape as every other error.
    """
    import app as flask_app

    failures = []
    client = flask_app.app.test_client()
    required = ["error", "response", "status", "steps"]

    cases = [
        ("empty prompt", {"prompt": ""}),
        ("no body", {}),
        ("whitespace-only prompt", {"prompt": "   "}),
        ("null prompt", {"prompt": None}),
        ("integer prompt", {"prompt": 123}),
        ("array prompt", {"prompt": ["hello"]}),
        ("object prompt", {"prompt": {"text": "hello"}}),
        ("boolean prompt", {"prompt": True}),
        ("body is an array", ["hello"]),
        ("body is a bare string", "hello"),
        ("body is a number", 7),
    ]

    for label, payload in cases:
        r = client.post("/api/execute", json=payload)
        if r.content_type and "json" not in r.content_type:
            failures.append(f"{label}: Content-Type is {r.content_type!r}, not JSON "
                            f"-- this is the 500 HTML page the contract forbids")
            continue
        body = r.get_json(silent=True)
        if not isinstance(body, dict):
            failures.append(f"{label}: body is {body!r}, expected the error object")
            continue
        if sorted(body) != required:
            failures.append(f"{label}: fields are {sorted(body)}, expected {required}")
        if body.get("status") != "error":
            failures.append(f"{label}: status is {body.get('status')!r}, expected 'error'")
        if body.get("response") is not None or body.get("steps") != []:
            failures.append(f"{label}: response/steps are "
                            f"{body.get('response')!r}/{body.get('steps')!r}, "
                            f"expected null and []")
        if r.status_code != 200:
            failures.append(f"{label}: HTTP {r.status_code}; errors belong in the body, "
                            f"and every other error path here answers 200")
    return failures


def main() -> int:
    total = 0
    for name, fn, what in [
        ("G01", g01_execute_contract,
         "/api/execute answers with exactly four fields on every malformed input"),
        ("G03", g03_module_names, "module names agree across diagram, agent_info and steps"),
        ("G04", g04_models, "the course-provided model deployments are the ones configured"),
        ("G05", g05_runtime_is_structured,
         "runtime is a filter argument, exact and inclusive, and cited back"),
        ("G06", g06_call_budget,
         "the total model-call cap binds planner, Observer and Verifier"),
        ("G07", g07_fusion_beats_greed,
         "rank fusion puts coverage above average rank, so greed loses"),
        ("G08", g08_no_follow_up_offers,
         "follow-up offers are caught and honest reports are not"),
        ("G09", g09_counts, "every displayed count comes from the shipped data"),
        ("G11", g11_screen_orders_not_filters,
         "a scan for an absence orders the candidates and removes none"),
        ("G10", g10_verdicts_need_evidence,
         "a verdict without a quote that shows it is not a verdict"),
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
