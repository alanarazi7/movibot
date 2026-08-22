"""Decomposer: the one place a request is interpreted.

It reads the user's words once and returns a plan -- which columns to filter
on, which words to scan for, which story conditions to search, and which
requirements each candidate must satisfy. Everything after it is either
deterministic Python or a check against one film's plot text. Nothing later
re-reads the request, so there is exactly one place where "for a family
evening" is decided to be framing rather than a condition.

The plan comes back through a tool schema rather than as free JSON, so the
provider enforces the shape and a missing field is a missing field rather than
a parse error three steps downstream.

Three outcomes, and the split matters:

  search      the request names something the catalog can be searched for
  refuse      it does not, and no amount of searching would help -- a film
              after 2017, a short, an actor's filmography
  about_self  it is a question about the agent, which no tool answers

Refusing is a decision made here, before any work, because the alternative is
what the old design did: filter, search, verify, and only then discover the
request was never answerable.
"""

from __future__ import annotations

import json
from typing import Any

from agent import llm_client, tools

# Conditions are stated as REQUIREMENTS the film must satisfy, never as
# questions, so a `yes` always means satisfied -- including for negatives.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": (
            "Break the request into the parts each stage of the pipeline "
            "needs. Call this exactly once; it is the only output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["search", "refuse", "about_self"],
                    "description": (
                        "'search' when the catalog can be searched for what "
                        "was asked. 'refuse' when nothing in the catalog "
                        "could answer it however hard you looked -- a film "
                        "released after 2017, anything under 47 minutes, a "
                        "request by actor or director, a studio other than "
                        "Disney or Pixar. 'about_self' for a question about "
                        "you rather than about films. A request that is "
                        "merely narrower than the user assumes is still "
                        "'search': answer it and say what universe it came "
                        "from."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "For 'refuse', what the catalog cannot do and why, in "
                        "one or two sentences, naming the nearest thing it "
                        "can do if there is one. For 'about_self', the answer "
                        "itself: who you are and what you can and cannot do, "
                        "two or three sentences, accurate rather than "
                        "promotional. Empty for 'search'."
                    ),
                },
                "conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Every condition the request contains, in the user's "
                        "terms, one per entry. This is the ledger a reader "
                        "checks the answer against, so include the "
                        "structured ones too even though they are enforced "
                        "elsewhere. Leave out framing that is not a "
                        "condition: 'for a family evening', 'for my nephew' "
                        "say who is watching, not what happens."
                    ),
                },
                "filter": {
                    "type": "object",
                    "description": (
                        "Arguments for the catalog filter: facts stored in "
                        "columns. Exact and free, so anything that fits here "
                        "belongs here and nowhere else."
                    ),
                    "properties": {
                        "year_min": {"type": "integer"},
                        "year_max": {"type": "integer"},
                        "runtime_min": {"type": "integer"},
                        "runtime_max": {
                            "type": "integer",
                            "description": "INCLUSIVE: 'under 110 minutes' is 109.",
                        },
                        "genres": {"type": "array", "items": {"type": "string"}},
                        "exclude_genres": {"type": "array", "items": {"type": "string"}},
                        "studio": {"type": "string"},
                        "languages": {"type": "array", "items": {"type": "string"}},
                        "exclude_titles": {
                            "type": "array", "items": {"type": "string"},
                            "description": "For 'besides X'. Exact titles only.",
                        },
                    },
                },
                "search_plots": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Conditions about what HAPPENS, each phrased as a "
                        "concrete event the way a plot summary narrates it: "
                        "'snow and ice cover the kingdom', not 'wintry'. "
                        "Retrieved over plot text. Do not put a column fact "
                        "here, and do not put a pure absence here -- nothing "
                        "can be retrieved for by not happening."
                    ),
                },
                "search_metadata": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Conditions about what a film IS or is known for, "
                        "phrased the way someone writing about it would: "
                        "'a strong female protagonist', 'a coming-of-age "
                        "story', 'praised for its animation'. Retrieved over "
                        "cast, production, reception and themes rather than "
                        "over the plot. Use this whenever the condition is a "
                        "description of the film rather than an event in it "
                        "-- those are written about far more often than they "
                        "are narrated. Both retrievals may be used at once, "
                        "and their rankings are merged."
                    ),
                },
                "verify": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "What each candidate film must satisfy, phrased as "
                        "REQUIREMENTS rather than questions: 'no character "
                        "dies', 'an animal appears'. This is what the "
                        "Verifier checks against one film's plot text, so "
                        "include every condition the story settles, negatives "
                        "included. **Never put a column fact here** -- year, "
                        "runtime, studio and language are already guaranteed "
                        "by the filter, plot text cannot confirm them, and "
                        "one of them makes an answerable request "
                        "unanswerable. Empty when the request has no story "
                        "condition at all."
                    ),
                },
                "max_films": {
                    "type": "integer",
                    "description": (
                        f"How many films to return, 1 to "
                        f"{tools.MAX_RECOMMENDATIONS_CEILING}. Default "
                        f"{tools.MAX_RECOMMENDATIONS_CEILING}; use fewer only "
                        "when the request asks for one."
                    ),
                },
            },
            "required": ["outcome", "conditions"],
        },
    },
}

DECOMPOSER_PROMPT = f"""\
You are the Decomposer inside MoviBot, a Disney and Pixar film \
recommender. You read one movie request and return a plan by calling `plan` \
exactly once. You never write prose for the user and you never recommend a \
film: something later does that, from evidence you will not see.

**You are a part of MoviBot, not a thing the user talks to.** When the \
request asks what this is or what it can do, the `message` you write is \
MoviBot's answer about MoviBot -- what it recommends, from what catalog, and \
what it cannot do. Never mention the Decomposer, the Verifier, or any \
other part by name: the user asked about a film recommender, not about its \
internals.

WHAT THE CATALOG IS

{tools.CATALOG_FACTS}

ROUTING, WHICH IS THE WHOLE JOB

Split the request into conditions, then send each to the stage that can settle \
it. Route by what evidence would settle a condition, never by how it was \
phrased -- a negation over a column is still a column lookup.

  a word about WHEN                -> `filter`, as a year range
  "modern", "recent", "new", "old", "classic", "from the 90s", "in the
  last ten years". These are dates, not themes: retrieving for "modern"
  matches text about modernity, which is a different subject. Resolve
  them against the catalog's range, which ENDS IN 2017 -- so "modern"
  or "recent" is roughly year_min 2000, "the last ten years" is about
  2007 onward, and "classic" or "old" is a year_max. Say so in the
  answer only if the user's sense of now and the catalog's differ
  enough to matter.

  a fact the catalog stores        -> `filter`
  "not Pixar", "no musicals", "besides Frozen", "after 2000",
  "under 110 minutes", "in Hindi". Free and exact. Never send one of
  these to `verify`: plot text cannot confirm a release year, and a
  requirement nothing can settle makes the whole request fail.

  something that HAPPENS            -> `search_plots`
  "snow covers the kingdom", "a prince reveals he never loved her".
  Retrieved over plot text, which narrates events.

  what a film IS or is known for    -> `search_metadata`
  "a princess movie", "a strong female protagonist", "a coming-of-age
  story". Retrieved over cast, production, reception and themes. These
  are written ABOUT a film far more often than they are narrated inside
  one, which is why looking for them in the plot finds little.

  An ABSENCE -- "nobody dies", "nothing scary" -- is retrieved for by
  neither: embed "no deaths" and you get the films where somebody dies,
  because that is what those plots say. Send it to `verify` alone and
  let the candidates come from the conditions that can be retrieved.

  a claim about what happens       -> `verify`
  Everything the story has to settle, including the absences you also
  screened for. The screen finds candidates; only the Verifier decides.

  nothing can settle it            -> `conditions` only
  Who it is for, what mood it suits. Recorded, not acted on.

**What the request is ABOUT is a condition, and the easiest one to lose.** "A \
princess movie", "a pirate film", "something with robots" name the subject. A \
request whose only routed condition is an exclusion has been read wrong: \
filtering "besides Frozen and Moana" and nothing else returns the best-rated \
films in the catalog, which answers nothing. The subject goes to \
`search_metadata` to find candidates AND to `verify` so it is actually \
checked.

**Retrieving is not checking.** A condition you retrieve for and do not put \
in `verify` is one the answer will never be able to stand behind: retrieval \
ranks, it does not decide. Anything in `search_plots` or `search_metadata` \
belongs in `verify` too, phrased as a requirement. Leave `verify` empty only \
when the request has no condition at all beyond stored columns.

Aim to route every condition somewhere; the only ones that go nowhere are the \
ones nothing could settle.

**Every condition goes somewhere, and `conditions` lists all of them.** That \
is the ledger: what you understood the request to be asking. Nothing is left \
out of it.

Where each one goes is a separate question, and some have no stage that can \
settle them. "For my daughter", "for a family evening", "something to relax \
to" say who is watching, not what happens in the film. There is nothing in a \
plot summary that confirms or denies one, and no fact about the catalog that \
does either. Record them in `conditions` and send them nowhere -- not to \
`verify`, where every film would come back unclear and a request with sixty \
answers would return none.

Do not invent a proxy for one. "For my daughter" is not a licence to add a \
princess, a rating or a genre the user did not ask for; you would be answering \
a request they did not make and reporting it as though they had.

A condition can appear in two places, and often should: "nobody dies" belongs \
in `screen` to order the candidates cheaply AND in `verify` to be settled.

REFUSING

Refuse when no amount of searching would help: a film after 2017, anything \
shorter than the catalog holds, a request by actor or director, a studio it \
does not carry. Say what is missing and offer the nearest real thing. Do not \
refuse a request that is merely narrower than the user assumes -- "a nice \
comedy" is answerable, it just needs saying which universe the answer came \
from.\
"""


def decompose(request: str) -> dict[str, Any]:
    """Turn one request into a plan. One model call.

    Returns the plan plus the step record the loop logs, so this call is
    traced exactly like the Verifier and the Answerer: module, prompt,
    response.
    """
    messages = [
        {"role": "system", "content": DECOMPOSER_PROMPT},
        {"role": "user", "content": request.strip()},
    ]
    message, usage = llm_client.complete(messages, tools=[PLAN_SCHEMA])

    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        # No plan is not an empty plan. Answering anyway would mean answering
        # a request nothing has read.
        return {"plan": None, "usage": usage,
                "error": "the Decomposer returned no plan",
                "raw": (message.content or "")[:500]}

    try:
        plan = json.loads(calls[0].function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return {"plan": None, "usage": usage,
                "error": f"the plan was not valid JSON: {exc}"}

    return {"plan": _normalise(plan, request), "usage": usage}


def _normalise(plan: dict[str, Any], request: str) -> dict[str, Any]:
    """Fill in what the schema allows to be absent, and drop what it forbids.

    The provider enforces types, not sense: `max_films` above the ceiling and
    a `verify` entry that is really a column fact both arrive well-formed, and
    both are wrong in ways the stages downstream cannot recover from.
    """
    out: dict[str, Any] = {
        # The words the plan was made from, carried with it. Every later stage
        # works from the plan, so without this the request itself stops
        # existing after the first call -- and the Verifier was judging
        # "an animal appears" with no idea what had been asked for.
        "request": request.strip(),
        "outcome": plan.get("outcome") or "search",
        "message": (plan.get("message") or "").strip(),
        "conditions": [c for c in (plan.get("conditions") or []) if str(c).strip()],
        "filter": {k: v for k, v in (plan.get("filter") or {}).items()
                   if v not in (None, [], "")},
        "screen": {k: v for k, v in (plan.get("screen") or {}).items()
                   if v not in (None, [], "")},
        "search_plots": [x for x in (plan.get("search_plots") or []) if str(x).strip()],
        "search_metadata": [x for x in (plan.get("search_metadata") or [])
                            if str(x).strip()],
    }

    out["verify"] = [v for v in (plan.get("verify") or []) if str(v).strip()]

    # A condition worth retrieving for is a condition worth checking. Left to
    # the prompt, `verify` came back empty on "a strong female character" and
    # the answer could only offer a ranking -- retrieval decides which films to
    # look at, never whether one qualifies. So anything retrieved for that was
    # not given a requirement becomes one. No vocabulary is involved: it is the
    # plan's own text, moved between its own fields.
    retrieved = out["search_plots"] + out["search_metadata"]
    if retrieved and not out["verify"]:
        out["verify"] = list(dict.fromkeys(retrieved))
        out["verify_filled_from_retrieval"] = True

    ceiling = tools.MAX_RECOMMENDATIONS_CEILING
    try:
        out["max_films"] = max(1, min(int(plan.get("max_films") or ceiling), ceiling))
    except (TypeError, ValueError):
        out["max_films"] = ceiling
    return out
