"""The four tools MoviBot can call, and their function-calling schemas.

Design note -- why four tools, and why in this order:

The catalog is 316 films. At that size the expensive thing is not searching, it
is reasoning: every model call costs money and latency. So the tools are ordered
cheapest-and-most-exhaustive first, and each one hands the next a smaller
candidate set:

    filter_catalog      columns    316 -> N     free, exact, exhaustive
    retrieve_plots      vectors    ranks N      one embedding per condition
    retrieve_metadata   vectors    ranks N      one embedding per condition
    verify_candidates   full text  <= 10 films  one model call per film

Each answers something the others cannot. Columns settle era, studio, language
and their negative forms -- "not Pixar" and "no musicals" are column lookups,
and nothing else needs to be spent on them. The two retrievals settle different
questions over disjoint corpora: plot text narrates what HAPPENS, while cast,
production and reception describe what a film IS. Both rank, neither decides.
Only the last tool decides, by reading one film's full plot text and answering
every requirement against it.

The consequence is that the token-heavy tool only ever sees what survived the
free ones, and a query pays for exactly the layers its conditions require.

Guardrails live here, not in the prompt. The model cannot forget them, and a
bad plan cannot bypass them:

  * results are ordered by `weighted_rating`, never raw `vote_average`
  * the candidate walk stops at MAX_VERIFICATIONS films or
    MAX_RECOMMENDATIONS_CEILING acceptances, whichever comes first
  * each film's plot text is trimmed to MAX_SYNOPSIS_CHARS, which is what
    bounds the context cost of a verification
"""

from __future__ import annotations

from typing import Any

import json
import re
from dataclasses import dataclass, field

from agent import catalog, shortlist, verifier
from rag import config as ragcfg
from rag import store as retrieval
from rag.corpora import DEFAULT_SOURCES


@dataclass
class ToolContext:
    """State that lives for one request, and never enters the prompt.

    The candidate set is the whole point. filter_catalog narrows the catalog
    and records every matching id here; search and read then scope themselves
    to it automatically. Previously the model had to carry ids forward itself,
    which cost tokens, invited transcription errors, and -- because the result
    was capped at 40 -- silently lost every match beyond the cap. A filter
    matching 212 films left 172 of them unreachable.
    """

    working_set: set[int] | None = None      # None = whole catalog
    scope_note: str = "whole catalog"
    trace: list[str] = field(default_factory=list)

    # The fused ordering from build_shortlist, and the conditions it fused.
    # Held here rather than passed through the prompt for the same reason the
    # working set is: it is bookkeeping, the model cannot improve it by seeing
    # it, and a list of ids re-typed by a model is a list of ids with a
    # transcription error in it.
    shortlist: list[int] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    # How many model calls the request can still afford. verify_candidates
    # spends calls inside a single tool call, so without this the loop's cap
    # would be enforced only between tool calls -- which is to say, after the
    # overspend had already happened.
    calls_remaining: int | None = None

    def narrow(self, ids: set[int], why: str) -> None:
        self.working_set = ids
        self.scope_note = f"{len(ids)} films ({why})"
        self.trace.append(self.scope_note)

    def candidates(self) -> list[int] | None:
        return sorted(self.working_set) if self.working_set is not None else None

# How many labels come back when a filter matches more than this. The full set
# is always in scope regardless; this only bounds what is *shown*, and
# list_all=true lifts it. Not a cost guard -- all 316 labels are ~3,100 tokens.
PREVIEW_FILMS = 15

# Full plot texts are large (median ~5 KB). One verification sees one film, so
# this is what bounds the context cost of a single call.
MAX_SYNOPSIS_CHARS = 6000

# Below this, the top hit is usually a sign the query was phrased as a theme
# rather than as an event. Measured on this corpus, seven queries:
#
#   abstract  0.3423  "a film about the dangers of vanity"
#             0.3492  "a story exploring themes of identity and belonging"
#             0.3528  "someone pretends to love another to seize power"
#             0.4063  "a character pretends to love another person, gains
#                      power or a throne, then reveals the deception"
#   concrete  0.4380  "he says he never loved her and leaves her to die so he
#                      can take the throne"
#             0.4767  "toys come alive when their owner leaves the room"
#             0.6635  "a rat cooks in a Paris restaurant kitchen"
#
# The fourth is why this is 0.42 rather than 0.40. It reads as concrete and is
# not -- it names an abstraction ("gains power or a throne") instead of a
# scene, scored just above the old threshold, and returned five films without
# Frozen among them. Everything observed still separates cleanly, but the gap
# is 0.4063 to 0.4380, not 0.3528 to 0.4380.
#
# Advisory only: a real answer can score below this, so it warns, never filters.
WEAK_MATCH_SIMILARITY = 0.42

# ---------------------------------------------------------------------
# Tool 1: structured filtering
# ---------------------------------------------------------------------

def filter_catalog(
    year_min: int | None = None,
    year_max: int | None = None,
    runtime_min: int | None = None,
    runtime_max: int | None = None,
    genres: list[str] | None = None,
    exclude_genres: list[str] | None = None,
    studio: str | None = None,
    languages: list[str] | None = None,
    exclude_english_only: bool = False,
    exclude_titles: list[str] | None = None,
    require_synopsis: bool = False,
    list_all: bool = False,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Filter the catalog on structured columns. Free, no model call.

    Every match becomes the request's candidate set, so the retrievals and
    the candidate walk are automatically limited to it -- nothing has to be
    carried forward. What comes back is a count and a handful of labels, not rows: the
    ids stay in Python, where they cost nothing.
    """
    df = catalog.movies()
    applied: dict[str, Any] = {}

    if year_min is not None:
        df = df[df["release_year"] >= int(year_min)]
        applied["year_min"] = int(year_min)
    if year_max is not None:
        df = df[df["release_year"] <= int(year_max)]
        applied["year_max"] = int(year_max)

    # Runtime is a column, and every one of the 316 rows has it. It was
    # reachable only through plot search before this, which cannot establish
    # it: no synopsis states how long its film runs, so "under 110 minutes"
    # scored as a weak thematic match and came back unverifiable. A fact the
    # catalog stores must never be answered by searching for it.
    #
    # Inclusive both ends, matching year. "Under 110" is runtime_max=109 and
    # the schema says so, because the alternative is the model passing 110 and
    # returning a 110-minute film to a user who asked for less.
    if runtime_min is not None:
        df = df[df["runtime_minutes"] >= int(runtime_min)]
        applied["runtime_min"] = int(runtime_min)
    if runtime_max is not None:
        df = df[df["runtime_minutes"] <= int(runtime_max)]
        applied["runtime_max"] = int(runtime_max)

    # Genres are AND-ed: "animated adventure" means both, not either.
    if genres:
        wanted = [g.lower() for g in genres]
        df = df[df["genres"].apply(
            lambda gs: all(any(w == g.lower() for g in gs) for w in wanted)
        )]
        applied["genres"] = genres

    if exclude_genres:
        unwanted = [g.lower() for g in exclude_genres]
        df = df[df["genres"].apply(
            lambda gs: not any(g.lower() in unwanted for g in gs)
        )]
        applied["exclude_genres"] = exclude_genres

    if studio:
        needle = studio.lower()
        df = df[df["production_companies"].apply(
            lambda cs: any(needle in c.lower() for c in cs)
        )]
        applied["studio"] = studio

    if languages:
        # "Hindi" has to reach हिन्दी, and "Mandarin" 普通话, or the filter
        # selects nothing while reporting itself as applied -- which is how a
        # catalog holding three Hindi films answered "a Disney movie in Hindi"
        # with "none are marked as Hindi-language".
        endonyms, isos, unmatched = [], [], []
        for name in languages:
            pair = catalog.resolve_language(name)
            if pair is None:
                unmatched.append(name)
                continue
            endonyms.append(pair[0].lower())
            if pair[1]:
                isos.append(pair[1])

        if endonyms or isos:
            spoken = df["spoken_languages"].apply(
                lambda ls: any(e == l.lower() for e in endonyms for l in ls)
            )
            # `original_language` separates a film made in Hindi from one that
            # merely has Hindi dialogue. Both answer "a movie in Hindi", and
            # the rating order decides which leads.
            original = df["original_language"].astype(str).str.lower().isin(isos)
            df = df[spoken | original]
        elif unmatched:
            df = df.iloc[0:0]

        applied["languages"] = languages
        if unmatched:
            # Never swallowed: an argument the catalog cannot match must be
            # reported, or an empty result reads as a fact about the films.
            applied["languages_unmatched"] = unmatched
            applied["languages_available"] = sorted(
                {v[0] for v in catalog.LANGUAGE_ALIASES.values()}
            )

    # "has non-English dialogue" is not a column; it is a property of the
    # language list. Encoding it here keeps the model from having to know the
    # catalog's language vocabulary.
    if exclude_english_only:
        df = df[df["spoken_languages"].apply(
            lambda ls: any(l not in ("English", "No Language") for l in ls)
        )]
        applied["exclude_english_only"] = True

    # Accepts either form a caller might reasonably use: the "Title (Year)"
    # label that every other tool speaks, or a bare title. Labels are resolved
    # to ids first, because matching them against the bare `title` column never
    # succeeds -- "toy story (1995)" is not "toy story". That mismatch made the
    # exclusion silently do nothing while still reporting itself as applied,
    # which is worse than failing: the planner believed a constraint had been
    # enforced and merely avoided naming the films instead.
    #
    # Bare titles stay an exact case-insensitive match, so excluding "Frozen"
    # does not also drop "Frozen Fever"; a substring match would over-exclude.
    if exclude_titles:
        blocked_ids, blocked_titles, unmatched = set(), set(), []
        for raw in exclude_titles:
            name = str(raw).strip()
            if not name:
                continue
            movie_id = catalog.resolve(name)
            if movie_id is not None:
                blocked_ids.add(movie_id)
                continue
            # Not a resolvable label. Fall back to a bare-title match, which
            # also drops every remake sharing that title -- what "besides The
            # Jungle Book" ought to mean.
            lowered = name.lower()
            if (df["title"].str.lower() == lowered).any():
                blocked_titles.add(lowered)
            else:
                unmatched.append(raw)

        if blocked_ids:
            df = df[~df["id"].isin(blocked_ids)]
        if blocked_titles:
            df = df[~df["title"].str.lower().isin(blocked_titles)]
        applied["exclude_titles"] = exclude_titles
        if unmatched:
            # Surfaced rather than swallowed: an exclusion the catalog cannot
            # match is something the planner must know about, not a no-op.
            applied["exclude_titles_unmatched"] = unmatched

    if require_synopsis:
        df = df[df["id"].apply(catalog.has_synopsis)]
        applied["require_synopsis"] = True

    df = df.sort_values("weighted_rating", ascending=False)
    matched = len(df)

    if ctx is not None:
        ctx.narrow(set(int(i) for i in df["id"]), _describe(applied))

    # Runtime rides along on every film, not only when a runtime filter was
    # passed. Citing "Moana (2016), 107 minutes" is the difference between the
    # planner reporting a constraint as satisfied and showing that it is, and a
    # field that appears only sometimes is a field the model learns not to
    # rely on. It costs about 4 tokens a film.
    labelled = [
        {"film": f"{r.title} ({int(r.release_year)})",
         "runtime": int(r.runtime_minutes),
         "rating": round(float(r.weighted_rating), 2)}
        for r in df.itertuples()
    ]

    out: dict[str, Any] = {
        "matched": matched,
        "filters_applied": applied or "none",
        "scope": "retrieval and the shortlist now cover exactly these films",
    }
    if list_all or matched <= PREVIEW_FILMS:
        # Labels are ~10 tokens each, so even all 316 costs less than a fifth
        # of one verification call. Completeness is affordable; hiding matches
        # behind a cap was not a saving worth making.
        out["films"] = labelled
    else:
        out["best_rated"] = labelled[:PREVIEW_FILMS]
        out["note"] = (
            f"Showing the {PREVIEW_FILMS} best rated of {matched}. All {matched} "
            "are in scope for search and reading; pass list_all=true to see every title."
        )
    return out


def _describe(applied: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in applied.items()) if applied else "no filter"



# ---------------------------------------------------------------------
# Tool 3: semantic search
# ---------------------------------------------------------------------



# Two corpora, two questions. Plot text narrates what happens; the rest of a
# Wikipedia page -- cast, production, reception, themes -- is where a film is
# described rather than told. "A strong female character" is written about a
# film far more often than it is narrated inside one, which is why retrieving
# only over plots answered it badly. Disjoint on purpose: a condition belongs
# to one or the other, and the split is only worth having if they differ.
# How many fused rows come back in the result. Bounds what is SHOWN, never
# what is ranked or walked -- the whole scope is ranked and the walk reads as
# far down it as the budget allows.
SHORTLIST_ROWS = 15

PLOT_SOURCES = ["mpst", "wiki_plot"]
METADATA_SOURCES = ["wiki_context", "overview"]


def retrieve_plots(conditions=None, top_k=None, ctx=None):
    """Rank films by what their plot text narrates."""
    return build_shortlist(conditions=conditions, top_k=top_k,
                           sources=PLOT_SOURCES, ctx=ctx)


def retrieve_metadata(conditions=None, top_k=None, ctx=None):
    """Rank films by how they are written about, rather than what happens."""
    return build_shortlist(conditions=conditions, top_k=top_k,
                           sources=METADATA_SOURCES, ctx=ctx)


def build_shortlist(
    conditions: list[str] | None = None,
    top_k: int | None = None,
    sources: list[str] | None = None,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Search EVERY story condition separately, then fuse the rankings.

    This is the tool that stops the agent being greedy. Searching one condition
    and reading its top hits assumes the other conditions hold in whatever came
    back, and nothing makes that true: a film ranking first on "a princess" and
    thirty-first on "snow and ice" gets read, while the film placing tenth on
    both is never seen.

    So each condition gets its own search -- one embedding each, no model call
    -- and the lists are fused before anything expensive happens. Films are
    ordered by how many conditions they placed for first, and by their average
    rank among those second, so satisfying everything moderately beats
    satisfying one thing perfectly.

    Searches are scoped to the current working set, so a catalog filter run
    beforehand removes films from every condition's list at once.
    """
    wanted = [c.strip() for c in (conditions or []) if str(c).strip()]
    if not wanted:
        return {"error": "build_shortlist needs at least one condition to search for."}

    sources = sources or PLOT_SOURCES
    scoped = ctx.candidates() if ctx is not None else None
    # Everything in scope, unless a caller asks for less. What is SHOWN is
    # bounded downstream; that has no business bounding what is ranked.
    top_k = int(top_k) if top_k else (len(scoped) if scoped else len(catalog.movies()))

    lists: dict[str, list[int]] = {}
    passages: dict[tuple[str, int], str] = {}
    for condition in wanted:
        hits = retrieval.search(condition, top_k=top_k, candidate_ids=scoped,
                                sources=sources)
        lists[condition] = [h["movie_id"] for h in hits]
        for h in hits:
            passages[(condition, h["movie_id"])] = h["passage"]

    ratings = {int(i): (catalog.rating_of(int(i)) or 0.0)
               for ids in lists.values() for i in ids}
    fused = shortlist.fuse(lists, ratings=ratings)

    if ctx is not None:
        # Merged with whatever a previous retrieval left, best-first, so two
        # retrievals over different corpora produce one ordering rather than
        # the second replacing the first.
        seen = set()
        merged = []
        for mid in [c.movie_id for c in fused] + list(ctx.shortlist):
            if mid not in seen:
                seen.add(mid); merged.append(mid)
        ctx.shortlist = merged
        ctx.conditions = list(dict.fromkeys(list(ctx.conditions) + wanted))

    rows = [shortlist.explain(c, wanted, catalog.label_of(c.movie_id) or str(c.movie_id))
            for c in fused[:SHORTLIST_ROWS]]

    full = len(fused)
    covered_all = sum(1 for c in fused if c.covered == len(wanted))
    # With full rankings a film is missing from a condition's list only when it
    # has no indexed text in that corpus at all, so this counts coverage of the
    # corpus rather than quality of the match.
    return {
        "conditions": wanted,
        "over": sources,
        "searched_within": "whole catalog" if scoped is None
                           else f"{len(scoped)} films in scope",
        "candidates": full,
        "matching_every_condition": covered_all,
        "shortlist": rows,
        "note": (
            f"{full} films ranked; {covered_all} have text for all {len(wanted)} "
            f"conditions. Ordered by conditions covered, then average rank -- so "
            "the top of this list is the best place to start verifying, and a film's "
            "position here is a search ranking, NOT evidence that it satisfies "
            "anything. Verify before recommending."
        ),
        "model_call": {"model": ragcfg.EMBED_MODEL, "kind": "embedding",
                       "calls": len(wanted), "inputs": wanted},
    }


# How many films the Verifier will look at before giving up. Ten one-film
# calls is the ceiling on a request that never finds anything; the walk stops
# the moment MAX_RECOMMENDATIONS are accepted, so the common case costs three
# or four. This is a demo, and a demo that cannot find an answer says so
# rather than searching forever.
MAX_VERIFICATIONS = 10

# How many films an answer may name. Owned here rather than in prompts.py
# because verify_candidates has to enforce it and prompts.py imports from this
# module, so the dependency only runs one way. prompts.MAX_RECOMMENDATIONS is
# a re-export, not a second copy.
MAX_RECOMMENDATIONS_CEILING = 3


def _catalog_facts() -> str:
    """What the catalog is, computed from the catalog.

    Every prompt that describes the data used to state its own numbers, and
    they went stale together the moment the studio filter was corrected and
    238 films became 316. Written once, from the CSV, so a prompt cannot claim
    a range the data does not have.
    """
    df = catalog.movies()
    lo, hi = int(df["release_year"].min()), int(df["release_year"].max())
    rlo, rhi = int(df["runtime_minutes"].min()), int(df["runtime_minutes"].max())
    readable = sum(1 for i in df["id"] if catalog.has_synopsis(int(i)))
    return (
        f"- {len(df)} Disney and Pixar feature films, {lo} to {hi}. No other "
        f"studio, no TV, no anime.\n"
        f"- Nothing later than {hi} exists for you -- not Frozen II, not "
        f"Encanto. You cannot know what is new, recent or trending.\n"
        f"- Feature films only, {rlo} to {rhi} minutes. Shorts were excluded, "
        f"so nothing shorter than {rlo} minutes can be found.\n"
        f"- No cast or crew data. \"Starring X\" and \"directed by Y\" cannot "
        f"be answered.\n"
        f"- {readable} of {len(df)} films have full plot text; the rest carry "
        f"only a one-line overview, and a story claim about those cannot be "
        f"settled."
    )


CATALOG_FACTS = _catalog_facts()


# Nothing here decides what a catalog fact looks like any more. A regex used to
# match "released in 1990", "runtime under 110 minutes", "a Pixar film" and
# strip them out of the verification list, because sending a column fact to the
# Verifier makes an answerable request unanswerable -- plot text cannot confirm
# a release year, and one unsettleable requirement fails every film.
#
# It was a stored list of the phrasings someone thought of, and it had already
# missed "year 1990+" once. The Decomposer is told where column facts belong,
# and the generic signal below catches what slips through: a requirement that
# comes back `unclear` on every film checked was never settleable from plot
# text, whatever it was about, and that needs no vocabulary to notice.


def verify_candidates(
    request: str = "",
    conditions: list[str] | None = None,
    films: list[str] | None = None,
    max_accept: int = 3,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Walk the shortlist best-first, checking EVERY condition on each film.

    One model call per film, each seeing that film's plot text and the whole
    list of requirements at once. A film is accepted only when every condition
    comes back `yes`; the accepted list is built here, in Python, so what the
    answer may claim is a count of a list rather than an assertion the model
    makes about its own reasoning.

    The walk stops at `max_accept` films or MAX_VERIFICATIONS calls, whichever
    comes first. Running out is a real outcome and is reported as one: this is
    a demo, and the honest answer to "nothing verified" is that nothing
    verified, not a longer search.
    """
    conditions = [c.strip() for c in (conditions or []) if str(c).strip()]
    if not conditions and ctx is not None:
        conditions = list(ctx.conditions)
    if not conditions:
        return {"error": "verify_candidates needs the conditions to check against."}



    # Named films go FIRST, they do not replace the walk. Letting them replace
    # it reopened the exact hole this tool was built to close: the planner named
    # three candidates it liked, the tool checked those three, and a request
    # that was entitled to ten verifications spent three and reported "I could
    # not verify any film after checking 3 films". Choosing which candidates to
    # check is the greedy move, and it does not stop being greedy because a
    # parameter invited it. So the named films are a head start on the fused
    # order, and the walk continues down that order until it accepts enough or
    # runs out of budget.
    named = [i for i in (catalog.resolve(f) for f in (films or [])) if i is not None]

    # The tail is the fused shortlist when there is one, and the working set
    # in rating order when there is not. Without that fallback a request that
    # never needed a semantic search -- everything settled by the filter --
    # could only ever check the films the plan happened to name, so thirteen
    # survivors got six verifications and the budget went unspent.
    if ctx is not None and ctx.shortlist:
        rest = list(ctx.shortlist)
    elif ctx is not None and ctx.working_set:
        rest = sorted(ctx.working_set,
                      key=lambda i: -(catalog.rating_of(i) or 0.0))
    else:
        rest = []

    tail = [i for i in rest if i not in set(named)]
    order = named + tail
    if not order:
        return {"error": "no candidates: retrieve or filter first, or name films."}

    max_accept = max(1, min(int(max_accept), MAX_RECOMMENDATIONS_CEILING))

    # Never spend the request's last call here: the planner still has to write
    # the answer, and evidence nobody gets to report is evidence wasted.
    affordable = MAX_VERIFICATIONS
    if ctx is not None and ctx.calls_remaining is not None:
        affordable = max(0, min(affordable, ctx.calls_remaining - 1))

    accepted, rejected, unresolved, rows = [], [], [], []
    checked = 0
    for movie_id in order:
        if len(accepted) >= max_accept or checked >= affordable:
            break
        label = catalog.label_of(movie_id)
        text = catalog.synopsis(movie_id)
        if not label or not text:
            # No text is not a verdict. Never counted as satisfying anything.
            unresolved.append(label or str(movie_id))
            continue

        checked += 1
        result = verifier.verify(label, conditions, _trim(text, MAX_SYNOPSIS_CHARS),
                                 request=request)
        rows.append(result)
        if result.get("accepted"):
            accepted.append(label)
        elif any(f.get("verdict") == "no" for f in result.get("findings") or []):
            rejected.append(label)
        else:
            unresolved.append(label)

    # Per-condition tallies. A condition that comes back `unclear` on every
    # film it was checked against is almost never a fact about those films --
    # it is a condition plot text cannot settle, like "fits a family evening".
    # Saying so lets the planner drop it and answer on the rest, instead of
    # reporting zero results for a request that had plenty.
    tally: dict[str, dict[str, int]] = {
        c: {"yes": 0, "no": 0, "unclear": 0} for c in conditions
    }
    for row in rows:
        for f in row.get("findings") or []:
            cell = tally.get(f.get("requirement"))
            if cell is not None and f.get("verdict") in cell:
                cell[f["verdict"]] += 1

    unsettleable = [c for c, t in tally.items()
                    if checked >= 3 and t["unclear"] >= checked and t["yes"] == 0]

    return {
        "conditions": conditions,
        "verified": checked,
        "by_condition": tally,
        **({"unsettleable": unsettleable,
            "unsettleable_note": (
                "Every film checked came back `unclear` on these, which means plot "
                "text cannot settle them -- they are not story facts. Do NOT report "
                "zero results because of one. Drop them, say you could not judge "
                "them from plot text, and answer on the conditions that remain."
            )} if unsettleable else {}),
        "accepted": accepted,
        "rejected": rejected,
        "unresolved": unresolved,
        "budget": {"checked": checked, "allowed": affordable,
                    "ceiling": MAX_VERIFICATIONS},
        "exhausted": checked >= affordable and len(accepted) < max_accept,
        "verdicts": rows,
        "note": (
            f"{len(accepted)} film(s) satisfied every condition. **Only these may be "
            "recommended, and the number you state must be this number.** A film in "
            "`rejected` failed a condition; one in `unresolved` was not settled by its "
            "text, which is not the same as satisfying it. If `accepted` is empty, say "
            "plainly that nothing could be verified and how many films were checked -- "
            "do not offer an unresolved title as a near-miss."
        ),
    }


def _trim(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# ---------------------------------------------------------------------
# Tool 3: full plot text
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# Schemas + dispatch
# ---------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filter_catalog",
            "description": (
                "Filter the movie catalog on structured facts: year, "
                "runtime, genre, keyword, studio, spoken language, title "
                "exclusions. Use this FIRST whenever the request contains any "
                "hard constraint, and for anything about language, era, "
                "runtime or studio, which plot search cannot see. Results "
                "come back best-first by a vote-count-weighted rating, and "
                "every film comes back with its runtime in minutes. The "
                "catalog is feature films only (316 Disney/Pixar titles, "
                "1937-2017, 47 to 172 minutes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year_min": {"type": "integer", "description": "Earliest release year, inclusive."},
                    "year_max": {"type": "integer", "description": "Latest release year, inclusive."},
                    "runtime_min": {
                        "type": "integer",
                        "description": (
                            "Shortest runtime in minutes, inclusive. Every "
                            "film in the catalog has a runtime, so this is "
                            "exact -- never try to settle a length constraint "
                            "by retrieval, which cannot see it."
                        ),
                    },
                    "runtime_max": {
                        "type": "integer",
                        "description": (
                            "Longest runtime in minutes, INCLUSIVE, so 'under "
                            "110 minutes' is runtime_max=109 and 'no more "
                            "than 110' is runtime_max=110. The catalog holds "
                            "nothing under 47 minutes -- shorts were excluded "
                            "-- so a small value returning nothing is a real "
                            "absence you may report, not a failed search."
                        ),
                    },
                    "genres": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Required genres, ALL must match. e.g. ['Animation','Adventure'].",
                    },
                    "exclude_genres": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Reject any movie carrying one of these genres.",
                    },
                    "studio": {"type": "string", "description": "Production company substring, e.g. 'Pixar'."},
                    "languages": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Languages, ANY may match. Pass the ordinary "
                            "English name -- ['Hindi'], ['French'], "
                            "['Mandarin'] -- and it is resolved to however the "
                            "catalog spells it; an endonym or ISO code works "
                            "too. Matches a film made in that language or one "
                            "with dialogue in it. A language the catalog does "
                            "not have comes back as languages_unmatched rather "
                            "than as an empty result, so an empty result here "
                            "is a real absence you may report."
                        ),
                    },
                    "exclude_english_only": {
                        "type": "boolean",
                        "description": "Keep only movies with some non-English dialogue.",
                    },
                    "exclude_titles": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Titles to drop, matched exactly. Use when the "
                            "user says 'besides X' or 'other than X'. "
                            "**Exclusion is never a franchise.** 'besides Toy "
                            "Story' drops the 1995 film and leaves Toy Story 2 "
                            "and 3 in scope, which is usually what was meant; "
                            "if the user clearly means the whole series, pass "
                            "each film by name. Two films sharing a title are "
                            "both dropped by the bare title ('The Jungle Book' "
                            "drops the 1967 and 2016 versions) and one by the "
                            "label ('The Jungle Book (1967)'). Never invent a "
                            "title to exclude: a title the catalog cannot "
                            "match comes back as exclude_titles_unmatched."
                        ),
                    },
                    "require_synopsis": {
                        "type": "boolean",
                        "description": "Keep only movies whose full plot text is available for later reading.",
                    },
                    "list_all": {
                        "type": "boolean",
                        "description": (
                            "Return every matching title instead of the best-rated "
                            f"{PREVIEW_FILMS}. Use when the user asks for all of them. "
                            "Every match is in scope for later tools either way."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_candidates",
            "description": (
                "Check films against EVERY condition, one model call per "
                "film, and return the ones that satisfy all of them. **This "
                "is the only thing that makes a film recommendable.** It "
                "walks the shortlist best-first, reads each film's plot text, "
                "and returns `accepted`, `rejected` and `unresolved`. Only "
                "films in `accepted` may appear in your answer, and the "
                "number you state must equal how many are in it. It stops at "
                "3 accepted or 10 films checked, whichever comes first; if it "
                "runs out having accepted nothing, say so plainly and say how "
                "many were checked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Every condition the film must satisfy, phrased "
                            "as a REQUIREMENT rather than a question -- 'no "
                            "character dies', 'a princess appears'. Include "
                            "the negative ones here: this is the only place "
                            "an absence is adjudicated, because nothing can "
                            "be retrieved for by not happening. Defaults to "
                            "the "
                            "conditions you passed to build_shortlist."
                        ),
                    },
                    "films": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Optional. Films to check FIRST, as "
                            "'Title (Year)'. This is a head start, not a "
                            "restriction: the walk continues down the fused "
                            "shortlist afterwards until enough films are "
                            "accepted or the budget runs out. You cannot "
                            "narrow verification to a few favourites, and "
                            "should not want to -- the shortlist is already "
                            "ordered by how well each film matched every "
                            "condition."
                        ),
                    },
                    "max_accept": {
                        "type": "integer",
                        "description": "Stop after this many films pass. Default 3.",
                    },
                },
                "required": ["conditions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_plots",
            "description": (
                "Rank films by what their plot text NARRATES. Every "
                "condition is retrieved separately and the rankings fused "
                "into one ordered shortlist. **Use this instead of "
                "search_plots whenever the request has more than one story "
                "condition.** Searching one condition and reading its top "
                "hits assumes the others hold in whatever came back, which "
                "nothing makes true: a film ranking 1st on 'a princess' and "
                "31st on 'snow and ice' gets read while the film placing "
                "10th on both is never seen. Costs one embedding per "
                "condition and no model call. Films are ordered by how many "
                "conditions they placed for, then by average rank, so "
                "satisfying everything moderately beats satisfying one thing "
                "perfectly. Scoped to the current working set, so run "
                "filter_catalog first when the request has structured "
                "constraints. A position in this list is a search ranking, "
                "NOT evidence -- verify_candidates decides, and nothing may "
                "be recommended before it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "One entry per story condition, each phrased as a "
                            "CONCRETE EVENT the way search_plots requires -- "
                            "'snow and ice cover the kingdom', not 'wintry'. "
                            "Do not put structured constraints here (year, "
                            "runtime, studio, language): those belong in "
                            "filter_catalog, which is exact. Do not put a "
                            "pure absence here ('nobody dies') either: "
                            "nothing can be retrieved for by not happening, "
                            "so send it straight to verify_candidates."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Films per condition. Defaults to everything in "
                            "scope, which is what you want: truncating a "
                            "ranking before the fusion is how a film that "
                            "satisfies every condition gets lost."
                        ),
                    },
                },
                "required": ["conditions"],
            },
        },
    },
]

# Human-readable trace names, matching the boxes in the Architecture tab.
# The metadata twin. Same mechanism, different corpus, and the difference is
# the whole point: a plot narrates events, the rest of a Wikipedia page
# describes the film.
_METADATA_SCHEMA = json.loads(json.dumps(
    next(x for x in TOOL_SCHEMAS if x["function"]["name"] == "retrieve_plots")))
_METADATA_SCHEMA["function"]["name"] = "retrieve_metadata"
_METADATA_SCHEMA["function"]["description"] = (
    "Rank films by how they are WRITTEN ABOUT rather than by what happens: "
    "cast, production, reception and themes. A film is described as having a "
    "strong female lead, or as a coming-of-age story, far more often than a "
    "plot summary narrates either, so a condition about what a film IS belongs "
    "here and one about what HAPPENS belongs in retrieve_plots. Both may run; "
    "their rankings merge into one shortlist. One embedding per condition."
)
TOOL_SCHEMAS.append(_METADATA_SCHEMA)


TRACE_NAMES = {
    "filter_catalog": "CatalogFilter",
    "retrieve_plots": "PlotRetrieval",
    "retrieve_metadata": "MetadataRetrieval",
    # The TOOL, not the module it calls. Naming this step "Verifier" logged a
    # tool under a subagent's name and made the Verifier look like something
    # the plan could invoke directly. It cannot: the walk chooses candidates
    # and sends one model call per film, and the Verifier answers one film.
    "verify_candidates": "CandidateWalk",
}


def dispatch(name: str, arguments: dict[str, Any],
             ctx: "ToolContext | None" = None) -> dict[str, Any]:
    """Run one tool call, threading the request's candidate set through it."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**{**arguments, "ctx": ctx})
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - returned to the model as an observation
        return {"error": f"{type(exc).__name__}: {exc}"}


_DISPATCH = {
    "filter_catalog": filter_catalog,
    "retrieve_plots": retrieve_plots,
    "retrieve_metadata": retrieve_metadata,
    "verify_candidates": verify_candidates,
}
