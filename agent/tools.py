"""The four tools MoviBot can call, and their function-calling schemas.

Design note -- why four tools, and why in this order:

The catalog is 316 films. At that size the expensive thing is not searching, it
is reasoning: every model turn costs money and latency. So the tools are ordered
cheapest-and-most-exhaustive first, and each one hands the next a smaller
candidate set:

    filter_catalog   columns      316 -> N     free, exact, exhaustive
    screen_out       regex        N -> clear   free, one-sided, word-list only
    search_plots     vectors      N -> ~10     one embedding, ~$0.0000002
    read_synopses    full text    <= 8 films   free, but token-heavy

Each answers something the others cannot. Columns settle era, studio, language
and their negative forms -- "not Pixar" and "no musicals" are column lookups,
not screens. A lexical screen tests an absence the way ranking cannot: embed
"nobody dies" and the top hits are the films where somebody does. It is
exhaustive over its word list, not over the event, so what it certifies is that
no listed word appears in the stored text. Vectors settle positive story
questions, and full text settles what actually happens -- including whatever a
screen left unresolved.

The consequence is that the token-heavy tool only ever sees what survived the
free ones, and a query pays for exactly the layers its conditions require.

Guardrails live here, not in the prompt. The model cannot forget them, and a
bad plan cannot bypass them:

  * results are ordered by `weighted_rating`, never raw `vote_average`
  * `screen_out` refuses to certify a film with less than MIN_SCREEN_TOKENS of
    plot text, so absence of evidence can never be reported as evidence
  * `read_synopses` reads at most MAX_SYNOPSES movies, truncated to
    MAX_SYNOPSIS_CHARS each, which is what bounds the context cost per turn
"""

from __future__ import annotations

from typing import Any

import re
from dataclasses import dataclass, field

from agent import catalog, shortlist, verifier
from rag import config as ragcfg
from rag import screen as screening
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

    # Films the lexical screen found clean, best-first. A PRIOR, not a verdict:
    # it says these are the cheapest candidates to check for an absence, and
    # nothing more. See screen_out for why it stopped being a filter.
    preferred: list[int] = field(default_factory=list)

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

# How many flagged films come back with a quote attached. A quote is ~100 tokens
# against ~1,500 for reading the same film's synopsis, so shipping the evidence
# with the flag is what lets most negations resolve without a fourth tool call.
MAX_FLAGGED_EVIDENCE = 8

# Full plot texts are large (median ~5 KB). These two caps together bound the
# worst-case tool payload at ~48 KB, which is what keeps a turn affordable.
MAX_SYNOPSES = 8
MAX_SYNOPSIS_CHARS = 6000

MAX_SEARCH_RESULTS = 25

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

# Each search hit carries the passage that matched, as evidence. Long enough
# to judge a story beat, short enough that 25 of them stay affordable.
MAX_PASSAGE_CHARS = 1200


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
    keywords: list[str] | None = None,
    studio: str | None = None,
    languages: list[str] | None = None,
    exclude_english_only: bool = False,
    exclude_titles: list[str] | None = None,
    require_synopsis: bool = False,
    list_all: bool = False,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Filter the catalog on structured columns. Free, no model call.

    Every match becomes the request's candidate set, so search_plots and
    read_synopses are automatically limited to it -- nothing has to be carried
    forward. What comes back is a count and a handful of labels, not rows: the
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

    # Keywords are OR-ed and substring-matched: the vocabulary is noisy
    # ("princess", "fairy tale", "royalty" all coexist), so requiring all of
    # them would usually return nothing.
    if keywords:
        wanted = [k.lower() for k in keywords]
        df = df[df["keywords"].apply(
            lambda ks: any(w in k.lower() for w in wanted for k in ks)
        )]
        applied["keywords"] = keywords

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
        "scope": "the screen and the shortlist now cover exactly these films",
    }
    if list_all or matched <= PREVIEW_FILMS:
        # Labels are ~10 tokens each, so even all 316 costs less than a fifth
        # of one read_synopses call. Completeness is affordable; hiding matches
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
# Tool 2: lexical screen, for negations
# ---------------------------------------------------------------------

def screen_out(
    words: list[str] | None = None,
    keep: str = "clear",
    and_words: list[str] | None = None,
    exclude_phrases: list[str] | None = None,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Scan every plot passage of every candidate for `words`. Free, both ways.

    One pass over the corpus splits the candidates into those where a listed
    word appears and those where none does; `keep` decides which half survives.
    Either way the scan is exhaustive over the word list, which is the property
    neither ranking nor reading can offer: no film escapes it by placing
    eleventh, and none is missed because the budget ran out at eight synopses.

      keep="clear"    (default) the negation case. "nobody dies", "nothing
                      scary". Searching for these returns the films that fail
                      them, because that is what those plots say, so they are
                      screened instead.
      keep="flagged"  the presence case. "an animal that wears a hat", "a film
                      with a train". A concrete noun that would literally be
                      written down in a plot description is found by scanning
                      for the word, not by ranking whole passages on overall
                      similarity -- ranking "an animal that wears a hat"
                      retrieves animal films, since one incidental word barely
                      moves a 300-token passage vector.

    keep="clear" ORDERS the working set; it does not shrink it. A word scan
    cannot tell an attempt from an outcome, so removing the flagged half made
    the scan into a verdict and put Toy Story, Monsters, Inc. and Zootopia out
    of reach of a "nobody dies" request over `killing`, `kill` and `murdered`
    that no character dies of. The clean films become the front of the
    verification queue instead, and the flagged ones are checked after.

    keep="flagged" does narrow, because there a match is the finding rather
    than a suspicion.
    """
    resolved = screening.resolve_words(words)
    if not resolved:
        return {"error": "Pass `words`: the words and phrases a plot might use "
                         "for the one thing you are scanning for."}

    keep = (keep or "clear").strip().lower()
    if keep not in ("clear", "flagged"):
        return {"error": "`keep` must be 'clear' (films where no listed word "
                         "appears) or 'flagged' (films where one does)."}

    # Always hand the screen an explicit candidate list. Left to infer its own
    # universe it can only see films that have plot passages, so the four films
    # with none would disappear from all three buckets rather than being
    # reported as unverifiable -- they would look considered when they were not.
    # rag/ does not import the catalog, so the fallback belongs here.
    candidates = ctx.candidates() if ctx is not None else None
    if candidates is None:
        candidates = [int(i) for i in catalog.movies()["id"]]

    result = screening.screen(resolved, candidate_ids=candidates,
                              and_words=and_words,
                              exclude_phrases=exclude_phrases)

    # Writing the list per request is more flexible than a fixed vocabulary and
    # has one failure the fixed one did not: a list too short to cover how the
    # thing is actually written. Three words for "nobody dies" reports a clean
    # scan that means almost nothing, and it reports it exhaustively, which is
    # what makes it convincing. Said here rather than assumed.
    thin_list = len(resolved) < 6 and keep == "clear"

    clear, flagged = result["clear"], result["flagged"]
    thin = result["insufficient_text"]

    if ctx is not None:
        if keep == "clear":
            # ORDERS, never narrows. Deleting the flagged half is what made a
            # word scan into a verdict: 194 of 316 films flag on the death
            # vocabulary, and Toy Story flags on "killing" and "murdered" for
            # a false belief, Monsters, Inc. and Zootopia on an attempt. All
            # three were removed from the request before the Verifier -- which
            # exists precisely to tell an attempt from an outcome -- could look
            # at one of them.
            #
            # So the clean films become a preference: CandidateWalk checks them
            # first because they are the likeliest to pass, and a flagged film
            # is checked afterwards rather than discarded. The count stays
            # exhaustive and free, which is the part nothing else can do.
            ranked = sorted(clear, key=lambda i: -(catalog.rating_of(i) or 0.0))
            ctx.preferred = ranked
            ctx.scope_note = (f"{len(ctx.working_set or [])} films "
                              f"({len(clear)} clear of {len(resolved)} words, "
                              f"{len(flagged)} flagged, none removed)")
            ctx.trace.append(ctx.scope_note)
        else:
            # The forward scan still narrows. Here a match IS the finding --
            # "a film with a train" -- so the films without one are genuinely
            # out of scope rather than merely unproven.
            ctx.narrow(set(flagged), f"kept the {len(flagged)} films mentioning "
                                     f"any of {len(resolved)} words")

    def labels(ids: list[int]) -> list[dict[str, Any]]:
        rows = [(catalog.label_of(i), catalog.rating_of(i)) for i in ids]
        rows = [(f, r) for f, r in rows if f]
        rows.sort(key=lambda fr: -(fr[1] or 0))
        return [{"film": f, "rating": round(r, 2) if r else None} for f, r in rows]

    def with_evidence(ids: list[int]) -> list[dict[str, Any]]:
        """Films best-rated first, each carrying the passage that matched."""
        rows = [(i, catalog.label_of(i), catalog.rating_of(i)) for i in ids]
        rows = [r for r in rows if r[1]]
        rows.sort(key=lambda r: -(r[2] or 0))
        return [
            {
                "film": lab,
                "matched": [h["word"] for h in result["evidence"].get(i, [])],
                "quote": (result["evidence"].get(i) or [{}])[0].get("quote"),
            }
            for i, lab, _ in rows
        ]

    out: dict[str, Any] = {
        "screened_for": resolved if len(resolved) <= 12 else
                        resolved[:12] + [f"... and {len(resolved) - 12} more"],
        **({"together_with": and_words} if and_words else {}),
        "kept": keep,
        "clear": len(clear),
        "flagged": len(flagged),
        "insufficient_text": len(thin),
        **({"thin_word_list": (
            f"Only {len(resolved)} word(s) were scanned for. An absence "
            "established over a short list is a weak finding stated "
            "exhaustively, which is the most convincing way to be wrong. If "
            "the thing has other common wordings, run the scan again with "
            "them before you rely on `clear`."
        )} if thin_list else {}),
        "meaning": (
            "clear = no listed word appears anywhere in the film's plot text, "
            f"and it has at least {result['min_screen_tokens']} tokens of plot "
            "for that absence to be meaningful. flagged = a word appears, which "
            "may be an attempt, threat or rumour rather than an outcome; read "
            "the quote before deciding what it shows. insufficient_text = too "
            "little plot text to screen; these were NOT verified either way."
        ),
    }

    if keep == "clear":
        clear_labels = labels(clear)
        out["scope"] = "search and reading now cover only the clear films"
        out["clear_films"] = clear_labels[:PREVIEW_FILMS]
        if len(clear_labels) > PREVIEW_FILMS:
            out["clear_note"] = (
                f"Showing the {PREVIEW_FILMS} best rated of {len(clear_labels)} "
                "clear films; all of them are in scope."
            )

        # Evidence only when the flagged set is small enough to be worth
        # quoting. Beyond that the model should not be resolving flags one by
        # one anyway -- it should recommend from the clear set.
        if flagged and len(flagged) <= MAX_FLAGGED_EVIDENCE:
            out["flagged_films"] = with_evidence(flagged)
        elif flagged:
            out["flagged_note"] = (
                f"{len(flagged)} films flagged -- too many to quote. Recommend "
                "from the clear set, or narrow further before screening."
            )
    else:
        # The matches ARE the answer here, so they always come back quoted --
        # a presence claim the model cannot cite is exactly the kind it should
        # not be making. The quote also carries the part a word list cannot
        # judge: "the hat lands on Tod" and "Bowler Hat Guy" both match `hat`,
        # and only the passage says which one involves an animal.
        matches = with_evidence(flagged)
        out["scope"] = "search and reading now cover only the matching films"
        if and_words:
            # Co-occurrence is proximity, not a relationship. Alice Through the
            # Looking Glass has a Cheshire Cat and a Hatter in one passage and
            # no cat in a hat anywhere. A word list cannot close that gap, and
            # naming the film with a note that the link is unproven is the
            # failure this whole path exists to avoid -- so say what the next
            # move is instead of leaving the planner to invent one.
            out["co_occurrence_warning"] = (
                "Both word lists appear in these passages. That is proximity, not a "
                "relationship: the text may pair the two things, or merely mention them "
                "in the same scene. Read the quote. If it does not plainly show the "
                "connection, call `read_synopses` on these films with `about` set to the "
                "connection you need -- and if that does not settle it either, say the "
                "plot text does not establish it and name nothing. Do not list a film "
                "alongside a note that the link is unproven."
            )
        out["matching_films"] = matches[:PREVIEW_FILMS]
        out["meaning"] = (
            "Every plot passage of every candidate was scanned, so this is "
            "every film in scope whose plot text uses one of these words -- "
            "exhaustive over the word list, not over the idea: a film can show "
            "the thing without naming it, and a match may not mean what you "
            f"want. Read the quote. {len(thin)} film(s) had under "
            f"{result['min_screen_tokens']} tokens of plot text and were not "
            "searched either way."
        )
        if not matches:
            out["no_matches"] = (
                "No film in scope uses any of these words. Plot text records "
                "events rather than appearance, so a visual detail may simply "
                "not be written down anywhere -- say that is what happened "
                "rather than recommending a film you could not verify."
            )
        elif len(matches) > PREVIEW_FILMS:
            out["matching_note"] = (
                f"Showing the {PREVIEW_FILMS} best rated of {len(matches)} "
                "matching films; all of them are in scope."
            )

        # A wide flat match on a presence query is nearly always a request that
        # paired two things, scanned as one list. The words are matched with
        # OR, so every film mentioning either one lands here, and answering
        # from that set means naming films and explaining why they do not fit.
        # The tool knows this before the planner does, so it says so.
        if not and_words and len(matches) > MAX_FLAGGED_EVIDENCE:
            out["consider_pairing"] = (
                f"{len(matches)} films matched, because these {len(resolved)} words are "
                "matched with OR: a film needs only one of them. If the request pairs "
                "two things -- 'a cat that wears a hat', 'a robot on a spaceship' -- put "
                "one thing's synonyms in `words` and the other's in `and_words`, and only "
                "films where both land in the same passage come back. Do not answer from "
                "this wide set by naming a film and noting which half it fails."
            )

    if thin:
        out["insufficient_films"] = [catalog.label_of(i) for i in thin][:PREVIEW_FILMS]

    return out


# ---------------------------------------------------------------------
# Tool 3: semantic search
# ---------------------------------------------------------------------

def search_plots(
    query: str,
    top_k: int = 10,
    ignore_scope: bool = False,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Rank movies by how well their plot matches a described story.

    Search is passage-level: each film's synopsis is split into ~300-token
    passages and scored independently, so a single story beat buried in a long
    plot is findable. Each result carries the passage that matched, which is
    evidence the planner can quote instead of inferring.
    """
    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    scoped = None if (ignore_scope or ctx is None) else ctx.candidates()

    # Plot-bearing corpora only. The index also holds cast lists and one-line
    # blurbs, which score plausibly on story questions without answering them:
    # asked "does anyone die", a cast list came second. They stay searchable
    # deliberately, but not when the question is about what happens.
    hits = retrieval.search(query, top_k=top_k, candidate_ids=scoped,
                            sources=DEFAULT_SOURCES)

    df = catalog.movies().set_index("id")
    results = []
    for hit in hits:
        movie_id = hit["movie_id"]
        if movie_id not in df.index:
            continue
        row = df.loc[movie_id]
        results.append({
            "film": f"{row['title']} ({int(row['release_year'])})",
            "similarity": round(float(hit["score"]), 4),
            "rating": round(float(row["weighted_rating"]), 2),
            "matching_passage": _trim(hit["passage"], MAX_PASSAGE_CHARS),
            "passages_matched": hit["passage_count"],
        })

    out: dict[str, Any] = {
        "query": query,
        # This tool makes a real model call -- the query is embedded before it
        # can be scored -- and the assignment asks for every model call to be
        # visible in the trace. Naming it here puts it in the step record
        # rather than leaving it implied by the tool's existence.
        "model_call": {
            "model": ragcfg.EMBED_MODEL,
            "kind": "embedding",
            "input": query,
            "dimensions": retrieval.coverage().get("dim"),
        },
        "searched_within": "whole catalog" if scoped is None else f"{len(scoped)} films in scope",
        "returned": len(results),
        "results": results,
        # Only films with plot text are indexed, so a query restricted to
        # candidates that have none legitimately returns nothing. Say so,
        # rather than letting it look like "no good match".
        "note": (
            "No indexed passages exist for the given candidates; semantic "
            "search cannot rank them. Filter with require_synopsis=true first."
            if not results else None
        ),
    }

    # The score the planner already has, read back to it as a verdict. Asked
    # "a film where someone pretends to love another to seize power", the
    # abstract phrasing returned five films in a 0.29-0.35 band and the answer
    # named three of them while explaining that none actually fit. The number
    # saying so was in the result all along.
    if results and results[0]["similarity"] < WEAK_MATCH_SIMILARITY:
        out["weak_match"] = (
            f"Top similarity {results[0]['similarity']:.2f}, below "
            f"{WEAK_MATCH_SIMILARITY}. On this corpus that band means the query "
            "was probably phrased as a theme rather than as an event, and these "
            "films are unlikely to fit. Re-run it as something a plot summary "
            "would literally narrate -- name the action, who does it and what "
            "happens -- before answering from these results. Do not recommend a "
            "film here while explaining that it does not match."
        )

    return out


# How many films each condition's own search contributes. Twenty is wide
# enough that a film satisfying every condition moderately still places on
# each list, which is the recall this whole mechanism depends on, and narrow
# enough that the fused table stays readable.
SHORTLIST_PER_CONDITION = 20

# How many fused rows come back to the planner. The full ordering is kept in
# the context regardless -- this bounds what is shown, not what is considered.
SHORTLIST_ROWS = 15


def build_shortlist(
    conditions: list[str] | None = None,
    top_k: int = SHORTLIST_PER_CONDITION,
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

    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    scoped = ctx.candidates() if ctx is not None else None

    lists: dict[str, list[int]] = {}
    passages: dict[tuple[str, int], str] = {}
    for condition in wanted:
        hits = retrieval.search(condition, top_k=top_k, candidate_ids=scoped,
                                sources=DEFAULT_SOURCES)
        lists[condition] = [h["movie_id"] for h in hits]
        for h in hits:
            passages[(condition, h["movie_id"])] = h["passage"]

    ratings = {int(i): (catalog.rating_of(int(i)) or 0.0)
               for ids in lists.values() for i in ids}
    fused = shortlist.fuse(lists, ratings=ratings)

    if ctx is not None:
        ctx.shortlist = [c.movie_id for c in fused]
        ctx.conditions = wanted

    rows = [shortlist.explain(c, wanted, catalog.label_of(c.movie_id) or str(c.movie_id))
            for c in fused[:SHORTLIST_ROWS]]

    full = len(fused)
    covered_all = sum(1 for c in fused if c.covered == len(wanted))
    return {
        "conditions": wanted,
        "searched_within": "whole catalog" if scoped is None
                           else f"{len(scoped)} films in scope",
        "candidates": full,
        "matching_every_condition": covered_all,
        "shortlist": rows,
        "note": (
            f"{full} films placed for at least one condition; {covered_all} placed for "
            f"all {len(wanted)}. Ordered by conditions matched, then average rank -- so "
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


# Conditions plot text can never settle, because they are catalog columns.
# Sent to the Verifier they come back `unclear` on every film, and since
# acceptance needs every condition to say yes, one of them makes the request
# unanswerable: "Disney studio" and "released in 1990 or later" together
# verified thirteen candidates to zero on a query with a real answer.
#
# Tight on purpose. "released from prison" and "a character counts to 1990"
# must not match, so each pattern needs the structural context that makes it a
# catalog fact rather than a story event.
_STRUCTURED_CONDITION = re.compile(
    r"""(
          \breleased?\s+(in|on|after|before|from|no\s+earlier\s+than)\s+(19|20)\d{2}
        | \b(19|20)\d{2}\s+or\s+(later|earlier|newer|older)
        | \bfrom\s+(the\s+)?(19|20)\d0s\b
        | \brelease\s+year\b
        | \byears?\s+(19|20)\d{2}\s*\+
        | ^\s*(19|20)\d{2}\s*\+\s*$
        | \b(from|after|since|before)\s+(19|20)\d{2}\s*\+?\s*$
        | ^\s*(a|an|the)?\s*(walt\s+)?(disney|pixar)(\s+(studio|film|movie|production|picture))?\s*$
        | \bis\s+a\s+(walt\s+)?(disney|pixar)\b
        | \bproduced\s+by\s+(walt\s+)?(disney|pixar)\b
        | \bruntime\b
        | \b(under|over|less\s+than|more\s+than|at\s+most|at\s+least)\s+\d+\s*(minutes|mins)\b
        | \b\d+\s*minutes\s+(long|or\s+(less|fewer|more))\b
        | \bspoken\s+language\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _drop_structured(conditions: list[str]) -> tuple[list[str], list[str]]:
    """Split conditions into ones plot text can settle and ones it cannot."""
    story, structured = [], []
    for c in conditions:
        (structured if _STRUCTURED_CONDITION.search(c) else story).append(c)
    return story, structured


def verify_candidates(
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

    conditions, structured = _drop_structured(conditions)
    if not conditions:
        return {
            "error": "Every condition given was a catalog fact, not a story fact.",
            "not_verifiable": structured,
            "note": (
                "Year, studio, runtime and language are columns. filter_catalog "
                "already guaranteed them for every film in scope -- plot text cannot "
                "confirm them and the Verifier would return `unclear` on all of them. "
                "Pass only conditions the story settles."
            ),
        }

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
    # never needed a semantic search -- everything settled by filter and
    # screen -- could only ever check the films the planner happened to name,
    # so thirteen survivors got six verifications and the budget went unspent.
    if ctx is not None and ctx.shortlist:
        rest = list(ctx.shortlist)
    elif ctx is not None and ctx.working_set:
        rest = sorted(ctx.working_set,
                      key=lambda i: -(catalog.rating_of(i) or 0.0))
    else:
        rest = []

    # A lexical screen for an absence puts its clean films first. They are the
    # likeliest to pass, so the budget reaches an answer sooner -- but the
    # flagged ones follow rather than being dropped, which is the whole point
    # of the screen no longer filtering.
    if ctx is not None and ctx.preferred:
        pref = [i for i in ctx.preferred if i in set(rest)]
        rest = pref + [i for i in rest if i not in set(pref)]
    tail = [i for i in rest if i not in set(named)]
    order = named + tail
    if not order:
        return {"error": "no candidates: call build_shortlist first, or name films."}

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
        result = verifier.verify(label, conditions, _trim(text, MAX_SYNOPSIS_CHARS))
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
        **({"not_verifiable": structured,
            "not_verifiable_note": (
                "Dropped before checking: these are catalog facts, not story facts, and "
                "filter_catalog already guaranteed them for every film in scope. They "
                "were NOT ignored -- they were enforced earlier and exactly."
            )} if structured else {}),
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
                            "with search_plots, which cannot see it."
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
                    "keywords": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Thematic keywords, ANY may match, substring. "
                            "e.g. ['princess','royalty']. Good for topics the "
                            "genre list is too coarse for."
                        ),
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
            "name": "screen_out",
            "description": (
                "Scan every plot passage of every candidate for a word list, "
                "and keep either the films where none of them appears "
                "(keep='clear', the default) or the films where one does "
                "(keep='flagged'). Free, and exhaustive over the word list in "
                "a way ranking cannot be -- no film escapes by placing "
                "eleventh. Use keep='clear' for an absence: 'nobody dies', "
                "'nothing scary'. Searching for those returns the films where "
                "somebody does, because that is what their plots say. Use "
                "keep='flagged' for the presence of something concrete enough "
                "to be written down in a plot description -- 'an animal that "
                "wears a hat', 'a film with a train' -- because ranking that "
                "request scores whole passages on overall similarity and "
                "returns animal films, one incidental word being far too small "
                "to move the vector. Do NOT use either direction for something "
                "the catalog stores ('not Pixar', 'no musicals'): those are "
                "filter_catalog arguments. Exhaustive over the word list is "
                "not the same as settled either way -- a film can narrate the "
                "event in other words or show the thing without naming it -- "
                "so quotes come back and you should read them before deciding "
                "what a match shows. Escalate to read_synopses when it matters "
                "and the scan left it unresolved. Run it AFTER filter_catalog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "**You write this list, for this request.** Every "
                            "word and phrase a plot summary might use for ONE "
                            "thing: inflections, synonyms, and the indirect "
                            "ways it gets written. For a death that is "
                            "['dies','died','dead','killed','murdered',"
                            "'perished','funeral','buried','sacrificed',"
                            "'passes away','loses his life','never returns'] "
                            "-- and for a hat, ['hat','hats','cap','bonnet',"
                            "'fez','top hat']. Multi-word phrases work; "
                            "matching is on word boundaries and "
                            "case-insensitive. Be generous: the scan is free "
                            "and exhaustive, so a missed synonym is a missed "
                            "film, while a spurious one only sends a film to "
                            "be checked. Ten to twenty entries is normal for "
                            "an abstract idea, three to six for a concrete "
                            "object. **Never mix two things into this list.** "
                            "['cat','hat'] matches a cat OR a hat anywhere in "
                            "a film, which is 51 films of near-misses; the "
                            "second thing belongs in `and_words`."
                        ),
                    },
                    "exclude_phrases": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Phrasings that contain one of your words without "
                            "meaning it, blanked before matching. **You write "
                            "these too**, for the words you chose: scanning "
                            "for death wants ['dead end','deadline','dead "
                            "heat','kill time','drop dead'], and scanning for "
                            "'shot' wants ['shot a photograph','camera shot'] "
                            "only when the question is violence. Nothing is "
                            "excluded unless you say so. Getting one wrong is "
                            "cheap in one direction only: a missing phrase "
                            "leaves a film flagged and therefore checked, "
                            "while an over-broad phrase can blank a real "
                            "event, so keep them literal and specific."
                        ),
                    },
                    "and_words": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "A SECOND word list that must appear in the same "
                            "passage as one from `words`. Use it whenever the "
                            "request pairs two things -- 'a cat that wears a "
                            "hat', 'a robot on a spaceship'. Put the synonyms "
                            "of one thing in `words` and of the other here. "
                            "Without it the scan matches either word anywhere "
                            "in a film: cat-or-hat returns 51 films, cat-words "
                            "alone 27, and cat-with-hat-in-one-passage 2. The "
                            "first two are lists of near-misses you would have "
                            "to explain away; the third is the answer."
                        ),
                    },
                    "keep": {
                        "type": "string",
                        "enum": ["clear", "flagged"],
                        "description": (
                            "Which half survives. 'clear' (default) keeps the "
                            "films where no listed word appears -- use for an "
                            "absence. 'flagged' keeps the films where one "
                            "does, each quoted -- use to find something."
                        ),
                    },
                },
                "required": ["words"],
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
                            "the negative ones here: this is where an absence "
                            "is actually adjudicated, and a lexical flag is "
                            "only a reason to look. Defaults to the "
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
            "name": "build_shortlist",
            "description": (
                "Search EVERY story condition separately and fuse the "
                "rankings into one ordered shortlist. **Use this instead of "
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
                "NOT evidence -- verify with read_synopses before "
                "recommending."
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
                            "nothing can be searched for by not happening, so "
                            "that is screen_out's job."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Films per condition, default 20. Raise it when a "
                            "condition is broad and you want more of the "
                            "middle of its ranking to reach the fusion."
                        ),
                    },
                },
                "required": ["conditions"],
            },
        },
    },
]

# Human-readable trace names, matching the boxes in the Architecture tab.
TRACE_NAMES = {
    "filter_catalog": "CatalogFilter",
    # Both of these read plot text, and the old names described how rather
    # than what: "LexicalScan" and "SemanticRetrieval" put a word scan and a
    # vector search at arm's length from each other when the useful
    # distinction is exact versus approximate over the same corpus. Fusing
    # several conditions' rankings is something SemanticRetrieval does, not a
    # thing of its own.
    "screen_out": "LexicalScan",
    "build_shortlist": "SemanticRetrieval",
    # The TOOL, not the module it calls. read_synopses/Observer already made
    # this distinction and verify_candidates did not, so a tool step was
    # logged under a subagent's name and the Verifier looked like something
    # the planner could invoke directly. It cannot: it walks candidates and
    # sends one model call per film, the way SynopsisReader sends one to the
    # Observer.
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


# search_plots is deliberately absent. It is still the primitive that
# build_shortlist calls once per condition, but it is no longer offered to the
# model, because "search one condition and read its top hits" is the greedy
# move this design exists to prevent -- and a tool the model can reach is a
# tool a prompt has to talk it out of using. One condition is build_shortlist
# with a list of one, so nothing is lost.
_DISPATCH = {
    "filter_catalog": filter_catalog,
    "screen_out": screen_out,
    "build_shortlist": build_shortlist,
    "verify_candidates": verify_candidates,
}
