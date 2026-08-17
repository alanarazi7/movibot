"""The four tools MoviBot can call, and their function-calling schemas.

Design note -- why four tools, and why in this order:

The catalog is 238 films. At that size the expensive thing is not searching, it
is reasoning: every model turn costs money and latency. So the tools are ordered
cheapest-and-most-exhaustive first, and each one hands the next a smaller
candidate set:

    filter_catalog   columns      238 -> N     free, exact, exhaustive
    screen_out       regex        N -> clear   free, exhaustive, one-sided
    search_plots     vectors      N -> ~10     one embedding, ~$0.0000002
    read_synopses    full text    <= 8 films   free, but token-heavy

Each answers something the others cannot. Columns settle era, studio and
language. A lexical screen settles negations ("nobody dies") exhaustively, which
ranking cannot: embed "nobody dies" and the top hits are the films where
somebody does. Vectors settle positive story questions. Full text settles what
actually happens.

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

from dataclasses import dataclass, field

from agent import catalog
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

    def narrow(self, ids: set[int], why: str) -> None:
        self.working_set = ids
        self.scope_note = f"{len(ids)} films ({why})"
        self.trace.append(self.scope_note)

    def candidates(self) -> list[int] | None:
        return sorted(self.working_set) if self.working_set is not None else None

# How many labels come back when a filter matches more than this. The full set
# is always in scope regardless; this only bounds what is *shown*, and
# list_all=true lifts it. Not a cost guard -- all 238 labels are ~2,300 tokens.
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

# Each search hit carries the passage that matched, as evidence. Long enough
# to judge a story beat, short enough that 25 of them stay affordable.
MAX_PASSAGE_CHARS = 1200


# ---------------------------------------------------------------------
# Tool 1: structured filtering
# ---------------------------------------------------------------------

def filter_catalog(
    year_min: int | None = None,
    year_max: int | None = None,
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
        wanted = [l.lower() for l in languages]
        df = df[df["spoken_languages"].apply(
            lambda ls: any(w in l.lower() for w in wanted for l in ls)
        )]
        applied["languages"] = languages

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

    labelled = [
        {"film": f"{r.title} ({int(r.release_year)})",
         "rating": round(float(r.weighted_rating), 2)}
        for r in df.itertuples()
    ]

    out: dict[str, Any] = {
        "matched": matched,
        "filters_applied": applied or "none",
        "scope": "search_plots and read_synopses now cover exactly these films",
    }
    if list_all or matched <= PREVIEW_FILMS:
        # Labels are ~10 tokens each, so even all 238 costs less than a fifth
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
    vocabulary: str | None = None,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Exclude every candidate whose plot mentions any of `words`. Free.

    This is how a negation gets answered: exhaustively and in one pass, rather
    than by ranking (which surfaces the films that fail the condition) or by
    reading (which is bounded at eight films and cannot cover a set of 200).

    The working set narrows to the `clear` films, so whatever follows is already
    restricted to candidates that passed. Flagged films are NOT rejected -- a
    match may be an attempt or a rumour -- and can still be read by name.
    """
    resolved = screening.resolve_words(words, vocabulary)
    if not resolved:
        return {"error": "Pass `words`, or a `vocabulary` from: "
                         + ", ".join(screening.VOCABULARIES)}

    # Always hand the screen an explicit candidate list. Left to infer its own
    # universe it can only see films that have plot passages, so the four films
    # with none would disappear from all three buckets rather than being
    # reported as unverifiable -- they would look considered when they were not.
    # rag/ does not import the catalog, so the fallback belongs here.
    candidates = ctx.candidates() if ctx is not None else None
    if candidates is None:
        candidates = [int(i) for i in catalog.movies()["id"]]

    result = screening.screen(resolved, candidate_ids=candidates)

    clear, flagged = result["clear"], result["flagged"]
    thin = result["insufficient_text"]

    if ctx is not None:
        ctx.narrow(set(clear), f"screened out {len(resolved)} words, "
                               f"{len(flagged)} flagged")

    def labels(ids: list[int]) -> list[dict[str, Any]]:
        rows = [(catalog.label_of(i), catalog.rating_of(i)) for i in ids]
        rows = [(f, r) for f, r in rows if f]
        rows.sort(key=lambda fr: -(fr[1] or 0))
        return [{"film": f, "rating": round(r, 2) if r else None} for f, r in rows]

    clear_labels = labels(clear)
    out: dict[str, Any] = {
        "screened_for": resolved if len(resolved) <= 12 else
                        resolved[:12] + [f"... and {len(resolved) - 12} more"],
        "clear": len(clear),
        "flagged": len(flagged),
        "insufficient_text": len(thin),
        "scope": "search and reading now cover only the clear films",
        "meaning": (
            "clear = no listed word appears anywhere in the film's plot text, "
            f"and it has at least {result['min_screen_tokens']} tokens of plot "
            "for that absence to be meaningful. flagged = a word appears, which "
            "may be an attempt, threat or rumour rather than an outcome; read "
            "the quote before rejecting. insufficient_text = too little plot "
            "text to screen; these were NOT verified either way."
        ),
    }

    out["clear_films"] = clear_labels[:PREVIEW_FILMS]
    if len(clear_labels) > PREVIEW_FILMS:
        out["clear_note"] = (
            f"Showing the {PREVIEW_FILMS} best rated of {len(clear_labels)} "
            "clear films; all of them are in scope."
        )

    # Evidence only when the flagged set is small enough to be worth quoting.
    # Beyond that the model should not be resolving flags one by one anyway --
    # it should recommend from the clear set.
    if flagged and len(flagged) <= MAX_FLAGGED_EVIDENCE:
        out["flagged_films"] = [
            {
                "film": catalog.label_of(i),
                "matched": [h["word"] for h in result["evidence"].get(i, [])],
                "quote": (result["evidence"].get(i) or [{}])[0].get("quote"),
            }
            for i in flagged
        ]
    elif flagged:
        out["flagged_note"] = (
            f"{len(flagged)} films flagged -- too many to quote. Recommend from "
            "the clear set, or narrow further before screening."
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

    return {
        "query": query,
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


def _trim(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# ---------------------------------------------------------------------
# Tool 3: full plot text
# ---------------------------------------------------------------------

def read_synopses(
    films: list[str],
    about: str | None = None,
    max_chars: int = MAX_SYNOPSIS_CHARS,
    ctx: "ToolContext | None" = None,
) -> dict[str, Any]:
    """Return plot text so constraints no column encodes can be judged.

    This is the only way to answer things like "nobody dies" or "warns about
    trusting strangers": those are properties of the story, absent from
    genres, keywords, and the one-line overview alike.

    Pass `about` whenever the question is about a specific part of the story.
    Without it, a long synopsis is truncated from the beginning, which silently
    hides late plot developments -- reading the first 6,000 characters of
    Frozen meets Hans as a charming prince and never reaches his betrayal.
    With it, the most relevant passages are returned instead, in story order.
    """
    requested = list(films or [])[:MAX_SYNOPSES]
    max_chars = max(500, min(int(max_chars), MAX_SYNOPSIS_CHARS))

    ids, unknown = [], []
    for label in requested:
        movie_id = catalog.resolve(label)
        (ids.append(movie_id) if movie_id is not None else unknown.append(label))

    # Rank passages once across all requested films, then group.
    relevant: dict[int, list[dict[str, Any]]] = {}
    if about and about.strip() and ids:
        for p in retrieval.search_passages(about, top_k=len(ids) * 6, candidate_ids=ids):
            relevant.setdefault(p["movie_id"], []).append(p)

    out = []
    for movie_id in ids:
        title = catalog.label_of(movie_id)
        hits = relevant.get(movie_id)

        if hits:
            # Story order, not score order: the planner is reading a narrative,
            # and passages shuffled by relevance read as incoherent.
            chosen, used = [], 0
            for p in sorted(hits, key=lambda h: -h["score"]):
                if used + len(p["text"]) > max_chars:
                    continue
                chosen.append(p)
                used += len(p["text"])
            chosen.sort(key=lambda h: h["chunk_index"])
            out.append({
                "film": title,
                "mode": "relevant_passages",
                "about": about,
                "passages": [
                    {"chunk_index": p["chunk_index"],
                     "relevance": round(p["score"], 4),
                     "text": p["text"]}
                    for p in chosen
                ],
                "note": "Excerpts selected for relevance, shown in story order.",
            })
            continue

        text = catalog.synopsis(movie_id)
        if text is None:
            out.append({
                "film": title, "synopsis": None,
                "note": "No plot text available for this title.",
            })
            continue

        truncated = len(text) > max_chars
        out.append({
            "film": title,
            "mode": "full_text",
            "synopsis": text[:max_chars],
            "truncated": truncated,
            "note": (
                f"Truncated to the first {max_chars} of {len(text)} characters. "
                "Later plot developments are NOT shown -- pass `about` to get "
                "the passages relevant to your question instead."
                if truncated else None
            ),
        })

    return {
        "requested": len(requested),
        "unresolved": unknown or None,
        "returned": len(out),
        "capped_at": MAX_SYNOPSES,
        "synopses": out,
    }


# ---------------------------------------------------------------------
# Schemas + dispatch
# ---------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filter_catalog",
            "description": (
                "Filter the movie catalog on structured facts: year, genre, "
                "keyword, studio, spoken language, title exclusions. Use this "
                "FIRST whenever the request contains any hard constraint, and "
                "for anything about language, era, or studio, which plot "
                "search cannot see. Results come back best-first by a "
                "vote-count-weighted rating. The catalog is feature films "
                "only (238 Disney/Pixar titles, 1940-2017)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year_min": {"type": "integer", "description": "Earliest release year, inclusive."},
                    "year_max": {"type": "integer", "description": "Latest release year, inclusive."},
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
                        "description": "Spoken languages, ANY may match, e.g. ['Français'].",
                    },
                    "exclude_english_only": {
                        "type": "boolean",
                        "description": "Keep only movies with some non-English dialogue.",
                    },
                    "exclude_titles": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Exact titles to drop. Use when the user says 'besides X' or 'other than X'.",
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
                "Exclude every film whose plot mentions any of the given words. "
                "This is the RIGHT tool for a negative condition -- 'nobody "
                "dies', 'nothing scary', 'no romance' -- and search_plots is "
                "the wrong one, because searching for 'nobody dies' returns the "
                "films where somebody does. Free, and it checks every plot "
                "passage of every candidate rather than a top-ranked few, so "
                "the result is exhaustive. Films that match are flagged, not "
                "rejected: the match may be an attempt or a rumour, and a quote "
                "comes back so you can judge. Run it AFTER filter_catalog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vocabulary": {
                        "type": "string",
                        "enum": sorted(screening.VOCABULARIES),
                        "description": (
                            "A curated word list. Prefer this over inventing "
                            "words: 'death' covers 30+ terms including "
                            "'funeral', 'sacrificed' and 'perished'."
                        ),
                    },
                    "words": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Extra words to exclude on, added to `vocabulary` "
                            "if both are given. Matched on word boundaries, so "
                            "pass each inflection you care about."
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
            "name": "search_plots",
            "description": (
                "Rank movies by how closely their plot matches a described "
                "story, premise, character, or theme. Use for anything not "
                "expressible as a column: 'a rat who wants to cook', 'a father "
                "learning to let go'. Automatically limited to whatever "
                "filter_catalog last selected -- you do not pass candidates. "
                "Only films with plot text are searchable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The story in plain language. Describe the plot, not the metadata.",
                    },
                    "ignore_scope": {
                        "type": "boolean",
                        "description": (
                            "Search the whole catalog rather than the filtered set. "
                            "Only when the request has no structured constraint, or "
                            "the filter returned nothing and needs widening."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"How many to return (hard cap {MAX_SEARCH_RESULTS}).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_synopses",
            "description": (
                "Read the plot text of specific movies. This is the ONLY way "
                "to verify constraints about what happens in the story -- "
                "whether a character dies, who betrays whom, whether it is "
                "frightening. Never assume such things from genre or title. "
                f"Reads at most {MAX_SYNOPSES} movies per call, so shortlist "
                "with the other tools first. ALWAYS pass `about` when your "
                "question concerns a specific part of the story: without it, "
                "long plots are cut off at the start and late developments are "
                "silently missing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "films": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Films to read, named exactly as returned: "
                            "\"Title (Year)\", e.g. \"Frozen (2013)\". At most "
                            f"{MAX_SYNOPSES}."
                        ),
                    },
                    "about": {
                        "type": "string",
                        "description": (
                            "What you need to find out, e.g. 'does any major "
                            "character die' or 'is the prince trustworthy'. "
                            "Returns the passages relevant to this instead of "
                            "the opening of the plot. Strongly recommended."
                        ),
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": f"Budget per movie (hard cap {MAX_SYNOPSIS_CHARS}).",
                    },
                },
                "required": ["films"],
            },
        },
    },
]

# Human-readable trace names, matching the boxes in the Architecture tab.
TRACE_NAMES = {
    "filter_catalog": "CatalogFilter",
    "screen_out": "LexicalScreen",
    "search_plots": "PlotSearch",
    "read_synopses": "SynopsisReader",
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
    "screen_out": screen_out,
    "search_plots": search_plots,
    "read_synopses": read_synopses,
}
