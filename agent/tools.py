"""The three tools MoviBot can call, and their function-calling schemas.

Design note -- why three tools and not six:

The catalog is 238 films. At that size the expensive thing is not searching,
it is reasoning: every LLM turn costs money and latency, so the tools are
shaped to answer a query in as few turns as possible rather than to mirror a
tidy pipeline. Each tool does one thing the others cannot:

    filter_catalog   answers from columns      (year, genre, language, studio)
    search_plots     answers from meaning      (theme, character, premise)
    read_synopses    answers from full text    (does anyone die, who betrays whom)

A query needs only the tools that its constraints actually require, so simple
queries finish in one round and hard ones in three.

Guardrails live here, not in the prompt. The model cannot forget them, and a
bad plan cannot bypass them:

  * results are ordered by `weighted_rating`, never raw `vote_average`
  * `filter_catalog` returns at most MAX_FILTER_RESULTS rows
  * `read_synopses` reads at most MAX_SYNOPSES movies, truncated to
    MAX_SYNOPSIS_CHARS each, which is what bounds the context cost per turn
"""

from __future__ import annotations

from typing import Any

from agent import catalog
from rag import store as retrieval

# A structured filter is for narrowing, not for dumping the catalog into the
# context window. 40 rows is enough for the model to choose from, and roughly
# 1.5 KB of JSON.
MAX_FILTER_RESULTS = 40

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
    limit: int = MAX_FILTER_RESULTS,
) -> dict[str, Any]:
    """Filter the catalog on structured columns. Free, no model call."""
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

    # Case-insensitive exact title match, so excluding "Frozen" does not also
    # drop "Frozen Fever" -- a substring match here would quietly over-exclude.
    if exclude_titles:
        blocked = {t.strip().lower() for t in exclude_titles}
        df = df[~df["title"].str.lower().isin(blocked)]
        applied["exclude_titles"] = exclude_titles

    if require_synopsis:
        df = df[df["id"].apply(catalog.has_synopsis)]
        applied["require_synopsis"] = True

    matched = len(df)
    df = df.sort_values("weighted_rating", ascending=False)
    capped = min(int(limit), MAX_FILTER_RESULTS)

    return {
        "matched": matched,
        "returned": min(matched, capped),
        "truncated": matched > capped,
        "filters_applied": applied,
        "movies": [_brief(row) for _, row in df.head(capped).iterrows()],
    }


def _brief(row) -> dict[str, Any]:
    """Compact row for the model: enough to choose, small enough to be cheap."""
    return {
        "movie_id": int(row["id"]),
        "title": str(row["title"]),
        "year": int(row["release_year"]),
        "runtime": int(row["runtime_minutes"]),
        "genres": list(row["genres"]),
        "rating": round(float(row["weighted_rating"]), 2),
        "raw_rating": round(float(row["vote_average"]), 1),
        "votes": int(row["vote_count"]),
        "has_synopsis": catalog.has_synopsis(int(row["id"])),
    }


# ---------------------------------------------------------------------
# Tool 2: semantic search
# ---------------------------------------------------------------------

def search_plots(
    query: str,
    candidate_ids: list[int] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Rank movies by how well their plot matches a described story.

    Search is passage-level: each film's synopsis is split into ~300-token
    passages and scored independently, so a single story beat buried in a long
    plot is findable. Each result carries the passage that matched, which is
    evidence the planner can quote instead of inferring.
    """
    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    hits = retrieval.search(query, top_k=top_k, candidate_ids=candidate_ids)

    df = catalog.movies().set_index("id")
    results = []
    for hit in hits:
        movie_id = hit["movie_id"]
        if movie_id not in df.index:
            continue
        row = df.loc[movie_id]
        results.append({
            "movie_id": int(movie_id),
            "title": str(row["title"]),
            "year": int(row["release_year"]),
            "similarity": round(float(hit["score"]), 4),
            "rating": round(float(row["weighted_rating"]), 2),
            "matching_passage": _trim(hit["passage"], MAX_PASSAGE_CHARS),
            "passages_matched": hit["passage_count"],
        })

    return {
        "query": query,
        "searched_within": len(candidate_ids) if candidate_ids else "whole catalog",
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
    movie_ids: list[int],
    about: str | None = None,
    max_chars: int = MAX_SYNOPSIS_CHARS,
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
    ids = [int(m) for m in (movie_ids or [])][:MAX_SYNOPSES]
    max_chars = max(500, min(int(max_chars), MAX_SYNOPSIS_CHARS))

    # Rank passages once across all requested films, then group.
    relevant: dict[int, list[dict[str, Any]]] = {}
    if about and about.strip() and ids:
        for p in retrieval.search_passages(about, top_k=len(ids) * 6, candidate_ids=ids):
            relevant.setdefault(p["movie_id"], []).append(p)

    out = []
    for movie_id in ids:
        title = catalog.title_of(movie_id)
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
                "movie_id": movie_id,
                "title": title,
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
                "movie_id": movie_id, "title": title, "synopsis": None,
                "note": "No plot text available for this title.",
            })
            continue

        truncated = len(text) > max_chars
        out.append({
            "movie_id": movie_id,
            "title": title,
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
        "requested": len(movie_ids or []),
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
                    "limit": {
                        "type": "integer",
                        "description": f"Max rows to return (hard cap {MAX_FILTER_RESULTS}).",
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
                "learning to let go'. Pass candidate_ids from a prior "
                "filter_catalog call to rank within that set instead of the "
                "whole catalog. Only films with full plot text are searchable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The story in plain language. Describe the plot, not the metadata.",
                    },
                    "candidate_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Restrict ranking to these movie_ids, typically from filter_catalog.",
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
                    "movie_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": f"Movie ids to read, at most {MAX_SYNOPSES}.",
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
                "required": ["movie_ids"],
            },
        },
    },
]

TOOLS = {
    "filter_catalog": filter_catalog,
    "search_plots": search_plots,
    "read_synopses": read_synopses,
}

# Human-readable trace names, matching the boxes in assets/architecture.png.
TRACE_NAMES = {
    "filter_catalog": "CatalogFilter",
    "search_plots": "PlotSearch",
    "read_synopses": "SynopsisReader",
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name. Bad names/arguments become results, not exceptions.

    A malformed tool call is something the model can recover from on the next
    turn if we hand the error back, but a raised exception would abort the
    whole request.
    """
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(TOOLS)}."}
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surfaced to the model deliberately
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
