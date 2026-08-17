"""The text corpora that can be indexed, and how to load each one.

Until now only MPST synopses were embedded, which is why 159 of 238 films are
semantically searchable while 234 are readable. The gap was an artifact of
build order rather than a decision: the index was built before the Wikipedia
cache existed and never revisited.

Each corpus is a different kind of evidence, and keeping them separable matters
because they answer different questions. A plot tells you whether someone dies;
a reception section tells you whether a film is considered frightening. Mixing
them into one undifferentiated index would make it impossible to ask only the
first, and would let a critic's phrasing outrank the story itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from rag.config import DATA_READY


@dataclass(frozen=True)
class Corpus:
    key: str
    label: str
    description: str
    load: Callable[[], pd.DataFrame]  # -> columns: movie_id, title, text


def _from_csv(filename: str, id_col: str, text_col: str) -> Callable[[], pd.DataFrame]:
    def load() -> pd.DataFrame:
        df = pd.read_csv(f"{DATA_READY}/{filename}")
        out = df[[id_col, "title", text_col]].rename(
            columns={id_col: "movie_id", text_col: "text"}
        )
        out = out[out["text"].apply(lambda t: isinstance(t, str) and bool(t.strip()))]
        return out.reset_index(drop=True)

    return load


# The CSV filenames below are historical: there is no Pinecone and no Supabase in
# this project, and neither ever ran. See agent/catalog.py for why the names were
# kept. "pinecone_candidates.csv" holds the MPST synopses; "supabase_movies.csv"
# is the catalog.
CORPORA: dict[str, Corpus] = {
    "mpst": Corpus(
        key="mpst",
        label="Plot synopses (MPST)",
        description=(
            "Full plot synopses, median 892 words. The richest source and the "
            "only one that narrates a whole story start to finish."
        ),
        load=_from_csv("pinecone_candidates.csv", "movie_id", "plot_synopsis"),
    ),
    "wiki_plot": Corpus(
        key="wiki_plot",
        label="Wikipedia plot",
        description=(
            "Wikipedia's Plot section. Shorter than MPST but covers 75 films "
            "MPST misses, which is most of the semantic coverage gap."
        ),
        load=_from_csv("wikipedia_cache.csv", "id", "plot_text"),
    ),
    "wiki_context": Corpus(
        key="wiki_context",
        label="Wikipedia context",
        description=(
            "Everything on the page that is not plot: cast, production, "
            "reception, themes. Answers tone questions the plot cannot settle."
        ),
        load=_from_csv("wikipedia_cache.csv", "id", "non_plot_text"),
    ),
    "overview": Corpus(
        key="overview",
        label="Catalog overview",
        description=(
            "The one-paragraph blurb from the catalog, median 53 words. Thin, "
            "but it is the only text that exists for every one of the 238 films."
        ),
        load=_from_csv("supabase_movies.csv", "id", "overview"),
    ),
}

DEFAULT_SOURCES = ["mpst", "wiki_plot"]


def resolve(sources: list[str] | None) -> list[str]:
    """Normalise a source selection. None or ["all"] means every corpus."""
    if not sources or "all" in sources:
        return list(CORPORA)
    return [s for s in sources if s in CORPORA]


def catalogue() -> list[dict]:
    """Describe the corpora, for the API and the Retrieval panel."""
    return [
        {"key": c.key, "label": c.label, "description": c.description}
        for c in CORPORA.values()
    ]
