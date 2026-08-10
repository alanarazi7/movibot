"""PlotSearch tool - semantic search over plot text.

Mock: word-overlap + IDF-based scoring over embedding_text.
Real (Chunk 3+4): cosine similarity search via Pinecone embeddings.
"""

import os
import re
import json
from typing import Any
from collections import Counter
import math

import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, "data_preprocessing", "data_ready", "pinecone_candidates.csv")

_candidates_cache = None
_idf_cache = None


def _tokenize(text: str) -> set[str]:
    """Tokenize text: lowercase, alphanumeric + apostrophe, drop stopwords and short tokens."""
    stopwords = frozenset([
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "he", "her", "hers", "him", "his", "how", "i",
        "if", "in", "into", "is", "it", "its", "me", "my", "myself",
        "no", "nor", "not", "of", "on", "or", "other", "our", "ours",
        "out", "over", "own", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them",
        "then", "there", "these", "they", "this", "those", "to",
        "too", "under", "until", "up", "very", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "you", "your", "yours", "yourself"
    ])
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return set(t for t in tokens if t not in stopwords and len(t) > 2)


def _load_candidates():
    """Load candidates CSV and build IDF index once."""
    global _candidates_cache, _idf_cache
    if _candidates_cache is not None:
        return

    _candidates_cache = pd.read_csv(_CSV_PATH)

    # Parse JSON columns
    _candidates_cache["keywords_list"] = _candidates_cache["keywords"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    _candidates_cache["mpst_tags_list"] = _candidates_cache["mpst_tags"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    # Tokenize embedding_text and compute document frequency
    doc_freq = Counter()
    doc_tokens_by_id = {}

    for _, row in _candidates_cache.iterrows():
        tokens = _tokenize(row["embedding_text"])
        doc_tokens_by_id[int(row["movie_id"])] = tokens
        doc_freq.update(tokens)

    # Compute IDF: log(N / df)
    N = len(_candidates_cache)
    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log(N / (freq + 1))  # +1 to avoid division by zero

    _idf_cache = idf
    _candidates_cache["_doc_tokens"] = _candidates_cache["movie_id"].apply(
        lambda mid: doc_tokens_by_id.get(int(mid), set())
    )


def run(
    query: str,
    top_k: int = 10,
    candidate_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Score candidates using word overlap + IDF + tag boost.

    Args:
        query: search query string.
        top_k: return top K results (default 10).
        candidate_ids: if provided, restrict scoring to these movie IDs.

    Returns:
        List of {"movie_id", "title", "release_year", "score", "matched_terms"}.
    """
    _load_candidates()

    df = _candidates_cache.copy()
    if candidate_ids:
        df = df[df["movie_id"].isin(candidate_ids)]

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    results = []
    for _, row in df.iterrows():
        doc_tokens = row["_doc_tokens"]
        tag_tokens = set(row["keywords_list"]) | set(row["mpst_tags_list"])

        # Score: IDF-weighted token overlap
        overlap = doc_tokens & query_tokens
        score = sum(_idf_cache.get(token, 0) for token in overlap)

        # Bonus for tag matches
        tag_overlap = tag_tokens & query_tokens
        score += len(tag_overlap) * 0.5  # Fixed boost per tag match

        if score > 0:
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
                "score": round(score, 3),
                "matched_terms": sorted(list(overlap | tag_overlap))[:5]  # Top 5 matched terms
            })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
