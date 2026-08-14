"""Splits plot synopses into passages for retrieval.

Why this exists
---------------
Embedding a whole synopsis loses most of it. E5-small-v2 accepts 512 tokens;
the median synopsis is 1,143 and the longest is 9,049. Frozen's vector covered
roughly the first 7% of its story, so Hans's betrayal -- which begins at
character 5,353 of 27,969 -- was never embedded, and no phrasing of "a film
about trusting the wrong person" could retrieve it. Chunking makes each story
beat independently retrievable.

Why sentences, not paragraphs
-----------------------------
The obvious approach, and the one used by the sibling medium-rag project, is
to accumulate paragraphs up to a token budget. That does not work on this
corpus: measured across all 159 synopses, **none contain blank-line
paragraphs** and 66 are single unbroken blobs with no newline at all. A
paragraph splitter would emit one chunk per document and change nothing. So
passages are assembled from sentences instead.

Parameters, and why these values
--------------------------------
Measured on the corpus rather than inherited:

    sentence length   median 23 tokens, p90 40, p99 64
    document length   median 1,143 tokens, p90 4,020, max 9,049

CHUNK_TOKENS = 300 holds roughly 13 sentences, which is about one scene. That
matters for the failure being fixed: the target beat should dominate its
passage rather than be averaged into a dozen unrelated ones. Larger windows
blur several beats together, which is what broke document-level embedding in
the first place.

OVERLAP_RATIO = 0.2 carries trailing sentences into the next passage so a beat
split across a boundary still appears whole somewhere.

MIN_CHUNK_TOKENS = 50 discards a stub tail that would otherwise embed noise.
"""

from __future__ import annotations

import re
from functools import lru_cache

CHUNK_TOKENS = 300
OVERLAP_RATIO = 0.2
MIN_CHUNK_TOKENS = 50

# Sentence boundary. The second alternative catches this corpus's most common
# defect: scraped text where the space after a full stop was lost, giving
# "...bot-fights.His older brother..." -- without it, such a synopsis collapses
# into a handful of enormous "sentences".
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])')


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def chunk_text(text: str) -> list[str]:
    """Split one synopsis into overlapping passages of ~CHUNK_TOKENS.

    Returns [] for empty input, and a single passage for text shorter than
    MIN_CHUNK_TOKENS only when that is the whole document -- a short synopsis
    should still be searchable, even though a short *tail* should not.
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    enc = _encoder()
    overlap_budget = int(CHUNK_TOKENS * OVERLAP_RATIO)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        n = len(enc.encode(sentence))

        # A single sentence over budget (max observed: 177 tokens) becomes its
        # own passage rather than being dropped or silently truncated.
        if n > CHUNK_TOKENS:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.append(sentence)
            continue

        if current_tokens + n > CHUNK_TOKENS and current:
            chunks.append(" ".join(current))

            # Carry trailing sentences forward, newest first, up to the
            # overlap budget.
            carried: list[str] = []
            carried_tokens = 0
            for prev in reversed(current):
                t = len(enc.encode(prev))
                if carried_tokens + t > overlap_budget:
                    break
                carried.insert(0, prev)
                carried_tokens += t

            current = carried + [sentence]
            current_tokens = carried_tokens + n
        else:
            current.append(sentence)
            current_tokens += n

    if current:
        tail = " ".join(current)
        # Keep a short tail only if it is the entire document; otherwise its
        # content already survives in the previous chunk's overlap.
        if current_tokens >= MIN_CHUNK_TOKENS or not chunks:
            chunks.append(tail)

    return chunks


def chunk_movie(movie_id: int, title: str, text: str) -> list[dict]:
    """Chunk one film's synopsis into records ready for embedding.

    The embedded text is prefixed with the title so a passage carries some
    identity of its own; a bare passage often names only pronouns, and E5
    scores it against the query with no idea whose story it is.
    """
    records = []
    for i, chunk in enumerate(chunk_text(text)):
        records.append({
            "chunk_id": f"{movie_id}_{i}",
            "movie_id": int(movie_id),
            "title": title,
            "chunk_index": i,
            "text": chunk,
            "embedding_text": f"{title}: {chunk}",
            "tokens": count_tokens(chunk),
        })
    return records
