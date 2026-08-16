"""Splits plot synopses into passages for retrieval.

Why chunk at all
----------------
Not because the model cannot read a whole synopsis -- text-embedding-3-small
accepts 8,191 tokens, and the median synopsis is 1,143, so whole-document
embedding is now possible. It is still wrong here, for two reasons:

  * Retrieval precision. One vector for a 28,000-character story averages
    thirty unrelated beats together. Frozen's betrayal, at 81% through the
    text, cannot outrank the film's own opening under that averaging.
  * Evidence. search_plots returns the matching passage to the planner as a
    quotable line of proof, and read_synopses returns relevant passages in
    story order. A document-level vector has no passage to return.

This used to be forced: E5-small-v2 accepted 512 tokens and covered roughly
the first 7% of a long synopsis. That constraint is gone. The choice remains.

Why sentences, not paragraphs
-----------------------------
The obvious approach, and the one used by the sibling medium-rag project, is
to accumulate paragraphs up to a token budget. That does not work on this
corpus: measured across all 159 synopses, **none contain blank-line
paragraphs** and 66 are single unbroken blobs with no newline at all. A
paragraph splitter would emit one chunk per document and change nothing. So
passages are assembled from sentences instead.

Parameters live in rag/config.py; the rationale is in rag/DECISIONS.md.
"""

from __future__ import annotations

import re
from functools import lru_cache

from rag.config import (  # noqa: E402
    CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    OVERLAP_RATIO,
    TOKENIZER,
)

# Sentence boundary. The second alternative catches this corpus's most common
# defect: scraped text where the space after a full stop was lost, giving
# "...bot-fights.His older brother..." -- without it, such a synopsis collapses
# into a handful of enormous "sentences".
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])')


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding(TOKENIZER)


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
        # A sub-threshold tail is dropped: too little content to embed well,
        # and short fragments score spuriously high on short queries. Some of
        # it is already duplicated by the previous chunk's overlap, but not
        # all -- the tail is carried overlap plus new sentences, so this can
        # lose text. Measured on this corpus it costs 1 sentence across all
        # 159 synopses, and that sentence is the scrape artifact
        # "[D-Man2010]". Kept when it is the whole document, so that a very
        # short synopsis still yields one searchable passage.
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
