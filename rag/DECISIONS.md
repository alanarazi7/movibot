# RAG Design Decisions

Every retrieval choice and the evidence for it. Parameters live in
`rag/config.py`; this file says why they hold those values.

The house style here follows the sibling `medium-rag-hw` project: measure the
corpus, then deviate from the course default only where the measurement says
to, and record the deviation.

---

## 1. Corpus

**1,254 passages from 159 films.** The source is
`data_preprocessing/data_ready/pinecone_candidates.csv` — the Disney/Pixar
films that matched an MPST plot synopsis by exact IMDb id.

Measured across the 159 synopses:

| Statistic | Value |
|---|---|
| Synopsis length, median | 1,143 tokens |
| Synopsis length, p90 | 4,020 tokens |
| Synopsis length, max | 9,049 tokens |
| Sentence length, median | 23 tokens |
| Sentence length, p90 | 40 tokens |
| Sentence length, p99 | 64 tokens |
| Synopses containing a blank-line paragraph | **0 of 159** |
| Synopses containing no newline at all | **66 of 159** |

That last pair is the load-bearing measurement — see §3.

---

## 2. Embedding model

**`MB5R2CF-azure/text-embedding-3-small`, 1,536 dimensions.** Confirmed against
`GET /v1/models`, which returns exactly two ids for this tenant.

One model everywhere, for ingest and for queries. An earlier design ran
`intfloat/e5-small-v2` locally and OpenAI in production. That was worse than it
sounds: the two produce different vectors, so **local testing never exercised
the rankings production would return**, and a bug in either path was invisible
from the other. It also dragged in torch at ~518 MB, against Vercel's 250 MB
serverless limit, which is why the local path could not deploy at all.

E5 was also the weakest link in retrieval quality. Frozen's rank for the same
underlying question, by phrasing, under E5:

| Query | Rank |
|---|---|
| "a prince reveals he never loved her and leaves her to die" | #3 |
| "a man pretends to love a woman so he can seize the throne" | #4 |
| "a charming stranger wins someone's trust and then betrays them" | #34 |
| "someone you just met turns out to be the villain" | #93 |

Total score spread across all 159 films was 0.076. Signal existed but was weak
and highly phrasing-sensitive.

A side effect worth noting: chunk sizes are counted with `cl100k_base`, which
is text-embedding-3-small's own tokeniser. Under E5 they were BERT tokens
counted with an OpenAI tokeniser — close enough to work, but not the same unit.

---

## 3. Chunking

### Sentences, not paragraphs

The sibling project accumulates **paragraphs** to a token budget. That approach
produces nothing here: **zero** of the 159 synopses contain blank-line
paragraphs and 66 have no newline at all, so a paragraph splitter emits one
chunk per document. Passages are assembled from **sentences** instead.

The splitter also handles this corpus's characteristic scrape defect, where the
space after a full stop was lost — `"...bot-fights.His older brother..."`.
Without that, such a synopsis collapses into a handful of enormous "sentences".

### Why chunk at all

Not because the model cannot read a whole synopsis. text-embedding-3-small
accepts 8,191 tokens; the median synopsis is 1,143. Whole-document embedding is
now *possible*. It is still wrong here:

- **Precision.** One vector for a 28,000-character story averages thirty
  unrelated beats. Frozen's betrayal sits at 81% through the text and cannot
  outrank the film's own opening under that averaging.
- **Evidence.** `search_plots` returns the matching passage to the planner as a
  quotable line of proof, and `read_synopses` returns relevant passages in story
  order. A document-level vector has no passage to return.

This used to be *forced*: E5 accepted 512 tokens and covered roughly the first
7% of a long synopsis. That constraint is gone; the choice remains.

### Parameters

| Parameter | Value | Course guidance | Why we differ |
|---|---:|---|---|
| `CHUNK_TOKENS` | 300 | 512–1024 for long-form prose | See below |
| `OVERLAP_RATIO` | 0.20 | 5–15% for long-form | See below |
| `MIN_CHUNK_TOKENS` | 50 | — | A trailing passage shorter than this is **merged into the previous one**, never discarded. See below |

**Chunk size.** The course recommends 512–1024 tokens for general long-form
text, and plot synopses are long-form narrative prose, so that row is the one
we nominally fall under. We use 300 because the *retrieval unit here is a story
beat*, not a section: the questions this system answers are "does anyone die",
"who betrays whom", "would this frighten a child". Those are local events,
typically a few sentences. At median 23 tokens per sentence, 300 tokens holds
roughly 13 sentences — about one scene, tight enough that a beat dominates its
passage rather than being averaged into a dozen unrelated ones. A 512–1024
token window spans several scenes and dilutes exactly the signal being searched
for.

This is the same *shape* of argument the sibling project made when it deviated
down to 300 for Medium articles, on the grounds that its paragraphs were short.
The reasoning differs; the conclusion coincides.

**Honest note on provenance.** 300 / 0.20 / 50 are the same three values the
sibling project uses. They were carried over and then justified against this
corpus, not derived from it independently. The measurement that *was* done
first, and that genuinely changed the design, is the paragraph one in §1.

**Nothing is discarded.** An earlier version dropped a trailing passage under
50 tokens on the grounds that a fragment embeds poorly — true in itself, but
the wrong remedy. A dropped ending is a story beat that can never be retrieved,
and endings are where the deaths and betrayals are. The tail is now folded into
the previous passage instead. Two details make that safe: the tail begins with
sentences already carried forward as overlap, so only the genuinely new
sentences are appended, and the result is checked to contain every input
sentence exactly once.

Measured across all 159 synopses after the change: **0 sentences lost, 0
duplicated, still 1,254 passages**, and exactly one passage exceeds
`CHUNK_TOKENS` — *The Santa Clause*, by a single token. The cost of never
discarding is one token.

**Overlap.** 0.20 sits above the 5–15% guidance. A beat split across a boundary
still appears whole in one passage or the other, which matters more here than in
article retrieval because a half-quoted betrayal is not usable as evidence. The
cost is roughly 20% more vectors — trivial at this scale.

**Still open:** whether 300 is right, or whether 200 or 450 retrieves better on
this corpus. Cheap to settle empirically (~$0.007 per variant) using the four
Frozen phrasings above as the probe. Not yet done.

---

## 4. Retrieval

| Parameter | Value | Course guidance |
|---|---:|---|
| `TOP_K` | 10 films | k = 3–5 for general text |
| `FETCH_MULTIPLIER` | 5 | — |
| per-film passage cap | 1 (best) | sibling uses 5 per article |

**Best-passage-per-film.** Passages are over-fetched (`top_k × 5`), then each
film is scored by its single strongest passage and that passage is returned as
evidence. Scoring by a film's best passage rather than its average is what makes
a specific beat findable at all.

**Why `top_k` is higher than the course default.** In the sibling project a
retrieved chunk is context pasted into the prompt, so k directly drives token
cost and 3–5 is right. Here a retrieved film is one line — an id, a score, and
one passage — that the planner then filters. The expensive step is
`read_synopses`, which is separately capped at 8 films. So a larger k widens the
candidate pool without materially widening the context.

---

## 5. Storage

Two stores, selected by `MOVIBOT_VECTOR_STORE`, both embedding with the same
model so both return the same ranking:

- **`matrix`** (default) — a committed `.npy`, scored with numpy. 1,254 vectors
  score in about **0.5 ms**, faster than a network round trip. Needs no
  credentials, so retrieval works from a fresh clone.
- **`pinecone`** — the same vectors in Pinecone, with `{movie_id, title,
  chunk_index, text}` as metadata. The text is carried because search returns it
  as evidence.

At this scale a vector database is not required. It is supported for the cloud
deployment path and because the course covers it.

**Guard:** `chunk_index_meta.json` records the model that built the index, and
loading refuses if it no longer matches `EMBED_MODEL`. Vectors from a different
model still *score* — they just score meaninglessly — so the failure is silent
unless something checks. This is not hypothetical: the E5 index was live when
the model changed.

---

## 6. Cost

| Operation | Tokens | Cost |
|---|---:|---:|
| Full index build (1,254 passages) | 340,808 | **~$0.0068** |
| One query embedding | ~10 | ~$0.000002 |

`python -m rag.ingest --dry-run` reports both without sending anything.
