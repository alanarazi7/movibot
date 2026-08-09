# MoviBot Data Design and Cleaning Rationale

> **Update — demo scope added.** Everything below describes the original
> full-catalog design (~43K movies, ~11K Pinecone candidates, popularity-ranked
> and cut at `priority_rank <= 3000`). By default the script now narrows
> further, right after cleaning, to `DEMO_STUDIOS` (Disney + Pixar) — 303
> movies, 170 with an MPST synopsis — which is small enough to embed in full,
> so `priority_rank`/cutoff no longer applies. The full-catalog behavior
> described here is still reachable via `--all-studios`; see
> `prepare_movibot_data usage.md` for current numbers of both modes.

## 1. Goal

MoviBot needs two different kinds of data:

1. **Structured movie facts** for deterministic filtering.
2. **Rich story text** for semantic candidate retrieval.

Trying to force both jobs into the same database is unnecessary.

The final design therefore separates the full movie catalog from the expensive semantic index.

---

## 2. Final architecture

```text
                         USER QUERY
                              |
                              v
                         Agent / Reasoner
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
             Supabase                    Pinecone
          full clean catalog          semantic subset
            ~43K movies               ranked candidates
                 |                          |
        title / year / runtime       one vector / movie
        genres / companies           MPST-rich semantic text
        popularity / overview                |
        keywords                             |
                 +------------+-------------+
                              |
                              v
                        candidate list
                              |
                              v
                    live Wikipedia checks
                 only when deeper evidence
                        is required
```

Wikipedia is intentionally **not cached as project data**. It is a live verification tool used after candidate generation.

---

## 3. Raw inputs

Only three CSVs are required.

### Kaggle: `movies_metadata.csv`

Used for:

- movie ID
- IMDb ID
- title
- release year
- runtime
- genres
- production companies
- popularity
- short overview

### Kaggle: `keywords.csv`

Used for compact story/topic keywords.

### MPST: `mpst_full_data.csv`

Used for:

- exact IMDb matching
- long plot synopsis
- story tags
- synopsis provenance

The large MPST `review` column is intentionally never loaded.

Ratings, credits, links, and transcripts are not part of this package.

---

## 4. Kaggle movie cleaning rules

The pipeline removes a movie only when it fails an objective usability requirement.

A movie is removed when it has:

- invalid/non-numeric movie ID
- missing/blank title
- invalid/missing release date
- invalid/missing runtime
- runtime <= 0
- missing/blank overview

The pipeline does **not** remove movies based on:

- popularity threshold
- year range
- language
- genre
- production company
- subjective quality

This keeps the Supabase catalog broad.

### Nested fields

TMDB-style nested values are simplified.

Example:

```text
[{"id": 16, "name": "Animation"}, {"id": 10751, "name": "Family"}]
```

becomes:

```json
["Animation","Family"]
```

The same rule is used for production-company names.

### Duplicate movie IDs

Duplicate Kaggle movie IDs are resolved deterministically.

Preference order:

1. richer useful metadata
2. higher popularity
3. longer overview

Only one row remains per Kaggle movie ID.

---

## 5. Keyword cleaning rules

Keyword IDs are converted to numeric movie IDs.

Invalid IDs are removed.

When multiple keyword rows exist for the same movie:

- keyword names are merged
- original order is preserved where possible
- duplicate keyword names are removed

Keywords for movies removed by movie cleaning are discarded.

Every final Supabase movie receives a keyword array.

If no keywords remain:

```json
[]
```

is stored.

---

## 6. MPST cleaning rules

Only these MPST fields are loaded:

```text
imdb_id
title
plot_synopsis
tags
synopsis_source
```

The large `review` field is intentionally ignored.

An MPST row is usable only when it has:

- valid IMDb ID of the form `tt...`
- non-empty title
- non-empty plot synopsis

If duplicate IMDb IDs ever occur, the row with the longest synopsis is retained.

MPST tags are converted to a compact JSON-array representation.

---

## 7. Matching rule

Kaggle and MPST are matched **only by exact normalized IMDb ID**.

Example:

```text
Kaggle imdb_id = tt2293640
MPST   imdb_id = tt2293640
```

is a match.

There is:

- no fuzzy title matching
- no title/year heuristic
- no manual correction

This makes the provenance of every semantic match easy to defend.

A title disagreement does not override an exact IMDb-ID match.

---

## 8. Why Supabase keeps the full catalog

The full cleaned Kaggle catalog is inexpensive compared with vectorized long text.

Supabase stores:

```text
id
imdb_id
title
release_year
runtime_minutes
genres
production_companies
popularity
overview
keywords
has_mpst_synopsis
```

This supports exact structured constraints such as:

- year
- runtime
- genre
- studio/company

The short overview and keywords are also retained as lightweight fallback evidence for movies that do not have MPST coverage.

`has_mpst_synopsis` tells the system whether the movie has a rich semantic source available for later Pinecone indexing.

With the dataset snapshot used during exploration, the full Supabase CSV proxy was about 21 MiB including overview and keywords.

---

## 9. Why Pinecone does not contain every movie

MPST has much richer story text than the Kaggle overview.

In the exploratory matched set:

- median Kaggle overview: about 48 words
- median MPST synopsis: about 693 words

The MPST synopsis therefore provides much more useful evidence for concepts such as:

- death
- murder
- violence
- animals
- children
- parents
- frightening content
- friendship

However, MPST does not cover the whole Kaggle catalog.

Therefore:

- **Supabase defines the movie universe**
- **MPST/Pinecone provides richer semantic retrieval for a subset**

A movie is not removed from MoviBot just because MPST does not contain it.

---

## 10. Pinecone candidate policy

This package deliberately does **not** delete Pinecone candidates.

Every exact Kaggle↔MPST match is saved.

Candidates are ranked by a deliberately simple demo policy:

1. higher Kaggle popularity first
2. newer release year as a tie-break
3. movie ID as a final deterministic tie-break

This produces:

```text
priority_rank = 1 ... N
```

For the current course demo, the intended first experiment is to embed:

```text
priority_rank <= 3000
```

Pure popularity was chosen because:

- the project is a course demonstration
- it is simple to explain
- it prioritizes recognizable movies
- MPST coverage is much stronger among popular movies than across the long tail
- it avoids inventing a complicated hand-tuned sampling policy before the agent itself is working

The full ranked file is preserved so the cutoff can later change without repeating cleaning or matching.

---

## 11. What gets embedded

For each MPST-backed movie, the local embedding input is:

```text
Title: <movie title>
Plot synopsis: <MPST synopsis>
Story tags: <MPST tags>
Keywords: <Kaggle keywords>
```

This creates one semantic document per movie.

The current planned embedding model is:

```text
text-embedding-3-small
```

with a 1536-dimensional vector.

One movie therefore produces one vector.

Full transcripts are intentionally avoided because they would require many chunks/vectors per movie and make the demo substantially more expensive and complex.

---

## 12. What is actually stored in Pinecone

The long `embedding_text` is an **ingestion artifact**, not proposed persistent Pinecone metadata.

Recommended persistent Pinecone record:

```text
vector:
    1536-D embedding

metadata:
    movie_id
    title
    release_year
```

The agent can use `movie_id` to recover the complete structured record from Supabase.

This keeps Pinecone compact and avoids duplicating the MPST synopsis online.

---

## 13. Why keep MPST tags and Kaggle keywords

They are very small relative to the synopsis and may add useful semantic labels.

During exploration, tags/keywords provided extra signals for concepts such as murder, violence, and horror even when the free text did not use the exact same wording.

Their storage cost in the local ingestion data is small, so there is little reason to remove them before embedding.

---

## 14. Deep verification

Pinecone is a **candidate-retrieval mechanism**, not final proof.

A query such as:

```text
"find a not-too-old animated movie with animals and no death"
```

contains both:

- structured constraints
- semantic/story constraints

The intended flow is approximately:

```text
1. use Supabase for structured filtering
2. use Pinecone for semantic candidate retrieval where rich vectors exist
3. use short Supabase overview/keywords as fallback context where useful
4. use live Wikipedia for deeper verification of ambiguous claims
5. synthesize the final answer
```

Wikipedia content is not downloaded into the persistent project dataset.

---

## 15. Storage interpretation

Earlier dry-run estimates suggested that:

- all ~43K Supabase rows with overview + keywords fit comfortably as a normal CSV-sized catalog
- a ~3K Pinecone subset leaves materially more headroom than a ~4K–5K subset
- the MPST synopsis itself dominates local text size
- the synopsis does not need to be stored in Pinecone after embedding

These are planning estimates only.

Actual Supabase and Pinecone storage includes database/index overhead, and the exact interpretation of any course-level 50 MB constraint should be verified separately.

---

## 16. Reproducibility principle

`data_full/` is treated as immutable raw input.

The complete final data package can be regenerated from:

```text
data_full/movies_metadata.csv
data_full/keywords.csv
data_full/mpst_full_data.csv
```

by running:

```bash
python prepare_movibot_data.py
```

No previous EDA outputs or intermediate cleaned files are required.
