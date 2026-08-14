# MoviBot Data Pipeline — Review & Exec Summary

Reviewed 2026-08-14. Every number here was recomputed from the code and the
files in `data_full/` / `data_ready/`, not copied from any earlier document.

This is now the single rationale doc for the pipeline. `DATA_SOURCES.md` and
`data cleaning rules.md` both described a superseded design and were deleted on
2026-08-14; their accurate content is here, and the how-to-run steps are in
`prepare_movibot_data usage.md`.

---

## 1. At a glance

Three raw sources go in: a movie catalog, a set of long plot synopses, and
Wikipedia. One script does the tabular work, one caches Wikipedia, and the agent
reads the results directly off disk — no database is required to run the system
locally.

The catalog is deliberately narrowed to **Disney and Pixar**, which keeps it in
family and kids' territory and makes the demo coherent. This is a demo
constraint rather than a product decision: the assignment caps stored data at
**50 MB** and the full multi-studio catalog does not fit. `--all-studios`
produces all 43,270 films from the same pipeline.

### The funnel

| Stage | Rows | Where |
|---|---:|---|
| Raw `movies_metadata.csv` | 45,466 | Kaggle |
| → after the `DEMO_STUDIOS` filter (runs **first**, on raw JSON) | 304 | `prepare_movibot_data.py:287` |
| → minus 1 non-positive runtime, minus 65 shorts under 45 min | 238 | `prepare_movibot_data.py:310` |
| **`supabase_movies.csv`** — the movie universe | **238** × 26 cols | `data_ready/` |
| Raw `keywords.csv` → merged by id → aligned to the catalog | 46,419 → 45,432 → 238 | `clean_keywords()` |
| Raw MPST `mpst_full_data.csv` | 14,828 | Kaggle |
| → exact IMDb-ID matches inside scope | 159 (66.8%) | `prepare_movibot_data.py:638` |
| **`pinecone_candidates.csv`** — the semantic subset | **159** × 30 cols | `data_ready/` |
| **`plot_chunks.parquet` + `chunk_embeddings.npy`** | **1,254** passages, 384-dim | `build_chunk_index.py` |
| **`wikipedia_cache.csv`** | 238 rows (237 articles, 233 with Plot) | `scrape_wikipedia.py` |

79 of the 238 catalog films have no MPST synopsis. They stay in the catalog and
remain answerable through structured filters and their short overview; they are
just not semantically searchable.

### Who consumes what

| Artifact | Consumer |
|---|---|
| `supabase_movies.csv` | `agent/catalog.py:29` → `filter_catalog` tool |
| `pinecone_candidates.csv` | `agent/catalog.py:30` → `read_synopses`; source for the chunk index |
| `wikipedia_cache.csv` | `agent/catalog.py:31` → verification context |
| `plot_chunks.parquet`, `chunk_embeddings.npy` | `agent/embeddings.py:35` → `search_plots` |

---

## 2. Sources

### Kaggle — *The Movies Dataset* (`rounakbanik/the-movies-dataset`) · **adopted**

`movies_metadata.csv` and `keywords.csv`. CC0.

**What it brings:** the structural spine — TMDB id, IMDb id, title, release
date, runtime, genres, production companies/countries, spoken languages,
collection, popularity, vote average/count, budget, revenue, overview, tagline.
This is the only source that supports deterministic filtering ("Pixar", "after
1990", "under 100 minutes"), so it defines the movie universe.

`keywords.csv` adds compact topic labels (`"betrayal"`, `"snowman"`,
`"reindeer"`) which are cheap to carry and add retrieval signal where the free
text does not use the same wording.

**What it does not bring:** rich story text. The median overview across the
catalog is **53 words** — enough to describe a film, not enough to answer
"does anyone die in it".

### Kaggle — *MPST: Movie Plot Synopses with Tags* (`cryptexcode/mpst-…`) · **adopted**

`mpst_full_data.csv`, 14,828 rows. CC0.

**What it brings:** long-form plot synopses — median **892 words** on the
matched Disney/Pixar set, against 53 for the Kaggle overview, a ~17× increase in
retrievable story detail. Plus story tags and synopsis provenance
(`synopsis_source`).

**Deliberately not loaded:** the `review` column, which dominates the file's
size and is irrelevant here. `clean_mpst()` passes an explicit `usecols`
(`prepare_movibot_data.py:500`) so it is never read into memory at all.

**Coverage:** 159 of 238 films, 66.8%.

### Wikipedia REST API · **adopted, pre-cached**

**What it brings:** independent verification text — Plot sections for
scene-level constraint checking, and Reception/Themes sections for tone
questions the plot itself cannot settle.

Scraped once by `scrape_wikipedia.py` into `wikipedia_cache.csv` so the agent
never makes a live call at query time. **237 of 238** films resolve to their
article; **233** carry a usable Plot section and 237 carry non-Plot text.

Resolution is the hard part, and getting it wrong is silent. A bare film title
is often a Wikipedia *disambiguation* page, and a redirect can land on an index
article or on the wrong film in a series. `wikipedia_client.py` therefore tries
the year-qualified title first, rejects disambiguation pages via the API's own
`pageprops.disambiguation` flag, rejects `List of …` pages, and requires the
article's lead sentence to name the right release year before accepting it.
See the [resolved findings](#resolved) for what this replaced.

### Not investigated

TMDB-5000 (adds cast/crew — no cast/crew queries at this scope), the large
Kaggle IMDB dumps (heavy, likely redundant with the source already in use), and
Wikidata (Wikipedia free text already covers the verification need). Full
transcripts were investigated and dropped: the HuggingFace corpus matched only 9
of 238 films, and `find_transcripts.py` has since been deleted.

---

## 3. Filters and thresholds

The pipeline's guiding rule is that a movie is removed only when it fails an
**objective usability** test, never a quality judgement. There are exactly two
deliberate exceptions, both documented below.

### Scope: `DEMO_STUDIOS` — 45,466 → 304

`prepare_movibot_data.py:50`. Keeps only films whose raw `production_companies`
field names Walt Disney Pictures, Walt Disney Animation Studios, or Pixar
Animation Studios.

Two properties worth noting:

- It runs **first**, straight off the raw nested-JSON field, before any parsing,
  cleaning or dedup (`prepare_movibot_data.py:287`). Everything downstream then
  operates on 304 rows instead of 45,466, which is why the pipeline can afford
  to keep *every* column rather than a curated subset.
- It is a **studio-membership** filter, not a content-rating one. It will admit
  a PG-13 title a Disney label distributes. Any kid-safety guarantee has to come
  from the agent's tools, not from this filter.

`--all-studios` bypasses it and reproduces the original 43,270-film catalog.

### Usability: seven predicates — 304 → 238

`prepare_movibot_data.py:302-318`. A row must have a numeric id, a non-blank
title, a parseable release date, a numeric runtime, runtime > 0, runtime ≥ 45,
and a non-blank overview.

At demo scope six of these seven fire on **nothing** — the Disney/Pixar set is
too well-curated to contain bad rows. Only two actually remove anything:

| Predicate | Removed |
|---|---:|
| runtime ≤ 0 | 1 |
| runtime < 45 min (shorts) | 65 |
| all five others combined | 0 |

They are still run, since nothing guarantees a future re-download is as clean.

### Exception 1 — `MIN_RUNTIME_MINUTES = 45`

This is a **format** filter, not a quality one: it removes a *kind* of title.

The Disney/Pixar scope drags in a long tail of animated shorts (*Lou* 6m,
*Presto* 5m, *Piper* 6m, *Paperman* 7m, the *Prep & Landing* specials). They
break ranking rather than merely padding it: shorts carry very high
`vote_average` on almost no votes — *Lou* scores 8.5 on **17 votes** — so eight
of the top ten rows by rating were shorts. A user who asks for "a movie" is
never served by a six-minute short, so every such answer is wrong regardless of
how well retrieval performed.

The cutoff sits in a clean gap in the runtime histogram: longest short 40m
(*Roving Mars*), shortest feature 47m (*Aliens of the Deep*). Nothing real sits
near the boundary. Dropping them also lifted MPST coverage from 56% to 66.8%,
since shorts rarely have a synopsis.

Enforced once, at preparation time, so every downstream artifact inherits it and
no agent tool has to remember a runtime floor.

### Exception 2 — deterministic dedup

`prepare_movibot_data.py:360-378`. When two rows share a TMDB id, keep the one
with richer metadata, then higher popularity, then the longer overview. Zero
duplicates occur at demo scope; the rule exists for the `--all-studios` path.

### Keywords: alignment, not filtering

`clean_keywords()` merges multiple keyword rows per film with a stable,
order-preserving union, then discards keywords for films the movie cleaning
removed (45,432 → 238). Every surviving film gets an array, `[]` if empty, so
downstream code never has to handle a missing key.

### MPST usability

A row is kept only with a valid `tt…` IMDb id, non-empty title, and non-empty
synopsis. At current scope **0** rows fail. Duplicate IMDb ids would be resolved
by longest synopsis; none occur.

### Matching: exact normalized IMDb ID only

`prepare_movibot_data.py:638`, an inner join with `validate="one_to_one"`.

No fuzzy title matching, no title/year heuristic, no manual correction. A title
disagreement does not override an ID match. This costs some coverage and buys
something more valuable: the provenance of every single semantic match is
trivially defensible, and a wrong synopsis can never be silently attached to the
wrong film.

---

## 4. Ranking instead of filtering: `weighted_rating`

Removing shorts fixed the *format* problem but not the *thin-votes* problem.
*Dangal* scores 8.0 on **140 votes**; *The Lion King* scores 8.0 on **5,520**.
Under a raw `vote_average` sort they tie, and *Dangal* can win.

The obvious fix — a hard `vote_count >= 500` floor — is far too destructive:

| Rule | Top of list | Films lost |
|---|---|---:|
| raw `vote_average` | *Dangal* (140 votes) | 0 |
| `vote_count >= 500` | *The Lion King* | **141 of 238 (59%)** |
| `weighted_rating` | *The Lion King* | **0** |

A floor also makes narrow queries unanswerable: asked for "a Disney movie in
Hindi", the only correct answer in scope *is* *Dangal*, and a 500-vote floor
deletes it.

So the catalog stores a precomputed Bayesian column instead — the IMDb Top-250
formula (`prepare_movibot_data.py:208`):

```
WR = (v / (v + m)) · R  +  (m / (v + m)) · C
```

with `C = 6.199` (the catalog mean, recomputed on the final deduped set every
run) and `m = 300` (the catalog's median vote count, 298.5, rounded). Every film
is scored as if it started with 300 votes at the catalog average.

The result is a **ranking** function, not a filter, which is the property that
matters:

- broad query ("best movie") → *Dangal* falls to **#40 of 238**, *The Lion King*
  takes #1
- narrow query ("in Hindi") → the structured filter leaves 3 films and *Dangal*
  is **#1**, still reachable

Nothing is deleted, so no answer becomes unreachable. `vote_average` and
`vote_count` are both retained for queries that genuinely want the raw figures.

---

## 5. Chunking: why passages, not documents

`agent/chunking.py`, built by `scripts/build_chunk_index.py`.

**The failure it fixes.** E5-small-v2 accepts 512 tokens. The median synopsis is
1,143 tokens and the longest is 9,049. Embedding whole documents meant Frozen's
vector covered roughly the **first 7%** of its story — so Hans, who first
appears at character 5,353 of 27,969 and betrays Anna at character 22,746, was
never embedded at all, and no phrasing of "a film about trusting the wrong
person" could retrieve it.

**Why sentences, not paragraphs.** The obvious approach — accumulate paragraphs
to a token budget, as the sibling `medium-rag-hw` project does — was measured
against this corpus and rejected: of the 159 synopses, **zero** contain
blank-line paragraphs and **66 have no newline at all**. A paragraph splitter
would emit exactly one chunk per document and change nothing.

**Parameters, measured rather than inherited.** Sentences are median 23 tokens
(p90 40, p99 64), so:

| Parameter | Value | Why |
|---|---:|---|
| `CHUNK_TOKENS` | 300 | ~13 sentences ≈ one scene, tight enough that a story beat dominates its passage rather than being averaged into a dozen others |
| `OVERLAP_RATIO` | 0.2 | a beat split across a boundary still appears whole somewhere |
| `MIN_CHUNK_TOKENS` | 50 | discards a stub tail that would otherwise embed noise — unless it is the whole document, in which case a short synopsis stays searchable |

The sentence regex also handles this corpus's characteristic scrape defect,
where the space after a full stop was lost (`"...bot-fights.His older
brother..."`); without the second alternative in `_SENTENCE_SPLIT`, such a
synopsis collapses into a handful of enormous "sentences".

**Result:** 1,254 passages from 159 films (7.9 per film, median 285 tokens),
each prefixed with its film's title so a passage carries some identity — a bare
passage often names only pronouns.

---

## 6. Worked example: *Frozen* (2013), end to end

### Stage 0 — raw

`movies_metadata.csv` contains **three** films titled "Frozen":

| id | imdb_id | vote_avg | votes | what it is |
|---:|---|---:|---:|---|
| 44363 | tt1323045 | 5.9 | 586 | 2010 ski-lift thriller (Adam Green) |
| 170986 | tt1071798 | 8.5 | 2 | unrelated |
| **109445** | **tt2294629** | **7.3** | **5,440** | the Disney film |

### Stage 1 — studio filter

`production_companies` reads
`[{'name': 'Walt Disney Pictures', ...}, {'name': 'Walt Disney Animation Studios', ...}]`,
so only `id=109445` survives. Note the side benefit: **scope disambiguation is
free**. The two impostors are gone before any title matching happens, so nothing
downstream can confuse them.

### Stage 2 — usability

Runtime 102 min (≥ 45 ✓), release date parses, title/overview non-blank, id
numeric. Kept. No duplicate id, so dedup is a no-op.

### Stage 3 — normalization

| Field | Raw | Cleaned |
|---|---|---|
| `genres` | `[{'id': 16, 'name': 'Animation'}, …]` | `["Animation","Adventure","Family"]` |
| `production_companies` | nested dicts | `["Walt Disney Pictures","Walt Disney Animation Studios"]` |
| `belongs_to_collection` | `{'id': 386382, 'name': 'Frozen Collection', 'poster_path': …}` | `Frozen Collection` |
| `spoken_languages` | nested dicts | `["English"]` |
| `release_date` | `2013-11-27` | `2013-11-27` + `release_year = 2013` |
| `adult` / `video` | `'False'` (string) | `False` (bool) |

### Stage 4 — `weighted_rating`

```
R = 7.3   v = 5,440   C = 6.199   m = 300
WR = (5440/5740)·7.3 + (300/5740)·6.199 = 7.2425
```

5,440 votes pull the film **95%** of the way to its own score, so smoothing
barely touches it. Rank moves from **#28** by raw `vote_average` to **#20** by
`weighted_rating` — Frozen gains ground precisely because the thinly-voted
titles above it lose it.

### Stage 5 — keyword merge

```json
["queen","musical","princess","betrayal","snowman","animation","reindeer",
 "curse","snow","troll","mountain climber","aftercreditsstinger",
 "woman director","3d"]
```

### Stage 6 — the row that ships to `supabase_movies.csv`

```
id                    109445
imdb_id               tt2294629
title                 Frozen
release_year          2013
runtime_minutes       102.0
genres                ["Animation","Adventure","Family"]
production_companies  ["Walt Disney Pictures","Walt Disney Animation Studios"]
belongs_to_collection Frozen Collection
popularity            24.248243
vote_average          7.3
vote_count            5440
weighted_rating       7.2425
budget                150000000
revenue               1274219009
tagline               Only the act of true love will thaw a frozen heart.
has_mpst_synopsis     True
```

### Stage 7 — MPST match

`tt2294629` matches exactly. The **66-word** overview is joined by a
**4,933-word / 27,969-character** synopsis, plus `mpst_tags = ["cute","fantasy"]`
and `synopsis_source = imdb`. Frozen lands at row **11 of 159** in
`pinecone_candidates.csv` (sorted by descending popularity).

Note what the tags illustrate: MPST's own labels for this film are `cute` and
`fantasy`. Nothing in them hints at the betrayal at the heart of the story. The
synopsis text is doing all the work here, which is exactly why the pipeline pays
for it.

### Stage 8 — `embedding_text` (28,178 chars)

```
Title: Frozen
Plot synopsis: The Walt Disney Pictures logo and the movie title appear to the
Norwegian song "Vuelie".In a winter landscape, ice harvesters use saws and hooks…
Story tags: cute, fantasy
Keywords: queen, musical, princess, betrayal, snowman, …
```

Note `"Vuelie".In a winter` — the missing-space defect, in the very first
sentence, on the flagship example.

### Stage 9 — chunking

**26 passages** (median 288 tokens, min 217, max 298), ids `109445_0` …
`109445_25`. This is the payoff. **Chunk 21**:

> …and says "Oh, Anna, if only there was someone here who loved you!" As Anna
> looks at him in shock, Hans explains that as the youngest of 13 brothers, he
> had no chance at claiming his family's throne, so he went looking for a royal
> family he could marry into. Unable to get to Elsa, he made Anna's
> acquaintance and played on her naivete…

This passage begins at character **22,726 of 27,969 — 81% of the way through**.
Under the old document-level index, which reached only the first ~7%, it was
invisible to both search *and* reading. It is now independently retrievable, and
`read_synopses` can quote it as evidence.

### Stage 10 — Wikipedia cache

`wiki_title = Frozen (2013 film)`, with a 2,982-character Plot section and
4,000 characters of cast/production/reception text.

This stage used to be where the example broke. `plot_text` was empty and
`non_plot_text` held Wikipedia's **disambiguation page** — a list of links
running from *Frozen (1997 film)* to *Frozen (album), by Sentenced*. The studio
filter disambiguated the Kaggle data, but nothing disambiguated the Wikipedia
lookup. Fixed; see [resolved findings](#resolved).

### Summary of the trace

| Stage | Frozen |
|---|---|
| Raw candidates named "Frozen" | 3 |
| Survives studio filter | 1 (`id=109445`) |
| In `supabase_movies.csv` | ✅ |
| In `pinecone_candidates.csv` | ✅ (row 11 of 159) |
| Passages in the chunk index | 26 |
| Wikipedia plot text | ✅ 2,982 chars from `Frozen (2013 film)` |

---

## Findings and what was done about them

<a id="resolved"></a>

Everything this review opened has now been closed — two by fixing code, the
rest by deleting or correcting the documents that had drifted. What follows
records what was wrong and what was done, so the decisions stay auditable.

### Fixed in code — the two that changed agent behaviour

**R1 — Wikipedia lookup resolved to disambiguation pages.**
`fetch_page_extract()` built its candidates as `[title, "{title} ({year} film)",
…]` and returned the first extract over 200 characters. A disambiguation page
always clears that bar, so for any film whose bare title is also a
disambiguation page the real article was never requested. Confirmed on Frozen,
whose cache held a list of links from *Frozen (1997 film)* to *Frozen (album)*.

The rewrite tries the year-qualified title first and rejects bad landings
explicitly: disambiguation pages via `pageprops.disambiguation`, `List of …`
index pages by title, and any article whose lead sentence does not name the
right release year. Two real redirect traps this caught: *The Prince and the
Pauper* redirected to *List of adaptations of The Prince and the Pauper*, and
*Beverly Hills Chihuahua 3* redirected to the 2008 first film, which would have
cached the wrong plot entirely — a wrong plot being considerably worse than no
plot. A Wikipedia search fallback then recovers films the title candidates miss.

**R2 — each page was fetched twice, with different disambiguation.**
`scrape_wikipedia.py` fetched once with the release year for the plot, then
again *without* it for the non-plot text, so the two halves of one cache row
could come from two different articles. It now fetches once and derives both
halves from that single extract. Section splitting also now matches Wikipedia's
real `== Heading ==` markers rather than guessing from line length.

Result across the catalog:

| | Before | After |
|---|---:|---:|
| Articles resolved | 228 | **237** |
| With a Plot section | 167 | **233** |
| With non-plot text | 208 | **237** |

The one unresolved film is *The Prince and the Pauper* (1962), where the guards
correctly reject an index page rather than cache the wrong thing. The four
articles with no Plot section are genuinely plot-less: *Frank and Ollie* (a
documentary), *Fantasia 2000*, *Return to Snowy River*, and a Hannah Montana
concert film.

---

### Also fixed in code — three that did not

**F1 — `schema.sql` declared two columns the CSV never produces.**
It declared `poster_path` and `homepage`, which `prepare_movibot_data.py`
deliberately drops since no tool uses them, while claiming "Columns match …
exactly". Both removed, and its header comment now names `filter_catalog`
rather than the long-deleted `CatalogFilter` module.

**F2 — `build_chunk_index.py` misstated its own outputs.**
Its docstring said the outputs were "all gitignored". `.gitignore` only excludes
`data_preprocessing/data_full/`, so `chunk_embeddings.npy` (1.9 MB) and
`plot_chunks.parquet` (1.6 MB) are committed — which is the right call, since it
makes the repo runnable without a rebuild. The docstring now says so.

**F3 — `scrape_wikipedia.py` only ran from its own directory.**
A bare `import wikipedia_client` meant the script needed
`data_preprocessing/` on `sys.path`. It now resolves its own paths, like every
other script in the repo.

### Fixed by deletion

**F4 — `data cleaning rules.md` described a superseded design.** Its §2 diagram
said ~43K movies; §2 and §14 said Wikipedia was "intentionally *not* cached as
project data" (it is, 238 rows); §10 documented a `priority_rank <= 3000` cutoff
that no longer exists; §11–12 described one 1536-dim vector per movie against a
shipped index of 1,254 passages × 384-dim. **Deleted.**

**F5 — `DATA_SOURCES.md` carried eight wrong figures.** MPST "56,216 rows"
(14,828), Wikipedia cache "287 movies" (238), transcripts quoted as both 3.3%
and 3.8%, "9 movies" beside a list of ten titles, "93% of catalog has no
transcript" (96.2%), "56% coverage" (66.8%), `imdb_id INT UNIQUE` against real
`tt…` strings, and "median ~693 words" (892 on the matched set). It also
documented the HuggingFace transcript source at length after that source was
dropped. **Deleted**; the Kaggle download instructions it held were moved into
`prepare_movibot_data usage.md` rather than lost.

**F6 — an empty `data_ready/` directory sat at the repo root**, shadowing the
real `data_preprocessing/data_ready/`. **Deleted**, along with `NEXT_STEPS.md`
and `DATA_IMPROVEMENTS_SUMMARY.md`, both of which documented the deleted
six-module ReAct architecture and the deleted transcript discovery.

### Corrected in place

**F7 — `prepare_movibot_data usage.md` contradicted itself**, saying "56%
coverage" on line 111 and 66.8% in §6. 66.8% is correct (159/238); fixed.

**F8 — `README.md` described an architecture that no longer existed** — a
four-tool ReAct loop, six modules marked "Mock (deterministic)", a
`MockLLMClient` that had been deleted, and the claim that "Wikipedia is fetched
live per-candidate rather than pre-indexed" when it is cached for 237 films. It
mentioned none of the three tools that actually exist. Rewritten.

## Appendix — reproducing every number here

```bash
cd ~/tabstar/movibot/data_preprocessing

# funnel stats, without writing any files
python3 -c "
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('p','prepare_movibot_data.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.clean_movies(pathlib.Path('data_full/movies_metadata.csv'), m.DEMO_STUDIOS)[1])
print(m.clean_mpst(pathlib.Path('data_full/mpst_full_data.csv'))[1])
"

# artifact shapes
python3 -c "
import pandas as pd, numpy as np
d = 'data_ready/'
print(pd.read_csv(d+'supabase_movies.csv').shape,
      pd.read_csv(d+'pinecone_candidates.csv').shape,
      np.load(d+'chunk_embeddings.npy').shape)
"
```

Full regeneration (writes to `data_ready/`):

```bash
python prepare_movibot_data.py            # Disney + Pixar (default)
python prepare_movibot_data.py --all-studios
python scrape_wikipedia.py                # from inside data_preprocessing/
python ../scripts/build_chunk_index.py
```
