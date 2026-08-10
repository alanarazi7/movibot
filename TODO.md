# Next Steps

Technical checklist, in build order. Due date: **2026-08-23**.

## Strategy: chunked, cost-gated

We build in small, reviewable chunks, and split them into a **free track** (data wrangling, database writes — no external model calls) and a **paid track** (anything that calls LLMod.ai or Pinecone). We finish the entire free track and review it before spending any budget on the paid track.

```
Chunk 1: Fetch & filter the dataset        [free, offline]
Chunk 2: Load structured data → Supabase   [free, no LLM]
          ── review checkpoint, explicit go-ahead needed ──
Chunk 3: Embed + index → Pinecone          [$ - first LLM spend]
Chunk 4: Build & test the agent            [$ - LLM calls]
Chunk 5: Deploy & polish                   [free/cheap]
```

Each chunk below is meant to be picked up cold in a fresh session — it says what's done, what's next, and what it's blocked on.

## Chunk 0: Blocked on team input

- [ ] Yair's email → `team_info.json`
- [ ] Andrei's email → `team_info.json`
- [x] Supabase project created (URL + secret key already in local `.env`, gitignored, never committed)
- [ ] LLMod.ai key + Pinecone key still needed in `.env` (only needed once we reach Chunk 3 — not blocking Chunks 1-2)
- [ ] **TMDB key** (`TMDB_ACCESS_TOKEN` v4 bearer, or `TMDB_API_KEY` v3) — free, registered at themoviedb.org. Blocks Chunk 1b's catalog refresh. Spends no LLM budget.
- [ ] **Kaggle credentials** (`~/.kaggle/kaggle.json`) — blocks both re-running Chunk 1 and the Chunk 1b transcript probe. Not present in a fresh clone, and the raw `data_full/` CSVs are gitignored, so nothing can be regenerated without them.

## Chunk 1: Fetch & filter the dataset — free, offline, DONE

Implemented in `data_preprocessing/prepare_movibot_data.py` (design rationale in `data_preprocessing/data cleaning rules.md`, usage in `data_preprocessing/prepare_movibot_data usage.md`). Ended up broader than the original single-CSV/~5K-downsample sketch below — kept for history, see the actual design instead:

- [x] Download 2 raw Kaggle sources into `data_preprocessing/data_full/` (gitignored): *The Movies Dataset* (`rounakbanik/the-movies-dataset`, 2 tables — `movies_metadata.csv` + `keywords.csv`) and MPST (`cryptexcode/mpst-movie-plot-synopses-with-tags`, `mpst_full_data.csv` — richer plot synopses, median ~693 words vs. ~48 for the Kaggle overview; not in the original team proposal doc but adopted for the semantic-search tool)
- [x] Narrow to a demo scope FIRST, straight off the raw data: `DEMO_STUDIOS` = Disney + Pixar (`--all-studios` reproduces the original full-catalog behavior instead) — **45,466 raw → 304 raw Disney + Pixar movies**
- [x] Clean that (now small) set: drop rows with invalid id, blank title, invalid/missing release date, invalid/non-positive runtime, or blank overview; dedupe by id — **304 → 303** (at this scope cleaning is nearly a no-op: only 1 row dropped, 0 duplicates)
- [x] Clean + merge keywords; clean MPST (skipping the huge irrelevant `review` column); match Kaggle↔MPST by exact normalized IMDb ID
- [x] Keep almost every column `movies_metadata.csv` has (25 in `supabase_movies.csv` incl. `keywords`/`has_mpst_synopsis`) — column count stopped being a size concern once row count dropped this far. Dropped `poster_path`/`homepage` since neither is used by any agent tool
- [x] Write two reviewable outputs to `data_preprocessing/data_ready/` (gitignored): `supabase_movies.csv` (303 movies, 0.23 MiB), `pinecone_candidates.csv` (170 of those 303 with an exact MPST match — 56% coverage, 2.70 MiB, full movie + MPST columns + `embedding_text`). Combined ~2.9 MiB — no ranking/cutoff column needed at this size, so `priority_rank` was dropped
- [x] Review together before moving to Chunk 2 — ran locally, sample rows sanity-checked (incl. tracing one movie, *Frozen* 2013, through all files)

This is pure pandas/CSV work — no network calls beyond the one-time Kaggle download, no LLM API keys needed, nothing that costs money.

## Chunk 1b: Catalog freshness + transcripts — free/cheap, in progress

Two gaps found while reviewing Chunk 1's output.

**Gap 1 — the catalog is eight years stale.** "The Movies Dataset" is a
MovieLens dump covering releases up to **July 2017**, so the 303 movies contain
nothing newer. The project's flagship demo query asks for a *"not-too-old
Disney movie"*, which the data currently cannot answer: Encanto, Turning Red,
Luca, Soul, Onward, Raya, Strange World, Wish, Lightyear, Elemental, Inside Out
2, Moana 2 and Zootopia 2 are all missing.

Source chosen: **the TMDB API**, not a static IMDb scrape. The catalog is
already TMDB-shaped (`id` is a TMDB movie id, every list column comes from
TMDB's nested-dict fields), so TMDB keeps the same id space, column names and
`production_companies` values that `DEMO_STUDIOS` matches on — new rows append
with no reconciliation and no title-based guessing. The Kaggle
`imdb-movies-1960-2023` alternative was rejected: it ends in 2023, so it is
itself already stale, and it carries no TMDB ids.

**Gap 2 — no source can answer "no deaths" / "not scary" exhaustively.**
Transcripts are the only source where risky events are visible scene by scene.
Screenplay corpora are the wrong shape for us (`mocboch/movie_scripts` was
sampled: it is live-action classics — Grand Hotel 1932, Citizen Kane 1941 —
with near-zero animation, no IMDb id, and inverted titles like
`"Wizard of Oz, The"`). Subtitle-derived corpora such as Kaggle's
`movie-transcripts-59k` cover animation properly and are the candidate to
probe.

- [x] `data_preprocessing/title_matching.py` — remake-safe `(title, year)` join for sources with no IMDb id. Every ambiguity is reported rather than guessed, and a candidate claimed by two movies is rejected for both, reproducing the one-to-one guarantee the MPST IMDb-id join gets for free. This matters most for exactly our scope: The Lion King, The Jungle Book, Aladdin, Cinderella, Dumbo, Mulan, Beauty and the Beast, Pinocchio, Lady and the Tramp and 101 Dalmatians all have live-action remakes sharing an identical title. 100% line coverage.
- [x] `data_preprocessing/fetch_tmdb_updates.py` — discovers Disney/Pixar releases since the Kaggle cutoff and emits `data_ready/tmdb_new_movies.csv` with the same 25 columns, in the same order, as `supabase_movies.csv`. Applies the same studio-scope and usability rules as `clean_movies()`. Verifies TMDB company ids against their live names before fetching, so a reassigned id aborts instead of silently pulling a different studio's catalog. 98% line coverage.
- [x] Rewrites TMDB's current `"Pixar"` spelling to the catalog's `"Pixar Animation Studios"` — without this every new Pixar film would be dropped as out of scope and the studio column would disagree with itself
- [ ] Run the fetcher (**blocked on a free TMDB key**), review the new rows, then concatenate into `supabase_movies.csv`
- [ ] Decide whether new rows keep TMDB ids as-is — they are in the same id space, so collisions with the existing 303 should be impossible, but verify before merging
- [ ] Transcript coverage probe (**blocked on Kaggle credentials**): join `movie-transcripts-59k` titles against the catalog with `title_matching`, report matched / ambiguous / unmatched. That single number decides the design: >60% makes transcripts the basis of `SceneSearch`; 20-60% makes them a first layer with live Wikipedia as fallback; <20% means dropping the idea. Load titles only — do not read transcript bodies during the probe.
- [ ] Transcripts are copyrighted and this repo is public: raw text stays in gitignored `data_full/`, and only derived vectors go to Pinecone

Embedding cost is not the constraint here — roughly 200 films at ~20K words is
~5M tokens, about **$0.10** with `text-embedding-3-small` against the $13 cap.
Re-embedding later is therefore cheap, which is why this chunk does not have to
block Chunks 2 and 4.

## Chunk 2: Load structured data into Supabase — free, no LLM, not started

- [ ] Fix `data_preprocessing/schema.sql` first: it still declares `poster_path` and `homepage`, which the "drop unused columns" pass removed — `prepare_movibot_data.py` no longer even reads them. 27 columns in the schema vs. 25 in the CSV. Inserts would not fail (both are nullable), we would just get two permanently empty columns.
- [ ] Run `data_preprocessing/schema.sql` in the Supabase SQL editor (creates the `movies` table, now including `imdb_id`, `keywords`, `has_mpst_synopsis`) — this one step needs to happen through the Supabase web UI; API keys alone don't grant DDL access
- [ ] Implement the Supabase-write half of `scripts/ingest.py`: read `data_preprocessing/data_ready/supabase_movies.csv` (Chunk 1's output), insert rows into `movies`
- [ ] Spot-check row count (303) and a few rows in the Supabase table editor
- [ ] **Review checkpoint** — once this is done, the `CatalogFilter` tool (Supabase queries) can already be sanity-tested with plain SQL, with zero LLM spend. This is the natural pause point before touching Pinecone/LLMod.ai.

## Chunk 3: Embeddings + Pinecone — first $ spend, needs explicit go-ahead

- [ ] Create the Pinecone index (`movibot-plots`, cosine, dim=1536 for `text-embedding-3-small`)
- [ ] Implement the embedding half of `scripts/ingest.py`: read `data_preprocessing/data_ready/pinecone_candidates.csv` (Chunk 1's output, 170 rows), embed each movie's `embedding_text` via LLMod.ai, upsert to Pinecone with metadata `{movie_id, title, release_year}` only (never store `embedding_text` itself as metadata)
- [ ] Test with a small `--limit` (e.g. 20 movies) first, check the resulting vector count, **then** run the rest — at 170 rows total there's no need for a ranked cutoff, just embed the whole file
- [ ] This is the first step that spends LLMod.ai budget — do not start until explicitly told to.

## Chunk 4: Agent core — LLM calls for reasoning, needs explicit go-ahead

All currently stubs / `NotImplementedError`:

- [ ] `agent/llm_client.py` — sanity-check `get_client()` against real LLMod.ai with one cheap call
- [ ] `agent/tools/catalog_filter.py::run()` — LLM translates structured constraints → Supabase query
- [ ] `agent/tools/plot_search.py::run()` — embed query, Pinecone search, return matches
- [ ] `agent/tools/scene_search.py::run()` — live Wikipedia "Plot" section fetch + LLM check
- [ ] `agent/tools/external_context.py::run()` — live Wikipedia non-Plot section fetch + LLM check
- [ ] `agent/react_loop.py::execute()` — Reasoner loop, `steps[]` construction, iteration cap, budget/time guard (Vercel's 300s limit)
- [ ] `app.py::execute()` — swap the hardcoded stub body for `agent.react_loop.execute(prompt)`

## Chunk 5: Deploy & polish

- [ ] Optionally polish `assets/architecture.png` past the current placeholder
- [ ] `agent_info.json` — swap the `STUB` `prompt_examples` entry for a captured real run (the Disney/toddler query)
- [ ] Connect `alanarazi7/movibot` to Vercel, set env vars in the dashboard: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `SUPABASE_URL`, `SUPABASE_KEY`
- [ ] Deploy, verify all 4 endpoints in production
- [ ] Re-run the Disney/toddler demo query against the prod URL — confirm well under 300s and that the `steps` trace module names match `assets/architecture.png` exactly
- [ ] `README.md` — fill in the Vercel URL once deployed

## Budget

- [ ] Track LLMod.ai spend against the $13 cap starting from Chunk 3, the first real spend

## Open design decision (not yet made)

`SceneSearch`/`ExternalContext` data sourcing. Three options now on the table:

1. Live Wikipedia fetch per candidate (current default per code comments) — no pre-indexing, works for any candidate, but costs latency inside Vercel's 300s budget and only sees plot summaries.
2. Pre-indexed transcripts (Chunk 1b) — far richer, scene-level evidence, but partial coverage and a chunking pipeline of its own.
3. Both, layered: transcript first, Wikipedia as fallback where no transcript matched.

Option 3 is the likely answer, but the choice is gated on the Chunk 1b probe's
coverage number. Decide once that number exists, before Chunk 4 hardens.
