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

## Chunk 2: Load structured data into Supabase — free, no LLM, not started

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

`SceneSearch`/`ExternalContext` data sourcing: live Wikipedia fetch (current default per code comments) vs. pre-indexing ~5K pages upfront. Revisit once Chunk 4 is underway.
