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

## Chunk 1: Fetch & filter the dataset — free, offline, not started

- [ ] Download Kaggle "The Movies Dataset" (`movies_metadata.csv`)
- [ ] Clean it: drop rows with broken/missing year, runtime, or overview
- [ ] Downsample to ~5K movies per `docs/team-idea-proposal-and-data-sources.pdf`'s scope
- [ ] Select just the columns we need: `title, release_year, runtime_minutes, genres, production_companies, popularity, overview`
- [ ] Write the filtered result to a local file (e.g. `data/movies_filtered.csv`) so it can be reviewed before anything touches a database
- [ ] Review together before moving to Chunk 2

This is pure pandas/CSV work — no network calls, no API keys needed, nothing that costs money.

## Chunk 2: Load structured data into Supabase — free, no LLM, not started

- [ ] Run `scripts/schema.sql` in the Supabase SQL editor (creates the `movies` table) — this one step needs to happen through the Supabase web UI; API keys alone don't grant DDL access
- [ ] Implement the Supabase-write half of `scripts/ingest.py`: read the Chunk 1 output, insert rows into `movies`
- [ ] Spot-check row count and a few rows in the Supabase table editor
- [ ] **Review checkpoint** — once this is done, the `CatalogFilter` tool (Supabase queries) can already be sanity-tested with plain SQL, with zero LLM spend. This is the natural pause point before touching Pinecone/LLMod.ai.

## Chunk 3: Embeddings + Pinecone — first $ spend, needs explicit go-ahead

- [ ] Create the Pinecone index (`movibot-plots`, cosine, dim=1536 for `text-embedding-3-small`)
- [ ] Implement the embedding half of `scripts/ingest.py`: embed each movie's `overview` via LLMod.ai, upsert to Pinecone with metadata `{movie_id, title, release_year}`
- [ ] Test with a small `--limit` (e.g. 100 movies) first, check the resulting vector count, **then** run the full ~5K ingest
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
