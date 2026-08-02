# Next Steps

Technical checklist, in build order, grouped by blocker. Due date: **2026-08-23**.

## 0. Blocked on input from the team

- [ ] Yair's email → `team_info.json`
- [ ] Andrei's email → `team_info.json`
- [ ] Create the Supabase project (web dashboard, not scriptable) → fill `SUPABASE_URL` / `SUPABASE_KEY` in `.env`
- [ ] Go-ahead to start spending LLMod.ai budget (first real ingestion + agent test calls)

## 1. Data layer

- [ ] Run `scripts/schema.sql` in the Supabase SQL editor once the project exists
- [ ] Download Kaggle "The Movies Dataset" (`movies_metadata.csv`), downsample to ~5K movies per `docs/team-idea-proposal-and-data-sources.pdf`
- [ ] Implement `scripts/ingest.py` (currently raises `NotImplementedError`):
  - [ ] Load + downsample CSV, select columns: `title, release_year, runtime_minutes, genres, production_companies, popularity, overview`
  - [ ] Insert rows into Supabase `movies` table
  - [ ] Create Pinecone index (`movibot-plots`, cosine, dim=1536 for `text-embedding-3-small`)
  - [ ] Embed `overview` via `agent/llm_client.py`'s `EMBEDDING_MODEL`, upsert to Pinecone with metadata `{movie_id, title, release_year}`
  - [ ] Test with `--limit 100` first, check Supabase row count + Pinecone vector count, **then** run the full ~5K ingest

## 2. Agent core (all currently stubs/`NotImplementedError`)

- [ ] `agent/llm_client.py` — sanity-check `get_client()` against real LLMod.ai with one cheap call
- [ ] `agent/tools/catalog_filter.py::run()` — LLM translates structured constraints → Supabase query
- [ ] `agent/tools/plot_search.py::run()` — embed query, Pinecone search, return matches
- [ ] `agent/tools/scene_search.py::run()` — live Wikipedia "Plot" section fetch + LLM check (decide fetch method: `wikipedia` pip lib vs raw REST call)
- [ ] `agent/tools/external_context.py::run()` — live Wikipedia non-Plot section fetch + LLM check
- [ ] `agent/react_loop.py::execute()` — Reasoner loop, `steps[]` construction, iteration cap, budget/time guard (Vercel's 300s limit)
- [ ] `app.py::execute()` — swap the hardcoded stub body for `agent.react_loop.execute(prompt)`

## 3. Architecture diagram

- [ ] Optionally polish `assets/architecture.png` past the current placeholder (arrows around Stop?/Synthesizer are cluttered)

## 4. GUI

- [ ] No code changes expected — `public/index.html` already renders the real `steps[]` shape. Just re-verify visually once `/api/execute` returns real data.

## 5. Content to replace with real data

- [ ] `agent_info.json` — swap the `STUB` `prompt_examples` entry for a captured real run (the Disney/toddler query)
- [ ] `README.md` — fill in the Vercel URL once deployed

## 6. Deployment

- [ ] Connect `alanarazi7/movibot` to Vercel
- [ ] Set env vars in the Vercel dashboard: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `SUPABASE_URL`, `SUPABASE_KEY`
- [ ] Deploy, verify all 4 endpoints in production
- [ ] Re-run the Disney/toddler demo query against the prod URL — confirm well under 300s and that the `steps` trace module names match `assets/architecture.png` exactly

## 7. Budget

- [ ] Track LLMod.ai spend against the $13 cap starting from the first real ingestion run

## Open design decision (not yet made)

`SceneSearch`/`ExternalContext` data sourcing: live Wikipedia fetch (current default per code comments) vs. pre-indexing ~5K pages upfront. Revisit if live fetch proves too slow/flaky once real agent calls start.
