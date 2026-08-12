# Next Steps: Real Backend Integration

**Current state:** Mock agent deployed & live at https://movibot-gamma.vercel.app  
**Due date:** 2026-08-23  
**Progress:** Chunks 1 & 5 complete ✅ | Chunks 2-4 remain

---

## Immediate Priority

### Chunk 0: Team Contact Info (Quick Win)
- [ ] Get Yair Zack's email → add to `team_info.json`
- [ ] Get Andrei Nekliudov's email → add to `team_info.json`
- [ ] Commit and redeploy
- **Status:** Alan's email ✅, two blanks remain

### Chunk 2: Supabase (FREE, no LLM - Cost-Gated)
**Prerequisite:** Supabase project exists (URL + key in `.env`, gitignored)

1. [ ] **Create `movies` table in Supabase**
   - Run `data_preprocessing/schema.sql` via Supabase web UI (SQL editor)
   - Creates: `id`, `imdb_id`, `title`, `release_year`, `runtime_minutes`, `genres` (JSON), `overview`, `keywords` (JSON), `has_mpst_synopsis` (bool), + 16 more columns
   - **Why web UI:** Supabase API keys don't grant DDL access

2. [ ] **Implement `scripts/ingest.py` — Supabase write half**
   - Read `data_preprocessing/data_ready/supabase_movies.csv` (303 rows, 25 columns)
   - Insert all rows into `movies` table via Supabase Python client
   - Handle duplicates gracefully (upsert if id exists)
   - Test: `python scripts/ingest.py --supabase-only`

3. [ ] **Sanity check**
   - Query Supabase: `SELECT COUNT(*) FROM movies` → should be 303
   - Spot-check a few rows (e.g., Frozen, Tinker Bell)
   - Verify all 25 columns present

4. [ ] **Review checkpoint — explicit go-ahead needed before Chunk 3**
   - At this point, `CatalogFilter` tool can be tested with real Supabase queries
   - Zero LLM cost, zero Pinecone cost
   - This is the natural pause before paid services

---

### Local Sandbox: E5 Embedding Backend Test (OPTIONAL but RECOMMENDED - FREE)
**Prerequisite:** Chunk 1 complete (data ready)  
**Status:** No budget wasted; test semantic search quality before Chunks 3-4 ($13 spend)  
**See:** [LOCAL_SANDBOX.md](LOCAL_SANDBOX.md)

1. [ ] **Install local dependencies**
   ```bash
   pip install -r requirements-local.txt
   ```

2. [ ] **Build embedding cache**
   - Embed 170 movies with E5-small-v2 (384-dim, local, free)
   - Store in `data_preprocessing/data_ready/plot_embeddings.*` (gitignored, regenerable)
   - `python scripts/local_sandbox_setup.py`

3. [ ] **Test PlotSearch with embedding backend**
   - Enable: `export PLOT_SEARCH_BACKEND=embedding`
   - Run queries via: `python app.py` then `/api/execute`
   - Verify: embedding results are more semantic than IDF mock (different matches on thematic queries)
   - Disable: `unset PLOT_SEARCH_BACKEND` to revert to mock (IDF) for regression testing

4. [ ] **Verify full ReAct loop with embeddings**
   - Run Disney/toddler demo query with embedding backend enabled
   - Confirm all 6 steps in trace (Reasoner → CatalogFilter → PlotSearch → SceneSearch → ExternalContext → Synthesizer)
   - Confirm PlotSearch step shows embedding-based results
   - Confirm zero external API calls (MockLLMClient still used for reasoning)

5. [ ] **Decision point: proceed to Chunks 2-4?**
   - If semantic search quality looks good → proceed with Chunks 2-4 (Supabase, Pinecone, LLMod.ai)
   - If results need tuning → iterate on prompts/queries locally before spending budget

---

### Chunk 3: Pinecone Embeddings (PAID: first LLMod.ai spend - Cost-Gated)
**Prerequisite:** Chunk 2 complete, LLMod.ai key in `.env`, Pinecone key in `.env`  
**Approval gate:** Explicit go-ahead needed (will spend budget)

1. [ ] **Create Pinecone index**
   - Name: `movibot-plots`
   - Metric: `cosine`
   - Dimension: `1536` (for `text-embedding-3-small`)
   - Serverless (cheapest)

2. [ ] **Implement `scripts/ingest.py` — Pinecone embedding half**
   - Read `data_preprocessing/data_ready/pinecone_candidates.csv` (170 rows with plot synopses)
   - For each row:
     - Call LLMod.ai to embed `embedding_text` → 1536-dim vector
     - Upsert to Pinecone with metadata: `{movie_id, title, release_year}`
     - Never store `embedding_text` itself as metadata (bloats index)
   - **Test with limit first:** `python scripts/ingest.py --pinecone-only --limit 20`
   - Check Pinecone dashboard: should have 20 vectors
   - **Then full run:** `python scripts/ingest.py --pinecone-only` (170 total)

3. [ ] **Cost tracking**
   - 170 embeddings @ text-embedding-3-small ≈ $0.02–0.03
   - Track actual spend against $13 course budget
   - Document in commit message

4. [ ] **Test PlotSearch tool**
   - Run a local query via `agent.tools.plot_search.run()`
   - Verify Pinecone cosine search returns sensible matches

---

### Chunk 4: Real LLM Calls (PAID: main budget spend)
**Prerequisite:** Chunks 2 & 3 complete, LLMod.ai authenticated  
**Approval gate:** Explicit go-ahead needed (will spend most of budget)

1. [ ] **Swap mock → real LLM in `agent/llm_client.py`**
   - `reason_next_action()` → call LLMod.ai with `REASONER_SYSTEM_PROMPT`
   - `verify_scene_constraint()` → call LLMod.ai with `SCENE_SEARCH_SYSTEM_PROMPT`
   - `verify_external_constraint()` → call LLMod.ai with `EXTERNAL_CONTEXT_SYSTEM_PROMPT`
   - `synthesize_answer()` → call LLMod.ai with `SYNTHESIZER_SYSTEM_PROMPT`

2. [ ] **Replace mock tools with real ones**
   - `CatalogFilter`: CSV → Supabase queries (from Chunk 2)
   - `PlotSearch`: IDF mock → Pinecone cosine search (from Chunk 3)
   - `SceneSearch`: mock logic → live Wikipedia fetch + LLM reasoning
   - `ExternalContext`: mock logic → live Wikipedia fetch + LLM reasoning

3. [ ] **End-to-end test**
   - Run the Disney/toddler demo query locally
   - Verify: `steps` trace shows real Reasoner/CatalogFilter/PlotSearch/SceneSearch/ExternalContext/Synthesizer
   - Verify: module names in trace match `assets/architecture.png`
   - Track LLMod.ai spend

4. [ ] **Test against production Supabase + Pinecone**
   - Confirm response completes in <300s
   - Verify responses are sensible (not hallucinated)

---

## Follow-Up (Post-Due-Date)

### Chunk 4 Polish (if time + budget remain)
- [ ] Improve system prompts based on real runs
- [ ] Handle edge cases (empty results, query timeouts, etc.)
- [ ] Add logging for debugging

### Future: Scale to full catalog
- [ ] Regenerate data with `--all-studios` (not just Disney + Pixar)
- [ ] Re-run Chunks 2-3 (Supabase + Pinecone)
- [ ] Re-test Chunk 4 at scale

---

## Blockers & Dependencies

| Blocker | Status | Owner |
|---------|--------|-------|
| Yair's email | 🔴 Missing | @yair |
| Andrei's email | 🔴 Missing | @andrei |
| Supabase project URL + key | ✅ Ready | (in `.env`) |
| LLMod.ai key | 🔴 Needed for Chunk 3 | @alan |
| Pinecone API key | 🔴 Needed for Chunk 3 | @alan |

---

## Commands Cheat Sheet

```bash
# Local testing (Chunks 2-4)
cd ~/tabstar/movibot
python scripts/ingest.py --supabase-only       # Chunk 2: load to Supabase
python scripts/ingest.py --pinecone-only --limit 20  # Chunk 3: test 20 vectors
python scripts/ingest.py --pinecone-only       # Chunk 3: full 170 vectors

# Local agent test (after Chunk 4)
python3 -c "
from agent import react_loop
result = react_loop.execute('Find me a 1990s Disney movie')
print(result['response'])
"

# Deploy to production
git add . && git commit -m "..." && git push origin master && vercel --prod
```

---

## Budget Summary

| Phase | Service | Est. Cost | Status |
|-------|---------|-----------|--------|
| Chunks 1-2 | (none) | Free | ✅ Complete |
| **Local Sandbox** | **Claude API** | **~$0.50** | 🟡 Optional (recommended) |
| Chunk 3 | Pinecone + LLMod.ai embeddings | ~$0.03 | 🟡 Planned |
| Chunk 4 | LLMod.ai LLM calls | ~$12.97 | 🟡 Planned |
| **Total (with Sandbox)** | | **~$13.50** | 🎯 Slight overage OK |
| **Total (without Sandbox)** | | **~$13.00** | 🎯 On budget |

**Note:** Local Sandbox ($0.50) is worthwhile to catch issues before committing to Chunk 4 ($12.97). Paying $0.50 to validate before $13 spend = good ROI.

All estimates based on 170 embeddings + ~3 LLM calls per user query (conservatively ~20 queries max before hitting budget cap).

---

## Notes

- **No secrets in repo:** All keys (OPENAI_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY) stay in `.env` (gitignored). Set via Vercel dashboard only when deploying Chunks 2-4-real.
- **Mock is production-ready:** Current deployment works 100% without real backends. Chunks 2-4 are strictly improvements, not bug fixes.
- **Review points:** Explicit sign-off needed before Chunk 3 (cost) and Chunk 4 (main spend). Chunk 2 is free & reversible.

