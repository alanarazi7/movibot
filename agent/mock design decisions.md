# Mock Agent Design Decisions

## Why a mock?

Chunk 4 (real agent implementation with LLM calls) is gated behind:
- Chunk 2 (Supabase table population) — not started
- Chunk 3 (Pinecone index + embeddings) — not started
- Explicit go-ahead, since Chunk 3+ spends the $13 LLMod.ai budget

This mock lets the full ReAct loop control flow (`Reasoner → CatalogFilter → PlotSearch → SceneSearch/ExternalContext → Synthesizer`) be built, tested, and demoed **end-to-end at zero cost** ahead of that gate. It directly serves the course spec's requirement #1: "avoid unnecessary LLM calls, stay within budget."

## Stand-in mapping

| Mock piece | Stands in for | Real implementation | Replaced by (TODO chunk) |
|---|---|---|---|
| `MockLLMClient.reason_next_action()` + deterministic policy | Reasoner LLM reasoning | `get_client()` + real LLMod.ai call | Chunk 4 |
| `catalog_filter.py` reading `supabase_movies.csv` | SQL queries on Supabase table | `supabase` Python client | Chunk 2 (populate) + Chunk 4 (wire) |
| `plot_search.py` local IDF + word-overlap scorer | Pinecone cosine-similarity search | `pinecone` Python client + real embeddings | Chunk 3 (embed & index) + Chunk 4 (wire) |
| `wikipedia_client.py` live fetch (already real, not mocked) | Same — no change | Same — already working | — |
| `_keyword_verdict()` heuristic (death, scary keywords) | LLM reasoning over text | Real LLMod.ai call with SCENE/EXTERNAL_SEARCH system prompts | Chunk 4 |

## Extraction approach — CSV-derived vocab

Rather than hardcoding studio/genre lists, all mock reasoning derives vocabulary directly from the local CSV files at load time:
- `catalog_filter.known_studios()` → flattened set of all `production_companies` in `supabase_movies.csv`
- `catalog_filter.known_genres()` → flattened set of all `genres` in `supabase_movies.csv`
- `MockLLMClient.translate_catalog_constraints()` calls these helpers to match user text against real data vocabulary

**Example:** user says "Disney movies" → regex extract "disney" → substring-match "disney" against known studios including `"Walt Disney Pictures"`, `"Walt Disney Animation Studios"`, `"DisneyToon Studios"` — all match at once, which is actual behavior in the 303-row dataset. If the dataset changes (e.g., `--all-studios` regeneration), the vocab auto-updates with zero code changes.

## Known limitations

1. **Word-overlap + IDF scoring cannot capture true semantics.** Pinecone's cosine similarity on dense embeddings finds thematic meaning; the mock's token overlap finds keyword matches. A query like "family-friendly adventure" will only match if those exact words appear in the plot text or tags.

2. **Constraint verdict taxonomy is minimal.** Only "death" and "scary" categories are recognized out of the box. Unrecognized constraint types (e.g., "involves animals") return `satisfied: None` with an honest `evidence` explanation rather than guessing.

3. **Wikipedia section-splitting is heuristic.** The `split_into_sections()` function assumes short lines flanked by blank lines are headings. Malformed Wikipedia extracts may split incorrectly.

4. **Regex-based constraint extraction is brittle.** Year/studio/genre extraction from free text uses regex and substring matching. Unusual phrasing (e.g., "Pixar's 2010 film") may not extract correctly.

5. **Each tool fires at most once per turn.** The mock's deterministic `reason_next_action()` has no organic way to decide "retry CatalogFilter with refined args." Re-entry is prevented by tracking `actions_taken` as a set + `MAX_ITERATIONS = 6` hard cap. The real agent (with LLM reasoning) will naturally support iterative refinement.

6. **SceneSearch prefers local `plot_synopsis` over live Wikipedia.** For the 170 titles with MPST data, the local 693-word synopsis is used directly (faster, more reliable, richer text than raw Wikipedia overviews). Live Wikipedia is only fetched for the remaining 133 titles. This is a deliberate trade-off: graceful degradation > strict "always fetch live."

## How to swap each mock for the real thing

### `MockLLMClient` → real LLM calls (Chunk 4)

**Current:** `react_loop.py` imports `from agent.llm_client import get_mock_client` and uses it throughout.

**To swap:**
1. Verify `OPENAI_API_KEY` and `OPENAI_BASE_URL` env vars are set.
2. Change the import to `from agent.llm_client import get_client` (or introduce a factory that checks an env var).
3. Ensure the real client's method signatures match `MockLLMClient` (they should: both return the same JSON shapes defined in `agent/prompts.py`).
4. Each real method sends the corresponding `*_SYSTEM_PROMPT` + user context to the LLMod.ai `MB5R2CF-azure/gpt-5.4-mini` model and parses the JSON response.

### `catalog_filter.py` local CSV → Supabase query (Chunk 2+4)

**Current:** reads from `data_preprocessing/data_ready/supabase_movies.csv` in memory.

**To swap:**
1. Ensure Supabase project is created and the `movies` table is populated (Chunk 2).
2. Replace CSV load with `supabase.from("movies").select("*")...` using the supabase Python client.
3. Apply the same filter logic (year, runtime, genre exact-match, studio substring-match) as SQL WHERE clauses.
4. Return the same candidate dict structure.

### `plot_search.py` local scoring → Pinecone (Chunk 3+4)

**Current:** tokenizes `embedding_text`, computes corpus-local IDF, scores by token overlap + tag boost.

**To swap:**
1. Ensure Pinecone index `movibot-plots` is created and populated with embeddings (Chunk 3).
2. Replace scoring with:
   - Embed the query using `EMBEDDING_MODEL` (`text-embedding-3-small` via LLMod.ai).
   - Query Pinecone with cosine metric, return top-K results with scores.
3. Return the same result dict structure (movie_id, title, release_year, score, matched_terms).

### `SceneSearch`/`ExternalContext` keyword heuristics → LLM reasoning (Chunk 4)

**Current:** `_keyword_verdict()` classifies constraint type (death, scary) and searches for keywords in text, returning `satisfied: true/false/None`.

**To swap:**
1. After fetching plot/external text via `wikipedia_client` (which stays unchanged), send the text + constraint to LLMod.ai with the `SCENE_SEARCH_SYSTEM_PROMPT` or `EXTERNAL_CONTEXT_SYSTEM_PROMPT`.
2. Parse the JSON response: `{"title", "constraint", "satisfied": true|false|null, "evidence"}`.
3. Return that verdict directly, skipping the heuristic logic entirely.

### Wikipedia live fetch → pre-indexed pages (open decision)

**Current:** `wikipedia_client.py` always fetches live via MediaWiki API, with `timeout=5.0` and graceful `None` on failure. No pre-indexing.

**Alternative (not implemented):** pre-download ~5K Wikipedia plot/non-plot sections, embed them, and index in Pinecone or a local SQLite index. This would eliminate network latency during /api/execute and respect the 300s Vercel timeout more robustly. Downside: ~5K pages × 693 words ≈ 3.5M words; embedding cost is non-trivial even at `text-embedding-3-small` rates.

**Decision:** live fetch is acceptable for the 303-movie demo set (max ~50 Wikipedia requests per query, ~5s worst-case). Pre-indexing can be revisited if timeouts become an issue.

## Files changed

- `agent/llm_client.py` — added `MockLLMClient` class + `get_mock_client()` factory
- `agent/tools/catalog_filter.py` — replaced `NotImplementedError` with CSV-backed filtering
- `agent/tools/plot_search.py` — replaced `NotImplementedError` with IDF + word-overlap scoring
- `agent/tools/scene_search.py` — replaced `NotImplementedError` with local-synopsis-first + Wikipedia fallback + heuristic verdict
- `agent/tools/external_context.py` — replaced `NotImplementedError` with live Wikipedia fetch + heuristic verdict
- `agent/tools/wikipedia_client.py` — new, shared Wikipedia fetching utilities
- `agent/react_loop.py` — replaced `NotImplementedError` with full ReAct loop orchestration
- `app.py` — swapped hardcoded stub response for call to `react_loop.execute()`

## No changes to

- `agent/prompts.py` — all system prompts remain unchanged
- `agent_info.json` — `prompt_examples` still have STUB data (will be replaced after first real run)
- `.env`, `requirements.txt`, dataset files — all unchanged
