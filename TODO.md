# TODO

Working checklist. Due **2026-08-23**. Last revised **2026-08-13**.

Ordered by what unblocks what. Anything that spends budget is marked and needs
explicit go-ahead first — nothing has been spent to date.

---

## Blocked on other people

- [ ] Yair Zack's email → `team_info.json` — the GUI shows a warning banner
      until this lands; it is driven from `/api/team_info`, so it clears itself
- [x] Andrei Nekliudov's email → `team_info.json`

## Blocked on credentials

`.env` still holds placeholder values (`your-llmod-api-key`, etc). Everything
below the "free" line works without them; nothing above it can start.

- [ ] Real `OPENAI_API_KEY` + `OPENAI_BASE_URL` (LLMod.ai)
- [ ] Real `PINECONE_API_KEY`
- [ ] **Confirm the LLMod.ai chat model id.** The sibling `medium-rag-hw`
      project uses `4UHRUIN-text-embedding-3-small` and `4UHRUIN-gpt-5-mini` —
      `<TENANT>-<model>`, no `azure/` segment, and gpt-5-**mini**. So the
      current default `MB5R2CF-azure/gpt-4o-mini` is probably wrong on both
      counts; likely `MB5R2CF-gpt-5-mini`. Override with `MOVIBOT_MODEL`.

---

## Free track — can start now, costs nothing

### 1. Retrieval quality — partially solved, needs a decision

Chunking landed (see Done). It fixed reading completely and improved retrieval
substantially, but exposed a second, different problem: **E5-small-v2 ranks by
surface events, not by theme**, and its scores sit in a very narrow band.

Frozen's rank for the same underlying question, by phrasing:

| Query | Rank |
|---|---|
| "a prince reveals he never loved her and leaves her to die" | **#3** |
| "a man pretends to love a woman so he can seize the throne" | **#4** |
| "a charming stranger wins someone's trust and then betrays them" | #34 |
| "someone you just met turns out to be the villain" | #93 |

Total score spread across all 159 films is 0.076; across the top 20 it is
0.033. Signal exists but is weak and highly phrasing-sensitive.

Mitigated for now by instructing the planner to search for concrete events
rather than themes (`agent/prompts.py`). Options if that proves insufficient
after the first real run:

- [ ] Try `text-embedding-3-small` for retrieval instead of E5 — 1536-dim and
      much stronger; costs ~$0.01 to index all 1,254 passages
- [ ] Have the planner issue 2–3 differently-phrased searches and union them
- [ ] Accept it: the planner reads `matching_passage` as evidence, so a weak
      ranker degrades to "reads a few more candidates" rather than to a wrong
      answer

### 2. Decide whether to keep the local embedding backend

torch is 518 MB, cannot deploy to Vercel (250 MB limit), and creates a
local/cloud divergence risk. The sibling `medium-rag-hw` project ships the same
kind of system with a 3-line `requirements.txt` and embeds via the API.

- [ ] Decide: keep local E5 for free offline dev, or drop it and use the API in
      both dev and prod (query embedding is ~$0.000002 per call)

### 3. Load Supabase (free, no model calls)

- [ ] Run `data_preprocessing/schema.sql` in the Supabase SQL editor — must go
      through the web UI, API keys don't grant DDL. Now includes `weighted_rating`.
- [ ] Implement the `--supabase-only` half of `scripts/ingest.py` (still a stub)
- [ ] Verify 238 rows, and that `genres`/`keywords` land as real `jsonb` not
      JSON-encoded strings
- [ ] Sanity-check `MOVIBOT_BACKEND=cloud` returns identical results to local

### 4. Close the semantic coverage gap (optional)

220 films have readable plot text but only **159 are semantically searchable**,
because vectors were built only for the MPST subset. Closing it is free locally
but would make the local and Pinecone indexes diverge — decide before doing it.

---

## Paid track — needs explicit go-ahead

### 5. First real planner run  💰 small

- [ ] Unset `MOVIBOT_OFFLINE`, put real credentials in `.env`
- [ ] One query end to end via `agent.loop.execute(...)`
- [ ] Confirm the planner picks sane tools, respects exclusions, and stays
      inside `MAX_ROUNDS = 5`
- [ ] Check the returned `budget` block for actual token usage
- [ ] Capture the real response into `agent_info.json` `prompt_examples` —
      both entries currently say the prose is pending this run
- [ ] Re-test query D ("warns about trusting strangers") — it is the one that
      exposed the truncation defect

### 6. Pinecone index  💰 ~$0.01

- [ ] Create index `movibot-plots`, cosine, dim **1536**, serverless
- [ ] Implement the `--pinecone-only` half of `scripts/ingest.py`
- [ ] Embed via LLMod.ai, batched (~50 per call), upsert metadata
      `{movie_id, title, release_year}` only — never `embedding_text`
- [ ] Test with `--limit 20` first, verify vector count, then the full run
- [ ] Index **1,254 passages**, not 159 documents — chunking has landed, so
      ingestion must use `agent/chunking.py` to stay identical to the local path
- [ ] Metadata per vector: `{movie_id, title, chunk_index, text}`

### 7. Deploy  💰 inherits above

- [ ] **Production must set `MOVIBOT_EMBEDDINGS=cloud`.** The local E5 backend
      cannot run on Vercel: torch is ~518 MB against a 250 MB serverless limit.
      This makes item 5 a hard prerequisite for deploy, not an optimisation.
- [ ] Set Vercel env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
      `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `SUPABASE_URL`, `SUPABASE_KEY`,
      `MOVIBOT_BACKEND=cloud`, `MOVIBOT_EMBEDDINGS=cloud`
- [ ] Deploy, verify all four endpoints in production
- [ ] Confirm a real query finishes well inside Vercel's 300s limit
- [ ] Confirm `steps` module names match `assets/architecture.png` exactly
- [ ] `README.md` — fill in the deployed URL

---

## Budget

$13 cap. **Spent so far: $0.** Every tool runs locally; only planner turns and
Pinecone embedding cost anything.

| Item | Estimate |
|---|---|
| Pinecone embeddings (1,254 passages, ~350K tokens) | ~$0.01 |
| Planner turns | ≤5 model calls per query, hard-capped in `agent/loop.py` |

`MOVIBOT_OFFLINE=1` blocks all spending at the client, not just in the loop.
Placeholder credentials are detected and rejected rather than attempted.

---

## Done

### Data pipeline

- [x] Two Kaggle sources → Disney/Pixar scope → cleaned, deduped, merged with MPST
- [x] **Feature films only** — 65 shorts under 45 min dropped at preparation
      time. They held 8 of the top 10 rating slots (*Lou*: 8.5 on 17 votes)
- [x] **Bayesian `weighted_rating`** (IMDb Top-250 formula, m=300, C=6.199) as
      the default sort. Chosen over a `vote_count` floor, which would have
      deleted 141 of 238 films and made narrow queries unanswerable
- [x] Outputs: `supabase_movies.csv` (238), `pinecone_candidates.csv` (159)
- [x] `wikipedia_cache.csv` pruned to match (238)
- [x] Pipeline reproduces exactly from raw input

### Wikipedia resolution fixed (2026-08-14)

- [x] **`wikipedia_client.py` rewritten.** It tried the bare title first and
      accepted any extract over 200 chars — a bar every *disambiguation* page
      clears, so films like Frozen cached a list of links instead of an article
- [x] Now tries `"{title} ({year} film)"` first and rejects bad landings:
      disambiguation pages via `pageprops.disambiguation`, `List of …` index
      pages, and any article whose lead sentence names the wrong year
- [x] Caught two redirect traps: *The Prince and the Pauper* → *List of
      adaptations of…*, and *Beverly Hills Chihuahua 3* → the 2008 first film,
      which had been caching the **wrong plot** entirely
- [x] Scraper fetched each page twice, the second time without the year, so the
      two halves of a row could come from different articles. Now fetches once
- [x] Coverage: articles 228 → **237**, Plot sections 167 → **233**,
      non-plot 208 → **237**

### Cleanup (2026-08-14)

- [x] **Transcripts dropped entirely** — `find_transcripts.py` and
      `transcript_matches.csv` deleted; 9/238 coverage was never usable
- [x] **`data_preprocessing/PIPELINE_REVIEW.md`** is now the single rationale
      doc for the pipeline
- [x] Deleted as superseded: `DATA_SOURCES.md`, `data cleaning rules.md`,
      `NEXT_STEPS.md`, `DATA_IMPROVEMENTS_SUMMARY.md`, and the empty root
      `data_ready/`. Kaggle download steps preserved in the usage doc
- [x] `README.md` rewritten — it still described a four-tool ReAct loop with
      mock modules and claimed Wikipedia was fetched live
- [x] `schema.sql` no longer declares `poster_path`/`homepage`, which the CSV
      never produced
- [x] Project Decisions tab added to the GUI, with the full Frozen record

### Agent

- [x] Rebuilt as a **tool-calling agent**, replacing the six-module ReAct loop
- [x] Three tools: `filter_catalog`, `search_plots`, `read_synopses`
- [x] `agent/loop.py` — native tool calling, bounded at 5 model turns
- [x] `agent/catalog.py`, `agent/embeddings.py` — local/cloud backends, local default
- [x] Guardrails enforced in data and tool code, never the prompt
- [x] Deleted `react_loop.py` and all four mock tool modules; no mock model by
      design, so a broken config cannot masquerade as a working agent
- [x] `MOVIBOT_OFFLINE` kill switch + placeholder-credential detection
- [x] `wikipedia_client.py` moved to `data_preprocessing/` — the agent reads the
      offline cache and never fetches Wikipedia live
- [x] Verified end to end at zero cost: all endpoints, both guard paths, and
      semantic retrieval (*Ratatouille*, *Mulan* both #1 from paraphrase)

### Chunked retrieval

- [x] `agent/chunking.py` — **sentence**-boundary chunking at 300 tokens / 20%
      overlap. Paragraph chunking (the `medium-rag-hw` approach) was measured
      and rejected: **zero** synopses contain blank-line paragraphs and 66 of
      159 have no newline at all, so it would emit one chunk per document
- [x] Parameters chosen from measurement, not inherited: sentences are median
      23 tokens, so 300 holds ~13 — about one scene, tight enough that a story
      beat dominates its passage
- [x] Handles this corpus's scrape defect (`...bot-fights.His older brother`)
      where the space after a full stop was lost
- [x] `scripts/build_chunk_index.py` — **1,254 passages from 159 films**
      (7.9 per film), embedded locally with E5. Free, offline
- [x] `search_plots` over-fetches passages, scores each film by its **best**
      passage, and returns that passage as quotable evidence
- [x] `read_synopses` takes `about` and returns relevant passages in story
      order instead of the opening N characters
- [x] Frozen's betrayal beat is now retrievable and readable — chunk 21 carries
      "if only there was someone here who loved you". Previously invisible at
      both stages
- [x] Superseded document-level index (`plot_embeddings.*`) deleted

### Docs

- [x] `assets/architecture.png` regenerated for the new module names
- [x] `agent_info.json` rewritten; `data_preprocessing/PIPELINE_REVIEW.md` documents
      the pipeline, its guardrails, and the open findings
- [x] Row counts corrected across all six docs (37 substitutions, recomputed not scaled)
- [x] `requirements-local.txt` split out, since torch cannot ship to Vercel
- [x] Architecture page published:
      https://claude.ai/code/artifact/1119c2df-5651-4fbf-903b-ba170a49ff5a

---

## Notes

- Transcripts: investigated and dropped — only 9 of 238 films matched the
  HuggingFace corpus (3.8%). Script and coverage report deleted 2026-08-14.
