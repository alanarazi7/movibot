# TODO

Working checklist. Due **2026-08-23**. Last revised **2026-08-16**.

Ordered by what unblocks what. Anything that spends budget is marked and needs
explicit go-ahead first — **nothing has been spent to date**.

The app is deployed and working at
[movibot-gamma.vercel.app](https://movibot-gamma.vercel.app); everything except
answering an actual query works there today. What remains is almost entirely
gated on credentials.

---

## Blocked on other people

- [ ] Yair Zack's email → `team_info.json` — the GUI shows a warning banner
      until this lands; it is driven from `/api/team_info`, so it clears itself

## Blocked on credentials

`.env` still holds placeholder values. Everything below the "free" line works
without them; nothing on the paid track can start.

- [ ] Real `OPENAI_API_KEY` + `OPENAI_BASE_URL` (LLMod.ai) — **these two alone
      unlock the 11 test cases locally**; Pinecone and Supabase are not involved
- [ ] Real `PINECONE_API_KEY` — only for `MOVIBOT_EMBEDDINGS=cloud`, i.e. production
- [ ] `SUPABASE_KEY` looks wrong: 41 chars, but the anon/service key is a JWT
      starting `eyJ` and usually 200+ chars. `SUPABASE_URL` looks fine

Check what is usable at any point, without spending anything:

```bash
python scripts/check_credentials.py          # free, no network
python scripts/check_credentials.py --ping   # 💰 one tiny call, settles the model id
```
- [x] **LLMod.ai model ids confirmed** against `GET /v1/models`, which lists
      exactly two for this tenant: `MB5R2CF-azure/gpt-5.4-mini` and
      `MB5R2CF-azure/text-embedding-3-small`. Both hardcoded defaults were
      wrong — the chat id said `gpt-4o-mini`, and the embedding id was missing
      the `azure/` segment — but the `MB5R2CF` prefix and the `azure/` segment
      themselves were right, contrary to what this file previously guessed

Check what is usable at any point, without spending anything:

```bash
python scripts/check_credentials.py          # free, no network
python scripts/check_credentials.py --ping   # 💰 one tiny call, settles the model id
```
- [ ] **Confirm the LLMod.ai chat model id.** The sibling `medium-rag-hw`
      project uses `4UHRUIN-text-embedding-3-small` and `4UHRUIN-gpt-5-mini` —
      `<TENANT>-<model>`, no `azure/` segment, and gpt-5-**mini**. So the
      current default `MB5R2CF-azure/gpt-4o-mini` is probably wrong on both
      counts; likely `MB5R2CF-gpt-5-mini`. Override with `MOVIBOT_MODEL`.

---

## Next up — free, agreed, not yet done

### 1. Explain chunking where it is actually seen

The Data tab shows "N passages in the search index" with no explanation, which
is where the question gets asked. The parameters are only documented in Project
Decisions → Retrieval and in `agent/chunking.py`.

- [ ] Add a line under the passages expandable: 300 tokens, 20% overlap, and
      why consecutive passages repeat a little

### 2. Close the semantic coverage gap

**234** films have readable plot text but only **159 are semantically
searchable**, because the index was built from the MPST file before the
Wikipedia cache existed. The other **75** can be read but never *found* — they
surface only if a structured filter lands on them first. Their Wikipedia plots
are not scraps: median 611 words against MPST's 892.

- [ ] Point `scripts/build_chunk_index.py` at the Wikipedia plot text as well,
      re-embed locally (free), and confirm 159 → 234 searchable
- [ ] Regenerate `public/data/` afterwards — the Data tab's passage counts and
      the architecture diagram both read from the index
- [ ] Note the trade-off before doing it: the local index would then differ
      from a Pinecone index built only from MPST, so item 6 must match

---

## Free track — larger, still open

### 3. Retrieval quality — partially solved, needs a decision after the first run

Chunking fixed reading completely and improved retrieval substantially, but
exposed a different problem: **E5-small-v2 ranks by surface events, not by
theme**, and its scores sit in a very narrow band.

Frozen's rank for the same underlying question, by phrasing:

| Query | Rank |
|---|---|
| "a prince reveals he never loved her and leaves her to die" | **#3** |
| "a man pretends to love a woman so he can seize the throne" | **#4** |
| "a charming stranger wins someone's trust and then betrays them" | #34 |
| "someone you just met turns out to be the villain" | #93 |

Total score spread across all 159 films is 0.076; across the top 20 it is
0.033. Signal exists but is weak and highly phrasing-sensitive. Mitigated for
now by instructing the planner to search for concrete events rather than themes.

- [ ] Try `text-embedding-3-small` instead of E5 — 1536-dim and much stronger;
      ~$0.01 to index all 1,254 passages
- [ ] Or have the planner issue 2–3 differently-phrased searches and union them
- [ ] Or accept it: the planner reads `matching_passage` as evidence, so a weak
      ranker degrades to "reads a few more candidates", not to a wrong answer

### 4. Decide whether to keep the local embedding backend

torch is ~518 MB against Vercel's 250 MB limit, so the local backend cannot ship
to production — which means local and cloud already take different paths.

- [ ] Decide: keep local E5 for free offline dev, or drop it and use the API in
      both (query embedding is ~$0.000002 per call)

### 5. Load Supabase (free, no model calls)

- [ ] Run `data_preprocessing/schema.sql` in the Supabase SQL editor — must go
      through the web UI, API keys don't grant DDL
- [ ] Implement the `--supabase-only` half of `scripts/ingest.py` (36 lines,
      still a stub)
- [ ] Verify 238 rows, and that `genres`/`keywords` land as real `jsonb` rather
      than JSON-encoded strings
- [ ] Sanity-check `MOVIBOT_BACKEND=cloud` returns identical results to local

---

## Paid track — needs explicit go-ahead

### 6. First real planner run  💰 small

**This is the highest-value remaining step.** Every behaviour below is
currently a prediction; none has been observed.

- [x] Credentials in `.env`; **first real call succeeded 2026-08-16**
      ("what is your name and purpose" → 1 model call, 0 tool calls, 2,613 +
      73 tokens, ~$0.0008). Two fixes it exposed: gpt-5 models reject
      `temperature != 1`, and token counts were structurally always zero
      because `complete()` discarded the response carrying `.usage`
- [ ] **Trim the system prompt.** It is 8,373 chars / ~2,600 tokens on every
      turn, and the brief asks to "minimize prompt/context size (only what's
      needed)". At 5 turns that is ~13K tokens of instructions per query
- [ ] Run the **11 test-bed cases** on the front page, in order. Each shows its
      expected behaviour; compare against what actually happens
- [ ] The three that are traps rather than exercises:
      - "starring Tom Hanks" — must refuse for want of cast data, *not* answer
        Toy Story from pretraining
      - "besides Frozen and Moana" — must become an `exclude_titles` filter,
        not merely a title it avoids naming
      - "a Disney movie in Hindi" — Dangal has 140 votes and sits #40 of 238 on
        broad queries; it must still win this narrow one
- [ ] Confirm the planner stays inside `MAX_ROUNDS = 5` and check the returned
      `budget` block for real token usage
- [ ] Watch for over-refusal on "a nice comedy" (should qualify, not refuse) and
      for invented post-2017 titles on "the latest Disney hit"
- [ ] Capture a real response into `agent_info.json` `prompt_examples` — both
      entries still say the prose is pending this run

### 7. Pinecone index  💰 ~$0.01

- [ ] Create index `movibot-plots`, cosine, dim **1536**, serverless
- [ ] Implement the `--pinecone-only` half of `scripts/ingest.py`
- [ ] Index **1,254 passages** (or 2,000-odd if item 2 lands first), not 159
      documents — ingestion must use `agent/chunking.py` to stay identical to
      the local path
- [ ] Test with `--limit 20`, verify vector count, then the full run
- [ ] Metadata per vector: `{movie_id, title, chunk_index, text}` — never
      `embedding_text`

### 8. Production with credentials  💰 inherits above

- [ ] Set Vercel env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
      `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `SUPABASE_URL`, `SUPABASE_KEY`,
      `MOVIBOT_BACKEND=cloud`, **`MOVIBOT_EMBEDDINGS=cloud`**
- [ ] `MOVIBOT_EMBEDDINGS=cloud` is mandatory in production — the local E5
      backend cannot run there, so item 7 is a hard prerequisite
- [ ] Confirm a real query finishes well inside Vercel's 300s limit
- [ ] Re-run the 11 test cases against production, not just locally

---

## How to deploy

**Deploys do not happen on `git push`.** This project is not Git-connected on
Vercel; a push updates GitHub only. Production changes require:

```bash
vercel --prod --yes
```

Do not trust the exit code — it returns 0 without necessarily promoting.
Verify by comparing bytes:

```bash
wc -c < public/index.html
vercel curl -sI https://movibot-gamma.vercel.app/ | grep -i content-length
```

`.vercelignore` keeps 113 MB of raw Kaggle input, the course PDFs, and the local
`.env` files out of the upload. It deliberately does **not** exclude
`data_preprocessing/data_ready/`, which the agent reads at runtime.

---

## Budget

$13 cap. **Spent so far: $0.**

| Item | Estimate |
|---|---|
| Pinecone embeddings (1,254 passages, ~350K tokens) | ~$0.01 |
| Planner turns | ≤5 model calls per query, hard-capped in `agent/loop.py` |
| The 11 test cases, once | a few cents |

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
- [x] Pipeline reproduces exactly from raw input

### Agent

- [x] Rebuilt as a **tool-calling agent**, replacing the six-module ReAct loop
- [x] Three tools: `filter_catalog`, `search_plots`, `read_synopses`
- [x] `agent/loop.py` — native tool calling, bounded at 5 model turns
- [x] Guardrails enforced in data and tool code, never the prompt
- [x] No mock model by design, so a broken config cannot masquerade as working
- [x] `MOVIBOT_OFFLINE` kill switch + placeholder-credential detection

### Chunked retrieval

- [x] `agent/chunking.py` — **sentence**-boundary chunking at 300 tokens / 20%
      overlap. Paragraph chunking was measured and rejected: **zero** synopses
      contain blank-line paragraphs and 66 of 159 have no newline at all
- [x] **1,254 passages from 159 films**, embedded locally with E5, free
- [x] `search_plots` scores each film by its **best** passage and returns that
      passage as quotable evidence; `read_synopses` returns relevant passages in
      story order rather than the opening N characters
- [x] Frozen's betrayal beat (chunk 21) is now retrievable — at 81% through the
      synopsis, it was invisible to the old document-level index

### Wikipedia resolution fixed (2026-08-14)

- [x] `wikipedia_client.py` rewritten. It tried the bare title first and took
      any extract over 200 chars — a bar every *disambiguation* page clears, so
      Frozen cached a list of links instead of an article
- [x] Now tries `"{title} ({year} film)"` first and rejects bad landings:
      disambiguation pages via `pageprops.disambiguation`, `List of …` pages,
      and any article whose lead sentence names the wrong year
- [x] Caught two redirect traps: *The Prince and the Pauper* → *List of
      adaptations of…*, and *Beverly Hills Chihuahua 3* → the 2008 first film,
      which had been caching the **wrong plot** entirely
- [x] Coverage: articles 228 → **237**, Plot sections 167 → **233**

### Scope and honesty (2026-08-15)

- [x] The prompt now states the catalog's bounds as data properties — Disney and
      Pixar only, 1940–2017, above 45 minutes, no cast or crew — and separates
      refusing a premise from qualifying an answer
- [x] `MAX_RECOMMENDATIONS = 3`; exhaustive requests are an explicit exception
- [x] **ON BEING EXHAUSTIVE** — completeness over 238 films is possible but
      prohibitively expensive, so the tools filter, rank, then read at most
      `MAX_SYNOPSES`. The prompt names that a heuristic and forbids presenting a
      shortlist as "the best in the catalog"
- [x] 2017 boundary documented as a property of the Kaggle snapshot

### GUI

- [x] **🗃️ Data tab** — all 238 films, searchable and filterable, with each
      film's complete stored record (metadata, synopsis, passages, Wikipedia)
      lazy-loaded from static JSON so the page stays small
- [x] **📋 Project Decisions** — Retrieval written; Agent and Deployment pending
- [x] **Test bed** — 11 hardcoded cases with expected behaviour, covering each
      tool, both scope limits, and the plausible failure modes
- [x] Team-email warning banner, derived from `/api/team_info` so it clears
      itself
- [x] **🗒️ Status tab** — this checklist, served live from `TODO.md` via
      `/api/status`, with progress counted from the checkboxes themselves

### Docs and deployment

- [x] `data_preprocessing/PIPELINE_REVIEW.md` is the single pipeline rationale;
      `DATA_SOURCES.md`, `data cleaning rules.md`, `NEXT_STEPS.md` and
      `DATA_IMPROVEMENTS_SUMMARY.md` deleted as superseded
- [x] `README.md` rewritten — it had described a four-tool ReAct loop with mock
      modules and claimed Wikipedia was fetched live
- [x] Transcripts dropped entirely (9 of 238 matched; too sparse to use)
- [x] `assets/architecture.png` redrawn with the data layer, the metered/free
      split, and every figure read at render time so it cannot drift
- [x] `.vercelignore` added — a deploy would otherwise have uploaded 113 MB of
      raw Kaggle input and the local `.env` files
- [x] Deployed to production and verified: all four endpoints, the static
      `/data/` routes, and graceful failure on `/api/execute` without credentials
