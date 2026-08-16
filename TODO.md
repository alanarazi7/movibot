# TODO

Working checklist. Due **2026-08-23**.

Spending needs explicit go-ahead. **Spent so far: ~$0.018 of $13** — the corpus
embedded once (~$0.0156), plus a handful of planner calls.

The app is live at [movibot-gamma.vercel.app](https://movibot-gamma.vercel.app)
and **answers real queries in production**. All four required endpoints work,
semantic search works, and the passage index covers every film in the catalog.

---

## Credentials

- [x] `OPENAI_API_KEY` + `OPENAI_BASE_URL` in `.env` **and in Vercel production**
- [x] Model ids confirmed against the tenant's list:
      `MB5R2CF-azure/gpt-5.4-mini`, `MB5R2CF-azure/text-embedding-3-small`
- [ ] `PINECONE_API_KEY` — **optional now.** The committed matrix serves search
      in production, so Pinecone is only for demonstrating the vector-DB path. A
      real key exists in the sibling `medium-rag-hw/.env` if we want it
- [ ] `SUPABASE_KEY` — unusable at 41 chars; the anon key is a JWT of 200+.
      Also optional: the CSV backend works

```bash
python scripts/check_credentials.py     # free, no network
```

---

## Next up

### 1. Guard the public endpoint  ⚠️ the one real exposure

`/api/execute` is public and ungated at up to ~$0.0143 per request, so roughly
**900 requests would exhaust the $13**. `MAX_ROUNDS` caps cost per request but
nothing caps requests.

- [ ] Accumulate real token usage against a `MOVIBOT_BUDGET_USD` cap and refuse
      once hit — the same shape as `MOVIBOT_OFFLINE`, but automatic. The budget
      block now reports true token counts, so the numbers exist

### 2. Retrieval quality — the open question

Moving from E5 to `text-embedding-3-small` did **not** fix phrasing
sensitivity, which was the hoped-for outcome. Measured on the same probe:

| Query | Rank |
|---|---|
| "a prince reveals he never loved her and leaves her to die" | **#2** |
| "someone you just met turns out to be the villain" | outside top 25 |

Mitigated by instructing the planner to search for concrete events, which is a
workaround rather than a fix.

- [ ] Try 200 / 300 / 450 token chunks — ~$0.007 each, and the content cache
      means only changed passages are re-embedded
- [ ] Or have the planner issue 2–3 differently-phrased searches and union them

### 3. Explain chunking where it is seen

- [x] The Retrieval tab now carries the parameters, the course defaults, and
      why we differ

---

## Remaining, in rough order

### 4. Run the 11 test cases  💰 ~$0.09

The agent answers in production, but only the identity prompt has actually been
run. Every other expected behaviour is still a prediction.

- [ ] Run all 11 from the front page and compare against the stated expectation
- [ ] The three that are traps: "starring Tom Hanks" must refuse rather than
      answer Toy Story from pretraining; "besides Frozen and Moana" must become
      a filter; "a Disney movie in Hindi" must still surface Dangal at 140 votes
- [ ] Watch for over-refusal on "a nice comedy" and invented post-2017 titles
- [ ] Capture a real response into `agent_info.json` `prompt_examples`, which
      still says the prose is pending

### 5. Trim the system prompt

2,613 tokens on every turn, and the brief asks to minimise prompt size. At five
turns that is ~13K tokens of instructions per query.

- [ ] Cut it, measuring that behaviour on the 11 cases does not regress

### 6. Optional: Supabase and Pinecone

Neither is needed. Both are supported and would only demonstrate the cloud path.

- [ ] Supabase: run `schema.sql`, implement the loader, verify 238 rows
- [ ] Pinecone: `python -m rag.ingest --sources all --pinecone`, then set
      `MOVIBOT_VECTOR_STORE=pinecone` in production

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

$13 cap. **Spent so far: ~$0.018** — one full corpus embedding
(~$0.0156) plus a handful of planner calls.

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

### Retrieval rebuilt (2026-08-16)

- [x] **`rag/` package** — chunking, embedding, storage, search, ingest and the
      decisions doc in one place, replacing logic spread over four files
- [x] **E5 dropped entirely.** It meant dev and production embedded with
      different models, so local testing never exercised production's rankings,
      and torch's 518 MB could not deploy at all
- [x] **Coverage gap closed.** Four corpora indexed, not one: 3,159 passages
      covering **all 238 films**, up from 1,254 covering 159
- [x] **Content-addressed embedding cache** — a passage is never embedded
      twice; verified against six invariants with a stubbed embedder
- [x] `--debug` ingests 10 passages per corpus for ~$0.0002
- [x] Tail passages are **merged, never discarded**: 0 sentences lost across
      the corpus, at a cost of one token on one passage
- [x] Ingest removed from the app — it writes into the repo, and a serverless
      filesystem is read-only, so it could only ever spend and fail
- [x] Three undeclared dependencies found by production and fixed: `tiktoken`,
      `pyarrow`, and `MOVIBOT_OFFLINE` failing to gate embedding
- [x] GUI: **Data**, **Retrieval** and **Status** tabs, with the four required
      endpoints grouped and labelled separately from the extras

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
- [x] **LLMod.ai model ids confirmed** against `GET /v1/models`, which lists
      exactly two for this tenant: `MB5R2CF-azure/gpt-5.4-mini` and
      `MB5R2CF-azure/text-embedding-3-small`. Both hardcoded defaults were
      wrong — the chat id said `gpt-4o-mini`, and the embedding id was missing
      the `azure/` segment — but the `MB5R2CF` prefix and the `azure/` segment
      themselves were right, contrary to what this file previously guessed
- [x] All three team emails in `team_info.json`; the warning banner that
      tracked them clears itself, so it is now gone from the page
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
