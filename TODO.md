# TODO

Open items only. Due **2026-08-23**.

Spending needs explicit go-ahead. **Spent so far: ~$0.018 of $13** — the corpus
embedded once (~$0.0156), plus a handful of planner calls.

The app is live at [movibot-gamma.vercel.app](https://movibot-gamma.vercel.app)
and answers real queries in production. Nothing is blocked on a credential:
the Pinecone and Supabase backends were removed, and the two that remain
(`OPENAI_API_KEY`, `OPENAI_BASE_URL`) are set locally and in production.

---

## Architecture

The shape we agreed: decompose a request into conditions, exhaust the
structured ones, then spend money only on what is left.

### Where it stands

```
POST /api/execute
  └─ loop.py            up to MAX_ROUNDS = 5 model turns; last turn gets no
                        tools, so it must answer rather than ask
       └─ Planner       one prompt, 2,005 tokens, every turn. The only paid step
            ├─ filter_catalog   238 -> N by column. Free. Sets the working set
            ├─ search_plots     ranks within it. One embedding, ~$0.0000002
            └─ read_synopses    <= 8 films, <= 6,000 chars each. Free
```

In place:

- **Working set in Python.** `filter_catalog` records every matching id in a
  request-scoped `ToolContext`; search and read scope to it automatically.
  Nothing is passed between tools, and no match is lost to a display cap —
  which it used to be, 172 films at a time.
- **Films addressed by name.** `Title (Year)` is unique across all 238, so no
  id enters the prompt. A filter response is 373 tokens where it was 13,791.
- **Four corpora, 3,159 passages, 1536-dim, scored in memory.** Search covers
  the two plot-bearing ones by default; `wiki_context` and `overview` stay
  reachable but no longer outrank plots on story questions.
- **Data**: 238 films × 26 columns from committed CSVs (2 ms); 234 films
  readable in full; no database of any kind.
- **Guardrails in code, not prose**: the 45-minute floor and `weighted_rating`
  ordering are properties of the data; `MAX_RECOMMENDATIONS = 3`,
  `PREVIEW_FILMS = 15`, `MAX_SEARCH_RESULTS = 25`, `MAX_SYNOPSES = 8`.

What is missing is making the decomposition explicit, and describing the whole
thing honestly.

### A1. Condition plan — the remaining piece

`filter_catalog` establishes the working set, but *which* conditions are
structured is still free-form model judgement rather than a visible step.

- [ ] A `plan` tool the planner calls first, classifying each condition as
      structured / semantic / subjective, so the split is traced and a
      mis-classification is diagnosable rather than mysterious

### A2. Architecture tab

- [ ] Show the diagram immediately instead of behind a button
- [ ] "How a request flows", using the narrowing trace as the worked example
- [ ] A section per component — Planner, each tool, each store — with its
      guardrail constants live, and the Planner's prompt served from
      `prompts.py` so it cannot drift
- [ ] Redraw the diagram: it mentions neither the working set nor the four
      corpora, which are now the two most interesting things about the flow

### A3. Prompt review

Never reviewed whole; it grew one section per problem we hit. ~2,100 tokens on
every turn, against a brief that asks to minimise context.

- [ ] Read it end to end against each component, once the tab puts them side
      by side
- [ ] Trim, then confirm the 11 cases do not regress

---

## Correctness and cost

### C1. Guard the public endpoint  ⚠️ the one real exposure

`/api/execute` is public and ungated at up to ~$0.0143 per request, so roughly
**900 requests would exhaust the $13**. `MAX_ROUNDS` caps cost per request but
nothing caps requests.

- [ ] Accumulate real token usage against a `MOVIBOT_BUDGET_USD` cap and refuse
      once hit — the budget block now reports true counts, so the numbers exist

### C2. Retrieval quality

The stronger embedding model did **not** fix phrasing sensitivity:

| Query | Rank |
|---|---|
| "a prince reveals he never loved her and leaves her to die" | **#2** |
| "someone you just met turns out to be the villain" | outside top 25 |

- [ ] Try 200 / 300 / 450 token chunks — ~$0.007 each, and the content cache
      means only changed passages are re-embedded
- [ ] Or have the planner issue 2–3 differently-phrased searches and union them

---

## Validation

### V1. Revise the 11 cases, then run them  💰 ~$0.09

Written before the corpora, the model change, `MAX_RECOMMENDATIONS`, and the
working set. Stale in several ways, and none has ever been run.

- [ ] Update the expectations: films are named not numbered, chunk ids are now
      `mpst_<id>_<n>`, answers may name up to 3 films, results carry a corpus
- [ ] Add a case for the corpus defect — a story question where a cast list
      could plausibly win — so it cannot silently return
- [ ] Run all 11 and compare against what each says should happen
- [ ] The three traps: "starring Tom Hanks" must refuse rather than answer Toy
      Story from pretraining; "besides Frozen and Moana" must become a filter;
      "a Disney movie in Hindi" must still surface Dangal at 140 votes
- [ ] Watch for over-refusal on "a nice comedy" and invented post-2017 titles
- [ ] Capture a real response into `agent_info.json` `prompt_examples`

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

`.vercelignore` keeps 113 MB of raw Kaggle input, the course PDFs, and the
local `.env` files out of the upload. It deliberately does **not** exclude
`data_preprocessing/data_ready/`, which the agent reads at runtime.

---

## Budget

$13 cap. **Spent so far: ~$0.018.**

| Item | Cost |
|---|---|
| Full corpus embedding (3,159 passages), already done | $0.0156 |
| One query embedding | ~$0.0000002 |
| One planner turn | $0.0011 cheap, $0.0041 after reading synopses |
| Worst-case request (5 turns) | $0.0143 |
| The 11 test cases, once | ~$0.09 |

`MOVIBOT_OFFLINE=1` blocks all spending — both planner calls and embedding.
