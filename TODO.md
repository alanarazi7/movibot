# TODO

Open items only. Due **2026-08-23**.

Spending needs explicit go-ahead. **The live figure is on the TODO tab**, read
from LLMod.ai's own accounting via `/api/budget` rather than typed here, where
it went stale the moment anyone ran a query. The bulk of it is the corpus,
embedded once (~$0.0156); the rest is planner calls.

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
       └─ Planner       one prompt, 2,552 tokens, every turn. The only paid step
                        turn 1 also writes the condition ledger, at no extra cost
            ├─ filter_catalog   238 -> N by column.  Free, exhaustive
            ├─ screen_out       N -> clear/flagged/insufficient.  Free, exhaustive
            ├─ search_plots     ranks what remains. One embedding, ~$0.0000002
            └─ read_synopses    <= 8 films, <= 6,000 chars each.  Free
```

Cheapest and most exhaustive first, so the token-heavy layer only ever sees
what survived the free ones.

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
- **Negations screened, not ranked.** `screen_out` scans every plot passage of
  every candidate (2,080 passages, 66 ms) rather than a top-ranked few, because
  embedding "nobody dies" returns the films where somebody does. On the full
  catalog: 76 clear, 149 flagged, 13 with too little plot text to screen. Its
  error is one-sided by design, and `scripts/check_screen.py` asserts that —
  13 known-death films must never come back clear.
- **The ledger rides turn 1.** The planner writes its typed conditions as
  content alongside its first tool call, so decomposition is visible in the
  trace without a dedicated planning turn, which would have doubled the paid
  turns of every request to produce the same text.
- **Guardrails in code, not prose**: the 45-minute floor and `weighted_rating`
  ordering are properties of the data; `MAX_RECOMMENDATIONS = 3`,
  `PREVIEW_FILMS = 15`, `MAX_SEARCH_RESULTS = 25`, `MAX_SYNOPSES = 8`,
  `MIN_SCREEN_TOKENS = 600`.

### A3. Prompt review  ← the remaining architecture item

### A3. Prompt review

Never reviewed whole; it grew one section per problem we hit. Now **2,552
tokens** on every turn plus 1,277 of tool schemas, against a brief that asks to
minimise context — and the four-layer rewrite added 547 of that.

The growth is arguably paid for: one avoided wrong-tool call (searching for a
negation) costs a full round at ~3,800 tokens, so preventing one covers the
increase seven times over. That is an argument, not a measurement.

- [ ] Read it end to end against each component — the Architecture tab now puts
      the prompt and every tool description side by side, served live
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
- [ ] Add a negation case that exercises all three screen buckets, and one
      where the right answer is a *flagged* film (an attempted killing, not a
      death) so `flagged` is not treated as `rejected`
- [ ] **`MAX_RECOMMENDATIONS = 3` is not being honoured.** A live production
      run on "a Pixar film where nobody dies, besides Toy Story" listed all
      **7** clear films. The screen returns a complete set and the model
      presented the set, which reads as reasonable but contradicts the cap.
      Decide which wins — the cap, or completeness when a layer is exhaustive
      — and say so in one place rather than two
- [ ] **"besides Toy Story" is ambiguous** and the model resolved it
      differently across two runs: once as the franchise (four labels, one of
      them the non-existent *Toy Story 4*), once as the single 1995 film. The
      second answer then recommended Toy Story 2 and 3. Both readings are
      defensible; pick one and make the tool description say it
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

Two things will bite you here.

**The exit code proves nothing.** `vercel --prod` returns 0 without necessarily
promoting. Verify by comparing what is actually served:

```bash
wc -c < public/index.html
vercel curl -sI https://movibot-gamma.vercel.app/ | grep -i content-length
```

**A change to a non-code file may not deploy at all.** Vercel reuses its build
cache when it sees no change worth rebuilding for, so editing only `TODO.md`,
`rag/DECISIONS.md` or another file the app *reads* can leave production serving
the old copy — with `x-vercel-cache: MISS`, so it does not look like a cache
problem. Use:

```bash
vercel --prod --yes --force
```

This matters more than it sounds: the TODO tab is served from `TODO.md`, so the
page whose whole purpose is to be current is exactly the one that can silently
go stale. Verify by content, not bytes, when the change is textual — `wc -c`
counts bytes and the file has multibyte characters, so the two numbers differ
legitimately.

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
