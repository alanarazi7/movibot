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

Deployment steps live in the README. Costs live on the Budget tab.
