## What is left

Three days out. Almost everything remaining needs live model calls, so it is
one budgeted round rather than a queue. Spend to date is on the Budget tab.

### Blocked on one paid round  💰 ~$0.10 of $12.91 remaining

Nothing below has ever met a live model. The last two queries tried by hand
each found a real bug that no expectation predicted, which is the argument for
running the whole set rather than spot-checking.

- [ ] **Run all 13 test cases** and compare each against what it says should
      happen. The three traps: "starring Tom Hanks" must refuse rather than
      answer Toy Story from pretraining; "besides Frozen and Moana" must become
      a real filter; "a Disney movie in Hindi" must still surface Dangal at 140
      votes. Watch for over-refusal on "a nice comedy" and invented post-2017
      titles
- [ ] **Six behaviour changes are unverified.** C05 routing (negations reach
      different tools by evidence, not by the word "no"), C06 escalation (a
      negative no word list settles must reach search or reading), C08 ceiling
      (a request for everything returns at most 3 and says so), the weak-search
      rule (`weak_match` under 0.40 forces a re-query), the answer-shape rule
      (a rejected film gets no heading), the evidence rule (no event described
      that was not in a passage the model was shown)
- [ ] **C01 — `prompt_examples` are missing required fields.** The spec requires
      `full_response` and `steps` on every example. Capture both, for both
      examples, from a real run. `steps[].module` must match a real trace
      exactly
- [ ] **C02 — both `status` fields still say the response "will be captured
      once the LLM endpoint is enabled".** It has been live for weeks. Replace
      with a capture note, or drop the field once `full_response` carries the
      evidence. The rest of the user-visible surface swept clean 2026-08-20
- [ ] **C03 — "empowering" has no evidence step.** `prompt_examples[0]` settles
      princess and the exclusions but not the adjective; `require_synopsis=true`
      only proves text exists. Give it its own ledger line and a tool that can
      settle it, and make the rationale cite that evidence
- [ ] **G02** — confirm on a real trace that every model call is there in
      order. Planner and Observer carry `usage`; `PlotSearch` now records a
      `model_call` naming the embedding deployment, its input and its
      dimensions, so the one known gap is closed pending verification
- [ ] **Verify the paired scan.** `and_words` requires two word lists to land
      in the same passage. Confirm the planner reaches for it on "a cat that
      wears a hat" rather than scanning one list and rationalising over the
      near-misses
- [ ] **Verify the Observer end to end.** Never run. Confirm it returns usable
      JSON, that its verdicts are sane, that the quote check rejects a
      paraphrase rather than passing it, and that the planner cites the
      findings instead of asserting past them. Measure the real token saving
      against the ~5,000-token synopsis payload it replaces
- [ ] **G05** — a high-complexity bounded request finishes well under 300 s
- [ ] **Read `cached_tokens` off the first run.** Captured and shown per run
      since 2026-08-20 but never yet observed. It decides whether trimming the
      prompt saves 500 tokens or 2,000, so it comes before A3

### Free, but needs a decision

- [x] ~~"besides Toy Story" is ambiguous~~ — settled 2026-08-20 in favour of
      what the code already does: exclusion is **exact title, never a
      franchise**. "Toy Story" drops the 1995 film and leaves 2 and 3 in scope;
      "The Jungle Book" drops two only because both remakes share that exact
      title. Written into the `exclude_titles` schema, along with an
      instruction never to invent a title to exclude

### After the round

- [ ] **A3 — trim the prompt.** See below; it is the weakest part of the
      project and the brief grades it. Deliberately sequenced last: six
      unverified changes are in there now, so a regression after a trim would
      be unattributable
- [ ] **G10 — at handover**, record the commit SHA and confirm the live GUI and
      API match the reviewed source

### Decided against

- **Instrumenting the ledger against the trace.** Would tell us whether the
  loop earns its keep over a static executor. Good question, does not fit in
  three days.
- **Sweeping for other queries the abstract phrasing loses.** The two we found
  were found by hand; a systematic sweep needs a corpus of queries we do not
  have.

---

## A3. The prompt is the weakest part of the project

**It is bad, and it is bad in a way the brief specifically marks down.**
Requirement 1 of the assignment is *"Build the agent in an optimized way: avoid
unnecessary LLM calls, **minimize prompt/context size (only what's needed)**."*
That is graded, not a nicety, and the prompt has moved the wrong way all week:

| | system prompt | + schemas | per turn |
|---|---|---|---|
| start of review, 2026-08-19 | 3,150 | 1,344 | 4,494 |
| now, 2026-08-20 | **4,197** | **1,670** | **5,867** |

**+33% in two days.** Every addition had a reproduction behind it, which is the
defence, and it is not good enough: the same lessons are now taught over and
over in different words, and that is a cause of failures rather than a cure.

Counted by hand:

- **"do not claim what you did not verify" — 6 separate places**
- **"screen_out tests words, not events" — 4 places**
- **"cheapest and most exhaustive first" — 4 places**
- **the 3-film ceiling — 4 places**

Redundancy is not the worst of it. Two restatements were in outright
**contradiction** until 2026-08-20 — "where you relied on the approximate
layers, say so" against "if you are writing *but I did not verify*, cut it" —
and the model resolved that by satisfying one and breaking the other. That is
exactly the class of failure that kept turning up in review: a rejected film
under an "Also possible" heading, an apologetic second recommendation. One
prompt saying the same thing six ways is not emphasis; it is six chances to
disagree with itself.

`HOW TO WORK` alone is **1,336 tokens, a third of the whole prompt**, and holds
two overlapping structures: the condition-type table and the numbered layer
order encode much the same routing knowledge twice.

- [ ] Merge the four repeated lessons into one statement each; target ~3,400
- [ ] Collapse the two structures in `HOW TO WORK` into one
- [ ] Re-run the 13 cases against the round's baseline, so any regression is
      attributable to the trim
- [ ] Put the measured before/after in the write-up. "We cut the prompt 33% with
      no regression on 13 cases" is a requirement-1 answer; "it grew because
      every addition was justified" is not

---

## Where it stands

```
POST /api/execute        returns exactly status / error / response / steps
  └─ loop.py             up to MAX_ROUNDS = 5 model turns; the last gets no
                         tools, so it must answer rather than ask
       └─ Planner        one prompt, 4,197 tokens + 1,670 of schemas, every
                         turn. The only paid step. Turn 1 also writes the
                         condition ledger, at no extra cost
            ├─ filter_catalog   238 -> N by column.  Free, exhaustive
            ├─ screen_out       N -> the half you keep.  Free, exhaustive
            ├─ search_plots     ranks what remains. One embedding, ~$0.0000002
            └─ read_synopses    <= 8 films, <= 6,000 chars each.  Free
```

The prompt encourages cheapest-and-most-exhaustive first, so the token-heavy
tool only ever sees what survived the free ones — a preference, not a path:
nothing in the code enforces the order.

- **Working set in Python.** `filter_catalog` records every matching id in a
  request-scoped `ToolContext`; search and read scope to it automatically.
  Nothing is passed between tools, and no match is lost to a display cap.
- **Films addressed by name.** `Title (Year)` is unique across all 238, so no
  id enters the prompt. A filter response is 373 tokens where it was 13,791.
- **Four corpora, 3,159 passages, 1536-dim, scored in memory.** Search covers
  the two plot-bearing ones by default.
- **Data**: 238 films × 26 columns from committed CSVs (2 ms); 234 readable in
  full, 4 overview-only; no database of any kind.
- **The scan runs both ways.** `screen_out` reads every plot passage of every
  candidate and keeps either half. Backwards for an absence — embedding "nobody
  dies" returns the films where somebody does; on the full catalog, 76 clear,
  149 flagged, 13 too thin to screen. Forwards for a concrete presence, because
  ranking "an animal that wears a hat" scores whole passages and returns films
  about animals.
- **The ledger rides turn 1**, as content alongside the first tool call, so
  decomposition is visible without a dedicated planning turn.
- **Guardrails in code, not prose**: the 45-minute floor and `weighted_rating`
  ordering are properties of the data; `MAX_RECOMMENDATIONS = 3`,
  `PREVIEW_FILMS = 15`, `MAX_SEARCH_RESULTS = 25`, `MAX_SYNOPSES = 8`,
  `MIN_SCREEN_TOKENS = 600`, `WEAK_MATCH_SIMILARITY = 0.40`.
- **Two check scripts, free and offline.** `scripts/check_screen.py` asserts the
  screen's one-sided error, title exclusion, the forward scan and language
  resolution. `scripts/check_gates.py` asserts module-name consistency, the
  model ids, and that every displayed count still derives from the build.

---

## Decisions on record

Recorded so they are not relitigated.

- **A cheap router in front of the planner** — rejected on arithmetic. The
  tenant offers one chat model and one embedding model, so a "cheaper subagent"
  can only mean fewer tokens to the same model. A router saves 4,608 tokens per
  refusal and costs 425 on everything else: +4.4% at the test bed's 27% refusal
  rate, +0.3% at a plausible 10%. Prefix caching erodes it further.
- **A separate Observer prompt** — rejected. `Observe` is
  `messages.append({"role": "tool", ...})` and `Stop?` is `if not tool_calls`;
  neither is inference, so neither can have a prompt. Native tool calling puts
  observation in the next Reason turn.
- **Plan-and-execute instead of ReAct** — rejected for this deadline, not on
  principle. The critique is fair: the ledger is written and then not executed.
  But the death screen returns 149 flagged against a `MAX_SYNOPSES = 8` cap, so
  which films to read depends on what the screen returned, not on the query. A
  static plan needs branches, and then it is the loop again.
- **A `meta` key on `/api/execute`** — rejected. The spec says the response
  "must match exactly these top-level fields", not "must include", so a fifth
  key is as non-conformant as three extra ones.
- **Qualifying toddler-safety rather than dropping it** — rejected. The catalog
  has no age rating of any kind, and of the 27 animated films the old path
  called clear on death, 14 are flagged on the `scary` vocabulary, including
  both it led with. A child-safety judgement the data cannot support is the
  worst place to hedge.

---

## Closed

- **G01** — `/api/execute` returned seven top-level fields against a spec that
  fixes four exactly. `plan` was byte-identical to `steps[0].response.content`
  and `narrowing` was the accumulation of the `scope` already on every tool
  step, so two of the three were duplicates; token usage moved onto the Planner
  step that spent it. *2026-08-20*
- **G03 / G04 / G09** — module names, model ids and every displayed count, now
  asserted by `scripts/check_gates.py`. It immediately found the Data tab
  labelling the MPST count "with a plot synopsis", the same framing that had
  told the system prompt it could not verify 79 films it can read in full.
  *2026-08-20*
- **C04** — the toddler example is gone; replaced by "An animated film where
  nobody dies", same tool path and same verified figures. *2026-08-20*
- **C05–C14** — ten workbook defects: routing negations by evidence rather than
  grammar, stopping the screen overclaiming, the hard recommendation ceiling,
  corrected coverage and `tool_order` in `agent_info`, and the removal of the
  reference-project mentions. *2026-08-19*
- **V1 revision** — 11 stale cases rewritten as 13 verified ones, three of them
  covering behaviour that had no test at all. *2026-08-20*
- The language filter reached 2 of 21 languages; the catalog stores endonyms, so
  `languages=['Hindi']` matched zero while reporting itself as applied. *2026-08-20*
- The system prompt claimed 159 of 238 films had a plot synopsis "and for the
  rest you have only the short overview", so the model hedged about 79 films it
  can read in full. *2026-08-20*
- The diagram said "at most 5 model turns" and "the only metered step", naming a
  number and a cost without saying what a turn was or why that box differed.
  *2026-08-20*

Deployment steps live in the README. Costs live on the Budget tab.
