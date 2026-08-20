## What is left

Two working items and one handover check. Everything else is closed.
Spend to date $0.33 of $13; the whole verification round cost 4 cents.

### G10 — the handover check

Deploys are **not** git-connected here: `git push` updates GitHub and nothing
else, and `vercel --prod` returns 0 without necessarily promoting. So the
repository and the live site can disagree silently, and every claim in the
write-up is about the live site.

It goes last because it is a statement about the final state. Running it before
the last commit verifies a build that is about to be replaced.

- [ ] Record the commit SHA that is actually serving
- [ ] Confirm the served diagram, prompt and counts match the repo — the two
      check scripts plus a hash comparison of `/api/model_architecture`
- [ ] Confirm `/api/execute`, `/api/agent_info`, `/api/team_info` and
      `/api/model_architecture` all answer on the live URL

### Decided against

- **Instrumenting the ledger against the trace.** Would tell us whether the
  loop earns its keep over a static executor. Good question, does not fit.
- **Sweeping for other queries the abstract phrasing loses.** The ones we found
  were found by hand; a systematic sweep needs a corpus of queries we do not
  have.
- **Chasing the animal-in-a-hat case further.** Three mechanism fixes narrowed
  it from 51 candidates to 2 and it still names a near-miss rather than saying
  nothing qualifies. Recorded in the test bed as the weakest result rather than
  hidden.

---

## A3. Prompt trimmed 43%  ✅ 2026-08-20

| | system prompt | + schemas | per turn |
|---|---|---|---|
| start of review, 2026-08-19 | 3,150 | 1,344 | 4,494 |
| peak, 2026-08-20 | 4,197 | 1,670 | 5,867 |
| **after the trim** | **2,374** | 2,067 | **4,441** |

**−43% on the prompt.** Requirement 1 of the brief asks to minimize prompt
size, so this is a graded number, not housekeeping.

What was actually wrong was not length but repetition. The same four lessons
were each stated three to six times — "do not claim what you did not verify"
in six places, "screen_out tests words not events" in four — and two of those
restatements had been in outright contradiction until earlier the same day.
One prompt saying the same thing six ways is not emphasis; it is six chances
to disagree with itself.

The rewrite states the governing rule once, at the top, and makes everything
else an application of it. `HOW TO WORK` held two overlapping structures — a
condition-type table and a numbered layer list encoding the same routing
knowledge twice — now merged into one table that carries the operational
detail with the routing. Rules that belong at the point of use moved into the
tool schemas, which is why schemas grew while the prompt shrank.

**The trim caused one regression, and testing caught it.** The
pretend-to-love case had returned Frozen; afterwards it returned Princess
Diaries 2. The dropped worked example was doing real work: without it the
model wrote "a character pretends to love another person, gains power or a
throne", which scored 0.4063 — just above the old 0.40 `weak_match`
threshold, so nothing warned, and Frozen was not in the top five.

Both halves are fixed. The threshold is 0.42, which the seven observed
queries still separate cleanly, and a compact three-line version of the
example is back with the middle case as the trap: a query can read as
concrete and still name an abstraction. Re-run returns Frozen in one tool
call.

All thirteen cases were re-run against the trimmed prompt on 2026-08-20; the
expectations in the test bed are written from that run.

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

- **C01 / C02 / C03 — the graded endpoint is compliant.** Both prompt examples
  now carry `full_response` and `steps` captured verbatim from a real run,
  stamped with the commit that produced them; `scripts/capture_examples.py`
  regenerates them so they can never be hand-written again. The status fields
  promising a capture "once the LLM endpoint is enabled" are gone, and the
  wider sweep for future-work language is clean. C03 closed with them: the
  empowering example now runs CatalogFilter -> PlotSearch -> SynopsisReader ->
  Observer and cites the plot text rather than inferring a theme from
  keywords. *2026-08-20*
- **`verified_tool_output` dropped.** A hand-written narrative of what the
  tools returned, sitting beside the captured steps, and it had already
  drifted -- claiming 13 matches next to a step showing 11, because the model
  picks its own filter arguments. The steps hold every number verbatim.
  *2026-08-20*
- **The full test bed ran, 13/13.** Eleven clean; two honest but imperfect —
  the animal-in-a-hat case names a near-miss, and the pretend-to-love case
  reaches Frozen on a concrete rewrite and reports "nothing supported" on an
  abstract one. The run found two live defects: the recommendation ceiling was
  not holding (six films listed, with the "at most three" line quoted
  underneath), and a weak search was being offered back to the user rather than
  re-run. *2026-08-20*
- **The ceiling is enforced in code**, not prose. `catalog.labels_in()` counts
  the films named in an answer and the loop rejects an over-long one at the
  cost of one turn. Two rounds of prompt wording had failed first. *2026-08-20*
- **C05, C06, C08 and the weak-search, answer-shape and evidence rules** all
  verified against a live model. *2026-08-20*
- **The Observer, the paired scan and G02** verified: the Observer returns
  usable JSON and its substring check rejected a quote for real; `and_words`
  took the cat-and-hat scan from 27 films to 2; Planner and Observer steps
  carry `usage` and `PlotSearch` names its embedding deployment. *2026-08-20*
- **G05** — the slowest of thirteen cases took 14.1s against a 300s limit.
  *2026-08-20*
- **`cached_tokens` read at last**: 72–83% of prompt tokens come back cached,
  which is why the A3 trim is worth less in money than it looked and still
  worth doing for the graded requirement. *2026-08-20*

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
