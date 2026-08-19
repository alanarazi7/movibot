## Architecture

The shape we agreed: decompose a request into conditions, exhaust the
structured ones, then spend money only on what is left.

### Where it stands

```
POST /api/execute
  └─ loop.py            up to MAX_ROUNDS = 5 model turns; last turn gets no
                        tools, so it must answer rather than ask
       └─ Planner       one prompt, 3,150 tokens, every turn. The only paid step
                        turn 1 also writes the condition ledger, at no extra cost
            ├─ filter_catalog   238 -> N by column.  Free, exhaustive
            ├─ screen_out       N -> clear/flagged/insufficient.  Free, exhaustive
            ├─ search_plots     ranks what remains. One embedding, ~$0.0000002
            └─ read_synopses    <= 8 films, <= 6,000 chars each.  Free
```

The prompt encourages cheapest-and-most-exhaustive first, so the token-heavy
tool only ever sees what survived the free ones — a preference, not a path:
nothing in the code enforces the order.

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

Never reviewed whole; it grew one section per problem we hit. Now **3,150
tokens** on every turn plus 1,344 of tool schemas — 4,494 in all — against a
brief that asks to minimise context. The QA-workbook rewrites (routing by
evidence, the claim rules, the output ceiling) added roughly 600 of that, and
were correctness fixes rather than padding, but the section has never been read
end to end for redundancy.

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
- [ ] **Confirm the ceiling holds.** `MAX_RECOMMENDATIONS = 3` is now a hard
      ceiling that no request raises, including one asking for everything; such
      a request is told the answer is not complete and given the true match
      count where the exhaustive tools settled the set. Decided; the production
      run that listed all 7 clear films predates it. Verify against the ceiling
      test in the test bed
- [ ] **The Hindi case expectation is now wrong.** It says "only 2 Hindi films
      exist". Three do: Dangal, Khoobsurat and Million Dollar Arm, the last
      English-language with Hindi dialogue. The filter used to match zero of
      them because the catalog stores endonyms; fixed 2026-08-20, and Dangal
      leads on rating as the case expects. Update the count and decide whether
      Million Dollar Arm belongs in the answer
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

## QA workbook — the four remaining

A pre-submission review (14 confirmed defects) was worked through on 2026-08-19.
Ten are closed: C05–C07 route negations by evidence rather than grammar and stop
the screen overclaiming, C08–C09 made `MAX_RECOMMENDATIONS` a hard ceiling with
no listing mode, C10–C13 corrected coverage, cast/crew, `keywords` and
`tool_order` in `agent_info`, and C14 removed the reference-project mentions and
the superseded pipeline doc.

The four below all live in `agent_info.json` → `prompt_examples`, which is a
**graded** endpoint. None can be finished without live runs, so this is the
budgeted round. Costs live on the Budget tab.

### C01 — examples miss `full_response` and `steps`  💰

The course spec requires both on every prompt example. Ours carry `prompt`,
`expected_tool_path`, `status`, `verified_tool_output`. Trivially checkable by a
grader, and a format defect rather than a documentation preference.

- [ ] Run the representative prompts and store each exact final `response` as
      `full_response`, plus the full returned `steps`
- [ ] Keep the diagnostic fields only if they still earn their place; do not let
      them stand in for the required ones
- [ ] `steps[].module` must match `/api/model_architecture` and a real
      `/api/execute` trace exactly

### C02 — examples advertise unfinished work

Both `status` fields say the response "will be captured ... once the LLM
endpoint is enabled". It has been enabled for weeks; the app answers production
queries and reports its own spend. A graded endpoint currently tells a reviewer
the project is half-built.

- [ ] Replace with a factual note (captured from build `<sha>` on `<date>`) or
      drop the field once `full_response` carries the evidence
- [ ] Sweep every user-visible surface for "will be", "once enabled", "stub",
      "TODO" and similar future-work language

### C03 — "empowering" has no evidence step

`prompt_examples[0]` asks for the *best empowering princess movie besides Frozen
and Moana*, and the path is `CatalogFilter(keywords=['princess','royalty'],
exclude_titles=[...], require_synopsis=true)`. Princess and the exclusions are
settled; **empowering is not**. `require_synopsis=true` only guarantees text
exists, and `agent_info` now documents `keywords` as a broad topical filter and
explicitly not proof of a theme — so by our own description this path cannot
support the adjective.

- [ ] Give "empowering" its own line in the condition ledger and a tool that can
      settle it (`search_plots`, or `read_synopses` on the shortlist)
- [ ] The final rationale must cite that evidence rather than inferring
      empowerment from princess/royalty metadata

### C04 — "safe for a toddler" is not established

`prompt_examples[1]` checks `genres=['Animation']` and a death screen. Animation
plus the absence of a death-vocabulary hit is not toddler suitability: peril,
frightening imagery, intensity and emotional weight are independent conditions.
The catalog stores no age rating, and the studio filter is membership rather
than content rating, so a Disney label can carry a PG-13 title.

**Decision needed before spending.** Either verify what can be verified and
qualify the rest, or decline the broader claim outright and say why. Whichever
is chosen has to hold in the prompt, the example and the answer text alike.

- [ ] Decide, then make the ledger carry both conditions separately
- [ ] The answer must never assert general toddler safety on the strength of the
      death screen alone

### Also waiting on this round

Three rewrites changed behaviour and have never met a live model:

- [ ] **C05 routing** — "not Pixar", "no musicals", "nothing scary",
      "doesn't centre on romance" should reach *different* tools, chosen by
      evidence rather than by the word "no"
- [ ] **C06 escalation** — a negative that no word list can settle should reach
      search or reading rather than dead-ending
- [ ] **C08 ceiling** — a request for everything must return at most
      `MAX_RECOMMENDATIONS` films and say it is not the complete list

## Architecture questions settled 2026-08-20

Raised while reviewing the app; recorded so they are not relitigated.

- **A cheap router in front of the planner** -- rejected on arithmetic. The
  tenant offers one chat model and one embedding model, so a "cheaper subagent"
  can only mean fewer tokens to the same model. A router saves 4,608 tokens per
  refusal and costs 425 on everything else: +4.4% at the test bed's 27% refusal
  rate, +0.3% at a plausible 10%. Prefix caching would erode it further.
- **A separate Observer prompt** -- rejected. `Observe` is
  `messages.append({"role": "tool", ...})` and `Stop?` is `if not tool_calls`;
  neither is inference, so neither can have a prompt. Native tool calling puts
  observation in the next Reason turn. A dedicated observer call would double
  the paid turns to restate what that turn already derives.
- **Plan-and-execute instead of ReAct** -- rejected for this deadline, not on
  principle. The critique is fair: the ledger is written and then not executed.
  But the death screen returns 149 flagged against a `MAX_SYNOPSES = 8` cap, so
  which films to read is a function of what the screen returned, not of the
  query. A static plan needs branches, and then it is the loop again. A rewrite
  also moves every module name, which G03 checks across three graded endpoints.

- [ ] **Verify the weak-search rule against a live model.** `search_plots` now
      returns `weak_match` under 0.40 similarity, and the prompt says a weak
      search must be re-run rather than narrated. Confirm on "a film where
      someone pretends to love another to seize power" that it re-queries and
      reaches Frozen (2013). Add it to the test bed as a case in its own right
- [ ] **Re-run the power/deception query.** The concrete phrasing now happens on
      round 1 and Frozen leads, verified 2026-08-20. Still to confirm: no
      rejected film gets a heading, and no plot event is described that was not
      in a passage the model was shown
- [ ] **A3 is now urgent, not deferred.** The prompt has gone 3,150 -> 4,008
      tokens across this review, +27%, and 5,678 per turn with schemas. Every
      addition had a reproduction behind it, but several are near-duplicates:
      three separate passages now say some version of "do not claim what you did
      not verify". Read it end to end and merge them
- [ ] **Sweep for other queries the abstract phrasing loses.** This was found by
      hand. The same failure is invisible on any request whose weak answer
      happens to look plausible
- [ ] **Instrument the ledger against the trace.** `plan` holds the typed
      conditions and `steps` holds the tools that actually ran; nothing compares
      them. If the model always follows its own ledger, a static executor would
      do, and that is a finding. If it deviates usefully -- escalating on a
      flag, widening an empty filter -- each deviation is evidence the loop
      earns its keep. Free to build, answered by the round already planned
- [ ] **Read `cached_tokens` off the first paid run.** Now captured and shown
      per run. It decides whether trimming the prompt saves 500 tokens or 2,000,
      so it comes before A3
- [ ] **Relabel the diagram.** `Reason` is marked "the only metered step",
      which says paid, not "the only inference". Four equal boxes read as four
      prompted components. Say what each box is: model call / local Python /
      appended to context / did the turn request a tool

## Release gates

From the same QA review. Not defects; verification that would violate the course
contract if it failed. The free ones can run before any spending.

### G01 — `/api/execute` disagrees with itself  ← free, decision needed

The spec wants `status`, `error`, `response`, `steps`. The error paths in
`app.py` return exactly those four. The success path returns `loop.execute()`
straight through, which is **seven**: it adds `plan`, `narrowing` and `budget`.
Those three drive the ledger, narrowing and budget panels in the GUI, so they
earn their place — but a strict reading of the contract fails on them.

- [ ] Decide: nest the extras (under `steps`, or a `meta` key) for a strict
      reading, or keep them flat and accept a lenient one
- [ ] Whichever is chosen, make the success and error paths agree

### Free to run now

- [ ] **G03** — module names identical across `/api/model_architecture`,
      `/api/agent_info` and `/api/execute` `steps[].module`, by exact string
- [ ] **G04** — deployed text and embedding model ids are the course-provided
      deployments; record them in a non-secret diagnostic view
- [ ] **G09** — every displayed count derives from the current build

### Needs the paid round

- [ ] **G02** — every model call appears in `steps`, in order, with module,
      prompt and response. No unlogged final synthesis call
- [ ] **G05** — a high-complexity bounded request finishes well under 300 s

### At handover

- [ ] **G10** — record the commit SHA and confirm the live GUI and API match the
      reviewed source

Deployment steps live in the README. Costs live on the Budget tab.
