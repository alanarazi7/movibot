# TODO

Open items only. Due **2026-08-23**.

Spending needs explicit go-ahead. **Spent so far: ~$0.018 of $13** — the corpus
embedded once (~$0.0156), plus a handful of planner calls.

The app is live at [movibot-gamma.vercel.app](https://movibot-gamma.vercel.app)
and **answers real queries in production**. All four required endpoints work,
semantic search works, and the passage index covers every film in the catalog.

---

## Credentials


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
