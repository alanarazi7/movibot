# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "nothing where the hero is betrayed") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `docs/pitch-deck.html` for the full writeup and a live example of this failure).

## How it works

A **Planner** model runs in a bounded loop: it reads the request, breaks it into conditions, and chooses which tools to call, bounded at `MAX_ROUNDS = 5` planner rounds and `MAX_TOTAL_LLM_CALLS = 16` model calls per query. Every tool is available on every round and none is mandatory; nothing in the code sequences them.

A round that requests no tool is the planner's **attempt** at an answer, not the answer. `agent/loop.py` checks it first, and a failure costs one correction round: it may not name a film verification did not accept, omit one it did, name films after a search without verifying them, exceed the recommendation ceiling, or offer a follow-up this stateless API cannot honour. The planner decides the route; it does not decide whether a film it recommends was checked.

| Tool | Answers from | Narrows | Cost |
|---|---|---|---|
| `filter_catalog` | catalog columns | all → N | free, exhaustive |
| `screen_out` | a word scan | N → the half you keep | free, exhaustive |
| `build_shortlist` | meaning, every condition at once | N → a fused ranking | one embedding per condition |
| `read_synopses` | full text, one question | ≤ 8 films | free; its output is read by the Observer |
| `verify_candidates` | full text, every condition | → the accepted list | one model call per film |

**Every story condition gets its own search, and the rankings are fused before anything expensive happens.** This is the difference between finding an answer and finding a plausible one. Searching a single condition and reading its top hits quietly assumes the rest hold in whatever came back — ask for *a princess and ice* and the film ranking first on "a princess" gets read, while **Frozen, which ranks tenth on princess and first on ice, is never seen at all**. So `build_shortlist` searches each condition separately and orders films by how many conditions they placed for, then by average rank. Coverage has to dominate: average rank alone makes a film ranked 1st, 1st and absent beat one ranked 10th, 10th and 10th, which is the greedy answer arriving by arithmetic.

`search_plots` is not offered to the model. It is the single-condition primitive `build_shortlist` calls once per condition, and it stays out of the model's reach because a tool that can be called greedily is a tool the prompt has to talk the model out of using.

**Verification runs one film at a time against every condition at once.** `verify_candidates` walks the fused shortlist best-first, giving each film's plot text and the whole list of requirements to one model call, and returns a verdict per condition — `yes`, `no`, or `unclear` — with the sentence that decides it quoted verbatim. A film is accepted only when every condition says yes, and **the accepted list is assembled in Python**, so the count an answer states is a count of a list rather than a claim the model makes about its own reasoning. It stops at three accepted films or ten checked, whichever comes first; running out having accepted nothing is a real answer and is reported as one.

`screen_out` is what answers the query in the pitch above. A negation cannot be retrieved for: embed *"no deaths"* and the top hits are the films where somebody dies, because that is what those plots say. So it is screened instead — every plot passage of every candidate is scanned, which is exhaustive over the word list in a way fixed-K retrieval cannot be. Its error is one-sided by design: *"dead heat"* over-excludes, it never under-excludes. A match makes a film **flagged**, not rejected, since a word list cannot tell an attempt from an outcome.

The same scan runs forwards. *"An animal that wears a hat"* is one small detail inside a 300-token passage, so ranking it returns films *about* animals while the film whose plot says the hat *"lands on Tod"* places nowhere — scanning for the word finds it, and keeping the matched half returns each film with the passage that proves it. When nothing matches, that is the answer: plot text records what happens rather than what things look like.

**Guardrails live in the data and tool code, never in the prompt** — the model cannot forget them and a bad plan cannot bypass them. Results are always ordered by `weighted_rating` rather than raw `vote_average`; `read_synopses` reads at most 8 films, truncated to 6,000 characters each, which is what bounds the cost of a turn; and `screen_out` refuses to certify a film with under 600 tokens of plot text, so absence of evidence is never reported as evidence.

**Plot text is read by separate modules, never by the planner.** A synopsis read is ~5,000 tokens; left in the planner's context it would be re-sent every turn. The **Verifier** reads it instead — one film, every condition, its own prompt with no tool schemas and no routing rules — and the **Observer** does the same for the narrower case where the planner has one specific question about films it has already named. Both check every quote to be a literal substring of the source and discard it otherwise, downgrading an unsupported `yes` to `unclear`, so what reaches the planner is evidence rather than a summary of it. The full text stays in the `steps` trace, which is why what a reviewer can inspect and what the model must carry are deliberately different things.

`python scripts/check_screen.py` and `python scripts/check_gates.py` assert the screen's safety property, the language filter, the `/api/execute` contract against eleven malformed inputs, the total call cap against an adversary that never stops calling tools, the fusion ordering that keeps greed from winning, and every displayed count — offline and for free.

**The app's [Architecture tab](https://movibot-gamma.vercel.app) is the full account** — the loop, every tool description, and every prompt verbatim, served live from the source. The diagram alone is at `GET /api/model_architecture`.

## Data

Three sources, prepared offline into `data_preprocessing/data_ready/`:
a Disney/Pixar catalog from Kaggle, MPST plot synopses, and a Wikipedia cache
scraped once. Raw Kaggle downloads (`data_preprocessing/data_full/`) are
gitignored; everything in `data_ready/` is committed, so the repo runs without
a rebuild.

The catalog is deliberately narrowed to Disney and Pixar, which keeps it in
family territory and makes the demo coherent. That is a demo constraint rather
than a product decision — the assignment caps stored data at 50 MB and the full
multi-studio catalog does not fit. `prepare_movibot_data.py --all-studios`
produces the full catalog from the same pipeline.

**Every current figure — counts, coverage, per-film records, the passage index —
is in the app's [Data](https://movibot-gamma.vercel.app) and Retrieval tabs, generated from the shipped
artifacts.** They are not repeated here, so there is one source of truth rather
than two that can drift apart. Retrieval parameters and their rationale live in
[`rag/DECISIONS.md`](rag/DECISIONS.md), also served live on the Retrieval tab.

## Status

Live and answering real queries at **[movibot-gamma.vercel.app](https://movibot-gamma.vercel.app)**.

The app is the current record of how it works: the **Architecture** tab
documents the loop and every prompt verbatim, and **Budget** reports live spend
against the cap. [`TODO.md`](TODO.md) keeps the project log — decisions, the
numbers that settled them, and what was fixed — in the repo rather than in the
GUI.

## API

- `GET /api/team_info` — team roster.
- `GET /api/agent_info` — description, purpose, prompt template, worked examples.
- `GET /api/model_architecture` — architecture diagram (PNG).
- `POST /api/execute` — `{"prompt": "..."}` in; `{"status", "error", "response", "steps"}` out.

## Local setup

```bash
pip install -r requirements.txt
python app.py           # http://localhost:5000
```

The catalog reads from committed CSVs and needs no credentials. Planner turns and query embedding both call LLMod.ai, so copy `.env.example` → `.env` and fill in the key; `python scripts/check_credentials.py` reports what is missing without spending anything.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | LLMod.ai key. Required for planner turns and query embedding. |
| `OPENAI_BASE_URL` | — | LLMod.ai endpoint. Required alongside the key. |
| `MOVIBOT_MODEL` | `MB5R2CF-azure/gpt-5.4-mini` | Overrides the planner model id. |

## Retrieval

Chunking, embedding, storage, search and the ingest command all live in `rag/`. Parameters are in `rag/config.py`; the reasoning, including where we deviate from the course defaults and why, is in [`rag/DECISIONS.md`](rag/DECISIONS.md).

```bash
python -m rag.ingest --dry-run   # free: how many passages, what it would cost
python -m rag.ingest             # 💰 ~$0.007 to embed the whole corpus
```

## Docs

- `docs/course-assignment-instructions.pdf` — the course's official assignment spec (API contract, deployment, budget, deadline).
- `docs/team-idea-proposal-and-data-sources.pdf` — the team's own idea writeup.
- `docs/pitch-deck.pptx` / `docs/pitch-deck.html` — the pitch deck (problem, architecture, demo).

## Deployment

Vercel, Python serverless (`vercel.json`).

- **Live URL:** https://movibot-gamma.vercel.app
- **GitHub Repo:** https://github.com/alanarazi7/movibot
- **Vercel Project Dashboard:** https://vercel.com/alan-agents-course/movibot

**Deploys do not happen on `git push`.** This project is not Git-connected on
Vercel; a push updates GitHub only. Production changes require:

```bash
vercel --prod --yes --scope alan-agents-course
```

`--scope alan-agents-course` is required. Without it the deploy fails with
`Not authorized` even when `vercel whoami` shows you logged in: the project
sits under a team, and the error names auth rather than scope.

Two further things will bite you.

**The exit code proves nothing.** `vercel --prod` returns 0 without necessarily
promoting. Verify against what is actually served:

```bash
curl -s https://movibot-gamma.vercel.app/ | grep -c "some string you just added"
```

Check by content, not by byte count — `wc -c` counts bytes and these files
contain multibyte characters, so local and served lengths differ legitimately.

**A change to a non-code file may not deploy at all.** Vercel reuses its build
cache when it sees no change worth rebuilding for, so editing only
`rag/DECISIONS.md`, `agent_info.json`, or another file the app *reads* can
leave production serving the old copy — with `x-vercel-cache: MISS`, so it does
not look like a cache problem. Use:

```bash
vercel --prod --yes --force --scope alan-agents-course
```

`.vercelignore` keeps 113 MB of raw Kaggle input, the course PDFs, and the
local `.env` files out of the upload. It deliberately does **not** exclude
`data_preprocessing/data_ready/`, which the agent reads at runtime.
