# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "safe for a toddler") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `docs/pitch-deck.html` for the full writeup and a live example of this failure).

## How it works

MoviBot is a **ReAct agent**: a planner model reasons, calls tools, observes what came back, and decides whether to go again or answer, bounded at `MAX_ROUNDS = 5` model turns per query. All four tools are available on every turn and none is mandatory. The system prompt encourages **cheapest and most exhaustive first**, so that each call hands the next a smaller candidate set and the token-heavy tool only ever sees what survived the free ones — but that is a preference, not a path: nothing in the code enforces an order, and the planner skips whatever a request does not need.

| Tool | Answers from | Narrows | Cost |
|---|---|---|---|
| `filter_catalog` | catalog columns | all → N | free, exhaustive |
| `screen_out` | a word scan | N → the half you keep | free, exhaustive |
| `search_plots` | meaning | N → a top handful | ~$0.0000002 |
| `read_synopses` | full text | ≤ 8 films | free, token-heavy |

`screen_out` is what answers the query in the pitch above. A negation cannot be retrieved for: embed *"no deaths"* and the top hits are the films where somebody dies, because that is what those plots say. So it is screened instead — every plot passage of every candidate is scanned, which is exhaustive over the word list in a way fixed-K retrieval cannot be. Its error is one-sided by design: *"dead heat"* over-excludes, it never under-excludes. A match makes a film **flagged**, not rejected, since a word list cannot tell an attempt from an outcome.

The same scan runs forwards. *"An animal that wears a hat"* is one small detail inside a 300-token passage, so ranking it returns films *about* animals while the film whose plot says the hat *"lands on Tod"* places nowhere — scanning for the word finds it, and keeping the matched half returns each film with the passage that proves it. When nothing matches, that is the answer: plot text records what happens rather than what things look like.

**Guardrails live in the data and tool code, never in the prompt** — the model cannot forget them and a bad plan cannot bypass them. Results are always ordered by `weighted_rating` rather than raw `vote_average`; `read_synopses` reads at most 8 films, truncated to 6,000 characters each, which is what bounds the cost of a turn; and `screen_out` refuses to certify a film with under 600 tokens of plot text, so absence of evidence is never reported as evidence.

`python scripts/check_screen.py` asserts the screen's safety property offline and for free.

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

The app is the current record of what works and what is open: the
**Architecture** tab documents the loop and every prompt verbatim, **Budget**
reports live spend against the cap, and **TODO** serves
[`TODO.md`](TODO.md) directly.

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
cache when it sees no change worth rebuilding for, so editing only `TODO.md`,
`rag/DECISIONS.md`, or another file the app *reads* can leave production
serving the old copy — with `x-vercel-cache: MISS`, so it does not look like a
cache problem. Use:

```bash
vercel --prod --yes --force --scope alan-agents-course
```

This matters more than it sounds: the TODO tab is served from `TODO.md`, so the
page whose whole purpose is to be current is exactly the one that can silently
go stale.

`.vercelignore` keeps 113 MB of raw Kaggle input, the course PDFs, and the
local `.env` files out of the upload. It deliberately does **not** exclude
`data_preprocessing/data_ready/`, which the agent reads at runtime.
