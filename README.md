# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "safe for a toddler") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `presentation.html` for the full writeup and a live example of this failure).

## How it works

MoviBot runs a ReAct loop (`Reason -> Act -> Observe -> Stop?`) over four tools, narrowing from cheap structured filtering down to expensive per-title verification:

| Module | Role |
|---|---|
| `Reasoner` | Plans the next action each loop iteration |
| `CatalogFilter` | Structured filter (year, studio, genre, runtime) — Supabase |
| `PlotSearch` | Thematic/character semantic search over plot overviews — Pinecone |
| `SceneSearch` | Per-candidate check for risky events (deaths, scary scenes) — live Wikipedia plot fetch + reasoning |
| `ExternalContext` | Per-candidate check for tone/subtext not in the plot — live Wikipedia fetch + reasoning |
| `Synthesizer` | Composes the final, evidence-backed answer |

See `assets/architecture.png` (served at `GET /api/model_architecture`) — module names there match the `steps` trace exactly.

**Data**: Kaggle "The Movies Dataset," downsampled to ~5K movies. Structured fields live in Supabase; `overview` text is embedded into Pinecone; Wikipedia is fetched live per-candidate rather than pre-indexed.

## Status

**Current state: skeleton only.** All 4 required API routes exist and return correctly-shaped data, but `/api/execute` returns a **hardcoded stub response** — no LLMod.ai, Pinecone, or Supabase calls have been made yet. The `agent/` package has the intended module structure and prompts written, but every tool raises `NotImplementedError` until the real ReAct loop is wired in and reviewed.

Blocked on:
- Supabase project creation (web dashboard, not scriptable) — `scripts/schema.sql` is ready to run the moment it exists.
- Team review of this skeleton before any real ingestion or agent calls (which will spend LLMod.ai budget).
- Yair's and Andrei's emails for `team_info.json` (currently placeholders).

## API

- `GET /api/team_info` — team roster.
- `GET /api/agent_info` — description, purpose, prompt template, worked examples.
- `GET /api/model_architecture` — architecture diagram (PNG).
- `POST /api/execute` — `{"prompt": "..."}` in; `{"status", "error", "response", "steps"}` out.

## Local setup

```bash
cp .env.example .env   # fill in credentials once available
pip install -r requirements.txt
python app.py           # http://localhost:5000
```

## Deployment

Vercel, Python serverless (`vercel.json`, same pattern as the team's prior `medium-rag-hw` assignment). Environment variables are set in the Vercel dashboard only — this repo is public, so no real keys are ever committed (see `.env.example` for the required names).

```
Vercel URL: TBD
GitHub Repo URL: https://github.com/alanarazi7/movibot
```
