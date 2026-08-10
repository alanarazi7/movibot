# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "safe for a toddler") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `docs/pitch-deck.html` for the full writeup and a live example of this failure).

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

**Data**: 2 Kaggle sources — "The Movies Dataset" (2 tables: `movies_metadata.csv` + `keywords.csv`) and MPST (`mpst_full_data.csv`, richer plot synopses matched by exact IMDb ID). Narrowed to a Disney + Pixar demo scope first, straight off the raw data, then cleaned (45,466 raw → 304 Disney + Pixar → 303 clean). All 303 go to Supabase for structured filtering, with every original movie column kept; the 170 with an MPST synopsis (56% coverage) are embedded into Pinecone for semantic search. Wikipedia is fetched live per-candidate rather than pre-indexed. See `data_preprocessing/data cleaning rules.md` for the full design rationale.

## Status

**Current state: Mock agent complete, ready for real implementation.**

- ✅ **Chunk 1 (fetch & filter):** Done. `data_preprocessing/prepare_movibot_data.py` produces:
  - `supabase_movies.csv` (303 Disney + Pixar movies, 25 columns)
  - `pinecone_candidates.csv` (170 with MPST synopsis, 56% coverage)
  - Both gitignored/regenerable from raw Kaggle CSVs (`--all-studios` for full catalog)

- ✅ **Chunk 4 (mock agent):** Done. Full ReAct loop implemented end-to-end:
  - `MockLLMClient` with deterministic reasoning for validation
  - All 4 tools (`CatalogFilter`, `PlotSearch`, `SceneSearch`, `ExternalContext`) fully functional
  - `/api/execute` wired to working agent, returns real results + full `steps` trace
  - Architecture diagram (`GET /api/model_architecture`) complete, module names consistent

- 🔄 **Next:** Chunk 2 (Supabase load) + Chunk 3 (Pinecone embeddings) — then swap mock → real Chunk 4. Supabase project exists (credentials in `.env`, gitignored); `movies` table not yet created. LLMod.ai/Pinecone keys still needed (only for paid track).

See **[`TODO.md`](TODO.md)** for the full chunk-by-chunk technical checklist.

## Docs

- `docs/course-assignment-instructions.pdf` — the course's official assignment spec (API contract, deployment, budget, deadline).
- `docs/team-idea-proposal-and-data-sources.pdf` — the team's own idea writeup, including the dataset decision (Kaggle "The Movies Dataset," downsampled to ~5K movies).
- `docs/pitch-deck.pptx` / `docs/pitch-deck.html` — the pitch deck (problem, architecture, demo).

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
