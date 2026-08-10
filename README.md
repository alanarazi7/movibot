# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "safe for a toddler") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `docs/pitch-deck.html` for the full writeup and a live example of this failure).

## How it works

MoviBot runs a ReAct loop (`Reason -> Act -> Observe -> Stop?`) over four tools, narrowing from cheap structured filtering down to expensive per-title verification:

| Module | Role | Current |
|---|---|---|
| `Reasoner` | Plans the next action each loop iteration | Mock (deterministic) |
| `CatalogFilter` | Structured filter (year, studio, genre, runtime) | CSV-backed (commits data) |
| `PlotSearch` | Thematic/character semantic search over plot overviews | IDF mock (CSV-backed) |
| `SceneSearch` | Per-candidate check for risky events (deaths, scary scenes) | Mock (deterministic reasoning) |
| `ExternalContext` | Per-candidate check for tone/subtext not in the plot | Mock (deterministic reasoning) |
| `Synthesizer` | Composes the final, evidence-backed answer | Mock (deterministic) |

**Future (Chunks 2–4-real):** CatalogFilter → Supabase queries, PlotSearch → Pinecone vectors, Reasoner/SceneSearch/ExternalContext → real LLM calls via LLMod.ai.

See `assets/architecture.png` (served at `GET /api/model_architecture`) — module names there match the `steps` trace exactly.

**Data**: 2 Kaggle sources — "The Movies Dataset" (2 tables: `movies_metadata.csv` + `keywords.csv`) and MPST (`mpst_full_data.csv`, richer plot synopses matched by exact IMDb ID). Narrowed to a Disney + Pixar demo scope, then cleaned (45,466 raw → 304 Disney + Pixar → 303 clean). All 303 are in `data_preprocessing/data_ready/supabase_movies.csv` (committed, 223 KB); the 170 with an MPST synopsis (56% coverage) are in `data_preprocessing/data_ready/pinecone_candidates.csv` (committed, 2.7 MB). Raw Kaggle downloads (`data_preprocessing/data_full/`) are gitignored but regenerable via `python data_preprocessing/prepare_movibot_data.py --all-studios`. When Chunks 2–4-real are live, all 303 will be in Supabase with every original column; the 170 will be vectorized and indexed in Pinecone. Wikipedia is fetched live per-candidate rather than pre-indexed. See `data_preprocessing/data cleaning rules.md` for the full design rationale.

## Status

**Current state: Mock agent deployed to Vercel. Live at [movibot-gamma.vercel.app](https://movibot-gamma.vercel.app).**

- ✅ **Chunk 1 (fetch & filter):** Done. `data_preprocessing/prepare_movibot_data.py` produces:
  - `supabase_movies.csv` (303 Disney + Pixar movies, 25 columns) — **now committed**
  - `pinecone_candidates.csv` (170 with MPST synopsis, 56% coverage) — **now committed**
  - Raw Kaggle downloads (`data_preprocessing/data_full/`) remain gitignored/regenerable

- ✅ **Chunk 4 (mock agent):** Done. Full ReAct loop implemented end-to-end:
  - `MockLLMClient` with deterministic reasoning for validation
  - All 4 tools (`CatalogFilter`, `PlotSearch`, `SceneSearch`, `ExternalContext`) fully functional
  - `/api/execute` wired to working agent, returns real results + full `steps` trace
  - Architecture diagram (`GET /api/model_architecture`) complete, module names consistent

- ✅ **Chunk 5 (deploy & polish):** Done. Deployed to Vercel. All 4 endpoints live and verified. `agent_info.json` contains captured real mock run example.

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
pip install -r requirements.txt
python app.py           # http://localhost:5000
```

The mock agent needs no external credentials (no `.env` required to run locally). For Chunks 2–4-real, copy `.env.example` → `.env` and fill in LLMod.ai/Pinecone/Supabase keys.

## Deployment

Vercel, Python serverless (`vercel.json`, same pattern as the team's prior `medium-rag-hw` assignment). The mock agent currently runs with no external dependencies (no API keys needed). When Chunks 2–4-real are implemented (Supabase + Pinecone + real LLM), environment variables will be set in the Vercel dashboard (see `.env.example` for the required names).

- **Live URL:** https://movibot-gamma.vercel.app
- **GitHub Repo:** https://github.com/alanarazi7/movibot
- **Vercel Project Dashboard:** https://vercel.com/alan-agents-course/movibot
