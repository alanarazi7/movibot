# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "safe for a toddler") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `docs/pitch-deck.html` for the full writeup and a live example of this failure).

## How it works

MoviBot is a **tool-calling agent**: a planner model with three tools, bounded at `MAX_ROUNDS = 5` model turns per query. Each tool answers a different *kind* of question, so a query uses only the ones its constraints actually need — simple queries finish in one round, hard ones in three.

| Tool | Answers from | Used for |
|---|---|---|
| `filter_catalog` | columns | year, genre, language, studio, runtime |
| `search_plots` | meaning | theme, character, premise (passage-level semantic search) |
| `read_synopses` | full text | "does anyone die", "who betrays whom" |

**Guardrails live in the data and tool code, never in the prompt** — the model cannot forget them and a bad plan cannot bypass them. Results are always ordered by `weighted_rating` rather than raw `vote_average`; `filter_catalog` returns at most 40 rows; `read_synopses` reads at most 8 films, truncated to 6,000 characters each, which is what bounds the cost of a turn.

See `assets/architecture.png`, served at `GET /api/model_architecture`.

## Data

Three sources, prepared offline into `data_preprocessing/data_ready/`:

| Artifact | Contents |
|---|---|
| `supabase_movies.csv` | **238** Disney + Pixar feature films, 26 columns — the movie universe |
| `pinecone_candidates.csv` | the **159** with a full MPST plot synopsis (66.8%) — the name is a leftover; no vector database is used |
| `chunk_index.npz` | **3,159** passages and their 1536-dim vectors, scored in memory |
| `wikipedia_cache.csv` | **237** films' Wikipedia articles, plot and non-plot text, scraped once offline |

The catalog is deliberately narrowed to Disney and Pixar, which keeps it in family territory and makes the demo coherent. That's a demo constraint rather than a product decision — the assignment caps stored data at 50 MB and the full multi-studio catalog doesn't fit. `prepare_movibot_data.py --all-studios` produces all 43,270 films from the same pipeline.

Raw Kaggle downloads (`data_preprocessing/data_full/`) are gitignored; everything in `data_ready/` is committed so the repo runs without a rebuild.

**Full rationale — every source, filter, threshold, and a worked example traced end to end — is in [`data_preprocessing/PIPELINE_REVIEW.md`](data_preprocessing/PIPELINE_REVIEW.md).** To regenerate the data, see [`data_preprocessing/prepare_movibot_data usage.md`](data_preprocessing/prepare_movibot_data%20usage.md).

## Status

Retrieval, tools, and the agent loop are complete and exercised end to end at zero cost. What remains needs credentials and a small budget.

- ✅ Data pipeline, chunked passage index, Wikipedia cache
- ✅ Three tools + bounded tool-calling loop (`agent/loop.py`)
- ✅ Catalog reads from committed CSVs; the passage index is a committed matrix, so retrieval needs no vector database
- ⏳ Running the 11 test cases

There is **no mock model by design**: a broken config fails loudly rather than masquerading as a working agent. `MOVIBOT_OFFLINE=1` hard-disables all spending.

See **[`TODO.md`](TODO.md)** for the current checklist.

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
| `MOVIBOT_BACKEND` | `local` | `local` reads the prepared CSVs; `cloud` uses Supabase |

| `MOVIBOT_OFFLINE` | unset | `1` disables every paid call |

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

Vercel, Python serverless (`vercel.json`, same pattern as the team's prior `medium-rag-hw` assignment).

- **Live URL:** https://movibot-gamma.vercel.app
- **GitHub Repo:** https://github.com/alanarazi7/movibot
- **Vercel Project Dashboard:** https://vercel.com/alan-agents-course/movibot
