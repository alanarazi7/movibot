# MoviBot

An agent for movie discovery under mixed constraints: structured facts (year, studio, runtime, genre) combined with fuzzy, subjective ones ("no deaths," "not scary," "safe for a toddler") that no catalog column can pre-encode. Course project for *Introduction to Modern AI Agents*.

Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

## The problem

Classic RAG fails on queries like *"bring me a not-too-old Disney movie with no deaths, ideally involving animals, for my 3-year-old"*: fixed-K retrieval can't guarantee an exhaustive answer, and there's no pre-built "no deaths" column to filter on. Even asking an LLM directly gives incomplete, falsely-confident lists (see `docs/pitch-deck.html` for the full writeup and a live example of this failure).

## How it works

MoviBot is a **tool-calling agent**: a planner model with four tools, bounded at `MAX_ROUNDS = 5` model turns per query. The tools are ordered **cheapest and most exhaustive first**, and each hands the next a smaller candidate set, so a query pays for only the layers its constraints actually require — and the token-heavy layer only ever sees what survived the free ones.

| Tool | Answers from | Narrows | Cost |
|---|---|---|---|
| `filter_catalog` | columns | 238 → N | free, exhaustive |
| `screen_out` | a word scan | N → clear | free, exhaustive |
| `search_plots` | meaning | N → ~10 | ~$0.0000002 |
| `read_synopses` | full text | ≤ 8 films | free, token-heavy |

The `screen_out` layer is what answers the query in the pitch above. A negation cannot be retrieved for: embed *"no deaths"* and the top hits are the films where somebody dies, because that is what those plots say. So it is screened instead — every plot passage of every candidate scanned in 66 ms, which is exhaustive in a way fixed-K retrieval can never be. Its error is one-sided by design: *"dead heat"* over-excludes, it never under-excludes. A match makes a film **flagged**, not rejected, since a word list cannot tell an attempt from an outcome.

**Guardrails live in the data and tool code, never in the prompt** — the model cannot forget them and a bad plan cannot bypass them. Results are always ordered by `weighted_rating` rather than raw `vote_average`; `read_synopses` reads at most 8 films, truncated to 6,000 characters each, which is what bounds the cost of a turn; and `screen_out` refuses to certify a film with under 600 tokens of plot text, so absence of evidence is never reported as evidence.

`python scripts/check_screen.py` asserts the screen's safety property offline and for free.

See `assets/architecture.png`, served at `GET /api/model_architecture`.

## Data

Three sources, prepared offline into `data_preprocessing/data_ready/`:

| Artifact | Contents |
|---|---|
| `supabase_movies.csv` | **238** Disney + Pixar feature films, 26 columns — the movie universe (the name is a leftover; there is no database) |
| `pinecone_candidates.csv` | the **159** with a full MPST plot synopsis (66.8%) — the name is a leftover; no vector database is used |
| `chunk_index.npz` | **3,159** passages and their 1536-dim vectors, scored in memory |
| `wikipedia_cache.csv` | **237** films' Wikipedia articles, plot and non-plot text, scraped once offline |

The catalog is deliberately narrowed to Disney and Pixar, which keeps it in family territory and makes the demo coherent. That's a demo constraint rather than a product decision — the assignment caps stored data at 50 MB and the full multi-studio catalog doesn't fit. `prepare_movibot_data.py --all-studios` produces all 43,270 films from the same pipeline.

Raw Kaggle downloads (`data_preprocessing/data_full/`) are gitignored; everything in `data_ready/` is committed so the repo runs without a rebuild.

Retrieval decisions (chunking, embedding model, why there is no vector database) are in [`rag/DECISIONS.md`](rag/DECISIONS.md), served live in the app's Retrieval tab.

> **The two CSV filenames are historical and now misleading.** `supabase_movies.csv` and `pinecone_candidates.csv` are plain committed CSVs — there is no Supabase and no Pinecone in this project, and neither was ever deployed. Both backends were removed rather than finished (a hosted service for 238 rows and 3,159 vectors buys nothing and costs a credential). The names were kept only because renaming them touches 37 references across 19 files for no functional gain. Read them as `catalog.csv` and `synopses.csv`.

## Status

Retrieval, tools, and the agent loop are complete and exercised end to end at zero cost. What remains needs credentials and a small budget.

- ✅ Data pipeline, chunked passage index, Wikipedia cache
- ✅ Three tools + bounded tool-calling loop (`agent/loop.py`)
- ✅ Catalog reads from committed CSVs; the passage index is a committed matrix, so retrieval needs no vector database
- ⏳ Running the 11 test cases

There is **no mock model by design**: a broken config fails loudly rather than masquerading as a working agent.

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
