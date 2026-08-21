# MoviBot

A movie recommender for requests that mix hard constraints with soft ones —
*"a Disney film from 1990+, with animals and no death"* — over a curated
catalog of **316 Disney and Pixar feature films (1937–2017)**.

Course project for *Introduction to Modern AI Agents*.
Team: Yair Zack, Andrei Nekliudov, Alan Arazi (batch 2, order 10).

**Live: [movibot-gamma.vercel.app](https://movibot-gamma.vercel.app)**

## The problem

Year and studio are columns; *"no deaths"* and *"not scary"* are not. Retrieval
alone cannot answer the second kind — embed "no deaths" and you get the films
where somebody dies, because that is what those plots say. Asking a model
directly gets a confident list with no way to tell which parts were checked.

MoviBot answers both kinds in one request, and **never names a film it did not
verify against the film's own plot text**. Every recommendation cites a sentence
quoted from the source.

## How it works

The app is the documentation. Its **Architecture** tab has the diagram and a
short account of each stage; the **Prompts** tab serves every prompt live from
the source, so nothing here can drift from what runs.

## Run it

```bash
pip install -r requirements.txt
python app.py           # http://localhost:5000
```

The catalog reads from committed CSVs and needs no credentials. Answering a
request calls LLMod.ai, so copy `.env.example` → `.env` and fill in
`OPENAI_API_KEY` and `OPENAI_BASE_URL`. `python scripts/check_credentials.py`
reports what is missing without spending anything.

Two checks run free and offline, and are the fastest way to see the project
still holds together:

```bash
python scripts/check_screen.py    # the word screen's safety property
python scripts/check_gates.py     # the API contract, the call cap, every displayed count
```

## API

- `GET /api/team_info` — team roster
- `GET /api/agent_info` — description, prompt template, worked examples
- `GET /api/model_architecture` — architecture diagram (PNG)
- `POST /api/execute` — `{"prompt": "..."}` in; `{"status", "error", "response", "steps"}` out

## Deployment

Vercel, Python serverless. **Deploys do not happen on `git push`** — the project
is not Git-connected, so a push updates GitHub only:

```bash
vercel --prod --yes --scope alan-agents-course
```

## Also here

- [`TODO.md`](TODO.md) — the project log: what was decided, and why
- [`rag/DECISIONS.md`](rag/DECISIONS.md) — retrieval parameters and the reasoning
- `docs/` — the assignment spec, the team proposal, the pitch deck
