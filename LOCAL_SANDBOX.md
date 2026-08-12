# Local Sandbox: E5 Embeddings Backend

Run MoviBot with **real semantic search (E5-small-v2 embeddings)** instead of mock IDF scoring — **entirely locally, zero external API calls or budget spend**.

---

## What You Get

| Component | Local Sandbox | Production (Chunks 3-4) |
|-----------|---------------|------------------------|
| **PlotSearch** | E5-small-v2 (384-dim, local) | Pinecone + LLMod.ai |
| **Reasoner** | MockLLMClient (deterministic) | LLMod.ai GPT-5.4-mini |
| **SceneSearch** | Keyword heuristics (mock) | LLMod.ai reasoning |
| **ExternalContext** | Keyword heuristics (mock) | LLMod.ai reasoning |
| **Cost** | Free (one-time 130 MB model download) | ~$13 |

**Why?** Test semantic search quality before committing budget to Chunks 3-4. The embedding backend uses the same interface as Pinecone but runs locally with zero keys/credits.

---

## Setup: One-Time

### Step 1: Install Local Dependencies

```bash
cd ~/tabstar/movibot
pip install -r requirements-local.txt
```

This installs `sentence-transformers` (which pulls `torch`). First run downloads the E5-small-v2 model (~130 MB) from HuggingFace Hub, cached under `~/.cache/huggingface/`.

### Step 2: Build the Embedding Cache

```bash
python scripts/local_sandbox_setup.py
```

This:
- Loads `data_preprocessing/data_ready/pinecone_candidates.csv` (170 movies with plot synopses)
- Encodes each movie's `embedding_text` with E5-small-v2 using `"passage: "` prefix (E5 requirement)
- L2-normalizes embeddings for cosine similarity
- Writes cache to:
  - `data_preprocessing/data_ready/plot_embeddings.npy` (embedding matrix)
  - `data_preprocessing/data_ready/plot_embeddings_mapping.json` (movie_id ordering)

Takes ~30 seconds. Regenerable if deleted (cache files added to `.gitignore`).

### Step 3: Enable the Embedding Backend

```bash
export PLOT_SEARCH_BACKEND=embedding
python app.py
# http://localhost:5000
```

Unset the env var to revert to IDF mock (the default, used by Vercel deployment):

```bash
unset PLOT_SEARCH_BACKEND
python app.py
```

---

## Testing

### Quick Direct Test

```bash
python3 << 'EOF'
import os
os.environ["PLOT_SEARCH_BACKEND"] = "embedding"

from agent.tools import plot_search

# Query with thematic content (few exact keywords)
query = "a heartwarming adventure with a loyal animal companion"
results = plot_search.run(query, top_k=5)

for r in results:
    print(f"{r['title']} ({r['release_year']}) - score: {r['score']:.3f}")
EOF
```

### Full ReAct Loop (Flask App)

```bash
export PLOT_SEARCH_BACKEND=embedding
python app.py
```

Then in another terminal:

```bash
curl -X POST http://localhost:5000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find me a Disney movie from the 1990s with no deaths, good for a toddler"}'
```

Check the response's `steps` trace: `PlotSearch` should show embedding-based results (different scores/matches than IDF), and the full ReAct loop should complete with zero external API calls.

### Compare IDF vs. Embedding Backends

Side-by-side comparison on a semantic query:

```bash
python3 << 'EOF'
import os

# Load both backends
os.environ["PLOT_SEARCH_BACKEND"] = "idf"
from agent.tools import plot_search as ps_idf

os.environ["PLOT_SEARCH_BACKEND"] = "embedding"
import importlib
importlib.reload(ps_idf.__module__)
from agent.tools import plot_search as ps_embed

query = "magical adventure with talking animals"

print("IDF Backend:")
for r in ps_idf.run(query, top_k=3):
    print(f"  {r['title']} - {r['matched_terms']}")

print("\nEmbedding Backend:")
for r in ps_embed.run(query, top_k=3):
    print(f"  {r['title']} - score: {r['score']:.3f}")
EOF
```

---

## Architecture: Embedding Backend

```
PlotSearch.run(query)
    ↓
    [env PLOT_SEARCH_BACKEND = "embedding"] ?
    ↓ Yes
    plot_search_embed.run(query)
        ↓
        E5 encode query with "query: " prefix
        ↓
        L2-normalize query embedding
        ↓
        Load cached embeddings (170 × 384)
        ↓
        Cosine similarity: dot product with L2-norm
        ↓
        Sort by score, return top-k
        ↓
        Return {"movie_id", "title", "release_year", "score", "matched_terms": []}
    ↓ No (default)
    Use IDF mock backend (existing behavior)
```

---

## Verification Checklist

- [ ] `python scripts/local_sandbox_setup.py` completes without error
- [ ] `data_preprocessing/data_ready/plot_embeddings.npy` (~550 KB) exists
- [ ] `data_preprocessing/data_ready/plot_embeddings_mapping.json` exists
- [ ] `export PLOT_SEARCH_BACKEND=embedding && python app.py` starts server
- [ ] `/api/execute` with a thematic query returns results
- [ ] Results differ from IDF backend (more semantic matches, fewer exact-word matches)
- [ ] Default (no env var) still uses IDF backend for regression testing
- [ ] Vercel deployment (with `requirements.txt`, not `requirements-local.txt`) still works

---

## Limitations vs. Production

| Aspect | Local | Production |
|--------|-------|-----------|
| **Model** | E5-small-v2 (384-dim) | text-embedding-3-small (1536-dim, OpenAI) |
| **Speed** | ~10ms per query (local) | ~100ms per query (Pinecone network round-trip) |
| **Scalability** | 170 movies only (Disney/Pixar demo) | 300+ movies (full catalog, future) |
| **Infrastructure** | In-process memory | Pinecone serverless index |

E5-small-v2 is weaker than OpenAI embeddings, but good enough to demo semantic search behavior. Results may differ from production once Pinecone is live (expected, model difference).

---

## Next Steps (After Local Validation)

If semantic search quality looks good locally, proceed to:
1. **Chunk 2:** Supabase load (free, already ready)
2. **Chunk 3:** Pinecone + LLMod.ai embeddings (~$0.03, gated)
3. **Chunk 4:** LLMod.ai LLM calls for Reasoner/SceneSearch/ExternalContext (~$12.97, gated)

If results need tuning, iterate on system prompts locally first, before spending budget.

---

## Troubleshooting

### ImportError: No module named 'sentence_transformers'

```bash
pip install -r requirements-local.txt
```

### FileNotFoundError: plot_embeddings.npy not found

```bash
python scripts/local_sandbox_setup.py
```

### Results look the same as IDF backend

- Confirm `echo $PLOT_SEARCH_BACKEND` is `embedding` (not empty or `idf`)
- Try a more semantic query (e.g., "heartwarming adventure" vs. "Disney 1990s")
- IDF backend only matches exact words; embedding backend matches themes

### Slow startup (first use)

First run downloads E5-small-v2 model (~130 MB) from HuggingFace. Cached locally, no re-download. Embedding computation takes ~30 seconds for 170 movies (one-time).

---

## Files

| File | Purpose | Size |
|------|---------|------|
| `agent/tools/plot_search.py` | Backend dispatch (env var check) | Modified |
| `agent/tools/plot_search_embed.py` | E5 embedding backend | New |
| `scripts/local_sandbox_setup.py` | Cache builder | New |
| `requirements-local.txt` | Local dependencies only | New |
| `data_preprocessing/data_ready/plot_embeddings.npy` | Cached embedding matrix | Generated (~550 KB) |
| `data_preprocessing/data_ready/plot_embeddings_mapping.json` | Movie ID ordering | Generated (~5 KB) |

