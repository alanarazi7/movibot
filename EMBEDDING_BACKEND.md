# E5 Embedding Backend: Local Testing Guide

**TL;DR:** Test MoviBot with real semantic search instead of keyword matching, entirely locally with zero external API calls.

## Quick Start (5 minutes)

```bash
# 1. Install dependencies (~30 MB, one-time)
pip install -r requirements-local.txt

# 2. Build embedding cache (~30 seconds, one-time)
python scripts/local_sandbox_setup.py

# 3. Enable embedding backend and run
export PLOT_SEARCH_BACKEND=embedding
python app.py
# Now http://localhost:5000 uses E5 embeddings for PlotSearch

# 4. Query the agent
curl -X POST http://localhost:5000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find me an animated adventure about a magical kingdom"}'
```

## What You Get

| Feature | IDF (Default) | E5 Embedding |
|---------|---------------|--------------|
| Search type | Keyword matching | Semantic similarity |
| Model | Local tokenizer | E5-small-v2 (384-dim) |
| Example query | "animal" matches "animal" in text | "loyal companion" matches "faithful sidekick" |
| Speed | ~1ms per query | ~10ms per query |
| Cost | Free | Free (local inference) |
| API keys | None needed | None needed |

## Backends in Action

**Query:** "heartwarming adventure with a loyal animal companion"

**IDF Backend (keyword-based):**
1. Tangled - matched terms: animal, loyal
2. Homeward Bound - matched terms: animal, loyal
3. Lion King - matched terms: companion

**E5 Backend (semantic):**
1. Wild Hearts Can't Be Broken - semantic score: 0.8319
2. Return to Snowy River - semantic score: 0.8235
3. Homeward Bound - semantic score: 0.8146

Notice: E5 finds "Wild Hearts Can't Be Broken" (about a woman and her horse), which has no exact keywords but is semantically similar.

## Files

| File | Purpose |
|------|---------|
| `agent/tools/plot_search_embed.py` | E5 embedding backend |
| `agent/tools/plot_search.py` | Dispatcher (reads `PLOT_SEARCH_BACKEND` env var) |
| `scripts/local_sandbox_setup.py` | Cache builder |
| `requirements-local.txt` | Local dependencies (not in production) |
| `data_preprocessing/data_ready/plot_embeddings.npy` | Cached embeddings (170 × 384) |
| `data_preprocessing/data_ready/plot_embeddings_mapping.json` | Movie ID ordering |

## Environment Variable

- **`PLOT_SEARCH_BACKEND=embedding`** → Use E5 embeddings
- **Unset (default)** → Use IDF keyword matching (production mode)

## Comparison Script

Compare both backends on sample queries:

```bash
python scripts/demo_backends.py
```

Output shows side-by-side results for quick visual comparison.

## How It Works

1. **Setup phase** (`local_sandbox_setup.py`):
   - Load 170 movies with plot synopses
   - Encode each with E5-small-v2 using `"passage: "` prefix (E5 requirement)
   - L2-normalize embeddings (for cosine similarity via dot product)
   - Write cache to `plot_embeddings.npy` and `plot_embeddings_mapping.json`

2. **Query phase** (via `plot_search_embed.run()`):
   - Encode user query with `"query: "` prefix
   - L2-normalize query embedding
   - Compute cosine similarity (dot product) with cached embeddings
   - Return top-k matches with scores

3. **Backend dispatch** (in `plot_search.py`):
   - Check `PLOT_SEARCH_BACKEND` env var
   - If `"embedding"` → delegate to `plot_search_embed.run()`
   - If unset → use IDF keyword matching (existing mock)

## Production Safety

The embedding backend is **100% isolated from production**:

- ✅ `requirements.txt` (production) doesn't include `sentence-transformers`
- ✅ `requirements-local.txt` is local-development-only
- ✅ Default (no env var) uses IDF backend
- ✅ Vercel deployment unchanged

To verify production unaffected:

```bash
# Simulate production environment
unset PLOT_SEARCH_BACKEND
python3 << 'EOF'
from agent import react_loop
result = react_loop.execute("Find me a Disney movie from the 1990s")
print(f"Status: {result['status']}")  # Should be "ok"
EOF
```

## Troubleshooting

### ImportError: No module named 'sentence_transformers'

```bash
pip install -r requirements-local.txt
```

### FileNotFoundError: plot_embeddings.npy not found

```bash
python scripts/local_sandbox_setup.py
```

### Results look identical to IDF backend

- Confirm `echo $PLOT_SEARCH_BACKEND` outputs `embedding`
- Try a more semantic query: "loyal sidekick" (E5 handles fuzzy similarity)
- IDF backend only matches exact words; if query has common words like "find", "movie", "Disney", both backends may look similar

### Slow startup (first use)

First run downloads E5-small-v2 model (~130 MB) from HuggingFace. Cached under `~/.cache/huggingface/`, no re-download. Setup takes ~30 seconds.

## Next Steps (Real Backend Testing)

Once you're satisfied with E5 results locally:

1. **Chunk 2:** Load data to Supabase (free)
2. **Chunk 3:** Create Pinecone index and embed with LLMod.ai (~$0.03)
3. **Chunk 4:** Replace LLMod.ai LLM calls for Reasoner/SceneSearch/ExternalContext (~$12.97)

The embedding backend can stay as a permanent local-development tool alongside production.

## References

- [E5 Model](https://huggingface.co/intfloat/e5-small-v2) — Semantic search embeddings, 384-dim
- [Sentence-Transformers](https://www.sbert.net/) — Local embedding library
- [LOCAL_SANDBOX.md](LOCAL_SANDBOX.md) — Full sandbox architecture
- [NEXT_STEPS.md](NEXT_STEPS.md) — Chunk-by-chunk build plan
