# 🎬 MoviBot E5 Embedding Backend: Complete Demo

## What's New

You now have a **production-ready local sandbox** for testing MoviBot's semantic search capabilities before committing to paid services (Chapters 3-4).

### Core Components

1. **E5 Embedding Backend** (`agent/tools/plot_search_embed.py`)
   - Real semantic search using E5-small-v2 embeddings (384-dim)
   - Pre-computed cache for 170 movies (~550 KB, regenerable)
   - Cosine similarity scoring
   - Same interface as Pinecone production backend

2. **Backend Dispatcher** (`agent/tools/plot_search.py`)
   - Reads `PLOT_SEARCH_BACKEND` env var
   - Default: IDF keyword matching (production mode)
   - Opt-in: E5 embeddings (local testing mode)
   - Zero changes to react_loop or call sites

3. **Setup Infrastructure**
   - `scripts/local_sandbox_setup.py` — One-time cache builder
   - `requirements-local.txt` — Isolated dependencies
   - `.gitignore` — Cache files excluded from repo

4. **Interactive UI**
   - `streamlit_demo.py` — Streamlit app for side-by-side testing
   - Query input + backend toggle
   - Results visualization + full ReAct trace

5. **Documentation**
   - `EMBEDDING_BACKEND.md` — Technical guide
   - `STREAMLIT_DEMO.md` — UI documentation
   - `LOCAL_SANDBOX.md` — Architecture (updated)
   - `NEXT_STEPS.md` — Build plan (updated)

---

## Quick Start (5 Minutes)

### Setup (One-Time)

```bash
cd ~/tabstar/movibot

# Install dependencies
pip install -r requirements-local.txt

# Build embedding cache (30 seconds)
python scripts/local_sandbox_setup.py
```

### Run the Demo

```bash
# Interactive Streamlit UI
streamlit run streamlit_demo.py
# Opens at http://localhost:8501

# Or: Direct Python testing
export PLOT_SEARCH_BACKEND=embedding
python3 -c "from agent import react_loop; print(react_loop.execute('Find me an animated adventure')['response'])"
```

---

## Feature Comparison

### IDF Backend (Keyword Matching)

| Aspect | Details |
|--------|---------|
| **Model** | Local tokenizer + word frequency |
| **Speed** | ~1 ms per query |
| **Approach** | Exact keyword match + tag boost |
| **Strength** | Fast, structured queries |
| **Weakness** | Misses thematic similarity |

**Example Query:** "heartwarming adventure with a loyal animal companion"

**Results:**
1. Tangled (2010) — matched: "animal", "loyal"
2. Homeward Bound (1993) — matched: "animal", "loyal"
3. Lion King (1994) — matched: "companion"

---

### E5 Backend (Semantic Search)

| Aspect | Details |
|--------|---------|
| **Model** | E5-small-v2 embeddings (384-dim) |
| **Speed** | ~10 ms per query |
| **Approach** | Cosine similarity in embedding space |
| **Strength** | Finds thematic matches, handles fuzzy queries |
| **Weakness** | Slightly slower, different from production (OpenAI embeddings) |

**Example Query:** "heartwarming adventure with a loyal animal companion"

**Results:**
1. Wild Hearts Can't Be Broken (1991) — score: 0.8319
2. Return to Snowy River (1988) — score: 0.8235
3. Homeward Bound (1993) — score: 0.8146

💡 **Notice:** E5 discovers "Wild Hearts Can't Be Broken" (woman + horse), which has no exact keywords but is semantically similar.

---

## Streamlit UI Demo

### Interface Sections

**Sidebar Controls:**
- Backend selection (E5, IDF, Compare Both)
- Number of results (1-20)
- Show full ReAct trace (toggle)
- Sample query selector

**Main Area:**
- Query input field
- Results dataframe (sortable, searchable)
- Detailed result cards (expandable)
- Backend comparison insights
- Full ReAct loop trace (if enabled)

### Example: "Find me an animated adventure about a magical kingdom"

**Backend: E5 Embedding**

```
1. Moana (2016)                    Semantic Score: 0.8095
2. Maleficent (2014)               Semantic Score: 0.7992
3. Atlantis: The Lost Empire (2001) Semantic Score: 0.7977
4. Tangled (2010)                  Semantic Score: 0.7968
5. Enchanted (2007)                Semantic Score: 0.7961
```

**Backend: IDF Keyword Matching**

```
1. Moana (2016)                    Score: 7.7970  Terms: about, animated, find
2. Tinker Bell (2009)              Score: 7.3340  Terms: about, adventure, find
3. Cinderella (2015)               Score: 7.2590  Terms: about, find, kingdom
4. Saving Mr. Banks (2013)         Score: 7.0350  Terms: about, adventure, animated
5. Pirates at World's End (2007)   Score: 6.6980  Terms: about, find, lead
```

**Full ReAct Loop Trace (E5 backend):**
```
1. Reasoner        → "Use CatalogFilter for structured filters"
2. CatalogFilter   → 107 Disney/Pixar candidates
3. Reasoner        → "Use PlotSearch for thematic content"
4. PlotSearch      → 20 semantic matches (E5 embeddings)
   ├─ Moana (0.8095)
   ├─ Maleficent (0.7992)
   └─ ... 18 more
5. Reasoner        → "All tools run, synthesize answer"
6. Synthesizer     → Final list of 20 recommendations
```

---

## Production Safety

✅ **Verified:**
- `requirements.txt` (Vercel) **unchanged** — no `sentence-transformers`
- `app.py` **unchanged** — no new code paths
- `react_loop.py` **unchanged** — delegation happens in `plot_search.py`
- Default behavior (no env var) **uses IDF backend** (production)
- Embedding cache **in `.gitignore`** — never committed

### Deployment Status

| Component | Production | Local Sandbox |
|-----------|-----------|--------------|
| Flask app | ✅ Works | ✅ Works |
| PlotSearch (IDF) | ✅ Active | ✅ Active (default) |
| PlotSearch (E5) | ❌ Not needed | ✅ Available |
| Vercel build | ✅ Unaffected | — |
| Env vars | No E5 needed | Optional `PLOT_SEARCH_BACKEND` |

---

## Files Overview

### New Files

```
agent/tools/plot_search_embed.py    # E5 embedding backend
scripts/local_sandbox_setup.py       # Cache builder
scripts/demo_backends.py             # Side-by-side comparison
streamlit_demo.py                    # Interactive UI
requirements-local.txt               # Local-only dependencies
EMBEDDING_BACKEND.md                 # Technical guide
STREAMLIT_DEMO.md                    # UI guide
DEMO_SUMMARY.md                      # This file
```

### Modified Files

```
agent/tools/plot_search.py           # Backend dispatcher (+env var check)
LOCAL_SANDBOX.md                     # Clarified scope (E5-only)
NEXT_STEPS.md                        # Updated Local Sandbox section
.gitignore                           # Added embedding cache entries
requirements-local.txt               # Added streamlit
```

### Unchanged Files (Production Safe)

```
app.py                               # No changes
react_loop.py                        # No changes
requirements.txt                     # No changes
vercel.json                          # No changes
agent/llm_client.py                  # No changes (MockLLMClient still active)
```

---

## Usage Scenarios

### Scenario 1: Quick Query Test (CLI)

```bash
export PLOT_SEARCH_BACKEND=embedding
python3 << 'EOF'
from agent.tools import plot_search
results = plot_search.run("magical adventure with animals", top_k=3)
for r in results:
    print(f"{r['title']} ({r['release_year']}) - {r['score']:.4f}")
EOF
```

### Scenario 2: Full Agent Test (CLI)

```bash
export PLOT_SEARCH_BACKEND=embedding
python3 << 'EOF'
from agent import react_loop
result = react_loop.execute("Find me a Disney movie for a 5-year-old with no scary parts")
print(result['response'])
print("\nSteps:", [s['module'] for s in result['steps']])
EOF
```

### Scenario 3: Side-by-Side Comparison

```bash
python scripts/demo_backends.py
```

Outputs: IDF vs. E5 results for sample queries.

### Scenario 4: Interactive Testing (Streamlit)

```bash
streamlit run streamlit_demo.py
```

Opens web UI at http://localhost:8501. Features:
- Live query input
- Backend toggle
- Results visualization
- Full ReAct trace inspection
- Configuration sidebar

---

## Next Steps (After Local Validation)

If semantic search quality looks good:

1. **Chunk 2:** Load movies to Supabase (free, no LLM calls)
2. **Chunk 3:** Create Pinecone index, embed with LLMod.ai (~$0.03)
3. **Chunk 4:** Replace MockLLMClient with LLMod.ai LLM calls (~$12.97)

The E5 backend can remain as a permanent local dev tool.

---

## Performance Notes

| Metric | IDF | E5 |
|--------|-----|-----|
| Query time | ~1 ms | ~10 ms |
| Model load | N/A | ~2 sec (first run) |
| Cache size | N/A | ~550 KB |
| Memory (model) | ~100 MB | ~500 MB |
| Dependencies | 0 extra | sentence-transformers + torch |

**Notes:**
- E5 model cached on first run (~130 MB download from HuggingFace)
- CPU inference (no GPU needed)
- Subsequent queries are ~10 ms on modern hardware
- Streamlit subprocess overhead: ~100 ms per execution (acceptable for UI)

---

## Troubleshooting

### "Cannot find module plot_embeddings.npy"

```bash
python scripts/local_sandbox_setup.py
```

### "ImportError: No module named 'sentence_transformers'"

```bash
pip install -r requirements-local.txt
```

### "Streamlit not found"

```bash
python3 -m pip install streamlit
```

### Results identical on both backends

Some queries have common keywords. Try:
- "loyal sidekick" (E5 should find thematic matches)
- "magical adventure" (both work similarly)

### App runs but returns no results

- Check `echo $PLOT_SEARCH_BACKEND` (should be "embedding" or unset)
- Verify cache: `ls data_preprocessing/data_ready/plot_embeddings*`
- Try IDF backend as baseline: `unset PLOT_SEARCH_BACKEND`

---

## Summary

You now have:

✅ **Local E5 semantic search backend** — test real embedding-based search
✅ **Backward-compatible dispatch** — flip env var to toggle backends
✅ **Production-safe design** — Vercel deployment unaffected
✅ **Interactive Streamlit UI** — visualize both backends side-by-side
✅ **Comprehensive documentation** — setup, usage, troubleshooting
✅ **Zero external API calls** — runs entirely locally

Ready to explore semantic search before committing $13 to production!

---

## See Also

- [EMBEDDING_BACKEND.md](EMBEDDING_BACKEND.md) — Technical deep-dive
- [STREAMLIT_DEMO.md](STREAMLIT_DEMO.md) — UI features and customization
- [LOCAL_SANDBOX.md](LOCAL_SANDBOX.md) — Architecture overview
- [NEXT_STEPS.md](NEXT_STEPS.md) — Build plan (Chunks 1-5)
