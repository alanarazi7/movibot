# 🎬 MoviBot Streamlit Demo

Interactive UI for testing and comparing the IDF keyword matching vs. E5 semantic embedding backends.

## Quick Start

```bash
# Install dependencies (one-time)
pip install -r requirements-local.txt
python scripts/local_sandbox_setup.py

# Run the Streamlit app
streamlit run streamlit_demo.py
```

Browser opens at `http://localhost:8501`

## Features

### 🔀 Backend Selection
- **E5 Embedding (Semantic):** Find thematic similarity even without exact keywords
- **IDF Keyword Matching:** Match exact words in plot text
- **Compare Both:** Side-by-side visualization of both backends

### 📝 Sample Queries
Click the dropdown for pre-loaded queries:
- "Find me an animated adventure about a magical kingdom with a strong female lead"
- "heartwarming adventure with a loyal animal companion"
- "a fun family movie with talking animals"
- "Find me a Disney movie from the 1990s with no deaths, safe for a toddler"
- "magical adventure involving love and music"

### 🎯 Results Visualization
- **Dataframe view:** Quick scan of top-k results
- **Detailed view:** Expandable cards with full movie info
- **Configurable:** Adjust number of results (1-20)

### 🔄 Full ReAct Loop Trace
Enable **"Show full ReAct trace"** to see:
- Reasoner decisions (which tool to run next)
- CatalogFilter results (structured filtering)
- PlotSearch results (semantic or keyword matching)
- SceneSearch/ExternalContext verifications
- Synthesizer final answer

## Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  🎬 MoviBot: PlotSearch Backend Demo                        │
├─────────────────────────────────────────────────────────────┤
│  SIDEBAR                          │  MAIN AREA               │
│  ⚙️ Configuration                 │  📝 Query Input          │
│   - Backend selection              │  - Custom or sample      │
│   - # results (1-20)               │  - Real-time results     │
│   - Show ReAct trace               │  - Dataframe view        │
│  📝 Sample Queries                 │  - Detailed results      │
│   - Click to load example           │                          │
│                                    │  💡 Insights             │
│                                    │  - Backend comparison    │
│                                    │  - Strengths/weaknesses  │
│                                    │                          │
│                                    │  🔄 Full Trace (opt)     │
│                                    │  - Steps breakdown       │
│                                    │  - Module details        │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Example 1: Semantic vs. Keyword Match

**Query:** "heartwarming adventure with a loyal animal companion"

**IDF Results** (keyword-based):
- Tangled (score: 4.95) — matched terms: "animal", "loyal"
- Homeward Bound (score: 4.95) — matched terms: "animal", "loyal"
- Lion King (score: 3.75) — matched terms: "companion"

**E5 Results** (semantic):
- Wild Hearts Can't Be Broken (score: 0.8319)
- Return to Snowy River (score: 0.8235)
- Homeward Bound (score: 0.8146)

💡 E5 finds "Wild Hearts Can't Be Broken" (woman + horse adventure) without exact keywords, while IDF only matches literal words.

### Example 2: Full ReAct Loop

Enable "Show full ReAct trace" and query: "Find me an animated adventure about a magical kingdom"

See the agent's decision process:
1. **Reasoner** → "Use CatalogFilter for structured filters"
2. **CatalogFilter** → 107 Disney/Pixar candidates
3. **Reasoner** → "Use PlotSearch for thematic content"
4. **PlotSearch** → Top 20 semantic matches (E5 backend)
5. **Reasoner** → "No risky content constraints, synthesize answer"
6. **Synthesizer** → Final list of recommendations

## Configuration

### Environment Variables

The Streamlit app always runs in isolated subprocesses, so env vars are clean:
- E5 backend: `PLOT_SEARCH_BACKEND=embedding`
- IDF backend: `PLOT_SEARCH_BACKEND=idf` (or unset)

No need to set them manually — the app handles it.

### Sidebar Controls

| Control | Default | Options |
|---------|---------|---------|
| Backend | IDF | E5 Embedding, IDF, Compare Both |
| Results | 5 | 1-20 |
| Show Trace | Off | On/Off |

## Performance

| Metric | IDF | E5 |
|--------|-----|-----|
| Query time | ~1 ms | ~10 ms |
| Model load | N/A | ~2 sec (first run) |
| Inference | Tokenizer | CPU (no GPU needed) |
| Memory | ~100 MB | ~500 MB (model + embeddings) |

**Notes:**
- IDF is instant (local tokenizer)
- E5 first load downloads model (~130 MB), then caches
- Subsequent E5 queries are ~10ms (CPU inference on modern MacBook)

## Troubleshooting

### Streamlit not found

```bash
python3 -m pip install streamlit
```

### "No module named 'agent'"

Run from the movibot directory:
```bash
cd ~/tabstar/movibot
streamlit run streamlit_demo.py
```

### App fails to load results

Check the browser console (F12) for error details. Common issues:
- Embedding cache missing: `python scripts/local_sandbox_setup.py`
- Dependencies missing: `pip install -r requirements-local.txt`
- Working directory wrong: ensure `cd movibot` first

### Results look the same for both backends

This is OK! Some queries have keywords that both backends match. Try:
- "loyal sidekick adventure" (E5 should find more thematic matches)
- "magical adventure" (both should work similarly)

## Customization

### Add More Sample Queries

Edit `streamlit_demo.py`, find `sample_queries` list:

```python
sample_queries = [
    "your new query here",
    "another custom query",
    # ... existing queries ...
]
```

Then restart Streamlit (`Ctrl+C`, `streamlit run streamlit_demo.py`).

### Adjust UI Layout

Streamlit config at top of `streamlit_demo.py`:

```python
st.set_page_config(
    page_title="MoviBot: PlotSearch Backends",
    layout="wide",  # or "centered"
    # ... other options
)
```

### Change Results Display

Modify the DataFrame columns in the backend comparison sections:

```python
df = pd.DataFrame([
    {
        "Title": r["title"],
        "Year": r["release_year"],
        "Score": f"{r['score']:.4f}",
        # Add more columns here
    }
    for r in results
])
```

## Under the Hood

The Streamlit app:
1. Accepts user query + backend choice
2. Spawns isolated Python subprocesses (one per backend)
3. Sets `PLOT_SEARCH_BACKEND` env var before importing agent code
4. Captures JSON results and renders in Streamlit
5. Visualizes PlotSearch + full ReAct trace on demand

This isolation ensures clean env var handling and prevents module caching issues.

## See Also

- [EMBEDDING_BACKEND.md](EMBEDDING_BACKEND.md) — Technical guide
- [LOCAL_SANDBOX.md](LOCAL_SANDBOX.md) — Architecture details
- [streamlit_demo.py](streamlit_demo.py) — Source code
