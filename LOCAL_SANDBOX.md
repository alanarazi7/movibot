# Local Sandbox: Zero-Cost Testing

Run the full MoviBot agent locally **without any external API calls or budget spend** by using:
- **Local embeddings** (sentence-transformers instead of LLMod.ai)
- **Local vector search** (Chroma or SQLite instead of Pinecone)
- **Claude API** (or mock) for LLM reasoning instead of LLMod.ai

---

## Why Local Sandbox?

| Phase | Data | Embeddings | LLM Calls | Cost | Purpose |
|-------|------|-----------|-----------|------|---------|
| **Chunk 2** | Supabase ✅ | (N/A yet) | Mock | Free | Test structured queries |
| **Local Sandbox** | SQLite | Sentence-transformers | Claude API | ~$0.50 | Full end-to-end test |
| **Chunk 3** | Supabase ✅ | LLMod.ai | (N/A yet) | ~$0.03 | Prod embeddings |
| **Chunk 4** | Supabase ✅ | Pinecone | LLMod.ai | ~$12.97 | Prod LLM reasoning |

**Decision point:** After Chunk 2 (free), test with Local Sandbox before committing to Chunks 3-4 (paid).

---

## Architecture: Local vs. Production

```
┌─────────────────────────────────────────────────────────────────┐
│                         MoviBot Agent (react_loop.py)           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
           ┌───────────────────┼───────────────────┐
           ↓                   ↓                   ↓
      Reasoner            CatalogFilter        PlotSearch
    (LLM calls)           (SQL queries)       (Vector search)
           ↓                   ↓                   ↓
    ┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
    │  LOCAL SANDBOX   │ │  Supabase   │ │ LOCAL SANDBOX    │
    │  Claude API      │ │  (always)   │ │ Chroma/SQLite    │
    │  ~$0.50 total    │ │  (always)   │ │ + sentence-trans │
    └──────────────────┘ └─────────────┘ └──────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │         PRODUCTION (Chunks 3-4)                          │
    │  SceneSearch → LLMod.ai calls                           │
    │  ExternalContext → LLMod.ai calls                       │
    │  PlotSearch → Pinecone vectors                          │
    └──────────────────────────────────────────────────────────┘
```

---

## Setup: Local Sandbox

### Step 1: Install Dependencies

```bash
cd ~/tabstar/movibot
pip install -r requirements.txt

# Add local sandbox dependencies
pip install sentence-transformers chromadb torch  # or chroma[all]
```

### Step 2: Configure Local Mode

Create `config_local.py`:

```python
# config_local.py
import os

# Local Sandbox Settings
USE_LOCAL_SANDBOX = True

# Embeddings: use local sentence-transformers instead of LLMod.ai
EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"  # 384-dim, ~80 MB, fast
EMBEDDING_DIM_LOCAL = 384

# Vector store: use Chroma (in-memory or persistent)
VECTOR_STORE_TYPE = "chroma"  # or "sqlite" for pure SQL
CHROMA_DB_PATH = "./data_chroma"  # persisted locally

# LLM: use Claude API (paid per-token, ~$0.50 for 20 test queries)
# or use mock for zero cost
LLM_MODE = "claude"  # or "mock" for zero cost
CLAUDE_MODEL = "claude-opus-4.0"  # or any available model
```

### Step 3: Prepare Local Embeddings

```bash
# 1. Load Pinecone candidates from CSV
# 2. Generate embeddings with sentence-transformers
# 3. Store in Chroma (1-time setup, ~2 min)

python scripts/local_sandbox_setup.py \
  --input data_preprocessing/data_ready/pinecone_candidates.csv \
  --output data_chroma \
  --model all-MiniLM-L6-v2
```

### Step 4: Test the Agent Locally

```bash
python3 << 'EOF'
import config_local
from agent import react_loop

# Test query
query = "Find me a Disney movie from the 1990s"

result = react_loop.execute(query)
print(f"Response: {result['response']}")
print(f"\nSteps: {len(result['steps'])} modules run")
for step in result['steps']:
    print(f"  - {step['module']}")
EOF
```

---

## Implementation Details

### Option A: Chroma (Recommended)

**Pros:**
- No database setup needed
- Simple Python API
- In-memory or persistent
- Faster than SQLite for vectors

**Cons:**
- New dependency

```python
# agent/tools/plot_search_local.py
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./data_chroma")
collection = client.get_collection(name="movies")
model = SentenceTransformer("all-MiniLM-L6-v2")

def run(query: str, top_k: int = 20, **kwargs) -> list[dict]:
    """Local semantic search via Chroma + sentence-transformers."""
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Convert to same format as Pinecone version
    matches = []
    for i, doc_id in enumerate(results["ids"][0]):
        matches.append({
            "movie_id": results["metadatas"][0][i]["movie_id"],
            "title": results["metadatas"][0][i]["title"],
            "release_year": results["metadatas"][0][i]["release_year"],
            "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            "matched_terms": []  # Not used in local mode
        })
    return matches
```

### Option B: SQLite (Pure SQL)

**Pros:**
- No dependencies (SQLite built-in)
- Pure SQL, familiar to everyone

**Cons:**
- Slower for vector similarity (need brute-force)
- Requires table schema

```python
# agent/tools/plot_search_local_sqlite.py
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
db = sqlite3.connect("./data_movies_local.db")

def run(query: str, top_k: int = 20, **kwargs) -> list[dict]:
    """Local semantic search via SQLite + sentence-transformers."""
    # Get all vectors from DB
    cursor = db.execute(
        "SELECT movie_id, title, release_year, embedding FROM movies"
    )
    rows = cursor.fetchall()
    
    # Compute cosine similarity on device
    query_embedding = model.encode(query)
    similarities = []
    for row in rows:
        movie_id, title, year, emb_bytes = row
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        similarities.append((sim, movie_id, title, year))
    
    # Sort by similarity, take top-k
    similarities.sort(reverse=True)
    matches = [
        {
            "movie_id": m[1],
            "title": m[2],
            "release_year": m[3],
            "score": float(m[0]),
            "matched_terms": []
        }
        for m in similarities[:top_k]
    ]
    return matches
```

### Option C: LLM Reasoning (Claude API)

Replace mock LLM client with real Claude:

```python
# agent/llm_client.py - LOCAL_SANDBOX variant

import os
from anthropic import Anthropic

client = Anthropic()

def get_client():
    """Return Claude client (no OpenAI key needed)."""
    return client

class ClaudeLLMClient:
    def reason_next_action(self, user_prompt: str, gathered: dict) -> dict:
        """Use Claude to decide next action (Reasoner)."""
        response = client.messages.create(
            model="claude-opus-4.0",
            max_tokens=500,
            system_prompt=prompts.REASONER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        # Parse Claude's response as JSON
        return json.loads(response.content[0].text)
    
    def verify_scene_constraint(self, title: str, prompt: str, plot_text: str) -> dict:
        """Use Claude to check scene constraints (SceneSearch)."""
        response = client.messages.create(
            model="claude-opus-4.0",
            max_tokens=300,
            system_prompt=prompts.SCENE_SEARCH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Title: {title}\nPlot: {plot_text[:1000]}\nConstraint: {prompt}"}]
        )
        return json.loads(response.content[0].text)
    
    # ... similar for other methods
```

---

## Setup Script: `scripts/local_sandbox_setup.py`

```python
#!/usr/bin/env python3
"""Initialize local Chroma/SQLite with embeddings from pinecone_candidates.csv."""

import argparse
import json
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_preprocessing/data_ready/pinecone_candidates.csv")
    parser.add_argument("--output", default="data_chroma")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Loading embedding model {args.model}...")
    model = SentenceTransformer(args.model)
    
    print(f"Initializing Chroma at {args.output}...")
    client = chromadb.PersistentClient(path=args.output)
    collection = client.get_or_create_collection(name="movies")
    
    print(f"Embedding {len(df)} movies and storing in Chroma...")
    embeddings = model.encode(df["embedding_text"].tolist(), show_progress_bar=True)
    
    collection.upsert(
        ids=[str(row["movie_id"]) for _, row in df.iterrows()],
        embeddings=embeddings.tolist(),
        metadatas=[{
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "release_year": int(row["release_year"])
        } for _, row in df.iterrows()],
        documents=df["embedding_text"].tolist()
    )
    
    print(f"✅ Done! {len(df)} movies embedded and stored locally.")
    print(f"   Chroma DB: {args.output}")
    print(f"   Model: {args.model} ({model.get_sentence_embedding_dimension()}-dim)")

if __name__ == "__main__":
    main()
```

---

## Testing Workflow

### 1. Build Supabase (Chunk 2)
```bash
# Load 303 movies to Supabase
python scripts/ingest.py --supabase-only
```

### 2. Setup Local Sandbox
```bash
# Embed 170 movies locally
python scripts/local_sandbox_setup.py
```

### 3. Test Agent End-to-End (Free!)
```bash
# Run queries, iterate on prompts, verify architecture
python3 -c "from agent import react_loop; print(react_loop.execute('Find a 1990s Disney movie')['response'])"
```

### 4. Verify Results
- Check that `steps` trace shows all 6 modules (Reasoner → CatalogFilter → PlotSearch → SceneSearch → ExternalContext → Synthesizer)
- Verify module names match `assets/architecture.png`
- Confirm responses are sensible (not hallucinated)

### 5. Decide: Commit to Chunks 3-4?
- If results look good → proceed with Pinecone + LLMod.ai (Chunks 3-4)
- If results need tuning → iterate on prompts locally before spending budget

---

## Cost Comparison

| Phase | Embeddings Cost | LLM Cost | Total Cost | Time |
|-------|-----------------|----------|-----------|------|
| **Local Sandbox** | $0.00 (local) | ~$0.50 (Claude) | ~$0.50 | 5 min setup + testing |
| **Production (Ch 3-4)** | ~$0.03 (LLMod.ai) | ~$12.97 (LLMod.ai) | ~$13.00 | 1-2 hours |
| **Total (if full cycle)** | — | — | **~$13.50** | — |

**Verdict:** Spend $0.50 locally to catch issues before spending $13 on production.

---

## Model Choices: Local Embeddings

| Model | Dim | Speed | Accuracy | Size | Recommended |
|-------|-----|-------|----------|------|-------------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡ Fast | ⭐⭐⭐⭐ Good | 80 MB | ✅ YES |
| all-mpnet-base-v2 | 768 | Medium | ⭐⭐⭐⭐⭐ Better | 420 MB | Only if accuracy critical |
| all-DistilRoBERTa-v1 | 768 | Medium | ⭐⭐⭐ OK | 270 MB | If speed matters |
| sentence-t5-base | 768 | Slow | ⭐⭐⭐⭐ Good | 500 MB | Not recommended |

**Recommendation:** Start with `all-MiniLM-L6-v2` (fast, small, good accuracy). Upgrade to `all-mpnet-base-v2` if semantic search quality is poor.

---

## Debugging: Local vs. Production Differences

When switching from Local Sandbox → Production, watch for:

| Issue | Local Cause | Production Fix |
|-------|------------|-----------------|
| Different top-k results | Embedding model difference (MiniLM vs text-embedding-3-small) | Expected; retune prompts if needed |
| Slower responses | No GPU acceleration (sentence-transformers on CPU) | Pinecone + LLMod.ai will be faster |
| Different LLM reasoning | Claude vs GPT-5.4-mini | May need to retune system prompts |
| Hallucinations | Local Claude might invent movies | Production LLM will too; check data quality |

---

## Next: After Local Sandbox Validation

Once you've tested end-to-end locally and confirmed the agent works:

1. ✅ Chunk 2: Supabase loaded
2. ✅ Local Sandbox: Full agent tested
3. → **Proceed to Chunk 3:** Create Pinecone index + embed with LLMod.ai
4. → **Proceed to Chunk 4:** Swap Claude for LLMod.ai LLM calls, swap Chroma for Pinecone

---

## Files to Create/Modify

| File | Purpose | Status |
|------|---------|--------|
| `config_local.py` | Local sandbox configuration | New |
| `scripts/local_sandbox_setup.py` | Initialize Chroma DB with embeddings | New |
| `agent/tools/plot_search_local.py` | Chroma-based semantic search | New |
| `agent/llm_client.py` | Add `ClaudeLLMClient` variant | Modify |
| `.env` | Add `ANTHROPIC_API_KEY` (Claude, not LLMod.ai) | Add |

---

## Summary

**Local Sandbox = $0.50 to test everything before committing $13 to production.**

Test the full ReAct loop, iterate on prompts, verify the architecture, and then confidently deploy to Supabase + Pinecone + LLMod.ai.

