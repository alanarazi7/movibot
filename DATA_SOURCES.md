# Data Sources

MoviBot uses three main data sources:

## 1. Kaggle: The Movies Dataset

**Source:** [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  
**Owner:** Rounak Banik  
**License:** CC0 (Public Domain)  
**Size:** ~5 MB (compressed)

### Files Used
- `movies_metadata.csv` — 45,466 movies with 24 columns (title, release_year, overview, genres, production_companies, etc.)
- `keywords.csv` — Plot keywords indexed by movie_id

### How to Download

1. **Install Kaggle CLI**
   ```bash
   pip install kaggle
   ```

2. **Authenticate**
   - Go to https://www.kaggle.com/settings/account
   - Click "Create new API token" → downloads `kaggle.json`
   - Move to `~/.kaggle/kaggle.json` and `chmod 600`

3. **Download to `data_preprocessing/data_full/`**
   ```bash
   cd ~/tabstar/movibot/data_preprocessing/data_full
   kaggle datasets download -d rounakbanik/the-movies-dataset
   unzip the-movies-dataset.zip
   ```

### In This Project
- Narrowed to **Disney + Pixar demo scope** (304 movies → 303 after cleaning)
- All 25 columns kept: `id`, `imdb_id`, `title`, `release_year`, `runtime_minutes`, `overview`, `genres`, `production_companies`, `keywords`, `vote_average`, etc.
- Used by: `CatalogFilter` tool (structured queries: year, studio, genre, runtime)
- Output: `data_preprocessing/data_ready/supabase_movies.csv` (committed)

---

## 2. Kaggle: MPST (Movie Plot Synopses with Tags)

**Source:** [MPST Movie Plot Synopses with Tags](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags)  
**Owner:** CrypTex Code  
**License:** CC0 (Public Domain)  
**Size:** ~20 MB

### Files Used
- `mpst_full_data.csv` — Plot synopses (56,216 rows) with IMDb ID, Wikipedia URL, and tags

### How to Download

1. **Use Kaggle CLI** (same auth as above)
   ```bash
   cd ~/tabstar/movibot/data_preprocessing/data_full
   kaggle datasets download -d cryptexcode/mpst-movie-plot-synopses-with-tags
   unzip mpst-movie-plot-synopses-with-tags.zip
   ```

### Why This Source
- **Richer plots:** Median ~693 words vs. Kaggle's ~48 word overviews
- **Exact ID matching:** Matched to Kaggle movies by IMDb ID
- **Coverage:** 170 of 303 Disney + Pixar movies have MPST synopses (56%)

### In This Project
- Exact IMDb ID match with Kaggle movies
- Skipped `review` column (irrelevant, ~50 MB alone)
- Kept: `plot`, `tags`, IMDb ID for matching
- Used by: `PlotSearch` tool (semantic search on plot text)
- Output: `data_preprocessing/data_ready/pinecone_candidates.csv` (committed, 170 rows)

---

## 3. Wikipedia (Live API)

**Source:** Wikipedia REST API  
**URL:** `https://en.wikipedia.org/api/rest_v1/`  
**License:** CC-BY-SA 3.0  
**Used by:** `SceneSearch` & `ExternalContext` tools

### How It Works

1. **SceneSearch Tool:** Fetches Wikipedia plot section for each movie candidate
   - Looks for "Plot" section in movie's Wikipedia page
   - Searches for keywords: "dies", "death", "killed", "scary", "horror"
   - Returns: true/false/null for "does this movie satisfy the constraint?"
   - Example: "Find movies with no deaths" → checks plot text for death mentions

2. **ExternalContext Tool:** Fetches Wikipedia reception/audience info
   - Looks for "Reception", "Themes", "Audience", "Critical response" sections
   - Searches for tone descriptors: "lighthearted", "dark", "intense", "family-friendly"
   - Returns: true/false/null for subjective constraints

### Current Implementation (Mock)
- **Location:** `agent/tools/wikipedia_client.py`
- **Mock behavior:** Returns deterministic results based on keywords in plot text
- **Real implementation (Chunk 4):** Will fetch live Wikipedia pages via REST API

### Example Query
```python
from agent.tools.wikipedia_client import fetch_movie_plot

plot = fetch_movie_plot("Frozen")
# Returns: Wikipedia plot section (or None if not found)

# SceneSearch checks for deaths:
if "dies" in plot.lower():
    satisfied = False  # Has deaths
else:
    satisfied = True   # No deaths found
```

---

## 4. LLMod.ai (Embeddings & LLM)

**Service:** LLMod.ai (OpenAI-compatible endpoint)  
**Used for:**
- **Chunk 3:** Text embeddings (text-embedding-3-small, 1536-dim) for Pinecone
- **Chunk 4:** LLM reasoning (GPT-5.4-mini) for Reasoner, SceneSearch, ExternalContext, Synthesizer

### Authentication
- **API Key:** `OPENAI_API_KEY` (in `.env`, gitignored)
- **Base URL:** `OPENAI_BASE_URL` (in `.env`, gitignored)
- **Never commit:** Keys stay in `.env.example` only as placeholders

### Cost Estimate
- **Embeddings:** 170 movies × $0.00002/1k tokens ≈ $0.003
- **LLM calls:** ~3 calls/query × 20 queries × $0.00015/1k tokens ≈ $9.00
- **Total budget:** $13.00 (with buffer)

---

## 5. Supabase (Structured Data)

**Service:** Supabase (PostgreSQL + REST API)  
**Purpose:** Store 303 movies for CatalogFilter queries (year, studio, genre, runtime)  
**Status:** Project created, table schema ready (`data_preprocessing/schema.sql`)

### Schema (Chunk 2 Deliverable)
```sql
CREATE TABLE movies (
  id INT PRIMARY KEY,
  imdb_id INT UNIQUE,
  title VARCHAR(255),
  release_year INT,
  runtime_minutes FLOAT,
  overview TEXT,
  genres JSON,
  production_companies JSON,
  keywords JSON,
  has_mpst_synopsis BOOLEAN,
  ... (25 columns total)
);
```

### Used by
- `CatalogFilter` tool: SQL queries like `SELECT * FROM movies WHERE release_year >= 1990 AND production_companies LIKE '%Disney%'`

---

## 6. Pinecone (Vector Search)

**Service:** Pinecone  
**Purpose:** Semantic search on 170 plot synopses (1536-dim vectors from text-embedding-3-small)  
**Status:** Index not yet created (Chunk 3)

### Index Config
```
Name: movibot-plots
Metric: cosine
Dimension: 1536
Serverless: true (cheapest)
Metadata: {movie_id, title, release_year}
```

### Used by
- `PlotSearch` tool: Find movies matching thematic queries (e.g., "involves animals", "lighthearted tone")

---

## Data Flow

```
Kaggle Movies + MPST
        ↓
data_preprocessing/prepare_movibot_data.py
        ↓
   ┌────┴────┐
   ↓         ↓
supabase_movies.csv  pinecone_candidates.csv
(303 rows, 25 cols)  (170 rows + plot text)
   ↓                 ↓
Chunk 2: Load to    Chunk 3: Embed &
Supabase 'movies'   index in Pinecone
   ↓                 ↓
CatalogFilter      PlotSearch
(SQL queries)      (cosine similarity)
   ↓                 ↓
   └────┬────┐      └────┬────┐
        ↓              ↓
   Reasoner → decides next action
        ↓
   SceneSearch ← fetches Wikipedia
   ExternalContext ← fetches Wikipedia
        ↓
   Synthesizer → final answer
```

---

## Storage & Gitignore

| Path | Contents | Gitignored | Size | Regenerable |
|------|----------|-----------|------|-------------|
| `data_preprocessing/data_full/` | Raw Kaggle CSVs | ✅ Yes | 113 MB | Yes (via `kaggle` CLI) |
| `data_preprocessing/data_ready/` | Cleaned CSVs | ❌ No | 2.9 MB | Yes (via `prepare_movibot_data.py`) |
| `.env` | API keys (Supabase, LLMod.ai, Pinecone) | ✅ Yes | <1 KB | No (must obtain manually) |

---

## To Regenerate Data

```bash
# 1. Download raw Kaggle sources to data_full/
cd ~/tabstar/movibot/data_preprocessing/data_full
kaggle datasets download -d rounakbanik/the-movies-dataset
kaggle datasets download -d cryptexcode/mpst-movie-plot-synopses-with-tags
unzip -q "*.zip"

# 2. Run the cleaning pipeline
cd ~/tabstar/movibot
python data_preprocessing/prepare_movibot_data.py --all-studios  # Full catalog
python data_preprocessing/prepare_movibot_data.py                # Disney + Pixar only (default)

# 3. Output appears in data_preprocessing/data_ready/
ls -lh data_preprocessing/data_ready/
```

---

## References

- **Kaggle Datasets:**
  - The Movies Dataset: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
  - MPST: https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags

- **APIs:**
  - Wikipedia REST API: https://en.wikipedia.org/api/rest_v1/
  - OpenAI API (via LLMod.ai): https://openai.com/api/

- **Services:**
  - Supabase: https://supabase.io/
  - Pinecone: https://www.pinecone.io/
  - LLMod.ai: https://llmod.ai/

