# Data Sources

## Original Proposal: Three Candidate Categories

The team's proposal identified three broad data-source patterns. Here's what we investigated and adopted:

| Category | Candidate Source | Investigated | Adopted | Finding |
|----------|------------------|--------------|---------|---------|
| **Movies Database** | Kaggle "The Movies Dataset" | ✅ | ✅ Yes | 303 Disney+Pixar movies, 25 columns. Primary source for structured queries. |
| **Movies Database** | HuggingFace tmdb-5000 (13MB) | ❌ | ❌ No | Not investigated; Kaggle source already sufficient. |
| **Movies Database** | Kaggle IMDB datasets (241–312MB) | ❌ | ❌ No | Not investigated; Kaggle source already sufficient. |
| **Movie Transcripts** | HuggingFace mocboch/movie_scripts (90MB) | ✅ | ❌ No | Only 10/303 coverage (3.3%) — too sparse to be useful. |
| **Movie Transcripts** | Kaggle fayaznoor10 (2.28GB) | ❌ | ❌ No | Not investigated (mocboch result was too low; larger corpus unlikely to help Disney+Pixar). |
| **Movie Transcripts** | Kaggle gufukuro (2.22GB) | ❌ | ❌ No | Not investigated. |
| **Movie Transcripts** | Kaggle ismaeldwikat (246MB) | ❌ | ❌ No | Not investigated. |
| **Wikipedia/Wikidata** | Wikipedia REST API | ✅ | ✅ Yes | 287/303 pages (94.7%), 213/303 Plot sections (70.3%). Pre-cached offline. |
| **Wikipedia/Wikidata** | Wikidata API | ❌ | ❌ No | Not investigated; Wikipedia alone sufficient. |

## Detailed: What We Found

### ✅ Adopted Sources

1. **Kaggle "The Movies Dataset"** — primary structured catalog
   - 303 Disney+Pixar movies after cleaning
   - 25 columns (id, title, year, genres, companies, keywords, etc.)
   - Used by: `CatalogFilter` tool (Supabase)

2. **Kaggle MPST (Movie Plot Synopses)** — rich plot text
   - 170 of 303 movies matched (56% coverage)
   - Median 693-word synopses (vs. 48-word Kaggle overviews)
   - Used by: `PlotSearch` tool (Pinecone embeddings)

3. **Wikipedia** — verification & context (pre-cached)
   - 287/303 pages found (94.7%)
   - Plot sections: 213/303 (70.3%)
   - Non-Plot text (Reception, Themes): 259/303 (85.5%)
   - Used by: `SceneSearch` & `ExternalContext` tools (cached offline)

### ❌ Rejected Sources

1. **Movie Transcripts (HuggingFace mocboch/movie_scripts)**
   - Coverage: 10/303 (3.3%) — **too sparse**
   - Matched movies: Finding Nemo, Tron, Aladdin, Toy Story, Mulan, Up, Newsies, Frankenweenie, Saving Mr. Banks, Into the Woods
   - Why rejected: Low coverage makes per-movie transcript enhancement unviable at current scope
   - Future: Could revisit if scope expands beyond Disney+Pixar demo

---

## Current Data Sources in Use

MoviBot currently uses three main data sources:

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

## 3. Wikipedia (Pre-Cached Offline)

**Source:** Wikipedia REST API (scraped once, cached offline)  
**License:** CC-BY-SA 3.0  
**Cache location:** `data_preprocessing/data_ready/wikipedia_cache.csv`  
**Used by:** `SceneSearch` & `ExternalContext` tools

### How It Works

1. **SceneSearch Tool:** Uses cached Wikipedia plot section for each movie candidate
   - Looks for "Plot" section in movie's pre-cached Wikipedia page
   - Searches for keywords: "dies", "death", "killed", "scary", "horror"
   - Returns: true/false/null for "does this movie satisfy the constraint?"
   - Fallback: If cache miss, fetches live Wikipedia (same as before)

2. **ExternalContext Tool:** Uses cached Wikipedia reception/audience info
   - Looks for "Reception", "Themes", "Audience", "Critical response" sections in cache
   - Searches for tone descriptors: "lighthearted", "dark", "intense", "family-friendly"
   - Returns: true/false/null for subjective constraints
   - Fallback: If cache miss, fetches live Wikipedia

### Pre-Caching Strategy
- **Scraper:** `data_preprocessing/scrape_wikipedia.py` fetches all 303 catalog movies once
- **Cache format:** CSV with columns `id`, `title`, `imdb_id`, `wiki_page_found`, `plot_text`, `non_plot_text`
- **Coverage:** ~75% of 303 movies have Wikipedia pages; ~50% have explicit Plot sections
- **Size:** Comparable to `pinecone_candidates.csv` (~3 MB)
- **Benefit:** No live Wikipedia calls at agent runtime (faster, no rate limits, reproducible)

### Example Query (via cache)
```python
from agent.tools import scene_search

result = scene_search.run("Frozen", "no deaths")
# Returns: {
#   "title": "Frozen",
#   "constraint": "no deaths",
#   "satisfied": true,
#   "evidence": "... plot synopsis text from Wikipedia cache ..."
# }
```

---

## 4. Movie Transcripts (Coverage Discovery)

**Source:** HuggingFace `mocboch/movie_scripts` dataset  
**License:** Dataset-specific (verify before use)  
**Size:** ~423 movie scripts in HF; 10 of 303 catalog movies matched (3.3% coverage)  
**Match report:** `data_preprocessing/data_ready/transcript_matches.csv`  

### Coverage & Matched Movies

From the 303 Disney/Pixar catalog:
- **10 movies with transcripts found:**
  - Finding Nemo, Tron, Aladdin, Toy Story, Mulan, Up, Newsies, Frankenweenie, Saving Mr. Banks, Into the Woods
- **93% of catalog has no matching transcript** in this source

### Why Transcripts Are Not Yet Used

The original design explicitly avoids full-script inclusion because:
- Scripts require many chunks/vectors per movie (expensive for semantic indexing)
- MPST synopses (170 movies, 56% coverage) already provide rich plot text for `PlotSearch`
- Live Wikipedia Plot sections (fallback) handle `SceneSearch` verification cases

### Future Enhancement Opportunity

For the 10 matched titles, transcripts could supplement:
- **SceneSearch:** Verify scene-level constraints (deaths, scary moments) with more precision than synopsis
- **PlotSearch:** Richer dialogue/character analysis for subjective themes ("lighthearted", "family-friendly")

Requires: (1) Transcript text loaded into `data_full/`, (2) Document chunking strategy, (3) Separate embedding index if pursuing semantic search over transcripts. **Not in scope for Chunk 1 (data preparation) — listed here for future reference.**

---

## 5. LLMod.ai (Embeddings & LLM)

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

## 6. Supabase (Structured Data)

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

## 7. Pinecone (Vector Search)

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

## Investigation & Decisions (2026-08-11/12)

### Why We Adopted Wikipedia (§3)

**Decision:** Pre-cache Wikipedia offline instead of live fetches.

**Rationale:**
- High coverage: 94.7% of catalog has Wikipedia pages; 70.3% have explicit Plot sections
- Verification use case perfect for cached text (risky-event detection: deaths, scary content, tone)
- No live API dependency at agent runtime → faster, no rate limits, reproducible
- Pre-cache cost: one-time scrape (~2.5 min), then reusable offline

**Implementation:** `data_preprocessing/scrape_wikipedia.py` produces `wikipedia_cache.csv`, used by `SceneSearch` and `ExternalContext` tools with fallback to live fetch if needed.

---

### Why We Rejected Movie Transcripts (§4)

**Decision:** Transcripts not adopted for current pipeline.

**Coverage Finding:**
- Searched: HuggingFace `mocboch/movie_scripts` (423 scripts)
- Result: 10 of 303 catalog movies (3.3%) — **too sparse**
- Matched: Finding Nemo, Tron, Aladdin, Toy Story, Mulan, Up, Newsies, Frankenweenie, Saving Mr. Banks, Into the Woods

**Rationale for Rejection:**
1. **Low coverage:** 3.3% means 293 movies would need fallback (synopsis or Wikipedia)
2. **Complexity:** Transcripts require chunking & separate embedding strategy (vs. one vector per movie via MPST)
3. **Cost:** Multi-chunk embedding per movie would exceed Pinecone budget for minimal marginal benefit
4. **Data quality:** MPST synopses (56% coverage) already provide rich plot text; transcripts add little signal for the demo scope

**Future Consideration:**
If scope expands beyond Disney+Pixar (e.g., full ~43K catalog), larger transcript corpora may become viable, especially for fine-grained constraint verification (scene-level deaths, character-specific content).

**Note:** Transcript coverage data preserved in `transcript_matches.csv` for future reference.

---

### Why We Didn't Adopt Other Database Variants

**HuggingFace tmdb-5000-movies (13 MB, 4,803 movies):**

Investigated in detail. **Unique fields not in our data:**
- **Cast** — structured list of actors & character names
- **Crew** — structured list of directors, writers, producers, etc.
- Better **budget/revenue coverage**: 78% & 70% (vs. our 54% & 51%)

**Potential value:**
- Actor/director filtering: "movies with [actor name]", "directed by [director]"
- Collaborative filtering ("if you like actor X...")
- Financial-based queries ("big-budget blockbusters")

**Why not adopted:**
- Our demo scope (Disney+Pixar, 303 movies, fixed set) has no actor/director queries
- Queries focus on: kid-friendliness, no deaths, family-suitable tone — none require cast/crew data
- Cast/crew would be **highly valuable for full catalog** (43K+ movies) but overkill for demo

**Future consideration:** If expanding beyond Disney+Pixar demo, TMDB-5000 becomes attractive for cast/crew filtering.

**Kaggle IMDB datasets** (ahmedosamamath 312MB, raedaddala 241MB):
- Not investigated in detail (large downloads without clear upside)
- Likely overlap significantly with existing Kaggle "The Movies Dataset"
- Would need explicit review if project moves beyond demo scope

**Wikidata:**
- Wikipedia free-text provides sufficient context for SceneSearch/ExternalContext tasks
- Wikidata structured fields (ratings, awards, cast) not required for current agent tool set
- Can be added later if filtering on those fields becomes a requirement

---

## Summary: Current Data Pipeline

**Adopted sources (in order of query):**

1. **Kaggle "The Movies Dataset"** → `supabase_movies.csv` (303 movies, Supabase, structured filters)
2. **Kaggle MPST synopses** → `pinecone_candidates.csv` (170 movies, Pinecone, semantic search)
3. **Wikipedia (pre-cached)** → `wikipedia_cache.csv` (287 movies, agent verification)

**Rejected sources:**
- Movie Transcripts: 3.3% coverage (too sparse)
- Other movie databases: redundant with Kaggle source
- Wikidata: not required for current tasks

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

