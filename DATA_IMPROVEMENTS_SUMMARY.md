# MoviBot Data Improvements (Chunk 1, 2026-08-11/12)

## Overview

Implemented two data-discovery and data-caching improvements to resolve open design decisions and address sources named in the original project proposal. All work is offline and free (Chunk 1).

## Part A: Movie Transcripts Coverage Discovery

### What Was Done

Created `data_preprocessing/find_transcripts.py` to discover which catalog movies have transcripts available in a public dataset.

**Data Source:** HuggingFace `mocboch/movie_scripts` (~423 scripts)

**Coverage Results:**
- **10 of 303 catalog movies matched (3.3%)**
  - Finding Nemo, Tron, Aladdin, Toy Story, Mulan
  - Up, Newsies, Frankenweenie, Saving Mr. Banks, Into the Woods

**Output:** `data_preprocessing/data_ready/transcript_matches.csv`
- Columns: `id`, `title`, `imdb_id`, `transcript_found` (bool), `transcript_source_file`
- Size: 13 KB
- Committed to repo

### Usage Status

**Not yet integrated.** Transcripts are kept out of the pipeline because:
1. Full scripts require many chunks per movie (expensive for semantic indexing)
2. MPST synopses (170 movies, 56% coverage) already provide rich plot text
3. Original design rationale (data cleaning rules.md §11) stands: complex, costly, lower priority

**Future Enhancement Opportunity:**
The 10 matched titles could supplement `SceneSearch` verification (scene-level precision on deaths/scary content). Requires: transcript text loading, document chunking, separate embedding index. Not in scope for current Chunk 1 but listed as a reference.

## Part B: Wikipedia Pre-Caching (Resolves Open Design Decision)

### What Was Done

Created `data_preprocessing/scrape_wikipedia.py` to replace live Wikipedia fetches with offline pre-cache. Resolves the `TODO.md` open design decision: live-fetch vs. pre-indexing → **chosen: pre-indexing**.

**Coverage Results:**
| Metric | Count | % |
|--------|-------|------|
| Pages found | 287 | 94.7% |
| With Plot section | 213 | 70.3% |
| With non-Plot text | 259 | 85.5% |

**Output:** `data_preprocessing/data_ready/wikipedia_cache.csv`
- Columns: `id`, `title`, `imdb_id`, `wiki_page_found` (bool), `plot_text`, `non_plot_text`
- Size: 1.6 MB (comparable to `pinecone_candidates.csv`)
- Committed to repo

### Benefits

1. **No live API calls at agent runtime** — faster queries, no rate limits, reproducible
2. **Offline verification** — agent can run without external network dependency (except model calls in Chunk 4)
3. **Resilient** — cached text is permanent; Wikipedia changes won't affect results
4. **One-time scrape cost** — 303 movies × 0.5s polite delay ≈ 2.5 min, done once

### Technical Improvements

**Fixed wikipedia_client.py:**
- Added User-Agent header (Wikipedia blocks requests without it)
- Enhanced `fetch_page_extract()` to accept `release_year` parameter
- Tries multiple title variants (exact, with year, with "(film)", etc.) for better disambiguation

**Updated agent tools:**
- `agent/tools/scene_search.py`: prefers MPST synopsis → Wikipedia cache → live Wikipedia
- `agent/tools/external_context.py`: prefers Wikipedia cache → live Wikipedia
- Both maintain fallback to live Wikipedia if cache is unavailable (graceful degradation)

## Documentation Updates

### DATA_SOURCES.md
- **§3 Wikipedia:** Reframed as "pre-cached offline" (was: live API)
- **§4 Movie Transcripts (new):** Discovery results, coverage stats, future enhancement notes
- Section numbers updated (4→6 Supabase, 5→7 Pinecone, etc.)

### TODO.md
- ✅ Marked **Wikipedia pre-indexing** resolved (offline pre-cache)
- 🔄 Added **Movie Transcripts** discovery results (coverage: 10/303 = 3.3%)

## Verification

All tools tested and working:

1. **Transcript discovery:**
   ```bash
   python3 data_preprocessing/find_transcripts.py
   # Output: transcript_matches.csv with 10 matches
   ```

2. **Wikipedia pre-cache:**
   ```bash
   python3 data_preprocessing/scrape_wikipedia.py
   # Output: wikipedia_cache.csv with 287/303 pages
   ```

3. **Agent tools with cache:**
   ```python
   from agent.tools import scene_search
   result = scene_search.run("Frozen", "no deaths")
   # Returns: {"satisfied": False, "evidence": "...from Wikipedia cache..."}
   ```

4. **Full mock agent:**
   ```python
   from agent.react_loop import execute
   result = execute("Find me a 3-year-old-friendly Pixar movie")
   # Returns: 20 matching movies with full ReAct trace
   ```

## Next Steps (for user review)

### Before Chunk 2 (Supabase)
- Spot-check a few cache entries (e.g., *Frozen*, *Toy Story*) against live Wikipedia
- Decide: use transcripts in future work? (can defer decision; data is now discoverable)

### Optional Polish
- Add `find_transcripts.py` and `scrape_wikipedia.py` to a `scripts/` makefile or README for one-command regeneration
- Document cache regeneration (if raw Kaggle datasets are re-downloaded and cleaned)

### Chunk 2 Readiness
All Chunk 1 offline data is complete and committed:
- ✅ `supabase_movies.csv` (303 movies, 25 columns)
- ✅ `pinecone_candidates.csv` (170 with MPST synopses)
- ✅ `wikipedia_cache.csv` (287 with pages)
- ✅ `transcript_matches.csv` (10 with transcripts found)
- Ready for Chunk 2: Supabase schema & ingest

## Files Modified/Created

**New:**
- `data_preprocessing/find_transcripts.py` (script)
- `data_preprocessing/scrape_wikipedia.py` (script)
- `data_preprocessing/data_ready/transcript_matches.csv` (data)
- `data_preprocessing/data_ready/wikipedia_cache.csv` (data)
- `DATA_IMPROVEMENTS_SUMMARY.md` (this file)

**Modified:**
- `agent/tools/wikipedia_client.py` (User-Agent, year parameter)
- `agent/tools/scene_search.py` (cache lookup)
- `agent/tools/external_context.py` (cache lookup)
- `DATA_SOURCES.md` (sections 3-7 updated, new §4)
- `TODO.md` (design decisions resolved, findings added)

## Commit

```
commit 85efe74
Author: Claude Haiku 4.5 <noreply@anthropic.com>
Date:   2026-08-12

    Improve MoviBot data state: transcripts discovery + Wikipedia pre-cache

    Part A: Transcript coverage discovery
    Part B: Wikipedia pre-caching (resolves TODO.md open design decision)
    All work in Chunk 1 (offline, free).
```
