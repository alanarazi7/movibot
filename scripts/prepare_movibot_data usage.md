# MoviBot Data Package — How to Run

This folder contains the final offline preparation pipeline for the MoviBot course project.

## 1. Required raw files

Place these **three original CSVs** in `data_full/`:

```text
project/
├── prepare_movibot_data.py
└── data_full/
    ├── movies_metadata.csv
    ├── keywords.csv
    └── mpst_full_data.csv
```

Sources:

- `movies_metadata.csv` — Kaggle *The Movies Dataset*
- `keywords.csv` — Kaggle *The Movies Dataset*
- `mpst_full_data.csv` — MPST movie plot synopsis dataset

No ratings, credits, links, or review columns are needed.

## 2. Python environment

The script only requires pandas.

With the project virtual environment activated:

```powershell
python -m pip install pandas
```

Or install it explicitly through the project environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas
```

## 3. Run

From the project root:

```powershell
.\.venv\Scripts\python.exe prepare_movibot_data.py
```

Mac/Linux:

```bash
./.venv/bin/python prepare_movibot_data.py
```

Optional custom paths:

```bash
python prepare_movibot_data.py --data-full data_full --out-dir data_ready
```

By default the script narrows to a demo scope: only movies produced by
Walt Disney Pictures, Walt Disney Animation Studios, or Pixar Animation
Studios (`DEMO_STUDIOS` in the script) — including ones with no MPST
synopsis. Pass `--all-studios` to skip this and keep the full multi-studio
catalog (the original, much larger, output):

```bash
python prepare_movibot_data.py --all-studios
```

## 4. Outputs

The script creates exactly two final data files:

```text
data_ready/
├── supabase_movies.csv
└── pinecone_candidates.csv
```

### `supabase_movies.csv`

Contains **all usable cleaned Kaggle movies within the demo studio scope**
(all 303 Disney + Pixar movies by default — including ones with no MPST
synopsis; pass `--all-studios` for the full 43,270-movie catalog instead).

Columns:

```text
id
imdb_id
title
release_year
runtime_minutes
genres
production_companies
popularity
overview
keywords
has_mpst_synopsis
```

This is the source file for the Supabase catalog.

`genres`, `production_companies`, and `keywords` are JSON arrays stored as CSV strings.

### `pinecone_candidates.csv`

Contains every movie **within the demo studio scope** that also has an
exact IMDb-ID match to MPST (170 of the 303 Disney + Pixar movies —
56% coverage). At this size the whole file is meant to be embedded in
full, so there's no ranking/cutoff column; rows are simply sorted by
descending Kaggle popularity for readability.

Columns:

```text
movie_id
imdb_id
title
release_year
popularity
genres
production_companies
synopsis_source
embedding_text
```

`embedding_text` is the text that should later be sent to the embedding model.

It contains:

```text
Title
MPST plot synopsis
MPST story tags
Kaggle keywords
```

## 5. Building the Pinecone index

At demo scope (170 rows), embed the whole file — no subsampling needed:

```python
import pandas as pd

df = pd.read_csv("data_ready/pinecone_candidates.csv")
```

Do **not** permanently store `embedding_text` in Pinecone metadata.

Recommended Pinecone metadata:

```python
{
    "movie_id": ...,
    "title": ...,
    "release_year": ...
}
```

The workflow should be:

```text
pinecone_candidates.csv
        |
        | embedding_text
        v
text-embedding-3-small
        |
        v
1536-D vector
        |
        v
Pinecone
(vector + compact metadata)
```

The local synopsis text can remain local after vectors are generated.

## 6. Expected sanity checks

With the dataset snapshots used during development, a default (demo-scope) run produces approximately:

```text
Raw Kaggle movies:               45,466
Clean Kaggle movies:              43,270
  ...within demo studio scope:      303
Raw MPST movies:                 14,828
Exact matches within scope:         170  (56% coverage)
```

Running with `--all-studios` reproduces the original full-catalog numbers instead
(43,270 clean movies, 11,328 exact MPST matches).

The exact console output from this script is the authority for the files on your machine.

If these numbers differ significantly, check that the expected dataset versions were downloaded.

## 7. Important

- The script does not call Supabase.
- The script does not call Pinecone.
- The script does not call LLMod.ai/OpenAI.
- The script does not call Wikipedia.
- The script does not generate embeddings.
- Running it is free/offline after the CSVs have been downloaded.
- Raw files in `data_full/` are never modified.
