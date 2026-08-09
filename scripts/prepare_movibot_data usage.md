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

## 4. Outputs

The script creates exactly two final data files:

```text
data_ready/
├── supabase_movies.csv
└── pinecone_candidates.csv
```

### `supabase_movies.csv`

Contains **all usable cleaned Kaggle movies**.

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

Contains **all movies that have an exact IMDb-ID match between the cleaned Kaggle catalog and MPST**.

Nothing is removed to reach a fixed Pinecone size.

Rows are sorted by pure Kaggle popularity and receive:

```text
priority_rank = 1, 2, 3, ...
```

Columns:

```text
priority_rank
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

## 5. Building the demo Pinecone index later

For the current demo plan, use the first 3,000 ranked rows:

```python
import pandas as pd

df = pd.read_csv("data_ready/pinecone_candidates.csv")
df = df[df["priority_rank"] <= 3000]
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

With the dataset snapshots used during development, previous runs produced approximately:

```text
Raw Kaggle movies:        45,466
Clean Kaggle movies:      43,270
Raw MPST movies:          14,828
Exact Kaggle/MPST matches:11,328
```

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
