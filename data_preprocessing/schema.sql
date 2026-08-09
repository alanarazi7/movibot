-- MoviBot Supabase schema for the `movies` table (CatalogFilter's data source).
-- Run this in the Supabase SQL editor once the project exists, before running
-- scripts/ingest.py. Columns match data_preprocessing/prepare_movibot_data.py's
-- data_preprocessing/data_ready/supabase_movies.csv output exactly.

create table if not exists movies (
  id bigint primary key,
  imdb_id text,
  title text not null,
  original_title text,
  release_year integer,
  release_date date,
  runtime_minutes integer,
  genres jsonb,
  production_companies jsonb,
  production_countries jsonb,
  spoken_languages jsonb,
  belongs_to_collection text,
  popularity real,
  vote_average real,
  vote_count integer,
  budget bigint,
  revenue bigint,
  overview text,
  tagline text,
  status text,
  original_language text,
  adult boolean,
  video boolean,
  poster_path text,
  homepage text,
  keywords jsonb,
  has_mpst_synopsis boolean
);

create index if not exists movies_release_year_idx on movies (release_year);
create index if not exists movies_popularity_idx on movies (popularity);
create index if not exists movies_imdb_id_idx on movies (imdb_id);
