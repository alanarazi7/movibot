-- MoviBot Supabase schema for the `movies` table (CatalogFilter's data source).
-- Run this in the Supabase SQL editor once the project exists, before running
-- scripts/ingest.py. Columns match scripts/prepare_movibot_data.py's
-- data_ready/supabase_movies.csv output exactly.

create table if not exists movies (
  id bigint primary key,
  imdb_id text,
  title text not null,
  release_year integer,
  runtime_minutes integer,
  genres jsonb,
  production_companies jsonb,
  popularity real,
  overview text,
  keywords jsonb,
  has_mpst_synopsis boolean
);

create index if not exists movies_release_year_idx on movies (release_year);
create index if not exists movies_popularity_idx on movies (popularity);
create index if not exists movies_imdb_id_idx on movies (imdb_id);
