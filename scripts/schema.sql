-- MoviBot Supabase schema for the `movies` table (CatalogFilter's data source).
-- Run this in the Supabase SQL editor once the project exists, before running
-- scripts/ingest.py.

create table if not exists movies (
  id bigint primary key,
  title text not null,
  release_year integer,
  runtime_minutes integer,
  genres jsonb,
  production_companies jsonb,
  popularity real,
  overview text
);

create index if not exists movies_release_year_idx on movies (release_year);
create index if not exists movies_popularity_idx on movies (popularity);
