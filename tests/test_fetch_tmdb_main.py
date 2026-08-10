"""End-to-end test of fetch_tmdb_updates.main() with the network mocked out.

Covers the part that actually touches the catalog: which movies survive the
scope and usability filters, and that the CSV written is byte-compatible with
supabase_movies.csv so the two files can simply be concatenated.
"""

import csv

import pytest

from data_preprocessing import fetch_tmdb_updates as mod
from data_preprocessing.fetch_tmdb_updates import OUTPUT_COLUMNS, TmdbError


def detail(movie_id, title, year=2021, companies=None, **overrides):
    payload = {
        "id": movie_id,
        "imdb_id": f"tt{movie_id:07d}",
        "title": title,
        "original_title": title,
        "overview": f"Overview of {title}.",
        "release_date": f"{year}-06-17",
        "runtime": 95,
        "genres": [{"id": 16, "name": "Animation"}],
        "production_companies": companies
        if companies is not None
        else [{"id": 3, "name": "Pixar"}],
        "production_countries": [{"name": "United States of America"}],
        "spoken_languages": [{"name": "English"}],
        "belongs_to_collection": None,
        "popularity": 10.0,
        "vote_average": 7.0,
        "vote_count": 100,
        "budget": 0,
        "revenue": 0,
        "tagline": "",
        "status": "Released",
        "original_language": "en",
        "adult": False,
        "video": False,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def fake_tmdb(monkeypatch):
    """Wires main() to an in-memory TMDB. Returns the mutable movie table."""
    movies: dict[int, dict] = {}

    def fake_get_json(_session, path, params=None):
        if path.endswith("/keywords"):
            return {"keywords": [{"name": "friendship"}]}
        movie_id = int(path.rsplit("/", 1)[-1])
        return movies[movie_id]

    monkeypatch.setattr(mod, "build_session", lambda: object())
    monkeypatch.setattr(mod, "verify_company_ids", lambda _s: None)
    monkeypatch.setattr(mod, "discover_movie_ids", lambda _s, _since: iter(sorted(movies)))
    monkeypatch.setattr(mod, "get_json", fake_get_json)
    return movies


def run_main(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        mod.sys, "argv", ["fetch_tmdb_updates", "--out-dir", str(tmp_path), *extra]
    )
    return mod.main()


def read_output(tmp_path):
    with (tmp_path / "tmdb_new_movies.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestMain:
    def test_writes_rows_with_the_catalog_columns(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca")

        assert run_main(monkeypatch, tmp_path) == 0

        rows = read_output(tmp_path)
        assert len(rows) == 1
        assert list(rows[0].keys()) == OUTPUT_COLUMNS
        assert rows[0]["title"] == "Luca"
        assert rows[0]["production_companies"] == '["Pixar Animation Studios"]'

    def test_skips_movies_outside_the_studio_scope(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca")
        fake_tmdb[101] = detail(
            101, "Unrelated", companies=[{"id": 9, "name": "Warner Bros. Pictures"}]
        )

        assert run_main(monkeypatch, tmp_path) == 0

        assert [r["title"] for r in read_output(tmp_path)] == ["Luca"]

    def test_skips_unusable_movies(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca")
        fake_tmdb[101] = detail(101, "Unreleased", release_date="")
        fake_tmdb[102] = detail(102, "No overview", overview="")
        fake_tmdb[103] = detail(103, "No runtime", runtime=0)

        assert run_main(monkeypatch, tmp_path) == 0

        assert [r["title"] for r in read_output(tmp_path)] == ["Luca"]

    def test_rows_are_sorted_by_id(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[300] = detail(300, "Third")
        fake_tmdb[100] = detail(100, "First")
        fake_tmdb[200] = detail(200, "Second")

        run_main(monkeypatch, tmp_path)

        assert [r["id"] for r in read_output(tmp_path)] == ["100", "200", "300"]

    def test_limit_caps_the_number_fetched(self, monkeypatch, tmp_path, fake_tmdb):
        for i in range(5):
            fake_tmdb[100 + i] = detail(100 + i, f"Movie {i}")

        run_main(monkeypatch, tmp_path, "--limit", "2")

        assert len(read_output(tmp_path)) == 2

    def test_keywords_are_attached(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca")

        run_main(monkeypatch, tmp_path)

        assert read_output(tmp_path)[0]["keywords"] == '["friendship"]'

    def test_empty_result_still_writes_a_header_only_file(
        self, monkeypatch, tmp_path, fake_tmdb
    ):
        assert run_main(monkeypatch, tmp_path) == 0

        assert read_output(tmp_path) == []
        assert (tmp_path / "tmdb_new_movies.csv").exists()

    def test_malformed_since_is_rejected(self, monkeypatch, tmp_path, fake_tmdb):
        assert run_main(monkeypatch, tmp_path, "--since", "July 2017") == 2

    def test_credential_failure_exits_nonzero_without_writing(
        self, monkeypatch, tmp_path, fake_tmdb
    ):
        def boom():
            raise TmdbError("No TMDB credentials found.")

        monkeypatch.setattr(mod, "build_session", boom)

        assert run_main(monkeypatch, tmp_path) == 1
        assert not (tmp_path / "tmdb_new_movies.csv").exists()

    def test_drops_shorts_below_the_feature_length_threshold(
        self, monkeypatch, tmp_path, fake_tmdb
    ):
        # TMDB lists Pixar SparkShorts and "Forky Asks a Question" episodes as
        # standalone movies. They are 3-9 minutes long and would pollute any
        # "recommend me a movie" answer.
        fake_tmdb[100] = detail(100, "Luca", runtime=95)
        fake_tmdb[101] = detail(101, "Forky Asks a Question: What Is Money?", runtime=3)
        fake_tmdb[102] = detail(102, "Bao", runtime=8)

        run_main(monkeypatch, tmp_path)

        assert [r["title"] for r in read_output(tmp_path)] == ["Luca"]

    def test_min_runtime_zero_keeps_everything(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca", runtime=95)
        fake_tmdb[101] = detail(101, "Bao", runtime=8)

        run_main(monkeypatch, tmp_path, "--min-runtime", "0")

        assert len(read_output(tmp_path)) == 2

    def test_min_runtime_is_configurable(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca", runtime=95)
        fake_tmdb[101] = detail(101, "Mid-length special", runtime=45)

        run_main(monkeypatch, tmp_path, "--min-runtime", "60")

        assert [r["title"] for r in read_output(tmp_path)] == ["Luca"]

    def test_negative_min_runtime_is_rejected(self, monkeypatch, tmp_path, fake_tmdb):
        assert run_main(monkeypatch, tmp_path, "--min-runtime", "-5") == 2

    def test_creates_the_output_directory(self, monkeypatch, tmp_path, fake_tmdb):
        fake_tmdb[100] = detail(100, "Luca")
        nested = tmp_path / "deep" / "nested"

        assert run_main(monkeypatch, nested) == 0
        assert (nested / "tmdb_new_movies.csv").exists()
