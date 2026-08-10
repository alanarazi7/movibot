"""Tests for data_preprocessing/fetch_tmdb_updates.py.

Network calls are not exercised here; what matters for correctness is that
rows built from live TMDB payloads are indistinguishable from rows the
offline pipeline produces, and that the usability rules match
prepare_movibot_data.clean_movies() exactly. A row that passes here but
would have been dropped there corrupts the catalog silently.
"""

import json

from data_preprocessing.fetch_tmdb_updates import (
    OUTPUT_COLUMNS,
    build_row,
    canonicalize_companies,
    is_in_demo_scope,
    names_of,
)
from data_preprocessing.prepare_movibot_data import DEMO_STUDIOS


def tmdb_detail(**overrides):
    """A minimal well-formed TMDB /movie/{id} payload."""
    detail = {
        "id": 508943,
        "imdb_id": "tt3006740",
        "title": "Luca",
        "original_title": "Luca",
        "overview": "A boy spends an unforgettable summer on the Italian Riviera.",
        "release_date": "2021-06-17",
        "runtime": 95,
        "genres": [{"id": 16, "name": "Animation"}, {"id": 10751, "name": "Family"}],
        "production_companies": [{"id": 3, "name": "Pixar"}],
        "production_countries": [{"iso_3166_1": "US", "name": "United States of America"}],
        "spoken_languages": [{"name": "English"}],
        "belongs_to_collection": None,
        "popularity": 42.5,
        "vote_average": 7.8,
        "vote_count": 9000,
        "budget": 0,
        "revenue": 0,
        "tagline": "Are you ready to meet Luca?",
        "status": "Released",
        "original_language": "en",
        "adult": False,
        "video": False,
    }
    detail.update(overrides)
    return detail


class TestNamesOf:
    def test_extracts_and_dedupes_names(self):
        assert names_of([{"name": "Pixar"}, {"name": "Pixar"}, {"name": "Disney"}]) == [
            "Pixar",
            "Disney",
        ]

    def test_skips_blank_and_malformed_entries(self):
        assert names_of([{"name": "  "}, {"nope": 1}, "junk", None, {"name": "Disney"}]) == [
            "Disney"
        ]

    def test_non_list_is_empty(self):
        assert names_of(None) == []
        assert names_of({"name": "Disney"}) == []


class TestCanonicalizeCompanies:
    def test_tmdb_pixar_maps_to_the_catalog_spelling(self):
        # TMDB company 3 is named "Pixar" today; the 2017 Kaggle dump spells it
        # "Pixar Animation Studios". Left unmapped, every new Pixar film would
        # be dropped as out of scope and the catalog would disagree with itself.
        assert canonicalize_companies(["Pixar"]) == ["Pixar Animation Studios"]

    def test_already_canonical_names_pass_through(self):
        assert canonicalize_companies(["Walt Disney Pictures"]) == ["Walt Disney Pictures"]

    def test_unrelated_companies_are_untouched(self):
        assert canonicalize_companies(["Lucasfilm Ltd."]) == ["Lucasfilm Ltd."]

    def test_canonicalization_does_not_introduce_duplicates(self):
        assert canonicalize_companies(["Pixar", "Pixar Animation Studios"]) == [
            "Pixar Animation Studios"
        ]


class TestIsInDemoScope:
    def test_accepts_tmdb_spelling_of_pixar(self):
        assert is_in_demo_scope(["Pixar"]) is True

    def test_accepts_canonical_studio_names(self):
        for studio in DEMO_STUDIOS:
            assert is_in_demo_scope([studio]) is True

    def test_rejects_unrelated_studio(self):
        assert is_in_demo_scope(["Warner Bros. Pictures"]) is False

    def test_rejects_empty(self):
        assert is_in_demo_scope([]) is False


class TestBuildRow:
    def test_emits_exactly_the_catalog_columns(self):
        row = build_row(tmdb_detail(), [])
        assert list(row.keys()) == OUTPUT_COLUMNS

    def test_derives_release_year_from_release_date(self):
        assert build_row(tmdb_detail(), [])["release_year"] == 2021

    def test_serializes_list_columns_like_the_offline_pipeline(self):
        row = build_row(tmdb_detail(), ["friendship", "sea monster"])

        assert json.loads(row["genres"]) == ["Animation", "Family"]
        assert json.loads(row["keywords"]) == ["friendship", "sea monster"]
        # Compact separators, matching prepare_movibot_data.json_list.
        assert row["genres"] == '["Animation","Family"]'

    def test_production_companies_are_stored_canonically(self):
        row = build_row(tmdb_detail(), [])
        assert json.loads(row["production_companies"]) == ["Pixar Animation Studios"]

    def test_has_mpst_synopsis_is_always_false(self):
        # MPST is a static 2018 corpus; nothing past the Kaggle cutoff is in it.
        assert build_row(tmdb_detail(), [])["has_mpst_synopsis"] is False

    def test_collection_name_is_flattened(self):
        row = build_row(
            tmdb_detail(belongs_to_collection={"id": 10194, "name": "Toy Story Collection"}),
            [],
        )
        assert row["belongs_to_collection"] == "Toy Story Collection"

    def test_missing_collection_becomes_empty_string(self):
        assert build_row(tmdb_detail(), [])["belongs_to_collection"] == ""

    def test_invalid_imdb_id_is_blanked_not_kept(self):
        assert build_row(tmdb_detail(imdb_id="12345"), [])["imdb_id"] == ""
        assert build_row(tmdb_detail(imdb_id=None), [])["imdb_id"] == ""

    # The usability rules below must mirror clean_movies() exactly.

    def test_blank_overview_is_rejected(self):
        assert build_row(tmdb_detail(overview="   "), []) is None

    def test_blank_title_is_rejected(self):
        assert build_row(tmdb_detail(title=""), []) is None

    def test_non_positive_runtime_is_rejected(self):
        assert build_row(tmdb_detail(runtime=0), []) is None
        assert build_row(tmdb_detail(runtime=None), []) is None

    def test_incomplete_release_date_is_rejected(self):
        # TMDB returns "" for unreleased titles and sometimes a bare year.
        assert build_row(tmdb_detail(release_date=""), []) is None
        assert build_row(tmdb_detail(release_date="2021"), []) is None

    def test_missing_id_is_rejected(self):
        assert build_row(tmdb_detail(id=None), []) is None
