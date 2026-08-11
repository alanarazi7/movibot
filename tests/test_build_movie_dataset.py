"""Tests for data_preprocessing/build_movie_dataset.py.

The join must be a LEFT join on the catalog: a movie with no Wikipedia article
still belongs in the dataset, it just has no plot. Dropping those rows would
quietly shrink the catalog the agent can filter over, so that every
"recommend me a Disney movie" answer silently excludes a third of the library.
"""

import pytest

from data_preprocessing.build_movie_dataset import (
    PLOT_COLUMNS,
    attach_plots,
    build_rows,
)


def movie(movie_id, title="A Movie", **extra):
    row = {"id": str(movie_id), "title": title, "release_year": "2000"}
    row.update(extra)
    return row


def plot(movie_id, text="Something happens.", **extra):
    row = {
        "movie_id": str(movie_id),
        "plot_text": text,
        "plot_words": str(len(text.split())),
        "wikipedia_title": "A Movie",
        "wikipedia_url": "https://en.wikipedia.org/wiki/A_Movie",
    }
    row.update(extra)
    return row


class TestAttachPlots:
    def test_attaches_a_plot_to_its_movie(self):
        merged, _ = attach_plots([movie(1)], [plot(1, "Simba flees.")])

        assert merged[0]["plot_text"] == "Simba flees."
        assert merged[0]["has_plot"] == "true"

    def test_keeps_movies_that_have_no_plot(self):
        merged, _ = attach_plots([movie(1), movie(2)], [plot(1)])

        assert len(merged) == 2
        assert merged[1]["has_plot"] == "false"

    def test_missing_plot_columns_are_blank_not_absent(self):
        merged, _ = attach_plots([movie(1)], [])

        for column in PLOT_COLUMNS:
            assert merged[0][column] == ""

    def test_every_catalog_row_survives(self):
        catalog = [movie(i) for i in range(10)]

        merged, _ = attach_plots(catalog, [plot(3)])

        assert len(merged) == 10
        assert [r["id"] for r in merged] == [str(i) for i in range(10)]

    def test_plot_for_an_unknown_movie_is_reported_not_silently_dropped(self):
        _, stats = attach_plots([movie(1)], [plot(1), plot(999)])

        assert stats["orphan_plots"] == 1

    def test_counts_matched_and_unmatched(self):
        _, stats = attach_plots([movie(1), movie(2), movie(3)], [plot(2)])

        assert stats["with_plot"] == 1
        assert stats["without_plot"] == 2
        assert stats["total"] == 3

    def test_blank_plot_text_does_not_count_as_having_a_plot(self):
        merged, stats = attach_plots([movie(1)], [plot(1, "   ")])

        assert merged[0]["has_plot"] == "false"
        assert stats["with_plot"] == 0

    def test_first_plot_wins_when_duplicated(self):
        merged, _ = attach_plots([movie(1)], [plot(1, "First."), plot(1, "Second.")])

        assert merged[0]["plot_text"] == "First."

    def test_does_not_mutate_the_input_rows(self):
        catalog = [movie(1)]
        original = dict(catalog[0])

        attach_plots(catalog, [plot(1)])

        assert catalog[0] == original


class TestBuildRows:
    def test_column_order_is_catalog_then_plot(self):
        rows, columns = build_rows(
            [movie(1, budget="100")], [plot(1)], catalog_columns=["id", "title", "budget"]
        )

        assert columns == ["id", "title", "budget", *PLOT_COLUMNS, "has_plot"]
        assert list(rows[0].keys()) == columns

    def test_rows_are_sorted_by_movie_id_numerically(self):
        catalog = [movie(100), movie(9), movie(20)]

        rows, _ = build_rows(catalog, [], catalog_columns=["id", "title", "release_year"])

        assert [r["id"] for r in rows] == ["9", "20", "100"]

    def test_rejects_an_empty_catalog(self):
        with pytest.raises(ValueError):
            build_rows([], [], catalog_columns=["id"])
