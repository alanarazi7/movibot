"""Tests for data_preprocessing/fetch_wikipedia_plots.py.

The risky part is section extraction. Wikipedia film articles do not agree on
what the plot section is called, and an over-greedy match would swallow the
Cast and Production sections -- which would poison the "no deaths" signal with
sentences about actors and box office rather than the story.
"""

import pytest

from data_preprocessing import fetch_wikipedia_plots as mod
from data_preprocessing.fetch_wikipedia_plots import (
    PLOT_SECTION_NAMES,
    article_title_from_url,
    chunked,
    extract_plot,
    fetch_extracts,
    parse_sparql_articles,
)


def article(*sections):
    """Builds a plain-text MediaWiki extract from (heading, body) pairs."""
    return "Lead paragraph.\n\n" + "\n\n".join(
        f"== {name} ==\n{body}" for name, body in sections
    )


class TestExtractPlot:
    def test_extracts_the_plot_section(self):
        text = article(("Plot", "Simba flees."), ("Cast", "Matthew Broderick."))

        assert extract_plot(text) == "Simba flees."

    def test_stops_before_the_next_section(self):
        text = article(("Plot", "Simba flees."), ("Cast", "Matthew Broderick as Simba."))

        assert "Broderick" not in extract_plot(text)

    def test_accepts_every_known_heading_variant(self):
        for name in PLOT_SECTION_NAMES:
            text = article((name.title(), "The story happens."), ("Cast", "Someone."))
            assert extract_plot(text) == "The story happens.", name

    def test_keeps_subsections_of_the_plot(self):
        # "=== Act I ===" is nested inside Plot and must not terminate it.
        text = (
            "Lead.\n\n== Plot ==\nOpening.\n\n=== Act I ===\nMiddle.\n\n== Cast ==\nActors."
        )
        plot = extract_plot(text)

        assert "Opening." in plot
        assert "Middle." in plot
        assert "Actors" not in plot

    def test_handles_a_plot_section_at_the_end_of_the_article(self):
        text = "Lead.\n\n== Plot ==\nSimba flees."

        assert extract_plot(text) == "Simba flees."

    def test_heading_match_is_case_insensitive(self):
        text = "Lead.\n\n== PLOT ==\nSimba flees.\n\n== Cast ==\nActors."

        assert extract_plot(text) == "Simba flees."

    def test_reads_anthology_segment_headings(self):
        # Disney's package films (Fantasia, Make Mine Music, Melody Time) have
        # no single plot -- their articles use Program / Film segments /
        # Vignettes instead, and were the only well-known titles missing.
        for name in ("Program", "Film segments", "Vignettes"):
            text = article((name, "Segment one. Segment two."), ("Cast", "Actors."))
            assert extract_plot(text) == "Segment one. Segment two.", name

    def test_plot_wins_over_a_segment_heading(self):
        text = article(
            ("Program", "Segments listing."),
            ("Plot", "The actual story."),
            ("Cast", "Actors."),
        )

        assert extract_plot(text) == "The actual story."

    def test_returns_none_when_there_is_no_plot_section(self):
        text = article(("Cast", "Actors."), ("Production", "Filmed in 1993."))

        assert extract_plot(text) is None

    def test_returns_none_for_empty_input(self):
        assert extract_plot("") is None
        assert extract_plot(None) is None

    def test_returns_none_when_the_plot_section_is_blank(self):
        assert extract_plot("Lead.\n\n== Plot ==\n\n== Cast ==\nActors.") is None

    def test_does_not_match_a_heading_that_merely_contains_plot(self):
        text = article(("Plot holes and criticism", "Critics noted."), ("Cast", "Actors."))

        assert extract_plot(text) is None


class TestArticleTitleFromUrl:
    def test_decodes_underscores(self):
        assert article_title_from_url("https://en.wikipedia.org/wiki/The_Lion_King") == (
            "The Lion King"
        )

    def test_decodes_percent_escapes(self):
        assert article_title_from_url(
            "https://en.wikipedia.org/wiki/Luca_%282021_film%29"
        ) == "Luca (2021 film)"

    def test_handles_parentheses_left_unescaped(self):
        assert article_title_from_url(
            "https://en.wikipedia.org/wiki/Pinocchio_(1940_film)"
        ) == "Pinocchio (1940 film)"


class TestParseSparqlArticles:
    def test_maps_imdb_ids_to_article_titles(self):
        payload = {
            "results": {
                "bindings": [
                    {
                        "imdb": {"value": "tt0110357"},
                        "article": {"value": "https://en.wikipedia.org/wiki/The_Lion_King"},
                    }
                ]
            }
        }

        assert parse_sparql_articles(payload) == {"tt0110357": "The Lion King"}

    def test_empty_results_give_an_empty_mapping(self):
        assert parse_sparql_articles({"results": {"bindings": []}}) == {}

    def test_malformed_payload_does_not_raise(self):
        assert parse_sparql_articles({}) == {}

    def test_first_binding_wins_for_a_duplicated_id(self):
        # A film can have several sitelinks; take one deterministically.
        payload = {
            "results": {
                "bindings": [
                    {
                        "imdb": {"value": "tt1"},
                        "article": {"value": "https://en.wikipedia.org/wiki/First"},
                    },
                    {
                        "imdb": {"value": "tt1"},
                        "article": {"value": "https://en.wikipedia.org/wiki/Second"},
                    },
                ]
            }
        }

        assert parse_sparql_articles(payload) == {"tt1": "First"}


class TestFetchExtracts:
    def test_requests_exactly_one_title_per_call(self, monkeypatch):
        """Regression: Wikipedia refuses to batch whole-article extracts. Given
        several titles it answers for the FIRST page only and drops the rest,
        which reads as "no plot section" for 19 of every 20 films."""
        calls = []

        class FakeSession:
            def get(self, url, params=None, timeout=None, headers=None):
                calls.append(params)

                class R:
                    status_code = 200
                    ok = True

                    @staticmethod
                    def json():
                        return {"query": {"pages": []}}

                return R()

        monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
        fetch_extracts(FakeSession(), ["A", "B", "C"])

        assert len(calls) == 3
        assert all("|" not in params["titles"] for params in calls)

    def test_maps_titles_to_extract_text(self, monkeypatch):
        class FakeSession:
            def get(self, url, params=None, timeout=None, headers=None):
                class R:
                    status_code = 200
                    ok = True

                    @staticmethod
                    def json():
                        return {
                            "query": {
                                "pages": [
                                    {"title": "A", "extract": "text a"},
                                    {"title": "B", "extract": "text b"},
                                    {"title": "C"},  # missing page, no extract
                                ]
                            }
                        }

                return R()

        monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

        assert fetch_extracts(FakeSession(), ["A", "B", "C"]) == {
            "A": "text a",
            "B": "text b",
        }


class TestChunked:
    def test_splits_into_even_chunks(self):
        assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_last_chunk_may_be_short(self):
        assert list(chunked([1, 2, 3], 2)) == [[1, 2], [3]]

    def test_empty_input_yields_nothing(self):
        assert list(chunked([], 3)) == []

    def test_rejects_a_non_positive_size(self):
        with pytest.raises(ValueError):
            list(chunked([1, 2], 0))
