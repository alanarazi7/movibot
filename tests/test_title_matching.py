"""Tests for data_preprocessing/title_matching.py.

The catalog is joined to MPST by exact IMDb ID, which is safe. External
transcript corpora and most scraped catalogs carry no IMDb ID, so they can
only be joined on (title, year). For a Disney/Pixar scope that is the worst
possible case: nearly every animated classic has a live-action remake under
an identical title. These tests pin down the rules that keep such a join
from silently attaching the wrong film to a movie.
"""

import pytest

from data_preprocessing.title_matching import (
    MatchCandidate,
    match_titles,
    normalize_title,
    summarize,
)


# ---------------------------------------------------------------------
# normalize_title
# ---------------------------------------------------------------------

class TestNormalizeTitle:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_title("  The   LION   King ") == "lion king"

    def test_inverted_article_matches_natural_order(self):
        # Script corpora frequently store "Wizard of Oz, The".
        assert normalize_title("Wizard of Oz, The") == normalize_title("The Wizard of Oz")

    def test_leading_article_is_stripped(self):
        assert normalize_title("The Jungle Book") == normalize_title("Jungle Book")

    def test_strips_diacritics(self):
        assert normalize_title("Ratatouille") == "ratatouille"
        assert normalize_title("Amélie") == "amelie"

    def test_punctuation_variants_collapse(self):
        # TMDB writes "WALL·E"; scrapes commonly write "Wall-E" or "WALL E".
        assert normalize_title("WALL·E") == normalize_title("Wall-E") == normalize_title("WALL E")

    def test_ampersand_becomes_and(self):
        assert normalize_title("Lilo & Stitch") == normalize_title("Lilo and Stitch")

    def test_trailing_year_parentheses_removed(self):
        assert normalize_title("Cinderella (1950)") == normalize_title("Cinderella")

    def test_trailing_roman_numeral_becomes_arabic(self):
        # "Frozen II" (TMDB) vs "Frozen 2" (scrapes) is a real Disney case.
        assert normalize_title("Frozen II") == normalize_title("Frozen 2")
        assert normalize_title("Mulan II") == normalize_title("Mulan 2")

    def test_ambiguous_roman_numerals_are_left_alone(self):
        # "X"/"I" are far more often letters than numerals; converting them
        # would corrupt titles like "Malcolm X".
        assert normalize_title("Malcolm X") == "malcolm x"

    def test_empty_and_none_are_safe(self):
        assert normalize_title("") == ""
        assert normalize_title(None) == ""


# ---------------------------------------------------------------------
# match_titles
# ---------------------------------------------------------------------

def target(key, title, year):
    return MatchCandidate(key=key, title=title, year=year)


class TestMatchTitles:
    def test_exact_title_and_year_matches(self):
        targets = [target("m1", "Toy Story", 1995)]
        candidates = [target("c1", "Toy Story", 1995)]

        (result,) = match_titles(targets, candidates)

        assert result.status == "matched"
        assert result.candidate_key == "c1"

    def test_year_within_tolerance_matches(self):
        # Release-year disagreements of one year are common between sources.
        targets = [target("m1", "Bambi", 1942)]
        candidates = [target("c1", "Bambi", 1943)]

        (result,) = match_titles(targets, candidates, year_tolerance=1)

        assert result.status == "matched"

    def test_year_outside_tolerance_does_not_match(self):
        targets = [target("m1", "Bambi", 1942)]
        candidates = [target("c1", "Bambi", 1950)]

        (result,) = match_titles(targets, candidates, year_tolerance=1)

        assert result.status == "unmatched"
        assert result.candidate_key is None

    def test_remake_is_disambiguated_by_year(self):
        # THE core case: the 1994 animated film must not get the 2019 script.
        targets = [target("m1", "The Lion King", 1994)]
        candidates = [
            target("c1", "The Lion King", 1994),
            target("c2", "The Lion King", 2019),
        ]

        (result,) = match_titles(targets, candidates)

        assert result.status == "matched"
        assert result.candidate_key == "c1"

    def test_closest_year_wins_when_unique(self):
        targets = [target("m1", "Fantasia", 1940)]
        candidates = [
            target("c1", "Fantasia", 1940),
            target("c2", "Fantasia", 1941),
        ]

        (result,) = match_titles(targets, candidates, year_tolerance=2)

        assert result.status == "matched"
        assert result.candidate_key == "c1"

    def test_equidistant_years_are_ambiguous_not_guessed(self):
        targets = [target("m1", "Cinderella", 1950)]
        candidates = [
            target("c1", "Cinderella", 1949),
            target("c2", "Cinderella", 1951),
        ]

        (result,) = match_titles(targets, candidates, year_tolerance=2)

        assert result.status == "ambiguous"
        assert result.candidate_key is None
        assert set(result.alternatives) == {"c1", "c2"}

    def test_yearless_candidate_matches_only_when_unique(self):
        targets = [target("m1", "Dumbo", 1941)]
        candidates = [target("c1", "Dumbo", None)]

        (result,) = match_titles(targets, candidates)

        assert result.status == "matched"
        assert result.candidate_key == "c1"

    def test_multiple_yearless_candidates_are_ambiguous(self):
        targets = [target("m1", "Aladdin", 1992)]
        candidates = [
            target("c1", "Aladdin", None),
            target("c2", "Aladdin", None),
        ]

        (result,) = match_titles(targets, candidates)

        assert result.status == "ambiguous"

    def test_dated_candidate_beats_yearless_candidate(self):
        targets = [target("m1", "Pinocchio", 1940)]
        candidates = [
            target("c1", "Pinocchio", 1940),
            target("c2", "Pinocchio", None),
        ]

        (result,) = match_titles(targets, candidates)

        assert result.status == "matched"
        assert result.candidate_key == "c1"

    def test_candidate_claimed_by_two_targets_is_rejected_for_both(self):
        # Enforces the one-to-one guarantee the MPST join gets from IMDb IDs:
        # one transcript must never be attached to two different movies.
        targets = [
            target("m1", "The Jungle Book", 2016),
            target("m2", "The Jungle Book", 2015),
        ]
        candidates = [target("c1", "The Jungle Book", 2016)]

        results = match_titles(targets, candidates, year_tolerance=1)

        assert {r.status for r in results} == {"ambiguous"}
        assert all(r.candidate_key is None for r in results)

    def test_unmatched_target_is_reported_not_dropped(self):
        targets = [target("m1", "Encanto", 2021)]
        candidates = [target("c1", "Moana", 2016)]

        (result,) = match_titles(targets, candidates)

        assert result.status == "unmatched"
        assert result.target_key == "m1"

    def test_every_target_gets_exactly_one_result(self):
        targets = [
            target("m1", "Up", 2009),
            target("m2", "Coco", 2017),
            target("m3", "Brave", 2012),
        ]
        candidates = [target("c1", "Up", 2009)]

        results = match_titles(targets, candidates)

        assert len(results) == len(targets)
        assert [r.target_key for r in results] == ["m1", "m2", "m3"]

    def test_target_without_year_matches_only_a_sole_candidate(self):
        targets = [target("m1", "Tangled", None)]
        candidates = [target("c1", "Tangled", 2010)]

        (result,) = match_titles(targets, candidates)

        assert result.status == "matched"

    def test_target_without_year_is_ambiguous_against_a_remake_pair(self):
        targets = [target("m1", "Beauty and the Beast", None)]
        candidates = [
            target("c1", "Beauty and the Beast", 1991),
            target("c2", "Beauty and the Beast", 2017),
        ]

        (result,) = match_titles(targets, candidates)

        assert result.status == "ambiguous"

    def test_falls_back_to_sole_undated_candidate_when_dates_are_implausible(self):
        targets = [target("m1", "Peter Pan", 1953)]
        candidates = [
            target("c1", "Peter Pan", 2003),
            target("c2", "Peter Pan", None),
        ]

        (result,) = match_titles(targets, candidates, year_tolerance=1)

        assert result.status == "matched"
        assert result.candidate_key == "c2"

    def test_results_are_immutable(self):
        (result,) = match_titles([target("m1", "Up", 2009)], [])

        with pytest.raises(Exception):
            result.status = "matched"  # type: ignore[misc]


class TestSummarize:
    def test_counts_every_status(self):
        targets = [
            target("m1", "Up", 2009),
            target("m2", "Coco", 2017),
            target("m3", "Cinderella", 1950),
        ]
        candidates = [
            target("c1", "Up", 2009),
            target("c2", "Cinderella", 1949),
            target("c3", "Cinderella", 1951),
        ]

        stats = summarize(match_titles(targets, candidates, year_tolerance=2))

        assert stats == {"total": 3, "matched": 1, "ambiguous": 1, "unmatched": 1}
