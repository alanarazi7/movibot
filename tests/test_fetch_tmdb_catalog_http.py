"""Tests for the TMDB HTTP/discovery layer in fetch_tmdb_catalog.py.

No real network calls. What is pinned down here is the failure behaviour:
credentials must fail loudly rather than silently fetching nothing, rate
limits and transient 5xx must be retried, and a wrong company id must abort
before it can pull somebody else's catalog into the movie table.
"""

import pytest
import requests

from data_preprocessing import fetch_tmdb_catalog as mod
from data_preprocessing.fetch_tmdb_catalog import (
    TmdbError,
    build_session,
    discover_movie_ids,
    get_json,
    verify_company_ids,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("TMDB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeSession:
    """Replays a scripted sequence of responses; exceptions are raised."""

    def __init__(self, *scripted):
        self._scripted = list(scripted)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        item = self._scripted.pop(0) if self._scripted else FakeResponse()
        if isinstance(item, Exception):
            raise item
        return item


class TestBuildSession:
    def test_missing_credentials_fail_loudly(self):
        with pytest.raises(TmdbError, match="No TMDB credentials"):
            build_session()

    def test_v4_token_becomes_a_bearer_header(self, monkeypatch):
        monkeypatch.setenv("TMDB_ACCESS_TOKEN", "tok123")

        session = build_session()

        assert session.headers["Authorization"] == "Bearer tok123"

    def test_v3_key_becomes_a_query_param(self, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "key456")

        session = build_session()

        assert session.params == {"api_key": "key456"}
        assert "Authorization" not in session.headers

    def test_bearer_token_wins_over_v3_key(self, monkeypatch):
        monkeypatch.setenv("TMDB_ACCESS_TOKEN", "tok123")
        monkeypatch.setenv("TMDB_API_KEY", "key456")

        assert build_session().headers["Authorization"] == "Bearer tok123"

    def test_blank_credentials_are_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("TMDB_ACCESS_TOKEN", "   ")

        with pytest.raises(TmdbError):
            build_session()


class TestGetJson:
    def test_returns_payload_on_success(self):
        session = FakeSession(FakeResponse(200, {"id": 7}))

        assert get_json(session, "/movie/7") == {"id": 7}

    def test_unauthorized_is_not_retried(self):
        session = FakeSession(FakeResponse(401))

        with pytest.raises(TmdbError, match="rejected the credentials"):
            get_json(session, "/movie/7")

        assert len(session.calls) == 1

    def test_rate_limit_is_retried_then_succeeds(self):
        session = FakeSession(
            FakeResponse(429, headers={"Retry-After": "1"}),
            FakeResponse(200, {"id": 7}),
        )

        assert get_json(session, "/movie/7") == {"id": 7}
        assert len(session.calls) == 2

    def test_server_error_is_retried_then_succeeds(self):
        session = FakeSession(FakeResponse(503), FakeResponse(200, {"id": 7}))

        assert get_json(session, "/movie/7") == {"id": 7}

    def test_persistent_server_error_raises(self):
        session = FakeSession(*[FakeResponse(500) for _ in range(4)])

        with pytest.raises(TmdbError, match="server error"):
            get_json(session, "/movie/7")

    def test_client_error_raises_immediately(self):
        session = FakeSession(FakeResponse(404))

        with pytest.raises(TmdbError, match="404"):
            get_json(session, "/movie/7")

    def test_network_error_is_retried_then_succeeds(self):
        session = FakeSession(
            requests.ConnectionError("boom"),
            FakeResponse(200, {"id": 7}),
        )

        assert get_json(session, "/movie/7") == {"id": 7}

    def test_persistent_network_error_raises(self):
        session = FakeSession(*[requests.ConnectionError("boom") for _ in range(4)])

        with pytest.raises(TmdbError, match="Network error"):
            get_json(session, "/movie/7")

    def test_builds_the_full_tmdb_url(self):
        session = FakeSession(FakeResponse(200, {}))

        get_json(session, "/movie/7")

        assert session.calls[0][0] == f"{mod.TMDB_BASE_URL}/movie/7"


class TestVerifyCompanyIds:
    def test_accepts_matching_names(self):
        session = FakeSession(
            *[FakeResponse(200, {"name": name}) for name in mod.STUDIO_COMPANY_IDS.values()]
        )

        verify_company_ids(session)  # must not raise

    def test_name_comparison_is_case_insensitive(self):
        session = FakeSession(
            *[
                FakeResponse(200, {"name": name.upper()})
                for name in mod.STUDIO_COMPANY_IDS.values()
            ]
        )

        verify_company_ids(session)

    def test_rejects_a_reassigned_company_id(self):
        session = FakeSession(FakeResponse(200, {"name": "Warner Bros. Pictures"}))

        with pytest.raises(TmdbError, match="Refusing to fetch a wrong catalog"):
            verify_company_ids(session)


class TestDiscoverMovieIds:
    def test_collects_ids_across_pages(self):
        session = FakeSession(
            FakeResponse(200, {"results": [{"id": 1}, {"id": 2}], "total_pages": 2}),
            FakeResponse(200, {"results": [{"id": 3}], "total_pages": 2}),
        )

        assert list(discover_movie_ids(session, "2017-07-01")) == [1, 2, 3]

    def test_stops_on_an_empty_page(self):
        session = FakeSession(
            FakeResponse(200, {"results": [{"id": 1}], "total_pages": 9}),
            FakeResponse(200, {"results": [], "total_pages": 9}),
        )

        assert list(discover_movie_ids(session, "2017-07-01")) == [1]

    def test_deduplicates_ids_repeated_across_pages(self):
        session = FakeSession(
            FakeResponse(200, {"results": [{"id": 1}, {"id": 2}], "total_pages": 2}),
            FakeResponse(200, {"results": [{"id": 2}, {"id": 3}], "total_pages": 2}),
        )

        assert list(discover_movie_ids(session, "2017-07-01")) == [1, 2, 3]

    def test_ignores_malformed_ids(self):
        session = FakeSession(
            FakeResponse(200, {"results": [{"id": None}, {"id": "x"}, {"id": 5}], "total_pages": 1}),
        )

        assert list(discover_movie_ids(session, "2017-07-01")) == [5]

    def test_sends_the_studio_and_date_filters(self):
        session = FakeSession(FakeResponse(200, {"results": [{"id": 1}], "total_pages": 1}))

        list(discover_movie_ids(session, "2017-07-01"))

        _, params = session.calls[0]
        # Derived, not hardcoded: adding a studio must not break this test.
        assert params["with_companies"] == "|".join(
            str(i) for i in mod.STUDIO_COMPANY_IDS
        )
        assert params["primary_release_date.gte"] == "2017-07-01"
        assert params["include_adult"] == "false"

    def test_queries_every_configured_studio(self):
        session = FakeSession(FakeResponse(200, {"results": [{"id": 1}], "total_pages": 1}))

        list(discover_movie_ids(session, "1920-01-01"))

        requested = set(session.calls[0][1]["with_companies"].split("|"))
        assert requested == {str(i) for i in mod.STUDIO_COMPANY_IDS}
