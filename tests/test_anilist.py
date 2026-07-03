# pyright: strict
# pyright: reportPrivateUsage=false
# These are deliberately under-test private helpers; the repo already disables
# reportPrivateUsage for all of tests/ (test code reads private members), but the
# strict directive above re-enables it, so restore the repo's test policy here.
"""Direct unit tests for the pure helpers and retry path in ``anilist``.

These functions are otherwise only exercised incidentally. The classification /
parsing / extraction helpers take plain ``dict`` bodies, so they are tested with
no mocks at all; ``_post_with_retry`` is faked at the ``requests`` boundary with
the ``responses`` library (no ``unittest.mock``).
"""

import re
import time

import pytest
import responses

from seadexarr.modules.anilist import (
    _MEDIA_FIELDS,
    API_URL,
    QUERY,
    RETRYABLE_ERROR_SUBSTRINGS,
    RETRYABLE_STATUS,
    _errors_are_retryable,
    _extract,
    _media_from,
    _parse_errors,
    _post_with_retry,
)
from seadexarr.modules.seadex_types import AniListError, AniListMediaNode


def _no_sleep(_seconds: float) -> None:
    """Replace ``time.sleep`` so the retry backoffs don't actually wait."""


# --- _errors_are_retryable --------------------------------------------------


def test_errors_are_retryable_none_body() -> None:
    """A ``None`` body carries no errors, so it is never retryable."""

    assert _errors_are_retryable(None) is False


def test_errors_are_retryable_no_errors() -> None:
    """An empty body, a missing ``errors`` key, and an empty list are all misses.

    A legitimate "not found" is HTTP 200 with no ``errors`` array, so it must not
    be treated as a throttle.
    """

    assert _errors_are_retryable({}) is False
    assert _errors_are_retryable({"data": {"Media": None}}) is False
    assert _errors_are_retryable({"errors": []}) is False


def test_errors_are_retryable_throttle_substring_is_case_insensitive() -> None:
    """A throttle/rate-limit message matches regardless of case, even without a status."""

    body: dict[str, object] = {"errors": [{"message": "Rate Limit Exceeded"}]}
    assert _errors_are_retryable(body) is True


def test_errors_are_retryable_status_path() -> None:
    """A 5xx-style status in the error entry is retryable even with a benign message."""

    body: dict[str, object] = {"errors": [{"message": "internal", "status": 503}]}
    assert _errors_are_retryable(body) is True


def test_errors_are_retryable_non_retryable_error() -> None:
    """A genuine validation error (non-throttle status, non-throttle message) is not retried."""

    body: dict[str, object] = {"errors": [{"message": "Validation failed", "status": 400}]}
    assert _errors_are_retryable(body) is False


# --- _parse_errors ----------------------------------------------------------


def test_parse_errors_none_and_missing_key() -> None:
    """No body or no ``errors`` key yields an empty list (never raises)."""

    assert _parse_errors(None) == []
    assert _parse_errors({}) == []


def test_parse_errors_non_list_errors() -> None:
    """A non-list ``errors`` value (malformed body) yields an empty list."""

    body: dict[str, object] = {"errors": "boom"}
    assert _parse_errors(body) == []


def test_parse_errors_wellformed() -> None:
    """A well-formed entry maps to a typed ``AniListError`` with its status and message."""

    body: dict[str, object] = {"errors": [{"message": "Too Many Requests", "status": 429}]}
    parsed = _parse_errors(body)
    assert parsed == [AniListError(message="Too Many Requests", status=429)]


def test_parse_errors_skips_malformed_entries() -> None:
    """Non-object junk in the array is skipped; only the dict entries are parsed.

    A dict without a ``message`` defaults to ``""``; a non-int ``status`` becomes
    ``None`` (here the second dict carries only a status).
    """

    body: dict[str, object] = {
        "errors": [{"message": "ok", "status": 429}, "junk", 5, None, {"status": 503}],
    }
    parsed = _parse_errors(body)
    assert parsed == [
        AniListError(message="ok", status=429),
        AniListError(message="", status=503),
    ]


# --- _extract ---------------------------------------------------------------


def test_extract_present_path() -> None:
    """A fully present path returns the leaf dict node verbatim."""

    body: dict[str, object] = {"data": {"Media": {"id": 5, "episodes": 12}}}
    assert _extract(body, "data", "Media") == {"id": 5, "episodes": 12}


def test_extract_missing_key_yields_empty() -> None:
    """A key missing at the final hop yields ``{}`` rather than ``None``."""

    body: dict[str, object] = {"data": {}}
    assert _extract(body, "data", "Media") == {}


def test_extract_null_intermediate_yields_empty() -> None:
    """A null intermediate level (``{"data": null}``) is coerced to ``{}`` and walked safely."""

    body: dict[str, object] = {"data": None}
    assert _extract(body, "data", "Media") == {}
    assert _extract(None, "data", "Media") == {}


# --- _media_from ------------------------------------------------------------


def test_media_from_full_body() -> None:
    """A complete body crosses into a fully-populated typed node, preferring ``large`` cover."""

    body: dict[str, object] = {
        "data": {
            "Media": {
                "id": 5,
                "title": {"english": "English Title", "romaji": "Romaji Title"},
                "coverImage": {"large": "https://img/large", "medium": "https://img/medium"},
                "episodes": 12,
                "format": "TV",
            },
        },
    }
    assert _media_from(body) == AniListMediaNode(
        id=5,
        title_english="English Title",
        title_romaji="Romaji Title",
        episodes=12,
        cover_image="https://img/large",
        format="TV",
    )


def test_media_from_missing_fields_defaults() -> None:
    """Absent nested ``title``/``coverImage`` and scalar fields default to ``None``."""

    body: dict[str, object] = {"data": {"Media": {"id": 7}}}
    assert _media_from(body) == AniListMediaNode(id=7)


def test_media_from_none_body_is_all_none() -> None:
    """A ``None`` body (a miss) parses to an all-``None`` node, not a crash."""

    assert _media_from(None) == AniListMediaNode()


# --- constants sanity -------------------------------------------------------


def test_media_fields_fragment_covers_from_api_reads() -> None:
    """Every field ``AniListMediaNode.from_api`` reads is selected by the fragment.

    The read set below is derived from ``from_api``'s body: the top-level
    ``id``/``title``/``coverImage``/``episodes``/``format`` keys plus the nested
    ``title.english``/``title.romaji``/``coverImage.large`` selections. A field
    dropped from ``_MEDIA_FIELDS`` would silently parse to ``None`` downstream,
    so the shared fragment (not any full query string) is what's pinned here.
    """

    read_fields = ("id", "title", "english", "romaji", "coverImage", "large", "episodes", "format")
    for name in read_fields:
        assert re.search(rf"\b{name}\b", _MEDIA_FIELDS), f"fragment no longer selects {name!r}"


def test_retryable_constants() -> None:
    """The retry vocabulary matches the helpers' assumptions: 429 is retryable and
    every throttle substring is lower-case (the message is lower-cased before matching).
    """

    assert 429 in RETRYABLE_STATUS
    assert 200 not in RETRYABLE_STATUS
    assert all(substring == substring.lower() for substring in RETRYABLE_ERROR_SUBSTRINGS)


# --- _post_with_retry (responses-faked requests boundary) -------------------


def test_post_with_retry_returns_valid_200_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 200 with a valid body is returned verbatim after one request."""

    monkeypatch.setattr(time, "sleep", _no_sleep)
    expected: dict[str, object] = {"data": {"Media": {"id": 1, "episodes": 12}}}
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, API_URL, json=expected, status=200)
        body = _post_with_retry(QUERY, {"id": 1})
        assert len(rsps.calls) == 1
    assert body == expected
    assert _media_from(body).episodes == 12


def test_post_with_retry_retries_after_http_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 is retried; the following 200 succeeds, for two total requests."""

    monkeypatch.setattr(time, "sleep", _no_sleep)
    success: dict[str, object] = {"data": {"Media": {"id": 2, "episodes": 24}}}
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, API_URL, json={"data": None}, status=429)
        rsps.add(responses.POST, API_URL, json=success, status=200)
        body = _post_with_retry(QUERY, {"id": 2})
        assert len(rsps.calls) == 2
    assert body == success


def test_post_with_retry_retries_on_throttle_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A soft-throttle (HTTP 200 + throttle error payload) takes the same retry path as a 429."""

    monkeypatch.setattr(time, "sleep", _no_sleep)
    throttled: dict[str, object] = {
        "data": None,
        "errors": [{"message": "Too Many Requests", "status": 429}],
    }
    success: dict[str, object] = {"data": {"Media": {"id": 3, "episodes": 1}}}
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, API_URL, json=throttled, status=200)
        rsps.add(responses.POST, API_URL, json=success, status=200)
        body = _post_with_retry(QUERY, {"id": 3})
        assert len(rsps.calls) == 2
    assert body == success
