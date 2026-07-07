# pyright: strict
"""Tests for the Notifier pushes: the grab embed and the wait-complete summary.

``Notifier.push_wait_summary`` posts the wait-pass outcome (colored by its
worst outcome class) to Discord and/or a generic webhook; ``push_grab`` posts
the per-title grab embed. Both are best-effort, so these pin the happy paths,
the no-url no-op, and the containment invariant: a notification failure warns
and returns False, it must never abort a grab or the end-of-run cache save.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import pytest
import requests

from seadexarr.modules import notify
from seadexarr.modules.config import Arr
from seadexarr.modules.discord import (
    COLOR_DEFERRED,
    COLOR_FAILED,
    COLOR_GRAB,
    COLOR_SUCCESS,
    DiscordEmbed,
    EmbedField,
)
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.notify import Notifier
from seadexarr.modules.wait_view import WaitOutcomeRow, WaitResult

from .builders import SEP, make_logger
from .fakes import CaptureHandler


@pytest.fixture
def pushes(monkeypatch: pytest.MonkeyPatch) -> list[DiscordEmbed]:
    """Route ``notify.discord_push`` into a recording list (no network)."""

    recorded: list[DiscordEmbed] = []

    def fake_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url
        recorded.append(embed)

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    return recorded


@contextmanager
def _capture(logger: logging.Logger) -> Generator[CaptureHandler]:
    """Collect the notifier's WARNING records off the shared test logger."""

    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _http_error(status: int) -> requests.HTTPError:
    """An ``HTTPError`` carrying a response, as ``raise_for_status`` raises it.

    The response url stands in for the webhook credential: the containment
    tests assert it never reaches a warning message.
    """

    response = requests.Response()
    response.status_code = status
    response.url = "https://discord.example/api/webhooks/1/secret-token"
    return requests.HTTPError(response=response)


def _result() -> WaitResult:
    return WaitResult(
        (
            WaitOutcomeRow("Frieren", Outcome.IMPORTED),
            WaitOutcomeRow("Apothecary Diaries", Outcome.IMPORTED),
            WaitOutcomeRow("Spy x Family", Outcome.DOWNLOAD_TIMED_OUT),
            WaitOutcomeRow("Bleach TYBW", Outcome.DOWNLOAD_ERRORED),
        ),
        elapsed_s=4264,
    )


def test_push_wait_summary_no_url_is_noop() -> None:
    notifier = Notifier(discord_url=None, webhook_url=None, logger=make_logger())

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is False


def test_push_wait_summary_empty_result_is_noop() -> None:
    notifier = Notifier(discord_url="https://discord", webhook_url="https://hook", logger=make_logger())

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=WaitResult((), 0.0)) is False


def test_push_wait_summary_posts_to_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[tuple[str, dict[str, object], int]] = []

    def fake_post(url: str, *, json: dict[str, object], timeout: int) -> object:
        posts.append((url, json, timeout))
        return object()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = Notifier(discord_url=None, webhook_url="https://hook.example", logger=make_logger())

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is True
    url, payload, _timeout = posts[0]
    assert url == "https://hook.example"
    assert payload["imported"] == 2
    assert payload["failed"] == 1


def test_push_wait_summary_webhook_failure_warns_and_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # The generic-webhook twin of the Discord containment: a request failure is
    # warned about and swallowed, never propagated to the finalize tail.
    def raising_post(url: str, *, json: dict[str, object], timeout: int) -> object:
        del url, json, timeout
        raise requests.ConnectionError("webhook down")

    monkeypatch.setattr(requests, "post", raising_post)
    logger = make_logger()
    notifier = Notifier(discord_url=None, webhook_url="https://hook.example", logger=logger)

    with _capture(logger) as handler:
        posted = notifier.push_wait_summary(arr=Arr.SONARR, result=_result())  # must not raise

    assert posted is False
    [warning] = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    # The exception is never interpolated: its str embeds the webhook URL,
    # which IS the credential. The config key points the user at the fix.
    assert warning == "Wait-report webhook POST failed (ConnectionError) - check notifications.wait_webhook_url"
    assert "hook.example" not in warning


def test_push_wait_summary_builds_discord_embed(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())

    assert notifier.push_wait_summary(arr=Arr.RADARR, result=_result()) is True
    embed = pushes[0]
    assert embed.title == "Radarr wait complete"
    assert f"2 imported{SEP}1 left{SEP}1 failed" in embed.description
    names = [field.name for field in embed.fields]
    assert names == ["Imported (2)", "Left for a later run (1)", "Failed (1)"]
    assert "Frieren" in embed.fields[0].value
    # Deferred/failed rows carry the outcome detail; a failure colors the embed red.
    assert embed.fields[2].value == "Bleach TYBW — download errored; left pending"
    assert embed.color == COLOR_FAILED


def test_push_wait_summary_all_imported_is_green(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert pushes[0].color == COLOR_SUCCESS


def test_push_wait_summary_deferred_only_is_orange(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.STILL_IMPORTING),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert pushes[0].color == COLOR_DEFERRED


def test_pushes_are_paced_only_within_a_burst(
    monkeypatch: pytest.MonkeyPatch,
    pushes: list[DiscordEmbed],
) -> None:
    # Pacing lives in the Notifier (discord_push is a pure POST), so a single or
    # final push never pays a trailing sleep; only a burst's later pushes wait
    # out the remainder of the 1s spacing.
    del pushes  # the pushes fixture supplies the no-network discord_push
    sleeps: list[float] = []
    now = {"t": 100.0}

    def fake_monotonic() -> float:
        return now["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert sleeps == []  # the first (and possibly only) push never sleeps

    now["t"] += 0.25  # a burst: the next push arrives 0.25s later
    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert sleeps == [0.75]  # topped up to the 1s spacing

    now["t"] += 5.0  # a slow follow-up needs no pacing
    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert sleeps == [0.75]


def test_push_grab_builds_linked_embed(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())

    posted = notifier.push_grab(
        arr_title="Sousou no Frieren",
        al_title="Frieren: Beyond Journey's End",
        seadex_url="https://releases.moe/154587/",
        fields=[EmbedField(name="n", value="v")],
        thumb_url="https://img.anili.st/cover.png",
    )

    assert posted is True
    embed = pushes[0]
    assert embed.author_name == "Sousou no Frieren"
    assert embed.title == "Frieren: Beyond Journey's End"
    assert embed.url == "https://releases.moe/154587/"
    assert embed.color == COLOR_GRAB
    assert embed.fields == (EmbedField(name="n", value="v"),)
    assert embed.thumb_url == "https://img.anili.st/cover.png"


def _push_grab(notifier: Notifier) -> bool:
    """Drive one grab push with minimal embed data (the containment tests' focus)."""

    return notifier.push_grab(
        arr_title="Show",
        al_title="Show",
        seadex_url="https://releases.moe/1",
        fields=[],
        thumb_url=None,
    )


def test_push_grab_discord_failure_warns_and_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # The invariant the engine relies on: a Discord failure is contained in
    # _push (warn, False) - it must never abort the grab that triggered it.
    def raising_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url, embed
        raise requests.ConnectionError("discord down")

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    logger = make_logger()
    notifier = Notifier(discord_url="https://discord.example", logger=logger)

    with _capture(logger) as handler:
        posted = _push_grab(notifier)  # must not raise

    assert posted is False
    [warning] = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    # The exception type only - a requests exception's str embeds the webhook
    # URL (the credential) - and the config key to check.
    assert warning == "Discord notification failed (ConnectionError) - check notifications.discord_url"
    assert "discord.example" not in warning


def test_push_grab_4xx_disables_discord_for_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 4xx (e.g. a deleted webhook's 404) is permanent: one actionable warning,
    # then Discord pushes are disabled - no warn-per-grab retry storm.
    posts: list[str] = []

    def raising_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del embed
        posts.append(url)
        raise _http_error(404)

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    logger = make_logger()
    notifier = Notifier(discord_url="https://discord.example", logger=logger)

    with _capture(logger) as handler:
        assert _push_grab(notifier) is False
        assert _push_grab(notifier) is False

    assert posts == ["https://discord.example"]  # the dead webhook is POSTed once
    assert notifier.enabled is False
    warnings = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    assert warnings == [
        "Discord notification failed (HTTP 404) - disabling Discord notifications "
        "for this run; check notifications.discord_url",
    ]
    assert all("secret-token" not in message for message in warnings)


def test_push_grab_429_stays_per_push_and_names_the_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 429 is Discord throttling a HEALTHY webhook (a first-sync burst can outrun
    # the 1s pacing), not a dead one: it must NOT disable pushes for the run, and
    # the warning must name the rate limit rather than blaming the config.
    posts: list[str] = []

    def raising_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del embed
        posts.append(url)
        raise _http_error(429)

    def no_sleep(seconds: float) -> None:
        del seconds  # the second push would otherwise pay the real 1s pacing

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    monkeypatch.setattr(time, "sleep", no_sleep)
    logger = make_logger()
    notifier = Notifier(discord_url="https://discord.example", logger=logger)

    with _capture(logger) as handler:
        assert _push_grab(notifier) is False
        assert _push_grab(notifier) is False

    assert notifier.enabled is True  # NOT disabled - later pushes still go out
    assert posts == ["https://discord.example"] * 2  # both pushes were attempted
    warnings = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    assert (
        warnings
        == [
            "Discord notification failed (HTTP 429) - rate limited by Discord; later notifications will still be sent",
        ]
        * 2
    )
    assert all("discord_url" not in message for message in warnings)  # config isn't at fault
    assert all("secret-token" not in message for message in warnings)


def test_push_grab_5xx_stays_per_push(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 5xx is transient (Discord hiccup): keep trying - and warning - per push.
    def raising_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url, embed
        raise _http_error(500)

    def no_sleep(seconds: float) -> None:
        del seconds  # the second push would otherwise pay the real 1s pacing

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    monkeypatch.setattr(time, "sleep", no_sleep)
    logger = make_logger()
    notifier = Notifier(discord_url="https://discord.example", logger=logger)

    with _capture(logger) as handler:
        assert _push_grab(notifier) is False
        assert _push_grab(notifier) is False

    assert notifier.enabled is True
    warnings = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    assert warnings == ["Discord notification failed (HTTP 500) - check notifications.discord_url"] * 2
