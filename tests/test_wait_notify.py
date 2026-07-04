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
    assert any(r.levelno == logging.WARNING and "webhook POST failed" in r.getMessage() for r in handler.records)


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
        posted = notifier.push_grab(  # must not raise
            arr_title="Show",
            al_title="Show",
            seadex_url="https://releases.moe/1",
            fields=[],
            thumb_url=None,
        )

    assert posted is False
    assert any(r.levelno == logging.WARNING and "Discord push failed" in r.getMessage() for r in handler.records)
