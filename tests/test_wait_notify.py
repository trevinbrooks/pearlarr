# pyright: strict
"""Tests for the Notifier pushes: the grab embed and the wait-complete summary.

``Notifier.push_wait_summary`` posts the wait-pass outcome (colored by its
worst outcome class) to Discord and/or a generic webhook; ``push_grab`` posts
the per-title grab embed. Both are best-effort (the engine swallows their
errors), so these pin the happy paths and the no-url no-op.
"""

import time

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

from .builders import make_logger


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


def test_push_wait_summary_builds_discord_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[DiscordEmbed] = []

    def fake_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url
        pushes.append(embed)

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())

    assert notifier.push_wait_summary(arr=Arr.RADARR, result=_result()) is True
    embed = pushes[0]
    assert embed.title == "Radarr wait complete"
    assert "2 imported · 1 left · 1 failed" in embed.description
    names = [field.name for field in embed.fields]
    assert names == ["Imported (2)", "Left for a later run (1)", "Failed (1)"]
    assert "Frieren" in embed.fields[0].value
    # Deferred/failed rows carry the outcome detail; a failure colors the embed red.
    assert embed.fields[2].value == "Bleach TYBW — download errored; left pending"
    assert embed.color == COLOR_FAILED


def test_push_wait_summary_all_imported_is_green(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[DiscordEmbed] = []

    def fake_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url
        pushes.append(embed)

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert pushes[0].color == COLOR_SUCCESS


def test_push_wait_summary_deferred_only_is_orange(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[DiscordEmbed] = []

    def fake_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url
        pushes.append(embed)

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    notifier = Notifier(discord_url="https://discord.example", logger=make_logger())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.STILL_IMPORTING),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert pushes[0].color == COLOR_DEFERRED


def test_pushes_are_paced_only_within_a_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pacing lives in the Notifier (discord_push is a pure POST), so a single or
    # final push never pays a trailing sleep; only a burst's later pushes wait
    # out the remainder of the 1s spacing.
    sleeps: list[float] = []
    now = {"t": 100.0}

    def fake_monotonic() -> float:
        return now["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    def fake_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url, embed

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
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


def test_push_grab_builds_linked_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[DiscordEmbed] = []

    def fake_discord_push(*, url: str, embed: DiscordEmbed) -> None:
        del url
        pushes.append(embed)

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
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
