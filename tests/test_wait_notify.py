# pyright: strict
"""Tests for the walk-away graft: the wait-complete notification push.

``Notifier.push_wait_summary`` posts the wait-pass outcome to Discord and/or a
generic webhook. It is best-effort (the engine swallows its errors), so these
pin the happy paths and the no-url no-op.
"""

import pytest
import requests

from seadexarr.modules import notify
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.notify import Notifier
from seadexarr.modules.wait_view import WaitOutcomeRow, WaitResult


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
    notifier = Notifier(discord_url=None, webhook_url=None)

    assert notifier.push_wait_summary(arr="sonarr", result=_result()) is False


def test_push_wait_summary_empty_result_is_noop() -> None:
    notifier = Notifier(discord_url="https://discord", webhook_url="https://hook")

    assert notifier.push_wait_summary(arr="sonarr", result=WaitResult((), 0.0)) is False


def test_push_wait_summary_posts_to_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[tuple[str, dict[str, object], int]] = []

    def fake_post(url: str, *, json: dict[str, object], timeout: int) -> object:
        posts.append((url, json, timeout))
        return object()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = Notifier(discord_url=None, webhook_url="https://hook.example")

    assert notifier.push_wait_summary(arr="sonarr", result=_result()) is True
    url, payload, _timeout = posts[0]
    assert url == "https://hook.example"
    assert payload["imported"] == 2
    assert payload["failed"] == 1


def test_push_wait_summary_builds_discord_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[tuple[str, list[dict[str, str]]]] = []

    def fake_discord_push(
        *,
        url: str,
        arr_title: str,
        al_title: str,
        seadex_url: str,
        fields: list[dict[str, str]],
        thumb_url: str | None,
    ) -> bool:
        del url, al_title, seadex_url, thumb_url
        pushes.append((arr_title, fields))
        return True

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    notifier = Notifier(discord_url="https://discord.example")

    assert notifier.push_wait_summary(arr="radarr", result=_result()) is True
    arr_title, fields = pushes[0]
    names = [field["name"] for field in fields]
    assert names == ["Imported", "Left for a later run", "Failed"]
    assert "Frieren" in fields[0]["value"]
    assert "Radarr wait complete" in arr_title
