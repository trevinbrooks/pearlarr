"""Tests for the walk-away graft: the wait-complete notification push.

``Notifier.push_wait_summary`` posts the wait-pass outcome to Discord and/or a
generic webhook. It is best-effort (the engine swallows its errors), so these
pin the happy paths and the no-url no-op.
"""

from typing import Any

import requests

from seadexarr.modules import notify
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.notify import Notifier
from seadexarr.modules.wait_view import WaitOutcomeRow, WaitResult


def _result() -> WaitResult:
    return WaitResult(
        (
            WaitOutcomeRow("h1", "Frieren", Outcome.IMPORTED),
            WaitOutcomeRow("h2", "Apothecary Diaries", Outcome.IMPORTED),
            WaitOutcomeRow("h3", "Spy x Family", Outcome.DOWNLOAD_TIMED_OUT),
            WaitOutcomeRow("h4", "Bleach TYBW", Outcome.DOWNLOAD_ERRORED),
        ),
        elapsed_s=4264,
    )


def test_push_wait_summary_no_url_is_noop() -> None:
    notifier = Notifier(discord_url=None, webhook_url=None)

    assert notifier.push_wait_summary(arr="sonarr", result=_result()) is False


def test_push_wait_summary_empty_result_is_noop() -> None:
    notifier = Notifier(discord_url="https://discord", webhook_url="https://hook")

    assert notifier.push_wait_summary(arr="sonarr", result=WaitResult((), 0.0)) is False


def test_push_wait_summary_posts_to_webhook(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> object:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return object()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = Notifier(discord_url=None, webhook_url="https://hook.example")

    assert notifier.push_wait_summary(arr="sonarr", result=_result()) is True
    assert captured["url"] == "https://hook.example"
    assert captured["json"]["imported"] == 2
    assert captured["json"]["failed"] == 1


def test_push_wait_summary_builds_discord_fields(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_discord_push(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    notifier = Notifier(discord_url="https://discord.example")

    assert notifier.push_wait_summary(arr="radarr", result=_result()) is True
    names = [field["name"] for field in captured["fields"]]
    assert names == ["Imported", "Left for a later run", "Failed"]
    assert "Frieren" in captured["fields"][0]["value"]
    assert "Radarr wait complete" in captured["arr_title"]
