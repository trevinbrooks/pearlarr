"""Tests for the walk-away grafts: the run-report artifact + the completion push.

``write_wait_report`` serializes a :class:`WaitResult` to a markdown + json pair
next to the log file; ``Notifier.push_wait_summary`` posts that outcome to Discord
and/or a generic webhook. Both are best-effort (the engine swallows their errors),
so these pin the happy paths and the no-url no-op.
"""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import requests

from seadexarr.modules import notify
from seadexarr.modules.config import Arr
from seadexarr.modules.manual_import import Outcome
from seadexarr.modules.notify import Notifier
from seadexarr.modules.seadex_arr import write_wait_report
from seadexarr.modules.wait_view import WaitOutcomeRow, WaitResult

_WHEN = datetime(2026, 6, 25, 12, 0, 0)


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


def _logger_to(tmp_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"wait-report-{tmp_path}")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(RotatingFileHandler(str(tmp_path / "SeaDexArr.log"), delay=True))
    return logger


def test_write_wait_report_emits_md_and_json(tmp_path: Path) -> None:
    logger = _logger_to(tmp_path)

    md_path, json_path = write_wait_report(logger, Arr.SONARR, _result(), when=_WHEN)

    assert Path(md_path).parent == tmp_path  # lands next to the log file
    assert Path(md_path).name == "run-report-sonarr-20260625-120000.md"

    md = Path(md_path).read_text(encoding="utf-8")
    assert "2 imported - 1 left - 1 failed" in md
    assert "| ✔ imported | Frieren |" in md
    assert "| ⚠ timed out | Spy x Family |" in md
    assert "| ✖ errored | Bleach TYBW |" in md

    payload: dict[str, Any] = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["imported"] == 2
    assert payload["left"] == 1
    assert payload["failed"] == 1
    assert payload["arr"] == "sonarr"
    assert [row["label"] for row in payload["rows"]] == [
        "Frieren", "Apothecary Diaries", "Spy x Family", "Bleach TYBW",
    ]


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
