# pyright: strict
"""Tests for `pearlarr replay`: the JSON envelope stream re-rendered as text.

Replay is a generic envelope formatter with no per-event knowledge, so these
pin: a golden round-trip (real `JsonRenderer` output fed back through replay
reproduces the `ts LEVEL [bracket] message k=v` grammar, with path > component >
origin > event bracket precedence and an appended traceback); tolerance of an
unknown newer event; the old-capture fallback when `component` is absent;
docker-style interleaved non-event lines counted and skipped; a foreign
`schema_version` warned once; and the failure arms (no events / unreadable
file) returning False and exiting 1.
"""

import io
import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pearlarr.cli import pearlarr_cli
from pearlarr.cli import replay as cli_replay
from pearlarr.config import Arr
from pearlarr.log import EntryState
from pearlarr.output import (
    CacheStatsReported,
    CapturedTrace,
    Diagnostic,
    EntryHeader,
    Event,
    ItemStarted,
    JsonRenderer,
    RunStarted,
    ScanStarted,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
)

_EPOCH = 1_751_990_000.0
_TS = datetime.fromtimestamp(_EPOCH).strftime("%Y-%m-%d %H:%M:%S")

# The synthetic hand-built lines below fix `time` to this instant.
_FIXED_TIME = "2026-01-01T18:00:00+00:00"
_FIXED_TS = "2026-01-01 18:00:00"

_ENTRY = ScopeId(ScopeKind.ENTRY, 2)


def _json_capture(events: list[Event]) -> str:
    """The real JSON stream (one object per line) for `events`, at the DEBUG floor."""

    stream = io.StringIO()
    renderer = JsonRenderer(stream)
    renderer.set_level(logging.DEBUG)
    for event in events:
        renderer.handle(event, _EPOCH)
    return stream.getvalue()


def _envelope(**fields: object) -> str:
    """One hand-built envelope line, keys kept in insertion order."""

    return json.dumps(fields, ensure_ascii=False)


def _replay(
    tmp_path: Path,
    content: str,
    capsys: pytest.CaptureFixture[str],
) -> tuple[bool, list[str], str]:
    """Write `content` to a capture file, replay it, and return (result, out lines, err)."""

    path = tmp_path / "capture.jsonl"
    path.write_text(content, encoding="utf-8")
    result = cli_replay(str(path))
    captured = capsys.readouterr()
    return result, captured.out.splitlines(), captured.err


def test_golden_round_trip_reproduces_the_text_grammar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        raise ValueError("request failed")
    except ValueError as exc:
        trace = CapturedTrace.from_exception(exc)
    events: list[Event] = [
        RunStarted(version="v1.0.0", data_dir="/data"),
        Diagnostic(severity=Severity.ERROR, message="boom", origin="anilist", trace=trace),
        ScanStarted(arr=Arr.SONARR, total=182),
        ItemStarted(arr=Arr.SONARR, index=3, total=182, title="Frieren"),
        ScopeOpened(scope=_ENTRY, label="Frieren"),
        EntryHeader(state=EntryState.CHECKING, title="Frieren", scope=_ENTRY),
        CacheStatsReported(
            entries=1, torrent_hashes=1, anilist_meta=1, sonarr_parse=1, pending_imports=0, size_bytes=4096
        ),
    ]

    result, out, err = _replay(tmp_path, _json_capture(events), capsys)

    assert result is True
    assert err == ""  # schema 1, nothing skipped
    # component bracket for the process banner and the cli command event.
    assert f"{_TS} INFO [run] Pearlarr started version=v1.0.0 data_dir=/data" in out
    assert (
        f"{_TS} INFO [cli] cache stats entries=1 torrent_hashes=1 anilist_meta=1 "
        f"sonarr_parse=1 pending_imports=0 size_bytes=4096"
    ) in out
    # path wins over component for a scoped event.
    assert f"{_TS} INFO [sonarr › [3/182] Frieren › entry] checking title=Frieren" in out
    # the diagnostic's bracket is its origin/component, and the traceback appends verbatim.
    assert f"{_TS} ERROR [anilist] boom" in out
    assert "ValueError: request failed" in out


def test_an_unknown_future_event_renders_best_effort(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    line = _envelope(
        schema_version=1,
        time=_FIXED_TIME,
        event="future_thing",
        level="INFO",
        message="the future is here",
        component="future_mod",
        shiny="yes",
    )

    result, out, err = _replay(tmp_path, line + "\n", capsys)

    assert result is True
    assert err == ""
    # No per-event knowledge: bracket is the component, the new field rides the tail.
    assert out == [f"{_FIXED_TS} INFO [future_mod] the future is here shiny=yes"]


def test_a_pre_component_capture_falls_back_to_origin_then_event(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Old captures predate `component`: origin, then the event name, seed the bracket.
    with_origin = _envelope(
        schema_version=1,
        time=_FIXED_TIME,
        event="diagnostic",
        level="ERROR",
        message="boom",
        origin="runlock",
    )
    bare = _envelope(
        schema_version=1,
        time=_FIXED_TIME,
        event="cache_removed",
        level="INFO",
        message="cache removed",
        cache_path="/d/cache.db",
    )

    result, out, err = _replay(tmp_path, f"{with_origin}\n{bare}\n", capsys)

    assert result is True
    assert err == ""
    assert out == [
        f"{_FIXED_TS} ERROR [runlock] boom",
        f"{_FIXED_TS} INFO [cache_removed] cache removed cache_path=/d/cache.db",
    ]


def test_interleaved_non_event_lines_are_skipped_and_counted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    envelope = _envelope(
        schema_version=1,
        time=_FIXED_TIME,
        event="run_started",
        level="INFO",
        message="Pearlarr started",
        component="run",
    )
    content = "\n".join(
        [
            "plain stderr text a container interleaved",
            "",  # blank lines are skipped silently, not counted
            envelope,
            "not json at all {oops}",
            "[]",  # valid JSON but not an object
            '{"not": "an envelope"}',  # object, but missing the envelope essentials
        ]
    )

    result, out, err = _replay(tmp_path, content + "\n", capsys)

    assert result is True  # skips never fail the command
    assert out == [f"{_FIXED_TS} INFO [run] Pearlarr started"]
    assert "Skipped 4 non-event lines" in err


def test_a_single_skip_is_singular(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    envelope = _envelope(
        schema_version=1, time=_FIXED_TIME, event="run_started", level="INFO", message="started", component="run"
    )

    _result, _out, err = _replay(tmp_path, f"garbage\n{envelope}\n", capsys)

    assert "Skipped 1 non-event line" in err
    assert "non-event lines" not in err  # singular, no trailing s


def test_a_foreign_schema_version_warns_once_and_still_renders(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = "\n".join(
        _envelope(
            schema_version=2,
            time=_FIXED_TIME,
            event="run_started",
            level="INFO",
            message=f"line {n}",
            component="run",
        )
        for n in range(2)
    )

    result, out, err = _replay(tmp_path, lines + "\n", capsys)

    assert result is True
    assert len(out) == 2  # both lines still render best-effort
    assert err.count("schema_version 2") == 1  # one heads-up for the whole stream
    assert "rendering best-effort" in err


def test_no_events_found_is_a_loud_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result, out, err = _replay(tmp_path, "just some plain text\nnothing json here\n", capsys)

    assert result is False
    assert out == []
    assert "No events found" in err
    assert "log_format: json" in err


def test_an_empty_capture_is_a_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result, out, err = _replay(tmp_path, "", capsys)

    assert result is False
    assert out == []
    assert "No events found" in err


def test_a_missing_file_is_reported_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli_replay("/no/such/capture.jsonl")
    err = capsys.readouterr().err

    assert result is False
    assert "Cannot read /no/such/capture.jsonl" in err


def test_stdin_is_read_with_a_dash(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    envelope = _envelope(
        schema_version=1,
        time=_FIXED_TIME,
        event="run_started",
        level="INFO",
        message="from stdin",
        component="run",
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(envelope + "\n"))

    result = cli_replay("-")
    captured = capsys.readouterr()

    assert result is True
    assert captured.out.splitlines() == [f"{_FIXED_TS} INFO [run] from stdin"]


class TestCliRunnerExitCodes:
    """End-to-end through typer: the returned bool becomes the process exit code."""

    def test_success_exits_zero(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture.jsonl"
        capture.write_text(
            _envelope(
                schema_version=1,
                time=_FIXED_TIME,
                event="run_started",
                level="INFO",
                message="started",
                component="run",
            )
            + "\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(pearlarr_cli, ["replay", str(capture)])

        assert result.exit_code == 0
        assert f"{_FIXED_TS} INFO [run] started" in result.stdout

    def test_no_events_exits_one(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture.jsonl"
        capture.write_text("not an event\n", encoding="utf-8")

        result = CliRunner().invoke(pearlarr_cli, ["replay", str(capture)])

        assert result.exit_code == 1

    def test_missing_file_exits_one(self) -> None:
        result = CliRunner().invoke(pearlarr_cli, ["replay", "/no/such/file.jsonl"])

        assert result.exit_code == 1
