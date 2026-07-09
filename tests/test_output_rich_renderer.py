# pyright: strict
"""Tests for the rich console surface's diagnostic path (``output.rich_renderer``).

Pin ambient placement against the PR2 cockpit scopes (boot-ledger indent while
the boot section is open, wait indent while the wait region is open, column 0
otherwise - including mid-scan, bug-parity until PR4), the S4 floors (third-party
WARNING floor unless DEBUG; first-party INFO renders dim), the ``file_only`` and
no-rich-console no-ops, trace rendering without locals, markup literalness, and
the begin_cycle fold reset.
"""

import io
import logging

from rich.console import Console

from seadexarr.modules.config import Arr
from seadexarr.modules.log import INDENT, LOG_NAME
from seadexarr.modules.output import (
    CapturedTrace,
    Diagnostic,
    Event,
    ItemStarted,
    RichRenderer,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    diagnostic_text,
    diagnostic_threshold,
)

from .fakes import strip_ansi

_BOOT = ScopeId(ScopeKind.BOOT_SECTION, 1)
_WAIT = ScopeId(ScopeKind.WAIT_REGION, 2)


def _renderer(width: int = 120) -> tuple[RichRenderer, io.StringIO]:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=width)
    return RichRenderer(lambda: console), stream


def _feed(renderer: RichRenderer, *events: Event) -> None:
    for event in events:
        renderer.handle(event, 0.0)


def _lines(stream: io.StringIO) -> list[str]:
    return strip_ansi(stream.getvalue()).splitlines()


def _warning(message: str, origin: str = LOG_NAME) -> Diagnostic:
    return Diagnostic(severity=Severity.WARNING, message=message, origin=origin)


class TestPlacement:
    def test_open_boot_section_indents_to_the_ledger(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, ScopeOpened(scope=_BOOT, label="boot"), _warning("perms loose"))

        assert _lines(stream) == [f"{INDENT}WARNING  perms loose"]

    def test_closed_boot_section_renders_at_column_zero(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, ScopeOpened(scope=_BOOT, label="boot"), ScopeClosed(scope=_BOOT), _warning("late"))

        assert _lines(stream) == ["WARNING  late"]

    def test_open_wait_region_indents(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, ScopeOpened(scope=_WAIT, label="wait"), _warning("webhook flaked"))

        assert _lines(stream) == [f"{INDENT}WARNING  webhook flaked"]

    def test_mid_scan_stays_at_column_zero_until_pr4(self) -> None:
        """Bug parity: scan scopes aren't emitted yet, so mid-scan diagnostics
        keep today's column-0 placement (PR4 moves them in-context)."""

        renderer, stream = _renderer()

        _feed(
            renderer,
            ScanStarted(arr=Arr.SONARR, total=182),
            ItemStarted(arr=Arr.SONARR, index=12, total=182, title="Frieren"),
            _warning("mid-scan"),
        )

        assert _lines(stream) == ["WARNING  mid-scan"]

    def test_begin_cycle_resets_the_fold(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, ScopeOpened(scope=_BOOT, label="boot"))
        renderer.begin_cycle()
        _feed(renderer, _warning("fresh cycle"))

        assert _lines(stream) == ["WARNING  fresh cycle"]


class TestFloors:
    def test_third_party_info_is_floored_at_the_default_level(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, Diagnostic(severity=Severity.INFO, message="handshake ok", origin="httpx"))

        assert stream.getvalue() == ""

    def test_third_party_warning_renders_with_its_origin(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, _warning("flaky pool", origin="httpx"))

        assert _lines(stream) == ["WARNING  httpx: flaky pool"]

    def test_debug_level_admits_third_party_chatter(self) -> None:
        renderer, stream = _renderer()
        renderer.set_level(logging.DEBUG)

        _feed(renderer, Diagnostic(severity=Severity.INFO, message="handshake ok", origin="httpx"))

        assert _lines(stream) == ["httpx: handshake ok"]

    def test_first_party_info_renders_dim_without_a_badge(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, Diagnostic(severity=Severity.INFO, message="Radarr not configured", origin=LOG_NAME))

        assert _lines(stream) == ["Radarr not configured"]
        text = diagnostic_text(
            Diagnostic(severity=Severity.INFO, message="Radarr not configured", origin=LOG_NAME),
            indented=False,
        )
        assert str(text.style) == "grey50"

    def test_threshold_table(self) -> None:
        """S4: first-party keeps console_level semantics; third-party floors at
        WARNING unless the configured level is DEBUG."""

        assert diagnostic_threshold(logging.INFO, first_party=True) == logging.INFO
        assert diagnostic_threshold(logging.ERROR, first_party=True) == logging.INFO
        assert diagnostic_threshold(logging.DEBUG, first_party=True) == logging.DEBUG
        assert diagnostic_threshold(logging.INFO, first_party=False) == logging.WARNING
        assert diagnostic_threshold(logging.ERROR, first_party=False) == logging.WARNING
        assert diagnostic_threshold(logging.DEBUG, first_party=False) == logging.DEBUG
        assert diagnostic_threshold(logging.CRITICAL, first_party=False) == logging.CRITICAL


class TestNoOps:
    def test_file_only_diagnostics_never_render(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
            Diagnostic(severity=Severity.ERROR, message="containment note", origin="output.hub", file_only=True),
        )

        assert stream.getvalue() == ""

    def test_without_a_rich_console_the_renderer_no_ops(self) -> None:
        """plain/json mode: the LegacyRenderer/file path still carries the record."""

        renderer = RichRenderer(lambda: None)

        _feed(renderer, ScopeOpened(scope=_BOOT, label="boot"), _warning("quiet"))


class TestRendering:
    def test_trace_renders_a_capped_traceback_but_never_locals(self) -> None:
        """The secrets pin, renderer-side: CapturedTrace was extracted with
        show_locals=False, so a frame local's VALUE can never render."""

        renderer, stream = _renderer()
        sentinel = "hunter2-" + "sentinel"  # never a contiguous string in this file
        try:
            leaked = sentinel
            raise ValueError(f"qbit exploded holding {len(leaked)} secret bytes")
        except ValueError as exc:
            trace = CapturedTrace.from_exception(exc)

        _feed(renderer, Diagnostic(severity=Severity.ERROR, message="sync failed", origin=LOG_NAME, trace=trace))

        out = strip_ansi(stream.getvalue())
        assert out.splitlines()[0] == "ERROR    sync failed"
        assert "Traceback" in out
        assert "ValueError" in out
        assert "qbit exploded" in out
        assert sentinel not in out

    def test_messages_render_literally_never_as_markup(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, _warning("[1/182] Frieren [MARKED INCOMPLETE]"))

        assert _lines(stream) == ["WARNING  [1/182] Frieren [MARKED INCOMPLETE]"]

    def test_badge_words_match_the_legacy_look(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
            Diagnostic(severity=Severity.ERROR, message="broke", origin=LOG_NAME),
            Diagnostic(severity=Severity.CRITICAL, message="dead", origin=LOG_NAME),
        )

        assert _lines(stream) == ["ERROR    broke", "CRITICAL dead"]
