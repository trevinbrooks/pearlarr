# pyright: strict
"""Tests for the rich console surface's diagnostic path (``output.rich_renderer``).

Pin ambient placement against the cockpit scopes (boot-ledger indent while the
boot section is open, wait indent while the wait region is open, column 0
otherwise - including under bare RUN/ITEM nodes), the unwind close that empties
the frontier for a leg-fatal error, the S4 floors (third-party WARNING floor
unless DEBUG; first-party INFO renders dim), the ``file_only`` and
no-rich-console no-ops, trace rendering without locals, markup literalness,
the begin_cycle fold reset, and the durable loop lines (NextRunScheduled; the
ReleaseSkipped/GrabFailed scan routing). The ENTRY-indent arm is pinned in
``test_output_scan_render``.
"""

import io
import logging
from datetime import datetime, timedelta, timezone

from rich.console import Console

from pearlarr.modules.config import Arr
from pearlarr.modules.log import INDENT, LOG_NAME
from pearlarr.modules.output import (
    CapturedTrace,
    Diagnostic,
    Event,
    GrabFailed,
    ItemStarted,
    NextRunScheduled,
    ReleaseSkipped,
    RichRenderer,
    RunFinished,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    SkipReason,
    diagnostic_text,
    diagnostic_threshold,
)

from .fakes import strip_ansi

_BOOT = ScopeId(ScopeKind.BOOT_SECTION, 1)
_WAIT = ScopeId(ScopeKind.WAIT_REGION, 2)
_ENTRY = ScopeId(ScopeKind.ENTRY, 3)


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

    def test_mid_scan_stays_at_column_zero_under_run_and_item_alone(self) -> None:
        """RUN/ITEM are structural, not indented contexts: between entries (the
        item's own rows), a diagnostic keeps column 0. An open entry scope indents
        it - see test_output_scan_render."""

        renderer, stream = _renderer()

        _feed(
            renderer,
            ScanStarted(arr=Arr.SONARR, total=182),
            ItemStarted(arr=Arr.SONARR, index=12, total=182, title="Frieren"),
            _warning("mid-scan"),
        )

        # The scan arm renders the banner/header lines (PR4); the diagnostic
        # itself lands un-indented at column 0.
        assert _lines(stream)[-1] == "WARNING  mid-scan"

    def test_begin_cycle_resets_the_fold(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, ScopeOpened(scope=_BOOT, label="boot"))
        renderer.begin_cycle()
        _feed(renderer, _warning("fresh cycle"))

        assert _lines(stream) == ["WARNING  fresh cycle"]


class TestUnwindPlacement:
    """The other half of bootstrap's unwind emit: RunFinished clears the frontier.

    ``bootstrap`` emits RunFinished from an inner finally, before its except arms
    log the leg-fatal error (``test_unwind_teardown`` pins that ordering). These
    pin what the ordering BUYS - the error lands at column 0, whatever frame the
    leg died in - and the regression it closes.
    """

    def test_run_finished_clears_a_mid_scan_entry_to_column_zero(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
            ScanStarted(arr=Arr.SONARR, total=182),
            ItemStarted(arr=Arr.SONARR, index=12, total=182, title="Frieren"),
            ScopeOpened(scope=_ENTRY, label="Frieren"),
            RunFinished(arr=Arr.SONARR),
            _warning("Sonarr run failed"),
        )

        # RunFinished renders nothing itself; it just empties the frontier.
        assert _lines(stream)[-1] == "WARNING  Sonarr run failed"

    def test_without_the_unwind_emit_the_error_indents_under_the_entry(self) -> None:
        # The negative control - Band D review finding #7, the state Band E closes.
        renderer, stream = _renderer()

        _feed(
            renderer,
            ScanStarted(arr=Arr.SONARR, total=182),
            ItemStarted(arr=Arr.SONARR, index=12, total=182, title="Frieren"),
            ScopeOpened(scope=_ENTRY, label="Frieren"),
            _warning("Sonarr run failed"),
        )

        assert _lines(stream)[-1] == f"{INDENT}WARNING  Sonarr run failed"

    def test_run_finished_also_clears_an_open_boot_section(self) -> None:
        # A leg that dies inside RunDeps.build never opens a scan: the boot section
        # is depth-1 too, so the same close evicts it and the error lands at column 0.
        renderer, stream = _renderer()

        _feed(
            renderer,
            ScopeOpened(scope=_BOOT, label="boot"),
            RunFinished(arr=Arr.SONARR),
            _warning("cache.db was written by a newer Pearlarr"),
        )

        assert _lines(stream) == ["WARNING  cache.db was written by a newer Pearlarr"]

    def test_a_repeat_run_finished_is_a_no_op(self) -> None:
        # A repeat close must not disturb an already-empty frontier (defense in
        # depth; production emits it once per leg).
        renderer, stream = _renderer()

        _feed(
            renderer,
            ScanStarted(arr=Arr.SONARR, total=1),
            RunFinished(arr=Arr.SONARR),
            RunFinished(arr=Arr.SONARR),
            _warning("after the leg"),
        )

        assert _lines(stream)[-1] == "WARNING  after the leg"


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
        """plain/json mode: the FileLogSink/text seats still carry the record."""

        renderer = RichRenderer(lambda: None)

        _feed(renderer, ScopeOpened(scope=_BOOT, label="boot"), _warning("quiet"))


class TestDurableLoopLines:
    """PR6 Band D: the scheduled-loop footer arm + the flipped release facts."""

    # A Thursday; aware (the producer's contract) — the footer shows wall time.
    _AT = datetime(2026, 1, 1, 23, 5, tzinfo=timezone(timedelta(hours=-5)))

    def test_next_run_scheduled_prints_the_plain_footer(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, NextRunScheduled(at=self._AT))

        assert _lines(stream) == ["Next scheduled run at Thu 23:05"]

    def test_next_run_scheduled_is_hidden_by_a_raised_level(self) -> None:
        # Matches the old raw INFO record's behavior at a configured WARNING.
        renderer, stream = _renderer()
        renderer.set_level(logging.WARNING)

        _feed(renderer, NextRunScheduled(at=self._AT))

        assert stream.getvalue() == ""

    def test_release_skipped_and_grab_failed_render_on_the_scan_surface(self) -> None:
        # No longer the pass arm: the flipped producers' events draw the same
        # detail rows the raw warnings did (bytes pinned in test_scan_parity).
        renderer, stream = _renderer()

        _feed(
            renderer,
            ReleaseSkipped(group="GroupA", tracker="Nyaa", reason=SkipReason.PRIVATE_ONLY, url="https://x/1"),
            GrabFailed(group="GroupA", url="https://x/1", error="tracker down"),
        )

        assert _lines(stream) == [
            "    skipped   GroupA on Nyaa (private-only)",
            "    failed    could not grab https://x/1: tracker down; will retry next run",
        ]


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
