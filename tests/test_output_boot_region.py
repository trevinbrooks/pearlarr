# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the frame-text tests exercise BootRegion._frame_text (a pure render helper).
"""Tests for the boot cockpit's renderer side (``output.boot_region``).

The RichRenderer's boot region draws the banner, graduates finished steps to
durable scrollback lines, prints the capstone, degrades to the heads-up digest
on a non-live rich console, and tears its single Live slot down whenever the
boot section leaves the fold's frontier (ScopeClosed, ScanStarted, run/cycle
boundaries). The pure ledger-line builders are pinned directly; the file/plain
surfaces render the same events through the ``output.textline`` grammar.
"""

import io
import logging

import pytest
from rich.console import Console
from rich.live import Live

from pearlarr.modules.config import Arr
from pearlarr.modules.console_caps import Capabilities
from pearlarr.modules.log import LOG_NAME
from pearlarr.modules.manual_import import OutcomeCategory
from pearlarr.modules.output import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    CycleStarted,
    Diagnostic,
    Event,
    RichRenderer,
    RunFinished,
    RunStarted,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
)
from pearlarr.modules.output.boot_region import (
    BootRegion,
    format_step_secs,
    graduation_line,
    ready_line,
    slow_line,
)

from .builders import SEP
from .fakes import strip_ansi

_SECTION = ScopeId(ScopeKind.BOOT_SECTION, 900)
_SECTION_TWO = ScopeId(ScopeKind.BOOT_SECTION, 901)


def _opened(scope: ScopeId = _SECTION) -> ScopeOpened:
    """Production shape: the section opens via ScopeOpened before any step."""

    return ScopeOpened(scope=scope, label="boot")


_ASCII_CAPS = Capabilities(live=False, color=False, unicode=False, width=80, height=24)
_UNICODE_CAPS = Capabilities(live=False, color=True, unicode=True, width=100, height=40)


def _started(label: str, serial: int = 1) -> BootStepStarted:
    return BootStepStarted(scope=ScopeId(ScopeKind.BOOT_STEP, serial), label=label)


def _finished(
    label: str,
    serial: int = 1,
    *,
    outcome: OutcomeCategory = OutcomeCategory.SUCCESS,
    detail: str | None = None,
    elapsed_s: float = 0.61,
) -> BootStepFinished:
    return BootStepFinished(
        scope=ScopeId(ScopeKind.BOOT_STEP, serial),
        label=label,
        outcome=outcome,
        detail=detail,
        elapsed_s=elapsed_s,
    )


def _renderer(*, width: int = 100) -> tuple[RichRenderer, io.StringIO]:
    # legacy_windows pinned: Windows CI auto-detects a legacy console, which
    # would drop the caps to ASCII glyphs and break the unicode assertions.
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, legacy_windows=False, width=width)
    return RichRenderer(lambda: console), stream


def _feed(renderer: RichRenderer, *events: Event) -> None:
    for event in events:
        renderer.handle(event, 0.0)


def _plain(stream: io.StringIO) -> str:
    return strip_ansi(stream.getvalue()).replace("\r", "")


# --- the shared ledger-line grammar -------------------------------------------------


def test_graduation_line_unicode_and_ascii_glyphs() -> None:
    event = _finished("Reading config", detail="config.yml")
    assert graduation_line(event, _UNICODE_CAPS) == f"  ✔ Reading config{SEP}config.yml{SEP}0.61s"
    assert graduation_line(event, _ASCII_CAPS) == f"  ok Reading config{SEP}config.yml{SEP}0.61s"

    warned = _finished("Connecting to qBittorrent", outcome=OutcomeCategory.DEFERRED, elapsed_s=0.05)
    assert graduation_line(warned, _UNICODE_CAPS) == f"  ⚠ Connecting to qBittorrent{SEP}0.05s"
    assert graduation_line(warned, _ASCII_CAPS) == f"  ~ Connecting to qBittorrent{SEP}0.05s"

    failed = _finished("Opening cache", outcome=OutcomeCategory.FAILED, elapsed_s=0.01)
    assert graduation_line(failed, _UNICODE_CAPS) == f"  ✖ Opening cache{SEP}0.01s"
    assert graduation_line(failed, _ASCII_CAPS) == f"  x Opening cache{SEP}0.01s"


def test_slow_line_ellipsis_degrades_to_ascii_dots() -> None:
    assert slow_line("Refreshing mappings", _UNICODE_CAPS) == "  Refreshing mappings…"
    assert slow_line("Refreshing mappings", _ASCII_CAPS) == "  Refreshing mappings..."


def test_format_step_secs_bands() -> None:
    assert format_step_secs(0.024) == "0.02s"
    assert format_step_secs(15.34) == "15.3s"
    assert format_step_secs(64.0) == "1m 04s"
    assert ready_line(1.27) == "  ready in 1.27s"


# --- the rich console's boot region --------------------------------------------------


class TestBootRegion:
    def test_banner_renders_title_blank_and_data_dir(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, RunStarted(version="v9.9.9", data_dir="/data/dir"))

        lines = _plain(stream).splitlines()
        assert "Pearlarr v9.9.9" in lines
        assert "  Data directory: /data/dir" in lines

    def test_steps_graduate_to_durable_scrollback_lines(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
            _opened(),
            _started("Reading config"),
            _finished("Reading config", detail="config.yml"),
            _started("Connecting to qBittorrent", serial=2),
            _finished(
                "Connecting to qBittorrent",
                serial=2,
                outcome=OutcomeCategory.DEFERRED,
                detail="not configured - preview mode",
                elapsed_s=0.05,
            ),
            BootReady(elapsed_s=1.27),
            ScopeClosed(scope=_SECTION),
        )

        out = _plain(stream)
        assert f"  ✔ Reading config{SEP}config.yml{SEP}0.61s" in out
        assert f"  ⚠ Connecting to qBittorrent{SEP}not configured - preview mode{SEP}0.05s" in out
        assert "  ready in 1.27s" in out

    def test_labels_render_literally_never_as_markup(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
            _opened(),
            _started("[1/2] weird [bold]label[/bold]"),
            _finished("[1/2] weird [bold]label[/bold]"),
            ScopeClosed(scope=_SECTION),
        )

        assert "[1/2] weird [bold]label[/bold]" in _plain(stream)

    def test_slow_heads_up_prints_only_on_a_non_live_console(self) -> None:
        # width 20 < MIN_LIVE_WIDTH: a rich but non-live console degrades the way
        # LogBootView did - the heads-up line carries liveness instead of a spinner.
        renderer, narrow = _renderer(width=20)

        _feed(
            renderer,
            _opened(),
            _started("Refreshing mappings"),
            BootStepSlow(scope=ScopeId(ScopeKind.BOOT_STEP, 1), label="Refreshing mappings"),
            _finished("Refreshing mappings"),
            ScopeClosed(scope=_SECTION),
        )

        assert "  Refreshing mappings…" in _plain(narrow).splitlines()

        live_renderer, wide = _renderer()
        _feed(
            live_renderer,
            _opened(),
            _started("Refreshing mappings"),
            BootStepSlow(scope=ScopeId(ScopeKind.BOOT_STEP, 1), label="Refreshing mappings"),
            _finished("Refreshing mappings"),
            ScopeClosed(scope=_SECTION),
        )
        # The live spinner shows liveness; no bare heads-up LINE lands in
        # scrollback (spinner frames carry a glyph + padding, never this form).
        assert "  Refreshing mappings…" not in _plain(wide).splitlines()

    def test_scope_closed_tears_down_the_live_slot(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("One"), _finished("One"))
        live = renderer._boot._live
        assert live is not None and live.is_started
        _feed(renderer, ScopeClosed(scope=_SECTION))
        assert renderer._boot._live is None and not live.is_started
        # A second section must start a FRESH Live: rich refuses two concurrent
        # Lives on one console, so a leaked slot would raise LiveError here.
        _feed(
            renderer,
            _opened(_SECTION_TWO),
            _started("Two", serial=2),
            _finished("Two", serial=2),
            ScopeClosed(scope=_SECTION_TWO),
        )

        out = _plain(stream)
        assert f"  ✔ One{SEP}0.61s" in out
        assert f"  ✔ Two{SEP}0.61s" in out

    def test_scan_started_tears_down_the_live_slot(self) -> None:
        # The PR4 trap: ScanStarted evicts the boot section in the fold WITHOUT
        # any ScopeClosed; the spinner must not survive over the scan output.
        renderer, _stream = _renderer()

        _feed(renderer, _opened(), _started("Fetching library"))
        live = renderer._boot._live
        assert live is not None and live.is_started
        _feed(renderer, ScanStarted(arr=Arr.SONARR, total=182))

        assert renderer._boot._live is None and not live.is_started

    def test_run_and_cycle_boundaries_tear_down_the_live_slot(self) -> None:
        # Teardown keys on the frontier departure, not on WHICH event caused it.
        for boundary in (RunStarted(version="v9.9.9", data_dir="/data"), CycleStarted(number=2)):
            renderer, _stream = _renderer()
            _feed(renderer, _opened(), _started("Fetching library"))
            live = renderer._boot._live
            assert live is not None and live.is_started
            _feed(renderer, boundary)
            assert renderer._boot._live is None and not live.is_started

    def test_run_finished_leg_boundary_evicts_the_live_and_the_next_leg_starts_fresh(self) -> None:
        # G4: within one scheduled cycle each arr leg ends with RunFinished (never a
        # begin_cycle). The leg close must evict the boot Live so the next leg's boot
        # section starts a FRESH slot — a leaked Live would be reused (stale region,
        # nested on rich's live stack), so the load-bearing pin is object identity.
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("One"))
        leg_one_live = renderer._boot._live
        assert leg_one_live is not None and leg_one_live.is_started

        _feed(renderer, RunFinished(arr=Arr.SONARR))
        assert renderer._boot._live is None and not leg_one_live.is_started

        _feed(renderer, _opened(_SECTION_TWO), _started("Two", serial=2), _finished("Two", serial=2))
        leg_two_live = renderer._boot._live
        assert leg_two_live is not None and leg_two_live.is_started
        assert leg_two_live is not leg_one_live  # a fresh slot, never the stale one

        renderer.close()
        assert renderer._boot._live is None and not leg_two_live.is_started

        out = _plain(stream)
        assert f"  ✔ Two{SEP}0.61s" in out  # leg two's ledger rendered under the fresh Live

    def test_begin_cycle_and_close_stop_a_live_slot(self) -> None:
        renderer, _stream = _renderer()

        _feed(renderer, _opened(), _started("One"))
        renderer.begin_cycle()
        _feed(renderer, _opened(), _started("Two", serial=2))
        renderer.close()
        _feed(renderer, _opened(), _started("Three", serial=3), ScopeClosed(scope=_SECTION))

    def test_a_raising_live_stop_still_prints_the_capstone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The shared spine (LiveRegion._stop_live) contains a stop() raise so a
        # failed teardown can't eat the durable print that follows it.
        real_stop = Live.stop

        def exploding_stop(live: Live) -> None:
            real_stop(live)  # the real teardown, so the console leaves live mode
            raise RuntimeError("live stop exploded")

        monkeypatch.setattr(Live, "stop", exploding_stop)
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("Fetching library"), BootReady(elapsed_s=0.61))

        assert renderer._boot._live is None
        assert "ready in 0.61s" in _plain(stream)

    def test_progress_updates_never_raise_and_stay_transient(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
            _opened(),
            _started("Refreshing mappings"),
            BootStepProgressed(scope=ScopeId(ScopeKind.BOOT_STEP, 1), fraction=0.5, detail="1/2 MB"),
            _finished("Refreshing mappings", detail="1/2 MB"),
            ScopeClosed(scope=_SECTION),
        )

        # The durable line carries the final detail; the bar frames were transient.
        assert f"  ✔ Refreshing mappings{SEP}1/2 MB{SEP}0.61s" in _plain(stream)

    def test_raised_level_silences_the_durable_boot_lines(self) -> None:
        # Parity with the logger-driven ledger: at level WARNING the INFO boot
        # lines never rendered pre-PR3 (the logger gate), so the region mirrors it.
        renderer, stream = _renderer()
        renderer.set_level(logging.WARNING)

        _feed(
            renderer,
            RunStarted(version="v9.9.9", data_dir="/data"),
            _opened(),
            _started("Reading config"),
            _finished("Reading config"),
            BootReady(elapsed_s=0.61),
            ScopeClosed(scope=_SECTION),
        )

        out = _plain(stream)
        assert "Pearlarr" not in out
        assert "✔" not in out  # no graduated ledger line (spinner frames may remain)
        assert "ready in" not in out

    def test_warning_level_keeps_first_party_info_diagnostics(self) -> None:
        # The two gates stay distinct at configured WARNING: the boot ledger
        # vanishes (logger parity, above) while a first-party INFO diagnostic
        # keeps rendering (diagnostic_threshold's console INFO floor).
        renderer, stream = _renderer()
        renderer.set_level(logging.WARNING)

        _feed(
            renderer,
            _opened(),
            _started("Reading config"),
            _finished("Reading config"),
            Diagnostic(severity=Severity.INFO, message="Radarr not configured", origin=LOG_NAME),
            ScopeClosed(scope=_SECTION),
        )

        out = _plain(stream)
        assert "✔" not in out
        assert "Radarr not configured" in out

    def test_level_source_is_live_not_a_construction_snapshot(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("One"), _finished("One"))
        renderer.set_level(logging.WARNING)
        _feed(renderer, _started("Two", serial=2), _finished("Two", serial=2), ScopeClosed(scope=_SECTION))

        out = _plain(stream)
        assert "✔ One" in out
        assert "✔ Two" not in out

    def test_without_a_rich_console_every_boot_event_no_ops(self) -> None:
        renderer = RichRenderer(lambda: None)

        _feed(
            renderer,
            RunStarted(version="v9.9.9", data_dir="/data"),
            _started("Reading config"),
            BootStepProgressed(scope=ScopeId(ScopeKind.BOOT_STEP, 1), fraction=0.5),
            BootStepSlow(scope=ScopeId(ScopeKind.BOOT_STEP, 1), label="Reading config"),
            _finished("Reading config"),
            BootReady(elapsed_s=0.61),
            ScopeClosed(scope=_SECTION),
        )


# --- the pure spinner-frame builder ---------------------------------------------------


def _frame(label: str, caps: Capabilities, fraction: float | None, detail: str | None) -> str:
    region = BootRegion(lambda: None, level_source=lambda: logging.INFO)
    region._label = label
    return region._frame_text(caps, fraction, detail).plain


class TestFrameText:
    def test_unicode_frame_draws_a_bar_with_percent_and_detail(self) -> None:
        text = _frame("Refreshing mappings", _UNICODE_CAPS, 0.5, "anime_ids.json")
        assert "50%" in text
        assert "█" in text and "░" in text  # half-filled unicode bar
        assert "anime_ids.json" in text

    def test_ascii_frame_degrades_the_bar(self) -> None:
        text = _frame("Refreshing mappings", _ASCII_CAPS, 0.5, None)
        assert "#" in text and "-" in text  # ascii fallback bar
        assert "█" not in text

    def test_bare_frame_degrades_the_ellipsis_to_ascii_dots(self) -> None:
        assert _frame("Reading config", _ASCII_CAPS, None, None) == "Reading config..."
