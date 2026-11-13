# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the frame-text tests exercise BootRegion._frame_text (a pure render helper).
"""Tests for the boot cockpit's renderer side (`output.boot_region`).

The RichRenderer's boot region draws the banner, graduates finished steps to
durable scrollback lines, prints the capstone, degrades to the heads-up digest
on a non-live rich console, and tears its single Live slot down whenever the
boot section leaves the fold's frontier (ScopeClosed, ScanStarted, run/cycle
boundaries). The pure ledger-line builders are pinned directly. The file/plain
surfaces render the same events through the `output.line_renderers` grammar.
"""

import io
import logging

import pytest
from rich.console import Console
from rich.live import Live

from pearlarr.config import Arr
from pearlarr.console_features import ConsoleFeatures
from pearlarr.log import LOG_NAME
from pearlarr.output import (
    ArrRunFinished,
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    Diagnostic,
    Event,
    PearlarrStarted,
    RichRenderer,
    ScanStarted,
    ScheduledRunStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
)
from pearlarr.output.boot_region import (
    BootRegion,
    ready_line,
    slow_line,
    step_finished_line,
    step_seconds,
)
from pearlarr.wait_types import OutcomeKind

from .builders import SEP
from .fakes import strip_ansi

_BOOT_SCOPE = ScopeId(ScopeKind.BOOT_SECTION, 900)
_BOOT_SCOPE_TWO = ScopeId(ScopeKind.BOOT_SECTION, 901)


def _opened(scope: ScopeId = _BOOT_SCOPE) -> ScopeOpened:
    """Production shape: the section opens via ScopeOpened before any step."""

    return ScopeOpened(scope=scope, label="boot")


_ASCII_FEATURES = ConsoleFeatures(can_redraw=False, can_color=False, can_unicode=False, width=80, height=24)
_UNICODE_FEATURES = ConsoleFeatures(can_redraw=False, can_color=True, can_unicode=True, width=100, height=40)


def _started(label: str, serial: int = 1) -> BootStepStarted:
    return BootStepStarted(scope=ScopeId(ScopeKind.BOOT_STEP, serial), label=label)


def _finished(
    label: str,
    serial: int = 1,
    *,
    outcome: OutcomeKind = OutcomeKind.IMPORTED,
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
    # would drop the features to ASCII glyphs and break the unicode assertions.
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
    assert step_finished_line(event, _UNICODE_FEATURES) == f"  ✔ Reading config{SEP}config.yml{SEP}0.61s"
    assert step_finished_line(event, _ASCII_FEATURES) == f"  ok Reading config{SEP}config.yml{SEP}0.61s"

    warned = _finished("Connecting to qBittorrent", outcome=OutcomeKind.LEFT, elapsed_s=0.05)
    assert step_finished_line(warned, _UNICODE_FEATURES) == f"  ⚠ Connecting to qBittorrent{SEP}0.05s"
    assert step_finished_line(warned, _ASCII_FEATURES) == f"  ~ Connecting to qBittorrent{SEP}0.05s"

    failed = _finished("Opening cache", outcome=OutcomeKind.FAILED, elapsed_s=0.01)
    assert step_finished_line(failed, _UNICODE_FEATURES) == f"  ✖ Opening cache{SEP}0.01s"
    assert step_finished_line(failed, _ASCII_FEATURES) == f"  x Opening cache{SEP}0.01s"


def test_slow_line_ellipsis_degrades_to_ascii_dots() -> None:
    assert slow_line("Refreshing mappings", _UNICODE_FEATURES) == "  Refreshing mappings…"
    assert slow_line("Refreshing mappings", _ASCII_FEATURES) == "  Refreshing mappings..."


def test_format_step_secs_bands() -> None:
    assert step_seconds(0.024) == "0.02s"
    assert step_seconds(15.34) == "15.3s"
    assert step_seconds(64.0) == "1m 04s"
    assert ready_line(1.27) == "  ready in 1.27s"


# --- the rich console's boot region --------------------------------------------------


class TestBootRegion:
    """Boot region: steps graduate to durable lines. The Live slot tears down at every scope/run/cycle boundary."""

    def test_banner_renders_title_blank_and_data_dir(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, PearlarrStarted(version="v9.9.9", data_dir="/data/dir"))

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
                outcome=OutcomeKind.LEFT,
                detail="not configured - preview mode",
                elapsed_s=0.05,
            ),
            BootReady(elapsed_s=1.27),
            ScopeClosed(scope=_BOOT_SCOPE),
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
            ScopeClosed(scope=_BOOT_SCOPE),
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
            ScopeClosed(scope=_BOOT_SCOPE),
        )

        assert "  Refreshing mappings…" in _plain(narrow).splitlines()

        live_renderer, wide = _renderer()
        _feed(
            live_renderer,
            _opened(),
            _started("Refreshing mappings"),
            BootStepSlow(scope=ScopeId(ScopeKind.BOOT_STEP, 1), label="Refreshing mappings"),
            _finished("Refreshing mappings"),
            ScopeClosed(scope=_BOOT_SCOPE),
        )
        # The live spinner shows liveness. No bare heads-up LINE lands in
        # scrollback (spinner frames carry a glyph + padding, never this form).
        assert "  Refreshing mappings…" not in _plain(wide).splitlines()

    def test_scope_closed_tears_down_the_live_slot(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("One"), _finished("One"))
        live = renderer._boot._live
        assert live is not None and live.is_started
        _feed(renderer, ScopeClosed(scope=_BOOT_SCOPE))
        assert renderer._boot._live is None and not live.is_started
        # A second section must start a FRESH Live: rich refuses two concurrent
        # Lives on one console, so a leaked slot would raise LiveError here.
        _feed(
            renderer,
            _opened(_BOOT_SCOPE_TWO),
            _started("Two", serial=2),
            _finished("Two", serial=2),
            ScopeClosed(scope=_BOOT_SCOPE_TWO),
        )

        out = _plain(stream)
        assert f"  ✔ One{SEP}0.61s" in out
        assert f"  ✔ Two{SEP}0.61s" in out

    def test_scan_started_tears_down_the_live_slot(self) -> None:
        # The trap: ScanStarted evicts the boot section in the fold WITHOUT
        # any ScopeClosed. The spinner must not survive over the scan output.
        renderer, _stream = _renderer()

        _feed(renderer, _opened(), _started("Fetching library"))
        live = renderer._boot._live
        assert live is not None and live.is_started
        _feed(renderer, ScanStarted(arr=Arr.SONARR, total=182))

        assert renderer._boot._live is None and not live.is_started

    def test_run_and_cycle_boundaries_tear_down_the_live_slot(self) -> None:
        # Teardown keys on the frontier departure, not on WHICH event caused it.
        for boundary in (PearlarrStarted(version="v9.9.9", data_dir="/data"), ScheduledRunStarted(number=2)):
            renderer, _stream = _renderer()
            _feed(renderer, _opened(), _started("Fetching library"))
            live = renderer._boot._live
            assert live is not None and live.is_started
            _feed(renderer, boundary)
            assert renderer._boot._live is None and not live.is_started

    def test_run_finished_arr_run_boundary_evicts_the_live_and_the_next_arr_run_starts_fresh(self) -> None:
        # G4: within one scheduled cycle each arr leg ends with ArrRunFinished (never a
        # begin_run). The leg close must evict the boot Live so the next leg's boot
        # section starts a FRESH slot - a leaked Live would be reused (stale region,
        # nested on rich's live stack), so the load-bearing pin is object identity.
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("One"))
        first_arr_run_live = renderer._boot._live
        assert first_arr_run_live is not None and first_arr_run_live.is_started

        _feed(renderer, ArrRunFinished(arr=Arr.SONARR))
        assert renderer._boot._live is None and not first_arr_run_live.is_started

        _feed(renderer, _opened(_BOOT_SCOPE_TWO), _started("Two", serial=2), _finished("Two", serial=2))
        second_arr_run_live = renderer._boot._live
        assert second_arr_run_live is not None and second_arr_run_live.is_started
        assert second_arr_run_live is not first_arr_run_live  # a fresh slot, never the stale one

        renderer.close()
        assert renderer._boot._live is None and not second_arr_run_live.is_started

        out = _plain(stream)
        assert f"  ✔ Two{SEP}0.61s" in out  # leg two's ledger rendered under the fresh Live

    def test_begin_cycle_and_close_stop_a_live_slot(self) -> None:
        renderer, _stream = _renderer()

        _feed(renderer, _opened(), _started("One"))
        renderer.begin_run()
        _feed(renderer, _opened(), _started("Two", serial=2))
        renderer.close()
        _feed(renderer, _opened(), _started("Three", serial=3), ScopeClosed(scope=_BOOT_SCOPE))

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
            ScopeClosed(scope=_BOOT_SCOPE),
        )

        # The durable line carries the final detail. The bar frames were transient.
        assert f"  ✔ Refreshing mappings{SEP}1/2 MB{SEP}0.61s" in _plain(stream)

    def test_raised_level_silences_the_durable_boot_lines(self) -> None:
        # Parity with the logger-driven ledger: at level WARNING the INFO boot
        # lines never rendered under the old logger gate, so the region mirrors it.
        renderer, stream = _renderer()
        renderer.set_level(logging.WARNING)

        _feed(
            renderer,
            PearlarrStarted(version="v9.9.9", data_dir="/data"),
            _opened(),
            _started("Reading config"),
            _finished("Reading config"),
            BootReady(elapsed_s=0.61),
            ScopeClosed(scope=_BOOT_SCOPE),
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
            ScopeClosed(scope=_BOOT_SCOPE),
        )

        out = _plain(stream)
        assert "✔" not in out
        assert "Radarr not configured" in out

    def test_level_source_is_live_not_a_construction_snapshot(self) -> None:
        renderer, stream = _renderer()

        _feed(renderer, _opened(), _started("One"), _finished("One"))
        renderer.set_level(logging.WARNING)
        _feed(renderer, _started("Two", serial=2), _finished("Two", serial=2), ScopeClosed(scope=_BOOT_SCOPE))

        out = _plain(stream)
        assert "✔ One" in out
        assert "✔ Two" not in out

    def test_without_a_rich_console_every_boot_event_no_ops(self) -> None:
        renderer = RichRenderer(lambda: None)

        _feed(
            renderer,
            PearlarrStarted(version="v9.9.9", data_dir="/data"),
            _started("Reading config"),
            BootStepProgressed(scope=ScopeId(ScopeKind.BOOT_STEP, 1), fraction=0.5),
            BootStepSlow(scope=ScopeId(ScopeKind.BOOT_STEP, 1), label="Reading config"),
            _finished("Reading config"),
            BootReady(elapsed_s=0.61),
            ScopeClosed(scope=_BOOT_SCOPE),
        )


# --- the pure spinner-frame builder ---------------------------------------------------


def _frame(label: str, features: ConsoleFeatures, fraction: float | None, detail: str | None) -> str:
    region = BootRegion(lambda: None, level_source=lambda: logging.INFO)
    region._label = label
    return region._frame_text(features, fraction, detail).plain


class TestFrameText:
    """The pure spinner-frame builder draws a unicode percent bar, degrading to ASCII without unicode support."""

    def test_unicode_frame_draws_a_bar_with_percent_and_detail(self) -> None:
        text = _frame("Refreshing mappings", _UNICODE_FEATURES, 0.5, "anime_ids.json")
        assert "50%" in text
        assert "█" in text and "░" in text  # half-filled unicode bar
        assert "anime_ids.json" in text

    def test_ascii_frame_degrades_the_bar(self) -> None:
        text = _frame("Refreshing mappings", _ASCII_FEATURES, 0.5, None)
        assert "#" in text and "-" in text  # ascii fallback bar
        assert "█" not in text

    def test_bare_frame_degrades_the_ellipsis_to_ascii_dots(self) -> None:
        assert _frame("Reading config", _ASCII_FEATURES, None, None) == "Reading config..."
