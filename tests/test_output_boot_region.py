# pyright: strict
# pyright: reportPrivateUsage=false
# ^ the frame-text tests exercise BootRegion._frame_text (a pure render helper).
"""Tests for the boot cockpit's renderer side (``output.boot_region``) + parity.

Renderer half: the RichRenderer's boot region draws the banner, graduates
finished steps to durable scrollback lines, prints the capstone, degrades to the
heads-up digest on a non-live rich console, and tears its single Live slot down
on the boot section's close (rich refuses two concurrent Lives, so a leaked slot
raises). Parity half: the file/plain ledger lines the LegacyRenderer echoes for
a scripted boot flow are byte-identical to the pre-PR3 view's output - the
goldens below were captured against ``boot_view`` at 78dd3e3, not derived from
the new builders - and LogCounter's tallies are unchanged.
"""

import io
import logging
from importlib.metadata import version

from rich.console import Console

from seadexarr.modules.boot_flow import BootFlow
from seadexarr.modules.console_caps import Capabilities
from seadexarr.modules.log import LOG_NAME, LogCounter, RichConsoleHandler
from seadexarr.modules.manual_import import OutcomeCategory
from seadexarr.modules.output import (
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    Event,
    LegacyRenderer,
    OutputHub,
    RichRenderer,
    RunStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    install_hub,
)
from seadexarr.modules.output.boot_region import (
    BootRegion,
    format_step_secs,
    graduation_line,
    ready_line,
    slow_line,
)

from .builders import SEP
from .fakes import FakeClock, strip_ansi

_SECTION = ScopeId(ScopeKind.BOOT_SECTION, 900)
_SECTION_TWO = ScopeId(ScopeKind.BOOT_SECTION, 901)

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
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=width)
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
        assert "SeaDexArr v9.9.9" in lines
        assert "  Data directory: /data/dir" in lines

    def test_steps_graduate_to_durable_scrollback_lines(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
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
            _started("Refreshing mappings"),
            BootStepSlow(scope=ScopeId(ScopeKind.BOOT_STEP, 1), label="Refreshing mappings"),
            _finished("Refreshing mappings"),
            ScopeClosed(scope=_SECTION),
        )

        assert "  Refreshing mappings…" in _plain(narrow).splitlines()

        live_renderer, wide = _renderer()
        _feed(
            live_renderer,
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

        _feed(renderer, _started("One"), _finished("One"), ScopeClosed(scope=_SECTION))
        # A second section must start a FRESH Live: rich refuses two concurrent
        # Lives on one console, so a leaked slot would raise LiveError here.
        _feed(renderer, _started("Two", serial=2), _finished("Two", serial=2), ScopeClosed(scope=_SECTION_TWO))

        out = _plain(stream)
        assert f"  ✔ One{SEP}0.61s" in out
        assert f"  ✔ Two{SEP}0.61s" in out

    def test_begin_cycle_and_close_stop_a_live_slot(self) -> None:
        renderer, _stream = _renderer()

        _feed(renderer, _started("One"))
        renderer.begin_cycle()
        _feed(renderer, _started("Two", serial=2))
        renderer.close()
        _feed(renderer, _started("Three", serial=3), ScopeClosed(scope=_SECTION))

    def test_progress_updates_never_raise_and_stay_transient(self) -> None:
        renderer, stream = _renderer()

        _feed(
            renderer,
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
            _started("Reading config"),
            _finished("Reading config"),
            BootReady(elapsed_s=0.61),
            ScopeClosed(scope=_SECTION),
        )

        out = _plain(stream)
        assert "SeaDexArr" not in out
        assert "✔" not in out  # no graduated ledger line (spinner frames may remain)
        assert "ready in" not in out

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
    region = BootRegion(lambda: None)
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


# --- strangler byte-parity: file/plain lines + LogCounter tallies ---------------------


def _scripted_flow(clock: FakeClock) -> None:
    """The parity script: banner, three steps (note/progress/warn), a mid-boot
    warning, capstone - the exact sequence the pre-PR3 goldens were captured from."""

    boot = BootFlow("/data/dir", clock=clock)
    boot.banner()
    with boot.step("Reading config") as step:
        clock.tick(0.61)
        step.note("config.yml")
    logging.getLogger(LOG_NAME).warning("config file is readable by other users")
    with boot.step("Refreshing mappings") as step:
        clock.tick(0.30)
        step.progress(0.5, "1/2 MB")
        clock.tick(0.31)
    with boot.step("Connecting to qBittorrent") as step:
        clock.tick(0.05)
        step.warn("not configured - preview mode")
    boot.end_section()
    boot.close()


def _run_scripted(app_logger: logging.Logger, *, console: Console | None) -> list[str]:
    """Run the script through the PR3 pipeline; return the file-surface lines."""

    app_logger.setLevel(logging.INFO)  # the goldens were captured at INFO
    buffer = io.StringIO()
    file_handler = logging.StreamHandler(buffer)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    app_logger.addHandler(file_handler)
    if console is not None:
        app_logger.addHandler(RichConsoleHandler(console))
    install_hub(OutputHub([LegacyRenderer()]))
    _scripted_flow(FakeClock())
    return buffer.getvalue().splitlines()


_BANNER = f"SeaDexArr v{version('seadexarr')}"

# Captured from boot_view at 78dd3e3 (pre-PR3): no rich console -> LogBootView,
# ascii glyphs, the one-time heads-up between the mid-boot warning and the
# mappings graduation.
_GOLDEN_PLAIN = [
    f"INFO: {_BANNER}",
    "INFO: ",
    "INFO:   Data directory: /data/dir",
    "INFO:   ok Reading config · config.yml · 0.61s",
    "WARNING: config file is readable by other users",
    "INFO:   Refreshing mappings...",
    "INFO:   ok Refreshing mappings · 1/2 MB · 0.61s",
    "INFO:   ~ Connecting to qBittorrent · not configured - preview mode · 0.05s",
    "INFO:   ready in 1.27s",
]

# Captured from boot_view at 78dd3e3 (pre-PR3): live TTY -> LiveBootView,
# unicode glyphs, NO heads-up line (the spinner carried liveness).
_GOLDEN_TTY = [
    f"INFO: {_BANNER}",
    "INFO: ",
    "INFO:   Data directory: /data/dir",
    "INFO:   ✔ Reading config · config.yml · 0.61s",
    "WARNING: config file is readable by other users",
    "INFO:   ✔ Refreshing mappings · 1/2 MB · 0.61s",
    "INFO:   ⚠ Connecting to qBittorrent · not configured - preview mode · 0.05s",
    "INFO:   ready in 1.27s",
]


class TestLedgerParity:
    def test_plain_surface_is_byte_identical_to_the_pre_pr3_golden(self, app_logger: logging.Logger) -> None:
        assert _run_scripted(app_logger, console=None) == _GOLDEN_PLAIN

    def test_live_tty_file_surface_is_byte_identical_to_the_pre_pr3_golden(self, app_logger: logging.Logger) -> None:
        console = Console(file=io.StringIO(), force_terminal=True, width=100)
        assert _run_scripted(app_logger, console=console) == _GOLDEN_TTY

    def test_log_counter_tallies_are_unchanged(self, app_logger: logging.Logger) -> None:
        # Pre-PR3 the view logged 8 INFO lines + the 1 direct WARNING on this
        # script; the echo must keep feeding LogCounter identically.
        counter = LogCounter()
        app_logger.addFilter(counter)

        _run_scripted(app_logger, console=None)

        assert counter.counts.get(logging.INFO, 0) == 8
        assert counter.counts.get(logging.WARNING, 0) == 1
