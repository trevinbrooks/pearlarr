"""Boot cockpit: a live, real-time view of startup IO before the first scan.

Startup does real work before any series is scanned - read+validate config,
download/parse the id-mapping sources (when stale), open the cache, log into
qBittorrent, fetch the library from Sonarr/Radarr, and prefetch AniList + SeaDex
metadata. This module shows that work happening, instead of a frozen screen or a
wall of ad-hoc log lines, and times each step.

It mirrors :mod:`.wait_view`: an animated cockpit on a capable TTY
(:class:`LiveBootView`) and a calm one-line-per-step digest on a non-TTY
(:class:`LogBootView`), chosen once by :func:`make_boot_view`. Each step
GRADUATES through the logger when it finishes - a glyph/color-coded ledger line
that lands in scrollback AND the plain-text file log - reusing the same
:class:`~.manual_import.OutcomeCategory` look the wait ledger uses.

Unlike the wait view (driven by an external poll loop, so single-threaded), a
boot step is a single blocking call. The live region therefore animates on rich's
own refresh thread (``auto_refresh=True``); it shares the logger's Console, so a
line logged mid-step (a graduation, a warning, the failure path) reflows ABOVE
the spinner rather than corrupting it. Every method is total: a presentation bug
degrades to a no-op and never aborts the real startup work it wraps.
"""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from importlib.metadata import PackageNotFoundError, version
from typing import final, override

from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from .console_caps import (
    Capabilities,
    TerminalEnv,
    block_bar,
    console_of,
    detect_capabilities,
    make_live,
    spinner_name,
)
from .log import INDENT, format_elapsed, indent_string, log_counter, log_styled, log_titled_rule
from .manual_import import OutcomeCategory

# Width of the live download bar (mapping refresh); only the live cockpit draws
# it, so it doesn't need to scale with the terminal like the wait cockpit's does.
_BAR_WIDTH = 16


@final
class BootStep:
    """A handle to the in-flight step: drive its progress + tune how it graduates.

    Yielded by :meth:`BootView.step`. The default graduation is SUCCESS (``✔``)
    with whatever detail was set; call :meth:`warn` for a DEFERRED (``⚠``) finish,
    or let an exception escape the ``with`` block for a FAILED (``✖``) one. The
    fields are read by the view to render/graduate the step; callers mutate them
    only through the methods.
    """

    def __init__(self, notify: Callable[[BootStep], None], label: str) -> None:
        self._notify = notify
        self.label = label
        self.detail: str | None = None
        self.fraction: float | None = None
        self.category = OutcomeCategory.SUCCESS
        # One-time heads-up flag (LogBootView): rides the step itself because an
        # id()-keyed set can collide once a dead step's address is reused.
        self.announced = False

    def progress(self, fraction: float, detail: str | None = None) -> None:
        """Report 0-1 progress (and optional detail) - refreshes the live bar."""

        self.fraction = max(0.0, min(1.0, fraction))
        if detail is not None:
            self.detail = detail
        self._notify(self)

    def note(self, text: str) -> None:
        """Set the detail shown on the graduated ledger line (e.g. "42 series")."""

        self.detail = text

    def warn(self, text: str | None = None) -> None:
        """Graduate this step as a warning (``⚠``) rather than a success."""

        self.category = OutcomeCategory.DEFERRED
        if text is not None:
            self.detail = text


class BootView(ABC):
    """The small interface the composition root drives while starting a run.

    The root logs the :meth:`banner`, runs each IO step inside :meth:`step`, calls
    :meth:`end_section` right before a per-arr scan starts logging (so the live
    region is gone first), and :meth:`close` once at the end. Every method is
    total by contract.
    """

    @abstractmethod
    def banner(self) -> None:
        """Render the instant brand+version title (once, before any step)."""

    @abstractmethod
    def step(self, label: str) -> contextlib.AbstractContextManager[BootStep]:
        """Run one IO step: animate it, time it, then graduate a ledger line."""

    @abstractmethod
    def end_section(self) -> None:
        """Tear down the live region + log the "ready" capstone before a scan.

        A run scans Sonarr then Radarr; each scan's per-series logging must start
        with no live region above it. The next :meth:`step` reopens one, so a
        section is "shared prefix + this arr's steps", capped here.
        """

    @abstractmethod
    def close(self) -> None:
        """Final teardown (idempotent) - safety net if a section was left open."""


class _DurableBootView(BootView):
    """Shared spine: time + graduate steps to the log, summarize, stay total.

    Both concrete views graduate a finished step the same way - a single
    ``logger`` call, so the durable ledger line hits the styled console (reflowed
    ABOVE any live region) AND the plain-text file log. Subclasses add only the
    live frame (:meth:`_begin` / :meth:`_on_change`) and its teardown
    (:meth:`_stop_live`).
    """

    def __init__(
        self,
        logger: logging.Logger,
        caps: Capabilities,
        *,
        time_source: Callable[[], float],
    ) -> None:
        self._logger = logger
        self._caps = caps
        self._time = time_source
        self._closed = False
        self._section_started: float | None = None
        self._section_count = 0
        self._section_failed = False
        self._section_errors_at = 0

    @final
    @override
    def banner(self) -> None:
        title = f"SeaDexArr {_app_version()}".rstrip()
        self._safe(lambda: log_titled_rule(self._logger, title, heavy=True))
        # A blank under the title gives the step ledger a gap below the header,
        # matching every other section. Printed before the Live spinner starts,
        # so it stays in durable scrollback above the transient spinner.
        self._safe(lambda: self._logger.info(""))

    @final
    @override
    def step(self, label: str) -> contextlib.AbstractContextManager[BootStep]:
        return self._step(label)

    @contextlib.contextmanager
    def _step(self, label: str) -> Generator[BootStep]:
        step = BootStep(self._notify, label)
        start = self._time()
        if self._section_started is None:
            self._section_started = start
            self._section_errors_at = self._errors_logged()
        self._safe(lambda: self._begin(step))
        failed = False
        try:
            yield step
        except Exception:
            # The CALLER's work raised: graduate it FAILED, then re-raise so the
            # root's own error handling (skip-this-run, etc.) still runs. Only the
            # presentation is swallowed (via _safe), never the real exception.
            failed = True
            raise
        finally:
            elapsed = self._time() - start
            category = OutcomeCategory.FAILED if failed else step.category
            self._safe(lambda: self._graduate(step, category, elapsed))
            self._section_count += 1
            if category is OutcomeCategory.FAILED:
                self._section_failed = True

    @final
    @override
    def end_section(self) -> None:
        self._safe(self._stop_live)
        self._safe(self._emit_capstone)
        self._section_started = None
        self._section_count = 0
        self._section_failed = False
        self._section_errors_at = 0

    @final
    @override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.end_section()

    def _notify(self, step: BootStep) -> None:
        """Re-render after a progress() report - total, never reaches the caller."""

        self._safe(lambda: self._on_change(step))

    def _graduate(self, step: BootStep, category: OutcomeCategory, elapsed: float) -> None:
        glyph = category.glyph if self._caps.unicode else category.ascii_glyph
        text = step.label
        if step.detail:
            text += f" · {step.detail}"
        text += f" · {_format_secs(elapsed)}"
        log_styled(self._logger, indent_string(f"{glyph} {text}"), self._style(category.style))

    def _emit_capstone(self) -> None:
        # No "ready" line on an empty or failed section, or one that logged an
        # ERROR without raising (a refused arm mid-section): the error log
        # already carries the meaning, and claiming "ready" after it would lie.
        if (
            self._section_count == 0
            or self._section_started is None
            or self._section_failed
            or self._errors_logged() > self._section_errors_at
        ):
            return
        elapsed = self._time() - self._section_started
        log_styled(self._logger, indent_string(f"ready in {_format_secs(elapsed)}"), self._style("grey50"))

    def _errors_logged(self) -> int:
        """ERROR+ records the logger's LogCounter has seen (0 on a counterless logger).

        Explicit LookupError handling, not ``_safe``: the snapshot read in
        ``_step`` runs OUTSIDE ``_safe`` (a raise would escape into the wrapped
        work), and letting ``_safe`` eat a capstone-side raise would swallow the
        whole capstone for counterless loggers.
        """

        try:
            counter = log_counter(self._logger)
        except LookupError:
            return 0
        return sum(n for level, n in counter.counts.items() if level >= logging.ERROR)

    def _style(self, style: str) -> str | None:
        return style if self._caps.color else None

    def _safe(self, fn: Callable[[], object]) -> None:
        try:
            fn()
        except Exception:
            self._logger.debug("boot view presentation error", exc_info=True)

    @abstractmethod
    def _begin(self, step: BootStep) -> None:
        """Start animating the step (may be a no-op)."""

    @abstractmethod
    def _on_change(self, step: BootStep) -> None:
        """React to a progress() report (refresh the bar, or a calm heads-up)."""

    @abstractmethod
    def _stop_live(self) -> None:
        """Stop any live region (idempotent)."""


@final
class LiveBootView(_DurableBootView):
    """The animated terminal cockpit: a single ``rich.Live`` spinner per step.

    A boot step blocks, so the spinner animates on rich's refresh thread
    (``auto_refresh=True``). The Live shares the logger's Console, so graduation /
    warning / error lines reflow ABOVE the spinner (the shared Console lock
    serializes the refresh thread against the log writes). ``transient=True``
    erases the spinner on :meth:`end_section`, leaving only the durable ledger.
    """

    def __init__(self, env: TerminalEnv) -> None:
        super().__init__(env.logger, env.caps, time_source=env.time_source)
        self._console = env.console
        self._live: Live | None = None
        self._spinner: Spinner | None = None
        self._spinner_name = spinner_name(env.caps)

    @override
    def _begin(self, step: BootStep) -> None:
        if self._live is None:
            self._live = make_live(self._console)
            self._live.start()
        # A fresh Spinner per step restarts the animation at frame 0 (intended);
        # within a step we mutate .text so the dots keep their phase. Pad the
        # whole spinner left by one indent so the running glyph sits in the same
        # column as the graduated "✔" line (indent_string). Padding holds the
        # spinner by reference and re-renders it each frame, so _on_change still
        # mutates self._spinner directly.
        self._spinner = Spinner(self._spinner_name, text=self._frame_text(step), style="cyan")
        self._live.update(Padding(self._spinner, (0, 0, 0, len(INDENT))))

    @override
    def _on_change(self, step: BootStep) -> None:
        if self._spinner is not None:
            self._spinner.update(text=self._frame_text(step))

    @override
    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._spinner = None

    def _frame_text(self, step: BootStep) -> Text:
        line = Text(step.label, style="bold")
        if step.fraction is not None:
            line.append("  ")
            line.append(block_bar(step.fraction, _BAR_WIDTH, self._caps))
            line.append(f" {round(step.fraction * 100)}%", style="cyan")
            if step.detail:
                line.append("  ")
                line.append(step.detail, style="grey50")
        elif step.detail:
            line.append("  ")
            line.append(step.detail, style="grey50")
        else:
            line.append("…" if self._caps.unicode else "...", style="bold")
        return line


@final
class LogBootView(_DurableBootView):
    """Calm one-line-per-step digest for a non-TTY (Docker, a pipe, CI).

    No live region: each step graduates a single durable ledger line, plus - the
    first time a slow step reports progress - one "<label>..." heads-up, so a long
    mapping download isn't a silent pause in container logs without becoming a
    per-MB flood.
    """

    @override
    def _begin(self, step: BootStep) -> None:
        return

    @override
    def _on_change(self, step: BootStep) -> None:
        if step.announced:
            return
        step.announced = True
        ellipsis = "…" if self._caps.unicode else "..."
        self._logger.info(indent_string(f"{step.label}{ellipsis}"))

    @override
    def _stop_live(self) -> None:
        return


@final
class NullBootView(BootView):
    """A no-op view for the no-cockpit paths (tests, ``RunDeps.build`` standalone).

    Lets the startup code call ``boot.step(...)`` unconditionally - with no view it
    just runs the wrapped work, no rendering, no logging. Stateless, so a single
    shared instance is safe.
    """

    @override
    def banner(self) -> None:
        return

    @override
    def step(self, label: str) -> contextlib.AbstractContextManager[BootStep]:
        return self._noop()

    @contextlib.contextmanager
    def _noop(self) -> Generator[BootStep]:
        yield BootStep(lambda _step: None, "")

    @override
    def end_section(self) -> None:
        return

    @override
    def close(self) -> None:
        return


def make_boot_view(
    logger: logging.Logger,
    *,
    time_source: Callable[[], float] = time.monotonic,
) -> BootView:
    """Build the animated cockpit on a capable TTY, else the calm log digest.

    Reuses the logger's rich Console (so the live region and the log share one
    Console - a line logged mid-step reflows ABOVE the spinner) and the same
    capability probe the wait view uses, so the choice is consistent across both.
    ``time_source`` is injectable purely so tests can drive the step clock.
    """

    console = console_of(logger)
    caps = detect_capabilities(console)
    if console is not None and caps.live:
        return LiveBootView(TerminalEnv(console, caps, logger, time_source))
    return LogBootView(logger, caps, time_source=time_source)


def _app_version() -> str:
    """The installed package version as ``"vX.Y.Z"`` (empty if undeterminable)."""

    try:
        return f"v{version('seadexarr')}"
    except PackageNotFoundError:  # pragma: no cover - only when run from a non-install
        return ""


def _format_secs(seconds: float) -> str:
    """A compact step duration: ``"0.02s"`` / ``"1.3s"`` / ``"1m 04s"``."""

    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return format_elapsed(seconds)
