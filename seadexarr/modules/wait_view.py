"""Presentation for the wait-for-completion + manual-import blocking pass.

The engine drives a :class:`WaitView` while it waits on each grabbed torrent to
download and then import. On an attached terminal the view is a live region of
per-torrent progress bars - a download percentage plus an elapsed/timeout
countdown; on a non-TTY (Docker, a pipe) it degrades to concise, throttled
heartbeat log lines so container logs stay clean. :func:`make_wait_view` picks
the right one once, so the engine drives a single small interface either way and
stays free of any rich/presentation detail. The plain-text file log is untouched
(a live region only ever touches the console handler's Console).
"""

import logging
import time
from typing import Protocol

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskID, TextColumn

from .log import LogFormatter, RichConsoleHandler, indent_string


class WaitView(Protocol):
    """The small interface the engine drives while waiting on downloads/imports.

    Every method is keyed by a stable ``key`` (the torrent infohash); the display
    ``label`` is registered once in :meth:`start`. Implementations must tolerate an
    unknown key (a no-op) so the engine never has to guard its calls.
    """

    def start(self, torrents: list[tuple[str, str]]) -> None:
        """Register the torrents to wait on, as ``(key, label)`` pairs."""

    def download(self, key: str, pct: float, elapsed: float, timeout: float) -> None:
        """Update one torrent's download progress (``pct`` is 0.0-1.0)."""

    def phase_sonarr(self, key: str, elapsed: float, timeout: float) -> None:
        """Mark a torrent downloaded and now waiting on Sonarr to import."""

    def done(self, key: str, outcome: str) -> None:
        """Mark a torrent terminal with a short outcome word."""

    def close(self) -> None:
        """Tear the view down (restore the terminal / stop refreshing)."""


def make_wait_view(logger: logging.Logger, *, poll_s: int) -> WaitView:
    """Build a live view on a TTY, else a heartbeat view (Docker / pipe / no console).

    Args:
        logger (logging.Logger): The app logger; its rich console handler is reused
            so the live region and the log lines share one Console.
        poll_s (int): The poll cadence, used to throttle the non-TTY heartbeat.
    """

    console = _console_of(logger)
    if console is not None and console.is_terminal:
        return _LiveWaitView(console)
    return _HeartbeatWaitView(logger, poll_s=poll_s)


def _console_of(logger: logging.Logger) -> Console | None:
    """The rich Console behind the logger's console handler, if any."""

    for handler in logger.handlers:
        if isinstance(handler, RichConsoleHandler):
            return handler.console
    return None


def _countdown(elapsed: float, timeout: float) -> str:
    """An "12m 03s / 1h 00m 00s" elapsed-vs-timeout label in house style."""

    return (
        f"{LogFormatter.format_elapsed(elapsed)} / "
        f"{LogFormatter.format_elapsed(timeout)}"
    )


class _LiveWaitView:
    """A ``rich.Live`` region of per-torrent progress bars (TTY only).

    Single-threaded by contract: the engine's poll loop calls these methods and we
    refresh explicitly, so ``auto_refresh`` is off and no background thread can
    contend with the logging handler that shares this Console. A warning/error
    logged mid-wait renders ABOVE the live region automatically (rich's render
    hook), so nothing is torn or lost. INVARIANT: keep the wait loop
    single-threaded; a threaded/async poll loop would need a different
    Console-sharing story.
    """

    def __init__(self, console: Console) -> None:
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=24),
            TextColumn("{task.fields[note]}"),
            console=console,
            auto_refresh=False,
        )
        self._live = Live(
            self._progress,
            console=console,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._tasks: dict[str, TaskID] = {}

    def start(self, torrents: list[tuple[str, str]]) -> None:
        self._live.start()
        for key, label in torrents:
            self._tasks[key] = self._progress.add_task(
                label, total=100.0, completed=0, note="queued",
            )
        self._live.refresh()

    def download(self, key: str, pct: float, elapsed: float, timeout: float) -> None:
        task = self._tasks.get(key)
        if task is None:
            return
        self._progress.update(
            task,
            completed=max(0.0, min(100.0, pct * 100.0)),
            note=_countdown(elapsed, timeout),
        )
        self._live.refresh()

    def phase_sonarr(self, key: str, elapsed: float, timeout: float) -> None:
        task = self._tasks.get(key)
        if task is None:
            return
        self._progress.update(
            task, completed=100.0, note=f"importing  {_countdown(elapsed, timeout)}",
        )
        self._live.refresh()

    def done(self, key: str, outcome: str) -> None:
        task = self._tasks.get(key)
        if task is None:
            return
        self._progress.update(task, completed=100.0, note=outcome)
        self._live.refresh()

    def close(self) -> None:
        self._live.stop()


class _HeartbeatWaitView:
    """Concise, throttled heartbeat log lines for non-TTY output (Docker / pipe).

    Each terminal ``done`` always logs; the per-poll ``download``/``phase_sonarr``
    lines are throttled to at most one per torrent per ``max(poll_s, 30)`` seconds,
    so a short poll interval can't carpet container logs.
    """

    def __init__(self, logger: logging.Logger, *, poll_s: int) -> None:
        self._logger = logger
        self._labels: dict[str, str] = {}
        self._min_gap = max(float(poll_s), 30.0)
        self._last: dict[str, float] = {}

    def start(self, torrents: list[tuple[str, str]]) -> None:
        self._labels = dict(torrents)
        self._logger.info(
            f"Waiting for {len(torrents)} download(s) to complete and import...",
        )

    def _should_emit(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last.get(key)
        if last is not None and now - last < self._min_gap:
            return False
        self._last[key] = now
        return True

    def download(self, key: str, pct: float, elapsed: float, timeout: float) -> None:
        if not self._should_emit(key):
            return
        label = self._labels.get(key, key)
        self._logger.info(
            indent_string(
                f"{label}: downloading {pct * 100:.0f}%  ({_countdown(elapsed, timeout)})",
            ),
        )

    def phase_sonarr(self, key: str, elapsed: float, timeout: float) -> None:
        if not self._should_emit(key):
            return
        label = self._labels.get(key, key)
        self._logger.info(
            indent_string(f"{label}: importing  ({_countdown(elapsed, timeout)})"),
        )

    def done(self, key: str, outcome: str) -> None:
        label = self._labels.get(key, key)
        self._logger.info(indent_string(f"{label}: {outcome}"))
        self._last.pop(key, None)

    def close(self) -> None:
        pass
