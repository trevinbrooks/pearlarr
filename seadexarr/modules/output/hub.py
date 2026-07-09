"""OutputHub: enqueue under one RLock; a single drainer dispatches outside it.

``emit`` appends to a bounded queue under the hub lock, and the first emitter
becomes the combiner: it drains the queue and calls renderers with the lock
RELEASED, re-checking for new entries before handing the baton back — so no
event ever waits for a future emit, and the hub-lock x console-lock (ABBA)
deadlock can never form. Deliberately not a consumer thread: this hub is the
app's UI, and synchronous main-path dispatch keeps output ordered with program
actions and crash-faithful; cross-thread events are rare stragglers the active
drain picks up. Lifecycle calls (``begin_cycle``/``set_level``/``close``) park
until no drain is in flight, so they never race a renderer's ``handle``.

Renderers that raise strike out (3 per cycle, S9) and are skipped until
``begin_cycle`` re-arms them — never a process-latching quarantine (the verified
daemon hazard). Striking out also closes the renderer, so a quarantined seat
releases its resources (a live spinner must stop repainting). ``emit`` itself never raises on a renderer bug, so presentation
can never abort a run or the cache save; KeyboardInterrupt/SystemExit still
propagate (Ctrl-C must unwind). The hub reads the clock once per emit and hands
every renderer the same instant, so cross-sink timestamps can never disagree.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, final

from .events import Diagnostic, Event, Severity, severity_of
from .trace import CapturedTrace
from ..config import LogFormat

# Strikes per renderer per cycle before it is skipped until the next begin_cycle.
STRIKE_LIMIT: Final = 3

# Bound on the pending-event queue; overflow sheds the NEWEST event (one file-only
# note per episode) — a blocking enqueue would re-form the ABBA deadlock.
QUEUE_CAP: Final = 10_000


class Renderer(Protocol):
    """One output surface subscribed to the hub.

    ``handle`` receives the hub-stamped emit instant (epoch seconds) so all
    surfaces timestamp identically. It must be an exhaustive ``match`` over
    :data:`~.events.Event` ending in ``assert_never``, so a new event type fails
    type-checking in every surface instead of silently dropping on one. Trivial
    pass-through renderers (Null/Recording) are exempt by nature.
    """

    def handle(self, event: Event, when: float) -> None: ...

    def begin_cycle(self) -> None: ...

    def set_level(self, level: int) -> None: ...

    def close(self) -> None: ...


@final
class NullRenderer:
    """A no-op surface (tests / headless runs)."""

    def handle(self, event: Event, when: float) -> None:
        pass

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class SeverityTally:
    """A frozen per-severity count snapshot; marks and deltas are both this type."""

    debug: int = 0
    info: int = 0
    warning: int = 0
    error: int = 0
    critical: int = 0

    @property
    def warnings(self) -> int:
        return self.warning

    @property
    def errors(self) -> int:
        return self.error + self.critical

    def delta(self, earlier: SeverityTally) -> SeverityTally:
        return SeverityTally(
            debug=self.debug - earlier.debug,
            info=self.info - earlier.info,
            warning=self.warning - earlier.warning,
            error=self.error - earlier.error,
            critical=self.critical - earlier.critical,
        )


@final
class SeverityCounts:
    """Monotonic per-process tallies; runs take a mark() and read counts_since().

    Replaces LogCounter: N13 — never "reset per cycle", always deltas via marks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[Severity, int] = dict.fromkeys(Severity, 0)

    def record(self, severity: Severity) -> None:
        with self._lock:
            self._counts[severity] += 1

    def mark(self) -> SeverityTally:
        """The current totals — hold it and diff later via counts_since()."""

        with self._lock:
            return SeverityTally(
                debug=self._counts[Severity.DEBUG],
                info=self._counts[Severity.INFO],
                warning=self._counts[Severity.WARNING],
                error=self._counts[Severity.ERROR],
                critical=self._counts[Severity.CRITICAL],
            )

    def counts_since(self, mark: SeverityTally) -> SeverityTally:
        return self.mark().delta(mark)


class _Sub:
    """One subscription: a renderer plus its per-cycle strike count."""

    __slots__ = ("renderer", "strikes")

    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer
        self.strikes = 0


@final
class OutputHub:
    """The one dispatch point: every surface sees every event, in emit order.

    ``renderers`` are the stable sinks (file/json/...) and never contain a
    console; the console renderer lives in its own tracked seat (``console`` at
    construction, thereafter swapped by ``begin_cycle`` via ``console_factory``).
    """

    def __init__(
        self,
        renderers: Sequence[Renderer],
        *,
        console: Renderer | None = None,
        console_factory: Callable[[LogFormat], Renderer] | None = None,
        clock: Callable[[], float] = time.time,
        strike_limit: int = STRIKE_LIMIT,
    ) -> None:
        self._lock = threading.RLock()
        self._subs = [_Sub(renderer) for renderer in renderers]
        self._console_sub: _Sub | None = None
        if console is not None:
            self._console_sub = _Sub(console)
            self._subs.append(self._console_sub)
        self._console_factory = console_factory
        self._console_format: LogFormat | None = None
        self._clock = clock
        self._strike_limit = strike_limit
        self._once_keys: set[tuple[str, str]] = set()
        self._counts = SeverityCounts()
        self._cycle_mark = SeverityTally()
        self._level = int(Severity.INFO)
        self._closed = False
        self._pending: deque[tuple[Event, float]] = deque()
        self._drainer: threading.Thread | None = None
        self._idle = threading.Condition(self._lock)
        self._overflowing = False

    @property
    def counts(self) -> SeverityCounts:
        return self._counts

    @property
    def level(self) -> int:
        """The configured level — a lock-free int read for the bridge's early-out."""

        return self._level

    def console_render_active(self) -> bool:
        """True when the console seat exists and is armed (not struck out).

        Deliberately lock-free (plain attribute reads): the rich handler calls this
        under handler locks, and hub dispatch re-enters the logger — taking the hub
        lock here is the inversion bridge.py documents. A one-record stale read at
        a begin_cycle boundary is acceptable.
        """

        sub = self._console_sub
        return sub is not None and sub.strikes < self._strike_limit

    def cycle_counts(self) -> SeverityTally:
        """Severity deltas since the last begin_cycle."""

        return self._counts.counts_since(self._cycle_mark)

    def record_severity(self, severity: Severity) -> None:
        """Count-only bump, no event: first-party WARNING+ payload records render
        fully legacy, but their severity must still reach the capstone/summary tallies."""

        if self._closed:
            return
        self._counts.record(severity)

    def emit(self, event: Event) -> None:
        """Enqueue under the hub lock; the combiner dispatches outside it.

        Under the lock: closed/once-key gating, the severity tally (at enqueue,
        so counts survive a dying renderer), one clock read, one queue append.
        If no drain is active the caller takes the baton and drains synchronously
        before returning; otherwise the active drainer picks the event up (its
        loop re-checks the queue before releasing the baton) — this covers both
        cross-thread emits and re-entrant emits from inside a renderer's handle.
        Overflow past the cap sheds the newest event, never blocks (bounded
        blocking would re-form the ABBA deadlock through backpressure). Total
        against renderer bugs: a raise strikes the renderer and surfaces as a
        file-only diagnostic to the survivors; emit itself never raises
        (KeyboardInterrupt/SystemExit still propagate). Emits on a closed hub
        drop silently.
        """

        with self._lock:
            if self._closed:
                return
            if isinstance(event, Diagnostic) and event.once_key is not None:
                # Deduped events are dropped whole: not enqueued, not counted.
                key = (event.origin, event.once_key)
                if key in self._once_keys:
                    return
                self._once_keys.add(key)
            self._counts.record(severity_of(event))
            when = self._clock()
            if len(self._pending) >= QUEUE_CAP:
                # Shed rendering only (the tally above stands); one note per episode.
                if not self._overflowing:
                    self._overflowing = True
                    self._pending.append((self._overflow_note(), when))
            else:
                self._pending.append((event, when))
            if self._drainer is not None:
                return
            self._drainer = threading.current_thread()
        self._drain()

    def _drain(self) -> None:
        """The combiner loop: pop + snapshot under the lock, dispatch outside it.

        The empty re-check before releasing the baton is the combiner guarantee:
        no event ever waits for a future emit.
        """

        try:
            while True:
                with self._lock:
                    if self._closed or not self._pending:
                        self._pending.clear()
                        self._overflowing = False
                        self._drainer = None
                        self._idle.notify_all()
                        break
                    event, when = self._pending.popleft()
                    armed = [sub for sub in self._subs if sub.strikes < self._strike_limit]
                self._dispatch(event, when, armed)
        finally:
            # A propagating KeyboardInterrupt/SystemExit must not strand the baton.
            with self._lock:
                if self._drainer is threading.current_thread():
                    self._drainer = None
                    self._idle.notify_all()

    def _await_no_drainer(self) -> None:
        """Park (lock held) until no OTHER thread holds the drain baton — the
        current-thread escape prevents self-deadlock on a mid-drain lifecycle call."""

        while self._drainer is not None and self._drainer is not threading.current_thread():
            self._idle.wait()

    def _overflow_note(self) -> Diagnostic:
        return Diagnostic(
            severity=Severity.WARNING,
            message=f"output queue overflowed ({QUEUE_CAP} pending); shedding newest events until it drains",
            origin="output.hub",
            file_only=True,
        )

    def begin_cycle(self, *, console_format: LogFormat, level: int) -> None:
        """Per-cycle turnover: re-arm strikes, clear once-keys, swap the console
        renderer when the re-peeked format changed, rotate/reset sinks, mark counts."""

        with self._lock:
            self._await_no_drainer()
            if self._closed:
                return
            self._once_keys.clear()
            for sub in self._subs:
                sub.strikes = 0
            if self._console_factory is not None and console_format != self._console_format:
                self._swap_console(console_format)
            for sub in self._subs:
                try:
                    sub.renderer.begin_cycle()
                except Exception:
                    self._strike(sub)
            self.set_level(level)
            self._cycle_mark = self._counts.mark()

    def set_level(self, level: int) -> None:
        """Forward the configured level; each surface applies its own floor semantics (S4)."""

        with self._lock:
            self._await_no_drainer()
            self._level = level
            for sub in self._subs:
                try:
                    sub.renderer.set_level(level)
                except Exception:
                    self._strike(sub)

    def close(self) -> None:
        """Idempotent teardown of every surface (file close, Live stop in PR2+)."""

        with self._lock:
            self._await_no_drainer()
            if self._closed:
                return
            self._closed = True
            for sub in self._subs:
                with contextlib.suppress(Exception):
                    sub.renderer.close()

    def _strike(self, sub: _Sub) -> None:
        """One strike (under a brief lock re-acquire); crossing the limit also closes
        the renderer — a quarantined seat must release its resources (a struck boot
        Live keeps repainting otherwise)."""

        with self._lock:
            sub.strikes += 1
            crossed = sub.strikes == self._strike_limit
        if crossed:
            with contextlib.suppress(Exception):
                sub.renderer.close()

    def _dispatch(self, event: Event, when: float, subs: Sequence[_Sub]) -> None:
        """Fan one event out to the snapshotted subs — no hub lock held here."""

        failures: list[tuple[_Sub, Exception]] = []
        for sub in subs:
            try:
                sub.renderer.handle(event, when)
            except Exception as exc:
                self._strike(sub)
                failures.append((sub, exc))
        # Notes go out after the event, so survivors keep the emit order.
        for sub, exc in failures:
            quarantined = " (quarantined until next cycle)" if sub.strikes >= self._strike_limit else ""
            message = f"renderer {type(sub.renderer).__name__} failed on {type(event).__name__}: {exc}{quarantined}"
            self._note(message, exc, when, skip=sub)

    def _swap_console(self, console_format: LogFormat) -> None:
        """Build the fresh console FIRST; only on success tear down the old seat."""

        factory = self._console_factory
        if factory is None:  # pragma: no cover - guarded by the caller
            return
        try:
            fresh = factory(console_format)
        except Exception as exc:
            message = f"console factory failed for format {console_format!r}: {exc}; keeping the current console"
            self._note(message, exc, self._clock(), skip=None)
            return
        if self._console_sub is not None:
            with contextlib.suppress(Exception):
                self._console_sub.renderer.close()
            self._subs.remove(self._console_sub)
        sub = _Sub(fresh)
        self._subs.append(sub)
        self._console_sub = sub
        self._console_format = console_format
        # The stored level is load-bearing: the fresh console starts at it.
        try:
            fresh.set_level(self._level)
        except Exception:
            self._strike(sub)

    def _note(self, message: str, exc: Exception, when: float, *, skip: _Sub | None) -> None:
        """A file-only containment note to the armed survivors — never counted
        (a presentation bug must not inflate the user-visible warnings tally)."""

        note = Diagnostic(
            severity=Severity.WARNING,
            message=message,
            origin="output.hub",
            trace=CapturedTrace.from_exception(exc),
            file_only=True,
        )
        with self._lock:
            armed = [sub for sub in self._subs if sub is not skip and sub.strikes < self._strike_limit]
        for sub in armed:
            try:
                sub.renderer.handle(note, when)
            except Exception:
                # A failure while reporting a failure just strikes — no recursion.
                self._strike(sub)
