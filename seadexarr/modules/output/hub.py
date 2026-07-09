"""OutputHub: synchronous fan-out under one RLock, with per-event containment.

Renderers that raise strike out (3 per cycle, S9) and are skipped until
``begin_cycle`` re-arms them — never a process-latching quarantine (the verified
daemon hazard). ``emit`` itself never raises on a renderer bug, so presentation
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

# Bound on events queued by re-entrant emits (a renderer emitting mid-dispatch);
# overflow drops silently — re-entrant floods are a bug, not a data path.
REENTRANT_QUEUE_CAP: Final = 100


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
        self._dispatch_depth = 0
        self._queue: deque[tuple[Event, float]] = deque()

    @property
    def counts(self) -> SeverityCounts:
        return self._counts

    def cycle_counts(self) -> SeverityTally:
        """Severity deltas since the last begin_cycle."""

        return self._counts.counts_since(self._cycle_mark)

    def emit(self, event: Event) -> None:
        """Dispatch to every armed renderer, in order, under the hub lock.

        Total against renderer bugs: a raise strikes the renderer and surfaces as
        a file-only diagnostic to the survivors; emit itself never raises. A
        re-entrant emit (a renderer emitting mid-dispatch) is queued and drained
        by the outermost dispatch — bounded, order-preserving (pre-implements the
        S5 bridge contract). Emits on a closed hub drop silently.
        """

        with self._lock:
            if self._closed:
                return
            when = self._clock()
            if self._dispatch_depth > 0:
                if len(self._queue) < REENTRANT_QUEUE_CAP:
                    self._queue.append((event, when))
                return
            self._dispatch_depth += 1
            try:
                self._dispatch(event, when)
                while self._queue:
                    queued, queued_when = self._queue.popleft()
                    self._dispatch(queued, queued_when)
            finally:
                self._dispatch_depth -= 1

    def begin_cycle(self, *, console_format: LogFormat, level: int) -> None:
        """Per-cycle turnover: re-arm strikes, clear once-keys, swap the console
        renderer when the re-peeked format changed, rotate/reset sinks, mark counts."""

        with self._lock:
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
                    sub.strikes += 1
            self.set_level(level)
            self._cycle_mark = self._counts.mark()

    def set_level(self, level: int) -> None:
        """Forward the configured level; each surface applies its own floor semantics (S4)."""

        with self._lock:
            self._level = level
            for sub in self._subs:
                try:
                    sub.renderer.set_level(level)
                except Exception:
                    sub.strikes += 1

    def close(self) -> None:
        """Idempotent teardown of every surface (file close, Live stop in PR2+)."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            for sub in self._subs:
                with contextlib.suppress(Exception):
                    sub.renderer.close()

    def _dispatch(self, event: Event, when: float) -> None:
        if isinstance(event, Diagnostic) and event.once_key is not None:
            # Deduped events are dropped whole: not dispatched, not counted.
            key = (event.origin, event.once_key)
            if key in self._once_keys:
                return
            self._once_keys.add(key)
        self._counts.record(severity_of(event))
        failures: list[tuple[_Sub, Exception]] = []
        for sub in self._subs:
            if sub.strikes >= self._strike_limit:
                continue
            try:
                sub.renderer.handle(event, when)
            except Exception as exc:
                sub.strikes += 1
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
            sub.strikes += 1

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
        for sub in self._subs:
            if sub is skip or sub.strikes >= self._strike_limit:
                continue
            try:
                sub.renderer.handle(note, when)
            except Exception:
                # A failure while reporting a failure just strikes — no recursion.
                sub.strikes += 1
