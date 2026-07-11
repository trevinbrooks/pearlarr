"""OutputHub: enqueue under one RLock; a single drainer dispatches outside it.

``emit`` appends to a bounded queue under the hub lock, and the first emitter
becomes the combiner: it drains the queue and calls renderers with the lock
RELEASED, re-checking for new entries before handing the baton back — so no
event ever waits for a future emit. Dispatch is lock-free, so it can never form
the hub-lock→console-lock (ABBA) inversion; the lifecycle calls that DO hold the
hub lock across a renderer (``begin_cycle``/``close``/``_swap_console``, where
``close`` reaches ``Live.stop``'s Console lock) stay deadlock-free by the
separate lock-ordering guarantee, not by this drain. Deliberately not a consumer
thread: this hub is the
app's UI, and synchronous main-path dispatch keeps output ordered with program
actions and crash-faithful; cross-thread events are rare stragglers the active
drain picks up. Lifecycle calls (``begin_cycle``/``set_level``/``close``) park
until no drain is in flight, so they never race a renderer's ``handle`` — and
they hold the drain baton for their body, so a re-entrant emit from inside a
renderer's lifecycle call (a bridge-adopted ``logger.debug`` in a teardown path)
enqueues instead of draining against a half-mutated subscriber list.

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
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol, final

from .events import Diagnostic, Event, Severity, severity_of
from .trace import CapturedTrace
from ..config import LogFormat

# Strikes per renderer per cycle before it is skipped until the next begin_cycle.
STRIKE_LIMIT: Final = 3

# Bound on the pending-event queue; overflow sheds the newest DIAGNOSTIC only (one
# file-only note per episode) — structural/scan events are never shed. A blocking
# enqueue would re-form the ABBA deadlock.
QUEUE_CAP: Final = 10_000


class Renderer(Protocol):
    """One output surface subscribed to the hub.

    ``handle`` receives the hub-stamped emit instant (epoch seconds) so all
    surfaces timestamp identically. It must be an exhaustive ``match`` over
    :data:`~.events.Event` ending in ``assert_never``, so a new event type fails
    type-checking in every surface instead of silently dropping on one. Trivial
    pass-through renderers (Null/Recording) are exempt by nature.
    """

    # True only for the surface that renders file_only diagnostics (the file sink);
    # the hub reads it to decide whether a containment note still has a home.
    writes_file_only: ClassVar[bool]

    def handle(self, event: Event, when: float) -> None: ...

    def begin_cycle(self) -> None: ...

    def set_level(self, level: int) -> None: ...

    def close(self) -> None: ...


@final
class NullRenderer:
    """A no-op surface (tests / headless runs)."""

    writes_file_only: ClassVar[bool] = False

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
    def errors(self) -> int:
        """ERROR and CRITICAL together — the summary's "errors" notion."""

        return self.error + self.critical

    def delta(self, earlier: SeverityTally) -> SeverityTally:
        return SeverityTally(
            debug=self.debug - earlier.debug,
            info=self.info - earlier.info,
            warning=self.warning - earlier.warning,
            error=self.error - earlier.error,
            critical=self.critical - earlier.critical,
        )


@dataclass(frozen=True, slots=True)
class CountsMark:
    """A baseline stamped on ONE counter; since() can never diff across hubs."""

    counts: SeverityCounts
    baseline: SeverityTally

    def since(self) -> SeverityTally:
        return self.counts.counts_since(self.baseline)


@final
class SeverityCounts:
    """Monotonic per-process tallies; runs take a mark() and read counts_since().

    N13 — never "reset per cycle", always deltas via marks.
    """

    def __init__(self) -> None:
        # RLock: a SIGTERM handler's emit can land while THIS thread is inside
        # record(); re-entry may lose one increment, a Lock would deadlock.
        self._lock = threading.RLock()
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

    def bound_mark(self) -> CountsMark:
        """A mark carrying THIS counter, so a later hub swap can't skew the diff."""

        return CountsMark(self, self.mark())


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
        self._level = int(Severity.INFO)
        self._closed = False
        self._pending: deque[tuple[Event, float]] = deque()
        # The popped event currently fanning out (baton holder only): the unwind
        # flush re-dispatches it, so an interrupt mid-dispatch can't lose it.
        self._in_flight: tuple[Event, float] | None = None
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

    @property
    def console_format(self) -> LogFormat | None:
        """The seated console format (None before any begin_cycle) — a lock-free
        read for the bridge; a one-record stale read at a begin_cycle boundary is
        acceptable."""

        return self._console_format

    def console_render_active(self) -> bool:
        """True when the console seat exists and is armed (not struck out).

        Deliberately lock-free (plain attribute reads): consulted from dispatch
        paths where taking the hub lock is the inversion bridge.py documents. A
        one-record stale read at a begin_cycle boundary is acceptable.
        """

        sub = self._console_sub
        return sub is not None and sub.strikes < self._strike_limit

    def dispatch_active(self) -> bool:
        """True while THIS thread is inside hub dispatch (drain or a lifecycle body).

        Deliberately lock-free: the bridge consults it under handler locks for
        the N2 file-only downgrade. The identity read is exact for the calling
        thread — only the thread itself can have set the baton to itself.
        """

        return self._drainer is threading.current_thread()

    def emit(self, event: Event) -> None:
        """Enqueue under the hub lock; the combiner dispatches outside it.

        Under the lock: closed/once-key gating, the severity tally (at enqueue,
        so counts survive a dying renderer; file_only diagnostics are never
        counted), one clock read, one queue append.
        If no drain is active the caller enters the combiner, which takes the
        baton interrupt-safely and drains synchronously before returning;
        otherwise the active drainer picks the event up (its loop re-checks the
        queue before releasing the baton) — this covers both cross-thread emits
        and re-entrant emits from inside a renderer's handle.
        Overflow past the cap sheds only diagnostics — the unbounded bridge-fed
        class; structural/scan events are appended even past the cap (fold inputs
        must never be lost), and never blocks (bounded blocking would re-form the
        ABBA deadlock through backpressure). Total
        against renderer bugs: a raise strikes the renderer and surfaces as a
        file-only diagnostic to the survivors; emit itself never raises
        (KeyboardInterrupt/SystemExit still propagate). Emits on a closed hub
        drop silently.
        """

        with self._lock:
            if self._closed:
                return
            key: tuple[str, str] | None = None
            if isinstance(event, Diagnostic) and event.once_key is not None:
                # Deduped events are dropped whole: not enqueued, not counted.
                key = (event.origin, event.once_key)
                if key in self._once_keys:
                    return
            when = self._clock()
            if isinstance(event, Diagnostic) and len(self._pending) >= QUEUE_CAP:
                # Shed ONLY diagnostics — the unbounded bridge-fed class. Structural/
                # scan events are O(library) and can't themselves overflow the cap, so
                # its memory bound survives while fold inputs are never lost. A
                # keyless diagnostic sheds rendering only (its tally stands); a
                # once-keyed one is shed WHOLE — key unregistered, uncounted, dedup
                # parity with the duplicate arm — so a re-emit after the queue
                # drains still lands instead of dying on a consumed key.
                if key is None and not event.file_only:
                    self._counts.record(severity_of(event))
                if not self._overflowing:
                    self._overflowing = True
                    self._pending.append((self._overflow_note(), when))
            else:
                if key is not None:
                    self._once_keys.add(key)
                # Counts = what a visible surface could show: file_only forensics
                # never inflate the issues row or suppress the capstone.
                if not (isinstance(event, Diagnostic) and event.file_only):
                    self._counts.record(severity_of(event))
                self._pending.append((event, when))
            if self._drainer is not None:
                return
        self._drain()

    def _drain(self) -> None:
        """The combiner entry: take the baton, run the loop, flush on unwind.

        The baton is taken INSIDE the try (emit only checks-and-calls), so an
        interrupt can never strand it between assignment and the clearing
        finally. ``took`` marks THIS frame's take: a loser of the take race —
        including a nested same-thread call from a mid-dispatch lifecycle call
        (set_level/begin_cycle/close) — returns without touching the outer
        frame's baton or flushing re-entrantly.
        """

        took = False
        try:
            with self._lock:
                if self._drainer is not None:
                    return
                took = True
                self._drainer = threading.current_thread()
            self._drain_loop()
        finally:
            # A propagating KeyboardInterrupt/SystemExit must not strand the baton —
            # or the queued tail (a SIGTERM handler's exit marker lands mid-unwind):
            # flush best-effort; the flush releases the baton itself.
            if took:
                with self._lock:
                    stranded = self._drainer is threading.current_thread()
                if stranded:
                    self._flush_on_unwind()

    def _drain_loop(self) -> None:
        """The combiner loop: pop + snapshot under the lock, dispatch outside it.

        The popped event rides ``_in_flight`` until the next locked section, so
        an interrupt mid-dispatch can't lose it — the unwind flush re-dispatches
        it (duplicates over loss, crash fidelity) — while the queue's occupancy
        (the overflow cap) never counts it twice. The terminal arm clears
        queue/overflow state and releases the baton in ONE locked section, so a
        cross-thread emit can never park behind a baton about to be cleared; the
        empty re-check before that release is the combiner guarantee — no event
        ever waits for a future emit.
        """

        while True:
            with self._lock:
                self._in_flight = None
                if self._closed or not self._pending:
                    self._pending.clear()
                    self._overflowing = False
                    self._drainer = None
                    self._idle.notify_all()
                    return
                event, when = self._pending.popleft()
                self._in_flight = (event, when)
                armed = [sub for sub in self._subs if sub.strikes < self._strike_limit]
            self._dispatch(event, when, armed)

    def _flush_on_unwind(self) -> None:
        """Re-dispatch the in-flight event, then flush the queued tail, while an
        exception unwinds ``_drain`` (crash fidelity: the tail must be on
        screen/in the file when the process dies). The in-flight re-dispatch is
        best-effort per renderer — even a repeat interrupt there is contained,
        so one hostile arm can't cost the whole tail (the tail flush itself
        stays interruptible: a second Ctrl-C still kills it). The finally
        releases the baton no matter what — a wedged baton would silence the
        hub for the process's remaining lifetime (the PR5 D-1 hazard)."""

        try:
            with self._lock:
                in_flight, self._in_flight = self._in_flight, None
                armed = [sub for sub in self._subs if sub.strikes < self._strike_limit]
            if in_flight is not None:
                event, when = in_flight
                for sub in armed:
                    try:
                        sub.renderer.handle(event, when)
                    except Exception:
                        self._strike(sub)
                    except BaseException:  # noqa: S112 - already unwinding; best-effort
                        continue
            self._drain_loop()
        finally:
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

    @contextlib.contextmanager
    def _baton_held(self) -> Generator[None]:
        """Hold the drain baton for a lifecycle body (lock held, no other drainer).

        A renderer's lifecycle call can log; the bridge adopts that into a
        re-entrant emit, which must enqueue — never start a nested drain against
        a half-mutated subscriber list. Callers drain after releasing the lock.
        """

        owned = self._drainer is threading.current_thread()
        if not owned:
            self._drainer = threading.current_thread()
        try:
            yield
        finally:
            if not owned:
                self._drainer = None
                self._idle.notify_all()

    def begin_cycle(self, *, console_format: LogFormat, level: int) -> None:
        """Per-cycle turnover: re-arm strikes, clear once-keys, swap the console
        renderer when the re-peeked format changed, rotate/reset sinks."""

        with self._lock:
            self._await_no_drainer()
            if self._closed:
                return
            with self._baton_held():
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
                self._set_level_locked(level)
        self._drain()

    def set_level(self, level: int) -> None:
        """Forward the configured level; each surface applies its own floor semantics (S4)."""

        with self._lock:
            self._await_no_drainer()
            with self._baton_held():
                self._set_level_locked(level)
        self._drain()

    def _set_level_locked(self, level: int) -> None:
        self._level = level
        for sub in self._subs:
            try:
                sub.renderer.set_level(level)
            except Exception:
                self._strike(sub)

    def close(self) -> None:
        """Idempotent teardown of every surface (file close, Live stop in PR2+).

        The file sinks close LAST, after the teardown chatter the other closes
        enqueued (a bridge-adopted Live.stop failure, a region's contained-
        teardown note) is handed to them — a close-path failure still leaves a
        file trace. Emits after the file sinks close drop at the ``_closed``
        gate: with no file surface left they are unrecordable on every route.
        """

        with self._lock:
            self._await_no_drainer()
            if self._closed:
                return
            with self._baton_held():
                file_subs = [sub for sub in self._subs if sub.renderer.writes_file_only]
                for sub in self._subs:
                    if sub in file_subs:
                        continue
                    with contextlib.suppress(Exception):
                        sub.renderer.close()
                self._closed = True
                # Dispatching under the lock is safe here: file sinks never touch
                # the Console lock (begin_cycle's rotation is the precedent).
                while self._pending:
                    event, when = self._pending.popleft()
                    for sub in file_subs:
                        if sub.strikes < self._strike_limit:
                            with contextlib.suppress(Exception):
                                sub.renderer.handle(event, when)
                for sub in file_subs:
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
        self._stderr_fallback(event)
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
        # No set_level here: begin_cycle (the only caller) applies the cycle's
        # level to every sub — the fresh console included — right after the swap,
        # and nothing dispatches in between (re-entrant emits enqueue under the
        # baton). A second call would just burn a strike twice on one bug.

    def _note(self, message: str, exc: Exception, when: float, *, skip: _Sub | None) -> None:
        """A containment note to the armed survivors.

        Forensic (file_only, uncounted — a presentation bug must not inflate the
        user-visible warnings tally) while a file_only surface survives to write
        it. When the casualty IS the file sink, the note escalates: visible,
        ERROR, counted — a run must never silently lose its whole log file.
        """

        with self._lock:
            armed = [sub for sub in self._subs if sub is not skip and sub.strikes < self._strike_limit]
        file_only = any(sub.renderer.writes_file_only for sub in armed)
        note = Diagnostic(
            severity=Severity.WARNING if file_only else Severity.ERROR,
            message=message,
            origin="output.hub",
            trace=CapturedTrace.from_exception(exc),
            file_only=file_only,
        )
        if not file_only:
            # P5: counts = what a visible surface could show; this one is visible.
            self._counts.record(note.severity)
        for sub in armed:
            try:
                sub.renderer.handle(note, when)
            except Exception:
                # A failure while reporting a failure just strikes — no recursion.
                self._strike(sub)
        self._stderr_fallback(note)

    def _stderr_fallback(self, event: Event) -> None:
        """No armed console seat (pre-begin_cycle, or quarantined): visible
        WARNING+ diagnostics still reach a human via stderr. Factory-less hubs
        (the renderer-less default, recording hubs) keep dropping silently —
        library and test use must stay quiet."""

        if self._console_factory is None or self.console_render_active():
            return
        if isinstance(event, Diagnostic) and not event.file_only and event.severity >= Severity.WARNING:
            # Contained like every renderer write: a broken/closed stderr must not
            # break the emit-never-raises contract (a last-resort surface has no
            # fallback of its own — the failure just drops).
            with contextlib.suppress(Exception):
                sys.stderr.write(f"{event.severity.name} [{event.origin}] {event.message}\n")
                sys.stderr.flush()
