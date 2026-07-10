"""Typed capability scope handles: position rides the handle, never the call site.

The handle set is exactly Step / Entry / Wait plus the position-free
:class:`Diagnostics` (S2: no SummaryScope — the summary is one atomic event;
Run/Item boundaries are plain events with no handle ceremony). Handles are
runtime-total: emitting on a closed handle demotes to an attributed
:class:`~.events.Diagnostic` (``placed_by=HANDLE``) instead of raising or
corrupting layout; tests pin that no production path ever demotes.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from types import TracebackType
from typing import ClassVar, Final, assert_never, final

from .events import (
    Accent,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFailed,
    LedgerRow,
    PlacedBy,
    ReleaseSkipped,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    StyledValue,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    severity_of,
)
from .runtime import emit_to_hub
from .trace import CapturedTrace
from ..manual_import import OutcomeCategory

type Emit = Callable[[Event], None]
"""The one producer-side seam: the hub satisfies it; tests pass a recorder."""

type EntryFact = EntryDetail | LedgerRow | ReleaseSkipped | GrabFailed | GrabAction
"""The entry-block facts an EntryScope can post (stamped with its ScopeId)."""


@final
class ScopeIds:
    """Thread-safe ScopeId minter (serials are monotonic per minter; factories
    default to the process-wide minter so serials never collide across factories)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._serials = itertools.count(1)

    def mint(self, kind: ScopeKind) -> ScopeId:
        with self._lock:
            return ScopeId(kind, next(self._serials))


# The process-wide default minter (see ScopeIds docstring).
PROCESS_SCOPE_IDS: Final = ScopeIds()


@final
class ScopeMark:
    """The cockpit views' ambient-scope mark ceremony: idempotent open/close.

    Mints from :data:`PROCESS_SCOPE_IDS` and emits through :func:`~.runtime.emit_to_hub`
    at call time (the hub may be installed after the view is built). Only the mark
    pair — no handle semantics, no demotion.
    """

    def __init__(self, kind: ScopeKind, label: str) -> None:
        self._kind = kind
        self._label = label
        self._scope: ScopeId | None = None

    def open(self) -> None:
        if self._scope is not None:
            return
        self._scope = PROCESS_SCOPE_IDS.mint(self._kind)
        emit_to_hub(ScopeOpened(scope=self._scope, label=self._label))

    def close(self) -> None:
        if self._scope is None:
            return
        emit_to_hub(ScopeClosed(scope=self._scope))
        self._scope = None


def _describe_fact(fact: EntryFact) -> str:
    """A compact one-line description of a demoted fact (for the late diagnostic)."""

    match fact:
        case EntryDetail(label=label, value=value):
            return f"{label}: {value.text}"
        case LedgerRow(state=state, label=label):
            return f"{state} {label}"
        case ReleaseSkipped(group=group, tracker=tracker, reason=reason):
            return f"release skipped: {group} on {tracker} ({reason.name.lower()})"
        case GrabFailed(group=group, error=error):
            return f"grab failed: {group} ({error})"
        case GrabAction(status=status):
            return f"grab action: {status.name.lower()}"
    assert_never(fact)


class _ScopeBase:
    """Shared handle spine: emitter, label, open flag, and late-demotion.

    ``_late`` binds the emitter/kind word/label once, so call sites state only
    WHAT was attempted and at which severity — runtime-total.
    """

    _KIND_WORD: ClassVar[str] = "scope"

    def __init__(self, emit: Emit, label: str) -> None:
        self._emit = emit
        self._label = label
        self._open = True

    def _late(self, what: str, severity: Severity) -> None:
        kind = type(self)._KIND_WORD
        self._emit(
            Diagnostic(
                severity=max(severity, Severity.INFO),
                message=f"{what} [after {kind} '{self._label}' closed]",
                origin=f"output.late.{kind}",
                placed_by=PlacedBy.HANDLE,
            ),
        )


@final
class StepScope(_ScopeBase):
    """One boot step: progress/note/warn producer-side, timing here, events out.

    Usable as a context manager: the step finishes on exit (FAILED when the body
    raised; the exception still propagates — only presentation is owned here).
    """

    _KIND_WORD: ClassVar[str] = "step"

    def __init__(self, emit: Emit, scope: ScopeId, label: str, clock: Callable[[], float]) -> None:
        super().__init__(emit, label)
        self._scope = scope
        self._clock = clock
        self._started = clock()
        self._category = OutcomeCategory.SUCCESS
        self._detail: str | None = None
        self._slow_sent = False
        emit(BootStepStarted(scope=scope, label=label))

    @property
    def scope_id(self) -> ScopeId:
        return self._scope

    def progress(self, fraction: float, detail: str | None = None) -> None:
        """Report 0-1 progress; the first report also emits the one-time slow heads-up."""

        if not self._open:
            self._late(f"progress {fraction:.2f}", Severity.INFO)
            return
        if not self._slow_sent:
            self._slow_sent = True
            self._emit(BootStepSlow(scope=self._scope, label=self._label))
        if detail is not None:
            self._detail = detail
        clamped = max(0.0, min(1.0, fraction))
        self._emit(BootStepProgressed(scope=self._scope, fraction=clamped, detail=self._detail))

    def note(self, text: str) -> None:
        """Set the detail the finished ledger line carries (e.g. "42 series")."""

        if not self._open:
            self._late(f"note: {text}", Severity.INFO)
            return
        self._detail = text

    def warn(self, text: str | None = None) -> None:
        """Finish this step as a warning (DEFERRED) rather than a success."""

        if not self._open:
            self._late(f"warn: {text or ''}", Severity.WARNING)
            return
        self._category = OutcomeCategory.DEFERRED
        if text is not None:
            self._detail = text

    def finish(self, *, failed: bool = False) -> None:
        """Emit the terminal BootStepFinished exactly once (idempotent)."""

        if not self._open:
            return
        self._open = False
        outcome = OutcomeCategory.FAILED if failed else self._category
        self._emit(
            BootStepFinished(
                scope=self._scope,
                label=self._label,
                outcome=outcome,
                detail=self._detail,
                elapsed_s=self._clock() - self._started,
            ),
        )

    def __enter__(self) -> StepScope:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.finish(failed=exc_type is not None)


@final
class EntryScope(_ScopeBase):
    """One entry block: opened WITH its header (header-at-open, B5); details stream."""

    _KIND_WORD: ClassVar[str] = "entry"

    def __init__(self, emit: Emit, scope: ScopeId, header: EntryHeader) -> None:
        super().__init__(emit, header.title)
        self._scope = scope
        emit(ScopeOpened(scope=scope, label=header.title))
        emit(replace(header, scope=scope))

    @property
    def scope_id(self) -> ScopeId:
        return self._scope

    def detail(
        self,
        label: str,
        value: str | StyledValue,
        *,
        severity: Severity = Severity.INFO,
        tail: str | None = None,
    ) -> None:
        styled = StyledValue(value) if isinstance(value, str) else value
        self.post(EntryDetail(label=label, value=styled, severity=severity, tail=tail))

    def warn(self, label: str, message: str) -> None:
        """An in-block, counted warning — there is no API here for a column-0 badge."""

        self.detail(label, StyledValue(message, accent=Accent.CAUTION), severity=Severity.WARNING)

    def fail(self, label: str, message: str) -> None:
        self.detail(label, StyledValue(message, accent=Accent.BAD), severity=Severity.ERROR)

    def post(self, fact: EntryFact) -> None:
        """Emit an entry-block fact stamped with this scope's id (demotes when stale)."""

        if not self._open:
            self._late(_describe_fact(fact), severity_of(fact))
            return
        self._emit(replace(fact, scope=self._scope))

    def close(self) -> None:
        """Idempotent; the reporter closes the previous entry before opening a sibling."""

        if not self._open:
            return
        self._open = False
        self._emit(ScopeClosed(scope=self._scope))


@final
class WaitScope(_ScopeBase):
    """The wait region: snapshot progress + graduations, opened/closed explicitly."""

    _KIND_WORD: ClassVar[str] = "wait"

    def __init__(self, emit: Emit, scope: ScopeId, total: int) -> None:
        super().__init__(emit, "wait")
        self._scope = scope
        emit(ScopeOpened(scope=scope, label="wait"))
        emit(WaitStarted(total=total, scope=scope))

    @property
    def scope_id(self) -> ScopeId:
        return self._scope

    def progress(self, snapshot: WaitSnapshot) -> None:
        if not self._open:
            self._late("wait progress", Severity.INFO)
            return
        self._emit(WaitProgress(snapshot=snapshot, scope=self._scope))

    def graduated(self, graduation: TorrentGraduated) -> None:
        if not self._open:
            self._late(f"{graduation.outcome.word} {graduation.label}", severity_of(graduation))
            return
        self._emit(replace(graduation, scope=self._scope))

    def finish(self, finished: WaitFinished) -> None:
        """Emit the wait tally stamped whole, then close (demotes when already closed)."""

        if not self._open:
            self._late("wait finished", Severity.INFO)
            return
        self._emit(replace(finished, scope=self._scope))
        self.close()

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._emit(ScopeClosed(scope=self._scope))


@final
class Diagnostics:
    """The weakest handle: report problems from anywhere, claim no position.

    Replaces a leaf client's ``logger`` parameter 1:1; ``origin`` is bound once at
    construction. ``once=`` keys dedup in the hub per (origin, key), cleared per
    cycle (S9).
    """

    def __init__(self, emit: Emit, origin: str) -> None:
        self._emit = emit
        self._origin = origin

    @property
    def origin(self) -> str:
        return self._origin

    def child(self, suffix: str) -> Diagnostics:
        """A narrower origin, e.g. "arr_http" -> "arr_http:Sonarr"."""

        return Diagnostics(self._emit, f"{self._origin}:{suffix}")

    def info(self, message: str, *, once: str | None = None) -> None:
        self._diag(Severity.INFO, message, once, None)

    def warn(self, message: str, *, once: str | None = None) -> None:
        self._diag(Severity.WARNING, message, once, None)

    def error(self, message: str, *, exc: BaseException | None = None, once: str | None = None) -> None:
        trace = CapturedTrace.from_exception(exc) if exc is not None else None
        self._diag(Severity.ERROR, message, once, trace)

    def _diag(self, severity: Severity, message: str, once: str | None, trace: CapturedTrace | None) -> None:
        self._emit(
            Diagnostic(
                severity=severity,
                message=message,
                origin=self._origin,
                once_key=once,
                trace=trace,
            ),
        )


@final
class ScopeFactory:
    """Bind-once producer bundle: one emitter, one id minter, one clock.

    Defaults to :data:`PROCESS_SCOPE_IDS` so two factories can never mint
    colliding serials; tests inject a fresh :class:`ScopeIds` for determinism.
    """

    def __init__(
        self,
        emit: Emit,
        *,
        clock: Callable[[], float] = time.monotonic,
        ids: ScopeIds | None = None,
    ) -> None:
        self._emit = emit
        self._clock = clock
        self._ids = ids if ids is not None else PROCESS_SCOPE_IDS

    def step(self, label: str) -> StepScope:
        return StepScope(self._emit, self._ids.mint(ScopeKind.BOOT_STEP), label, self._clock)

    def entry(self, header: EntryHeader) -> EntryScope:
        return EntryScope(self._emit, self._ids.mint(ScopeKind.ENTRY), header)

    def wait(self, total: int) -> WaitScope:
        return WaitScope(self._emit, self._ids.mint(ScopeKind.WAIT_REGION), total)

    def diagnostics(self, origin: str) -> Diagnostics:
        return Diagnostics(self._emit, origin)
