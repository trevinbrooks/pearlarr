"""Typed capability scope handles: position rides the handle, never the call site.

Emitting on a closed handle demotes to a `Diagnostic` instead of raising.
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
    TorrentGraduated,
    WaitFinished,
    WaitKind,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    clamp01,
    severity_of,
)
from .hub import SeverityCounts
from .runtime import emit_to_hub
from ..manual_import import OutcomeCategory

type Emit = Callable[[Event], None]

type CountsSource = Callable[[], SeverityCounts]

type EntryFact = EntryDetail | LedgerRow | ReleaseSkipped | GrabFailed | GrabAction


@final
class ScopeIds:
    """Thread-safe ScopeId minter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._serials = itertools.count(1)

    def mint(self, kind: ScopeKind) -> ScopeId:
        with self._lock:
            return ScopeId(kind, next(self._serials))


# The process-wide default minter: factories share it so serials never collide across factories.
PROCESS_SCOPE_IDS: Final = ScopeIds()


@final
class ScopeMark:
    """The boot flow's ambient-scope mark ceremony: idempotent open/close."""

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
    """Shared handle spine, including the closed-handle demotion."""

    _KIND_WORD: ClassVar[str] = "scope"

    def __init__(self, emit: Emit, label: str, scope: ScopeId) -> None:
        self._emit = emit
        self._label = label
        self._scope = scope
        self._open = True

    @property
    def scope_id(self) -> ScopeId:
        return self._scope

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
    """One boot step: progress/note/warn producer-side, timing here, events out."""

    _KIND_WORD: ClassVar[str] = "step"

    def __init__(self, emit: Emit, scope: ScopeId, label: str, clock: Callable[[], float]) -> None:
        super().__init__(emit, label, scope)
        self._clock = clock
        self._started = clock()
        self._category = OutcomeCategory.SUCCESS
        self._detail: str | None = None
        self._slow_sent = False
        emit(BootStepStarted(scope=scope, label=label))

    def progress(self, fraction: float, detail: str | None = None) -> None:
        """Report 0-1 progress. The first report also emits the one-time slow heads-up."""

        if not self._open:
            self._late(f"progress {fraction:.2f}", Severity.INFO)
            return
        if not self._slow_sent:
            self._slow_sent = True
            self._emit(BootStepSlow(scope=self._scope, label=self._label))
        if detail is not None:
            self._detail = detail
        self._emit(BootStepProgressed(scope=self._scope, fraction=clamp01(fraction), detail=self._detail))

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
        """Idempotent."""

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
    """One entry block, opened with its header."""

    _KIND_WORD: ClassVar[str] = "entry"

    def __init__(self, emit: Emit, scope: ScopeId, header: EntryHeader) -> None:
        super().__init__(emit, header.title, scope)
        emit(ScopeOpened(scope=scope, label=header.title))
        emit(replace(header, scope=scope))

    def post(self, fact: EntryFact) -> None:
        """Emit an entry-block fact stamped with this scope's id (demotes when stale)."""

        if not self._open:
            self._late(_describe_fact(fact), severity_of(fact))
            return
        self._emit(replace(fact, scope=self._scope))

    def close(self) -> None:
        """Idempotent."""

        if not self._open:
            return
        self._open = False
        self._emit(ScopeClosed(scope=self._scope))


@final
class WaitScope(_ScopeBase):
    """The wait region: snapshot progress and graduations."""

    _KIND_WORD: ClassVar[str] = "wait"

    def __init__(self, emit: Emit, scope: ScopeId, total: int, *, pulse_s: float, kind: WaitKind) -> None:
        super().__init__(emit, "wait", scope)
        emit(ScopeOpened(scope=scope, label="wait"))
        emit(WaitStarted(total=total, pulse_s=pulse_s, kind=kind, scope=scope))

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
        """Idempotent."""

        if not self._open:
            return
        self._open = False
        self._emit(ScopeClosed(scope=self._scope))


@final
class ScopeFactory:
    """Bind-once producer bundle: emitter, id minter, clock."""

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

    def wait(self, total: int, *, pulse_s: float, kind: WaitKind = WaitKind.MONITOR) -> WaitScope:
        return WaitScope(self._emit, self._ids.mint(ScopeKind.WAIT_REGION), total, pulse_s=pulse_s, kind=kind)
