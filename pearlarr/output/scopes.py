"""Typed capability scope handles: position travels with the handle, never the call site.

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
    Event,
    GrabAction,
    GrabFailed,
    PlacedBy,
    ReleaseSkipped,
    RowFinished,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    Severity,
    TitleDetail,
    TitleHeader,
    TitleRow,
    WaitFinished,
    WaitKind,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    clamp01,
    severity_of,
)
from .hub import SeverityCounter
from .process_hub import emit_to_hub
from ..wait_types import OutcomeKind

type Emit = Callable[[Event], None]

type CountsSource = Callable[[], SeverityCounter]

type TitleEvent = TitleDetail | TitleRow | ReleaseSkipped | GrabFailed | GrabAction


@final
class ScopeIdCounter:
    """Thread-safe ScopeId minter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._serials = itertools.count(1)

    def mint(self, kind: ScopeKind) -> ScopeId:
        with self._lock:
            return ScopeId(kind, next(self._serials))


# The process-wide default minter: factories share it so serials never collide across factories.
PROCESS_SCOPE_IDS: Final = ScopeIdCounter()


@final
class BootScope:
    """The boot flow's ambient-scope mark: idempotent open and close calls."""

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


def _describe_fact(fact: TitleEvent) -> str:
    """A compact one-line description of a demoted fact (for the late diagnostic)."""

    match fact:
        case TitleDetail(label=label, value=value):
            return f"{label}: {value.text}"
        case TitleRow(state=state, label=label):
            return f"{state} {label}"
        case ReleaseSkipped(group=group, tracker=tracker, reason=reason):
            return f"release skipped: {group} on {tracker} ({reason.name.lower()})"
        case GrabFailed(group=group, error=error):
            return f"grab failed: {group} ({error})"
        case GrabAction(status=status):
            return f"grab action: {status.name.lower()}"
    assert_never(fact)


class _ScopeBase:
    """Shared handle base class, including the closed-handle demotion."""

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
        self._category = OutcomeKind.IMPORTED
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
        """Set the detail the finished boot step line carries (e.g. "42 series")."""

        if not self._open:
            self._late(f"note: {text}", Severity.INFO)
            return
        self._detail = text

    def warn(self, text: str | None = None) -> None:
        """Finish this step as a warning (DEFERRED) rather than a success."""

        if not self._open:
            self._late(f"warn: {text or ''}", Severity.WARNING)
            return
        self._category = OutcomeKind.LEFT
        if text is not None:
            self._detail = text

    def finish(self, *, failed: bool = False) -> None:
        """Idempotent."""

        if not self._open:
            return
        self._open = False
        outcome = OutcomeKind.FAILED if failed else self._category
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
class TitleScope(_ScopeBase):
    """One title block, opened with its header."""

    _KIND_WORD: ClassVar[str] = "entry"

    def __init__(self, emit: Emit, scope: ScopeId, header: TitleHeader) -> None:
        super().__init__(emit, header.title, scope)
        emit(ScopeOpened(scope=scope, label=header.title))
        emit(replace(header, scope=scope))

    def post(self, fact: TitleEvent) -> None:
        """Emit a title block fact tagged with this scope's id (demotes when closed)."""

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
    """The wait region: snapshot progress and finished rows."""

    _KIND_WORD: ClassVar[str] = "wait"

    def __init__(self, emit: Emit, scope: ScopeId, total: int, *, progress_line_s: float, kind: WaitKind) -> None:
        super().__init__(emit, "wait", scope)
        emit(ScopeOpened(scope=scope, label="wait"))
        emit(WaitStarted(total=total, progress_line_s=progress_line_s, kind=kind, scope=scope))

    def progress(self, snapshot: WaitSnapshot) -> None:
        if not self._open:
            self._late("wait progress", Severity.INFO)
            return
        self._emit(WaitProgress(snapshot=snapshot, scope=self._scope))

    def row_finished(self, event: RowFinished) -> None:
        if not self._open:
            self._late(f"{event.outcome.word} {event.label}", severity_of(event))
            return
        self._emit(replace(event, scope=self._scope))

    def finish(self, finished: WaitFinished) -> None:
        """Emit the whole wait tally tagged with this scope's id, then close (demotes when already closed)."""

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
        ids: ScopeIdCounter | None = None,
    ) -> None:
        self._emit = emit
        self._clock = clock
        self._ids = ids if ids is not None else PROCESS_SCOPE_IDS

    def step(self, label: str) -> StepScope:
        return StepScope(self._emit, self._ids.mint(ScopeKind.BOOT_STEP), label, self._clock)

    def title(self, header: TitleHeader) -> TitleScope:
        return TitleScope(self._emit, self._ids.mint(ScopeKind.ENTRY), header)

    def wait(self, total: int, *, progress_line_s: float, kind: WaitKind) -> WaitScope:
        return WaitScope(
            self._emit, self._ids.mint(ScopeKind.WAIT_REGION), total, progress_line_s=progress_line_s, kind=kind
        )
