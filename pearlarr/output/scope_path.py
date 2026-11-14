"""The shared pure reduction of scope/boundary events to the open scope path.

One implementation, instantiated per renderer: the rich renderer's instance is the
sole position authority. Text renderer instances derive labels only - the
`[path]` scope path for handle-carried ScopeIds and the advisory `during=` tail
for diagnostics - never position, never layout.

`OpenScopes` is replay-deterministic: the same event stream always yields the same
state. The fixed transition table IS `KIND_DEPTH` plus the `match` in
`OpenScopes.apply` - boundary events close strictly-deeper scopes, never
heuristic inference. A close on an unknown/already-closed id is a no-op, so
defensive closes (the unwind teardown, the doubled arr run close) are idempotent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, assert_never, final

from .events import (
    ArrRunFinished,
    ArrRunSummaryReady,
    BootReady,
    BootStepFinished,
    BootStepProgressed,
    BootStepSlow,
    BootStepStarted,
    CacheBackedUp,
    CacheIntegrityReported,
    CacheRemoved,
    CacheRestored,
    CacheStatsReported,
    CapReached,
    ConfigMigrated,
    ConfigUpToDate,
    ConfigValidated,
    Diagnostic,
    EffectiveConfigShown,
    Event,
    GrabAction,
    GrabFailed,
    ItemStarted,
    NextRunScheduled,
    PathsShown,
    PearlarrStarted,
    ReleaseSkipped,
    RowFinished,
    ScanFinished,
    ScanStarted,
    ScheduledRunStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    StarterConfigWritten,
    TitleDetail,
    TitleHeader,
    TitleRow,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)

KIND_DEPTH: Final[Mapping[ScopeKind, int]] = {
    ScopeKind.BOOT: 1,
    ScopeKind.ARR_RUN: 1,
    ScopeKind.BOOT_STEP: 2,
    ScopeKind.ITEM: 2,
    # Same depth as ITEM is deliberate: a wait pass is mutually exclusive
    # with an item. Revisit only if a per-item wait ever appears.
    ScopeKind.WAIT: 2,
    ScopeKind.TITLE: 3,
}

# The scope kinds whose path segment is the kind's word. Other kinds render their label.
WORD_SEGMENT_KINDS: Final = frozenset({ScopeKind.BOOT, ScopeKind.TITLE, ScopeKind.WAIT})

# Path separator for the text renderers' [path] scope path.
PATH_SEP: Final = " › "


@dataclass(frozen=True, slots=True)
class OpenScope:
    """One open scope."""

    kind: ScopeKind
    label: str
    scope: ScopeId | None
    """`None` for boundary-opened ARR_RUN/ITEM scopes."""


def _segment(open_scope: OpenScope) -> str:
    """A scope's path segment: the kind's word for the `WORD_SEGMENT_KINDS`, the label otherwise."""

    return str(open_scope.kind) if open_scope.kind in WORD_SEGMENT_KINDS else open_scope.label


@final
class OpenScopes:
    """The open scope path state, built purely from the event stream (one per renderer)."""

    def __init__(self) -> None:
        self._stack: list[OpenScope] = []

    def apply(self, event: Event) -> None:
        """Advance the open scopes by one event, per the fixed transition table above."""

        match event:
            case ScopeOpened(scope=scope, label=label):
                self._push(scope.kind, label, scope)
            case ScopeClosed(scope=scope):
                self._pop_through(scope)
            case BootStepStarted(scope=scope, label=label):
                self._push(ScopeKind.BOOT_STEP, label, scope)
            case BootStepFinished(scope=scope):
                self._pop_through(scope)
            case PearlarrStarted() | ScheduledRunStarted() | ArrRunFinished():
                self._close_at(1)  # unwind to root (all levels)
            case ScanStarted(arr=arr):
                self._push(ScopeKind.ARR_RUN, str(arr))
            case ItemStarted(index=index, total=total, title=title):
                self._push(ScopeKind.ITEM, f"[{index}/{total}] {title}")
            case ScanFinished() | ArrRunSummaryReady():
                self._close_at(2)  # close item/wait/title, keep the arr run
            case WaitStarted():
                self._close_at(3)  # close the title before the wait pass
            case (
                NextRunScheduled()
                | BootStepProgressed()
                | BootStepSlow()
                | BootReady()
                | TitleHeader()
                | TitleDetail()
                | TitleRow()
                | ReleaseSkipped()
                | GrabFailed()
                | GrabAction()
                | CapReached()
                | WaitProgress()
                | RowFinished()
                | WaitFinished()
                | Diagnostic()
            ):
                pass
            case (
                PathsShown()
                | StarterConfigWritten()
                | ConfigValidated()
                | ConfigUpToDate()
                | ConfigMigrated()
                | EffectiveConfigShown()
                | CacheBackedUp()
                | CacheRestored()
                | CacheRemoved()
                | CacheStatsReported()
                | CacheIntegrityReported()
            ):
                # cli command facts - never emitted during a run.
                pass
            case _:
                assert_never(event)

    # -- read APIs (the only interface: labels/during only, never position) --------

    def scopes(self) -> tuple[OpenScope, ...]:
        """The open scopes, root to top."""

        return tuple(self._stack)

    def path_text(self) -> str:
        """The full PATH_SEP-joined scope path, e.g. "sonarr" then "[3/182] Title" then "title"."""

        return PATH_SEP.join(_segment(open_scope) for open_scope in self._stack)

    def path_for(self, scope: ScopeId) -> str | None:
        """The scope path down to the named open scope, or None when it isn't open."""

        for i, open_scope in enumerate(self._stack):
            if open_scope.scope == scope:
                return PATH_SEP.join(_segment(n) for n in self._stack[: i + 1])
        return None

    def during(self) -> str | None:
        """The top scope's label - the advisory `during=` tail for diagnostics."""

        return self._stack[-1].label if self._stack else None

    def reset(self) -> None:
        """Clear all open scopes (begin_run)."""

        self._stack.clear()

    # -- the transition primitives -----------------------------------------------

    def _push(self, kind: ScopeKind, label: str, scope: ScopeId | None = None) -> None:
        # Closing >= own depth first keeps the stack strictly increasing in depth.
        self._close_at(KIND_DEPTH[kind])
        self._stack.append(OpenScope(kind, label, scope))

    def _close_at(self, depth: int) -> None:
        while self._stack and KIND_DEPTH[self._stack[-1].kind] >= depth:
            self._stack.pop()

    def _pop_through(self, scope: ScopeId) -> None:
        for i, open_scope in enumerate(self._stack):
            if open_scope.scope == scope:
                del self._stack[i:]
                return
