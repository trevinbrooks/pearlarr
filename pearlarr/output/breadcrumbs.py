"""The shared pure fold of scope/boundary events into the open-node path.

One implementation, instantiated per surface: the rich renderer's instance is the
sole placement authority. Text-sink instances derive labels only - the
`[path]` breadcrumb for handle-carried ScopeIds and the advisory `during=` tail
for diagnostics - never position, never layout.

The fold is replay-deterministic: the same event stream always yields the same
state. The fixed transition table IS `KIND_DEPTH` plus the `match` in
`BreadcrumbFold.apply` - boundary events close strictly-deeper nodes, never
heuristic inference. A close on an unknown/already-closed id is a no-op, so
defensive closes (the unwind teardown, the doubled run-close) are idempotent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, assert_never, final

from .events import (
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
    CycleStarted,
    Diagnostic,
    EffectiveConfigShown,
    EntryDetail,
    EntryHeader,
    Event,
    GrabAction,
    GrabFailed,
    ItemStarted,
    LedgerRow,
    NextRunScheduled,
    PathsShown,
    ReleaseSkipped,
    RunFinished,
    RunStarted,
    RunSummaryReady,
    ScanFinished,
    ScanStarted,
    ScopeClosed,
    ScopeId,
    ScopeKind,
    ScopeOpened,
    StarterConfigWritten,
    TorrentGraduated,
    WaitFinished,
    WaitProgress,
    WaitStarted,
)

KIND_DEPTH: Final[Mapping[ScopeKind, int]] = {
    ScopeKind.BOOT_SECTION: 1,
    ScopeKind.RUN: 1,
    ScopeKind.BOOT_STEP: 2,
    ScopeKind.ITEM: 2,
    # Same depth as ITEM is deliberate: a wait pass is mutually exclusive
    # with an item. Revisit only if a per-item wait ever appears.
    ScopeKind.WAIT_REGION: 2,
    ScopeKind.ENTRY: 3,
}

# Fixed path segments for section-like kinds. Other kinds render their label.
SEGMENT_WORD: Final[Mapping[ScopeKind, str]] = {
    ScopeKind.BOOT_SECTION: "boot",
    ScopeKind.ENTRY: "entry",
    ScopeKind.WAIT_REGION: "wait",
}

# Path separator for the text sinks' [path] breadcrumb.
PATH_SEP: Final = " › "


@dataclass(frozen=True, slots=True)
class OpenNode:
    """One open frontier node."""

    kind: ScopeKind
    label: str
    scope: ScopeId | None
    """`None` for boundary-opened RUN/ITEM nodes."""


def _segment(node: OpenNode) -> str:
    """A node's path segment: fixed words for section kinds, the label otherwise."""

    return SEGMENT_WORD.get(node.kind, node.label)


@final
class BreadcrumbFold:
    """The open-path state, folded purely from the event stream (one per surface)."""

    def __init__(self) -> None:
        self._stack: list[OpenNode] = []

    def apply(self, event: Event) -> None:
        """Advance the fold by one event, per the fixed transition table above."""

        match event:
            case ScopeOpened(scope=scope, label=label):
                self._push(scope.kind, label, scope)
            case ScopeClosed(scope=scope):
                self._pop_through(scope)
            case BootStepStarted(scope=scope, label=label):
                self._push(ScopeKind.BOOT_STEP, label, scope)
            case BootStepFinished(scope=scope):
                self._pop_through(scope)
            case RunStarted() | CycleStarted() | RunFinished():
                self._close_at(1)  # unwind to root (all levels)
            case ScanStarted(arr=arr):
                self._push(ScopeKind.RUN, str(arr))
            case ItemStarted(index=index, total=total, title=title):
                self._push(ScopeKind.ITEM, f"[{index}/{total}] {title}")
            case ScanFinished() | RunSummaryReady():
                self._close_at(2)  # close item/wait/entry, keep the run
            case WaitStarted():
                self._close_at(3)  # close the entry before the wait pass
            case (
                NextRunScheduled()
                | BootStepProgressed()
                | BootStepSlow()
                | BootReady()
                | EntryHeader()
                | EntryDetail()
                | LedgerRow()
                | ReleaseSkipped()
                | GrabFailed()
                | GrabAction()
                | CapReached()
                | WaitProgress()
                | TorrentGraduated()
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

    # -- read APIs (the only surface: labels/during only, never placement) ---------

    def nodes(self) -> tuple[OpenNode, ...]:
        """The open nodes, root to top."""

        return tuple(self._stack)

    def path_text(self) -> str:
        """The full PATH_SEP-joined breadcrumb, e.g. "sonarr" then "[3/182] Frieren" then "entry"."""

        return PATH_SEP.join(_segment(node) for node in self._stack)

    def path_for(self, scope: ScopeId) -> str | None:
        """The breadcrumb down to the named open scope, or None when it isn't open."""

        for i, node in enumerate(self._stack):
            if node.scope == scope:
                return PATH_SEP.join(_segment(n) for n in self._stack[: i + 1])
        return None

    def during(self) -> str | None:
        """The top node's label - the advisory `during=` tail for diagnostics."""

        return self._stack[-1].label if self._stack else None

    def reset(self) -> None:
        """Clear all open nodes (begin_cycle)."""

        self._stack.clear()

    # -- the transition primitives -----------------------------------------------

    def _push(self, kind: ScopeKind, label: str, scope: ScopeId | None = None) -> None:
        # Closing >= own depth first keeps the stack strictly increasing in depth.
        self._close_at(KIND_DEPTH[kind])
        self._stack.append(OpenNode(kind, label, scope))

    def _close_at(self, depth: int) -> None:
        while self._stack and KIND_DEPTH[self._stack[-1].kind] >= depth:
            self._stack.pop()

    def _pop_through(self, scope: ScopeId) -> None:
        for i, node in enumerate(self._stack):
            if node.scope == scope:
                del self._stack[i:]
                return
