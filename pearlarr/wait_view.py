"""The wait pass's view: wait pass snapshots in, hub events out. Nothing renders here."""

import contextlib
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import NamedTuple, final, override

from .console_features import console_of, detect_features
from .output import (
    Phase,
    RowFinished,
    ScopeFactory,
    TorrentView,
    WaitFinished,
    WaitKind,
    WaitScope,
    WaitSnapshot,
    emit_to_hub,
)
from .wait_types import NO_HAND_IMPORT_FILES, LeftForHandImport, Outcome, OutcomeKind


@dataclass(frozen=True, slots=True)
class WaitOutcomeRow:
    """One finished torrent's result."""

    label: str
    outcome: Outcome
    carried_over: bool = False
    """Whether the pending import predates this run. A fresh grab tallies as `grabbed`, never `imported`."""
    left_for_hand_import: LeftForHandImport = NO_HAND_IMPORT_FILES
    """On the `NOTHING_PLACED` and `PARTIAL` outcomes, the files in the folder left for a hand import."""


@dataclass(frozen=True, slots=True)
class WaitResult:
    """The outcome of a whole wait pass."""

    rows: tuple[WaitOutcomeRow, ...]
    elapsed_s: float

    @property
    def waited(self) -> int:
        """How many torrents the pass finished."""

        return len(self.rows)

    @property
    def imported(self) -> int:
        """Count of imported torrents."""

        return self._count(OutcomeKind.IMPORTED)

    @property
    def carried_over_imported(self) -> int:
        """Count of carried-over imported torrents, the run tally's `imported` bucket."""

        return sum(1 for row in self.rows if row.carried_over and row.outcome.kind is OutcomeKind.IMPORTED)

    @property
    def left(self) -> int:
        """Count of torrents left for a later run."""

        return self._count(OutcomeKind.LEFT)

    @property
    def failed(self) -> int:
        """Count of failed torrents."""

        return self._count(OutcomeKind.FAILED)

    def _count(self, kind: OutcomeKind) -> int:
        return sum(1 for row in self.rows if row.outcome.kind is kind)


class FinishedRow(NamedTuple):
    """A newly finished torrent paired with its outcome."""

    view: TorrentView
    outcome: Outcome


def newly_finished(seen: AbstractSet[str], snapshot: WaitSnapshot) -> list[FinishedRow]:
    """The finished torrents not yet emitted, in snapshot order."""

    return [
        FinishedRow(torrent, torrent.outcome)
        for torrent in snapshot.torrents
        if torrent.phase is Phase.FINISHED and torrent.outcome is not None and torrent.key not in seen
    ]


class WaitView(ABC):
    """The interface the wait pass drives while waiting. Every method MUST be total (never raise)."""

    wants_download_progress: bool = True
    """Whether the renderers show per-row download progress."""

    @abstractmethod
    def update(self, snapshot: WaitSnapshot) -> None:
        """Narrate the latest snapshot (finishing the rows of newly finished torrents)."""

    @abstractmethod
    def close(self) -> None:
        """Finish the pass and emit the closing tally (idempotent)."""


@final
class HubWaitView(WaitView):
    """The wait-pass narrator: turns wait pass snapshots into hub events."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        progress_line_s: float,
        wants_download_progress: bool,
        kind: WaitKind,
    ) -> None:
        self._logger = logger
        self._progress_line_s = progress_line_s
        self._kind = kind
        self.wants_download_progress = wants_download_progress
        # Process-global ids through the late-resolving `emit_to_hub`.
        self._factory = ScopeFactory(emit_to_hub)
        self._scope: WaitScope | None = None
        self._seen: set[str] = set()
        self._tally: Counter[OutcomeKind] = Counter()
        self._last_elapsed = 0.0
        self._closed = False

    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        try:
            if self._closed:  # Defensive: the wait pass never updates after close.
                return
            # Stamped first, so an interrupted narration still reports fresh elapsed.
            self._last_elapsed = snapshot.elapsed_s
            if self._scope is None:
                # WaitStarted precedes any finished rows of the first snapshot.
                self._scope = self._factory.wait(
                    total=snapshot.total(), progress_line_s=self._progress_line_s, kind=self._kind
                )
            scope = self._scope
            for view, outcome in newly_finished(self._seen, snapshot):
                self._seen.add(view.key)
                self._tally[outcome.kind] += 1
                scope.row_finished(
                    RowFinished(
                        label=view.label,
                        outcome=outcome,
                        files=view.import_total,
                        waited_s=view.phase_elapsed_s,
                        left_for_hand_import=view.left_for_hand_import,
                    ),
                )
            scope.progress(snapshot)
        except Exception:
            # Total by contract: a narration bug becomes a no-op, never aborting the wait pass's
            # loop or the end-of-run cache save.
            self._logger.debug("wait view update failed", exc_info=True)

    @override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        scope = self._scope
        # Never updated, so the region never opened and there is nothing to emit.
        if scope is None:
            return
        try:
            # A zero-tally pass still finishes: the builders render [] for the empty tally, so the file stays silent.
            scope.finish(
                WaitFinished(
                    imported=self._tally[OutcomeKind.IMPORTED],
                    left=self._tally[OutcomeKind.LEFT],
                    failed=self._tally[OutcomeKind.FAILED],
                    elapsed_s=self._last_elapsed,
                    in_progress=self._tally[OutcomeKind.IN_PROGRESS],
                    kind=self._kind,
                ),
            )
        except Exception:
            self._logger.debug("wait view close failed", exc_info=True)
        finally:
            # Contains a failing close, and still closes when an interrupt aborts finish mid-dispatch.
            # An interrupt already propagating is unaffected.
            with contextlib.suppress(BaseException):
                scope.close()


def make_wait_view(
    logger: logging.Logger,
    *,
    poll_s: int,
    progress_line_interval: int = 300,
    kind: WaitKind,
) -> WaitView:
    """The production narrator, set up from the logger's console."""

    console = console_of(logger)
    features = detect_features(console)
    return HubWaitView(
        logger,
        progress_line_s=float(max(poll_s, progress_line_interval)),
        wants_download_progress=console is not None and features.can_redraw,
        kind=kind,
    )
