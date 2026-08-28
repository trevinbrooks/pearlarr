"""The wait-pass producer: engine snapshots in, hub events out. Nothing renders here."""

import contextlib
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import NamedTuple, final, override

from .console_caps import console_of, detect_capabilities
from .manual_import import Outcome, OutcomeCategory
from .output import (
    Phase,
    ScopeFactory,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitKind,
    WaitScope,
    WaitSnapshot,
    emit_to_hub,
)


@dataclass(frozen=True, slots=True)
class WaitOutcomeRow:
    """One torrent's terminal result."""

    label: str
    outcome: Outcome
    carried_over: bool = False
    """Whether the record predates this run. A fresh grab tallies as `added`, never `imported`."""


@dataclass(frozen=True, slots=True)
class WaitResult:
    """The outcome of a whole wait pass."""

    rows: tuple[WaitOutcomeRow, ...]
    elapsed_s: float

    @property
    def waited(self) -> int:
        """How many torrents the pass reached a terminal outcome for."""

        return len(self.rows)

    @property
    def imported(self) -> int:
        """Count of imported (SUCCESS) torrents."""

        return self._count(OutcomeCategory.SUCCESS)

    @property
    def carried_over_imported(self) -> int:
        """Count of carried-over imported torrents, the run tally's `imported` bucket."""

        return sum(1 for row in self.rows if row.carried_over and row.outcome.category is OutcomeCategory.SUCCESS)

    @property
    def left(self) -> int:
        """Count of deferred ("left for a later run") torrents."""

        return self._count(OutcomeCategory.DEFERRED)

    @property
    def failed(self) -> int:
        """Count of failed torrents."""

        return self._count(OutcomeCategory.FAILED)

    def _count(self, category: OutcomeCategory) -> int:
        return sum(1 for row in self.rows if row.outcome.category is category)


class Graduation(NamedTuple):
    """A newly-terminal torrent paired with its outcome."""

    view: TorrentView
    outcome: Outcome


def graduations(seen: AbstractSet[str], snapshot: WaitSnapshot) -> list[Graduation]:
    """The terminal torrents not yet emitted, in snapshot order."""

    return [
        Graduation(torrent, torrent.outcome)
        for torrent in snapshot.torrents
        if torrent.phase is Phase.TERMINAL and torrent.outcome is not None and torrent.key not in seen
    ]


class WaitView(ABC):
    """The interface the engine drives while waiting. Every method MUST be total (never raise)."""

    wants_telemetry: bool = True
    """Whether the render surfaces show per-row download telemetry."""

    @abstractmethod
    def update(self, snapshot: WaitSnapshot) -> None:
        """Narrate the latest snapshot (graduating any newly-terminal torrents)."""

    @abstractmethod
    def close(self) -> None:
        """Finish the pass and emit the closing tally (idempotent)."""


@final
class HubWaitView(WaitView):
    """The wait-pass narrator: turns engine snapshots into hub events."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        pulse_s: float,
        wants_telemetry: bool,
        kind: WaitKind,
    ) -> None:
        self._logger = logger
        self._pulse_s = pulse_s
        self._kind = kind
        self.wants_telemetry = wants_telemetry
        # Process-global ids through the late-resolving hub seam.
        self._factory = ScopeFactory(emit_to_hub)
        self._scope: WaitScope | None = None
        self._seen: set[str] = set()
        self._tally: Counter[OutcomeCategory] = Counter()
        self._last_elapsed = 0.0
        self._closed = False

    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        try:
            if self._closed:  # Defensive: the engine never updates after close.
                return
            # Stamped first, so an interrupted narration still reports fresh elapsed.
            self._last_elapsed = snapshot.elapsed_s
            if self._scope is None:
                # WaitStarted precedes any first-snapshot graduations.
                self._scope = self._factory.wait(total=snapshot.total(), pulse_s=self._pulse_s, kind=self._kind)
            scope = self._scope
            for view, outcome in graduations(self._seen, snapshot):
                self._seen.add(view.key)
                self._tally[outcome.category] += 1
                scope.graduated(
                    TorrentGraduated(
                        label=view.label,
                        outcome=outcome,
                        files=view.import_total,
                        waited_s=view.phase_elapsed_s,
                    ),
                )
            scope.progress(snapshot)
        except Exception:
            # Total by contract: a narration bug degrades to a no-op, never aborting the engine's
            # wait loop or the end-of-run cache save.
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
                    imported=self._tally[OutcomeCategory.SUCCESS],
                    deferred=self._tally[OutcomeCategory.DEFERRED],
                    failed=self._tally[OutcomeCategory.FAILED],
                    elapsed_s=self._last_elapsed,
                    pending=self._tally[OutcomeCategory.PENDING],
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
    digest_interval: int = 300,
    kind: WaitKind,
) -> WaitView:
    """The production narrator, probed off the logger's console."""

    console = console_of(logger)
    caps = detect_capabilities(console)
    return HubWaitView(
        logger,
        pulse_s=float(max(poll_s, digest_interval)),
        wants_telemetry=console is not None and caps.live,
        kind=kind,
    )
