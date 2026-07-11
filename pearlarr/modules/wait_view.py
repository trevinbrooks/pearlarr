"""The wait-pass producer: engine snapshots in, hub events out (PR5).

The engine drives a :class:`WaitView` while it waits on each grabbed torrent to
download and then import, pushing one immutable :class:`WaitSnapshot` per poll
cycle. Nothing renders here: :class:`HubWaitView` - the narrator - turns each
push into hub events (``WaitStarted`` on the first snapshot via a
:class:`~.output.scopes.WaitScope`, one ``TorrentGraduated`` per newly-terminal
torrent, ``WaitProgress`` per poll, the ``WaitFinished`` tally on close), and
the renderers own every look decision: the RichRenderer's
:class:`~.output.wait_region.WaitRegion` draws the live cockpit / non-TTY
digest, and the hub's text sinks write the structured file/plain/json lines.
Every method is total: a presentation bug degrades to a no-op, never aborting
the wait loop or the end-of-run cache save.
"""

import contextlib
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import final, override

from .console_caps import console_of, detect_capabilities
from .manual_import import Outcome, OutcomeCategory
from .output import (
    Phase,
    ScopeFactory,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitScope,
    WaitSnapshot,
    emit_to_hub,
)


@dataclass(frozen=True, slots=True)
class WaitOutcomeRow:
    """One torrent's terminal result, captured by the monitor for the run report."""

    label: str
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class WaitResult:
    """The outcome of a whole wait pass - the completion notification's payload.

    Returned by :meth:`ImportWaitManager.run_monitor` so the run loop can push
    the Discord/webhook completion notification (``Notifier.push_wait_summary``)
    without re-deriving state.
    """

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
    def left(self) -> int:
        """Count of deferred ("left for a later run") torrents."""

        return self._count(OutcomeCategory.DEFERRED)

    @property
    def failed(self) -> int:
        """Count of failed torrents."""

        return self._count(OutcomeCategory.FAILED)

    def _count(self, category: OutcomeCategory) -> int:
        return sum(1 for row in self.rows if row.outcome.category is category)


def graduations(seen: AbstractSet[str], snapshot: WaitSnapshot) -> list[TorrentView]:
    """The terminal torrents not yet emitted - pure, deterministic.

    A torrent graduates exactly once: the narrator tracks the keys it has already
    emitted and this returns the newly-terminal ones in snapshot order.
    """

    return [
        torrent
        for torrent in snapshot.torrents
        if torrent.phase is Phase.TERMINAL and torrent.outcome is not None and torrent.key not in seen
    ]


class WaitView(ABC):
    """The small interface the engine drives while waiting on downloads/imports.

    The engine pushes a full :class:`WaitSnapshot` each poll cycle. Both methods
    MUST be total (never raise) so a presentation bug can't abort the wait loop
    or the end-of-run cache save.
    """

    # Whether this pass's render surfaces show per-row download telemetry between
    # heavy polls; the engine skips the fast-lane qBittorrent read when it can't
    # be seen. Per-instance (one narrator class serves both seats).
    wants_telemetry: bool = True

    @abstractmethod
    def update(self, snapshot: WaitSnapshot) -> None:
        """Narrate the latest snapshot (graduating any newly-terminal torrents)."""

    @abstractmethod
    def close(self) -> None:
        """Finish the pass and emit the closing tally (idempotent)."""


@final
class HubWaitView(WaitView):
    """The wait-pass narrator: turns engine snapshots into hub events (P1).

    Holds producer state only (seen keys, the outcome tally, the last elapsed
    clock); the renderers decide every look. The wait scope opens lazily on the
    first snapshot, so a pass that never polls emits nothing.
    """

    def __init__(self, logger: logging.Logger, *, pulse_s: float, wants_telemetry: bool) -> None:
        self._logger = logger
        self._pulse_s = pulse_s
        self.wants_telemetry = wants_telemetry
        # Process-global ids through the late-resolving hub seam (the same path
        # the old views' ScopeMark graft used).
        self._factory = ScopeFactory(emit_to_hub)
        self._scope: WaitScope | None = None
        self._seen: set[str] = set()
        self._tally: Counter[OutcomeCategory] = Counter()
        self._last_elapsed = 0.0
        self._closed = False

    @override
    def update(self, snapshot: WaitSnapshot) -> None:
        try:
            if self._closed:  # defensive; the engine never updates after close
                return
            # Stamped first, so an interrupted narration still reports fresh elapsed.
            self._last_elapsed = snapshot.elapsed_s
            if self._scope is None:
                # Deliberate order flip vs the old views: WaitStarted now precedes
                # any first-snapshot graduations (they logged graduations first).
                self._scope = self._factory.wait(total=snapshot.total(), pulse_s=self._pulse_s)
            scope = self._scope
            for torrent in graduations(self._seen, snapshot):
                outcome = torrent.outcome
                if outcome is None:  # pragma: no cover - graduations() guarantees this
                    continue
                self._seen.add(torrent.key)
                self._tally[outcome.category] += 1
                scope.graduated(
                    TorrentGraduated(
                        label=torrent.label,
                        outcome=outcome,
                        files=torrent.import_total,
                        waited_s=torrent.phase_elapsed_s,
                    ),
                )
            scope.progress(snapshot)
        except Exception:
            # Total by contract: a narration bug degrades to a no-op, it never
            # aborts the engine's wait loop or the end-of-run cache save.
            self._logger.debug("wait view update failed", exc_info=True)

    @override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        scope = self._scope
        if scope is None:
            # Never updated: the region never opened, so there is nothing to emit.
            return
        try:
            # A zero-tally pass still finishes: the builders render [] for the
            # empty tally, so the file stays silent (parity with the old views).
            scope.finish(
                WaitFinished(
                    imported=self._tally[OutcomeCategory.SUCCESS],
                    deferred=self._tally[OutcomeCategory.DEFERRED],
                    failed=self._tally[OutcomeCategory.FAILED],
                    elapsed_s=self._last_elapsed,
                ),
            )
        except Exception:
            self._logger.debug("wait view close failed", exc_info=True)
        finally:
            # The placement scope must still close (the old unconditional close),
            # even when an interrupt aborts finish mid-dispatch; a no-op after a
            # clean finish, and suppress keeps a propagating interrupt intact.
            with contextlib.suppress(BaseException):
                scope.close()


def make_wait_view(logger: logging.Logger, *, poll_s: int, digest_interval: int = 300) -> WaitView:
    """The production narrator, probed off the logger's console.

    Args:
        logger (logging.Logger): The app logger; its rich console handler is
            probed so ``wants_telemetry`` matches what the console will draw
            (per-row telemetry exists only on the live-TTY cockpit).
        poll_s (int): The poll cadence - the floor for the non-TTY digest interval.
        digest_interval (int): Target seconds between non-TTY aggregate pulses.
    """

    console = console_of(logger)
    caps = detect_capabilities(console)
    return HubWaitView(
        logger,
        pulse_s=float(max(poll_s, digest_interval)),
        wants_telemetry=console is not None and caps.live,
    )
