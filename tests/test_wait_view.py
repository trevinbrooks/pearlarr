# pyright: strict
"""Tests for the wait-pass narrator (``wait_view``).

The engine pushes one immutable :class:`WaitSnapshot` per poll cycle; the view
side is a single narrator (:class:`HubWaitView`) that turns those pushes into
hub events - rendering lives behind the hub (WaitRegion for the console, the
textline sinks for file/plain/json; pinned in test_output_wait_region /
test_output_textline). These pin the narrator's event grammar (lazy open,
graduation dedup + exact field mapping, the close tally), the ``make_wait_view``
probe (wants_telemetry + the pulse interval), the pure ``graduations`` helper,
the WaitResult counting, and the no-throw contract.
"""

import io
import logging

import pytest
from rich.console import Console

from seadexarr.modules.log import RichConsoleHandler
from seadexarr.modules.manual_import import Outcome, OutcomeCategory
from seadexarr.modules.output import (
    Event,
    Phase,
    ScopeClosed,
    ScopeKind,
    ScopeOpened,
    TorrentGraduated,
    TorrentView,
    WaitFinished,
    WaitProgress,
    WaitSnapshot,
    WaitStarted,
    install_hub,
)
from seadexarr.modules.output.recording import RecordingHub
from seadexarr.modules.wait_view import HubWaitView, WaitOutcomeRow, WaitResult, graduations, make_wait_view

from .fakes import CaptureHandler


def _logger_with_console(*, force_terminal: bool, width: int = 100) -> logging.Logger:
    logger = logging.getLogger(f"wait-view-test-{force_terminal}-{width}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(RichConsoleHandler(Console(file=io.StringIO(), force_terminal=force_terminal, width=width)))
    return logger


def _quiet_logger(name: str = "wait-narrator-test") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def _downloading(key: str, label: str, frac: float = 0.5) -> TorrentView:
    return TorrentView(key=key, label=label, phase=Phase.DOWNLOADING, fraction=frac)


def _terminal(key: str, label: str, outcome: Outcome, *, files: int | None = None, elapsed: float = 0.0) -> TorrentView:
    return TorrentView(
        key=key,
        label=label,
        phase=Phase.TERMINAL,
        outcome=outcome,
        import_done=files,
        import_total=files,
        phase_elapsed_s=elapsed,
    )


# --- factory / capability probe ------------------------------------------------


def test_factory_wants_telemetry_on_a_live_tty() -> None:
    view = make_wait_view(_logger_with_console(force_terminal=True), poll_s=30)

    assert isinstance(view, HubWaitView)
    assert view.wants_telemetry is True


def test_factory_skips_telemetry_on_a_non_tty() -> None:
    # The non-TTY digest renders no per-row telemetry, so the engine's fast-lane
    # qBittorrent read is skipped for it (pure waste on Docker/cron).
    assert make_wait_view(_logger_with_console(force_terminal=False), poll_s=30).wants_telemetry is False


def test_factory_skips_telemetry_without_a_console() -> None:
    # plain/json logging: no rich console handler at all.
    assert make_wait_view(_quiet_logger("wait-view-null"), poll_s=30).wants_telemetry is False


def test_factory_skips_telemetry_when_too_narrow() -> None:
    # A real TTY too narrow for a legible cockpit folds to the digest path.
    assert make_wait_view(_logger_with_console(force_terminal=True, width=20), poll_s=30).wants_telemetry is False


def test_factory_pulse_is_the_max_of_poll_and_digest_as_a_float() -> None:
    recording = RecordingHub()
    install_hub(recording.hub)
    logger = _quiet_logger()

    make_wait_view(logger, poll_s=600, digest_interval=300).update(WaitSnapshot(()))  # the poll floor rules
    make_wait_view(logger, poll_s=30, digest_interval=300).update(WaitSnapshot(()))  # the digest target rules

    floor, target = recording.of_type(WaitStarted)
    assert floor.pulse_s == 600.0
    assert target.pulse_s == 300.0
    assert type(floor.pulse_s) is float


# --- the narrator's event grammar ------------------------------------------------


class TestHubWaitViewNarration:
    """The narrator's hub-event grammar, pinned through a real recording hub."""

    @staticmethod
    def _view(*, pulse_s: float = 300.0) -> HubWaitView:
        return HubWaitView(_quiet_logger(), pulse_s=pulse_s, wants_telemetry=True)

    def test_first_update_opens_starts_and_progresses_exactly_once(self) -> None:
        recording = RecordingHub()
        install_hub(recording.hub)
        view = self._view()
        first = WaitSnapshot((_downloading("h1", "A"), _downloading("h2", "B")), elapsed_s=5)

        view.update(first)

        assert [type(e) for e in recording.events] == [ScopeOpened, WaitStarted, WaitProgress]
        (opened,) = recording.of_type(ScopeOpened)
        assert opened.scope.kind is ScopeKind.WAIT_REGION
        (started,) = recording.of_type(WaitStarted)
        assert started == WaitStarted(total=2, pulse_s=300.0, scope=opened.scope)
        assert recording.of_type(WaitProgress) == [WaitProgress(snapshot=first, scope=opened.scope)]

        # Later updates never re-open; the total stays the FIRST snapshot's.
        view.update(WaitSnapshot((_downloading("h1", "A"), _downloading("h2", "B"), _downloading("h3", "C"))))
        view.update(WaitSnapshot((_downloading("h1", "A"),)))
        assert len(recording.of_type(ScopeOpened)) == 1
        assert recording.of_type(WaitStarted) == [started]

    def test_graduation_dedup_and_exact_field_mapping(self) -> None:
        recording = RecordingHub()
        install_hub(recording.hub)
        view = self._view()
        done = _terminal("h1", "Bocchi the Rock!", Outcome.IMPORTED, files=12, elapsed=192.0)

        view.update(WaitSnapshot((done, _downloading("h2", "B")), elapsed_s=200))
        # h1 stays terminal in the next snapshot: it must NOT graduate twice.
        view.update(WaitSnapshot((done, _terminal("h2", "B", Outcome.DOWNLOAD_TIMED_OUT)), elapsed_s=500))

        (opened,) = recording.of_type(ScopeOpened)
        assert recording.of_type(TorrentGraduated) == [
            TorrentGraduated(
                label="Bocchi the Rock!",
                outcome=Outcome.IMPORTED,
                files=12,  # = import_total
                waited_s=192.0,  # = phase_elapsed_s
                scope=opened.scope,
            ),
            TorrentGraduated(
                label="B",
                outcome=Outcome.DOWNLOAD_TIMED_OUT,
                files=None,
                waited_s=0.0,
                scope=opened.scope,
            ),
        ]

    def test_first_snapshot_terminals_graduate_after_the_start(self) -> None:
        # The documented Band C divergence, pinned deliberately: the old views
        # logged a first-snapshot graduation BEFORE their start line; the
        # narrator opens (ScopeOpened + WaitStarted) first.
        recording = RecordingHub()
        install_hub(recording.hub)

        self._view().update(WaitSnapshot((_terminal("h1", "A", Outcome.IMPORTED),), elapsed_s=10))

        assert [type(e) for e in recording.events] == [ScopeOpened, WaitStarted, TorrentGraduated, WaitProgress]

    def test_close_finishes_with_the_tally_then_closes_the_scope(self) -> None:
        recording = RecordingHub()
        install_hub(recording.hub)
        view = self._view()
        view.update(
            WaitSnapshot(
                (
                    _terminal("h1", "A", Outcome.IMPORTED),
                    _terminal("h2", "B", Outcome.DOWNLOAD_TIMED_OUT),
                    _terminal("h3", "C", Outcome.DOWNLOAD_ERRORED),
                ),
                elapsed_s=200,
            ),
        )

        view.close()

        (opened,) = recording.of_type(ScopeOpened)
        assert recording.of_type(WaitFinished) == [
            WaitFinished(imported=1, deferred=1, failed=1, elapsed_s=200.0, scope=opened.scope),
        ]
        assert [type(e) for e in recording.events[-2:]] == [WaitFinished, ScopeClosed]
        (closed,) = recording.of_type(ScopeClosed)
        assert closed.scope == opened.scope

        emitted = len(recording.events)
        view.close()  # idempotent: a second close emits nothing
        assert len(recording.events) == emitted

    def test_zero_update_close_emits_nothing(self) -> None:
        # Never updated -> the region never opened, so there is nothing to close.
        recording = RecordingHub()
        install_hub(recording.hub)

        self._view().close()

        assert recording.events == []

    def test_zero_tally_close_still_emits_the_finish(self) -> None:
        # The renderers show nothing for an empty tally (the builders return []),
        # but the event MUST flow: the scope needs its close on every surface.
        recording = RecordingHub()
        install_hub(recording.hub)
        view = self._view()
        view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=42))

        view.close()

        (opened,) = recording.of_type(ScopeOpened)
        assert recording.of_type(WaitFinished) == [
            WaitFinished(imported=0, deferred=0, failed=0, elapsed_s=42.0, scope=opened.scope),
        ]
        assert len(recording.of_type(ScopeClosed)) == 1

    def test_finish_elapsed_is_the_last_snapshots(self) -> None:
        # The narrator has no clock: elapsed rides the snapshots, so the tally
        # reports the LAST pushed elapsed_s (the Ctrl-C partial close relies on it).
        recording = RecordingHub()
        install_hub(recording.hub)
        view = self._view()
        view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=30))
        view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=95))

        view.close()

        (finished,) = recording.of_type(WaitFinished)
        assert finished.elapsed_s == 95.0

    def test_interrupted_finish_still_closes_the_scope(self) -> None:
        # A KeyboardInterrupt inside finish's synchronous dispatch propagates
        # (never swallowed), but the placement scope still closes on the way out
        # (the finally), so WAIT_REGION can't leak open behind the unwind.
        recording = RecordingHub()
        install_hub(recording.hub)
        real_emit = recording.hub.emit

        def interrupt_on_finish(event: Event) -> None:
            if isinstance(event, WaitFinished):
                raise KeyboardInterrupt
            real_emit(event)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("seadexarr.modules.wait_view.emit_to_hub", interrupt_on_finish)
            view = self._view()
            view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=5))
            with pytest.raises(KeyboardInterrupt):
                view.close()

        assert recording.of_type(WaitFinished) == []
        assert len(recording.of_type(ScopeClosed)) == 1  # the finally's fallback close

    def test_update_after_close_emits_nothing(self) -> None:
        # Defensive: the engine never updates a closed view, but a bug there must
        # not re-open the scope or emit past the finish.
        recording = RecordingHub()
        install_hub(recording.hub)
        view = self._view()
        view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=5))
        view.close()
        emitted = len(recording.events)

        view.update(WaitSnapshot((_terminal("h1", "A", Outcome.IMPORTED),), elapsed_s=60))

        assert len(recording.events) == emitted


# --- no-throw contract ---------------------------------------------------------


class _FlakyEmit:
    """An emit stand-in that starts raising after ``allow`` successful events."""

    def __init__(self, allow: int) -> None:
        self.calls = 0
        self.allow = allow

    def __call__(self, event: Event) -> None:
        self.calls += 1
        if self.calls > self.allow:
            raise RuntimeError("hub boom")


def test_narrator_is_total_when_emission_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # An emission raise must degrade to a debug no-op, never propagate (which
    # would abort the engine's wait loop or the end-of-run cache save).
    monkeypatch.setattr("seadexarr.modules.wait_view.emit_to_hub", _FlakyEmit(allow=3))
    logger = _quiet_logger("wait-narrator-boom")
    logger.setLevel(logging.DEBUG)
    capture = CaptureHandler()
    logger.addHandler(capture)
    view = HubWaitView(logger, pulse_s=300.0, wants_telemetry=True)

    view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=5))  # open+start+progress: the 3 allowed
    view.update(WaitSnapshot((_downloading("h1", "A"),), elapsed_s=35))  # progress raises -> swallowed
    view.close()  # finish raises -> swallowed (the fallback scope close too)

    assert [record.levelno for record in capture.records] == [logging.DEBUG, logging.DEBUG]


# --- pure model helpers --------------------------------------------------------


def test_graduations_returns_only_unseen_terminals() -> None:
    # frozenset inputs also prove purity: graduations() can never mutate seen.
    snap = WaitSnapshot(
        (
            _terminal("h1", "A", Outcome.IMPORTED),
            _downloading("h2", "B", 0.3),
            _terminal("h3", "C", Outcome.DOWNLOAD_ERRORED),
        ),
    )

    assert [t.key for t in graduations(frozenset(), snap)] == ["h1", "h3"]  # snapshot order
    assert [t.key for t in graduations(frozenset({"h1"}), snap)] == ["h3"]
    assert graduations(frozenset({"h1", "h3"}), snap) == []


# --- WaitResult ----------------------------------------------------------------


def test_wait_result_counts_by_category() -> None:
    result = WaitResult(
        (
            WaitOutcomeRow("A", Outcome.IMPORTED),
            WaitOutcomeRow("B", Outcome.IMPORTED),
            WaitOutcomeRow("C", Outcome.DOWNLOAD_TIMED_OUT),
            WaitOutcomeRow("D", Outcome.DOWNLOAD_ERRORED),
        ),
        elapsed_s=600,
    )

    assert result.waited == 4
    assert result.imported == 2
    assert result.left == 1
    assert result.failed == 1


def test_outcome_dropped_is_exactly_imported_and_missing() -> None:
    # Pins the word<->drop parity the engine relies on (outcome.dropped drives
    # _drop_pending), so a displayed word can never diverge from the store mutation.
    assert {o for o in Outcome if o.dropped} == {Outcome.IMPORTED, Outcome.MISSING}


def test_outcome_glyph_falls_back_to_ascii() -> None:
    assert Outcome.IMPORTED.glyph(use_unicode=True) == "✔"
    assert Outcome.IMPORTED.glyph(use_unicode=False) == "ok"
    assert Outcome.IMPORTED.category is OutcomeCategory.SUCCESS
