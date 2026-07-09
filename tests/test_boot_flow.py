# pyright: strict
"""Tests for the boot flow's producer facade (``boot_flow``).

``BootFlow`` is render-free: it opens/closes the boot-section scope mark, runs
each IO step as a :class:`~seadexarr.modules.output.StepScope`, and emits the
capstone when the section earned one. These pin the emitted event stream
(RecordingHub), the capstone gate (empty/failed sections and ERROR+ recorded on
the hub since the first step suppress it; a DEFERRED warn does not), the
section reset across arrs, the one-shot slow heads-up, the close() safety net,
and the production qBittorrent-unconfigured wiring through ``RunDeps.build``.
"""

import logging
from pathlib import Path

import httpx
import pytest

from seadexarr.modules.boot_flow import BootFlow
from seadexarr.modules.config import Arr
from seadexarr.modules.log import LOG_NAME
from seadexarr.modules.manual_import import OutcomeCategory
from seadexarr.modules.mappings import MappingResolver
from seadexarr.modules.output import (
    BootReady,
    BootStepFinished,
    BootStepSlow,
    BootStepStarted,
    Diagnostic,
    RunStarted,
    ScopeClosed,
    ScopeKind,
    ScopeOpened,
    Severity,
    install_bridge,
    install_hub,
)
from seadexarr.modules.output.recording import RecordingHub
from seadexarr.modules.run_services import RunDeps

from .builders import make_bare_instance, make_config
from .fakes import FakeClock


def _flow(data_dir: str = "/data/dir") -> tuple[BootFlow, RecordingHub, FakeClock]:
    # conftest's autouse teardown uninstalls the hub after every test.
    recording = RecordingHub()
    install_hub(recording.hub)
    clock = FakeClock()
    return BootFlow(data_dir, clock=clock), recording, clock


def _error(message: str) -> Diagnostic:
    return Diagnostic(severity=Severity.ERROR, message=message)


# --- banner + section scope ------------------------------------------------------


def test_banner_opens_the_section_and_states_the_run_facts() -> None:
    flow, recording, _ = _flow()

    flow.banner()

    (opened,) = recording.of_type(ScopeOpened)
    assert opened.scope.kind is ScopeKind.BOOT_SECTION
    (started,) = recording.of_type(RunStarted)
    assert started.data_dir == "/data/dir"
    assert started.version.startswith("v")  # the installed package version
    # RunStarted is the run boundary (the fold closes stale nodes on it), so it
    # precedes the section open.
    assert recording.events == [started, opened]


def test_a_step_after_end_section_opens_a_fresh_scope() -> None:
    # The second arr's boot steps reopen a section (bootstrap's per-arr loop).
    flow, recording, _ = _flow()

    flow.banner()
    flow.end_section()
    with flow.step("Fetching library"):
        pass
    flow.end_section()

    first, second = recording.of_type(ScopeOpened)
    assert first.scope != second.scope
    assert [c.scope for c in recording.of_type(ScopeClosed)] == [first.scope, second.scope]


def test_close_is_the_scope_safety_net() -> None:
    # bootstrap's finally-guarded teardown: a failed section still closes.
    flow, recording, _ = _flow()

    flow.banner()
    flow.close()
    flow.close()  # idempotent

    assert len(recording.of_type(ScopeOpened)) == 1
    assert len(recording.of_type(ScopeClosed)) == 1


# --- step lifecycle ---------------------------------------------------------------


def test_step_graduates_with_detail_and_timing() -> None:
    flow, recording, clock = _flow()

    with flow.step("Fetching Sonarr library") as step:
        clock.tick(1.2)
        step.note("42 series")

    (started,) = recording.of_type(BootStepStarted)
    (finished,) = recording.of_type(BootStepFinished)
    assert finished == BootStepFinished(
        scope=started.scope,
        label="Fetching Sonarr library",
        outcome=OutcomeCategory.SUCCESS,
        detail="42 series",
        elapsed_s=1.2,
    )


def test_step_failure_graduates_failed_and_reraises() -> None:
    flow, recording, clock = _flow()

    with pytest.raises(ValueError, match="bad config"):
        with flow.step("Reading config"):
            clock.tick(0.01)
            raise ValueError("bad config")
    flow.close()

    (finished,) = recording.of_type(BootStepFinished)
    assert finished.outcome is OutcomeCategory.FAILED
    assert recording.of_type(BootReady) == []  # no false "ready" after a failure


def test_slow_heads_up_is_emitted_once_per_step() -> None:
    # A slow step reports many progress ticks; the digest surfaces collapse them
    # to a single heads-up, so the one-shot lives producer-side (S6).
    flow, recording, clock = _flow()

    with flow.step("Refreshing mappings") as step:
        for frac in (0.25, 0.5, 0.75, 1.0):
            clock.tick(0.3)
            step.progress(frac, "anime_ids.json")

    assert [s.label for s in recording.of_type(BootStepSlow)] == ["Refreshing mappings"]


# --- the capstone gate ------------------------------------------------------------


def test_capstone_measures_from_the_first_step() -> None:
    flow, recording, clock = _flow()

    flow.banner()
    with flow.step("Reading config"):
        clock.tick(0.61)
    with flow.step("Opening cache"):
        clock.tick(0.61)
    flow.end_section()

    (ready,) = recording.of_type(BootReady)
    assert ready.elapsed_s == pytest.approx(1.22)


def test_error_recorded_mid_section_suppresses_the_capstone() -> None:
    # An ERROR that doesn't raise still means the section isn't "ready"; the
    # facade sees it through the hub's severity counts (the LogCounter gate died
    # with the view). Errors logged through the app logger reach these counts
    # via the PR2 bridge in production.
    flow, recording, clock = _flow()

    with flow.step("Reading config"):
        clock.tick(0.02)
    recording.emit(_error("no runnable arr selected"))
    flow.end_section()

    assert recording.of_type(BootReady) == []


def test_a_warning_does_not_gate_the_capstone() -> None:
    flow, recording, clock = _flow()

    with flow.step("Reading config"):
        clock.tick(0.02)
    recording.emit(Diagnostic(severity=Severity.WARNING, message="config perms loose"))
    flow.end_section()

    assert len(recording.of_type(BootReady)) == 1


def test_deferred_warn_still_allows_the_capstone() -> None:
    # A DEFERRED (warn) finish - qBittorrent unconfigured, a SeaDex outage - is a
    # degraded but READY section; only failures and recorded errors gate it.
    flow, recording, clock = _flow()

    with flow.step("Connecting to qBittorrent") as step:
        clock.tick(0.05)
        step.warn("not configured")
    flow.end_section()

    (finished,) = recording.of_type(BootStepFinished)
    assert finished.outcome is OutcomeCategory.DEFERRED
    assert len(recording.of_type(BootReady)) == 1


def test_error_before_the_first_step_does_not_gate() -> None:
    # Parity with the old view: the gate's window opens at the FIRST step (the
    # banner->step gap holds only import work), so earlier errors don't count.
    flow, recording, clock = _flow()

    flow.banner()
    recording.emit(_error("pre-step noise"))
    with flow.step("Reading config"):
        clock.tick(0.02)
    flow.end_section()

    assert len(recording.of_type(BootReady)) == 1


def test_error_gate_resets_with_the_section() -> None:
    # end_section resets the mark: an error during one arr's section must not
    # suppress the NEXT arr's capstone (multi-arr runs share the flow).
    flow, recording, clock = _flow()

    with flow.step("First step"):
        pass
    recording.emit(_error("first section error"))
    flow.end_section()
    with flow.step("Second step"):
        clock.tick(0.1)
    flow.end_section()

    assert len(recording.of_type(BootReady)) == 1  # second section only


def test_empty_section_emits_no_capstone() -> None:
    flow, recording, _ = _flow()

    flow.banner()
    flow.end_section()

    assert recording.of_type(BootReady) == []
    assert len(recording.of_type(ScopeClosed)) == 1  # the mark still closes


def test_logger_error_via_the_bridge_suppresses_the_capstone() -> None:
    # The full production chain: logger.error -> HubBridgeHandler -> hub counts ->
    # capstone gate. conftest's autouse teardown uninstalls the bridge + hub and
    # resets the app logger's handlers/level; propagate (flipped by install_bridge)
    # is restored here.
    flow, recording, clock = _flow()
    app_logger = logging.getLogger(LOG_NAME)
    propagate_before = app_logger.propagate
    install_bridge(recording.hub)
    try:
        with flow.step("Reading config"):
            clock.tick(0.02)
            app_logger.error("half-configured arr refused")
        flow.end_section()

        assert recording.of_type(BootReady) == []

        # Positive control: the same wiring without the error still graduates.
        with flow.step("Opening cache"):
            clock.tick(0.02)
        flow.end_section()

        assert len(recording.of_type(BootReady)) == 1
    finally:
        app_logger.propagate = propagate_before


# --- no-hub totality --------------------------------------------------------------


def test_without_an_installed_hub_the_flow_is_a_silent_no_op() -> None:
    # The renderer-less default hub drops emissions: library/standalone use of
    # RunDeps.build / run_sync needs no cockpit and no ceremony.
    flow = BootFlow()
    flow.banner()
    with flow.step("anything") as step:
        step.progress(0.5, "x")
        step.note("y")
        step.warn()
    flow.end_section()
    flow.close()


def test_no_hub_step_still_propagates_caller_errors() -> None:
    flow = BootFlow()
    with pytest.raises(RuntimeError, match="boom"):
        with flow.step("anything"):
            raise RuntimeError("boom")


# --- production wiring: the unconfigured-qBittorrent boot warning ------------------


def test_rundeps_build_warns_deferred_when_qbit_unconfigured(tmp_path: Path) -> None:
    # Missing qBittorrent credentials put the whole run in perpetual preview; the
    # boot ledger must say so via a DEFERRED step instead of silently skipping
    # the qBittorrent step.
    flow, recording, _ = _flow()

    deps = RunDeps.build(
        Arr.SONARR,
        cache=str(tmp_path / "cache.db"),
        logger=logging.getLogger("boot-flow-qbit"),
        mappings=make_bare_instance(MappingResolver),
        app_config=make_config(),  # no qbittorrent credentials
        web=httpx.Client(),
        boot=flow,
    )
    flow.close()
    deps.cache_store.close()  # don't leak the sqlite handle past the test

    assert deps.qbit is None
    [qbit_step] = [e for e in recording.of_type(BootStepFinished) if e.label == "Connecting to qBittorrent"]
    assert qbit_step.outcome is OutcomeCategory.DEFERRED
    assert qbit_step.detail == "not configured - preview mode"
