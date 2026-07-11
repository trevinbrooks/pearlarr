# pyright: strict
"""Tests for the logging bridge (`output.bridge`) + the real-seat wiring.

Pin the adoption table (WARNING+ visible; first-party sub-WARNING file_only at
INFO+ config or under a rich seat, visible at DEBUG config on plain/json;
third-party sub-WARNING gated by the hub level, then file_only unless
configured DEBUG), the root-logger gate `setup_logger` opens for the bridge,
construct-never-mutate (caplog safety), warnings capture,
idempotent install surviving `setup_logger` rebuilds, the per-thread
reentrancy downgrade, and the full-stack seams over the REAL sinks: one console
render + one structured file line per record. The motivating scenario - a
plain WARNING fired between boot steps - is pinned end-to-end through the real
boot flow, bridge, hub and renderer.
"""

import logging
import sys
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import pytest

from pearlarr.modules.boot_flow import BootFlow
from pearlarr.modules.log import LOG_NAME, setup_logger
from pearlarr.modules.output import (
    Diagnostic,
    Event,
    FileLogSink,
    HubBridgeHandler,
    NullRenderer,
    OutputHub,
    RichRenderer,
    Severity,
    attributed_message,
    current_hub,
    install_bridge,
    install_hub,
    is_first_party,
    uninstall_bridge,
    uninstall_hub,
)
from pearlarr.modules.output.recording import RecordingRenderer

from .fakes import TtyStringIO, strip_ansi


@pytest.fixture
def bridged(app_logger: logging.Logger) -> Generator[tuple[RecordingRenderer, logging.Logger]]:
    """A recording hub bridged to the real root + app loggers, torn down after."""

    recorder = RecordingRenderer()
    install_hub(OutputHub([recorder]))
    install_bridge()
    yield recorder, app_logger
    uninstall_bridge()


def _third_party(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """A propagating stand-in for a library logger (fresh handlers, own level)."""

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = True
    return logger


# --- the adoption table -------------------------------------------------------------


class TestAdoption:
    def test_first_party_plain_warning_is_adopted(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        recorder, logger = bridged

        logger.warning("watch out")

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic == Diagnostic(severity=Severity.WARNING, message="watch out", origin=LOG_NAME)

    def test_first_party_sub_warning_adopts_file_only_at_info_config(
        self, bridged: tuple[RecordingRenderer, logging.Logger]
    ) -> None:
        """P3: at INFO+ config (the bridged hub's default) DEBUG chatter +
        unmigrated INFO stragglers stay in the FILE; stdout omits them."""

        recorder, logger = bridged

        logger.info("checking Frieren")
        logger.debug("cache hit")

        assert [(d.severity, d.file_only) for d in recorder.of_type(Diagnostic)] == [
            (Severity.INFO, True),
            (Severity.DEBUG, True),
        ]

    def test_first_party_sub_warning_visible_at_debug_config_on_a_plain_seat(self, app_logger: logging.Logger) -> None:
        """At DEBUG config with a plain/json seat the bridge is a first-party
        record's ONLY console route, so DEBUG/INFO adopt visible."""

        recorder = RecordingRenderer()
        hub = OutputHub([recorder], console_factory=lambda console_format: NullRenderer())
        hub.begin_cycle(console_format="plain", level=logging.DEBUG)
        install_hub(hub)
        install_bridge()

        try:
            app_logger.debug("cache hit")
            app_logger.info("checking Frieren")
        finally:
            uninstall_bridge()

        assert [(d.severity, d.file_only) for d in recorder.of_type(Diagnostic)] == [
            (Severity.DEBUG, False),
            (Severity.INFO, False),
        ]

    def test_first_party_sub_warning_stays_file_only_at_debug_config_under_rich(
        self, app_logger: logging.Logger
    ) -> None:
        """Under a rich seat RichConsoleHandler already prints the raw record;
        visible adoption would double it, so file_only stands even at DEBUG."""

        recorder = RecordingRenderer()
        hub = OutputHub([recorder], console_factory=lambda console_format: NullRenderer())
        hub.begin_cycle(console_format="rich", level=logging.DEBUG)
        install_hub(hub)
        install_bridge()

        try:
            app_logger.debug("cache hit")
        finally:
            uninstall_bridge()

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.file_only

    def test_third_party_adopts_all_levels_visible_at_configured_debug(self, app_logger: logging.Logger) -> None:
        """At a configured DEBUG the early-out never fires and nothing is demoted
        to file_only: the hub is a library record's only console route."""

        del app_logger
        recorder = RecordingRenderer()
        hub = OutputHub([recorder])
        hub.set_level(logging.DEBUG)
        install_hub(hub)
        install_bridge()
        library = _third_party("bridge-test-httpx")

        try:
            library.debug("d")
            library.info("i")
            library.warning("w")
            library.error("e")
            library.critical("c")
        finally:
            uninstall_bridge()

        diagnostics = recorder.of_type(Diagnostic)
        assert [d.severity for d in diagnostics] == [
            Severity.DEBUG,
            Severity.INFO,
            Severity.WARNING,
            Severity.ERROR,
            Severity.CRITICAL,
        ]
        assert {d.origin for d in diagnostics} == {"bridge-test-httpx"}
        assert all(not d.file_only for d in diagnostics)

    def test_sub_threshold_third_party_records_are_never_constructed(
        self, bridged: tuple[RecordingRenderer, logging.Logger]
    ) -> None:
        """The early-out: below WARNING and below the hub's level (INFO here), a
        third-party record is dropped before adoption - not constructed, not
        counted. INFO itself adopts file_only (the file keeps the forensics;
        stdout loses library chatter at INFO+)."""

        recorder, _logger = bridged
        library = _third_party("bridge-test-quiet")

        library.debug("handshake bytes")
        library.info("handshake ok")

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.severity is Severity.INFO
        assert diagnostic.message == "handshake ok"
        assert diagnostic.file_only

    def test_third_party_warning_adopts_visible(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        recorder, _logger = bridged

        _third_party("bridge-test-loud").warning("flaky pool")

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.severity is Severity.WARNING
        assert not diagnostic.file_only

    def test_exc_info_becomes_a_captured_trace(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        recorder, logger = bridged

        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("sync failed", exc_info=True)

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.trace is not None
        assert "ValueError: boom" in diagnostic.trace.plain

    def test_adoption_never_mutates_the_record(self) -> None:
        """caplog safety: the bridge constructs a new event and hands back the
        record byte-identical - no attribute added, consumed or flattened."""

        recorder = RecordingRenderer()
        install_hub(OutputHub([recorder]))
        bridge = HubBridgeHandler()
        record = logging.LogRecord("bridge-test-mutation", logging.WARNING, __file__, 1, "flaky %s", ("pool",), None)
        snapshot = dict(record.__dict__)

        bridge.handle(record)

        assert record.__dict__ == snapshot
        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.message == "flaky pool"

    def test_capture_warnings_adopts_the_warnings_module(
        self, bridged: tuple[RecordingRenderer, logging.Logger]
    ) -> None:
        recorder, _logger = bridged

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("dusty corner", UserWarning, stacklevel=1)

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.origin == "py.warnings"
        assert "dusty corner" in diagnostic.message

    def test_reentrant_records_downgrade_to_file_only(self, app_logger: logging.Logger) -> None:
        """A record fired from inside hub dispatch (renderer/SIGTERM logging) is
        adopted file-only: no frontier placement mid-fold."""

        recorder = RecordingRenderer()
        install_hub(OutputHub([recorder, _MidDispatchLogger("bridge-test-reentrant")]))
        install_bridge()
        try:
            app_logger.warning("outer")
        finally:
            uninstall_bridge()

        outer, inner = recorder.of_type(Diagnostic)
        assert outer.message == "outer"
        assert not outer.file_only
        assert inner.origin == "bridge-test-reentrant"
        assert inner.file_only

    def test_reentrant_records_downgrade_on_a_producer_entered_drain(self, app_logger: logging.Logger) -> None:
        """N2 holds for EVERY drain: a renderer logging while a DIRECT producer
        emit (reporter/hub_note — the majority path) drives dispatch is adopted
        file-only too. The hub's own baton read decides, not bridge-private
        state that only its own emits used to set."""

        del app_logger
        recorder = RecordingRenderer()
        hub = OutputHub([recorder, _MidDispatchLogger("bridge-test-producer-drain")])
        install_hub(hub)
        install_bridge()
        try:
            hub.emit(Diagnostic(severity=Severity.WARNING, message="outer", origin=LOG_NAME))
        finally:
            uninstall_bridge()

        outer, inner = recorder.of_type(Diagnostic)
        assert outer.message == "outer"
        assert inner.origin == "bridge-test-producer-drain"
        assert inner.file_only

    def test_a_hub_swap_repoints_the_bridge_without_a_reinstall(self, app_logger: logging.Logger) -> None:
        """The bridge resolves the hub through the registry per record: an
        install_hub swap must never orphan it onto the closed old hub (silent
        loss of every third-party WARNING/ERROR)."""

        del app_logger
        first, second = RecordingRenderer(), RecordingRenderer()
        install_hub(OutputHub([first]))
        install_bridge()
        try:
            install_hub(OutputHub([second]))  # closes the first hub; NO re-install
            _third_party("bridge-test-swap").warning("flaky pool")
        finally:
            uninstall_bridge()

        assert first.of_type(Diagnostic) == []
        (diagnostic,) = second.of_type(Diagnostic)
        assert diagnostic.message == "flaky pool"


class _MidDispatchLogger:
    """A renderer that logs a third-party record from inside handle (once)."""

    writes_file_only: ClassVar[bool] = False

    def __init__(self, logger_name: str) -> None:
        self._name = logger_name
        self._fired = False

    def handle(self, event: Event, when: float) -> None:
        del when
        if isinstance(event, Diagnostic) and not event.file_only and not self._fired:
            self._fired = True
            logging.getLogger(self._name).warning("from inside dispatch")

    def begin_cycle(self) -> None:
        pass

    def set_level(self, level: int) -> None:
        pass

    def close(self) -> None:
        pass


# --- the root-logger gate (setup_logger opens it for the bridge) -----------------------


class TestRootGate:
    """A self-silent (NOTSET) library logger defers to the ROOT level; stdlib's
    WARNING default would drop its sub-WARNING records before the bridge runs."""

    def test_third_party_info_reaches_the_bridge_at_info_config(self, app_logger: logging.Logger) -> None:
        del app_logger
        recorder = RecordingRenderer()
        install_hub(OutputHub([recorder]))
        install_bridge()
        setup_logger(log_level="INFO", console_format="plain")
        library = _third_party("bridge-test-rootgate-info", level=logging.NOTSET)

        try:
            library.debug("handshake bytes")  # below root INFO: never constructed
            library.info("handshake ok")
        finally:
            uninstall_bridge()

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.severity is Severity.INFO
        assert diagnostic.message == "handshake ok"
        assert diagnostic.file_only

    def test_third_party_debug_reaches_the_bridge_visible_at_debug_config(self, app_logger: logging.Logger) -> None:
        del app_logger
        recorder = RecordingRenderer()
        hub = OutputHub([recorder])
        hub.set_level(logging.DEBUG)
        install_hub(hub)
        install_bridge()
        setup_logger(log_level="DEBUG", console_format="plain")
        library = _third_party("bridge-test-rootgate-debug", level=logging.NOTSET)

        try:
            library.debug("handshake bytes")
        finally:
            uninstall_bridge()

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.severity is Severity.DEBUG
        assert not diagnostic.file_only


# --- install lifecycle ----------------------------------------------------------------


class TestInstall:
    def test_install_is_idempotent(self, app_logger: logging.Logger) -> None:
        first = install_bridge()
        second = install_bridge()
        try:
            root_bridges = [h for h in logging.getLogger().handlers if isinstance(h, HubBridgeHandler)]
            app_bridges = [h for h in app_logger.handlers if isinstance(h, HubBridgeHandler)]
            assert root_bridges == [second]
            assert app_bridges == [second]
            assert first is not second  # replaced, never doubled
        finally:
            uninstall_bridge()

    def test_setup_logger_rebuilds_preserve_the_bridge(self, app_logger: logging.Logger) -> None:
        """Scheduled mode re-runs setup_logger per cycle; the bridge must survive
        the handler teardown with its identity intact (installed once, S3)."""

        del app_logger
        bridge = install_bridge()
        try:
            for _ in range(2):
                logger = setup_logger(log_level="INFO", console_format="plain")
                bridges = [h for h in logger.handlers if isinstance(h, HubBridgeHandler)]
                assert bridges == [bridge]
        finally:
            uninstall_bridge()

    def test_uninstall_detaches_from_both_loggers(self, app_logger: logging.Logger) -> None:
        install_bridge()

        uninstall_bridge()

        assert not any(isinstance(h, HubBridgeHandler) for h in logging.getLogger().handlers)
        assert not any(isinstance(h, HubBridgeHandler) for h in app_logger.handlers)


# --- origin helpers ---------------------------------------------------------------


class TestOriginHelpers:
    def test_first_party_is_the_app_logger_and_its_children(self) -> None:
        assert is_first_party(LOG_NAME)
        assert is_first_party(f"{LOG_NAME}.child")
        assert not is_first_party("httpx")
        assert not is_first_party(f"{LOG_NAME}x")  # a prefix, not a child

    def test_third_party_messages_carry_their_origin(self) -> None:
        ours = Diagnostic(severity=Severity.WARNING, message="m", origin=LOG_NAME)
        theirs = Diagnostic(severity=Severity.WARNING, message="m", origin="httpx")
        assert attributed_message(ours) == "m"
        assert attributed_message(theirs) == "httpx: m"


# --- the full-stack seams: once on console, once in the structured file ----------------


def _console_text(stream: TtyStringIO) -> str:
    return strip_ansi(stream.getvalue())


def _file_text(log_file: Path) -> str:
    # The FileLogSink flushes per line, so the file is current after every emit.
    return log_file.read_text(encoding="utf-8")


@pytest.fixture
def full_stack(
    app_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[logging.Logger, TtyStringIO, Path]]:
    """The real production wiring under rich: hub (FileLogSink first) + bridge
    installed BEFORE setup_logger builds the rich console over a fake TTY."""

    del app_logger  # isolation + teardown ordering only
    stream = TtyStringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    hub = OutputHub([FileLogSink(str(tmp_path / "logs"))], console=RichRenderer())
    install_hub(hub)
    install_bridge()
    logger = setup_logger(log_level="INFO", console_format="rich")
    yield logger, stream, tmp_path / "logs" / "Pearlarr.log"
    uninstall_bridge()
    uninstall_hub()


class TestFullStackSeams:
    def test_first_party_warning_renders_once_on_console_once_in_file(
        self, full_stack: tuple[logging.Logger, TtyStringIO, Path]
    ) -> None:
        logger, stream, log_file = full_stack

        logger.warning("watch out")

        console = _console_text(stream)
        assert console.count("watch out") == 1
        assert "WARNING  watch out" in console  # the hub renderer's badge line
        file_text = _file_text(log_file)
        assert file_text.count("watch out") == 1  # the FileLogSink's grammar line
        assert "WARNING [Pearlarr] watch out" in file_text

    def test_first_party_info_reaches_the_file_and_the_raw_tty_once_each(
        self, full_stack: tuple[logging.Logger, TtyStringIO, Path]
    ) -> None:
        """P3: an unmigrated INFO one-liner adopts file_only — the FileLogSink
        keeps it while the rich TTY shows the RAW record (no double render)."""

        logger, stream, log_file = full_stack

        logger.info("checking Frieren")

        assert _console_text(stream).count("checking Frieren") == 1
        assert _file_text(log_file).count("checking Frieren") == 1

    def test_third_party_warning_renders_once_on_console_once_in_file(
        self, full_stack: tuple[logging.Logger, TtyStringIO, Path]
    ) -> None:
        logger, stream, log_file = full_stack
        del logger
        library = _third_party("bridge-test-pool")

        library.warning("flaky pool")

        console = _console_text(stream)
        assert console.count("flaky pool") == 1
        assert "WARNING  bridge-test-pool: flaky pool" in console
        # The FileLogSink is the file path, and the hub counts third-party
        # stragglers (N1, the deliberate tally fix).
        assert _file_text(log_file).count("flaky pool") == 1
        assert current_hub().counts.mark().warning == 1

    def test_third_party_info_reaches_the_file_only(self, full_stack: tuple[logging.Logger, TtyStringIO, Path]) -> None:
        logger, stream, log_file = full_stack
        del logger
        library = _third_party("bridge-test-chatty")

        library.info("handshake ok")

        assert "handshake ok" not in _console_text(stream)  # file_only at INFO+
        assert _file_text(log_file).count("handshake ok") == 1

    def test_warning_between_boot_steps_renders_at_the_ledger_indent(
        self, full_stack: tuple[logging.Logger, TtyStringIO, Path]
    ) -> None:
        """THE motivating scenario: the chmod-style warning fired after a boot
        step closes lands under the boot ledger, not at column 0."""

        logger, stream, _log_file = full_stack
        boot = BootFlow()
        boot.banner()
        with boot.step("Reading config"):
            pass

        logger.warning("config is readable by other users")
        boot.end_section()
        logger.warning("after the cockpit")

        lines = _console_text(stream).splitlines()
        (mid_boot,) = [line for line in lines if "readable by other users" in line]
        (after,) = [line for line in lines if "after the cockpit" in line]
        assert mid_boot.startswith("  WARNING  ")
        assert after.startswith("WARNING  ")

    def test_dual_run_cycles_keep_one_bridge_and_the_fresh_console(
        self,
        app_logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two setup_logger + begin_cycle rounds (scheduled mode): the bridge stays
        singly-installed and the renderer resolves the CURRENT cycle's console."""

        del app_logger
        stream_one = TtyStringIO()
        monkeypatch.setattr(sys, "stdout", stream_one)
        hub = OutputHub([], console=RichRenderer())
        install_hub(hub)
        bridge = install_bridge()
        try:
            logger = setup_logger(log_level="INFO", console_format="rich")
            hub.begin_cycle(console_format="rich", level=logging.INFO)

            stream_two = TtyStringIO()
            monkeypatch.setattr(sys, "stdout", stream_two)
            logger = setup_logger(log_level="INFO", console_format="rich")
            hub.begin_cycle(console_format="rich", level=logging.INFO)

            logger.warning("second cycle")

            assert [h for h in logger.handlers if isinstance(h, HubBridgeHandler)] == [bridge]
            assert "second cycle" not in _console_text(stream_one)
            assert "WARNING  second cycle" in _console_text(stream_two)
        finally:
            uninstall_bridge()
            uninstall_hub()
