# pyright: strict
"""Tests for the logging bridge (``output.bridge``) + the LegacyRenderer echo.

Pin the adoption table (first-party plain WARNING+ only; third-party WARNING+
always plus sub-hub-level DEBUG/INFO early-outed; payload/HUB_EVENT skips),
construct-never-mutate (caplog safety), warnings capture, idempotent install
surviving ``setup_logger`` rebuilds, the per-thread reentrancy downgrade, and
the loop/double-render seams: one console render + one file line per record,
and a LegacyRenderer re-emission is never re-adopted. The motivating scenario -
a plain WARNING fired between boot steps - is pinned end-to-end through the
real boot flow, bridge, hub and renderer.
"""

import io
import logging
import sys
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

from seadexarr.modules.boot_flow import BootFlow
from seadexarr.modules.log import (
    CONSOLE_EXTRA,
    HUB_EVENT,
    LOG_NAME,
    KvLine,
    StyledLine,
    hub_event_marked,
    log_counter,
    setup_logger,
)
from seadexarr.modules.output import (
    Diagnostic,
    Event,
    HubBridgeHandler,
    LegacyRenderer,
    OutputHub,
    RichRenderer,
    Severity,
    attributed_message,
    install_bridge,
    install_hub,
    is_first_party,
    uninstall_bridge,
    uninstall_hub,
)
from seadexarr.modules.output.recording import RecordingRenderer

from .fakes import CaptureHandler, TtyStringIO, strip_ansi


@pytest.fixture
def bridged(app_logger: logging.Logger) -> Generator[tuple[RecordingRenderer, logging.Logger]]:
    """A recording hub bridged to the real root + app loggers, torn down after."""

    recorder = RecordingRenderer()
    install_bridge(OutputHub([recorder]))
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

    def test_first_party_info_and_debug_stay_legacy(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        recorder, logger = bridged

        logger.info("checking Frieren")
        logger.debug("cache hit")

        assert recorder.of_type(Diagnostic) == []

    def test_first_party_payload_records_stay_legacy(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        """Even WARNING+ payload records (kv/styled lines) keep their legacy render."""

        recorder, logger = bridged

        logger.warning("missing S01E03", extra={CONSOLE_EXTRA: KvLine(key="missing", value="S01E03", key_width=9)})
        logger.error("dim note", extra={CONSOLE_EXTRA: StyledLine(style="grey50")})

        assert recorder.of_type(Diagnostic) == []

    def test_first_party_payload_records_tally_count_only(self, app_logger: logging.Logger) -> None:
        """The capstone gap: a WARNING+ payload record stays legacy-rendered but
        its severity still reaches the hub tally — with ZERO events dispatched."""

        recorder = RecordingRenderer()
        hub = OutputHub([recorder])
        install_bridge(hub)
        mark = hub.counts.mark()

        try:
            app_logger.error("dim note", extra={CONSOLE_EXTRA: StyledLine(style="grey50")})
            app_logger.warning(
                "missing S01E03", extra={CONSOLE_EXTRA: KvLine(key="missing", value="S01E03", key_width=9)}
            )
            app_logger.info("plain kv", extra={CONSOLE_EXTRA: KvLine(key="cache", value="hit", key_width=9)})
        finally:
            uninstall_bridge()

        since = hub.counts.counts_since(mark)
        assert (since.errors, since.warnings, since.info) == (1, 1, 0)
        assert recorder.events == []  # count-only: nothing constructed or dispatched

    def test_hub_event_marked_records_are_dropped(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        recorder, logger = bridged

        logger.warning("httpx: flaky pool", extra={HUB_EVENT: True})

        assert recorder.of_type(Diagnostic) == []

    def test_third_party_adopts_all_levels_with_origin(self, app_logger: logging.Logger) -> None:
        """At a configured DEBUG the early-out never fires: every level is adopted (S4)."""

        del app_logger
        recorder = RecordingRenderer()
        hub = OutputHub([recorder])
        hub.set_level(logging.DEBUG)
        install_bridge(hub)
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

    def test_sub_threshold_third_party_records_are_never_constructed(
        self, bridged: tuple[RecordingRenderer, logging.Logger]
    ) -> None:
        """The early-out: below WARNING and below the hub's level (INFO here), a
        third-party record is dropped before adoption - not constructed, not
        counted (pre-PR2 parity). INFO itself still adopts (the file admits it)."""

        recorder, _logger = bridged
        library = _third_party("bridge-test-quiet")

        library.debug("handshake bytes")
        library.info("handshake ok")

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.severity is Severity.INFO
        assert diagnostic.message == "handshake ok"

    def test_exc_info_becomes_a_captured_trace(self, bridged: tuple[RecordingRenderer, logging.Logger]) -> None:
        recorder, logger = bridged

        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("sync failed", exc_info=True)

        (diagnostic,) = recorder.of_type(Diagnostic)
        assert diagnostic.trace is not None
        assert "ValueError: boom" in diagnostic.trace.plain_text()

    def test_adoption_never_mutates_the_record(self) -> None:
        """caplog safety: the bridge constructs a new event and hands back the
        record byte-identical - no attribute added, consumed or flattened."""

        recorder = RecordingRenderer()
        bridge = HubBridgeHandler(OutputHub([recorder]))
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
        adopted file-only: no frontier placement mid-fold (S5 pin 4 / N2)."""

        recorder = RecordingRenderer()
        install_bridge(OutputHub([recorder, _MidDispatchLogger("bridge-test-reentrant")]))
        try:
            app_logger.warning("outer")
        finally:
            uninstall_bridge()

        outer, inner = recorder.of_type(Diagnostic)
        assert outer.message == "outer"
        assert not outer.file_only
        assert inner.origin == "bridge-test-reentrant"
        assert inner.file_only


class _MidDispatchLogger:
    """A renderer that logs a third-party record from inside handle (once)."""

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


# --- install lifecycle ----------------------------------------------------------------


class TestInstall:
    def test_install_is_idempotent(self, app_logger: logging.Logger) -> None:
        hub = OutputHub([RecordingRenderer()])

        first = install_bridge(hub)
        second = install_bridge(hub)
        try:
            root_bridges = [h for h in logging.getLogger().handlers if isinstance(h, HubBridgeHandler)]
            app_bridges = [h for h in app_logger.handlers if isinstance(h, HubBridgeHandler)]
            assert root_bridges == [second]
            assert app_bridges == [second]
            assert first is not second  # replaced, never doubled
        finally:
            uninstall_bridge()

    def test_setup_logger_rebuilds_preserve_the_bridge(
        self,
        app_logger: logging.Logger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scheduled mode re-runs setup_logger per cycle; the bridge must survive
        the handler teardown with its identity intact (installed once, S3)."""

        bridge = install_bridge(OutputHub([RecordingRenderer()]))
        try:
            monkeypatch.setattr(sys, "stdout", io.StringIO())
            for _ in range(2):
                logger = setup_logger(log_level="INFO", log_dir=str(tmp_path / "logs"), console_format="plain")
                bridges = [h for h in logger.handlers if isinstance(h, HubBridgeHandler)]
                assert bridges == [bridge]
        finally:
            uninstall_bridge()

    def test_uninstall_detaches_from_both_loggers(self, app_logger: logging.Logger) -> None:
        install_bridge(OutputHub([RecordingRenderer()]))

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


# --- the strangler seams: once on console, once in file, never a loop -----------------


def _console_text(stream: io.StringIO) -> str:
    return strip_ansi(stream.getvalue())


def _file_text(log_file: Path, logger: logging.Logger) -> str:
    for handler in logger.handlers:
        handler.flush()
    return log_file.read_text(encoding="utf-8")


@pytest.fixture
def full_stack(
    app_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[logging.Logger, TtyStringIO, Path]]:
    """The real PR2 wiring: setup_logger (rich console over a fake TTY) plus a
    hub carrying the LegacyRenderer echo and the RichRenderer console seat."""

    del app_logger  # isolation + teardown ordering only
    stream = TtyStringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    logger = setup_logger(log_level="INFO", log_dir=str(tmp_path / "logs"), console_format="rich")
    hub = OutputHub([LegacyRenderer()], console=RichRenderer())
    install_hub(hub)
    install_bridge(hub)
    yield logger, stream, tmp_path / "logs" / "SeaDexArr.log"
    uninstall_bridge()
    uninstall_hub()


class TestStranglerSeam:
    def test_first_party_warning_renders_once_on_console_once_in_file(
        self, full_stack: tuple[logging.Logger, TtyStringIO, Path]
    ) -> None:
        logger, stream, log_file = full_stack

        logger.warning("watch out")

        console = _console_text(stream)
        assert console.count("watch out") == 1
        assert "WARNING  watch out" in console  # the hub renderer's badge line
        assert _file_text(log_file, logger).count("watch out") == 1  # legacy file path

    def test_third_party_warning_renders_once_on_console_once_in_file(
        self, full_stack: tuple[logging.Logger, TtyStringIO, Path]
    ) -> None:
        logger, stream, log_file = full_stack
        library = _third_party("bridge-test-pool")

        library.warning("flaky pool")

        console = _console_text(stream)
        assert console.count("flaky pool") == 1
        assert "WARNING  bridge-test-pool: flaky pool" in console
        # The LegacyRenderer echo is the file path - and LogCounter now counts
        # third-party stragglers (N1, the deliberate tally fix).
        assert _file_text(log_file, logger).count("bridge-test-pool: flaky pool") == 1
        assert log_counter(logger).counts[logging.WARNING] == 1

    def test_third_party_info_reaches_the_file_only(self, full_stack: tuple[logging.Logger, TtyStringIO, Path]) -> None:
        logger, stream, log_file = full_stack
        library = _third_party("bridge-test-chatty")

        library.info("handshake ok")

        assert "handshake ok" not in _console_text(stream)  # floored (S4)
        assert _file_text(log_file, logger).count("handshake ok") == 1

    def test_hub_event_records_never_re_adopt(self, full_stack: tuple[logging.Logger, TtyStringIO, Path]) -> None:
        """The loop pin: bridge -> hub -> LegacyRenderer -> logger -> bridge ends
        at the HUB_EVENT mark; the echo lands in the file exactly once."""

        logger, _stream, log_file = full_stack
        capture = CaptureHandler()
        logger.addHandler(capture)

        _third_party("bridge-test-loop").warning("around we go")

        echoes = [r for r in capture.records if hub_event_marked(r)]
        assert len(echoes) == 1
        assert _file_text(log_file, logger).count("around we go") == 1

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
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two setup_logger + begin_cycle rounds (scheduled mode): the bridge stays
        singly-installed and the renderer resolves the CURRENT cycle's console."""

        del app_logger
        stream_one = TtyStringIO()
        monkeypatch.setattr(sys, "stdout", stream_one)
        logger = setup_logger(log_level="INFO", log_dir=str(tmp_path / "logs"), console_format="rich")
        hub = OutputHub([LegacyRenderer()], console=RichRenderer())
        install_hub(hub)
        bridge = install_bridge(hub)
        try:
            hub.begin_cycle(console_format="rich", level=logging.INFO)

            stream_two = TtyStringIO()
            monkeypatch.setattr(sys, "stdout", stream_two)
            logger = setup_logger(log_level="INFO", log_dir=str(tmp_path / "logs"), console_format="rich")
            hub.begin_cycle(console_format="rich", level=logging.INFO)

            logger.warning("second cycle")

            assert [h for h in logger.handlers if isinstance(h, HubBridgeHandler)] == [bridge]
            assert "second cycle" not in _console_text(stream_one)
            assert "WARNING  second cycle" in _console_text(stream_two)
        finally:
            uninstall_bridge()
            uninstall_hub()


# --- the file_only echo: forensics reach the file, never stdout or the tally ----------


@pytest.fixture
def plain_stack(
    app_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[OutputHub, logging.Logger, io.StringIO, Path]]:
    """The PR2 wiring under plain format: the LegacyRenderer echo carries both
    the stdout and file surfaces (no rich console)."""

    del app_logger  # isolation + teardown ordering only
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    logger = setup_logger(log_level="INFO", log_dir=str(tmp_path / "logs"), console_format="plain")
    hub = OutputHub([LegacyRenderer()])
    install_hub(hub)
    install_bridge(hub)
    yield hub, logger, stream, tmp_path / "logs" / "SeaDexArr.log"
    uninstall_bridge()
    uninstall_hub()


class TestFileOnlyEcho:
    """A file_only diagnostic still echoes (pre-PR6 the legacy FileHandler is the
    ONLY persistence - a full skip would lose containment forensics), but the
    ``HUB_FILE_ONLY`` mark keeps it off stdout and out of the LogCounter tally."""

    def test_containment_note_reaches_the_file_but_not_stdout_or_the_tally(
        self, plain_stack: tuple[OutputHub, logging.Logger, io.StringIO, Path]
    ) -> None:
        hub, logger, stream, log_file = plain_stack

        hub.emit(
            Diagnostic(severity=Severity.WARNING, message="renderer struck out", origin="output.hub", file_only=True),
        )

        assert "output.hub: renderer struck out" in _file_text(log_file, logger)
        assert "renderer struck out" not in stream.getvalue()
        assert log_counter(logger).counts.get(logging.WARNING, 0) == 0

    def test_a_normal_third_party_echo_still_prints_and_counts(
        self, plain_stack: tuple[OutputHub, logging.Logger, io.StringIO, Path]
    ) -> None:
        """The N1 pin holds: only HUB_FILE_ONLY records skip stdout + the tally."""

        _hub, logger, stream, log_file = plain_stack

        _third_party("bridge-test-plain-pool").warning("flaky pool")

        assert "bridge-test-plain-pool: flaky pool" in stream.getvalue()
        assert _file_text(log_file, logger).count("bridge-test-plain-pool: flaky pool") == 1
        assert log_counter(logger).counts[logging.WARNING] == 1
