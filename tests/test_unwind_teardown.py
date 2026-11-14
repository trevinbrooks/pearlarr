# pyright: strict
"""The composition root's unwind teardown: close the arr run before reporting its death.

`run_arrs` wraps each arr's body in an INNER `except BaseException` that emits
`ArrRunFinished` and re-raises. Being inner, it runs during the unwind BEFORE the
outer `except` handlers log the arr-run-fatal error - so that error is a run-level fact
rendered at column 0, never a detail indented inside whatever title / item / boot
step the arr run happened to die in.

These pin the EMIT ORDERING (the half bootstrap owns). What the ordering buys - the
error actually landing at column 0 - is pinned against the real renderer in
`test_output_rich_renderer.TestUnwindPosition`. The boundary calls of the arr run's finish
are pinned in `test_import_wait.TestFinishArrRunOrdering` / `TestFinishArrRunUnwind`.
`ArrRunFinished` is emitted exactly once per arr run: the finish emits on success, and
bootstrap's inner handler emits only when the arr run died first. The open scopes' close
idempotency (`test_output_scope_path`) and the renderer no-op
(`TestUnwindPosition`) remain as defense in depth.

`ArrRunDeps.build` is the failure site: it is the earliest thing inside the inner
try, so it also exercises the `deps is None` path - no reporter exists yet, which
is why bootstrap must emit through the process hub rather than the reporter.
"""

import logging
import os
from pathlib import Path

import httpx
import pytest
import yaml

import pearlarr.bootstrap as bootstrap
from pearlarr.arr_run_services import ArrRunDeps
from pearlarr.boot_flow import BootFlow
from pearlarr.bootstrap import run_arrs
from pearlarr.cache import CacheSchemaError
from pearlarr.config import AppConfig, Arr, ArrTarget
from pearlarr.mappings import MappingResolver, MappingSources
from pearlarr.output import ArrRunFinished, Diagnostic, FileLogRenderer, Severity, install_bridge, install_hub
from pearlarr.paths import resolve_paths
from pearlarr.wait_types import ImportWaitMode

from .builders import make_config
from .fakes import RecordingHub

# The message each scripted failure carries, so the adopted ERROR is identifiable
# in the recorded stream (the clean handler logs `str(e)`, the traceback handler doesn't).
_SCHEMA_ERROR = "cache.db was written by a newer Pearlarr"


def _memory_resolver(
    app_config: AppConfig,
    mappings_db: str,
    logger: logging.Logger,
    boot: BootFlow,
    retry: str,
    web: httpx.Client,
) -> MappingResolver | None:
    """A network-free stand-in for `bootstrap.build_resolver` (no sources enabled)."""

    del app_config, mappings_db, boot, retry
    return MappingResolver(
        refresh_days=1,
        ignore_anilist_ids=set(),
        web=web,
        sources=MappingSources(anime={}, anidb=False, anibridge=False),
        logger=logger,
    )


def _failing_build(exc: Exception) -> object:
    """A `ArrRunDeps.build` replacement that raises `exc` with the real signature.

    Patched onto the class, so it is reached as a plain function (no `cls`). The
    real `build` is a classmethod and the call site passes `arr` positionally.
    """

    def build(
        arr: Arr,
        cache: str = "cache.db",
        *,
        logger: logging.Logger,
        resolver: MappingResolver,
        app_config: AppConfig,
        web: httpx.Client,
        boot: BootFlow,
    ) -> ArrRunDeps:
        del arr, cache, logger, resolver, app_config, web, boot
        raise exc

    return build


def _write_config() -> None:
    """A valid, tight-permissioned Sonarr config where `resolve_paths` looks for it."""

    paths = resolve_paths()
    os.makedirs(paths.data_dir)
    config = Path(paths.config_file)
    config.write_text(
        yaml.safe_dump(make_config(url="http://sonarr.test", api_key="k").model_dump(mode="json")),
        encoding="utf-8",
    )
    # 0600 keeps the loose-permissions warning out of the recorded stream.
    config.chmod(0o600)


def _run_failing_arr(
    monkeypatch: pytest.MonkeyPatch,
    app_logger: logging.Logger,
    exc: Exception,
) -> tuple[bool, RecordingHub]:
    """Drive one Sonarr run whose `ArrRunDeps.build` raises `exc`. Record the stream.

    The real hub AND the real logging bridge are installed (production parity), so
    the composition root's `hub_note` ERROR lands in the same event stream as the
    unwind's `ArrRunFinished` - which is the only way their relative order is observable.
    """

    _write_config()
    monkeypatch.setattr(bootstrap, "build_resolver", _memory_resolver)
    monkeypatch.setattr(ArrRunDeps, "build", _failing_build(exc))

    recording = RecordingHub()
    install_hub(recording.hub)
    install_bridge()

    paths = resolve_paths()
    completed = run_arrs(
        [ArrTarget(Arr.SONARR)], paths=paths, logger=app_logger, file_log_renderer=FileLogRenderer(paths.log_dir)
    )
    return completed, recording


def _arr_run_fatal_error(recording: RecordingHub) -> tuple[int, Diagnostic]:
    """Where the arr-run-fatal ERROR the except handlers logged ended up, and what it said."""

    return next(
        (i, event)
        for i, event in enumerate(recording.events)
        if isinstance(event, Diagnostic) and event.severity is Severity.ERROR
    )


class TestUnwindEmitsArrRunFinished:
    """Every arr-run-fatal path closes the run before its error is reported."""

    def test_clean_branch_closes_the_run_before_logging(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_logger: logging.Logger,
    ) -> None:
        completed, recording = _run_failing_arr(monkeypatch, app_logger, CacheSchemaError(_SCHEMA_ERROR))

        assert completed is False
        # Exactly one close for the arr run - the arr run's finish never ran, so this is
        # bootstrap's defensive emit and nothing doubled it.
        assert recording.of_type(ArrRunFinished) == [ArrRunFinished(arr=Arr.SONARR)]
        error_at, error = _arr_run_fatal_error(recording)
        assert recording.events.index(ArrRunFinished(arr=Arr.SONARR)) < error_at
        # The clean (no-traceback) handler's Diagnostic: the message, nothing captured.
        assert error.message == _SCHEMA_ERROR
        assert error.trace is None

    def test_traceback_branch_closes_the_run_before_logging(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_logger: logging.Logger,
    ) -> None:
        # The ordering is a property of the inner finally, not of which except handler
        # catches: an unrecognized failure takes the "Unexpected error" handler and the
        # close still precedes it.
        completed, recording = _run_failing_arr(monkeypatch, app_logger, RuntimeError("boom"))

        assert completed is False
        assert recording.of_type(ArrRunFinished) == [ArrRunFinished(arr=Arr.SONARR)]
        error_at, error = _arr_run_fatal_error(recording)
        assert recording.events.index(ArrRunFinished(arr=Arr.SONARR)) < error_at
        assert "Unexpected error during Sonarr run" in error.message
        assert error.trace is not None  # the exc_info handler carries its traceback


class _StubDeps:
    """A `ArrRunDeps` stand-in whose only exercised method is the outer finally's close."""

    def close(self) -> None:
        """No run-scoped resources to release (bootstrap's outer finally calls this)."""


class _FakeRunServices:
    """The per-id hub bootstrap builds. Never driven, since `ArrRunLoop` is faked."""

    def __init__(self, deps: ArrRunDeps, arr: Arr) -> None:
        del deps, arr


class _FakeSonarrSync:
    """The strategy handed to `run_arr`. The faked loop never consults it."""

    def __init__(self, deps: ArrRunDeps, services: _FakeRunServices) -> None:
        del deps, services


class _FakeRunLoop:
    """A run loop whose `run_arr` returns without emitting the finish's ArrRunFinished."""

    def __init__(self, deps: ArrRunDeps, services: _FakeRunServices) -> None:
        del deps, services

    def run_arr(
        self,
        strategy: _FakeSonarrSync,
        *,
        item_id: int | None,
        dry_run: bool,
        import_wait_mode: ImportWaitMode | None,
        boot: BootFlow,
    ) -> None:
        del strategy, item_id, dry_run, import_wait_mode, boot


def _completing_build() -> object:
    """A `ArrRunDeps.build` replacement returning a close-only stub (real signature)."""

    def build(
        arr: Arr,
        cache: str = "cache.db",
        *,
        logger: logging.Logger,
        resolver: MappingResolver,
        app_config: AppConfig,
        web: httpx.Client,
        boot: BootFlow,
    ) -> _StubDeps:
        del arr, cache, logger, resolver, app_config, web, boot
        return _StubDeps()

    return build


def _run_completing_arr(
    monkeypatch: pytest.MonkeyPatch,
    app_logger: logging.Logger,
) -> tuple[bool, RecordingHub]:
    """Drive one Sonarr run that completes normally. Record the event stream.

    The construction interfaces are faked so the arr run finishes without real deps: `build`
    returns a close-only stub, and `ArrRunServices` / `ArrRunLoop` / `SonarrStrategy` are
    minimal no-ops. bootstrap lazy-imports the latter three INSIDE `run_arrs`, so the
    source-module attributes are what the arr run resolves.
    """

    _write_config()
    monkeypatch.setattr(bootstrap, "build_resolver", _memory_resolver)
    monkeypatch.setattr(ArrRunDeps, "build", _completing_build())
    monkeypatch.setattr("pearlarr.arr_run_services.ArrRunServices", _FakeRunServices)
    monkeypatch.setattr("pearlarr.arr_run_loop.ArrRunLoop", _FakeRunLoop)
    monkeypatch.setattr("pearlarr.sonarr_strategy.SonarrStrategy", _FakeSonarrSync)

    recording = RecordingHub()
    install_hub(recording.hub)
    install_bridge()

    paths = resolve_paths()
    completed = run_arrs(
        [ArrTarget(Arr.SONARR)], paths=paths, logger=app_logger, file_log_renderer=FileLogRenderer(paths.log_dir)
    )
    return completed, recording


class TestCompletedArrRunSingleClose:
    """An arr run that returns normally adds no ArrRunFinished of its own - the arr run's finish owns it."""

    def test_completed_arr_run_emits_no_arr_run_finished_from_bootstrap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_logger: logging.Logger,
    ) -> None:
        completed, recording = _run_completing_arr(monkeypatch, app_logger)

        assert completed is True
        # The unwind emit fires only on a dying arr run. A normal return leaves the close
        # to the arr run's finish (faked to a no-op here). The real finish's emit is pinned in
        # test_import_wait.TestFinishArrRunOrdering, so zero here proves nothing doubled.
        assert recording.of_type(ArrRunFinished) == []
