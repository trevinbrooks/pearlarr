# pyright: strict
"""The composition root's unwind teardown: close the leg before reporting its death.

``run_arrs`` wraps each arr's body in an INNER ``try`` whose ``finally`` emits
``RunFinished``. Being inner, it runs during the unwind BEFORE the outer ``except``
arms log the leg-fatal error - so that error is a cycle-level fact rendered at
column 0, never a detail indented inside whatever entry / item / boot step the leg
happened to die in.

These pin the EMIT ORDERING (the half bootstrap owns). What the ordering buys - the
error actually landing at column 0 - is pinned against the real renderer in
``test_output_rich_renderer.TestUnwindPlacement``; the run tail's own boundary calls
are pinned in ``test_import_wait.TestFinalizeRunOrdering`` / ``TestFinalizeRunUnwind``.
A completed leg therefore emits ``RunFinished`` twice (the tail's, then this one);
the fold's close is idempotent (``test_output_breadcrumbs``) and a repeat is a
renderer no-op (``TestUnwindPlacement``).

``RunDeps.build`` is the failure site: it is the earliest thing inside the inner
try, so it also exercises the ``deps is None`` path - no reporter exists yet, which
is why bootstrap must emit through the process hub seam rather than the reporter.
"""

import logging
import os
from pathlib import Path

import httpx
import pytest
import yaml

import seadexarr.modules.bootstrap as bootstrap
from seadexarr.modules.boot_flow import BootFlow
from seadexarr.modules.bootstrap import run_arrs
from seadexarr.modules.cache import CacheSchemaError
from seadexarr.modules.config import AppConfig, Arr
from seadexarr.modules.mappings import MappingResolver, MappingSources
from seadexarr.modules.output import Diagnostic, RunFinished, Severity, install_bridge, install_hub
from seadexarr.modules.output.recording import RecordingHub
from seadexarr.modules.paths import resolve_paths
from seadexarr.modules.run_services import RunDeps

from .builders import make_config

# The message each scripted failure carries, so the adopted ERROR is identifiable
# in the recorded stream (the clean arm logs ``str(e)``; the traceback arm doesn't).
_SCHEMA_ERROR = "cache.db was written by a newer SeaDexArr"


def _memory_resolver(
    app_config: AppConfig,
    mappings_db: str,
    logger: logging.Logger,
    boot: BootFlow,
    retry: str,
    web: httpx.Client,
) -> MappingResolver | None:
    """A network-free stand-in for ``bootstrap.build_resolver`` (no sources enabled)."""

    del app_config, mappings_db, boot, retry
    return MappingResolver(
        cache_time=1,
        ignore_anilist_ids=set(),
        web=web,
        sources=MappingSources(anime={}, anidb=False, anibridge=False),
        logger=logger,
    )


def _failing_build(exc: Exception) -> object:
    """A ``RunDeps.build`` replacement that raises ``exc`` with the real signature.

    Patched onto the class, so it is reached as a plain function (no ``cls``); the
    real ``build`` is a classmethod and the call site passes ``arr`` positionally.
    """

    def build(
        arr: Arr,
        cache: str = "cache.db",
        *,
        logger: logging.Logger,
        mappings: MappingResolver,
        app_config: AppConfig,
        web: httpx.Client,
        boot: BootFlow,
    ) -> RunDeps:
        del arr, cache, logger, mappings, app_config, web, boot
        raise exc

    return build


def _write_config() -> None:
    """A valid, tight-permissioned Sonarr config where ``resolve_paths`` looks for it."""

    paths = resolve_paths()
    os.makedirs(paths.data_dir)
    config = Path(paths.config)
    config.write_text(
        yaml.safe_dump(make_config(url="http://sonarr.test", api_key="k").model_dump(mode="json")),
        encoding="utf-8",
    )
    # 0600 keeps the loose-permissions warning out of the recorded stream.
    config.chmod(0o600)


def _run_failing_leg(
    monkeypatch: pytest.MonkeyPatch,
    app_logger: logging.Logger,
    exc: Exception,
) -> tuple[bool, RecordingHub]:
    """Drive one Sonarr leg whose ``RunDeps.build`` raises ``exc``; record the stream.

    The real hub AND the real logging bridge are installed, so the composition root's
    ``logger.error`` is adopted into the same event stream as the unwind's
    ``RunFinished`` - which is the only way their relative order is observable.
    """

    _write_config()
    monkeypatch.setattr(bootstrap, "build_resolver", _memory_resolver)
    monkeypatch.setattr(RunDeps, "build", _failing_build(exc))

    recording = RecordingHub()
    install_hub(recording.hub)
    install_bridge(recording.hub)

    completed = run_arrs([(Arr.SONARR, None)], paths=resolve_paths(), logger=app_logger)
    return completed, recording


def _leg_fatal_error(recording: RecordingHub) -> tuple[int, Diagnostic]:
    """Where the leg-fatal ERROR the except arms logged landed, and what it said."""

    return next(
        (i, event)
        for i, event in enumerate(recording.events)
        if isinstance(event, Diagnostic) and event.severity is Severity.ERROR
    )


class TestUnwindEmitsRunFinished:
    """Every leg-fatal path closes the run before its error is reported."""

    def test_clean_arm_closes_the_run_before_logging(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_logger: logging.Logger,
    ) -> None:
        completed, recording = _run_failing_leg(monkeypatch, app_logger, CacheSchemaError(_SCHEMA_ERROR))

        assert completed is False
        # Exactly one close for the leg - the run tail never ran, so this is
        # bootstrap's defensive emit and nothing doubled it.
        assert recording.of_type(RunFinished) == [RunFinished(arr=Arr.SONARR)]
        error_at, error = _leg_fatal_error(recording)
        assert recording.events.index(RunFinished(arr=Arr.SONARR)) < error_at
        # The adopted record is the one the clean (no-traceback) arm logged.
        assert error.message == _SCHEMA_ERROR

    def test_traceback_arm_closes_the_run_before_logging(
        self,
        monkeypatch: pytest.MonkeyPatch,
        app_logger: logging.Logger,
    ) -> None:
        # The ordering is a property of the inner finally, not of which except arm
        # catches: an unrecognised failure takes the "Unexpected error" arm and the
        # close still precedes it.
        completed, recording = _run_failing_leg(monkeypatch, app_logger, RuntimeError("boom"))

        assert completed is False
        assert recording.of_type(RunFinished) == [RunFinished(arr=Arr.SONARR)]
        error_at, error = _leg_fatal_error(recording)
        assert recording.events.index(RunFinished(arr=Arr.SONARR)) < error_at
        assert "Unexpected error during Sonarr run" in error.message
