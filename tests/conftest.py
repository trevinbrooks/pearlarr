# pyright: strict
"""Shared pytest fixtures.

Kept deliberately small: fixtures land here as the migration consumes them, so
they're shaped by real use rather than guessed up front. The first is the
isolated logger that replaces the single process-global `builders.make_logger`.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from pearlarr.log import LOG_NAME
from pearlarr.mapping_store import MappingStore
from pearlarr.output import uninstall_bridge, uninstall_hub
from pearlarr.paths import DATA_DIR_ENV

from .builders import make_logger


@pytest.fixture(autouse=True)
def close_leaked_handles(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Close the sqlite stores + file log handlers a test leaves open.

    Two GC-timed `ResourceWarning` sources that `filterwarnings=["error"]`
    would otherwise turn into failures at a nondeterministic later moment (whoever
    is running when the GC finalizes the object):

    * `MappingStore`: `make_run_deps` / `make_sonarr_sync` build a real
      `MappingResolver` whose `:memory:` store the tests can't close in the
      builder (they query `deps.mappings` after construction). Wrapping the
      `open` factory registers every store regardless of construction path;
      `close()` is idempotent, so stores that already close themselves are fine.
    * The file handler `setup_logger` attaches to the `"Pearlarr"`
      logger (only the e2e smoke drives the real logging path); left open, its
      file handle leaks.

    The process-global `"Pearlarr"` logger is also fully reset (all handlers
    removed, level back to NOTSET): a test that ran `setup_logger` /
    `apply_log_level` would otherwise leak a console handler bound to its own
    captured stdout and a raised level into whichever test runs next under
    randomized ordering. The output seam gets the same treatment: a test that
    installed the hub/bridge (the cli run commands do) would otherwise leave the
    bridge on the ROOT logger, echoing every later test's third-party warnings.
    `setup_logger`/`apply_log_level` also open the ROOT logger's level (the
    bridge's gate); it is restored to the stdlib WARNING default here.
    """

    opened: list[MappingStore] = []
    real_open = MappingStore.open

    def tracking_open(path: str) -> MappingStore:
        store = real_open(path)
        opened.append(store)
        return store

    monkeypatch.setattr(MappingStore, "open", tracking_open)
    yield
    for store in opened:
        store.close()
    uninstall_bridge()
    uninstall_hub()
    app_logger = logging.getLogger("Pearlarr")
    for handler in list(app_logger.handlers):
        handler.close()
        app_logger.removeHandler(handler)
    app_logger.setLevel(logging.NOTSET)
    logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def isolate_data_dir(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point `PEARLARR_DATA_DIR` at a per-test tmp dir for every test.

    A backstop: `resolve_paths()` is read at runtime across `cli.py` and
    `mappings.py`, and `test_cli` does destructive cache ops keyed off it - one
    forgotten `setenv` would touch the developer's real data dir. Tests that set
    the env themselves still override this default; `@pytest.mark.real_data_dir`
    opts fully out (`test_paths` and the `@realdata` parity suite, which need
    the real dir / manage the env directly).
    """

    # `request.keywords` aggregates the item's markers (incl. class/module
    # pytestmark); `FixtureRequest.node` is untyped, so read markers off it.
    if "real_data_dir" in request.keywords:
        return
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "pearlarr_data"))


@pytest.fixture
def app_logger() -> Iterator[logging.Logger]:
    """The real app logger, isolated: handlers/filters/level/propagate restored.

    Starts at DEBUG so nothing is level-gated by default; tests pinning gated
    output set the level they need on the yielded logger.
    """

    log = logging.getLogger(LOG_NAME)
    saved_handlers = list(log.handlers)
    saved_filters = list(log.filters)
    saved_level = log.level
    saved_propagate = log.propagate
    log.handlers.clear()
    log.filters.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False
    yield log
    for handler in log.handlers:
        if handler not in saved_handlers:
            handler.close()
    log.handlers[:] = saved_handlers
    log.filters[:] = saved_filters
    log.setLevel(saved_level)
    log.propagate = saved_propagate


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    """An isolated quiet logger (NullHandler, no propagation, WARNING).

    Config (logger name, no-propagation, level) comes from the single shared
    `builders.make_logger` factory, so the fixture and the construction-time
    builders that also call it can't drift. On top of that the fixture brackets a
    handler reset on BOTH setup and teardown, so a level bump or an attached
    handler in one test can't leak into the next under randomized ordering (tests
    run sequentially, so the bracketing is total).
    """

    log = make_logger()
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    yield log
    log.handlers.clear()
    log.setLevel(logging.WARNING)
