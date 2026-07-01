# pyright: strict
"""Shared pytest fixtures.

Kept deliberately small: fixtures land here as the migration consumes them, so
they're shaped by real use rather than guessed up front. The first is the
isolated logger that replaces the single process-global ``builders.make_logger``.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from seadexarr.modules.paths import DATA_DIR_ENV


@pytest.fixture(autouse=True)
def isolate_data_dir(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point ``SEADEX_ARR_DATA_DIR`` at a per-test tmp dir for every test.

    A backstop: ``resolve_paths()`` is read at runtime across ``cli.py`` and
    ``mappings.py``, and ``test_cli`` does destructive cache ops keyed off it - one
    forgotten ``setenv`` would touch the developer's real data dir. Tests that set
    the env themselves still override this default; ``@pytest.mark.real_data_dir``
    opts fully out (``test_paths`` and the ``@realdata`` parity suite, which need
    the real dir / manage the env directly).
    """

    # ``request.keywords`` aggregates the item's markers (incl. class/module
    # pytestmark); ``FixtureRequest.node`` is untyped, so read markers off it.
    if "real_data_dir" in request.keywords:
        return
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "seadexarr_data"))


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    """An isolated quiet logger (NullHandler, no propagation, WARNING).

    Brackets the test logger with a reset on BOTH setup and teardown, so a level
    bump or an attached handler in one test can't leak into the next under
    randomized ordering - the disciplined replacement for the bare process-global
    ``builders.make_logger`` (tests run sequentially, so the bracketing is total).
    """

    log = logging.getLogger("seadexarr-test")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.propagate = False
    log.setLevel(logging.WARNING)
    yield log
    log.handlers.clear()
    log.setLevel(logging.WARNING)
