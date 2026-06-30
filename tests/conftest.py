# pyright: strict
"""Shared pytest fixtures.

Kept deliberately small: fixtures land here as the migration consumes them, so
they're shaped by real use rather than guessed up front. The first is the
isolated logger that replaces the single process-global ``builders.make_logger``.
"""

import logging
from collections.abc import Iterator

import pytest


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
