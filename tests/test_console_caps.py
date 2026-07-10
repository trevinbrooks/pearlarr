# pyright: strict
"""Tests for the shared caps cache (``console_caps.CapsCache``).

One CapsCache instance is shared by the console seat's regions (boot + wait),
so a mid-boot resize can never flip one surface's slow-heads-up decision only:
these pin the identity-keyed hit, the re-probe on a new console, and the cycle
reset.
"""

import io

from rich.console import Console

from seadexarr.modules.console_caps import CapsCache, detect_capabilities


def _console(width: int = 100) -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=width)


def test_same_console_identity_returns_the_cached_probe() -> None:
    cache = CapsCache()
    console = _console()

    first = cache.for_console(console)

    # detect_capabilities builds a fresh value per call, so an `is` hit proves
    # the second lookup never re-probed.
    assert cache.for_console(console) is first


def test_a_new_console_identity_reprobes_and_replaces() -> None:
    cache = CapsCache()
    cache.for_console(_console(width=100))

    narrow = cache.for_console(_console(width=20))

    assert narrow.width == 20
    assert not narrow.live  # below MIN_LIVE_WIDTH: the fresh probe was used


def test_reset_drops_the_cached_probe() -> None:
    cache = CapsCache()
    console = _console()
    first = cache.for_console(console)

    cache.reset()
    cache.reset()  # idempotent: both seats reset at cycle start

    assert cache.for_console(console) is not first


def test_no_console_takes_the_constant_path_without_touching_the_cache() -> None:
    cache = CapsCache()
    console = _console()
    cached = cache.for_console(console)

    assert cache.for_console(None) == detect_capabilities(None)
    assert cache.for_console(console) is cached
