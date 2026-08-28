"""The injectable time seam: monotonic readings plus sleep behind one handle."""

import time
from abc import ABC, abstractmethod
from typing import override


class Clock(ABC):
    """Time faculties for code that waits or throttles, injected so tests drive them."""

    @abstractmethod
    def now(self) -> float:
        """A monotonic reading in seconds, comparable only against this clock's own readings."""

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Block for `seconds`."""


class SystemClock(Clock):
    """The real clock: `time.monotonic` readings and `time.sleep` waits."""

    @override
    def now(self) -> float:
        return time.monotonic()

    @override
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
