"""The typed test currency: record events, assert dataclass equality (no MagicMock)."""

from __future__ import annotations

from typing import final, override

from .events import Event
from .hub import OutputHub, Renderer


@final
class RecordingRenderer(Renderer):
    """A renderer that records everything it is handed (events + lifecycle calls)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.whens: list[float] = []
        self.cycles = 0
        self.levels: list[int] = []
        self.closed = False

    @override
    def handle(self, event: Event, when: float) -> None:
        self.events.append(event)
        self.whens.append(when)

    @override
    def begin_cycle(self) -> None:
        self.cycles += 1

    @override
    def set_level(self, level: int) -> None:
        self.levels.append(level)

    @override
    def close(self) -> None:
        self.closed = True

    def of_type[E](self, cls: type[E]) -> list[E]:
        """The recorded events of one type, in emit order."""

        return [event for event in self.events if isinstance(event, cls)]


@final
class RecordingHub:
    """A real OutputHub over one RecordingRenderer.

    Handles mint real ScopeIds and dedup/containment behave exactly as in
    production.
    """

    def __init__(self) -> None:
        self.renderer = RecordingRenderer()
        self.hub = OutputHub([self.renderer])

    @property
    def events(self) -> list[Event]:
        return self.renderer.events

    def emit(self, event: Event) -> None:
        self.hub.emit(event)

    def of_type[E](self, cls: type[E]) -> list[E]:
        return self.renderer.of_type(cls)
