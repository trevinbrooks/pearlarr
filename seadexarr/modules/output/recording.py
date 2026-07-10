"""The typed test currency: record events, assert dataclass equality (no MagicMock)."""

from __future__ import annotations

from typing import ClassVar, final

from .events import Event
from .hub import OutputHub


@final
class RecordingRenderer:
    """A renderer that records everything it is handed (events + lifecycle calls)."""

    writes_file_only: ClassVar[bool] = False

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.whens: list[float] = []
        self.cycles = 0
        self.levels: list[int] = []
        self.closed = False

    def handle(self, event: Event, when: float) -> None:
        self.events.append(event)
        self.whens.append(when)

    def begin_cycle(self) -> None:
        self.cycles += 1

    def set_level(self, level: int) -> None:
        self.levels.append(level)

    def close(self) -> None:
        self.closed = True

    def of_type[E](self, cls: type[E]) -> list[E]:
        """The recorded events of one type, in emit order."""

        return [event for event in self.events if isinstance(event, cls)]


@final
class RecordingHub:
    """A real OutputHub over one RecordingRenderer, so handles mint real ScopeIds
    and dedup/containment behave exactly as in production."""

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
