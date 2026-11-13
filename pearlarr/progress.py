"""The progress sink a long step reports through."""

from typing import Protocol


class ProgressCallback(Protocol):
    """Sink for step progress (`fraction` is 0-1), driving the boot cockpit's live bar.

    Protocol, not an ABC (house rule), so the data and gateway modules need not import the output layer.
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...
