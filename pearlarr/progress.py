"""The progress callback a long step reports through."""

from typing import Protocol


class ProgressCallback(Protocol):
    """Callback for step progress (`fraction` is 0-1), driving the boot region's live bar.

    Protocol, not an ABC (house rule), so the data and gateway modules need not import the output layer.
    """

    def progress(self, fraction: float, detail: str | None = None) -> None: ...
