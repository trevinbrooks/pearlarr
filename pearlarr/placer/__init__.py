"""Place a torrent's files onto a title's Sonarr episodes. No I/O.

One way in: `place_batch` places a batch under a window, and its `PlacerResult` also says what the passes read
off each name of the torrent before the first pass (its episode title, whether its parse reads inside the window,
whether it is an extra, whether its number is its own title's).
"""

from ._passes import PlacerResult, place_batch
from ._state import NameFacts

__all__ = ["NameFacts", "PlacerResult", "place_batch"]
