"""Place a torrent's files onto an entry's Sonarr episodes. No I/O. `assign_episode_ids` is the only way in.

`_run_passes` holds the four passes that read, place, or set aside a numbered run whole by its own release numbers.
`_passes` holds `assign_episode_ids`, the pass list and glossary, and every other pass. `_state` holds the state
every pass reads and writes.
"""

from ._passes import assign_episode_ids

__all__ = ["assign_episode_ids"]
