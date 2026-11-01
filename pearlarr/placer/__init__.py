"""Place a torrent's files onto an entry's Sonarr episodes. No I/O. `assign_episode_ids` is the only way in."""

from ._passes import assign_episode_ids

__all__ = ["assign_episode_ids"]
