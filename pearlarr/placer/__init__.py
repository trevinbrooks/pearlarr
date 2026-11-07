"""Place a torrent's files onto an entry's Sonarr episodes. No I/O.

Two ways in: `assign_episode_ids` places a batch under a scope, and `name_facts` reads what the passes would judge
each name by (its title, whether its parse reads inside the scope, whether it is an extra) without placing anything.
"""

from ._facts import NameFacts, name_facts
from ._passes import assign_episode_ids

__all__ = ["NameFacts", "assign_episode_ids", "name_facts"]
