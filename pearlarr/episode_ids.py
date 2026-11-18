"""The ids an episode is held by: a Sonarr row's own, or a stand-in's for an aired episode Sonarr has no row for.

A stand-in's id is its TVDB episode's, negated. It is the same on every run and never a Sonarr row's, which are
positive. 0 is no id. Every question of which kind an id is gets asked here.
"""

from collections.abc import Iterable


def is_stand_in(episode_id: int) -> bool:
    """Whether the id is a stand-in's, an episode Sonarr has no row for, and not a Sonarr episode's."""

    return episode_id < 0


def is_row_id(episode_id: int) -> bool:
    """Whether the id is a Sonarr row's: no stand-in's, and not the 0 of an episode with no id."""

    return episode_id > 0


def stand_in_id(tvdb_id: int) -> int:
    """The id of the stand-in for a TVDB episode: the same on every run, and never a Sonarr episode's."""

    return -tvdb_id


def stand_in_tvdb_id(episode_id: int) -> int:
    """The TVDB episode id a stand-in's id (`stand_in_id`) stands for."""

    return -episode_id


def stand_ins_among(ids: Iterable[int]) -> tuple[int, ...]:
    """The stand-ins' ids among `ids`, in their order."""

    return tuple(ep_id for ep_id in ids if is_stand_in(ep_id))


def without_stand_ins(ids: Iterable[int]) -> tuple[int, ...]:
    """`ids` less the stand-ins', in their order."""

    return tuple(ep_id for ep_id in ids if not is_stand_in(ep_id))
