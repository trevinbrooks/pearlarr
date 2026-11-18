"""A series' scene numbers, worked out from TheXEM's mappings.

Scene numbers are the season, episode and absolute number releases give an episode where they number a series
differently from TVDB. TheXEM maps them per TVDB episode, and Sonarr keeps them on its episode rows, except for a
series it numbers by another TVDB order. This is a copy of how Sonarr works them out for a series in aired order
(`XemService`; Sonarr and pearlarr are both GPL-3), on season and episode numbers in place of episode rows:
`scene_numbers` is `PerformUpdate`, and `_numbered_on` is `ExtrapolateMappings`.

Where it departs from Sonarr:

- Sonarr saves nothing for a series where a mapping finds two episodes on one season and episode number. The
  episodes handed in are one per number.
- Sonarr marks some of the episodes without a mapping as unverified, and numbers on only a marked one. The mark
  isn't kept. For an episode the numbering reaches it says one of two things, and each is tested in its place: in
  the last mapped TVDB season, that an episode of the season has a mapping, and in a later season, that releases
  and TVDB don't end on the same season. Its third reason, a mapping whose scene numbers are the episode's own
  numbers, only marks an episode with a mapping after it, which is never numbered on.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Self

from .releases import EpisodeKey


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneNumbers:
    """The numbers releases give one episode."""

    season: int
    episode: int
    absolute: int | None
    """None on an episode numbered on into a season no mapping names: Sonarr gives it no absolute number."""


class SceneMapping(NamedTuple):
    """One of TheXEM's mappings: the numbers releases give the episode TVDB numbers `season` and `episode`.

    Flat because it is stored as a JSON array and read back by position.
    """

    season: int
    episode: int
    scene_season: int
    scene_episode: int
    scene_absolute: int

    @property
    def tvdb(self) -> EpisodeKey:
        """The episode as TVDB numbers it."""

        return EpisodeKey(self.season, self.episode)

    @property
    def scene(self) -> SceneNumbers:
        """The episode as releases number it."""

        return SceneNumbers(season=self.scene_season, episode=self.scene_episode, absolute=self.scene_absolute)

    @property
    def names_no_episode(self) -> bool:
        """Whether all three scene numbers are 0, which Sonarr reads as a mapping to nothing."""

        return self.scene_season == 0 and self.scene_episode == 0 and self.scene_absolute == 0


@dataclass(frozen=True, slots=True, kw_only=True)
class CachedSceneMappings:
    """One series' mappings as TheXEM last sent them, replaced whole on each read."""

    fresh_until: str
    """When the mappings are due to be read again, in `stamps.STAMP_FORMAT`."""

    mappings: tuple[SceneMapping, ...]
    """The mappings that have a scene side, as TheXEM lists them. Empty when it has none for the series."""

    def to_json(self) -> dict[str, Any]:
        """The record as the cache stores it, each mapping a five-number array in `SceneMapping`'s field order."""

        return {"fresh_until": self.fresh_until, "mappings": [list(mapping) for mapping in self.mappings]}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self | None:
        """Rebuild a stored record, or None when it doesn't have the stored shape (the series is then read again)."""

        try:
            return cls(fresh_until=raw["fresh_until"], mappings=tuple(SceneMapping(*row) for row in raw["mappings"]))
        except (KeyError, TypeError):
            return None


def scene_numbers(aired: Iterable[EpisodeKey], mappings: Sequence[SceneMapping]) -> dict[EpisodeKey, SceneNumbers]:
    """The scene numbers of each aired episode that has any (`XemService.PerformUpdate`).

    `aired` are the season and episode numbers of every episode of one series as aired, one episode per number.
    `mappings` are all of TheXEM's mappings of the series that have a scene side, the ones Sonarr keeps
    (`XemProxy.GetSceneTvdbMappings`): the last TVDB season and the last scene season are taken over all of them,
    those of episodes the series lacks too.

    An episode with a mapping takes the mapping's scene numbers. The episodes after the mappings are numbered on
    from them (`_numbered_on`), unless releases number every mapped episode as a special. Any other episode is left
    out.
    """

    episodes = tuple(aired)
    listed = frozenset(episodes)
    numbers: dict[EpisodeKey, SceneNumbers] = {}
    for mapping in mappings:
        # TheXEM can map an episode before TVDB lists it.
        if mapping.tvdb not in listed:
            continue
        if mapping.names_no_episode:
            continue
        numbers[mapping.tvdb] = mapping.scene
    if any(scene.season != 0 for scene in numbers.values()):
        numbers |= _numbered_on(episodes, numbers, mappings)
    return numbers


def _numbered_on(
    episodes: Iterable[EpisodeKey], mapped: Mapping[EpisodeKey, SceneNumbers], mappings: Sequence[SceneMapping]
) -> dict[EpisodeKey, SceneNumbers]:
    """Scene numbers for the episodes after the mappings, counted on from them (`XemService.ExtrapolateMappings`).

    An episode after every mapping of the last mapped TVDB season continues from that season's last mapping. An
    episode of a later season takes its own episode number, in the scene season as far past the last scene season,
    when releases and TVDB don't end on the same season. `mapped` are the episodes that took a mapping's numbers.
    """

    last_scene_season = max(mapping.scene_season for mapping in mappings)
    last_tvdb_season = max(mapping.season for mapping in mappings)
    last_season_mapped = any(key.season == last_tvdb_season for key in mapped)
    last = _last(
        [mapping for mapping in mappings if mapping.season == last_tvdb_season], lambda mapping: mapping.episode
    )
    last_of_scene_season = _last(
        [mapping for mapping in mappings if mapping.scene_season == last.scene_season],
        lambda mapping: mapping.scene_episode,
    )

    numbered: dict[EpisodeKey, SceneNumbers] = {}
    for key in episodes:
        season, episode = key
        if season == 0:
            continue
        if key in mapped:
            continue
        if season < last_tvdb_season:
            continue
        if season > last_tvdb_season:
            # Sonarr marks an episode of a later season only when the last seasons differ.
            if last_scene_season != last_tvdb_season:
                numbered[key] = SceneNumbers(
                    season=last_scene_season + season - last_tvdb_season, episode=episode, absolute=None
                )
            continue
        # Sonarr marks an episode of the last mapped season only when an episode of the season has a mapping.
        if not last_season_mapped:
            continue
        if last.episode >= episode:
            continue
        # Nothing is numbered on in a scene season whose last episode is a TVDB special.
        if last_of_scene_season.season == 0:
            continue
        past = episode - last.episode
        numbered[key] = SceneNumbers(
            season=last.scene_season, episode=last.scene_episode + past, absolute=last.scene_absolute + past
        )
    return numbered


def _last(mappings: Sequence[SceneMapping], key: Callable[[SceneMapping], int]) -> SceneMapping:
    """The mapping with the highest key, the last listed of those that share it (Sonarr's `OrderBy(key).Last()`)."""

    return sorted(mappings, key=key)[-1]
