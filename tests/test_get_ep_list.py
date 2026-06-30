# pyright: strict
"""``SonarrEpisodes.get_ep_list`` AniBridge empty-season-map handling.

Pins the fix for the latent AniBridge bug: a mapping whose ``tvdb_mappings`` is an
empty dict (AniBridge registered the series' TVDB id but parsed no usable per-season
ranges) resolves to ``[]`` instead of silently selecting episodes. The caller
(``process_al_id``) surfaces the visible NO_EPISODES skip. The sibling ``{season: []}``
whole-season-covered case must NOT be treated as empty.
"""

from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.seadex_types import SonarrEpisode
from seadexarr.modules.sonarr_episodes import check_ep_by_anibridge, check_ep_by_anime_ids

from .builders import make_sonarr_episodes, sonarr_ep


class _FakeSonarr:
    """Minimal Sonarr-client stand-in: ``episodes`` returns a fixed list.

    ``get_ep_list`` reaches the empty-map short-circuit only after the per-series
    episode fetch, so the fake just returns the scripted list (called positionally,
    no ``quiet``).
    """

    def __init__(self, ep_list: list[SonarrEpisode]) -> None:
        self._ep_list = ep_list

    def episodes(self, series_id: int) -> list[SonarrEpisode]:
        del series_id
        return self._ep_list


def test_empty_anibridge_season_map_resolves_to_no_episodes() -> None:
    """An empty ``tvdb_mappings`` -> ``[]`` (no silent grab; caller logs the skip)."""

    sonarr = _FakeSonarr([sonarr_ep(1, 1)])
    episodes = make_sonarr_episodes(sonarr=sonarr)

    mapping = MappingEntry(anilist_id=123, tvdb_id=456, tvdb_mappings={})
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert result == []


def test_empty_map_is_distinct_from_whole_season_marker() -> None:
    """``{}`` covers nothing, but ``{1: []}`` covers all of season 1.

    The guard keys on ``if not tvdb_mappings``, so it must short-circuit only for
    ``{}`` and never for the present-but-empty ``{season: []}`` whole-season marker.
    """

    ep = sonarr_ep(1, 1)
    assert check_ep_by_anibridge(ep=ep, tvdb_mappings={}) is False
    assert check_ep_by_anibridge(ep=ep, tvdb_mappings={1: []}) is True


def test_season_zero_is_grabbed_at_season_zero_but_dropped_at_minus_one() -> None:
    """Movies-as-specials live in season 0 and must grab at ``tvdb_season=0``.

    The VU1 fix matters here: a degraded AniBridge entry carries ``tvdb_season=-1``,
    which DROPS every season-0 episode (``-1 & season 0 -> False``), so a movie
    shadowed to -1 was never grabbed as a special. Restoring Kometa's season 0
    (Part 1) makes the s0 episode selectable again.
    """

    s0 = sonarr_ep(0, 1)
    assert check_ep_by_anime_ids(ep=s0, tvdb_season=0) is True  # grabbed as a special
    assert check_ep_by_anime_ids(ep=s0, tvdb_season=-1) is False  # the shadowing -1 drops it
