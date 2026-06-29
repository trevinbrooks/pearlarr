"""``SonarrEpisodes.get_ep_list`` AniBridge empty-season-map handling.

Pins the fix for the latent AniBridge bug: a mapping whose ``tvdb_mappings`` is an
empty dict (AniBridge registered the series' TVDB id but parsed no usable per-season
ranges) resolves to ``[]`` instead of silently selecting episodes. The caller
(``process_al_id``) surfaces the visible NO_EPISODES skip. The sibling ``{season: []}``
whole-season-covered case must NOT be treated as empty.
"""

from unittest import mock

from seadexarr.modules.mappings import MappingEntry
from seadexarr.modules.sonarr_episodes import check_ep_by_anibridge

from .builders import make_sonarr_episodes, sonarr_ep


def test_empty_anibridge_season_map_resolves_to_no_episodes() -> None:
    """An empty ``tvdb_mappings`` -> ``[]`` (no silent grab; caller logs the skip)."""

    sonarr = mock.MagicMock()
    sonarr.episodes.return_value = [sonarr_ep(1, 1)]
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
