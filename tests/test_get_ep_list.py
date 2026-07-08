# pyright: strict
"""``SonarrEpisodes.get_ep_list`` AniBridge empty-season-map handling.

Pins the fix for the latent AniBridge bug: a mapping whose ``tvdb_mappings`` is an
empty dict (AniBridge registered the series' TVDB id but parsed no usable per-season
ranges) resolves to ``[]`` instead of silently selecting episodes. The caller
(``process_al_id``) surfaces the visible NO_EPISODES skip. The sibling ``{season: []}``
whole-season-covered case must NOT be treated as empty.

Also pins that the anime-ids path resolves the AniList format / episode count
through the gateway (whose retry log narrates backoffs), never the bare helpers.
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


class _RecordingAniList:
    """Records the gateway resolver calls ``get_ep_list`` routes through."""

    def __init__(self, *, media_format: str | None = None, n_eps: int | None = None) -> None:
        self._media_format = media_format
        self._n_eps = n_eps
        self.format_calls: list[int] = []
        self.n_eps_calls: list[int] = []

    def media_format(self, al_id: int) -> str | None:
        self.format_calls.append(al_id)
        return self._media_format

    def n_eps(self, al_id: int) -> int | None:
        self.n_eps_calls.append(al_id)
        return self._n_eps


class _NoAniDbMappings:
    """A mappings stand-in with no AniDB data (the anime-ids fast path)."""

    has_anidb = False


def test_anime_ids_lookups_route_through_the_gateway() -> None:
    """Format + episode-count lookups go through the AniList gateway.

    The gateway carries the warm run cache and the bound wire client (with its
    per-run retry narration), so an AniList backoff here narrates instead of
    hanging silently - these lookups must never bypass it to the bare wire.
    """

    sonarr = _FakeSonarr([sonarr_ep(1, 1), sonarr_ep(1, 2)])
    anilist = _RecordingAniList(n_eps=1)
    episodes = make_sonarr_episodes(sonarr=sonarr, _anilist=anilist, _mappings=_NoAniDbMappings())

    mapping = MappingEntry(anilist_id=123, tvdb_season=1)
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert anilist.format_calls == [123]
    assert anilist.n_eps_calls == [123]
    # The gateway-resolved episode count drove the offset slice (2 eps -> 1).
    assert result is not None
    assert [ep.episode_number for ep in result] == [1]


def test_empty_anibridge_season_map_resolves_to_no_episodes() -> None:
    """An empty ``tvdb_mappings`` -> ``[]`` (no silent grab; caller logs the skip)."""

    sonarr = _FakeSonarr([sonarr_ep(1, 1)])
    episodes = make_sonarr_episodes(sonarr=sonarr)

    mapping = MappingEntry(anilist_id=123, tvdb_mappings={})
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
