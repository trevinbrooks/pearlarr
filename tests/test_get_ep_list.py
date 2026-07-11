# pyright: strict
"""`SonarrEpisodes.get_ep_list` AniBridge empty-season-map handling.

Pins the fix for the latent AniBridge bug: a mapping whose `tvdb_mappings` is an
empty dict (AniBridge registered the series' TVDB id but parsed no usable per-season
ranges) resolves to `[]` instead of silently selecting episodes. The caller
(`process_al_id`) surfaces the visible NO_EPISODES skip. The sibling `{season: []}`
whole-season-covered case must NOT be treated as empty.

Also pins that the anime-ids path resolves the AniList format / episode count
through the gateway (whose retry log narrates backoffs), never the bare helpers.

The `has_anidb=True` path is pinned too: the lookup gate (non-TV format or
season 0, with an AniDB id), the either/or with the offset slice (a non-empty
AniDB map bypasses it entirely; an empty one falls through to it), and the
None-key skip (an episode without a season/episode number can't hit the map).
"""

from pearlarr.modules.mappings import MappingEntry
from pearlarr.modules.seadex_types import SonarrEpisode
from pearlarr.modules.sonarr_episodes import check_ep_by_anibridge, check_ep_by_anime_ids

from .builders import make_sonarr_episodes, sonarr_ep


class _FakeSonarr:
    """Minimal Sonarr-client stand-in: `episodes` returns a fixed list.

    `get_ep_list` reaches the empty-map short-circuit only after the per-series
    episode fetch, so the fake just returns the scripted list (called positionally,
    no `quiet`).
    """

    def __init__(self, ep_list: list[SonarrEpisode]) -> None:
        self._ep_list = ep_list

    def episodes(self, series_id: int) -> list[SonarrEpisode]:
        del series_id
        return self._ep_list


class _RecordingAniList:
    """Records the gateway resolver calls `get_ep_list` routes through."""

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


class _FakeAniDbMappings:
    """An AniDB-bearing mappings stand-in: records lookups, returns a scripted map."""

    has_anidb = True

    def __init__(self, mapping: dict[int, dict[int, int]]) -> None:
        self._mapping = mapping
        self.calls: list[tuple[int, int]] = []

    def anidb_mapping_dict(self, anidb_id: int, tvdb_season: int) -> dict[int, dict[int, int]]:
        self.calls.append((anidb_id, tvdb_season))
        return self._mapping


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
    """An empty `tvdb_mappings` -> `[]` (no silent grab; caller logs the skip)."""

    sonarr = _FakeSonarr([sonarr_ep(1, 1)])
    episodes = make_sonarr_episodes(sonarr=sonarr)

    mapping = MappingEntry(anilist_id=123, tvdb_mappings={})
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert result == []


def test_empty_map_is_distinct_from_whole_season_marker() -> None:
    """`{}` covers nothing, but `{1: []}` covers all of season 1.

    The guard keys on `if not tvdb_mappings`, so it must short-circuit only for
    `{}` and never for the present-but-empty `{season: []}` whole-season marker.
    """

    ep = sonarr_ep(1, 1)
    assert check_ep_by_anibridge(ep=ep, tvdb_mappings={}) is False
    assert check_ep_by_anibridge(ep=ep, tvdb_mappings={1: []}) is True


def test_anidb_map_filters_episodes_and_bypasses_the_offset_slice() -> None:
    """A non-empty AniDB map keeps only its (season, episode) hits - no offset slice.

    The two mechanisms are either/or: with a map in hand the offset slice never
    runs, so the gateway's episode count is never consulted (`n_eps_calls` empty).
    A wrong slice here would double-apply an offset the AniDB map already encodes.
    """

    sonarr = _FakeSonarr([sonarr_ep(0, 1), sonarr_ep(0, 2), sonarr_ep(0, 3)])
    anilist = _RecordingAniList(media_format="MOVIE")
    mappings = _FakeAniDbMappings({0: {1: 100, 3: 300}})
    episodes = make_sonarr_episodes(sonarr=sonarr, _anilist=anilist, _mappings=mappings)

    mapping = MappingEntry(anilist_id=123, tvdb_season=0, anidb_id=99)
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert mappings.calls == [(99, 0)]
    assert result is not None
    assert [ep.episode_number for ep in result] == [1, 3]
    assert anilist.n_eps_calls == []  # the AniDB map bypasses the offset slice


def test_tv_format_outside_season_zero_never_consults_anidb() -> None:
    """Format TV with a regular season skips the AniDB lookup even with an id.

    The gate is `not TV or season 0`: a plain TV season uses the offset slice,
    so the scripted map (which would drop episode 2) must never be consulted.
    """

    sonarr = _FakeSonarr([sonarr_ep(1, 1), sonarr_ep(1, 2)])
    anilist = _RecordingAniList(media_format="TV", n_eps=2)
    mappings = _FakeAniDbMappings({1: {1: 1}})
    episodes = make_sonarr_episodes(sonarr=sonarr, _anilist=anilist, _mappings=mappings)

    mapping = MappingEntry(anilist_id=123, tvdb_season=1, anidb_id=99)
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert mappings.calls == []
    assert result is not None
    assert [ep.episode_number for ep in result] == [1, 2]  # the offset slice applied


def test_missing_anidb_id_never_consults_anidb() -> None:
    """No AniDB id on the mapping -> no lookup, even when the gate would pass."""

    sonarr = _FakeSonarr([sonarr_ep(0, 1), sonarr_ep(0, 2)])
    anilist = _RecordingAniList(media_format="MOVIE", n_eps=1)
    mappings = _FakeAniDbMappings({0: {1: 100}})
    episodes = make_sonarr_episodes(sonarr=sonarr, _anilist=anilist, _mappings=mappings)

    mapping = MappingEntry(anilist_id=123, tvdb_season=0)
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert mappings.calls == []
    assert result is not None
    assert [ep.episode_number for ep in result] == [1]  # the offset slice applied


def test_episodes_without_numbers_are_skipped_on_the_anidb_path() -> None:
    """A None season/episode number can never hit the AniDB map - skipped, no crash.

    `tvdb_season=-1` (anything but specials) lets the number-less episodes past
    the season prefilter, so this pins the anidb loop's own None guard.
    """

    sonarr = _FakeSonarr([sonarr_ep(1, 1), SonarrEpisode(), SonarrEpisode(season_number=1)])
    anilist = _RecordingAniList(media_format="MOVIE")
    mappings = _FakeAniDbMappings({1: {1: 100}})
    episodes = make_sonarr_episodes(sonarr=sonarr, _anilist=anilist, _mappings=mappings)

    mapping = MappingEntry(anilist_id=123, tvdb_season=-1, anidb_id=99)
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert mappings.calls == [(99, -1)]
    assert result is not None
    assert [(ep.season_number, ep.episode_number) for ep in result] == [(1, 1)]


def test_empty_anidb_map_falls_through_to_the_offset_slice() -> None:
    """An empty AniDB map is "no mapping": the offset slice applies as usual.

    The lookup happened (calls recorded) but returned nothing, so the episode
    count is resolved and the slice runs - matching the has_anidb=False tests.
    """

    sonarr = _FakeSonarr([sonarr_ep(0, 1), sonarr_ep(0, 2)])
    anilist = _RecordingAniList(media_format="MOVIE", n_eps=1)
    mappings = _FakeAniDbMappings({})
    episodes = make_sonarr_episodes(sonarr=sonarr, _anilist=anilist, _mappings=mappings)

    mapping = MappingEntry(anilist_id=123, tvdb_season=0, anidb_id=99)
    result = episodes.get_ep_list(sonarr_series_id=10, al_id=123, mapping=mapping)

    assert mappings.calls == [(99, 0)]
    assert anilist.n_eps_calls == [123]  # fell through to the offset slice
    assert result is not None
    assert [ep.episode_number for ep in result] == [1]


def test_season_zero_is_grabbed_at_season_zero_but_dropped_at_minus_one() -> None:
    """Movies-as-specials live in season 0 and must grab at `tvdb_season=0`.

    The VU1 fix matters here: a degraded AniBridge entry carries `tvdb_season=-1`,
    which DROPS every season-0 episode (`-1 & season 0 -> False`), so a movie
    shadowed to -1 was never grabbed as a special. Restoring Kometa's season 0
    (Part 1) makes the s0 episode selectable again.
    """

    s0 = sonarr_ep(0, 1)
    assert check_ep_by_anime_ids(ep=s0, tvdb_season=0) is True  # grabbed as a special
    assert check_ep_by_anime_ids(ep=s0, tvdb_season=-1) is False  # the shadowing -1 drops it
