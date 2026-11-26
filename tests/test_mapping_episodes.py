# pyright: strict
"""`SonarrEpisodes.title_episodes` AniBridge empty-season-map handling.

Pins the fix for the latent AniBridge bug: a mapping whose `tvdb_mappings` is an
empty dict (AniBridge registered the series' TVDB id but parsed no usable per-season
ranges) resolves to no episodes instead of silently selecting episodes. The caller
(`check_title`) surfaces the visible NO_EPISODES row. The sibling `{season: []}`
whole-season-covered case must NOT be treated as empty. An AniBridge mapping with no
TVDB ranges at all, found by another id, resolves to no episodes too.

Also pins that the anime-ids path and `episode_count` read the AniList format and
episode count from the run's AniList gateway.

The AniDB path is pinned too: the lookup gate (non-TV format or season 0, with an
AniDB id), and the either/or with the offset slice (a non-empty AniDB map bypasses it
entirely, while an empty one falls through to it).

Last, the mapped specials AniList's dates disagree with give way to the ones
`specials_by_date` chooses, logged once per run.

The episodes are a run's, over `make_run_deps`: Sonarr is a `FakeSonarrClient` listing
series 10, AniList's answer for title 123 is in the gateway's run cache, as the run's
prefetch leaves it, and the AniDB maps are an anime-lists file given to the real resolver.
"""

from dataclasses import replace
from datetime import date
from xml.etree import ElementTree

import pytest

from pearlarr.anilist_client import AniListAiring, AniListBody
from pearlarr.arr_models import SonarrEpisode
from pearlarr.mappings import MappingSource, TitleMapping
from pearlarr.output import Severity
from pearlarr.sonarr_episodes import SonarrEpisodes, TitleEpisodes, check_ep_by_anibridge, check_ep_by_anime_ids

from .builders import ScriptedAniListClient, anilist_body, make_resolver, make_run_deps, make_sonarr_episodes, sonarr_ep
from .fakes import FakeSonarrClient, diagnostic_messages, install_recording_hub

_NO_AIRING = AniListAiring(start=None, end=None, episodes=None, duration=None)


def _counted(episodes: int) -> AniListAiring:
    """AniList's facts for a title with `episodes` episodes and no air dates."""

    return AniListAiring(start=None, end=None, episodes=episodes, duration=None)


def _answer(media_format: str, airing: AniListAiring = _NO_AIRING) -> AniListBody:
    """AniList's answer for title 123: its format, and when it aired, how many episodes and how long each."""

    def fuzzy(day: date | None) -> dict[str, int] | None:
        return None if day is None else {"year": day.year, "month": day.month, "day": day.day}

    media = {
        "id": 123,
        "format": media_format,
        "episodes": airing.episodes,
        "duration": airing.duration,
        "startDate": fuzzy(airing.start),
        "endDate": fuzzy(airing.end),
    }
    return {"data": {"Media": media}}


def _anime_lists(season: int, tvdb_episodes: tuple[int, ...]) -> ElementTree.Element:
    """An anime-lists file mapping AniDB 99's episodes, in order, onto `tvdb_episodes` of TVDB `season`.

    With no episodes, AniDB knows the title and maps none of its episodes.
    """

    pairs = "".join(f";{anidb}-{tvdb}" for anidb, tvdb in enumerate(tvdb_episodes, start=1))
    mapping = f'<mapping-list><mapping tvdbseason="{season}">{pairs};</mapping></mapping-list>' if pairs else ""
    return ElementTree.fromstring(f'<anime-list><anime anidbid="99">{mapping}</anime></anime-list>')


def _episodes(
    listed: list[SonarrEpisode], anilist: AniListBody | None = None, *, anime_lists: ElementTree.Element | None = None
) -> SonarrEpisodes:
    """A run's episodes, with Sonarr listing `listed` and AniList answering `anilist` for title 123.

    With no `anilist` AniList knows no title, and with no `anime_lists` the AniDB source is off.
    """

    deps = make_run_deps()
    if anilist is not None:
        deps.anilist.answers[123] = anilist
    if anime_lists is not None:
        deps = replace(deps, resolver=make_resolver(deps.config, deps.http, anime_lists=anime_lists))
    return make_sonarr_episodes(deps, sonarr=FakeSonarrClient(episodes=listed))


def _resolved(episodes: SonarrEpisodes, mapping: TitleMapping) -> tuple[SonarrEpisode, ...]:
    """The episodes `mapping` resolves to for title 123 on series 10."""

    result = episodes.title_episodes(10, 123, mapping)
    assert isinstance(result, TitleEpisodes)
    return result.aired


def test_anime_ids_lookups_route_through_the_gateway() -> None:
    """The episode count the offset slice reads is the run's AniList gateway's.

    The gateway carries the warm arr run cache and the bound wire client (with its
    per-arr-run retry narration), so an AniList backoff here narrates instead of
    hanging silently. The wire client knows no title, so only the gateway's answer
    can make the count.
    """

    episodes = _episodes([sonarr_ep(1, 1), sonarr_ep(1, 2)], _answer("TV", _counted(1)))

    result = _resolved(episodes, TitleMapping(anilist_id=123, tvdb_season=1))

    # The gateway's episode count drove the offset slice (2 eps -> 1).
    assert [ep.episode_number for ep in result] == [1]


def test_the_episode_count_is_the_gateways() -> None:
    """`episode_count` reads AniList's count through the gateway too."""

    episodes = _episodes([], _answer("TV", _counted(12)))

    assert episodes.episode_count(123) == 12


def test_a_title_the_gateway_left_unread_is_unread() -> None:
    # AniList turns unreachable on title 5. Title 4 is looked up after that, from its answer in the arr run cache.
    deps = make_run_deps(anilist_client=ScriptedAniListClient(failing=frozenset({5})))
    deps.anilist.answers[4] = anilist_body(4)
    deps.anilist.episode_count(5)
    deps.anilist.episode_count(4)
    episodes = make_sonarr_episodes(deps)

    assert episodes.metadata_unread([4, 5]) is True
    assert episodes.metadata_unread([4]) is False


def test_empty_anibridge_season_map_resolves_to_no_episodes() -> None:
    """An empty `tvdb_mappings` -> no episodes (no silent grab, caller logs the no-episodes row)."""

    episodes = _episodes([sonarr_ep(1, 1)])

    mapping = TitleMapping(anilist_id=123, tvdb_mappings={})
    result = _resolved(episodes, mapping)

    assert result == ()


def test_an_anibridge_mapping_without_tvdb_ranges_resolves_to_no_episodes() -> None:
    """AniBridge found the title by IMDb or TMDB, so its season reads as -1, every regular season: no episodes."""

    episodes = _episodes([sonarr_ep(1, 1)])

    assert _resolved(episodes, TitleMapping(anilist_id=123, source=MappingSource.ANIBRIDGE)) == ()


def test_empty_map_is_distinct_from_whole_season_marker() -> None:
    """`{}` covers nothing, but `{1: []}` covers all of season 1.

    The guard keys on `if not tvdb_mappings`, so it must short-circuit only for
    `{}` and never for the present-but-empty `{season: []}` whole-season marker.
    """

    ep = sonarr_ep(1, 1)
    assert check_ep_by_anibridge(ep=ep, tvdb_mappings={}) is False
    assert check_ep_by_anibridge(ep=ep, tvdb_mappings={1: []}) is True


@pytest.mark.parametrize("media_format", [pytest.param("MOVIE", id="movie"), pytest.param("TV", id="tv")])
def test_anidb_map_filters_episodes_and_bypasses_the_offset_slice(media_format: str) -> None:
    """A non-empty AniDB map keeps only its (season, episode) hits - no offset slice.

    The gate lets in any title mapped to the specials, whatever its format. The two
    mechanisms are either/or: with a map in hand the offset slice never runs. AniList
    counts one episode, so a slice would keep S00E01 alone, and a wrong slice here
    would double-apply an offset the AniDB map already encodes.
    """

    episodes = _episodes(
        [sonarr_ep(0, 1), sonarr_ep(0, 2), sonarr_ep(0, 3)],
        _answer(media_format, _counted(1)),
        anime_lists=_anime_lists(0, (1, 3)),
    )

    result = _resolved(episodes, TitleMapping(anilist_id=123, tvdb_season=0, anidb_id=99))

    assert [ep.episode_number for ep in result] == [1, 3]


def test_tv_format_outside_season_zero_never_consults_anidb() -> None:
    """Format TV with a regular season skips the AniDB lookup even with an id.

    The gate is `not TV or season 0`: a plain TV season uses the offset slice,
    so AniDB's map (which would drop episode 2) must never be consulted.
    """

    episodes = _episodes(
        [sonarr_ep(1, 1), sonarr_ep(1, 2)], _answer("TV", _counted(2)), anime_lists=_anime_lists(1, (1,))
    )

    result = _resolved(episodes, TitleMapping(anilist_id=123, tvdb_season=1, anidb_id=99))

    assert [ep.episode_number for ep in result] == [1, 2]  # the offset slice applied


def test_a_mapping_without_an_anidb_id_takes_the_offset_slice() -> None:
    """No AniDB id on the mapping -> the offset slice, even when the gate would pass."""

    episodes = _episodes(
        [sonarr_ep(0, 1), sonarr_ep(0, 2)], _answer("MOVIE", _counted(1)), anime_lists=_anime_lists(0, (2,))
    )

    result = _resolved(episodes, TitleMapping(anilist_id=123, tvdb_season=0))

    assert [ep.episode_number for ep in result] == [1]  # the offset slice applied


def test_empty_anidb_map_falls_through_to_the_offset_slice() -> None:
    """An empty AniDB map is "no mapping": the offset slice applies as usual.

    AniDB knows the title but maps none of its episodes, so the episode count is
    resolved and the slice runs, as with the AniDB source off.
    """

    episodes = _episodes(
        [sonarr_ep(0, 1), sonarr_ep(0, 2)], _answer("MOVIE", _counted(1)), anime_lists=_anime_lists(0, ())
    )

    result = _resolved(episodes, TitleMapping(anilist_id=123, tvdb_season=0, anidb_id=99))

    assert [ep.episode_number for ep in result] == [1]


def test_an_absolute_mapping_slices_by_position_across_seasons() -> None:
    """A season of -1 numbers the title across every regular season, so its offset counts rows, not episode numbers.

    The second cour starts two episodes in, and Sonarr numbers it as season 2 from episode 1.
    """

    listed = [sonarr_ep(1, 1, ep_id=1), sonarr_ep(1, 2, ep_id=2), sonarr_ep(2, 1, ep_id=3), sonarr_ep(2, 2, ep_id=4)]
    episodes = _episodes(listed, _answer("TV", _counted(2)))

    result = _resolved(episodes, TitleMapping(anilist_id=123, tvdb_season=-1, tvdb_epoffset=2))

    assert [(ep.season_number, ep.episode_number) for ep in result] == [(2, 1), (2, 2)]


def test_season_zero_is_grabbed_at_season_zero_but_dropped_at_minus_one() -> None:
    """Movies-as-specials live in season 0 and must grab at `tvdb_season=0`.

    The VU1 fix matters here: an AniBridge mapping without TVDB mappings carries `tvdb_season=-1`,
    which DROPS every season-0 episode (`-1 & season 0 -> False`), so a movie
    shadowed to -1 was never grabbed as a special. Restoring Kometa's season 0
    (Part 1) makes the s0 episode selectable again.
    """

    s0 = sonarr_ep(0, 1)
    assert check_ep_by_anime_ids(ep=s0, tvdb_season=0) is True  # grabbed as a special
    assert check_ep_by_anime_ids(ep=s0, tvdb_season=-1) is False  # the shadowing -1 drops it


# The tables count every title below from S00E01, a short special from the year before it aired.
_MAPPED_FROM_SPECIAL_ONE = TitleMapping(anilist_id=123, tvdb_season=0)
_EARLIER_SPECIAL = sonarr_ep(0, 1, ep_id=1, air_date="2019-02-20", runtime=12)


def _dated_specials(later: list[SonarrEpisode], airing: AniListAiring) -> SonarrEpisodes:
    series = [_EARLIER_SPECIAL, *later, sonarr_ep(1, 1, ep_id=9, air_date="2019-01-05", runtime=24)]
    return _episodes(series, _answer("MOVIE", airing))


def _film() -> SonarrEpisodes:
    # S00E02 is film-length and aired on AniList's date.
    premiere = date(2020, 8, 29)
    return _dated_specials(
        [sonarr_ep(0, 2, ep_id=2, air_date="2020-08-29", runtime=95)],
        AniListAiring(start=premiere, end=premiere, episodes=1, duration=95),
    )


def test_mapped_specials_give_way_to_the_ones_aired_on_anilists_date() -> None:
    result = _resolved(_film(), _MAPPED_FROM_SPECIAL_ONE)

    assert [(ep.season_number, ep.episode_number) for ep in result] == [(0, 2)]


def test_a_pick_by_date_is_logged_once_a_run() -> None:
    recording = install_recording_hub()
    episodes = _film()

    episodes.title_episodes(10, 123, _MAPPED_FROM_SPECIAL_ONE)
    episodes.title_episodes(10, 123, _MAPPED_FROM_SPECIAL_ONE)

    assert diagnostic_messages(recording, Severity.DEBUG) == [
        "AniList 123 maps to S00E01 (aired 2019-02-20) but is dated 2020-08-29 - using S00E02 (aired 2020-08-29)"
    ]


def test_a_logged_pick_of_several_specials_names_each_and_both_dates() -> None:
    recording = install_recording_hub()
    # Two 45-minute parts, aired a week apart on AniList's start and end dates.
    episodes = _dated_specials(
        [
            sonarr_ep(0, 2, ep_id=2, air_date="2020-08-29", runtime=45),
            sonarr_ep(0, 3, ep_id=3, air_date="2020-09-05", runtime=45),
        ],
        AniListAiring(start=date(2020, 8, 29), end=date(2020, 9, 5), episodes=2, duration=45),
    )

    episodes.title_episodes(10, 123, _MAPPED_FROM_SPECIAL_ONE)

    note = (
        "AniList 123 maps to S00E01 (aired 2019-02-20), S00E02 (aired 2020-08-29) "
        "but is dated 2020-08-29 to 2020-09-05 - using S00E02 (aired 2020-08-29), S00E03 (aired 2020-09-05)"
    )
    assert diagnostic_messages(recording, Severity.DEBUG) == [note]
