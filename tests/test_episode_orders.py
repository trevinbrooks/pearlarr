# pyright: strict
# pyright: reportPrivateUsage=false
# The tests read the strategy's private collaborators (_episodes / _reconciler),
# which strict re-flags. The repo disables reportPrivateUsage for tests.
"""Each series' TVDB orders through the Sonarr strategy: what it fetches, and what grab and import are served.

The grab is served a file's parse too, matched as an aired Sonarr matches it.
"""

from collections.abc import Iterable, Mapping
from datetime import timedelta

import pytest

from pearlarr.aired_view import AiredSeries, NoView
from pearlarr.arr_models import (
    AlternateTitle,
    MatchedEpisode,
    ParsedFileInfo,
    SonarrEpisode,
    SonarrItem,
    SonarrSeries,
)
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.grab_placement import ListedFile
from pearlarr.mappings import TitleMapping
from pearlarr.parse_cache import parse_to_json
from pearlarr.releases import EpisodeKey, SeadexDict
from pearlarr.sonarr_matching import AiredMatcher
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.tvdb_orders import OrderRow
from pearlarr.wait_types import ImportWaitMode

from .builders import (
    FakeCacheStore,
    make_release,
    make_sonarr_sync,
    pending_import,
    rg_group,
    sonarr_ep,
    stand_in_ep,
    stored_orders,
)
from .fakes import FakeSkyHookClient, FakeSonarrClient, debug_logger

_DVD = (OrderRow(1, 1, 9002, None), OrderRow(1, 2, 9001, None))
"""The dvd order of a two-episode season that swaps its episodes."""

_ORDERS = {"official": (OrderRow(1, 1, 9001, 1), OrderRow(1, 2, 9002, 2)), "dvd": _DVD}
"""A stored answer's orders: official and `_DVD`."""

_LONGER_AIRED = {"official": (*_ORDERS["official"], OrderRow(1, 3, 9003, 3)), "dvd": _DVD}
"""A stored answer whose official order has a third episode `_DVD` lacks."""

_ROWS = [sonarr_ep(1, 1, ep_id=11, tvdb_id=9001), sonarr_ep(1, 2, ep_id=12, tvdb_id=9002)]
"""Sonarr's rows of the season, numbered as aired."""

_DVD_ROWS = [sonarr_ep(1, 1, ep_id=11, tvdb_id=9002), sonarr_ep(1, 2, ep_id=12, tvdb_id=9001)]
"""Sonarr's rows of the season, numbered by `_DVD`."""

_SEASON_ONE = TitleMapping(anilist_id=500, tvdb_season=1)
"""A title mapped to the whole of season 1."""


def _series(
    series_id: int, tvdb_id: int, *, season_type: str = "official", counts: Mapping[str, int] | None = None
) -> SonarrSeries:
    """A Sonarr series row as the wire sends it, listing `counts` as v5 does (none, as on v4, when None)."""

    raw: dict[str, object] = {
        "id": series_id,
        "title": f"Show {series_id}",
        "tvdbId": tvdb_id,
        "seasonType": season_type,
    }
    if counts is not None:
        raw["seasonTypes"] = [{"type": name, "episodeCount": count} for name, count in counts.items()]
    return SonarrSeries.model_validate(raw)


def _sync(
    monkeypatch: pytest.MonkeyPatch,
    series: list[SonarrItem],
    cache: FakeCacheStore,
    client: FakeSkyHookClient | None = None,
    *,
    sonarr: FakeSonarrClient | None = None,
) -> SonarrStrategy:
    """The Sonarr strategy after the Sonarr run's library fetch of `series`, its orders fetched through `client`.

    Every series' episodes are `_ROWS` unless `sonarr` serves others. Its AniList knows no title.
    """

    strat = make_sonarr_sync(
        sonarr=sonarr or FakeSonarrClient(episodes=list(_ROWS)), cache_store=cache, skyhook_client=client
    )

    def _library() -> list[SonarrItem]:
        return series

    monkeypatch.setattr(strat._episodes, "get_all_sonarr_series", _library)
    strat.get_items()
    return strat


def _numbered(episodes: tuple[SonarrEpisode, ...]) -> list[tuple[int, EpisodeKey]]:
    """Each episode's id and its `(season, episode)`, in the order given."""

    return [(ep.id, EpisodeKey(ep.season_number or 0, ep.episode_number or 0)) for ep in episodes]


@pytest.mark.parametrize(
    ("series_id", "season_type", "other_orders"),
    [
        pytest.param(1, "official", {"dvd"}, id="official"),
        # A null or junk `seasonType` names no order.
        pytest.param(1, "", set[str](), id="no-order"),
        pytest.param(2, "", set[str](), id="not-in-the-library"),
    ],
)
def test_a_series_not_numbered_by_another_order_is_served_sonarrs_rows_as_sent(
    series_id: int, season_type: str, other_orders: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    strat = _sync(monkeypatch, [_series(1, 50, season_type=season_type)], cache)

    view = strat._episodes.aired_series(series_id)
    index = strat._episodes.series_index(series_id)

    assert isinstance(view, AiredSeries)
    assert (view.order, view.aired) == (season_type, tuple(_ROWS))
    assert index is not None
    assert (set(index.by_id), set(index.orders)) == ({11, 12}, other_orders)


def test_a_series_numbered_by_another_order_is_served_as_aired_with_its_stored_orders_read_onto_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    view = strat._episodes.aired_series(1)
    index = strat._episodes.series_index(1)

    assert isinstance(view, AiredSeries)
    assert view.order == "dvd"
    assert _numbered(view.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert index is not None
    assert index.id_by_key == {(1, 1): 12, (1, 2): 11, (1, 3): -9003}
    assert {name: order.id_by_key for name, order in index.orders.items()} == {"dvd": {(1, 1): 11, (1, 2): 12}}


def test_a_titles_episodes_of_a_series_numbered_by_another_order_are_its_aired_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    title = strat._episodes.title_episodes(1, 500, _SEASON_ONE)

    assert not isinstance(title, NoView)
    assert title.order == "dvd"
    assert _numbered(title.aired) == [(12, (1, 1)), (11, (1, 2)), (-9003, (1, 3))]
    assert _numbered(title.in_sonarr) == [(12, (1, 1)), (11, (1, 2))]
    assert title.index().by_id[-9003] == stand_in_ep(1, 3, tvdb_id=9003).model_copy(
        update={"absolute_episode_number": 3}
    )


def test_a_series_numbered_by_another_order_with_nothing_stored_has_no_view(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], FakeCacheStore(), sonarr=sonarr)

    assert strat._episodes.aired_series(1) is NoView.ORDERS_UNREAD
    assert strat._episodes.series_index(1) is None
    assert strat._episodes.title_episodes(1, 500, _SEASON_ONE) is NoView.ORDERS_UNREAD
    assert strat._series_indexes({1}) == {}


def test_a_series_index_is_built_once_for_a_read_of_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore())

    first = strat._episodes.series_index(1)

    assert first is not None
    assert strat._episodes.series_index(1) is first


def test_a_new_read_of_a_series_replaces_its_view_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = FakeSonarrClient(episodes=list(_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50)], FakeCacheStore(), sonarr=sonarr)
    before = strat._episodes.series_index(1)
    third = sonarr_ep(1, 3, ep_id=13, tvdb_id=9003)
    sonarr.episodes_return = [*_ROWS, third]

    strat._episodes.read_episodes(1)

    view = strat._episodes.aired_series(1)
    index = strat._episodes.series_index(1)
    assert before is not None
    assert set(before.by_id) == {11, 12}
    assert isinstance(view, AiredSeries)
    assert view.aired == (*_ROWS, third)
    assert index is not None
    assert set(index.by_id) == {11, 12, 13}


@pytest.mark.parametrize(
    "stored",
    [pytest.param(True, id="aired-view"), pytest.param(False, id="no-view")],
)
def test_the_import_reads_a_series_numbered_by_another_order_as_sonarr_numbers_it(
    stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    if stored:
        cache.put_tvdb_orders(50, stored_orders(_LONGER_AIRED))
    sonarr = FakeSonarrClient(episodes=list(_DVD_ROWS))
    strat = _sync(monkeypatch, [_series(1, 50, season_type="dvd")], cache, sonarr=sonarr)

    statuses = strat._reconciler._read_statuses(pending_import(series_id=1), [11])

    assert statuses is not None
    assert statuses.snapshot.by_series[1].episodes.key_by_id == {11: (1, 1), 12: (1, 2)}


@pytest.mark.parametrize(
    ("series_id", "unread"),
    [
        pytest.param(1, False, id="stored"),
        pytest.param(2, True, id="no-row"),
        pytest.param(3, False, id="no-tvdb-id"),
        pytest.param(4, False, id="not-in-the-library"),
    ],
)
def test_a_library_series_with_no_stored_answer_is_unread(
    series_id: int, unread: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60), _series(3, 0)], cache)

    assert strat._episodes.metadata_unread([], [series_id]) is unread


def test_orders_are_fetched_once_per_tvdb_series(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two Sonarr series of one TVDB series, one with no TVDB id, and one the Sonarr run's library fetch lacks.
    cache = FakeCacheStore()
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 50), _series(3, 0)], cache)

    assert strat._episodes.prefetch_orders([1, 2, 3, 9]) == 1
    assert cache.stats().tvdb_orders == 1


def test_sonarr_v5_counts_are_sent_with_the_fetch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # The stored answer is fresh, so only the count Sonarr's series row carries can make it due.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, sonarr_counts={"official": 2}))
    strat = _sync(monkeypatch, [_series(1, 50, counts={"official": 3})], cache)

    assert strat._episodes.prefetch_orders([1]) == 1


@pytest.mark.parametrize(
    ("mode", "fetched"),
    [pytest.param(ImportWaitMode.BLOCKING, 1, id="waiting"), pytest.param(ImportWaitMode.OFF, 0, id="off")],
)
def test_a_pending_imports_series_is_fetched_while_imports_are_waited_on(
    mode: ImportWaitMode, fetched: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing in the scan needs series 2, but grab and import read the pending import's series anyway.
    pending = pending_import(infohash="h", series_id=2)
    cache = FakeCacheStore(pending={str(Arr.SONARR): {pending.infohash: pending.to_json()}})
    strat = _sync(monkeypatch, [_series(2, 60)], cache)
    strat._services.bind_arr_run(ArrRunState(arr=Arr.SONARR, import_wait_mode=mode))

    assert strat.prefetch_orders([]) == fetched


def test_an_unreachable_source_leaves_a_new_series_without_orders_and_serves_the_stored_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Series 1's stored answer is due but SkyHook is unreachable: the stale answer still serves the grab.
    # Series 2 was never fetched, so its index has no orders.
    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders(_ORDERS, fresh_for=timedelta(days=-1)))
    client = FakeSkyHookClient(failing=frozenset({(50, "official"), (60, "official")}))
    strat = _sync(monkeypatch, [_series(1, 50), _series(2, 60)], cache, client)

    assert strat._episodes.prefetch_orders([1, 2]) == 0

    indexes = strat._series_indexes({1, 2})
    assert indexes[1].orders["dvd"].id_by_key == {(1, 1): 12, (1, 2): 11}
    assert indexes[2].orders == {}


_TWO_SEASONS = (
    OrderRow(0, 1, 9100, None),
    *(OrderRow(1, n, 9000 + n, n) for n in (1, 2, 3)),
    *(OrderRow(2, n, 9003 + n, 3 + n) for n in (1, 2, 3, 4)),
)
"""The aired order of a series with a special and two seasons, of three and four episodes."""

_AS_ONE_SEASON = (OrderRow(0, 1, 9100, None), *(OrderRow(1, n, 9000 + n, n) for n in range(1, 7)))
"""Its dvd order: the special, and the first six episodes as one season. It lacks the seventh."""

_ONE_SEASON_ROWS = [
    sonarr_ep(0, 1, ep_id=20, tvdb_id=9100, title="Recap"),
    *(sonarr_ep(1, n, ep_id=10 + n, tvdb_id=9000 + n, absolute=n) for n in range(1, 7)),
]
"""Sonarr's rows of that series, numbered by `_AS_ONE_SEASON`."""

_ZOKU = AlternateTitle(title="Show Zoku", scene_season_number=2)
"""The series' name for its second season."""

_URL = "https://nyaa.si/1"


def _matched(season: int, episode: int, ep_id: int) -> MatchedEpisode:
    return MatchedEpisode(season_number=season, episode_number=episode, id=ep_id)


def _stored_as_one_season(aired: tuple[OrderRow, ...] = _TWO_SEASONS) -> FakeCacheStore:
    """A cache with TVDB series 50's orders stored: `aired`, and `_AS_ONE_SEASON` as its dvd order."""

    cache = FakeCacheStore()
    cache.put_tvdb_orders(50, stored_orders({"official": aired, "dvd": _AS_ONE_SEASON}))
    return cache


def _in_one_season(*alternate_titles: AlternateTitle) -> SonarrSeries:
    """Series 1 of TVDB series 50, numbered by its dvd order, with its other names in Sonarr."""

    return _series(1, 50, season_type="dvd").model_copy(update={"alternateTitles": alternate_titles})


def _synced_in_one_season(
    monkeypatch: pytest.MonkeyPatch, sonarr: FakeSonarrClient | None = None, series: SonarrSeries | None = None
) -> SonarrStrategy:
    """The strategy after a library read of series 1, numbered by `_AS_ONE_SEASON`, before anything reads its rows.

    Sonarr's rows are `_ONE_SEASON_ROWS` unless `sonarr` serves others, and the series has no alternate title
    unless `series` has.
    """

    sonarr = sonarr or FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS))
    return _sync(monkeypatch, [series or _in_one_season()], _stored_as_one_season(), sonarr=sonarr)


def _parsing(parses: Mapping[str, ParsedFileInfo]) -> FakeSonarrClient:
    """A Sonarr with `_ONE_SEASON_ROWS` that answers each file name with its parse."""

    return FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS), parse_fn=parses.get)


def _listing(*names: str) -> SeadexDict:
    """One release listing the files, each of 100 bytes."""

    return {"Group": rg_group({_URL: make_release(url=_URL, files=list(names), size=[100] * len(names))})}


def test_a_name_with_only_an_absolute_number_is_matched_to_the_aired_episode_with_that_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sonarr counts the fifth episode as S01E05. As aired it is the second season's second.
    strat = _synced_in_one_season(monkeypatch)
    info = ParsedFileInfo(
        season_number=0,
        absolute_episode_numbers=(5,),
        series_title="Show",
        matched_episodes=(_matched(1, 5, 15),),
        series_id=1,
    )

    assert strat._episodes.aired_parse(info) == info.with_matched((_matched(2, 2, 15),))


def test_a_name_with_a_season_and_episode_is_matched_to_the_aired_episode_numbered_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sonarr's order has no second season, so it matched the name to nothing.
    strat = _synced_in_one_season(monkeypatch)
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_title="Show", series_id=1)

    assert strat._episodes.aired_parse(info) == info.with_matched((_matched(2, 2, 15),))


def test_an_episode_sonarr_matched_by_its_own_order_is_not_passed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # S01E05 is an episode of Sonarr's one season. The aired first season has three.
    strat = _synced_in_one_season(monkeypatch)
    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(5,),
        series_title="Show",
        matched_episodes=(_matched(1, 5, 15),),
        series_id=1,
    )

    assert strat._episodes.aired_parse(info) == info.with_matched(())


def test_a_name_is_matched_to_an_aired_episode_sonarrs_order_has_no_row_for(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _synced_in_one_season(monkeypatch)
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(7,), series_title="Show", series_id=1)

    assert strat._episodes.aired_parse(info) == info.with_matched((_matched(2, 4, -9007),))


def test_a_name_under_an_alternate_title_of_a_season_is_matched_within_that_season(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read by the series' own name, the third episode is S01E03.
    strat = _synced_in_one_season(monkeypatch, series=_in_one_season(_ZOKU))
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(3,), series_title="Show Zoku", series_id=1)

    assert strat._episodes.aired_parse(info) == info.with_matched((_matched(2, 3, 16),))


@pytest.mark.parametrize(
    "series_id",
    [
        pytest.param(2, id="aired-order"),
        pytest.param(9, id="not-in-the-library"),
        pytest.param(None, id="no-series"),
    ],
)
def test_a_parse_not_of_a_series_in_another_order_comes_back_as_it_is(
    series_id: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    strat = _sync(monkeypatch, [_in_one_season(), _series(2, 60)], _stored_as_one_season())
    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(2,),
        series_title="Show 2",
        matched_episodes=(_matched(1, 2, 12),) if series_id is not None else (),
        series_id=series_id,
    )

    assert strat._episodes.aired_parse(info) is info


def test_a_parse_reads_the_same_whatever_the_run_read_of_its_series_before(monkeypatch: pytest.MonkeyPatch) -> None:
    # One run has read nothing of series 1, one checked a title of it, and one re-read it in an import poll after.
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_title="Show", series_id=1)
    unread = _synced_in_one_season(monkeypatch)
    checked = _synced_in_one_season(monkeypatch)
    checked._episodes.title_episodes(1, 500, _SEASON_ONE)
    polled = _synced_in_one_season(monkeypatch)
    polled._episodes.title_episodes(1, 500, _SEASON_ONE)
    polled._episodes.read_episodes(1)

    answers = [strat._episodes.aired_parse(info) for strat in (unread, checked, polled)]

    assert answers == [info.with_matched((_matched(2, 2, 15),))] * 3


def test_a_series_matcher_is_built_once_for_a_read_of_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[AiredSeries] = []
    build = AiredMatcher.of

    def recording(series: AiredSeries, aliases: Iterable[AlternateTitle]) -> AiredMatcher:
        built.append(series)
        return build(series, aliases)

    monkeypatch.setattr(AiredMatcher, "of", recording)
    strat = _synced_in_one_season(monkeypatch)
    info = ParsedFileInfo(season_number=2, episode_numbers=(2,), series_title="Show", series_id=1)

    strat._episodes.aired_parse(info)
    strat._episodes.aired_parse(info)

    assert built == [strat._episodes.aired_series(1)]


def test_a_new_read_of_a_series_matches_a_name_to_its_new_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # An import poll re-reads the series after Sonarr gains a row for the seventh episode, where the stand-in was.
    sonarr = FakeSonarrClient(episodes=list(_ONE_SEASON_ROWS))
    strat = _synced_in_one_season(monkeypatch, sonarr)
    info = ParsedFileInfo(season_number=0, absolute_episode_numbers=(7,), series_title="Show", series_id=1)
    before = strat._episodes.aired_parse(info)
    sonarr.episodes_return = [*_ONE_SEASON_ROWS, sonarr_ep(1, 7, ep_id=17, tvdb_id=9007)]

    strat._episodes.read_episodes(1)

    assert before == info.with_matched((_matched(2, 4, -9007),))
    assert strat._episodes.aired_parse(info) == info.with_matched((_matched(2, 4, 17),))


_FIFTH = "Show - 05.mkv"

_FIFTH_BY_SONARR = ParsedFileInfo(
    season_number=0,
    absolute_episode_numbers=(5,),
    series_title="Show",
    matched_episodes=(_matched(1, 5, 15),),
    series_id=1,
)
"""Sonarr's own parse of `_FIFTH`: the fifth episode of its one season."""


def test_a_listed_file_of_a_series_in_another_order_carries_its_aired_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing in the run has read the series yet.
    strat = _synced_in_one_season(monkeypatch, _parsing({_FIFTH: _FIFTH_BY_SONARR}))

    files = strat._parse.parsed_files(_listing(_FIFTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))),)}


def test_the_parse_cache_keeps_sonarrs_own_answer_for_a_series_in_another_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strat = _synced_in_one_season(monkeypatch, _parsing({_FIFTH: _FIFTH_BY_SONARR}))

    strat._parse.parsed_files(_listing(_FIFTH))

    row = strat._parse.cache_store.get_sonarr_parse(_FIFTH)
    assert row is not None
    assert row["parse"] == parse_to_json(_FIFTH_BY_SONARR)


def test_a_cached_parse_is_matched_as_aired_again_without_asking_sonarr(monkeypatch: pytest.MonkeyPatch) -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR})
    strat = _synced_in_one_season(monkeypatch, sonarr)
    strat._parse.parsed_files(_listing(_FIFTH))

    ((again,),) = strat._parse.parsed_files(_listing(_FIFTH)).values()

    assert again.parse == _FIFTH_BY_SONARR.with_matched((_matched(2, 2, 15),))
    assert sonarr.parse_calls == [_FIFTH]


_SIXTH = "Show - 06.mkv"


def _with_two_sixth_episodes(monkeypatch: pytest.MonkeyPatch) -> SonarrStrategy:
    """`_synced_in_one_season` where TVDB's aired order counts the special as a sixth episode too.

    Two aired episodes carry the number, and Sonarr answers a name it reads by such a number with an error.
    """

    by_sonarr = ParsedFileInfo(
        season_number=0,
        absolute_episode_numbers=(6,),
        series_title="Show",
        matched_episodes=(_matched(1, 6, 16),),
        series_id=1,
    )
    cache = _stored_as_one_season((OrderRow(0, 1, 9100, 6), *_TWO_SEASONS[1:]))
    return _sync(monkeypatch, [_in_one_season()], cache, sonarr=_parsing({_SIXTH: by_sonarr}))


def test_a_name_an_aired_sonarr_answers_with_an_error_is_an_unread_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _with_two_sixth_episodes(monkeypatch)

    files = strat._parse.parsed_files(_listing(_SIXTH))

    assert files == {_URL: (ListedFile(_SIXTH, 100, None),)}


def test_the_debug_log_names_a_file_an_aired_sonarr_answers_with_an_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    strat = _with_two_sixth_episodes(monkeypatch)
    strat._parse.logger = debug_logger("pearlarr-aired-parse")

    with caplog.at_level("DEBUG"):
        strat._parse.parsed_files(_listing(_SIXTH))

    assert [record.message for record in caplog.records] == [
        f"No parse for {_SIXTH}: two aired episodes of its series share a number it is matched by"
    ]


def test_a_name_an_alternate_title_may_not_count_for_has_no_matched_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sonarr reads only some release names by the alternate title. By it the third episode is S02E03, without it
    # S01E03.
    limited = _ZOKU.model_copy(update={"comment": "Some releases only"})
    name = "Show Zoku - 03.mkv"
    by_sonarr = ParsedFileInfo(
        season_number=0,
        absolute_episode_numbers=(3,),
        series_title="Show Zoku",
        matched_episodes=(_matched(1, 3, 13),),
        series_id=1,
    )
    strat = _synced_in_one_season(monkeypatch, _parsing({name: by_sonarr}), _in_one_season(limited))

    files = strat._parse.parsed_files(_listing(name))

    assert files == {_URL: (ListedFile(name, 100, by_sonarr.with_matched(())),)}


def test_a_name_sonarr_searches_for_a_specials_title_has_no_matched_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sonarr looks for a special's title in the name of an episode 0 of a season, and the parse doesn't keep the name.
    name = "Show S01E00 Recap.mkv"
    by_sonarr = ParsedFileInfo(
        season_number=1,
        episode_numbers=(0,),
        series_title="Show",
        matched_episodes=(_matched(0, 1, 20),),
        series_id=1,
    )
    strat = _synced_in_one_season(monkeypatch, _parsing({name: by_sonarr}))

    files = strat._parse.parsed_files(_listing(name))

    assert files == {_URL: (ListedFile(name, 100, by_sonarr.with_matched(())),)}


def test_a_listed_file_of_a_series_whose_aired_order_is_not_stored_has_no_matched_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strat = _sync(monkeypatch, [_in_one_season()], FakeCacheStore(), sonarr=_parsing({_FIFTH: _FIFTH_BY_SONARR}))

    files = strat._parse.parsed_files(_listing(_FIFTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, _FIFTH_BY_SONARR.with_matched(())),)}


def test_a_listed_file_of_a_series_whose_episodes_fail_to_read_is_an_unread_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR})
    sonarr.episodes_return = None
    strat = _synced_in_one_season(monkeypatch, sonarr)

    files = strat._parse.parsed_files(_listing(_FIFTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, None),)}


def test_a_series_whose_episodes_fail_to_read_is_asked_for_once_however_many_files_it_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sonarr = _parsing({_FIFTH: _FIFTH_BY_SONARR, _SIXTH: _FIFTH_BY_SONARR})
    sonarr.episodes_return = None
    strat = _synced_in_one_season(monkeypatch, sonarr)
    asked_before = len(sonarr.episodes_calls)

    files = strat._parse.parsed_files(_listing(_FIFTH, _SIXTH))

    assert files == {_URL: (ListedFile(_FIFTH, 100, None), ListedFile(_SIXTH, 100, None))}
    assert sonarr.episodes_calls[asked_before:] == [1]
