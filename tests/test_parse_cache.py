# pyright: strict
"""The stored Sonarr `/parse` answers, and a run's parse of a release's files through them.

A name goes to Sonarr once. A later run reads its stored answer while it is fresh, one with no matched episodes only
under the series hash of the library it was stored under.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import pytest

from pearlarr.arr_models import MatchedEpisode, ParsedFileInfo, SonarrSeries
from pearlarr.grab_placement import ListedFile
from pearlarr.parse_cache import (
    SONARR_PARSE_CACHE_TTL_DAYS,
    SONARR_PARSE_NO_EPISODES_TTL_DAYS,
    ParseCacheRows,
    ParseFreshness,
    parse_to_json,
)
from pearlarr.releases import SeadexDict
from pearlarr.sonarr_episodes import sonarr_series_hash
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.stamps import stamp_of

from .builders import (
    FakeCacheStore,
    make_config,
    make_parse_cache,
    make_release,
    make_run_deps,
    make_series_reads,
    rg_group,
)
from .fakes import FakeSonarrClient

_NAME = "[X] Show - 01.mkv"
_OTHER_NAME = "[X] Show - 02.mkv"

# The parse Sonarr returns when it matched no series at all. It has no matched
# episodes, so its row is pinned to the series hash on write.
_NO_SERIES = ParsedFileInfo()


def _of_series(series_id: int, episode: int = 1) -> ParsedFileInfo:
    """A parse Sonarr matched to season 1 episode `episode` of the series with that id.

    Sonarr matches episodes only once it has matched a series, so every parse with matched episodes names one.
    """

    return ParsedFileInfo(
        season_number=1,
        episode_numbers=(episode,),
        series_title="Show",
        matched_episodes=(MatchedEpisode(season_number=1, episode_number=episode, id=episode),),
        series_id=series_id,
    )


def _stamp(days_ago: int) -> str:
    """A row stamp `days_ago` days before now.

    A run's window opens on the wall clock (`ParseFreshness.open`), so each stamp sits a day or more off a cutoff.
    """

    return stamp_of(datetime.now() - timedelta(days=days_ago))


def _window(series_hash: str = "fp", *, orders: Mapping[int, str] | None = None) -> ParseFreshness:
    """A run's parse freshness, opened now under the series hash `series_hash`.

    Each series of `orders` is numbered by its order there, and every other series as aired.
    """

    return ParseFreshness.open(series_hash, orders or {})


def _record(info: ParsedFileInfo, *, days_ago: int = 1, series_hash: str | None = None) -> dict[str, object]:
    """One persisted parse row, stamped `days_ago` and optionally pinned to a series hash."""

    row: dict[str, object] = {"fetched_at": _stamp(days_ago), "parse": parse_to_json(info)}
    if series_hash is not None:
        row["series_fp"] = series_hash
    return row


def _untitled(info: ParsedFileInfo) -> dict[str, object]:
    """A row stamped a day ago whose parse has no `series_title` key, as older builds wrote it."""

    parse = parse_to_json(info)
    del parse["series_title"]
    return {"fetched_at": _stamp(1), "parse": parse}


class TestSeriesHash:
    """`sonarr_series_hash` differs across different id sets, whatever order the library lists them in."""

    def test_the_listing_order_does_not_change_it(self) -> None:
        assert sonarr_series_hash([3, 1, 2]) == sonarr_series_hash([1, 2, 3])

    def test_different_sets_differ(self) -> None:
        # Sets of one size, so a hash of how many series there are can't pass.
        assert sonarr_series_hash([1, 2]) != sonarr_series_hash([1, 3])


class TestParseFreshness:
    """A stored parse with matched episodes is read for the 30-day TTL.

    One with none (pinned with `series_fp`) is read only under the same series hash and the short backstop. A row
    older builds stored without a whole parse or without one of its fields, or one this build can't validate, is never
    read.
    """

    @pytest.mark.parametrize(
        ("row", "read_back"),
        [
            pytest.param(
                _record(_of_series(7), days_ago=SONARR_PARSE_CACHE_TTL_DAYS - 1),
                _of_series(7),
                id="matched-inside-the-ttl",
            ),
            pytest.param(
                _record(_of_series(7), days_ago=SONARR_PARSE_CACHE_TTL_DAYS + 1), None, id="matched-past-the-ttl"
            ),
            pytest.param(
                _record(_NO_SERIES, days_ago=SONARR_PARSE_NO_EPISODES_TTL_DAYS - 1, series_hash="fp"),
                _NO_SERIES,
                id="no-matched-episode-inside-the-backstop",
            ),
            pytest.param(
                _record(_NO_SERIES, days_ago=SONARR_PARSE_NO_EPISODES_TTL_DAYS + 1, series_hash="fp"),
                None,
                id="no-matched-episode-past-the-backstop",
            ),
            pytest.param(
                _record(_NO_SERIES, series_hash="old"), None, id="no-matched-episode-under-another-series-hash"
            ),
            # Rows older builds stored carry an episode list, not a whole parse: they
            # re-parse once however recently they were stamped.
            pytest.param(
                {"fetched_at": _stamp(1), "episodes": [{"season": 1, "episode": 1}]}, None, id="a-legacy-episode-list"
            ),
            # The model would read the missing title as "", so the key decides.
            pytest.param(_untitled(_of_series(7)), None, id="a-parse-without-a-series-title"),
            # Every field is present, so the key check passes it and validation is what misses.
            pytest.param(
                {"fetched_at": _stamp(1), "parse": {**parse_to_json(_of_series(7)), "episode_numbers": "one"}},
                None,
                id="a-parse-this-build-cannot-validate",
            ),
        ],
    )
    def test_a_stored_parse_is_read_only_while_fresh(
        self, row: dict[str, object], read_back: ParsedFileInfo | None
    ) -> None:
        rows = ParseCacheRows(FakeCacheStore(sonarr_parse={_NAME: row}))

        assert rows.read(_NAME, window=_window("fp")) == read_back

    def test_a_row_stored_under_one_order_is_stale_once_its_series_is_in_another_of_its_orders(self) -> None:
        rows = ParseCacheRows(FakeCacheStore())

        rows.write(_NAME, _of_series(7), window=_window(orders={7: "dvd"}))

        same_order = rows.read(_NAME, window=_window(orders={7: "dvd"}))
        another_order = rows.read(_NAME, window=_window(orders={7: "altdvd"}))
        assert (same_order, another_order) == (_of_series(7), None)


class TestParseCacheRows:
    """A written parse reads back as Sonarr gave it.

    One with no matched episodes reads back only under the series hash it was written under.
    """

    @pytest.mark.parametrize(
        ("info", "under_another_hash"),
        [
            pytest.param(_of_series(7), _of_series(7), id="matched"),
            pytest.param(_NO_SERIES, None, id="no-series"),
            pytest.param(_of_series(7).with_matched(()), None, id="a-series-and-no-episode"),
        ],
    )
    def test_a_written_parse_reads_back_under_another_series_hash_only_with_matched_episodes(
        self, info: ParsedFileInfo, under_another_hash: ParsedFileInfo | None
    ) -> None:
        rows = ParseCacheRows(FakeCacheStore())

        rows.write(_NAME, info, window=_window("fp"))

        same_hash = rows.read(_NAME, window=_window("fp"))
        another_hash = rows.read(_NAME, window=_window("other"))
        assert (same_hash, another_hash) == (info, under_another_hash)


_LIBRARY = (SonarrSeries(id=1),)
_HASH = sonarr_series_hash([1])
"""The series hash of a run over `_LIBRARY`."""


def _parse_cache(
    sonarr: FakeSonarrClient, cache: FakeCacheStore, *, library: Sequence[SonarrSeries] = _LIBRARY
) -> SonarrParseCache:
    """The parse cache of a run over `library` and the stored parses in `cache` (`make_parse_cache`)."""

    return make_parse_cache(sonarr, cache, library=library)


def _listing(files: Mapping[str, int], urls: Sequence[str] = ("u",)) -> SeadexDict:
    """One group's release at each of `urls`, each listing every one of `files` with its size."""

    return {"GroupA": rg_group({url: make_release(files=list(files), size=list(files.values())) for url in urls})}


class TestParsedFilesCache:
    """`parsed_files` asks Sonarr once for each name, then reads its stored answer while it is fresh.

    It never stores a failed answer, it drops a stored one past the TTL, and it parses only the video files.
    """

    @pytest.mark.parametrize(
        ("answer", "asked"),
        [
            pytest.param(_of_series(1), [_NAME], id="matched"),
            pytest.param(_NO_SERIES, [_NAME], id="no-matched-episode"),
            pytest.param(None, [_NAME, _NAME], id="no-answer"),
        ],
    )
    def test_a_later_run_asks_sonarr_again_only_after_no_answer(
        self, answer: ParsedFileInfo | None, asked: list[str]
    ) -> None:
        cache = FakeCacheStore()
        sonarr = FakeSonarrClient(parse_fn=lambda _name: answer)

        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100}))
        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100}))

        assert sonarr.parse_calls == asked

    def test_a_name_that_matched_no_series_is_asked_again_once_the_library_gains_a_series(self) -> None:
        cache = FakeCacheStore()
        sonarr = FakeSonarrClient(parse_fn=lambda _name: _NO_SERIES)

        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100}))
        larger = (*_LIBRARY, SonarrSeries(id=2))
        _parse_cache(sonarr, cache, library=larger).parsed_files(_listing({_NAME: 100}))

        assert sonarr.parse_calls == [_NAME, _NAME]

    def test_a_stored_parse_with_no_matched_episodes_is_asked_again_sooner(self) -> None:
        # The same age for both: past the backstop of a parse with no matched episodes, inside the TTL of one with.
        days_ago = SONARR_PARSE_NO_EPISODES_TTL_DAYS + 1
        cache = FakeCacheStore(
            sonarr_parse={
                _NAME: _record(_of_series(1), days_ago=days_ago),
                _OTHER_NAME: _record(_NO_SERIES, days_ago=days_ago, series_hash=_HASH),
            }
        )
        sonarr = FakeSonarrClient()

        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100, _OTHER_NAME: 100}))

        assert sonarr.parse_calls == [_OTHER_NAME]

    def test_a_stored_parse_past_the_ttl_is_dropped_though_no_release_lists_it(self) -> None:
        # Nothing asks for a name no release lists any more, so without the drop its row would stay for good.
        cache = FakeCacheStore(
            sonarr_parse={_OTHER_NAME: _record(_of_series(1), days_ago=SONARR_PARSE_CACHE_TTL_DAYS + 1)}
        )

        _parse_cache(FakeSonarrClient(), cache).parsed_files(_listing({_NAME: 100}))

        assert cache.get_sonarr_parse(_OTHER_NAME) is None

    def test_each_video_file_is_paired_with_its_size_and_a_parse_asked_for_once(self) -> None:
        # Video files only, each with its own listed size and its parse. Both urls list the same files, as a release's
        # public and private torrents do.
        answers = {_NAME: _of_series(1), _OTHER_NAME: _of_series(1, episode=2)}
        sonarr = FakeSonarrClient(parse_fn=answers.get)
        files = {_NAME: 100, "[X] Show - 01.ass": 1, "[X] OST - 01.flac": 2, _OTHER_NAME: 200}

        gathered = _parse_cache(sonarr, FakeCacheStore()).parsed_files(_listing(files, urls=("u", "v")))

        paired = (ListedFile(_NAME, 100, _of_series(1)), ListedFile(_OTHER_NAME, 200, _of_series(1, episode=2)))
        assert (gathered, sorted(sonarr.parse_calls)) == ({"u": paired, "v": paired}, [_NAME, _OTHER_NAME])

    @pytest.mark.parametrize(
        ("sleep_time", "wait", "met"),
        [
            # Each ask waits up to `wait` seconds for the other. Unthrottled the two meet; the long wait keeps a slow
            # machine from failing them.
            pytest.param(0, 30.0, [_NAME, _OTHER_NAME], id="unthrottled"),
            # Throttled they go one at a time, so neither can meet the other however long it waits. The short wait
            # keeps the run quick. On a slow machine two asks at once can still miss each other, so this row can let
            # through a warm that ignores the throttle, but it never fails one that keeps it.
            pytest.param(1, 0.01, [], id="throttled"),
        ],
    )
    def test_a_name_never_asked_and_one_stored_stale_go_to_sonarr_at_once_unless_throttled(
        self, sleep_time: int, wait: float, met: list[str]
    ) -> None:
        both_asked = threading.Barrier(2)
        asked_together: list[str] = []

        def answer(name: str) -> ParsedFileInfo:
            with contextlib.suppress(threading.BrokenBarrierError):
                both_asked.wait(timeout=wait)
                asked_together.append(name)
            return _of_series(1)

        sonarr = FakeSonarrClient(parse_fn=answer)
        stale = _record(_NO_SERIES, days_ago=SONARR_PARSE_NO_EPISODES_TTL_DAYS + 1, series_hash=_HASH)
        cache = FakeCacheStore(sonarr_parse={_NAME: stale})
        deps = make_run_deps(config=make_config(sleep_time=sleep_time), cache_store=cache)
        parse_cache = SonarrParseCache(deps, sonarr, make_series_reads(sonarr, library=_LIBRARY))

        parse_cache.parsed_files(_listing({_NAME: 100, _OTHER_NAME: 100}))

        assert sorted(asked_together) == met
