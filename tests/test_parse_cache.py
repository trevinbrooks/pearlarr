# pyright: strict
"""The stored Sonarr `/parse` answers, and a run's parse of a release's files through them.

A name goes to Sonarr once. A later run reads its stored answer while it is fresh, one with no matched episodes only
under the series hash of the library it was stored under.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
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


def _matched(season: int, episode: int) -> ParsedFileInfo:
    """A parse Sonarr resolved to one series episode, its episode id left out."""

    return ParsedFileInfo(matched_episodes=(MatchedEpisode(season_number=season, episode_number=episode),))


def _of_series(series_id: int) -> ParsedFileInfo:
    """A parse Sonarr matched to the first episode of the series with that id."""

    return ParsedFileInfo(
        season_number=1,
        episode_numbers=(1,),
        series_title="Show",
        matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=11),),
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

    One with none (pinned with `series_fp`) is read only under the same series hash and the short backstop. A legacy
    row carrying no whole parse is never read.
    """

    @pytest.mark.parametrize(
        ("row", "read_back"),
        [
            pytest.param(
                _record(_matched(1, 1), days_ago=SONARR_PARSE_CACHE_TTL_DAYS - 1),
                _matched(1, 1),
                id="matched-inside-the-ttl",
            ),
            pytest.param(
                _record(_matched(1, 1), days_ago=SONARR_PARSE_CACHE_TTL_DAYS + 1), None, id="matched-past-the-ttl"
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
    """The parse-cache leaf: fresh-only reads and shape-owning writes."""

    @pytest.mark.parametrize(
        ("info", "under_another_hash"),
        [
            pytest.param(_of_series(7), _of_series(7), id="matched"),
            pytest.param(_NO_SERIES, None, id="no-series"),
            pytest.param(_of_series(7).with_matched(()), None, id="a-series-and-no-episode"),
        ],
    )
    def test_a_written_parse_with_no_matched_episode_reads_back_only_under_its_series_hash(
        self, info: ParsedFileInfo, under_another_hash: ParsedFileInfo | None
    ) -> None:
        rows = ParseCacheRows(FakeCacheStore())

        rows.write(_NAME, info, window=_window("fp"))

        same_hash = rows.read(_NAME, window=_window("fp"))
        another_hash = rows.read(_NAME, window=_window("other"))
        assert (same_hash, another_hash) == (info, under_another_hash)

    def test_unreadable_parse_reads_as_a_miss(self) -> None:
        # A row whose values this build cannot validate re-parses: every field is present, so the key check passes
        # it and validation is what misses.
        row: dict[str, object] = {
            "fetched_at": _stamp(1),
            "parse": {**parse_to_json(_matched(1, 1)), "episode_numbers": "one"},
        }
        store = FakeCacheStore(sonarr_parse={_NAME: row})

        assert ParseCacheRows(store).read(_NAME, window=_window()) is None

    def test_a_parse_without_a_series_title_is_stale_for_the_warm_check_and_the_read(self) -> None:
        # The model would read the missing title as "", so the key decides, inside the TTL.
        store = FakeCacheStore(sonarr_parse={_NAME: _untitled(_matched(1, 1))})
        rows = ParseCacheRows(store)

        assert not rows.is_fresh(_NAME, window=_window())
        assert rows.read(_NAME, window=_window()) is None


_LIBRARY = (SonarrSeries(id=1),)
_HASH = sonarr_series_hash([1])
"""The series hash of a run over `_LIBRARY`."""


def _parse_cache(sonarr: FakeSonarrClient, cache: FakeCacheStore) -> SonarrParseCache:
    """A run's parse cache over the stored parses in `cache`, its library `_LIBRARY`, built as a run builds it."""

    return SonarrParseCache(make_run_deps(cache_store=cache), sonarr, make_series_reads(sonarr, library=_LIBRARY))


def _listing(files: Mapping[str, int]) -> SeadexDict:
    """One group's release `u`, listing each of `files` with its size."""

    return {"GroupA": rg_group({"u": make_release(files=list(files), size=list(files.values()))})}


class TestParsedFilesCache:
    """`parsed_files` asks Sonarr once for each name, then reads its stored answer while it is fresh.

    It never stores a failed answer, and it parses only the video files.
    """

    @pytest.mark.parametrize(
        ("answer", "asked"),
        [
            pytest.param(_matched(1, 1), [_NAME], id="matched"),
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

    def test_a_stored_parse_with_no_matched_episodes_is_asked_again_sooner(self) -> None:
        # The same age for both: past the backstop of a parse with no matched episodes, inside the TTL of one with.
        days_ago = SONARR_PARSE_NO_EPISODES_TTL_DAYS + 1
        cache = FakeCacheStore(
            sonarr_parse={
                _NAME: _record(_matched(1, 1), days_ago=days_ago),
                _OTHER_NAME: _record(_NO_SERIES, days_ago=days_ago, series_hash=_HASH),
            }
        )
        sonarr = FakeSonarrClient()

        _parse_cache(sonarr, cache).parsed_files(_listing({_NAME: 100, _OTHER_NAME: 100}))

        assert sonarr.parse_calls == [_OTHER_NAME]

    def test_parsed_files_pairs_each_video_file_with_its_size_and_parse(self) -> None:
        # Video files only, each with its own listed size and its parse.
        answers = {_NAME: _matched(1, 1), _OTHER_NAME: _matched(1, 2)}
        sonarr = FakeSonarrClient(parse_fn=answers.get)
        listing = _listing({_NAME: 100, "[X] Show - 01.ass": 1, "[X] OST - 01.flac": 2, _OTHER_NAME: 200})

        gathered = _parse_cache(sonarr, FakeCacheStore()).parsed_files(listing)

        assert gathered == {"u": (ListedFile(_NAME, 100, _matched(1, 1)), ListedFile(_OTHER_NAME, 200, _matched(1, 2)))}

    def test_names_not_yet_stored_go_to_sonarr_at_once(self) -> None:
        # Each answer waits, 30 seconds at most, for the other name to be asked, so asking one at a time fails.
        both_asked = threading.Barrier(2, timeout=30.0)

        def answer(name: str) -> ParsedFileInfo:
            del name
            both_asked.wait()
            return _matched(1, 1)

        sonarr = FakeSonarrClient(parse_fn=answer)

        gathered = _parse_cache(sonarr, FakeCacheStore()).parsed_files(_listing({_NAME: 100, _OTHER_NAME: 100}))

        assert gathered == {"u": (ListedFile(_NAME, 100, _matched(1, 1)), ListedFile(_OTHER_NAME, 100, _matched(1, 1)))}
