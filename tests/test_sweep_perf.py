# pyright: strict
"""Tests for the Sonarr sweep speedups.

Covers the parse cache rows (freshness, the leaf's reads and writes), the
series-id hash, worker gating, and concurrent fresh episode prefetch.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import override

import pytest

from pearlarr.aired_view import AiredSeries, NoView
from pearlarr.arr_models import MatchedEpisode, ParsedFileInfo, SonarrEpisode, SonarrItem, SonarrSeries
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import AppConfig, Arr
from pearlarr.grab_placement import ListedFile
from pearlarr.parse_cache import (
    SONARR_PARSE_CACHE_TTL_DAYS,
    SONARR_PARSE_NO_EPISODES_TTL_DAYS,
    ParseCacheRows,
    ParseFreshness,
    parse_to_json,
)
from pearlarr.releases import SeadexDict
from pearlarr.series_reads import SeriesReads
from pearlarr.sonarr_episodes import SonarrEpisodes, sonarr_series_hash
from pearlarr.sonarr_parse import SonarrParseCache
from pearlarr.stamps import stamp_of

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    make_config,
    make_entry_record,
    make_release,
    make_run_deps,
    make_series_reads,
    rg_group,
    sonarr_ep,
)
from .fakes import FakeSonarrClient, RecordingProgress

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


class _Failure(Enum):
    """How a Sonarr episode read fails."""

    NO_ANSWER = auto()
    RAISES = auto()


class _Sonarr(FakeSonarrClient):
    """Sonarr holding one episode per series, numbered by the series id, recording each read and whether it was quiet.

    A read of a series in `failing` fails as its value says.
    """

    def __init__(self, failing: Mapping[int, _Failure] | None = None) -> None:
        super().__init__()
        self.failing: dict[int, _Failure] = dict(failing or {})
        self.reads: list[tuple[int, bool]] = []

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        self.reads.append((series_id, quiet))
        match self.failing.get(series_id):
            case _Failure.RAISES:
                raise ValueError("boom")
            case _Failure.NO_ANSWER:
                return None
            case None:
                return [sonarr_ep(1, series_id)]


class _GatedSonarr(_Sonarr):
    """A `_Sonarr` whose reads after the first wait for `gate`, so the one worker can't run ahead of the stop.

    The wait is bounded once: a gate that never opens fails the test in 30 seconds, never hangs it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        if self.reads and not self.gate.wait(30.0):
            self.gate.set()
        return super().episodes(series_id, quiet=quiet)


class _TogetherSonarr(_Sonarr):
    """A `_Sonarr` whose every read waits, 30 seconds at most, until two reads are under way."""

    def __init__(self) -> None:
        super().__init__()
        self._both_started = threading.Barrier(2, timeout=30.0)

    @override
    def episodes(self, series_id: int, *, quiet: bool = False) -> list[SonarrEpisode] | None:
        self._both_started.wait()
        return super().episodes(series_id, quiet=quiet)


class _StopOnFirstProgress:
    """A `ProgressCallback` that opens the fetch gate, then raises `SystemExit` as the SIGTERM handler would."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def progress(self, fraction: float, detail: str | None = None) -> None:
        del fraction, detail
        self._gate.set()
        raise SystemExit(0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Run:
    """The library a run's prefetch reads, and what SeaDex, the cache and the run hold for its titles."""

    series: tuple[int, ...] = (1,)
    """The library's series, by id (each its own TVDB id)."""
    titles: Mapping[int, int] = field(default_factory=lambda: {7: 1})
    """Each AniList title the mappings know, with the series it maps to."""
    without_entry: tuple[int, ...] = ()
    """The titles SeaDex has no entry for."""
    unchanged: tuple[int, ...] = ()
    """The titles cached under their entry's current update."""
    files_changed: tuple[int, ...] = ()
    """The titles whose files changed in Sonarr since the last run."""
    unmonitored: tuple[int, ...] = ()
    """The series Sonarr doesn't monitor."""
    settings: AppConfig = field(default_factory=make_config)


_TWO_SERIES = _Run(series=(1, 2), titles={7: 1, 8: 2})
"""Two series of one title each, neither title checked before."""


def _episodes(sonarr: _Sonarr, run: _Run) -> tuple[SonarrEpisodes, SeriesReads]:
    """The run's episode collaborator over `sonarr` and the reads it warms, built as a run builds them."""

    entries = {aid: make_entry_record(anilist_id=aid) for aid in run.titles}
    cache = FakeCacheStore()
    for aid in run.unchanged:
        cache.update_cached_title(Arr.SONARR, aid, {"updated_at": entries[aid].updated_at})
    deps = make_run_deps(
        config=run.settings,
        cache_store=cache,
        seadex=FakeSeaDexSource({aid: entry for aid, entry in entries.items() if aid not in run.without_entry}),
        anime_ids={f"Title {aid}": {"anilist_id": aid, "tvdb_id": sid} for aid, sid in run.titles.items()},
    )
    services = ArrRunServices(deps, Arr.SONARR)
    services.mark_dirty(run.files_changed)
    reads = make_series_reads(sonarr)
    return SonarrEpisodes(deps, sonarr, services, reads), reads


def _library(run: _Run) -> list[SonarrItem]:
    return [SonarrSeries(id=sid, tvdbId=sid, monitored=sid not in run.unmonitored) for sid in run.series]


class TestPrefetchEpisodes:
    """`SonarrEpisodes.prefetch` reads at once the episodes of each series the run will check.

    A series whose read fails is read again when the check asks for it, and the count and the progress take in
    every read made.
    """

    @pytest.mark.parametrize(
        ("run", "reads"),
        [
            pytest.param(_Run(), [(1, True)], id="a-title-never-checked"),
            pytest.param(_Run(unchanged=(7,)), [], id="its-title-unchanged"),
            pytest.param(_Run(titles={7: 1, 8: 1}, unchanged=(7,)), [(1, True)], id="one-of-its-two-titles-unchanged"),
            pytest.param(_Run(titles={}), [], id="no-anilist-mapping"),
            pytest.param(_Run(without_entry=(7,)), [], id="no-seadex-entry"),
            pytest.param(
                _Run(without_entry=(7,), settings=make_config(ignore_seadex_update_times=True)),
                [],
                id="no-seadex-entry-update-times-ignored",
            ),
            pytest.param(_Run(without_entry=(7,), files_changed=(7,)), [], id="no-seadex-entry-files-changed"),
            pytest.param(
                _Run(unmonitored=(1,), settings=make_config(sonarr_ignore_unmonitored=True)),
                [],
                id="unmonitored-and-ignored",
            ),
            pytest.param(_Run(unmonitored=(1,)), [(1, True)], id="unmonitored-not-ignored"),
        ],
    )
    def test_a_series_is_read_only_when_one_of_its_titles_will_be_checked(
        self, run: _Run, reads: list[tuple[int, bool]]
    ) -> None:
        sonarr = _Sonarr()
        episodes, _ = _episodes(sonarr, run)

        episodes.prefetch(_library(run))

        assert sonarr.reads == reads

    @pytest.mark.parametrize(
        ("failing", "reads"),
        [
            pytest.param({}, [(1, True)], id="answered"),
            pytest.param({1: _Failure.NO_ANSWER}, [(1, True), (1, False)], id="no-answer"),
            pytest.param({1: _Failure.RAISES}, [(1, True), (1, False)], id="raises"),
        ],
    )
    def test_the_check_reads_a_series_again_only_when_the_prefetch_could_not(
        self, failing: Mapping[int, _Failure], reads: list[tuple[int, bool]]
    ) -> None:
        run = _Run()
        sonarr = _Sonarr(failing)
        episodes, series_reads = _episodes(sonarr, run)
        episodes.prefetch(_library(run))
        sonarr.failing.clear()  # Sonarr answers again by the time the check asks

        view = series_reads.aired_series(1)

        assert isinstance(view, AiredSeries)
        assert (view.in_sonarr, sonarr.reads) == ((sonarr_ep(1, 1),), reads)

    def test_a_read_with_no_answer_still_counts_as_done(self) -> None:
        sonarr = _Sonarr({2: _Failure.NO_ANSWER})
        episodes, _ = _episodes(sonarr, _TWO_SERIES)
        progress = RecordingProgress()

        warmed = episodes.prefetch(_library(_TWO_SERIES), progress=progress)

        assert (warmed, progress.updates) == (2, [(0.5, "1/2"), (1.0, "2/2")])

    def test_unthrottled_reads_are_under_way_at_once(self) -> None:
        # Each read waits for the other to start, so reading one series at a time fails.
        sonarr = _TogetherSonarr()
        episodes, _ = _episodes(sonarr, _TWO_SERIES)

        episodes.prefetch(_library(_TWO_SERIES))

        assert sorted(sonarr.reads) == [(1, True), (2, True)]

    def test_a_stop_mid_sweep_drops_the_fetches_not_yet_started(self) -> None:
        sonarr = _GatedSonarr()
        # One worker, and a third series it can't have started by the stop.
        run = _Run(series=(1, 2, 3), titles={7: 1, 8: 2, 9: 3}, settings=make_config(sleep_time=2))
        episodes, _ = _episodes(sonarr, run)

        with pytest.raises(SystemExit):
            episodes.prefetch(_library(run), progress=_StopOnFirstProgress(sonarr.gate))

        # The first fetch, plus at most the one the worker had in hand.
        assert sonarr.reads in ([(1, True)], [(1, True), (2, True)])


class TestAiredSeriesRead:
    """`aired_series` fetches a cold series once per run and reports a failed fetch as `NoView.READ_FAILED`."""

    def test_second_call_serves_the_run_cache(self) -> None:
        sonarr = _Sonarr()
        reads = make_series_reads(sonarr)

        reads.aired_series(1)
        view = reads.aired_series(1)

        assert isinstance(view, AiredSeries)
        assert (view.in_sonarr, sonarr.reads) == ((sonarr_ep(1, 1),), [(1, False)])

    def test_failed_fetch_is_no_view_and_not_cached(self) -> None:
        # A failed read is the unreadable list, distinct from a series with no episodes:
        # every verdict that depends on the episode index places nothing on it instead of reading empty.
        sonarr = _Sonarr({1: _Failure.NO_ANSWER})
        reads = make_series_reads(sonarr)

        assert (reads.aired_series(1), reads.series_index(1), sonarr.reads) == (
            NoView.READ_FAILED,
            None,
            [(1, False), (1, False)],
        )


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
