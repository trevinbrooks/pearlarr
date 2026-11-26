# pyright: strict
"""Tests for the run services the strategies call, and for the fake that stands in for them.

The title check tests read what a strategy hands `FakeRunServices`. These pin the details the real services print for
the lines the fake only records, the read of a series' titles for its listings, and that the fake overrides every
public member of the services. The cached-title check and the services' other title interfaces are in
`test_cached_skip.py`.
"""

from collections.abc import Generator
from contextlib import contextmanager

import httpx

from pearlarr.anibridge import AniBridgeGraph
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.arr_run_state import ArrRunState
from pearlarr.config import Arr
from pearlarr.mappings import ExternalIds, MappingResolver, MappingSources, ResolvedMappings, TitleMapping
from pearlarr.output import Accent, StyledValue, TitleDetail

from .builders import make_config, make_run_deps, make_services
from .fakes import install_recording_hub
from .title_check_builders import FakeRunServices


def test_fake_overrides_the_full_public_surface() -> None:
    """`@override` catches renames and signature drift but NOT additions.

    A new public `ArrRunServices` member would be silently inherited here (real
    body over fake state) and only fail at test runtime. Reflection closes
    that gap - extend the fake when this fails.
    """

    # Loop-driven members (the arr run loop calls these, strategies never do), so the
    # fake doesn't script them.
    lifecycle = {"bind_arr_run", "state", "is_dry_run", "mark_dirty", "pay_pause", "selection_changed"}
    surface = {
        name
        for name, member in vars(ArrRunServices).items()
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }
    overridden = {name for name in vars(FakeRunServices) if not name.startswith("_")}
    assert surface - lifecycle <= overridden


def test_the_services_log_missing_episodes_as_a_caution_detail() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_missing_episodes("S01 E02")

    assert recording.of_type(TitleDetail) == [TitleDetail("missing", StyledValue("S01 E02", Accent.CAUTION))]


def test_the_services_log_whose_numbers_a_titles_lines_print_as_a_dim_detail_naming_sonarrs_order() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_numbers_order("dvd")

    assert recording.of_type(TitleDetail) == [TitleDetail("numbers", StyledValue("by Sonarr's DVD order", Accent.DIM))]


def test_the_services_log_the_files_left_out_as_a_note_detail_labeled_aired() -> None:
    recording = install_recording_hub()

    ArrRunServices(make_run_deps(), Arr.SONARR).log_aired_only("S01 E02 (1 of 2 files, NAN0)")

    assert recording.of_type(TitleDetail) == [
        TitleDetail(
            "left out",
            StyledValue("aired episodes not in Sonarr's order: S01 E02 (1 of 2 files, NAN0)", Accent.NOTE),
        )
    ]


# One inert shared client for the resolver ctors (inline sources: nothing downloads).
_WEB = httpx.Client()

GRAPH: AniBridgeGraph = {
    "anilist:269": {
        "tvdb_show:74796:s2": {"21-41": "1-21"},
        "tvdb_show:74796:s3": {"42-52": "1-9,11-12"},
        "anidb:1234": {},
        "imdb_show:tt0269": {},
        # Providers a lookup doesn't read. Real graphs carry them.
        "mal:111": {},
        "tmdb_show:5000:s1": {"1-12": "1-12"},
    },
    "anilist:270": {
        "tvdb_show:74796:s1": {},
        "tmdb_movie:888": {},
        "tvdb_movie:999": {},
    },
}


class _IgnoredIdLog:
    """Records each `log_ignored_anilist_id` call's AniList id."""

    def __init__(self) -> None:
        self.ids: list[int] = []

    def log_ignored_anilist_id(self, state: ArrRunState, anilist_id: int) -> None:
        del state
        self.ids.append(anilist_id)


class TestArrRunServicesIgnoredIds:
    """The run's two reads of one series' mappings: the titles it checks, and every title its listings fetch reads."""

    @staticmethod
    @contextmanager
    def _services(log: _IgnoredIdLog) -> Generator[ArrRunServices]:
        # tvdb 74796 maps 269 (both sources) and 270 (AniBridge only), and 270 is on the ignore list.
        resolver = MappingResolver(
            refresh_days=1,
            ignore_anilist_ids={270},
            web=_WEB,
            sources=MappingSources(
                anime={"x": {"anilist_id": 269, "tvdb_id": 74796, "tvdb_season": 7}},
                anidb=False,
                anibridge=GRAPH,
            ),
        )
        try:
            yield make_services(_resolver=resolver, _reporter=log)
        finally:
            resolver.close()

    def test_the_checked_titles_leave_out_the_ignored_one_and_log_it(self) -> None:
        log = _IgnoredIdLog()
        with self._services(log) as services:
            assert list(services.title_mappings(ExternalIds(tvdb=74796))) == [269]
        assert log.ids == [270]


def test_the_listing_read_gets_the_ignored_title_set_apart_and_logs_nothing() -> None:
    # The series' listings still give an ignored title its window. The check's own read shows the title once.
    config = make_config(ignore_anilist_ids={2})
    deps = make_run_deps(
        config=config,
        anime_ids={
            "Frieren": {"anilist_id": 1, "tvdb_id": 50, "tvdb_season": 1},
            "Frieren Season 2": {"anilist_id": 2, "tvdb_id": 50, "tvdb_season": 2},
        },
    )
    recording = install_recording_hub()

    resolved = ArrRunServices(deps, Arr.SONARR).mappings_with_ignored(ExternalIds(tvdb=50))

    assert resolved == ResolvedMappings(
        kept={1: TitleMapping(1, tvdb_season=1)}, ignored={2: TitleMapping(2, tvdb_season=2)}
    )
    assert recording.events == []
