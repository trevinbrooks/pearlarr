# pyright: strict
"""Tests for the hooks the arr run calls on a strategy outside a title's check.

They cover the titles an item maps to, the one item a run is narrowed to, the id a series' carried-over pending
imports are reported under, and the start of a run.
"""

import pytest

from pearlarr.arr_models import RadarrItem, RadarrMovie, SonarrItem, SonarrSeries
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import Accent, Severity, TitleRow
from pearlarr.output.events import TitleState
from pearlarr.radarr_strategy import RadarrStrategy
from pearlarr.sonarr_strategy import SonarrStrategy

from .builders import make_config, make_entry_record, make_radarr_sync, make_run_deps, make_sonarr_sync, sonarr_ep
from .fakes import FakeRadarrClient, FakeSonarrClient, diagnostic_messages, install_recording_hub
from .title_check_builders import (
    FakeRunServices,
    mapping_one_ignored_imdb_title,
    mapping_one_imdb_title,
    mapping_one_series,
)


class TestItemMappings:
    """item_mappings finds an item's titles by every id the arr holds, and shows the ones the ignore list names."""

    def test_radarr_finds_a_movie_mapped_only_by_its_imdb_id(self) -> None:
        deps = mapping_one_imdb_title(make_run_deps(), imdb_id="tt7", anilist_id=7)
        strat = RadarrStrategy(deps, ArrRunServices(deps, Arr.RADARR), FakeRadarrClient())

        mappings = strat.item_mappings(RadarrMovie(tmdbId=42, imdbId="tt7"))

        assert mappings == {7: TitleMapping(anilist_id=7, imdb_id="tt7")}

    def test_sonarr_finds_a_series_mapped_only_by_its_imdb_id(self) -> None:
        deps = mapping_one_imdb_title(make_run_deps(), imdb_id="tt9", anilist_id=9)
        strat = SonarrStrategy(deps, ArrRunServices(deps, Arr.SONARR), sonarr_client=FakeSonarrClient())

        mappings = strat.item_mappings(SonarrSeries(tvdbId=99, imdbId="tt9"))

        assert mappings == {9: TitleMapping(anilist_id=9, imdb_id="tt9")}

    def test_radarr_shows_a_title_on_the_ignore_list_and_leaves_it_out_of_the_check(self) -> None:
        deps = mapping_one_ignored_imdb_title(make_run_deps(), imdb_id="tt7", anilist_id=7)
        strat = RadarrStrategy(deps, ArrRunServices(deps, Arr.RADARR), FakeRadarrClient())
        recording = install_recording_hub()

        mappings = strat.item_mappings(RadarrMovie(tmdbId=42, imdbId="tt7"))

        assert mappings == {}
        assert recording.events == [TitleRow(state=TitleState.IGNORED, label="AniList #7", accent=Accent.DIM)]

    def test_sonarr_shows_a_title_on_the_ignore_list_and_leaves_it_out_of_the_check(self) -> None:
        deps = mapping_one_ignored_imdb_title(make_run_deps(), imdb_id="tt9", anilist_id=9)
        strat = SonarrStrategy(deps, ArrRunServices(deps, Arr.SONARR), sonarr_client=FakeSonarrClient())
        recording = install_recording_hub()

        mappings = strat.item_mappings(SonarrSeries(tvdbId=99, imdbId="tt9"))

        assert mappings == {}
        assert recording.events == [TitleRow(state=TitleState.IGNORED, label="AniList #9", accent=Accent.DIM)]


class TestFilterToSingle:
    """filter_to_single keeps the item with the arr's external id, and warns when there is none."""

    @pytest.mark.parametrize(
        ("wanted", "kept", "warnings"),
        [
            pytest.param(20, [SonarrSeries(tvdbId=20)], [], id="listed"),
            pytest.param(
                7, [], ["No anime series with TVDB ID 7 found in Sonarr - check the --tvdb-id value"], id="not-listed"
            ),
        ],
    )
    def test_sonarr_keeps_the_series_of_the_tvdb_id(
        self, wanted: int, kept: list[SonarrItem], warnings: list[str]
    ) -> None:
        recording = install_recording_hub()

        result = make_sonarr_sync().filter_to_single([SonarrSeries(tvdbId=10), SonarrSeries(tvdbId=20)], wanted)

        assert (result, diagnostic_messages(recording, Severity.WARNING)) == (kept, warnings)

    @pytest.mark.parametrize(
        ("wanted", "kept", "warnings"),
        [
            pytest.param(2, [RadarrMovie(tmdbId=2)], [], id="listed"),
            pytest.param(
                7, [], ["No anime movie with TMDB ID 7 found in Radarr - check the --tmdb-id value"], id="not-listed"
            ),
        ],
    )
    def test_radarr_keeps_the_movie_of_the_tmdb_id(
        self, wanted: int, kept: list[RadarrItem], warnings: list[str]
    ) -> None:
        recording = install_recording_hub()

        result = make_radarr_sync().filter_to_single([RadarrMovie(tmdbId=1), RadarrMovie(tmdbId=2)], wanted)

        assert (result, diagnostic_messages(recording, Severity.WARNING)) == (kept, warnings)


def _sonarr_mapping_tvdb_50(sonarr: FakeSonarrClient, run: ArrRunServices | None = None) -> SonarrStrategy:
    """A `SonarrStrategy` built by its constructor over `sonarr`, where only TVDB series 50 maps, to AniList 500."""

    deps = mapping_one_series(make_run_deps(config=make_config(sleep_time=0)), tvdb_id=50, anilist_id=500)
    return SonarrStrategy(deps, run or ArrRunServices(deps, Arr.SONARR), sonarr_client=sonarr)


class TestPerItemSnapshotHook:
    """The arr run reports a series' carried-over pending imports under the id this hook names.

    A movie names none, as Radarr's pending imports are reconciled in the wait pass.
    """

    def test_sonarr_names_the_series_by_its_sonarr_id(self) -> None:
        # Pending imports belong to a series by its Sonarr id, not its TVDB id.
        assert make_sonarr_sync().pending_import_series_id(SonarrSeries(id=7, tvdbId=70)) == 7

    def test_pending_import_series_id_is_none(self) -> None:
        # Radarr pending imports carry series_id=0, so the per-item snapshot hook key is
        # None (short-circuits the per-item snapshot - reconcile runs in the wait pass).
        strat = make_radarr_sync()

        assert strat.pending_import_series_id(RadarrMovie(id=5)) is None


class TestArrRunStartHook:
    """get_items doubles as the arr run's start: it lists the mapped series, and the run reads them afresh."""

    def test_sonarr_lists_only_the_series_with_an_anilist_mapping(self) -> None:
        mapped = SonarrSeries(id=5, title="Show", tvdbId=50)
        strat = _sonarr_mapping_tvdb_50(FakeSonarrClient(all_series=[mapped, SonarrSeries(id=7, tvdbId=70)]))

        assert strat.get_items() == [mapped]

    def test_a_title_checked_in_the_next_run_reads_the_episodes_sonarr_holds_then(self) -> None:
        # The first run reads episode 1 with no file. Its file lands before the second run, whose check must not
        # go by the first run's rows.
        series = SonarrSeries(id=5, title="Show", tvdbId=50)
        sonarr = FakeSonarrClient(all_series=[series], episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)])
        run = FakeRunServices(prologue_entry=make_entry_record(), anilist_title="Show")
        strat = _sonarr_mapping_tvdb_50(sonarr, run)

        strat.get_items()
        strat.check_title(series, 500, TitleMapping(anilist_id=500))
        sonarr.episodes_return = [sonarr_ep(1, 1, ep_id=101, release_group="SubGroup")]
        strat.get_items()
        strat.check_title(series, 500, TitleMapping(anilist_id=500))

        assert run.missing_episodes == ["S01 E01"]
