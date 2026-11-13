# pyright: strict
"""Direct tests for the Sonarr and Radarr API models: how each reads its wire JSON, junk fields included."""

from datetime import date

import pytest

from pearlarr.arr_models import (
    AlternateTitle,
    CommandResource,
    HistoryPage,
    HistoryRecord,
    MatchedEpisode,
    ParsedFileInfo,
    SeasonTypeCount,
    SonarrEpisode,
    SonarrHistoryRecord,
    SonarrSeries,
)


def test_sonarr_episode_reads_its_air_date_and_runtime() -> None:
    episode = SonarrEpisode.model_validate({"airDate": "2020-08-29", "runtime": 95})

    assert (episode.air_date, episode.runtime) == (date(2020, 8, 29), 95)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"airDate": None, "runtime": None},
        {"airDate": "", "runtime": ""},
        {"airDate": "next summer", "runtime": "long"},
        {"airDate": 20200829, "runtime": [95]},
    ],
)
def test_sonarr_episode_folds_a_missing_or_junk_air_date_and_runtime_to_none(raw: dict[str, object]) -> None:
    episode = SonarrEpisode.model_validate({"id": 7, **raw})

    assert episode == SonarrEpisode(id=7)


def test_command_resource_junk_file_entry_skips_without_dropping_the_command() -> None:
    """A junk `body.files[]` element is skipped while the command survives.

    A dropped command would blind the in-flight ManualImport guard.
    """

    command = CommandResource.model_validate(
        {
            "id": 7,
            "name": "ManualImport",
            "status": "started",
            "body": {"files": ["junk", {"downloadId": "ABC", "episodeIds": [1, "x", 2]}]},
        },
    )

    assert command.id == 7
    assert command.name == "ManualImport"
    [file] = command.files
    assert file.download_id == "ABC"
    assert file.episode_ids == (1, 2)


def test_sonarr_series_reads_its_orders_and_drops_a_junk_one_alone() -> None:
    """v4 sends no order fields. A null `seasonType` names no order, and a junk `seasonTypes[]` element drops alone."""

    assert SonarrSeries.model_validate({"id": 1, "title": "Show"}).seasonType == "official"
    assert SonarrSeries.model_validate({"id": 1, "title": "Show", "seasonType": None}).seasonType == ""

    series = SonarrSeries.model_validate(
        {
            "id": 1,
            "title": "Show",
            "seasonType": "dvd",
            "seasonTypes": [
                {"type": "official", "episodeCount": 4},
                {"type": "dvd"},
                {"type": "dvd", "episodeCount": 2},
            ],
        },
    )
    assert series.seasonType == "dvd"
    assert series.seasonTypes == (
        SeasonTypeCount(type="official", episode_count=4),
        SeasonTypeCount(type="dvd", episode_count=2),
    )


def test_sonarr_series_reads_its_alternate_titles_and_drops_a_junk_one_alone() -> None:
    """A series with no `alternateTitles` has none, and a junk element (a null title among them) drops alone."""

    assert SonarrSeries.model_validate({"id": 1, "title": "Show"}).alternateTitles == ()

    series = SonarrSeries.model_validate(
        {
            "id": 1,
            "title": "Show",
            "alternateTitles": [
                {"title": "To Glimmer-Zu", "sceneSeasonNumber": -1},
                {"title": None, "sceneSeasonNumber": 2},
                "junk",
                {"title": "Shin To Glimmer-Zu", "sceneSeasonNumber": 2},
            ],
        },
    )
    assert series.alternateTitles == (AlternateTitle(title="To Glimmer-Zu"), AlternateTitle(title="Shin To Glimmer-Zu"))


def test_parsed_file_info_null_number_arrays_fold_to_empty() -> None:
    """Sonarr nulls empty number arrays. They reduce to () like the absent case."""

    info = ParsedFileInfo.model_validate(
        {"parsedEpisodeInfo": {"seasonNumber": 1, "episodeNumbers": None, "absoluteEpisodeNumbers": None}},
    )
    assert info == ParsedFileInfo(season_number=1)


def test_parsed_file_info_reads_the_lifted_series_title_and_folds_a_null_one() -> None:
    """The parsed series name comes along. A null one, or a null `parsedEpisodeInfo`, reads as "" and never fails."""

    lifted = ParsedFileInfo.model_validate({"parsedEpisodeInfo": {"seriesTitle": "To Glimmer-Zu"}})
    assert lifted.series_title == "To Glimmer-Zu"
    assert ParsedFileInfo.model_validate({"parsedEpisodeInfo": {"seriesTitle": None}}) == ParsedFileInfo()
    assert ParsedFileInfo.model_validate({"parsedEpisodeInfo": None, "episodes": []}) == ParsedFileInfo()


def test_parsed_file_info_with_numbers_replaces_only_the_numbers() -> None:
    """`with_numbers` swaps the episode numbers, absolute numbers, and matched pairs and keeps every other field."""

    info = ParsedFileInfo(
        season_number=1,
        episode_numbers=(5,),
        absolute_episode_numbers=(30,),
        matched_episodes=(MatchedEpisode(season_number=1, episode_number=5, id=105),),
        special=True,
        series_title="Show",
    )
    matched = (
        MatchedEpisode(season_number=1, episode_number=5, id=105),
        MatchedEpisode(season_number=1, episode_number=6),
    )

    assert info.with_numbers(episodes=(5, 6), absolutes=(), matched=matched) == ParsedFileInfo(
        season_number=1,
        episode_numbers=(5, 6),
        matched_episodes=matched,
        special=True,
        series_title="Show",
    )


def test_history_record_field_name_construction_matches_alias_parse() -> None:
    """Field-name kwargs (the tests/fakes idiom) build the same record the aliased wire parse does.

    This is the `validate_by_name` contract.
    """

    by_name = HistoryRecord(id=2, date="d", item_id=5, event_type="grabbed", download_id="A")
    from_wire = HistoryRecord.model_validate(
        {"id": 2, "date": "d", "seriesId": 5, "eventType": "grabbed", "downloadId": "A"},
    )
    assert by_name == from_wire


def test_sonarr_history_record_reads_the_import_row_fields() -> None:
    """`episodeId` and the `data` map's `droppedPath` land on the Sonarr row; the two-arr base carries neither."""

    row: dict[str, object] = {
        "id": 1,
        "eventType": "downloadFolderImported",
        "date": "d",
        "seriesId": 5,
        "episodeId": 77,
        "data": {"DroppedPath": "/d/a.mkv", "Reason": "x"},
    }
    record = SonarrHistoryRecord.model_validate(row)
    assert (record.item_id, record.episode_id, record.dropped_path, record.reason) == (5, 77, "/d/a.mkv", "x")
    # A top-level reason never blocks taking the path: each validator reads its own key.
    with_reason = SonarrHistoryRecord.model_validate({**row, "reason": "top"})
    assert (with_reason.reason, with_reason.dropped_path) == ("top", "/d/a.mkv")
    bare = SonarrHistoryRecord.model_validate({"eventType": "grabbed"})
    assert (bare.episode_id, bare.dropped_path) == (0, None)
    assert set(SonarrHistoryRecord.model_fields) - set(HistoryRecord.model_fields) == {"episode_id", "dropped_path"}
    assert "dropped_path" not in HistoryRecord.model_validate(row).model_dump()


def test_history_page_reads_total_records() -> None:
    """The envelope's `totalRecords` parses; an envelope without it reads as uncut."""

    assert HistoryPage.model_validate({"records": [], "totalRecords": 250}).total_records == 250
    assert HistoryPage.model_validate({"records": []}).total_records == 0
