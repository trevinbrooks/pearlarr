# pyright: strict
"""Direct tests for the `seadex_types` pydantic boundary machinery.

Pins the contracts the client boundaries build on: `validate_each`'s
skip-with-scrubbed-warning posture (warnings NEVER embed payload values), the
all-None empty-dict validation of `AniListMediaNode` (the miss-node
contract the AniList gateway depends on), and the frozen-model violation shape
(pydantic `ValidationError`, not `FrozenInstanceError`).
"""

from datetime import date

import pytest
from pydantic import ValidationError

from pearlarr.output import Diagnostic, Severity, install_hub
from pearlarr.output.recording import RecordingHub
from pearlarr.seadex_types import (
    AlternateTitle,
    AniListMediaNode,
    BoundaryContractError,
    CommandResource,
    EpisodeKey,
    HistoryPage,
    HistoryRecord,
    MatchedEpisode,
    MovieFile,
    OrderOnlyKey,
    ParsedFileInfo,
    QueueRecord,
    SeasonTypeCount,
    SonarrEpisode,
    SonarrHistoryRecord,
    SonarrSeries,
    normalized_infohash,
    validate_each,
)

# --- validate_each ------------------------------------------------------------


def test_validate_each_skips_bad_records_and_scrubs_the_warning() -> None:
    """A junk record is skipped with ONE warning naming the model, index, and failing field.

    It never includes the payload value itself (it may carry titles/paths).
    """

    recording = RecordingHub()
    install_hub(recording.hub)  # conftest teardown restores the default
    records = validate_each(
        MovieFile,
        [{"releaseGroup": "SubsPlease", "size": 1}, {"size": "not-a-number-SECRET"}],
    )

    assert records == [MovieFile(release_group="SubsPlease", size=1)]
    [warning] = recording.of_type(Diagnostic)
    assert warning.severity is Severity.WARNING
    message = warning.message
    assert "MovieFile" in message
    assert "[1]" in message
    assert "size" in message
    assert "SECRET" not in message  # the raw input value must never reach a log


def test_validate_each_empty_list_is_empty() -> None:
    """An empty payload validates to an empty list (an empty library is legal)."""

    recording = RecordingHub()
    install_hub(recording.hub)  # conftest teardown restores the default
    assert validate_each(MovieFile, []) == []
    assert recording.of_type(Diagnostic) == []


def test_validate_each_strict_raises_when_nothing_validates() -> None:
    """strict=True: a non-empty payload with ZERO valid records raises `BoundaryContractError`.

    An all-invalid library must never read as empty.
    """

    with pytest.raises(BoundaryContractError):
        validate_each(SonarrSeries, ["junk", 42], strict=True)


def test_validate_each_strict_accepts_empty_and_partial_payloads() -> None:
    """strict=True still reads an EMPTY payload as [] (a legitimate empty library).

    It keeps the valid records of a partially-junk payload too.
    """

    assert validate_each(SonarrSeries, [], strict=True) == []
    records = validate_each(
        SonarrSeries,
        [{"id": 1, "title": "Show", "tvdbId": 7}, {"id": None, "title": "Null id"}],
        strict=True,
    )
    assert records == [SonarrSeries(id=1, title="Show", tvdbId=7)]


# --- model contracts ----------------------------------------------------------


def test_anilist_media_node_empty_dict_is_the_all_none_miss_node() -> None:
    """`{}` (the reduced `Media: null` miss) validates to the all-None node."""

    assert AniListMediaNode.model_validate({}) == AniListMediaNode()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"year": None, "month": None, "day": None},
        {"year": 2020, "month": 8, "day": None},
        {"year": 2020, "month": 8},
        # February has no 30th.
        {"year": 2020, "month": 2, "day": 30},
        "2020-08-29",
    ],
)
def test_an_anilist_date_without_every_part_is_no_date(raw: object) -> None:
    assert AniListMediaNode.model_validate({"id": 1, "startDate": raw}) == AniListMediaNode(id=1)


def test_anilist_media_node_takes_a_date_by_field_name() -> None:
    by_alias = AniListMediaNode.model_validate({"startDate": {"year": 2020, "month": 8, "day": 29}})
    by_name = AniListMediaNode(start_date=date(2020, 8, 29))

    assert by_alias.start_date == by_name.start_date == date(2020, 8, 29)


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


def test_frozen_model_mutation_raises_validation_error() -> None:
    """Ported models are frozen: assignment raises pydantic's ValidationError.

    The attribute name rides a variable because the checkers (pyrefly) already
    reject a direct assignment statically - this pins the RUNTIME contract.
    """

    record = QueueRecord(download_id="ABC", state="downloading", status="ok")
    frozen_field = "state"
    with pytest.raises(ValidationError):
        setattr(record, frozen_field, "importing")
    assert record.state == "downloading"


def test_command_resource_junk_file_entry_skips_without_dropping_the_command() -> None:
    """A junk `body.files[]` entry is skipped while the command survives.

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
    """v4 sends no order fields. A null `seasonType` names no order, and a junk `seasonTypes[]` entry drops alone."""

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
    """A row with no `alternateTitles` has none, and a junk entry (a null title among them) drops alone."""

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
    """Sonarr nulls empty number arrays. They fold to () like the absent case."""

    info = ParsedFileInfo.model_validate(
        {"parsedEpisodeInfo": {"seasonNumber": 1, "episodeNumbers": None, "absoluteEpisodeNumbers": None}},
    )
    assert info == ParsedFileInfo(season_number=1)


def test_parsed_file_info_reads_the_lifted_series_title_and_folds_a_null_one() -> None:
    """The title Sonarr lifted rides along. A null one, or a null `parsedEpisodeInfo`, reads as "" and never fails."""

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
    # A top-level reason never blocks the path lift: each validator reads its own key.
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


# --- normalized_infohash ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ABCDEF0123  ", "abcdef0123"),
        ("abcdef0123", "abcdef0123"),
        ("   ", None),
        ("", None),
        (None, None),
    ],
)
def test_normalized_infohash_is_stripped_lowercase_or_none(raw: str | None, expected: str | None) -> None:
    assert normalized_infohash(raw) == expected


# --- OrderOnlyKey -------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("dvd", "DVD S07E01"),
        ("altdvd", "alternate DVD S07E01"),
        ("alttwo", "second alternate S07E01"),
        ("regional", "regional S07E01"),
    ],
)
def test_an_order_only_key_is_labelled_under_its_orders_name(order: str, expected: str) -> None:
    assert OrderOnlyKey(order, EpisodeKey(7, 1)).label == expected


def test_an_order_only_key_survives_its_json_form() -> None:
    key = OrderOnlyKey("dvd", EpisodeKey(7, 1))

    assert key.to_json() == {"order": "dvd", "season": 7, "episode": 1}
    assert OrderOnlyKey.from_json(key.to_json()) == key
