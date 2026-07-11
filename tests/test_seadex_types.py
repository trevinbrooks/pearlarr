# pyright: strict
"""Direct tests for the `seadex_types` pydantic boundary machinery.

Pins the contracts the client boundaries build on: `validate_each`'s
skip-with-scrubbed-warning posture (warnings NEVER embed payload values), the
all-None empty-dict validation of `AniListMediaNode` (the miss-node
contract the AniList gateway depends on), and the frozen-model violation shape
(pydantic `ValidationError`, not `FrozenInstanceError`).
"""

import pytest
from pydantic import ValidationError

from pearlarr.modules.output import Diagnostic, Severity, install_hub
from pearlarr.modules.output.recording import RecordingHub
from pearlarr.modules.seadex_types import (
    AniListMediaNode,
    BoundaryContractError,
    CommandResource,
    HistoryRecord,
    MovieFile,
    ParsedFileInfo,
    QueueRecord,
    SonarrSeries,
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


def test_parsed_file_info_null_number_arrays_fold_to_empty() -> None:
    """Sonarr nulls empty number arrays; they fold to () like the absent case."""

    info = ParsedFileInfo.model_validate(
        {"parsedEpisodeInfo": {"seasonNumber": 1, "episodeNumbers": None, "absoluteEpisodeNumbers": None}},
    )
    assert info == ParsedFileInfo(season_number=1)


def test_history_record_field_name_construction_matches_alias_parse() -> None:
    """Field-name kwargs (the tests/fakes idiom) build the same record the aliased wire parse does.

    This is the `validate_by_name` contract.
    """

    by_name = HistoryRecord(id=2, date="d", item_id=5, event_type="grabbed", download_id="A")
    from_wire = HistoryRecord.model_validate(
        {"id": 2, "date": "d", "seriesId": 5, "eventType": "grabbed", "downloadId": "A"},
    )
    assert by_name == from_wire
