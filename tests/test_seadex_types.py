# pyright: strict
"""Direct tests for the ``seadex_types`` pydantic boundary machinery.

Pins the contracts the client boundaries build on: ``validate_each``'s
skip-with-scrubbed-warning posture (warnings NEVER embed payload values), the
all-None empty-dict validation of :class:`AniListMediaNode` (the miss-node
contract the AniList gateway depends on), and the frozen-model violation shape
(pydantic ``ValidationError``, not ``FrozenInstanceError``).
"""

import logging

import pytest
from pydantic import ValidationError

from seadexarr.modules.seadex_types import (
    AniListMediaNode,
    CommandResource,
    HistoryRecord,
    MovieFile,
    QueueRecord,
    validate_each,
)

from .fakes import CaptureHandler


def _capture_logger(name: str) -> tuple[logging.Logger, CaptureHandler]:
    """An isolated logger with a recording handler attached."""

    logger = logging.getLogger(name)
    logger.propagate = False
    capture = CaptureHandler()
    logger.handlers = [capture]
    return logger, capture


# --- validate_each ------------------------------------------------------------


def test_validate_each_skips_bad_records_and_scrubs_the_warning() -> None:
    """A junk record is skipped with ONE warning naming the model, index and
    failing field - never the payload value itself (it may carry titles/paths).
    """

    logger, capture = _capture_logger("seadexarr.test.validate-each")
    records = validate_each(
        MovieFile,
        [{"releaseGroup": "SubsPlease", "size": 1}, {"size": "not-a-number-SECRET"}],
        logger=logger,
    )

    assert records == [MovieFile(release_group="SubsPlease", size=1)]
    [warning] = [r for r in capture.records if r.levelno == logging.WARNING]
    message = warning.getMessage()
    assert "MovieFile" in message
    assert "[1]" in message
    assert "size" in message
    assert "SECRET" not in message  # the raw input value must never reach a log


def test_validate_each_empty_list_is_empty() -> None:
    """An empty payload validates to an empty list (an empty library is legal)."""

    logger, capture = _capture_logger("seadexarr.test.validate-each-empty")
    assert validate_each(MovieFile, [], logger=logger) == []
    assert capture.records == []


# --- model contracts ----------------------------------------------------------


def test_anilist_media_node_empty_dict_is_the_all_none_miss_node() -> None:
    """``{}`` (the reduced ``Media: null`` miss) validates to the all-None node."""

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
    """A junk ``body.files[]`` entry is skipped while the command survives - a
    dropped command would blind the in-flight ManualImport guard.
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


def test_history_record_field_name_construction_matches_alias_parse() -> None:
    """Field-name kwargs (the tests/fakes idiom) build the same record the
    aliased wire parse does - the ``validate_by_name`` contract.
    """

    by_name = HistoryRecord(id=2, date="d", item_id=5, event_type="grabbed", download_id="A")
    from_wire = HistoryRecord.model_validate(
        {"id": 2, "date": "d", "seriesId": 5, "eventType": "grabbed", "downloadId": "A"},
    )
    assert by_name == from_wire
