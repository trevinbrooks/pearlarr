# pyright: strict
"""Direct tests for the `boundary` pydantic machinery.

Pins `validate_each`'s skip-with-scrubbed-warning posture (warnings NEVER embed payload values) and the frozen-model
violation shape (pydantic `ValidationError`, not `FrozenInstanceError`).
"""

import pytest
from pydantic import ValidationError

from pearlarr.arr_models import MovieFile, QueueRecord, SonarrSeries
from pearlarr.boundary import BoundaryContractError, validate_each
from pearlarr.output import Diagnostic, Severity, install_hub

from .fakes import RecordingHub

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
