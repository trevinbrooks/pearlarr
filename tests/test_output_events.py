# pyright: strict
"""Tests for the output-event vocabulary (`output.events`).

Pin the value model (frozen, equality-comparable facts), the severity mapping
that drives counts and renderer floors, and the single ArrRunStats -> ArrRunTally
conversion site (never two hand-maintained field lists).
"""

import dataclasses
from typing import Final

import pytest

from pearlarr.arr_run_state import ArrRunStats
from pearlarr.config import Arr
from pearlarr.output import (
    Accent,
    ArrRunFinished,
    ArrRunSummary,
    ArrRunTally,
    BootStepFinished,
    Diagnostic,
    GrabbedRow,
    GrabFailed,
    ItemStarted,
    NeedsActionCause,
    NeedsActionRow,
    PlacedBy,
    ReleaseSkipped,
    RowFinished,
    ScanFinished,
    ScanStarted,
    ScopeId,
    ScopeKind,
    Severity,
    StyledValue,
    TitleDetail,
    TitleHeader,
    TitleRow,
    severity_of,
)
from pearlarr.output.events import TitleState
from pearlarr.releases import SkipReason
from pearlarr.wait_types import Outcome, OutcomeKind

# --- value model -----------------------------------------------------------------


def test_events_are_frozen() -> None:
    header = TitleHeader(state=TitleState.CHECKING, title="Frieren")
    field_name = "title"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(header, field_name, "other")


def test_events_compare_by_value() -> None:
    a = ReleaseSkipped(group="Okay-Subs", tracker="AB", reason=SkipReason.PRIVATE_ONLY)
    b = ReleaseSkipped(group="Okay-Subs", tracker="AB", reason=SkipReason.PRIVATE_ONLY)
    assert a == b


def test_styled_value_defaults_to_plain() -> None:
    value = StyledValue("S01 E01-E28")
    assert value.accent is Accent.PLAIN


def test_dry_run_derives_from_the_note() -> None:
    """One field carries both the dry-run marker and its note.

    A noteless "DRY RUN" title (the literal-None render bug class) is unrepresentable.
    """

    real = ArrRunSummary(
        arr=Arr.SONARR,
        dry_run_note=None,
        grabbed_count=0,
        tally=ArrRunTally.from_stats(ArrRunStats()),
        wait_mode_on=False,
        warnings=0,
        errors=0,
        elapsed_s=None,
        tip=None,
    )
    assert not real.dry_run
    assert dataclasses.replace(real, dry_run_note="nothing grabbed").dry_run


def test_scope_id_is_a_value() -> None:
    assert ScopeId(ScopeKind.TITLE, 3) == ScopeId(ScopeKind.TITLE, 3)
    assert ScopeId(ScopeKind.TITLE, 3) != ScopeId(ScopeKind.ITEM, 3)


# --- severity mapping --------------------------------------------------------------


def test_diagnostic_and_title_detail_carry_their_own_severity() -> None:
    diag = Diagnostic(severity=Severity.ERROR, message="boom")
    detail = TitleDetail(label="skipped", value=StyledValue("x"), severity=Severity.WARNING)
    assert severity_of(diag) is Severity.ERROR
    assert severity_of(detail) is Severity.WARNING


def test_skip_reason_severity_marks_the_users_own_choice_info() -> None:
    def severity(reason: SkipReason) -> Severity:
        return ReleaseSkipped(group="G", tracker="Nyaa", reason=reason).severity

    assert severity(SkipReason.PRIVATE_ONLY) is Severity.WARNING
    assert severity(SkipReason.UNSUPPORTED_TRACKER) is Severity.WARNING
    assert severity(SkipReason.MISNUMBERED) is Severity.WARNING
    assert severity(SkipReason.PLACEMENT_INPUT_UNREAD) is Severity.WARNING
    assert severity(SkipReason.TRACKER_NOT_SELECTED) is Severity.INFO
    skipped = ReleaseSkipped(group="G", tracker="Nyaa", reason=SkipReason.TRACKER_NOT_SELECTED)
    assert severity_of(skipped) is Severity.INFO


def test_grab_failed_is_a_warning() -> None:
    assert severity_of(GrabFailed(group="G", url="u", error="tracker down")) is Severity.WARNING


def test_boot_step_finished_tallies_info_regardless_of_outcome() -> None:
    """The outcome drives glyphs/styles only.

    A failed/deferred step's caller logs the problem itself, so an outcome-based
    tally would double-count it.
    """

    scope = ScopeId(ScopeKind.BOOT_STEP, 1)

    def finished(outcome: OutcomeKind) -> BootStepFinished:
        return BootStepFinished(scope=scope, label="Reading config", outcome=outcome, detail=None, elapsed_s=0.1)

    assert severity_of(finished(OutcomeKind.IMPORTED)) is Severity.INFO
    assert severity_of(finished(OutcomeKind.LEFT)) is Severity.INFO
    assert severity_of(finished(OutcomeKind.FAILED)) is Severity.INFO


def test_row_finished_severity_follows_its_outcome_category() -> None:
    def row_finished(outcome: Outcome) -> RowFinished:
        return RowFinished(label="T", outcome=outcome, files=None, waited_s=1.0)

    assert severity_of(row_finished(Outcome.IMPORTED)) is Severity.INFO
    assert severity_of(row_finished(Outcome.DOWNLOAD_TIMED_OUT)) is Severity.WARNING
    assert severity_of(row_finished(Outcome.NOTHING_PLACED)) is Severity.WARNING
    assert severity_of(row_finished(Outcome.GONE)) is Severity.ERROR


def test_structural_events_default_to_info() -> None:
    assert severity_of(ScanStarted(arr=Arr.SONARR, total=182)) is Severity.INFO
    assert severity_of(ItemStarted(arr=Arr.SONARR, index=1, total=2, title="X")) is Severity.INFO
    assert severity_of(TitleRow(state=TitleState.IGNORED, label="AniList #1")) is Severity.INFO
    # The close boundaries are markers, not tallied facts: INFO tallies as nothing
    # in the warnings / errors counts.
    assert severity_of(ScanFinished(arr=Arr.SONARR)) is Severity.INFO
    assert severity_of(ArrRunFinished(arr=Arr.SONARR)) is Severity.INFO


def test_diagnostic_defaults_are_ambient_and_shared() -> None:
    diag = Diagnostic(severity=Severity.WARNING, message="m")
    assert diag.placed_by is PlacedBy.AMBIENT
    assert diag.origin == "app"
    assert diag.once_key is None
    assert diag.trace is None
    assert not diag.file_only


# --- ArrRunTally: the single conversion site ------------------------------------------


def test_arr_run_tally_freezes_arr_run_stats_whole() -> None:
    stats = ArrRunStats(checked=182, up_to_date=161, unchanged=14)
    stats.no_mappings = 2
    stats.no_seadex_entry = 3
    stats.seadex_unreachable = 4
    stats.no_releases = 5
    stats.unmonitored = 6
    stats.downloading = 1
    stats.downloaded = 2
    stats.imported = 3
    grab = GrabbedRow(title="T", coverage="S01", url="u", name="n", group="G")
    needs = NeedsActionRow(
        title="T2",
        coverage=None,
        group="G2",
        url=None,
        reason="private tracker",
        cause=NeedsActionCause.PRIVATE_ONLY,
    )
    stats.grabbed.append(grab)
    stats.needs_action.append(needs)

    tally = ArrRunTally.from_stats(stats)

    assert tally == ArrRunTally(
        checked=182,
        grabbed=(grab,),
        up_to_date=161,
        unchanged=14,
        no_seadex_entry=3,
        seadex_unreachable=4,
        no_releases=5,
        no_mappings=2,
        needs_action=(needs,),
        unmonitored=6,
        downloading=1,
        downloaded=2,
        imported=3,
        held_by_cap=0,
    )
    # Frozen tuples of owned facts: later stats mutation can't reach into the tally.
    stats.grabbed.append(grab)
    assert tally.grabbed == (grab,)


# Fields allowed to differ between ArrRunStats and ArrRunTally - deliberately empty.
_PARITY_ALLOWLIST: Final[frozenset[str]] = frozenset()


def test_arr_run_tally_field_names_mirror_arr_run_stats() -> None:
    stats_names = {f.name for f in dataclasses.fields(ArrRunStats)} - _PARITY_ALLOWLIST
    tally_names = {f.name for f in dataclasses.fields(ArrRunTally)} - _PARITY_ALLOWLIST
    assert stats_names == tally_names
