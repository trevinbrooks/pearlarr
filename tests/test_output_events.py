# pyright: strict
"""Tests for the output-event vocabulary (``output.events``).

Pin the value model (frozen, equality-comparable facts), the severity mapping
that drives counts and sink floors, and the single RunStats -> RunTally
conversion site (S10: never two hand-maintained field lists).
"""

import dataclasses
from typing import Final

import pytest

from seadexarr.modules.config import Arr
from seadexarr.modules.log import EntryState
from seadexarr.modules.manual_import import Outcome, OutcomeCategory
from seadexarr.modules.output import (
    Accent,
    BootStepFinished,
    Diagnostic,
    EntryDetail,
    EntryHeader,
    GrabFact,
    GrabFailed,
    ItemStarted,
    LedgerRow,
    NeedsActionCause,
    NeedsActionFact,
    PlacedBy,
    ReleaseSkipped,
    RunTally,
    ScanStarted,
    ScopeId,
    ScopeKind,
    Severity,
    SkipReason,
    Span,
    StyledValue,
    TorrentGraduated,
    severity_of,
)
from seadexarr.modules.reporter import GrabRecord, NeedsActionKind, NeedsActionRecord, RunStats

# --- value model -----------------------------------------------------------------


def test_events_are_frozen() -> None:
    header = EntryHeader(state=EntryState.CHECKING, title="Frieren")
    field_name = "title"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(header, field_name, "other")


def test_events_compare_by_value() -> None:
    a = ReleaseSkipped(group="Okay-Subs", tracker="AB", reason=SkipReason.PRIVATE_ONLY)
    b = ReleaseSkipped(group="Okay-Subs", tracker="AB", reason=SkipReason.PRIVATE_ONLY)
    assert a == b


def test_styled_value_defaults_to_plain_with_no_spans() -> None:
    value = StyledValue("S01 E01-E28")
    assert value.accent is Accent.PLAIN
    assert value.spans == ()
    assert Span(0, 4, Accent.ACCENT).accent is Accent.ACCENT


def test_scope_id_is_a_value() -> None:
    assert ScopeId(ScopeKind.ENTRY, 3) == ScopeId(ScopeKind.ENTRY, 3)
    assert ScopeId(ScopeKind.ENTRY, 3) != ScopeId(ScopeKind.ITEM, 3)


# --- severity mapping --------------------------------------------------------------


def test_diagnostic_and_entry_detail_carry_their_own_severity() -> None:
    diag = Diagnostic(severity=Severity.ERROR, message="boom")
    detail = EntryDetail(label="skipped", value=StyledValue("x"), severity=Severity.WARNING)
    assert severity_of(diag) is Severity.ERROR
    assert severity_of(detail) is Severity.WARNING


def test_skip_reason_severity_marks_the_users_own_choice_info() -> None:
    assert SkipReason.PRIVATE_ONLY.severity is Severity.WARNING
    assert SkipReason.UNSUPPORTED_TRACKER.severity is Severity.WARNING
    assert SkipReason.TRACKER_NOT_SELECTED.severity is Severity.INFO
    skipped = ReleaseSkipped(group="G", tracker="Nyaa", reason=SkipReason.TRACKER_NOT_SELECTED)
    assert severity_of(skipped) is Severity.INFO


def test_grab_failed_is_a_warning() -> None:
    assert severity_of(GrabFailed(group="G", url="u", error="tracker down")) is Severity.WARNING


def test_boot_step_finished_tallies_info_regardless_of_outcome() -> None:
    """The outcome drives glyphs/styles only: a failed/deferred step's caller logs
    the problem itself, so an outcome-based tally would double-count it."""

    scope = ScopeId(ScopeKind.BOOT_STEP, 1)

    def finished(outcome: OutcomeCategory) -> BootStepFinished:
        return BootStepFinished(scope=scope, label="Reading config", outcome=outcome, detail=None, elapsed_s=0.1)

    assert severity_of(finished(OutcomeCategory.SUCCESS)) is Severity.INFO
    assert severity_of(finished(OutcomeCategory.DEFERRED)) is Severity.INFO
    assert severity_of(finished(OutcomeCategory.FAILED)) is Severity.INFO


def test_torrent_graduated_severity_follows_its_outcome_category() -> None:
    def graduated(outcome: Outcome) -> TorrentGraduated:
        return TorrentGraduated(label="T", outcome=outcome, files=None, waited_s=1.0)

    assert severity_of(graduated(Outcome.IMPORTED)) is Severity.INFO
    assert severity_of(graduated(Outcome.DOWNLOAD_TIMED_OUT)) is Severity.WARNING
    assert severity_of(graduated(Outcome.MISSING)) is Severity.ERROR


def test_structural_events_default_to_info() -> None:
    assert severity_of(ScanStarted(arr=Arr.SONARR, total=182)) is Severity.INFO
    assert severity_of(ItemStarted(arr=Arr.SONARR, index=1, total=2, title="X")) is Severity.INFO
    assert severity_of(LedgerRow(state=EntryState.IGNORED, label="AniList #1")) is Severity.INFO


def test_diagnostic_defaults_are_ambient_and_shared() -> None:
    diag = Diagnostic(severity=Severity.WARNING, message="m")
    assert diag.placed_by is PlacedBy.AMBIENT
    assert diag.origin == "app"
    assert diag.once_key is None
    assert diag.trace is None
    assert not diag.file_only


# --- RunTally: the single conversion site (S10) ------------------------------------


def test_run_tally_freezes_run_stats_whole() -> None:
    stats = RunStats(checked=182, up_to_date=161, cached=14)
    stats.no_mappings = 2
    stats.no_seadex_entry = 3
    stats.seadex_unreachable = 4
    stats.no_releases = 5
    stats.unmonitored = 6
    stats.queued = 1
    stats.importing = 2
    stats.imported = 3
    grab = GrabRecord(title="T", coverage="S01", url="u", name="n", group="G")
    needs = NeedsActionRecord(
        title="T2",
        coverage=None,
        group="G2",
        url=None,
        reason="private tracker",
        kind=NeedsActionKind.PRIVATE_ONLY,
    )
    stats.added.append(grab)
    stats.needs_action.append(needs)

    tally = RunTally.from_stats(stats)

    grab_fact = GrabFact(title="T", coverage="S01", url="u", name="n", group="G")
    needs_fact = NeedsActionFact(
        title="T2",
        coverage=None,
        group="G2",
        url=None,
        reason="private tracker",
        cause=NeedsActionCause.PRIVATE_ONLY,
    )
    assert tally == RunTally(
        checked=182,
        added=(grab_fact,),
        up_to_date=161,
        cached=14,
        no_seadex_entry=3,
        seadex_unreachable=4,
        no_releases=5,
        no_mappings=2,
        needs_action=(needs_fact,),
        unmonitored=6,
        queued=1,
        importing=2,
        imported=3,
    )
    # Frozen tuples of owned facts: later stats mutation can't reach into the tally.
    stats.added.append(grab)
    assert tally.added == (grab_fact,)


# Fields allowed to differ between RunStats and RunTally - deliberately empty (S10).
_PARITY_ALLOWLIST: Final[frozenset[str]] = frozenset()


def test_run_tally_field_names_mirror_run_stats() -> None:
    stats_names = {f.name for f in dataclasses.fields(RunStats)} - _PARITY_ALLOWLIST
    tally_names = {f.name for f in dataclasses.fields(RunTally)} - _PARITY_ALLOWLIST
    assert stats_names == tally_names


def test_needs_action_cause_member_names_mirror_needs_action_kind() -> None:
    # from_stats maps by member NAME (NeedsActionCause[kind.name]); a rename breaks here.
    assert [m.name for m in NeedsActionCause] == [m.name for m in NeedsActionKind]
