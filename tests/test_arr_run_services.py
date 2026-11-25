# pyright: strict
"""Tests for the run services the strategies call, and for the fake that stands in for them.

The title check tests read what a strategy hands `FakeRunServices`. These pin the details the real services print for
the lines the fake only records, and that the fake overrides every public member of the services. The cached-title
check and the services' other title interfaces are in `test_cached_skip.py`.
"""

from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import Arr
from pearlarr.output import Accent, StyledValue, TitleDetail

from .builders import make_run_deps
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
