"""An arr run's state and stats, the grabbed and needs-action rows, per-title flags, and titles."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import qbittorrentapi

from .config import Arr
from .output import CountsMark, GrabbedRow, NeedsActionRow, SeverityCounter
from .pending_import import PendingImport, TitleNames
from .releases import SkipReason
from .wait_types import ImportWaitMode, PendingImportState

if TYPE_CHECKING:
    # Annotation-only (PerTitleState.add_private_only_notices). planner doesn't import us.
    from .planner import PrivateOnlyNotices


def unresolved_label(al_id: int) -> str:
    """The id-form label a title falls back to when no name resolved."""

    return f"AniList #{al_id}"


@dataclass(frozen=True, slots=True)
class Title:
    """A title's resolved name: what the run shows, and the AniList name when one resolved."""

    al_id: int
    name: str
    """The AniList name, else the arr item's own name, else the id form."""
    names: TitleNames = field(default_factory=TitleNames)
    """The arr item's name and every AniList name, the placement's tie-break between look-alike files."""

    @property
    def anilist(self) -> str | None:
        """The first AniList name alone, the only value the cache stores as the title's name."""

        return self.names.anilist[0] if self.names.anilist else None

    @property
    def resolved(self) -> bool:
        """False when nothing named the title and `name` fell back to the id form."""

        return self.name != unresolved_label(self.al_id)


def title_from_names(al_id: int, anilist_titles: tuple[str, ...], arr_title: str) -> Title:
    """The title, named by the first AniList name, else the arr item's own name, else the id form."""

    names = TitleNames(arr_title, tuple(title for title in anilist_titles if title))
    named = names.anilist[0] if names.anilist else arr_title
    return Title(al_id=al_id, name=named or unresolved_label(al_id), names=names)


@dataclass
class ArrRunStats:
    """The arr run's tally rendered by the arr run summary."""

    checked: int = 0
    grabbed: list[GrabbedRow] = field(default_factory=list[GrabbedRow])
    up_to_date: int = 0
    unchanged: int = 0
    no_seadex_entry: int = 0
    seadex_unreachable: int = 0
    """Lookups left undone with SeaDex unreachable this arr run, counted apart from `no_seadex_entry`."""
    no_releases: int = 0
    no_mappings: int = 0
    needs_action: list[NeedsActionRow] = field(
        default_factory=list[NeedsActionRow],
    )
    unmonitored: int = 0
    downloading: int = 0
    """Carried-over `DOWNLOADING` pending imports (a grab of this arr run stays in `grabbed`)."""
    downloaded: int = 0
    """Carried-over `DOWNLOADED` pending imports (a grab of this arr run stays in `grabbed`)."""
    imported: int = 0
    """Carried-over `IMPORTED` pending imports (a grab of this arr run stays in `grabbed`)."""
    held_by_cap: int = 0
    """Titles with a grabbable release held past the cap (re-checked and grabbed next run)."""


@dataclass
class PerTitleState:
    """Per-title scratch flags, reset at the top of each title."""

    held_by_cap: bool = False
    """A grabbable release was held by the cap, so the title must not be cached."""
    private_only_skipped: bool = False
    """A private-only release forced a skip, so the title must not be cached."""
    private_only_groups: list[str] = field(default_factory=list[str])
    """Group names of the private-only skip, carried for the arr run summary's "needs action" list."""
    private_only_outdated: bool = False
    """The arr's copy of a preferred private release is outdated, and the release is skipped because only a fallback
    covers it."""
    unsupported_tracker_skipped: bool = False
    """The tracker has no parser, so the title must not be cached."""
    unsupported_tracker_groups: list[str] = field(default_factory=list[str])
    """Group names of the unsupported-tracker skip, carried for the summary."""
    grab_failed_groups: list[str] = field(default_factory=list[str])
    """Groups whose release hit a contained grab failure (tracker or client unreachable), so the title stays
    uncached."""
    skipped_groups: dict[SkipReason, list[str]] = field(default_factory=dict[SkipReason, list[str]])
    """The groups a skip stopped, by skip reason. While any are here the title stays uncached and is checked again next
    run, and a hand-import skip re-warns until the import covers it. `SkipReason.PLACEMENT_INPUT_UNREAD` also carries
    the groups whose placement waited on a failed read, though a release of such a group may have been grabbed. Filled
    by `add_skipped`."""
    metadata_unread: bool = False
    """A title of the series was looked up without its AniList answer, or a series was read without its TVDB
    orders, so the title stays uncached and re-checks. Nothing for the user to do."""
    title_name: str | None = None
    """Name of the title being checked, so grabs and the summary can attribute what they grab."""
    entry_url: str | None = None
    """SeaDex URL of the title being checked, so grabs and the summary can link what they grab."""
    entry_coverage: str | None = None
    """Coverage of the title being checked, so grabs and the summary can attribute what they grab."""

    def add_skipped(self, skip_reason: SkipReason, groups: Sequence[str]) -> None:
        """List `groups` under `skip_reason` in `skipped_groups`.

        No groups adds no key, so an empty `skipped_groups` means nothing was skipped for the title.
        """

        if groups:
            self.skipped_groups.setdefault(skip_reason, []).extend(groups)

    def add_private_only_notices(self, skips: "PrivateOnlyNotices") -> None:
        """Merge a planner private-only skip result into this title's flags."""

        self.private_only_skipped |= skips.warned
        self.private_only_groups.extend(skips.groups)
        self.private_only_outdated |= skips.outdated


@dataclass
class ArrRunState:
    """One arr run's state."""

    arr: Arr
    dry_run: bool = False
    import_wait_mode: ImportWaitMode = ImportWaitMode.OFF
    """The arr run's resolved wait mode (cli > config > default). `OFF` makes every pending-import path a no-op."""
    stats: ArrRunStats = field(default_factory=ArrRunStats)
    torrents_added: int = 0
    per_title: PerTitleState = field(default_factory=PerTitleState)
    """Per-title scratch flags, reassigned fresh at the top of each title so none leak into the next."""
    arr_title: str = ""
    """The name of the arr item under scan, the display fallback for its titles when AniList names none."""
    started_monotonic: float | None = None
    """Arr run clock (monotonic, so an NTP or DST step cannot move it)."""
    counts_mark: CountsMark = field(default_factory=lambda: SeverityCounter().bound_mark())
    """Marked at the arr run's start and diffed for the summary's issues row (an unmarked state diffs to zero)."""
    grabbed_this_run: dict[str, PendingImport] = field(default_factory=dict[str, PendingImport])
    """Pending imports written THIS arr run, keyed by hash for the run-list writes. The durable copies live in
    `cache_store`."""
    reacquired_keys: set[str] = field(default_factory=set[str])
    """Stored pending imports this arr run saw again (`ALREADY_IN_QBIT`) without grabbing them, left out of the
    snapshot but counted by the tally. Never also in `grabbed_this_run`."""
    pending_import_states: dict[str, PendingImportState] = field(default_factory=dict[str, PendingImportState])
    """Observed status of each carried-over pending import, keyed by hash. Never a grab of this arr run, which stays
    `grabbed`."""
    pause_owed: bool = False
    """A title's queries owe the `advanced.sleep_time` pause. The arr run loop sleeps it before the next title."""


def is_dry_run(state: ArrRunState, qbit: qbittorrentapi.Client | None) -> bool:
    """A run is a dry run when one was requested OR qBittorrent is not configured."""

    return state.dry_run or qbit is None
