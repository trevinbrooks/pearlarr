"""A leg's state: the run context and stats, the grab and needs-action records, per-title flags, and entry titles."""

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
    """The id-form label an entry falls back to when no title resolved."""

    return f"AniList #{al_id}"


@dataclass(frozen=True, slots=True)
class Title:
    """An entry's resolved title: what the run shows, and the AniList title when one resolved."""

    al_id: int
    name: str
    """The AniList title, else the arr item's own title, else the id form."""
    names: TitleNames = field(default_factory=TitleNames)
    """The arr title and every AniList title, the placement's tie-break between look-alike files."""

    @property
    def anilist(self) -> str | None:
        """The first AniList title alone, the only value the cache stores as the entry's name."""

        return self.names.anilist[0] if self.names.anilist else None

    @property
    def resolved(self) -> bool:
        """False when nothing named the entry and `display` fell back to the id form."""

        return self.name != unresolved_label(self.al_id)


def title_from_names(al_id: int, anilist_titles: tuple[str, ...], arr_title: str) -> Title:
    """The title ladder: the first AniList title, else the arr item's own title, else the id form."""

    names = TitleNames(arr_title, tuple(title for title in anilist_titles if title))
    named = names.anilist[0] if names.anilist else arr_title
    return Title(al_id=al_id, name=named or unresolved_label(al_id), names=names)


@dataclass
class ArrRunStats:
    """The per-run tally rendered by the end-of-run summary."""

    checked: int = 0
    added: list[GrabbedRow] = field(default_factory=list[GrabbedRow])
    up_to_date: int = 0
    unchanged: int = 0
    no_seadex_entry: int = 0
    seadex_unreachable: int = 0
    """Lookups skipped this run, counted apart from `no_seadex_entry`."""
    no_releases: int = 0
    no_mappings: int = 0
    needs_action: list[NeedsActionRow] = field(
        default_factory=list[NeedsActionRow],
    )
    unmonitored: int = 0
    downloading: int = 0
    """Carried-over `QUEUED` records (a this-run grab stays in `added`)."""
    downloaded: int = 0
    """Carried-over `DOWNLOADED` records (a this-run grab stays in `added`)."""
    imported: int = 0
    """Carried-over `IMPORTED` records (a this-run grab stays in `added`)."""
    held_by_cap: int = 0
    """Titles with a grabbable release held past the run cap (re-checked and grabbed next run)."""


@dataclass
class PerTitleState:
    """Per-title scratch flags, reset at the top of each title."""

    held_by_cap: bool = False
    """A grabbable release was held by the run cap, so the title must not be cached as done."""
    private_only_skipped: bool = False
    """A private-only release forced a skip, so the title must not be cached as done."""
    private_only_groups: list[str] = field(default_factory=list[str])
    """Group names of the private-only skip, riding along for the run summary's "needs action" list."""
    private_only_outdated: bool = False
    """A stale owned private pick is held because only a fallback covers it."""
    unsupported_tracker_skipped: bool = False
    """The tracker has no parser, so the title must not be cached as done."""
    unsupported_tracker_groups: list[str] = field(default_factory=list[str])
    """Group names of the unsupported-tracker skip, riding along for the summary."""
    grab_failed_groups: list[str] = field(default_factory=list[str])
    """Groups whose release hit a contained grab failure (tracker or client down), so the title stays uncached."""
    skipped_groups: dict[SkipReason, list[str]] = field(default_factory=dict[SkipReason, list[str]])
    """The groups a hold stopped, by hold. While any are here the title stays uncached and is checked again next
    run, and a hand-import hold re-warns until the import covers it. `SkipReason.INPUT_UNREAD` also carries the
    groups whose placement waited on a failed read, though a url of such a group may have been grabbed. Filled by
    `add_held`."""
    metadata_unread: bool = False
    """An entry of the series was looked up without its AniList record, or a series was read without its TVDB
    orders, so the title stays uncached and re-checks. Nothing for the user to do."""
    title_name: str | None = None
    """Title of the entry currently being processed, so grabs and the summary can attribute what they grab."""
    entry_url: str | None = None
    """SeaDex URL of the entry currently being processed, so grabs and the summary can link what they grab."""
    entry_coverage: str | None = None
    """Coverage of the entry currently being processed, so grabs and the summary can attribute what they grab."""

    def add_skipped(self, skip_reason: SkipReason, groups: Sequence[str]) -> None:
        """List `groups` under `hold` in `held_groups`.

        No groups adds no key, so an empty `held_groups` means nothing held the title.
        """

        if groups:
            self.skipped_groups.setdefault(skip_reason, []).extend(groups)

    def add_private_only_notices(self, skips: "PrivateOnlyNotices") -> None:
        """Fold a planner private-only skip result onto this title's flags."""

        self.private_only_skipped |= skips.warned
        self.private_only_groups.extend(skips.groups)
        self.private_only_outdated |= skips.outdated


@dataclass
class ArrRunState:
    """Per-run state."""

    arr: Arr
    dry_run: bool = False
    import_wait_mode: ImportWaitMode = ImportWaitMode.OFF
    """The run's resolved wait mode (cli > config > default). `OFF` makes every pending-import path a no-op."""
    stats: ArrRunStats = field(default_factory=ArrRunStats)
    torrents_added: int = 0
    per_title: PerTitleState = field(default_factory=PerTitleState)
    """Per-title scratch flags, reassigned fresh at the top of each title so none leak into the next."""
    arr_title: str = ""
    """The arr item under scan, the display fallback for its ids when AniList names none."""
    started_monotonic: float | None = None
    """Run clock (monotonic, so an NTP or DST step cannot move it)."""
    counts_mark: CountsMark = field(default_factory=lambda: SeverityCounter().bound_mark())
    """Stamped at run start and diffed for the summary's issues row (an unstamped ctx diffs to zero)."""
    grabbed_this_run: dict[str, PendingImport] = field(default_factory=dict[str, PendingImport])
    """Records written THIS run, keyed by infohash for the run-list writes. The durable copies live in
    `cache_store`."""
    reacquired_keys: set[str] = field(default_factory=set[str])
    """Store-resident records this run re-saw (`ALREADY_ADDED`) without grabbing them, skipped by the snapshot
    but counted by the tally. Never also a `pending_imports` record."""
    pending_import_states: dict[str, PendingImportState] = field(default_factory=dict[str, PendingImportState])
    """Observed status of each carried-over record, keyed by infohash. Never a this-run grab, which stays
    `added`."""
    pause_owed: bool = False
    """An id's queries owe the `advanced.sleep_time` pause. The run loop sleeps it before the next id."""


def is_dry_run(state: ArrRunState, qbit: qbittorrentapi.Client | None) -> bool:
    """A run is a no-op preview when a dry run was requested OR qBittorrent is not configured."""

    return state.dry_run or qbit is None
