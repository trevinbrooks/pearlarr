"""Pure wait/outcome vocabulary for the wait-for-completion import path.

This module holds the domain shapes the wait side of the manual-import feature
speaks: the configurable `ImportWaitMode`, the durable
`PendingImport` record persisted through the cache store, the per-poll
probe/outcome enums the engine and views consume (`WaitOutcome`,
`ImportReadiness`, `PendingState`, `Outcome`), qBittorrent
telemetry sanitization, and the basename/group normalizers every collaborator
matches through.

Everything here is deliberately side-effect free - no network, no disk, no
qBittorrent. The pure *planning* helpers (queue verdict, episode assignment,
import plan, quality/language resolution) live in `sonarr_import_plan`,
which imports from this module - never the other way around.
"""

import math
import os
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum, auto
from typing import Any, NamedTuple

from .seadex_types import coerce_int


def normalize_basename(name: str) -> str:
    """Normalize a filename leaf for cross-source matching.

    Args:
        name: A filename (basename or full path - only the text is folded).

    Returns:
        The NFC-normalized, stripped, case-folded leaf.
    """

    return unicodedata.normalize("NFC", name).strip().casefold()


def fold_path_separators(path: str) -> str:
    r"""Fold `\` to `/` for cross-platform comparison."""

    return path.replace("\\", "/")


def path_leaf(name: str) -> str:
    """A path's leaf with case and unicode preserved, for parser and display use."""

    return os.path.basename(fold_path_separators(name).rstrip("/"))


def normalized_leaf(name: str) -> str:
    """Fold a listing path or on-disk path to its normalized leaf (`path_leaf`, then `normalize_basename`)."""

    return normalize_basename(path_leaf(name))


def normalize_group(group: str) -> str:
    """Normalize a release group for comparison: strip whitespace/wrapping dashes, casefold."""

    return group.strip().strip("-").casefold()


def normalize_rg(name: str | None) -> str | None:
    """`normalize_group` with None-tolerance: None for a missing/blank name."""

    if not name:
        return None
    return normalize_group(name)


class ImportWaitMode(StrEnum):
    """Controls if and when the manual-import wait/import runs."""

    OFF = "off"
    """Disabled: no waiting, no pending-import records, no manual import."""

    DEFERRED = "deferred"
    """Never wait on a download: record this run's grabs and import earlier runs' finished downloads in one
    pass at the end of the run."""

    BLOCKING = "blocking"
    """Same as `hybrid`."""

    HYBRID = "hybrid"
    """The default: wait at the end of the run for downloads to finish, then import - this run's grabs and
    any download still pending from an earlier run."""


class WaitOutcome(Enum):
    """The result of waiting on a torrent's completion in qBittorrent."""

    COMPLETE = auto()
    """Import now."""

    ERRORED = auto()
    """Leave the record pending for a later retry (TTL eventually drops it)."""

    MISSING = auto()
    """The torrent is gone from qBittorrent, so the record should be dropped."""


class ImportReadiness(Enum):
    """The result of one Sonarr import attempt, telling the engine what to do.

    The strategy's `import_completed` returns this each poll so the engine's
    blocking wait loop knows whether to stop or keep polling.
    """

    IMPORTED = auto()
    """The files are imported (we queued a ManualImport, or Sonarr already handled them). Drop the durable
    record."""

    RETRY = auto()
    """Not ready yet (Sonarr hasn't seen/parsed the files, is mid-import, or a call failed transiently). Poll
    again until the readiness deadline."""

    LEAVE = auto()
    """The attempt raised (contained by the manager) or no strategy is bound - leave the record pending for a
    later run."""


class AttemptKind(Enum):
    """Which kind of import attempt a poll is (see `ImportCompleter.import_completed`)."""

    POLL = auto()
    """An ordinary poll: a clean `importPending` is Sonarr's while its completed download handling is on."""

    DEADLINE = auto()
    """The final in-bound attempt: steps in past a clean `importPending`."""

    @property
    def at_deadline(self) -> bool:
        """The final attempt for the record."""

        return self is AttemptKind.DEADLINE


class PendingState(StrEnum):
    """The current status of one carried-over pending import, for reporting.

    A `StrEnum` (so each member IS its rendered word) shared by the inline
    snapshot ledger row, the WaitView live region, and the end-of-run scoreboard
    counters, so one vocabulary describes a carried-over record everywhere.
    """

    QUEUED = "queued"
    """Still downloading."""

    DOWNLOADED = "downloaded"
    """The download finished and will be imported by the end-of-run pass."""

    IMPORTED = "imported"
    """The episode files are verified present. The record is dropped."""

    ERRORED = "errored"
    """The download errored in qBittorrent. Left for a later run."""

    MISSING = "missing"
    """The torrent is gone from qBittorrent. The record is dropped."""


def classify_pending(
    wait_outcome: "WaitOutcome | None",
    files_present: bool,
) -> PendingState:
    """Map a poll's outcome to a single carried-over `PendingState`.

    Args:
        wait_outcome: The torrent's terminal outcome this
            poll, or `None` while it is still downloading.
        files_present: Whether every intended episode file is verified
            present in Sonarr (the only signal that promotes to `IMPORTED`).
    """

    if wait_outcome is WaitOutcome.MISSING:
        return PendingState.MISSING
    if wait_outcome is WaitOutcome.ERRORED:
        return PendingState.ERRORED
    if wait_outcome is None:
        return PendingState.QUEUED
    if files_present:
        return PendingState.IMPORTED
    return PendingState.DOWNLOADED


@dataclass(frozen=True)
class ImportProbe:
    """The outcome of one `import_completed` poll, richer than readiness alone.

    Lets the engine tell `imported` (every intended episode file is verified
    present) from `importing` (an import command was accepted but the copy is
    still running) - a distinction the bare `ImportReadiness` collapses.
    """

    readiness: ImportReadiness
    """What the engine should do (drop / retry / leave)."""

    files_present: bool
    """Whether every intended episode file is verified present in Sonarr."""

    command_issued: bool
    """Whether a manual-import command covering this download was accepted."""

    imported_count: int = 0
    """How many of the intended episodes already hold the recommended file"""

    target_count: int = 0
    """The intended episodes we mapped, 0 means the seed map is incomplete (indeterminate)."""

    deferred: bool = False
    """Whether this poll waited on OUR OWN Sonarr work."""


class ImportProgress(NamedTuple):
    """A cheap, read-only files-landed count for the wait cockpit's import bar."""

    done: int
    total: int
    determinate: bool
    """True only when the persisted seed map covers every intended file, so `done`/`total` are the true full
    set. When False the importing row stays indeterminate (spinner only) and must NOT promote."""

    @property
    def files_present(self) -> bool:
        """Every intended file verified present."""

        return self.determinate and 0 < self.total <= self.done


class OutcomeCategory(Enum):
    """The wait view's ledger glyph + color."""

    SUCCESS = ("✔", "ok", "green")
    """The torrent imported."""

    DEFERRED = ("⚠", "~", "yellow")
    """Left pending for a later run (a download timeout, or an import that hasn't landed yet). Not a failure,
    just unfinished."""

    FAILED = ("✖", "x", "bold red")
    """The download errored or vanished from qBittorrent."""

    PENDING = ("·", "-", "grey50")
    """Left for the next run on purpose - the one-cycle check saw it and nothing is wrong."""

    glyph: str
    """The unicode glyph (`✔`/`⚠`/`✖`/`·`)."""

    ascii_glyph: str
    """The ASCII fallback, for dumb terminals / legacy Windows, where `✔` can't be encoded."""

    style: str
    """The rich style its ledger row is colored with."""

    def __init__(self, glyph: str, ascii_glyph: str, style: str) -> None:
        self.glyph = glyph
        self.ascii_glyph = ascii_glyph
        self.style = style

    def glyph_for(self, *, use_unicode: bool) -> str:
        """The ledger glyph: unicode `✔/⚠/✖/·` or its ASCII fallback."""

        return self.glyph if use_unicode else self.ascii_glyph


class Outcome(Enum):
    """A torrent's terminal result in the wait pass, with its rendering vocab."""

    IMPORTED = ("imported", "imported", OutcomeCategory.SUCCESS, True)
    MISSING = ("gone", "gone from qBittorrent", OutcomeCategory.FAILED, True)
    DOWNLOAD_ERRORED = ("errored", "download errored; left pending", OutcomeCategory.FAILED, False)
    DOWNLOAD_TIMED_OUT = ("timed out", "download timed out; left pending", OutcomeCategory.DEFERRED, False)
    NO_CONTENT_PATH = (
        "no path",
        "complete but no content path reported; left pending",
        OutcomeCategory.DEFERRED,
        False,
    )
    STILL_IMPORTING = ("unfinished", "still importing; left pending", OutcomeCategory.DEFERRED, False)
    NOT_READY = ("not ready", "import not ready; left pending", OutcomeCategory.DEFERRED, False)
    ATTEMPT_FAILED = ("failed", "import attempt failed; left pending", OutcomeCategory.DEFERRED, False)
    NOT_CHECKED = ("not checked", "qBittorrent unreachable; checked again next run", OutcomeCategory.DEFERRED, False)
    IMPORT_IN_PROGRESS = ("in progress", "import in progress; checked again next run", OutcomeCategory.PENDING, False)
    AWAITING_IMPORT = ("awaiting", "awaiting import; checked again next run", OutcomeCategory.PENDING, False)
    STILL_DOWNLOADING = ("downloading", "still downloading; checked again next run", OutcomeCategory.PENDING, False)

    word: str
    """The short ledger token (every one fits `STATE_WIDTH` = 11)."""

    detail: str
    """The longer human phrase the run report / notification use."""

    category: OutcomeCategory
    """The `OutcomeCategory` driving glyph + color + tally."""

    dropped: bool
    """Whether the engine removes the record from the durable store on this outcome.
    True for `IMPORTED` (files verified present) and `MISSING` (gone from qBittorrent)."""

    def __init__(
        self,
        word: str,
        detail: str,
        category: OutcomeCategory,
        dropped: bool,
    ) -> None:
        self.word = word
        self.detail = detail
        self.category = category
        self.dropped = dropped

    @property
    def style(self) -> str:
        """The rich style for this outcome's ledger row (from its category)."""

        return self.category.style

    def glyph(self, *, use_unicode: bool) -> str:
        """The leading ledger glyph: unicode `✔/⚠/✖/·` or its ASCII fallback."""

        return self.category.glyph_for(use_unicode=use_unicode)


# Each non-dropped Outcome's tally bucket. Dropped outcomes (IMPORTED, MISSING) leave the store,
# so a wait pass folds every record it leaves resident through exactly this table.
PENDING_STATE_FOR_OUTCOME: dict[Outcome, PendingState] = {
    Outcome.STILL_DOWNLOADING: PendingState.QUEUED,
    Outcome.DOWNLOAD_TIMED_OUT: PendingState.QUEUED,
    Outcome.NOT_CHECKED: PendingState.QUEUED,
    Outcome.AWAITING_IMPORT: PendingState.DOWNLOADED,
    Outcome.IMPORT_IN_PROGRESS: PendingState.DOWNLOADED,
    Outcome.STILL_IMPORTING: PendingState.DOWNLOADED,
    Outcome.NOT_READY: PendingState.DOWNLOADED,
    Outcome.ATTEMPT_FAILED: PendingState.DOWNLOADED,
    Outcome.NO_CONTENT_PATH: PendingState.DOWNLOADED,
    Outcome.DOWNLOAD_ERRORED: PendingState.ERRORED,
}


# qBittorrent reports a torrent with no meaningful ETA as 8_640_000 seconds
# (100 days), its "infinite" sentinel. Treat it (and anything at/above it) as
# "unknown" rather than rendering a nonsense countdown.
_QBIT_ETA_INFINITE = 8_640_000


@dataclass(frozen=True)
class TorrentProbe:
    """One qBittorrent completion poll, with live download telemetry."""

    outcome: "WaitOutcome | None"
    """The terminal outcome this poll, or None while still downloading (or on a transient qB error)."""

    content_path: str | None
    """The completed download's path (COMPLETE only)."""

    progress: float
    """0.0-1.0 download fraction (0.0 when unknown)."""

    speed_bps: int | None = None
    """Download speed in bytes/s, None when idle/unknown."""

    eta_s: int | None = None
    """qBittorrent's ETA in seconds, None when unknown/∞."""

    bytes_done: int | None = None
    """Bytes downloaded so far, None when unknown."""

    bytes_total: int | None = None
    """Total size in bytes, None when unknown."""

    observed: bool = True
    """False when qBittorrent could not actually be read (no client / a transient error)."""


class TorrentTelemetry(NamedTuple):
    """One info row's sanitized telemetry, field-for-field what `TorrentProbe` carries."""

    progress: float
    speed_bps: int | None
    eta_s: int | None
    bytes_done: int | None
    bytes_total: int | None


def sanitize_torrent_telemetry(
    progress: object,
    dlspeed: object,
    eta: object,
    completed: object,
    size: object,
) -> TorrentTelemetry:
    """Fold one qBittorrent info row's raw telemetry into sanitized fields."""

    frac = _as_float(progress)
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))

    raw_speed = coerce_int(dlspeed)
    speed_bps = raw_speed if raw_speed is not None and raw_speed > 0 else None

    raw_eta = coerce_int(eta)
    eta_s = raw_eta if raw_eta is not None and 0 < raw_eta < _QBIT_ETA_INFINITE else None

    raw_total = coerce_int(size)
    bytes_total = raw_total if raw_total is not None and raw_total > 0 else None
    raw_done = coerce_int(completed)
    bytes_done = raw_done if raw_done is not None and raw_done > 0 else None
    if bytes_done is not None and bytes_total is not None:
        bytes_done = min(bytes_done, bytes_total)
    return TorrentTelemetry(frac, speed_bps, eta_s, bytes_done, bytes_total)


def _as_float(value: object) -> float | None:
    """Best-effort float, or None for a non-numeric / NaN value."""

    if isinstance(value, (int, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return None if math.isnan(parsed) else parsed
    return None


class PendingKey(NamedTuple):
    """One pending record's composite identity: the torrent plus the entry claiming it."""

    infohash: str
    al_id: int

    @property
    def row_key(self) -> str:
        """The per-record string key snapshot rows carry (`TorrentView.key`)."""

        return f"{self.infohash}:{self.al_id}"


def _normalized_names(names: Iterable[str]) -> set[str]:
    """Normalized-leaf SET, deliberately not a multiset."""

    return {normalized_leaf(name) for name in names}


class SeedCoverage(NamedTuple):
    """A record's two seed trust levels over its grabbed video files."""

    mapped: bool
    """the map ALONE covers every file"""
    accounted: bool
    """map + knowably-excluded files"""


class OwnedEpisode(NamedTuple):
    """One grab-time ownership claim over an untagged on-disk file."""

    ep_id: int
    size: int


@dataclass(frozen=True)
class GuardFacts:
    """The plan's per-entry overwrite-guard evidence, carried whole."""

    entry_groups: tuple[str, ...] = ()
    """Pick groups the plan verified current by size."""

    stale_groups: tuple[str, ...] = ()
    """Pick groups the plan judged stale on disk."""

    owned_episodes: tuple[OwnedEpisode, ...] = ()
    """Episodes where the on-disk file has no release group, but matched a pick's listed size exactly."""

    @property
    def owned_sizes(self) -> dict[int, int]:
        """`owned_episodes` as the id -> size mapping the classifier reads."""

        return dict(self.owned_episodes)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "GuardFacts":
        """Rebuild from the persisted dict (missing keys fall back empty)."""

        return cls(
            entry_groups=tuple(raw.get("entry_groups", [])),
            stale_groups=tuple(raw.get("stale_groups", [])),
            owned_episodes=tuple(OwnedEpisode(pair[0], pair[1]) for pair in raw.get("owned_episodes", [])),
        )


@dataclass(frozen=True)
class PendingImport:
    """A durable record of one added torrent awaiting a series-pinned import."""

    infohash: str
    """The qBittorrent tracking key (never None). Also the dedup `downloadId` sent to Sonarr."""

    series_id: int
    """The Sonarr series id the files belong to."""

    al_id: int
    """The AniList entry this record's episode slice belongs to."""

    file_episode_map: dict[str, list[int]]
    """The primary file (Basename) to episode (Sonarr episode ids) mapping."""

    episode_ids: list[int]
    """Legacy read-only fallback: new seeds always write `[]`"""

    release_group: str
    """The SeaDex release group (authoritative)."""

    is_dual_audio: bool
    """Whether the SeaDex release is dual-audio. Selects the dual vs. single language list."""

    seadex_files: list[str]
    """SeaDex filenames, for our regex quality parse."""

    title: str | None
    """Display title (logging only)."""

    added_at: str
    """When the record was written, in `UPDATED_AT_STR_FORMAT`, used for the TTL drop."""

    coverage: str | None = None
    """The entry's season/episode coverage at grab time (e.g. `"S01 E01-E13"`)."""

    url: str | None = None
    """The SeaDex entry URL at grab time, for the carried-over record's inline `link` line."""

    slice_coverage: str | None = None
    """THIS record's own episode slice (e.g. `"S02 E06"`), from the grab-time map."""

    ordered_episode_ids: list[int] = field(default_factory=list[int])
    """The resolved episode ids for this entry, in season order"""

    excluded_files: list[str] = field(default_factory=list[str])
    """Normalized basenames of grabbed video files this record knowably never imports: a sibling entry's
    slice or a collision-refused duplicate."""

    guards: GuardFacts = field(default_factory=GuardFacts)
    """The plan's overwrite-guard evidence"""

    release_sizes: list[int] = field(default_factory=list[int])
    """The grabbed listing's file sizes. Lets the import tell this release's own files from a stale
    same-group copy."""

    preowned_episode_ids: list[int] = field(default_factory=list[int])
    """Target episodes that already held a recommended file at grab time."""

    @property
    def key(self) -> PendingKey:
        """The record's composite store/tracking key (see `PendingKey`)."""

        return PendingKey(self.infohash, self.al_id)

    @property
    def display_label(self) -> str:
        """The cockpit/ledger/report row label: `title · group[ · episode slice]`."""

        base = self.title or self.infohash
        if self.release_group:
            base = f"{base} · {self.release_group}"
        if self.slice_coverage:
            base = f"{base} · {self.slice_coverage}"
        return base

    def target_ids(self) -> list[int]:
        """Our intended episode ids: map values first-claim order, then the legacy fallback."""

        ids: list[int] = []
        seen: set[int] = set()
        for file_ids in self.file_episode_map.values():
            for ep_id in file_ids:
                if ep_id and ep_id not in seen:
                    seen.add(ep_id)
                    ids.append(ep_id)
        for ep_id in self.episode_ids:
            if ep_id and ep_id not in seen:
                seen.add(ep_id)
                ids.append(ep_id)
        return ids

    def seed_coverage(self) -> SeedCoverage:
        """Coverage from normalized-name SUPERSETS (never lengths): a healed extra can't fake it."""

        needed = _normalized_names(self.seadex_files)
        if not needed:
            return SeedCoverage(mapped=False, accounted=False)
        mapped_names = _normalized_names(self.file_episode_map)
        if mapped_names >= needed:
            return SeedCoverage(mapped=True, accounted=True)
        covered = mapped_names | _normalized_names(self.excluded_files)
        return SeedCoverage(mapped=False, accounted=covered >= needed)

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain dict persisted under `pending_imports`."""

        raw = asdict(self)
        del raw["guards"]
        return raw

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, guards: GuardFacts | None = None) -> "PendingImport":
        """Rebuild a record from its persisted cache-store dict."""

        return cls(
            infohash=raw.get("infohash", ""),
            series_id=raw.get("series_id", 0),
            al_id=raw.get("al_id", 0),
            file_episode_map=raw.get("file_episode_map", {}),
            episode_ids=raw.get("episode_ids", []),
            release_group=raw.get("release_group", ""),
            is_dual_audio=raw.get("is_dual_audio", False),
            seadex_files=raw.get("seadex_files", []),
            title=raw.get("title"),
            added_at=raw.get("added_at", ""),
            coverage=raw.get("coverage"),
            url=raw.get("url"),
            ordered_episode_ids=raw.get("ordered_episode_ids", []),
            slice_coverage=raw.get("slice_coverage"),
            excluded_files=raw.get("excluded_files", []),
            guards=guards or GuardFacts(),
            release_sizes=raw.get("release_sizes", []),
            preowned_episode_ids=raw.get("preowned_episode_ids", []),
        )
