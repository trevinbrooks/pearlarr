"""Pure wait/outcome vocabulary for the wait-for-completion import path.

This module holds the domain shapes the wait side of the manual-import feature
speaks: the configurable `ImportWaitMode`, the durable
`PendingImport` record persisted through the cache store, the per-poll
probe/outcome enums the engine and views consume (`WaitOutcome`,
`PendingState`, `Outcome`), qBittorrent telemetry sanitization, the
basename/group normalizers every collaborator matches through, and the
download-path translation beside the path folders.

Everything here is deliberately side-effect free - no network, no disk, no
qBittorrent. The pure *planning* helpers (the probe verdicts, the episode
state, the placement and the names it reads, the grab-time seeds, the import
plan, quality/language resolution) live in the sibling planning modules that
`docs/architecture.md` lists. This module imports none of them.
"""

import math
import os
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Any, NamedTuple

from .seadex_types import RemotePathMapping, coerce_int
from .stamps import parse_stamp_or_none


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


def _path_segments(path: str) -> list[str]:
    """Split a path into non-empty segments for the boundary-aware compare (separators folded)."""

    return [segment for segment in fold_path_separators(path).split("/") if segment]


def translate_download_path(
    content_path: str,
    mappings: Sequence[RemotePathMapping],
    qbit_host: str | None,
) -> str:
    """Map a download-client path into Sonarr's filesystem view.

    Longest-`remotePath`-prefix match is the PRIMARY rule. Host equality (our
    `qbit_host`, folded here) only tiebreaks equally-long prefixes, and host
    inequality never excludes a mapping: Sonarr's `host` is the download-client
    host as SONARR configured it, routinely a different string from our
    qBittorrent host (localhost vs container name vs IP). Matching is per path
    segment, so it is separator-boundary-aware (`/downloads` never matches
    `/downloads-x/f`), tolerant of trailing slashes and Windows backslashes on
    either side, and case-insensitive, while the suffix keeps its ORIGINAL case
    (POSIX targets are case-sensitive). No match returns the path untranslated
    (the same-filesystem no-op). `content_path` is qBittorrent's, a folder or a
    single file.
    """

    content_segments = _path_segments(content_path)
    folded = [segment.casefold() for segment in content_segments]
    target_host = qbit_host.casefold() if qbit_host else None

    best_rank: tuple[int, bool] | None = None
    best_local = ""
    for mapping in mappings:
        if not mapping.remote_path or not mapping.local_path:
            continue
        remote = [segment.casefold() for segment in _path_segments(mapping.remote_path)]
        if not remote or folded[: len(remote)] != remote:
            continue
        host_matches = target_host is not None and (mapping.host or "").casefold() == target_host
        rank = (len(remote), host_matches)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_local = mapping.local_path

    if best_rank is None:
        return content_path
    base = best_local.rstrip("/\\") or "/"
    suffix = content_segments[best_rank[0] :]
    if not suffix:
        return base
    joined = "/".join(suffix)
    return f"/{joined}" if base == "/" else f"{base}/{joined}"


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


class AttemptKind(Enum):
    """Which kind of import attempt a poll is (see `ImportCompleter.import_completed`)."""

    POLL = auto()
    """An ordinary poll: a clean `importPending` is Sonarr's while its completed download handling is on."""

    DEADLINE = auto()
    """The attempt past the stall bound: steps in past a clean `importPending`, and an uncredited
    result graduates the row."""

    @property
    def at_deadline(self) -> bool:
        """The final attempt for the record."""

        return self is AttemptKind.DEADLINE


class EffectStatus(Enum):
    """One post-import cleanup effect's result (the retire contract, never rendered)."""

    DONE = auto()
    """Ran, or the goal was verifiably already met."""

    SKIPPED = auto()
    """Not applicable or deferred to a sibling: nothing outstanding for THIS record."""

    RETRY = auto()
    """Transient, the manager may re-attempt in-run."""

    FAILED = auto()
    """Given up this run, kept for a later run."""


class CleanupEffect(StrEnum):
    """One post-import cleanup effect. Each member IS its keep-warn noun."""

    CATEGORY_MOVE = "category move"
    QUEUE_REMOVAL = "queue removal"


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
    """The episode files are verified present. The record is dropped, or kept only for its post-import
    cleanup."""

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


class Deferral(Enum):
    """Why a poll waited on Sonarr's work rather than this record's (the monitor credits every reason but NONE)."""

    NONE = auto()
    """Not deferred: the wait is this record's own and burns its ready clock."""

    ISSUED = auto()
    """Our ManualImport was accepted this poll (credited once per record)."""

    IMPORT = auto()
    """An import is in flight: ours from an earlier poll, an unproven one, or Sonarr's own `importing` row."""

    BUSY = auto()
    """A foreign disk command holds the line."""


type FileEpisodeMap = Mapping[str, Sequence[int]]
"""A read view of normalized basename -> episode ids: a seed's map, a poll's placements, or both merged."""


@dataclass(frozen=True)
class ImportProbe:
    """The outcome of one `import_completed` poll.

    Lets the engine tell `imported` (every intended episode file is verified
    present) from `importing` (an import command was accepted but the copy is
    still running).
    """

    files_present: bool
    """Whether every intended episode file is verified present in Sonarr."""

    command_issued: bool
    """Whether a manual-import command covering this download was accepted."""

    attempted: bool = True
    """False only when no attempt ran (it raised, or no strategy is bound), so the record stays pending."""

    imported_count: int = 0
    """How many of the intended episodes already hold the recommended file"""

    target_count: int = 0
    """The intended episodes we mapped, 0 means the seed map is incomplete (indeterminate)."""

    deferral: Deferral = Deferral.NONE
    """Why this poll waited on Sonarr's work instead of this record's."""

    placements: FileEpisodeMap = field(default_factory=dict[str, list[int]])
    """Import-time placements this poll made (normalized basename -> ids), for the record seam to persist.
    Empty when the poll placed nothing."""

    exclusions: tuple[str, ...] = ()
    """On-disk files this poll proved never this record's to import (a sibling's slice, a duplicate), for the
    record seam to persist. Empty when the poll excluded nothing."""

    unmatched_files: tuple[str, ...] = ()
    """The download's on-disk files no episode claimed (one per distinct basename), when that is all the poll
    found. Non-empty only from `unmatched()`: `files_present` and `command_issued` False, `deferral` NONE."""

    @property
    def deferred(self) -> bool:
        """Whether this poll waited on Sonarr's work (the monitor credits the interval back)."""

        return self.deferral is not Deferral.NONE

    @classmethod
    def imported(cls, *, imported_count: int = 0, target_count: int = 0) -> "ImportProbe":
        """The intended files are verified present, with no import command of ours in flight."""

        return cls(
            files_present=True,
            command_issued=False,
            imported_count=imported_count,
            target_count=target_count,
        )

    @classmethod
    def waiting(
        cls,
        *,
        command_issued: bool = False,
        deferral: Deferral = Deferral.NONE,
        imported_count: int = 0,
        target_count: int = 0,
    ) -> "ImportProbe":
        """Not imported yet: poll again until the readiness deadline."""

        return cls(
            files_present=False,
            command_issued=command_issued,
            imported_count=imported_count,
            target_count=target_count,
            deferral=deferral,
        )

    @classmethod
    def unmatched(cls, names: tuple[str, ...]) -> "ImportProbe":
        """Nothing to import or verify: every file on disk matched no episode."""

        return cls(files_present=False, command_issued=False, unmatched_files=names)


LEAVE_PROBE = ImportProbe(attempted=False, files_present=False, command_issued=False)
"""The fail-open probe: no attempt ran, so leave the record pending for a later run."""


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


NO_PROGRESS = ImportProgress(0, 0, determinate=False)
"""The indeterminate zero reading a failed or strategy-less poll reports."""


class OutcomeCategory(Enum):
    """The wait view's ledger glyph + color."""

    SUCCESS = ("✔", "ok", "green")
    """The torrent imported."""

    DEFERRED = ("⚠", "~", "yellow")
    """Left pending for a later run (a download timeout, an import that hasn't landed yet, or files only a
    human can place). Not a failure, just unfinished."""

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
    SONARR_BUSY = ("sonarr busy", "Sonarr busy with a disk command; left pending", OutcomeCategory.DEFERRED, False)
    NOT_READY = ("not ready", "import not ready; left pending", OutcomeCategory.DEFERRED, False)
    UNMATCHED = ("unmatched", "no file matched an episode; import by hand in Sonarr", OutcomeCategory.DEFERRED, False)
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
    Outcome.SONARR_BUSY: PendingState.DOWNLOADED,
    Outcome.NOT_READY: PendingState.DOWNLOADED,
    Outcome.UNMATCHED: PendingState.DOWNLOADED,
    Outcome.ATTEMPT_FAILED: PendingState.DOWNLOADED,
    Outcome.NO_CONTENT_PATH: PendingState.DOWNLOADED,
    Outcome.DOWNLOAD_ERRORED: PendingState.ERRORED,
}


# qBittorrent reports a torrent with no meaningful ETA as 8_640_000 seconds
# (100 days), its "infinite" sentinel. Treat it (and anything at/above it) as
# "unknown" rather than rendering a nonsense countdown.
_QBIT_ETA_INFINITE = 8_640_000


class TorrentTelemetry(NamedTuple):
    """One info row's sanitized live telemetry: download fraction, speed, ETA, and byte counts."""

    progress: float
    speed_bps: int | None
    eta_s: int | None
    bytes_done: int | None
    bytes_total: int | None


_ZERO_TELEMETRY = TorrentTelemetry(0.0, None, None, None, None)
"""The unread placeholder (no client, or a transient qBittorrent error)."""


@dataclass(frozen=True)
class TorrentProbe:
    """One qBittorrent completion poll, with live download telemetry."""

    outcome: "WaitOutcome | None"
    """The terminal outcome this poll, or None while still downloading (or on a transient qB error)."""

    content_path: str | None
    """The completed download's path (COMPLETE only)."""

    telemetry: TorrentTelemetry = _ZERO_TELEMETRY
    """The poll's sanitized live reading (zeroed when nothing was read)."""

    observed: bool = True
    """False when qBittorrent could not actually be read (no client / a transient error)."""

    @property
    def ready_path(self) -> str | None:
        """The completed download's importable path: COMPLETE with a reported content path, else None."""

        if self.outcome is WaitOutcome.COMPLETE and self.content_path:
            return self.content_path
        return None


def sanitize_torrent_telemetry(row: object) -> TorrentTelemetry:
    """Fold one qBittorrent info row into sanitized telemetry (fields read best-effort off the row)."""

    frac = _as_float(getattr(row, "progress", None))
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))

    raw_speed = coerce_int(getattr(row, "dlspeed", None))
    speed_bps = raw_speed if raw_speed is not None and raw_speed > 0 else None

    raw_eta = coerce_int(getattr(row, "eta", None))
    eta_s = raw_eta if raw_eta is not None and 0 < raw_eta < _QBIT_ETA_INFINITE else None

    raw_total = coerce_int(getattr(row, "size", None))
    bytes_total = raw_total if raw_total is not None and raw_total > 0 else None
    raw_done = coerce_int(getattr(row, "completed", None))
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


def _normalized_names(names: Iterable[str]) -> set[str]:
    """Normalized-leaf SET, deliberately not a multiset."""

    return {normalized_leaf(name) for name in names}


def _normalized_map(entries: FileEpisodeMap) -> dict[str, list[int]]:
    """The map with every key normalized and zero ids dropped, minus the entries that leaves empty."""

    normalized: dict[str, list[int]] = {}
    for name, ids in entries.items():
        clean = [i for i in ids if i]
        if clean:
            normalized[normalized_leaf(name)] = clean
    return normalized


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


class OwnGroup(NamedTuple):
    """A torrent's own release group and the sizes its current listing carries (the trust policy's last vote)."""

    release_group: str
    sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EntryNames:
    """What names an entry: the arr series title and the AniList titles (English first, then romaji).

    The placement's tie-break between look-alike files scores on the words
    an AniList title adds to the series title, so both ride together.
    """

    series: str = ""
    anilist: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """The plain dict persisted inside a claim."""

        return {"series": self.series, "anilist": list(self.anilist)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "EntryNames":
        """Rebuild from the persisted dict (a record written before the names were kept reads empty)."""

        return cls(series=raw.get("series", ""), anilist=tuple(raw.get("anilist", [])))


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
    def merged(cls, parts: Iterable["GuardFacts"]) -> "GuardFacts":
        """Several entries' evidence as one: the groups unioned in order, the owned episodes concatenated."""

        entry_groups: dict[str, None] = {}
        stale_groups: dict[str, None] = {}
        owned: list[OwnedEpisode] = []
        for part in parts:
            entry_groups.update(dict.fromkeys(part.entry_groups))
            stale_groups.update(dict.fromkeys(part.stale_groups))
            owned.extend(part.owned_episodes)
        return cls(tuple(entry_groups), tuple(stale_groups), tuple(owned))

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "GuardFacts":
        """Rebuild from the persisted dict (missing keys fall back empty)."""

        return cls(
            entry_groups=tuple(raw.get("entry_groups", [])),
            stale_groups=tuple(raw.get("stale_groups", [])),
            owned_episodes=tuple(OwnedEpisode(pair[0], pair[1]) for pair in raw.get("owned_episodes", [])),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryClaim:
    """One AniList entry's claim on a torrent: the window its files are judged under and the slice it intends."""

    al_id: int
    """The AniList entry id, also the key of the entry's `guard_facts` row."""

    series_id: int
    """The Sonarr series id the entry's files belong to (0 on Radarr, which has no series windows)."""

    title: str | None
    """Display title (logging only)."""

    coverage: str | None
    """The entry's season/episode coverage when it claimed (e.g. `"S01 E01-E13"`, logging only)."""

    url: str | None
    """The SeaDex entry URL when it claimed, for the carried-over record's inline `link` line."""

    ordered_episode_ids: tuple[int, ...]
    """The entry's resolved episode ids in season order. Empty means unscoped: any id the series map holds."""

    names: EntryNames
    """The series and AniList titles the placement breaks ties by (see `EntryNames`)."""

    preowned_episode_ids: tuple[int, ...]
    """Claimed ids that already held a recommended file at the FIRST claim (the wait bar's net-out)."""

    slice_coverage: str | None
    """This claim's own ids as a coverage string (e.g. `"S02 E06"`)."""

    claimed_at: str
    """The claim's clock in `UPDATED_AT_STR_FORMAT`, stamped by the pipeline. The TTL drop ages the newest."""

    guards: GuardFacts = field(default_factory=GuardFacts)
    """The plan's overwrite-guard evidence. Not serialized: hydrated from `guard_facts` by `al_id`."""

    def admits(self, ep_id: int) -> bool:
        """Whether the claim's window holds `ep_id` (an unscoped claim admits every id)."""

        return not self.ordered_episode_ids or ep_id in self.ordered_episode_ids

    def to_json(self) -> dict[str, Any]:
        """The plain dict persisted inside the record (the guards ride their own row)."""

        return {
            "al_id": self.al_id,
            "series_id": self.series_id,
            "title": self.title,
            "coverage": self.coverage,
            "url": self.url,
            "ordered_episode_ids": list(self.ordered_episode_ids),
            "names": self.names.to_json(),
            "preowned_episode_ids": list(self.preowned_episode_ids),
            "slice_coverage": self.slice_coverage,
            "claimed_at": self.claimed_at,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, guards: Mapping[int, GuardFacts]) -> "EntryClaim":
        """Rebuild a claim from its persisted dict, fed the arr's guard rows by entry."""

        al_id = raw.get("al_id", 0)
        return cls(
            al_id=al_id,
            series_id=raw.get("series_id", 0),
            title=raw.get("title"),
            coverage=raw.get("coverage"),
            url=raw.get("url"),
            ordered_episode_ids=tuple(raw.get("ordered_episode_ids", [])),
            names=EntryNames.from_json(raw.get("names", {})),
            preowned_episode_ids=tuple(raw.get("preowned_episode_ids", [])),
            slice_coverage=raw.get("slice_coverage"),
            claimed_at=raw.get("claimed_at", ""),
            guards=guards.get(al_id, GuardFacts()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingImport:
    """One torrent awaiting import: its whole file map and every entry's claim on it."""

    infohash: str
    """The qBittorrent tracking key, lowercase (never None). Also the dedup `downloadId` sent to Sonarr."""

    release_group: str
    """The SeaDex release group (authoritative)."""

    is_dual_audio: bool
    """Whether the SeaDex release is dual-audio. Selects the dual vs. single language list."""

    seadex_files: tuple[str, ...]
    """SeaDex filenames, for our regex quality parse."""

    added_at: str
    """The torrent's birth in `UPDATED_AT_STR_FORMAT`: the pipeline's stamp at the add (Radarr's history floor)."""

    file_episode_map: FileEpisodeMap
    """Normalized basename -> Sonarr episode ids over the WHOLE torrent: the grab-time seeds plus the placements
    later imports made. Wrapped read-only at construction."""

    claims: tuple[EntryClaim, ...]
    """Every entry's claim on the torrent, accretion order (the import's window order)."""

    excluded_files: tuple[str, ...] = ()
    """Normalized basenames of video files this record knowably never imports (a collision-refused duplicate,
    a file every claim's window resolves outside), found at grab time or by an import poll."""

    release_sizes: tuple[int, ...] = ()
    """The grabbed listing's file sizes. Lets the import tell this release's own files from a stale
    same-group copy."""

    awaiting_cleanup: bool = False
    """The import verified but a post-import effect (category move / queue close) still needs to run."""

    def __post_init__(self) -> None:
        # Detach from the caller's map, then wrap read-only (the ids as tuples, so nothing inside mutates).
        object.__setattr__(
            self,
            "file_episode_map",
            MappingProxyType({name: tuple(ids) for name, ids in self.file_episode_map.items()}),
        )

    @property
    def own_group(self) -> OwnGroup:
        """The torrent's own group at its listing's sizes, the trust policy's last vote."""

        return OwnGroup(self.release_group, self.release_sizes)

    @property
    def series_ids(self) -> tuple[int, ...]:
        """The distinct series the claims span, claim order."""

        return tuple(dict.fromkeys(claim.series_id for claim in self.claims))

    @property
    def al_ids(self) -> tuple[int, ...]:
        """The distinct entries claiming the torrent, claim order."""

        return tuple(dict.fromkeys(claim.al_id for claim in self.claims))

    def claim_for(self, series_id: int) -> EntryClaim | None:
        """The first claim on `series_id`, if any."""

        return next((claim for claim in self.claims if claim.series_id == series_id), None)

    def claim_of(self, al_id: int) -> EntryClaim | None:
        """The entry's claim, if it holds one."""

        return next((claim for claim in self.claims if claim.al_id == al_id), None)

    def claim_holding(self, ep_id: int) -> EntryClaim | None:
        """The first claim whose window names `ep_id` (an unscoped claim names nothing), if any."""

        return next((claim for claim in self.claims if ep_id in claim.ordered_episode_ids), None)

    @property
    def display_label(self) -> str:
        """The cockpit/ledger/report row label: `titles · group[ · episode slices]`, every claim named."""

        base = " & ".join(dict.fromkeys(claim.title for claim in self.claims if claim.title)) or self.infohash
        if self.release_group:
            base = f"{base} · {self.release_group}"
        if slices := ", ".join(dict.fromkeys(c.slice_coverage for c in self.claims if c.slice_coverage)):
            base = f"{base} · {slices}"
        return base

    def seeded_map(self) -> dict[str, list[int]]:
        """The map as the placement reads it: keys normalized, zero ids dropped (see `_normalized_map`)."""

        return _normalized_map(self.file_episode_map)

    def target_ids(self) -> list[int]:
        """Our intended episode ids: the map's values in first-claim order."""

        return list(dict.fromkeys(ep_id for file_ids in self.file_episode_map.values() for ep_id in file_ids if ep_id))

    def resolved_ids(self) -> list[int]:
        """The episode set unplaced files assign into: the claims' windows in order, else the seeds' ids."""

        windows = list(dict.fromkeys(ep_id for claim in self.claims for ep_id in claim.ordered_episode_ids))
        return windows or sorted(self.target_ids())

    def preowned_ids(self) -> list[int]:
        """Every claim's preowned ids, claim order."""

        return list(dict.fromkeys(ep_id for claim in self.claims for ep_id in claim.preowned_episode_ids))

    def guards_for(self, series_id: int) -> GuardFacts:
        """The guard evidence of every claim on `series_id`, merged."""

        return GuardFacts.merged(claim.guards for claim in self.claims if claim.series_id == series_id)

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

    def unplaced_names(self) -> set[str]:
        """Normalized listing names the map does not cover and no exclusion claims: what a rebuild may still place."""

        return (
            _normalized_names(self.seadex_files)
            - _normalized_names(self.file_episode_map)
            - _normalized_names(self.excluded_files)
        )

    def with_placements(self, placements: FileEpisodeMap) -> "PendingImport":
        """The record with placements folded into its map (normalized, zero ids dropped), their names no longer excluded."""

        merged = {**_normalized_map(self.file_episode_map), **_normalized_map(placements)}
        excluded = tuple(name for name in self.excluded_files if name not in merged)
        return replace(self, file_episode_map=merged, excluded_files=excluded)

    def with_exclusions(self, names: Iterable[str]) -> "PendingImport":
        """The record with import-time exclusions appended, normalized, deduplicated, order kept."""

        merged = dict.fromkeys(normalized_leaf(name) for name in (*self.excluded_files, *names))
        return replace(self, excluded_files=tuple(merged))

    def with_claim(self, claim: EntryClaim) -> "PendingImport":
        """The record with `claim` joined and its cleanup flag cleared (new files to import make it active again).

        A re-flag by an entry already claiming replaces its claim, keeping the stored claim's first clock and
        preowned ids: the window follows the newest plan, the TTL and the wait bar's net-out never restart.
        """

        for index, stored in enumerate(self.claims):
            if stored.al_id == claim.al_id:
                kept = replace(claim, preowned_episode_ids=stored.preowned_episode_ids, claimed_at=stored.claimed_at)
                claims = (*self.claims[:index], kept, *self.claims[index + 1 :])
                break
        claims = (*self.claims, claim)
        return replace(self, claims=claims, awaiting_cleanup=False)

    def restamped(self, stamp: str) -> "PendingImport":
        """The record with its birth and every claim's clock set to `stamp` (a fresh add starts every clock)."""

        return replace(self, added_at=stamp, claims=tuple(replace(c, claimed_at=stamp) for c in self.claims))

    def newest_claimed_at(self) -> datetime | None:
        """The newest parseable claim stamp, the age the TTL drop reads (None when no claim stamp parses)."""

        stamps = [moment for claim in self.claims if (moment := parse_stamp_or_none(claim.claimed_at)) is not None]
        return max(stamps) if stamps else None

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain dict persisted under `pending_imports`."""

        return {
            "infohash": self.infohash,
            "release_group": self.release_group,
            "is_dual_audio": self.is_dual_audio,
            "seadex_files": list(self.seadex_files),
            "added_at": self.added_at,
            "file_episode_map": {name: list(ids) for name, ids in self.file_episode_map.items()},
            "claims": [claim.to_json() for claim in self.claims],
            "excluded_files": list(self.excluded_files),
            "release_sizes": list(self.release_sizes),
            "awaiting_cleanup": self.awaiting_cleanup,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, guards: Mapping[int, GuardFacts]) -> "PendingImport":
        """Rebuild a record from its persisted cache-store dict, each claim fed its entry's guard row."""

        return cls(
            infohash=raw.get("infohash", ""),
            release_group=raw.get("release_group", ""),
            is_dual_audio=raw.get("is_dual_audio", False),
            seadex_files=tuple(raw.get("seadex_files", [])),
            added_at=added_at_of(raw),
            file_episode_map=raw.get("file_episode_map", {}),
            claims=tuple(EntryClaim.from_json(claim, guards=guards) for claim in raw.get("claims", [])),
            excluded_files=tuple(raw.get("excluded_files", [])),
            release_sizes=tuple(raw.get("release_sizes", [])),
            awaiting_cleanup=is_awaiting_cleanup(raw),
        )


def is_awaiting_cleanup(raw: dict[str, Any]) -> bool:
    """The cleanup flag off a stored row's raw dict, sparing keys-only readers a rehydration."""

    return bool(raw.get("awaiting_cleanup"))


def added_at_of(raw: dict[str, Any]) -> str:
    """The birth stamp off a stored row's raw dict (`""` when unset or not a string)."""

    stamp = raw.get("added_at", "")
    return stamp if isinstance(stamp, str) else ""


def hydrate_pending(rows: Mapping[str, dict[str, Any]], guards: Mapping[int, GuardFacts]) -> dict[str, PendingImport]:
    """Rehydrate stored rows into records, keyed by infohash, each claim fed its entry's guard row."""

    return {key: PendingImport.from_json(raw, guards=guards) for key, raw in rows.items()}
