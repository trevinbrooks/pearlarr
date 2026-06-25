"""Pure helpers for the wait-for-completion + series-pinned manual import path.

This module holds the *pure* domain vocabulary and decision helpers that drive
seadexarr's Sonarr manual-import feature: the configurable wait mode, the durable
:class:`PendingImport` record that is persisted in ``cache.json``, and the small
deterministic functions that map files to authoritative Sonarr episode ids,
parse a quality name out of a filename, and layer the quality/language/episode-id
decisions.

Everything here is deliberately side-effect free - no network, no disk, no
qBittorrent. The actual Sonarr HTTP calls (manual-import candidates, command
execution, quality/language resolution) and the qBittorrent completion poll live
in the engine and the Sonarr strategy; this module only owns the data shapes and
the rules they share, so the rules can be unit-tested without any I/O.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum, auto
from typing import Any, cast

from .seadex_types import (
    SONARR_MISSING_KEY,
    Language,
    QualityDefinition,
    QualityModel,
    Revision,
    SonarrEpisode,
)


def normalize_basename(name: str) -> str:
    """Normalize a filename leaf for cross-source matching.

    SeaDex/PocketBase JSON is NFC, but a macOS (APFS/HFS) disk scan can hand
    Sonarr the same name in NFD, so ``"é"`` (NFC) != ``"é"`` (NFD) under a plain
    ``dict`` lookup; trailing whitespace and case can drift too. Normalizing
    BOTH keyspaces (our SeaDex-recorded names and the on-disk leaves) through
    this one function is what lets our authoritative map match the files Sonarr
    actually found - so a grabbed file is never skipped over a unicode/whitespace
    mismatch.

    Args:
        name (str): A filename (basename or full path; only the text is folded).

    Returns:
        str: The NFC-normalized, stripped, case-folded leaf.
    """

    return unicodedata.normalize("NFC", name).strip().casefold()


def normalize_group(group: str) -> str:
    """Casefold a release group for recommended-set membership.

    Mirrors the casefold the planner uses to compare release groups, so the
    never-overwrite check and the grab-time group filter agree on what counts as
    "the same group". Blank/None is handled by the caller (only real groups are
    ever passed here).
    """

    return group.strip().casefold()


class ImportWaitMode(StrEnum):
    """When (if ever) the manual-import wait/import runs, resolved cli > config.

    A ``StrEnum`` so each member IS its config/CLI string (``ImportWaitMode.OFF``
    is and serializes as ``"off"``), matching the :class:`CacheField` style. The
    mode only controls *when* the import runs; all non-off modes share the same
    durable :class:`PendingImport` substrate.
    """

    OFF = "off"
    DEFERRED = "deferred"
    BLOCKING = "blocking"
    HYBRID = "hybrid"


class WaitOutcome(Enum):
    """The result of waiting on a torrent's completion in qBittorrent.

    ``COMPLETE`` -> import now; ``ERRORED``/``TIMED_OUT`` -> leave the record
    pending for a later retry (TTL eventually drops it); ``MISSING`` -> the
    torrent is gone from qBittorrent, so the record should be dropped.
    """

    COMPLETE = auto()
    ERRORED = auto()
    TIMED_OUT = auto()
    MISSING = auto()


class ImportReadiness(Enum):
    """The result of one Sonarr import attempt, telling the engine what to do.

    The strategy's ``import_completed`` returns this each poll so the engine's
    blocking wait loop knows whether to stop or keep polling:

    ``IMPORTED`` -> the files are imported (we queued a ManualImport, or Sonarr
    already handled them); drop the durable record.
    ``RETRY`` -> not ready yet (Sonarr hasn't seen/parsed the files, is mid-import,
    or a call failed transiently); poll again until the readiness deadline.
    ``LEAVE`` -> nothing we can import right now (no candidate maps to one of our
    episodes, or the attempt raised); leave the record pending for a later run.
    """

    IMPORTED = auto()
    RETRY = auto()
    LEAVE = auto()


class PendingState(StrEnum):
    """The current status of one carried-over pending import, for reporting.

    A ``StrEnum`` (so each member IS its rendered word) shared by the inline
    snapshot ledger row, the WaitView live region, and the end-of-run scoreboard
    counters, so one vocabulary describes a carried-over record everywhere:

    ``QUEUED`` -> still downloading (or never reached completion this poll); it
    waits.
    ``IMPORTING`` -> the download finished and an import command was accepted, but
    the episode files haven't landed yet (a remote-mount copy is in flight).
    ``IMPORTED`` -> the episode files are verified present; the record is dropped.
    ``ERRORED`` -> the download errored in qBittorrent; left for a later run.
    ``MISSING`` -> the torrent is gone from qBittorrent; the record is dropped.
    """

    QUEUED = "queued"
    IMPORTING = "importing"
    IMPORTED = "imported"
    ERRORED = "errored"
    MISSING = "missing"


def classify_pending(
    wait_outcome: "WaitOutcome | None",
    files_present: bool,
) -> PendingState:
    """Map a poll's outcome to a single carried-over :class:`PendingState`.

    Pure, no I/O (mirrors :func:`classify_queue`): the engine reads the torrent's
    completion outcome and the strategy's import probe, and this folds them into
    one status word. The verified-files check dominates the completed case, so a
    finished-but-not-yet-copied import always reads ``IMPORTING`` until the files
    actually land.

    Args:
        wait_outcome (WaitOutcome | None): The torrent's terminal outcome this
            poll, or ``None`` while it is still downloading.
        files_present (bool): Whether every intended episode file is verified
            present in Sonarr (the only signal that promotes to ``IMPORTED``).

    Returns:
        PendingState: ``MISSING`` / ``ERRORED`` for those terminal outcomes; while
        not COMPLETE -> ``QUEUED``; COMPLETE with files present -> ``IMPORTED``;
        COMPLETE without files present -> ``IMPORTING``.
    """

    if wait_outcome is WaitOutcome.MISSING:
        return PendingState.MISSING
    if wait_outcome is WaitOutcome.ERRORED:
        return PendingState.ERRORED
    if wait_outcome is not WaitOutcome.COMPLETE:
        return PendingState.QUEUED
    if files_present:
        return PendingState.IMPORTED
    return PendingState.IMPORTING


@dataclass(frozen=True)
class ImportProbe:
    """The outcome of one ``import_completed`` poll, richer than readiness alone.

    Lets the engine tell ``imported`` (every intended episode file is verified
    present) from ``importing`` (an import command was accepted but the copy is
    still running) - a distinction the bare :class:`ImportReadiness` collapses.

    Args:
        readiness (ImportReadiness): What the engine should do (drop / retry /
            leave), as before.
        files_present (bool): Whether every intended episode file is verified
            present in Sonarr. Only this promotes a record to ``imported``.
        command_issued (bool): Whether a manual-import command was accepted this
            poll (its copy may still be in flight - so not yet ``files_present``).
    """

    readiness: ImportReadiness
    files_present: bool
    command_issued: bool


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a tracked download THIS poll.

    Derived purely from the queue records sharing a ``downloadId`` (a season pack
    has one record per episode), reading ``trackedDownloadState`` AND
    ``trackedDownloadStatus`` AND whether ``statusMessages`` is populated - the
    last two disambiguate a healthy pending item from a stuck/blocked one, which
    the state alone cannot. "Already imported" is NOT decided here (a successful
    import is removed from the queue); the caller reads the episode files for that.

    ``WAIT`` -> something is genuinely in motion (downloading / importing); let
    Sonarr finish so we never race an in-flight import.
    ``PENDING_CLEAN`` -> a clean ``importPending`` (status ok, no messages):
    Sonarr parsed it and is waiting to import. With Completed Download Handling on
    it will import shortly; with CDH off it sits here forever - so the caller waits
    a grace, then forces our import.
    ``STEP_IN`` -> Sonarr can't / won't progress it (``importBlocked`` / ``failed`` /
    ``ignored`` / status ``error`` / a pending item carrying warnings), or it isn't
    tracking the download at all (empty); drive our authoritative manual import.
    """

    WAIT = auto()
    PENDING_CLEAN = auto()
    STEP_IN = auto()


@dataclass(frozen=True)
class QueueRecordView:
    """The fields of one Sonarr queue record the verdict actually depends on.

    A flat value object so the queue decision is pure and unit-testable: the
    strategy reduces each raw queue dict to this (case preserved; folded in
    :func:`classify_queue`) and drops records that don't match the download.

    Args:
        state (str): ``trackedDownloadState`` (e.g. ``importPending``).
        status (str): ``trackedDownloadStatus`` (``ok`` / ``warning`` / ``error``).
        has_messages (bool): Whether ``statusMessages`` was non-empty (a populated
            message array on a pending item means trouble, not progress).
    """

    state: str
    status: str
    has_messages: bool


# trackedDownloadState values (camelCase from Sonarr, compared case-folded) that
# mean Sonarr is genuinely working the download right now - wait rather than race
# it. ``queued``/``delay``/``paused`` are QueueStatus-ish transients Sonarr may
# surface in the same field; treat them as "still working" too.
_QUEUE_IN_MOTION_STATES = frozenset(
    {"downloading", "importing", "queued", "delay", "paused"},
)
_QUEUE_STEP_IN_STATES = frozenset(
    {"importblocked", "failed", "failedpending", "ignored"},
)


def classify_queue(records: list[QueueRecordView]) -> QueueVerdict:
    """Reduce a download's queue records to a single verdict for this poll.

    Side-effect free so the decision can be unit-tested without any HTTP.
    Priority, highest first:

      1. anything in motion (downloading / importing / ...) -> ``WAIT`` (never race
         an in-flight Sonarr import; re-evaluate next poll).
      2. any troubled record (``importBlocked`` / ``failed`` / ``ignored`` / status
         ``error``) -> ``STEP_IN``.
      3. any ``importPending`` -> ``PENDING_CLEAN``, regardless of its status or
         status messages. Sonarr is mid-import, so we wait for it to settle rather
         than step in - stepping in on a still-pending record races Sonarr's own
         import and double-imports the torrent.
      4. otherwise (empty because Sonarr isn't tracking it, all ``imported``, or an
         unknown state) -> ``STEP_IN``.

    Args:
        records (list[QueueRecordView]): Every queue record sharing the download's
            infohash (matched + reduced by the caller).

    Returns:
        QueueVerdict: The action this poll, BEFORE the episode-file "already
        imported" check the caller layers on top.
    """

    in_motion = False
    troubled = False
    clean_pending = False
    for record in records:
        state = record.state.casefold()
        status = record.status.casefold()
        if state in _QUEUE_IN_MOTION_STATES:
            in_motion = True
        elif state in _QUEUE_STEP_IN_STATES or status == "error":
            troubled = True
        elif state == "importpending":
            clean_pending = True

    if in_motion:
        return QueueVerdict.WAIT
    if troubled:
        return QueueVerdict.STEP_IN
    if clean_pending:
        return QueueVerdict.PENDING_CLEAN
    return QueueVerdict.STEP_IN


@dataclass(frozen=True)
class PendingImport:
    """A durable record of one added torrent awaiting a series-pinned import.

    Written at the add site (keyed by ``infohash`` in
    ``cache_store.data["pending_imports"]``) and read back to drive the manual
    import. It carries every field we have *authoritative* data for - the
    Sonarr ``series_id``, our own ``(basename -> episode ids)`` mapping, the
    SeaDex release group, dual-audio flag and season - so the import never has
    to trust Sonarr's blind title parse.

    Args:
        infohash (str): The qBittorrent tracking key (never None). Also the
            dedup ``downloadId`` sent to Sonarr.
        series_id (int): The Sonarr series id the files belong to.
        file_episode_map (dict[str, list[int]]): Basename -> authoritative
            Sonarr episode ids; the primary file->episode mapping. Repaired and
            extended in place at import time when a grabbed file wasn't parseable
            at grab time, so the map self-heals.
        episode_ids (list[int]): Flat fallback ids, used ONLY for a genuine
            single-file torrent whose one file our parse couldn't resolve.
        release_group (str): The SeaDex release group (authoritative).
        is_dual_audio (bool): Whether the SeaDex release is dual-audio; selects
            the dual vs. single language list.
        season_number (int | None): The single season, or None for multi-season
            / absolute-numbered packs.
        seadex_files (list[str]): SeaDex filenames, for our regex quality parse.
        seadex_sizes (list[int]): File sizes parallel to ``seadex_files``, for the
            order-based last-resort mapping of a fully-unparseable sequential pack.
        title (str | None): Display title (logging only).
        added_at (str): When the record was written, in
            :data:`UPDATED_AT_STR_FORMAT`, used for the TTL drop.
        coverage (str | None): The entry's season/episode coverage at grab time
            (e.g. ``"S01 E01-E13"``), so a carried-over record can render its
            ``files`` line inline next run without re-deriving it. Logging only.
        url (str | None): The SeaDex entry URL at grab time, for the carried-over
            record's inline ``link`` line. Logging only.
    """

    infohash: str
    series_id: int
    file_episode_map: dict[str, list[int]]
    episode_ids: list[int]
    release_group: str
    is_dual_audio: bool
    season_number: int | None
    seadex_files: list[str]
    seadex_sizes: list[int]
    title: str | None
    added_at: str
    coverage: str | None = None
    url: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain dict persisted under ``pending_imports``.

        Every field is JSON-native (str / int / bool / list / dict / None), so
        ``asdict`` is the whole serializer - and a field added to the dataclass
        can't be silently dropped from the persisted form.
        """

        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "PendingImport":
        """Rebuild a record from its persisted ``cache.json`` dict.

        Missing keys fall back to safe empties so a partially written or older
        record still rehydrates rather than raising.
        """

        return cls(
            infohash=raw.get("infohash", ""),
            series_id=raw.get("series_id", 0),
            file_episode_map=raw.get("file_episode_map", {}),
            episode_ids=raw.get("episode_ids", []),
            release_group=raw.get("release_group", ""),
            is_dual_audio=raw.get("is_dual_audio", False),
            season_number=raw.get("season_number"),
            seadex_files=raw.get("seadex_files", []),
            seadex_sizes=raw.get("seadex_sizes", []),
            title=raw.get("title"),
            added_at=raw.get("added_at", ""),
            coverage=raw.get("coverage"),
            url=raw.get("url"),
        )


def build_episode_id_map(ep_list: list[SonarrEpisode]) -> dict[tuple[int, int], int]:
    """Index Sonarr episodes by ``(season, episode)`` -> episode id.

    Mirrors the planner's keying: a missing ``season_number``/``episode_number``
    collapses to :data:`SONARR_MISSING_KEY` (an out-of-range value that never
    collides with a real key). On a duplicate key the first episode wins
    (``setdefault``), and episodes with a falsy id (0) are skipped - a 0 id can
    never be POSTed to Sonarr.

    Args:
        ep_list (list[SonarrEpisode]): Episodes parsed from ``/api/v3/episode``.

    Returns:
        dict[tuple[int, int], int]: ``(season, episode) -> ep.id`` for every
        episode carrying a real id.
    """

    by_key: dict[tuple[int, int], int] = {}
    for ep in ep_list:
        if not ep.id:
            continue
        season = ep.season_number if ep.season_number is not None else SONARR_MISSING_KEY
        episode = (
            ep.episode_number if ep.episode_number is not None else SONARR_MISSING_KEY
        )
        by_key.setdefault((season, episode), ep.id)
    return by_key


class EpisodeFileStatus(Enum):
    """How an intended target episode's CURRENT Sonarr file relates to ours.

    One read of the episode list drives both invariants - never overwrite a
    recommended file, never skip an episode we intended to import:

    ``ABSENT`` -> no file yet; import ours.
    ``RECOMMENDED`` -> already holds a file from a recommended group (ours, or
    another preferred torrent we grabbed for this series); it is done - do NOT
    overwrite it.
    ``OTHER_GROUP`` -> holds a file from a non-recommended group; import ours over
    it (the user's intended replacement).
    ``UNKNOWN_GROUP`` -> holds a file whose group Sonarr couldn't parse; import
    ours rather than trust an unidentifiable file as recommended.
    """

    ABSENT = auto()
    RECOMMENDED = auto()
    OTHER_GROUP = auto()
    UNKNOWN_GROUP = auto()


def episode_file_statuses(
    target_ep_ids: list[int],
    episodes_by_id: dict[int, SonarrEpisode],
    recommended_groups: set[str],
) -> dict[int, EpisodeFileStatus]:
    """Classify each intended target episode by its current on-disk file.

    Pure: reads only the fetched episode list and the (normalized) set of
    recommended release groups for the series (every group we grabbed). "Already
    imported" is decided HERE from the episode files - not from the queue, since
    Sonarr drops an imported item from its queue almost immediately.

    Args:
        target_ep_ids (list[int]): The episode ids our mapping intends to fill.
        episodes_by_id (dict[int, SonarrEpisode]): Current episodes keyed by id.
        recommended_groups (set[str]): Normalized recommended groups for the
            series (via :func:`normalize_group`).

    Returns:
        dict[int, EpisodeFileStatus]: One status per de-duplicated target id.
    """

    statuses: dict[int, EpisodeFileStatus] = {}
    for ep_id in target_ep_ids:
        if ep_id in statuses:
            continue
        ep = episodes_by_id.get(ep_id)
        if ep is None or not ep.episode_file_id:
            statuses[ep_id] = EpisodeFileStatus.ABSENT
            continue
        group = ep.episode_file.release_group if ep.episode_file else None
        if not group:
            statuses[ep_id] = EpisodeFileStatus.UNKNOWN_GROUP
        elif normalize_group(group) in recommended_groups:
            statuses[ep_id] = EpisodeFileStatus.RECOMMENDED
        else:
            statuses[ep_id] = EpisodeFileStatus.OTHER_GROUP
    return statuses


def all_targets_done(statuses: dict[int, EpisodeFileStatus]) -> bool:
    """True only when EVERY intended target already holds a recommended file.

    The "already imported / drop the record" signal. An UNKNOWN_GROUP or
    OTHER_GROUP file is NOT done (we still intend to import ours), so a present-
    but-unidentifiable file never makes us drop a record prematurely.
    """

    return bool(statuses) and all(
        s is EpisodeFileStatus.RECOMMENDED for s in statuses.values()
    )


def targets_needing_import(statuses: dict[int, EpisodeFileStatus]) -> set[int]:
    """The never-skip set: every intended id NOT already a recommended file.

    ABSENT / OTHER_GROUP / UNKNOWN_GROUP all need our import; only RECOMMENDED is
    excluded (it is done and must not be overwritten).
    """

    return {
        ep_id
        for ep_id, status in statuses.items()
        if status is not EpisodeFileStatus.RECOMMENDED
    }


def episode_ids_for_parsed(
    parsed: list[dict[str, Any]],
    ep_id_map: dict[tuple[int, int], int],
) -> list[int]:
    """Map Sonarr ``/parse`` ``(season, episode)`` dicts to OUR episode ids.

    The season/episode numbers come from Sonarr ``/parse`` (an internal tool of
    our pipeline), but the assignment stays ours: the ``(season, episode) -> id``
    index is built from the episode list OUR mapping selected. Numbers that don't
    resolve (or resolve to a 0 id) are dropped.
    """

    ids: list[int] = []
    for ep in parsed:
        season = ep.get("season")
        episode = ep.get("episode")
        if season is None or episode is None:
            continue
        ep_id = ep_id_map.get((season, episode))
        if ep_id:
            ids.append(ep_id)
    return ids


def build_authoritative_map(
    seeded_map: dict[str, list[int]],
    repaired: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Merge our grab-time map with import-time repairs into the final map.

    Both inputs are keyed by NORMALIZED basename (:func:`normalize_basename`) ->
    episode-id lists. ``seeded_map`` is what we computed at grab time; ``repaired``
    is what an import-time re-parse resolved for files the seed didn't cover. The
    seed wins on a key collision (it reflects the original per-torrent partition),
    and only non-empty id lists survive.

    Returns:
        dict[str, list[int]]: ``normalized_basename -> [episode_id]`` for every
        intended file we can authoritatively place.
    """

    merged: dict[str, list[int]] = {}
    for basename, ids in repaired.items():
        clean = [i for i in ids if i]
        if clean:
            merged[basename] = clean
    for basename, ids in seeded_map.items():
        clean = [i for i in ids if i]
        if clean:
            merged[basename] = clean
    return merged


@dataclass(frozen=True)
class CandidateFile:
    """An on-disk manual-import candidate, reduced to what planning needs.

    Built by the strategy from one raw ManualImportResource. The normalized
    ``basename`` is the match key against our authoritative map; ``path`` is what
    we POST; ``quality`` is reused if our own quality parse comes up empty; the
    two rejection flags fold Sonarr's per-file rejections into the plan.
    """

    basename: str
    path: str
    quality: QualityModel | None
    is_sample: bool
    is_already_imported: bool


@dataclass(frozen=True)
class ImportDecision:
    """One decision per entry in OUR authoritative map (the source of truth).

    Candidates only supply the on-disk ``path`` + rejection flags; the episode
    assignment is strictly ``episode_ids`` from our map. ``action`` is one of
    ``import`` / ``skip_done`` / ``sample`` / ``already`` / ``missing``.
    """

    basename: str
    action: str
    path: str | None
    quality: QualityModel | None
    episode_ids: list[int]


def plan_import_files(
    authoritative_map: dict[str, list[int]],
    candidates_by_basename: dict[str, CandidateFile],
    needing_import: set[int],
) -> list[ImportDecision]:
    """Decide, per intended file, whether/how to import it - strictly from our map.

    Iterates OUR map (never the candidates): a file Sonarr found that isn't in our
    map is never imported, and a file our map intends that isn't on disk is
    surfaced as ``missing`` (never silently skipped). For a present file both
    invariants are honored via ``needing_import`` (the non-recommended target
    set): a file whose every episode already holds a recommended release is
    ``skip_done`` (not overwritten); otherwise it is imported for exactly its
    needing-import episodes.

    Args:
        authoritative_map (dict[str, list[int]]): normalized basename -> our ids.
        candidates_by_basename (dict[str, CandidateFile]): on-disk files by key.
        needing_import (set[int]): episode ids still needing our file (from
            :func:`targets_needing_import`).

    Returns:
        list[ImportDecision]: one decision per map entry, in map order.
    """

    decisions: list[ImportDecision] = []
    for basename, ep_ids in authoritative_map.items():
        candidate = candidates_by_basename.get(basename)
        if candidate is None:
            decisions.append(ImportDecision(basename, "missing", None, None, ep_ids))
            continue
        if candidate.is_sample:
            decisions.append(ImportDecision(basename, "sample", candidate.path, None, []))
            continue
        if candidate.is_already_imported:
            decisions.append(ImportDecision(basename, "already", candidate.path, None, []))
            continue
        import_ids = [i for i in ep_ids if i in needing_import]
        if not import_ids:
            decisions.append(
                ImportDecision(basename, "skip_done", candidate.path, None, ep_ids),
            )
            continue
        decisions.append(
            ImportDecision(
                basename, "import", candidate.path, candidate.quality, import_ids,
            ),
        )
    return decisions


def resolve_wait_mode(
    cli_mode: ImportWaitMode | None,
    config_mode: ImportWaitMode | None,
) -> ImportWaitMode:
    """Resolve the effective wait mode with precedence cli > config > default.

    Args:
        cli_mode (ImportWaitMode | None): The ``--import-wait-mode`` CLI value.
        config_mode (ImportWaitMode | None): The configured ``import_wait_mode``.

    Returns:
        ImportWaitMode: ``cli_mode`` if set, else ``config_mode`` if set, else
        :attr:`ImportWaitMode.OFF`.
    """

    if cli_mode is not None:
        return cli_mode
    if config_mode is not None:
        return config_mode
    return ImportWaitMode.OFF


# Source/resolution -> Sonarr quality name, mirroring Sonarr's quality naming
# (e.g. "Bluray-2160p", "WEBDL-1080p"). Keyed by a normalized source token; the
# resolution suffix is appended from the matched resolution.
_SOURCE_TO_NAME: dict[str, str] = {
    "bluray": "Bluray",
    "remux": "Remux",
    "webdl": "WEBDL",
    "webrip": "WEBRip",
    "web": "WEBDL",
    "hdtv": "HDTV",
}

# Source patterns, ordered most-specific first so "WEB-DL" wins over a bare
# "WEB" and "Remux" is detected before "BluRay" in a "BluRay Remux" name.
_SOURCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("remux", re.compile(r"remux", re.IGNORECASE)),
    ("webdl", re.compile(r"web-?dl", re.IGNORECASE)),
    ("webrip", re.compile(r"webrip", re.IGNORECASE)),
    ("bluray", re.compile(r"blu-?ray", re.IGNORECASE)),
    ("hdtv", re.compile(r"hdtv", re.IGNORECASE)),
    ("web", re.compile(r"web", re.IGNORECASE)),
]

_RESOLUTION_PATTERN: re.Pattern[str] = re.compile(r"(2160p|1080p|720p|480p)", re.IGNORECASE)


def parse_quality_from_filename(filename: str) -> str | None:
    """Best-effort Sonarr quality name from a SeaDex filename.

    Detects a resolution (``2160p``/``1080p``/``720p``/``480p``) and a source
    (BluRay, Remux, WEB-DL, WEBRip, WEB, HDTV), case-insensitively, and joins
    them into a Sonarr-style name like ``"Bluray-2160p"`` or ``"WEBDL-1080p"``.
    A ``Remux`` source maps to ``"Remux-2160p"``; a bare ``WEB`` is treated as
    WEB-DL. When no source is recognized the name defaults to a ``WEBDL`` prefix
    (the most common anime case), so a resolution is always enough to produce a
    name. None is returned only when no resolution can be found at all.

    Args:
        filename (str): The SeaDex filename (or full path; only the name text
            is matched).

    Returns:
        str | None: A Sonarr quality name, or None when no resolution is found.
    """

    res_match = _RESOLUTION_PATTERN.search(filename)
    if res_match is None:
        return None
    resolution = res_match.group(1).lower()

    source_token: str | None = None
    for token, pattern in _SOURCE_PATTERNS:
        if pattern.search(filename):
            source_token = token
            break

    # No recognizable source: fall back to a WEBDL-style name (the most common
    # case for anime releases that omit an explicit source tag).
    source_name = _SOURCE_TO_NAME.get(source_token, "WEBDL") if source_token else "WEBDL"
    return f"{source_name}-{resolution}"


@dataclass(frozen=True)
class QualitySelection:
    """The outcome of the layered quality decision.

    ``name`` is the Sonarr quality name to resolve (the ours/default layers) and
    ``model`` is the candidate's in-context quality model dict to reuse verbatim
    (the sonarr layer); exactly one is set, or both are None for the unknown case
    (the caller warns). The winning layer isn't recorded - the consumer branches
    on which of ``name``/``model`` is set, never on a layer tag.
    """

    name: str | None
    model: QualityModel | None


def _quality_name(blob: object) -> str | None:
    """The ``name`` of a quality-ish object, or None when absent/non-str.

    Reads ``name`` off an arbitrary JSON object null-safely; used to walk the
    schema ``QualityModel.quality.name`` path and its cross-version variants.
    """

    if isinstance(blob, dict):
        name: object = cast("dict[str, Any]", blob).get("name")
        if isinstance(name, str) and name:
            return name
    return None


def _candidate_quality_name(candidate_quality: QualityModel | None) -> str | None:
    """Pull the nested quality name out of a Sonarr candidate quality model.

    The schema path is ``QualityModel.quality.name``, but Sonarr nests the name
    differently across endpoints/versions (``quality.quality.name``, or a bare
    ``name`` on the model itself). The model is therefore probed as an open
    mapping for these variant paths, which the strict ``QualityModel`` schema
    does not cover.
    """

    if not candidate_quality:
        return None
    quality = candidate_quality.get("quality")
    # Schema path + the ``quality.quality.name`` variant: ``quality`` is the
    # nested ``Quality`` whose ``name`` is canonical, but a variant double-nests
    # another quality object under it. ``Quality`` is read as an open mapping
    # here because that inner ``quality`` key is outside the schema.
    if isinstance(quality, dict):
        inner_name = _quality_name(cast("dict[str, Any]", quality).get("quality"))
        if inner_name is not None:
            return inner_name
        direct = _quality_name(quality)
        if direct is not None:
            return direct
    # Variant: a bare ``name`` on the model itself (outside the schema, which puts
    # ``name`` on the nested ``Quality``), so probe the model as an open mapping.
    return _quality_name(cast("dict[str, Any]", candidate_quality))


def select_quality(
    our_name: str | None,
    candidate_quality: QualityModel | None,
    default_name: str | None,
) -> QualitySelection:
    """Choose a quality with precedence ours > sonarr-in-context > default.

    Layers, in order:
      1. ``our_name`` (our regex parse of the SeaDex filename) -> carry the name.
      2. ``candidate_quality`` if present *and* its nested quality name is a real
         value (not missing and not ``"Unknown"``) -> reuse the model verbatim.
      3. ``default_name`` (the configured fallback) -> carry the name.
      4. otherwise both None (the caller warns; Sonarr re-grab risk).

    Args:
        our_name (str | None): Our parsed Sonarr quality name, if any.
        candidate_quality (QualityModel | None): The candidate's in-context model.
        default_name (str | None): The configured default quality name.

    Returns:
        QualitySelection: The value to carry forward (a name, a model, or neither).
    """

    if our_name:
        return QualitySelection(name=our_name, model=None)

    candidate_name = _candidate_quality_name(candidate_quality)
    if candidate_quality and candidate_name and candidate_name != "Unknown":
        return QualitySelection(name=None, model=candidate_quality)

    if default_name:
        return QualitySelection(name=default_name, model=None)

    return QualitySelection(name=None, model=None)


def derive_languages(
    is_dual_audio: bool,
    dual: list[str],
    single: list[str],
) -> list[str]:
    """Pick the language name list for the import.

    Args:
        is_dual_audio (bool): Whether the SeaDex release is dual-audio.
        dual (list[str]): Language names for a dual-audio release.
        single (list[str]): Language names for a single-audio release.

    Returns:
        list[str]: ``dual`` when ``is_dual_audio`` else ``single``.
    """

    return dual if is_dual_audio else single


def resolve_quality_model(
    name: str,
    quality_defs: list[QualityDefinition],
) -> QualityModel | None:
    """Resolve a Sonarr quality NAME to a manual-import ``QualityModel``.

    Looks the name up (case-insensitive) against the nested ``quality.name`` of
    each ``/api/v3/qualitydefinition`` entry and wraps the matched quality dict
    in the ``{"quality": ..., "revision": ...}`` shape Sonarr expects on a
    manual-import file. Returns None when no definition matches the name, so the
    caller can omit the quality key (Sonarr falls back to Unknown).

    Args:
        name (str): A Sonarr quality name (e.g. ``"WEBDL-1080p"``).
        quality_defs (list[QualityDefinition]): The ``/api/v3/qualitydefinition``
            list; each entry nests a ``quality`` dict with
            ``id``/``name``/``source``/``resolution`` (re-emitted verbatim).

    Returns:
        QualityModel | None: A ``QualityModel``, or None when no definition matches.
    """

    target = name.casefold()
    for definition in quality_defs:
        # The matched definition's nested quality is the schema ``Quality`` the
        # outgoing QualityModel re-emits verbatim.
        quality = definition.get("quality")
        if quality is None:
            continue
        quality_name = quality.get("name")
        if isinstance(quality_name, str) and quality_name.casefold() == target:
            revision: Revision = {"version": 1, "real": 0, "isRepack": False}
            return {"quality": quality, "revision": revision}
    return None


def resolve_language_objects(
    names: Sequence[object],
    lang_defs: list[Language],
) -> list[Language]:
    """Resolve language names to Sonarr ``{id, name}`` language objects.

    Matches each requested name (case-insensitive) against the
    ``/api/v3/language`` list and returns the matched ``{"id", "name"}`` objects
    in request order, skipping any name with no match (so an unknown configured
    language is simply dropped rather than failing the import).

    ``names`` is typed ``Sequence[object]`` rather than ``list[str]`` because the
    contract is "configured language names" but the *runtime* value is sourced
    from open YAML: a blank/malformed ``import_languages_*`` key can hand this a
    ``None`` or a non-string entry (a bare int, etc.). The per-entry ``isinstance``
    guard below is therefore live, not dead - the honest looser type is what keeps
    it necessary while still accepting the documented ``list[str]`` callers pass.

    Args:
        names (Sequence[object]): Language names to resolve (e.g. ``["Japanese"]``);
            non-string entries from a malformed config are skipped.
        lang_defs (list[Language]): The ``/api/v3/language`` list; each entry has
            ``id`` and ``name`` (a ``LanguageResource`` ``{id, name}``).

    Returns:
        list[Language]: The matched ``{"id", "name"}`` objects (unknown names
        omitted).
    """

    by_name: dict[str, Language] = {
        name.casefold(): definition
        for definition in lang_defs
        if isinstance((name := definition.get("name")), str)
    }
    resolved: list[Language] = []
    # ``names or []`` and the str guard keep a blank/None or malformed configured
    # language list from raising (a blank YAML value parses to None, which would
    # otherwise blow up on ``for name in None`` / ``None.casefold()``).
    for name in names or []:
        if not isinstance(name, str):
            continue
        definition = by_name.get(name.casefold())
        if definition is not None:
            resolved.append({"id": definition.get("id"), "name": definition.get("name")})
    return resolved
