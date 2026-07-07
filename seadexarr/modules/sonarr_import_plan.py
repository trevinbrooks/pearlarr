"""Pure decision logic for Sonarr manual imports.

The deterministic planning vocabulary the import subsystem shares: the queue
verdict (:func:`classify_queue`) and the in-flight ManualImport guard, the
``(season, episode) -> id`` index and the episode-file status / never-overwrite
checks, the file -> episode assignment (:func:`assign_episode_ids`), the
per-file import plan (:func:`plan_import_files`), and the layered
quality/language resolution.

Everything here is deliberately side-effect free - no network, no disk, no
qBittorrent - so every rule can be unit-tested without I/O. This module imports
from :mod:`.manual_import` (the wait/outcome vocabulary and the normalizers);
the dependency is strictly one-way.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Any, NamedTuple

from .manual_import import normalize_group
from .seadex_types import (
    CommandResource,
    Language,
    ParsedFileInfo,
    Quality,
    QualityDefinition,
    QualityModel,
    QualitySource,
    Revision,
    SonarrEpisode,
    season_episode_key,
)


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a tracked download THIS poll.

    Derived purely from the queue records sharing a ``downloadId`` (a season pack
    has one record per episode), reading only ``trackedDownloadState``. "Already
    imported" is NOT decided here (a successful import is removed from the queue);
    the caller reads the episode files.

    ``WAIT`` -> something is genuinely in motion (downloading / importing); let
    Sonarr finish so we never race an in-flight import.
    ``PENDING_CLEAN`` -> any ``importPending`` record, regardless of its status or
    status messages: Sonarr parsed it and is waiting to import. With Completed
    Download Handling on it will import shortly; with CDH off it sits here forever -
    so the caller waits a grace, then forces our import. Stepping in on a still-
    pending record would race Sonarr's own import and double-import.
    ``STEP_IN`` -> Sonarr can't / won't progress it (``importBlocked`` / ``failed`` /
    ``failedPending`` / ``ignored``), or it isn't tracking the download at all
    (empty); drive our authoritative manual import.
    """

    WAIT = auto()
    PENDING_CLEAN = auto()
    STEP_IN = auto()


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


def classify_queue(states: list[str]) -> QueueVerdict:
    """Reduce a download's queue-record states to a single verdict for this poll.

    Side-effect free so the decision can be unit-tested without any HTTP.
    Priority, highest first:

      1. anything in motion (downloading / importing / ...) -> ``WAIT`` (never race
         an in-flight Sonarr import; re-evaluate next poll).
      2. any troubled record (``importBlocked`` / ``failed`` / ``failedPending`` /
         ``ignored``) -> ``STEP_IN``.
      3. any ``importPending`` -> ``PENDING_CLEAN``, regardless of its status or
         status messages. Sonarr is mid-import, so we wait for it to settle rather
         than step in - stepping in on a still-pending record races Sonarr's own
         import and double-imports the torrent.
      4. otherwise (empty because Sonarr isn't tracking it, all ``imported``, or an
         unknown state) -> ``STEP_IN``.

    Args:
        states (list[str]): Every matching queue record's ``trackedDownloadState``
            (matched + reduced by the caller; case preserved, folded here).

    Returns:
        QueueVerdict: The action this poll, BEFORE the episode-file "already
        imported" check the caller layers on top.
    """

    in_motion = False
    troubled = False
    clean_pending = False
    for raw_state in states:
        state = raw_state.casefold()
        # Bucket by state only: an importPending record must always wait
        # (routing it to STEP_IN would race Sonarr's import and double-import).
        if state in _QUEUE_IN_MOTION_STATES:
            in_motion = True
        elif state in _QUEUE_STEP_IN_STATES:
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


# A command counts as a ManualImport only under this name (Sonarr's command
# ``name``, compared case-folded), and only these statuses mean it is still
# running - a terminal command (completed / failed / aborted / cancelled /
# orphaned) is no longer in flight, so it never wedges a re-import.
_MANUAL_IMPORT_COMMAND_NAME = "manualimport"
_COMMAND_IN_FLIGHT_STATES = frozenset({"queued", "started"})


def _norm_path(path: str) -> str:
    """Normalize a path for a pure (no-disk) prefix compare: ``\\`` -> ``/``, folded."""

    return path.replace("\\", "/").casefold()


def manual_import_in_flight(
    commands: list[CommandResource],
    infohash: str,
    content_path: str,
    target_ep_ids: set[int],
) -> bool:
    """Whether a ManualImport already in flight covers THIS download.

    Pure, no I/O (mirrors :func:`classify_queue`): the strategy reads the
    ``/api/v3/command`` list and asks this whether to re-issue our import. A
    ManualImport command's copy is async and Sonarr drops the torrent from the
    regular queue while importing it server-side, so the queue alone reads
    "empty -> step in" and we'd stack a duplicate every poll. Matching the durable
    ``infohash`` against the still-running commands closes that loop, and because
    the match key lives in the command (not an in-memory id) it also survives a
    process restart - so a carried-over record re-driven on a LATER run won't
    re-stack a command run A POSTed that is still running.

    A command qualifies only when its ``name`` is ``ManualImport`` and its
    ``status`` is ``queued``/``started`` (a terminal command is not in flight).
    Such a command is taken to cover this download when:

      * PRIMARY: any of its files' ``download_id`` equals ``infohash``
        (case-insensitively) - the infohash a queue-driven import carries; this is
        the common, robust case.
      * FALLBACK (a folder / season-pack import whose files carry NO download id):
        any file path sits under ``content_path``, OR any file's episode id is one
        of ``target_ep_ids`` (our intended set). This is deliberately broad: a
        false positive only makes us WAIT (bounded by the import deadline, which
        forces through), whereas a missed match re-opens the duplicate-import loop.

    Args:
        commands (list[CommandResource]): The parsed ``/api/v3/command`` list.
        infohash (str): This download's infohash (the Sonarr download id).
        content_path (str): The qBittorrent ``content_path`` we import from, used
            for the no-download-id folder-import fallback.
        target_ep_ids (set[int]): Our intended episode ids, for the same fallback.

    Returns:
        bool: True when a still-running ManualImport already covers this download.
    """

    target_hash = infohash.casefold()
    content_prefix = _norm_path(content_path)
    for command in commands:
        name = (command.name or "").casefold()
        status = (command.status or "").casefold()
        if name != _MANUAL_IMPORT_COMMAND_NAME or status not in _COMMAND_IN_FLIGHT_STATES:
            continue
        file_hashes = {f.download_id.casefold() for f in command.files if f.download_id is not None}
        if target_hash in file_hashes:
            return True
        # Fallback only for a command whose files carry no download id at all (a
        # folder / season-pack import); a command that DOES carry download ids but
        # for a different torrent must not be swept up by a path/episode overlap.
        if file_hashes:
            continue
        for file in command.files:
            if file.path is not None and _norm_path(file.path).startswith(content_prefix):
                return True
            if any(ep_id in target_ep_ids for ep_id in file.episode_ids):
                return True
    return False


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
        by_key.setdefault(season_episode_key(ep.season_number, ep.episode_number), ep.id)
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


class EpisodeSnapshot(NamedTuple):
    """One poll's coherent view of a series: the fresh episode index plus the
    normalized recommended-group (overwrite-guard) set - fetched together, so
    consumers never mix state from two different polls."""

    episodes_by_id: dict[int, SonarrEpisode]
    recommended_groups: set[str]


def episode_file_statuses(
    target_ep_ids: list[int],
    snapshot: EpisodeSnapshot,
) -> dict[int, EpisodeFileStatus]:
    """Classify each intended target episode by its current on-disk file.

    Pure: reads only the snapshot's episode list and (normalized) set of
    recommended release groups for the series (every group we grabbed). "Already
    imported" is decided HERE from the episode files - not from the queue, since
    Sonarr drops an imported item from its queue almost immediately.

    Args:
        target_ep_ids (list[int]): The episode ids our mapping intends to fill.
        snapshot (EpisodeSnapshot): The same-poll episode index + recommended
            groups (via :func:`normalize_group`).

    Returns:
        dict[int, EpisodeFileStatus]: One status per de-duplicated target id.
    """

    statuses: dict[int, EpisodeFileStatus] = {}
    for ep_id in target_ep_ids:
        if ep_id in statuses:
            continue
        ep = snapshot.episodes_by_id.get(ep_id)
        if ep is None or not ep.episode_file_id:
            statuses[ep_id] = EpisodeFileStatus.ABSENT
            continue
        group = ep.episode_file.release_group if ep.episode_file else None
        if not group:
            statuses[ep_id] = EpisodeFileStatus.UNKNOWN_GROUP
        elif normalize_group(group) in snapshot.recommended_groups:
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

    return bool(statuses) and all(s is EpisodeFileStatus.RECOMMENDED for s in statuses.values())


def targets_needing_import(statuses: dict[int, EpisodeFileStatus]) -> set[int]:
    """The never-skip set: every intended id NOT already a recommended file.

    ABSENT / OTHER_GROUP / UNKNOWN_GROUP all need our import; only RECOMMENDED is
    excluded (it is done and must not be overwritten).
    """

    return {ep_id for ep_id, status in statuses.items() if status is not EpisodeFileStatus.RECOMMENDED}


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


_SXXEXX: re.Pattern[str] = re.compile(r"[Ss](\d{1,2})[\s._-]*[Ee](\d{1,3})")


def parse_se_from_filename(name: str) -> ParsedFileInfo | None:
    """Offline ``SxxExx`` fallback for when Sonarr's ``/parse`` is unreachable.

    Pure + regex-only: pulls a single ``SxxExx`` out of a leaf and returns it as a
    :class:`ParsedFileInfo` (season + episode). Returns None when the name carries
    no ``SxxExx`` (an absolute-numbered or unparseable leaf) - those are left to
    Sonarr's parse or the absolute-index leg, never guessed from a bare number.
    """

    m = _SXXEXX.search(name)
    if not m:
        return None
    return ParsedFileInfo(
        season_number=int(m.group(1)),
        episode_numbers=(int(m.group(2)),),
    )


@dataclass(frozen=True)
class EpisodeAssignment:
    """The outcome of assigning a torrent's on-disk files to resolved episode ids.

    ``assigned`` is ``normalized basename -> [episode id]`` for every file we could
    place with confidence (each id is in the resolved set and used exactly once).
    ``skipped`` lists the files we could NOT place - the caller warns on these and
    leaves them, rather than risk a wrong assignment (the chosen safe posture).
    """

    assigned: dict[str, list[int]]
    skipped: list[str]


def _exact_episode_ids(
    info: ParsedFileInfo | None,
    ep_id_map: Mapping[tuple[int, int], int],
    resolved_set: set[int],
    allow_unscoped: bool = False,
) -> list[int]:
    """The ids for a file's exact ``(season, episode)`` parse.

    Honors a file only when EVERY parsed episode resolves to a real series episode
    id (a partial hit means the file spans an episode we can't place, so it is
    treated as unplaced and skipped rather than half-imported). A missing season
    collapses to :data:`SONARR_MISSING_KEY`, matching :func:`build_episode_id_map`.

    Normally an id must also be inside ``resolved_set`` (our per-entry scope, which
    keeps an episode another preferred torrent owns out). When ``allow_unscoped`` is
    set - only when we have NO resolved set to scope against (an empty
    ``ordered_episode_ids``, e.g. a record grabbed before specials resolution
    populated it) - the membership check is dropped so a correctly-named file still
    lands on its real series episode instead of sticking forever. This trusts Sonarr
    for an UNAMBIGUOUS ``(season, episode)`` only; absolute numbers never reach here.
    """

    if info is None or not info.episode_numbers:
        return []
    ids: list[int] = []
    for episode in info.episode_numbers:
        ep_id = ep_id_map.get(season_episode_key(info.season_number, episode))
        if ep_id and (allow_unscoped or ep_id in resolved_set):
            ids.append(ep_id)
    if len(ids) != len(info.episode_numbers):
        return []
    return ids


def _has_no_signal(info: ParsedFileInfo | None) -> bool:
    """Whether a file carries no usable episode number at all (parse miss)."""

    return info is None or (not info.episode_numbers and not info.absolute_episode_numbers)


def assign_episode_ids(
    ordered_files: Sequence[str],
    parsed_by_file: Mapping[str, ParsedFileInfo | None],
    ordered_episode_ids: Sequence[int],
    ep_id_map: Mapping[tuple[int, int], int],
    allow_unscoped: bool | None = None,
) -> EpisodeAssignment:
    """Map a torrent's on-disk files to OUR resolved episode ids - names never override.

    The resolved set (``ordered_episode_ids``, season-sorted, lifted from the
    add-flow ``ep_list``) is authoritative; a release's own numbering is only ever
    used to *index into* it, never to decide identity. Two legs, in strict
    precedence, then skip:

    1. **Exact (season, episode):** a file whose parsed ``(season, episode)``
       resolves to an id *inside* the resolved set is placed there (handles
       correctly-named files Sonarr just couldn't match to the series, and
       per-season multi-season packs). With NO resolved set (an empty
       ``ordered_episode_ids``), this leg places against the live series episode
       map directly, so a correctly-named file still imports rather than sticking.
    2. **Absolute index:** the leftover files are mapped onto the leftover resolved
       ids by absolute number - but ONLY when every leftover file carries a single
       absolute number, the counts match 1:1, and no two files share an absolute
       (a shared absolute is the tell of per-title-restart numbering across a
       season boundary, e.g. a "... - 01" from two different sub-series, and is
       refused rather than scrambled). Handles mis-numbered specials and
       continuous absolute batches.
    3. **Skip:** anything still unplaced is returned in ``skipped`` for the caller
       to warn on - never guessed.

    Args:
        ordered_files (Sequence[str]): On-disk video files (normalized basenames)
            in SeaDex order - the order only fixes deterministic output.
        parsed_by_file (Mapping[str, ParsedFileInfo | None]): Series-agnostic parse
            per file (None when Sonarr's parse was unavailable and no SxxExx fell
            out of the name).
        ordered_episode_ids (Sequence[int]): The resolved episode ids, season-order.
        ep_id_map (Mapping[tuple[int, int], int]): ``(season, episode) -> id`` over
            ALL the series' episodes; membership in the resolved set does the
            scoping, so an exact parse outside our entry is rejected.
        allow_unscoped (bool | None): Scope-gate override; None (default) derives it
            from an empty ``ordered_episode_ids``. A caller that pre-subtracts
            already-placed ids passes it explicitly (off its FULL resolved set) so a
            fully-seeded record isn't mistaken for "no scope to enforce".

    Returns:
        EpisodeAssignment: the placed files and the skipped ones.
    """

    resolved = [i for i in ordered_episode_ids if i]
    resolved_set = set(resolved)
    # With NO resolved set to scope against, the exact leg falls back to the live
    # series episode map (a correctly-named file still lands on its real episode);
    # the absolute/positional legs stay disabled below (no leftover ids), so an
    # ambiguous file is skipped, never guessed. A caller that pre-subtracts seeded
    # ids passes allow_unscoped explicitly (off the FULL resolved set), so a fully-
    # seeded record's empty remainder doesn't masquerade as "no scope".
    if allow_unscoped is None:
        allow_unscoped = not resolved_set

    assigned: dict[str, list[int]] = {}
    used: set[int] = set()
    deferred: list[str] = []

    # Leg 1: exact (season, episode) - inside the resolved set, or against the live
    # series map when there is no set to scope against.
    for name in ordered_files:
        ids = _exact_episode_ids(
            parsed_by_file.get(name),
            ep_id_map,
            resolved_set,
            allow_unscoped,
        )
        if ids and not any(i in used for i in ids):
            assigned[name] = ids
            used.update(ids)
        else:
            deferred.append(name)

    # Leg 2: absolute index over the leftovers, only on a clean 1:1.
    leftover_ids = [i for i in resolved if i not in used]
    abs_by_file: dict[str, int] = {}
    for name in deferred:
        info = parsed_by_file.get(name)
        if info is not None and len(info.absolute_episode_numbers) == 1:
            abs_by_file[name] = info.absolute_episode_numbers[0]

    clean_absolute = (
        bool(abs_by_file)
        and len(abs_by_file) == len(deferred)  # every leftover has one absolute
        and len(abs_by_file) == len(leftover_ids)  # 1:1 with the leftover ids
        and len(set(abs_by_file.values())) == len(abs_by_file)  # no shared absolute (restart numbering)
    )

    skipped: list[str] = []
    if clean_absolute:
        for name, _abs in sorted(abs_by_file.items(), key=lambda kv: kv[1]):
            assigned[name] = [leftover_ids.pop(0)]
    elif (
        len(deferred) == 1
        and len(leftover_ids) == 1
        and _has_no_signal(
            parsed_by_file.get(deferred[0]),
        )
    ):
        # Degenerate positional: one leftover file, one leftover episode, and the
        # file carries NO usable number - it is that episode (the single-file
        # fallback). A file that parsed to a concrete episode OUTSIDE our set is
        # not swept up here; it stays skipped.
        assigned[deferred[0]] = [leftover_ids[0]]
    else:
        skipped = list(deferred)

    return EpisodeAssignment(assigned=assigned, skipped=skipped)


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


class ImportAction(StrEnum):
    """What :func:`plan_import_files` decided for one entry in OUR map.

    A ``StrEnum`` (so each member IS its rendered word, matching the
    :class:`PendingState` / :class:`QueueVerdict` / :class:`EpisodeFileStatus`
    style) - the consumer branches on a typed value instead of a magic string.
    Only ``IMPORT`` and ``MISSING`` drive behavior; the three "nothing to import
    for this file" members are kept distinct purely for reporting:

    ``IMPORT`` -> POST a manual import for this file.
    ``MISSING`` -> our map intends this file but it isn't on disk (surfaced, never
    silently skipped).
    ``SAMPLE`` -> a sample (never our intended file).
    ``ALREADY`` -> not needed, and Sonarr flagged an already-imported rejection.
    ``SKIP_DONE`` -> not needed (every target already holds a recommended file),
    with no Sonarr rejection.
    """

    IMPORT = "import"
    SKIP_DONE = "skip_done"
    SAMPLE = "sample"
    ALREADY = "already"
    MISSING = "missing"


@dataclass(frozen=True)
class ImportDecision:
    """One decision per entry in OUR authoritative map (the source of truth).

    Candidates only supply the on-disk ``path`` + rejection flags; the episode
    assignment is strictly ``episode_ids`` from our map. ``action`` is an
    :class:`ImportAction`.
    """

    basename: str
    action: ImportAction
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

    ``needing_import`` (derived from the EPISODE FILES via
    :func:`episode_file_statuses`) - not Sonarr's per-candidate already-imported
    rejection - is authoritative for whether we still want a file. Sonarr raises
    that rejection whenever the episode already holds *any* file on disk, including
    a non-recommended or unidentifiable-group one we flagged as still-needing
    replacement; honoring it as a skip there is the grab-then-skip bug (we grab a
    missing-group replacement, then Sonarr's "already imported" makes us skip
    importing it). So ``is_already_imported`` only yields ``already`` when NONE of
    the file's episodes still need us (every target already holds a recommended
    file - Sonarr and our episode-file check agree); when a target still needs us
    we import over it, as the never-skip invariant requires. ``is_sample`` still
    wins (a sample is never our intended file).

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
            decisions.append(ImportDecision(basename, ImportAction.MISSING, None, None, ep_ids))
            continue
        if candidate.is_sample:
            decisions.append(ImportDecision(basename, ImportAction.SAMPLE, candidate.path, None, []))
            continue
        import_ids = [i for i in ep_ids if i in needing_import]
        if not import_ids:
            # Nothing of ours still needs this file. Sonarr's already-imported
            # rejection and our episode-file done-check agree here, so report the
            # more specific ``ALREADY`` when Sonarr flagged it, else ``SKIP_DONE``.
            action = ImportAction.ALREADY if candidate.is_already_imported else ImportAction.SKIP_DONE
            decisions.append(
                ImportDecision(basename, action, candidate.path, None, ep_ids),
            )
            continue
        # A target still needs our file: import it over whatever is there, even
        # when Sonarr raised an already-imported rejection (that on-disk file is
        # the non-recommended / unidentifiable one we grabbed to replace).
        decisions.append(
            ImportDecision(
                basename,
                ImportAction.IMPORT,
                candidate.path,
                candidate.quality,
                import_ids,
            ),
        )
    return decisions


# Filename source tokens -> QualitySource, ordered most-specific first so a
# "BluRay Remux" name resolves to BLURAY_RAW (not BLURAY), "BD" counts as BluRay,
# and "WEB-DL" wins over a bare "WEB". A token that matches nothing leaves the
# source axis undetermined (None) - it is NEVER defaulted to WEB here; the
# configured default fills it.
_SOURCE_PATTERNS: list[tuple[re.Pattern[str], QualitySource]] = [
    (re.compile(r"remux", re.IGNORECASE), QualitySource.BLURAY_RAW),
    (re.compile(r"blu-?ray|\bbd\b", re.IGNORECASE), QualitySource.BLURAY),
    (re.compile(r"web-?dl", re.IGNORECASE), QualitySource.WEB),
    (re.compile(r"webrip", re.IGNORECASE), QualitySource.WEBRIP),
    (re.compile(r"hdtv", re.IGNORECASE), QualitySource.TELEVISION),
    (re.compile(r"\bdvd\b", re.IGNORECASE), QualitySource.DVD),
    (re.compile(r"\bweb\b", re.IGNORECASE), QualitySource.WEB),
]

_RESOLUTION_PATTERN: re.Pattern[str] = re.compile(r"(2160|1080|720|480)p", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedQuality:
    """Quality as two independent axes: ``source`` and ``resolution``.

    Either axis is ``None`` when it could not be authoritatively determined, which
    is what lets the quality decision layer the axes across Sonarr's parse, our
    filename parse, and the configured default (each fills only the axes the
    higher-precedence layers left ``None``). The resulting ``(source, resolution)``
    pair is matched against Sonarr's quality definitions to pick the real quality.
    """

    source: QualitySource | None = None
    resolution: int | None = None


def parse_quality_from_filename(filename: str) -> ParsedQuality:
    """Best-effort ``(source, resolution)`` parse of a SeaDex filename.

    Detects a resolution (``2160``/``1080``/``720``/``480``) and a source
    (Remux, BluRay, WEB-DL, WEBRip, WEB, HDTV, DVD), case-insensitively and
    independently. Either axis is ``None`` when not found - notably an
    unrecognized source is left ``None`` (NOT defaulted to WEB), so the configured
    default can fill it rather than the file being silently mislabeled.

    Args:
        filename (str): The SeaDex filename (or full path; only the text matched).

    Returns:
        ParsedQuality: The parsed axes; either may be ``None``.
    """

    res_match = _RESOLUTION_PATTERN.search(filename)
    resolution = int(res_match.group(1)) if res_match is not None else None

    source: QualitySource | None = None
    for pattern, candidate in _SOURCE_PATTERNS:
        if pattern.search(filename):
            source = candidate
            break
    return ParsedQuality(source=source, resolution=resolution)


def quality_axes_from_model(model: QualityModel | None) -> ParsedQuality:
    """The ``(source, resolution)`` axes of a Sonarr ``QualityModel``.

    Reads the canonical schema path ``model.quality.source`` /
    ``model.quality.resolution`` via ``.get()`` (the ``QualityModel``/``Quality``
    keys are ``NotRequired``, so subscripting is runtime-unsafe). An ``"unknown"``
    source or a ``0``/absent resolution maps to ``None`` (undetermined), so an
    unparsed candidate cleanly yields ``ParsedQuality()`` and falls through to the
    next precedence layer.

    Args:
        model (QualityModel | None): A candidate's in-context quality model.

    Returns:
        ParsedQuality: The structured axes Sonarr determined, each possibly None.
    """

    if not model:
        return ParsedQuality()
    quality = model.get("quality")
    if not quality:
        return ParsedQuality()
    resolution = quality.get("resolution")
    if not isinstance(resolution, int) or resolution <= 0:
        resolution = None
    return ParsedQuality(source=QualitySource.parse(quality.get("source")), resolution=resolution)


def quality_axes_from_name(
    name: str | None,
    quality_defs: list[QualityDefinition],
) -> ParsedQuality:
    """The ``(source, resolution)`` axes of a configured default quality NAME.

    Resolves the user's ``imports.default_quality`` (a Sonarr quality name like
    ``"Bluray-2160p"``) to its structured axes by matching it, case-insensitively,
    against the ``/api/v3/qualitydefinition`` list - so the default contributes a
    real ``(source, resolution)`` the decision fills gaps from. An unset name, or
    one that matches no definition, yields ``ParsedQuality()`` (no default).

    Args:
        name (str | None): The configured default quality name, if any.
        quality_defs (list[QualityDefinition]): The ``/api/v3/qualitydefinition``
            list.

    Returns:
        ParsedQuality: The default's axes, or empty when unset/unmatched.
    """

    if not name:
        return ParsedQuality()
    target = name.casefold()
    for definition in quality_defs:
        quality = definition.get("quality")
        if not quality:
            continue
        def_name = quality.get("name")
        if def_name is not None and def_name.casefold() == target:
            return quality_axes_from_model({"quality": quality})
    return ParsedQuality()


def derive_languages(
    is_dual_audio: bool,
    dual: list[str],
    single: list[str],
) -> list[str]:
    """Pick the import language list: ``dual`` when dual-audio, else ``single``."""

    return dual if is_dual_audio else single


def _find_definition(
    source: QualitySource,
    resolution: int,
    quality_defs: list[QualityDefinition],
) -> Quality | None:
    """The nested ``Quality`` whose ``(source, resolution)`` matches, or None.

    Scans the ``/api/v3/qualitydefinition`` list for the definition whose nested
    quality has the given structured source and resolution. ``(source, resolution)``
    is unique across Sonarr's standard definitions (the only near-collision, Raw-HD
    vs HDTV-1080p, differs by source), so the pair identifies the quality without
    ever matching on its display name.
    """

    for definition in quality_defs:
        quality = definition.get("quality")
        if quality is None:
            continue
        if quality.get("resolution") == resolution and QualitySource.parse(quality.get("source")) is source:
            return quality
    return None


def _candidate_revision(candidate_model: QualityModel | None) -> Revision:
    """The candidate's revision (proper/repack), or a fresh ``version 1`` default."""

    if candidate_model is not None:
        revision = candidate_model.get("revision")
        if revision is not None:
            return revision
    return {"version": 1, "real": 0, "isRepack": False}


def resolve_quality(
    sonarr: ParsedQuality,
    ours: ParsedQuality,
    default: ParsedQuality,
    quality_defs: list[QualityDefinition],
    candidate_model: QualityModel | None,
) -> QualityModel:
    """Resolve the final manual-import ``QualityModel`` - never omitted.

    The source and resolution axes are decided independently, each taking the
    first authoritative value in precedence order: Sonarr's parse, then our
    filename parse, then the configured default. When both axes are determined the
    quality definition matching the ``(source, resolution)`` pair is emitted, so
    the payload always carries a quality Sonarr actually defines (a valid id+name).
    A determined ``BLURAY_RAW``/``TELEVISION_RAW`` with no matching remux/raw
    definition at that resolution gracefully downgrades to ``BLURAY``/``TELEVISION``
    rather than failing.

    Crucially this never returns ``None`` and the caller never omits the quality:
    omitting it is exactly what made Sonarr crash in
    ``FileNameBuilder.AddQualityTokens``. When nothing resolves, Sonarr's own
    candidate model (valid by construction) is re-emitted verbatim; only if the
    candidate carries no quality at all is an explicit ``Unknown`` synthesized.

    Args:
        sonarr (ParsedQuality): Axes from Sonarr's candidate parse (highest).
        ours (ParsedQuality): Axes from our filename parse.
        default (ParsedQuality): Axes from the configured default quality.
        quality_defs (list[QualityDefinition]): The ``/api/v3/qualitydefinition``
            list to match against.
        candidate_model (QualityModel | None): Sonarr's in-context model, the
            last-resort verbatim fallback.

    Returns:
        QualityModel: The quality to POST; never omitted.
    """

    source = sonarr.source or ours.source or default.source
    resolution = sonarr.resolution or ours.resolution or default.resolution
    revision = _candidate_revision(candidate_model)

    if source is not None and resolution is not None:
        quality = _find_definition(source, resolution, quality_defs)
        if quality is None and source is QualitySource.BLURAY_RAW:
            quality = _find_definition(QualitySource.BLURAY, resolution, quality_defs)
        if quality is None and source is QualitySource.TELEVISION_RAW:
            quality = _find_definition(QualitySource.TELEVISION, resolution, quality_defs)
        if quality is not None:
            return {"quality": quality, "revision": revision}

    # No confident match: re-emit Sonarr's own candidate (valid by construction)
    # rather than omit the quality, else synthesize an explicit Unknown.
    if candidate_model is not None and candidate_model.get("quality"):
        return candidate_model
    unknown: Quality = {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0}
    return {"quality": unknown, "revision": revision}


def resolve_language_objects(
    names: list[str],
    lang_defs: list[Language],
) -> list[Language]:
    """Resolve configured language names to Sonarr ``{id, name}`` objects.

    Case-insensitive match against the ``/api/v3/language`` list, in request
    order; a name with no match is dropped rather than failing the import.
    """

    by_name: dict[str, Language] = {
        name.casefold(): definition for definition in lang_defs if isinstance((name := definition.get("name")), str)
    }
    resolved: list[Language] = []
    for name in names:
        definition = by_name.get(name.casefold())
        if definition is not None:
            resolved.append({"id": definition.get("id"), "name": definition.get("name")})
    return resolved
