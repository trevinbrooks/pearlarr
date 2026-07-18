"""Pure decision logic for Sonarr manual imports.

The deterministic planning vocabulary the import subsystem shares: the queue
verdict (`classify_queue`) and the in-flight ManualImport guard, the
`(season, episode) -> id` index and the episode-file status / never-overwrite
checks, the file -> episode assignment (`assign_episode_ids`), the
per-file import plan (`plan_import_files`), and the layered
quality/language resolution.

Side-effect free like `manual_import`, from which it imports the wait/outcome
vocabulary and the normalizers.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import NamedTuple

from .manual_import import normalize_group
from .seadex_types import (
    CommandResource,
    HistoryRecord,
    Language,
    ParsedEpisode,
    ParsedFileInfo,
    Quality,
    QualityDefinition,
    QualityModel,
    QualitySource,
    RemotePathMapping,
    Revision,
    SonarrEpisode,
    season_episode_key,
)


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a tracked download THIS poll.

    Derived purely from the queue records sharing a `downloadId` (a season pack
    has one record per episode), reading only `trackedDownloadState`. "Already
    imported" is NOT decided here (a successful import is removed from the queue).
    The caller reads the episode files.
    """

    WAIT = auto()
    """Something is genuinely in motion (downloading / importing). Let Sonarr finish so we never race an
    in-flight import."""

    PENDING_CLEAN = auto()
    """An `importPending` record: Sonarr parsed it and is waiting to import. See `classify_queue` rule 3."""

    STEP_IN = auto()
    """Sonarr can't / won't progress it (`importBlocked` / `failed` / `failedPending` / `ignored`), or it isn't
    tracking the download at all (empty). Drive our authoritative manual import."""


# trackedDownloadState values (camelCase from Sonarr, compared case-folded) that
# mean Sonarr is genuinely working the download right now - wait rather than race
# it. `queued`/`delay`/`paused` are QueueStatus-ish transients Sonarr may
# surface in the same field. Treat them as "still working" too.
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

      1. anything in motion (downloading / importing / ...) -> `WAIT` (never race
         an in-flight Sonarr import, re-evaluate next poll).
      2. any troubled record (`importBlocked` / `failed` / `failedPending` /
         `ignored`) -> `STEP_IN`.
      3. any `importPending` -> `PENDING_CLEAN`, regardless of its status or
         status messages. Sonarr has parsed it and is waiting to import, so we let
         it settle rather than step in - stepping in on a still-pending record
         races Sonarr's own import and double-imports the torrent.
      4. otherwise (empty because Sonarr isn't tracking it, all `imported`, or an
         unknown state) -> `STEP_IN`.

    Args:
        states: Every matching queue record's `trackedDownloadState`
            (matched + reduced by the caller, case preserved, folded here).

    Returns:
        The action this poll, BEFORE the episode-file "already imported" check
        the caller layers on top.
    """

    in_motion = False
    troubled = False
    clean_pending = False
    for raw_state in states:
        state = raw_state.casefold()
        if state in _QUEUE_IN_MOTION_STATES:
            in_motion = True
        elif state in _QUEUE_STEP_IN_STATES:
            troubled = True
        # Invariant: importPending always buckets as PENDING_CLEAN, never STEP_IN -
        # stepping in races Sonarr's own import and double-imports the files.
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
# `name`, compared case-folded), and only these statuses mean it is still
# running - a terminal command (completed / failed / aborted / cancelled /
# orphaned) is no longer in flight, so it never wedges a re-import.
_MANUAL_IMPORT_COMMAND_NAME = "manualimport"
_COMMAND_IN_FLIGHT_STATES = frozenset({"queued", "started"})


def _norm_path(path: str) -> str:
    r"""Normalize a path for a pure (no-disk) prefix compare: `\` -> `/`, folded."""

    return path.replace("\\", "/").casefold()


class ContentPaths(NamedTuple):
    """One download's import folder in both filesystem views the guard must match.

    A dead-tracked folder import POSTs the TRANSLATED (Sonarr-visible) path, so
    a command read back carries that form while the durable record carries the
    raw qBittorrent one - the guard needs both prefixes.
    """

    raw: str
    """The qBittorrent `content_path`."""

    sonarr_visible: str
    """The remote-path-mapped view (equal to `raw` when no translation was
    computed or none applies)."""


def manual_import_in_flight(
    commands: list[CommandResource],
    infohash: str,
    content_paths: ContentPaths,
    target_ep_ids: set[int],
) -> bool:
    """Whether a ManualImport already in flight covers THIS download.

    Pure, no I/O (mirrors `classify_queue`): the strategy reads the
    `/api/v3/command` list and asks this whether to re-issue our import. A
    ManualImport command's copy is async and Sonarr drops the torrent from the
    regular queue while importing it server-side, so the queue alone reads
    "empty -> step in" and we'd stack a duplicate every poll. Matching the durable
    `infohash` against the still-running commands closes that loop, and because
    the match key lives in the command (not an in-memory id) it also survives a
    process restart - so a carried-over record re-driven on a LATER run won't
    re-stack a command run A POSTed that is still running.

    A command qualifies only when its `name` is `ManualImport` and its
    `status` is `queued`/`started` (a terminal command is not in flight).
    Such a command is taken to cover this download when:

      * PRIMARY: any of its files' `download_id` equals `infohash`
        (case-insensitively) - the infohash a queue-driven import carries. This is
        the common, robust case.
      * FALLBACK (a folder / dead-tracked import whose files carry NO download
        id): any file path sits under either `content_paths` prefix, OR any
        file's episode id is one of `target_ep_ids` (our intended set). This is
        deliberately broad: a false positive only makes us WAIT (bounded by the
        import deadline, which forces through), whereas a missed match re-opens
        the duplicate-import loop.

    Args:
        commands: The parsed `/api/v3/command` list.
        infohash: This download's infohash (the Sonarr download id).
        content_paths: The import folder's raw + Sonarr-visible views, for
            the no-download-id fallback arm.
        target_ep_ids: Our intended episode ids, for the same fallback.

    Returns:
        True when a still-running ManualImport already covers this download.
    """

    target_hash = infohash.casefold()
    content_prefixes = {_norm_path(content_paths.raw), _norm_path(content_paths.sonarr_visible)}
    for command in commands:
        name = (command.name or "").casefold()
        status = (command.status or "").casefold()
        if name != _MANUAL_IMPORT_COMMAND_NAME or status not in _COMMAND_IN_FLIGHT_STATES:
            continue
        file_hashes = {f.download_id.casefold() for f in command.files if f.download_id is not None}
        if target_hash in file_hashes:
            return True
        # Fallback only for a command whose files carry no download id at all (a
        # folder / season-pack import). A command that DOES carry download ids but
        # for a different torrent must not be swept up by a path/episode overlap.
        if file_hashes:
            continue
        for file in command.files:
            if file.path is not None and any(_norm_path(file.path).startswith(p) for p in content_prefixes):
                return True
            if any(ep_id in target_ep_ids for ep_id in file.episode_ids):
                return True
    return False


# The episode-history events that map a re-appeared download to a queue-hidden
# tracked state (Imported / Failed / Ignored) - states Sonarr never runs its
# completed-download Check on, so `manualimport?downloadId=` NREs (HTTP 500)
# forever. Keyed by casefolded eventType, valued by the human label the hub
# note renders. `grabbed` (or none of the four) means genuinely Downloading.
_DEAD_TRACKED_HISTORY_EVENTS = {
    "downloadfolderimported": "imported",
    "downloadfailed": "failed",
    "downloadignored": "ignored",
}
_GRABBED_HISTORY_EVENT = "grabbed"


@dataclass(frozen=True, slots=True)
class DownloadHistoryVerdict:
    """What a download's newest relevant Sonarr history event says about its state."""

    dead_tracked: bool
    """True when Sonarr's history maps the download to a queue-hidden state it
    will never serve by id - import from its folder instead."""

    event: str | None = None
    """The dead-tracked event label (`imported`/`failed`/`ignored`), for the
    hub note. None when clean."""

    date: str | None = None
    """The dead-tracked event's raw ISO date, for the hub note."""


def classify_download_history(records: Sequence[HistoryRecord]) -> DownloadHistoryVerdict:
    """Classify a download's tracked state from its newest relevant history event.

    The episode-history mirror of Sonarr's own `GetStateFromHistory`
    latest-event rule: walk `records` (page 1, date-DESCENDING, as
    `history_for_download` returns them) and decide on the first event among
    `grabbed` / `downloadFolderImported` / `downloadFailed` /
    `downloadIgnored`, skipping every other type (e.g. `episodeFileDeleted`).
    Imported/failed/ignored -> dead-tracked. `grabbed` -> clean (a hash Sonarr
    itself re-grabbed after an old failure is genuinely Downloading). None of
    the four found -> clean. Probing only for prior imports would misroute
    re-grabs of previously-FAILED/IGNORED hashes - the same NREs apply there.
    """

    for record in records:
        event = record.event_type.casefold()
        if event == _GRABBED_HISTORY_EVENT:
            return DownloadHistoryVerdict(dead_tracked=False)
        label = _DEAD_TRACKED_HISTORY_EVENTS.get(event)
        if label is not None:
            return DownloadHistoryVerdict(dead_tracked=True, event=label, date=record.date)
    return DownloadHistoryVerdict(dead_tracked=False)


def _path_segments(path: str) -> list[str]:
    r"""Split a path into non-empty segments for the boundary-aware compare (`\` -> `/`)."""

    return [segment for segment in path.replace("\\", "/").split("/") if segment]


def translate_download_path(
    content_path: str,
    mappings: Sequence[RemotePathMapping],
    qbit_host: str | None,
) -> str:
    """Map a download-client path into Sonarr's filesystem view.

    Longest-`remotePath`-prefix match is the PRIMARY rule. Host equality only
    tiebreaks equally-long prefixes, and host inequality never excludes a
    mapping (Sonarr's `host` is the download-client host as SONARR configured
    it - routinely a different string from our qBittorrent host: localhost vs
    container name vs IP). Matching is per path segment, so it is
    separator-boundary-aware (`/downloads` never matches `/downloads-x/f`),
    tolerant of trailing slashes and Windows backslashes on either side, and
    case-insensitive - while the suffix keeps its ORIGINAL case (POSIX targets
    are case-sensitive). No match returns the path untranslated (the
    same-filesystem no-op).

    Args:
        content_path: The qBittorrent `content_path` (a folder or a single
            file).
        mappings: Sonarr's remote path mappings.
        qbit_host: Our qBittorrent hostname (casefolded upstream or not -
            folded here), for the tiebreak only.

    Returns:
        The Sonarr-visible path, or `content_path` unchanged.
    """

    content_segments = _path_segments(content_path)
    folded = [segment.casefold() for segment in content_segments]
    target_host = qbit_host.casefold() if qbit_host else None

    best_rank: tuple[int, bool] | None = None
    best_local = ""
    best_length = 0
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
            best_length = len(remote)

    if best_rank is None:
        return content_path
    base = best_local.rstrip("/\\") or "/"
    suffix = content_segments[best_length:]
    if not suffix:
        return base
    joined = "/".join(suffix)
    return f"/{joined}" if base == "/" else f"{base}/{joined}"


def build_episode_id_map(ep_list: list[SonarrEpisode]) -> dict[tuple[int, int], int]:
    """Index Sonarr episodes by `(season, episode)` -> episode id.

    Mirrors the planner's keying: a missing `season_number`/`episode_number`
    collapses to `SONARR_MISSING_KEY` (an out-of-range value that never
    collides with a real key). On a duplicate key the first episode wins
    (`setdefault`), and episodes with a falsy id (0) are skipped - a 0 id can
    never be POSTed to Sonarr.

    Args:
        ep_list: Episodes parsed from `/api/v3/episode`.

    Returns:
        `(season, episode) -> ep.id` for every episode carrying a real id.
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
    recommended file, never skip an episode we intended to import.
    """

    ABSENT = auto()
    """No file yet. Import ours."""

    RECOMMENDED = auto()
    """Already holds a file from a recommended group (ours, or another preferred torrent we grabbed for this
    series). It is done - do NOT overwrite it."""

    OTHER_GROUP = auto()
    """Holds a file from a non-recommended group. Import ours over it (the operator's intended replacement)."""

    UNKNOWN_GROUP = auto()
    """Holds a file whose group Sonarr couldn't parse. Import ours rather than trust an unidentifiable file as
    recommended."""


class EpisodeSnapshot(NamedTuple):
    """One poll's coherent view of a series: the fresh episode index plus the recommended-group set.

    The two fields are fetched together, so consumers never mix state from two different polls.
    """

    episodes_by_id: dict[int, SonarrEpisode]
    """The fresh episode index."""

    recommended_groups: set[str]
    """The normalized (overwrite-guard) recommended-group set."""


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
        target_ep_ids: The episode ids our mapping intends to fill.
        snapshot: The same-poll episode index + recommended
            groups (via `normalize_group`).

    Returns:
        One status per de-duplicated target id.
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

    ABSENT / OTHER_GROUP / UNKNOWN_GROUP all need our import. Only RECOMMENDED is
    excluded (it is done and must not be overwritten).
    """

    return {ep_id for ep_id, status in statuses.items() if status is not EpisodeFileStatus.RECOMMENDED}


def episode_ids_for_parsed(
    parsed: list[ParsedEpisode],
    ep_id_map: dict[tuple[int, int], int],
) -> list[int]:
    """Map Sonarr `/parse` `(season, episode)` pairs to OUR episode ids.

    The season/episode numbers come from Sonarr `/parse` (an internal tool of
    our pipeline), but the assignment stays ours: the `(season, episode) -> id`
    index is built from the episode list OUR mapping selected. Numbers that don't
    resolve (or resolve to a 0 id) are dropped.
    """

    ids: list[int] = []
    for ep in parsed:
        ep_id = ep_id_map.get((ep.season, ep.episode))
        if ep_id:
            ids.append(ep_id)
    return ids


_SXXEXX: re.Pattern[str] = re.compile(r"[Ss](\d{1,2})[\s._-]*[Ee](\d{1,3})")


def parse_se_from_filename(name: str) -> ParsedFileInfo | None:
    """Offline `SxxExx` fallback for when Sonarr's `/parse` is unreachable.

    Pure + regex-only: pulls a single `SxxExx` out of a leaf and returns it as a
    `ParsedFileInfo` (season + episode). Returns None when the name carries
    no `SxxExx` (an absolute-numbered or unparseable leaf) - those are left to
    Sonarr's parse or the absolute-index leg, never guessed from a bare number.
    Marked `offline` because the regex knows nothing about absolute numbers: a
    dual-numbered name ("S01E12 - 12") parsed here would otherwise launder its
    lost absolute into a "known" parse and blind the positional leg's tell.
    """

    m = _SXXEXX.search(name)
    if not m:
        return None
    return ParsedFileInfo(
        season_number=int(m.group(1)),
        episode_numbers=(int(m.group(2)),),
        offline=True,
    )


# One file plausibly spans a double or triple episode, never more.
_MATCHED_SPAN_CAP = 3


@dataclass(frozen=True)
class EpisodeAssignment:
    """The outcome of assigning a torrent's on-disk files to resolved episode ids."""

    assigned: dict[str, list[int]]
    """Normalized basename -> `[episode id]` for every file we could place with confidence (each id is in the
    resolved set and used exactly once)."""

    skipped: list[str]
    """The files we could NOT place - the caller warns on these and leaves them, rather than risk a wrong
    assignment (the chosen safe posture)."""


def _exact_episode_ids(
    info: ParsedFileInfo | None,
    ep_id_map: Mapping[tuple[int, int], int],
    resolved_set: set[int],
    allow_unscoped: bool = False,
) -> list[int]:
    """The ids for a file's exact `(season, episode)` parse.

    Honors a file only when EVERY parsed episode resolves to a real series episode
    id (a partial hit means the file spans an episode we can't place, so it is
    treated as unplaced and skipped rather than half-imported). A missing season
    collapses to `SONARR_MISSING_KEY`, matching `build_episode_id_map`.

    A name with no `(season, episode)` of its own falls back to Sonarr's
    series-MATCHED pairs (`matched_episodes` - how an absolute-only name gets
    its concrete season mapping), but ONLY under scope enforcement: membership
    in `resolved_set` is what keeps Sonarr's series match from deciding
    identity on its own, so the matched pairs never apply unscoped. A matched
    pair also carries Sonarr's own episode id when present. It must AGREE with
    our map's id for the same numbers, so a wrong-series title match whose
    numbers coincide with ours is refused. Junk duplicate pairs collapse to
    one claim before the every-pair check. A full-season parse (or more
    distinct matched claims than `_MATCHED_SPAN_CAP`) is never borrowed -
    Sonarr matches a bare "S01" name to the WHOLE season, and one junk file
    must not swallow it. A borrowed span must also cover the name's own
    absolute numbers: a "12-13" file whose match resolved only E12 would
    otherwise half-import.

    Normally an id must also be inside `resolved_set` (our per-entry scope, which
    keeps an episode another preferred torrent owns out). When `allow_unscoped` is
    set - only when we have NO resolved set to scope against (an empty
    `ordered_episode_ids`, e.g. a record grabbed before specials resolution
    populated it) - the membership check is dropped so a correctly-named file still
    lands on its real series episode instead of sticking forever. This trusts Sonarr
    for an UNAMBIGUOUS name-parsed `(season, episode)` only. Absolute numbers and
    matched pairs never reach the unscoped arm.
    """

    if info is None:
        return []
    claims: list[tuple[int | None, int, int | None]] = [
        (info.season_number, episode, None) for episode in info.episode_numbers
    ]
    borrowed = False
    # Full-season parses (bare "S01") match the whole season and never borrow.
    if not claims and not allow_unscoped and not info.full_season:
        claims = [(matched.season_number, matched.episode_number, matched.id) for matched in info.matched_episodes]
        borrowed = True
    claims = list(dict.fromkeys(claims))
    # The cap counts DISTINCT borrowed (season, episode) pairs (junk wire
    # duplicates collapse): a wider span is the season-pack shape sans flag.
    span = len({(season, episode) for season, episode, _ in claims})
    if not claims or (borrowed and span > _MATCHED_SPAN_CAP):
        return []
    # A borrowed span must also COVER the name's own absolutes: fewer matched
    # pairs than absolute numbers is a partially-resolved multi-episode file,
    # and placing the resolved half would half-import it.
    if borrowed and info.absolute_episode_numbers and span != len(set(info.absolute_episode_numbers)):
        return []
    ids: list[int] = []
    for season, episode, claimed_id in claims:
        ep_id = ep_id_map.get(season_episode_key(season, episode))
        if ep_id and claimed_id in (None, ep_id) and (allow_unscoped or ep_id in resolved_set):
            ids.append(ep_id)
    if len(ids) != len(claims):
        return []
    # The triple dedup keeps (s,e,None) and (s,e,id) apart. Collapse the
    # resolved ids so one episode never reaches the wire twice.
    return list(dict.fromkeys(ids))


def _has_no_signal(info: ParsedFileInfo | None) -> bool:
    """Whether a file's NAME carries no usable episode number at all (parse miss).

    Deliberately blind to `matched_episodes`: the degenerate single-file
    fallback keys on this, and an out-of-set Sonarr match must not veto the
    placement OUR resolution intends (Sonarr informs identity, never decides -
    in either direction). Cardinality is `_spans_multiple`'s question.
    """

    return info is None or (not info.episode_numbers and not info.absolute_episode_numbers)


def _signal_is_bogus(info: ParsedFileInfo, ep_id_map: Mapping[tuple[int, int], int]) -> bool:
    """Whether the name's numbers provably describe no episode of this series.

    A movie year read as SxxEyy ("Chronicle.2020" parsing S20E20) is a parse
    artifact, not identity: when EVERY name-parsed key misses the WHOLE series
    map and the name carries no absolutes, the signal is noise and the file
    counts as numberless. A key that resolves anywhere in the series is real
    evidence and is never downgraded.
    """

    if not info.episode_numbers or info.absolute_episode_numbers:
        return False
    return all(not ep_id_map.get(season_episode_key(info.season_number, episode)) for episode in info.episode_numbers)


def _spans_multiple(info: ParsedFileInfo) -> bool:
    """Whether Sonarr's evidence says the file holds MORE than one episode.

    Identity never comes from the series match alone, but cardinality is a
    different question: a full-season parse or a multi-pair match means
    placing the file as ONE episode is a structural half-import, so the
    degenerate single-file fallback refuses it. A single pair - agreeing,
    disagreeing, or out of set - stays the fallback's business.
    """

    pairs = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
    return info.full_season or len(pairs) > 1


def _natural_key(name: str) -> str:
    """Digit-aware sort key ("sp10" sorts after "sp2"): zero-pad digit runs."""

    return re.sub(r"\d+", lambda match: match.group().zfill(12), name)


def assign_episode_ids(
    ordered_files: Sequence[str],
    parsed_by_file: Mapping[str, ParsedFileInfo | None],
    ordered_episode_ids: Sequence[int],
    ep_id_map: Mapping[tuple[int, int], int],
    allow_unscoped: bool | None = None,
) -> EpisodeAssignment:
    """Map a torrent's on-disk files to OUR resolved episode ids - names never override.

    The resolved set (`ordered_episode_ids`, season-sorted, lifted from the
    add-flow `ep_list`) is authoritative. A release's own numbering is only ever
    used to *index into* it, never to decide identity. Two legs, then two
    narrow fallbacks, in strict precedence, then skip:

    1. **Exact (season, episode):** a file whose parsed `(season, episode)`
       resolves to an id *inside* the resolved set is placed there (handles
       correctly-named files Sonarr just couldn't match to the series, and
       per-season multi-season packs). An absolute-only name borrows Sonarr's
       series-matched pair (`matched_episodes`) under the same in-set scoping,
       so a batch spanning entries places exactly - never positionally. With NO
       resolved set (an empty `ordered_episode_ids`), this leg places against
       the live series episode map directly, so a correctly-named file still
       imports rather than sticking (name-parsed pairs only, see
       `_exact_episode_ids`).
    2. **Absolute index:** the leftover files are mapped onto the leftover resolved
       ids by absolute number - but ONLY when every leftover file carries a single
       absolute number, the counts match 1:1, every parse in the batch is known
       (a None parse - or the offline regex stand-in for one, which is blind
       to absolutes - could be hiding a duplicate, so the leg fails closed), and
       no two files ANYWHERE in the batch share an absolute (a shared absolute
       is the tell of per-title-restart numbering across a season boundary,
       e.g. a "... - 01" from two different sub-series, or a v2 of an
       already-placed file. Every absolute of every parse supplied is counted -
       seeded files included - so neither a leg-1 placement nor an earlier
       poll's can hide one sharer from this leg). Handles mis-numbered
       specials and continuous absolute batches. Anything short of that clean
       shape is refused rather than scrambled.
    3. **Single-file fallback:** one leftover file onto one leftover id, when
       the name carries no number at all - or only a provably-bogus key, one
       missing the WHOLE series map with no absolutes (a movie year read as
       SxxEyy) - and Sonarr's matched evidence doesn't span multiple episodes.
    4. **Ordered zip:** a pristine numberless batch (every parse known, real,
       numberless, and single-span, with the parse map covering EXACTLY the
       deferred files - so nothing was placed or seeded) zips natural name
       order onto the leftover ids on a 1:1 count (the "Special 1..N" shape).
       Any numbered, bogus-keyed, seeded, or ambiguous file refuses it whole.
    5. **Skip:** anything still unplaced is returned in `skipped` for the caller
       to warn on - never guessed.

    Args:
        ordered_files: On-disk video files (normalized basenames)
            in SeaDex order - the order only fixes deterministic output.
        parsed_by_file: Series-agnostic parse per file (None when Sonarr's
            parse was unavailable and no SxxExx fell out of the name). May
            cover MORE files than `ordered_files` - the shared-absolute tell
            scans every parse supplied so an already-seeded file still exposes
            a duplicate, but only `ordered_files` are ever placed.
        ordered_episode_ids: The resolved episode ids, season-order.
        ep_id_map: `(season, episode) -> id` over ALL the series' episodes.
            Membership in the resolved set does the scoping, so an exact parse
            outside our entry is rejected.
        allow_unscoped: Scope-gate override. None (default) derives it
            from an empty `ordered_episode_ids`. A caller that pre-subtracts
            already-placed ids passes it explicitly (off its FULL resolved set) so a
            fully-seeded record isn't mistaken for "no scope to enforce".

    Returns:
        The placed files and the skipped ones.
    """

    resolved = [i for i in ordered_episode_ids if i]
    resolved_set = set(resolved)
    # With NO resolved set to scope against, the exact leg falls back to the live
    # series episode map (a correctly-named file still lands on its real episode).
    # The absolute/positional legs stay disabled below (no leftover ids), so an
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

    # The restart-numbering tell is a BATCH property, counting every absolute
    # of every parse supplied - seeded files included, or a v1 placed on an
    # earlier poll would hide its v2 from this leg. Deduped per parse: the
    # tell is two FILES sharing an absolute, not junk repeats within one.
    batch_absolutes = [
        number
        for parsed in parsed_by_file.values()
        if parsed is not None
        for number in dict.fromkeys(parsed.absolute_episode_numbers)
    ]
    # A parse the caller couldn't get (None), or the offline regex stand-in
    # for one (blind to absolutes: "S01E12 - 12" would launder its lost 12),
    # may be hiding a duplicate - the tell's input is incomplete, so the leg
    # fails CLOSED, the same posture a hiccuped leftover gets from the count.
    all_parses_known = all(parsed is not None and not parsed.offline for parsed in parsed_by_file.values())

    clean_absolute = (
        bool(abs_by_file)
        and all_parses_known
        and len(abs_by_file) == len(deferred)  # every leftover has one absolute
        and len(abs_by_file) == len(leftover_ids)  # 1:1 with the leftover ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restart numbering)
    )

    skipped: list[str] = []
    if clean_absolute:
        for name, _abs in sorted(abs_by_file.items(), key=lambda kv: kv[1]):
            assigned[name] = [leftover_ids.pop(0)]
    elif (
        len(deferred) == 1
        and len(leftover_ids) == 1
        and (single := parsed_by_file.get(deferred[0])) is not None
        and (_has_no_signal(single) or _signal_is_bogus(single, ep_id_map))
        and not _spans_multiple(single)
    ):
        # Degenerate positional: one leftover file, one leftover episode, and
        # Sonarr SAW the name and found no number - or only a provably-bogus
        # key that exists nowhere in the series - it is that episode (the
        # single-file fallback). A None parse is no evidence at all (a blip
        # heals next poll), and multi-episode matched evidence means placing
        # as one episode would half-import - both refuse instead.
        assigned[deferred[0]] = [leftover_ids[0]]
    elif (
        len(deferred) > 1
        and len(deferred) == len(leftover_ids)
        and len(deferred) == len(parsed_by_file)
        and all(
            (parsed := parsed_by_file.get(name)) is not None
            and not parsed.offline
            and _has_no_signal(parsed)
            and not _spans_multiple(parsed)
            for name in deferred
        )
    ):
        # Pristine numberless batch: the parse-map equality proves NOTHING in
        # the batch was placed or seeded (a mixed batch never zips, so an
        # extra can never fill a missing episode's slot), counts match 1:1,
        # and every parse is a real numberless one. Order is the only signal
        # left: zip name order onto airing order (the "Special 1..N" shape).
        for name, ep_id in zip(sorted(deferred, key=_natural_key), leftover_ids, strict=True):
            assigned[name] = [ep_id]
    else:
        skipped = list(deferred)

    return EpisodeAssignment(assigned=assigned, skipped=skipped)


@dataclass(frozen=True)
class CandidateFile:
    """An on-disk manual-import candidate, reduced to what planning needs.

    Built by the strategy from one raw ManualImportResource.
    """

    basename: str
    """The normalized match key against our authoritative map."""

    path: str
    """What we POST."""

    quality: QualityModel | None
    """Reused if our own quality parse comes up empty."""

    is_sample: bool
    """Folds Sonarr's per-file sample rejection into the plan."""

    is_already_imported: bool
    """Folds Sonarr's per-file already-imported rejection into the plan."""


class ImportAction(StrEnum):
    """What `plan_import_files` decided for one entry in OUR map.

    A `StrEnum` (so each member IS its rendered word, matching the
    `PendingState` / `QueueVerdict` / `EpisodeFileStatus`
    style) - the consumer branches on a typed value instead of a magic string.
    Only `IMPORT` and `MISSING` drive behavior. The three "nothing to import
    for this file" members are kept distinct purely for reporting.
    """

    IMPORT = "import"
    """POST a manual import for this file."""

    SKIP_DONE = "skip_done"
    """Not needed (every target already holds a recommended file), with no Sonarr rejection."""

    SAMPLE = "sample"
    """A sample (never our intended file)."""

    ALREADY = "already"
    """Not needed, and Sonarr flagged an already-imported rejection."""

    MISSING = "missing"
    """Our map intends this file but it isn't on disk (surfaced, never silently skipped)."""


@dataclass(frozen=True)
class ImportDecision:
    """One decision per entry in OUR authoritative map (the source of truth).

    Candidates only supply the on-disk `path` and rejection flags (folded into `action`).
    """

    basename: str
    action: ImportAction
    path: str | None
    """The on-disk path, supplied by the matched candidate."""

    quality: QualityModel | None
    episode_ids: list[int]
    """The episode assignment, strictly from our map, never the candidate's own parse."""


def plan_import_files(
    authoritative_map: dict[str, list[int]],
    candidates_by_basename: dict[str, CandidateFile],
    needing_import: set[int],
) -> list[ImportDecision]:
    """Decide, per intended file, whether/how to import it - strictly from our map.

    Iterates OUR map (never the candidates): a file Sonarr found that isn't in our
    map is never imported, and a file our map intends that isn't on disk is
    surfaced as `missing` (never silently skipped). For a present file both
    invariants are honored via `needing_import` (the non-recommended target
    set): a file whose every episode already holds a recommended release is
    `skip_done` (not overwritten). Otherwise it is imported for exactly its
    needing-import episodes.

    `needing_import` (derived from the EPISODE FILES via
    `episode_file_statuses`) - not Sonarr's per-candidate already-imported
    rejection - is authoritative for whether we still want a file. Sonarr raises
    that rejection whenever the episode already holds *any* file on disk, including
    a non-recommended or unidentifiable-group one we flagged as still-needing
    replacement. Honoring it as a skip there is the grab-then-skip bug (we grab a
    missing-group replacement, then Sonarr's "already imported" makes us skip
    importing it). So `is_already_imported` only yields `already` when NONE of
    the file's episodes still need us (every target already holds a recommended
    file - Sonarr and our episode-file check agree). When a target still needs us
    we import over it, as the never-skip invariant requires. `is_sample` still
    wins (a sample is never our intended file).

    Args:
        authoritative_map: normalized basename -> our ids.
        candidates_by_basename: on-disk files by key.
        needing_import: episode ids still needing our file (from
            `targets_needing_import`).

    Returns:
        One decision per map entry, in map order.
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
            # more specific `ALREADY` when Sonarr flagged it, else `SKIP_DONE`.
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
# source axis undetermined (None) - it is NEVER defaulted to WEB here. The
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
    """Quality as two independent axes: `source` and `resolution`.

    Either axis is `None` when it could not be authoritatively determined, which
    is what lets the quality decision layer the axes across Sonarr's parse, our
    filename parse, and the configured default (each fills only the axes the
    higher-precedence layers left `None`). The resulting `(source, resolution)`
    pair is matched against Sonarr's quality definitions to pick the real quality.
    """

    source: QualitySource | None = None
    resolution: int | None = None


def parse_quality_from_filename(filename: str) -> ParsedQuality:
    """Best-effort `(source, resolution)` parse of a SeaDex filename.

    Detects a resolution (`2160`/`1080`/`720`/`480`) and a source
    (Remux, BluRay, WEB-DL, WEBRip, WEB, HDTV, DVD), case-insensitively and
    independently. Either axis is `None` when not found - notably an
    unrecognized source is left `None` (NOT defaulted to WEB), so the configured
    default can fill it rather than the file being silently mislabeled.

    Args:
        filename: The SeaDex filename (or full path, only the text matched).

    Returns:
        The parsed axes. Either may be `None`.
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
    """The `(source, resolution)` axes of a Sonarr `QualityModel`.

    Reads the canonical schema path `model.quality.source` /
    `model.quality.resolution` (every field defaults to None on the partial
    models the helpers build). An `"unknown"` source or a `0`/absent
    resolution maps to `None` (undetermined), so an unparsed candidate
    cleanly yields `ParsedQuality()` and falls through to the next
    precedence layer.

    Args:
        model: A candidate's in-context quality model.

    Returns:
        The structured axes Sonarr determined, each possibly None.
    """

    if model is None:
        return ParsedQuality()
    quality = model.quality  # an empty/null wire quality already folded to None
    if quality is None:
        return ParsedQuality()
    resolution = quality.resolution
    if resolution is None or resolution <= 0:
        resolution = None
    return ParsedQuality(source=QualitySource.parse(quality.source), resolution=resolution)


def quality_axes_from_name(
    name: str | None,
    quality_defs: list[QualityDefinition],
) -> ParsedQuality:
    """The `(source, resolution)` axes of a configured default quality NAME.

    Resolves the configured `imports.default_quality` (a Sonarr quality name like
    `"Bluray-2160p"`) to its structured axes by matching it, case-insensitively,
    against the `/api/v3/qualitydefinition` list - so the default contributes a
    real `(source, resolution)` the decision fills gaps from. An unset name, or
    one that matches no definition, yields `ParsedQuality()` (no default).

    Args:
        name: The configured default quality name, if any.
        quality_defs: The `/api/v3/qualitydefinition` list.

    Returns:
        The default's axes, or empty when unset/unmatched.
    """

    if not name:
        return ParsedQuality()
    target = name.casefold()
    for definition in quality_defs:
        quality = definition.quality
        if quality is None:
            continue
        def_name = quality.name
        if def_name is not None and def_name.casefold() == target:
            return quality_axes_from_model(QualityModel(quality=quality))
    return ParsedQuality()


def derive_languages(
    is_dual_audio: bool,
    dual: list[str],
    single: list[str],
) -> list[str]:
    """Pick the import language list: `dual` when dual-audio, else `single`."""

    return dual if is_dual_audio else single


def _find_definition(
    source: QualitySource,
    resolution: int,
    quality_defs: list[QualityDefinition],
) -> Quality | None:
    """The nested `Quality` whose `(source, resolution)` matches, or None.

    Scans the `/api/v3/qualitydefinition` list for the definition whose nested
    quality has the given structured source and resolution. `(source, resolution)`
    is unique across Sonarr's standard definitions (the only near-collision, Raw-HD
    vs HDTV-1080p, differs by source), so the pair identifies the quality without
    ever matching on its display name.
    """

    for definition in quality_defs:
        quality = definition.quality
        if quality is None:
            continue
        if quality.resolution == resolution and QualitySource.parse(quality.source) is source:
            return quality
    return None


# A RAW source degrades to its base when no remux/raw definition exists at that
# resolution (try the raw definition first, then this base).
_RAW_DOWNGRADE: dict[QualitySource, QualitySource] = {
    QualitySource.BLURAY_RAW: QualitySource.BLURAY,
    QualitySource.TELEVISION_RAW: QualitySource.TELEVISION,
}


def _candidate_revision(candidate_model: QualityModel | None) -> Revision:
    """The candidate's revision (proper/repack), or a fresh `version 1` default."""

    if candidate_model is not None and candidate_model.revision is not None:
        return candidate_model.revision
    return Revision(version=1, real=0, isRepack=False)


def resolve_quality(
    sonarr: ParsedQuality,
    ours: ParsedQuality,
    default: ParsedQuality,
    quality_defs: list[QualityDefinition],
    candidate_model: QualityModel | None,
) -> QualityModel:
    """Resolve the final manual-import `QualityModel` - never omitted.

    The source and resolution axes are decided independently, each taking the
    first authoritative value in precedence order: Sonarr's parse, then our
    filename parse, then the configured default. When both axes are determined the
    quality definition matching the `(source, resolution)` pair is emitted, so
    the payload always carries a quality Sonarr actually defines (a valid id+name).
    A determined `BLURAY_RAW`/`TELEVISION_RAW` with no matching remux/raw
    definition at that resolution gracefully downgrades to `BLURAY`/`TELEVISION`
    rather than failing.

    Crucially this never returns `None` and the caller never omits the quality:
    omitting it is exactly what made Sonarr crash in
    `FileNameBuilder.AddQualityTokens`. When nothing resolves, Sonarr's own
    candidate model (valid by construction) is re-emitted verbatim. Only if the
    candidate carries no quality at all is an explicit `Unknown` synthesized.

    Args:
        sonarr: Axes from Sonarr's candidate parse (highest).
        ours: Axes from our filename parse.
        default: Axes from the configured default quality.
        quality_defs: The `/api/v3/qualitydefinition` list to match against.
        candidate_model: Sonarr's in-context model, the
            last-resort verbatim fallback.

    Returns:
        The quality to POST. Never omitted.
    """

    # Invariant: the import payload always carries a quality key - omitting it
    # crashes Sonarr in FileNameBuilder.AddQualityTokens (observed on Sonarr 4.x).
    source = sonarr.source or ours.source or default.source
    resolution = sonarr.resolution or ours.resolution or default.resolution
    revision = _candidate_revision(candidate_model)

    if source is not None and resolution is not None:
        quality = _find_definition(source, resolution, quality_defs)
        base = _RAW_DOWNGRADE.get(source)
        if quality is None and base is not None:
            quality = _find_definition(base, resolution, quality_defs)
        if quality is not None:
            return QualityModel(quality=quality, revision=revision)

    # No confident match: re-emit Sonarr's own candidate (valid by construction)
    # rather than omit the quality, else synthesize an explicit Unknown. An
    # EMPTY candidate quality already folded to None at the parse boundary, so
    # this None test guards the already-folded empty quality.
    if candidate_model is not None and candidate_model.quality is not None:
        return candidate_model
    unknown = Quality(id=0, name="Unknown", source="unknown", resolution=0)
    return QualityModel(quality=unknown, revision=revision)


def resolve_language_objects(
    names: list[str],
    lang_defs: list[Language],
) -> list[Language]:
    """Resolve configured language names to Sonarr `{id, name}` objects.

    Case-insensitive match against the `/api/v3/language` list, in request
    order. A name with no match is dropped rather than failing the import.
    """

    by_name: dict[str, Language] = {
        name.casefold(): definition for definition in lang_defs if (name := definition.name) is not None
    }
    resolved: list[Language] = []
    for name in names:
        definition = by_name.get(name.casefold())
        if definition is not None:
            # Re-built fresh with BOTH fields set, so the exclude_unset write
            # dump always carries them (a null id included).
            resolved.append(Language(id=definition.id, name=definition.name))
    return resolved
