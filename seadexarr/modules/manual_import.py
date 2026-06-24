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
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Any

from .seadex_types import SONARR_MISSING_KEY, SonarrEpisode


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


class QueueVerdict(Enum):
    """What Sonarr's queue says to do with a tracked download.

    Derived purely from the aggregate ``trackedDownloadState`` of the queue
    records sharing a ``downloadId`` (a season pack has one record per episode):

    ``DONE`` -> Sonarr already imported every record; drop the pending record.
    ``WAIT`` -> Sonarr is still downloading or actively importing; keep polling.
    ``STEP_IN`` -> Sonarr can't / won't auto-import (``importBlocked``, ``failed``,
    ``ignored``) or isn't tracking the download at all; drive our authoritative
    manual import.
    """

    DONE = auto()
    WAIT = auto()
    STEP_IN = auto()


# trackedDownloadState values (camelCase from Sonarr, compared case-folded) that
# mean Sonarr is still working the download and we should keep waiting rather than
# step in. ``queued``/``delay``/``paused`` are QueueStatus-ish transients Sonarr
# may surface in the same field; treat them as "still working" too.
_QUEUE_ACTIVE_STATES = frozenset(
    {"downloading", "importpending", "importing", "queued", "delay", "paused"},
)


def classify_queue_states(states: list[str]) -> QueueVerdict:
    """Reduce the per-episode ``trackedDownloadState`` list to a single verdict.

    Side-effect free so the queue decision can be unit-tested without any HTTP.
    Priority, highest first:

      1. any ``importBlocked`` -> ``STEP_IN`` (Sonarr gave up; our authoritative
         mapping is exactly what unblocks it).
      2. any still-active state (downloading / importPending / importing / ...) ->
         ``WAIT`` (let Sonarr finish; it may still flip to importBlocked, which a
         later poll re-evaluates).
      3. all ``imported`` -> ``DONE``.
      4. otherwise (``failed`` / ``failedPending`` / ``ignored`` / unknown, or an
         empty list because Sonarr isn't tracking the download) -> ``STEP_IN``.

    Args:
        states (list[str]): The ``trackedDownloadState`` of every queue record
            sharing the download's infohash (any case; missing values dropped by
            the caller).

    Returns:
        QueueVerdict: The action the strategy should take this poll.
    """

    folded = [s.casefold() for s in states]

    if any(s == "importblocked" for s in folded):
        return QueueVerdict.STEP_IN
    if any(s in _QUEUE_ACTIVE_STATES for s in folded):
        return QueueVerdict.WAIT
    if folded and all(s == "imported" for s in folded):
        return QueueVerdict.DONE
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
            Sonarr episode ids; the primary file->episode mapping.
        episode_ids (list[int]): Flat fallback ids for the single-file /
            unparsed last-resort assignment.
        release_group (str): The SeaDex release group (authoritative).
        is_dual_audio (bool): Whether the SeaDex release is dual-audio; selects
            the dual vs. single language list.
        season_number (int | None): The single season, or None for multi-season
            / absolute-numbered packs.
        seadex_files (list[str]): SeaDex filenames, for our regex quality parse.
        title (str | None): Display title (logging only).
        added_at (str): When the record was written, in
            :data:`UPDATED_AT_STR_FORMAT`, used for the TTL drop.
    """

    infohash: str
    series_id: int
    file_episode_map: dict[str, list[int]]
    episode_ids: list[int]
    release_group: str
    is_dual_audio: bool
    season_number: int | None
    seadex_files: list[str]
    title: str | None
    added_at: str

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain dict persisted under ``pending_imports``."""

        return {
            "infohash": self.infohash,
            "series_id": self.series_id,
            "file_episode_map": self.file_episode_map,
            "episode_ids": self.episode_ids,
            "release_group": self.release_group,
            "is_dual_audio": self.is_dual_audio,
            "season_number": self.season_number,
            "seadex_files": self.seadex_files,
            "title": self.title,
            "added_at": self.added_at,
        }

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
            title=raw.get("title"),
            added_at=raw.get("added_at", ""),
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

    ``source`` records which layer won (``"ours"``/``"sonarr"``/``"default"``/
    ``"unknown"``); ``name`` is the Sonarr quality name to resolve (for the
    ours/default layers) and ``model`` is the candidate's in-context quality
    model dict to reuse verbatim (for the sonarr layer). The unused field is
    None in each case.
    """

    source: str
    name: str | None
    model: dict | None


def _candidate_quality_name(candidate_quality: dict | None) -> str | None:
    """Pull the nested quality name out of a Sonarr candidate quality dict.

    Sonarr nests the name at either ``quality.name`` or ``quality.quality.name``
    depending on the endpoint; read both null-safely.
    """

    if not candidate_quality:
        return None
    quality = candidate_quality.get("quality")
    if isinstance(quality, dict):
        inner = quality.get("quality")
        if isinstance(inner, dict) and inner.get("name"):
            return inner.get("name")
        if quality.get("name"):
            return quality.get("name")
    if candidate_quality.get("name"):
        return candidate_quality.get("name")
    return None


def select_quality(
    our_name: str | None,
    candidate_quality: dict | None,
    default_name: str | None,
) -> QualitySelection:
    """Choose a quality with precedence ours > sonarr-in-context > default.

    Layers, in order:
      1. ``our_name`` (our regex parse of the SeaDex filename) -> ``"ours"``.
      2. ``candidate_quality`` if present *and* its nested quality name is a real
         value (not missing and not ``"Unknown"``) -> ``"sonarr"`` (reuse the
         model verbatim).
      3. ``default_name`` (the configured fallback) -> ``"default"``.
      4. otherwise ``"unknown"`` (the caller warns; Sonarr re-grab risk).

    Args:
        our_name (str | None): Our parsed Sonarr quality name, if any.
        candidate_quality (dict | None): The candidate's in-context quality model.
        default_name (str | None): The configured default quality name.

    Returns:
        QualitySelection: The winning layer and the value to carry forward.
    """

    if our_name:
        return QualitySelection(source="ours", name=our_name, model=None)

    candidate_name = _candidate_quality_name(candidate_quality)
    if candidate_quality and candidate_name and candidate_name != "Unknown":
        return QualitySelection(source="sonarr", name=None, model=candidate_quality)

    if default_name:
        return QualitySelection(source="default", name=default_name, model=None)

    return QualitySelection(source="unknown", name=None, model=None)


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


def assign_episode_ids(
    candidate_basenames: list[str],
    file_episode_map: dict[str, list[int]],
    flat_fallback: list[int],
) -> dict[str, list[int]]:
    """Map manual-import candidate files to authoritative Sonarr episode ids.

    For each candidate basename we use ``file_episode_map[basename]`` when
    present. As a single-file last resort, if exactly one basename is left
    unmatched *and* ``flat_fallback`` carries ids, that fallback is assigned to
    it. Any other unmatched file is left out of the result (the caller skips it
    rather than guessing). Episode id 0 is never assigned (a 0 id can't be
    POSTed to Sonarr).

    Args:
        candidate_basenames (list[str]): Basenames of the files Sonarr found on
            disk for this torrent.
        file_episode_map (dict[str, list[int]]): Our authoritative
            ``basename -> episode ids`` mapping.
        flat_fallback (list[int]): Flat fallback ids for the single-file rule.

    Returns:
        dict[str, list[int]]: ``basename -> episode ids`` for the files we can
        confidently map (unmappable files omitted; 0 ids stripped).
    """

    resolved: dict[str, list[int]] = {}
    unmatched: list[str] = []
    for basename in candidate_basenames:
        mapped = file_episode_map.get(basename)
        if mapped:
            ids = [ep_id for ep_id in mapped if ep_id]
            if ids:
                resolved[basename] = ids
                continue
        unmatched.append(basename)

    fallback_ids = [ep_id for ep_id in flat_fallback if ep_id]
    if len(unmatched) == 1 and fallback_ids:
        resolved[unmatched[0]] = fallback_ids

    return resolved


def resolve_quality_model(name: str, quality_defs: list[dict]) -> dict | None:
    """Resolve a Sonarr quality NAME to a manual-import ``QualityModel`` dict.

    Looks the name up (case-insensitive) against the nested ``quality.name`` of
    each ``/api/v3/qualitydefinition`` entry and wraps the matched quality dict
    in the ``{"quality": ..., "revision": ...}`` shape Sonarr expects on a
    manual-import file. Returns None when no definition matches the name, so the
    caller can omit the quality key (Sonarr falls back to Unknown).

    Args:
        name (str): A Sonarr quality name (e.g. ``"WEBDL-1080p"``).
        quality_defs (list[dict]): The raw ``/api/v3/qualitydefinition`` list;
            each entry nests a ``quality`` dict with ``id``/``name``/``source``/
            ``resolution``.

    Returns:
        dict | None: A ``QualityModel`` dict, or None when no definition matches.
    """

    target = name.casefold()
    for definition in quality_defs:
        quality = definition.get("quality")
        if not isinstance(quality, dict):
            continue
        quality_name = quality.get("name")
        if isinstance(quality_name, str) and quality_name.casefold() == target:
            return {
                "quality": quality,
                "revision": {"version": 1, "real": 0, "isRepack": False},
            }
    return None


def resolve_language_objects(names: list[str], lang_defs: list[dict]) -> list[dict]:
    """Resolve language names to Sonarr ``{id, name}`` language objects.

    Matches each requested name (case-insensitive) against the
    ``/api/v3/language`` list and returns the matched ``{"id", "name"}`` objects
    in request order, skipping any name with no match (so an unknown configured
    language is simply dropped rather than failing the import).

    Args:
        names (list[str]): Language names to resolve (e.g. ``["Japanese"]``).
        lang_defs (list[dict]): The raw ``/api/v3/language`` list; each entry has
            ``id`` and ``name``.

    Returns:
        list[dict]: The matched ``{"id", "name"}`` objects (unknown names
        omitted).
    """

    by_name = {
        definition["name"].casefold(): definition
        for definition in lang_defs
        if isinstance(definition.get("name"), str)
    }
    resolved: list[dict] = []
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
