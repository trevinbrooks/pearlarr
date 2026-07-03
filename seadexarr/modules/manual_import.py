"""Pure wait/outcome vocabulary for the wait-for-completion import path.

This module holds the domain shapes the wait side of the manual-import feature
speaks: the configurable :class:`ImportWaitMode`, the durable
:class:`PendingImport` record persisted through the cache store, the per-poll
probe/outcome enums the engine and views consume (:class:`WaitOutcome`,
:class:`ImportReadiness`, :class:`PendingState`, :class:`Outcome`), qBittorrent
telemetry sanitization, and the basename/group normalizers every collaborator
matches through.

Everything here is deliberately side-effect free - no network, no disk, no
qBittorrent. The pure *planning* helpers (queue verdict, episode assignment,
import plan, quality/language resolution) live in :mod:`.sonarr_import_plan`,
which imports from this module - never the other way around.
"""

import math
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum, auto
from typing import Any, NamedTuple

from .seadex_types import coerce_int


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
    """Normalize a release group for comparison: strip whitespace/wrapping dashes, casefold.

    The single source of truth for group comparison - ``planner.normalize_rg``
    delegates here - so the never-overwrite check and the grab-time group filter
    agree on what counts as "the same group" (a dash-wrapped "-Aergia-" equals
    "Aergia" in both). Blank/None is handled by the caller (only real groups are
    ever passed here).
    """

    return group.strip().strip("-").casefold()


class ImportWaitMode(StrEnum):
    """When (if ever) the manual-import wait/import runs, resolved cli > config.

    A ``StrEnum`` so each member IS its config/CLI string (``ImportWaitMode.OFF``
    is and serializes as ``"off"``). The mode only controls *when* the import
    runs; all non-off modes share the same durable :class:`PendingImport`
    substrate.
    """

    OFF = "off"
    DEFERRED = "deferred"
    BLOCKING = "blocking"
    HYBRID = "hybrid"


class WaitOutcome(Enum):
    """The result of waiting on a torrent's completion in qBittorrent.

    ``COMPLETE`` -> import now; ``ERRORED`` -> leave the record pending for a
    later retry (TTL eventually drops it); ``MISSING`` -> the torrent is gone
    from qBittorrent, so the record should be dropped.
    """

    COMPLETE = auto()
    ERRORED = auto()
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
        PendingState: ``MISSING`` / ``ERRORED`` for those terminal outcomes;
        ``None`` (still downloading) -> ``QUEUED``; COMPLETE with files present ->
        ``IMPORTED``; COMPLETE without files present -> ``IMPORTING``.
    """

    if wait_outcome is WaitOutcome.MISSING:
        return PendingState.MISSING
    if wait_outcome is WaitOutcome.ERRORED:
        return PendingState.ERRORED
    if wait_outcome is None:
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
        imported_count (int): How many of the intended episodes already hold the
            recommended file - the "files inserted" bar numerator. Meaningful only
            with ``target_count`` > 0 (a complete seed map); 0 otherwise.
        target_count (int): The intended-episode denominator for the bar, fixed to
            the persisted seed set so the bar can't rescale mid-import. 0 means the
            seed map is incomplete, so the importing row stays indeterminate.
    """

    readiness: ImportReadiness
    files_present: bool
    command_issued: bool
    imported_count: int = 0
    target_count: int = 0


class ImportProgress(NamedTuple):
    """A cheap, read-only files-landed count for the wait cockpit's import bar.

    Returned by the strategy's ``import_progress`` (the Tier-2 poll): no refresh,
    no queue, no command - just the fresh episode files counted against the seed
    set. ``determinate`` is True only when the persisted seed map covers every
    intended file, so ``done``/``total`` are the true full set; when False the
    importing row stays indeterminate (spinner only) and must NOT promote.
    """

    done: int
    total: int
    determinate: bool


class OutcomeCategory(Enum):
    """The visual class of a terminal wait outcome - how it reads at a glance.

    Drives the wait view's ledger glyph + color and the end-of-wait tally:

    ``SUCCESS`` -> the torrent imported.
    ``DEFERRED`` -> left pending for a later run (a download timeout, or an
    import that hasn't landed yet); not a failure, just unfinished.
    ``FAILED`` -> the download errored or vanished from qBittorrent.

    Each member carries the unicode glyph, an ASCII fallback (dumb terminals /
    legacy Windows, where ``✔`` can't be encoded), and the rich style its ledger
    row is colored with - so one place owns the look of each outcome class.
    """

    SUCCESS = ("✔", "ok", "green")
    DEFERRED = ("⚠", "~", "yellow")
    FAILED = ("✖", "x", "bold red")

    glyph: str
    ascii_glyph: str
    style: str

    def __init__(self, glyph: str, ascii_glyph: str, style: str) -> None:
        self.glyph = glyph
        self.ascii_glyph = ascii_glyph
        self.style = style


class Outcome(Enum):
    """A torrent's terminal result in the wait pass, with its rendering vocab.

    Replaces the free-form outcome strings the engine used to hand the WaitView,
    so success and failure read distinctly AND the displayed word can't drift
    from the durable-store decision. Each member carries:

    ``word`` -> the short ledger token (every one fits ``STATE_WIDTH`` = 11).
    ``detail`` -> the longer human phrase the run report / notification use.
    ``category`` -> the :class:`OutcomeCategory` driving glyph + color + tally.
    ``dropped`` -> whether the engine removes the record from the durable store
    on this outcome. True for EXACTLY ``IMPORTED`` (files verified present) and
    ``MISSING`` (gone from qBittorrent) - the two records that must never be
    retried; a test pins this set so the word and the drop can't diverge.
    """

    IMPORTED = ("imported", "imported", OutcomeCategory.SUCCESS, True)
    MISSING = ("gone", "gone from qBittorrent", OutcomeCategory.FAILED, True)
    DOWNLOAD_ERRORED = ("errored", "download errored; left pending", OutcomeCategory.FAILED, False)
    DOWNLOAD_TIMED_OUT = ("timed out", "download timed out; left pending", OutcomeCategory.DEFERRED, False)
    STILL_IMPORTING = ("unfinished", "still importing; left pending", OutcomeCategory.DEFERRED, False)
    NOT_READY = ("not ready", "import not ready; left pending", OutcomeCategory.DEFERRED, False)
    NOTHING_TO_IMPORT = ("no files", "nothing to import; left pending", OutcomeCategory.DEFERRED, False)

    word: str
    detail: str
    category: OutcomeCategory
    dropped: bool

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
        """The leading ledger glyph: unicode ``✔/⚠/✖`` or its ASCII fallback."""

        return self.category.glyph if use_unicode else self.category.ascii_glyph


# qBittorrent reports a torrent with no meaningful ETA as 8_640_000 seconds
# (100 days), its "infinite" sentinel; treat it (and anything at/above it) as
# "unknown" rather than rendering a nonsense countdown.
_QBIT_ETA_INFINITE = 8_640_000


@dataclass(frozen=True)
class TorrentProbe:
    """One qBittorrent completion poll, with live download telemetry.

    Widens the old ``(outcome, content_path, progress)`` tuple so the wait view
    can show real speed / ETA / bytes. :meth:`~.import_wait.ImportWaitManager.poll_torrent`
    is the one place that builds this and the one place that SANITIZES qBittorrent's junk
    (via :func:`sanitize_torrent_telemetry`), so nothing downstream ever sees a
    sentinel: ``eta_s`` drops the 8_640_000 "∞" value to None, ``speed_bps`` drops
    a 0/idle speed to None (the view renders that as "stalled"), bytes are
    clamped, and a NaN/blank progress folds to 0.0.

    Args:
        outcome (WaitOutcome | None): The terminal outcome this poll, or None
            while still downloading (or on a transient qB error -> keep waiting).
        content_path (str | None): The completed download's path (COMPLETE only).
        progress (float): 0.0-1.0 download fraction (0.0 when unknown).
        speed_bps (int | None): Download speed in bytes/s, None when idle/unknown.
        eta_s (int | None): qBittorrent's ETA in seconds, None when unknown/∞.
        bytes_done (int | None): Bytes downloaded so far, None when unknown.
        bytes_total (int | None): Total size in bytes, None when unknown.
        observed (bool): False when qBittorrent could not actually be read (no
            client / a transient error), so the zeroed telemetry is a placeholder
            - the monitor keeps the row's last real bar/speed instead of painting
            a fake 0% + stall sample.
    """

    outcome: "WaitOutcome | None"
    content_path: str | None
    progress: float
    speed_bps: int | None = None
    eta_s: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    observed: bool = True


class TorrentTelemetry(NamedTuple):
    """One info row's sanitized telemetry, field-for-field what ``TorrentProbe`` carries."""

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
    """Fold one qBittorrent info row's raw telemetry into sanitized fields.

    Pure (no I/O), so the sentinel handling is unit-testable without a client.
    A NaN/blank ``progress`` folds to 0.0 and is clamped to ``[0, 1]``; a 0/idle
    or negative ``dlspeed`` and the 8_640_000 "∞" ``eta`` become None; bytes are
    coerced to non-negative ints (None when unknown) and ``completed`` is clamped
    to ``size``. Inputs are typed ``object`` because the values come off an
    untyped qBittorrent attribute read (``getattr``), which can hand back None.
    """

    frac = _as_float(progress)
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))

    raw_speed = coerce_int(dlspeed)
    speed_bps = raw_speed if raw_speed is not None and raw_speed > 0 else None

    raw_eta = coerce_int(eta)
    eta_s = raw_eta if raw_eta is not None and 0 < raw_eta < _QBIT_ETA_INFINITE else None

    raw_total = coerce_int(size)
    bytes_total = raw_total if raw_total is not None and raw_total > 0 else None
    raw_done = coerce_int(completed)
    bytes_done = max(0, raw_done) if raw_done is not None and raw_done > 0 else None
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


@dataclass(frozen=True)
class PendingImport:
    """A durable record of one added torrent awaiting a series-pinned import.

    Written at the add site through the cache facade (keyed by ``infohash`` via
    ``cache_store.put_pending``/``get_pending``/``drop_pending``) and read back to
    drive the manual import. It carries every field we have *authoritative* data
    for - the
    Sonarr ``series_id``, our own ``(basename -> episode ids)`` mapping, the
    SeaDex release group, dual-audio flag and coverage - so the import never has
    to trust Sonarr's blind title parse.

    Args:
        infohash (str): The qBittorrent tracking key (never None). Also the
            dedup ``downloadId`` sent to Sonarr.
        series_id (int): The Sonarr series id the files belong to.
        file_episode_map (dict[str, list[int]]): Basename -> authoritative
            Sonarr episode ids; the primary file->episode mapping. Repaired and
            extended in place at import time when a grabbed file wasn't parseable
            at grab time, so the map self-heals.
        episode_ids (list[int]): Legacy read-only fallback: new seeds always
            write ``[]`` (a value could only duplicate ``file_episode_map``);
            readers still fold it in so an old persisted record rehydrates.
        ordered_episode_ids (list[int]): The resolved episode ids for this entry,
            in season order - the authoritative set the import assigns into. Lifted
            straight from the add-flow ``ep_list`` (which already applied the
            specials/offset mapping), so import-time assignment never has to trust
            Sonarr's title parse: a file's parsed ``(season, episode)`` is honored
            only when it lands in this set, and an absolute-numbered pack is mapped
            positionally onto it. Empty for records written before this field
            existed (such a record falls back to the seeded ``file_episode_map``).
        release_group (str): The SeaDex release group (authoritative).
        is_dual_audio (bool): Whether the SeaDex release is dual-audio; selects
            the dual vs. single language list.
        seadex_files (list[str]): SeaDex filenames, for our regex quality parse.
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
    seadex_files: list[str]
    title: str | None
    added_at: str
    coverage: str | None = None
    url: str | None = None
    ordered_episode_ids: list[int] = field(default_factory=list[int])

    @property
    def display_label(self) -> str:
        """The cockpit/ledger/report row label: ``title · group``.

        The release group disambiguates a series that grabbed several torrents
        (their titles are identical); the infohash is the last-resort fallback.
        """

        base = self.title or self.infohash
        if self.release_group:
            return f"{base} · {self.release_group}"
        return base

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain dict persisted under ``pending_imports``.

        Every field is JSON-native (str / int / bool / list / dict / None), so
        ``asdict`` is the whole serializer - and a field added to the dataclass
        can't be silently dropped from the persisted form.
        """

        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "PendingImport":
        """Rebuild a record from its persisted cache-store dict.

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
            seadex_files=raw.get("seadex_files", []),
            title=raw.get("title"),
            added_at=raw.get("added_at", ""),
            coverage=raw.get("coverage"),
            url=raw.get("url"),
            ordered_episode_ids=raw.get("ordered_episode_ids", []),
        )


def resolve_wait_mode(
    cli_mode: ImportWaitMode | None,
    config_mode: ImportWaitMode | None,
) -> ImportWaitMode:
    """Resolve the effective wait mode with precedence cli > config > default.

    Args:
        cli_mode (ImportWaitMode | None): The ``--import-wait-mode`` CLI value.
        config_mode (ImportWaitMode | None): The configured ``imports.wait_mode``.

    Returns:
        ImportWaitMode: ``cli_mode`` if set, else ``config_mode`` if set, else
        :attr:`ImportWaitMode.OFF`.
    """

    if cli_mode is not None:
        return cli_mode
    if config_mode is not None:
        return config_mode
    return ImportWaitMode.OFF
