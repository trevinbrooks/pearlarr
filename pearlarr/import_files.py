"""Pure per-file import plan: which of our intended files import onto which episodes, and which are left."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from .arr_models import ManualImportCandidate, QualityModel
from .episode_state import EpisodeStatuses
from .file_paths import normalized_leaf, path_leaf
from .pending_import import FileEpisodeMap, PendingImport
from .video_files import is_video_candidate

# Rejection-reason substrings, matched case-insensitively. Only ALREADY_IMPORTED means Sonarr holds the file:
# a SAMPLE is just a file to skip, never a sign the episode imported, so the two stay apart.
_ALREADY_IMPORTED_TOKENS = ("already", "exist")
_SAMPLE_TOKENS = ("sample",)


@dataclass(frozen=True, slots=True)
class CandidateFile:
    """An on-disk manual-import candidate, reduced to what planning needs (`candidate_files`)."""

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


def _rejection_matches(candidate: ManualImportCandidate, tokens: tuple[str, ...]) -> bool:
    """True if any of a candidate's rejection reasons contains one of `tokens` (lowercase), case-insensitively.

    A bare-string rejection from an older Sonarr is folded into `reason` at the client boundary.
    """

    for rejection in candidate.rejections:
        if not rejection.reason:
            continue
        lowered = rejection.reason.casefold()
        if any(token in lowered for token in tokens):
            return True
    return False


def candidate_files(candidates: Iterable[ManualImportCandidate]) -> dict[str, CandidateFile]:
    """Index on-disk candidates by normalized basename: a basename duplicated across folders carries one verdict.

    The candidates arrive parsed at the Sonarr client boundary (`SonarrClient.manual_import_candidates`), so each is
    read by attribute and the raw DTO never reaches the decision path.
    """

    by_basename: dict[str, CandidateFile] = {}
    for candidate in candidates:
        path = candidate.path
        if not path:
            continue
        base = normalized_leaf(path)
        by_basename[base] = CandidateFile(
            basename=base,
            path=path,
            quality=candidate.quality,
            is_sample=_rejection_matches(candidate, _SAMPLE_TOKENS),
            is_already_imported=_rejection_matches(candidate, _ALREADY_IMPORTED_TOKENS),
        )
    return by_basename


@dataclass(frozen=True, slots=True)
class UnplacedInFolder:
    """One poll's video files on disk that the import's map lacks. The import reports them and never places one."""

    names: tuple[str, ...]
    """Their normalized basenames, in SeaDex's order, then the scan's for a file SeaDex doesn't list."""

    reported: tuple[str, ...]
    """The names the record doesn't exclude (`PendingImport.excluded_files`), in the same order."""

    hand_import_files: tuple[str, ...]
    """The reported names the grab left for a hand import (`PendingImport.hand_import_files`)."""

    @property
    def quiet(self) -> int:
        """How many names the record excludes: reported at the grab, not again."""

        return len(self.names) - len(self.reported)

    @property
    def skips(self) -> tuple[str, ...]:
        """The reported names not left for a hand import: the files the skip warning counts."""

        return tuple(name for name in self.reported if name not in self.hand_import_files)

    @property
    def unmatched(self) -> tuple[str, ...]:
        """The names a hand import must place when nothing is mapped.

        The reported ones, or every name when the record excluded them all.
        """

        return self.reported or self.names


def disk_leftover(
    pending: PendingImport, candidates_by_basename: Mapping[str, CandidateFile], mapped: FileEpisodeMap
) -> UnplacedInFolder:
    """The video files among `candidates_by_basename` that `mapped` lacks, split by what `pending` says of them."""

    video = {
        name for name, candidate in candidates_by_basename.items() if is_video_candidate(path_leaf(candidate.path))
    }
    # SeaDex's order keeps the output stable from poll to poll. A file the list doesn't name follows in disk order.
    listed = dict.fromkeys(name for name in (normalized_leaf(f) for f in pending.seadex_files) if name in video)
    ordered = [*listed, *(name for name in candidates_by_basename if name in video and name not in listed)]
    names = tuple(name for name in ordered if name not in mapped)
    excluded = set(pending.excluded_files)
    reported = tuple(name for name in names if name not in excluded)
    hand_import = set(pending.hand_import_files)
    return UnplacedInFolder(names, reported, tuple(name for name in reported if name in hand_import))


class ImportAction(StrEnum):
    """What `plan_import_files` decided for one entry in OUR map.

    A `StrEnum` (so each member IS its rendered word, matching the
    `PendingState` / `QueueVerdict` / `EpisodeFileStatus`
    style): the consumer branches on a typed value instead of a magic string.
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


@dataclass(frozen=True, slots=True)
class ImportDecision:
    """One decision per entry in OUR authoritative map (the source of truth).

    Candidates only supply the on-disk `path` and rejection flags (folded into `action`).
    """

    basename: str
    action: ImportAction
    path: str | None
    """The on-disk path, supplied by the matched candidate."""

    quality: QualityModel | None
    episode_ids: tuple[int, ...]
    """The episode assignment, strictly from our map, never the candidate's own parse."""


def plan_import_files(
    authoritative_map: FileEpisodeMap,
    candidates_by_basename: Mapping[str, CandidateFile],
    statuses: EpisodeStatuses,
) -> list[ImportDecision]:
    """Decide, for each file in OUR map (never the candidates), whether to import it and onto which episodes.

    A mapped file Sonarr didn't find is `missing`, one no episode needs is `skip_done` or `already`, and a needed
    one goes onto the episodes `TargetStatuses.import_ids` picks. One decision per map entry, in map order.
    """

    decisions: list[ImportDecision] = []
    for basename, ep_ids in authoritative_map.items():
        candidate = candidates_by_basename.get(basename)
        if candidate is None:
            decisions.append(ImportDecision(basename, ImportAction.MISSING, None, None, ep_ids))
            continue
        if candidate.is_sample:
            decisions.append(ImportDecision(basename, ImportAction.SAMPLE, candidate.path, None, ()))
            continue
        import_ids = statuses.import_ids(ep_ids)
        if not import_ids:
            # No episode needs this file, so Sonarr's already-imported rejection agrees with us. Report `ALREADY`
            # when Sonarr raised it, else `SKIP_DONE`.
            action = ImportAction.ALREADY if candidate.is_already_imported else ImportAction.SKIP_DONE
            decisions.append(ImportDecision(basename, action, candidate.path, None, ep_ids))
            continue
        # An episode still needs this file, so import it even when Sonarr says it's already imported. Sonarr says
        # that about any file already on the episode, including the one we're replacing.
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
