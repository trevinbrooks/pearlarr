"""Pure per-file import plan: which of the file map's files import onto which episodes, and which are left."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from .arr_models import ManualImportCandidate, QualityModel
from .episode_state import EpisodeStatuses
from .file_paths import file_name, normalize_file_name
from .pending_import import FileEpisodeMap, PendingImport
from .video_files import is_video_candidate

# Rejection-reason substrings, matched case-insensitively. Only ALREADY_IMPORTED means Sonarr has the file:
# a SAMPLE is just a file to skip, never a sign the episode imported, so the two stay apart.
_ALREADY_IMPORTED_TOKENS = ("already", "exist")
_SAMPLE_TOKENS = ("sample",)


@dataclass(frozen=True, slots=True)
class CandidateFile:
    """A ManualImport candidate in the download's folder, reduced to what planning needs (`candidate_files`)."""

    normalized_name: str
    """The normalized match key against the authoritative file map."""

    path: str
    """What pearlarr POSTs."""

    quality: QualityModel | None
    """Reused if pearlarr's quality parse comes up empty."""

    is_sample: bool
    """Merges Sonarr's per-file sample rejection into the plan."""

    is_already_imported: bool
    """Merges Sonarr's per-file already-imported rejection into the plan."""


def _rejection_matches(candidate: ManualImportCandidate, tokens: tuple[str, ...]) -> bool:
    """True if any of a candidate's rejection reasons contains one of `tokens` (lowercase), case-insensitively.

    A plain-string rejection from an older Sonarr is converted to `reason` at the client boundary.
    """

    for rejection in candidate.rejections:
        if not rejection.reason:
            continue
        lowered = rejection.reason.casefold()
        if any(token in lowered for token in tokens):
            return True
    return False


def candidate_files(candidates: Iterable[ManualImportCandidate]) -> dict[str, CandidateFile]:
    """Index the folder's candidates by normalized file name: a file name duplicated across folders carries one answer.

    The candidates arrive parsed at the Sonarr client boundary (`SonarrClient.manual_import_candidates_by_download_id`), so each is
    read by attribute and the raw DTO never reaches the decision path.
    """

    by_name: dict[str, CandidateFile] = {}
    for candidate in candidates:
        path = candidate.path
        if not path:
            continue
        base = normalize_file_name(path)
        by_name[base] = CandidateFile(
            normalized_name=base,
            path=path,
            quality=candidate.quality,
            is_sample=_rejection_matches(candidate, _SAMPLE_TOKENS),
            is_already_imported=_rejection_matches(candidate, _ALREADY_IMPORTED_TOKENS),
        )
    return by_name


@dataclass(frozen=True, slots=True)
class UnplacedInFolder:
    """One poll's video files in the folder the file map lacks. The import reports them and never places one."""

    names: tuple[str, ...]
    """Their normalized file names, in SeaDex's order, then the fetch's for a file SeaDex doesn't list."""

    reported: tuple[str, ...]
    """The names the pending import doesn't exclude (`PendingImport.excluded_files`), in the same order."""

    hand_import_files: tuple[str, ...]
    """The reported names the grab left for a hand import (`PendingImport.hand_import_files`)."""

    @property
    def quiet(self) -> int:
        """How many names the pending import excludes: reported at the grab, not again."""

        return len(self.names) - len(self.reported)

    @property
    def skips(self) -> tuple[str, ...]:
        """The reported names not left for a hand import: the files the skip warning counts."""

        return tuple(name for name in self.reported if name not in self.hand_import_files)

    @property
    def when_nothing_placed(self) -> tuple[str, ...]:
        """The names a hand import must place when the file map places nothing.

        The reported ones, or every name when the pending import excluded them all.
        """

        return self.reported or self.names


def unplaced_in_folder(
    pending: PendingImport, candidates_by_name: Mapping[str, CandidateFile], file_map: FileEpisodeMap
) -> UnplacedInFolder:
    """The video files among `candidates_by_name` that `file_map` lacks, split by what `pending` says of them."""

    video = {name for name, candidate in candidates_by_name.items() if is_video_candidate(file_name(candidate.path))}
    # SeaDex's order keeps the output stable from poll to poll. A file the list doesn't name follows in fetch order.
    listed = dict.fromkeys(name for name in (normalize_file_name(f) for f in pending.seadex_files) if name in video)
    ordered = [*listed, *(name for name in candidates_by_name if name in video and name not in listed)]
    names = tuple(name for name in ordered if name not in file_map)
    excluded = set(pending.excluded_files)
    reported = tuple(name for name in names if name not in excluded)
    hand_import = set(pending.hand_import_files)
    return UnplacedInFolder(names, reported, tuple(name for name in reported if name in hand_import))


class ImportAction(StrEnum):
    """What `plan_import_files` decided for one file in the file map.

    A `StrEnum` (so each member IS its rendered word, matching the
    `PendingImportState` / `QueueState` / `EpisodeFileStatus`
    style): the consumer branches on a typed value instead of a magic string.
    Only `IMPORT` and `MISSING` drive behavior. The three "nothing to import
    for this file" members are kept distinct purely for reporting.
    """

    IMPORT = "import"
    """POST a ManualImport for this file."""

    SKIP_DONE = "skip_done"
    """Not needed (each of its episodes already has a trusted file), with no Sonarr rejection."""

    SAMPLE = "sample"
    """A sample (never an intended file)."""

    ALREADY = "already"
    """Not needed, and Sonarr raised an already-imported rejection."""

    MISSING = "missing"
    """The file map intends this file but it isn't in the folder (reported, never silently skipped)."""


@dataclass(frozen=True, slots=True)
class ImportDecision:
    """One decision per file in the authoritative file map (the source of truth).

    Candidates only supply the folder's `path` and their rejections (reduced to `action`).
    """

    normalized_name: str
    action: ImportAction
    path: str | None
    """The path in the folder, supplied by the matched candidate."""

    quality: QualityModel | None
    episode_ids: tuple[int, ...]
    """The episodes, strictly from the file map, never the candidate's parse."""


def plan_import_files(
    file_map: FileEpisodeMap,
    candidates_by_name: Mapping[str, CandidateFile],
    statuses: EpisodeStatuses,
) -> list[ImportDecision]:
    """Decide, for each file in the file map (never the candidates), whether to import it and onto which episodes.

    A file map file Sonarr didn't find is `missing`, one no episode needs is `skip_done` or `already`, and a needed
    one goes onto the episodes `EpisodeStatuses.import_ids` returns. One decision per file, in file map order.
    """

    decisions: list[ImportDecision] = []
    for name, ep_ids in file_map.items():
        candidate = candidates_by_name.get(name)
        if candidate is None:
            decisions.append(ImportDecision(name, ImportAction.MISSING, None, None, ep_ids))
            continue
        if candidate.is_sample:
            decisions.append(ImportDecision(name, ImportAction.SAMPLE, candidate.path, None, ()))
            continue
        import_ids = statuses.import_ids(ep_ids)
        if not import_ids:
            # No episode needs this file, so Sonarr's already-imported rejection agrees with pearlarr. Report `ALREADY`
            # when Sonarr raised it, else `SKIP_DONE`.
            action = ImportAction.ALREADY if candidate.is_already_imported else ImportAction.SKIP_DONE
            decisions.append(ImportDecision(name, action, candidate.path, None, ep_ids))
            continue
        # An episode still needs this file, so import it even when Sonarr says it's already imported. Sonarr says
        # that about any file already on the episode, including the one being replaced.
        decisions.append(
            ImportDecision(
                name,
                ImportAction.IMPORT,
                candidate.path,
                candidate.quality,
                import_ids,
            ),
        )
    return decisions
