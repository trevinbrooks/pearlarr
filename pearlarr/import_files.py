"""Pure per-file import plan: which of our intended files import onto which episodes, and which are left."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .episode_state import TargetStatuses
from .manual_import import FileEpisodeMap
from .seadex_types import QualityModel


@dataclass(frozen=True, slots=True)
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
    statuses: TargetStatuses,
) -> list[ImportDecision]:
    """Decide, for each file in OUR map (never the candidates), whether to import it and onto which episodes.

    A mapped file Sonarr didn't find is `missing`, one no episode needs is `skip_done` or `already`, and a needed
    one goes onto the episodes `TargetStatuses.import_ids` picks. One decision per map entry, in map order.
    """

    decisions: list[ImportDecision] = []
    for basename, ep_ids in authoritative_map.items():
        candidate = candidates_by_basename.get(basename)
        if candidate is None:
            decisions.append(ImportDecision(basename, ImportAction.MISSING, None, None, tuple(ep_ids)))
            continue
        if candidate.is_sample:
            decisions.append(ImportDecision(basename, ImportAction.SAMPLE, candidate.path, None, ()))
            continue
        import_ids = statuses.import_ids(ep_ids)
        if not import_ids:
            # No episode needs this file, so Sonarr's already-imported rejection agrees with us. Report `ALREADY`
            # when Sonarr raised it, else `SKIP_DONE`.
            action = ImportAction.ALREADY if candidate.is_already_imported else ImportAction.SKIP_DONE
            decisions.append(
                ImportDecision(basename, action, candidate.path, None, tuple(ep_ids)),
            )
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
