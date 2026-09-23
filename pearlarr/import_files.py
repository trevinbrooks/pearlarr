"""Pure per-file import plan: which of our intended files import onto which episodes, and which are left."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum

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
    needing_import: AbstractSet[int],
) -> list[ImportDecision]:
    """Decide, per intended file, whether/how to import it, strictly from our map.

    The map is normalized basename -> our episode ids, and the candidates are
    keyed the same way. Iterates OUR map (never the candidates): a file Sonarr
    found that isn't in our map is never imported, and a file our map intends
    that isn't on disk is surfaced as `missing` (never silently skipped). For a
    present file both invariants are honored via `needing_import` (the
    non-recommended target set): a file whose every episode already holds a
    recommended release is `skip_done` (not overwritten). Otherwise it is
    imported for exactly its needing-import episodes.

    `needing_import` (derived from the EPISODE FILES via
    `EpisodeSnapshot.statuses` and `TargetStatuses.needing_import`) is
    authoritative for whether we still want a file, not Sonarr's per-candidate
    already-imported rejection. Sonarr raises that rejection whenever the
    episode already holds *any* file on disk, including a non-recommended or
    unidentifiable-group one we flagged as still-needing replacement. Honoring
    it as a skip there is the grab-then-skip bug (we grab a missing-group
    replacement, then Sonarr's "already imported" makes us skip importing it).
    So `is_already_imported` only yields `already` when NONE of the file's
    episodes still need us (every target already holds a recommended file,
    Sonarr and our episode-file check agree). When a target still needs us we
    import over it, as the never-skip invariant requires. `is_sample` still
    wins (a sample is never our intended file). One decision per map entry,
    in map order.
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
        import_ids = tuple(i for i in ep_ids if i in needing_import)
        if not import_ids:
            # Nothing of ours still needs this file. Sonarr's already-imported
            # rejection and our episode-file done-check agree here, so report the
            # more specific `ALREADY` when Sonarr flagged it, else `SKIP_DONE`.
            action = ImportAction.ALREADY if candidate.is_already_imported else ImportAction.SKIP_DONE
            decisions.append(
                ImportDecision(basename, action, candidate.path, None, tuple(ep_ids)),
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
