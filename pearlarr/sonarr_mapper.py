"""Import-time file -> episode mapping: the gnarliest Sonarr logic.

`FileEpisodeMapper` turns the on-disk manual-import candidates for a completed
download into the authoritative `basename -> episode ids` map, honoring OUR
resolved set (Sonarr's parse informs, never decides): the grab-time map is
taken as-is, every other on-disk leaf is parsed and placed into our resolved
set via the pure `assign_episode_ids`. Owns the per-run on-disk parse cache.
"""

from collections.abc import Mapping
from typing import NamedTuple

from .import_files import CandidateFile
from .manual_import import PendingImport, normalized_leaf, path_leaf
from .placement_types import EpisodeAssignment, EpisodeIndex, Placement, PlacementBatch
from .release_names import parse_se_from_filename
from .seadex_types import ManualImportCandidate, ParsedFileInfo
from .sonarr_client import AbstractSonarrClient
from .sonarr_parse import is_video_candidate
from .window_placement import place_leftover, windows_of

# Rejection-reason substrings, matched case-insensitively against each
# rejection's reason/message text. `ALREADY_IMPORTED` means Sonarr already has
# the file (it imported it itself, or it exists) - seeing only these means the
# download is effectively done. `SAMPLE` is just a file to skip, not a sign the
# real episode imported, so the two are kept apart.
_ALREADY_IMPORTED_TOKENS = ("already", "exist")
_SAMPLE_TOKENS = ("sample",)


def _rejection_matches(candidate: ManualImportCandidate, tokens: tuple[str, ...]) -> bool:
    """True if any of a candidate's rejections contains one of `tokens`.

    Best-effort and case-insensitive (`tokens` must be lowercase). Each
    rejection is an `ImportRejection` view whose `reason` carries the
    human text (a bare-string rejection from an older Sonarr is folded into the
    same `reason` field at the client boundary).
    """

    for rejection in candidate.rejections:
        if not rejection.reason:
            continue
        lowered = rejection.reason.casefold()
        if any(token in lowered for token in tokens):
            return True
    return False


class FileAssignment(NamedTuple):
    """One poll's file -> episode map: the seeded entries plus this poll's verdict on every other leaf."""

    result: EpisodeAssignment
    """This poll's verdicts on the unseeded on-disk video leaves, SeaDex order."""
    seeded: dict[str, list[int]]
    """The grab-time map's entries, keyed by normalized basename."""
    settled: bool
    """Whether the skips are a verdict against real inputs: every parse known (`PlacementBatch.all_parses_known`)
    and the episode index served. False makes a skip tentative, re-asked next poll."""

    @property
    def placed(self) -> dict[str, list[int]]:
        """This poll's fresh placements alone, for the caller to persist."""

        return self.result.assigned

    @property
    def assigned(self) -> dict[str, list[int]]:
        """The seeded entries plus `placed`."""

        return {**self.seeded, **self.placed}

    @property
    def skipped(self) -> tuple[str, ...]:
        """Unplaceable on-disk video leaves nothing proved foreign, for the executor to warn about."""

        return self.result.skipped

    @property
    def excluded(self) -> tuple[Placement, ...]:
        """The leaves this record knowably never imports, with their verdicts (for the caller to persist)."""

        return self.result.excluded

    @property
    def unplaced(self) -> tuple[str, ...]:
        """Every leaf without ids: the skips plus the exclusions."""

        return self.result.unplaced


class FileEpisodeMapper:
    """Owns import-time file -> episode assignment + the per-run on-disk parse cache.

    Constructed once per run in `SonarrSync` from the
    strategy's Sonarr client. The import executor calls `candidate_files` then
    `assign` for each completed download. `assign` returns the unplaceable
    basenames for the executor to warn about (producer/consumer split).
    """

    def __init__(self, sonarr: AbstractSonarrClient) -> None:
        """Bind the strategy's Sonarr client, whose `/parse` the on-disk parse reads."""

        self.sonarr = sonarr

        # Per-run, in-memory cache of the `/parse` of an on-disk
        # leaf (raw basename -> ParsedFileInfo), so the import poll loop sends a
        # given filename to Sonarr's parser at most once a run rather than every
        # poll. A /parse miss (None) is treated as transient and deliberately NOT
        # cached, so a hiccup doesn't strand a correctly-named file for the run.
        self._parse_info_cache: dict[str, ParsedFileInfo] = {}

    def reset(self) -> None:
        """Drop the per-run on-disk parse cache (run-start, via get_items)."""

        self._parse_info_cache = {}

    def candidate_files(
        self,
        candidates: list[ManualImportCandidate],
    ) -> dict[str, CandidateFile]:
        """Index on-disk manual-import candidates by normalized basename.

        The candidates arrive already parsed at the Sonarr client boundary
        (`SonarrClient.manual_import_candidates`), so each is read by
        attribute and the raw DTO never reaches the decision path.
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

    def assign(
        self,
        pending: PendingImport,
        candidates_by_basename: Mapping[str, CandidateFile],
        indexes: Mapping[int, EpisodeIndex],
    ) -> FileAssignment:
        """Build the final `basename -> episode ids` map from OUR resolved set, never from Sonarr's parse alone.

        Seeded files keep their grab-time ids. The on-disk leftover is placed under one window per claim, in
        claim order (`indexes` holds each claim's series), and anything ambiguous comes back skipped for the
        caller to warn about. The record is never mutated: fresh placements ride `placed` for the caller to
        persist. A basename duplicated across folders carries one verdict.
        """

        on_disk = {
            norm_base: candidate
            for norm_base, candidate in candidates_by_basename.items()
            if is_video_candidate(path_leaf(candidate.path))
        }

        # SeaDex order first (so output is stable and the absolute leg's input is
        # deterministic), then any on-disk leaf the SeaDex list didn't name.
        ordered = [norm for norm in (normalized_leaf(name) for name in pending.seadex_files) if norm in on_disk]
        placed = set(ordered)
        ordered += [norm_base for norm_base in on_disk if norm_base not in placed]

        # Honor our grab-time map (OUR add-time assignment) - seeded ids are
        # taken as-is. Intended files not yet on disk stay in the map so the
        # planner detects them missing and retries (never silent-drops). Only
        # the on-disk leftovers the seed doesn't cover (e.g. a specials pack
        # whose grab-time parse found nothing) are resolved from their parse.
        seeded = pending.seeded_map()
        leftover = [norm for norm in ordered if norm not in seeded]
        # Parse the WHOLE batch when anything is left to place: the positional
        # leg's shared-absolute tell scans seeded files too, or a v1 placed on
        # an earlier poll would hide its v2 (parses are cached per run).
        # Mapped names are parsed BY NAME even when their files have moved out
        # (a completed move-mode import) - a gone v1 must not blind the tell.
        parsed_by_file: dict[str, ParsedFileInfo | None] = {}
        if leftover:
            for norm_base in ordered:
                parsed_by_file[norm_base] = self._parsed_file_info(path_leaf(on_disk[norm_base].path))
            for name in pending.file_episode_map:
                norm_base = normalized_leaf(name)
                if norm_base not in parsed_by_file:
                    parsed_by_file[norm_base] = self._parsed_file_info(path_leaf(name))

        # The leftovers assign into each claim's window in turn, the seeded ids already used there.
        batch = PlacementBatch(leftover, parsed_by_file)
        windowed = place_leftover(seeded, batch, windows_of(pending.claims, indexes))
        # An empty index means the exact leg could not have matched a numbered name this poll.
        settled = batch.all_parses_known and all(indexes[sid].id_by_key for sid in pending.series_ids)
        return FileAssignment(windowed.merged, seeded, settled=settled)

    def _parsed_file_info(self, raw_base: str) -> ParsedFileInfo | None:
        """Sonarr `/parse` of one on-disk leaf, cached per run.

        Carries the name-parsed numbers plus Sonarr's series-matched pairs.
        on a transient parse failure (None) falls back to an offline
        `SxxExx` regex - without caching - so a momentary Sonarr hiccup neither
        strands a correctly-named file nor sticks for the rest of the run.
        """

        if raw_base in self._parse_info_cache:
            return self._parse_info_cache[raw_base]
        info = self.sonarr.parse(raw_base)
        if info is None:
            return parse_se_from_filename(raw_base)
        self._parse_info_cache[raw_base] = info
        return info
