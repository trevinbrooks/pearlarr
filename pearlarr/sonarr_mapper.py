"""Import-time file -> episode mapping: a completed download's on-disk leaves placed into OUR resolved set.

The grab-time map is taken as-is and every other leaf is parsed and placed (Sonarr's parse informs, never decides).
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

# Rejection-reason substrings, matched case-insensitively. Only ALREADY_IMPORTED means Sonarr holds the file:
# a SAMPLE is just a file to skip, never a sign the episode imported, so the two stay apart.
_ALREADY_IMPORTED_TOKENS = ("already", "exist")
_SAMPLE_TOKENS = ("sample",)


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
    """Owns import-time file -> episode assignment and the per-run on-disk parse cache.

    The executor calls `candidate_files` then `assign` per completed download, and reads the `FileAssignment`.
    """

    def __init__(self, sonarr: AbstractSonarrClient) -> None:
        """Bind the strategy's Sonarr client, whose `/parse` the on-disk parse reads."""

        self.sonarr = sonarr

        # Raw leaf -> its `/parse`, so a name reaches Sonarr once a run, not once a poll. A miss (None) is
        # transient and deliberately NOT cached, so a hiccup never strands a correctly named file for the run.
        self._parse_info_cache: dict[str, ParsedFileInfo] = {}

    def reset(self) -> None:
        """Drop the per-run on-disk parse cache (run-start, via get_items)."""

        self._parse_info_cache = {}

    def candidate_files(
        self,
        candidates: list[ManualImportCandidate],
    ) -> dict[str, CandidateFile]:
        """Index on-disk candidates by normalized basename: a basename duplicated across folders carries one verdict.

        The candidates arrive parsed at the Sonarr client boundary (`SonarrClient.manual_import_candidates`),
        so each is read by attribute and the raw DTO never reaches the decision path.
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

        Seeded files keep their ids, the on-disk leftover is placed under one window per claim in claim
        order, and anything ambiguous comes back skipped. The record is never mutated: fresh placements ride `placed`.
        """

        on_disk = {
            norm_base: candidate
            for norm_base, candidate in candidates_by_basename.items()
            if is_video_candidate(path_leaf(candidate.path))
        }

        # SeaDex order first (stable output, a deterministic absolute leg), then any leaf the list didn't name.
        ordered = [norm for norm in (normalized_leaf(name) for name in pending.seadex_files) if norm in on_disk]
        listed = set(ordered)
        ordered += [norm_base for norm_base in on_disk if norm_base not in listed]

        # Seeded ids are taken as-is, an intended file not yet on disk kept so the planner retries it.
        # Only the on-disk leftover the seed doesn't cover is placed from its parse.
        seeded = pending.seeded_map()
        leftover = [norm for norm in ordered if norm not in seeded]
        # Parse the WHOLE batch when anything is left: the shared-absolute tell scans seeded files too, and
        # mapped names are parsed BY NAME even once moved out, so a placed or gone v1 never hides its v2.
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
        """Sonarr `/parse` of one on-disk leaf (name numbers and series-matched pairs), cached per run.

        A transient failure (None) falls back to the offline `SxxExx` reading, uncached, so a hiccup never sticks.
        """

        if raw_base in self._parse_info_cache:
            return self._parse_info_cache[raw_base]
        info = self.sonarr.parse(raw_base)
        if info is None:
            return parse_se_from_filename(raw_base)
        self._parse_info_cache[raw_base] = info
        return info
