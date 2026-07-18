"""Import-time file -> episode mapping: the gnarliest Sonarr logic.

`FileEpisodeMapper` turns the on-disk manual-import candidates for a completed
download into the authoritative `basename -> episode ids` map, honoring OUR
resolved set (Sonarr's parse informs, never decides): the grab-time map is
taken as-is, every other on-disk leaf is parsed and placed into our resolved
set via the pure `assign_episode_ids`. Owns the per-run on-disk parse cache.
"""

import os

from .manual_import import PendingImport, normalize_basename
from .seadex_types import ManualImportCandidate, ParsedFileInfo
from .sonarr_client import AbstractSonarrClient
from .sonarr_import_plan import CandidateFile, assign_episode_ids, parse_se_from_filename
from .sonarr_parse import is_video_candidate

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
            base = normalize_basename(os.path.basename(path))
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
        candidates_by_basename: dict[str, CandidateFile],
        ep_id_map: dict[tuple[int, int], int],
    ) -> tuple[dict[str, list[int]], list[str]]:
        """Build the final `basename -> episode ids` map from OUR resolved set.

        Identity never comes from Sonarr's series-matched title parse alone: a
        file's parsed `(season, episode)` - from its name, or from Sonarr's
        matched resolution of an absolute-only name - is honored only *inside*
        our resolved set, an absolute-numbered pack is mapped positionally onto
        it, and anything ambiguous is returned as skipped (the caller warns and
        leaves it - the chosen safe posture).

        Files our grab-time `file_episode_map` already covers (the add-time
        assignment) keep their seeded ids untouched. When anything is left to
        place, their parses are still fetched so the positional leg's
        shared-absolute tell sees the whole batch (an earlier poll's placement
        must not hide a v2 duplicate). Every uncovered on-disk video leaf is
        handed to the pure `assign_episode_ids`, which places it into our
        resolved set
        (`ordered_episode_ids`, the add-flow's season-sorted episodes - or, for a
        record predating that field, one synthesized from its seeds). When there is
        no set to scope against (an on-disk specials record whose grab-time parse
        found nothing), `assign_episode_ids` falls back to the live series map
        for exactly named files (see `allow_unscoped`). Fresh placements self-heal
        onto the record. SeaDex order keeps output and the absolute leg stable.

        Returns `(merged_map, unplaceable_basenames)`. A basename duplicated
        across folders collapses in the basename-keyed pool, so it is reported
        unplaceable at most once - and never when the map placed it.
        """

        on_disk = {
            norm_base: candidate
            for norm_base, candidate in candidates_by_basename.items()
            if is_video_candidate(os.path.basename(candidate.path))
        }

        # SeaDex order first (so output is stable and the absolute leg's input is
        # deterministic), then any on-disk leaf the SeaDex list didn't name.
        ordered = [norm for norm in (normalize_basename(name) for name in pending.seadex_files) if norm in on_disk]
        placed = set(ordered)
        ordered += [norm_base for norm_base in on_disk if norm_base not in placed]

        # Honor our grab-time map (OUR add-time assignment) - seeded ids are
        # taken as-is. Intended files not yet on disk stay in the map so the
        # planner detects them missing and retries (never silent-drops). Only
        # the on-disk leftovers the seed doesn't cover (e.g. a specials pack
        # whose grab-time parse found nothing) are resolved from their parse.
        seeded: dict[str, list[int]] = {}
        for name, ids in pending.file_episode_map.items():
            clean = [i for i in ids if i]
            if clean:
                seeded[normalize_basename(name)] = clean
        seeded_ids = {i for ids in seeded.values() for i in ids}

        leftover = [norm for norm in ordered if norm not in seeded]
        # Parse the WHOLE batch when anything is left to place: the positional
        # leg's shared-absolute tell scans seeded files too, or a v1 placed on
        # an earlier poll would hide its v2 (parses are cached per run).
        # Mapped names are parsed BY NAME even when their files have moved out
        # (a completed move-mode import) - a gone v1 must not blind the tell.
        parsed_by_file: dict[str, ParsedFileInfo | None] = {}
        if leftover:
            for norm_base in ordered:
                parsed_by_file[norm_base] = self._parsed_file_info(os.path.basename(on_disk[norm_base].path))
            for name in pending.file_episode_map:
                norm_base = normalize_basename(name)
                if norm_base not in parsed_by_file:
                    parsed_by_file[norm_base] = self._parsed_file_info(os.path.basename(name))

        # The set the leftovers assign into: ordered_episode_ids, or - for a record
        # predating that field - one synthesized from its seeds (so the old
        # seed/single-file scoping survives). Ids the seed already owns are removed,
        # so a leftover file can't be handed an episode that's already placed.
        resolved_ids = pending.ordered_episode_ids or sorted(
            seeded_ids | {i for i in pending.episode_ids if i},
        )
        leftover_resolved = [i for i in resolved_ids if i not in seeded_ids]

        result = assign_episode_ids(
            leftover,
            parsed_by_file,
            leftover_resolved,
            ep_id_map,
            # Gate on the FULL resolved set, not the post-seed remainder: a fully-
            # seeded record (empty leftover_resolved) must keep scope enforced, or
            # an out-of-scope on-disk leftover would be imported (allow_unscoped).
            allow_unscoped=not resolved_ids,
        )

        # Self-heal: keep every fresh placement on the record for the run.
        for norm_base, ids in result.assigned.items():
            pending.file_episode_map[norm_base] = ids

        merged = {**seeded, **result.assigned}
        # A duplicate leaf (one basename in two folders) collapses in the
        # basename-keyed pool: its second occurrence defers off the used-set
        # and lands in skipped even though the name WAS placed. The warning
        # follows the map - placed names drop out, repeats collapse.
        skipped = [name for name in dict.fromkeys(result.skipped) if name not in merged]
        return merged, skipped

    def _parsed_file_info(self, raw_base: str) -> ParsedFileInfo | None:
        """Sonarr `/parse` of one on-disk leaf, cached per run.

        Carries the name-parsed numbers plus Sonarr's series-matched pairs.
        on a transient parse failure (None) falls back to an offline
        `SxxExx` regex - without caching - so a momentary Sonarr hiccup neither
        strands a correctly-named file nor sticks for the rest of the run.
        """

        if raw_base in self._parse_info_cache:
            return self._parse_info_cache[raw_base]
        info = self.sonarr.parse_episode_info(raw_base)
        if info is None:
            return parse_se_from_filename(raw_base)
        self._parse_info_cache[raw_base] = info
        return info
