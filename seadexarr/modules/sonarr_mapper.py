"""Import-time file -> episode mapping: the gnarliest Sonarr logic.

Extracted from :class:`~.seadex_sonarr.SonarrSync`. ``FileEpisodeMapper`` turns the
on-disk manual-import candidates for a completed download into the authoritative
``basename -> episode ids`` map, honoring OUR resolved set (never Sonarr's
series-matched title parse): the grab-time map is taken as-is, every other on-disk
leaf is parsed series-agnostically and placed into our resolved set via the pure
:func:`~.manual_import.assign_episode_ids`. Owns the per-run on-disk parse cache.
"""

import os

from .manual_import import (
    CandidateFile,
    PendingImport,
    assign_episode_ids,
    normalize_basename,
    parse_se_from_filename,
)
from .seadex_types import ManualImportCandidate, ParsedFileInfo
from .sonarr_client import SonarrClient
from .sonarr_parse import is_video_candidate

# Rejection-reason substrings, matched case-insensitively against each
# rejection's reason/message text. ``ALREADY_IMPORTED`` means Sonarr already has
# the file (it imported it itself, or it exists) - seeing only these means the
# download is effectively done. ``SAMPLE`` is just a file to skip, not a sign the
# real episode imported, so the two are kept apart.
_ALREADY_IMPORTED_TOKENS = ("already", "exist")
_SAMPLE_TOKENS = ("sample",)


def _rejection_matches(candidate: ManualImportCandidate, tokens: tuple[str, ...]) -> bool:
    """True if any of a candidate's rejections contains one of ``tokens``.

    Best-effort and case-insensitive. Each rejection is an
    :class:`~.seadex_types.ImportRejection` view whose ``reason`` carries the
    human text (a bare-string rejection from an older Sonarr is folded into the
    same ``reason`` field at the client boundary).

    Args:
        candidate (ManualImportCandidate): The parsed candidate (reads
            ``rejections``).
        tokens (tuple[str, ...]): Lowercase substrings to look for.
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

    Constructed once per run in :class:`~.seadex_sonarr.SonarrSync` from the
    strategy's Sonarr client. The import executor calls ``candidate_files`` then
    ``assign`` for each completed download; ``assign`` returns the unplaceable
    basenames for the executor to warn about (producer/consumer split).
    """

    def __init__(self, sonarr: SonarrClient) -> None:
        """Bind the Sonarr client the on-disk parse reads.

        Args:
            sonarr (SonarrClient): The strategy's Sonarr client (its ``/parse``).
        """

        self.sonarr = sonarr

        # Per-run, in-memory cache of the series-agnostic ``/parse`` of an on-disk
        # leaf (raw basename -> ParsedFileInfo | None), so the import poll loop sends
        # a given filename to Sonarr's parser at most once a run rather than every
        # poll. A None value caches a confirmed "Sonarr can't parse this" miss.
        self._parse_info_cache: dict[str, ParsedFileInfo | None] = {}

    def reset(self) -> None:
        """Drop the per-run on-disk parse cache (run-start, via get_items)."""

        self._parse_info_cache = {}

    def candidate_files(
        self,
        candidates: list[ManualImportCandidate],
    ) -> dict[str, CandidateFile]:
        """Index on-disk manual-import candidates by normalized basename.

        The candidates arrive already parsed at the Sonarr client boundary
        (:meth:`SonarrClient.manual_import_candidates`), so each is read by
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
        """Build the final ``basename -> episode ids`` map from OUR resolved set.

        Identity is assigned off each file's series-agnostic parse against the live
        series episode map - never Sonarr's series-matched title parse: a file's
        ``(season, episode)`` is honored only *inside* our resolved set, an
        absolute-numbered pack is mapped positionally onto it, and anything ambiguous
        is returned as skipped (the caller warns and leaves it - the chosen safe
        posture).

        Files our grab-time ``file_episode_map`` already covers (the add-time
        assignment) are taken as-is - no need to re-parse what we resolved at grab
        time. Every other on-disk video leaf is parsed series-agnostically and handed
        to the pure :func:`assign_episode_ids`, which places it into our resolved set
        (``ordered_episode_ids``, the add-flow's season-sorted episodes - or, for a
        record predating that field, one synthesized from its seeds). When there is
        no set to scope against (an on-disk specials record whose grab-time parse
        found nothing), :func:`assign_episode_ids` falls back to the live series map
        for exactly named files (see ``allow_unscoped``). Fresh placements self-heal
        onto the record; SeaDex order keeps output and the absolute leg stable.

        Returns ``(merged_map, unplaceable_basenames)``.
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

        # Honor our grab-time map (OUR add-time assignment) - no need to re-parse
        # what we resolved at grab time. Intended files not yet on disk stay in the
        # map so the planner detects them missing and retries (never silent-drops);
        # only the on-disk leftovers the seed doesn't cover (e.g. a specials pack
        # whose grab-time parse found nothing) are resolved from their parse.
        seeded: dict[str, list[int]] = {}
        for name, ids in pending.file_episode_map.items():
            clean = [i for i in ids if i]
            if clean:
                seeded[normalize_basename(name)] = clean
        seeded_ids = {i for ids in seeded.values() for i in ids}

        leftover = [norm for norm in ordered if norm not in seeded]
        parsed_by_file = {
            norm_base: self._parsed_file_info(os.path.basename(on_disk[norm_base].path)) for norm_base in leftover
        }

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

        return {**seeded, **result.assigned}, result.skipped

    def _parsed_file_info(self, raw_base: str) -> ParsedFileInfo | None:
        """Series-agnostic parse of one on-disk leaf, cached per run.

        Prefers Sonarr's ``/parse`` ``parsedEpisodeInfo`` (it handles absolute
        numbering); on a transient parse failure (None) falls back to an offline
        ``SxxExx`` regex - without caching - so a momentary Sonarr hiccup neither
        strands a correctly-named file nor sticks for the rest of the run.
        """

        if raw_base in self._parse_info_cache:
            return self._parse_info_cache[raw_base]
        info = self.sonarr.parse_episode_info(raw_base)
        if info is None:
            return parse_se_from_filename(raw_base)
        self._parse_info_cache[raw_base] = info
        return info
