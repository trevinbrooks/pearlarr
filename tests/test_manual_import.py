# pyright: strict
"""The pure manual-import vocabulary: normalizers, the `PendingImport` record, the wait states, telemetry.

The planning modules' tests sit beside this file, one per module: `test_placement_types`, `test_placer`,
`test_episode_state`, `test_import_files`, `test_probe_verdicts`, `test_import_quality`.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pearlarr.manual_import import (
    LEAVE_PROBE,
    Deferral,
    GuardFacts,
    ImportProbe,
    OwnedEpisode,
    PendingImport,
    PendingKey,
    PendingState,
    TorrentTelemetry,
    WaitOutcome,
    classify_pending,
    normalize_basename,
    normalize_group,
    normalize_rg,
    normalized_leaf,
    path_leaf,
    sanitize_torrent_telemetry,
)

from .builders import SEP, pending_import


class TestNormalize:
    """`normalize_basename`/`normalize_group` casefold and strip for stable comparison.

    Interior dashes in a group name are never stripped, only wrapping ones.
    """

    def test_nfc_nfd_match(self) -> None:
        # Same text, NFC (composed) vs NFD (decomposed) "é". Both fold equal.
        nfc = "Café - 01.mkv"
        nfd = "Café - 01.mkv"
        assert normalize_basename(nfc) == normalize_basename(nfd)

    def test_strips_and_casefolds(self) -> None:
        assert normalize_basename("  Show - 01.MKV  ") == "show - 01.mkv"

    def test_windows_paths_and_trailing_separators_still_fold_to_a_leaf(self) -> None:
        # MUTATION PIN: a Windows arr's paths reach a POSIX host verbatim, where
        # os.path.basename sees no separator at all and would key the whole path.
        assert normalized_leaf("C:\\downloads\\Show\\Show - 01.mkv") == "show - 01.mkv"
        # And a directory entry never folds to the empty leaf, which every other
        # unnamed thing would then collide with.
        assert normalized_leaf("Show/NC/") == "nc"

    def test_path_leaf_folds_separators_but_preserves_the_name(self) -> None:
        # The parser-input twin of normalized_leaf: same separator folding, but
        # case/unicode intact - Sonarr's /parse and the video gate must see the
        # real filename, never a whole Windows path or a folded one.
        assert path_leaf("C:\\downloads\\Show\\Show - 01 [1080p].MKV") == "Show - 01 [1080p].MKV"
        assert path_leaf("Show/NC/") == "NC"

    def test_group_casefold(self) -> None:
        assert normalize_group("SubGroup") == normalize_group("subgroup")

    def test_group_strip_only_removes_wrapping_dashes(self) -> None:
        # MUTATION PIN: strip("-") widened to a multi-char strip set would eat
        # the X off an X-edged group. Only wrapping dashes (and whitespace) go,
        # and interior dashes always stay.
        assert normalize_group("Xrays-") == "xrays"
        assert normalize_group("X-Raws") == "x-raws"

    def test_group_dash_wrapped_agrees_with_planner(self) -> None:
        # normalize_group is the single source of truth normalize_rg delegates to.
        # A dash-wrapped group must compare equal on both ends or a release the
        # planner grabbed gets re-imported over by the overwrite guard.
        assert normalize_group("-Aergia-") == "aergia"
        assert normalize_group("-Aergia-") == normalize_rg("-Aergia-")
        assert normalize_group("Aergia") == normalize_rg("-Aergia-")


class TestPendingImportPlacements:
    """`with_placements` folds import-time placements into the map; `unplaced_names` is what a rebuild may fill."""

    def test_with_exclusions_normalizes_dedupes_and_appends(self) -> None:
        # The recorded order is the reading order: what the record already held, then this poll's
        # additions, each normalized once and never repeated.
        pending = pending_import(excluded_files=["show - 01.mkv"])

        healed = pending.with_exclusions(["SHOW - 03.MKV", " Show - 02.mkv ", "show - 03.mkv", "Show - 01.mkv"])

        assert healed.excluded_files == ["show - 01.mkv", "show - 03.mkv", "show - 02.mkv"]
        assert pending.excluded_files == ["show - 01.mkv"]

    def test_with_exclusions_of_nothing_leaves_the_record_equal(self) -> None:
        # The no-op the record seam gates its write on.
        pending = pending_import(excluded_files=["show - 01.mkv"])

        assert pending.with_exclusions(()) == pending

    def test_with_placements_normalizes_and_merges(self) -> None:
        # A raw-cased seed key collapses onto its normalized placement, a
        # zero-id seed entry with no placement is dropped, a mixed entry keeps
        # its real ids, and the original record is untouched.
        seed = {"Show - 01 [1080p].MKV": [101], "Show - 02.mkv": [0], "Show - 03.mkv": [0, 103]}
        pending = pending_import(file_episode_map=dict(seed))

        healed = pending.with_placements({"SHOW - 01 [1080p].mkv": [111], "Show - 04.mkv": [104]})

        assert healed.file_episode_map == {
            "show - 01 [1080p].mkv": [111],
            "show - 03.mkv": [103],
            "show - 04.mkv": [104],
        }
        assert pending.file_episode_map == seed

    def test_unplaced_names_is_the_listing_minus_map_and_exclusions(self) -> None:
        pending = pending_import(
            file_episode_map={"Show - 01.mkv": [101]},
            seadex_files=["Show - 01.mkv", "Show - 02.mkv", "Show - 03.mkv"],
            excluded_files=["show - 03.mkv"],
        )
        assert pending.unplaced_names() == {"show - 02.mkv"}

    @pytest.mark.parametrize(
        ("file_episode_map", "excluded_files"),
        [
            ({}, []),
            ({"Show - 01.mkv": [101]}, []),
            ({"Show - 01.mkv": [101]}, ["show - 02.mkv"]),
            ({"Show - 01.mkv": [101], "Show - 02.mkv": [102]}, []),
        ],
    )
    def test_seed_coverage_accounted_matches_unplaced_names(
        self,
        file_episode_map: dict[str, list[int]],
        excluded_files: list[str],
    ) -> None:
        # The two reads must not drift: accounted means nothing is left unplaced.
        pending = pending_import(
            file_episode_map=file_episode_map,
            seadex_files=["Show - 01.mkv", "Show - 02.mkv"],
            excluded_files=excluded_files,
        )
        assert pending.seed_coverage().accounted == (not pending.unplaced_names())


class TestPendingImportRoundTrip:
    """`PendingImport`'s JSON round-trip tolerates missing/unknown keys and defaults coverage/url to None.

    `display_label` falls back title -> infohash when the title/group is absent.
    """

    def test_to_json_from_json_round_trip(self) -> None:
        pending = PendingImport(
            infohash="abc123",
            series_id=55,
            al_id=990,
            file_episode_map={"ep1.mkv": [11], "ep2.mkv": [12]},
            episode_ids=[11, 12],
            release_group="Era-Raws",
            is_dual_audio=True,
            seadex_files=["ep1.mkv", "ep2.mkv"],
            title="Some Show",
            added_at="2026-06-24 12:00:00",
            coverage="S02 E01-E12",
            url="https://releases.moe/1",
            slice_coverage="S02 E01-E02",
            excluded_files=["other-slice.mkv"],
            guards=GuardFacts(entry_groups=("Era-Raws", "OtherPick"), owned_episodes=(OwnedEpisode(11, 700),)),
            awaiting_cleanup=True,
        )
        raw = pending.to_json()
        # Guard evidence is entry-level (its own guard_facts row): the per-torrent
        # blob never carries a copy, so a bare rehydrate comes back guard-empty.
        assert "guards" not in raw
        assert PendingImport.from_json(raw) == replace(pending, guards=GuardFacts())
        # The caller-supplied row (the read seams' join) hydrates it back whole.
        assert PendingImport.from_json(raw, guards=pending.guards) == pending

    def test_healed_map_round_trips_and_flips_coverage(self) -> None:
        # The placements live in the same blob field as the seed, so a healed
        # record rehydrates mapped on the next run.
        pending = pending_import(
            file_episode_map={},
            episode_ids=[],
            seadex_files=["Show - 01 [1080p].mkv"],
            ordered_episode_ids=[101],
        )

        healed = pending.with_placements({"show - 01 [1080p].mkv": [101]})

        assert PendingImport.from_json(healed.to_json()) == healed
        assert (pending.seed_coverage().mapped, healed.seed_coverage().mapped) == (False, True)

    def test_from_json_ignores_a_legacy_blob_guards_key(self) -> None:
        # A pre-v3 blob carries a frozen grab-time copy - the divergence the
        # guard_facts row exists to kill - so it must never resurrect.
        raw = {"infohash": "h", "series_id": 1, "guards": {"entry_groups": ["Stale"]}}
        assert PendingImport.from_json(raw).guards == GuardFacts()

    def test_from_json_tolerates_missing_keys(self) -> None:
        rebuilt = PendingImport.from_json({"infohash": "h", "series_id": 1})
        assert rebuilt.infohash == "h"
        assert rebuilt.file_episode_map == {}
        assert rebuilt.title is None
        # Pre-excluded_files records rehydrate empty (completeness stays
        # conservative for them).
        assert rebuilt.excluded_files == []
        # Pre-guards records guard on grabbed groups alone.
        assert rebuilt.guards == GuardFacts()
        # A legacy record predates the cleanup flag: nothing is owed.
        assert rebuilt.awaiting_cleanup is False
        # A legacy record with no al_id rehydrates under the 0 sentinel and keys
        # as its hash's singleton.
        assert rebuilt.al_id == 0
        assert rebuilt.key == PendingKey("h", 0)

    def test_display_label_is_title_dot_group_with_fallbacks(self) -> None:
        # The group disambiguates a series that grabbed several torrents. A
        # groupless record shows the bare title, a titleless one its infohash.
        rebuilt = PendingImport.from_json({"infohash": "h", "series_id": 1})
        assert rebuilt.display_label == "h"
        titled = PendingImport.from_json({"infohash": "h", "series_id": 1, "title": "Show"})
        assert titled.display_label == "Show"
        grouped = PendingImport.from_json(
            {"infohash": "h", "series_id": 1, "title": "Show", "release_group": "Era-Raws"},
        )
        assert grouped.display_label == f"Show{SEP}Era-Raws"

    def test_display_label_appends_the_record_episode_slice(self) -> None:
        # Sibling records from ONE group (a per-episode torrent each) share the
        # title and group - only the slice tells a wait/notification row apart.
        sliced = PendingImport.from_json(
            {
                "infohash": "h",
                "series_id": 1,
                "title": "Show",
                "release_group": "Era-Raws",
                "slice_coverage": "S02 E06",
            },
        )
        assert sliced.display_label == f"Show{SEP}Era-Raws{SEP}S02 E06"

    def test_old_record_with_unknown_keys_rehydrates(self) -> None:
        # Back-compat: a record persisted with since-removed keys still loads
        # (from_json reads only the known keys and ignores the rest).
        raw = {
            "infohash": "h",
            "series_id": 1,
            "file_episode_map": {"a.mkv": [1]},
            "episode_ids": [1],
            "release_group": "RG",
            "is_dual_audio": False,
            "seadex_files": ["a.mkv"],
            "seadex_sizes": [1000],
            "title": "T",
            "added_at": "2026-06-24 00:00:00",
        }
        assert PendingImport.from_json(raw).infohash == "h"

    def test_old_record_without_coverage_url_defaults_to_none(self) -> None:
        # Migration-safe: a record persisted before coverage/url existed loads with
        # both defaulting to None (via from_json's .get).
        raw = {"infohash": "h", "series_id": 1}
        rebuilt = PendingImport.from_json(raw)
        assert rebuilt.coverage is None
        assert rebuilt.url is None

    def test_coverage_url_default_none_on_dataclass(self) -> None:
        # The dataclass defaults coverage/url to None so callers (and old records)
        # need not supply them.
        pending = PendingImport(
            infohash="h",
            series_id=1,
            al_id=1,
            file_episode_map={},
            episode_ids=[],
            release_group="RG",
            is_dual_audio=False,
            seadex_files=[],
            title=None,
            added_at="2026-06-24 00:00:00",
        )
        assert pending.coverage is None
        assert pending.url is None


class TestPendingStateAndProbe:
    """The shared carried-over status vocabulary + the import probe value object."""

    def test_pending_state_members(self) -> None:
        assert {s.name for s in PendingState} == {
            "QUEUED",
            "DOWNLOADED",
            "IMPORTED",
            "ERRORED",
            "MISSING",
        }

    def test_pending_state_is_its_string(self) -> None:
        assert PendingState.DOWNLOADED == "downloaded"
        assert PendingState.QUEUED == "queued"

    def test_imported_probe_claims_the_files_without_a_command(self) -> None:
        probe = ImportProbe.imported(imported_count=2, target_count=2)
        assert probe.files_present is True
        assert probe.command_issued is False
        assert (probe.imported_count, probe.target_count) == (2, 2)

    def test_waiting_probe_carries_the_attempt_flags(self) -> None:
        probe = ImportProbe.waiting(command_issued=True, deferral=Deferral.IMPORT)
        assert probe.files_present is False
        assert probe.command_issued is True
        assert probe.deferral is Deferral.IMPORT
        assert probe.deferred is True
        assert probe.attempted is True

    def test_waiting_probe_defaults_to_no_deferral(self) -> None:
        probe = ImportProbe.waiting()
        assert probe.deferral is Deferral.NONE
        assert probe.deferred is False

    def test_leave_probe_records_no_attempt(self) -> None:
        assert LEAVE_PROBE.attempted is False
        assert LEAVE_PROBE.files_present is False
        assert LEAVE_PROBE.command_issued is False


class TestClassifyPending:
    """`classify_pending` folds a poll's outcome + the files-present flag into a state."""

    def test_missing(self) -> None:
        assert classify_pending(WaitOutcome.MISSING, False) is PendingState.MISSING

    def test_errored(self) -> None:
        assert classify_pending(WaitOutcome.ERRORED, False) is PendingState.ERRORED

    def test_still_downloading_is_queued(self) -> None:
        assert classify_pending(None, False) is PendingState.QUEUED

    def test_complete_and_files_present_is_imported(self) -> None:
        assert classify_pending(WaitOutcome.COMPLETE, True) is PendingState.IMPORTED

    def test_complete_without_files_is_importing(self) -> None:
        # The copy is still in flight -> downloaded, never imported, until the
        # files are verified present.
        assert classify_pending(WaitOutcome.COMPLETE, False) is PendingState.DOWNLOADED


class TestSanitizeTorrentTelemetry:
    """MUTATION PIN: pins the pure telemetry sanitizer's clamps and sentinel folds.

    Covers the numeric-string `_as_float` path too - a cluster of ~10 surviving mutants.
    """

    @pytest.mark.parametrize(
        ("progress", "dlspeed", "eta", "completed", "size", "expected"),
        [
            # All-None getattr reads fold to the empty telemetry.
            (None, None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # A clean row passes through untouched.
            (0.64, 3_200_000, 130, 1_800, 2_900, TorrentTelemetry(0.64, 3_200_000, 130, 1_800, 2_900)),
            # NaN progress (float and string) folds to 0.0, not a poisoned bar.
            (float("nan"), 100, 130, 50, 200, TorrentTelemetry(0.0, 100, 130, 50, 200)),
            ("nan", None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Numeric-string progress parses. Junk folds to 0.0.
            ("0.75", None, None, None, None, TorrentTelemetry(0.75, None, None, None, None)),
            ("fast", None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Progress clamps to [0, 1] on both ends.
            (1.5, None, None, None, None, TorrentTelemetry(1.0, None, None, None, None)),
            (-0.25, None, None, None, None, TorrentTelemetry(0.0, None, None, None, None)),
            # Idle (0) and negative speeds read as "no speed", never a 0 B/s row.
            (0.5, 0, None, None, None, TorrentTelemetry(0.5, None, None, None, None)),
            (0.5, -5, None, None, None, TorrentTelemetry(0.5, None, None, None, None)),
            # qBittorrent's 8_640_000 "infinite" eta and a 0/negative eta are unknown.
            # The last finite second still renders.
            (0.5, 100, 8_640_000, None, None, TorrentTelemetry(0.5, 100, None, None, None)),
            (0.5, 100, 0, None, None, TorrentTelemetry(0.5, 100, None, None, None)),
            (0.5, 100, 8_639_999, None, None, TorrentTelemetry(0.5, 100, 8_639_999, None, None)),
            # Zero/negative byte counts are unknown, not empty-progress readings.
            (0.5, None, None, 0, 0, TorrentTelemetry(0.5, None, None, None, None)),
            (0.5, None, None, -3, -1, TorrentTelemetry(0.5, None, None, None, None)),
            # An over-count clamps done to the total, never a >100% bar.
            (0.5, None, None, 500, 200, TorrentTelemetry(0.5, None, None, 200, 200)),
            # Bytes done without a known total still renders.
            (0.5, None, None, 100, None, TorrentTelemetry(0.5, None, None, 100, None)),
        ],
    )
    def test_edge_inputs(
        self,
        progress: object,
        dlspeed: object,
        eta: object,
        completed: object,
        size: object,
        expected: TorrentTelemetry,
    ) -> None:
        row = SimpleNamespace(progress=progress, dlspeed=dlspeed, eta=eta, completed=completed, size=size)

        assert sanitize_torrent_telemetry(row) == expected

    def test_attrless_row_folds_to_the_zero_reading(self) -> None:
        # The fields are read best-effort off the row: a row missing them all is the empty telemetry.
        assert sanitize_torrent_telemetry(object()) == TorrentTelemetry(0.0, None, None, None, None)


def test_wait_outcome_members_exist() -> None:
    assert {o.name for o in WaitOutcome} == {"COMPLETE", "ERRORED", "MISSING"}
