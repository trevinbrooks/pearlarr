# pyright: strict
"""The per-file import plan: strict-honor, never-overwrite, never-skip."""

from pearlarr.import_files import CandidateFile, plan_import_files


def _candidate(basename: str, *, sample: bool = False, already: bool = False) -> CandidateFile:
    """A file Sonarr found under the download, flagged as a sample or as already imported on request."""

    return CandidateFile(
        basename=basename,
        path=f"/dl/{basename}",
        quality=None,
        is_sample=sample,
        is_already_imported=already,
    )


class TestPlanImportFiles:
    """`plan_import_files` decides each candidate's action from the file->episode map and the needing-import set.

    Actions: import, skip_done, missing, sample, already.
    """

    def test_imports_only_needing_episodes(self) -> None:
        amap = {"a.mkv": [11], "b.mkv": [12]}
        cands = {"a.mkv": _candidate("a.mkv"), "b.mkv": _candidate("b.mkv")}
        # episode 12 already holds a recommended file -> only 11 needs import.
        decisions = plan_import_files(amap, cands, needing_import={11})
        by_base = {d.basename: d for d in decisions}
        assert by_base["a.mkv"].action == "import"
        assert by_base["a.mkv"].episode_ids == [11]
        assert by_base["b.mkv"].action == "skip_done"

    def test_candidate_not_in_map_is_never_imported(self) -> None:
        amap = {"a.mkv": [11]}
        cands = {"a.mkv": _candidate("a.mkv"), "rogue.mkv": _candidate("rogue.mkv")}
        decisions = plan_import_files(amap, cands, needing_import={11})
        # Only our mapped file is decided on. The rogue on-disk file is ignored.
        assert {d.basename for d in decisions} == {"a.mkv"}

    def test_intended_file_missing_from_disk_is_flagged_not_dropped(self) -> None:
        amap = {"a.mkv": [11], "b.mkv": [12]}
        cands = {"a.mkv": _candidate("a.mkv")}
        decisions = plan_import_files(amap, cands, needing_import={11, 12})
        by_base = {d.basename: d for d in decisions}
        assert by_base["a.mkv"].action == "import"
        assert by_base["b.mkv"].action == "missing"

    def test_sample_is_not_imported(self) -> None:
        # A sample is never our intended file regardless of need.
        amap = {"s.mkv": [11]}
        cands = {"s.mkv": _candidate("s.mkv", sample=True)}
        decisions = plan_import_files(amap, cands, needing_import={11})
        assert decisions[0].action == "sample"

    def test_already_imported_does_not_skip_a_needed_target(self) -> None:
        # Bug fix: Sonarr's "already imported" rejection fires whenever the episode
        # already holds ANY file (including a missing-group one we grabbed to
        # replace). It must NOT veto importing a target that still needs our file.
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv", already=True)}
        decisions = plan_import_files(amap, cands, needing_import={12})
        assert decisions[0].action == "import"
        assert decisions[0].episode_ids == [12]

    def test_already_imported_skips_only_when_no_target_needs_us(self) -> None:
        # When every target already holds a recommended file (none in needing),
        # Sonarr's rejection and our episode-file check agree -> the more specific
        # `already` (never overwrite).
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv", already=True)}
        decisions = plan_import_files(amap, cands, needing_import=set())
        assert decisions[0].action == "already"

    def test_not_needed_without_rejection_is_skip_done(self) -> None:
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv")}
        decisions = plan_import_files(amap, cands, needing_import=set())
        assert decisions[0].action == "skip_done"
