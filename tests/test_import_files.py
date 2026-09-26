# pyright: strict
"""The per-file import plan: strict-honor, never-overwrite, never-skip."""

from typing import ClassVar

from pearlarr.episode_state import EpisodeFileStatus
from pearlarr.import_files import CandidateFile, plan_import_files

from .builders import target_statuses


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
    """`plan_import_files` decides each candidate's action from the file->episode map and the target statuses.

    Actions: import, skip_done, missing, sample, already.
    """

    def test_imports_only_needing_episodes(self) -> None:
        amap = {"a.mkv": [11], "b.mkv": [12]}
        cands = {"a.mkv": _candidate("a.mkv"), "b.mkv": _candidate("b.mkv")}
        # episode 12 already holds a recommended file -> only 11 needs import.
        decisions = plan_import_files(
            amap, cands, target_statuses({11: EpisodeFileStatus.ABSENT, 12: EpisodeFileStatus.RECOMMENDED})
        )
        by_base = {d.basename: d for d in decisions}
        assert by_base["a.mkv"].action == "import"
        assert by_base["a.mkv"].episode_ids == (11,)
        assert by_base["b.mkv"].action == "skip_done"

    def test_candidate_not_in_map_is_never_imported(self) -> None:
        amap = {"a.mkv": [11]}
        cands = {"a.mkv": _candidate("a.mkv"), "rogue.mkv": _candidate("rogue.mkv")}
        decisions = plan_import_files(amap, cands, target_statuses({11: EpisodeFileStatus.ABSENT}))
        # Only our mapped file is decided on. The rogue on-disk file is ignored.
        assert {d.basename for d in decisions} == {"a.mkv"}

    def test_intended_file_missing_from_disk_is_flagged_not_dropped(self) -> None:
        amap = {"a.mkv": [11], "b.mkv": [12]}
        cands = {"a.mkv": _candidate("a.mkv")}
        decisions = plan_import_files(
            amap, cands, target_statuses({11: EpisodeFileStatus.ABSENT, 12: EpisodeFileStatus.ABSENT})
        )
        by_base = {d.basename: d for d in decisions}
        assert by_base["a.mkv"].action == "import"
        assert by_base["b.mkv"].action == "missing"

    def test_sample_is_not_imported(self) -> None:
        # A sample is never our intended file regardless of need.
        amap = {"s.mkv": [11]}
        cands = {"s.mkv": _candidate("s.mkv", sample=True)}
        decisions = plan_import_files(amap, cands, target_statuses({11: EpisodeFileStatus.ABSENT}))
        assert decisions[0].action == "sample"

    def test_already_imported_does_not_skip_a_needed_target(self) -> None:
        # Bug fix: Sonarr's "already imported" rejection fires whenever the episode
        # already holds ANY file (including a missing-group one we grabbed to
        # replace). It must NOT veto importing a target that still needs our file.
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv", already=True)}
        decisions = plan_import_files(amap, cands, target_statuses({12: EpisodeFileStatus.UNKNOWN_GROUP}))
        assert decisions[0].action == "import"
        assert decisions[0].episode_ids == (12,)

    def test_already_imported_skips_only_when_no_target_needs_us(self) -> None:
        # When every target already holds a recommended file (none in needing),
        # Sonarr's rejection and our episode-file check agree -> the more specific
        # `already` (never overwrite).
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv", already=True)}
        decisions = plan_import_files(amap, cands, target_statuses({12: EpisodeFileStatus.RECOMMENDED}))
        assert decisions[0].action == "already"

    def test_not_needed_without_rejection_is_skip_done(self) -> None:
        amap = {"i.mkv": [12]}
        cands = {"i.mkv": _candidate("i.mkv")}
        decisions = plan_import_files(amap, cands, target_statuses({12: EpisodeFileStatus.RECOMMENDED}))
        assert decisions[0].action == "skip_done"


class TestOwnCopyAlongside:
    """A needed file also goes onto its episodes holding a copy of our own, so Sonarr replaces the copy.

    An episode holding any other recommended file is left alone.
    """

    _MAP: ClassVar[dict[str, list[int]]] = {"range.mkv": [21, 22]}
    _CANDIDATES: ClassVar[dict[str, CandidateFile]] = {"range.mkv": _candidate("range.mkv")}

    def test_an_episode_holding_our_own_copy_is_posted_with_the_needing_one(self) -> None:
        statuses = target_statuses({21: EpisodeFileStatus.RECOMMENDED, 22: EpisodeFileStatus.ABSENT}, frozenset({21}))

        [decision] = plan_import_files(self._MAP, self._CANDIDATES, statuses)

        assert (decision.action, decision.episode_ids) == ("import", (21, 22))

    def test_another_releases_recommended_file_is_left_alone(self) -> None:
        statuses = target_statuses({21: EpisodeFileStatus.RECOMMENDED, 22: EpisodeFileStatus.ABSENT})

        [decision] = plan_import_files(self._MAP, self._CANDIDATES, statuses)

        assert (decision.action, decision.episode_ids) == ("import", (22,))

    def test_our_own_copies_alone_never_trigger_an_import(self) -> None:
        statuses = target_statuses(
            {21: EpisodeFileStatus.RECOMMENDED, 22: EpisodeFileStatus.RECOMMENDED}, frozenset({21, 22})
        )

        [decision] = plan_import_files(self._MAP, self._CANDIDATES, statuses)

        assert decision.action == "skip_done"

    def test_our_copy_on_an_episode_the_file_isnt_placed_on_stays_put(self) -> None:
        statuses = target_statuses({21: EpisodeFileStatus.ABSENT, 23: EpisodeFileStatus.RECOMMENDED}, frozenset({23}))

        [decision] = plan_import_files({"one.mkv": [21]}, {"one.mkv": _candidate("one.mkv")}, statuses)

        assert decision.episode_ids == (21,)
