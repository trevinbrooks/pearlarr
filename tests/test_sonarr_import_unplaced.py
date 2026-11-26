# pyright: strict
"""Tests for the files a Sonarr import doesn't place.

They cover what an attempt warns of, a download whose every file the file map lacks, the files the grab left for a
hand import, and the halves of a split opener only another order numbers.
"""

from dataclasses import replace
from typing import ClassVar

import pytest

from pearlarr.arr_models import SonarrEpisode
from pearlarr.file_paths import normalize_file_name
from pearlarr.output import Severity
from pearlarr.pending_import import PendingImport
from pearlarr.releases import EpisodeKey, OrderOnlyKey
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import AttemptKind, AttemptResult, HandImportFile, LeftForHandImport, PauseReason

from .builders import manual_candidate, pending_import, sonarr_ep
from .fakes import FakeSonarrClient, diagnostic_messages, install_recording_hub
from .sonarr_import_builders import make_sonarr_for_import


class TestAutoImportWarningGating:
    """An import warns of what is final and new, once per run. What is early or already reported is not warned of.

    An intended file absent on an early poll is expected (the copy hasn't finished), and a file the grab
    excluded was reported when it was set aside, so neither is warned of. A file that can't be placed is.
    """

    @pytest.mark.parametrize(
        ("attempt_kind", "warnings"),
        [
            pytest.param(
                AttemptKind.ORDINARY,
                ["Show · SubGroup: 1 file could not be placed on an episode and was not imported"],
                id="before-the-deadline",
            ),
            pytest.param(
                AttemptKind.FINAL,
                [
                    "Show · SubGroup: 1 file could not be placed on an episode and was not imported",
                    "/d: 1 intended file not visible to Sonarr for Show · SubGroup - will retry",
                ],
                id="at-the-deadline",
            ),
        ],
    )
    def test_a_file_not_on_disk_is_warned_of_only_at_the_deadline(
        self, attempt_kind: AttemptKind, warnings: list[str]
    ) -> None:
        # The file map intends a file that isn't on disk yet. Early on the copy may just not have finished. The
        # unrelated file beside it can't be placed on any episode, which is final, so either attempt warns of it.
        pending = pending_import(file_episode_map={"Show - 01 [1080p].mkv": [101]})
        strat, _ = make_sonarr_for_import(candidates=[manual_candidate("/d/Unrelated.mkv")])
        recording = install_recording_hub()

        result = strat.import_completed(pending, "/d", attempt_kind)

        assert result.all_done is False
        assert diagnostic_messages(recording, Severity.WARNING) == warnings

    @staticmethod
    def _strat_with_unplaceable(*names: str) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # The file map's file plus `names`, which the file map lacks, beside the empty file map episode.
        return make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), *(manual_candidate(f"/d/{n}") for n in names)],
            episodes=[sonarr_ep(1, 1, ep_id=101, episode_file_id=0)],
        )

    def test_grab_time_excluded_unplaceable_file_never_warns(self) -> None:
        # A sibling pending import's part of a shared torrent was placed at grab time:
        # this pending import still imports its own file, and the unplaceable sibling
        # file was reported when the grab set it aside, so it is not a "could not be placed" warning.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling], excluded_files=[normalize_file_name(sibling)]
        )
        strat, sonarr = self._strat_with_unplaceable(sibling)
        recording = install_recording_hub()

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.command_issued is True
        assert len(sonarr.execute_calls) == 1
        assert not any("could not be placed" in m for m in diagnostic_messages(recording, Severity.WARNING))

    def test_partial_exclusion_warns_the_remainder_once(self) -> None:
        # One excluded plus one genuinely unplaceable file: the warning names
        # only the remainder, and the once-per-run memo is consumed by it.
        sibling = "Show - S00E01 [1080p].mkv"
        pending = pending_import(
            seadex_files=["Show - 01 [1080p].mkv", sibling, "Extras.mkv"],
            excluded_files=[normalize_file_name(sibling)],
        )
        strat, _ = self._strat_with_unplaceable(sibling, "Extras.mkv")
        recording = install_recording_hub()

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        warnings = [m for m in diagnostic_messages(recording, Severity.WARNING) if "could not be placed" in m]
        assert warnings == ["Show · SubGroup: 1 file could not be placed on an episode and was not imported"]


class TestUnmatchedAttemptResult:
    """A download whose every on-disk file the file map lacks is an outcome on its first poll, not a wait.

    The import places nothing, so the next poll would find the same files: no parse or episode read is waited on.
    """

    _NAMES = ("Movie Part 1.mkv", "Movie Part 2.mkv")

    @classmethod
    def _unplaced_pair(cls, *, excluded: tuple[str, ...] = ()) -> PendingImport:
        # Two files the grab placed nowhere, for one window episode.
        return pending_import(
            file_episode_map={},
            ordered_episode_ids=[101],
            seadex_files=list(cls._NAMES),
            excluded_files=[normalize_file_name(name) for name in excluded],
        )

    @classmethod
    def _strat(cls, *, served: bool = True, file_group: str | None = None) -> tuple[SonarrStrategy, FakeSonarrClient]:
        # `file_group` names the group whose file the window episode already holds (None: no file). With `served`
        # off the series has no episodes at all.
        episode = sonarr_ep(1, 1, ep_id=101, release_group=file_group, episode_file_id=1 if file_group else 0)
        return make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in cls._NAMES],
            episodes=[episode] if served else [],
        )

    def test_all_unmatched_returns_the_names_without_a_warning(self) -> None:
        pending = self._unplaced_pair()
        strat, sonarr = self._strat()
        recording = install_recording_hub()

        first = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        second = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert first.unplaced_files == self._NAMES
        assert first.all_done is False
        assert first.command_issued is False
        assert first.pause_reason is PauseReason.NONE
        assert second == first
        assert sonarr.execute_calls == []
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_a_series_with_no_episodes_waits(self) -> None:
        pending = self._unplaced_pair()
        strat, _ = self._strat(served=False)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result == replace(AttemptResult.waiting(), pending=pending)

    @pytest.mark.parametrize(
        ("excluded", "file_group", "unplaced"),
        [
            pytest.param(_NAMES, None, _NAMES, id="every-file-excluded-at-the-grab"),
            pytest.param(_NAMES[1:], None, _NAMES[:1], id="one-file-excluded-at-the-grab"),
            pytest.param((), "OtherPick", _NAMES, id="another-groups-copy-on-the-window-episode"),
        ],
    )
    def test_the_unmatched_names_are_the_files_still_to_import_by_hand(
        self, excluded: tuple[str, ...], file_group: str | None, unplaced: tuple[str, ...]
    ) -> None:
        pending = self._unplaced_pair(excluded=excluded)
        strat, sonarr = self._strat(file_group=file_group)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.unplaced_files == unplaced
        assert sonarr.execute_calls == []


class TestLeftForHandImportFiles:
    """A file the grab left for a hand import keeps the pending import stored once every placed file is in."""

    # Each file's number is one episode early. The episode titles say where the titled two go. The plain file
    # belongs on 601.
    _NAMES = ("show - 01 - The Return [grp].mkv", "show - 02 - Festival Night [grp].mkv", "show - 03 [grp].mkv")
    _TITLES = ("Pilot Flight", "The Return", "Festival Night")
    _SIZE_ON: ClassVar[dict[int, int]] = {601: 1003, 602: 1001, 603: 1002}

    @classmethod
    def _record(cls) -> PendingImport:
        return pending_import(
            file_episode_map={normalize_file_name(cls._NAMES[0]): [602], normalize_file_name(cls._NAMES[1]): [603]},
            hand_import_files=(normalize_file_name(cls._NAMES[2]),),
            seadex_files=list(cls._NAMES),
            release_sizes=tuple(sorted(cls._SIZE_ON.values())),
            ordered_episode_ids=(601, 602, 603),
        )

    @classmethod
    def _episodes(cls, *with_files: int) -> list[SonarrEpisode]:
        """The window, each episode of `with_files` holding the pending import's file at its listed size."""

        return [
            sonarr_ep(1, n, ep_id=ep_id, title=title, size=cls._SIZE_ON[ep_id], release_group="SubGroup")
            if ep_id in with_files
            else sonarr_ep(1, n, ep_id=ep_id, title=title, episode_file_id=0)
            for n, (ep_id, title) in enumerate(zip(cls._SIZE_ON, cls._TITLES, strict=True), start=1)
        ]

    @classmethod
    def _strat(cls, *extras: str) -> tuple[SonarrStrategy, FakeSonarrClient]:
        """The download's files plus `extras` on disk, the placed two in."""

        return make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in (*cls._NAMES, *extras)],
            episodes=cls._episodes(602, 603),
        )

    def test_the_file_left_for_hand_import_keeps_the_pending_import_and_is_named_every_poll(self) -> None:
        strat, sonarr = self._strat()
        recording = install_recording_hub()

        first = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)
        second = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert first.left_for_hand_import.names == (self._NAMES[2],)
        assert first.all_done is False
        assert second == first
        assert sonarr.execute_calls == []
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_a_hand_import_with_no_group_at_a_listed_size_finishes_the_pending_import(self) -> None:
        strat, sonarr = self._strat()
        no_group = sonarr_ep(1, 1, ep_id=601, size=self._SIZE_ON[601], release_group=None)
        sonarr.episodes_return = [no_group, *self._episodes(602, 603)[1:]]

        done = strat.import_completed(self._record(), "/d", AttemptKind.ORDINARY)

        assert done.all_done is True
        assert done.left_for_hand_import.names == ()

    def test_a_plain_skip_alone_finishes(self) -> None:
        strat, _ = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv"), manual_candidate("/d/Extras.mkv")],
            episodes=[
                sonarr_ep(1, 1, ep_id=101, release_group="SubGroup"),
                sonarr_ep(1, 2, ep_id=102, episode_file_id=0),
            ],
        )
        pending = pending_import(seadex_files=["Show - 01 [1080p].mkv", "Extras.mkv"], ordered_episode_ids=[101, 102])

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is True
        assert result.left_for_hand_import.names == ()


class TestOrderOnlyHalves:
    """A name-numbered torrent whose split opener's halves only the DVD order numbers, its other eleven files in."""

    _RUN: ClassVar[list[str]] = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]

    @pytest.mark.parametrize(
        ("opener", "missing", "untrusted"),
        [
            pytest.param(sonarr_ep(4, 1, ep_id=401, episode_file_id=0), (EpisodeKey(4, 1),), (), id="opener-empty"),
            pytest.param(
                sonarr_ep(4, 1, ep_id=401, release_group="other"),
                (),
                (EpisodeKey(4, 1),),
                id="opener-holds-another-groups-file",
            ),
        ],
    )
    def test_the_attempt_result_keys_each_half_by_its_file_name_and_names_the_opener_still_needed(
        self, opener: SonarrEpisode, missing: tuple[EpisodeKey, ...], untrusted: tuple[EpisodeKey, ...]
    ) -> None:
        run = [normalize_file_name(name) for name in self._RUN]
        pending = pending_import(
            release_group="grp",
            seadex_files=self._RUN,
            file_episode_map={name: [400 + n] for n, name in enumerate(run[2:], start=2)},
            hand_import_files=run[:2],
            order_only={
                run[0]: OrderOnlyKey("dvd", EpisodeKey(7, 1)),
                run[1]: OrderOnlyKey("dvd", EpisodeKey(7, 2)),
            },
            ordered_episode_ids=[400 + n for n in range(1, 13)],
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}") for name in self._RUN],
            episodes=[opener, *(sonarr_ep(4, n, ep_id=400 + n, release_group="grp") for n in range(2, 13))],
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.left_for_hand_import == LeftForHandImport(
            files=(
                HandImportFile("[grp] Show Ending S1 - 01 [ABCD1234].mkv", OrderOnlyKey("dvd", EpisodeKey(7, 1))),
                HandImportFile("[grp] Show Ending S1 - 02 [ABCD1234].mkv", OrderOnlyKey("dvd", EpisodeKey(7, 2))),
            ),
            missing=missing,
            untrusted=untrusted,
        )
        assert sonarr.execute_calls == []
