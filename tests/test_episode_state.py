# pyright: strict
"""Episode-file statuses and the per-target snapshot (never-overwrite, the trust map it reads)."""

from pearlarr.episode_state import (
    EpisodeFileStatus,
    EpisodeSnapshot,
    GroupVotes,
    RecordSnapshot,
    Route,
)
from pearlarr.manual_import import GuardFacts, OwnGroup, normalize_group
from pearlarr.placement_types import episode_index
from pearlarr.seadex_types import SonarrEpisode

from .builders import entry_claim, episode_snapshot, pending_import, sonarr_ep, target_statuses


class TestEpisodeFileStatuses:
    """`EpisodeSnapshot.statuses` judges each episode's file, and `TargetStatuses` answers from those judgments.

    `all_done` needs every status recommended, `needing_import` leaves out only the recommended ones, and
    `import_ids` takes an episode holding a copy of our own along with a needing one.
    """

    def test_absent_recommended_other_unknown(self) -> None:
        episodes = [
            sonarr_ep(1, 1, ep_id=1, episode_file_id=0),
            sonarr_ep(1, 2, ep_id=2, episode_file_id=20, release_group="SubGroup"),
            sonarr_ep(1, 3, ep_id=3, episode_file_id=30, release_group="OtherGroup"),
            sonarr_ep(1, 4, ep_id=4, episode_file_id=40, release_group=None),
        ]
        statuses = episode_snapshot(episodes=episode_index(episodes), trusted={"subgroup": None}).statuses(
            [1, 2, 3, 4], {}
        )
        assert statuses.by_id == {
            1: EpisodeFileStatus.ABSENT,
            2: EpisodeFileStatus.RECOMMENDED,
            3: EpisodeFileStatus.OTHER_GROUP,
            4: EpisodeFileStatus.UNKNOWN_GROUP,
        }

    def test_missing_episode_is_absent(self) -> None:
        statuses = episode_snapshot(episodes=episode_index([]), trusted={"subgroup": None}).statuses([99], {})
        assert statuses.by_id == {99: EpisodeFileStatus.ABSENT}

    def test_dash_wrapped_group_counts_as_recommended(self) -> None:
        # Sonarr can report a file's group dash-wrapped ("-Aergia-"). The overwrite
        # guard must still match it against the recommended set built from "Aergia".
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="-Aergia-")]
        statuses = episode_snapshot(
            episodes=episode_index(episodes), trusted={normalize_group("Aergia"): None}
        ).statuses([5], {})
        assert statuses.by_id == {5: EpisodeFileStatus.RECOMMENDED}

    def test_untagged_file_the_grab_identified_is_recommended(self) -> None:
        # The planner declined to re-download this file because a pick's listed
        # size named it. The import must agree, or it copies over the very file
        # the grab decision called ours - an untagged twin stays unidentifiable.
        episodes = [
            sonarr_ep(1, 1, ep_id=6, episode_file_id=60, release_group=None, size=600),
            sonarr_ep(1, 2, ep_id=7, episode_file_id=70, release_group=None),
        ]
        snapshot = episode_snapshot(
            episodes=episode_index(episodes), trusted={"subgroup": None}, owned_episode_sizes={6: 600}
        )
        assert snapshot.statuses([6, 7], {}).by_id == {
            6: EpisodeFileStatus.RECOMMENDED,
            7: EpisodeFileStatus.UNKNOWN_GROUP,
        }

    def test_owned_claim_lapses_when_the_file_changed_size(self) -> None:
        # The identification froze at grab time; a DIFFERENT untagged file
        # landing mid-wait must not inherit the claim - honoring the id alone
        # would mark an unverified file done and never send ours.
        episodes = [sonarr_ep(1, 1, ep_id=6, episode_file_id=60, release_group=None, size=601)]
        snapshot = episode_snapshot(
            episodes=episode_index(episodes), trusted={"subgroup": None}, owned_episode_sizes={6: 600}
        )
        assert snapshot.statuses([6], {}).by_id == {6: EpisodeFileStatus.UNKNOWN_GROUP}

    def test_owned_claim_needs_a_readable_file_record(self) -> None:
        # A truthy episodeFileId with a null/empty episodeFile payload carries
        # no size to verify against, so the claim is refused - not promoted
        # sight-unseen onto a file record Sonarr couldn't even describe.
        ep = SonarrEpisode.model_validate({"id": 6, "seasonNumber": 1, "episodeNumber": 1, "episodeFileId": 60})
        snapshot = episode_snapshot(
            episodes=episode_index([ep]), trusted={"subgroup": None}, owned_episode_sizes={6: 600}
        )
        assert snapshot.statuses([6], {}).by_id == {6: EpisodeFileStatus.UNKNOWN_GROUP}

    def test_own_group_file_at_a_listed_size_is_recommended(self) -> None:
        # Our own group's file at a size a current listing carries is our copy
        # (just imported, or an already-current episode of the pack).
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="SubGroup", size=1000)]
        snapshot = episode_snapshot(episodes=episode_index(episodes), trusted={"subgroup": frozenset({1000})})
        assert snapshot.statuses([5], {}).by_id == {5: EpisodeFileStatus.RECOMMENDED}

    def test_own_group_file_at_an_unlisted_size_is_replaced(self) -> None:
        # A same-group file at a size NO current listing carries is the stale
        # copy this grab replaces. Reading it as done would close the record
        # with zero imports and strand the upgrade forever.
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="SubGroup", size=999)]
        snapshot = episode_snapshot(episodes=episode_index(episodes), trusted={"subgroup": frozenset({1000})})
        assert snapshot.statuses([5], {}).by_id == {5: EpisodeFileStatus.OTHER_GROUP}

    def test_size_gate_off_for_a_name_trusted_group(self) -> None:
        # A None trust value (an older record, or a blind listing, recorded no
        # sizes) keeps the group-name-only behavior rather than replacing blindly.
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="SubGroup", size=999)]
        snapshot = episode_snapshot(episodes=episode_index(episodes), trusted={"subgroup": None})
        assert snapshot.statuses([5], {}).by_id == {5: EpisodeFileStatus.RECOMMENDED}

    @staticmethod
    def _holding(size: int, group: str | None = "SubGroup") -> EpisodeSnapshot:
        """Episode 5 holding `group`'s file at `size`, our release listing 1000 and 2000, an untagged 2000 ours."""

        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group=group, size=size)]
        trusted: dict[str, frozenset[int] | None] = {"subgroup": frozenset({1000, 2000, 3000}), "other": None}
        return episode_snapshot(
            episodes=episode_index(episodes),
            trusted=trusted,
            own=OwnGroup("SubGroup", (1000, 2000)),
            owned_episode_sizes={5: 2000},
        )

    def test_own_file_not_the_intended_one_is_misplaced(self) -> None:
        # Another listed file of our release sits where file 1000 belongs: the episode still lacks its own.
        assert self._holding(2000).judge(5, 1000).status is EpisodeFileStatus.MISPLACED

    def test_own_file_that_is_the_intended_one_is_recommended(self) -> None:
        assert self._holding(2000).judge(5, 2000).status is EpisodeFileStatus.RECOMMENDED

    def test_own_file_with_no_intended_size_is_recommended(self) -> None:
        assert self._holding(2000).judge(5, None).status is EpisodeFileStatus.RECOMMENDED

    def test_statuses_read_each_targets_intended_size(self) -> None:
        assert self._holding(2000).statuses([5], {5: 1000}).by_id == {5: EpisodeFileStatus.MISPLACED}

    def test_own_file_at_an_unlisted_size_is_other_group(self) -> None:
        # The stale-size check runs first: a size no listing carries is the copy this grab replaces.
        assert self._holding(4000).judge(5, 1000).status is EpisodeFileStatus.OTHER_GROUP

    def test_own_file_at_a_size_only_a_sibling_lists_is_recommended(self) -> None:
        # 3000 is trusted through a sibling's listing: not one of our torrent's files, so not a misplaced one.
        assert self._holding(3000).judge(5, 1000).status is EpisodeFileStatus.RECOMMENDED

    def test_a_trusted_groups_file_at_one_of_our_sizes_not_intended_here_is_misplaced(self) -> None:
        # A same-files sibling lists byte-identical files, so its misfile is ours to repair too.
        assert self._holding(2000, "Other").judge(5, 1000).status is EpisodeFileStatus.MISPLACED

    def test_untagged_file_at_the_recorded_size_not_the_intended_one_is_misplaced(self) -> None:
        assert self._holding(2000, None).judge(5, 1000).status is EpisodeFileStatus.MISPLACED

    def test_untagged_file_at_the_recorded_size_that_is_the_intended_one_is_recommended(self) -> None:
        assert self._holding(2000, None).judge(5, 2000).status is EpisodeFileStatus.RECOMMENDED

    @staticmethod
    def _guarded() -> EpisodeSnapshot:
        """An empty index guarded by no facts, voting SubGroup's release at size 1000."""

        return EpisodeSnapshot.guarded(episode_index([]), GuardFacts(), GroupVotes(OwnGroup("-SubGroup-", (1000,))))

    def test_guarded_carries_the_own_release(self) -> None:
        assert self._guarded().own == OwnGroup("-SubGroup-", (1000,))

    def test_a_misplaced_file_is_not_done(self) -> None:
        assert target_statuses({1: EpisodeFileStatus.MISPLACED}).all_done() is False

    def test_all_done_only_when_all_recommended(self) -> None:
        rec = target_statuses({1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.RECOMMENDED})
        mixed = target_statuses({1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.OTHER_GROUP})
        assert rec.all_done() is True
        assert mixed.all_done() is False
        assert target_statuses({}).all_done() is False

    def test_needing_import_excludes_only_recommended(self) -> None:
        statuses = target_statuses(
            {
                1: EpisodeFileStatus.ABSENT,
                2: EpisodeFileStatus.RECOMMENDED,
                3: EpisodeFileStatus.OTHER_GROUP,
                4: EpisodeFileStatus.UNKNOWN_GROUP,
                5: EpisodeFileStatus.MISPLACED,
            }
        )
        assert statuses.needing_import() == {1, 3, 4, 5}

    def test_import_ids_takes_our_copy_along_with_a_needing_id_in_order(self) -> None:
        statuses = target_statuses(
            {1: EpisodeFileStatus.ABSENT, 2: EpisodeFileStatus.RECOMMENDED, 3: EpisodeFileStatus.RECOMMENDED},
            frozenset({2}),
        )
        assert statuses.import_ids([2, 1, 3]) == (2, 1)

    def test_import_ids_is_empty_when_no_id_needs_the_file(self) -> None:
        statuses = target_statuses(
            {1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.RECOMMENDED}, frozenset({1, 2})
        )
        assert statuses.import_ids([1, 2]) == ()

    def test_holding_own_copy_is_the_targets_with_a_file_at_one_of_our_sizes(self) -> None:
        # Episodes 1 and 3 are both done, but only episode 1's file is at one of our listed sizes.
        episodes = [
            sonarr_ep(1, 1, ep_id=1, episode_file_id=10, release_group="SubGroup", size=300),
            sonarr_ep(1, 2, ep_id=2, episode_file_id=0),
            sonarr_ep(1, 3, ep_id=3, episode_file_id=30, release_group="OtherPick", size=900),
        ]
        snapshot = episode_snapshot(
            episodes=episode_index(episodes),
            trusted={"subgroup": frozenset({300}), "otherpick": None},
            own=OwnGroup("SubGroup", (300,)),
        )

        statuses = snapshot.statuses([1, 2, 3], {1: 300, 2: 300, 3: 400})

        assert statuses.by_id == {
            1: EpisodeFileStatus.RECOMMENDED,
            2: EpisodeFileStatus.ABSENT,
            3: EpisodeFileStatus.RECOMMENDED,
        }
        assert statuses.holding_own_copy == {1}


def _series(
    *ep_ids: int, group: str = "SubGroup", trusted: dict[str, frozenset[int] | None] | None = None
) -> EpisodeSnapshot:
    """A one-series snapshot holding `ep_ids`, each with a `group` file, trusting `trusted` (nothing by default)."""

    episodes = [
        sonarr_ep(1, n, ep_id=ep_id, episode_file_id=10 * n, release_group=group)
        for n, ep_id in enumerate(ep_ids, start=1)
    ]
    return episode_snapshot(episodes=episode_index(episodes), trusted=trusted or {})


class TestRecordSnapshot:
    """`RecordSnapshot`: one series snapshot per claim series, the window indexes derived, each target routed."""

    @staticmethod
    def _snapshot(by_claim: dict[int, EpisodeSnapshot] | None = None) -> RecordSnapshot:
        pending = pending_import(
            claims=(entry_claim(al_id=1, series_id=7, ordered_episode_ids=[1]), entry_claim(al_id=2, series_id=8)),
        )
        first = episode_snapshot(
            episodes=episode_index([sonarr_ep(1, 1, ep_id=1, episode_file_id=10, release_group="SubGroup")]),
            trusted={"subgroup": None},
        )
        second = episode_snapshot(
            episodes=episode_index([sonarr_ep(1, 1, ep_id=2, episode_file_id=20, release_group="SubGroup")])
        )
        return RecordSnapshot(pending, {7: first, 8: second}, by_claim or {})

    def test_statuses_judge_each_target_against_its_intended_size(self) -> None:
        # Both episodes hold our file at 200, which is intended on episode 2 only.
        episodes = [
            sonarr_ep(1, 1, ep_id=1, episode_file_id=10, release_group="SubGroup", size=200),
            sonarr_ep(1, 2, ep_id=2, episode_file_id=20, release_group="SubGroup", size=200),
        ]
        series = episode_snapshot(
            episodes=episode_index(episodes),
            trusted={"subgroup": frozenset({100, 200})},
            own=OwnGroup("SubGroup", (100, 200)),
        )
        pending = pending_import(
            file_episode_map={"a.mkv": [1], "b.mkv": [2]}, sizes_by_name={"a.mkv": 100, "b.mkv": 200}
        )
        snapshot = RecordSnapshot(pending, {7: series}, {})

        assert snapshot.statuses([1, 2], pending.file_episode_map).by_id == {
            1: EpisodeFileStatus.MISPLACED,
            2: EpisodeFileStatus.RECOMMENDED,
        }

    def test_statuses_note_the_targets_holding_a_copy_of_our_own(self) -> None:
        # Episode 1 holds a file at one of our listed sizes, episode 2 one at another, and no snapshot judges 3.
        episodes = [
            sonarr_ep(1, 1, ep_id=1, episode_file_id=10, release_group="SubGroup", size=200),
            sonarr_ep(1, 2, ep_id=2, episode_file_id=20, release_group="SubGroup", size=900),
        ]
        series = episode_snapshot(episodes=episode_index(episodes), own=OwnGroup("SubGroup", (100, 200)))
        snapshot = RecordSnapshot(pending_import(), {7: series}, {})

        assert snapshot.statuses([1, 2, 3], {}).holding_own_copy == {1}

    def test_indexes_are_each_series_episode_index(self) -> None:
        snapshot = self._snapshot()

        assert dict(snapshot.indexes) == {7: snapshot.by_series[7].episodes, 8: snapshot.by_series[8].episodes}
        assert (snapshot.series_of(1), snapshot.series_of(2), snapshot.series_of(3)) == (7, 8, None)

    def test_statuses_route_each_target_to_its_series(self) -> None:
        # Each id classifies under the series whose index holds it (the same group is trusted on one
        # and not the other), target order is kept, and an id no index holds is absent.
        statuses = self._snapshot().statuses([2, 3, 1, 2], {})

        assert list(statuses.by_id.items()) == [
            (2, EpisodeFileStatus.OTHER_GROUP),
            (3, EpisodeFileStatus.ABSENT),
            (1, EpisodeFileStatus.RECOMMENDED),
        ]

    def test_a_claimed_id_is_judged_under_its_holding_claims_snapshot(self) -> None:
        # Claim 1's window names id 1, so its own snapshot (the same index, the group untrusted) judges
        # it ahead of the series' merged one. Id 2 routes to the lone unscoped claim 2, whose own snapshot
        # is absent here, so it reads the series'.
        own = episode_snapshot(
            episodes=episode_index([sonarr_ep(1, 1, ep_id=1, episode_file_id=10, release_group="SubGroup")])
        )
        snapshot = self._snapshot(by_claim={1: own})

        assert snapshot.snapshot_for(1) is snapshot.by_claim[1]
        assert snapshot.snapshot_for(2) is snapshot.by_series[8]
        assert snapshot.snapshot_for(3) is None
        assert snapshot.statuses([1, 2], {}).by_id == {
            1: EpisodeFileStatus.OTHER_GROUP,
            2: EpisodeFileStatus.OTHER_GROUP,
        }
        # Without a per-claim snapshot the holding claim's id falls back to the series' view.
        assert self._snapshot().statuses([1], {}).by_id == {1: EpisodeFileStatus.RECOMMENDED}

    def test_route_names_the_first_scoped_claim_holding_the_id(self) -> None:
        # Two windows hold id 2: the earlier claim judges it, as the placer's first window decides, so the
        # later claim's preowned 2 does not read as preowned. Id 3 holds one claim, and 9 holds none.
        first = entry_claim(al_id=1, ordered_episode_ids=[1, 2])
        second = entry_claim(al_id=2, ordered_episode_ids=[2, 3], preowned_episode_ids=[2, 3])
        snapshot = RecordSnapshot(pending_import(claims=(first, second)), {7: _series(1, 2, 3)}, {})

        assert snapshot.route(2) == Route(first, 7)
        assert snapshot.route(3) == Route(second, 7)
        assert snapshot.route(9) == Route(None, None)
        assert (snapshot.preowned(2), snapshot.preowned(3)) == (False, True)

    def test_route_judges_a_series_under_its_lone_unscoped_claim(self) -> None:
        # One unscoped claim on series 7 judges every id its index holds. Two unscoped claims on series 8
        # merge: their ids route to the series alone, and preowned reads off either of them.
        lone = entry_claim(al_id=1)
        pair = (entry_claim(al_id=2, series_id=8), entry_claim(al_id=3, series_id=8, preowned_episode_ids=[5]))
        snapshot = RecordSnapshot(pending_import(claims=(lone, *pair)), {7: _series(1), 8: _series(5)}, {})

        assert snapshot.route(1) == Route(lone, 7)
        assert snapshot.route(5) == Route(None, 8)
        assert snapshot.preowned(5) is True

    def test_an_overlapped_id_is_classified_under_the_first_claims_evidence(self) -> None:
        # A's plan judged group G stale and B's picks carry it. Both windows hold id 2: judged under A, the
        # G file on it still needs importing, where B's own snapshot calls its id 3 done.
        first = entry_claim(al_id=1, ordered_episode_ids=[1, 2])
        second = entry_claim(al_id=2, ordered_episode_ids=[2, 3])
        by_claim = {1: _series(1, 2, 3, group="G"), 2: _series(1, 2, 3, group="G", trusted={"g": None})}
        snapshot = RecordSnapshot(pending_import(claims=(first, second)), {7: _series(1, 2, 3, group="G")}, by_claim)

        assert snapshot.snapshot_for(2) is by_claim[1]
        assert snapshot.statuses([2, 3], {}).by_id == {
            2: EpisodeFileStatus.OTHER_GROUP,
            3: EpisodeFileStatus.RECOMMENDED,
        }
