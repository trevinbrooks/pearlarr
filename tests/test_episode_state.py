# pyright: strict
"""Episode-file statuses and the per-target snapshot (never-overwrite, the trust map it reads)."""

from pearlarr.episode_state import EpisodeFileStatus, EpisodeSnapshot, RecordSnapshot, TargetStatuses
from pearlarr.manual_import import normalize_group
from pearlarr.placement_types import episode_index
from pearlarr.seadex_types import SonarrEpisode

from .builders import sonarr_ep


class TestEpisodeFileStatuses:
    """`EpisodeSnapshot.statuses` classifies each episode's file; the `TargetStatuses` folds derive from that.

    `all_done` is true only when every status is recommended. `needing_import`
    excludes only the recommended ones.
    """

    def test_absent_recommended_other_unknown(self) -> None:
        episodes = [
            sonarr_ep(1, 1, ep_id=1, episode_file_id=0),
            sonarr_ep(1, 2, ep_id=2, episode_file_id=20, release_group="SubGroup"),
            sonarr_ep(1, 3, ep_id=3, episode_file_id=30, release_group="OtherGroup"),
            sonarr_ep(1, 4, ep_id=4, episode_file_id=40, release_group=None),
        ]
        statuses = EpisodeSnapshot(episode_index(episodes), {"subgroup": None}).statuses([1, 2, 3, 4])
        assert statuses.by_id == {
            1: EpisodeFileStatus.ABSENT,
            2: EpisodeFileStatus.RECOMMENDED,
            3: EpisodeFileStatus.OTHER_GROUP,
            4: EpisodeFileStatus.UNKNOWN_GROUP,
        }

    def test_missing_episode_is_absent(self) -> None:
        statuses = EpisodeSnapshot(episode_index([]), {"subgroup": None}).statuses([99])
        assert statuses.by_id == {99: EpisodeFileStatus.ABSENT}

    def test_dash_wrapped_group_counts_as_recommended(self) -> None:
        # Sonarr can report a file's group dash-wrapped ("-Aergia-"). The overwrite
        # guard must still match it against the recommended set built from "Aergia".
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="-Aergia-")]
        statuses = EpisodeSnapshot(episode_index(episodes), {normalize_group("Aergia"): None}).statuses([5])
        assert statuses.by_id == {5: EpisodeFileStatus.RECOMMENDED}

    def test_untagged_file_the_grab_identified_is_recommended(self) -> None:
        # The planner declined to re-download this file because a pick's listed
        # size named it. The import must agree, or it copies over the very file
        # the grab decision called ours - an untagged twin stays unidentifiable.
        episodes = [
            sonarr_ep(1, 1, ep_id=6, episode_file_id=60, release_group=None, size=600),
            sonarr_ep(1, 2, ep_id=7, episode_file_id=70, release_group=None),
        ]
        snapshot = EpisodeSnapshot(episode_index(episodes), {"subgroup": None}, owned_episode_sizes={6: 600})
        assert snapshot.statuses([6, 7]).by_id == {
            6: EpisodeFileStatus.RECOMMENDED,
            7: EpisodeFileStatus.UNKNOWN_GROUP,
        }

    def test_owned_claim_lapses_when_the_file_changed_size(self) -> None:
        # The identification froze at grab time; a DIFFERENT untagged file
        # landing mid-wait must not inherit the claim - honoring the id alone
        # would mark an unverified file done and never send ours.
        episodes = [sonarr_ep(1, 1, ep_id=6, episode_file_id=60, release_group=None, size=601)]
        snapshot = EpisodeSnapshot(episode_index(episodes), {"subgroup": None}, owned_episode_sizes={6: 600})
        assert snapshot.statuses([6]).by_id == {6: EpisodeFileStatus.UNKNOWN_GROUP}

    def test_owned_claim_needs_a_readable_file_record(self) -> None:
        # A truthy episodeFileId with a null/empty episodeFile payload carries
        # no size to verify against, so the claim is refused - not promoted
        # sight-unseen onto a file record Sonarr couldn't even describe.
        ep = SonarrEpisode.model_validate({"id": 6, "seasonNumber": 1, "episodeNumber": 1, "episodeFileId": 60})
        snapshot = EpisodeSnapshot(episode_index([ep]), {"subgroup": None}, owned_episode_sizes={6: 600})
        assert snapshot.statuses([6]).by_id == {6: EpisodeFileStatus.UNKNOWN_GROUP}

    def test_own_group_file_at_a_listed_size_is_recommended(self) -> None:
        # Our own group's file at a size a current listing carries is our copy
        # (just imported, or an already-current episode of the pack).
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="SubGroup", size=1000)]
        snapshot = EpisodeSnapshot(episode_index(episodes), {"subgroup": frozenset({1000})})
        assert snapshot.statuses([5]).by_id == {5: EpisodeFileStatus.RECOMMENDED}

    def test_own_group_file_at_an_unlisted_size_is_replaced(self) -> None:
        # A same-group file at a size NO current listing carries is the stale
        # copy this grab replaces. Reading it as done would close the record
        # with zero imports and strand the upgrade forever.
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="SubGroup", size=999)]
        snapshot = EpisodeSnapshot(episode_index(episodes), {"subgroup": frozenset({1000})})
        assert snapshot.statuses([5]).by_id == {5: EpisodeFileStatus.OTHER_GROUP}

    def test_size_gate_off_for_a_name_trusted_group(self) -> None:
        # A None trust value (an older record, or a blind listing, recorded no
        # sizes) keeps the group-name-only behavior rather than replacing blindly.
        episodes = [sonarr_ep(1, 1, ep_id=5, episode_file_id=50, release_group="SubGroup", size=999)]
        snapshot = EpisodeSnapshot(episode_index(episodes), {"subgroup": None})
        assert snapshot.statuses([5]).by_id == {5: EpisodeFileStatus.RECOMMENDED}

    def test_all_done_only_when_all_recommended(self) -> None:
        rec = TargetStatuses({1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.RECOMMENDED})
        mixed = TargetStatuses({1: EpisodeFileStatus.RECOMMENDED, 2: EpisodeFileStatus.OTHER_GROUP})
        assert rec.all_done() is True
        assert mixed.all_done() is False
        assert TargetStatuses({}).all_done() is False

    def test_needing_import_excludes_only_recommended(self) -> None:
        statuses = TargetStatuses(
            {
                1: EpisodeFileStatus.ABSENT,
                2: EpisodeFileStatus.RECOMMENDED,
                3: EpisodeFileStatus.OTHER_GROUP,
                4: EpisodeFileStatus.UNKNOWN_GROUP,
            }
        )
        assert statuses.needing_import() == {1, 3, 4}


class TestRecordSnapshot:
    """`RecordSnapshot`: one series snapshot per claim series, the window indexes derived, each target routed."""

    @staticmethod
    def _snapshot() -> RecordSnapshot:
        first = EpisodeSnapshot(
            episode_index([sonarr_ep(1, 1, ep_id=1, episode_file_id=10, release_group="SubGroup")]),
            {"subgroup": None},
        )
        second = EpisodeSnapshot(
            episode_index([sonarr_ep(1, 1, ep_id=2, episode_file_id=20, release_group="SubGroup")]),
            {},
        )
        return RecordSnapshot({7: first, 8: second})

    def test_indexes_are_each_series_episode_index(self) -> None:
        snapshot = self._snapshot()

        assert dict(snapshot.indexes) == {7: snapshot.by_series[7].episodes, 8: snapshot.by_series[8].episodes}
        assert (snapshot.series_of(1), snapshot.series_of(2), snapshot.series_of(3)) == (7, 8, None)

    def test_statuses_route_each_target_to_its_series(self) -> None:
        # Each id classifies under the series whose index holds it (the same group is trusted on one
        # and not the other), target order is kept, and an id no index holds is absent.
        statuses = self._snapshot().statuses([2, 3, 1, 2])

        assert list(statuses.by_id.items()) == [
            (2, EpisodeFileStatus.OTHER_GROUP),
            (3, EpisodeFileStatus.ABSENT),
            (1, EpisodeFileStatus.RECOMMENDED),
        ]
