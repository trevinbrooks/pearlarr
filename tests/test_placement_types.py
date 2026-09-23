# pyright: strict
"""The `(season, episode) -> id` index: both facets, the fetch order, zero ids dropped, the read-only detachment.

The other placement types are exercised through `assign_episode_ids` in `test_placer` and
`test_manual_import_fixtures`.
"""

from collections.abc import MutableMapping
from typing import cast

import pytest

from pearlarr.placement_types import EpisodeIndex, episode_index
from pearlarr.seadex_types import SONARR_MISSING_KEY, EpisodeKey, SonarrEpisode

from .builders import sonarr_ep


class TestEpisodeIndex:
    """`episode_index` folds an episode fetch into the import family's two facets.

    `id_by_key` maps `(season, episode)` to the first episode id, missing numbers
    folding to a sentinel key (no collision with real pairs). Zero-id episodes are
    dropped from every facet before keying. `by_id` keeps the fetch order - the
    resolved set the add flow persists rides `list(by_id)`.
    """

    def test_normal_seasoned_episodes(self) -> None:
        eps = [
            sonarr_ep(1, 1, ep_id=11, episode_file_id=0),
            sonarr_ep(1, 2, ep_id=12, episode_file_id=0),
            sonarr_ep(2, 1, ep_id=21, episode_file_id=0),
        ]
        index = episode_index(eps)
        assert index.id_by_key == {(1, 1): 11, (1, 2): 12, (2, 1): 21}
        assert tuple(index.by_id) == (11, 12, 21)

    def test_missing_season_and_episode_use_sentinel_no_collision(self) -> None:
        eps = [
            sonarr_ep(None, None, ep_id=5, episode_file_id=0),
            sonarr_ep(1, 1, ep_id=6, episode_file_id=0),
        ]
        result = episode_index(eps).id_by_key
        assert result[EpisodeKey(SONARR_MISSING_KEY, SONARR_MISSING_KEY)] == 5
        assert result[EpisodeKey(1, 1)] == 6

    def test_first_wins_on_duplicate_key(self) -> None:
        eps = [sonarr_ep(1, 1, ep_id=7, episode_file_id=0), sonarr_ep(1, 1, ep_id=8, episode_file_id=0)]
        assert episode_index(eps).id_by_key == {(1, 1): 7}

    def test_zero_id_dropped_before_keying(self) -> None:
        # A real-id twin behind a zero-id record must still win its key.
        eps = [sonarr_ep(1, 2, ep_id=0, episode_file_id=0), sonarr_ep(1, 2, ep_id=9, episode_file_id=0)]
        index = episode_index(eps)
        assert index.id_by_key == {(1, 2): 9}
        assert tuple(index.by_id) == (9,)

    def test_the_title_and_absolute_number_ride_the_record(self) -> None:
        raw = {"id": 6, "seasonNumber": 1, "episodeNumber": 1, "absoluteEpisodeNumber": 13, "title": "Beach Day"}
        ep = SonarrEpisode.model_validate({**raw, "episodeFileId": 0})
        bare = SonarrEpisode.model_validate({"id": 6, "seasonNumber": 1, "episodeNumber": 1, "episodeFileId": 0})

        assert (ep.absolute_episode_number, ep.title) == (13, "Beach Day")
        assert (bare.absolute_episode_number, bare.title) == (None, "")

    def test_facets_detach_and_reject_mutation(self) -> None:
        source = {EpisodeKey(1, 1): 11}
        index = EpisodeIndex(by_id={}, id_by_key=source)

        source[EpisodeKey(1, 2)] = 12

        assert EpisodeKey(1, 2) not in index.id_by_key
        with pytest.raises(TypeError):
            cast("MutableMapping[EpisodeKey, int]", index.id_by_key)[EpisodeKey(1, 3)] = 13
