# pyright: strict
"""The series' episode titles folded into words beyond the series' name."""

from collections.abc import MutableMapping
from typing import cast

import pytest

from pearlarr.episode_titles import EpisodeTitles
from pearlarr.releases import EpisodeKey

from .builders import series_index


class TestEpisodeTitlesOf:
    """`EpisodeTitles.of` converts an index's episode titles into read-only groupings."""

    def test_the_groupings_are_read_only(self) -> None:
        index = series_index(
            {EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12}, titles={11: "Show: The Long Road Home", 12: "A Quiet Morning"}
        )

        view = EpisodeTitles.of(index.by_id, index.id_by_key, frozenset({"show"}))

        with pytest.raises(TypeError):
            cast("MutableMapping[tuple[str, ...], tuple[()]]", view.by_first)[("x",)] = ()
        with pytest.raises(TypeError):
            cast("MutableMapping[tuple[str, ...], tuple[()]]", view.alone_by_beyond)[("x",)] = ()
