# pyright: strict
"""Reading file names without Sonarr: the episode range a name spells."""

import pytest

from pearlarr.release_names import RangeKey, range_key


class TestRangeKey:
    """`range_key` reads a tight `SxxEyy-zz` range in any letter case, whether or not it counts up."""

    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("show s01e05-07 [1080p].mkv", RangeKey(1, 5, 7)),
            ("show s01e05-e07.mkv", RangeKey(1, 5, 7)),
            ("Show.S01E05-06v2.mkv", RangeKey(1, 5, 6)),
            ("Show.S01E05-06.1080p.mkv", RangeKey(1, 5, 6)),
            ("show s01e07-05.mkv", RangeKey(1, 7, 5)),
        ],
    )
    def test_a_tight_range_is_read(self, name: str, key: RangeKey) -> None:
        assert range_key(name) == key

    @pytest.mark.parametrize(
        "name",
        [
            "show s01e05 - 06 - title.mkv",
            "show s01e05-720p.mkv",
            "show s01e05-10bit.mkv",
            "show s01e05-576i.mkv",
            "show s01e05-06v2a.mkv",
            "show s01e05-6.mkv",
            "show s01e01-2.0.mkv",
            "shows01e05-06.mkv",
            "show s01e05.mkv",
        ],
    )
    def test_a_name_that_spells_no_range_reads_none(self, name: str) -> None:
        assert range_key(name) is None
