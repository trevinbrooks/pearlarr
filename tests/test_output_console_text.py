# pyright: strict
"""Direct tests for the console text formatters: `compact_duration` across every magnitude."""

from pearlarr.output.console_text import compact_duration


class TestCompactDuration:
    """`compact_duration` covers every magnitude and floors negatives to "0s"."""

    def test_covers_every_magnitude(self) -> None:
        assert compact_duration(4000) == "1h06m"
        assert compact_duration(130) == "2m"
        assert compact_duration(40) == "40s"
        assert compact_duration(-5) == "0s"
