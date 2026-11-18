# pyright: strict
"""How a series' numbering at one read compares with its numbering at another."""

import pytest

from pearlarr.tvdb_orders import SeriesNumbering

type OrderAndCount = tuple[str | None, int | None]
"""An order's type name and Sonarr's episode count, as a cached title or a series row holds them."""


@pytest.mark.parametrize(
    ("cached", "now", "differs"),
    [
        pytest.param(("dvd", 12), ("dvd", 12), False, id="same-order-and-count"),
        pytest.param(("dvd", 12), ("altdvd", 12), True, id="two-other-orders"),
        pytest.param(("official", 12), ("dvd", 12), True, id="aired-then-another-order"),
        pytest.param(("dvd", 12), ("official", 12), True, id="another-order-then-aired"),
        pytest.param(("dvd", 12), ("dvd", 13), True, id="another-count-under-another-order"),
        pytest.param(("dvd", None), ("dvd", 13), False, id="no-count-cached"),
        pytest.param(("dvd", 12), ("dvd", None), False, id="no-count-now"),
        pytest.param(("official", 12), ("official", 13), False, id="another-count-as-aired"),
        pytest.param(("", 12), ("official", 12), False, id="no-order-named-and-aired"),
        pytest.param((None, None), ("official", 12), False, id="no-order-cached-and-aired"),
        pytest.param((None, None), ("", 12), False, id="no-order-cached-and-none-named"),
        pytest.param((None, None), ("dvd", 12), True, id="no-order-cached-then-another-order"),
    ],
)
def test_two_numberings_differ_by_their_orders_or_by_the_counts_both_have(
    cached: OrderAndCount, now: OrderAndCount, differs: bool
) -> None:
    assert SeriesNumbering.of(*cached).differs_from(SeriesNumbering.of(*now)) is differs
