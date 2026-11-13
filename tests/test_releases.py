# pyright: strict
"""Direct tests for the release helpers: the infohash spelling and the order-only episode key."""

import pytest

from pearlarr.releases import EpisodeKey, OrderOnlyKey, normalized_infohash

# --- normalized_infohash ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ABCDEF0123  ", "abcdef0123"),
        ("abcdef0123", "abcdef0123"),
        ("   ", None),
        ("", None),
        (None, None),
    ],
)
def test_normalized_infohash_is_stripped_lowercase_or_none(raw: str | None, expected: str | None) -> None:
    assert normalized_infohash(raw) == expected


# --- OrderOnlyKey -------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("dvd", "DVD S07E01"),
        ("altdvd", "alternate DVD S07E01"),
        ("alttwo", "second alternate S07E01"),
        ("regional", "regional S07E01"),
    ],
)
def test_an_order_only_key_is_labelled_under_its_orders_name(order: str, expected: str) -> None:
    assert OrderOnlyKey(order, EpisodeKey(7, 1)).label == expected


def test_an_order_only_key_survives_its_json_form() -> None:
    key = OrderOnlyKey("dvd", EpisodeKey(7, 1))

    assert key.to_json() == {"order": "dvd", "season": 7, "episode": 1}
    assert OrderOnlyKey.from_json(key.to_json()) == key
