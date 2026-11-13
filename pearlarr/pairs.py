"""Key-value pairs into a dict that keeps only the keys whose value never disagrees."""

from collections.abc import Iterable


def unambiguous[K, V](pairs: Iterable[tuple[K, V]]) -> dict[K, V]:
    """Map each key to its value, dropping any key that shows up with two different values."""

    values: dict[K, V] = {}
    ambiguous: set[K] = set()
    for key, value in pairs:
        if values.setdefault(key, value) != value:
            ambiguous.add(key)
    return {key: value for key, value in values.items() if key not in ambiguous}
