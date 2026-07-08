"""Runtime JSON narrowing: ``TypeIs`` guards for walking untyped parsed JSON.

``response.json()`` yields ``Any``; these guards narrow one container level at
a time to the :data:`~.seadex_types.Json` alias, so a walk stays typed without
casts. ``isinstance`` can't inspect element types, so the guarantee is one
level deep - each hop re-narrows the ``Json`` it pulled out, which is exactly
how a JSON walk proceeds.
"""

from typing import TypeIs

from .seadex_types import Json


def is_json_obj(value: object) -> TypeIs[dict[str, Json]]:
    """True when ``value`` is a JSON object (keys are str by construction)."""

    return isinstance(value, dict)


def is_json_list(value: object) -> TypeIs[list[Json]]:
    """True when ``value`` is a JSON array."""

    return isinstance(value, list)
