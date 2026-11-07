"""`name_facts`: what the passes read off each name of a torrent, for the code that runs after the windows merge."""

from collections.abc import Mapping
from dataclasses import dataclass

from ._state import Placer
from ..placement_types import PlacementBatch, TargetScope


@dataclass(frozen=True, slots=True, kw_only=True)
class NameFacts:
    """What the passes read off one name under one scope, without placing it."""

    titled_as: int | None
    """The episode the name is titled as (`TorrentFile.titled_as`), else None."""
    reads_inside: bool
    """Whether the name's parse resolves wholly inside the scope: the exact pass would place it there, were its ids
    open and nothing disputing them."""
    extra: bool
    """Whether the name is an extra (an opening, an ending, and so on): never an episode."""
    title_counted: bool
    """Whether Sonarr counted the name by its own title's number and nothing confirms it (`Placer.counted_by_title`):
    the number is the title's, not a count."""


def name_facts(batch: PlacementBatch, scope: TargetScope) -> Mapping[str, NameFacts]:
    """The passes' reading of every name of `batch` under `scope`, seeded and gone names included.

    The names are read as `assign_episode_ids` reads them before its first pass, so the facts are the same ones the
    passes judge by. Nothing is placed.
    """

    state = Placer.start(batch, scope)
    return {
        file.name: NameFacts(
            titled_as=file.titled_as,
            reads_inside=bool(file.reading.exact_ids),
            extra=file.extra,
            title_counted=state.counted_by_title(file),
        )
        for file in state.files
    }
