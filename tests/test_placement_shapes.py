# pyright: strict
"""Placement shapes captured from a live Sonarr, one case per shape.

Two tiers: this catalog holds real captures (whole-series episode lists and raw
`/parse` payloads) with expectations taken from independent truth, never from the
placer's own output. The synthetic edge cases live in `test_placer.py`. Each case
runs through the grab's placement, whose map the import imports as is.
"""

import json
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, RootModel

from pearlarr.arr_models import ParsedFileInfo, SonarrEpisode
from pearlarr.boundary import Json
from pearlarr.file_paths import normalize_basename
from pearlarr.grab_placement import ListedFile, place_release
from pearlarr.pending_import import TitleNames
from pearlarr.placement_types import GrabbingTitle, episode_index
from pearlarr.releases import EpisodeRecord

from .builders import new_torrent, seed_scope

_CATALOG = Path(__file__).parent / "fixtures" / "sonarr" / "placement_shapes.json"


class ShapeTag(StrEnum):
    """The placement shape a case exercises. Every member carries at least one case."""

    RELEASE_RUN_UNREAD = "release run unread"
    """A 1..N run Sonarr read nothing for."""
    RELEASE_RUN_MATCHED_INSIDE = "release run matched inside"
    """Matched pairs inside the window but shifted by an interleaved special."""
    RELEASE_RUN_BOGUS_KEY = "release run bogus key"
    """Members carry an SxxEyy that resolves nowhere in the series."""
    RELEASE_RUN_KEY_OUTSIDE = "release run key outside"
    """Members' own keys resolve to another season and the window is season 0."""
    RELEASE_RUN_MATCHED_OUTSIDE = "release run matched outside"
    """Matched pairs fall outside the window."""
    RELEASE_RUN_KEY_INSIDE = "release run key inside"
    """Members' own keys resolve inside the window, so the reading stands."""
    COHERENT_PERMUTED = "coherent permuted"
    """Every member reads one distinct id inside the window, not in run order."""
    NUMBERED_RUN = "numbered run"
    """A blind 1..N run beside placed files, the mixed batch the ordered zip refuses."""
    TITLED_RUN = "titled run"
    """Several 1..N runs fit the window (a franchise pack), and an entry title picks one."""
    FOREIGN = "foreign"
    """A file whose complete reading resolves outside the entry."""
    EXACT = "exact"
    ABSOLUTE = "absolute"
    ORDERED = "ordered"
    SINGLE = "single"
    TITLED = "titled"
    """One numberless leftover among several, named by an entry title."""
    RUN_TITLED_MEMBERS = "run titled members"
    """Several runs fit, and the episode titles the members carry pick one."""
    RUN_NAMED_ELSEWHERE = "run named elsewhere"
    """The one run that fits is not the one the entry's title names: refused."""
    SEASON_RUN_REREAD = "season run reread"
    """A 1..N run Sonarr matched into the N-episode season and its specials is the season's own numbering."""
    ABSOLUTE_WINDOW = "absolute window"
    """The entry holds a special TVDB interleaves, so the series' absolute numbering orders the window."""
    COVERING_RUN = "covering run"
    """A whole-season run listed on one cour's entry places that cour's slice."""
    VERSIONED_RUN = "versioned run"
    """Members come in two versions: the later one is the member, the earlier its duplicate."""
    SEASON_COUNTED = "season counted"
    """The entry's title and the release count the season differently ("3" against "III")."""
    EXTRA = "extra"
    """An opening, an ending, a preview, a menu, a commercial, a trailer, a teaser, or a promo: set aside first."""
    NAMED_OUTSIDE = "named outside"
    """A file titled as another season's episode, another slice's once the window is full."""


class Provenance(StrEnum):
    """What establishes a case's expectation."""

    TITLE = "title"
    """The titles the file names carry, matched against the series list or the entry's own titles."""
    COUNT = "count"
    """A release run as wide as the window it covers."""
    DISK_AGREES = "disk agrees"
    """Sonarr's own placement, upheld by one of the two above."""
    PINNED = "pinned"
    """A decision, where no evidence decides it."""


class ShapeCase(BaseModel, frozen=True):
    """One captured torrent: its inputs, and where its files belong."""

    name: str
    provenance: Provenance
    tags: tuple[ShapeTag, ...]
    episodes: tuple[SonarrEpisode, ...]
    """The WHOLE series, as the episode fetch returns it."""
    entry_ids: tuple[int, ...]
    """The record's resolved set, season order."""
    series_title: str = ""
    """The Sonarr series title, the words every file shares."""
    titles: tuple[str, ...] = ()
    """The entry's AniList titles, the placement's tie-break."""
    parses: dict[str, dict[str, Json]]
    """Raw `/parse` payload per file name, in listing order."""
    expected: dict[str, tuple[int, ...]]
    expected_excluded: tuple[str, ...] = ()


class ShapeCatalog(RootModel[tuple[ShapeCase, ...]]):
    """The catalog file: one entry per shape case."""


CASES = ShapeCatalog.model_validate(json.loads(_CATALOG.read_text(encoding="utf-8"))).root
_PARAMS = [pytest.param(case, id=case.name) for case in CASES]


def _names(case: ShapeCase) -> TitleNames:
    return TitleNames(case.series_title, case.titles)


def _scope(case: ShapeCase) -> GrabbingTitle:
    """The grab-time scope: the entry's slice of the whole-series index, and the names."""

    index = episode_index(case.episodes)
    return seed_scope(
        episode_index([index.by_id[episode_id] for episode_id in case.entry_ids]), index, names=_names(case)
    )


def _parsed(case: ShapeCase) -> dict[str, ParsedFileInfo]:
    """The case's raw payloads through the boundary model, listing order kept."""

    return {name: ParsedFileInfo.model_validate(payload) for name, payload in case.parses.items()}


@pytest.mark.parametrize("case", _PARAMS)
def test_the_grab_places_the_captured_shape(case: ShapeCase) -> None:
    """The grab places every captured file where independent truth puts it.

    The records the planner judges coverage by are the placed episodes' numbers, sized per file.
    """

    scope = _scope(case)
    parsed = _parsed(case)
    files = [ListedFile(name, size, parsed[name]) for size, name in enumerate(case.parses, start=1)]

    placement = place_release(files, scope, new_torrent(scope))

    assignment = placement.assignment
    assert assignment.placed == {normalize_basename(name): ids for name, ids in case.expected.items()}
    assert {p.name for p in assignment.excluded} == {normalize_basename(name) for name in case.expected_excluded}
    assert list(placement.records) == [
        EpisodeRecord(
            scope.series_index.by_id[ep_id].season_number, scope.series_index.by_id[ep_id].episode_number, size
        )
        for size, name in enumerate(case.parses, start=1)
        for ep_id in case.expected.get(name, ())
    ]
    assert placement.inputs_read


def test_every_shape_tag_carries_a_case() -> None:
    """A shape with no capture behind it is a gap in the catalog."""

    assert {tag for case in CASES for tag in case.tags} == set(ShapeTag)
