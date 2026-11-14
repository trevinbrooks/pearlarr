# pyright: strict
"""Placement shapes captured from a live Sonarr, one case per shape.

Two tiers: this catalog holds real captures (whole-series episode lists and raw
`/parse` payloads) with expectations taken from independent truth, never from the
placer's own output. The synthetic edge cases live in `test_placer.py`. Each case
runs through the grab's placement, whose file map the import imports as is.
"""

import json
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, RootModel

from pearlarr.arr_models import ParsedFileInfo, SonarrEpisode
from pearlarr.boundary import Json
from pearlarr.file_paths import normalize_file_name
from pearlarr.grab_placement import ListedFile, place_release
from pearlarr.pending_import import TitleNames
from pearlarr.placement_types import GrabbingTitle, episode_index
from pearlarr.releases import ReleaseEpisode

from .builders import make_grabbing_title, new_torrent

_CATALOG = Path(__file__).parent / "fixtures" / "sonarr" / "placement_shapes.json"


class ShapeTag(StrEnum):
    """The placement shape a case exercises. Every member carries at least one case."""

    SEQUENCE_NO_PARSED_NUMBER = "sequence no parsed number"
    """A 1..N numbered sequence Sonarr parsed nothing for."""
    SEQUENCE_MATCHED_IN_WINDOW = "sequence matched in window"
    """Matched episodes inside the window but shifted by an interleaved special."""
    SEQUENCE_SPELLED_NOT_IN_SERIES = "sequence spelled not in series"
    """Members carry an SxxEyy that resolves nowhere in the series."""
    SEQUENCE_SPELLED_OUTSIDE_WINDOW = "sequence spelled outside window"
    """Members' parsed seasons and episodes resolve to another season and the window is season 0."""
    SEQUENCE_MATCHED_OUTSIDE_WINDOW = "sequence matched outside window"
    """Matched episodes fall outside the window."""
    SEQUENCE_SPELLED_IN_WINDOW = "sequence spelled in window"
    """Members' parsed seasons and episodes resolve inside the window, so the parse stands."""
    SEQUENCE_PARSED_OUT_OF_ORDER = "sequence parsed out of order"
    """Every member reads one distinct id inside the window, not in sequence order."""
    SEQUENCE_BESIDE_PLACED_FILES = "sequence beside placed files"
    """A 1..N numbered sequence with no parsed number beside placed files, the mixed batch that blocks pairing in
    order."""
    SEQUENCE_PICKED_BY_TITLE_NAME = "sequence picked by title name"
    """Several 1..N numbered sequences fit the window (a franchise torrent), and one of the title's names picks one."""
    OUTSIDE_WINDOW = "outside window"
    """A file whose complete parse resolves outside the title."""
    EXACT = "exact"
    ABSOLUTE = "absolute"
    NAME_ORDER = "name order"
    SINGLE = "single"
    TITLE_NAME = "title name"
    """One unplaced file with no parsed number among several, named by one of the title's names."""
    SEQUENCE_PICKED_BY_EPISODE_TITLES = "sequence picked by episode titles"
    """Several numbered sequences fit, and the episode titles the members carry pick one."""
    TITLE_NAMES_ANOTHER_SEQUENCE = "title names another sequence"
    """The one numbered sequence that fits is not the one the title's names point at: rejected."""
    SEASON_SEQUENCE_MATCHED_INTO_SPECIALS = "season sequence matched into specials"
    """A 1..N sequence Sonarr matched into the N-episode season and its specials is the season's own numbering."""
    WINDOW_IN_ABSOLUTE_ORDER = "window in absolute order"
    """The title holds a special TVDB interleaves, so the series' absolute numbering orders the window."""
    WHOLE_SEASON_SEQUENCE = "whole-season sequence"
    """A whole-season numbered sequence listed on one cour's entry places that cour's part."""
    SEQUENCE_WITH_VERSIONS = "sequence with versions"
    """Members come in two versions: the later one is the member, the earlier its duplicate."""
    SEASON_NUMERAL = "season numeral"
    """The title's name and the release number the season differently ("3" against "III")."""
    EXTRA = "extra"
    """An opening, an ending, a preview, a menu, a commercial, a trailer, a teaser, or a promo: set aside first."""
    EPISODE_TITLE_OUTSIDE_WINDOW = "episode title outside window"
    """A file titled as another season's episode, or as one outside the window once the window is full."""


class Basis(StrEnum):
    """What a case's expectation rests on."""

    TITLES_IN_NAMES = "titles in names"
    """The titles the file names carry, matched against the series' episode titles or the title's names."""
    COUNT = "count"
    """A numbered sequence as wide as the window it covers."""
    SONARR_AGREES = "sonarr agrees"
    """Sonarr's own placement, upheld by one of the two above."""
    PINNED = "pinned"
    """A decision, where no evidence decides it."""


class ShapeCase(BaseModel, frozen=True):
    """One captured torrent: its inputs, and where its files belong."""

    name: str
    basis: Basis
    tags: tuple[ShapeTag, ...]
    episodes: tuple[SonarrEpisode, ...]
    """The WHOLE series, as the episode fetch returns it."""
    window_ids: tuple[int, ...]
    """The ids of the title's window, season order."""
    series_name: str = ""
    """Sonarr's name for the series, the words every file shares."""
    anilist_names: tuple[str, ...] = ()
    """The title's AniList names, the placement's tie-break."""
    parses: dict[str, dict[str, Json]]
    """Raw `/parse` payload per file name, in listing order."""
    expected: dict[str, tuple[int, ...]]
    expected_excluded: tuple[str, ...] = ()


class ShapeCatalog(RootModel[tuple[ShapeCase, ...]]):
    """The catalog file: one entry per shape case."""


CASES = ShapeCatalog.model_validate(json.loads(_CATALOG.read_text(encoding="utf-8"))).root
_PARAMS = [pytest.param(case, id=case.name) for case in CASES]


def _names(case: ShapeCase) -> TitleNames:
    return TitleNames(case.series_name, case.anilist_names)


def _grabbing_title(case: ShapeCase) -> GrabbingTitle:
    """The grab-time target: the title's window of the whole-series index, and the names."""

    index = episode_index(case.episodes)
    return make_grabbing_title(
        episode_index([index.by_id[episode_id] for episode_id in case.window_ids]), index, names=_names(case)
    )


def _parsed(case: ShapeCase) -> dict[str, ParsedFileInfo]:
    """The case's raw payloads through the boundary model, listing order kept."""

    return {name: ParsedFileInfo.model_validate(payload) for name, payload in case.parses.items()}


@pytest.mark.parametrize("case", _PARAMS)
def test_the_grab_places_the_captured_shape(case: ShapeCase) -> None:
    """The grab places every captured file where independent truth puts it.

    The records the planner judges coverage by are the placed episodes' numbers, sized per file.
    """

    grabbing_title = _grabbing_title(case)
    parsed = _parsed(case)
    files = [ListedFile(name, size, parsed[name]) for size, name in enumerate(case.parses, start=1)]

    placement = place_release(files, grabbing_title, new_torrent(grabbing_title))

    placements = placement.placements
    assert placements.placed == {normalize_file_name(name): ids for name, ids in case.expected.items()}
    assert {p.name for p in placements.excluded} == {normalize_file_name(name) for name in case.expected_excluded}
    assert list(placement.episodes) == [
        ReleaseEpisode(*grabbing_title.series_index.key_by_id[ep_id], size)
        for size, name in enumerate(case.parses, start=1)
        for ep_id in case.expected.get(name, ())
    ]
    assert placement.inputs_read


def test_every_shape_tag_carries_a_case() -> None:
    """A shape with no capture behind it is a gap in the catalog."""

    assert {tag for case in CASES for tag in case.tags} == set(ShapeTag)
