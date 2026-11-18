"""A Sonarr series' episodes as TVDB's aired order numbers them, whatever order Sonarr numbers the series by."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Self

from .arr_models import SonarrEpisode, episode_sort_key
from .releases import EpisodeKey, season_episode_key
from .scene_numbers import SceneNumbers
from .tvdb_orders import CachedOrders, EpisodeDetails, OrderRow, SonarrSeriesOrders

_UNTITLED = "TBA"
"""The title Sonarr gives an episode TVDB has no title for."""


class NoView(Enum):
    """Why a series has no aired view this run."""

    READ_FAILED = auto()
    """Sonarr's episodes of the series couldn't be read, its order couldn't be read after them, or the series is
    held for the run (`SonarrEpisodes.aired_series`)."""

    ORDERS_UNREAD = auto()
    """Sonarr numbers the series by another order, and its aired order or its episode details aren't stored."""

    SCENE_UNREAD = auto()
    """Sonarr numbers the series by another order, and TheXEM's scene mappings of it were never read. Sonarr keeps
    no scene numbers for such a series, so without the mappings a release's own numbers can't be matched."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AiredSeries:
    """One series' episodes, numbered as aired.

    For a series in another order, Sonarr's rows are `in_sonarr` plus `own_only` plus the rows with no positive TVDB
    episode id, and the aired episodes are `in_sonarr` plus `stand_ins`.
    """

    order: str
    """The order Sonarr numbers the series by (`seasonType`)."""

    in_sonarr: tuple[SonarrEpisode, ...]
    """Sonarr's episodes, numbered as aired. For a series in aired order they are Sonarr's rows as sent."""

    stand_ins: tuple[SonarrEpisode, ...]
    """One per aired episode Sonarr's order of the series has no row for. Empty for a series in aired order."""

    own_only: tuple[SonarrEpisode, ...] = ()
    """Sonarr's rows for the TVDB episodes the aired episodes lack, as sent: an episode only the series' own order
    lists, or the second of two the aired order gives one number. Empty for a series in aired order."""

    aired: tuple[SonarrEpisode, ...] = field(init=False)
    """Both, in aired order."""

    def __post_init__(self) -> None:
        aired = self.in_sonarr
        if self.stand_ins:
            aired = tuple(sorted((*self.in_sonarr, *self.stand_ins), key=episode_sort_key))
        object.__setattr__(self, "aired", aired)

    def with_scene_numbers(self, numbers: Mapping[EpisodeKey, SceneNumbers]) -> Self:
        """The view with the scene numbers `numbers` has for an episode set on it, a stand-in included.

        `numbers` go by aired season and episode number. An episode they leave out stays as it is.
        """

        return replace(
            self,
            in_sonarr=tuple(_scene_numbered(ep, numbers) for ep in self.in_sonarr),
            stand_ins=tuple(_scene_numbered(ep, numbers) for ep in self.stand_ins),
        )


def is_stand_in(episode_id: int) -> bool:
    """Whether the id is a stand-in's, an episode Sonarr has no row for, and not a Sonarr episode's."""

    return episode_id < 0


def stand_in_id(tvdb_id: int) -> int:
    """The id of the stand-in for a TVDB episode: the same on every run, and never a Sonarr episode's."""

    return -tvdb_id


def stand_in_tvdb_id(episode_id: int) -> int:
    """The TVDB episode id a stand-in's id (`stand_in_id`) stands for."""

    return -episode_id


def stand_in(row: OrderRow, details: EpisodeDetails) -> SonarrEpisode:
    """An aired episode Sonarr's order of the series leaves out, as a Sonarr row with no file.

    Its id is its TVDB episode's `stand_in_id`. A missing title or runtime reads as Sonarr's own rows have it.
    """

    return SonarrEpisode(
        id=stand_in_id(row.tvdb_id),
        season_number=row.season,
        episode_number=row.episode,
        absolute_episode_number=row.absolute,
        title=details.title or _UNTITLED,
        tvdb_id=row.tvdb_id,
        air_date=details.air_date,
        runtime=details.runtime or 0,
    )


def aired_series(
    rows: Sequence[SonarrEpisode], facts: SonarrSeriesOrders, stored: CachedOrders | None
) -> AiredSeries | NoView:
    """The series' aired view, from Sonarr's `rows`, its order `facts` and its `stored` orders.

    A series in aired order keeps its rows untouched, whatever is stored. A series in another order has each row
    renumbered to the aired episode with the same TVDB episode id, and a stand-in for each aired episode none of its
    rows has. Where the aired order gives two episodes one number, the first listed is kept and the other left out,
    as Sonarr keeps them. A row whose TVDB episode is none of the aired episodes kept is kept apart, as sent
    (`AiredSeries.own_only`).

    A renumbered row loses its scene numbers: they were given to the row under another number, so they may be
    another episode's.
    """

    if not facts.in_another_order:
        return AiredSeries(order=facts.season_type, in_sonarr=tuple(rows), stand_ins=())
    if stored is None or (aired := stored.aired) is None or not aired.rows:
        return NoView.ORDERS_UNREAD
    by_tvdb = _one_per_number(aired.rows)
    in_sonarr = sorted(
        (
            ep.model_copy(
                update={
                    "season_number": row.season,
                    "episode_number": row.episode,
                    "absolute_episode_number": row.absolute,
                    "scene_season_number": None,
                    "scene_episode_number": None,
                    "scene_absolute_episode_number": None,
                }
            )
            for ep in rows
            if ep.tvdb_id is not None and (row := by_tvdb.get(ep.tvdb_id)) is not None
        ),
        key=episode_sort_key,
    )
    held = {ep.tvdb_id for ep in in_sonarr}
    return AiredSeries(
        order=facts.season_type,
        in_sonarr=tuple(in_sonarr),
        # Every aired row was stored with its details, from one reply.
        stand_ins=tuple(
            stand_in(row, aired.details[tvdb_id]) for tvdb_id, row in by_tvdb.items() if tvdb_id not in held
        ),
        own_only=tuple(ep for ep in rows if ep.tvdb_id is not None and ep.tvdb_id > 0 and ep.tvdb_id not in by_tvdb),
    )


def numbered_by(rows: Sequence[SonarrEpisode], order: Sequence[OrderRow]) -> bool:
    """Whether Sonarr's `rows` are numbered by `order`'s rows, as far as the TVDB episodes both hold can tell.

    False when a row whose TVDB episode `order` lists sits at a season and episode number `order` doesn't list that
    episode at. One such row is enough, a special among them. A row with no TVDB episode id, or one `order` lacks,
    says nothing.

    Every row of `order` counts, the second of two episodes it gives one number too: a row for that episode at
    any other number isn't numbered by `order`.

    Rows numbered by an order that shares no episode with `order`, or that differs from it only in episodes of its
    own, pass.
    """

    listed_at: dict[int, set[tuple[int, int]]] = {}
    for row in order:
        listed_at.setdefault(row.tvdb_id, set()).add((row.season, row.episode))
    return all(
        (ep.season_number, ep.episode_number) in numbers
        for ep in rows
        if ep.tvdb_id is not None and ep.tvdb_id > 0 and (numbers := listed_at.get(ep.tvdb_id)) is not None
    )


def _scene_numbered(episode: SonarrEpisode, numbers: Mapping[EpisodeKey, SceneNumbers]) -> SonarrEpisode:
    """The episode with its scene numbers from `numbers`, or as it is when they have none for it."""

    scene = numbers.get(season_episode_key(episode.season_number, episode.episode_number))
    if scene is None:
        return episode
    return episode.model_copy(
        update={
            "scene_season_number": scene.season,
            "scene_episode_number": scene.episode,
            "scene_absolute_episode_number": scene.absolute,
        }
    )


def _one_per_number(aired: Sequence[OrderRow]) -> dict[int, OrderRow]:
    """The aired rows by TVDB episode id, the first row of each `(season, episode)` only.

    Sonarr keeps the first of two episodes TVDB gives one number, in the order SkyHook lists them
    (`RefreshEpisodeService.RefreshEpisodeInfo`), and the stored rows keep that order. A row whose TVDB episode id
    isn't positive is left out: negated, it wouldn't mark a stand-in.
    """

    by_tvdb: dict[int, OrderRow] = {}
    numbers: set[tuple[int, int]] = set()
    for row in aired:
        number = (row.season, row.episode)
        if row.tvdb_id > 0 and number not in numbers:
            numbers.add(number)
            by_tvdb[row.tvdb_id] = row
    return by_tvdb
