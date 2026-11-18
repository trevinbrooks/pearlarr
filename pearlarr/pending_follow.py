"""Following a stored pending import's episode ids onto the Sonarr rows that hold their TVDB episodes now.

A pending import stores the TVDB episode id of every Sonarr row id it holds (`TvdbIds`). Sonarr can give an episode
another row, so a stored id is followed before it is read: put on the row that holds its TVDB episode among the rows
read. `follow` is the one place an id moves.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .arr_models import SonarrEpisode
from .episode_ids import is_stand_in, stand_in_id, stand_in_tvdb_id, without_stand_ins
from .pending_import import FileEpisodeMap, PendingImport, TitleClaim, TvdbIds, tvdb_id_of

type _IdMoves = Mapping[int, int]
"""An episode id a pending import holds -> the id it is now, for each one that moved: the row that holds its TVDB
episode, or that episode's stand-in (`aired_view.stand_in`) when no row does."""

type _RowByTvdb = Mapping[int, int]
"""TVDB episode id -> the id of the row that holds it now, for each TVDB episode a pending import holds."""


@dataclass(frozen=True, slots=True)
class Followed:
    """A stored pending import followed under the rows of every series it claims (`follow`)."""

    pending: PendingImport
    """The pending import with every id on the row that holds its TVDB episode. The stored one itself when no id
    moved, so a caller tells a no-op by identity."""

    set_aside: Mapping[str, tuple[int, ...]]
    """Each file the follow took out of the file map -> its TVDB episodes with no row. Empty when no id moved."""


def follow(pending: PendingImport, rows_by_series: Mapping[int, Sequence[SonarrEpisode]]) -> Followed | None:
    """`pending` with every id on the row that holds its TVDB episode among the rows read, None when it can't be.

    `rows_by_series` holds each series' rows as read, a series that wasn't read left out. None when the pending
    import stores no TVDB ids, when a claimed series wasn't read or has no rows, and when a TVDB episode the
    pending import holds sits on two of the rows: its ids may name other episodes, so nothing may be read from it.

    An id whose episode has no row becomes the episode's stand-in in a window and leaves the preowned ids, and a
    file on it leaves the file map whole for the exclusions. A window's stand-in whose episode has a row becomes
    the row's id. An id stored with no TVDB id stays.
    """

    tvdb_ids = pending.tvdb_ids
    if tvdb_ids is None:
        return None
    row_by_tvdb = _rows_held(pending, tvdb_ids, rows_by_series)
    if row_by_tvdb is None:
        return None
    moves = _moves(row_by_tvdb, tvdb_ids, pending.window_stand_ins())
    if not moves:
        return Followed(pending, {})
    file_map = {
        name: tuple(dict.fromkeys(moves.get(ep_id, ep_id) for ep_id in ids))
        for name, ids in pending.file_episode_map.items()
    }
    followed = replace(
        pending,
        file_episode_map=file_map,
        claims=tuple(_moved_claim(claim, moves) for claim in pending.claims),
        tvdb_ids=_moved_tvdb_ids(tvdb_ids, moves),
    )
    return Followed(followed, _set_aside(pending.file_episode_map, tvdb_ids, row_by_tvdb))


def _rows_held(
    pending: PendingImport, tvdb_ids: TvdbIds, rows_by_series: Mapping[int, Sequence[SonarrEpisode]]
) -> dict[int, int] | None:
    """The row of each TVDB episode `pending` holds (`_RowByTvdb`), None when it can't be followed (`follow`)."""

    held = {tvdb_id for tvdb_id in tvdb_ids.by_row_id.values() if tvdb_id is not None}
    held.update(stand_in_tvdb_id(ep_id) for ep_id in pending.window_stand_ins())
    row_by_tvdb: dict[int, int] = {}
    for series_id in pending.series_ids:
        rows = rows_by_series.get(series_id)
        if not rows:
            return None
        for row in rows:
            tvdb_id = tvdb_id_of(row)
            if tvdb_id not in held or not row.id:
                continue
            if row_by_tvdb.setdefault(tvdb_id, row.id) != row.id:
                return None
    return row_by_tvdb


def _moves(row_by_tvdb: _RowByTvdb, tvdb_ids: TvdbIds, stand_ins: Iterable[int]) -> dict[int, int]:
    """Where each id that isn't where its TVDB episode is now went (`_IdMoves`), from `tvdb_ids` and `stand_ins`.

    Worked out whole before any id is replaced: rows can swap episodes, and replacing ids one at a time would
    undo a swap. A row id held with no TVDB id never moves.
    """

    moves: dict[int, int] = {}
    for row_id, tvdb_id in tvdb_ids.by_row_id.items():
        if tvdb_id is None:
            continue
        now = row_by_tvdb.get(tvdb_id, stand_in_id(tvdb_id))
        if now != row_id:
            moves[row_id] = now
    for ep_id in stand_ins:
        if (now := row_by_tvdb.get(stand_in_tvdb_id(ep_id))) is not None:
            moves[ep_id] = now
    return moves


def _moved_claim(claim: TitleClaim, moves: _IdMoves) -> TitleClaim:
    """The claim with each moved id replaced by the id it is now, both lists deduplicated.

    The window takes it in place, a stand-in too, so it never gets smaller. The preowned ids drop an episode
    that has no row. The coverage strings stay as stored.
    """

    window = dict.fromkeys(moves.get(ep_id, ep_id) for ep_id in claim.ordered_episode_ids)
    preowned = dict.fromkeys(moves.get(ep_id, ep_id) for ep_id in claim.preowned_episode_ids)
    return replace(claim, ordered_episode_ids=tuple(window), preowned_episode_ids=without_stand_ins(preowned))


def _moved_tvdb_ids(tvdb_ids: TvdbIds, moves: _IdMoves) -> TvdbIds:
    """The stored TVDB ids after `moves`.

    A moved row id's entry goes to the id it is now, one whose episode has no row is dropped, and a stand-in
    that has a row gives the row its TVDB episode id.
    """

    moved: dict[int, int | None] = {}
    for row_id, tvdb_id in tvdb_ids.by_row_id.items():
        now = moves.get(row_id, row_id)
        if is_stand_in(now):
            continue
        if tvdb_id is None:
            # Never moved, so `now` is its own id. A row id that moved onto it brings the TVDB id.
            moved.setdefault(now, None)
        else:
            moved[now] = tvdb_id
    for was, now in moves.items():
        if is_stand_in(was):
            moved[now] = stand_in_tvdb_id(was)
    return TvdbIds(moved)


def _set_aside(file_map: FileEpisodeMap, tvdb_ids: TvdbIds, row_by_tvdb: _RowByTvdb) -> dict[str, tuple[int, ...]]:
    """Each file of the stored `file_map` the follow takes out of it -> its TVDB episodes with no row."""

    tvdb_by_row = tvdb_ids.by_row_id
    without_row = {
        name: tuple(
            tvdb_id for ep_id in ids if (tvdb_id := tvdb_by_row.get(ep_id)) is not None and tvdb_id not in row_by_tvdb
        )
        for name, ids in file_map.items()
    }
    return {name: no_row for name, no_row in without_row.items() if no_row}
