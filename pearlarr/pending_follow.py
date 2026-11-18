"""Following a stored pending import's episode ids onto the Sonarr rows that hold their TVDB episodes now.

A pending import stores the TVDB episode id of every Sonarr row id it holds (`TvdbIds`). Sonarr can give an episode
another row, so a stored id is followed before it is read: put on the row that holds its TVDB episode among the rows
read. `follow` is the one place an id moves.

`PendingFollower` follows the stored pending imports a title's check reads, and gives one stored before pearlarr
kept TVDB ids its own at the start of a run. The import follows the one it is importing itself
(`ImportReconciler`), under the rows it reads then. Both follow only under rows the run reads the series by
(`SeriesRows`), and both warn of a file a follow set aside through the follower, so it is said once.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .arr_models import SonarrEpisode
from .episode_ids import is_stand_in, stand_in_id, stand_in_tvdb_id, without_stand_ins
from .output import hub_warn
from .output.console_text import pluralize
from .pending_import import FileEpisodeMap, PendingImport, TitleClaim, TvdbIds, tvdb_id_of
from .pending_import_store import PendingImportStore
from .series_reads import SeriesRows

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
    by_file = {
        name: tuple(
            tvdb_id for ep_id in ids if (tvdb_id := tvdb_by_row.get(ep_id)) is not None and tvdb_id not in row_by_tvdb
        )
        for name, ids in file_map.items()
    }
    return {name: episodes for name, episodes in by_file.items() if episodes}


class PendingFollower:
    """Follows a run's stored pending imports for the title checks, and stores the TVDB ids of one that has none.

    Built once in `SonarrStrategy`, over the run's pending import store and its series rows.
    """

    def __init__(self, pending_imports: PendingImportStore, rows: SeriesRows, logger: logging.Logger) -> None:
        """Bind the run's pending import store, its series rows and the logger. No run has started."""

        self._pending_imports = pending_imports
        self._rows = rows
        self.logger = logger

        # The series of every pending import stored once the run has started (`begin_run`).
        self._claimed_series: tuple[int, ...] = ()

        # Each (torrent hash, file name) this run warned of as set aside (`warn_set_aside`).
        self._set_aside_warned: set[tuple[str, str]] = set()

    def begin_run(self, *, waits_on_imports: bool) -> None:
        """Start a run: give each pending import stored with no TVDB ids its own, and note every stored one's series.

        The stored pending imports are loaded once, here. While imports aren't waited on none is loaded, as at
        every other read of the stored ones.
        """

        self._set_aside_warned.clear()
        claimed: dict[int, None] = {}
        if waits_on_imports:
            for pending in self._pending_imports.load(self._pending_imports.rows()).values():
                if pending.tvdb_ids is None and not self._store_tvdb_ids(pending):
                    continue
                claimed.update(dict.fromkeys(pending.series_ids))
        self._claimed_series = tuple(claimed)

    @property
    def claimed_series(self) -> tuple[int, ...]:
        """The series of every pending import stored as the run started, each once.

        A pending import's series is read at grab and import whether or not its own item needs a check, and
        whether or not the series has an AniList mapping. None while imports aren't waited on.
        """

        return self._claimed_series

    def followed(self, stored: Mapping[str, PendingImport]) -> dict[str, PendingImport]:
        """The stored pending imports by hash, each followed under the rows of the series it claims.

        Those are the rows each series' view is built from (`SeriesRows.viewed_episodes`), and so its index: every
        id a placement joins names the episode the index has it on. One whose ids moved is saved, and its titles
        are checked again next run, with a warning for each file the follow set aside (`warn_set_aside`). One
        that can't be followed under them (`follow`) is left out, since its ids may name other episodes. That
        includes one claiming a series with no view this run: the rows kept of a series the run holds may be part
        way through a change of order, with an episode on none of them.
        """

        followed: dict[str, PendingImport] = {}
        for infohash, pending in stored.items():
            rows = {
                series_id: viewed
                for series_id in pending.series_ids
                if (viewed := self._rows.viewed_episodes(series_id)) is not None
            }
            result = follow(pending, rows)
            if result is None:
                self.logger.debug(
                    f"{pending.display_label}: its episode ids can't be matched to the episodes Sonarr sent - "
                    "not placing its torrent's files this run"
                )
                continue
            followed[infohash] = result.pending
            if result.pending is not pending:
                self.warn_set_aside(pending, result.set_aside)
                self._pending_imports.save_followed(result.pending)
        return followed

    def warn_set_aside(self, pending: PendingImport, set_aside: Mapping[str, tuple[int, ...]]) -> None:
        """Warn of each file a follow of `pending` set aside (`Followed.set_aside`), once per file per run.

        A check and an import both follow a stored pending import, and an import follows it again on each attempt
        until one is saved. Whichever sets a file aside first says so.
        """

        for name, tvdb_ids in set_aside.items():
            if (pending.infohash, name) in self._set_aside_warned:
                continue
            self._set_aside_warned.add((pending.infohash, name))
            hub_warn(
                f"{pending.display_label}: Sonarr no longer has "
                f"{pluralize(len(tvdb_ids), 'the episode', 'the episodes')} {name} was matched to "
                f"(TVDB episode {', '.join(map(str, tvdb_ids))}) - leaving the file out of the import"
            )

    def _store_tvdb_ids(self, pending: PendingImport) -> bool:
        """Store the TVDB episode of every row id a pending import written without them holds, or drop it.

        They are read off the rows of its series only when every one is read and in aired order. Its ids were
        matched as though each series were aired, so on a series Sonarr numbers by another order they may name
        other episodes: it is dropped and its titles are checked again, as is one holding an id Sonarr sends no row
        for. One awaiting cleanup is never dropped, since its files are imported and its cleanup ends it by the ids
        it stores. One with a series that couldn't be read is left for the next run.

        Returns whether it is still stored.
        """

        viewed = {series_id: self._rows.viewed_episodes(series_id) for series_id in pending.series_ids}
        if any(rows is not None and self._rows.in_another_order(series_id) for series_id, rows in viewed.items()):
            return not self._drop_unless_imported(
                pending,
                "its episodes were matched before pearlarr could read the order Sonarr numbers its series by - "
                "checking its titles again (episodes it already imported may be on the wrong episodes)",
            )
        sent: list[SonarrEpisode] = []
        for rows in viewed.values():
            if not rows:
                return True
            sent.extend(rows)
        if not pending.row_ids() <= {row.id for row in sent}:
            return not self._drop_unless_imported(
                pending, "Sonarr no longer has an episode it was matched to - checking its titles again"
            )
        self._pending_imports.save(replace(pending, tvdb_ids=TvdbIds.of(sent)))
        return True

    def _drop_unless_imported(self, pending: PendingImport, why: str) -> bool:
        """Drop a pending import whose ids can't be trusted with one warning, unless it is awaiting cleanup.

        Returns whether it was dropped.
        """

        if pending.awaiting_cleanup:
            return False
        self._pending_imports.drop_unimported(pending)
        hub_warn(f"Pending import {pending.display_label} dropped: {why}")
        return True
