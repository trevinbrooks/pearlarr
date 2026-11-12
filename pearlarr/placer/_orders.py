"""Pass 16 of the list in `_passes`: place the files the other passes left open by another TVDB order."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from ._state import NumberedRun, Placer, TorrentFile
from ..placement_types import PlacementVerdict
from ..release_names import sole_title_match
from ..seadex_types import EpisodeKey, OrderOnlyKey


@dataclass(frozen=True, slots=True)
class _OrderFit:
    """How one order reads the files it explains (`_keyed_fit`, `_bare_fit`)."""

    readings: Mapping[TorrentFile, tuple[int, ...]]
    """File -> the episodes its number names in the order."""
    order_only: Mapping[TorrentFile, OrderOnlyKey]
    """File -> the order and the key of its number, which the order gives to an episode Sonarr doesn't list."""
    silent: bool = False
    """Whether a keyed fit places nothing, though it still counts among the explaining orders (`pass_other_orders`)."""

    def judges(self, file: TorrentFile) -> bool:
        """Whether the fit reads the file, onto episodes or onto a key Sonarr doesn't list."""

        return file in self.readings or file in self.order_only


@dataclass(frozen=True, slots=True)
class _KeyedFiles:
    """The torrent's keyed files, as the keyed half judges every order against them."""

    keys: Mapping[TorrentFile, tuple[EpisodeKey, ...]]
    """Each file that spells its own key -> the `(season, episode)` keys its parse names. Extras are left out, and so
    is a lower version of a number, which stays as the passes left it."""
    aired: Mapping[TorrentFile, tuple[int, ...] | None]
    """Each keyed file -> the episodes the aired order reads its keys as, None when Sonarr's list lacks one."""
    settled: Mapping[TorrentFile, tuple[int, ...]]
    """The keyed files already on their episodes -> those episodes (`_settled_ids`)."""
    left_open: tuple[TorrentFile, ...]
    """The keyed files to place that aren't settled and the passes left open, batch order."""
    strays: tuple[TorrentFile, ...]
    """The keyed files not settled whose keys the aired order reads, any of them onto an episode away from home:
    one the window doesn't hold and SeaDex doesn't list for the torrent (`_at_home`)."""

    @classmethod
    def of(cls, state: Placer) -> Self:
        """Read the keyed files off the state."""

        keys: dict[TorrentFile, tuple[EpisodeKey, ...]] = {}
        for file in state.counted:
            info = file.parsed
            if info is None or not file.name_read.explicit_key or info.season_number is None:
                continue
            if info.episode_numbers:
                keys[file] = tuple(EpisodeKey(info.season_number, episode) for episode in info.episode_numbers)
        for run in state.runs(keys):
            for file in run.superseded:
                del keys[file]
        aired = {file: _aired(state, named) for file, named in keys.items()}
        settled = {file: ids for file in keys if (ids := _settled_ids(state, file)) is not None}
        left_open = tuple(file for file in state.placing if file in keys and file not in settled and _left_open(file))
        strays = tuple(
            file for file, ids in aired.items() if file not in settled and ids is not None and not _at_home(state, ids)
        )
        return cls(keys, aired, settled, left_open, strays)


def pass_other_orders(state: Placer) -> None:
    """Place left-open keyed files where their own keys land in another TVDB order that explains the torrent.

    The keyed files are the names that spell their own `SxxEyy`. Home is an episode the window holds or SeaDex
    lists for the torrent. A stray is a keyed file not settled whose keys the aired order reads, any of them onto
    an episode away from home. An order explains the torrent when:

    - it does one thing the aired order can't. Either it reads another way a keyed file the aired order can't read
      (`_answers`): one a count or a title refused, one whose own title names another episode than its key, or one
      whose key Sonarr's list lacks. Or it reads every stray onto episodes at home (`_brings_home`),
    - it reads the keyed files onto distinct episodes,
    - no bracketed count or title in a keyed name contradicts it,
    - it reads every refused special onto an episode SeaDex lists for the torrent (`_refused_special`),
    - it numbers every keyed file already settled (placed by an earlier pass, seeded, or placed by an earlier
      window),
    - and it numbers every keyed file (the pack follows it whole), or a count or title confirms its reading of a
      file this window's own passes left skipped, misnumbered, refuted or foreign, whether or not an earlier
      window placed it ("S10E01 (0337)" where the order's S10E01 is episode 337).

    So a pack the aired order reads whole onto episodes at home is never read another way, and an order that reads
    the files as the aired order does explains nothing.

    When two explaining orders read a left-open file onto different episodes, only the orders whose readings of the
    left-open files SeaDex lists for the torrent are kept. A file the kept orders don't judge alike stays as it was:
    they read it onto different episodes, or one reads it onto an episode and another gives its key to an episode
    Sonarr doesn't list.

    Two kinds of explaining order are silent. One reads a settled file off the episodes it sits on, and no settled
    file moves. The other reads a misnumbered file onto the episodes the earlier passes read it as, which doesn't
    explain why its numbers were refused. A silent order places nothing and gives no file a new verdict, but it
    still counts as an explaining order: a file it judges unlike another order stays as it was, and while it is
    kept there is no one explaining order to name a key Sonarr doesn't list.

    A file the order reads onto the episodes the earlier passes read it as (`Reading.resolved`) is placed only
    when a title refuted it. A title refutes its run's numbers, not the file's own, and an order that explains the
    torrent accounts for the run, whichever file it explains the torrent by. Any other such file stays as the
    passes left it: the order adds nothing to what kept them from placing it there, such as a size that names
    another episode.

    A misnumbered or refuted file read outside the window is claimed for the window that holds the episode
    (`OTHER_ORDER_ELSEWHERE`). A skipped or foreign one keeps its verdict, which already lets that window place it.
    A skipped or foreign file whose key the one explaining order gives to an episode Sonarr doesn't list (a split
    half, a merged pair) is `ORDER_ONLY`: left for a hand import, unless another window places it or proves it a
    duplicate.

    Then the skipped bare-numbered runs: `_place_bare_runs`.
    """

    if not state.series.orders:
        return
    _place_keyed(state)
    _place_bare_runs(state)


def _place_keyed(state: Placer) -> None:
    """Place the left-open keyed files by an order that explains the torrent (the rules of `pass_other_orders`)."""

    files = _KeyedFiles.of(state)
    if not files.left_open:
        return
    fits = [fit for name in state.series.orders if (fit := _keyed_fit(state, files, name)) is not None]
    if _read_apart(fits, files.left_open) and (listed := state.scope.listing.ids):
        fits = [fit for fit in fits if all(listed.issuperset(fit.readings.get(file, ())) for file in files.left_open)]
    speaking = [fit for fit in fits if not fit.silent]
    # Only the one explaining order names a key Sonarr doesn't list, and only when it reads a file onto an episode.
    sole = fits[0] if len(fits) == 1 and speaking and fits[0].readings else None
    for file in files.left_open:
        # Only a file a count or a title refused is claimed for another window.
        refused_by_name = _refused_by_name(file)
        judged = {fit.readings.get(file) for fit in fits if fit.judges(file)}
        if len(judged) != 1:
            continue
        (ids,) = judged
        if ids is None:
            if sole is not None and not refused_by_name:
                state.set_aside_order_only(file, sole.order_only[file])
            continue
        if not any(file in fit.readings for fit in speaking):
            continue
        if ids == file.reading.resolved and file.verdict is not PlacementVerdict.TITLE_REFUTED:
            continue
        if all(state.scope.admits(ep_id) and ep_id not in state.used for ep_id in ids):
            state.place(file, ids, PlacementVerdict.OTHER_ORDER)
        elif refused_by_name and not any(state.scope.admits(ep_id) for ep_id in ids):
            state.set_aside_elsewhere(file, ids)


def _rereads_misnumbered(readings: Mapping[TorrentFile, tuple[int, ...]], left_open: Sequence[TorrentFile]) -> bool:
    """Whether an order's readings put a misnumbered file on the episodes the earlier passes read it as."""

    return any(
        file.verdict is PlacementVerdict.MISNUMBERED and readings.get(file) == file.reading.resolved
        for file in left_open
    )


def _read_apart(fits: Sequence[_OrderFit], left_open: Sequence[TorrentFile]) -> bool:
    """Whether two fits read some left-open file onto different episodes."""

    return any(len({fit.readings[file] for fit in fits if file in fit.readings}) > 1 for file in left_open)


def _left_open(file: TorrentFile) -> bool:
    """Whether the passes left the file open for another order (`PlacementVerdict.left_open`)."""

    return (verdict := file.verdict) is not None and verdict.left_open


def _refused_by_name(file: TorrentFile) -> bool:
    """Whether a count or a title in the names refused the file: left open, and for a hand import."""

    return (verdict := file.verdict) is not None and verdict.left_open and verdict.hand_import


def _settled_ids(state: Placer, file: TorrentFile) -> tuple[int, ...] | None:
    """The episodes the file sits on, else None: an earlier window's, else this call's, else the stored map's."""

    if (earlier := state.scope.earlier.get(file.name)) is not None:
        return earlier
    placed = file.placement
    if placed is not None and placed.verdict.placed:
        return placed.ids
    return file.seeded_on or None


def _keyed_fit(state: Placer, files: _KeyedFiles, name: str) -> _OrderFit | None:
    """How the order `name` reads the keyed files when it explains the torrent (see `pass_other_orders`), else None."""

    order = state.series.orders[name]
    listed = state.scope.listing.ids
    readings: dict[TorrentFile, tuple[int, ...]] = {}
    order_only: dict[TorrentFile, OrderOnlyKey] = {}
    whole = True
    answers = False
    for file, keys in files.keys.items():
        if not order.keys.issuperset(keys):
            if file in files.settled:
                return None
            whole = False
            continue
        if order_only_keys := [key for key in keys if key not in order.id_by_key]:
            order_only[file] = OrderOnlyKey(name, order_only_keys[0])
        else:
            readings[file] = tuple(order.id_by_key[key] for key in keys)
        answers |= _answers(files, file, readings.get(file))
    placed = [ep_id for ids in readings.values() for ep_id in ids]
    if not (answers or _brings_home(state, files, readings)) or len(placed) != len(set(placed)):
        return None
    confirmed = False
    for file, ids in readings.items():
        counted = _counted_episodes(state, file)
        if any(ep_id not in ids for ep_id in counted) or file.contradicts(ids):
            return None
        if _refused_special(file, files.keys[file]) and listed and not listed.issuperset(ids):
            return None
        if _left_open(file):
            confirmed |= any(ep_id in ids for ep_id in counted) or file.confirms(ids)
    if not (whole or confirmed):
        return None
    moves_settled = any(readings.get(file) != ids for file, ids in files.settled.items())
    return _OrderFit(readings, order_only, moves_settled or _rereads_misnumbered(readings, files.left_open))


def _refused_special(file: TorrentFile, keys: Sequence[EpisodeKey]) -> bool:
    """Whether a count or a title refused the file and its keys are specials'.

    A special's number is refused for missing the listing, so an order has to put it on the listing to explain
    it. A season's number is refused for running past the season's end, which an order explains by counting a
    special in, whether or not SeaDex lists that special.
    """

    return _refused_by_name(file) and all(key.season == 0 for key in keys)


def _answers(files: _KeyedFiles, file: TorrentFile, ids: tuple[int, ...] | None) -> bool:
    """Whether an order's reading answers a keyed file the aired order can't read.

    `ids` is the order's reading, None when the order gives the file's key to an episode Sonarr doesn't list. The
    aired order can't read the file when Sonarr's list lacks a key of it, its name refused it, or its title names an
    episode its key doesn't. A reading onto the aired order's own episodes answers nothing, and an episode Sonarr
    doesn't list answers only a file whose key the aired list lacks.
    """

    aired = files.aired[file]
    if aired is None:
        return True
    if ids is None or ids == aired:
        return False
    return _refused_by_name(file) or file.contradicts(aired)


def _brings_home(state: Placer, files: _KeyedFiles, readings: Mapping[TorrentFile, tuple[int, ...]]) -> bool:
    """Whether the torrent has stray keyed files and the order reads every one onto episodes at home (`_at_home`).

    A stray (`_KeyedFiles.strays`) is a file nothing placed, whose key the aired order reads onto an episode that
    the window doesn't hold and SeaDex doesn't list for the torrent. An order explains none of them when it reads
    one onto another such episode, gives its key to an episode Sonarr doesn't list, or has no key for it: the
    torrent still holds a file its listing doesn't account for.
    """

    strays = files.strays
    return bool(strays) and all((ids := readings.get(file)) is not None and _at_home(state, ids) for file in strays)


def _aired(state: Placer, keys: Sequence[EpisodeKey]) -> tuple[int, ...] | None:
    """The episodes the aired order reads the keys as, None when Sonarr's list lacks one of them."""

    id_by_key = state.scope.id_by_key
    ids = tuple(id_by_key[key] for key in keys if key in id_by_key)
    return ids if len(ids) == len(keys) else None


def _at_home(state: Placer, ids: Sequence[int]) -> bool:
    """Whether each episode is one the window holds or SeaDex lists for the torrent."""

    scope = state.scope
    return all(scope.admits(ep_id) or ep_id in scope.listing.ids for ep_id in ids)


def _counted_episodes(state: Placer, file: TorrentFile) -> list[int]:
    """The episodes the name's bracketed counts name as absolute numbers ("(0337)": episode 337)."""

    by_absolute = state.series.id_by_absolute
    return [ep_id for count in file.name_read.tagged_numbers if (ep_id := by_absolute.get(count))]


def _place_bare_runs(state: Placer) -> None:
    """Place the one skipped bare-numbered run an AniList title names, by an order season that holds the window.

    A bare run is the files sharing a prefix before a release number, none spelling a key ("Show S1 - 03"). Only a
    run with more members than the window has episodes is read, since no aired count explains one. The one season
    of an order that holds every window episode reads member n as its nth episode. The order explains the run
    when:

    - it reads every member, none outside the window or past the season's end: a settled one onto the episodes it
      holds, and each one still to place, which must be skipped, onto a distinct unused window episode or a key
      Sonarr doesn't list,
    - its numbering of the window differs from the aired one,
    - and no member's title contradicts it.

    A member every explaining order reads alike is placed `OTHER_ORDER`, or is `ORDER_ONLY` when they all read it
    as a key Sonarr doesn't list. Read apart, it stays skipped, and so does one an earlier window placed.
    """

    runs = state.runs(file for file in state.counted if not file.name_read.explicit_key)
    open_runs = [run for run in runs if any(_skipped(member) for member in run.members)]
    if not open_runs:
        return
    pick = sole_title_match([run.prefix for run in open_runs], state.scope.names)
    if pick is None:
        return
    run = open_runs[pick]
    if len(run.members) <= len(state.scope.real_ids):
        return
    fits = [fit for name in state.series.orders if (fit := _bare_fit(state, name, run)) is not None]
    for member in run.members:
        judging = [fit for fit in fits if fit.judges(member)]
        readings = {fit.readings.get(member) for fit in judging}
        if len(readings) != 1:
            continue
        (ids,) = readings
        if ids is None:
            state.set_aside_order_only(member, judging[0].order_only[member])
        else:
            state.place(member, ids, PlacementVerdict.OTHER_ORDER)


def _skipped(file: TorrentFile) -> bool:
    """Whether the passes left the file skipped."""

    return file.verdict is PlacementVerdict.SKIPPED


def _bare_fit(state: Placer, name: str, run: NumberedRun) -> _OrderFit | None:
    """How the order `name` reads the run's skipped members when it explains the run (`_place_bare_runs`), else None."""

    order = state.series.orders[name]
    window = state.scope.real_ids
    if (season := _season_holding(state, name)) is None:
        return None
    key_by_id = state.series.key_by_id
    order_keys = {key for key, ep_id in order.id_by_key.items() if ep_id in window}
    if order_keys == {key_by_id[ep_id] for ep_id in window if ep_id in key_by_id}:
        return None
    readings: dict[TorrentFile, tuple[int, ...]] = {}
    order_only: dict[TorrentFile, OrderOnlyKey] = {}
    taken: set[int] = set()
    for member in run.members:
        key = EpisodeKey(season, member.release_number)
        ep_id = order.id_by_key.get(key)
        if ep_id is None and key not in order.keys:
            return None
        if ep_id is not None and ep_id not in window:
            return None
        if (settled := _settled_ids(state, member)) is not None:
            if settled != (() if ep_id is None else (ep_id,)):
                return None
        elif not member.to_place:
            continue
        elif not _skipped(member):
            return None
        elif ep_id is None:
            order_only[member] = OrderOnlyKey(name, key)
        elif ep_id in state.used or ep_id in taken or member.contradicts((ep_id,)):
            return None
        else:
            taken.add(ep_id)
            readings[member] = (ep_id,)
    return _OrderFit(readings, order_only) if readings or order_only else None


def _season_holding(state: Placer, name: str) -> int | None:
    """The one season of the order `name` that holds every window episode, else None."""

    window = state.scope.real_ids
    id_by_key = state.series.orders[name].id_by_key
    seasons = {key.season for key, ep_id in id_by_key.items() if ep_id in window}
    if len(seasons) != 1:
        return None
    (season,) = seasons
    held = {ep_id for key, ep_id in id_by_key.items() if key.season == season}
    return season if window <= held else None
