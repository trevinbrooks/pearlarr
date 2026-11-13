"""Pass 16 of the list in `_passes`: place the files the other passes left open by another TVDB order."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from ._state import NumberedSequence, PlacerState, TorrentFile
from ..file_names import sole_title_match
from ..placement_types import PlacementVerdict
from ..releases import EpisodeKey, OrderOnlyKey


@dataclass(frozen=True, slots=True)
class _OrderFit:
    """How one order reads the files it explains (`_season_episode_fit`, `_name_number_fit`)."""

    episodes_by_file: Mapping[TorrentFile, tuple[int, ...]]
    """File -> the episodes its number names in the order."""
    order_only: Mapping[TorrentFile, OrderOnlyKey]
    """File -> the order and the key of its number, which the order gives to an episode Sonarr doesn't list."""
    places_nothing: bool = False
    """Whether a keyed fit places nothing, though it still counts among the explaining orders (`_place_season_episode_files`)."""

    def judges(self, file: TorrentFile) -> bool:
        """Whether the fit reads the file, onto episodes or onto a key Sonarr doesn't list."""

        return file in self.episodes_by_file or file in self.order_only


@dataclass(frozen=True, slots=True)
class _SeasonEpisodeFile:
    """One file that spells its own key, as the keyed half judges every order against it."""

    file: TorrentFile
    keys: tuple[EpisodeKey, ...]
    """The `(season, episode)` keys its parse names."""
    aired_ids: tuple[int, ...] | None
    """The episodes the aired order reads the keys as, None when Sonarr's list lacks one."""
    placed_ids: tuple[int, ...] | None
    """The episodes the file already sits on (`_final_ids`), None when it sits on none."""


@dataclass(frozen=True, slots=True)
class _SeasonEpisodeFiles:
    """The torrent's keyed files, as the keyed half judges every order against them."""

    files: tuple[_SeasonEpisodeFile, ...]
    """Every file that spells its own key. Extras are left out, and so is a lower version of a number, which stays
    as the passes left it."""
    still_to_place: tuple[TorrentFile, ...]
    """The keyed files to place that aren't settled and the passes left open, batch order."""
    outside_files: tuple[TorrentFile, ...]
    """The keyed files not settled whose keys the aired order reads, any of them onto an episode away from home:
    one the window doesn't hold and SeaDex doesn't list for the torrent (`_in_window_or_listing`)."""

    @classmethod
    def of(cls, state: PlacerState) -> Self:
        """Read the keyed files off the state."""

        keys: dict[TorrentFile, tuple[EpisodeKey, ...]] = {}
        for file in state.non_extras:
            info = file.parse
            if info is None or not file.name_parts.spells_season_episode or info.season_number is None:
                continue
            if info.episode_numbers:
                keys[file] = tuple(EpisodeKey(info.season_number, episode) for episode in info.episode_numbers)
        for sequence in state.sequences_among(keys):
            for file in sequence.superseded:
                del keys[file]
        keyed_files = tuple(
            _SeasonEpisodeFile(file, named, _aired(state, named), _final_ids(state, file))
            for file, named in keys.items()
        )
        provisional = {keyed.file: keyed for keyed in keyed_files if keyed.placed_ids is None}
        still_to_place = tuple(file for file in state.placing if file in provisional and _other_order_may_place(file))
        outside_files = tuple(
            keyed.file
            for keyed in provisional.values()
            if keyed.aired_ids is not None and not _in_window_or_listing(state, keyed.aired_ids)
        )
        return cls(keyed_files, still_to_place, outside_files)


def pass_other_orders(state: PlacerState) -> None:
    """Place left-open keyed files where their own keys land in another TVDB order that explains the torrent.

    It is the only pass that gives a judged file a new verdict, and it moves no file an earlier pass, a seed or an
    earlier window placed.

    The keyed files are the names that spell their own `SxxEyy`. Home is an episode the window holds or SeaDex
    lists for the torrent. A stray is a keyed file not settled whose keys the aired order reads, any of them onto
    an episode away from home. An order explains the torrent when:

    - it does one thing the aired order can't. Either it reads another way a keyed file the aired order can't read
      (`_answers`): one a count or a title refused, one whose own title names another episode than its key, or one
      whose key Sonarr's list lacks. Or it reads every stray onto episodes at home (`_brings_outside_files_in`),
    - it reads the keyed files onto distinct episodes,
    - no bracketed count or title in a keyed name contradicts it,
    - it reads every refused special onto an episode SeaDex lists for the torrent (`_final_unplaced_special`),
    - it numbers every keyed file already settled (placed by an earlier pass, seeded, or placed by an earlier
      window),
    - and it numbers every keyed file (the pack follows it whole), or a count or title confirms its reading of a
      file this window's own passes left skipped, misnumbered, refuted or foreign, whether or not an earlier
      window placed it ("S10E01 (0337)" where the order's S10E01 is episode 337).

    So a pack the aired order reads whole onto episodes at home is never read another way, and an order that reads
    the files as the aired order does explains nothing. `_place_season_episode_files` places the left-open files by the explaining
    orders.

    Then the skipped bare-numbered runs: `_place_name_number_sequences`.
    """

    if not state.index.orders:
        return
    _place_season_episode_files(state)
    _place_name_number_sequences(state)


def _place_season_episode_files(state: PlacerState) -> None:
    """Place the left-open keyed files by the orders that explain the torrent (`pass_other_orders`).

    A left-open file is placed where its keys land in the explaining orders (`OTHER_ORDER`), when the window holds
    those episodes and no file has them yet. When two explaining orders read a left-open file onto different
    episodes, only the orders whose readings of the left-open files SeaDex lists for the torrent are kept. A file
    the kept orders don't judge alike stays as it was: they read it onto different episodes, or one reads it onto
    an episode and another gives its key to an episode Sonarr doesn't list.

    Two kinds of explaining order are silent. One reads a settled file off the episodes it sits on, and no settled
    file moves. The other reads a misnumbered file onto the episodes the earlier passes read it as, which doesn't
    explain why its numbers were refused. A silent order places nothing and gives no file a new verdict, but it
    still counts as an explaining order: a file it judges unlike another order stays as it was, and while it is
    kept there is no one explaining order to name a key Sonarr doesn't list.

    A file the order reads onto the episodes the earlier passes read it as (`ParsedEpisodes.ids`) is placed only
    when a title refuted it. A title refutes its run's numbers, not the file's own, and an order that explains the
    torrent accounts for the run, whichever file it explains the torrent by. Any other such file stays as the
    passes left it: the order adds nothing to what kept them from placing it there, such as a size that names
    another episode.

    A misnumbered or refuted file read outside the window is claimed for the window that holds the episode
    (`OTHER_ORDER_OUTSIDE_WINDOW`). A skipped or foreign one keeps its verdict, which already lets that window place it.
    A skipped or foreign file whose key the one explaining order gives to an episode Sonarr doesn't list (a split
    half, a merged pair) is `ORDER_ONLY`: left for a hand import, unless another window places it or proves it a
    duplicate.
    """

    files = _SeasonEpisodeFiles.of(state)
    if not files.still_to_place:
        return
    fits = [fit for name in state.index.orders if (fit := _season_episode_fit(state, files, name)) is not None]
    if _fits_disagree(fits, files.still_to_place) and (listed := state.target.seadex.ids):
        fits = [
            fit
            for fit in fits
            if all(listed.issuperset(fit.episodes_by_file.get(file, ())) for file in files.still_to_place)
        ]
    placing_fits = [fit for fit in fits if not fit.places_nothing]
    # Only the one explaining order names a key Sonarr doesn't list, and only when it reads a file onto an episode.
    sole = fits[0] if len(fits) == 1 and placing_fits and fits[0].episodes_by_file else None
    for file in files.still_to_place:
        # Only a file a count or a title refused is claimed for another window.
        final_unplaced_by_name = _final_unplaced_by_name(file)
        judged = {fit.episodes_by_file.get(file) for fit in fits if fit.judges(file)}
        if len(judged) != 1:
            continue
        (ids,) = judged
        if ids is None:
            if sole is not None and not final_unplaced_by_name:
                state.set_aside_order_only(file, sole.order_only[file])
            continue
        if not any(file in fit.episodes_by_file for fit in placing_fits):
            continue
        if ids == file.parsed_episodes.ids and file.verdict is not PlacementVerdict.TITLE_REFUTED:
            continue
        if all(state.target.admits(ep_id) and ep_id not in state.used for ep_id in ids):
            state.place(file, ids, PlacementVerdict.OTHER_ORDER)
        elif final_unplaced_by_name and not any(state.target.admits(ep_id) for ep_id in ids):
            state.set_aside_outside_window(file, ids)


def _repeats_misnumbered(
    episodes_by_file: Mapping[TorrentFile, tuple[int, ...]], still_to_place: Sequence[TorrentFile]
) -> bool:
    """Whether an order's readings put a misnumbered file on the episodes the earlier passes read it as."""

    return any(
        file.verdict is PlacementVerdict.MISNUMBERED and episodes_by_file.get(file) == file.parsed_episodes.ids
        for file in still_to_place
    )


def _fits_disagree(fits: Sequence[_OrderFit], still_to_place: Sequence[TorrentFile]) -> bool:
    """Whether two fits read some left-open file onto different episodes."""

    return any(
        len({fit.episodes_by_file[file] for fit in fits if file in fit.episodes_by_file}) > 1 for file in still_to_place
    )


def _other_order_may_place(file: TorrentFile) -> bool:
    """Whether the passes left the file open for another order (`PlacementVerdict.left_open`)."""

    return (verdict := file.verdict) is not None and verdict.other_order_may_place


def _final_unplaced_by_name(file: TorrentFile) -> bool:
    """Whether a count or a title in the names refused the file: left open, and for a hand import."""

    return (verdict := file.verdict) is not None and verdict.other_order_may_place and verdict.left_for_hand_import


def _final_ids(state: PlacerState, file: TorrentFile) -> tuple[int, ...] | None:
    """The episodes the file sits on, else None: an earlier window's, else this call's, else the stored map's."""

    if (earlier := state.target.earlier.get(file.name)) is not None:
        return earlier
    placed = file.placement
    if placed is not None and placed.verdict.placed:
        return placed.ids
    return file.saved_ids or None


def _season_episode_fit(state: PlacerState, files: _SeasonEpisodeFiles, name: str) -> _OrderFit | None:
    """How the order `name` reads the keyed files when it explains the torrent (see `pass_other_orders`), else None."""

    order = state.index.orders[name]
    listed = state.target.seadex.ids
    episodes_by_file: dict[TorrentFile, tuple[int, ...]] = {}
    order_only: dict[TorrentFile, OrderOnlyKey] = {}
    whole = True
    answers = False
    for keyed in files.files:
        file = keyed.file
        if not order.keys.issuperset(keyed.keys):
            if keyed.placed_ids is not None:
                return None
            whole = False
            continue
        if order_only_keys := [key for key in keyed.keys if key not in order.id_by_key]:
            order_only[file] = OrderOnlyKey(name, order_only_keys[0])
        else:
            episodes_by_file[file] = tuple(order.id_by_key[key] for key in keyed.keys)
        answers |= _answers(keyed, episodes_by_file.get(file))
    placed = [ep_id for ids in episodes_by_file.values() for ep_id in ids]
    if not (answers or _brings_outside_files_in(state, files, episodes_by_file)) or len(placed) != len(set(placed)):
        return None
    confirmed = False
    for keyed in files.files:
        file = keyed.file
        if (ids := episodes_by_file.get(file)) is None:
            continue
        bracketed = _bracketed_number_episodes(state, file)
        if any(ep_id not in ids for ep_id in bracketed) or file.contradicts(ids):
            return None
        if _final_unplaced_special(keyed) and listed and not listed.issuperset(ids):
            return None
        if _other_order_may_place(file):
            confirmed |= any(ep_id in ids for ep_id in bracketed) or file.confirms(ids)
    if not (whole or confirmed):
        return None
    moves_final = any(
        keyed.placed_ids is not None and episodes_by_file.get(keyed.file) != keyed.placed_ids for keyed in files.files
    )
    return _OrderFit(
        episodes_by_file, order_only, moves_final or _repeats_misnumbered(episodes_by_file, files.still_to_place)
    )


def _final_unplaced_special(keyed: _SeasonEpisodeFile) -> bool:
    """Whether a count or a title refused the file and its keys are specials'.

    A special's number is refused for missing the listing, so an order has to put it on the listing to explain
    it. A season's number is refused for running past the season's end, which an order explains by counting a
    special in, whether or not SeaDex lists that special.
    """

    return _final_unplaced_by_name(keyed.file) and all(key.season == 0 for key in keyed.keys)


def _answers(keyed: _SeasonEpisodeFile, ids: tuple[int, ...] | None) -> bool:
    """Whether an order's reading answers a keyed file the aired order can't read.

    `ids` is the order's reading, None when the order gives the file's key to an episode Sonarr doesn't list. The
    aired order can't read the file when Sonarr's list lacks a key of it, its name refused it, or its title names an
    episode its key doesn't. A reading onto the aired order's own episodes answers nothing, and an episode Sonarr
    doesn't list answers only a file whose key the aired list lacks.
    """

    aired = keyed.aired_ids
    if aired is None:
        return True
    if ids is None or ids == aired:
        return False
    return _final_unplaced_by_name(keyed.file) or keyed.file.contradicts(aired)


def _brings_outside_files_in(
    state: PlacerState, files: _SeasonEpisodeFiles, episodes_by_file: Mapping[TorrentFile, tuple[int, ...]]
) -> bool:
    """Whether the torrent has stray keyed files and the order reads every one onto episodes at home (`_in_window_or_listing`).

    A stray (`_SeasonEpisodeFiles.outside_files`) is a file nothing placed, whose key the aired order reads onto an episode that
    the window doesn't hold and SeaDex doesn't list for the torrent. An order explains none of them when it reads
    one onto another such episode, gives its key to an episode Sonarr doesn't list, or has no key for it: the
    torrent still holds a file its listing doesn't account for.
    """

    outside_files = files.outside_files
    return bool(outside_files) and all(
        (ids := episodes_by_file.get(file)) is not None and _in_window_or_listing(state, ids) for file in outside_files
    )


def _aired(state: PlacerState, keys: Sequence[EpisodeKey]) -> tuple[int, ...] | None:
    """The episodes the aired order reads the keys as, None when Sonarr's list lacks one of them."""

    id_by_key = state.target.id_by_key
    ids = tuple(id_by_key[key] for key in keys if key in id_by_key)
    return ids if len(ids) == len(keys) else None


def _in_window_or_listing(state: PlacerState, ids: Sequence[int]) -> bool:
    """Whether each episode is one the window holds or SeaDex lists for the torrent."""

    target = state.target
    return all(target.admits(ep_id) or ep_id in target.seadex.ids for ep_id in ids)


def _bracketed_number_episodes(state: PlacerState, file: TorrentFile) -> list[int]:
    """The episodes the name's bracketed counts name as absolute numbers ("(0337)": episode 337)."""

    by_absolute = state.index.id_by_absolute
    return [ep_id for count in file.name_parts.bracketed_numbers if (ep_id := by_absolute.get(count))]


def _place_name_number_sequences(state: PlacerState) -> None:
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

    sequences = state.sequences_among(file for file in state.non_extras if not file.name_parts.spells_season_episode)
    unjudged_sequences = [
        sequence for sequence in sequences if any(_no_episode_found(member) for member in sequence.members)
    ]
    if not unjudged_sequences:
        return
    chosen = sole_title_match([sequence.prefix for sequence in unjudged_sequences], state.target.title_names)
    if chosen is None:
        return
    sequence = unjudged_sequences[chosen]
    if len(sequence.members) <= len(state.target.real_ids):
        return
    fits = [fit for name in state.index.orders if (fit := _name_number_fit(state, name, sequence)) is not None]
    for member in sequence.members:
        judging = [fit for fit in fits if fit.judges(member)]
        episodes_by_file = {fit.episodes_by_file.get(member) for fit in judging}
        if len(episodes_by_file) != 1:
            continue
        (ids,) = episodes_by_file
        if ids is None:
            state.set_aside_order_only(member, judging[0].order_only[member])
        else:
            state.place(member, ids, PlacementVerdict.OTHER_ORDER)


def _no_episode_found(file: TorrentFile) -> bool:
    """Whether the passes left the file skipped."""

    return file.verdict is PlacementVerdict.NO_EPISODE_FOUND


def _name_number_fit(state: PlacerState, name: str, sequence: NumberedSequence) -> _OrderFit | None:
    """How the order `name` reads the run's skipped members when it explains the run (`_place_name_number_sequences`), else None."""

    order = state.index.orders[name]
    window = state.target.real_ids
    if (season := _season_holding(state, name)) is None:
        return None
    key_by_id = state.index.key_by_id
    order_keys = {key for key, ep_id in order.id_by_key.items() if ep_id in window}
    if order_keys == {key_by_id[ep_id] for ep_id in window if ep_id in key_by_id}:
        return None
    episodes_by_file: dict[TorrentFile, tuple[int, ...]] = {}
    order_only: dict[TorrentFile, OrderOnlyKey] = {}
    used: set[int] = set()
    for member in sequence.members:
        key = EpisodeKey(season, member.name_number)
        ep_id = order.id_by_key.get(key)
        if ep_id is None and key not in order.keys:
            return None
        if ep_id is not None and ep_id not in window:
            return None
        if (final := _final_ids(state, member)) is not None:
            if final != (() if ep_id is None else (ep_id,)):
                return None
        elif not member.to_place:
            continue
        elif not _no_episode_found(member):
            return None
        elif ep_id is None:
            order_only[member] = OrderOnlyKey(name, key)
        elif ep_id in state.used or ep_id in used or member.contradicts((ep_id,)):
            return None
        else:
            used.add(ep_id)
            episodes_by_file[member] = (ep_id,)
    return _OrderFit(episodes_by_file, order_only) if episodes_by_file or order_only else None


def _season_holding(state: PlacerState, name: str) -> int | None:
    """The one season of the order `name` that holds every window episode, else None."""

    window = state.target.real_ids
    id_by_key = state.index.orders[name].id_by_key
    seasons = {key.season for key, ep_id in id_by_key.items() if ep_id in window}
    if len(seasons) != 1:
        return None
    (season,) = seasons
    in_season = {ep_id for key, ep_id in id_by_key.items() if key.season == season}
    return season if window <= in_season else None
