"""`place_batch`, the pass list below, and every pass not in `_sequence_passes` or `_orders`.

The title's episode ids are authoritative. Sonarr's parse of a file name, and the number the name carries, only
ever choose among them.

The passes run in this order. Each places files only onto the ids that earlier passes left unused:

1. Extras (`_pass_extras`): `EXTRA`.
2. Title contradictions (`_pass_reject_contradicted`): rejects the parses a title contradicts or no title confirms.
3. Numbers that disagree (`_pass_reject_disagreeing`): `DUPLICATE`, and rejects the parses whose numbers disagree.
4. Overlapping spans (`_pass_reject_overlaps`): rejects the multi-episode parses another file's parse overlaps.
5. Season re-read (`_sequence_passes.pass_sequence_as_matched_season`): re-reads a `1..N` sequence into one season,
   or finds it fits a season and the specials equally.
6. Misnumbered specials (`_sequence_passes.pass_misnumbered_specials`): `SPECIAL_ALIAS`, `ALIAS_OUTSIDE_WINDOW`,
   `MISNUMBERED`, `NO_EPISODE_FOUND`, `DUPLICATE`.
7. Size match (`_pass_size_match`): `SIZE_MATCH`, `SIZE_OVER_NAME`, `NO_EPISODE_FOUND`, `OUTSIDE_WINDOW`.
8. Sequence over parse (`_sequence_passes.pass_sequence_over_parse`): `SEQUENCE_OVER_PARSE`, `WAITING_ON_PARSE`,
   `OUTSIDE_WINDOW`, `DUPLICATE`.
9. Exact (`_pass_exact`): `EXACT`.
10. Episode title (`_pass_episode_title`): `EPISODE_TITLE`.
11. Absolute order (`_pass_pairing`, `_absolute_order`): `ABSOLUTE`.
12. Single file (`_pass_pairing`, `_place_single`): `SINGLE`, `TITLE_NAME`.
13. Name order (`_pass_pairing`, `_natural_order`): `NAME_ORDER`.
14. Numbered sequence (`_sequence_passes.pass_sequence`): `SEQUENCE`, `DUPLICATE`.
15. Classify (`_pass_classify`): `MISNUMBERED`, `DUPLICATE`, `OUTSIDE_WINDOW`, `TITLE_REFUTED`, `NO_EPISODE_FOUND`.
16. Other orders (`_orders.pass_other_orders`): `OTHER_ORDER`, `OTHER_ORDER_OUTSIDE_WINDOW`, `ORDER_ONLY`.

At most one of passes 11 to 13 is tried: pairing in absolute order when its conditions hold, else the single-file
pass when one id is unused, else pairing in name order. They consider only the unjudged files not wholly outside
the window. Every pass that pairs in order (passes 8, 11, 13, and 14) places nothing when a file carries the title
of an episode other than its pair, and that turns the pairing passes off.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ._orders import pass_other_orders
from ._sequence_passes import (
    MIN_TITLE_HITS,
    pass_misnumbered_specials,
    pass_sequence,
    pass_sequence_as_matched_season,
    pass_sequence_over_parse,
)
from ._state import NameFacts, NumberedSequence, PlacerState, TorrentFile
from ..file_names import FileIdentity, natural_key, sole_title_match
from ..parsed_episodes import glued_numbers, numbers_miss_the_series, parse_has_no_number
from ..placement_types import BatchPlacements, PlacementBatch, PlacementTarget, PlacementVerdict
from ..releases import EpisodeKey


@dataclass(frozen=True, slots=True)
class PlacerResult:
    """What one `place_batch` call gives: each file's verdict, and what the passes read off every name."""

    placements: BatchPlacements
    """One verdict per name in `PlacementBatch.new_names`, batch order."""
    facts: Mapping[str, NameFacts]
    """Every name of the torrent, the file map's and gone names included, as the names read before the first pass.
    Read-only (`PlacerState.facts`)."""


def place_batch(batch: PlacementBatch, target: PlacementTarget) -> PlacerResult:
    """Place the files of `batch` onto the ids of `target` by running the passes the module docstring lists, in order.

    One file name never overrides the window. With an empty episode index, no verdict that needs the index is
    given.
    """

    state = PlacerState.start(batch, target)
    _pass_extras(state)
    _pass_reject_contradicted(state)
    _pass_reject_disagreeing(state)
    _pass_reject_overlaps(state)
    pass_sequence_as_matched_season(state)
    pass_misnumbered_specials(state)
    _pass_size_match(state)
    pass_sequence_over_parse(state)
    _pass_exact(state)
    _pass_episode_title(state)
    _pass_pairing(state)
    pass_sequence(state)
    _pass_classify(state)
    pass_other_orders(state)
    return PlacerResult(BatchPlacements(state.placements()), state.facts)


def _pass_extras(state: PlacerState) -> None:
    """Set the extras aside as `PlacementVerdict.EXTRA`: never an episode, and left out of every count.

    An extra is a file whose name carries an extras word (an opening, an ending, a preview, a menu, and so on). A
    file whose spelled-out season and episode Sonarr matched is never an extra.
    """

    for file in state.unjudged_files():
        if file.extra:
            state.leave_unplaced(file, PlacementVerdict.EXTRA)


def _pass_reject_contradicted(state: PlacerState) -> None:
    """Reject each parse a file's title contradicts, its sequence's unconfirmed parses too, and unconfirmed numbers.

    A title contradicts a parse when the file's name carries the title of an episode other than the one the
    parse points at. In a numbered sequence, that rejects the in-window parses of the other members too, and the
    title of a member in the file map contradicts its sequence like any mate's. A member whose own title confirms its
    parse keeps it, and one rejected for a mate's title is noted (`TorrentFile.sequence_contradicted`), which pass 15
    reads as `TITLE_REFUTED`. A file Sonarr read onto an episode by the number its own title opens with, when no title
    confirms that number (`PlacerState.parsed_by_title_number`), is left for a hand import like one whose numbers
    disagree: `MISNUMBERED` unless a later pass places it or gives it a provisional verdict. Any rejection turns the
    pairing passes off.
    """

    files = state.non_extras
    titled = {file for file in files if (ids := file.parsed_episodes.ids) and file.contradicts(ids)}
    unconfirmed = {file for file in files if file.parsed_episodes.ids and state.parsed_by_title_number(file)}
    contradicted = set(titled)
    for sequence in state.non_extra_sequences:
        mates = set(sequence.all_versions)
        if titled & mates:
            contradicted |= {
                file for file in mates if file.parsed_episodes.in_window and not file.confirms(file.parsed_episodes.ids)
            }
    state.reject_each(contradicted | unconfirmed)
    for file in contradicted - titled:
        file.sequence_contradicted = True
    for file in unconfirmed:
        file.numbers_disagree = True
    state.pairing_off |= bool(contradicted or unconfirmed)


def _pass_reject_disagreeing(state: PlacerState) -> None:
    """Reject each parse that another number in its own name, or its sequence's shape, puts on a different episode.

    A name numbered two ways is right about at most one of them (`_numbers_disagree`), and a sequence spelling out
    season and episode that numbers past its season's end, in a season whose absolute numbering interleaves a special,
    disagrees from that special's position on (`_positions_disagree`). The window that admits one of the episodes the
    numbers disagree over rejects the parse, and the file is `MISNUMBERED` unless a later pass places it or gives it a
    provisional verdict. The sequence-over-parse pass (pass 8) never places a sequence with a member whose position
    makes its numbers disagree. Under any other window the file reads as it did. A lower version of a number that
    disagrees is `DUPLICATE`, a copy of its top version. The files a parse whose numbers disagree would have beaten
    to an episode are next in line for it (`_next_in_line`): the rejection never hands them the episode. The pairing
    passes are turned off under every window.
    """

    sequences = state.non_extra_sequences
    candidates = {file: ids for file in state.non_extras if (ids := _numbers_disagree(state, file)) is not None}
    by_position: set[TorrentFile] = set()
    for sequence in sequences:
        for file, ids in _positions_disagree(state, sequence).items():
            candidates[file] = candidates.get(file, frozenset()) | ids
            by_position.add(file)
    disagreeing = {file for file, ids in candidates.items() if any(state.target.admits(ep_id) for ep_id in ids)}
    copies = [file for sequence in sequences for file in sequence.superseded if file in disagreeing]
    # Next in line first: a rejected parse wins nothing, and a copy left unplaced is out of the running.
    for file, lost in _next_in_line(state, disagreeing).items():
        file.next_in_line_for = lost
    state.reject_each(disagreeing)
    state.set_aside_each([file for file in copies if file.unjudged], PlacementVerdict.DUPLICATE)
    for file in disagreeing.difference(copies):
        file.numbers_disagree = True
    for file in by_position & disagreeing:
        file.numbers_disagree_by_position = True
    for file in candidates.keys() - disagreeing:
        file.numbers_disagree_outside_window = True
    state.pairing_off |= bool(candidates)


def _numbers_disagree(state: PlacerState, file: TorrentFile) -> frozenset[int] | None:
    """The episodes the name's own numbers spread it over: its season and episode's and the other number's.

    None when they agree. Only a number that is another episode's absolute number disagrees. Beside a spelled-out
    season and episode, or parsed episodes taken from Sonarr's match, that is a bracketed number ("S10E01 (0337)").
    Beside a season and episode Sonarr guessed, it is any standalone number when Sonarr's series match doesn't back
    them, else the one they were glued from (`glued_numbers`).
    """

    info = file.parse
    if info is None:
        return None
    parsed = file.parsed_episodes
    name_parts = file.name_parts
    by_absolute = state.index.id_by_absolute
    resolved = frozenset(parsed.ids)
    if parsed.from_matched or not info.episode_numbers or info.season_number is None:
        if not resolved:
            return None
        counts: Iterable[int] = name_parts.bracketed_numbers
    elif name_parts.spells_season_episode:
        counts = name_parts.bracketed_numbers
    elif resolved and not parsed.sonarr_agrees:
        counts = name_parts.standalone_numbers
    else:
        counts = glued_numbers(info).intersection(name_parts.standalone_numbers)
    others = {ep_id for count in counts if (ep_id := by_absolute.get(count)) is not None} - resolved
    return resolved | others if others else None


def _next_in_line(state: PlacerState, disagreeing: set[TorrentFile]) -> dict[TorrentFile, frozenset[int]]:
    """Each unplaced file's episodes a parse whose numbers disagree takes first in the exact pass on the current parses.

    Those files are next in line, and the rejection must not hand them the episodes.
    """

    winner = {ep_id: file for file, ids in state.exact_pass_placements().items() for ep_id in ids}
    runner_ups: dict[TorrentFile, frozenset[int]] = {}
    for file in state.unjudged_files():
        lost = frozenset(
            ep_id
            for ep_id in file.parsed_episodes.exact_ids
            if (won := winner.get(ep_id)) in disagreeing and won is not file
        )
        if lost:
            runner_ups[file] = lost
    return runner_ups


def _positions_disagree(state: PlacerState, sequence: NumberedSequence) -> dict[TorrentFile, frozenset[int]]:
    """The members of a season and episode sequence numbered past its season's end, from the first special in it on.

    Such a sequence (`1..N+k` for an N-episode season with k interleaved specials) counted the specials or appended
    extras. From the first special's position on, the two numberings put a member on different episodes. A sequence
    numbered within its season is left alone. Each member whose numbers disagree maps to the episodes the two
    numberings spread those members over: the season's episodes from the special's position on, and the interleaved
    specials the sequence numbers.
    """

    seasons = {info.season_number for file in sequence.members if (info := file.parse) is not None}
    if len(seasons) != 1 or (season := seasons.pop()) in (None, 0):
        return {}
    if any(
        (info := file.parse) is None or info.episode_numbers[:1] != (file.name_number,) for file in sequence.members
    ):
        return {}
    if not all(file.name_parts.spells_season_episode for file in sequence.members):
        return {}
    count = state.index.season_counts.get(season, 0)
    # A season the index lacks is no season to count against (the season and episode miss, so no parsed number).
    if not count or max(sequence.numbers) <= count:
        return {}
    absolute_of, key_by_id = state.index.absolute_of, state.index.key_by_id
    regular = sorted(
        absolute_of[ep_id] for ep_id, key in key_by_id.items() if key.season == season and ep_id in absolute_of
    )
    if len(regular) != count:
        return {}
    first, last = regular[0], regular[-1]
    interleaved = {
        ep_id: number
        for ep_id, number in absolute_of.items()
        if key_by_id[ep_id].season == 0 and first <= number <= last
    }
    if not interleaved:
        return {}
    position = min(interleaved.values()) - first + 1
    spread = frozenset(
        ep_id for ep_id, key in key_by_id.items() if key.season == season and key.episode >= position
    ) | frozenset(ep_id for ep_id, number in interleaved.items() if number - first + 1 in sequence.numbers)
    return {file: spread for file in sequence.members if file.name_number >= position}


def _pass_reject_overlaps(state: PlacerState) -> None:
    """Reject each unplaced multi-episode parse that shares an episode with a different parse of another file.

    The other file may be one in the file map.
    """

    spans_by_ep: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    for file in state.non_extras:
        if not (parsed := file.parsed_episodes).rejected:
            for ep_id in parsed.ids:
                spans_by_ep[ep_id].add(parsed.ids)
    overlapping = [
        file
        for file in state.unjudged_files()
        if len(ids := file.parsed_episodes.ids) > 1 and any(len(spans_by_ep[ep_id]) > 1 for ep_id in ids)
    ]
    state.reject_each(overlapping)


def _pass_size_match(state: PlacerState) -> None:
    """Place files on the episode their size matched (`SeadexEvidence.size_matches`), even over their names.

    A file with the same size as a single-file release SeaDex lists for one episode goes on that episode
    (`SIZE_MATCH`). It is left for later passes if its parse already agrees, its title names another episode, or
    the matched episode belongs to another series. If the file's numbers point at a different episode
    (`_numbers_another`), it's `SIZE_OVER_NAME` and the caller warns. That means the batch's numbering is off, so the
    pairing passes are turned off and the sequence-over-parse pass (pass 8) passes over the file's numbered sequence.
    If the episode is already used the file is `NO_EPISODE_FOUND`, and if it's outside the window it's
    `OUTSIDE_WINDOW`. This runs after the misnumbered specials pass (pass 6), so it never pulls a file out of a batch
    that pass left `MISNUMBERED` or placed through its aliases.
    """

    for file in state.unjudged_files():
        ep_id = file.size_match
        if ep_id is None or ep_id not in state.target.index.by_id:
            continue
        parsed = file.parsed_episodes
        if (parsed.usable and parsed.ids == (ep_id,)) or file.contradicts((ep_id,)):
            continue
        if not state.target.admits(ep_id):
            state.leave_unplaced(file, PlacementVerdict.OUTSIDE_WINDOW)
            continue
        overrides = _numbers_another(state, file, ep_id)
        if overrides:
            # No pairing pass may place the rest, and the sequence-over-parse pass passes over this file's sequence.
            state.pairing_off = True
            if (member := file.name_parts.member) is not None:
                state.size_contradicted_prefixes.add(member.prefix)
        state.place_if_unused(
            file, ep_id, PlacementVerdict.SIZE_OVER_NAME if overrides else PlacementVerdict.SIZE_MATCH
        )


def _numbers_another(state: PlacerState, file: TorrentFile, ep_id: int) -> bool:
    """True if the name has a number and none of its numbers point at `ep_id`.

    Numbers here means its parsed episodes, its parsed seasons and episodes (ignored if none of them exist in the
    series), and its absolute numbers. A name with no season and episode also counts its name number, as an episode
    or an absolute.
    """

    resolved = file.parsed_episodes.ids
    if ep_id in resolved:
        return False
    info = file.parse
    member = file.name_parts.member
    keys: set[EpisodeKey] = set()
    absolutes: set[int] = set()
    if info is not None:
        absolutes.update(info.absolute_episode_numbers)
        if info.season_number is not None and not numbers_miss_the_series(info, state.target.id_by_key):
            keys.update(EpisodeKey(info.season_number, number) for number in info.episode_numbers)
    if not (resolved or keys or absolutes or member):
        return False
    key = state.index.key_by_id.get(ep_id)
    absolute = state.index.absolute_of.get(ep_id)
    if key in keys or absolute in absolutes:
        return False
    # In a name with a season and episode, the name number is no evidence beyond them.
    if keys:
        return True
    return member is None or member.number not in {absolute, None if key is None else key.episode}


def _pass_exact(state: PlacerState) -> None:
    """Place each unplaced file whose usable parsed episodes lie wholly inside the window on unused ids.

    With no window, a name's parsed season and episode place against the whole series. When two files resolve to one
    episode, the order is `ParsedEpisodes.rank`, then the higher `vN`, then batch order. So a "17.5 (S00E01)" beats a
    "- 17" whose match Sonarr shifted onto the same special, a "- 08" beats a "- Bonus" whose CRC tag parsed as its
    season and episode, and a "- 09v2" beats its "- 09". The loser stays unplaced.
    """

    for file, ids in state.exact_pass_placements().items():
        state.place(file, ids, PlacementVerdict.EXACT)


def _pass_episode_title(state: PlacerState) -> None:
    """Place each unplaced file on the unused window episode whose title its name carries, highest `vN` first.

    The file's parse must cover one episode, and the title places it whatever its number said (`_title_placement`
    lists the exceptions). It runs even when the pairing passes are off, and each placement turns them off.
    """

    for file in sorted(state.unjudged_files(), key=lambda f: f.name_parts.version, reverse=True):
        if (ep_id := _title_placement(state, file)) is not None:
            state.place(file, (ep_id,), PlacementVerdict.EPISODE_TITLE)
            state.pairing_off = True


def _title_placement(state: PlacerState, file: TorrentFile) -> int | None:
    """The unused window episode whose title the name carries, else None.

    Also None for a file next in line for that episode, for an unread or multi-episode parse, for a file that fits a
    season and the specials equally and carries the title of neither, and for a sequence's lone title whose episode a
    mate reads by number (`_outnumbered`).
    """

    ep_id = file.episode_by_title
    if ep_id is None or ep_id in state.used or not state.target.admits(ep_id) or state.single_parse(file) is None:
        return None
    if file.next_in_line((ep_id,)) or _outnumbered(state, file, ep_id):
        return None
    either_ids = file.parsed_episodes.either_ids
    return None if either_ids is not None and ep_id not in either_ids else ep_id


def _outnumbered(state: PlacerState, file: TorrentFile, ep_id: int) -> bool:
    """Whether another file of the name's sequence reads `ep_id`, the sequence's numbers standing.

    The numbers stand when fewer than `MIN_TITLE_HITS` files of the sequence carry titles against their parses.
    """

    for sequence in state.non_extra_sequences:
        mates = sequence.all_versions
        if file in mates:
            against = sum(mate.contradicts(mate.parsed_episodes.ids) for mate in mates)
            reads_it = any(mate is not file and ep_id in mate.parsed_episodes.ids for mate in mates)
            return against < MIN_TITLE_HITS and reads_it
    return False


def _pass_pairing(state: PlacerState) -> None:
    """Pair in absolute order, else try the single-file pass when one id is unused, else pair in name order."""

    # A file whose usable parsed episodes lie wholly outside the window belongs to another window.
    # It takes no unused id and does not spoil the count for the other files.
    candidate_files = [file for file in state.unjudged_files() if not file.parsed_episodes.wholly_outside]
    unused_ids = state.unused_ids()
    if not candidate_files or not unused_ids:
        return
    if (order := _absolute_order(state, candidate_files, unused_ids)) is not None:
        state.pair_in_order(order, unused_ids, PlacementVerdict.ABSOLUTE)
    elif len(unused_ids) == 1:
        _place_single(state, candidate_files, unused_ids[0])
    elif (order := _natural_order(state, candidate_files, unused_ids)) is not None:
        state.pair_in_order(order, unused_ids, PlacementVerdict.NAME_ORDER)


def _absolute_order(
    state: PlacerState, candidate_files: Sequence[TorrentFile], unused_ids: Sequence[int]
) -> list[TorrentFile] | None:
    """The candidate files in absolute-number order when pairing in absolute order applies, else None.

    It pairs the candidate files with the unused ids in absolute-number order, only when every candidate file carries
    exactly one absolute number, the counts match, no parse is unread, and no two files of the torrent share an
    absolute number. An absolute Sonarr took from the file's own title's number
    (`PlacerState.parsed_by_title_number`) is no absolute.
    """

    abs_by_file: dict[TorrentFile, int] = {}
    for file in candidate_files:
        info = file.parse
        if info is not None and len(info.absolute_episode_numbers) == 1 and not state.parsed_by_title_number(file):
            abs_by_file[file] = info.absolute_episode_numbers[0]
    # The shared-absolute check reads every non-extra parse, the file map's included. Repeats inside one parse collapse.
    batch_absolutes = [
        number
        for file in state.non_extras
        if (parse := file.parse) is not None
        for number in dict.fromkeys(parse.absolute_episode_numbers)
    ]
    applies = (
        bool(abs_by_file)
        and not state.pairing_off
        and state.all_parses_known
        and len(abs_by_file) == len(candidate_files)  # every candidate file has one absolute
        and len(abs_by_file) == len(unused_ids)  # as many files as unused ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restarted numbering)
    )
    return sorted(abs_by_file, key=abs_by_file.__getitem__) if applies else None


def _place_single(state: PlacerState, candidate_files: Sequence[TorrentFile], ep_id: int) -> None:
    """Give the one unused id to the only candidate file if it has no parsed number, else to the one a name points at.

    No parsed number here means the file's parse has no usable episode number (`_has_no_parsed_number`). The only
    candidate file takes the id as `SINGLE`, and the file with no parsed number a name points at takes it as
    `TITLE_NAME`. The name is one of the title's AniList names, and that check runs only when no parse is unread. A
    file next in line for the id takes it neither way, and no other file takes it in its place.
    """

    no_parsed_number = [file for file in candidate_files if _has_no_parsed_number(state, file)]
    if len(candidate_files) == 1 and no_parsed_number:
        chosen, verdict = 0, PlacementVerdict.SINGLE
    elif state.all_parses_known:
        stems = [file.name_parts.stem.text for file in no_parsed_number]
        chosen, verdict = sole_title_match(stems, state.target.title_names), PlacementVerdict.TITLE_NAME
    else:
        return
    if chosen is not None and not no_parsed_number[chosen].next_in_line((ep_id,)):
        state.place(no_parsed_number[chosen], (ep_id,), verdict)


def _has_no_parsed_number(state: PlacerState, file: TorrentFile) -> bool:
    """Whether the file's single-episode parse has no usable number: none at all, or only episodes the series lacks.

    False for a file whose numbers disagree (they name episodes, they just disagree), for a file that fits a season
    and the specials equally, and for a file carrying an episode title (that episode is used or outside the window).
    """

    if file.numbers_disagree or file.numbers_disagree_outside_window:
        return False
    info = state.single_parse(file)
    if info is None:
        return False
    if not (parse_has_no_number(info) or numbers_miss_the_series(info, state.target.id_by_key)):
        return False
    return file.parsed_episodes.either_ids is None and file.episode_by_title is None


def _natural_order(
    state: PlacerState, candidate_files: Sequence[TorrentFile], unused_ids: Sequence[int]
) -> list[TorrentFile] | None:
    """The candidate files in natural name order when pairing in name order applies ("Special 1..N"), else None.

    It pairs the candidate files with the unused ids in natural name order when every file of the torrent but the
    extras is unplaced, has no parsed number, and is not read outside the window, no id was used from the start (by
    the file map or an earlier window), and the counts match.
    """

    # Nothing judged: the batch's parses hold the candidate files and the extras and nothing else, and nothing but an
    # extra has a verdict, so nothing was placed, left unplaced, or in the file map.
    unused_set = set(candidate_files)
    nothing_judged = (
        not state.pairing_off
        and len(candidate_files) == len(unused_ids)
        and all(file.in_parsed == (file in unused_set or file.extra) for file in state.files)
        and state.nothing_judged
        and all(
            (info := state.single_parse(file)) is not None and parse_has_no_number(info) for file in candidate_files
        )
    )
    return sorted(candidate_files, key=lambda file: natural_key(file.name)) if nothing_judged else None


_BY_SIZE = frozenset({PlacementVerdict.SIZE_MATCH, PlacementVerdict.SIZE_OVER_NAME})
"""The verdicts of a file placed by size."""


@dataclass(frozen=True, slots=True)
class _DuplicateProof:
    """The evidence that an unplaced file is a duplicate, gathered once before classification."""

    proven: frozenset[int]
    """Ids we have a reason to think are used: placed by this call (except by size), or pointed at by the parse or
    title of a file in the file map or a gone one."""
    used_by_file: Mapping[FileIdentity, frozenset[int]]
    """The episodes each file's versions use: placed by this call, or carried as a title by a version in the file map
    or a gone one."""


def _pass_classify(state: PlacerState) -> None:
    """Classify every unplaced file.

    Each becomes `MISNUMBERED` (its own numbers disagree, or its number stands unconfirmed:
    `TorrentFile.numbers_disagree`), `DUPLICATE` (it reads inside the window onto episodes other files provably use,
    or a version of the same file uses the episode whose title it carries), `OUTSIDE_WINDOW` (its parsed episodes lie
    wholly outside the window, or it carries the title of an episode outside it, confirmed by its parse, by that
    parse's rejection while other files use its episodes, or by no unused id being left), `TITLE_REFUTED` (a sequence
    member pass 2 rejected), or `NO_EPISODE_FOUND`. The grab records the exclusions on the pending import, and the
    import warns about an unplaced file it finds in the torrent's folder. It never guesses.
    """

    proof = _duplicate_proof(state)
    nothing_left = not state.unused_ids()
    for file in state.unjudged_files():
        parsed = file.parsed_episodes
        titled = file.episode_by_title
        if file.numbers_disagree:
            # Its own numbers disagree, or its number stands unconfirmed, and no pass placed it: a hand import.
            state.leave_unplaced(file, PlacementVerdict.MISNUMBERED)
        elif ids := parsed.exact_ids:
            # The exact pass passed it over: an episode is used. A span only partly used stays NO_EPISODE_FOUND.
            proven = all(ep_id in state.used and ep_id in proof.proven for ep_id in ids)
            state.leave_unplaced(file, PlacementVerdict.DUPLICATE if proven else PlacementVerdict.NO_EPISODE_FOUND)
        elif titled is not None and titled in state.used:
            # Carries the title of a used episode: a duplicate only when a version of the same file uses it.
            twin = titled in proof.used_by_file.get(file.identity, ())
            state.leave_unplaced(file, PlacementVerdict.DUPLICATE if twin else PlacementVerdict.NO_EPISODE_FOUND)
        elif state.has_episode_index and (parsed.wholly_outside or _titled_outside(state, file, nothing_left)):
            state.leave_unplaced(file, PlacementVerdict.OUTSIDE_WINDOW)
        elif file.sequence_contradicted:
            state.leave_unplaced(file, PlacementVerdict.TITLE_REFUTED)
        else:
            state.leave_unplaced(file, PlacementVerdict.NO_EPISODE_FOUND)


def _duplicate_proof(state: PlacerState) -> _DuplicateProof:
    """Gather the duplicate evidence.

    An id used from the start (by the file map or an earlier window) proves nothing alone.
    """

    by_size: set[int] = set()
    evidenced: set[int] = set()
    used: defaultdict[FileIdentity, set[int]] = defaultdict(set)
    for file in state.files:
        if (placed := file.placement) is not None:
            used[file.identity].update(placed.ids)
            # A file placed by size only proves its own episode. It doesn't make another file on that episode a copy.
            if placed.verdict in _BY_SIZE:
                by_size.update(placed.ids)
        elif not file.to_place:
            evidenced.update(file.parsed_episodes.ids)
            if (titled := file.episode_by_title) is not None:
                evidenced.add(titled)
                used[file.identity].add(titled)
    proven = (state.used - state.target.used - by_size) | evidenced
    return _DuplicateProof(frozenset(proven), {identity: frozenset(ids) for identity, ids in used.items()})


def _titled_outside(state: PlacerState, file: TorrentFile, nothing_left: bool) -> bool:
    """Whether the name carries the title of an episode outside the window, confirmed another way.

    Confirmed when the file has complete parsed episodes wholly outside the window, when its parse was rejected
    and every episode it pointed at is used by another file of this torrent, or when no unused id is left. An
    episode title alone never excludes a file the title may still need.
    """

    ep_id = file.episode_by_title
    if ep_id is None or state.target.admits(ep_id):
        return False
    parsed = file.parsed_episodes
    used = bool(parsed.ids) and state.used.issuperset(parsed.ids)
    return (parsed.complete and not parsed.in_window) or (parsed.rejected and used) or nothing_left
