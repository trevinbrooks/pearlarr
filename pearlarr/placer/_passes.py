"""`assign_episode_ids`, the glossary and pass list below, and every pass not in `_run_passes` or `_orders`.

Our episode ids for the entry are authoritative. Sonarr's parse of a file name, and the number the name
carries, only ever choose among them.

Terms used across the placement modules:

- Own key: the `SxxExx` a file name carries. Matched pair: an episode Sonarr's series match points at.
  A name without an own key borrows Sonarr's matched pairs.
- Range key: an own key spelled as a tight range ("S01E05-06"). When Sonarr read only its first episode, the
  parse is widened to the whole range before any pass runs (`widen_range_key` says when it isn't).
- Scope (`TargetScope`): the entry's episode ids a file may be placed on, plus the whole series' episode
  map every key is looked up in. An empty id list means no scope ("unscoped"): a file's own keys then
  place against the whole series.
- Window (`window_placement`): the scope of one entry a torrent is listed on or claimed by. A torrent is
  placed under every window in turn, season windows first, then specials windows, each in AniList id order.
- Listing (`TargetScope.listing`): what SeaDex's listings tell us about the torrent. `ids` are the episodes of
  every entry that lists it, which a numbered specials pack is checked against (empty when no entry lists it, so
  the check is skipped). `identified` maps files to the episode whose single-file release has the same size.
  `special_aliases` maps a TMDB specials number to the TVDB special it stands for.
- Open ids: the scope's ids that no file holds yet.
- Seeded file: a file of the torrent that an earlier grab already mapped.
  Its ids start out used, as do ids an earlier window placed. Seeded files, and files since moved off disk
  ("gone"), are never placed again, but their parses and names still count as evidence.
- Alias match: Sonarr moved a name's episode numbers, unchanged, into one regular season (a sequel as season 1).
- Reading (`Reading`): the episode ids a file's parse resolves to in our series map, with flags saying how
  far to trust them. A rejected reading (`Reading.vetoed`) is kept as evidence, but no pass places the file
  by it.
- Complete reading: every episode the parse names resolved in our series map (a partial span is never placed).
- Trusted reading: a complete reading that was not rejected (`Reading.trusted`).
- Runner-up: a file whose reading the exact pass's order (`Reading.rank`, the higher `vN`, batch order) places
  after a disputed file's reading for an episode. A refusal opens no episode to a runner-up: it never takes the
  episode by its reading or its title, and its claim on it refuses no run.
- Unknown parse: a missing parse. `PlacementBatch.all_parses_known` is False while any file of the torrent has one.
- Numberless: the file name carries no episode or absolute number (a key the series lacks may count as none).
- Release number: the episode number the file name carries ("Show - 07"), read without Sonarr. A trailing "S2"
  counts as one only when the names sharing its prefix count up from "S1" (`read_names`).
- Numbered run (`NumberedRun`): the files whose names share the text before their release number, in number
  order. Of two versions of one number, the higher `vN` is the member and the lower is superseded.
- Tail: the words of a run member's name after its release number.
- Run window (`RunWindow`): the open ids, in number order, when they are consecutive episodes of one season or
  carry consecutive absolute numbers. Specials numbered apart are one too when they are every episode of the
  title and each aired after the one before it.
- Covering run: a run numbered `1..N` for a whole season when the run window holds only part of that season.
- Tied file: a run file the season re-read (pass 5) fits to a season and the specials equally well (`Reading.tied`).
- Open file: a file with no verdict yet whose trusted reading does not lie wholly outside the scope.
- Specific title: an episode title with enough words beyond the series title to place a file by.
- Titled as: the file name carries the specific title of exactly one episode, and no other file of the torrent
  carries that title (versions of one file count once).
- Run evidence: every episode title (a colon part only when specific) that starts a run member's tail.
  It can pick or veto a run.
- Zip: pair files with ids one to one, in order. A zip places nothing when a file is titled as an
  episode other than its pair.
- Count-based passes: the absolute zip, the ordered zip, and the numbered-run pass. They place files by
  position, not by what each file says, so any sign that the batch's count is off turns them off
  (`Placer.count_legs_barred`).
- Slice: the part of a series one entry covers when a season is split across several entries. A file of
  another slice is `FOREIGN`.
- Stored record: the pending import record already saved for a torrent, one per infohash.
- Claim: one entry's part of a stored record (`EntryClaim`). In `parse_reading`: a `(season, episode)` a parse names.
  In `window_placement`: one window's verdict on a file it neither placed nor refused, which a later window may still
  decide. A file no window decides ends with its strongest claim (`PlacementVerdict.claim_rank`).
- Lone-number file: a file Sonarr parsed to absolute numbers alone ("Show - 07"), no own key, no season above 0,
  neither a special nor a season pack. Not an extra, and not counted by its own title.
- Numberings reader (`numberings`): reads a torrent's lone-number files under every way the release group could
  have counted the series (from its first episode, from the window's, by Sonarr's episode numbers, straight through
  the torrent's windows, by TVDB's absolutes) and proposes each file's row when every fitting way agrees. It runs
  once per torrent, after the windows merge, and the fill (`WindowedAssignment.filled`) places a proposal only on a
  file no window placed, refused or read inside itself (`NUMBERINGS_AGREE`), so it never outranks a pass. No count
  sign turns it off: its own gates refuse instead.
- Other order (`EpisodeOrder`): another way TVDB numbers the series (DVD, alternate, and so on), read onto Sonarr's
  episodes. The index holds them only for a series Sonarr numbers as aired.
- Order-only key: a `(season, episode)` an other order gives to an episode Sonarr doesn't list, such as one half of
  an episode that order splits in two.
- Left open: a file no earlier window placed that reaches pass 16 `SKIPPED`, `MISNUMBERED`, `TITLE_REFUTED` or
  `FOREIGN` (`PlacementVerdict.left_open`). Nothing placed it, and nothing proved it a duplicate or an extra.

The passes run in this order. Each places files only onto the ids that earlier passes left open:

1. Extras (`_pass_extras`): `EXTRA`.
2. Title contradictions (`_pass_refuse_refuted`): rejects the readings a title contradicts or no title confirms.
3. Number disputes (`_pass_refuse_disputed`): `DUPLICATE`, and rejects the disputed readings.
4. Overlapping spans (`_pass_refuse_overlaps`): rejects the multi-episode readings another file's reading overlaps.
5. Season re-read (`_run_passes.pass_reread_season_runs`): re-reads a `1..N` run into one season, or ties it.
6. Misnumbered pack (`_run_passes.pass_misnumbered_pack`): `ALTERNATE`, `ALIASED_ELSEWHERE`, `MISNUMBERED`,
   `SKIPPED`, `DUPLICATE`.
7. Size identity (`_pass_identity`): `IDENTIFIED`, `OVERRIDDEN`, `SKIPPED`, `FOREIGN`.
8. Release run (`_run_passes.pass_release_run`): `RELEASE_RUN`, `HELD`, `FOREIGN`, `DUPLICATE`.
9. Exact (`_pass_exact`): `EXACT`.
10. Episode title (`_pass_episode_title`): `EPISODE_TITLE`.
11. Absolute zip (`_pass_counted`, `_absolute_order`): `ABSOLUTE`.
12. Single file (`_pass_counted`, `_place_single`): `SINGLE`, `TITLED`.
13. Ordered zip (`_pass_counted`, `_natural_order`): `ORDERED`.
14. Numbered run (`_run_passes.pass_numbered_run`): `NUMBERED_RUN`, `DUPLICATE`.
15. Classify (`_pass_classify`): `MISNUMBERED`, `DUPLICATE`, `FOREIGN`, `TITLE_REFUTED`, `SKIPPED`.
16. Other orders (`_orders.pass_other_orders`): `OTHER_ORDER`, `OTHER_ORDER_ELSEWHERE`, `ORDER_ONLY`.

At most one of passes 11 to 13 is tried: the absolute zip when its conditions hold, else the single-file
pass when one id is open, else the ordered zip. They consider only open files. Every zip (passes 8, 11,
13, and 14) places nothing when a file is titled as an episode other than its pair, and that turns the
count-based passes off.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ._orders import pass_other_orders
from ._run_passes import (
    MIN_TITLE_HITS,
    pass_misnumbered_specials,
    pass_numbered_run,
    pass_release_run,
    pass_reread_season_runs,
)
from ._state import NameFacts, NumberedSequence, PlacerState, TorrentFile
from ..parse_reading import glued_counts, numbers_miss_the_series, parse_has_no_number
from ..placement_types import BatchPlacements, PlacementBatch, PlacementTarget, PlacementVerdict
from ..release_names import FileIdentity, natural_key, sole_title_match
from ..releases import EpisodeKey


@dataclass(frozen=True, slots=True)
class PlacerResult:
    """What one `assign_episode_ids` call gives: each file's verdict, and what the passes read off every name."""

    assignment: BatchPlacements
    """One verdict per name in `PlacementBatch.names`, batch order."""
    facts: Mapping[str, NameFacts]
    """Every name of the torrent, seeded and gone names included, as the names read before the first pass. Read-only
    (`Placer.facts`)."""


def place_batch(batch: PlacementBatch, scope: PlacementTarget) -> PlacerResult:
    """Place the files of `batch` onto the ids of `scope` by running the passes the module docstring lists, in order.

    One file name never overrides the scope. With an empty series map, every verdict that needs the map is
    refused.
    """

    state = PlacerState.start(batch, scope)
    _pass_extras(state)
    _pass_reject_contradicted(state)
    _pass_reject_disagreeing(state)
    _pass_reject_overlaps(state)
    pass_reread_season_runs(state)
    pass_misnumbered_specials(state)
    _pass_size_match(state)
    pass_release_run(state)
    _pass_exact(state)
    _pass_episode_title(state)
    _pass_pairing(state)
    pass_numbered_run(state)
    _pass_classify(state)
    pass_other_orders(state)
    return PlacerResult(BatchPlacements(state.placements()), state.facts)


def _pass_extras(state: PlacerState) -> None:
    """Set the extras aside as `PlacementVerdict.EXTRA`: never an episode, and left out of every count.

    An extra is a file whose name carries an extras word (an opening, an ending, a preview, a menu, and so on). A
    file whose own key Sonarr matched is never an extra.
    """

    for file in state.unjudged_files():
        if file.extra:
            state.leave_unplaced(file, PlacementVerdict.EXTRA)


def _pass_reject_contradicted(state: PlacerState) -> None:
    """Reject each reading a file's title contradicts, its run's unconfirmed readings too, and hold unconfirmed counts.

    A title contradicts a reading when the file's name carries the title of an episode other than the one the
    reading points at. In a numbered run, that rejects the in-scope readings of the other members too, and a seeded
    member's title refutes its run like any mate's. A member whose own title confirms its reading keeps it, and one
    rejected for a mate's title is noted (`TorrentFile.run_refuted`), which pass 15 reads as `TITLE_REFUTED`. A file
    Sonarr read onto an episode by the number its own title opens with, when no title confirms the count
    (`Placer.counted_by_title`), is held for a hand import like a disputed one: `MISNUMBERED` unless a later pass
    places or claims it. Any rejection turns the count-based passes off.
    """

    files = state.non_extras
    titled = {file for file in files if (ids := file.reading.ids) and file.contradicts(ids)}
    unconfirmed = {file for file in files if file.reading.ids and state.counted_by_title(file)}
    refuted = set(titled)
    for run in state.counted_runs:
        mates = set(run.all_versions)
        if titled & mates:
            refuted |= {file for file in mates if file.reading.in_window and not file.confirms(file.reading.ids)}
    state.veto_each(refuted | unconfirmed)
    for file in refuted - titled:
        file.sequence_contradicted = True
    for file in unconfirmed:
        file.numbers_disagree = True
    state.pairing_off |= bool(refuted or unconfirmed)


def _pass_reject_disagreeing(state: PlacerState) -> None:
    """Reject each reading that another count of its own name, or its run's shape, puts on a different episode.

    A name that counts two ways is right about at most one of them (`_disputes`), and a keyed run that numbers past
    its season's end, in a season whose absolute numbering interleaves a special, is disputed from that special's
    position on (`_positions_disputed`). The window that admits one of the episodes a dispute is over rejects the
    reading, and the file is `MISNUMBERED` unless a later pass places or claims it. The release run (pass 8) never
    places a run with a member its position disputes. Under any other window the file reads as it did. A lower
    version of a disputed number is `DUPLICATE`, a copy of its top version. The files a disputed reading would have
    beaten to an episode are its runner-ups (`_runner_ups`): the refusal never hands them the episode. The
    count-based passes are turned off under every window.
    """

    runs = state.counted_runs
    candidates = {file: ids for file in state.non_extras if (ids := _disputes(state, file)) is not None}
    by_position: set[TorrentFile] = set()
    for run in runs:
        for file, ids in _positions_disputed(state, run).items():
            candidates[file] = candidates.get(file, frozenset()) | ids
            by_position.add(file)
    disputed = {file for file, ids in candidates.items() if any(state.scope.admits(ep_id) for ep_id in ids)}
    copies = [file for run in runs for file in run.superseded if file in disputed]
    # Runner-ups first: a rejected reading wins nothing, and a set-aside copy is out of the running.
    for file, lost in _runner_ups(state, disputed).items():
        file.next_in_line_for = lost
    state.veto_each(disputed)
    state.set_aside_each([file for file in copies if file.unjudged], PlacementVerdict.DUPLICATE)
    for file in disputed.difference(copies):
        file.numbers_disagree = True
    for file in by_position & disputed:
        file.numbers_disagree_by_position = True
    for file in candidates.keys() - disputed:
        file.numbers_disagree_elsewhere = True
    state.pairing_off |= bool(candidates)


def _disputes(state: PlacerState, file: TorrentFile) -> frozenset[int] | None:
    """The episodes the name's own counts spread it over: its key's and the other count's. None when they agree.

    Only a count that is another episode's absolute number disputes. Beside a spelled-out key or a reading borrowed
    from Sonarr's match, that is a bracketed count ("S10E01 (0337)"). Beside a key Sonarr guessed, it is any count
    when Sonarr's series match doesn't back the key, else the one the key was glued from (`glued_counts`).
    """

    info = file.parse
    if info is None:
        return None
    reading = file.reading
    name_read = file.name_read
    by_absolute = state.index.id_by_absolute
    resolved = frozenset(reading.ids)
    if reading.from_matched or not info.episode_numbers or info.season_number is None:
        if not resolved:
            return None
        counts: Iterable[int] = name_read.bracketed_numbers
    elif name_read.spells_season_episode:
        counts = name_read.bracketed_numbers
    elif resolved and not reading.sonarr_agrees:
        counts = name_read.standalone_numbers
    else:
        counts = glued_counts(info).intersection(name_read.standalone_numbers)
    others = {ep_id for count in counts if (ep_id := by_absolute.get(count)) is not None} - resolved
    return resolved | others if others else None


def _runner_ups(state: PlacerState, disputed: set[TorrentFile]) -> dict[TorrentFile, frozenset[int]]:
    """Each unplaced file's episodes a disputed reading takes first when the exact pass runs on the current readings.

    Those files are the runner-ups the refusal must not hand the episodes to.
    """

    winner = {ep_id: file for file, ids in state.exact_picks().items() for ep_id in ids}
    runner_ups: dict[TorrentFile, frozenset[int]] = {}
    for file in state.unjudged_files():
        lost = frozenset(
            ep_id for ep_id in file.reading.exact_ids if (won := winner.get(ep_id)) in disputed and won is not file
        )
        if lost:
            runner_ups[file] = lost
    return runner_ups


def _positions_disputed(state: PlacerState, run: NumberedSequence) -> dict[TorrentFile, frozenset[int]]:
    """The members of a keyed run numbered past its season's end, from the first special the season interleaves on.

    Such a run (`1..N+k` for an N-episode season with k interleaved specials) counted the specials or appended
    extras. From the first special's position on, the two counts put a member on different episodes. A run keyed
    within its season is left alone. Each disputed member maps to the episodes the two counts spread the disputed
    members over: the season's episodes from the special's position on, and the interleaved specials the run
    numbers.
    """

    seasons = {info.season_number for file in run.members if (info := file.parse) is not None}
    if len(seasons) != 1 or (season := seasons.pop()) in (None, 0):
        return {}
    if any((info := file.parse) is None or info.episode_numbers[:1] != (file.name_number,) for file in run.members):
        return {}
    if not all(file.name_read.spells_season_episode for file in run.members):
        return {}
    count = state.index.season_counts.get(season, 0)
    # A season the map lacks is no season to count against (the keys miss, and the file reads numberless).
    if not count or max(run.numbers) <= count:
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
    ) | frozenset(ep_id for ep_id, number in interleaved.items() if number - first + 1 in run.numbers)
    return {file: spread for file in run.members if file.name_number >= position}


def _pass_reject_overlaps(state: PlacerState) -> None:
    """Reject each unplaced multi-episode reading that shares an episode with a different reading of another file.

    The other file may be a seeded one.
    """

    spans_by_ep: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    for file in state.non_extras:
        if not (reading := file.reading).rejected:
            for ep_id in reading.ids:
                spans_by_ep[ep_id].add(reading.ids)
    overlapping = [
        file
        for file in state.unjudged_files()
        if len(ids := file.reading.ids) > 1 and any(len(spans_by_ep[ep_id]) > 1 for ep_id in ids)
    ]
    state.veto_each(overlapping)


def _pass_size_match(state: PlacerState) -> None:
    """Place files on the episode their size matched (`ListingEvidence.identified`), even over their names.

    A file with the same size as a single-file release SeaDex lists for one episode goes on that episode
    (`IDENTIFIED`). It is left for later passes if its reading already agrees, its title names another episode, or
    the matched episode belongs to another series. If the file's numbers point at a different episode
    (`_numbers_another`), it's `OVERRIDDEN` and the caller warns. That means the batch's numbering is off, so the
    count-based passes are turned off and the release run (pass 8) skips the file's numbered run. If the episode is
    already taken the file is `SKIPPED`, and if it's outside the scope it's `FOREIGN`. This runs after the
    misnumbered pack (pass 6), so it never pulls a file out of a pack that pass refused or placed through its
    aliases.
    """

    for file in state.unjudged_files():
        ep_id = file.size_match
        if ep_id is None or ep_id not in state.scope.index.by_id:
            continue
        reading = file.reading
        if (reading.trusted and reading.ids == (ep_id,)) or file.contradicts((ep_id,)):
            continue
        if not state.scope.admits(ep_id):
            state.leave_unplaced(file, PlacementVerdict.OUTSIDE_WINDOW)
            continue
        overrides = _numbers_another(state, file, ep_id)
        if overrides:
            # No count-based pass may place the rest, and the release run skips this file's run.
            state.pairing_off = True
            if (member := file.name_read.member) is not None:
                state.size_contradicted_prefixes.add(member.prefix)
        state.place_or_skip(file, ep_id, PlacementVerdict.SIZE_OVER_NAME if overrides else PlacementVerdict.SIZE_MATCH)


def _numbers_another(state: PlacerState, file: TorrentFile, ep_id: int) -> bool:
    """True if the name has a number and none of its numbers point at `ep_id`.

    Numbers here means its reading, its own keys (ignored if none of them exist in the series), and its
    absolute numbers. A name with no keys also counts its release number, as an episode or an absolute.
    """

    resolved = file.reading.ids
    if ep_id in resolved:
        return False
    info = file.parse
    member = file.name_read.member
    keys: set[EpisodeKey] = set()
    absolutes: set[int] = set()
    if info is not None:
        absolutes.update(info.absolute_episode_numbers)
        if info.season_number is not None and not numbers_miss_the_series(info, state.scope.id_by_key):
            keys.update(EpisodeKey(info.season_number, number) for number in info.episode_numbers)
    if not (resolved or keys or absolutes or member):
        return False
    key = state.index.key_by_id.get(ep_id)
    absolute = state.index.absolute_of.get(ep_id)
    if key in keys or absolute in absolutes:
        return False
    # In a keyed name the release number is no evidence beyond the key.
    if keys:
        return True
    return member is None or member.number not in {absolute, None if key is None else key.episode}


def _pass_exact(state: PlacerState) -> None:
    """Place each unplaced file whose trusted reading lies wholly inside the scope on unused ids.

    Unscoped, a name's own keys place against the whole series. When two files resolve to one episode,
    `Reading.rank` decides, then the higher `vN`, then batch order. So a "17.5 (S00E01)" beats a "- 17" whose match
    Sonarr shifted onto the same special, a "- 08" beats a "- Bonus" whose CRC tag parsed as its key, and a
    "- 09v2" beats its "- 09". The loser stays unplaced.
    """

    for file, ids in state.exact_picks().items():
        state.place(file, ids, PlacementVerdict.EXACT)


def _pass_episode_title(state: PlacerState) -> None:
    """Place each unplaced file on the open scope episode its name is titled as, highest `vN` first.

    The file's parse must cover one episode, and the title places it whatever its number said (`_title_placement`
    lists the exceptions). It runs even when the count-based passes are off, and each placement turns them off.
    """

    for file in sorted(state.unjudged_files(), key=lambda f: f.name_read.version, reverse=True):
        if (ep_id := _title_placement(state, file)) is not None:
            state.place(file, (ep_id,), PlacementVerdict.EPISODE_TITLE)
            state.pairing_off = True


def _title_placement(state: PlacerState, file: TorrentFile) -> int | None:
    """The open scope episode the name is titled as, else None.

    Also None for a runner-up on that episode, for a missing or multi-episode parse, for a tied file titled as
    neither of its candidates, and for a run's lone title whose episode a mate reads by number (`_outnumbered`).
    """

    ep_id = file.episode_by_title
    if ep_id is None or ep_id in state.used or not state.scope.admits(ep_id) or state.single_parse(file) is None:
        return None
    if file.runner_up((ep_id,)) or _outnumbered(state, file, ep_id):
        return None
    tied = file.reading.either_ids
    return None if tied is not None and ep_id not in tied else ep_id


def _outnumbered(state: PlacerState, file: TorrentFile, ep_id: int) -> bool:
    """Whether another file of the name's run reads `ep_id`, the run's count standing.

    The count stands when fewer than `MIN_TITLE_HITS` files of the run are titled against their readings.
    """

    for run in state.counted_runs:
        mates = run.all_versions
        if file in mates:
            against = sum(mate.contradicts(mate.reading.ids) for mate in mates)
            reads_it = any(mate is not file and ep_id in mate.reading.ids for mate in mates)
            return against < MIN_TITLE_HITS and reads_it
    return False


def _pass_pairing(state: PlacerState) -> None:
    """Try the absolute zip, else the single-file pass when one id is open, else the ordered zip."""

    # A file whose trusted reading lies wholly outside the scope belongs to another slice.
    # It takes no open id and does not spoil the count for the other files.
    open_files = [file for file in state.unjudged_files() if not file.reading.wholly_outside]
    open_ids = state.open_ids()
    if not open_files or not open_ids:
        return
    if (order := _absolute_order(state, open_files, open_ids)) is not None:
        state.pair_in_order(order, open_ids, PlacementVerdict.ABSOLUTE)
    elif len(open_ids) == 1:
        _place_single(state, open_files, open_ids[0])
    elif (order := _natural_order(state, open_files, open_ids)) is not None:
        state.pair_in_order(order, open_ids, PlacementVerdict.NAME_ORDER)


def _absolute_order(
    state: PlacerState, open_files: Sequence[TorrentFile], open_ids: Sequence[int]
) -> list[TorrentFile] | None:
    """The open files in absolute-number order when the absolute zip applies, else None.

    The zip pairs the open files with the open ids in absolute-number order, only when every open file carries
    exactly one absolute number, the counts match, every parse is known, and no two files of the torrent share an
    absolute number. An absolute Sonarr counted from the file's own title's number (`Placer.counted_by_title`) is
    no absolute.
    """

    abs_by_file: dict[TorrentFile, int] = {}
    for file in open_files:
        info = file.parse
        if info is not None and len(info.absolute_episode_numbers) == 1 and not state.counted_by_title(file):
            abs_by_file[file] = info.absolute_episode_numbers[0]
    # The shared-absolute check reads every non-extra parse, seeded files included. Repeats inside one parse collapse.
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
        and len(abs_by_file) == len(open_files)  # every open file has one absolute
        and len(abs_by_file) == len(open_ids)  # as many files as open ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restarted numbering)
    )
    return sorted(abs_by_file, key=abs_by_file.__getitem__) if applies else None


def _place_single(state: PlacerState, open_files: Sequence[TorrentFile], ep_id: int) -> None:
    """Fill the one open id: with the only open file if it is numberless, else the numberless file a title names.

    Numberless here means the file's parse has no usable episode number (`_reads_no_number`). The only open file
    takes the id as `SINGLE`, and the file a title names takes it as `TITLED`. The title is an AniList title of the
    entry, and that match runs only when every parse is known. A runner-up for the id takes it neither way, and no
    other file takes it in its place.
    """

    unnumbered = [file for file in open_files if _reads_no_number(state, file)]
    if len(open_files) == 1 and unnumbered:
        pick, verdict = 0, PlacementVerdict.SINGLE
    elif state.all_parses_known:
        stems = [file.name_read.stem.text for file in unnumbered]
        pick, verdict = sole_title_match(stems, state.scope.title_names), PlacementVerdict.TITLE_NAME
    else:
        return
    if pick is not None and not unnumbered[pick].runner_up((ep_id,)):
        state.place(unnumbered[pick], (ep_id,), verdict)


def _reads_no_number(state: PlacerState, file: TorrentFile) -> bool:
    """Whether the file's single-episode parse has no usable number: none at all, or only keys the series lacks.

    False for a disputed file (its numbers name episodes, they just disagree), for a tied file, and for a
    titled file (its titled episode is taken or outside the scope).
    """

    if file.numbers_disagree or file.numbers_disagree_elsewhere:
        return False
    info = state.single_parse(file)
    if info is None:
        return False
    if not (parse_has_no_number(info) or numbers_miss_the_series(info, state.scope.id_by_key)):
        return False
    return file.reading.either_ids is None and file.episode_by_title is None


def _natural_order(
    state: PlacerState, open_files: Sequence[TorrentFile], open_ids: Sequence[int]
) -> list[TorrentFile] | None:
    """The open files in natural name order when the ordered zip applies (the "Special 1..N" shape), else None.

    The zip pairs the open files with the open ids in natural name order when every file of the torrent but the
    extras is unplaced, numberless, and not read outside the scope, no id was used from the start (by a seeded file
    or an earlier window), and the counts match.
    """

    # Pristine: the parse map holds the open files and the extras and nothing else, and nothing but an extra
    # has a verdict, so nothing was placed, held, or seeded.
    open_set = set(open_files)
    pristine = (
        not state.pairing_off
        and len(open_files) == len(open_ids)
        and all(file.in_parse_map == (file in open_set or file.extra) for file in state.files)
        and state.untouched
        and all((info := state.single_parse(file)) is not None and parse_has_no_number(info) for file in open_files)
    )
    return sorted(open_files, key=lambda file: natural_key(file.name)) if pristine else None


_BY_SIZE = frozenset({PlacementVerdict.SIZE_MATCH, PlacementVerdict.SIZE_OVER_NAME})
"""The verdicts of a file placed by size."""


@dataclass(frozen=True, slots=True)
class _DuplicateProof:
    """The evidence that an unplaced file is a duplicate, gathered once before classification."""

    proven: frozenset[int]
    """Ids we have a reason to think are held: placed by this call (except by size), or pointed at by a seeded or
    gone file's reading or title."""
    held_by_file: Mapping[FileIdentity, frozenset[int]]
    """The episodes each file's versions hold: placed by this call, or a seeded or gone version titled as one."""


def _pass_classify(state: PlacerState) -> None:
    """Classify every unplaced file.

    Each becomes `MISNUMBERED` (its own numbers disagree, or its count stands unconfirmed: `TorrentFile.disputed`),
    `DUPLICATE` (it reads inside the scope onto episodes other files provably hold, or a version of the same file
    holds its titled episode), `FOREIGN` (its reading lies wholly outside the scope, or it is titled as an episode
    outside it and confirmed by its reading, by that reading's rejection while other files hold its episodes, or by
    no open id being left), `TITLE_REFUTED` (a run member pass 2 rejected), or `SKIPPED`. The grab records the
    exclusions on the pending record, and the import warns about a skipped file it finds on disk. It never guesses.
    """

    proof = _duplicate_proof(state)
    nothing_left = not state.open_ids()
    for file in state.unjudged_files():
        reading = file.reading
        titled = file.episode_by_title
        if file.numbers_disagree:
            # Its own numbers disagree, or its count stands unconfirmed, and no pass placed it: a hand import.
            state.leave_unplaced(file, PlacementVerdict.MISNUMBERED)
        elif ids := reading.exact_ids:
            # The exact pass skipped it: an episode is taken. A span only partly taken stays SKIPPED.
            proven = all(ep_id in state.used and ep_id in proof.proven for ep_id in ids)
            state.leave_unplaced(file, PlacementVerdict.DUPLICATE if proven else PlacementVerdict.NO_EPISODE_FOUND)
        elif titled is not None and titled in state.used:
            # Titled as a taken episode: a duplicate only when a version of the same file holds it.
            twin = titled in proof.held_by_file.get(file.identity, ())
            state.leave_unplaced(file, PlacementVerdict.DUPLICATE if twin else PlacementVerdict.NO_EPISODE_FOUND)
        elif state.map_known and (reading.wholly_outside or _titled_outside(state, file, nothing_left)):
            state.leave_unplaced(file, PlacementVerdict.OUTSIDE_WINDOW)
        elif file.sequence_contradicted:
            state.leave_unplaced(file, PlacementVerdict.TITLE_REFUTED)
        else:
            state.leave_unplaced(file, PlacementVerdict.NO_EPISODE_FOUND)


def _duplicate_proof(state: PlacerState) -> _DuplicateProof:
    """Gather the duplicate evidence.

    An id held from the start (a seeded file's or an earlier window's) proves nothing alone.
    """

    by_size: set[int] = set()
    evidenced: set[int] = set()
    held: defaultdict[FileIdentity, set[int]] = defaultdict(set)
    for file in state.files:
        if (placed := file.placement) is not None:
            held[file.identity].update(placed.ids)
            # A file placed by size only proves its own episode. It doesn't make another file on that episode a copy.
            if placed.verdict in _BY_SIZE:
                by_size.update(placed.ids)
        elif not file.to_place:
            evidenced.update(file.reading.ids)
            if (titled := file.episode_by_title) is not None:
                evidenced.add(titled)
                held[file.identity].add(titled)
    proven = (state.used - state.scope.used - by_size) | evidenced
    return _DuplicateProof(frozenset(proven), {identity: frozenset(ids) for identity, ids in held.items()})


def _titled_outside(state: PlacerState, file: TorrentFile, nothing_left: bool) -> bool:
    """Whether the name is titled as an episode outside the scope, confirmed another way.

    Confirmed when the file has a complete reading wholly outside the scope, when its reading was rejected
    and every episode it pointed at is held by another file of this torrent, or when no open id is left. A
    title alone never excludes a file the entry may still need.
    """

    ep_id = file.episode_by_title
    if ep_id is None or state.scope.admits(ep_id):
        return False
    reading = file.reading
    held = bool(reading.ids) and state.used.issuperset(reading.ids)
    return (reading.complete and not reading.in_window) or (reading.rejected and held) or nothing_left
