"""`assign_episode_ids`, the glossary and pass list below, and every pass not in `_run_passes`.

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
- Run window (`RunWindow`): the open ids, when they are consecutive episodes, in airing order.
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

The passes run in this order. Each places files only onto the ids that earlier passes left open:

1. Extras: a file whose name carries an extras word (an opening, an ending, a preview, a menu, and so on)
   is set aside as `EXTRA` and left out of every count below. A file whose own key Sonarr matched is never
   an extra.
2. Title contradictions: when a file's name carries the title of an episode other than the one its reading
   points at, that reading is rejected. If the file belongs to a numbered run, the in-scope readings of the
   other members are rejected too, except where a member's own title confirms its episode, and a member no later
   pass places is `TITLE_REFUTED`. A file Sonarr counted by the number its own title opens with, when no title
   confirms the count, is `MISNUMBERED` unless its run places it. The count-based passes are turned off.
3. Number disputes: a name that counts two ways is right about at most one of them (`_disputes`), and a keyed
   run that numbers past its season's end, in a season whose absolute numbering interleaves a special, is
   disputed from that special's position on (`_positions_disputed`). The window that admits one of the episodes
   a dispute is over rejects the reading, and the file is `MISNUMBERED` unless its run places it (pass 8), which
   a run with a member its position disputes never does. Under any other window the file reads as it did. A
   lower version of a disputed number is `DUPLICATE`, a copy of its top version. The files a disputed reading
   would have beaten to an episode are its runner-ups: the refusal never hands them the episode. The count-based
   passes are turned off under every window.
4. Overlapping spans: a multi-episode reading that shares an episode with a different reading of another
   file is rejected.
5. Season re-read: a run numbered `1..N` without own keys, which Sonarr matched into one season of exactly
   N episodes with a few members shifted onto specials, is read as that season's episodes 1 to N. When the
   specials season fits just as well, it is a tie: the readings are rejected and both candidates recorded.
6. Misnumbered pack: a pack of specials, one file per number, as wide as the listing, with a number that is no
   listed special, is numbered some other way (an older TVDB state, or TMDB's order). If the listing's
   `special_aliases` map its numbers one to one onto the listed specials, and no title, size match, or seeded
   file puts a member anywhere else, each file goes on its aliased special (`ALTERNATE`, or `SKIPPED` when
   taken) and a superseded version is `DUPLICATE`. A file whose special is outside the window is
   `ALIASED_ELSEWHERE`, a claim `window_placement` settles across the windows. Otherwise every file of the run is
   `MISNUMBERED` for a hand import and no later pass places any of it, unless the window already has used ids.
   Then the later passes take the pack by its numbers, so a pack the stored record partly holds off its aliases
   keeps its own numbers. `_run_passes._sole_specials_run` lists what else turns the pass off.
7. Size identity: a file with the same size as a single-file release SeaDex lists for one episode goes on that
   episode (`IDENTIFIED`). If the file's reading already agrees, or its title names a different episode, it's
   left for the later passes. If the file's numbers point at a different episode, it's `OVERRIDDEN` and the
   caller warns. That means the batch's numbering is off, so the count-based passes are turned off and pass 8
   skips the file's numbered run. If the episode is already taken the file is `SKIPPED`, and if it's outside the
   scope it's `FOREIGN`. This runs after pass 6, so it never pulls a file out of a pack that pass refused or
   placed through its aliases.
8. Release run: when the open ids form a run window and the tie-breaks leave one numbered run that fits it
   (a `_run_passes._Tier`), the run's own numbers place its members onto the window, unless Sonarr's reading
   already puts each member on its own window episode. While any parse is unknown the members are `HELD` instead.
   `pass_release_run` lists the tie-breaks and the refusals.
9. Exact: a file whose trusted reading lies wholly inside the scope, on open ids, is placed there. Unscoped, a
   name's own keys place against the whole series.
10. Episode title: a file titled as an open scope episode, whose parse covers one episode, is placed there,
   whatever its number said. Each such placement turns the count-based passes off.
11. Absolute zip: the remaining files pair with the open ids in absolute-number order, only when every file
    carries exactly one absolute number, the counts match, every parse is known, and no two files of the
    torrent share an absolute number.
12. Single file: with one open id left, the only remaining file takes it if its parse has no usable episode
    number. Otherwise, the numberless file an AniList title names takes it (`TITLED`).
13. Ordered zip: when every file of the torrent but the extras is unplaced, numberless, and not read
    outside the scope, no id was used from the start (by a seeded file or an earlier window), and the counts
    match, the files pair with the open ids in natural name order.
14. Numbered run: among the files Sonarr resolved no episode for, the one run numbered `1..N` places onto
    a run window of N ids.
15. Classify: every file still unplaced becomes `DUPLICATE` (it reads inside the scope onto episodes other
    files provably hold, or a version of the same file holds its titled episode), `FOREIGN` (its reading lies
    wholly outside the scope, or it is titled as an episode outside it and confirmed by its reading, by that
    reading's rejection while other files hold its episodes, or by no open id being left), `TITLE_REFUTED` (a
    run member pass 2 rejected), or `SKIPPED`. The grab records the exclusions on the pending record, and the
    import warns about a skipped file it finds on disk. It never guesses.

At most one of passes 11 to 13 is tried: the absolute zip when its conditions hold, else the single-file
pass when one id is open, else the ordered zip. They consider only open files. Every zip (passes 8, 11,
13, and 14) places nothing when a file is titled as an episode other than its pair, and that turns the
count-based passes off.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ._run_passes import (
    MIN_TITLE_HITS,
    pass_misnumbered_pack,
    pass_numbered_run,
    pass_release_run,
    pass_reread_season_runs,
)
from ._state import NumberedRun, Placer, TorrentFile
from ..parse_reading import glued_counts, numbers_miss_the_series, parse_has_no_number
from ..placement_types import EpisodeAssignment, PlacementBatch, PlacementVerdict, TargetScope
from ..release_names import FileIdentity, natural_key, sole_title_match
from ..seadex_types import EpisodeKey


def assign_episode_ids(batch: PlacementBatch, scope: TargetScope) -> EpisodeAssignment:
    """Place the files of `batch` onto the ids of `scope` by running the passes the module docstring lists, in order.

    One file name never overrides the scope. With an empty series map, every verdict that needs the map is
    refused. The result holds one `Placement` per name of `PlacementBatch.names`, in batch order.
    """

    if not batch.names:
        # The passes would read every parse of the torrent only to return this same empty result.
        return EpisodeAssignment(())
    state = Placer.start(batch, scope)
    _pass_extras(state)
    _pass_refuse_refuted(state)
    _pass_refuse_disputed(state)
    _pass_refuse_overlaps(state)
    pass_reread_season_runs(state)
    pass_misnumbered_pack(state)
    _pass_identity(state)
    pass_release_run(state)
    _pass_exact(state)
    _pass_episode_title(state)
    _pass_counted(state)
    pass_numbered_run(state)
    _pass_classify(state)
    return EpisodeAssignment(state.placements())


def _pass_extras(state: Placer) -> None:
    """Set the extras aside as `PlacementVerdict.EXTRA`: never an episode, and left out of every count."""

    for file in state.remaining():
        if file.extra:
            state.set_aside(file, PlacementVerdict.EXTRA)


def _pass_refuse_refuted(state: Placer) -> None:
    """Reject each reading a file's title contradicts, its run's unconfirmed readings too, and hold unconfirmed counts.

    A seeded member's title refutes its run like any mate's. A member whose own title confirms its reading keeps
    it, and one rejected for a mate's title is noted (`TorrentFile.run_refuted`). A count no title confirms
    (`Placer.counted_by_title`) is held for a hand import, like a disputed one. Any rejection turns the count-based
    passes off.
    """

    files = state.counted
    titled = {file for file in files if (ids := file.reading.resolved) and file.contradicts(ids)}
    unconfirmed = {file for file in files if file.reading.resolved and state.counted_by_title(file)}
    refuted = set(titled)
    for run in state.counted_runs:
        mates = set(run.whole)
        if titled & mates:
            refuted |= {file for file in mates if file.reading.inside and not file.confirms(file.reading.resolved)}
    state.veto_each(refuted | unconfirmed)
    for file in refuted - titled:
        file.run_refuted = True
    for file in unconfirmed:
        file.disputed = True
    state.count_legs_barred |= bool(refuted or unconfirmed)


def _pass_refuse_disputed(state: Placer) -> None:
    """Reject each reading that another count of its own name, or its run's shape, puts on a different episode.

    See pass 3 in the module docstring. The runner-ups are `_runner_ups`.
    """

    runs = state.counted_runs
    candidates = {file: ids for file in state.counted if (ids := _disputes(state, file)) is not None}
    by_position: set[TorrentFile] = set()
    for run in runs:
        for file, ids in _positions_disputed(state, run).items():
            candidates[file] = candidates.get(file, frozenset()) | ids
            by_position.add(file)
    disputed = {file for file, ids in candidates.items() if any(state.scope.admits(ep_id) for ep_id in ids)}
    copies = [file for run in runs for file in run.superseded if file in disputed]
    # Runner-ups first: a rejected reading wins nothing, and a set-aside copy is out of the running.
    for file, lost in _runner_ups(state, disputed).items():
        file.runner_up_for = lost
    state.veto_each(disputed)
    state.set_aside_each([file for file in copies if file.remaining], PlacementVerdict.DUPLICATE)
    for file in disputed.difference(copies):
        file.disputed = True
    for file in by_position & disputed:
        file.disputed_by_position = True
    for file in candidates.keys() - disputed:
        file.disputed_elsewhere = True
    state.count_legs_barred |= bool(candidates)


def _disputes(state: Placer, file: TorrentFile) -> frozenset[int] | None:
    """The episodes the name's own counts spread it over: its key's and the other count's. None when they agree.

    Only a count that is another episode's absolute number disputes. Beside a spelled-out key or a reading borrowed
    from Sonarr's match, that is a bracketed count ("S10E01 (0337)"). Beside a key Sonarr guessed, it is any count
    when Sonarr's series match doesn't back the key, else the one the key was glued from (`glued_counts`).
    """

    info = file.parsed
    if info is None:
        return None
    reading = file.reading
    name_read = file.name_read
    by_absolute = state.series.id_by_absolute
    resolved = frozenset(reading.resolved)
    if reading.borrowed or not info.episode_numbers or info.season_number is None:
        if not resolved:
            return None
        counts: Iterable[int] = name_read.tagged_numbers
    elif name_read.explicit_key:
        counts = name_read.tagged_numbers
    elif resolved and not reading.corroborated:
        counts = name_read.bare_numbers
    else:
        counts = glued_counts(info).intersection(name_read.bare_numbers)
    others = {ep_id for count in counts if (ep_id := by_absolute.get(count)) is not None} - resolved
    return resolved | others if others else None


def _runner_ups(state: Placer, disputed: set[TorrentFile]) -> dict[TorrentFile, frozenset[int]]:
    """Each unplaced file's episodes a disputed reading takes first when the exact pass runs on the current readings.

    Those files are the runner-ups the refusal must not hand the episodes to.
    """

    winner = {ep_id: file for file, ids in state.exact_picks().items() for ep_id in ids}
    runner_ups: dict[TorrentFile, frozenset[int]] = {}
    for file in state.remaining():
        lost = frozenset(
            ep_id for ep_id in file.reading.exact_ids if (won := winner.get(ep_id)) in disputed and won is not file
        )
        if lost:
            runner_ups[file] = lost
    return runner_ups


def _positions_disputed(state: Placer, run: NumberedRun) -> dict[TorrentFile, frozenset[int]]:
    """The members of a keyed run numbered past its season's end, from the first special the season interleaves on.

    Such a run (`1..N+k` for an N-episode season with k interleaved specials) counted the specials or appended
    extras. From the first special's position on, the two counts put a member on different episodes. A run keyed
    within its season is left alone. Each disputed member maps to the episodes the two counts spread the disputed
    members over: the season's episodes from the special's position on, and the interleaved specials the run
    numbers.
    """

    seasons = {info.season_number for file in run.members if (info := file.parsed) is not None}
    if len(seasons) != 1 or (season := seasons.pop()) in (None, 0):
        return {}
    if any((info := file.parsed) is None or info.episode_numbers[:1] != (file.release_number,) for file in run.members):
        return {}
    if not all(file.name_read.explicit_key for file in run.members):
        return {}
    count = state.series.season_counts.get(season, 0)
    # A season the map lacks is no season to count against (the keys miss, and the file reads numberless).
    if not count or max(run.numbers) <= count:
        return {}
    absolute_of, key_by_id = state.series.absolute_of, state.series.key_by_id
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
    return {file: spread for file in run.members if file.release_number >= position}


def _pass_refuse_overlaps(state: Placer) -> None:
    """Reject each unplaced multi-episode reading that shares an episode with a different reading of another file.

    The other file may be a seeded one.
    """

    spans_by_ep: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    for file in state.counted:
        if not (reading := file.reading).vetoed:
            for ep_id in reading.resolved:
                spans_by_ep[ep_id].add(reading.resolved)
    overlapping = [
        file
        for file in state.remaining()
        if len(ids := file.reading.resolved) > 1 and any(len(spans_by_ep[ep_id]) > 1 for ep_id in ids)
    ]
    state.veto_each(overlapping)


def _pass_identity(state: Placer) -> None:
    """Place files on the episode their size matched (`ListingEvidence.identified`), even over their names.

    A file is left for later passes if its reading already agrees, its title names another episode, or the
    matched episode belongs to another series. See pass 7 in the module docstring.
    """

    for file in state.remaining():
        ep_id = file.identified_as
        if ep_id is None or ep_id not in state.scope.series.by_id:
            continue
        reading = file.reading
        if (reading.trusted and reading.resolved == (ep_id,)) or file.contradicts((ep_id,)):
            continue
        if not state.scope.admits(ep_id):
            state.set_aside(file, PlacementVerdict.FOREIGN)
            continue
        overrides = _numbers_another(state, file, ep_id)
        if overrides:
            # No count-based pass may place the rest, and the release run skips this file's run.
            state.count_legs_barred = True
            if (member := file.name_read.member) is not None:
                state.size_refuted_runs.add(member.prefix)
        state.place_or_skip(file, ep_id, PlacementVerdict.OVERRIDDEN if overrides else PlacementVerdict.IDENTIFIED)


def _numbers_another(state: Placer, file: TorrentFile, ep_id: int) -> bool:
    """True if the name has a number and none of its numbers point at `ep_id`.

    Numbers here means its reading, its own keys (ignored if none of them exist in the series), and its
    absolute numbers. A name with no keys also counts its release number, as an episode or an absolute.
    """

    resolved = file.reading.resolved
    if ep_id in resolved:
        return False
    info = file.parsed
    member = file.name_read.member
    keys: set[EpisodeKey] = set()
    absolutes: set[int] = set()
    if info is not None:
        absolutes.update(info.absolute_episode_numbers)
        if info.season_number is not None and not numbers_miss_the_series(info, state.scope.id_by_key):
            keys.update(EpisodeKey(info.season_number, number) for number in info.episode_numbers)
    if not (resolved or keys or absolutes or member):
        return False
    key = state.series.key_by_id.get(ep_id)
    absolute = state.series.absolute_of.get(ep_id)
    if key in keys or absolute in absolutes:
        return False
    # In a keyed name the release number is no evidence beyond the key.
    if keys:
        return True
    return member is None or member.number not in {absolute, None if key is None else key.episode}


def _pass_exact(state: Placer) -> None:
    """Place each unplaced file whose trusted reading lies wholly inside the scope on unused ids.

    When two files resolve to one episode, `Reading.rank` decides, then the higher `vN`, then batch order.
    So a "17.5 (S00E01)" beats a "- 17" whose match Sonarr shifted onto the same special, a "- 08" beats a
    "- Bonus" whose CRC tag parsed as its key, and a "- 09v2" beats its "- 09". The loser stays unplaced.
    """

    for file, ids in state.exact_picks().items():
        state.place(file, ids, PlacementVerdict.EXACT)


def _pass_episode_title(state: Placer) -> None:
    """Place each unplaced file on the open scope episode its name is titled as, highest `vN` first.

    It runs even when the count-based passes are off, and each placement turns them off.
    """

    for file in sorted(state.remaining(), key=lambda f: f.name_read.version, reverse=True):
        if (ep_id := _title_placement(state, file)) is not None:
            state.place(file, (ep_id,), PlacementVerdict.EPISODE_TITLE)
            state.count_legs_barred = True


def _title_placement(state: Placer, file: TorrentFile) -> int | None:
    """The open scope episode the name is titled as, else None.

    Also None for a runner-up on that episode, for a missing or multi-episode parse, for a tied file titled as
    neither of its candidates, and for a run's lone title whose episode a mate reads by number (`_outnumbered`).
    """

    ep_id = file.titled_as
    if ep_id is None or ep_id in state.used or not state.scope.admits(ep_id) or state.single_parse(file) is None:
        return None
    if file.runner_up((ep_id,)) or _outnumbered(state, file, ep_id):
        return None
    tied = file.reading.tied
    return None if tied is not None and ep_id not in tied else ep_id


def _outnumbered(state: Placer, file: TorrentFile, ep_id: int) -> bool:
    """Whether another file of the name's run reads `ep_id`, the run's count standing.

    The count stands when fewer than `MIN_TITLE_HITS` files of the run are titled against their readings.
    """

    for run in state.counted_runs:
        mates = run.whole
        if file in mates:
            against = sum(mate.contradicts(mate.reading.resolved) for mate in mates)
            reads_it = any(mate is not file and ep_id in mate.reading.resolved for mate in mates)
            return against < MIN_TITLE_HITS and reads_it
    return False


def _pass_counted(state: Placer) -> None:
    """Try the absolute zip, else the single-file pass when one id is open, else the ordered zip."""

    # A file whose trusted reading lies wholly outside the scope belongs to another slice.
    # It takes no open id and does not spoil the count for the other files.
    open_files = [file for file in state.remaining() if not file.reading.outside]
    open_ids = state.open_ids()
    if not open_files or not open_ids:
        return
    if (order := _absolute_order(state, open_files, open_ids)) is not None:
        state.place_zip(order, open_ids, PlacementVerdict.ABSOLUTE)
    elif len(open_ids) == 1:
        _place_single(state, open_files, open_ids[0])
    elif (order := _natural_order(state, open_files, open_ids)) is not None:
        state.place_zip(order, open_ids, PlacementVerdict.ORDERED)


def _absolute_order(
    state: Placer, open_files: Sequence[TorrentFile], open_ids: Sequence[int]
) -> list[TorrentFile] | None:
    """The open files in absolute-number order when the absolute zip applies (see the module), else None.

    An absolute Sonarr counted from the file's own title's number (`Placer.counted_by_title`) is no absolute.
    """

    abs_by_file: dict[TorrentFile, int] = {}
    for file in open_files:
        info = file.parsed
        if info is not None and len(info.absolute_episode_numbers) == 1 and not state.counted_by_title(file):
            abs_by_file[file] = info.absolute_episode_numbers[0]
    # The shared-absolute check reads every non-extra parse, seeded files included. Repeats inside one parse collapse.
    batch_absolutes = [
        number
        for file in state.counted
        if (parse := file.parsed) is not None
        for number in dict.fromkeys(parse.absolute_episode_numbers)
    ]
    applies = (
        bool(abs_by_file)
        and not state.count_legs_barred
        and state.all_parses_known
        and len(abs_by_file) == len(open_files)  # every open file has one absolute
        and len(abs_by_file) == len(open_ids)  # as many files as open ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restarted numbering)
    )
    return sorted(abs_by_file, key=abs_by_file.__getitem__) if applies else None


def _place_single(state: Placer, open_files: Sequence[TorrentFile], ep_id: int) -> None:
    """Fill the one open id: with the only open file if it is numberless, else the numberless file a title names.

    The title is an AniList title of the entry, and that match runs only when every parse is known. A runner-up for
    the id takes it neither way, and no other file takes it in its place.
    """

    unnumbered = [file for file in open_files if _reads_no_number(state, file)]
    if len(open_files) == 1 and unnumbered:
        pick, verdict = 0, PlacementVerdict.SINGLE
    elif state.all_parses_known:
        stems = [file.name_read.stem.text for file in unnumbered]
        pick, verdict = sole_title_match(stems, state.scope.names), PlacementVerdict.TITLED
    else:
        return
    if pick is not None and not unnumbered[pick].runner_up((ep_id,)):
        state.place(unnumbered[pick], (ep_id,), verdict)


def _reads_no_number(state: Placer, file: TorrentFile) -> bool:
    """Whether the file's single-episode parse has no usable number: none at all, or only keys the series lacks.

    False for a disputed file (its numbers name episodes, they just disagree), for a tied file, and for a
    titled file (its titled episode is taken or outside the scope).
    """

    if file.disputed or file.disputed_elsewhere:
        return False
    info = state.single_parse(file)
    if info is None:
        return False
    if not (parse_has_no_number(info) or numbers_miss_the_series(info, state.scope.id_by_key)):
        return False
    return file.reading.tied is None and file.titled_as is None


def _natural_order(
    state: Placer, open_files: Sequence[TorrentFile], open_ids: Sequence[int]
) -> list[TorrentFile] | None:
    """The open files in natural name order when the ordered zip applies (the "Special 1..N" shape), else None."""

    # Pristine: the parse map holds the open files and the extras and nothing else, and nothing but an extra
    # has a verdict, so nothing was placed, held, or seeded.
    open_set = set(open_files)
    pristine = (
        not state.count_legs_barred
        and len(open_files) == len(open_ids)
        and all(file.in_parse_map == (file in open_set or file.extra) for file in state.files)
        and state.untouched
        and all((info := state.single_parse(file)) is not None and parse_has_no_number(info) for file in open_files)
    )
    return sorted(open_files, key=lambda file: natural_key(file.name)) if pristine else None


# Verdicts for files placed by size.
_BY_SIZE = frozenset({PlacementVerdict.IDENTIFIED, PlacementVerdict.OVERRIDDEN})


@dataclass(frozen=True, slots=True)
class _DuplicateProof:
    """The evidence that an unplaced file is a duplicate, gathered once before classification."""

    proven: frozenset[int]
    """Ids we have a reason to think are held: placed by this call (except by size), or pointed at by a seeded or
    gone file's reading or title."""
    held_by_file: Mapping[FileIdentity, frozenset[int]]
    """The episodes each file's versions hold: placed by this call, or a seeded or gone version titled as one."""


def _pass_classify(state: Placer) -> None:
    """Classify every unplaced file."""

    proof = _duplicate_proof(state)
    nothing_left = not state.open_ids()
    for file in state.remaining():
        reading = file.reading
        titled = file.titled_as
        if file.disputed:
            # Its own numbers disagree, or its count stands unconfirmed, and no run placed it: a hand import.
            state.set_aside(file, PlacementVerdict.MISNUMBERED)
        elif ids := reading.exact_ids:
            # The exact pass skipped it: an episode is taken. A span only partly taken stays SKIPPED.
            proven = all(ep_id in state.used and ep_id in proof.proven for ep_id in ids)
            state.set_aside(file, PlacementVerdict.DUPLICATE if proven else PlacementVerdict.SKIPPED)
        elif titled is not None and titled in state.used:
            # Titled as a taken episode: a duplicate only when a version of the same file holds it.
            twin = titled in proof.held_by_file.get(file.identity, ())
            state.set_aside(file, PlacementVerdict.DUPLICATE if twin else PlacementVerdict.SKIPPED)
        elif state.map_known and (reading.outside or _titled_outside(state, file, nothing_left)):
            state.set_aside(file, PlacementVerdict.FOREIGN)
        elif file.run_refuted:
            state.set_aside(file, PlacementVerdict.TITLE_REFUTED)
        else:
            state.set_aside(file, PlacementVerdict.SKIPPED)


def _duplicate_proof(state: Placer) -> _DuplicateProof:
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
            evidenced.update(file.reading.resolved)
            if (titled := file.titled_as) is not None:
                evidenced.add(titled)
                held[file.identity].add(titled)
    proven = (state.used - state.scope.used - by_size) | evidenced
    return _DuplicateProof(frozenset(proven), {identity: frozenset(ids) for identity, ids in held.items()})


def _titled_outside(state: Placer, file: TorrentFile, nothing_left: bool) -> bool:
    """Whether the name is titled as an episode outside the scope, confirmed another way.

    Confirmed when the file has a complete reading wholly outside the scope, when its reading was rejected
    and every episode it pointed at is held by another file of this torrent, or when no open id is left. A
    title alone never excludes a file the entry may still need.
    """

    ep_id = file.titled_as
    if ep_id is None or state.scope.admits(ep_id):
        return False
    reading = file.reading
    held = bool(reading.resolved) and state.used.issuperset(reading.resolved)
    return (reading.complete and not reading.inside) or (reading.vetoed and held) or nothing_left
