"""Pure placement: `assign_episode_ids` runs the ordered passes that map a batch's files onto the scope's episodes."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Self

from .parse_reading import Reading, claims_several_episodes, numbers_miss_the_series, parse_has_no_number, read_parse
from .placement_types import EpisodeAssignment, EpisodeIndex, Placement, PlacementBatch, PlacementVerdict, TargetScope
from .release_names import (
    NameRead,
    NumberedRun,
    RunMember,
    best_title_matches,
    folded_words,
    is_consecutive,
    is_extras_name,
    natural_key,
    numbered_runs,
    read_name,
    runs_from_one,
    runs_with_numbers,
    sole_title_match,
)
from .seadex_types import EpisodeKey, ParsedFileInfo, season_episode_key


def assign_episode_ids(
    batch: PlacementBatch,
    scope: TargetScope,
) -> EpisodeAssignment:
    """Map a torrent's files to OUR resolved episode ids. Names never override.

    The resolved set (`scope.resolved`, season-sorted, lifted from the
    add-flow `ep_list`) is authoritative. A release's own numbering is only ever
    used to *index into* it, never to decide identity. One reading per file
    (`read_parse`), then passes in strict precedence over one state, each placing
    into the ids the earlier ones left:

    1. **Release run:** the batch's one run fitting a consecutive window
       (one season's episodes, or the series' absolutes across several:
       numbered `1..N`, or as the window's own episodes, or `1..N` for the
       whole season a slice window belongs to, or N consecutive numbers
       Sonarr read nothing of) indexes it when Sonarr's reading of the
       members is incoherent (see `_pass_release_run`). A `1..N` run Sonarr
       matched into the one season of exactly N episodes, a few members
       shifted onto its specials, is first re-read as that season's own
       numbering. Members are HELD, not placed, while any parse in the batch
       is unknown.
    2. **Exact (season, episode):** a file whose reading resolves cleanly inside
       the resolved set is placed there (a name Sonarr just couldn't match, a
       per-season multi-season pack, an absolute-only name borrowing Sonarr's
       matched pair under the same in-set scoping). With NO resolved set
       (`scope.unscoped`) the name-parsed keys place against the live series
       map directly, so a correctly-named file still imports rather than sticking.
    3. **Absolute index:** the leftovers zip onto the leftover ids by absolute
       number, ONLY when every leftover carries a single absolute, the counts
       match 1:1, every parse in the batch is known, and no two files ANYWHERE
       in the batch share an absolute (the restart-numbering tell, counted over
       seeded files too).
    4. **Single file:** one leftover file onto one leftover id, when the name
       carries no number at all (or only a provably-bogus key, one missing the
       WHOLE series map) and Sonarr's matched evidence spans no episodes.
    5. **Ordered zip:** a pristine numberless batch (every parse known, real,
       numberless, single-span, covering EXACTLY the leftover files) zips
       natural name order onto the leftover ids 1:1 (the "Special 1..N" shape).
    6. **Numbered run:** the one `1..N` run among the files Sonarr read nothing
       from indexes a consecutive window of that width, mixed batch or not.
    7. **Classify:** what is left resolves inside the set onto a taken episode
       (`DUPLICATE`), or cleanly and entirely outside it (`FOREIGN`), or is
       simply `SKIPPED`. The caller warns on skips and records exclusions,
       never guesses.

    What rides in is documented on `PlacementBatch` (the parses may cover more
    files than are placed) and `TargetScope` (an empty resolved set is no scope
    at all). An empty series map refuses every map-dependent verdict. One
    `Placement` per distinct file, batch order.
    """

    state = _Placer.start(batch, scope)
    _pass_release_run(state)
    _pass_exact(state)
    _pass_counted(state)
    _pass_numbered_run(state)
    return state.finish()


# Episode titles decide between runs, or against one, from this many members on.
_MIN_TITLE_HITS = 2


@dataclass(frozen=True, slots=True)
class _RunWindow:
    """The leftover ids as consecutive episodes in airing order."""

    season: int | None
    """The one season the ids belong to, or None when the series' absolute numbering orders several."""
    ids: tuple[int, ...]
    """The leftover ids, airing order."""
    numbers: tuple[int, ...]
    """The ids' episode numbers (absolute numbers across seasons), the same order."""
    id_set: frozenset[int] = field(init=False, repr=False, compare=False)
    """The ids as a set, built once for the per-candidate reads."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id_set", frozenset(self.ids))

    @property
    def specials(self) -> bool:
        """The specials season's window."""

        return self.season == 0

    def slice_of(self, run: NumberedRun) -> tuple[RunMember, ...]:
        """The run's members numbered as the window's episodes (a whole-season run over a slice window)."""

        return tuple(member for member in run.members if member.number in self.numbers)


class _TitledEpisode(NamedTuple):
    """One series episode with a title, as folded words."""

    ep_id: int
    words: tuple[str, ...]


class _TitleEvidence(NamedTuple):
    """How many of a run's tails name an episode inside the window, and how many one outside it."""

    inside: int
    outside: int

    @property
    def selects(self) -> bool:
        """Enough tails name the window, and more than name anything else."""

        return self.inside >= _MIN_TITLE_HITS and self.inside > self.outside

    @property
    def vetoes(self) -> bool:
        """The tails name other episodes and none of the window's."""

        return self.outside >= _MIN_TITLE_HITS and not self.inside


@dataclass(frozen=True, slots=True)
class _SeriesFacts:
    """What the series map says on its own, read once per call and never changed by a pass."""

    key_by_id: Mapping[int, EpisodeKey]
    """The series map inverted."""
    season_counts: Mapping[int, int]
    """Episodes per season."""
    absolute_of: Mapping[int, int]
    """Episode id -> the series' absolute number, for the ids that carry one."""
    titled: tuple[_TitledEpisode, ...]
    """The episodes that carry a title, as folded words."""

    @classmethod
    def of(cls, series: EpisodeIndex) -> Self:
        """The one reader of the series index."""

        episodes = series.by_id
        return cls(
            key_by_id={ep_id: key for key, ep_id in series.id_by_key.items()},
            season_counts=Counter(key.season for key in series.id_by_key),
            absolute_of={
                ep_id: ep.absolute_episode_number
                for ep_id, ep in episodes.items()
                if ep.absolute_episode_number is not None
            },
            titled=tuple(
                _TitledEpisode(ep_id, words) for ep_id, ep in episodes.items() if (words := folded_words(ep.title))
            ),
        )

    def title_evidence(self, run: NumberedRun, window: _RunWindow) -> _TitleEvidence:
        """Count the run's tails naming an episode inside the window against those naming one outside it.

        A tail names an episode when the episode's title opens it (the rest
        is quality junk). A tail naming episodes on both sides counts for
        neither.
        """

        ids = window.id_set
        inside = outside = 0
        for member in run.members:
            words = member.tail_words
            if not words:
                continue
            sides = {episode.ep_id in ids for episode in self.titled if words[: len(episode.words)] == episode.words}
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return _TitleEvidence(inside, outside)


@dataclass(slots=True)
class _Placer:
    """One `assign_episode_ids` call's state: the readings, the verdicts so far, and the ids they used."""

    batch: PlacementBatch
    scope: TargetScope
    readings: dict[str, Reading]
    """One reading per distinct name to place, batch order."""
    name_reads: Mapping[str, NameRead]
    """One read of each name to place: its stem and run membership."""
    facts: _SeriesFacts
    verdicts: dict[str, Placement] = field(default_factory=dict[str, Placement])
    used: set[int] = field(default_factory=set[int])
    count_legs_barred: bool = False
    """The release-run pass found several runs, or refused the one it found: no count leg (the numbered run,
    the absolute and ordered zips) places what it would not."""

    @classmethod
    def start(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """Read every name once against the scope."""

        readings = {name: read_parse(batch.parsed.get(name), scope) for name in dict.fromkeys(batch.to_place)}
        name_reads = {name: read_name(name) for name in readings}
        state = cls(batch, scope, readings, name_reads, _SeriesFacts.of(scope.series), used=set(scope.used))
        state.reread_season_runs()
        return state

    @property
    def map_known(self) -> bool:
        """Whether the series map was served (every map-dependent verdict refuses on an empty one)."""

        return bool(self.scope.id_by_key)

    def reread_season_runs(self) -> None:
        """Re-read a `1..N` run Sonarr matched into one season of exactly N episodes as that season's own numbering.

        TVDB interleaves specials into the absolute numbering, so Sonarr's
        match of a season-only release drifts onto a special after each one.
        The count tells the shapes apart: a release that carried the special
        would number N + 1. Only a run of names without keys of their own,
        read by Sonarr into that one season and its specials, is re-read:
        more into the season, or as many when the specials are not N either.
        As many onto the specials when they count N too is a tie: the reads
        are vetoed, the episodes either numbering gives are `tied`, and only
        the run's count over an entry's window places it. A lower version of
        a member reads as the member does.
        """

        if not self.map_known:
            return
        for run in self.runs(self.readings):
            width = len(run.members)
            if run.numbers != tuple(range(1, width + 1)):
                continue
            infos = [self.batch.parsed.get(name) for name in run.names]
            if any(info is None or info.episode_numbers for info in infos):
                continue
            readings = [self.readings[name] for name in run.names]
            if any(len(r.resolved) > 1 for r in readings):
                continue
            read = [self.facts.key_by_id[r.resolved[0]].season for r in readings if r.resolved]
            seasons = {season for season in read if season != 0}
            if len(seasons) != 1:
                continue
            season = seasons.pop()
            if self.facts.season_counts.get(season) != width:
                continue
            into_season = read.count(season)
            onto_specials = read.count(0)
            if not onto_specials or into_season < onto_specials:
                continue
            if into_season == onto_specials and self.facts.season_counts.get(0) == width:
                self.tie(run, season)
            else:
                self.reread(run, season)

    def tie(self, run: NumberedRun, season: int) -> None:
        """Veto the run's reads and record the episodes each name may be under either numbering."""

        for numbered in run.whole:
            keys = (EpisodeKey(season, numbered.number), EpisodeKey(0, numbered.number))
            ids = tuple(i for key in keys if (i := self.scope.id_by_key.get(key)))
            self.readings[numbered.name] = self.readings[numbered.name]._replace(vetoed=True, tied=ids)

    def reread(self, run: NumberedRun, season: int) -> None:
        """Read every name of the run as the season's episode of its number."""

        for numbered in run.whole:
            ep_id = self.scope.id_by_key.get(EpisodeKey(season, numbered.number))
            if ep_id:
                inside = (ep_id,) if ep_id in self.scope.real_ids else ()
                self.readings[numbered.name] = Reading(
                    (ep_id,), inside, complete=True, borrowed=True, vetoed=False, corroborated=True
                )

    def veto(self, run: NumberedRun) -> None:
        """Veto every reading of the run, its displaced versions included."""

        for numbered in run.whole:
            self.readings[numbered.name] = self.readings[numbered.name]._replace(vetoed=True)

    def window(self) -> list[int]:
        """The leftover ids: the scope's resolved set minus every id used so far, scope order."""

        return [i for i in self.scope.resolved if i and i not in self.used]

    def run_window(self) -> _RunWindow | None:
        """The window as consecutive episodes in airing order, else None.

        One season's episode numbers, else the series' absolute numbers when
        every slot carries one (an entry holding a special TVDB interleaved,
        or two seasons' cours). Never trusts the scope's order: the ids are
        re-sorted by their keys.
        """

        window = self.window()
        if len(window) < 2:
            return None
        key_by_id = self.facts.key_by_id
        keyed = sorted((key_by_id[ep_id], ep_id) for ep_id in window if ep_id in key_by_id)
        if len(keyed) != len(window):
            return None
        seasons = {key.season for key, _ in keyed}
        episodes = [key.episode for key, _ in keyed]
        if len(seasons) == 1 and is_consecutive(episodes):
            return _RunWindow(seasons.pop(), tuple(ep_id for _, ep_id in keyed), tuple(episodes))
        absolute_of = self.facts.absolute_of
        absolute = sorted((absolute_of[ep_id], ep_id) for ep_id in window if ep_id in absolute_of)
        if len(absolute) != len(window) or not is_consecutive([number for number, _ in absolute]):
            return None
        return _RunWindow(None, tuple(ep_id for _, ep_id in absolute), tuple(number for number, _ in absolute))

    def remaining(self) -> list[str]:
        """The names without a verdict yet, batch order."""

        return [name for name in self.readings if name not in self.verdicts]

    def spans_multiple(self, info: ParsedFileInfo) -> bool:
        """Whether the file plausibly holds more than one episode: the name's claim, or a multi-pair series match.

        Cardinality, not identity: a file Sonarr matched to two episodes, placed as one, is a structural
        half-import. A full-season read (a name with no episode token) counts only when that season
        reaches into the entry, never when it names another season.
        """

        if claims_several_episodes(info):
            return True
        pairs = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
        if len(pairs) <= 1:
            return False
        if not info.full_season:
            return True
        return any(
            self.scope.id_by_key.get(season_episode_key(season, episode)) in self.scope.real_ids
            for season, episode in pairs
        )

    def settled_elsewhere(self, run: NumberedRun) -> bool:
        """Whether Sonarr read the run whole into distinct episodes outside the scope: another season's, or the other cour's.

        A tie's run is elsewhere when neither numbering reaches the scope.
        """

        readings = [self.readings[name] for name in run.names]
        if all(r.tied is not None for r in readings):
            return not any(i in self.scope.real_ids for r in readings for i in r.tied or ())
        if not all(r.outside and r.corroborated and len(r.resolved) == 1 for r in readings):
            return False
        return len({r.resolved[0] for r in readings}) == len(readings)

    def covering(self, runs: Iterable[NumberedRun], window: _RunWindow) -> list[NumberedRun]:
        """The `1..N` runs for the whole season a strict slice window belongs to: N members over fewer slots."""

        season = window.season
        if season is None:
            return []
        count = self.facts.season_counts.get(season, 0)
        return runs_from_one(runs, count) if count > len(window.ids) else []

    def covering_refused(self, window: _RunWindow) -> bool:
        """Whether a season run's count says nothing about the window.

        A specials window (a season run as wide as the specials count is a
        coincidence), or a season whose numbering has a gap (the run counts
        on past it).
        """

        season = window.season
        if season is None or window.specials:
            return True
        count = self.facts.season_counts.get(season, 0)
        return any(EpisodeKey(season, n) not in self.scope.id_by_key for n in range(1, count + 1))

    def seasons_read(self, run: NumberedRun) -> set[int]:
        """The seasons Sonarr read the run's members into."""

        return {self.facts.key_by_id[ep_id].season for name in run.names for ep_id in self.readings[name].resolved}

    def read_elsewhere(self, run: NumberedRun, season: int | None) -> bool:
        """Whether Sonarr read a member into another season than the window's: the reads dispute the run's count."""

        return any(read != season for read in self.seasons_read(run))

    def place(self, name: str, ids: Sequence[int], verdict: PlacementVerdict) -> None:
        """Record a placement and take its ids out of the window."""

        self.verdicts[name] = Placement(name, tuple(ids), verdict)
        self.used.update(ids)

    def set_aside(self, name: str, verdict: PlacementVerdict) -> None:
        """Record a verdict that carries no ids (held, excluded, or skipped)."""

        self.verdicts[name] = Placement(name, (), verdict)

    def set_aside_each(self, members: Iterable[RunMember], verdict: PlacementVerdict) -> None:
        """Record one id-less verdict for each run member."""

        for member in members:
            self.set_aside(member.name, verdict)

    def runs(self, names: Iterable[str]) -> list[NumberedRun]:
        """The numbered runs among `names` (each a name to place), by their reads."""

        members = (member for name in names if (member := self.name_reads[name].member) is not None)
        return numbered_runs(members, self.batch.parsed)

    def finish(self) -> EpisodeAssignment:
        """Classify what is still open, then fold the verdicts in batch order."""

        # An id is a proven duplicate's when its holder reads there too: placed this batch (no pass places
        # where a keyed open file resolves without evidence), or a name outside `to_place` (seeded or gone)
        # whose own parse resolves there. A seed that landed a file positionally is a disagreement the
        # caller reports, not a verdict.
        evidenced = frozenset(
            ep_id
            for name, info in self.batch.parsed.items()
            if name not in self.readings
            for ep_id in read_parse(info, self.scope).resolved
        )
        proven = (self.used - set(self.scope.used)) | evidenced
        for name in self.remaining():
            reading = self.readings[name]
            if reading.complete and not reading.vetoed and reading.inside == reading.resolved:
                # Resolves cleanly inside the set: the exact pass left it only because its episode is taken.
                taken = [ep_id for ep_id in reading.inside if ep_id in self.used]
                duplicate = bool(taken) and all(ep_id in proven for ep_id in taken)
                self.set_aside(name, PlacementVerdict.DUPLICATE if duplicate else PlacementVerdict.SKIPPED)
            elif reading.outside and self.map_known:
                self.set_aside(name, PlacementVerdict.FOREIGN)
            else:
                self.set_aside(name, PlacementVerdict.SKIPPED)
        return EpisodeAssignment(tuple(self.verdicts[name] for name in self.readings))


def _pass_release_run(state: _Placer) -> None:
    """Judge the release's own numbering against Sonarr's reading of its members.

    Stands down unless the window is consecutive episodes (one season's, or
    the series' absolutes across several) and one run is left to index it.
    The candidates, by tier: runs numbered `1..N` for its width N, runs
    numbered exactly as its episodes (a split cour's second half), runs
    numbered `1..N` for the whole season a slice window belongs to, and N
    consecutive numbers from anywhere when no member's reading is complete
    and no seed owns part of the scope (a release numbering the whole
    series across its seasons). Among several (a franchise pack), a run
    Sonarr read whole elsewhere stands aside for the rest, then the episode
    titles the members carry name one, then an AniList title does, then the
    highest tier holds. Several left is none, and the other count legs
    stand down too. With any parse in the batch unknown the members are HELD (no
    later pass may place what this one could not judge). Refuses (the exact
    pass proceeds) on `_run_refused`, on a covering run whose count says
    nothing about the window, on a covering run Sonarr read into another
    season, or on `_pick_refused`. Every refusal stands the other count
    legs down too, and a disputed covering run Sonarr did not read whole into
    one other season is vetoed before any pick. A coherent reading (every
    member one distinct id inside the window) stands. Otherwise Sonarr's reading is incoherent (a
    TVDB special shifted its match, the pairs point outside, the keys are
    bogus, or it read nothing) and the run indexes the window. A
    whole-season run's members past a slice window are the other slice's.
    """

    if not state.map_known:
        return
    window = state.run_window()
    if window is None:
        return
    width = len(window.ids)
    runs = state.runs(state.remaining())
    covering = state.covering(runs, window)
    # Reads into another season dispute a covering run's count, pick or not: read whole into one other
    # season, it is that season's (its members foreign), else its files are nowhere.
    disputed = [run for run in covering if state.read_elsewhere(run, window.season)]
    for run in disputed:
        if not (state.settled_elsewhere(run) and len(state.seasons_read(run)) == 1):
            state.veto(run)
    tiers = (
        runs_from_one(runs, width),
        runs_with_numbers(runs, window.numbers),
        covering,
        # A window a seed already took part of fits a run from anywhere by chance, never by count.
        [
            run
            for run in runs
            if not state.used and run.consecutive(width) and not any(state.readings[n].complete for n in run.names)
        ],
    )
    candidates = list(dict.fromkeys(run for tier in tiers for run in tier))
    state.count_legs_barred = len(candidates) > 1
    if state.count_legs_barred:
        candidates = [run for run in candidates if not state.settled_elsewhere(run)] or candidates
    if (
        len(candidates) > 1
        and len(selected := [r for r in candidates if state.facts.title_evidence(r, window).selects]) == 1
    ):
        candidates = selected
    if (
        len(candidates) > 1
        and (named := sole_title_match([run.prefix for run in candidates], state.scope.names)) is not None
    ):
        candidates = [candidates[named]]
    if len(candidates) > 1:
        candidates = next(kept for tier in tiers if (kept := [run for run in tier if run in candidates]))
    if len(candidates) != 1:
        return
    run = candidates[0]
    if not state.batch.all_parses_known:
        state.set_aside_each(run.whole, PlacementVerdict.HELD)
        return
    covers = run in covering
    refused = (
        _run_refused(state, run, window)
        or (covers and state.covering_refused(window))
        or run in disputed
        or _pick_refused(state, run, runs, window)
    )
    if refused:
        state.count_legs_barred = True
        return
    # A whole-season run over a slice window places the slice's numbers. The rest is the other slice's.
    members = window.slice_of(run) if covers else run.members
    readings = [state.readings[member.name] for member in members]
    coherent = all(
        r.complete and not r.vetoed and len(r.resolved) == 1 and r.resolved[0] in window.id_set for r in readings
    ) and len({r.resolved[0] for r in readings}) == len(readings)
    if coherent:
        return
    for member, ep_id in zip(members, window.ids, strict=True):
        state.place(member.name, [ep_id], PlacementVerdict.RELEASE_RUN)
    for member in run.members:
        if member not in members:
            state.set_aside(member.name, PlacementVerdict.FOREIGN)
    state.set_aside_each(run.superseded, PlacementVerdict.DUPLICATE)


def _pass_exact(state: _Placer) -> None:
    """Place every open file whose reading resolves cleanly inside the scope onto unused ids.

    Two files resolving to one episode are judged by `Reading.rank`, batch
    order within a rank: a "17.5 (S00E01)" beats the "- 17" whose match
    Sonarr shifted onto the same special, while a "- 08" beats an "- ED"
    whose CRC tag parsed as its key. Within that, a "- 09v2" beats its
    "- 09". The loser is left over (a duplicate).
    """

    for name in sorted(
        state.remaining(), key=lambda name: (state.readings[name].rank, -state.name_reads[name].version)
    ):
        reading = state.readings[name]
        if not reading.complete or reading.vetoed or reading.inside != reading.resolved:
            continue
        if any(i in state.used for i in reading.inside):
            continue
        state.place(name, reading.inside, PlacementVerdict.EXACT)


def _pass_counted(state: _Placer) -> None:
    """The count legs over the open files and the window: absolute zip, else single file, else ordered zip."""

    # A file reading wholly outside the scope is another slice's: it neither
    # takes a leftover id nor blocks the count for the files that could.
    open_names = [name for name in state.remaining() if not state.readings[name].outside]
    window = state.window()
    if not open_names or not window:
        return
    parsed = state.batch.parsed

    abs_by_file: dict[str, int] = {}
    for name in open_names:
        info = parsed.get(name)
        if info is not None and len(info.absolute_episode_numbers) == 1:
            abs_by_file[name] = info.absolute_episode_numbers[0]
    # The restart-numbering tell is a BATCH property, counting every absolute
    # of every parse supplied, seeded files included, or a v1 placed on an
    # earlier poll would hide its v2 from this leg. Deduped per parse: the
    # tell is two FILES sharing an absolute, not junk repeats within one.
    batch_absolutes = [
        number
        for info in parsed.values()
        if info is not None
        for number in dict.fromkeys(info.absolute_episode_numbers)
    ]
    # A parse the caller couldn't get (None), or the offline regex stand-in
    # for one (blind to absolutes: "S01E12 - 12" would launder its lost 12),
    # may be hiding a duplicate: the tell's input is incomplete, so the leg
    # fails CLOSED, the same posture a hiccuped leftover gets from the count.
    if (
        abs_by_file
        and not state.count_legs_barred
        and state.batch.all_parses_known
        and len(abs_by_file) == len(open_names)  # every leftover has one absolute
        and len(abs_by_file) == len(window)  # 1:1 with the leftover ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restart numbering)
    ):
        for name, _abs in sorted(abs_by_file.items(), key=lambda kv: kv[1]):
            state.place(name, [window.pop(0)], PlacementVerdict.ABSOLUTE)
        return

    if len(window) == 1:
        # Degenerate positional: one leftover episode, and a leftover file
        # Sonarr SAW and found no number in, or only a provably-bogus key
        # that exists nowhere in the series (a None parse is no evidence at
        # all, and multi-episode evidence would half-import, so both refuse).
        # The sole such file is that episode. Among several, once every parse
        # is known, the one an AniList title names is, never an extra. A tie's
        # file is neither: its number is the season's or the specials'.
        numberless = [
            name
            for name in open_names
            if (info := parsed.get(name)) is not None
            and (
                parse_has_no_number(info) or (state.map_known and numbers_miss_the_series(info, state.scope.id_by_key))
            )
            and not state.spans_multiple(info)
            and state.readings[name].tied is None
        ]
        if len(open_names) == 1 and numberless:
            state.place(numberless[0], [window[0]], PlacementVerdict.SINGLE)
            return
        if state.batch.all_parses_known:
            episodic = [name for name in numberless if not is_extras_name(name)]
            stems = [state.name_reads[name].stem.text for name in episodic]
            if (named := sole_title_match(stems, state.scope.names)) is not None:
                state.place(episodic[named], [window[0]], PlacementVerdict.TITLED)
                return

    if (
        len(open_names) > 1
        and not state.count_legs_barred
        and len(open_names) == len(window)
        and len(open_names) == len(parsed)
        and not state.verdicts
        and not state.scope.used
        and all(
            (info := parsed.get(name)) is not None
            and not info.offline
            and parse_has_no_number(info)
            and not state.spans_multiple(info)
            for name in open_names
        )
    ):
        # Pristine numberless batch: the parse-map equality proves NOTHING in
        # the batch was placed, held, or seeded (a mixed batch never zips, so
        # an extra can never fill a missing episode's slot), counts match 1:1,
        # and every parse is a real numberless one. Order is the only signal
        # left: zip name order onto airing order (the "Special 1..N" shape).
        for name, ep_id in zip(sorted(open_names, key=natural_key), window, strict=True):
            state.place(name, [ep_id], PlacementVerdict.ORDERED)


def _pass_numbered_run(state: _Placer) -> None:
    """Index a consecutive window by the one `1..N` run among the files Sonarr could not read at all.

    Unlike the ordered zip this survives a MIXED batch (a specials run beside
    a placed season pack). Blind means the reading resolved nothing, the name
    carries no `(season, episode)`, and the match spans no episodes: a file
    Sonarr placed anywhere in the series merely fell outside our scope, and a
    positional run must never re-home it.
    """

    if not state.map_known or not state.batch.all_parses_known or state.count_legs_barred:
        return
    window = state.run_window()
    if window is None:
        return
    parsed = state.batch.parsed
    blind = [
        name
        for name in state.remaining()
        if not state.readings[name].resolved
        and (info := parsed.get(name)) is not None
        and not info.offline
        and not info.episode_numbers
        and not state.spans_multiple(info)
    ]
    fits = runs_from_one(state.runs(blind), len(window.ids))
    if len(fits) != 1:
        return
    run = fits[0]
    for name, ep_id in zip(run.names, window.ids, strict=True):
        state.place(name, [ep_id], PlacementVerdict.NUMBERED_RUN)
    state.set_aside_each(run.superseded, PlacementVerdict.DUPLICATE)


def _run_refused(state: _Placer, run: NumberedRun, window: _RunWindow) -> bool:
    """Whether the batch proves the run does not own the window whole."""

    window_set = window.id_set
    for name in run.names:
        info = state.batch.parsed.get(name)
        # Sonarr matching a member to several episodes is the incoherence the
        # run overrides, and the listing's count backs the run.
        if info is None or claims_several_episodes(info):
            return True
        reading = state.readings[name]
        keyed_outside = not reading.borrowed and reading.complete and not any(i in window_set for i in reading.resolved)
        # A member named for another season of the series overrides only a
        # specials window: a torrent mislisted on a sequel entry never imports onto it.
        if keyed_outside and not window.specials:
            return True
    members_set = {numbered.name for numbered in run.whole}
    for name in state.remaining():
        if name in members_set:
            continue
        reading = state.readings[name]
        if any(i in window_set for i in reading.tied or ()):
            return True
        if not reading.complete or reading.vetoed:
            continue
        if reading.resolved and all(i in window_set for i in reading.resolved):
            return True
    return False


def _pick_refused(state: _Placer, run: NumberedRun, runs: Sequence[NumberedRun], window: _RunWindow) -> bool:
    """Whether the names put the window's files elsewhere.

    The pick's episode titles name only other episodes, or an AniList title
    names other runs of the batch (ones Sonarr did not read whole elsewhere)
    and not the pick. Either refuses, never promotes.
    """

    if state.facts.title_evidence(run, window).vetoes:
        return True
    open_runs = [candidate for candidate in runs if not state.settled_elsewhere(candidate)]
    named = best_title_matches([candidate.prefix for candidate in open_runs], state.scope.names)
    return bool(named) and (run not in open_runs or open_runs.index(run) not in named)
