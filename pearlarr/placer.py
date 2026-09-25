"""Pure placement: `assign_episode_ids` runs the ordered passes that map a batch's files onto the scope's episodes."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Self

from .parse_reading import Reading, claims_several_episodes, numbers_miss_the_series, parse_has_no_number, read_parse
from .placement_types import EpisodeAssignment, EpisodeIndex, Placement, PlacementBatch, PlacementVerdict, TargetScope
from .release_names import (
    FileIdentity,
    NameRead,
    Naming,
    NumberedRun,
    RunMember,
    TitleGround,
    folded_words,
    is_consecutive,
    is_distinct_title,
    is_extras_name,
    natural_key,
    numbered_runs,
    opens_title,
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

    The resolved set (`scope.resolved`, season order) is authoritative: a release's own numbering only ever
    indexes into it. One reading per file (`read_parse`) and one read of its name, then passes in strict
    precedence over one state, each placing into the ids the earlier ones left:

    1. **Extras:** a file named as one (`PlacementVerdict.EXTRA`) is set aside, no part of any count below.
    2. **Refuted readings:** a reading the file's own episode title contradicts is vetoed, with its run's
       readings inside the scope that no title confirms, and the count legs stand down.
    3. **Overlapping spans:** a span holding an episode another file reads differently is vetoed.
    4. **Season re-read:** a `1..N` run Sonarr matched into the one season of exactly N episodes, a few
       members shifted onto its specials, is re-read as that season's own numbering.
    5. **Release run:** the one run fitting a consecutive window (a `_Tier`) indexes it when Sonarr's reading
       of the members is incoherent (`_pass_release_run`). Members are HELD while any parse is unknown.
    6. **Exact:** a reading resolving cleanly inside the resolved set is placed there. With NO resolved set
       (`scope.unscoped`) the name's keys place against the live series map.
    7. **Episode title:** a file onto the one scope episode its name is titled as, whatever its number said.
       Every zip below refuses a pair a title contradicts.
    8. **Absolute index:** the leftovers zip onto the leftover ids by absolute number, only when every leftover
       carries one, the counts match, every parse is known, and no two files of the batch share an absolute.
    9. **Single file:** one numberless leftover onto one leftover id.
    10. **Ordered zip:** a pristine numberless batch zips natural name order onto the leftover ids.
    11. **Numbered run:** the one `1..N` run among the files Sonarr read nothing from indexes the window.
    12. **Classify:** what is left is `DUPLICATE` (inside the set onto a taken episode, a version of the file
        titled as one), `FOREIGN` (cleanly outside it, or titled as another slice's episode), or `SKIPPED`.
        The caller warns on skips and records exclusions, never guesses.

    `PlacementBatch` and `TargetScope` say what rides in. An empty series map refuses every map-dependent
    verdict. One `Placement` per distinct file, batch order.
    """

    state = _Placer.start(batch, scope)
    _pass_extras(state)
    _pass_refuse_refuted(state)
    _pass_refuse_overlaps(state)
    _pass_reread_season_runs(state)
    _pass_release_run(state)
    _pass_exact(state)
    _pass_episode_title(state)
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
    """One series episode's title, in one of its forms."""

    ep_id: int
    words: tuple[str, ...]
    """The title's folded words."""
    beyond: tuple[str, ...]
    """The words past the series title's."""
    distinct: bool
    """Whether `beyond` says enough to place a file. A run's title evidence counts every title."""
    part: bool
    """Whether the form is the head or the subtitle alone rather than the title as written."""


class _TitleForm(NamedTuple):
    """One form of an episode title: as written, or one part of it around a colon."""

    text: str
    part: bool


def _title_forms(title: str) -> tuple[_TitleForm, ...]:
    """A title as written and, around a colon, its head and its subtitle alone (a name often carries just one)."""

    head, colon, subtitle = title.rpartition(":")
    parts = (_TitleForm(head, True), _TitleForm(subtitle, True)) if colon and head.strip() and subtitle.strip() else ()
    return (_TitleForm(title, False), *parts)


def _sole(hits: Iterable[_TitledEpisode]) -> _TitledEpisode | None:
    """The one episode named by a title as written, else by a part, else None.

    Two is none: a recap repeating a title, a two-part title, a head that is another episode's subtitle.
    """

    by_rank: dict[bool, dict[int, _TitledEpisode]] = {False: {}, True: {}}
    for hit in hits:
        by_rank[hit.part].setdefault(hit.ep_id, hit)
    named = by_rank[False] or by_rank[True]
    return next(iter(named.values())) if len(named) == 1 else None


class _TitleLead(NamedTuple):
    """What the episode titles say of one run member's tail."""

    titled: _TitledEpisode | None
    """The one episode whose distinct title opens the tail, shared with other files or not."""
    leaders: frozenset[int]
    """The episodes whose title, distinct or not, leads the tail: the run's evidence."""


class _EpisodeTitles(NamedTuple):
    """The series' episode titles, as words: the distinct ones name a file, every one is a run's evidence."""

    series_words: frozenset[str]
    """The series title's words."""
    titled: tuple[_TitledEpisode, ...]
    """One per title form (an episode with a subtitle appears up to three times)."""

    @classmethod
    def of(cls, series: EpisodeIndex, ground: TitleGround) -> Self:
        """Read every episode's title forms past the series title's words."""

        titled: list[_TitledEpisode] = []
        for ep_id, ep in series.by_id.items():
            for form in _title_forms(ep.title):
                words = folded_words(form.text)
                beyond = tuple(word for word in words if word not in ground.series_words)
                distinct = is_distinct_title(beyond)
                # A part too short to place a file ("Chapter 4: Home") is no run evidence either.
                if words and (distinct or not form.part):
                    titled.append(_TitledEpisode(ep_id, words, beyond, distinct, form.part))
        return cls(ground.series_words, tuple(titled))

    def leading(self, words: tuple[str, ...]) -> _TitleLead:
        """The one episode whose distinct title opens a run member's tail, and every episode whose title leads it.

        Quality junk may follow the opening title. The led set, distinct or not, is the run's evidence.
        """

        led = [ep for ep in self.titled if words[: len(ep.words)] == ep.words]
        titled = _sole(ep for ep in led if ep.distinct and opens_title(words, ep.words))
        return _TitleLead(titled, frozenset(ep.ep_id for ep in led))

    def exactly(self, words: tuple[str, ...]) -> _TitledEpisode | None:
        """The one episode whose distinct title IS the words (a numberless name), both shed of the series title's."""

        beyond = tuple(word for word in words if word not in self.series_words)
        return _sole(ep for ep in self.titled if ep.distinct and ep.beyond == beyond)


class _TitleEvidence(NamedTuple):
    """How many of a run's members are titled as an episode inside the window, and how many as one outside it."""

    inside: int
    outside: int

    @property
    def selects(self) -> bool:
        """Enough members are titled as the window's episodes, and more than as anything else."""

        return self.inside >= _MIN_TITLE_HITS and self.inside > self.outside

    @property
    def vetoes(self) -> bool:
        """The members are titled as other episodes and as none of the window's."""

        return self.outside >= _MIN_TITLE_HITS and not self.inside


class _NameTitle(NamedTuple):
    """One name read against the episode titles."""

    episode: int | None
    """The episode the name is titled as: one episode's distinct title, carried by no other file of the torrent."""
    leaders: frozenset[int]
    """The episodes whose title leads a run member's tail, distinct or not: the run's evidence."""


class _TorrentNames(NamedTuple):
    """Every name of the torrent read once: its stem and run membership, its title, and whether it is an extra."""

    reads: Mapping[str, NameRead]
    titles: Mapping[str, _NameTitle]
    extras: frozenset[str]
    """The names that are extras: set aside, and no part of any count."""

    @classmethod
    def of(cls, batch: PlacementBatch, scope: TargetScope, readings: Mapping[str, Reading]) -> Self:
        """Read every name, seeded and gone ones included. Versions of one file share its title without dispute.

        A file Sonarr matched by its own key is an episode, whatever extras word its title carries.
        """

        ground = TitleGround.of(scope.names)
        episodes = _EpisodeTitles.of(scope.series, ground)
        reads = {name: read_name(name) for name in batch.torrent_names}
        carriers = Counter(words for _file, words in {(read.file, read.title_words) for read in reads.values()})
        titles: dict[str, _NameTitle] = {}
        extras: set[str] = set()
        for name, read in reads.items():
            words = read.title_words
            if read.member is not None:
                lead = episodes.leading(words)
            else:
                lead = _TitleLead(episodes.exactly(words), frozenset())
            titled = lead.titled
            episode = titled.ep_id if titled is not None and carriers[words] == 1 else None
            titles[name] = _NameTitle(episode, lead.leaders)
            # An extras word inside the episode title the name carries is title, not extra.
            shield = ground.entry_words | frozenset(titled.words if titled is not None else ())
            if not readings[name].keyed and is_extras_name(name, shield):
                extras.add(name)
        return cls(reads, titles, frozenset(extras))

    @property
    def counted(self) -> tuple[str, ...]:
        """Every name but the extras: the names whose readings count."""

        return tuple(name for name in self.reads if name not in self.extras)

    def episode(self, name: str) -> int | None:
        """The episode the name is titled as, else None."""

        return self.titles[name].episode

    def contradicts(self, name: str, ids: Sequence[int]) -> bool:
        """Whether placing the name on `ids` contradicts the episode it is titled as."""

        return (ep_id := self.episode(name)) is not None and ep_id not in ids

    def confirms(self, name: str, ids: Sequence[int]) -> bool:
        """Whether the episode the name is titled as is among `ids`."""

        return (ep_id := self.episode(name)) is not None and ep_id in ids

    def evidence(self, run: NumberedRun, window: _RunWindow) -> _TitleEvidence:
        """Count the members whose tail a window episode's title leads against those another's does.

        A tail led on both sides counts for neither.
        """

        inside = outside = 0
        for member in run.members:
            sides = {ep_id in window.id_set for ep_id in self.titles[member.name].leaders}
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return _TitleEvidence(inside, outside)


class _Tier(IntEnum):
    """The release-run pass's candidate tiers, best first."""

    WHOLE = 1
    """Numbered `1..N` for the window's width N."""
    NUMBERED = 2
    """Numbered exactly as the window's episodes."""
    COVERING = 3
    """Numbered `1..N` for the whole season a slice window belongs to."""
    ANYWHERE = 4
    """N consecutive numbers from anywhere, no member read completely, nothing seeded."""


class _Candidate(NamedTuple):
    """One run that could index the window: its position among the runs, the first tier it fits, its title evidence."""

    position: int
    run: NumberedRun
    tier: _Tier
    evidence: _TitleEvidence


class _SeasonFit(NamedTuple):
    """The one season a `1..N` run counts as, and whether the specials count it too."""

    season: int
    tied: bool


@dataclass(frozen=True, slots=True)
class _SeriesFacts:
    """What the series map says on its own, read once per call and never changed by a pass."""

    key_by_id: Mapping[int, EpisodeKey]
    """The series map inverted."""
    season_counts: Mapping[int, int]
    """Episodes per season."""
    absolute_of: Mapping[int, int]
    """Episode id -> the series' absolute number, for the ids that carry one."""

    @classmethod
    def of(cls, series: EpisodeIndex) -> Self:
        """Read once from the series index."""

        episodes = series.by_id
        return cls(
            key_by_id={ep_id: key for key, ep_id in series.id_by_key.items()},
            season_counts=Counter(key.season for key in series.id_by_key),
            absolute_of={
                ep_id: ep.absolute_episode_number
                for ep_id, ep in episodes.items()
                if ep.absolute_episode_number is not None
            },
        )


class _DuplicateProof(NamedTuple):
    """What proves a leftover a duplicate, gathered once before the leftovers are classified."""

    proven: frozenset[int]
    """The ids whose holder reads or is titled there: placed this batch, or a seeded or gone name's own evidence."""
    held_by_file: Mapping[FileIdentity, frozenset[int]]
    """The episodes each file's versions hold: placed this batch, or seeded (gone) and titled as one."""


@dataclass(slots=True)
class _Placer:
    """One `assign_episode_ids` call's state: the readings, the verdicts so far, and the ids they used."""

    batch: PlacementBatch
    scope: TargetScope
    readings: dict[str, Reading]
    """One reading per name of the torrent, seeded and gone names included: their evidence counts, though
    only `batch.names` are placed."""
    torrent: _TorrentNames
    """Every name of the torrent read once, seeded and gone names included."""
    facts: _SeriesFacts
    verdicts: dict[str, Placement] = field(default_factory=dict[str, Placement])
    used: set[int] = field(default_factory=set[int])
    count_legs_barred: bool = False
    """The release-run pass found several runs or refused its pick, or a title refuted a reading, placed a file
    its number did not, or contradicted a zip pair: no count leg (the numbered run, the absolute and ordered
    zips) places what it would not."""

    @classmethod
    def start(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """Read every name and every parse of the torrent once."""

        readings = {name: read_parse(batch.parsed.get(name), scope) for name in batch.torrent_names}
        torrent = _TorrentNames.of(batch, scope, readings)
        return cls(batch, scope, readings, torrent, _SeriesFacts.of(scope.series), used=set(scope.used))

    def single_parse(self, name: str) -> ParsedFileInfo | None:
        """The name's parse, when Sonarr served one and it spans one episode (several placed as one half-import)."""

        info = self.batch.parsed.get(name)
        return None if info is None or self.spans_multiple(info) else info

    def title_placement(self, name: str) -> int | None:
        """The open episode of the scope the name is titled as, else None (a span, an unread parse, a tie's file)."""

        ep_id = self.torrent.episode(name)
        if ep_id is None or ep_id in self.used or not self.scope.admits(ep_id) or self.single_parse(name) is None:
            return None
        tied = self.readings[name].tied
        return None if tied is not None and ep_id not in tied else ep_id

    def titled_outside(self, name: str, nothing_left: bool) -> bool:
        """Whether the name is titled as another slice's episode, and Sonarr read it outside too or nothing is left.

        A title alone never excludes a file the entry may still need.
        """

        ep_id = self.torrent.episode(name)
        if ep_id is None or self.scope.admits(ep_id):
            return False
        reading = self.readings[name]
        return (reading.complete and not reading.inside) or nothing_left

    def reads_no_number(self, name: str) -> bool:
        """Whether Sonarr saw the file and found no episode in it: no number, or only a key the whole series lacks.

        Never a tie's file (the season's or the specials'), or a titled one (that episode is not here).
        """

        info = self.single_parse(name)
        if info is None:
            return False
        bogus = self.map_known and numbers_miss_the_series(info, self.scope.id_by_key)
        if not (parse_has_no_number(info) or bogus):
            return False
        return self.readings[name].tied is None and self.torrent.episode(name) is None

    @property
    def map_known(self) -> bool:
        """Whether the series map was served (every map-dependent verdict refuses on an empty one)."""

        return bool(self.scope.id_by_key)

    def season_fit(self, run: NumberedRun) -> _SeasonFit | None:
        """The season a keyless `1..N` run counts as: Sonarr read it into that one season of exactly N episodes.

        Some members read onto the specials instead (never more than into
        the season), and the fit is tied when as many did and the specials
        count N too. None when the run is not that shape.
        """

        width = len(run.members)
        if run.numbers != tuple(range(1, width + 1)):
            return None
        infos = [self.batch.parsed.get(name) for name in run.names]
        if any(info is None or info.episode_numbers for info in infos):
            return None
        readings = [self.readings[name] for name in run.names]
        if any(len(r.resolved) > 1 for r in readings):
            return None
        read = [self.facts.key_by_id[r.resolved[0]].season for r in readings if r.resolved]
        seasons = {season for season in read if season != 0}
        if len(seasons) != 1:
            return None
        season = seasons.pop()
        if self.facts.season_counts.get(season) != width:
            return None
        into_season = read.count(season)
        onto_specials = read.count(0)
        if not onto_specials or into_season < onto_specials:
            return None
        return _SeasonFit(season, into_season == onto_specials and self.facts.season_counts.get(0) == width)

    def tie(self, run: NumberedRun, season: int) -> None:
        """Veto the run's reads and record the episodes each name may be under either numbering."""

        for numbered in run.whole:
            keys = (EpisodeKey(season, numbered.number), EpisodeKey(0, numbered.number))
            ids = tuple(i for key in keys if (i := self.scope.id_by_key.get(key)))
            self.readings[numbered.name] = self.readings[numbered.name]._replace(vetoed=True, tied=ids)

    def reread(self, run: NumberedRun, season: int) -> None:
        """Read every name of the run as the season's episode of its number, vetoed where its title says otherwise."""

        for numbered in run.whole:
            ep_id = self.scope.id_by_key.get(EpisodeKey(season, numbered.number))
            if ep_id:
                inside = (ep_id,) if self.scope.admits(ep_id) else ()
                vetoed = self.torrent.contradicts(numbered.name, (ep_id,))
                self.readings[numbered.name] = Reading(
                    (ep_id,), inside, complete=True, borrowed=True, vetoed=vetoed, corroborated=True
                )

    def veto(self, run: NumberedRun) -> None:
        """Veto every reading of the run, its displaced versions included."""

        self.veto_names(numbered.name for numbered in run.whole)

    def veto_names(self, names: Iterable[str]) -> None:
        """Veto each name's reading: its claims stay real but never place on their own."""

        for name in names:
            self.readings[name] = self.readings[name]._replace(vetoed=True)

    def open_ids(self) -> list[int]:
        """The leftover ids: the scope's resolved set minus every id used so far, scope order."""

        return [i for i in self.scope.resolved if i and i not in self.used]

    def run_window(self) -> _RunWindow | None:
        """The open ids as consecutive episodes in airing order, else None.

        One season's episode numbers, else the series' absolutes when every slot carries one (a special
        interleaved, two seasons' cours). The ids are re-sorted by their keys, never trusted in scope order.
        """

        open_ids = self.open_ids()
        if len(open_ids) < 2:
            return None
        key_by_id = self.facts.key_by_id
        keyed = sorted((key_by_id[ep_id], ep_id) for ep_id in open_ids if ep_id in key_by_id)
        if len(keyed) != len(open_ids):
            return None
        seasons = {key.season for key, _ in keyed}
        episodes = [key.episode for key, _ in keyed]
        if len(seasons) == 1 and is_consecutive(episodes):
            return _RunWindow(seasons.pop(), tuple(ep_id for _, ep_id in keyed), tuple(episodes))
        absolute_of = self.facts.absolute_of
        absolute = sorted((absolute_of[ep_id], ep_id) for ep_id in open_ids if ep_id in absolute_of)
        if len(absolute) != len(open_ids) or not is_consecutive([number for number, _ in absolute]):
            return None
        return _RunWindow(None, tuple(ep_id for _, ep_id in absolute), tuple(number for number, _ in absolute))

    def remaining(self) -> list[str]:
        """The names without a verdict yet, batch order."""

        return [name for name in self.batch.names if name not in self.verdicts]

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

    def place_zip(self, names: Sequence[str], ids: Sequence[int], verdict: PlacementVerdict) -> bool:
        """Place the names onto the ids pairwise, or none when a title says a pair is wrong (so the count was).

        A refused zip bars every count leg.
        """

        pairs = list(zip(names, ids, strict=True))
        if any(self.torrent.contradicts(name, (ep_id,)) for name, ep_id in pairs):
            self.count_legs_barred = True
            return False
        for name, ep_id in pairs:
            self.place(name, (ep_id,), verdict)
        return True

    def set_aside(self, name: str, verdict: PlacementVerdict) -> None:
        """Record a verdict that carries no ids (held, excluded, or skipped)."""

        self.verdicts[name] = Placement(name, (), verdict)

    def set_aside_each(self, members: Iterable[RunMember], verdict: PlacementVerdict) -> None:
        """Record one id-less verdict for each run member."""

        for member in members:
            self.set_aside(member.name, verdict)

    def runs(self, names: Iterable[str]) -> list[NumberedRun]:
        """The numbered runs among `names`, by their reads."""

        members = (member for name in names if (member := self.torrent.reads[name].member) is not None)
        return numbered_runs(members, self.batch.parsed)

    def runs_from_anywhere(self, runs: Iterable[NumberedRun], width: int) -> list[NumberedRun]:
        """The runs of `width` consecutive numbers wherever they start, no member of which Sonarr read completely.

        None once a seed owns part of the scope: such a run fits what is left by chance, never by count.
        """

        if self.used:
            return []
        return [run for run in runs if run.consecutive(width) and not any(self.readings[n].complete for n in run.names)]

    def release_candidates(self, runs: Sequence[NumberedRun], window: _RunWindow) -> list[_Candidate]:
        """Every run that could index the window, at the first tier it fits, tier order then run order."""

        width = len(window.ids)
        tiers = {
            _Tier.WHOLE: runs_from_one(runs, width),
            _Tier.NUMBERED: runs_with_numbers(runs, window.numbers),
            _Tier.COVERING: self.covering(runs, window),
            _Tier.ANYWHERE: self.runs_from_anywhere(runs, width),
        }
        tier_of: dict[NumberedRun, _Tier] = {}
        for tier, fits in tiers.items():
            for run in fits:
                tier_of.setdefault(run, tier)
        position = {run: index for index, run in enumerate(runs)}
        return [
            _Candidate(position[run], run, tier, self.torrent.evidence(run, window)) for run, tier in tier_of.items()
        ]

    def duplicate_proof(self) -> _DuplicateProof:
        """What proves a leftover a duplicate. A seed that landed a file positionally proves nothing."""

        placing = frozenset(self.batch.names)
        evidenced: set[int] = set()
        held: defaultdict[FileIdentity, set[int]] = defaultdict(set)
        for name, read in self.torrent.reads.items():
            if (placed := self.verdicts.get(name)) is not None:
                held[read.file].update(placed.ids)
            elif name not in placing:
                evidenced.update(self.readings[name].resolved)
                if (titled := self.torrent.episode(name)) is not None:
                    evidenced.add(titled)
                    held[read.file].add(titled)
        proven = (self.used - self.scope.used) | evidenced
        return _DuplicateProof(frozenset(proven), {file: frozenset(ids) for file, ids in held.items()})

    def finish(self) -> EpisodeAssignment:
        """Classify what is still open, then fold the verdicts in batch order."""

        proof = self.duplicate_proof()
        nothing_left = not self.open_ids()
        for name in self.remaining():
            reading = self.readings[name]
            titled = self.torrent.episode(name)
            if reading.complete and not reading.vetoed and reading.inside == reading.resolved:
                # Inside the set, so the exact pass left it because its episodes are taken. A span only partly
                # taken still holds an episode nothing has, so it stays loud.
                proven = all(ep_id in self.used and ep_id in proof.proven for ep_id in reading.inside)
                self.set_aside(name, PlacementVerdict.DUPLICATE if proven else PlacementVerdict.SKIPPED)
            elif titled is not None and titled in self.used:
                # Titled as a taken episode: a duplicate only when a version of the same file holds it.
                twin = titled in proof.held_by_file.get(self.torrent.reads[name].file, ())
                self.set_aside(name, PlacementVerdict.DUPLICATE if twin else PlacementVerdict.SKIPPED)
            elif self.map_known and (reading.outside or self.titled_outside(name, nothing_left)):
                self.set_aside(name, PlacementVerdict.FOREIGN)
            else:
                self.set_aside(name, PlacementVerdict.SKIPPED)
        return EpisodeAssignment(tuple(self.verdicts[name] for name in self.batch.names))


def _pass_extras(state: _Placer) -> None:
    """Set the extras aside (`PlacementVerdict.EXTRA`): never an episode, and no part of any count."""

    for name in state.remaining():
        if name in state.torrent.extras:
            state.set_aside(name, PlacementVerdict.EXTRA)


def _pass_refuse_refuted(state: _Placer) -> None:
    """Veto the readings a title refutes and, inside the scope, their runs' unconfirmed readings. Bar the count legs.

    The run's count, its own or Sonarr's, was wrong about the titled name, so about every member it numbered by
    the same shifted count, seeded members included: an import poll judges the leftover as the grab did. A member
    its own title confirms keeps its reading.
    """

    readings, torrent = state.readings, state.torrent
    names = torrent.counted
    refuted = {name for name in names if (ids := readings[name].resolved) and torrent.contradicts(name, ids)}
    for run in state.runs(names):
        members = {member.name for member in run.whole}
        if refuted & members:
            refuted |= {n for n in members if readings[n].inside and not torrent.confirms(n, readings[n].resolved)}
    state.veto_names(refuted)
    state.count_legs_barred |= bool(refuted)


def _pass_refuse_overlaps(state: _Placer) -> None:
    """Veto every open span holding an episode another name of the torrent reads differently.

    One of the two is wrong, and a span placed wrong half-imports. Seeded names count, so an import poll
    refuses the span the grab did.
    """

    spans_by_ep: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    for name in state.torrent.counted:
        if not (reading := state.readings[name]).vetoed:
            for ep_id in reading.resolved:
                spans_by_ep[ep_id].add(reading.resolved)
    overlapping = [
        name
        for name in state.remaining()
        if len(ids := state.readings[name].resolved) > 1 and any(len(spans_by_ep[ep_id]) > 1 for ep_id in ids)
    ]
    state.veto_names(overlapping)


def _pass_reread_season_runs(state: _Placer) -> None:
    """Re-read a `1..N` run Sonarr matched into one season of exactly N episodes as that season's own numbering.

    TVDB interleaves specials into the absolute numbering, so Sonarr's match of a season-only release drifts
    onto a special after each one, and a release carrying the special would number N + 1. Only a run without
    keys of its own is re-read. As many onto the specials when they count N too is a tie: the reads are
    vetoed and the episodes either numbering gives are `tied`, so only the run's count over a window places it.
    """

    if not state.map_known:
        return
    for run in state.runs(state.remaining()):
        fit = state.season_fit(run)
        if fit is None:
            continue
        if fit.tied:
            state.tie(run, fit.season)
        else:
            state.reread(run, fit.season)


def _pass_release_run(state: _Placer) -> None:
    """Judge the release's own numbering against Sonarr's reading of its members.

    Stands down unless the window is consecutive episodes and one run is left to index it. Candidates are
    the runs fitting a `_Tier`. A disputed covering run Sonarr did not read whole into one other season is
    vetoed before any pick. Tie-breaks, in order: a run Sonarr read whole elsewhere stands aside, then the
    episode titles the members carry name one, then an AniList title does, then the highest tier holds.
    Several left is none, and the other count legs stand down too. With any parse unknown the members are
    HELD. A refusal bars the count legs and lets the exact pass proceed. A coherent reading (every member
    one distinct id inside the window) stands. Otherwise the run indexes the window, and a whole-season
    run's members past a slice window are the other slice's.
    """

    if not state.map_known:
        return
    window = state.run_window()
    if window is None:
        return
    runs = state.runs(state.remaining())
    candidates = state.release_candidates(runs, window)
    # Reads into another season dispute a covering run's count, pick or not: read whole into one other
    # season, it is that season's (its members foreign), else its files are nowhere.
    disputed = [c.run for c in candidates if c.tier is _Tier.COVERING and state.read_elsewhere(c.run, window.season)]
    for run in disputed:
        if not (state.settled_elsewhere(run) and len(state.seasons_read(run)) == 1):
            state.veto(run)
    open_positions = frozenset(index for index, run in enumerate(runs) if not state.settled_elsewhere(run))
    naming = Naming.of([run.prefix for run in runs], state.scope.names)
    state.count_legs_barred |= len(candidates) > 1
    pick = _sole_candidate(candidates, open_positions, naming)
    if pick is None:
        return
    run = pick.run
    if not state.batch.all_parses_known:
        state.set_aside_each(run.whole, PlacementVerdict.HELD)
        return
    covers = pick.tier is _Tier.COVERING
    refused = (
        _run_refused(state, run, window)
        or (covers and state.covering_refused(window))
        or run in disputed
        or pick.evidence.vetoes
        or (naming is not None and naming.names_another(pick.position, open_positions))
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
    if not state.place_zip([member.name for member in members], window.ids, PlacementVerdict.RELEASE_RUN):
        return
    for member in run.members:
        if member not in members:
            state.set_aside(member.name, PlacementVerdict.FOREIGN)
    state.set_aside_each(run.superseded, PlacementVerdict.DUPLICATE)


def _sole_candidate(
    candidates: Sequence[_Candidate], open_positions: frozenset[int], naming: Naming | None
) -> _Candidate | None:
    """The one candidate the tie-breaks leave, else None (see `_pass_release_run`)."""

    if len(candidates) > 1:
        candidates = [c for c in candidates if c.position in open_positions] or candidates
    if len(candidates) > 1 and len(selected := [c for c in candidates if c.evidence.selects]) == 1:
        candidates = selected
    if (
        len(candidates) > 1
        and naming is not None
        and (named := naming.sole_among(c.position for c in candidates)) is not None
    ):
        candidates = [c for c in candidates if c.position == named]
    if len(candidates) > 1:
        top = min(c.tier for c in candidates)
        candidates = [c for c in candidates if c.tier is top]
    return candidates[0] if len(candidates) == 1 else None


def _pass_exact(state: _Placer) -> None:
    """Place every open file whose reading resolves cleanly inside the scope onto unused ids.

    Two files resolving to one episode are judged by `Reading.rank`, batch
    order within a rank: a "17.5 (S00E01)" beats the "- 17" whose match
    Sonarr shifted onto the same special, while a "- 08" beats a "- Bonus"
    whose CRC tag parsed as its key. Within that, a "- 09v2" beats its
    "- 09". The loser is left over (a duplicate).
    """

    for name in sorted(state.remaining(), key=lambda n: (state.readings[n].rank, -state.torrent.reads[n].version)):
        reading = state.readings[name]
        if not reading.complete or reading.vetoed or reading.inside != reading.resolved:
            continue
        if any(i in state.used for i in reading.inside):
            continue
        state.place(name, reading.inside, PlacementVerdict.EXACT)


def _pass_episode_title(state: _Placer) -> None:
    """Place every open file onto the one open episode of the scope its name is titled as, the highest version first.

    Positive evidence per file, so it runs past the count legs' bar and bars them itself: a file its number
    placed wrong (a shifted absolute match, a bogus key) says the batch's count is off.
    """

    for name in sorted(state.remaining(), key=lambda n: state.torrent.reads[n].version, reverse=True):
        if (ep_id := state.title_placement(name)) is not None:
            state.place(name, (ep_id,), PlacementVerdict.EPISODE_TITLE)
            state.count_legs_barred = True


def _pass_counted(state: _Placer) -> None:
    """The count legs over the open files and the window: absolute zip, else single file, else ordered zip."""

    # A file reading wholly outside the scope is another slice's: it neither
    # takes a leftover id nor blocks the count for the files that could.
    open_names = [name for name in state.remaining() if not state.readings[name].outside]
    open_ids = state.open_ids()
    if not open_names or not open_ids:
        return
    if (order := _absolute_order(state, open_names, open_ids)) is not None:
        state.place_zip(order, open_ids, PlacementVerdict.ABSOLUTE)
    elif len(open_ids) == 1:
        _place_single(state, open_names, open_ids[0])
    elif (order := _natural_order(state, open_names, open_ids)) is not None:
        state.place_zip(order, open_ids, PlacementVerdict.ORDERED)


def _absolute_order(state: _Placer, open_names: Sequence[str], open_ids: Sequence[int]) -> list[str] | None:
    """The open files by absolute number when each carries one, 1:1 with the window, and no tell refuses, else None."""

    parsed = state.batch.parsed
    abs_by_file: dict[str, int] = {}
    for name in open_names:
        info = parsed.get(name)
        if info is not None and len(info.absolute_episode_numbers) == 1:
            abs_by_file[name] = info.absolute_episode_numbers[0]
    # The restart-numbering tell counts every absolute of every parse supplied, seeded files included (a v1
    # placed on an earlier poll would hide its v2), deduped per parse: two FILES sharing one, not junk repeats.
    batch_absolutes = [
        number
        for name in state.torrent.counted
        if (parse := parsed.get(name)) is not None
        for number in dict.fromkeys(parse.absolute_episode_numbers)
    ]
    # An unknown parse (None, or the offline stand-in blind to absolutes) may hide a duplicate: the tell's
    # input is incomplete, so the leg fails CLOSED.
    applies = (
        bool(abs_by_file)
        and not state.count_legs_barred
        and state.batch.all_parses_known
        and len(abs_by_file) == len(open_names)  # every leftover has one absolute
        and len(abs_by_file) == len(open_ids)  # 1:1 with the leftover ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restart numbering)
    )
    return sorted(abs_by_file, key=abs_by_file.__getitem__) if applies else None


def _place_single(state: _Placer, open_names: Sequence[str], ep_id: int) -> None:
    """One leftover episode: the sole numberless file is it, else among several the one an AniList title names."""

    unnumbered = [name for name in open_names if state.reads_no_number(name)]
    if len(open_names) == 1 and unnumbered:
        state.place(unnumbered[0], (ep_id,), PlacementVerdict.SINGLE)
    elif state.batch.all_parses_known:
        stems = [state.torrent.reads[name].stem.text for name in unnumbered]
        if (pick := sole_title_match(stems, state.scope.names)) is not None:
            state.place(unnumbered[pick], (ep_id,), PlacementVerdict.TITLED)


def _natural_order(state: _Placer, open_names: Sequence[str], open_ids: Sequence[int]) -> list[str] | None:
    """A pristine numberless batch in natural name order (the "Special 1..N" shape), else None."""

    parsed = state.batch.parsed
    # Pristine: the parse map holds the open files and the extras and nothing else, and nothing but an extra
    # has a verdict, so nothing was placed, held, or seeded (an unread extra could fill a missing episode's slot).
    pristine = (
        not state.count_legs_barred
        and len(open_names) == len(open_ids)
        and set(parsed) == {*open_names, *state.torrent.extras}
        and state.verdicts.keys() <= state.torrent.extras
        and not state.scope.used
        and all(
            (info := parsed.get(name)) is not None
            and not info.offline
            and parse_has_no_number(info)
            and not state.spans_multiple(info)
            for name in open_names
        )
    )
    return sorted(open_names, key=natural_key) if pristine else None


def _pass_numbered_run(state: _Placer) -> None:
    """Index a consecutive window by the one `1..N` run among the files Sonarr could not read at all.

    Unlike the ordered zip this survives a mixed batch. Blind means the reading resolved nothing, the name
    carries no key, and the match spans no episodes: a file Sonarr placed elsewhere in the series is never
    re-homed by position.
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
    if state.place_zip(run.names, window.ids, PlacementVerdict.NUMBERED_RUN):
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
