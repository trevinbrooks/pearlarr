"""Place a torrent's files onto an entry's Sonarr episodes: `assign_episode_ids` and the passes it runs. No I/O.

Our episode ids for the entry are authoritative. Sonarr's parse of a file name, and the number the name
carries, only ever choose among them.

Terms used across the placement modules:

- Own key: the `SxxExx` a file name carries. Matched pair: an episode Sonarr's series match points at.
  A name without an own key borrows Sonarr's matched pairs.
- Scope (`TargetScope`): the entry's episode ids a file may be placed on, plus the whole series' episode
  map every key is looked up in. An empty id list means no scope ("unscoped"): a file's own keys then
  place against the whole series.
- Window (`window_placement`): the scope of one entry a torrent is listed on. A torrent listed on several
  entries is placed under each window in turn.
- Open ids: the scope's ids that no file holds yet.
- Seeded file: a file of the torrent that the grab-time placement or an earlier import poll already mapped.
  Its ids start out used, as do ids an earlier window placed. Seeded files, and files since moved off disk
  ("gone"), are never placed again, but their parses and names still count as evidence.
- Alias match: Sonarr moved a name's episode numbers, unchanged, into one regular season (a sequel as season 1).
- Reading (`Reading`): the episode ids a file's parse resolves to in our series map, with flags saying how
  far to trust them. A rejected reading (`Reading.vetoed`) is kept as evidence, but no pass places the file
  by it.
- Complete reading: every episode the parse names resolved in our series map (a partial span is never placed).
- Trusted reading: a complete reading that was not rejected.
- Unknown parse: a missing parse, or the offline `SxxExx` stand-in. `PlacementBatch.all_parses_known` is False
  while any file of the torrent has one.
- Numberless: the file name carries no episode or absolute number (a key the series lacks may count as none).
- Release number: the episode number read from the file name's text alone ("Show - 07"), without Sonarr.
- Numbered run (`NumberedRun`): the files whose names share the text before their release number, in number
  order. Of two versions of one number, the higher `vN` is the member and the lower is superseded.
- Tail: the words of a run member's name after its release number.
- Run window (`_RunWindow`): the open ids, when they are consecutive episodes, in airing order.
- Covering run: a run numbered `1..N` for a whole season when the run window holds only part of that season.
- Tied file: a run file the season re-read (pass 4) fits to a season and the specials equally well (`Reading.tied`).
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
  (`_Placer.count_legs_barred`).
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
   other members are rejected too, except where a member's own title confirms its episode. The count-based
   passes are turned off.
3. Overlapping spans: a multi-episode reading that shares an episode with a different reading of another
   file is rejected.
4. Season re-read: a run numbered `1..N` without own keys, which Sonarr matched into one season of exactly
   N episodes with a few members shifted onto specials, is read as that season's episodes 1 to N. When the
   specials season fits just as well, it is a tie: the readings are rejected and both candidates recorded.
5. Release run: when the open ids form a run window and the tie-breaks leave one numbered run that fits it
   (a `_Tier`), the run's own numbers place its members onto the window, unless Sonarr's reading already
   puts each member on its own window episode. While any parse is unknown the members are `HELD` instead.
   `_pass_release_run` lists the tie-breaks and the refusals.
6. Exact: a file whose complete, unrejected reading lies wholly inside the scope, on open ids, is placed
   there. Unscoped, a name's own keys place against the whole series.
7. Episode title: a file titled as an open scope episode, whose parse covers one episode, is placed there,
   whatever its number said. Each such placement turns the count-based passes off.
8. Absolute zip: the remaining files pair with the open ids in absolute-number order, only when every file
   carries exactly one absolute number, the counts match, every parse is known, and no two files of the
   torrent share an absolute number.
9. Single file: with one open id left, the only remaining file takes it if its parse has no usable episode
   number. Otherwise, the numberless file an AniList title names takes it (`TITLED`).
10. Ordered zip: when every file of the torrent but the extras is unplaced, numberless, and not read
    outside the scope, no id was used from the start (by a seeded file or an earlier window), and the counts
    match, the files pair with the open ids in natural name order.
11. Numbered run: among the files Sonarr resolved no episode for, the one run numbered `1..N` places onto
    a run window of N ids.
12. Classify: every file still unplaced becomes `DUPLICATE` (it reads inside the scope onto episodes other
    files provably hold, or a version of the same file holds its titled episode), `FOREIGN` (its reading lies
    wholly outside the scope, or it is titled as an episode outside it and confirmed by its reading or by no
    open id being left), or `SKIPPED`. The caller warns about skipped files and records the exclusions. It
    never guesses.

At most one of passes 8 to 10 is tried: the absolute zip when its conditions hold, else the single-file
pass when one id is open, else the ordered zip. They consider only open files. Every zip (passes 5, 8,
10, and 11) places nothing when a file is titled as an episode other than its pair, and that turns the
count-based passes off.
"""

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
    """Place the files of `batch` onto the ids of `scope` by running the passes the module docstring lists, in order.

    One file name never overrides the scope. With an empty series map, every verdict that needs the map is
    refused. The result holds one `Placement` per distinct file, in batch order.
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


# Episode titles pick a run, or refuse one, only when at least this many members carry them.
_MIN_TITLE_HITS = 2


@dataclass(frozen=True, slots=True)
class _RunWindow:
    """The open ids as consecutive episodes, in airing order: the run window."""

    season: int | None
    """The season every id belongs to, or None when the ids span seasons and absolute numbers order them."""
    ids: tuple[int, ...]
    """The open ids, in airing order."""
    numbers: tuple[int, ...]
    """The ids' episode numbers (absolute numbers when `season` is None), in the same order."""
    id_set: frozenset[int] = field(init=False, repr=False, compare=False)
    """The ids as a set, built once for membership checks."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id_set", frozenset(self.ids))

    @property
    def specials(self) -> bool:
        """Whether the window lies in the specials season (season 0)."""

        return self.season == 0

    def slice_of(self, run: NumberedRun) -> tuple[RunMember, ...]:
        """The run members whose numbers are the window's episode numbers (a whole-season run over part of it)."""

        return tuple(member for member in run.members if member.number in self.numbers)


class _TitledEpisode(NamedTuple):
    """One series episode's title, in one of its forms."""

    ep_id: int
    words: tuple[str, ...]
    """The title's folded words."""
    beyond: tuple[str, ...]
    """The title's words that are not words of the series title."""
    distinct: bool
    """Whether `beyond` is specific enough to place a file by. Run evidence counts every title, specific or not."""
    part: bool
    """Whether this form is only the part before or after a colon, not the whole title."""


class _TitleForm(NamedTuple):
    """One form of an episode title: the whole title, or the part before or after its last colon."""

    text: str
    part: bool


def _title_forms(title: str) -> tuple[_TitleForm, ...]:
    """The whole title, plus the parts before and after its last colon when both have text.

    A name often carries only one part.
    """

    head, colon, subtitle = title.rpartition(":")
    parts = (_TitleForm(head, True), _TitleForm(subtitle, True)) if colon and head.strip() and subtitle.strip() else ()
    return (_TitleForm(title, False), *parts)


def _sole(hits: Iterable[_TitledEpisode]) -> _TitledEpisode | None:
    """The one episode a whole title matches. When no whole title matches, the one a title part matches. Else None.

    Two matches count as none: a recap repeating a title, a two-part title, or a title's text before the colon
    that is another episode's subtitle.
    """

    by_rank: dict[bool, dict[int, _TitledEpisode]] = {False: {}, True: {}}
    for hit in hits:
        by_rank[hit.part].setdefault(hit.ep_id, hit)
    named = by_rank[False] or by_rank[True]
    return next(iter(named.values())) if len(named) == 1 else None


class _TitleLead(NamedTuple):
    """What the episode titles say about one run member's tail (the words after its release number)."""

    titled: _TitledEpisode | None
    """The one episode whose specific title starts the tail and ends it or meets a quality tag, whether or not other
    files carry it."""
    leaders: frozenset[int]
    """Every episode whose title, specific or not, starts the tail: the evidence a run is judged by."""


class _EpisodeTitles(NamedTuple):
    """The series' episode titles as words: specific titles can place a file, and every title is run evidence."""

    series_words: frozenset[str]
    """The series title's words."""
    titled: tuple[_TitledEpisode, ...]
    """One per title form (an episode with a subtitle appears up to three times)."""

    @classmethod
    def of(cls, series: EpisodeIndex, ground: TitleGround) -> Self:
        """Fold every form of every episode title into words, noting the words beyond the series title."""

        titled: list[_TitledEpisode] = []
        for ep_id, ep in series.by_id.items():
            for form in _title_forms(ep.title):
                words = folded_words(form.text)
                beyond = tuple(word for word in words if word not in ground.series_words)
                distinct = is_distinct_title(beyond)
                # A title part too short to place a file ("Chapter 4: Home") is not run evidence either.
                if words and (distinct or not form.part):
                    titled.append(_TitledEpisode(ep_id, words, beyond, distinct, form.part))
        return cls(ground.series_words, tuple(titled))

    def leading(self, words: tuple[str, ...]) -> _TitleLead:
        """Match a run member's tail: the one specific title that starts it, and every title that starts it.

        Quality tags may follow the specific title. The full set of matches, specific or not, is run evidence.
        """

        led = [ep for ep in self.titled if words[: len(ep.words)] == ep.words]
        titled = _sole(ep for ep in led if ep.distinct and opens_title(words, ep.words))
        return _TitleLead(titled, frozenset(ep.ep_id for ep in led))

    def exactly(self, words: tuple[str, ...]) -> _TitledEpisode | None:
        """The one episode whose specific title equals a name's words, with series title words dropped from both."""

        beyond = tuple(word for word in words if word not in self.series_words)
        return _sole(ep for ep in self.titled if ep.distinct and ep.beyond == beyond)


class _TitleEvidence(NamedTuple):
    """How many run members' tails start with only window titles, and how many with only titles outside it."""

    inside: int
    outside: int

    @property
    def selects(self) -> bool:
        """Whether at least `_MIN_TITLE_HITS` members carry window titles, and more than carry outside titles."""

        return self.inside >= _MIN_TITLE_HITS and self.inside > self.outside

    @property
    def vetoes(self) -> bool:
        """Whether at least `_MIN_TITLE_HITS` members carry outside titles and none carries a window title."""

        return self.outside >= _MIN_TITLE_HITS and not self.inside


class _NameTitle(NamedTuple):
    """One name read against the episode titles."""

    episode: int | None
    """The episode the name is titled as: its specific title, carried by no other file of the torrent."""
    leaders: frozenset[int]
    """For a run member, every episode whose title starts its tail (run evidence). Empty for other names."""


class _TorrentNames(NamedTuple):
    """Every file name of the torrent, read once: stem, run membership, episode title, and whether it is an extra."""

    reads: Mapping[str, NameRead]
    titles: Mapping[str, _NameTitle]
    extras: frozenset[str]
    """The extras: set aside, and left out of every count."""

    @classmethod
    def of(cls, batch: PlacementBatch, scope: TargetScope, readings: Mapping[str, Reading]) -> Self:
        """Read every name, seeded and gone ones included.

        Two versions of one file carrying the same title still count as titled as it. A file whose own key Sonarr
        matched (`Reading.keyed`) is an episode, whatever extras word its name carries.
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
            # An extras word that belongs to the entry's titles or the name's episode title marks no extra.
            shield = ground.entry_words | frozenset(titled.words if titled is not None else ())
            if not readings[name].keyed and is_extras_name(name, shield):
                extras.add(name)
        return cls(reads, titles, frozenset(extras))

    @property
    def counted(self) -> tuple[str, ...]:
        """Every name except the extras: the names the evidence checks read."""

        return tuple(name for name in self.reads if name not in self.extras)

    def episode(self, name: str) -> int | None:
        """The episode the name is titled as, else None."""

        return self.titles[name].episode

    def contradicts(self, name: str, ids: Sequence[int]) -> bool:
        """Whether the name is titled as an episode that is not among `ids`."""

        return (ep_id := self.episode(name)) is not None and ep_id not in ids

    def confirms(self, name: str, ids: Sequence[int]) -> bool:
        """Whether the episode the name is titled as is among `ids`."""

        return (ep_id := self.episode(name)) is not None and ep_id in ids

    def evidence(self, run: NumberedRun, window: _RunWindow) -> _TitleEvidence:
        """Count the members whose tail starts with only window titles, and those with only outside titles.

        A tail matching titles on both sides counts for neither.
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
    """How well a numbered run fits the run window, best first. The release-run pass ranks candidates by it."""

    WHOLE = 1
    """Numbered `1..N`, where N is the window's width."""
    NUMBERED = 2
    """Numbered exactly as the window's episode numbers."""
    COVERING = 3
    """Numbered `1..N` for the whole season, when the window holds only part of that season."""
    ANYWHERE = 4
    """Any N consecutive numbers, when no member has a complete reading and no id is used yet (by a seeded file, an
    earlier window, or an earlier pass)."""


class _Candidate(NamedTuple):
    """A run that could place onto the window: its position among the runs, its best tier, and its title evidence."""

    position: int
    run: NumberedRun
    tier: _Tier
    evidence: _TitleEvidence


class _SeasonFit(NamedTuple):
    """The season a `1..N` run matches, and whether the specials season matches it just as well (a tie)."""

    season: int
    tied: bool


@dataclass(frozen=True, slots=True)
class _SeriesFacts:
    """What the series map says on its own, read once per call and never changed by a pass."""

    key_by_id: Mapping[int, EpisodeKey]
    """Episode id -> `(season, episode)` key: the series map inverted."""
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
    """The evidence that an unplaced file is a duplicate, gathered once before classification."""

    proven: frozenset[int]
    """Ids held for a reason: placed by this call, or a seeded or gone file's reading or title points there."""
    held_by_file: Mapping[FileIdentity, frozenset[int]]
    """The episodes each file's versions hold: placed by this call, or a seeded or gone version titled as one."""


@dataclass(slots=True)
class _Placer:
    """One `assign_episode_ids` call's state: the readings, the verdicts so far, and the ids used so far."""

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
    """Whether the count-based passes (the absolute zip, the ordered zip, the numbered run) are turned off.

    Set when a title rejects a reading, when the release-run pass sees several candidate runs or refuses its
    pick, when a title places a file, and when a title contradicts a zip pair."""

    @classmethod
    def start(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """Read every name and every parse of the torrent once."""

        readings = {name: read_parse(batch.parsed.get(name), scope) for name in batch.torrent_names}
        torrent = _TorrentNames.of(batch, scope, readings)
        return cls(batch, scope, readings, torrent, _SeriesFacts.of(scope.series), used=set(scope.used))

    def single_parse(self, name: str) -> ParsedFileInfo | None:
        """The name's parse when there is one and it covers one episode, else None.

        A multi-episode file placed as one episode would import half.
        """

        info = self.batch.parsed.get(name)
        return None if info is None or self.spans_multiple(info) else info

    def title_placement(self, name: str) -> int | None:
        """The open scope episode the name is titled as, else None.

        Also None for a missing or multi-episode parse, and for a tied file titled as neither of its candidates.
        """

        ep_id = self.torrent.episode(name)
        if ep_id is None or ep_id in self.used or not self.scope.admits(ep_id) or self.single_parse(name) is None:
            return None
        tied = self.readings[name].tied
        return None if tied is not None and ep_id not in tied else ep_id

    def titled_outside(self, name: str, nothing_left: bool) -> bool:
        """Whether the name is titled as an episode outside the scope, confirmed another way.

        Confirmed when the file has a complete reading wholly outside the scope, or when no open id is left.
        A title alone never excludes a file the entry may still need.
        """

        ep_id = self.torrent.episode(name)
        if ep_id is None or self.scope.admits(ep_id):
            return False
        reading = self.readings[name]
        return (reading.complete and not reading.inside) or nothing_left

    def reads_no_number(self, name: str) -> bool:
        """Whether the file's single-episode parse has no usable number: none at all, or only keys the series lacks.

        False for a tied file, and for a titled file (its titled episode is taken or outside the scope).
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
        """Whether the series map is non-empty. With an empty map, every verdict that needs it is refused."""

        return bool(self.scope.id_by_key)

    def season_fit(self, run: NumberedRun) -> _SeasonFit | None:
        """The season a `1..N` run without own keys belongs to: Sonarr matched it into one season of exactly N episodes.

        Every member needs a parse and at most one resolved id. At least one member must be matched onto the
        specials, but no more than into the season. It is a tie when as many went to each and the specials season
        also has N episodes. None for any other shape.
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
        """Reject the run's readings and record each file's episode under the season's and the specials' numbering."""

        for numbered in run.whole:
            keys = (EpisodeKey(season, numbered.number), EpisodeKey(0, numbered.number))
            ids = tuple(i for key in keys if (i := self.scope.id_by_key.get(key)))
            self.readings[numbered.name] = self.readings[numbered.name]._replace(vetoed=True, tied=ids)

    def reread(self, run: NumberedRun, season: int) -> None:
        """Read each run file as the season's episode of its release number, rejected where its title disagrees."""

        for numbered in run.whole:
            ep_id = self.scope.id_by_key.get(EpisodeKey(season, numbered.number))
            if ep_id:
                inside = (ep_id,) if self.scope.admits(ep_id) else ()
                vetoed = self.torrent.contradicts(numbered.name, (ep_id,))
                self.readings[numbered.name] = Reading(
                    (ep_id,), inside, complete=True, borrowed=True, vetoed=vetoed, corroborated=True
                )

    def veto(self, run: NumberedRun) -> None:
        """Reject the reading of every file of the run, superseded versions included."""

        self.veto_names(numbered.name for numbered in run.whole)

    def veto_names(self, names: Iterable[str]) -> None:
        """Reject each name's reading: it still counts as evidence, but no pass places the file by it."""

        for name in names:
            self.readings[name] = self.readings[name]._replace(vetoed=True)

    def open_ids(self) -> list[int]:
        """The open ids: the scope's nonzero ids minus every id used so far, in scope order."""

        return [i for i in self.scope.resolved if i and i not in self.used]

    def run_window(self) -> _RunWindow | None:
        """The open ids as a run window, else None. It needs at least two open ids.

        The ids must be consecutive episodes of one season, or else all carry consecutive absolute numbers (a
        special in between, or two seasons' cours). The ids are sorted by their numbers, never taken in scope order.
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
        """Whether the file likely holds more than one episode: its name claims several, or Sonarr matched several.

        This asks how many episodes, not which: a file matched to two episodes but placed as one imports half.
        A full-season match (a name with no episode number) counts only when one of its episodes is in the scope.
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
        """Whether Sonarr matched every member to its own distinct episode outside the scope (another season or cour).

        A tied run counts as elsewhere when none of its candidate episodes is in the scope.
        """

        readings = [self.readings[name] for name in run.names]
        if all(r.tied is not None for r in readings):
            return not any(i in self.scope.real_ids for r in readings for i in r.tied or ())
        if not all(r.outside and r.corroborated and len(r.resolved) == 1 for r in readings):
            return False
        return len({r.resolved[0] for r in readings}) == len(readings)

    def covering(self, runs: Iterable[NumberedRun], window: _RunWindow) -> list[NumberedRun]:
        """The runs numbered `1..N` for the window's whole season, when the window holds fewer than N ids."""

        season = window.season
        if season is None:
            return []
        count = self.facts.season_counts.get(season, 0)
        return runs_from_one(runs, count) if count > len(window.ids) else []

    def covering_refused(self, window: _RunWindow) -> bool:
        """Whether a whole-season run cannot be trusted to number this window.

        True for a window spanning seasons, for a specials window (a run as long as the specials season is a
        coincidence), and for a season whose numbering has a gap (the run would count on past it).
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
        """Whether a member's reading lands in a season other than the window's, which disputes the run's numbers."""

        return any(read != season for read in self.seasons_read(run))

    def place(self, name: str, ids: Sequence[int], verdict: PlacementVerdict) -> None:
        """Record a placement and mark its ids used."""

        self.verdicts[name] = Placement(name, tuple(ids), verdict)
        self.used.update(ids)

    def place_zip(self, names: Sequence[str], ids: Sequence[int], verdict: PlacementVerdict) -> bool:
        """Place the names onto the ids pairwise, or none when a name is titled as an episode other than its pair.

        A refused zip turns the count-based passes off: the count that paired them was wrong.
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
        """The numbered runs among `names`, formed from their release numbers."""

        members = (member for name in names if (member := self.torrent.reads[name].member) is not None)
        return numbered_runs(members, self.batch.parsed)

    def runs_from_anywhere(self, runs: Iterable[NumberedRun], width: int) -> list[NumberedRun]:
        """The runs of `width` consecutive numbers, from any start, where no member has a complete reading.

        Empty once any id is used (by a seeded file, an earlier window, or an earlier pass): a run fitting the
        remaining ids would fit by chance, not by count.
        """

        if self.used:
            return []
        return [run for run in runs if run.consecutive(width) and not any(self.readings[n].complete for n in run.names)]

    def release_candidates(self, runs: Sequence[NumberedRun], window: _RunWindow) -> list[_Candidate]:
        """Every run that could place onto the window, at the best tier it fits, sorted by tier then by run order."""

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
        """Gather the duplicate evidence.

        An id held from the start (a seeded file's or an earlier window's) proves nothing alone.
        """

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
        """Classify every unplaced file, then return all verdicts in batch order."""

        proof = self.duplicate_proof()
        nothing_left = not self.open_ids()
        for name in self.remaining():
            reading = self.readings[name]
            titled = self.torrent.episode(name)
            if reading.complete and not reading.vetoed and reading.inside == reading.resolved:
                # Inside the scope, so the exact pass skipped it because an episode is taken. A span only partly
                # taken still covers a free episode, so it stays SKIPPED and gets reported.
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
    """Set the extras aside as `PlacementVerdict.EXTRA`: never an episode, and left out of every count."""

    for name in state.remaining():
        if name in state.torrent.extras:
            state.set_aside(name, PlacementVerdict.EXTRA)


def _pass_refuse_refuted(state: _Placer) -> None:
    """Reject each reading its file's title contradicts, and the in-scope readings of that file's run no title confirms.

    The run's numbering, its own or Sonarr's, was wrong about the titled file, so it is wrong by the same shift
    about every member. Seeded members count too, so an import poll judges the leftovers as the grab did. A
    member whose own title confirms its reading keeps it. Any rejection turns the count-based passes off.
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
    """Reject each unplaced multi-episode reading that shares an episode with a different reading of another file.

    One of the two is wrong, and a wrongly placed span imports half. Seeded files count as the other file,
    so an import poll rejects the same span the grab did.
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
    """Read a `1..N` run Sonarr matched into one N-episode season as that season's episodes 1 to N (`season_fit`).

    TVDB interleaves specials into the absolute numbering, so in Sonarr's match of a release without the specials,
    every episode after a special lands one number off, the first on the special. A release with the special
    would number N + 1. Only a run without own keys qualifies. In a tie (as many members matched onto an
    N-episode specials season), the readings are rejected and each file records both candidates in
    `Reading.tied`: the release-run pass can still place them, and the episode-title pass only onto one of the two.
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
    """Place one numbered run onto the run window by its own numbers when Sonarr's reading of it does not add up.

    Candidates are the unplaced runs that fit a `_Tier`. Tie-breaks, in order: drop runs Sonarr matched whole
    outside the scope (unless all are), then keep the one run the titles select, the one an AniList title names,
    or the best tier. With any parse unknown, the pick is `HELD`. It is refused by `_run_refused`,
    `covering_refused` for a covering pick, a dispute, a title veto (`_TitleEvidence.vetoes`), or an AniList
    title naming another run.
    """

    if not state.map_known:
        return
    window = state.run_window()
    if window is None:
        return
    runs = state.runs(state.remaining())
    candidates = state.release_candidates(runs, window)
    # A covering run read into another season is disputed, picked or not. Read whole into one other season, it
    # belongs there (its files end up FOREIGN). Otherwise its readings are rejected.
    disputed = [c.run for c in candidates if c.tier is _Tier.COVERING and state.read_elsewhere(c.run, window.season)]
    for run in disputed:
        if not (state.settled_elsewhere(run) and len(state.seasons_read(run)) == 1):
            state.veto(run)
    open_positions = frozenset(index for index, run in enumerate(runs) if not state.settled_elsewhere(run))
    naming = Naming.of([run.prefix for run in runs], state.scope.names)
    # Several candidates turn the count-based passes off, even when a tie-break picks one.
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
    # A whole-season run over a partial window places only the window's numbers. Its other members are FOREIGN.
    members = window.slice_of(run) if covers else run.members
    readings = [state.readings[member.name] for member in members]
    coherent = all(
        r.complete and not r.vetoed and len(r.resolved) == 1 and r.resolved[0] in window.id_set for r in readings
    ) and len({r.resolved[0] for r in readings}) == len(readings)
    # Sonarr already reads each member onto its own window episode: the exact pass places them.
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
    """Place each unplaced file whose complete, unrejected reading lies wholly inside the scope on unused ids.

    When two files resolve to one episode, `Reading.rank` decides, then the higher `vN`, then batch order.
    So a "17.5 (S00E01)" beats a "- 17" whose match Sonarr shifted onto the same special, a "- 08" beats a
    "- Bonus" whose CRC tag parsed as its key, and a "- 09v2" beats its "- 09". The loser stays unplaced.
    """

    for name in sorted(state.remaining(), key=lambda n: (state.readings[n].rank, -state.torrent.reads[n].version)):
        reading = state.readings[name]
        if not reading.complete or reading.vetoed or reading.inside != reading.resolved:
            continue
        if any(i in state.used for i in reading.inside):
            continue
        state.place(name, reading.inside, PlacementVerdict.EXACT)


def _pass_episode_title(state: _Placer) -> None:
    """Place each unplaced file on the open scope episode its name is titled as, highest `vN` first.

    A title speaks for one file, so this pass runs even when the count-based passes are off. Each placement
    turns them off: a file its number failed to place (a shifted absolute match, a bogus key) shows the
    batch's count is off.
    """

    for name in sorted(state.remaining(), key=lambda n: state.torrent.reads[n].version, reverse=True):
        if (ep_id := state.title_placement(name)) is not None:
            state.place(name, (ep_id,), PlacementVerdict.EPISODE_TITLE)
            state.count_legs_barred = True


def _pass_counted(state: _Placer) -> None:
    """Try the absolute zip, else the single-file pass when one id is open, else the ordered zip."""

    # A file whose trusted reading lies wholly outside the scope belongs to another slice.
    # It takes no open id and does not spoil the count for the other files.
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
    """The open files in absolute-number order when the absolute zip applies (see the module), else None."""

    parsed = state.batch.parsed
    abs_by_file: dict[str, int] = {}
    for name in open_names:
        info = parsed.get(name)
        if info is not None and len(info.absolute_episode_numbers) == 1:
            abs_by_file[name] = info.absolute_episode_numbers[0]
    # The shared-absolute check (a sign of restarted numbering) reads every non-extra parse, seeded files included,
    # so a v2 still collides with the v1 an earlier poll placed. Repeats inside one parse are junk and collapse.
    batch_absolutes = [
        number
        for name in state.torrent.counted
        if (parse := parsed.get(name)) is not None
        for number in dict.fromkeys(parse.absolute_episode_numbers)
    ]
    # An unknown parse (missing, or the offline fallback that cannot see absolutes) may hide a shared
    # absolute, so the check's input is incomplete and the zip refuses.
    applies = (
        bool(abs_by_file)
        and not state.count_legs_barred
        and state.batch.all_parses_known
        and len(abs_by_file) == len(open_names)  # every open file has one absolute
        and len(abs_by_file) == len(open_ids)  # as many files as open ids
        and len(set(batch_absolutes)) == len(batch_absolutes)  # no shared absolute (restarted numbering)
    )
    return sorted(abs_by_file, key=abs_by_file.__getitem__) if applies else None


def _place_single(state: _Placer, open_names: Sequence[str], ep_id: int) -> None:
    """Fill the one open id: with the only open file if it is numberless, else the numberless file a title names.

    The title is an AniList title of the entry, and that match runs only when every parse is known.
    """

    unnumbered = [name for name in open_names if state.reads_no_number(name)]
    if len(open_names) == 1 and unnumbered:
        state.place(unnumbered[0], (ep_id,), PlacementVerdict.SINGLE)
    elif state.batch.all_parses_known:
        stems = [state.torrent.reads[name].stem.text for name in unnumbered]
        if (pick := sole_title_match(stems, state.scope.names)) is not None:
            state.place(unnumbered[pick], (ep_id,), PlacementVerdict.TITLED)


def _natural_order(state: _Placer, open_names: Sequence[str], open_ids: Sequence[int]) -> list[str] | None:
    """The open files in natural name order when the ordered zip applies (the "Special 1..N" shape), else None."""

    parsed = state.batch.parsed
    # Pristine: the parse map holds the open files and the extras and nothing else, and nothing but an extra
    # has a verdict, so nothing was placed, held, or seeded.
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
    """Zip the one `1..N` run among the files Sonarr found no episode for onto a run window of N ids.

    Unlike the ordered zip, this works beside files that other passes placed. A file qualifies when its reading
    resolved nothing, its parse is known, its name carries no `SxxExx` key, and it is not multi-episode, so a
    file Sonarr matched elsewhere in the series is never moved by position. Skipped without a series map,
    with any parse unknown, or with the count-based passes off.
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
    """Whether the batch shows the run cannot own the whole window."""

    window_set = window.id_set
    for name in run.names:
        info = state.batch.parsed.get(name)
        # A missing parse, or a name that itself claims several episodes, refuses the run. A multi-episode
        # Sonarr match does not: the run's own numbering and file count override it.
        if info is None or claims_several_episodes(info):
            return True
        reading = state.readings[name]
        keyed_outside = not reading.borrowed and reading.complete and not any(i in window_set for i in reading.resolved)
        # A member whose own key lies outside the window refuses the run, so a torrent listed on the wrong
        # sequel entry never imports onto it. A specials window is exempt.
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
