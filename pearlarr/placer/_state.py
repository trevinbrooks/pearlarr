"""The state one `assign_episode_ids` call runs its passes over, and the run window the run passes read.

`Placer` holds the views two or more passes read. Verdicts and used ids change only through `place*` and
`set_aside*`, and vetoes through `veto*`, except the readings pass 5 rewrites whole. A helper of one pass sits with
that pass.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Self

from ..episode_titles import TorrentNames, spells_title
from ..parse_reading import Reading, read_parse, spans_multiple, widen_range_key
from ..placement_types import Placement, PlacementBatch, PlacementVerdict, SeriesFacts, TargetScope
from ..release_names import NumberedRun, RunMember, is_consecutive, numbered_runs
from ..seadex_types import ParsedFileInfo


@dataclass(frozen=True, slots=True)
class RunWindow:
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


@dataclass(slots=True)
class Placer:
    """One `assign_episode_ids` call's state: the readings, the verdicts so far, and the ids used so far."""

    batch: PlacementBatch
    scope: TargetScope
    readings: dict[str, Reading]
    """One reading per name of the torrent, seeded and gone names included: their evidence counts, though
    only `batch.names` are placed."""
    torrent: TorrentNames
    """Every name of the torrent read once, seeded and gone names included."""
    facts: SeriesFacts
    counted_runs: tuple[NumberedRun, ...] = field(init=False)
    """The `runs` among every name but the extras (`TorrentNames.counted`)."""
    verdicts: dict[str, Placement] = field(default_factory=dict[str, Placement])
    used: set[int] = field(default_factory=set[int])
    count_legs_barred: bool = False
    """Whether the count-based passes (the absolute zip, the ordered zip, the numbered run) are turned off: an
    earlier pass found a sign the batch's count is off."""
    size_refuted_runs: set[str] = field(default_factory=set[str])
    """Prefixes of runs where a member was placed by size against its own numbers. `pass_release_run` won't
    place these runs."""
    disputed: set[str] = field(default_factory=set[str])
    """Names whose own numbers disagree over an episode of this window (`_passes._pass_refuse_disputed`), or whose
    count no title confirms (`_passes._pass_refuse_refuted`): `MISNUMBERED` unless a run places them."""
    disputed_by_position: set[str] = field(default_factory=set[str])
    """The `disputed` members of a keyed run numbered past an interleaved special (`_passes._positions_disputed`):
    no run places them."""
    disputed_elsewhere: set[str] = field(default_factory=set[str])
    """Names whose own numbers disagree over other windows' episodes only: read as they are, never numberless."""
    run_refuted: set[str] = field(default_factory=set[str])
    """Run members rejected because a mate's title refutes the run's numbers (`_passes._pass_refuse_refuted`),
    their own names carrying no such title: `TITLE_REFUTED` unless a later pass places them."""
    runner_ups: dict[str, frozenset[int]] = field(default_factory=dict[str, frozenset[int]])
    """Each runner-up's lost episodes: the ids a disputed reading would have taken first in the exact pass's order."""

    def __post_init__(self) -> None:
        self.counted_runs = tuple(self.runs(self.torrent.counted))

    @classmethod
    def start(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """Read every name and every parse of the torrent once, after widening each range key Sonarr read short."""

        facts = SeriesFacts(scope.series)
        widened = {
            name: None if info is None else widen_range_key(name, info, facts) for name, info in batch.parsed.items()
        }
        # The readings and every pass see the widened parses.
        batch = batch._replace(parsed=widened)
        readings = {name: read_parse(batch.parsed.get(name), scope) for name in batch.torrent_names}
        torrent = TorrentNames.of(batch, scope, readings)
        return cls(batch, scope, readings, torrent, facts, used=set(scope.used))

    def single_parse(self, name: str) -> ParsedFileInfo | None:
        """The name's parse when there is one and it covers one episode, else None."""

        info = self.batch.parsed.get(name)
        return None if info is None or spans_multiple(info, self.scope) else info

    def counted_by_title(self, name: str) -> bool:
        """Whether Sonarr counted the file by its own title's number (`NameRead.titled_count`) and nothing confirms it.

        Every number Sonarr parsed is that count, and the words behind it spell a title (`spells_title`) that
        names no episode. Read as an episode, the count stands when those words are an episode's title: the
        file's own, its own a word off at most (`EpisodeTitles.nearly`), or another's that refutes it.
        "Show - 13 Rainy Days" is held when no title says either. Read as no episode, it stands when an episode of
        the series carries the count: "Show - 100 Days Later" in a 13-episode series is no absolute. A spelled-out
        key stands, and a one-word tail ("07 MHz") says too little.
        """

        read = self.torrent.reads[name]
        info = self.batch.parsed.get(name)
        if read.titled_count is None or info is None or read.explicit_key or not spells_title(read.title_words):
            return False
        season = info.season_number
        numbers = {*info.absolute_episode_numbers, *(season * 100 + ep for ep in info.episode_numbers if season)}
        if numbers != {read.titled_count} or self.torrent.episode(name) is not None:
            return False
        if resolved := self.readings[name].resolved:
            return len(resolved) != 1 or not self.torrent.episodes.nearly(resolved[0], read.title_words)
        return read.titled_count not in self.facts.id_by_absolute

    def runner_up(self, name: str, ids: Iterable[int]) -> bool:
        """Whether the file is a runner-up for any of `ids`."""

        return not self.runner_ups.get(name, frozenset()).isdisjoint(ids)

    def _exact_order(self) -> list[str]:
        """The unplaced names in the exact pass's order: `Reading.rank`, then the higher `vN`, then batch order."""

        return sorted(self.remaining(), key=lambda n: (self.readings[n].rank, -self.torrent.reads[n].version))

    def exact_ids(self, name: str) -> tuple[int, ...]:
        """The ids the exact pass places the file on: a trusted reading wholly inside the scope. Else ()."""

        reading = self.readings[name]
        return reading.inside if reading.trusted and reading.inside == reading.resolved else ()

    def exact_picks(self) -> dict[str, tuple[int, ...]]:
        """The files the exact pass places as the readings stand, in its order, each with its ids.

        A file is passed over when one of its ids is taken, by an earlier pick too, or when it is a runner-up for one.
        """

        used = set(self.used)
        picks: dict[str, tuple[int, ...]] = {}
        for name in self._exact_order():
            ids = self.exact_ids(name)
            if ids and used.isdisjoint(ids) and not self.runner_up(name, ids):
                used.update(ids)
                picks[name] = ids
        return picks

    @property
    def map_known(self) -> bool:
        """Whether the series map is non-empty. With an empty map, every verdict that needs it is refused."""

        return bool(self.scope.id_by_key)

    @property
    def unjudged(self) -> bool:
        """Whether no file other than an extra has a verdict yet."""

        return self.verdicts.keys() <= self.torrent.extras

    @property
    def untouched(self) -> bool:
        """Whether nothing has spoken for the batch: `unjudged`, and the scope started with no used ids."""

        return self.unjudged and not self.scope.used

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

    def run_window(self) -> RunWindow | None:
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
            return RunWindow(seasons.pop(), tuple(ep_id for _, ep_id in keyed), tuple(episodes))
        absolute_of = self.facts.absolute_of
        absolute = sorted((absolute_of[ep_id], ep_id) for ep_id in open_ids if ep_id in absolute_of)
        if len(absolute) != len(open_ids) or not is_consecutive([number for number, _ in absolute]):
            return None
        return RunWindow(None, tuple(ep_id for _, ep_id in absolute), tuple(number for number, _ in absolute))

    def remaining(self) -> list[str]:
        """The names without a verdict yet, batch order."""

        return [name for name in self.batch.names if name not in self.verdicts]

    def remaining_members(self, members: Iterable[RunMember]) -> list[RunMember]:
        """The members still to place, in the order given. Seeded, gone, and already judged ones drop out."""

        remaining = frozenset(self.remaining())
        return [member for member in members if member.name in remaining]

    def place(self, name: str, ids: Sequence[int], verdict: PlacementVerdict) -> None:
        """Record a placement and mark its ids used."""

        self.verdicts[name] = Placement(name, tuple(ids), verdict)
        self.used.update(ids)

    def place_or_skip(self, name: str, ep_id: int, verdict: PlacementVerdict) -> None:
        """Place the file on the episode, or set it aside as `SKIPPED` when the episode is already used."""

        if ep_id in self.used:
            self.set_aside(name, PlacementVerdict.SKIPPED)
        else:
            self.place(name, (ep_id,), verdict)

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
        """The numbered runs among `names`, formed by their release numbers, less the `_absolute_disagrees` members."""

        members = (member for name in names if (member := self.torrent.reads[name].member) is not None)
        return numbered_runs(member for member in members if not self._absolute_disagrees(member))

    def _absolute_disagrees(self, member: RunMember) -> bool:
        """Whether the member's parse has one absolute, neither its release number nor one of its `title_numbers`."""

        info = self.batch.parsed.get(member.name)
        if info is None or len(absolutes := set(info.absolute_episode_numbers)) != 1:
            return False
        (absolute,) = absolutes
        return absolute != member.number and absolute not in member.title_numbers
