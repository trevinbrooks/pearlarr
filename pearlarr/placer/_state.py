"""The state one `assign_episode_ids` call runs its passes over: its files, their numbered runs, the run window.

`TorrentFile` holds what one call knows and decides about one file, `NumberedRun` the files sharing one prefix,
`RunWindow` the open ids the run passes read, and `Placer` every file plus the views two or more passes read. A
helper of one pass sits with that pass.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

from ._names import TorrentNames
from ..episode_titles import EpisodeTitles, spells_title
from ..parse_reading import Reading, glued_counts, read_parse, spans_multiple, widen_range_key
from ..placement_types import EpisodeIndex, Placement, PlacementBatch, PlacementVerdict, TargetScope
from ..release_names import FileIdentity, NameRead, is_consecutive
from ..seadex_types import ParsedFileInfo


@dataclass(slots=True, eq=False, kw_only=True)
class TorrentFile:
    """One name of the torrent in one `assign_episode_ids` call: what its name and parse say, and the verdict.

    Versions of one file are separate records. Records hash by identity, so passes keep sets of them. The fields
    through `seeded_on` are set once in `Placer.start`. Placements change only through `Placer.place*` and
    `Placer.set_aside*`, and vetoes through `veto`, except that pass 5 re-reads or ties a run's readings. Passes 2
    and 3 set the flags and `runner_up_for`.
    """

    name: str
    """The normalized file name."""
    to_place: bool
    """Whether the batch places this name. False for a seeded or gone name, which is evidence only."""
    in_parse_map: bool
    """Whether the batch's parse map has an entry for the name, a None entry included."""
    parsed: ParsedFileInfo | None
    """The widened parse (`widen_range_key`), else None when the batch has no parse for the name."""
    name_read: NameRead
    """The name as `read_names` reads it within the torrent: stem, run membership, title words, numbers."""
    titled_as: int | None
    """The episode the name is titled as: its specific title, carried by no other file of the torrent, else None."""
    leaders: frozenset[int]
    """Every episode whose title starts the name's tail: run evidence, read for run members only."""
    extra: bool
    """Whether the file is an extra: set aside, and left out of every count."""
    identified_as: int | None
    """The episode whose listed single-file release has the file's size (`ListingEvidence.identified`), else None."""
    seeded_on: tuple[int, ...] | None
    """The ids the name is seeded on from an earlier placement, else None."""
    reading: Reading
    """The parse read against the scope. Pass 5 re-reads or ties it, and `veto` marks it rejected."""
    placement: Placement | None = None
    """The verdict and ids, once a pass judges the file."""
    disputed: bool = False
    """Whether the file's own numbers disagree over an episode of this window (`_passes._pass_refuse_disputed`), or
    no title confirms its count (`_passes._pass_refuse_refuted`): `MISNUMBERED` unless a run places it."""
    disputed_by_position: bool = False
    """Whether the file is a `disputed` member of a keyed run numbered past an interleaved special
    (`_passes._positions_disputed`): no run places it."""
    disputed_elsewhere: bool = False
    """Whether the file's own numbers disagree over other windows' episodes only (`_passes._pass_refuse_disputed`):
    read as they are, never numberless."""
    run_refuted: bool = False
    """Whether a mate's title refutes the numbers of the file's run and its own name carries no such title
    (`_passes._pass_refuse_refuted`): `TITLE_REFUTED` unless a later pass places it."""
    runner_up_for: frozenset[int] = frozenset()
    """The ids a disputed reading would have taken first in the exact pass's order (`_passes._pass_refuse_disputed`):
    the file is their runner-up."""

    @property
    def judged(self) -> bool:
        """Whether a pass has judged the file."""

        return self.placement is not None

    @property
    def remaining(self) -> bool:
        """Whether the batch places the file and no pass has judged it yet."""

        return self.to_place and not self.judged

    @property
    def identity(self) -> FileIdentity:
        """The file the name is a version of (`NameRead.file`)."""

        return self.name_read.file

    @property
    def release_number(self) -> int:
        """The release number of a run's file. Read it only on a run's files: a file in no run may have none."""

        member = self.name_read.member
        assert member is not None
        return member.number

    def contradicts(self, ids: Sequence[int]) -> bool:
        """Whether the name is titled as an episode that is not among `ids`."""

        return (ep_id := self.titled_as) is not None and ep_id not in ids

    def confirms(self, ids: Sequence[int]) -> bool:
        """Whether the episode the name is titled as is among `ids`."""

        return (ep_id := self.titled_as) is not None and ep_id in ids

    def runner_up(self, ids: Iterable[int]) -> bool:
        """Whether the file is a runner-up for any of `ids`."""

        return not self.runner_up_for.isdisjoint(ids)

    @property
    def absolute_disagrees(self) -> bool:
        """Whether the file's parse has one absolute, neither its release number nor one of its `title_numbers`.

        False for a name without a release number.
        """

        member = self.name_read.member
        info = self.parsed
        if member is None or info is None or len(absolutes := set(info.absolute_episode_numbers)) != 1:
            return False
        (absolute,) = absolutes
        return absolute != member.number and absolute not in member.title_numbers

    def veto(self) -> None:
        """Reject the reading: it still counts as evidence, but no pass places the file by it."""

        self.reading = self.reading._replace(vetoed=True)


@dataclass(frozen=True, slots=True)
class NumberedRun:
    """The files whose names share one prefix before their release number."""

    prefix: str
    """The text before the number that every member shares."""
    members: tuple[TorrentFile, ...]
    """The top version at each number, in number order. Files tied at the top version all stay, so a number can
    repeat."""
    superseded: tuple[TorrentFile, ...]
    """The lower versions a member's higher `vN` replaced: duplicates once the run is placed."""

    @classmethod
    def of(cls, prefix: str, by_number: Mapping[int, Sequence[TorrentFile]]) -> Self:
        """One prefix's run: the top version at each number kept, the lower ones superseded."""

        kept: list[TorrentFile] = []
        superseded: list[TorrentFile] = []
        for versions in by_number.values():
            top = max(file.name_read.version for file in versions)
            kept.extend(file for file in versions if file.name_read.version == top)
            superseded.extend(file for file in versions if file.name_read.version < top)
        members = sorted(kept, key=lambda file: (file.release_number, file.name))
        return cls(prefix, tuple(members), tuple(superseded))

    @property
    def whole(self) -> tuple[TorrentFile, ...]:
        """The members plus the lower versions they superseded."""

        return (*self.members, *self.superseded)

    @property
    def numbers(self) -> tuple[int, ...]:
        """The members' release numbers, in order."""

        return tuple(file.release_number for file in self.members)

    def consecutive(self, width: int) -> bool:
        """Whether the run is exactly `width` consecutive numbers, wherever it starts."""

        return len(self.members) == width and is_consecutive(self.numbers)

    def from_one(self, width: int) -> bool:
        """Whether the run is numbered `1..width`."""

        return self.numbers == tuple(range(1, width + 1))


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

    def slice_of(self, run: NumberedRun) -> tuple[TorrentFile, ...]:
        """The run members whose numbers are the window's episode numbers (a whole-season run over part of it)."""

        return tuple(file for file in run.members if file.release_number in self.numbers)


@dataclass(slots=True, kw_only=True)
class Placer:
    """One `assign_episode_ids` call's state: every file of the torrent, and the ids used so far."""

    scope: TargetScope
    """The caller's scope with the stored map's ids it admits used (`TargetScope.using`)."""
    episode_titles: EpisodeTitles
    """The episode titles the names were read against."""
    files: tuple[TorrentFile, ...]
    """Every file of the torrent: the files to place in batch order, then the seeded and gone ones."""
    placing: tuple[TorrentFile, ...]
    """The files to place, batch order."""
    counted: tuple[TorrentFile, ...]
    """Every file but the extras: the files the evidence checks read."""
    all_parses_known: bool
    """Whether every parse of the torrent is known (`PlacementBatch.all_parses_known`)."""
    counted_runs: tuple[NumberedRun, ...] = field(init=False)
    """The `runs` among the `counted` files."""
    used: set[int] = field(default_factory=set[int])
    """Every id placed on so far, the scope's used ids included. Only `place` adds to it."""
    count_legs_barred: bool = False
    """Whether the count-based passes (the absolute zip, the ordered zip, the numbered run) are turned off: an
    earlier pass found a sign the batch's count is off."""
    size_refuted_runs: set[str] = field(default_factory=set[str])
    """Prefixes of files placed by size against their own release numbers. `pass_release_run` won't place a run
    with one of these prefixes."""

    def __post_init__(self) -> None:
        self.counted_runs = tuple(self.runs(self.counted))

    @property
    def series(self) -> EpisodeIndex:
        """The scope's episode index, whose lookups (`EpisodeIndex.key_by_id` and the rest) the passes read."""

        return self.scope.series

    @classmethod
    def start(cls, batch: PlacementBatch, scope: TargetScope) -> Self:
        """Read every name and every parse of the torrent once, after widening each range key Sonarr read short.

        The stored map's ids that `scope` admits count as used from the start, on `Placer.scope` itself.
        """

        seeded = {name: tuple(ids) for name, ids in batch.seeded.items()}
        # On the scope, not just `used`: `untouched` and the duplicate proof read `scope.used` as held before this call.
        scope = scope.using(ep_id for ids in seeded.values() for ep_id in ids)
        widened = {
            name: None if info is None else widen_range_key(name, info, scope.series)
            for name, info in batch.parsed.items()
        }
        # The readings and every pass see the widened parses.
        batch = replace(batch, parsed=widened)
        readings = {name: read_parse(batch.parsed.get(name), scope) for name in batch.torrent_names}
        torrent = TorrentNames.of(batch, scope, readings)
        names_to_place = frozenset(batch.names)
        by_name = {
            name: TorrentFile(
                name=name,
                to_place=name in names_to_place,
                in_parse_map=name in batch.parsed,
                parsed=batch.parsed.get(name),
                name_read=title.read,
                titled_as=title.titled_as,
                leaders=title.leaders,
                extra=title.extra,
                identified_as=scope.listing.identified.get(name),
                seeded_on=seeded.get(name),
                reading=readings[name],
            )
            for name, title in torrent.by_name.items()
        }
        return cls(
            scope=scope,
            episode_titles=torrent.episode_titles,
            files=tuple(by_name.values()),
            placing=tuple(by_name[name] for name in batch.names),
            counted=tuple(file for file in by_name.values() if not file.extra),
            all_parses_known=batch.all_parses_known,
            used=set(scope.used),
        )

    def single_parse(self, file: TorrentFile) -> ParsedFileInfo | None:
        """The name's parse when there is one and it covers one episode, else None."""

        info = file.parsed
        return None if info is None or spans_multiple(info, self.scope) else info

    def counted_by_title(self, file: TorrentFile) -> bool:
        """Whether Sonarr counted the file by its own title's number (`NameRead.titled_count`) and nothing confirms it.

        Only a parse whose every number is that count is held. The count stands beside a spelled-out key, behind
        words too few to be a title (`spells_title`) or that name an episode, when they nearly name the one episode it
        reads as (`EpisodeTitles.nearly`), or, read as no episode, when it is an episode's absolute number.
        """

        name_read = file.name_read
        info = file.parsed
        if name_read.titled_count is None or info is None or name_read.explicit_key:
            return False
        if not spells_title(name_read.title_words):
            return False
        numbers = {*info.absolute_episode_numbers, *glued_counts(info)}
        if numbers != {name_read.titled_count} or file.titled_as is not None:
            return False
        if resolved := file.reading.resolved:
            return len(resolved) != 1 or not self.episode_titles.nearly(resolved[0], name_read.title_words)
        return name_read.titled_count not in self.series.id_by_absolute

    def _exact_order(self) -> list[TorrentFile]:
        """The unplaced files in the exact pass's order: `Reading.rank`, then the higher `vN`, then batch order."""

        return sorted(self.remaining(), key=lambda f: (f.reading.rank, -f.name_read.version))

    def exact_picks(self) -> dict[TorrentFile, tuple[int, ...]]:
        """The files the exact pass places as the readings stand, in its order, each with its ids.

        A file is passed over when one of its ids is taken, by an earlier pick too, or when it is a runner-up for one.
        """

        used = set(self.used)
        picks: dict[TorrentFile, tuple[int, ...]] = {}
        for file in self._exact_order():
            ids = file.reading.exact_ids
            if ids and used.isdisjoint(ids) and not file.runner_up(ids):
                used.update(ids)
                picks[file] = ids
        return picks

    @property
    def map_known(self) -> bool:
        """Whether the series map is non-empty. With an empty map, every verdict that needs it is refused."""

        return bool(self.scope.id_by_key)

    @property
    def unjudged(self) -> bool:
        """Whether no file other than an extra has a verdict yet."""

        return all(file.extra for file in self.files if file.judged)

    @property
    def untouched(self) -> bool:
        """Whether nothing has spoken for the batch: `unjudged`, and the scope started with no used ids."""

        return self.unjudged and not self.scope.used

    def veto_each(self, files: Iterable[TorrentFile]) -> None:
        """Reject each file's reading (`TorrentFile.veto`)."""

        for file in files:
            file.veto()

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
        key_by_id = self.series.key_by_id
        keyed = sorted((key_by_id[ep_id], ep_id) for ep_id in open_ids if ep_id in key_by_id)
        if len(keyed) != len(open_ids):
            return None
        seasons = {key.season for key, _ in keyed}
        episodes = [key.episode for key, _ in keyed]
        if len(seasons) == 1 and is_consecutive(episodes):
            return RunWindow(seasons.pop(), tuple(ep_id for _, ep_id in keyed), tuple(episodes))
        absolute_of = self.series.absolute_of
        absolute = sorted((absolute_of[ep_id], ep_id) for ep_id in open_ids if ep_id in absolute_of)
        if len(absolute) != len(open_ids) or not is_consecutive([number for number, _ in absolute]):
            return None
        return RunWindow(None, tuple(ep_id for _, ep_id in absolute), tuple(number for number, _ in absolute))

    def remaining(self) -> list[TorrentFile]:
        """The files to place without a verdict yet, batch order."""

        return [file for file in self.placing if file.remaining]

    def place(self, file: TorrentFile, ids: Sequence[int], verdict: PlacementVerdict) -> None:
        """Record a placement and mark its ids used."""

        file.placement = Placement(file.name, tuple(ids), verdict)
        self.used.update(ids)

    def place_or_skip(self, file: TorrentFile, ep_id: int, verdict: PlacementVerdict) -> None:
        """Place the file on the episode, or set it aside as `SKIPPED` when the episode is already used."""

        if ep_id in self.used:
            self.set_aside(file, PlacementVerdict.SKIPPED)
        else:
            self.place(file, (ep_id,), verdict)

    def place_zip(self, files: Sequence[TorrentFile], ids: Sequence[int], verdict: PlacementVerdict) -> bool:
        """Place the files onto the ids pairwise, or none when a file is titled as an episode other than its pair.

        A refused zip turns the count-based passes off: the count that paired them was wrong.
        """

        pairs = list(zip(files, ids, strict=True))
        if any(file.contradicts((ep_id,)) for file, ep_id in pairs):
            self.count_legs_barred = True
            return False
        for file, ep_id in pairs:
            self.place(file, (ep_id,), verdict)
        return True

    def set_aside(self, file: TorrentFile, verdict: PlacementVerdict) -> None:
        """Record a verdict that carries no ids (held, excluded, or skipped)."""

        file.placement = Placement(file.name, (), verdict)

    def set_aside_each(self, files: Iterable[TorrentFile], verdict: PlacementVerdict) -> None:
        """Record one id-less verdict for each file."""

        for file in files:
            self.set_aside(file, verdict)

    def runs(self, files: Iterable[TorrentFile]) -> list[NumberedRun]:
        """The numbered runs among `files`, formed by their release numbers, less the `absolute_disagrees` files."""

        by_prefix: dict[str, dict[int, list[TorrentFile]]] = {}
        for file in files:
            if (member := file.name_read.member) is not None and not file.absolute_disagrees:
                by_prefix.setdefault(member.prefix, {}).setdefault(member.number, []).append(file)
        return [NumberedRun.of(prefix, by_number) for prefix, by_number in by_prefix.items()]

    def placements(self) -> tuple[Placement, ...]:
        """The placements of the files to place, batch order. Each has one once the passes are done."""

        placed: list[Placement] = []
        for file in self.placing:
            assert file.placement is not None
            placed.append(file.placement)
        return tuple(placed)
