"""The state one `place_batch` call runs its passes over: its files, their numbered runs, the run window.

`TorrentFile` holds what one call knows and decides about one file, `NumberedSequence` the files sharing one prefix,
`ConsecutiveEpisodes` the open ids the run passes read, `NameFacts` what one name reads as before the first pass, and `PlacerState`
every file plus the views two or more passes read. A helper of one pass sits with that pass.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Self

from ._names import TorrentNames
from ..arr_models import ParsedFileInfo
from ..episode_titles import EpisodeTitles, spells_title
from ..file_names import FileIdentity, NameParts, is_consecutive
from ..parsed_episodes import ParsedEpisodes, glued_numbers, resolve_parse, spans_multiple, widen_range_key
from ..placement_types import EpisodeIndex, Placement, PlacementBatch, PlacementTarget, PlacementVerdict
from ..releases import OrderOnlyKey


@dataclass(slots=True, eq=False, kw_only=True)
class TorrentFile:
    """One name of the torrent in one `place_batch` call: what its name and parse say, and the verdict.

    Versions of one file are separate records. Records hash by identity, so passes keep sets of them. The fields
    through `saved_ids` are set once in `PlacerState.start`. Placements change only through `PlacerState.place*` and
    `PlacerState.leave_unplaced*`, and vetoes through `reject`, except that pass 5 re-reads or ties a run's readings. Passes 2
    and 3 set the flags and `next_in_line_for`. Pass 16 alone gives a judged file a new verdict.
    """

    name: str
    """The normalized file name."""
    to_place: bool
    """Whether the batch places this name. False for a seeded or gone name, which is evidence only."""
    in_parsed: bool
    """Whether the batch's parse map has an entry for the name, a None entry included."""
    parse: ParsedFileInfo | None
    """The widened parse (`widen_range_key`), else None when the batch has no parse for the name."""
    name_parts: NameParts
    """The name as `split_names` reads it within the torrent: stem, run membership, title words, numbers."""
    episode_by_title: int | None
    """The episode the name is titled as: its specific title, carried by no other file of the torrent, else None."""
    leading_title_ids: frozenset[int]
    """Every episode whose title starts the name's tail: run evidence, read for run members only."""
    extra: bool
    """Whether the file is an extra: set aside, and left out of every count."""
    size_match: int | None
    """The episode whose listed single-file release has the file's size (`SeadexEvidence.size_matches`), else None."""
    saved_ids: tuple[int, ...] | None
    """The ids the name is seeded on from an earlier placement, else None."""
    parsed_episodes: ParsedEpisodes
    """The parse read against the scope. Pass 5 re-reads or ties it, and `reject` marks it rejected."""
    placement: Placement | None = None
    """The verdict and ids, once a pass judges the file."""
    numbers_disagree: bool = False
    """Whether the file's own numbers disagree over an episode of this window (`_passes._pass_reject_disagreeing`), or
    no title confirms its count (`_passes._pass_reject_contradicted`): `MISNUMBERED` unless a later pass places it."""
    numbers_disagree_by_position: bool = False
    """Whether the file is a `disputed` member of a keyed run numbered past an interleaved special
    (`_passes._positions_disagree`): no run places it."""
    numbers_disagree_outside_window: bool = False
    """Whether the file's own numbers disagree over other windows' episodes only (`_passes._pass_reject_disagreeing`):
    read as they are, never numberless."""
    sequence_contradicted: bool = False
    """Whether a mate's title refutes the numbers of the file's run and its own name carries no such title
    (`_passes._pass_reject_contradicted`): `TITLE_REFUTED` when no pass places it and pass 15 reads it no other way
    (`_passes._pass_classify`)."""
    next_in_line_for: frozenset[int] = frozenset()
    """The ids a disputed reading would have taken first in the exact pass's order (`_passes._pass_reject_disagreeing`):
    the file is their runner-up."""

    @property
    def judged(self) -> bool:
        """Whether a pass has judged the file."""

        return self.placement is not None

    @property
    def verdict(self) -> PlacementVerdict | None:
        """The verdict a pass gave the file, else None."""

        return None if self.placement is None else self.placement.verdict

    @property
    def unjudged(self) -> bool:
        """Whether the batch places the file and no pass has judged it yet."""

        return self.to_place and not self.judged

    @property
    def identity(self) -> FileIdentity:
        """The file the name is a version of (`NameParts.identity`)."""

        return self.name_parts.identity

    @property
    def name_number(self) -> int:
        """The release number of a run's file. Read it only on a run's files: a file in no run may have none."""

        member = self.name_parts.member
        assert member is not None
        return member.number

    def contradicts(self, ids: Sequence[int]) -> bool:
        """Whether the name is titled as an episode that is not among `ids`."""

        return (ep_id := self.episode_by_title) is not None and ep_id not in ids

    def confirms(self, ids: Sequence[int]) -> bool:
        """Whether the episode the name is titled as is among `ids`."""

        return (ep_id := self.episode_by_title) is not None and ep_id in ids

    def next_in_line(self, ids: Iterable[int]) -> bool:
        """Whether the file is a runner-up for any of `ids`."""

        return not self.next_in_line_for.isdisjoint(ids)

    @property
    def absolute_disagrees(self) -> bool:
        """Whether the file's parse has one absolute, neither its release number nor one of its `title_numbers`.

        False for a name without a release number.
        """

        member = self.name_parts.member
        info = self.parse
        if member is None or info is None or len(absolutes := set(info.absolute_episode_numbers)) != 1:
            return False
        (absolute,) = absolutes
        return absolute != member.number and absolute not in member.title_numbers

    def reject(self) -> None:
        """Reject the reading: it still counts as evidence, but no pass places the file by it."""

        self.parsed_episodes = replace(self.parsed_episodes, rejected=True)


@dataclass(frozen=True, slots=True)
class NumberedSequence:
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
            top = max(file.name_parts.version for file in versions)
            kept.extend(file for file in versions if file.name_parts.version == top)
            superseded.extend(file for file in versions if file.name_parts.version < top)
        members = sorted(kept, key=lambda file: (file.name_number, file.name))
        return cls(prefix, tuple(members), tuple(superseded))

    @property
    def all_versions(self) -> tuple[TorrentFile, ...]:
        """The members plus the lower versions they superseded."""

        return (*self.members, *self.superseded)

    @property
    def numbers(self) -> tuple[int, ...]:
        """The members' release numbers, in order."""

        return tuple(file.name_number for file in self.members)

    def consecutive(self, width: int) -> bool:
        """Whether the run is exactly `width` consecutive numbers, wherever it starts."""

        return len(self.members) == width and is_consecutive(self.numbers)

    def from_one(self, width: int) -> bool:
        """Whether the run is numbered `1..width`."""

        return self.numbers == tuple(range(1, width + 1))


@dataclass(frozen=True, slots=True)
class ConsecutiveEpisodes:
    """The open ids as one run of episodes, in number order: the run window (`PlacerState.consecutive_unused`)."""

    season: int | None
    """The season every id belongs to, or None when absolute numbers order the ids."""
    ids: tuple[int, ...]
    """The open ids, by episode number, or by absolute number when `season` is None."""
    numbers: tuple[int, ...]
    """The ids' episode numbers (absolute numbers when `season` is None), in the same order. Consecutive, except
    over a scope's own specials numbered apart."""
    id_set: frozenset[int] = field(init=False, repr=False, compare=False)
    """The ids as a set, built once for membership checks."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id_set", frozenset(self.ids))

    @property
    def is_specials(self) -> bool:
        """Whether the window lies in the specials season (season 0)."""

        return self.season == 0

    def files_of(self, sequence: NumberedSequence) -> tuple[TorrentFile, ...]:
        """The run members whose numbers are the window's episode numbers (a whole-season run over part of it)."""

        return tuple(file for file in sequence.members if file.name_number in self.numbers)


@dataclass(frozen=True, slots=True, kw_only=True)
class NameFacts:
    """What the passes read off one name under one scope, before the first pass."""

    episode_by_title: int | None
    """The episode the name is titled as (`TorrentFile.episode_by_title`), else None."""
    parses_inside_window: bool
    """Whether a trusted reading of the name resolves wholly inside the scope (`ParsedEpisodes.exact_ids`): the exact pass
    would place it there, were its ids open and nothing disputing them."""
    extra: bool
    """Whether the name is an extra (an opening, an ending, and so on): never an episode."""
    parsed_by_title_number: bool
    """Whether Sonarr counted the name by its own title's number and nothing confirms it (`PlacerState.parsed_by_title_number`):
    the number is the title's, not a count."""


@dataclass(slots=True, kw_only=True)
class PlacerState:
    """One `place_batch` call's state: every file of the torrent, and the ids used so far."""

    target: PlacementTarget
    """The caller's scope with the stored map's ids it admits used (`PlacementTarget.using`)."""
    episode_titles: EpisodeTitles
    """The episode titles the names were read against."""
    files: tuple[TorrentFile, ...]
    """Every file of the torrent: the files to place in batch order, then the seeded and gone ones."""
    placing: tuple[TorrentFile, ...]
    """The files to place, batch order."""
    non_extras: tuple[TorrentFile, ...]
    """Every file but the extras: the files the evidence checks read."""
    all_parses_known: bool
    """Whether every parse of the torrent is known (`PlacementBatch.all_parses_known`)."""
    non_extra_sequences: tuple[NumberedSequence, ...] = field(init=False)
    """The `runs` among the `counted` files."""
    facts: Mapping[str, NameFacts] = field(init=False)
    """What every name of the torrent reads as before the first pass. Passes 2 to 5 reject and re-read readings, so
    this is set once at the start and never read off the files later."""
    used: set[int] = field(default_factory=set[int])
    """Every id placed on so far, the scope's used ids included. Only `place` adds to it."""
    pairing_off: bool = False
    """Whether the count-based passes (the absolute zip, the ordered zip, the numbered run) are turned off: an
    earlier pass found a sign the batch's count is off."""
    size_contradicted_prefixes: set[str] = field(default_factory=set[str])
    """Prefixes of files placed by size against their own release numbers. `pass_sequence_over_parse` won't place a run
    with one of these prefixes."""

    def __post_init__(self) -> None:
        self.non_extra_sequences = tuple(self.sequences_among(self.non_extras))
        self.facts = MappingProxyType(
            {
                file.name: NameFacts(
                    episode_by_title=file.episode_by_title,
                    parses_inside_window=bool(file.parsed_episodes.exact_ids),
                    extra=file.extra,
                    parsed_by_title_number=self.parsed_by_title_number(file),
                )
                for file in self.files
            }
        )

    @property
    def index(self) -> EpisodeIndex:
        """The scope's episode index, whose lookups (`EpisodeIndex.key_by_id` and the rest) the passes read."""

        return self.target.index

    @classmethod
    def start(cls, batch: PlacementBatch, target: PlacementTarget) -> Self:
        """Read every name and every parse of the torrent once, after widening each range key Sonarr read short.

        The stored map's ids that `scope` admits count as used from the start.
        """

        # On the scope, not just `used`: `nothing_judged` and the duplicate proof read `target.used` as held before this call.
        target = target.using(ep_id for ids in batch.file_map.values() for ep_id in ids)
        widened = {
            name: None if info is None else widen_range_key(name, info, target.index)
            for name, info in batch.parsed.items()
        }
        # The readings and every pass see the widened parses.
        parsed_by_name = {name: resolve_parse(widened.get(name), target) for name in batch.all_names}
        torrent = TorrentNames.of(batch, target, parsed_by_name)
        names_to_place = frozenset(batch.new_names)
        by_name = {
            name: TorrentFile(
                name=name,
                to_place=name in names_to_place,
                in_parsed=name in widened,
                parse=widened.get(name),
                name_parts=title.read,
                episode_by_title=title.episode_by_title,
                leading_title_ids=title.leading_title_ids,
                extra=title.extra,
                size_match=target.seadex.size_matches.get(name),
                saved_ids=batch.file_map.get(name),
                parsed_episodes=parsed_by_name[name],
            )
            for name, title in torrent.by_name.items()
        }
        return cls(
            target=target,
            episode_titles=torrent.episode_titles,
            files=tuple(by_name.values()),
            placing=tuple(by_name[name] for name in batch.new_names),
            non_extras=tuple(file for file in by_name.values() if not file.extra),
            all_parses_known=batch.all_parses_known,
            used=set(target.used),
        )

    def single_parse(self, file: TorrentFile) -> ParsedFileInfo | None:
        """The name's parse when there is one and it covers one episode, else None."""

        info = file.parse
        return None if info is None or spans_multiple(info, self.target) else info

    def parsed_by_title_number(self, file: TorrentFile) -> bool:
        """Whether Sonarr counted the file by its own title's number (`NameParts.episode_title_number`) and nothing confirms it.

        Only a parse whose every number is that count is held. The count stands beside a spelled-out key, behind
        words too few to be a title (`spells_title`) or that name an episode, when they nearly name the one episode it
        reads as (`EpisodeTitles.near_matches`), or, read as no episode, when it is an episode's absolute number.
        """

        name_parts = file.name_parts
        info = file.parse
        if name_parts.episode_title_number is None or info is None or name_parts.spells_season_episode:
            return False
        if not spells_title(name_parts.title_words):
            return False
        numbers = {*info.absolute_episode_numbers, *glued_numbers(info)}
        if numbers != {name_parts.episode_title_number} or file.episode_by_title is not None:
            return False
        if resolved := file.parsed_episodes.ids:
            return len(resolved) != 1 or not self.episode_titles.near_matches(resolved[0], name_parts.title_words)
        return name_parts.episode_title_number not in self.index.id_by_absolute

    def _exact_order(self) -> list[TorrentFile]:
        """The unplaced files in the exact pass's order: `ParsedEpisodes.rank`, then the higher `vN`, then batch order."""

        return sorted(self.unjudged_files(), key=lambda f: (f.parsed_episodes.rank, -f.name_parts.version))

    def exact_pass_placements(self) -> dict[TorrentFile, tuple[int, ...]]:
        """The files the exact pass places as the readings stand, in its order, each with its ids.

        A file is passed over when one of its ids is taken, by an earlier pick too, or when it is a runner-up for one.
        """

        used = set(self.used)
        picks: dict[TorrentFile, tuple[int, ...]] = {}
        for file in self._exact_order():
            ids = file.parsed_episodes.exact_ids
            if ids and used.isdisjoint(ids) and not file.next_in_line(ids):
                used.update(ids)
                picks[file] = ids
        return picks

    @property
    def has_episode_index(self) -> bool:
        """Whether the series map is non-empty. With an empty map, every verdict that needs it is refused."""

        return bool(self.target.id_by_key)

    @property
    def only_extras_judged(self) -> bool:
        """Whether no file other than an extra has a verdict yet."""

        return all(file.extra for file in self.files if file.judged)

    @property
    def nothing_judged(self) -> bool:
        """Whether nothing has spoken for the batch: `unjudged`, and the scope started with no used ids."""

        return self.only_extras_judged and not self.target.used

    def reject_each(self, files: Iterable[TorrentFile]) -> None:
        """Reject each file's reading (`TorrentFile.reject`)."""

        for file in files:
            file.reject()

    def unused_ids(self) -> list[int]:
        """The open ids: the scope's nonzero ids minus every id used so far, in scope order."""

        return [i for i in self.target.episode_ids if i and i not in self.used]

    def consecutive_unused(self) -> ConsecutiveEpisodes | None:
        """The open ids as a run window, else None. It needs at least two open ids.

        The ids must be consecutive episodes of one season, or else all carry consecutive absolute numbers (a
        special in between, or two seasons' cours). Failing both, specials whose numbers aren't consecutive form
        one when they are every episode of the title and aired in number order (`_whole_window_aired_in_turn`).
        The ids are sorted by their numbers, never taken in scope order.
        """

        unused_ids = self.unused_ids()
        if len(unused_ids) < 2:
            return None
        key_by_id = self.index.key_by_id
        keyed = sorted((key_by_id[ep_id], ep_id) for ep_id in unused_ids if ep_id in key_by_id)
        if len(keyed) != len(unused_ids):
            return None
        seasons = {key.season for key, _ in keyed}
        episodes = tuple(key.episode for key, _ in keyed)
        ids = tuple(ep_id for _, ep_id in keyed)
        if len(seasons) == 1 and is_consecutive(episodes):
            return ConsecutiveEpisodes(seasons.pop(), ids, episodes)
        absolute_of = self.index.absolute_of
        absolute = sorted((absolute_of[ep_id], ep_id) for ep_id in unused_ids if ep_id in absolute_of)
        if len(absolute) == len(unused_ids) and is_consecutive([number for number, _ in absolute]):
            return ConsecutiveEpisodes(
                None, tuple(ep_id for _, ep_id in absolute), tuple(number for number, _ in absolute)
            )
        if seasons == {0} and self._whole_window_aired_in_turn(ids):
            return ConsecutiveEpisodes(0, ids, episodes)
        return None

    def _whole_window_aired_in_turn(self, ids: Sequence[int]) -> bool:
        """Whether the open ids, in number order, are every episode of the title in the order they aired.

        None of the scope may be taken, and a scope that carries AniList's episode count must hold that many
        episodes: nothing says where in the title's order an episode Sonarr lacks falls. Each episode must have
        aired after the one before it. Two that share an air date are numbered in no order a release has to count
        them in.
        """

        target = self.target
        if len(ids) != len(target.episode_ids) or target.episode_count not in (None, len(ids)):
            return False
        aired = [self.index.by_id[ep_id].air_date for ep_id in ids]
        return all(first is not None and then is not None and first < then for first, then in pairwise(aired))

    def unjudged_files(self) -> list[TorrentFile]:
        """The files to place without a verdict yet, batch order."""

        return [file for file in self.placing if file.unjudged]

    def place(self, file: TorrentFile, ids: Sequence[int], verdict: PlacementVerdict) -> None:
        """Record a placement and mark its ids used."""

        file.placement = Placement(file.name, tuple(ids), verdict)
        self.used.update(ids)

    def place_if_unused(self, file: TorrentFile, ep_id: int, verdict: PlacementVerdict) -> None:
        """Place the file on the episode, or set it aside as `SKIPPED` when the episode is already used."""

        if ep_id in self.used:
            self.leave_unplaced(file, PlacementVerdict.NO_EPISODE_FOUND)
        else:
            self.place(file, (ep_id,), verdict)

    def pair_in_order(self, files: Sequence[TorrentFile], ids: Sequence[int], verdict: PlacementVerdict) -> bool:
        """Place the files onto the ids pairwise, or none when a file is titled as an episode other than its pair.

        A refused zip turns the count-based passes off: the count that paired them was wrong.
        """

        pairs = list(zip(files, ids, strict=True))
        if any(file.contradicts((ep_id,)) for file, ep_id in pairs):
            self.pairing_off = True
            return False
        for file, ep_id in pairs:
            self.place(file, (ep_id,), verdict)
        return True

    def leave_unplaced(self, file: TorrentFile, verdict: PlacementVerdict) -> None:
        """Record a verdict that carries no ids (held, excluded, or skipped)."""

        file.placement = Placement(file.name, (), verdict)

    def set_aside_order_only(self, file: TorrentFile, key: OrderOnlyKey) -> None:
        """Record `ORDER_ONLY` with the order and the key that read the file."""

        file.placement = Placement(file.name, (), PlacementVerdict.ORDER_ONLY, key)

    def set_aside_outside_window(self, file: TorrentFile, ids: tuple[int, ...]) -> None:
        """Record `OTHER_ORDER_OUTSIDE_WINDOW` with the episodes the order reads the file onto."""

        file.placement = Placement(file.name, (), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW, other_order_episodes=ids)

    def set_aside_each(self, files: Iterable[TorrentFile], verdict: PlacementVerdict) -> None:
        """Record one id-less verdict for each file."""

        for file in files:
            self.leave_unplaced(file, verdict)

    def sequences_among(self, files: Iterable[TorrentFile]) -> list[NumberedSequence]:
        """The numbered runs among `files`, formed by their release numbers, less the `absolute_disagrees` files."""

        by_prefix: dict[str, dict[int, list[TorrentFile]]] = {}
        for file in files:
            if (member := file.name_parts.member) is not None and not file.absolute_disagrees:
                by_prefix.setdefault(member.prefix, {}).setdefault(member.number, []).append(file)
        return [NumberedSequence.of(prefix, by_number) for prefix, by_number in by_prefix.items()]

    def placements(self) -> tuple[Placement, ...]:
        """The placements of the files to place, batch order. Each has one once the passes are done."""

        placed: list[Placement] = []
        for file in self.placing:
            assert file.placement is not None
            placed.append(file.placement)
        return tuple(placed)
