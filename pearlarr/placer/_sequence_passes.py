"""The passes that read, place, or leave unplaced a numbered sequence whole by its own name numbers.

They are passes 5, 6, 8, and 14 of the list in `_passes`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Self

from ._state import ConsecutiveEpisodes, NumberedSequence, PlacerState, TorrentFile
from ..file_names import Naming
from ..parsed_episodes import ParsedEpisodes, names_several_episodes
from ..placement_types import PlacementVerdict
from ..releases import EpisodeKey

# Episode titles select a sequence, or contradict one, only when at least this many members carry them.
MIN_TITLE_HITS = 2


@dataclass(frozen=True, slots=True)
class EpisodeTitleMatches:
    """How many sequence members carry only titles of the unused episodes, and how many only other titles.

    A member carries the titles its words after the number start with.
    """

    for_unused: int
    for_others: int

    @property
    def selects(self) -> bool:
        """Whether at least `MIN_TITLE_HITS` members carry unused episodes' titles, and more than carry other titles."""

        return self.for_unused >= MIN_TITLE_HITS and self.for_unused > self.for_others

    @property
    def contradicts(self) -> bool:
        """Whether at least `MIN_TITLE_HITS` members carry other titles and none carries an unused episode's title."""

        return self.for_others >= MIN_TITLE_HITS and not self.for_unused


@dataclass(frozen=True, slots=True)
class _SeasonFit:
    """The season a `1..N` sequence matches, and whether the specials season fits it just as well."""

    season: int
    fits_specials_too: bool


def pass_sequence_as_matched_season(state: PlacerState) -> None:
    """Read a `1..N` sequence Sonarr matched into one N-episode season as that season's episodes 1 to N (`_season_fit`).

    Only a sequence without a parsed season and episode qualifies, and Sonarr shifted some of its members onto
    specials. When it fits the specials equally (as many members matched onto an N-episode specials season), the
    parsed episodes are rejected and each file records both candidates in `ParsedEpisodes.either_ids`: the
    sequence-over-parse pass can still place them, and the episode-title pass only onto one of the two.
    """

    if not state.has_episode_index:
        return
    for sequence in state.sequences_among(state.unjudged_files()):
        fit = _season_fit(state, sequence)
        if fit is None:
            continue
        if fit.fits_specials_too:
            _set_season_or_specials(state, sequence, fit.season)
        else:
            _set_season_episodes(state, sequence, fit.season)


def _season_fit(state: PlacerState, sequence: NumberedSequence) -> _SeasonFit | None:
    """The season a `1..N` sequence without a parsed season and episode belongs to.

    Sonarr matched it into one season of exactly N episodes. Every member needs a parse and at most one resolved id.
    At least one member must be matched onto the specials, but no more than into the season. It fits the specials
    equally when as many went to each and the specials season also has N episodes. None for any other shape.
    """

    width = len(sequence.members)
    if not sequence.from_one(width):
        return None
    if any((info := file.parse) is None or info.episode_numbers for file in sequence.members):
        return None
    if any(len(file.parsed_episodes.ids) > 1 for file in sequence.members):
        return None
    read = [
        state.index.key_by_id[resolved[0]].season for file in sequence.members if (resolved := file.parsed_episodes.ids)
    ]
    seasons = {season for season in read if season != 0}
    if len(seasons) != 1:
        return None
    season = seasons.pop()
    if state.index.season_counts.get(season) != width:
        return None
    into_season = read.count(season)
    onto_specials = read.count(0)
    if not onto_specials or into_season < onto_specials:
        return None
    return _SeasonFit(
        season, fits_specials_too=into_season == onto_specials and state.index.season_counts.get(0) == width
    )


def _set_season_or_specials(state: PlacerState, sequence: NumberedSequence, season: int) -> None:
    """Reject the parsed episodes and record each file's episode under the season's and the specials' numbering."""

    for file in sequence.all_versions:
        keys = (EpisodeKey(season, file.name_number), EpisodeKey(0, file.name_number))
        ids = tuple(i for key in keys if (i := state.target.id_by_key.get(key)))
        file.parsed_episodes = replace(file.parsed_episodes, rejected=True, either_ids=ids)


def _set_season_episodes(state: PlacerState, sequence: NumberedSequence, season: int) -> None:
    """Read each sequence file as the season's episode of its name number, rejected where its title contradicts it."""

    for file in sequence.all_versions:
        ep_id = state.target.id_by_key.get(EpisodeKey(season, file.name_number))
        if ep_id:
            inside = (ep_id,) if state.target.admits(ep_id) else ()
            rejected = file.contradicts((ep_id,))
            file.parsed_episodes = ParsedEpisodes(
                ids=(ep_id,),
                in_window=inside,
                complete=True,
                from_matched=True,
                rejected=rejected,
                sonarr_agrees=True,
            )


@dataclass(frozen=True, slots=True)
class _ListedSpecials:
    """A batch of misnumbered specials and the listed specials it is as wide as."""

    sequence: NumberedSequence
    listed: frozenset[int]


def pass_misnumbered_specials(state: PlacerState) -> None:
    """Place a batch of misnumbered specials through the special aliases, or leave it all `MISNUMBERED`.

    The batch is one sequence of specials, one file per number, as wide as the listing, with a number that is no
    listed special (`_misnumbered_specials`). `_sole_specials_sequence` lists what else turns the pass off. The batch
    is numbered some other way (an older TVDB state, or TMDB's order), and its own numbers would put some files on the
    wrong specials, so only the aliases can place it. When the `special_aliases` map its numbers one to one onto the
    listed specials, and no title, size match, or file in the file map puts a member anywhere else
    (`_aliased_specials`), each file goes on its aliased special (`SPECIAL_ALIAS`, or `NO_EPISODE_FOUND` when used)
    and a superseded version is `DUPLICATE`. A file whose special is outside the window is `ALIAS_OUTSIDE_WINDOW`, a
    provisional verdict `window_placement` resolves across the windows.

    When the aliases can't place it, every file of the sequence is `MISNUMBERED` for a hand import and passes 7 to 15
    place none of it, unless some of the window's ids are already used (by the pending import's file map or an
    earlier window). Then the pass stands down and the later passes take the batch by its numbers, so on a window
    where the file map places a file off its aliased special the rest of the batch places by its numbers too.
    """

    misnumbered = _misnumbered_specials(state)
    if misnumbered is None:
        return
    specials = _aliased_specials(state, misnumbered)
    if specials is None:
        if state.nothing_judged:
            state.set_aside_each(_unjudged_of(misnumbered.sequence.all_versions), PlacementVerdict.MISNUMBERED)
        return
    for file in _unjudged_of(misnumbered.sequence.members):
        ep_id = specials[file]
        if state.target.admits(ep_id):
            state.place_if_unused(file, ep_id, PlacementVerdict.SPECIAL_ALIAS)
        else:
            state.leave_unplaced(file, PlacementVerdict.ALIAS_OUTSIDE_WINDOW)
    state.set_aside_each(_unjudged_of(misnumbered.sequence.superseded), PlacementVerdict.DUPLICATE)


def _unjudged_of(files: Iterable[TorrentFile]) -> list[TorrentFile]:
    """The files still to place, in the order given. File map, gone, and already judged ones drop out."""

    return [file for file in files if file.unjudged]


def _aliased_specials(state: PlacerState, misnumbered: _ListedSpecials) -> dict[TorrentFile, int] | None:
    """Each member -> its aliased special, or None if the aliases don't map the batch one to one onto the listing.

    Also None when a member's title, size match, or file map ids put it anywhere but its aliased special.
    """

    listing = state.target.seadex
    members = frozenset(misnumbered.sequence.members)
    specials: dict[TorrentFile, int] = {}
    # Every version is checked, superseded ones too.
    for file in misnumbered.sequence.all_versions:
        tvdb_number = listing.special_aliases.get(file.name_number)
        ep_id = None if tvdb_number is None else state.target.id_by_key.get(EpisodeKey(0, tvdb_number))
        if ep_id is None or file.contradicts((ep_id,)):
            return None
        if file.size_match not in (None, ep_id) or file.saved_ids not in (None, (ep_id,)):
            return None
        if file in members:
            specials[file] = ep_id
    # The map is one to one when it hits every listed special.
    return specials if frozenset(specials.values()) == misnumbered.listed else None


def _misnumbered_specials(state: PlacerState) -> _ListedSpecials | None:
    """The batch as one listing-wide batch of specials with a number that isn't a listed special, else None.

    For specials the curated listing (`PlacementTarget.specials_listing`) beats the file names. In a regular season,
    a number outside the listing leaves the file outside the window as usual.
    """

    listed = state.target.specials_listing()
    if listed is None or (sequence := _sole_specials_sequence(state, len(listed))) is None:
        return None
    if all(state.target.id_by_key.get(EpisodeKey(0, n)) in listed for n in sequence.numbers):
        return None
    return _ListedSpecials(sequence, listed)


def _sole_specials_sequence(state: PlacerState, width: int) -> NumberedSequence | None:
    """The batch as one sequence of `width` distinct `S00Exx` names, each parsed by Sonarr as its own number's special.

    File map and gone files count as members. None if a parse is unknown, the pairing passes are off, a file other
    than an extra already has a verdict, the batch holds another sequence or a file outside the sequence, a number
    repeats or spans several episodes, or every member's title confirms its number.
    """

    if not state.all_parses_read or state.pairing_off or not state.only_extras_judged:
        return None
    sequences = state.non_extra_sequences
    if len(sequences) != 1 or set(sequences[0].all_versions) != set(state.non_extras):
        return None
    sequence = sequences[0]
    if len(sequence.numbers) != width or len(set(sequence.numbers)) != width:
        return None
    confirmed = 0
    for file in sequence.all_versions:
        info = state.single_parse(file)
        if info is None or info.season_number != 0 or info.episode_numbers != (file.name_number,):
            return None
        # One title confirms its own file alone: a batch it does not cover whole stays judged by the listing.
        confirmed += file.confirms(file.parsed_episodes.ids)
    return None if confirmed == len(sequence.all_versions) else sequence


class _FitKind(IntEnum):
    """How well a sequence fits the unused episodes, best first. The sequence-over-parse pass ranks candidates by it."""

    ONE_TO_N = 1
    """Numbered `1..N`, where N is the number of unused episodes."""
    EPISODE_NUMBERS = 2
    """Numbered exactly as the unused episodes' numbers."""
    WHOLE_SEASON = 3
    """Numbered `1..N` for the whole season, when the unused episodes are only part of that season."""
    ANY_CONSECUTIVE = 4
    """Any N consecutive numbers (`_SequenceFit.at_any_start`)."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A sequence that fits the unused episodes: its position among the sequences, best fit kind, and title matches."""

    position: int
    sequence: NumberedSequence
    fit_kind: _FitKind
    title_matches: EpisodeTitleMatches


@dataclass(frozen=True, slots=True, eq=False)
class _SequenceFit:
    """The consecutive unused episodes and the unjudged sequences the sequence-over-parse pass fits to them.

    `unused` and `sequences` are taken once when the pass starts. A sequence holds the live file objects, so every
    check sees the latest parsed episodes and verdicts.
    """

    state: PlacerState
    unused: ConsecutiveEpisodes
    sequences: tuple[NumberedSequence, ...]
    """The numbered sequences among the unjudged files, in sequence order."""

    @classmethod
    def of(cls, state: PlacerState) -> Self | None:
        """The fit when the unused ids are consecutive episodes, else None."""

        unused = state.consecutive_unused()
        return None if unused is None else cls(state, unused, tuple(state.sequences_among(state.unjudged_files())))

    def candidates(self) -> list[_Candidate]:
        """Every sequence that fits the unused episodes, at its best fit kind, sorted by that then by sequence order."""

        width = len(self.unused.ids)
        fit_kinds = {
            _FitKind.ONE_TO_N: [sequence for sequence in self.sequences if sequence.from_one(width)],
            _FitKind.EPISODE_NUMBERS: [
                sequence for sequence in self.sequences if sequence.numbers == self.unused.numbers
            ],
            _FitKind.WHOLE_SEASON: self.whole_season(),
            _FitKind.ANY_CONSECUTIVE: self.at_any_start(),
        }
        fit_kind_of: dict[NumberedSequence, _FitKind] = {}
        for tier, fits in fit_kinds.items():
            for sequence in fits:
                fit_kind_of.setdefault(sequence, tier)
        position = {sequence: index for index, sequence in enumerate(self.sequences)}
        return [
            _Candidate(position[sequence], sequence, tier, self.title_matches(sequence))
            for sequence, tier in fit_kind_of.items()
        ]

    def title_matches(self, sequence: NumberedSequence) -> EpisodeTitleMatches:
        """Count the members carrying only titles of the unused episodes, and those carrying only other titles.

        A member carries the titles its words after the number start with. Words matching titles on both sides count
        for neither.
        """

        inside = outside = 0
        for file in sequence.members:
            sides = {ep_id in self.unused.id_set for ep_id in file.leading_title_ids}
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return EpisodeTitleMatches(inside, outside)

    def whole_season(self) -> list[NumberedSequence]:
        """The sequences numbered `1..N` for the unused episodes' whole season, when there are fewer than N of them."""

        season = self.unused.season
        if season is None:
            return []
        count = self.state.index.season_counts.get(season, 0)
        return (
            [sequence for sequence in self.sequences if sequence.from_one(count)]
            if count > len(self.unused.ids)
            else []
        )

    def whole_season_final_unplaced(self) -> bool:
        """Whether a whole-season sequence cannot be trusted to number these unused episodes.

        True when they span seasons or are specials, and for a season whose numbering has a gap.
        """

        season = self.unused.season
        if season is None or self.unused.is_specials:
            return True
        count = self.state.index.season_counts.get(season, 0)
        return any(EpisodeKey(season, n) not in self.state.target.id_by_key for n in range(1, count + 1))

    def at_any_start(self) -> list[NumberedSequence]:
        """The sequences of consecutive numbers as wide as the unused episodes, at any start.

        No member may have complete parsed episodes. Empty once any id is used (by the file map, an earlier window, or
        an earlier pass).
        """

        if self.state.used:
            return []
        width = len(self.unused.ids)
        return [
            sequence
            for sequence in self.sequences
            if sequence.consecutive(width) and not any(file.parsed_episodes.complete for file in sequence.members)
        ]

    def seasons_parsed(self, sequence: NumberedSequence) -> set[int]:
        """The seasons Sonarr parsed the sequence's members into."""

        key_by_id = self.state.index.key_by_id
        return {key_by_id[ep_id].season for file in sequence.members for ep_id in file.parsed_episodes.ids}

    def parsed_into_another_season(self, sequence: NumberedSequence) -> bool:
        """Whether a member is parsed into a season other than the unused episodes', against the sequence's numbers."""

        return any(read != self.unused.season for read in self.seasons_parsed(sequence))

    def final_outside_window(self, sequence: NumberedSequence) -> bool:
        """Whether Sonarr matched every member to its own distinct episode outside the window (another season or cour).

        A sequence that fits a season and the specials equally counts as outside when none of its candidate episodes
        is in the window.
        """

        parsed_by_name = [file.parsed_episodes for file in sequence.members]
        if all(r.either_ids is not None for r in parsed_by_name):
            return not any(i in self.state.target.real_ids for r in parsed_by_name for i in r.either_ids or ())
        if not all(r.wholly_outside and r.sonarr_agrees and len(r.ids) == 1 for r in parsed_by_name):
            return False
        return len({r.ids[0] for r in parsed_by_name}) == len(parsed_by_name)

    def final_unplaced(self, sequence: NumberedSequence) -> bool:
        """Whether the batch shows the sequence cannot take all the unused episodes."""

        state, unused_set = self.state, self.unused.id_set
        for file in sequence.members:
            info = file.parse
            # A missing parse, or a name that itself names several episodes, blocks the sequence. A multi-episode
            # Sonarr match does not.
            if info is None or names_several_episodes(info):
                return True
            parsed = file.parsed_episodes
            spelled_outside_unused = not parsed.from_matched and parsed.complete and unused_set.isdisjoint(parsed.ids)
            # A member whose spelled-out season and episode lie outside the unused episodes blocks the sequence,
            # except when those are specials.
            if spelled_outside_unused and not self.unused.is_specials:
                return True
        mates = set(sequence.all_versions)
        for file in state.unjudged_files():
            if file in mates:
                continue
            parsed = file.parsed_episodes
            if not unused_set.isdisjoint(parsed.either_ids or ()):
                return True
            # A file next in line already lost those episodes before the sequence was asked: it blocks nothing.
            if not parsed.usable or file.next_in_line(parsed.ids):
                continue
            if parsed.ids and unused_set.issuperset(parsed.ids):
                return True
        return False


def pass_sequence_over_parse(state: PlacerState) -> None:
    """Place one numbered sequence onto the unused episodes by its own numbers when Sonarr's parse of it doesn't add up.

    The unused ids must be consecutive episodes. Candidates are the unjudged sequences that fit a `_FitKind`.
    Tie-breaks, in order: drop sequences Sonarr matched whole outside the window (unless all are), then keep the one
    sequence the titles select, the one an AniList name of the title points at, or the best fit kind. The sequence's
    own numbers place its members onto the unused episodes (`SEQUENCE_OVER_PARSE`), unless Sonarr's parse already
    puts each member on its own unused episode. A sequence isn't placed when a file sharing its prefix was placed by
    size against that file's numbers (`PlacerState.size_contradicted_prefixes`). With any parse unknown, the pick is
    `WAITING_ON_PARSE`. It is blocked by `_SequenceFit.final_unplaced`, `whole_season_final_unplaced` for a
    whole-season pick, a member parsed into another season, a member whose numbers disagree by position
    (`TorrentFile.numbers_disagree_by_position`), titles that contradict it (`EpisodeTitleMatches.contradicts`), or an
    AniList name of the title pointing at another sequence.
    """

    if not state.has_episode_index or (fit := _SequenceFit.of(state)) is None:
        return
    candidates = fit.candidates()
    # A whole-season sequence parsed into another season is blocked, picked or not. Parsed whole into one other
    # season, it belongs there (its files end up OUTSIDE_WINDOW). Otherwise its parsed episodes are rejected.
    parsed_into_another_season = [
        c.sequence
        for c in candidates
        if c.fit_kind is _FitKind.WHOLE_SEASON and fit.parsed_into_another_season(c.sequence)
    ]
    # Rejections first: a sequence with rejected parsed episodes stays unjudged, which `unjudged_positions` then counts.
    for sequence in parsed_into_another_season:
        if not (fit.final_outside_window(sequence) and len(fit.seasons_parsed(sequence)) == 1):
            state.reject_each(sequence.all_versions)
    unjudged_positions = frozenset(
        index for index, sequence in enumerate(fit.sequences) if not fit.final_outside_window(sequence)
    )
    naming = Naming.of([sequence.prefix for sequence in fit.sequences], state.target.title_names)
    # Several candidates turn the pairing passes off, even when a tie-break picks one.
    state.pairing_off |= len(candidates) > 1
    chosen = _sole_candidate(candidates, unjudged_positions, naming)
    if chosen is None:
        return
    sequence = chosen.sequence
    if sequence.prefix in state.size_contradicted_prefixes:
        return
    if not state.all_parses_read:
        state.set_aside_each(sequence.all_versions, PlacementVerdict.WAITING_ON_PARSE)
        return
    whole_season = chosen.fit_kind is _FitKind.WHOLE_SEASON
    final_unplaced = (
        fit.final_unplaced(sequence)
        or (whole_season and fit.whole_season_final_unplaced())
        or sequence in parsed_into_another_season
        or any(file.numbers_disagree_by_position for file in sequence.members)
        or chosen.title_matches.contradicts
        or (naming is not None and naming.names_another(chosen.position, unjudged_positions))
    )
    if final_unplaced:
        state.pairing_off = True
        return
    unused = fit.unused
    # A whole-season sequence over part of its season places only the unused episodes' numbers. Its other members are
    # OUTSIDE_WINDOW.
    files = unused.files_of(sequence) if whole_season else sequence.members
    # Sonarr already parses each member onto its own unused episode: the exact pass places them.
    exact = [file.parsed_episodes.exact_ids for file in files]
    if all(len(ids) == 1 and ids[0] in unused.id_set for ids in exact) and len(set(exact)) == len(exact):
        return
    if not state.pair_in_order(files, unused.ids, PlacementVerdict.SEQUENCE_OVER_PARSE):
        return
    state.set_aside_each([file for file in sequence.members if file not in files], PlacementVerdict.OUTSIDE_WINDOW)
    state.set_aside_each(sequence.superseded, PlacementVerdict.DUPLICATE)


def _sole_candidate(
    candidates: Sequence[_Candidate], unjudged_positions: frozenset[int], naming: Naming | None
) -> _Candidate | None:
    """The one candidate the tie-breaks leave, else None (see `pass_sequence_over_parse`)."""

    if len(candidates) > 1:
        candidates = [c for c in candidates if c.position in unjudged_positions] or candidates
    if len(candidates) > 1 and len(selected := [c for c in candidates if c.title_matches.selects]) == 1:
        candidates = selected
    if (
        len(candidates) > 1
        and naming is not None
        and (named := naming.sole_among(c.position for c in candidates)) is not None
    ):
        candidates = [c for c in candidates if c.position == named]
    if len(candidates) > 1:
        top = min(c.fit_kind for c in candidates)
        candidates = [c for c in candidates if c.fit_kind is top]
    return candidates[0] if len(candidates) == 1 else None


def pass_sequence(state: PlacerState) -> None:
    """Pair in order the one `1..N` sequence among the files Sonarr found no episode for onto N consecutive unused ids.

    Unlike pairing in name order, this works beside files that other passes placed. A file qualifies when its parsed
    episodes resolved nothing, its name carries no `SxxExx` season and episode, and it is not multi-episode. It does
    nothing without an episode index, with any parse unknown, or with the pairing passes off.
    """

    if not state.has_episode_index or not state.all_parses_read or state.pairing_off:
        return
    unused = state.consecutive_unused()
    if unused is None:
        return
    without_number = [
        file
        for file in state.unjudged_files()
        if not file.parsed_episodes.ids and (info := state.single_parse(file)) is not None and not info.episode_numbers
    ]
    fits = [sequence for sequence in state.sequences_among(without_number) if sequence.from_one(len(unused.ids))]
    if len(fits) != 1:
        return
    sequence = fits[0]
    if state.pair_in_order(sequence.members, unused.ids, PlacementVerdict.SEQUENCE):
        state.set_aside_each(sequence.superseded, PlacementVerdict.DUPLICATE)
