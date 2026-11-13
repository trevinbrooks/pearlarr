"""The passes that read, place, or set aside a numbered run whole by its own release numbers.

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

# Episode titles pick a run, or refuse one, only when at least this many members carry them.
MIN_TITLE_HITS = 2


@dataclass(frozen=True, slots=True)
class EpisodeTitleMatches:
    """How many run members' tails start with only window titles, and how many with only titles outside it."""

    in_window: int
    outside_window: int

    @property
    def selects(self) -> bool:
        """Whether at least `MIN_TITLE_HITS` members carry window titles, and more than carry outside titles."""

        return self.in_window >= MIN_TITLE_HITS and self.in_window > self.outside_window

    @property
    def contradicts(self) -> bool:
        """Whether at least `MIN_TITLE_HITS` members carry outside titles and none carries a window title."""

        return self.outside_window >= MIN_TITLE_HITS and not self.in_window


@dataclass(frozen=True, slots=True)
class _SeasonFit:
    """The season a `1..N` run matches, and whether the specials season matches it just as well (a tie)."""

    season: int
    fits_specials_too: bool


def pass_sequence_as_matched_season(state: PlacerState) -> None:
    """Read a `1..N` run Sonarr matched into one N-episode season as that season's episodes 1 to N (`_season_fit`).

    Only a run without own keys qualifies, and Sonarr shifted some of its members onto specials. In a tie (as many
    members matched onto an N-episode specials season), the readings are rejected and each file records both
    candidates in `ParsedEpisodes.either_ids`: the release-run pass can still place them, and the episode-title pass only onto
    one of the two.
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
    """The season a `1..N` run without own keys belongs to: Sonarr matched it into one season of exactly N episodes.

    Every member needs a parse and at most one resolved id. At least one member must be matched onto the
    specials, but no more than into the season. It is a tie when as many went to each and the specials season
    also has N episodes. None for any other shape.
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
    """Reject the run's readings and record each file's episode under the season's and the specials' numbering."""

    for file in sequence.all_versions:
        keys = (EpisodeKey(season, file.name_number), EpisodeKey(0, file.name_number))
        ids = tuple(i for key in keys if (i := state.target.id_by_key.get(key)))
        file.parsed_episodes = replace(file.parsed_episodes, rejected=True, either_ids=ids)


def _set_season_episodes(state: PlacerState, sequence: NumberedSequence, season: int) -> None:
    """Read each run file as the season's episode of its release number, rejected where its title disagrees."""

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
    """A misnumbered specials pack and the listed specials it is as wide as."""

    sequence: NumberedSequence
    listed: frozenset[int]


def pass_misnumbered_specials(state: PlacerState) -> None:
    """Place a misnumbered specials pack through the listing's special aliases, or refuse it whole.

    The pack is the batch as one run of specials, one file per number, as wide as the listing, with a number that
    is no listed special (`_misnumbered_specials`). `_sole_specials_sequence` lists what else turns the pass off. The pack is
    numbered some other way (an older TVDB state, or TMDB's order), and its own numbers would put some files on the
    wrong specials, so only the aliases can place it. When the listing's `special_aliases` map its numbers one to
    one onto the listed specials, and no title, size match, or seeded file puts a member anywhere else
    (`_aliased_specials`), each file goes on its aliased special (`SPECIAL_ALIAS`, or `SKIPPED` when taken) and a
    superseded version is `DUPLICATE`. A file whose special is outside the window is `ALIAS_OUTSIDE_WINDOW`, a claim
    `window_placement` settles across the windows.

    When the aliases can't place it, every file of the run is `MISNUMBERED` for a hand import and passes 7 to 15
    place none of it, unless some of the window's ids are already used (by the stored record's map or an earlier
    window). Then the pass stands down and the later passes take the pack by its numbers, so on the window holding
    a seed off its aliased special the rest of the pack places by its numbers too.
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
    """The files still to place, in the order given. Seeded, gone, and already judged ones drop out."""

    return [file for file in files if file.unjudged]


def _aliased_specials(state: PlacerState, misnumbered: _ListedSpecials) -> dict[TorrentFile, int] | None:
    """Each member -> its aliased special, or None if the aliases don't map the pack one to one onto the listing.

    Also None when a member's title, size match, or seed puts it anywhere but its aliased special.
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
    """The batch as one listing-wide specials pack with a number that isn't a listed special, else None.

    For specials the curated listing (`PlacementTarget.seadex`) beats the file names. In a regular season, a number
    outside the listing leaves the file foreign as usual.
    """

    listed = state.target.specials_listing()
    if listed is None or (sequence := _sole_specials_sequence(state, len(listed))) is None:
        return None
    if all(state.target.id_by_key.get(EpisodeKey(0, n)) in listed for n in sequence.numbers):
        return None
    return _ListedSpecials(sequence, listed)


def _sole_specials_sequence(state: PlacerState, width: int) -> NumberedSequence | None:
    """The batch as one run of `width` distinct `S00Exx` names, each read by Sonarr as its own number's special.

    Seeded and gone files count as members. None if a parse is unknown, the count-based passes are off, a file
    other than an extra already has a verdict, the batch holds another run or a file outside the run, a number
    repeats or spans several episodes, or every member's title confirms its number.
    """

    if not state.all_parses_known or state.pairing_off or not state.only_extras_judged:
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
        # One title vouches for its own file alone: a pack it does not cover whole stays judged by the listing.
        confirmed += file.confirms(file.parsed_episodes.ids)
    return None if confirmed == len(sequence.all_versions) else sequence


class _FitKind(IntEnum):
    """How well a numbered run fits the run window, best first. The release-run pass ranks candidates by it."""

    ONE_TO_N = 1
    """Numbered `1..N`, where N is the window's width."""
    EPISODE_NUMBERS = 2
    """Numbered exactly as the window's episode numbers."""
    WHOLE_SEASON = 3
    """Numbered `1..N` for the whole season, when the window holds only part of that season."""
    ANY_CONSECUTIVE = 4
    """Any N consecutive numbers (`_SequenceFit.at_any_start`)."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A run that could place onto the window: its position among the runs, its best tier, and its title evidence."""

    position: int
    sequence: NumberedSequence
    fit_kind: _FitKind
    title_matches: EpisodeTitleMatches


@dataclass(frozen=True, slots=True, eq=False)
class _SequenceFit:
    """The run window and the unplaced runs the release-run pass fits to it.

    `window` and `runs` are taken once when the pass starts. A run holds the live file records, so every check
    sees the latest readings and verdicts.
    """

    state: PlacerState
    window: ConsecutiveEpisodes
    sequences: tuple[NumberedSequence, ...]
    """The numbered runs among the unplaced files, in run order."""

    @classmethod
    def of(cls, state: PlacerState) -> Self | None:
        """The fit when the open ids form a run window, else None."""

        window = state.consecutive_unused()
        return None if window is None else cls(state, window, tuple(state.sequences_among(state.unjudged_files())))

    def candidates(self) -> list[_Candidate]:
        """Every run that could place onto the window, at the best tier it fits, sorted by tier then by run order."""

        width = len(self.window.ids)
        fit_kinds = {
            _FitKind.ONE_TO_N: [sequence for sequence in self.sequences if sequence.from_one(width)],
            _FitKind.EPISODE_NUMBERS: [
                sequence for sequence in self.sequences if sequence.numbers == self.window.numbers
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
        """Count the members whose tail starts with only window titles, and those with only outside titles.

        A tail matching titles on both sides counts for neither.
        """

        inside = outside = 0
        for file in sequence.members:
            sides = {ep_id in self.window.id_set for ep_id in file.leading_title_ids}
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return EpisodeTitleMatches(inside, outside)

    def whole_season(self) -> list[NumberedSequence]:
        """The runs numbered `1..N` for the window's whole season, when the window holds fewer than N ids."""

        season = self.window.season
        if season is None:
            return []
        count = self.state.index.season_counts.get(season, 0)
        return (
            [sequence for sequence in self.sequences if sequence.from_one(count)]
            if count > len(self.window.ids)
            else []
        )

    def whole_season_final_unplaced(self) -> bool:
        """Whether a whole-season run cannot be trusted to number this window.

        True for a window spanning seasons, for a specials window, and for a season whose numbering has a gap.
        """

        season = self.window.season
        if season is None or self.window.is_specials:
            return True
        count = self.state.index.season_counts.get(season, 0)
        return any(EpisodeKey(season, n) not in self.state.target.id_by_key for n in range(1, count + 1))

    def at_any_start(self) -> list[NumberedSequence]:
        """The runs of consecutive numbers as wide as the window, at any start, where no member has a complete reading.

        Empty once any id is used (by a seeded file, an earlier window, or an earlier pass).
        """

        if self.state.used:
            return []
        width = len(self.window.ids)
        return [
            sequence
            for sequence in self.sequences
            if sequence.consecutive(width) and not any(file.parsed_episodes.complete for file in sequence.members)
        ]

    def seasons_parsed(self, sequence: NumberedSequence) -> set[int]:
        """The seasons Sonarr read the run's members into."""

        key_by_id = self.state.index.key_by_id
        return {key_by_id[ep_id].season for file in sequence.members for ep_id in file.parsed_episodes.ids}

    def parsed_outside_window(self, sequence: NumberedSequence) -> bool:
        """Whether a member's reading lands in a season other than the window's, which disputes the run's numbers."""

        return any(read != self.window.season for read in self.seasons_parsed(sequence))

    def final_outside_window(self, sequence: NumberedSequence) -> bool:
        """Whether Sonarr matched every member to its own distinct episode outside the scope (another season or cour).

        A tied run counts as elsewhere when none of its candidate episodes is in the scope.
        """

        parsed_by_name = [file.parsed_episodes for file in sequence.members]
        if all(r.either_ids is not None for r in parsed_by_name):
            return not any(i in self.state.target.real_ids for r in parsed_by_name for i in r.either_ids or ())
        if not all(r.wholly_outside and r.sonarr_agrees and len(r.ids) == 1 for r in parsed_by_name):
            return False
        return len({r.ids[0] for r in parsed_by_name}) == len(parsed_by_name)

    def final_unplaced(self, sequence: NumberedSequence) -> bool:
        """Whether the batch shows the run cannot own the whole window."""

        state, window_set = self.state, self.window.id_set
        for file in sequence.members:
            info = file.parse
            # A missing parse, or a name that itself claims several episodes, refuses the run. A multi-episode
            # Sonarr match does not.
            if info is None or names_several_episodes(info):
                return True
            parsed = file.parsed_episodes
            spelled_outside_window = not parsed.from_matched and parsed.complete and window_set.isdisjoint(parsed.ids)
            # A member whose own key lies outside the window refuses the run, except on a specials window.
            if spelled_outside_window and not self.window.is_specials:
                return True
        mates = set(sequence.all_versions)
        for file in state.unjudged_files():
            if file in mates:
                continue
            parsed = file.parsed_episodes
            if not window_set.isdisjoint(parsed.either_ids or ()):
                return True
            # A runner-up's claim on the window lost before the run was asked: it refuses nothing.
            if not parsed.usable or file.next_in_line(parsed.ids):
                continue
            if parsed.ids and window_set.issuperset(parsed.ids):
                return True
        return False


def pass_sequence_over_parse(state: PlacerState) -> None:
    """Place one numbered run onto the run window by its own numbers when Sonarr's reading of it does not add up.

    The open ids must form a run window. Candidates are the unplaced runs that fit a `_FitKind`. Tie-breaks, in order:
    drop runs Sonarr matched whole outside the scope (unless all are), then keep the one run the titles select, the
    one an AniList title names, or the best tier. The run's own numbers place its members onto the window
    (`SEQUENCE_OVER_PARSE`), unless Sonarr's reading already puts each member on its own window episode. A run isn't placed
    when a file sharing its prefix was placed by size against that file's numbers (`PlacerState.size_contradicted_prefixes`).
    With any parse unknown, the pick is `WAITING_ON_PARSE`. It is refused by `_SequenceFit.final_unplaced`, `whole_season_final_unplaced` for a
    covering pick, a dispute, a member its position disputes (`TorrentFile.numbers_disagree_by_position`), a title veto
    (`EpisodeTitleMatches.contradicts`), or an AniList title naming another run.
    """

    if not state.has_episode_index or (fit := _SequenceFit.of(state)) is None:
        return
    candidates = fit.candidates()
    # A covering run read into another season is disputed, picked or not. Read whole into one other season, it
    # belongs there (its files end up FOREIGN). Otherwise its readings are rejected.
    parsed_outside_window = [
        c.sequence for c in candidates if c.fit_kind is _FitKind.WHOLE_SEASON and fit.parsed_outside_window(c.sequence)
    ]
    # Vetoes first: a rejected reading leaves its run unsettled, which `unjudged_positions` then counts as open.
    for sequence in parsed_outside_window:
        if not (fit.final_outside_window(sequence) and len(fit.seasons_parsed(sequence)) == 1):
            state.reject_each(sequence.all_versions)
    unjudged_positions = frozenset(
        index for index, sequence in enumerate(fit.sequences) if not fit.final_outside_window(sequence)
    )
    naming = Naming.of([sequence.prefix for sequence in fit.sequences], state.target.title_names)
    # Several candidates turn the count-based passes off, even when a tie-break picks one.
    state.pairing_off |= len(candidates) > 1
    chosen = _sole_candidate(candidates, unjudged_positions, naming)
    if chosen is None:
        return
    sequence = chosen.sequence
    if sequence.prefix in state.size_contradicted_prefixes:
        return
    if not state.all_parses_known:
        state.set_aside_each(sequence.all_versions, PlacementVerdict.WAITING_ON_PARSE)
        return
    whole_season = chosen.fit_kind is _FitKind.WHOLE_SEASON
    final_unplaced = (
        fit.final_unplaced(sequence)
        or (whole_season and fit.whole_season_final_unplaced())
        or sequence in parsed_outside_window
        or any(file.numbers_disagree_by_position for file in sequence.members)
        or chosen.title_matches.contradicts
        or (naming is not None and naming.names_another(chosen.position, unjudged_positions))
    )
    if final_unplaced:
        state.pairing_off = True
        return
    window = fit.window
    # A whole-season run over a partial window places only the window's numbers. Its other members are FOREIGN.
    files = window.files_of(sequence) if whole_season else sequence.members
    # Sonarr already reads each member onto its own window episode: the exact pass places them.
    exact = [file.parsed_episodes.exact_ids for file in files]
    if all(len(ids) == 1 and ids[0] in window.id_set for ids in exact) and len(set(exact)) == len(exact):
        return
    if not state.pair_in_order(files, window.ids, PlacementVerdict.SEQUENCE_OVER_PARSE):
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
    """Zip the one `1..N` run among the files Sonarr found no episode for onto a run window of N ids.

    Unlike the ordered zip, this works beside files that other passes placed. A file qualifies when its reading
    resolved nothing, its name carries no `SxxExx` key, and it is not multi-episode. Skipped without a series map,
    with any parse unknown, or with the count-based passes off.
    """

    if not state.has_episode_index or not state.all_parses_known or state.pairing_off:
        return
    window = state.consecutive_unused()
    if window is None:
        return
    blind = [
        file
        for file in state.unjudged_files()
        if not file.parsed_episodes.ids and (info := state.single_parse(file)) is not None and not info.episode_numbers
    ]
    fits = [sequence for sequence in state.sequences_among(blind) if sequence.from_one(len(window.ids))]
    if len(fits) != 1:
        return
    sequence = fits[0]
    if state.pair_in_order(sequence.members, window.ids, PlacementVerdict.SEQUENCE):
        state.set_aside_each(sequence.superseded, PlacementVerdict.DUPLICATE)
