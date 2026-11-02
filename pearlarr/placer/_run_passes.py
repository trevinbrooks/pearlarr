"""The passes that read, place, or set aside a numbered run whole by its own release numbers.

They are passes 5, 6, 8, and 14 of the list in `_passes`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Self

from ._state import Placer, RunWindow, TorrentFile
from ..episode_titles import TitleEvidence
from ..parse_reading import Reading, claims_several_episodes
from ..placement_types import PlacementVerdict
from ..release_names import Naming, NumberedRun, RunMember, runs_from_one, runs_with_numbers
from ..seadex_types import EpisodeKey


class _SeasonFit(NamedTuple):
    """The season a `1..N` run matches, and whether the specials season matches it just as well (a tie)."""

    season: int
    tied: bool


def pass_reread_season_runs(state: Placer) -> None:
    """Read a `1..N` run Sonarr matched into one N-episode season as that season's episodes 1 to N (`_season_fit`).

    Only a run without own keys qualifies. In a tie (as many members matched onto an N-episode specials season),
    the readings are rejected and each file records both candidates in `Reading.tied`: the release-run pass can
    still place them, and the episode-title pass only onto one of the two.
    """

    if not state.map_known:
        return
    for run in state.runs(state.remaining()):
        fit = _season_fit(state, run)
        if fit is None:
            continue
        if fit.tied:
            _tie(state, run, fit.season)
        else:
            _reread(state, run, fit.season)


def _season_fit(state: Placer, run: NumberedRun) -> _SeasonFit | None:
    """The season a `1..N` run without own keys belongs to: Sonarr matched it into one season of exactly N episodes.

    Every member needs a parse and at most one resolved id. At least one member must be matched onto the
    specials, but no more than into the season. It is a tie when as many went to each and the specials season
    also has N episodes. None for any other shape.
    """

    width = len(run.members)
    if run.numbers != tuple(range(1, width + 1)):
        return None
    files = state.files_of(run.members)
    if any((info := file.parsed) is None or info.episode_numbers for file in files):
        return None
    if any(len(file.reading.resolved) > 1 for file in files):
        return None
    read = [state.facts.key_by_id[resolved[0]].season for file in files if (resolved := file.reading.resolved)]
    seasons = {season for season in read if season != 0}
    if len(seasons) != 1:
        return None
    season = seasons.pop()
    if state.facts.season_counts.get(season) != width:
        return None
    into_season = read.count(season)
    onto_specials = read.count(0)
    if not onto_specials or into_season < onto_specials:
        return None
    return _SeasonFit(season, into_season == onto_specials and state.facts.season_counts.get(0) == width)


def _tie(state: Placer, run: NumberedRun, season: int) -> None:
    """Reject the run's readings and record each file's episode under the season's and the specials' numbering."""

    for numbered, file in zip(run.whole, state.files_of(run.whole), strict=True):
        keys = (EpisodeKey(season, numbered.number), EpisodeKey(0, numbered.number))
        ids = tuple(i for key in keys if (i := state.scope.id_by_key.get(key)))
        file.reading = file.reading._replace(vetoed=True, tied=ids)


def _reread(state: Placer, run: NumberedRun, season: int) -> None:
    """Read each run file as the season's episode of its release number, rejected where its title disagrees."""

    for numbered, file in zip(run.whole, state.files_of(run.whole), strict=True):
        ep_id = state.scope.id_by_key.get(EpisodeKey(season, numbered.number))
        if ep_id:
            inside = (ep_id,) if state.scope.admits(ep_id) else ()
            vetoed = file.contradicts((ep_id,))
            file.reading = Reading((ep_id,), inside, complete=True, borrowed=True, vetoed=vetoed, corroborated=True)


class _ListedPack(NamedTuple):
    """A misnumbered specials pack and the listed specials it is as wide as."""

    run: NumberedRun
    listed: frozenset[int]


def pass_misnumbered_pack(state: Placer) -> None:
    """Place a misnumbered specials pack through the listing's special aliases, or refuse it whole.

    The pack's own numbers would put some files on the wrong specials, so only the aliases can place it
    (`_aliased_specials`). When they can't, the pack is `MISNUMBERED` for a hand import, unless some of the
    window's ids are already used (by the stored record's map or an earlier window). Then the pass stands down,
    so on the window holding a seed off its aliased special the rest of the pack places by its numbers too.
    """

    pack = _misnumbered_pack(state)
    if pack is None:
        return
    specials = _aliased_specials(state, pack)
    if specials is None:
        if state.untouched:
            state.set_aside_each(_remaining_of(state, pack.run.whole), PlacementVerdict.MISNUMBERED)
        return
    for file in _remaining_of(state, pack.run.members):
        ep_id = specials[file]
        if state.scope.admits(ep_id):
            state.place_or_skip(file, ep_id, PlacementVerdict.ALTERNATE)
        else:
            state.set_aside(file, PlacementVerdict.ALIASED_ELSEWHERE)
    state.set_aside_each(_remaining_of(state, pack.run.superseded), PlacementVerdict.DUPLICATE)


def _remaining_of(state: Placer, members: Iterable[RunMember]) -> list[TorrentFile]:
    """The members' files still to place, in the order given. Seeded, gone, and already judged ones drop out."""

    return [file for file in state.files_of(members) if file.remaining]


def _aliased_specials(state: Placer, pack: _ListedPack) -> dict[TorrentFile, int] | None:
    """Each member's file -> its aliased special, or None if the aliases don't map the pack one to one onto the listing.

    Also None when a member's title, size match, or seed puts it anywhere but its aliased special.
    """

    listing = state.scope.listing
    member_files = frozenset(state.files_of(pack.run.members))
    specials: dict[TorrentFile, int] = {}
    # Every version is checked, superseded ones too.
    for member, file in zip(pack.run.whole, state.files_of(pack.run.whole), strict=True):
        tvdb_number = listing.special_aliases.get(member.number)
        ep_id = None if tvdb_number is None else state.scope.id_by_key.get(EpisodeKey(0, tvdb_number))
        if ep_id is None or file.contradicts((ep_id,)):
            return None
        if file.identified_as not in (None, ep_id) or file.seeded_on not in (None, (ep_id,)):
            return None
        if file in member_files:
            specials[file] = ep_id
    # The map is one to one when it hits every listed special.
    return specials if frozenset(specials.values()) == pack.listed else None


def _misnumbered_pack(state: Placer) -> _ListedPack | None:
    """The batch as one listing-wide specials pack with a number that isn't a listed special, else None.

    For specials the curated listing (`TargetScope.listing`) beats the file names. In a regular season, a number
    outside the listing leaves the file foreign as usual.
    """

    listed = state.scope.specials_listing()
    if listed is None or (run := _sole_specials_run(state, len(listed))) is None:
        return None
    if all(state.scope.id_by_key.get(EpisodeKey(0, n)) in listed for n in run.numbers):
        return None
    return _ListedPack(run, listed)


def _sole_specials_run(state: Placer, width: int) -> NumberedRun | None:
    """The batch as one run of `width` distinct `S00Exx` names, each read by Sonarr as its own number's special.

    Seeded and gone files count as members. None if a parse is unknown, the count-based passes are off, a file
    other than an extra already has a verdict, the batch holds another run or a file outside the run, a number
    repeats or spans several episodes, or every member's title confirms its number.
    """

    if not state.all_parses_known or state.count_legs_barred or not state.unjudged:
        return None
    runs = state.counted_runs
    if len(runs) != 1 or set(state.files_of(runs[0].whole)) != set(state.counted):
        return None
    run = runs[0]
    if len(run.numbers) != width or len(set(run.numbers)) != width:
        return None
    confirmed = 0
    for member, file in zip(run.whole, state.files_of(run.whole), strict=True):
        info = state.single_parse(file)
        if info is None or info.season_number != 0 or info.episode_numbers != (member.number,):
            return None
        # One title vouches for its own file alone: a pack it does not cover whole stays judged by the listing.
        confirmed += file.confirms(file.reading.resolved)
    return None if confirmed == len(run.whole) else run


class _Tier(IntEnum):
    """How well a numbered run fits the run window, best first. The release-run pass ranks candidates by it."""

    WHOLE = 1
    """Numbered `1..N`, where N is the window's width."""
    NUMBERED = 2
    """Numbered exactly as the window's episode numbers."""
    COVERING = 3
    """Numbered `1..N` for the whole season, when the window holds only part of that season."""
    ANYWHERE = 4
    """Any N consecutive numbers (`_RunFit.from_anywhere`)."""


class _Candidate(NamedTuple):
    """A run that could place onto the window: its position among the runs, its best tier, and its title evidence."""

    position: int
    run: NumberedRun
    tier: _Tier
    evidence: TitleEvidence


@dataclass(frozen=True, slots=True, eq=False)
class _RunFit:
    """The run window and the unplaced runs the release-run pass fits to it.

    `window` and `runs` are taken once when the pass starts. Every check reads `state` live.
    """

    state: Placer
    window: RunWindow
    runs: tuple[NumberedRun, ...]
    """The numbered runs among the unplaced names, in run order."""

    @classmethod
    def of(cls, state: Placer) -> Self | None:
        """The fit when the open ids form a run window, else None."""

        window = state.run_window()
        return None if window is None else cls(state, window, tuple(state.runs(state.remaining())))

    def candidates(self) -> list[_Candidate]:
        """Every run that could place onto the window, at the best tier it fits, sorted by tier then by run order."""

        width = len(self.window.ids)
        tiers = {
            _Tier.WHOLE: runs_from_one(self.runs, width),
            _Tier.NUMBERED: runs_with_numbers(self.runs, self.window.numbers),
            _Tier.COVERING: self.covering(),
            _Tier.ANYWHERE: self.from_anywhere(),
        }
        tier_of: dict[NumberedRun, _Tier] = {}
        for tier, fits in tiers.items():
            for run in fits:
                tier_of.setdefault(run, tier)
        position = {run: index for index, run in enumerate(self.runs)}
        return [_Candidate(position[run], run, tier, self.evidence(run)) for run, tier in tier_of.items()]

    def evidence(self, run: NumberedRun) -> TitleEvidence:
        """Count the members whose tail starts with only window titles, and those with only outside titles.

        A tail matching titles on both sides counts for neither.
        """

        inside = outside = 0
        for file in self.state.files_of(run.members):
            sides = {ep_id in self.window.id_set for ep_id in file.leaders}
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return TitleEvidence(inside, outside)

    def covering(self) -> list[NumberedRun]:
        """The runs numbered `1..N` for the window's whole season, when the window holds fewer than N ids."""

        season = self.window.season
        if season is None:
            return []
        count = self.state.facts.season_counts.get(season, 0)
        return runs_from_one(self.runs, count) if count > len(self.window.ids) else []

    def covering_refused(self) -> bool:
        """Whether a whole-season run cannot be trusted to number this window.

        True for a window spanning seasons, for a specials window, and for a season whose numbering has a gap.
        """

        season = self.window.season
        if season is None or self.window.specials:
            return True
        count = self.state.facts.season_counts.get(season, 0)
        return any(EpisodeKey(season, n) not in self.state.scope.id_by_key for n in range(1, count + 1))

    def from_anywhere(self) -> list[NumberedRun]:
        """The runs of consecutive numbers as wide as the window, at any start, where no member has a complete reading.

        Empty once any id is used (by a seeded file, an earlier window, or an earlier pass).
        """

        if self.state.used:
            return []
        width = len(self.window.ids)
        return [
            run
            for run in self.runs
            if run.consecutive(width) and not any(file.reading.complete for file in self.state.files_of(run.members))
        ]

    def seasons_read(self, run: NumberedRun) -> set[int]:
        """The seasons Sonarr read the run's members into."""

        key_by_id = self.state.facts.key_by_id
        return {key_by_id[ep_id].season for file in self.state.files_of(run.members) for ep_id in file.reading.resolved}

    def read_elsewhere(self, run: NumberedRun) -> bool:
        """Whether a member's reading lands in a season other than the window's, which disputes the run's numbers."""

        return any(read != self.window.season for read in self.seasons_read(run))

    def settled_elsewhere(self, run: NumberedRun) -> bool:
        """Whether Sonarr matched every member to its own distinct episode outside the scope (another season or cour).

        A tied run counts as elsewhere when none of its candidate episodes is in the scope.
        """

        readings = [file.reading for file in self.state.files_of(run.members)]
        if all(r.tied is not None for r in readings):
            return not any(i in self.state.scope.real_ids for r in readings for i in r.tied or ())
        if not all(r.outside and r.corroborated and len(r.resolved) == 1 for r in readings):
            return False
        return len({r.resolved[0] for r in readings}) == len(readings)

    def refused(self, run: NumberedRun) -> bool:
        """Whether the batch shows the run cannot own the whole window."""

        state, window_set = self.state, self.window.id_set
        for file in state.files_of(run.members):
            info = file.parsed
            # A missing parse, or a name that itself claims several episodes, refuses the run. A multi-episode
            # Sonarr match does not.
            if info is None or claims_several_episodes(info):
                return True
            reading = file.reading
            keyed_outside = not reading.borrowed and reading.complete and window_set.isdisjoint(reading.resolved)
            # A member whose own key lies outside the window refuses the run, except on a specials window.
            if keyed_outside and not self.window.specials:
                return True
        member_files = set(state.files_of(run.whole))
        for file in state.remaining():
            if file in member_files:
                continue
            reading = file.reading
            if not window_set.isdisjoint(reading.tied or ()):
                return True
            # A runner-up's claim on the window lost before the run was asked: it refuses nothing.
            if not reading.trusted or file.runner_up(reading.resolved):
                continue
            if reading.resolved and window_set.issuperset(reading.resolved):
                return True
        return False


def pass_release_run(state: Placer) -> None:
    """Place one numbered run onto the run window by its own numbers when Sonarr's reading of it does not add up.

    Candidates are the unplaced runs that fit a `_Tier`. Tie-breaks, in order: drop runs Sonarr matched whole
    outside the scope (unless all are), then keep the one run the titles select, the one an AniList title names,
    or the best tier. A run with a member placed by size against its numbers (`Placer.size_refuted_runs`) isn't
    placed. With any parse unknown, the pick is `HELD`. It is refused by `_RunFit.refused`, `covering_refused` for
    a covering pick, a dispute, a member its position disputes (`TorrentFile.disputed_by_position`), a title veto
    (`TitleEvidence.vetoes`), or an AniList title naming another run.
    """

    if not state.map_known or (fit := _RunFit.of(state)) is None:
        return
    candidates = fit.candidates()
    # A covering run read into another season is disputed, picked or not. Read whole into one other season, it
    # belongs there (its files end up FOREIGN). Otherwise its readings are rejected.
    disputed = [c.run for c in candidates if c.tier is _Tier.COVERING and fit.read_elsewhere(c.run)]
    # Vetoes first: a rejected reading leaves its run unsettled, which `open_positions` then counts as open.
    for run in disputed:
        if not (fit.settled_elsewhere(run) and len(fit.seasons_read(run)) == 1):
            state.veto_each(state.files_of(run.whole))
    open_positions = frozenset(index for index, run in enumerate(fit.runs) if not fit.settled_elsewhere(run))
    naming = Naming.of([run.prefix for run in fit.runs], state.scope.names)
    # Several candidates turn the count-based passes off, even when a tie-break picks one.
    state.count_legs_barred |= len(candidates) > 1
    pick = _sole_candidate(candidates, open_positions, naming)
    if pick is None:
        return
    run = pick.run
    if run.prefix in state.size_refuted_runs:
        return
    if not state.all_parses_known:
        state.set_aside_each(state.files_of(run.whole), PlacementVerdict.HELD)
        return
    covers = pick.tier is _Tier.COVERING
    refused = (
        fit.refused(run)
        or (covers and fit.covering_refused())
        or run in disputed
        or any(file.disputed_by_position for file in state.files_of(run.members))
        or pick.evidence.vetoes
        or (naming is not None and naming.names_another(pick.position, open_positions))
    )
    if refused:
        state.count_legs_barred = True
        return
    window = fit.window
    # A whole-season run over a partial window places only the window's numbers. Its other members are FOREIGN.
    members = window.slice_of(run) if covers else run.members
    files = state.files_of(members)
    # Sonarr already reads each member onto its own window episode: the exact pass places them.
    exact = [file.reading.exact_ids for file in files]
    if all(len(ids) == 1 and ids[0] in window.id_set for ids in exact) and len(set(exact)) == len(exact):
        return
    if not state.place_zip(files, window.ids, PlacementVerdict.RELEASE_RUN):
        return
    state.set_aside_each([file for file in state.files_of(run.members) if file not in files], PlacementVerdict.FOREIGN)
    state.set_aside_each(state.files_of(run.superseded), PlacementVerdict.DUPLICATE)


def _sole_candidate(
    candidates: Sequence[_Candidate], open_positions: frozenset[int], naming: Naming | None
) -> _Candidate | None:
    """The one candidate the tie-breaks leave, else None (see `pass_release_run`)."""

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


def pass_numbered_run(state: Placer) -> None:
    """Zip the one `1..N` run among the files Sonarr found no episode for onto a run window of N ids.

    Unlike the ordered zip, this works beside files that other passes placed. A file qualifies when its reading
    resolved nothing, its name carries no `SxxExx` key, and it is not multi-episode. Skipped without a series map,
    with any parse unknown, or with the count-based passes off.
    """

    if not state.map_known or not state.all_parses_known or state.count_legs_barred:
        return
    window = state.run_window()
    if window is None:
        return
    blind = [
        file
        for file in state.remaining()
        if not file.reading.resolved and (info := state.single_parse(file)) is not None and not info.episode_numbers
    ]
    fits = runs_from_one(state.runs(blind), len(window.ids))
    if len(fits) != 1:
        return
    run = fits[0]
    if state.place_zip(state.files_of(run.members), window.ids, PlacementVerdict.NUMBERED_RUN):
        state.set_aside_each(state.files_of(run.superseded), PlacementVerdict.DUPLICATE)
