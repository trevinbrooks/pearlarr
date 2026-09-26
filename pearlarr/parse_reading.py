"""Resolve one Sonarr parse against the scope: the episodes a file name claims, and how far to trust them."""

from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from .placement_types import SeriesFacts, TargetScope
from .release_names import range_key
from .seadex_types import EpisodeKey, MatchedEpisode, ParsedFileInfo, season_episode_key

# Sonarr's matched pairs plausibly cover a double or triple episode, never more, and a range key Sonarr read short
# is widened only that far. A range Sonarr read whole from the name is trusted at any width.
_MATCHED_SPAN_CAP = 3


class _EpisodeClaim(NamedTuple):
    """One identity claim a file's parse makes: a `(season, episode)` pair, plus Sonarr's id when borrowed."""

    season: int | None
    episode: int
    claimed_id: int | None
    """Sonarr's own episode id from a borrowed matched pair. It must agree
    with our map's id. None for a name-parsed claim (no id to cross-check)."""


class Reading(NamedTuple):
    """What one file's parse resolves to in OUR series map, and how far to trust it.

    Every pass consults this one reading, so the grab-time placement and the
    import wait never disagree about what a name says.
    """

    resolved: tuple[int, ...]
    """Every id a claim resolved to (inside the scope or not), claim order, deduped."""

    inside: tuple[int, ...]
    """The subset inside the scope (all of `resolved` when the scope is `unscoped`)."""

    complete: bool
    """At least one claim, and every claim resolved (a partial span is never half-placed)."""

    borrowed: bool
    """The claims are Sonarr's matched pairs: the name carried no `(season, episode)` of its own, or its keys
    gave way to Sonarr's alias match (`_claims_of`)."""

    vetoed: bool
    """Rejected: the ids stay as evidence, but no pass places the file by them, and they never prove it foreign.

    Set for a full-season parse, a borrowed span wider than `_MATCHED_SPAN_CAP` or not matching the count of
    the name's own absolutes, a run whose readings dispute its numbers (a tie, or a covering run read into
    another season), a reading its title contradicts, and a span another file reads differently."""

    corroborated: bool
    """Sonarr's series match includes every claimed pair (a borrowed reading always does). An own key Sonarr
    could not match to the series is a parse it did not believe either (a CRC tag read as "E8")."""

    tied: tuple[int, ...] | None = None
    """For a tied file (see the placer's season re-read), its episode under the season's and the specials'
    numbering. No other run may take a window holding either, and a tied run counts as matched elsewhere
    only for a scope holding none of them."""

    @property
    def keyed(self) -> bool:
        """The name's own key resolved in the series and Sonarr's match agrees: an episode, whatever else it says."""

        return bool(self.resolved) and not self.borrowed and self.corroborated

    @property
    def outside(self) -> bool:
        """A complete, unrejected reading wholly outside the scope: the file belongs to another slice."""

        return self.complete and not self.vetoed and not self.inside

    @property
    def rank(self) -> tuple[bool, bool]:
        """Exact-pass priority, best first: an own key Sonarr matched, a borrowed pair, an own key it did not match."""

        return (not self.corroborated, self.borrowed)


_NO_READING = Reading((), (), complete=False, borrowed=False, vetoed=False, corroborated=False)


class _Claims(NamedTuple):
    """The claims a parse makes, and whether they are Sonarr's matched pairs rather than the name's own keys."""

    claims: tuple[_EpisodeClaim, ...]
    borrowed: bool


def _matched_claims(info: ParsedFileInfo) -> tuple[_EpisodeClaim, ...]:
    """Sonarr's series-matched pairs as claims, junk repeats collapsed."""

    return tuple(
        dict.fromkeys(
            _EpisodeClaim(matched.season_number, matched.episode_number, matched.id)
            for matched in info.matched_episodes
        )
    )


def _resolve(claim: _EpisodeClaim, scope: TargetScope) -> int | None:
    """Our map's id for the claim's numbers, when Sonarr's own id (if any) agrees."""

    ep_id = scope.id_by_key.get(season_episode_key(claim.season, claim.episode))
    return ep_id if ep_id and claim.claimed_id in (None, ep_id) else None


def _all_inside(claims: Iterable[_EpisodeClaim], scope: TargetScope) -> bool:
    """Whether every claim resolves to an episode of the scope."""

    return all(_resolve(claim, scope) in scope.real_ids for claim in claims)


def _is_alias_shift(own: Sequence[_EpisodeClaim], matched: Sequence[_EpisodeClaim]) -> bool:
    """Whether Sonarr moved the name's episode numbers, unchanged, into one regular season (an alias match)."""

    seasons = {claim.season for claim in matched}
    return (
        len(seasons) == 1
        and seasons.isdisjoint({None, 0})
        and [claim.episode for claim in own] == [claim.episode for claim in matched]
    )


def _claims_of(info: ParsedFileInfo, scope: TargetScope) -> _Claims:
    """The name's own keys, else Sonarr's matched pairs when there is a scope and the parse is not a full season.

    Own keys give way to Sonarr's alias match (the same numbers moved into one regular season) when not every
    key resolves inside the scope and every matched pair does: a sequel numbered as its own season 1.
    """

    own = tuple(dict.fromkeys(_EpisodeClaim(info.season_number, episode, None) for episode in info.episode_numbers))
    if not own:
        borrows = not scope.unscoped and not info.full_season
        return _Claims(_matched_claims(info) if borrows else (), borrowed=borrows)
    matched = _matched_claims(info)
    if _is_alias_shift(own, matched) and not _all_inside(own, scope) and _all_inside(matched, scope):
        return _Claims(matched, borrowed=True)
    return _Claims(own, borrowed=False)


def read_parse(info: ParsedFileInfo | None, scope: TargetScope) -> Reading:
    """Read one parse against the scope: the name's own keys are the claims (`_claims_of` has the alias shift).

    A name with none borrows Sonarr's matched pairs when there is a scope and the parse is not a full season. A pair
    may resolve anywhere in the series, but only when its own episode id agrees with our map's, so a wrong-series
    match never resolves. `inside` keeps the ids the scope admits.
    """

    if info is None:
        return _NO_READING
    claims, borrowed = _claims_of(info, scope)
    if not claims:
        return _NO_READING
    # Only a borrowed span is capped (DISTINCT pairs): Sonarr matches a bare
    # "S01" name to the WHOLE season, so a wide match is a season pack without
    # the flag, while the name's own "E11-E16" is an explicit claim. A borrowed
    # span must also COVER the name's own absolutes, or a "12-13" file whose
    # match resolved only E12 would import half.
    pairs = {(claim.season, claim.episode) for claim in claims}
    absolutes = set(info.absolute_episode_numbers)
    vetoed = info.full_season or (
        borrowed and (len(pairs) > _MATCHED_SPAN_CAP or (bool(absolutes) and len(pairs) != len(absolutes)))
    )
    matched = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
    corroborated = borrowed or pairs <= matched
    resolved: list[int] = []
    complete = True
    for claim in claims:
        if (ep_id := _resolve(claim, scope)) is not None:
            resolved.append(ep_id)
        else:
            complete = False
    # The claim dedup keeps (s,e,None) and (s,e,id) apart. Dedupe the resolved
    # ids too, so Sonarr never receives one episode twice.
    ids = tuple(dict.fromkeys(resolved))
    inside = tuple(i for i in ids if scope.admits(i))
    return Reading(ids, inside, complete=complete, borrowed=borrowed, vetoed=vetoed, corroborated=corroborated)


def widen_range_key(name: str, info: ParsedFileInfo, facts: SeriesFacts) -> ParsedFileInfo:
    """Widen a parse Sonarr read as just the first episode of the name's `SxxEyy-zz` range to the whole range.

    It stays as read if the range doesn't count up, is wider than a triple, ends on its first episode's absolute
    (dual numbering), or fits neither the name's season nor the one Sonarr matched its first episode into.
    """

    key = range_key(name)
    if key is None or info.season_number != key.season or info.episode_numbers != (key.first,):
        return info
    episodes = tuple(range(key.first, key.last + 1))
    if not 1 < len(episodes) <= _MATCHED_SPAN_CAP:
        return info
    first_id = facts.id_by_key.get(EpisodeKey(key.season, key.first))
    if first_id is not None and facts.absolute_of.get(first_id) == key.last:
        return info
    matched_season = _matched_season(key.first, info.matched_episodes)
    seasons = (key.season,) if matched_season is None else (key.season, matched_season)
    if not any(_range_fits(episodes, season, facts) for season in seasons):
        return info
    # Sonarr often reads the range's last number as an absolute. It isn't one, so drop it.
    absolutes = () if info.absolute_episode_numbers == (key.last,) else info.absolute_episode_numbers
    matched = _widened_matches(info.matched_episodes, episodes)
    return info.with_numbers(episodes=episodes, absolutes=absolutes, matched=matched)


def _matched_season(first: int, matched: Iterable[MatchedEpisode]) -> int | None:
    """The one regular season Sonarr matched the range's first episode into, else None.

    As in `_is_alias_shift`, a match into the specials or spread over several seasons isn't a shift.
    """

    seasons = {pair.season_number for pair in matched if pair.episode_number == first}
    return seasons.pop() if len(seasons) == 1 and 0 not in seasons else None


def _range_fits(episodes: Sequence[int], season: int, facts: SeriesFacts) -> bool:
    """Whether the season has every episode of the range plus at least one more."""

    return all(EpisodeKey(season, episode) in facts.id_by_key for episode in episodes) and (
        len(episodes) < facts.season_counts[season]
    )


def _widened_matches(matched: Iterable[MatchedEpisode], episodes: Sequence[int]) -> tuple[MatchedEpisode, ...]:
    """Sonarr's matched pairs, each pair on the range's first episode followed by the rest of the range.

    The added pairs keep that pair's season and have no Sonarr id, since Sonarr never matched them. Our map
    resolves them like a name's own key. A pair Sonarr already matched isn't added a second time.
    """

    present = {(pair.season_number, pair.episode_number) for pair in matched}
    widened: list[MatchedEpisode] = []
    for pair in matched:
        widened.append(pair)
        if pair.episode_number == episodes[0]:
            widened.extend(
                MatchedEpisode(season_number=pair.season_number, episode_number=episode)
                for episode in episodes[1:]
                if (pair.season_number, episode) not in present
            )
    return tuple(widened)


def parse_has_no_number(info: ParsedFileInfo | None) -> bool:
    """Whether a file's NAME carries no episode or absolute number at all, or there is no parse.

    It ignores `matched_episodes` on purpose: a Sonarr match outside the scope must not block the placement we
    intend (Sonarr informs identity, never decides). How many episodes a file holds is `spans_multiple`'s question.
    """

    return info is None or (not info.episode_numbers and not info.absolute_episode_numbers)


def numbers_miss_the_series(info: ParsedFileInfo, ep_id_map: Mapping[EpisodeKey, int]) -> bool:
    """Whether the name's numbers provably describe no episode of this series.

    A movie year read as SxxEyy is a parse artifact: when every name-parsed key misses the whole series map
    and the name carries no absolutes, the file counts as numberless. Only meaningful with a non-empty map.
    """

    if not info.episode_numbers or info.absolute_episode_numbers:
        return False
    return all(not ep_id_map.get(season_episode_key(info.season_number, episode)) for episode in info.episode_numbers)


def claims_several_episodes(info: ParsedFileInfo) -> bool:
    """Whether the NAME claims more than one episode, so placing the file as one would half-import.

    Only the name's own numbers count: a multi-pair series match may only map a number into another numbering
    (`spans_multiple` weighs it), and a full-season read of a name is a missing episode token, never a claim of several.
    """

    return len(set(info.episode_numbers)) > 1 or len(set(info.absolute_episode_numbers)) > 1
