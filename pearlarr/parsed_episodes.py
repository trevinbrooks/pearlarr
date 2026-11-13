"""Resolve one Sonarr parse against the scope: the episodes a file name claims, and how far to trust them."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .arr_models import MatchedEpisode, ParsedFileInfo
from .file_names import range_key
from .placement_types import EpisodeIndex, PlacementTarget
from .releases import EpisodeKey, season_episode_key

# Sonarr's matched pairs plausibly cover a double or triple episode, never more, and a range key Sonarr read short
# is widened only that far. A range Sonarr read whole from the name is trusted at any width.
_MATCHED_SPAN_CAP = 3


@dataclass(frozen=True, slots=True)
class _ParsedKey:
    """One identity claim a file's parse makes: a `(season, episode)` pair, plus Sonarr's id when borrowed."""

    season: int | None
    episode: int
    matched_id: int | None
    """Sonarr's own episode id from a borrowed matched pair. It must agree
    with our map's id. None for a name-parsed claim (no id to cross-check)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedEpisodes:
    """What one file's parse resolves to in our series map, and how far to trust it."""

    ids: tuple[int, ...]
    """Every id a claim resolved to (inside the scope or not), claim order, deduped."""

    in_window: tuple[int, ...]
    """The subset inside the scope (all of `resolved` when the scope is `unscoped`)."""

    complete: bool
    """At least one claim, and every claim resolved (a partial span is never half-placed)."""

    from_matched: bool
    """The claims are Sonarr's matched pairs: the name carried no `(season, episode)` of its own, or its keys
    gave way to Sonarr's alias match (`_claims_of`)."""

    rejected: bool
    """Rejected: the ids stay as evidence, but no pass places the file by them, and they never prove it foreign.

    Set for a full-season parse, and for a borrowed span wider than `_MATCHED_SPAN_CAP` or not matching the count
    of the name's own absolutes. The placer's passes reject the rest: a reading a title contradicts or no title
    confirms, a number dispute, a span another file reads differently, a tie, and a covering run read into another
    season."""

    sonarr_agrees: bool
    """Sonarr's series match includes every claimed pair (a borrowed reading always does). An own key Sonarr
    could not match to the series is a parse it did not believe either (a CRC tag read as "E8")."""

    either_ids: tuple[int, ...] | None = None
    """For a tied file (see the placer's season re-read), its episode under the season's and the specials'
    numbering. No other run may take a window holding either, and a tied run counts as matched elsewhere
    only for a scope holding none of them."""

    @property
    def key_confirmed(self) -> bool:
        """The name's own key resolved in the series and Sonarr's match agrees: an episode, whatever else it says."""

        return bool(self.ids) and not self.from_matched and self.sonarr_agrees

    @property
    def usable(self) -> bool:
        """A complete reading that was not rejected."""

        return self.complete and not self.rejected

    @property
    def wholly_outside(self) -> bool:
        """A trusted reading wholly outside the scope: the file belongs to another slice."""

        return self.usable and not self.in_window

    @property
    def exact_ids(self) -> tuple[int, ...]:
        """The ids the exact pass places the file on: a trusted reading wholly inside the scope. Else ()."""

        return self.in_window if self.usable and self.in_window == self.ids else ()

    @property
    def rank(self) -> tuple[bool, bool]:
        """Exact-pass priority, best first: an own key Sonarr matched, a borrowed pair, an own key it did not match."""

        return (not self.sonarr_agrees, self.from_matched)


_NO_PARSED_EPISODES = ParsedEpisodes(
    ids=(), in_window=(), complete=False, from_matched=False, rejected=False, sonarr_agrees=False
)


@dataclass(frozen=True, slots=True)
class _ParsedKeys:
    """The claims a parse makes, and whether they are Sonarr's matched pairs rather than the name's own keys."""

    keys: tuple[_ParsedKey, ...]
    from_matched: bool


def _matched_keys(info: ParsedFileInfo) -> tuple[_ParsedKey, ...]:
    """Sonarr's series-matched pairs as claims, junk repeats collapsed."""

    return tuple(
        dict.fromkeys(
            _ParsedKey(matched.season_number, matched.episode_number, matched.id) for matched in info.matched_episodes
        )
    )


def _resolve(key: _ParsedKey, target: PlacementTarget) -> int | None:
    """Our map's id for the claim's numbers, when Sonarr's own id (if any) agrees."""

    ep_id = target.id_by_key.get(season_episode_key(key.season, key.episode))
    return ep_id if ep_id and key.matched_id in (None, ep_id) else None


def _all_inside(keys: Iterable[_ParsedKey], target: PlacementTarget) -> bool:
    """Whether every claim resolves to an episode of the scope."""

    return all(_resolve(key, target) in target.real_ids for key in keys)


def _is_season_shift(own: Sequence[_ParsedKey], matched: Sequence[_ParsedKey]) -> bool:
    """Whether Sonarr moved the name's episode numbers, unchanged, into one regular season (an alias match)."""

    seasons = {key.season for key in matched}
    return (
        len(seasons) == 1
        and seasons.isdisjoint({None, 0})
        and [key.episode for key in own] == [key.episode for key in matched]
    )


def _parsed_keys_of(info: ParsedFileInfo, target: PlacementTarget) -> _ParsedKeys:
    """The name's own keys, else Sonarr's matched pairs when there is a scope and the parse is not a full season.

    Own keys give way to Sonarr's alias match (the same numbers moved into one regular season) when not every
    key resolves inside the scope and every matched pair does: a sequel numbered as its own season 1.
    """

    own = tuple(dict.fromkeys(_ParsedKey(info.season_number, episode, None) for episode in info.episode_numbers))
    if not own:
        uses_matched = not target.no_window and not info.full_season
        return _ParsedKeys(_matched_keys(info) if uses_matched else (), from_matched=uses_matched)
    matched = _matched_keys(info)
    if _is_season_shift(own, matched) and not _all_inside(own, target) and _all_inside(matched, target):
        return _ParsedKeys(matched, from_matched=True)
    return _ParsedKeys(own, from_matched=False)


def resolve_parse(info: ParsedFileInfo | None, target: PlacementTarget) -> ParsedEpisodes:
    """Read one parse against the scope: the name's own keys are the claims (`_claims_of` has the alias shift).

    A name with none borrows Sonarr's matched pairs when there is a scope and the parse is not a full season. A pair
    may resolve anywhere in the series, but only when its own episode id agrees with our map's, so a wrong-series
    match never resolves. `inside` keeps the ids the scope admits.
    """

    if info is None:
        return _NO_PARSED_EPISODES
    basis = _parsed_keys_of(info, target)
    if not basis.keys:
        return _NO_PARSED_EPISODES
    # Only a borrowed span is capped (DISTINCT pairs): Sonarr matches a bare
    # "S01" name to the WHOLE season, so a wide match is a season pack without
    # the flag, while the name's own "E11-E16" is an explicit claim. A borrowed
    # span must also COVER the name's own absolutes, or a "12-13" file whose
    # match resolved only E12 would import half.
    pairs = {(key.season, key.episode) for key in basis.keys}
    absolutes = set(info.absolute_episode_numbers)
    rejected = info.full_season or (
        basis.from_matched and (len(pairs) > _MATCHED_SPAN_CAP or (bool(absolutes) and len(pairs) != len(absolutes)))
    )
    matched = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
    sonarr_agrees = basis.from_matched or pairs <= matched
    resolved: list[int] = []
    complete = True
    for key in basis.keys:
        if (ep_id := _resolve(key, target)) is not None:
            resolved.append(ep_id)
        else:
            complete = False
    # The claim dedup keeps (s,e,None) and (s,e,id) apart. Dedupe the resolved
    # ids too, so Sonarr never receives one episode twice.
    ids = tuple(dict.fromkeys(resolved))
    inside = tuple(i for i in ids if target.admits(i))
    return ParsedEpisodes(
        ids=ids,
        in_window=inside,
        complete=complete,
        from_matched=basis.from_matched,
        rejected=rejected,
        sonarr_agrees=sonarr_agrees,
    )


def widen_range_key(name: str, info: ParsedFileInfo, series: EpisodeIndex) -> ParsedFileInfo:
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
    first_id = series.id_by_key.get(EpisodeKey(key.season, key.first))
    if first_id is not None and series.absolute_of.get(first_id) == key.last:
        return info
    matched_season = _matched_season(key.first, info.matched_episodes)
    seasons = (key.season,) if matched_season is None else (key.season, matched_season)
    if not any(_range_fits(episodes, season, series) for season in seasons):
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


def _range_fits(episodes: Sequence[int], season: int, series: EpisodeIndex) -> bool:
    """Whether the season has every episode of the range plus at least one more."""

    return all(EpisodeKey(season, episode) in series.id_by_key for episode in episodes) and (
        len(episodes) < series.season_counts[season]
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
    """Whether a file's name carries no episode or absolute number at all, or there is no parse.

    Sonarr's `matched_episodes` don't count. How many episodes a file holds is `spans_multiple`'s question.
    """

    return info is None or (not info.episode_numbers and not info.absolute_episode_numbers)


def glued_numbers(info: ParsedFileInfo) -> frozenset[int]:
    """The counts Sonarr glues the parse's key from: season, then two-digit episode ("Show.101": S01E01).

    Season 0 glues too ("Show.0013": S00E13). Empty without a season.
    """

    season = info.season_number
    if season is None:
        return frozenset()
    return frozenset(season * 100 + episode for episode in info.episode_numbers)


def numbers_miss_the_series(info: ParsedFileInfo, id_by_key: Mapping[EpisodeKey, int]) -> bool:
    """Whether every key the name parsed misses the series map and the name has no absolutes: a year read as SxxEyy.

    False with an empty map.
    """

    if not id_by_key or not info.episode_numbers or info.absolute_episode_numbers:
        return False
    return all(not id_by_key.get(season_episode_key(info.season_number, episode)) for episode in info.episode_numbers)


def names_several_episodes(info: ParsedFileInfo) -> bool:
    """Whether the name's own numbers claim more than one episode.

    A multi-pair series match doesn't count (`spans_multiple` weighs it), nor does a full-season read.
    """

    return len(set(info.episode_numbers)) > 1 or len(set(info.absolute_episode_numbers)) > 1


def spans_multiple(info: ParsedFileInfo, target: PlacementTarget) -> bool:
    """Whether the file likely holds more than one episode: its name claims several, or Sonarr matched several.

    It asks how many episodes, not which. A full-season match (a name with no episode number) counts only when one
    of its episodes is in the scope.
    """

    if names_several_episodes(info):
        return True
    pairs = {(matched.season_number, matched.episode_number) for matched in info.matched_episodes}
    if len(pairs) <= 1:
        return False
    if not info.full_season:
        return True
    return any(
        target.id_by_key.get(season_episode_key(season, episode)) in target.real_ids for season, episode in pairs
    )
