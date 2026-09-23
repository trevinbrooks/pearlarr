"""Pure reading of one Sonarr parse against the target scope: the claims a name makes and how far to trust them."""

from collections.abc import Mapping
from typing import NamedTuple

from .placement_types import TargetScope
from .seadex_types import EpisodeKey, ParsedFileInfo, season_episode_key

# A borrowed span (Sonarr's matched pairs) plausibly covers a double or triple
# episode, never more. A name's own explicit range is its claim at any width.
_MATCHED_SPAN_CAP = 3


class _EpisodeClaim(NamedTuple):
    """One identity claim a file's parse makes: a `(season, episode)` pair, plus Sonarr's id when borrowed."""

    season: int | None
    episode: int
    claimed_id: int | None
    """Sonarr's own episode id from a borrowed matched pair. It must agree
    with our map's id. None for a name-parsed claim (no id to cross-check)."""


class Reading(NamedTuple):
    """One file's identity reading: what its parse's claims resolve to in OUR map, and how far to trust it.

    The single reading every pass consults, so the seed and the import wait
    can never disagree on what a name says.
    """

    resolved: tuple[int, ...]
    """Every id a claim resolved to (inside the scope or not), claim order, deduped."""

    inside: tuple[int, ...]
    """The subset inside the resolved set (all of `resolved` when the scope is `unscoped`)."""

    complete: bool
    """At least one claim, and every claim resolved (a partial span is never half-placed)."""

    borrowed: bool
    """The claims are Sonarr's matched pairs (the name carried no `(season, episode)` of its own)."""

    vetoed: bool
    """A full-season parse, a borrowed span past `_MATCHED_SPAN_CAP` or short of the name's own absolutes,
    or a run's reads that dispute its count (a tie, or a covering run read into another season): the
    claims are real but never placed on their own, and never prove the file foreign."""

    corroborated: bool
    """Sonarr's series match names every claimed pair (a borrowed reading always is). An own key Sonarr
    could not match to the series is a parse it did not believe either (a CRC tag read as "E8")."""

    tied: tuple[int, ...] | None = None
    """The episodes a tie's name may be, under the season's and the specials' numbering: spoken for, so
    no other run takes a window holding one, and a run of them stands aside only from a scope holding
    none."""

    @property
    def outside(self) -> bool:
        """A trusted reading that resolves wholly outside the scope: the file is another slice's."""

        return self.complete and not self.vetoed and not self.inside

    @property
    def rank(self) -> tuple[bool, bool]:
        """Exact-pass precedence, lowest first: a corroborated own key, a borrowed pair, an unmatched own key."""

        return (not self.corroborated, self.borrowed)


_NO_READING = Reading((), (), complete=False, borrowed=False, vetoed=False, corroborated=False)


def read_parse(info: ParsedFileInfo | None, scope: TargetScope, resolved_set: frozenset[int]) -> Reading:
    """Read one parse against the scope.

    The name's own `(season, episode)` keys are the claims. A name with none
    borrows Sonarr's series-MATCHED pairs, but ONLY under scope enforcement:
    membership in `resolved_set` is what keeps Sonarr's series match from
    deciding identity on its own, so matched pairs never apply `unscoped`. A
    borrowed pair's own episode id must AGREE with our map's id for the same
    numbers, or a wrong-series title match whose numbers coincide with ours
    would resolve. Junk duplicate pairs collapse to one claim. A missing
    season collapses to `SONARR_MISSING_KEY`, matching `EpisodeIndex.id_by_key`.
    """

    if info is None:
        return _NO_READING
    claims: list[_EpisodeClaim] = [_EpisodeClaim(info.season_number, episode, None) for episode in info.episode_numbers]
    borrowed = False
    if not claims and not scope.unscoped and not info.full_season:
        claims = [
            _EpisodeClaim(matched.season_number, matched.episode_number, matched.id)
            for matched in info.matched_episodes
        ]
        borrowed = True
    claims = list(dict.fromkeys(claims))
    if not claims:
        return _NO_READING
    # Only a borrowed span is capped (DISTINCT pairs): Sonarr matches a bare
    # "S01" name to the WHOLE season, so a wide match is the season-pack shape
    # sans flag, while the name's own "E11-E16" is an explicit claim. A borrowed
    # span must also COVER the name's own absolutes, or a "12-13" file whose
    # match resolved only E12 would half-import.
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
        ep_id = scope.id_by_key.get(season_episode_key(claim.season, claim.episode))
        if ep_id and claim.claimed_id in (None, ep_id):
            resolved.append(ep_id)
        else:
            complete = False
    # The triple dedup keeps (s,e,None) and (s,e,id) apart. Collapse the
    # resolved ids so one episode never reaches the wire twice.
    ids = tuple(dict.fromkeys(resolved))
    inside = ids if scope.unscoped else tuple(i for i in ids if i in resolved_set)
    return Reading(ids, inside, complete=complete, borrowed=borrowed, vetoed=vetoed, corroborated=corroborated)


def parse_has_no_number(info: ParsedFileInfo | None) -> bool:
    """Whether a file's NAME carries no usable episode number at all (parse miss).

    Deliberately blind to `matched_episodes`: the degenerate single-file
    fallback keys on this, and an out-of-set Sonarr match must not veto the
    placement OUR resolution intends (Sonarr informs identity, never decides -
    in either direction). Cardinality is the placer's `spans_multiple` question.
    """

    return info is None or (not info.episode_numbers and not info.absolute_episode_numbers)


def numbers_miss_the_series(info: ParsedFileInfo, ep_id_map: Mapping[EpisodeKey, int]) -> bool:
    """Whether the name's numbers provably describe no episode of this series.

    A movie year read as SxxEyy ("Title.2020" parsing S20E20) is a parse
    artifact, not identity: when EVERY name-parsed key misses the WHOLE series
    map and the name carries no absolutes, the signal is noise and the file
    counts as numberless. A key that resolves anywhere in the series is real
    evidence and is never downgraded. Only meaningful over a served map: an
    empty map makes every key "miss", so the caller gates on `map_known`.
    """

    if not info.episode_numbers or info.absolute_episode_numbers:
        return False
    return all(not ep_id_map.get(season_episode_key(info.season_number, episode)) for episode in info.episode_numbers)


def claims_several_episodes(info: ParsedFileInfo) -> bool:
    """Whether the NAME claims more than one episode, so placing the file as one would half-import.

    Only the name's own numbers count. A multi-pair series match is a scene
    map for another numbering, and a full-season read of a file name is a
    missing episode token ("S2 - OVA", "S0101"), never a claim of several.
    """

    return len(set(info.episode_numbers)) > 1 or len(set(info.absolute_episode_numbers)) > 1
