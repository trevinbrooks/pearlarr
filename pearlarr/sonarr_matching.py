r"""Sonarr's matching of a parsed file name to a series' episodes, copied for a series numbered as aired.

The module is a copy of Sonarr's logic (Sonarr and pearlarr are both GPL-3), function for function:

- `sonarr_title_key` is `Parser.CleanSeriesTitle`, with `NormalizeRegex`, `PercentRegex` and the accent table in
  `sonarr_accents`.
- `AiredMatcher.matches` is `ParsingService.Map` and `GetEpisodes`, with `MapSeasonNumber`, `GetSeasonEpisodes`,
  `GetAnimeEpisodes`, `GetStandardEpisodes`, the scene-number test of `ParseSpecialEpisodeTitle`, and
  `SceneMappingService`'s `FindSceneMapping`, `FindMappings`, `FilterSceneMappings` and `GetSceneSeasonNumber`. Its
  lookups are `EpisodeRepository`'s `Find`, `GetEpisodes`, `GetEpisodesBySceneSeason` and
  `FindEpisodesBySceneNumbering`.

Each copied function names its source in parentheses. A step of Sonarr's that can't change an answer here is left
out and listed below with the other departures. `parse_without_answer` is pearlarr's own: what the placement reads
of a parse when there is no answer.

Where it departs from Sonarr:

- A character's class comes from Python's Unicode tables, where Sonarr's comes from .NET's. The two differ on a
  character only one of them is recent enough to know.
- Python's rules stand in for .NET's where no title has needed .NET's:
  - A whole number with a sign or spaces around it is cleaned like any other title. Sonarr keeps it as it is.
  - The joining-word and percent patterns find where a word starts and ends by Python's `\b`. It differs from
    .NET's next to a combining accent, a connector other than `_`, or a number that isn't a digit, like a fraction.
  - A zero-width joiner or non-joiner ends a word here, where .NET's `\b` counts it as part of one.
  - A character past U+FFFF is one character of its own class here. .NET holds it as two halves, neither a word
    character nor a digit.
  - A character is lowercased by Python's rule, which turns a few into two characters, the dotted capital I among
    them. .NET keeps each to one.
  - The edit cost between two titles counts characters, where .NET counts UTF-16 units.
- An alias is looked up among one series' alternate titles. Sonarr looks a title up among every series' mappings,
  where another series' mapping under the same key can change which one is picked.
- An alternate title carries no `FilterRegex`, the pattern Sonarr limits an alias to some release names by. It
  carries a comment, which Sonarr never leaves blank on an alias with a pattern. So a name is matched under each
  list of aliases the pattern of every alias with a comment could leave, and has an answer when all of them match
  it alike.
- An alternate title carries no mapping type, so "it has no TVDB season" stands in for `Type == "XemService"`. The
  XEM rule's test that the alias isn't of TVDB origin is left out, since XEM gives no alias an origin.
- A blank series title is looked up like any other, where `FindSceneMapping` reads it by no alias. It finds only an
  alias whose title has no letter or digit, and no alternate title is so.
- `Series.UseSceneNumbering` isn't read. Sonarr clears a series' scene numbers when it turns the flag off, so a
  lookup by scene numbers finds nothing for such a series.
- Sonarr's search of a release's name for a special's title (`FindEpisodeByTitle`) isn't copied. A name that Sonarr
  would run it for has no answer.
- A parse keeps no air date, season list or season-title flag. A name read as a date matches nothing, a pack of
  several seasons matches as a pack of its first, and a name read as a season's title matches nothing.
- Episodes found by one lookup come in season and episode order, where Sonarr's come in its database's order.
- Steps that can't change an answer here:
  - `FilterSceneMappings` keeps the latest mapping alias of each title. The aliases a name is read by all share one
    title, so the latest of them all is kept.
  - `MapSeasonNumber` tests that an alias with a TVDB season starts at or before the name's season. Any alias
    picked for that season does.
  - `MapSeasonNumber`'s XEM rule passes over a first season the alias already names, which it would set to the
    same.
"""

import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations
from types import MappingProxyType
from typing import Final, NamedTuple, Self, assert_never

from .aired_view import AiredSeries
from .arr_models import AlternateTitle, MatchedEpisode, ParsedFileInfo, SonarrEpisode
from .releases import EpisodeKey
from .sonarr_accents import UNACCENTED

type _Found = tuple[MatchedEpisode, ...]

_NO_SEASON: Final = -1
"""The season Sonarr writes on an alternate title for "none"."""

_TVDB_ORIGIN: Final = "tvdb"
"""The `sceneOrigin` of an alternate title whose releases are numbered as TVDB numbers the series."""

_PERCENT: Final = re.compile(r"\b\d+%")
_JOINING_WORD: Final = re.compile(r"(?:\b|_)(?<!^)(?:[aà](?!$)|an|the|and|or|of)(?!$)(?:\b|_)", re.IGNORECASE)

_WORD_CATEGORIES: Final = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Nd", "Pc"})
r"""The Unicode categories .NET's `\w` matches."""

_ADD: Final = 10
_DROP: Final = 1
_SWAP: Final = 10

_MOST_COMMENTED: Final = 4
"""The most aliases with a comment under one title key that `matches` tries every reading of.

No series measured has more than 2 under one key, and a cap of 1 would leave 60 of their names with no answer.
Each one more doubles the readings, so 4 leaves room and stays cheap.
"""


class NoAnswer(Enum):
    """Why an aired Sonarr's answer to a parse can't be known, where an empty answer says it matches no episode.

    `AiredMatcher.matches` gives the first three. The other two are a series with no aired view to match over.
    """

    SONARR_FAILS = auto()
    """Sonarr answers the name with an error: two episodes share a number it reads one episode by."""

    LIMITED_ALIAS = auto()
    """An alternate title may count for some release names only, and the name matches differently by it."""

    SPECIAL_BY_TITLE = auto()
    """Sonarr would search the release's name for a special's title (`FindEpisodeByTitle`), which isn't copied."""

    EPISODES_UNREAD = auto()
    """Sonarr's episodes of the series couldn't be read this run."""

    ORDERS_UNREAD = auto()
    """Sonarr numbers the series by another order, and its aired order or its episode details aren't stored."""


def parse_without_answer(info: ParsedFileInfo, reason: NoAnswer) -> ParsedFileInfo | None:
    """The parse the placement reads when an aired Sonarr's answer to it can't be known. None leaves it unread.

    An error answer, and a series whose episodes couldn't be read, leave the parse unread, as a failed `/parse` does.
    Every other reason gives it no matched episodes, so the placement reads the name's own numbers.
    """

    match reason:
        case NoAnswer.SONARR_FAILS | NoAnswer.EPISODES_UNREAD:
            return None
        case NoAnswer.LIMITED_ALIAS | NoAnswer.SPECIAL_BY_TITLE:
            return info.with_matched(())
        case NoAnswer.ORDERS_UNREAD:
            # Not unread: orders are fetched only for a series a run checks or waits on, so another series' may never
            # be stored, and leaving its files unread would hold their release on every run.
            return info.with_matched(())
    assert_never(reason)


def sonarr_title_key(title: str) -> str:
    """The key Sonarr files a series title or alternate title under: lowercase letters and digits only.

    A blank title comes back unchanged. In any other, a `%` after a number reads "percent", the joining words (a,
    an, the, and, or, of) are dropped unless the title starts or ends with them, and so is every character that
    isn't a word character. What is left is lowercased and stripped of accents.
    """

    if not title or title.isspace():
        return title
    if "%" in title:
        title = _spelling_percent(title)
    kept = _without_spans(title, [found.span() for found in _JOINING_WORD.finditer(title)])
    return "".join(char.lower() for char in kept if char != "_" and _is_word(char)).translate(UNACCENTED)


def _spelling_percent(title: str) -> str:
    """The title with "percent" for each `%` that ends a number starting at a word boundary (`PercentRegex`)."""

    ends = [found.end() for found in _PERCENT.finditer(title)]
    return _without_spans(title, [(end - 1, end) for end in ends], "percent")


def _without_spans(text: str, spans: Sequence[tuple[int, int]], filler: str = "") -> str:
    """`text` with each of the `spans`, in order and apart, replaced by `filler`."""

    parts: list[str] = []
    at = 0
    for start, end in spans:
        parts += (text[at:start], filler)
        at = end
    return "".join((*parts, text[at:]))


def _is_word(char: str) -> bool:
    r"""Whether .NET's `\w` matches the character, so `NormalizeRegex`'s `\W` keeps it."""

    return unicodedata.category(char) in _WORD_CATEGORIES


def _filtered(candidates: Sequence[AlternateTitle]) -> Iterator[Sequence[AlternateTitle]]:
    """Each reading of the candidates: a list Sonarr's release-name patterns could leave (`FilterSceneMappings`).

    An alias with no comment has no pattern. One with a comment may have one, and it then passes a release's name
    or doesn't. When any alias's pattern passes, Sonarr keeps the passing aliases only. When none does, it keeps
    the aliases that have no pattern.
    """

    commented = [index for index, alias in enumerate(candidates) if alias.comment]
    for count in range(len(commented) + 1):
        for chosen in map(frozenset, combinations(commented, count)):
            yield [alias for index, alias in enumerate(candidates) if index in chosen or not alias.comment]
            if chosen:
                yield [alias for index, alias in enumerate(candidates) if index in chosen]


class _SeasonMap(NamedTuple):
    """The scene season an alias starts at and the TVDB season that is, as Sonarr ranks aliases."""

    scene: int
    tvdb: int


def _season_map(alias: AlternateTitle) -> _SeasonMap | None:
    """The alias's two seasons, when it gives both (`FilterSceneMappings`' test of a mapping alias)."""

    scene, tvdb = alias.scene_season_number, alias.season_number
    if scene is None or tvdb is None or _NO_SEASON in (scene, tvdb):
        return None
    return _SeasonMap(scene, tvdb)


def _tvdb_season_descending(alias: AlternateTitle) -> tuple[bool, int]:
    """Sorts aliases by TVDB season, highest first and one with none last (`FindMappings`' `OrderByDescending`)."""

    return (alias.season_number is None, -(alias.season_number or 0))


def _titled(candidates: Sequence[AlternateTitle], title: str) -> Sequence[AlternateTitle]:
    """Several candidates narrowed to those titled exactly `title`, or else as the closest one is (`FindMappings`)."""

    if len(candidates) <= 1:
        return candidates
    exact = sorted((alias for alias in candidates if alias.title == title), key=_tvdb_season_descending)
    if exact:
        return exact
    closest = min(candidates, key=lambda alias: (_edit_cost(title, alias.title), _tvdb_season_descending(alias)))
    return [alias for alias in candidates if alias.title == closest.title]


def _for_season(candidates: Sequence[AlternateTitle], scene_season: int | None) -> AlternateTitle | None:
    """The candidate a name of that season is read by (`FilterSceneMappings`, as `FindSceneMapping` takes its first).

    A `scene_season` of None is a name read with no season, as `GetSceneSeasonNumber` reads one. Such a name, or
    any name when no candidate maps a season, gets the first candidate that doesn't. A name with a season gets, of
    the candidates that do and start at or before its season, the latest.
    """

    mapping = [(seasons, alias) for alias in candidates if (seasons := _season_map(alias)) is not None]
    if scene_season is None or not mapping:
        return next((alias for alias in candidates if _season_map(alias) is None), None)
    started = [(seasons, alias) for seasons, alias in mapping if seasons.scene <= scene_season]
    if not started:
        return None
    _, latest = max(started, key=lambda pair: pair[0])
    return latest


def _edit_cost(text: str, other: str) -> int:
    """What turning `text` into `other` costs at Sonarr's prices (`LevenshteinDistance`, as `FindMappings` calls it).

    Dropping a character costs 1, adding or swapping one costs 10.
    """

    row = [column * _ADD for column in range(len(other) + 1)]
    for char in text:
        diagonal, row[0] = row[0], row[0] + _DROP
        for column, wanted in enumerate(other, start=1):
            swapped = diagonal + (0 if char == wanted else _SWAP)
            diagonal = row[column]
            row[column] = min(row[column - 1] + _ADD, diagonal + _DROP, swapped)
    return row[-1]


class _SonarrFails(Exception):
    """Raised where Sonarr's lookup for one episode finds two."""


def _one(found: _Found) -> _Found:
    """What a lookup for one episode found, as Sonarr's `SingleOrDefault` takes it."""

    if len(found) > 1:
        raise _SonarrFails
    return found


def _grouped[K](
    episodes: Iterable[SonarrEpisode], key: Callable[[SonarrEpisode], K | None]
) -> dict[K, tuple[MatchedEpisode, ...]]:
    """The episodes that have a `key`, as matches, under it."""

    groups: dict[K, list[MatchedEpisode]] = {}
    for episode in episodes:
        if (
            (group := key(episode)) is not None
            and episode.season_number is not None
            and episode.episode_number is not None
        ):
            groups.setdefault(group, []).append(
                MatchedEpisode(
                    season_number=episode.season_number, episode_number=episode.episode_number, id=episode.id
                )
            )
    return {group: tuple(found) for group, found in groups.items()}


def _pair(season: int | None, episode: int | None) -> EpisodeKey | None:
    return EpisodeKey(season, episode) if season is not None and episode is not None else None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Episodes:
    """A series' episodes under each number Sonarr looks one up by (`EpisodeRepository`'s lookups)."""

    numbered: Mapping[EpisodeKey, _Found]
    absolute: Mapping[int, _Found]
    in_season: Mapping[int, _Found]
    scene_numbered: Mapping[EpisodeKey, _Found]
    scene_absolute: Mapping[int, _Found]
    in_scene_season: Mapping[int, _Found]
    titled_specials: bool
    """Whether any special has a title, so a search of a release's name for one could find it."""

    @classmethod
    def of(cls, episodes: Sequence[SonarrEpisode]) -> Self:
        return cls(
            numbered=_grouped(episodes, lambda ep: _pair(ep.season_number, ep.episode_number)),
            absolute=_grouped(episodes, lambda ep: ep.absolute_episode_number),
            in_season=_grouped(episodes, lambda ep: ep.season_number),
            scene_numbered=_grouped(episodes, lambda ep: _pair(ep.scene_season_number, ep.scene_episode_number)),
            scene_absolute=_grouped(episodes, lambda ep: ep.scene_absolute_episode_number),
            in_scene_season=_grouped(episodes, lambda ep: ep.scene_season_number),
            titled_specials=any(ep.season_number == 0 and ep.title for ep in episodes),
        )

    def one(self, season: int, episode: int, *, scene_source: bool) -> _Found:
        """The episodes releases number so, for a name numbered as releases are, else the one TVDB does.

        It is `GetStandardEpisodes`' lookup for one number (`FindEpisodesBySceneNumbering`, then `Find`).
        """

        key = EpisodeKey(season, episode)
        by_scene = self.scene_numbered.get(key, ()) if scene_source else ()
        return by_scene or _one(self.numbered.get(key, ()))

    def season(self, season: int, *, scene_source: bool) -> _Found:
        """The season as releases number it, for a name numbered as releases are, else as TVDB does.

        It is `GetSeasonEpisodes`' lookup (`GetEpisodesBySceneSeason`, then `GetEpisodes`).
        """

        by_scene = self.in_scene_season.get(season, ()) if scene_source else ()
        return by_scene or self.in_season.get(season, ())


@dataclass(frozen=True, slots=True, kw_only=True)
class AiredMatcher:
    """One series' aired episodes and alternate titles, ready for Sonarr's matching of any parse to them."""

    _episodes: _Episodes
    _aliases: Mapping[str, tuple[AlternateTitle, ...]]
    """The series' alternate titles under their title keys, in Sonarr's order."""

    @classmethod
    def of(cls, series: AiredSeries, aliases: Iterable[AlternateTitle]) -> Self:
        """The matcher of a series' aired view and its alternate titles."""

        by_key: dict[str, list[AlternateTitle]] = {}
        for alias in aliases:
            by_key.setdefault(sonarr_title_key(alias.title), []).append(alias)
        return cls(
            _episodes=_Episodes.of(series.aired),
            _aliases=MappingProxyType({key: tuple(found) for key, found in by_key.items()}),
        )

    def matches(self, info: ParsedFileInfo) -> tuple[MatchedEpisode, ...] | NoAnswer:
        """The episodes Sonarr matches the parse to, in the order of the name's numbers, or why there is no answer.

        A title key with more aliases with a comment than `_MOST_COMMENTED` has no answer.
        """

        candidates = self._aliases.get(sonarr_title_key(info.series_title), ())
        if sum(1 for alias in candidates if alias.comment) > _MOST_COMMENTED:
            return NoAnswer.LIMITED_ALIAS
        answers = [self._answer(info, _titled(left, info.series_title)) for left in _filtered(candidates)]
        return answers[0] if all(answer == answers[0] for answer in answers) else NoAnswer.LIMITED_ALIAS

    def _answer(self, info: ParsedFileInfo, candidates: Sequence[AlternateTitle]) -> _Found | NoAnswer:
        """The match under one reading of the aliases the name's series title can be read by."""

        try:
            return self._matched(info, candidates)
        except _SonarrFails:
            return NoAnswer.SONARR_FAILS

    def _matched(self, info: ParsedFileInfo, candidates: Sequence[AlternateTitle]) -> _Found | NoAnswer:
        """The episodes the parse matches when its title is read by `candidates` (`Map` and `GetEpisodes`).

        The alias the name's season is read by maps that season and says whether the name is numbered as releases
        number the series. A whole season, absolute numbers, an episode 0 and a special each have their own lookup.
        """

        season = _season(info)
        if season is None:
            return ()
        alias = _for_season(candidates, season)
        mapped = season if alias is None else _mapped_season(alias, season)
        scene_source = alias is None or alias.scene_origin != _TVDB_ORIGIN
        if info.full_season:
            return self._episodes.season(mapped, scene_source=scene_source)
        if info.absolute_episode_numbers:
            return self._by_absolute(info, _for_season(candidates, None), scene_source=scene_source)
        if season != 0 and info.episode_numbers == (0,):
            if EpisodeKey(season, 0) in self._episodes.scene_numbered:
                return self._episodes.one(season, 0, scene_source=False)
            if self._episodes.titled_specials:
                return NoAnswer.SPECIAL_BY_TITLE
        if info.special and mapped != 0:
            return ()
        return tuple(
            found
            for episode in info.episode_numbers
            for found in self._episodes.one(mapped, episode, scene_source=scene_source)
        )

    def _by_absolute(self, info: ParsedFileInfo, any_season: AlternateTitle | None, *, scene_source: bool) -> _Found:
        """The episode of each absolute number in the name (`GetAnimeEpisodes`).

        `any_season` is the alias the name is read by with no season (`GetSceneSeasonNumber`). A number is read
        within a season when that alias names one, or else the name gives one past the first and no episode number.
        Read within none, it is the one episode releases number so series-wide. A number that finds nothing so is
        the one TVDB numbers so.
        """

        season = any_season.scene_season_number if any_season is not None else None
        if season is None and (info.season_number or 0) > 1 and not info.episode_numbers:
            season = info.season_number
        matched: list[MatchedEpisode] = []
        for number in info.absolute_episode_numbers:
            found: _Found = ()
            if info.special:
                found = self._episodes.one(0, number, scene_source=False)
            elif scene_source and season is not None:
                found = self._episodes.one(season, number, scene_source=True)
            elif scene_source and len(by_scene := self._episodes.scene_absolute.get(number, ())) == 1:
                found = by_scene
            matched += found or _one(self._episodes.absolute.get(number, ()))
        return tuple(matched)


def _season(info: ParsedFileInfo) -> int | None:
    """The season the name gives. A negative one is Sonarr's way of writing none."""

    return info.season_number if info.season_number is not None and info.season_number >= 0 else None


def _mapped_season(alias: AlternateTitle, season: int) -> int:
    """The season a name's season is, read by the alias picked for it (`MapSeasonNumber`).

    An alias with a scene season and a TVDB season of 0 or more shifts the name's season by the gap between the two.
    An alias with a scene season and no TVDB season turns a first season into its own.
    """

    scene, tvdb = alias.scene_season_number, alias.season_number
    if tvdb is not None and tvdb >= 0 and scene is not None:
        return season + tvdb - scene
    if tvdb is None and scene is not None and scene != _NO_SEASON and season == 1:
        return scene
    return season
