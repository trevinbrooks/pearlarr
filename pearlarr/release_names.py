"""Pure name reading: the offline SxxExx parse, sort keys, stems and versions, numbered runs, the title tie-break."""

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import takewhile
from typing import NamedTuple, Self

from .manual_import import EntryNames
from .seadex_types import ParsedFileInfo

_SXXEXX: re.Pattern[str] = re.compile(r"[Ss](\d{1,2})[\s._-]*[Ee](\d{1,3})")


def parse_se_from_filename(name: str) -> ParsedFileInfo | None:
    """Offline `SxxExx` fallback for when Sonarr's `/parse` is unreachable: one key pulled from the leaf, else None.

    Marked `offline` because the regex knows nothing about absolute numbers: a dual-numbered name parsed here
    would otherwise launder its lost absolute into a "known" parse and blind the positional leg's tell.
    """

    m = _SXXEXX.search(name)
    if not m:
        return None
    return ParsedFileInfo(
        season_number=int(m.group(1)),
        episode_numbers=(int(m.group(2)),),
        offline=True,
    )


_BRACKETED = re.compile(r"[\[(][^\[\]()]*[\])]")
_TRAILING_TAG = re.compile(rf"\s*{_BRACKETED.pattern}$")
_NON_WORD = re.compile(r"[^0-9a-z]+")
_GROUP_TAG = re.compile(r"^\[([0-9a-z]+(?:[-_.][0-9a-z]+)*)\]", re.IGNORECASE)
_TRAILING_VERSION = re.compile(r"v(\d+)$")
# The episode of an "S02E01" key (a keyed run is judged by its keys), else
# the LAST " - NN - " (an "Episode" word may lead the number, a "vN" and
# the absolute in brackets may trail it, and a title follows), else the
# episode of a packed "S0101" (a season the series lacks, read as the
# release's count), else a trailing 1-3 digit integer not glued to more
# digits (a year or CRC tail is no release number).
_KEYED_NUMBER = re.compile(r"^(.*?[Ss]\d{1,2}[Ee])(\d{1,3})(?:v(?P<version>\d+))?(?!\d)")
_MIDDLE_NUMBER = re.compile(
    r"^(.*) - (?:[Ee]pisode |[Ee]p\.? )?(\d{1,3})(?:v(?P<version>\d+))?(?: \[\d{1,3}\])?(?= - )"
)
_PACKED_NUMBER = re.compile(r"^(.*?(?:^|[\s._-])[Ss]\d{2})(\d{2})(?=[\s._-]|$)")
_TRAILING_NUMBER = re.compile(r"^(.*?)(?<!\d)(\d{1,3})$")
_NUMBER_FORMS = (_KEYED_NUMBER, _MIDDLE_NUMBER, _PACKED_NUMBER, _TRAILING_NUMBER)
# The separators between a release number and the title after it.
_LEADING_SEPARATOR = re.compile(r"^[\s._-]+")
# A word that marks an extra, numbered or not ("PV 01", "NCOP2", "ED1a", "menus"):
# an opening or ending (creditless too), a preview, a menu, a commercial, a trailer, a teaser, a promo.
_EXTRAS_WORD = re.compile(
    r"^(?:nc(?:op|ed)|menu|preview|trailer|teaser|promo)(?:s|\d{1,2}[a-z]?)?$"
    r"|^(?:op|ed|pv|cm)(?:\d{1,2}[a-z]?)?$|^creditless$|^commercials?$"
)


class Stem(NamedTuple):
    """A name shed of its extension, trailing tags and `vN`s (to a fixpoint), and the highest `vN` shed (1 when none)."""

    text: str
    version: int


@dataclass(frozen=True, slots=True)
class RunMember:
    """One file's numbered-run membership, read purely from its name."""

    name: str
    prefix: str
    """The text before the release number: the grouping key."""
    number: int
    tail_words: tuple[str, ...]
    """The folded words after the release number, its separator stripped (empty when nothing follows)."""
    version: int
    """The `vN` after the number or trailing (1 when none): of two names sharing a number, the higher is the member."""


class FileIdentity(NamedTuple):
    """What a name is a version of: a member's prefix and release number, else its stem alone."""

    text: str
    """A member's prefix before its release number, else the whole stem."""
    number: int | None
    """A member's release number, else None."""


class NameRead(NamedTuple):
    """One name read once: its stem, its run membership when it carries a release number, and its title words."""

    stem: Stem
    member: RunMember | None
    title_words: tuple[str, ...]
    """The words an episode title is read from: a member's tail, else the whole stem."""

    @property
    def version(self) -> int:
        """The name's `vN` (1 when none), trailing or after its release number."""

        return self.member.version if self.member is not None else self.stem.version

    @property
    def file(self) -> FileIdentity:
        """The file the name is a version of."""

        member = self.member
        return FileIdentity(member.prefix, member.number) if member is not None else FileIdentity(self.stem.text, None)


def read_name(name: str) -> NameRead:
    """Read a name once: its stem and version, its run membership, and its title words."""

    stem = _stem_version(name)
    member = _member_of(name, stem)
    return NameRead(stem, member, member.tail_words if member is not None else folded_words(stem.text))


def _stem_version(name: str) -> Stem:
    """The stem plus the highest trailing `vN` it shed (1 when none)."""

    stem = (name.rsplit(".", 1)[0] if "." in name else name).replace("_", " ")
    version = 1
    while True:
        trimmed = _TRAILING_TAG.sub("", stem).rstrip(" .-")
        if (found := _TRAILING_VERSION.search(trimmed)) is not None:
            version = max(version, int(found.group(1)))
            trimmed = trimmed[: found.start()].rstrip(" .-")
        if trimmed == stem:
            return Stem(stem, version)
        stem = trimmed


def _member_of(name: str, stem: Stem) -> RunMember | None:
    """The release's own number in a name, read purely from the text.

    The stem's separators around the number are dropped, so
    "show_-_07v2_[bd 1080p].mkv" reads as ("show", 7). None when no form fits.
    """

    match = next((found for form in _NUMBER_FORMS if (found := form.match(stem.text)) is not None), None)
    if match is None:
        return None
    prefix = match.group(1).rstrip(" .-")
    tail = _LEADING_SEPARATOR.sub("", stem.text[match.end() :])
    version = stem.version
    if (middle := match.groupdict().get("version")) is not None:
        version = max(version, int(middle))
    return RunMember(name, prefix, int(match.group(2)), folded_words(tail), version)


def natural_key(name: str) -> str:
    """Digit-aware sort key ("sp10" sorts after "sp2"): zero-pad digit runs."""

    return re.sub(r"\d+", lambda match: match.group().zfill(12), name)


def _raw_words(text: str) -> list[str]:
    """The lowercase words of a name as written: bracket groups kept, nothing folded."""

    return [word for word in _NON_WORD.split(text.casefold()) if word]


def is_extras_name(name: str, shielded: frozenset[str]) -> bool:
    """Whether the name carries an extras word outside `shielded` (the entry's and its episode title's words).

    A leading one-token bracket ("[CMS]", "[Ed-Subs]") is the release group's tag, unless the token is an
    extras word itself ("[NCOP]").
    """

    tag = _GROUP_TAG.match(name)
    body = name if tag is None or _EXTRAS_WORD.match(tag.group(1).casefold()) is not None else name[tag.end() :]
    return any(word not in shielded and _EXTRAS_WORD.match(word) is not None for word in _raw_words(body))


def is_consecutive(numbers: Sequence[int]) -> bool:
    """Whether the numbers count up by one from the first (an empty sequence does)."""

    return not numbers or list(numbers) == list(range(numbers[0], numbers[0] + len(numbers)))


class NumberedRun(NamedTuple):
    """One prefix's numbered members, number order."""

    prefix: str
    """The text before the number every member shares: the grouping key."""
    members: tuple[RunMember, ...]
    superseded: tuple[RunMember, ...]
    """The lower versions a member's higher `vN` displaced: duplicates once the run is placed."""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self.members)

    @property
    def whole(self) -> tuple[RunMember, ...]:
        """The members and the lower versions they displaced."""

        return (*self.members, *self.superseded)

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(member.number for member in self.members)

    def consecutive(self, width: int) -> bool:
        """Whether the run is exactly `width` consecutive numbers, wherever it starts."""

        return len(self.members) == width and is_consecutive(self.numbers)


def numbered_runs(members: Iterable[RunMember], parsed: Mapping[str, ParsedFileInfo | None]) -> list[NumberedRun]:
    """Every numbered run among the members, grouped by prefix.

    A member whose one parsed absolute disagrees with its release number is dropped. Of several members
    sharing a number the highest `vN` stays and the rest are superseded (equal versions all stay).
    """

    by_prefix: dict[str, dict[int, list[RunMember]]] = {}
    for member in members:
        info = parsed.get(member.name)
        absolutes: set[int] = set(info.absolute_episode_numbers) if info is not None else set()
        if len(absolutes) == 1 and member.number not in absolutes:
            continue
        by_prefix.setdefault(member.prefix, {}).setdefault(member.number, []).append(member)
    return [_run_of(prefix, by_number) for prefix, by_number in by_prefix.items()]


def _member_order(member: RunMember) -> tuple[int, str]:
    """Number order, name order within a number (names are unique per batch)."""

    return (member.number, member.name)


def _run_of(prefix: str, by_number: Mapping[int, Sequence[RunMember]]) -> NumberedRun:
    """One prefix's run: the top version at each number kept, the lower ones superseded."""

    kept: list[RunMember] = []
    superseded: list[RunMember] = []
    for versions in by_number.values():
        top = max(member.version for member in versions)
        kept.extend(member for member in versions if member.version == top)
        superseded.extend(member for member in versions if member.version < top)
    return NumberedRun(prefix, tuple(sorted(kept, key=_member_order)), tuple(superseded))


def runs_with_numbers(runs: Iterable[NumberedRun], numbers: Iterable[int]) -> list[NumberedRun]:
    """The runs numbered exactly `numbers`."""

    expected = tuple(numbers)
    return [run for run in runs if run.numbers == expected]


def runs_from_one(runs: Iterable[NumberedRun], width: int) -> list[NumberedRun]:
    """The runs numbered `1..width`."""

    return runs_with_numbers(runs, range(1, width + 1))


# A title names a candidate when they share at least half their leftover words.
_MIN_TITLE_OVERLAP = 0.5
# An episode title names a file from this many words past the series title and the labels below.
_MIN_TITLE_WORDS = 2
# The words that label a title without being one: "Part 2" and "OVA 1" name nothing.
_TITLE_LABELS = frozenset({"part", "episode", "ep", "special", "sp", "ova", "oad", "movie", "recap", "vol", "chapter"})
# A title read from a name's tail ends where a quality word begins: one holding a letter and
# a digit ("1080p", "x265") or one of these. A bare number does not end it ("... Man 1").
_QUALITY_WORDS = frozenset({"bluray", "bdrip", "webrip", "remux", "repack", "proper"})
# A count label beside its number says nothing the number does not.
_COUNT_LABELS = frozenset({"season", "episode", "ep"})

# The words that count a season, folded to its plain number so "2nd Season",
# "Season 2", "S2" and "II" agree. Lone "I", "V" and "X" stay words.
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|[nr]d|th)$")
_SEASON_TOKEN = re.compile(r"^s(\d{1,2})$")
_COUNT_WORDS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "vi": "6",
    "vii": "7",
    "viii": "8",
}


def _count_word(word: str) -> str | None:
    """The plain number a word counts with ("02", "2nd", "second", "s2", "ii"), else None."""

    if word.isdigit():
        return str(int(word))
    if (found := _ORDINAL.match(word) or _SEASON_TOKEN.match(word)) is not None:
        return str(int(found.group(1)))
    return _COUNT_WORDS.get(word)


def folded_words(text: str) -> tuple[str, ...]:
    """The words of a title or name in order: case and accents folded, bracketed groups dropped, seasons counted plainly."""

    folded = unicodedata.normalize("NFKD", _BRACKETED.sub(" ", text)).encode("ascii", "ignore").decode()
    words = [_count_word(word) or word for word in _raw_words(folded)]
    return tuple(
        word
        for index, word in enumerate(words)
        if word not in _COUNT_LABELS
        or not any(words[near].isdigit() for near in (index - 1, index + 1) if 0 <= near < len(words))
    )


def _word_set(text: str) -> frozenset[str]:
    """The distinct words of a title or name."""

    return frozenset(folded_words(text))


class TitleGround(NamedTuple):
    """The title words an entry's names are read against."""

    series_words: frozenset[str]
    """The series title's words: an episode title names a file past them."""
    entry_words: frozenset[str]
    """The series and AniList titles' words: an extras word among them is a title word, not an extra."""

    @classmethod
    def of(cls, names: EntryNames) -> Self:
        """The series title's words, and those plus every AniList title's."""

        series = _word_set(names.series)
        return cls(series, series.union(*(_word_set(title) for title in names.anilist)))


def is_distinct_title(words: Iterable[str]) -> bool:
    """Whether the words say enough to name an episode: `_MIN_TITLE_WORDS` that are neither a number nor a label."""

    return sum(not word.isdigit() and word not in _TITLE_LABELS for word in words) >= _MIN_TITLE_WORDS


def _is_quality_word(word: str) -> bool:
    """Whether the word is quality junk: one of `_QUALITY_WORDS`, or letters and digits mixed ("1080p", "x265")."""

    return word in _QUALITY_WORDS or (any(c.isalpha() for c in word) and any(c.isdigit() for c in word))


def opens_title(words: tuple[str, ...], title: tuple[str, ...]) -> bool:
    """Whether the title's words open the words, and the words end there or go on with quality junk."""

    count = len(title)
    return words[:count] == title and (len(words) == count or _is_quality_word(words[count]))


class _Leftover(NamedTuple):
    """An AniList title's words beyond the series title, in title order, and as a set."""

    words: tuple[str, ...]
    distinct: frozenset[str]

    @classmethod
    def of(cls, title: str, ground: frozenset[str]) -> Self:
        words = tuple(word for word in folded_words(title) if word not in ground)
        return cls(words, frozenset(words))

    def overlap(self, rest: frozenset[str]) -> float:
        """The share of the combined leftover words a candidate's leftover has in common with this title."""

        return len(rest & self.distinct) / len(rest | self.distinct)

    def opening_only(self, rest: frozenset[str]) -> bool:
        """Whether a candidate shares nothing past this title's opening run of words.

        A title opens with the franchise name in its own language, which the
        series title's words cannot shed, and names the entry after it.
        """

        opening = frozenset(takewhile(rest.__contains__, self.words))
        return opening != self.distinct and opening == rest & self.distinct


class _Scored(NamedTuple):
    """One candidate's leftover words and its best overlap with a title."""

    rest: frozenset[str]
    score: float


@dataclass(frozen=True, slots=True)
class Naming:
    """Every candidate scored against the entry's AniList titles, both sides shed of the series title's words.

    Built once over every candidate. The questions take index subsets, and each answer depends only on
    the subset's own scores, so one naming serves every tie-break and refusal over the same candidates.
    """

    leftovers: tuple[_Leftover, ...]
    scored: tuple[_Scored, ...]

    @classmethod
    def of(cls, candidates: Sequence[str], names: EntryNames) -> Self | None:
        """Score the candidates, or None when nothing can name them: no ground, no titles, or a title that IS the series."""

        ground = _word_set(names.series)
        leftovers = tuple(_Leftover.of(title, ground) for title in names.anilist)
        if not ground or not leftovers or not all(leftover.distinct for leftover in leftovers):
            return None
        scored: list[_Scored] = []
        for candidate in candidates:
            rest = _word_set(candidate) - ground
            scored.append(_Scored(rest, max(leftover.overlap(rest) for leftover in leftovers)))
        return cls(leftovers, tuple(scored))

    def best_among(self, indices: Iterable[int]) -> frozenset[int]:
        """The candidates among `indices` at the top score, when it reaches `_MIN_TITLE_OVERLAP`."""

        among = frozenset(indices)
        top = max((self.scored[index].score for index in among), default=0.0)
        if top < _MIN_TITLE_OVERLAP:
            return frozenset()
        return frozenset(index for index in among if self.scored[index].score == top)

    def sole_among(self, indices: Iterable[int]) -> int | None:
        """The one candidate among `indices` a title names, else None: a tie or an opening-words winner refuses."""

        winners = self.best_among(indices)
        if len(winners) != 1:
            return None
        index = next(iter(winners))
        return index if self._named(index) else None

    def _named(self, index: int) -> bool:
        """Whether a winner shares more than the opening words of some title it scores best against."""

        rest, score = self.scored[index]
        return any(leftover.overlap(rest) == score and not leftover.opening_only(rest) for leftover in self.leftovers)

    def names_another(self, pick: int, among: Iterable[int]) -> bool:
        """Whether a title names candidates among `among` best and not `pick`. Refuses the pick, never promotes."""

        winners = self.best_among(among)
        return bool(winners) and pick not in winners


def sole_title_match(candidates: Sequence[str], names: EntryNames) -> int | None:
    """The index of the one candidate an AniList title names, else None.

    The unique best sharing at least `_MIN_TITLE_OVERLAP` of the combined leftover words wins, unless it
    shares only the opening words of every title it scores best against. Refuses, never promotes a runner-up.
    """

    naming = Naming.of(candidates, names)
    return None if naming is None else naming.sole_among(range(len(candidates)))
