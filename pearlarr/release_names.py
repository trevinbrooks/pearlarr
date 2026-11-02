"""Read file names without Sonarr.

The offline SxxExx parse, range keys, sort keys, stems, versions, runs, the entry's title told apart from a name's
numbers (`TitleGround`, `read_names`), and title matching.
"""

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import compress, takewhile
from typing import NamedTuple, Self

from .manual_import import EntryNames
from .seadex_types import EpisodeKey, ParsedFileInfo

_SXXEXX: re.Pattern[str] = re.compile(r"[Ss](\d{1,2})[\s._-]*[Ee](\d{1,3})")
# "S01E05-06" or "S01E05-E06" with no spaces, maybe a "v2" after it. A letter or digit touching either end, as in
# "S01E05-720p", or a one-digit last number, as in a version-like "S01E01-2.0", means the name says something else.
_RANGE_KEY = re.compile(
    r"(?<![a-z\d])s(?P<season>\d{1,2})e(?P<first>\d{1,3})-e?(?P<last>\d{2,3})(?:v\d+)?(?![a-z\d])", re.I
)


class RangeKey(NamedTuple):
    """The season and the first and last episode numbers of a name's `SxxEyy-zz` range."""

    season: int
    first: int
    last: int


def parse_se_from_filename(name: str) -> ParsedFileInfo | None:
    """Offline `SxxExx` fallback for when Sonarr's `/parse` is unreachable: one key pulled from the leaf, else None.

    Marked `offline` because the regex knows nothing about absolute numbers: a dual-numbered name parsed here
    would otherwise pass as a known parse and hide a shared absolute from the absolute zip's check.
    """

    m = _SXXEXX.search(name)
    if not m:
        return None
    return ParsedFileInfo(
        season_number=int(m.group(1)),
        episode_numbers=(int(m.group(2)),),
        offline=True,
    )


def range_key(name: str) -> RangeKey | None:
    """The tight `SxxEyy-zz` range in `name`, or None. It doesn't check the order, so `S01E07-05` reads too."""

    found = _RANGE_KEY.search(name)
    if found is None:
        return None
    return RangeKey(int(found["season"]), int(found["first"]), int(found["last"]))


_BRACKETED = re.compile(r"[\[(][^\[\]()]*[\])]")
_TRAILING_TAG = re.compile(rf"\s*{_BRACKETED.pattern}$")
_NON_WORD = re.compile(r"[^0-9a-z]+")
_GROUP_TAG = re.compile(r"^\[([0-9a-z]+(?:[-_.][0-9a-z]+)*)\]", re.IGNORECASE)
# A trailing "v2" or "v2.1": the minor part says nothing the major does not.
_TRAILING_VERSION = re.compile(r"v(\d+)(?:\.\d+)?$")
# The release number, by the first form that fits: the episode of an "S02E01" key (a keyed run is judged by its
# keys); the " - NN - " segment (`_middle_count`); the episode of a packed "S0101" (a season the series lacks,
# read as the release's count); a trailing number (`_trailing_count`); the scene-dotted count (`_dotted_count`).
_KEYED_NUMBER = re.compile(r"^(.*?[Ss]\d{1,2}[Ee])(?P<number>\d{1,3})(?:v(?P<version>\d+))?(?!\d)")
# An "Episode" word may lead the number, a "vN" and the absolute in brackets may trail it, and a title follows;
# "0517" and "1001" count past the thousandth episode. A second such segment is a title's number.
_MIDDLE_SEGMENT = re.compile(
    r" - (?:[Ee]pisode |[Ee]p\.? )?(?P<number>\d{1,4})(?:v(?P<version>\d+))?(?: \[\d{1,3}\])?(?= - )"
)
_MIDDLE_NUMBER = re.compile(rf"^(.*?){_MIDDLE_SEGMENT.pattern}")
_PACKED_NUMBER = re.compile(r"^(.*?(?:^|[\s._-])[Ss]\d{2})(?P<number>\d{2})(?=[\s._-]|$)")
# A trailing 1-3 digit integer not glued to more digits, or four behind a leading zero ("Show - 0517") or behind
# " - " ("Show - 1001"). Glued to a hyphen ("-Group-25") or a lone "s" ("Show S1") it is a group's suffix or a
# season token (`read_names` reads a pack's set of those); behind a digit and a dot ("Show - 21.5", "FLAC2.0") it
# is a decimal's fraction. A plain "Season N" ending the name is a season token too ("[BD Menu 01] Show - Season
# 01"); behind an ordinal, digit or word, the number counts ("Show 2nd Season 25", "The Final Season 25").
_TRAILING_NUMBER = re.compile(
    r"^(.*?)(?<!\d)(?<!\d\.)(?<![a-zA-Z0-9]-)(?<![\s._-][sS])(?P<number>\d{1,3}|0\d{3}|(?<= - )\d{4})$"
)
_ORDINAL_WORD = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|final|last"
# An ordinal and its separator, ahead of a "season" word: "2nd ", "Final.".
_ORDINAL_LEAD = rf"(?:\d{{1,2}}(?:st|[nr]d|th)|{_ORDINAL_WORD})[\s._-]+"
_SEASON_TAIL = re.compile(rf"(?:^|[^a-z0-9])(?P<ordinal>{_ORDINAL_LEAD})?season[\s._]*$", re.IGNORECASE)
# A pack's specials counted as a set beside its episodes: "Show S1", "Show S2", "Show S3" next to "Show - 01"..
_LONE_S = re.compile(r"^(.*[\s._-][sS])(\d{1,2})$")
# A scene name counts before its quality tail and names codecs, bit depths and volumes after it: the tail starts
# at the first resolution ("1080p", "1920x1080"), else at the first source word ("BluRay", "WEB-DL", "DVDRip").
_RESOLUTION = re.compile(r"(?<![a-z0-9])(?:\d{3,4}[pi]|\d{3,4}x\d{3,4})(?![a-z0-9])", re.IGNORECASE)
_SOURCE = r"blu-?ray|bd(?:rip|remux)?|web(?:-?dl|-?rip)?|hdtv|hdrip|dvd(?:rip)?|remux|uhd"
_SOURCE_WORD = re.compile(rf"(?<![a-z0-9])(?:{_SOURCE})(?![a-z0-9])", re.IGNORECASE)
# A codec's 264-266 behind a lone h or x ("H.264", "Show.X.264").
_CODEC = r"(?<=[^a-z0-9][hx]\.)26[456]"
# A four-digit count: behind a zero, or below 1900.
_LONG_COUNT = r"0\d{3}|1[0-8]\d{2}"
# The scene-dotted count ("Show.101.v4.480p", "Show.1069.1080p"): the head's first three digits between dots, or
# four behind a zero or below 1900 ("Show.2011.001" counts 1), an optional ".vN" behind it. Another count behind it
# in the head drops it ("Show.100.01": the 100 is the title's) unless only the first is zero-padded
# ("Show.007.Day.20.Again"), as does a key anywhere ("Show.101.S03"). A codec's number is neither the count nor
# one behind it.
_DOTTED_NUMBER = re.compile(
    rf"^(.*?[^\s._-])\.(?!{_CODEC}(?=\.|$))(?P<number>{_LONG_COUNT}|\d{{3}})(?:\.v(?P<version>\d+))?(?=\.|$)",
    re.IGNORECASE,
)
# A number a unit follows is a bit depth or a frame rate ("10.bit", "60.fps"), no count and no count's partner.
_NO_UNIT = r"(?![\s._-]*(?:bits?|fps)(?![a-z0-9]))"
# The count right behind the entry's title ("Show 100 Again 12", "Show.100.01", "Show! - 14"): two or three digits,
# four behind a zero or below 1900, an optional "vN", standing alone past the title's own punctuation; not a
# codec's 26x behind a lone h or x ("Show.X.264"), nor a bit depth or frame rate ("Show.10.bit", "Show.60.fps").
# Matched on the blanked text, so a number in a bracket is no count.
_SPAN_COUNT = re.compile(
    rf"[\W_]+(?:episode |ep\.? )?(?!{_CODEC}(?![a-z0-9]))(?P<number>\d{{2,3}}|{_LONG_COUNT})"
    rf"(?:[\s._-]?v(?P<version>\d+))?(?![a-z0-9]){_NO_UNIT}",
    re.IGNORECASE,
)
# What may follow that count: the stem's end, a bracket, a number ("12.1080p"; `_NUMBER_BEHIND` tells a range's
# or decimal's partner apart), a spaced dash, an "END" mark, a source word ("Show.009.01.DVDRip") or a quality
# word ("018.PROPER", "018.Hi10P": `_count_ends`), or release tags running to one of those ("06v2 TV (Dub)",
# "12 MULTi [1080p]", "13 Director's Cut [BDRip ...]"). Any other word there makes it an episode title's number
# ("Show - 20 Years Later", "Show.100.Steps.Home", "Show - 12 Raw Deal").
_COUNT_END = re.compile(
    rf"[\s._-]*(?:$|\d|[\[(])|\s+-\s|[\s._-]+(?:(?-i:END)|{_SOURCE})(?![a-z0-9])|[\s._-]+(?P<word>[^\W_]+)",
    re.IGNORECASE,
)
# A release tag says what the file is, not which episode: a broadcast or disc cut ("TV", "OVA"), a streaming
# source ("CR", "AMZN"), an audio, language or subtitle tag ("MULTi", "Dual-Audio", "Dub", "VOSTFR", "Hardsub"),
# a codec ("HEVC", "AVC", "FLAC"), an edit ("Director's Cut", "Uncut", "Uncensored", "Remastered"), a storyboard;
# any of them with "Version" or "Edition" behind it.
_RELEASE_TAG = re.compile(
    r"(?:tv|ova|ona|oad|cr|amzn|nf|funi|hidive|dsnp|multi|dual(?:[\s._-]*audio)?|audio|dub(?:bed)?|sub(?:bed|s)?"
    r"|vost(?:fr|a)|hardsub(?:bed|s)?|softsub(?:bed|s)?|raw|eng|jpn|jap|english|japanese|hevc|avc|aac|flac|opus"
    r"|ac3|eac3|dts|truehd|hdr|sdr|director'?s?[\s._-]*cut|uncut|uncensored|censored|extended|remaster(?:ed)?"
    r"|storyboard)(?:[\s._-]*(?:version|edition))?(?![a-z0-9])",
    re.IGNORECASE,
)
# A count a name sets off from the title's own number before it: behind a spaced dash or an "Episode" word
# ("Show - 2", "Show Episode 2" under a title "Show 2"), where a name writing the title alone has neither ("Show 2").
_DASH_BEHIND = re.compile(r"[\s._]*-\s")
_LABEL_LAST = re.compile(r"(?:^|[\s._-])(?:episode|ep)\.?$", re.IGNORECASE)
# Two counts a joiner ties into one range ("12 & 13", "12+13", "12, 13", "12 and 13"): the second ends no name.
_JOINED = re.compile(r"\d\s*[&+~,]\s*$|\d\s+(?:and|to)\s+$", re.IGNORECASE)
# A number standing alone behind a count with only separators between: a range's or decimal's partner
# ("01-02", "21.5", "01.5"), unlike a resolution's ("12.1080p") or a bit depth's ("001.10.bit").
_NUMBER_BEHIND = re.compile(rf"[\s._-]*\d+(?![a-z0-9]){_NO_UNIT}", re.IGNORECASE)
# An OVA, ONA or OAD tag right behind a count marks an extra, whose number is none of the plain run's
# ("Show - 13 OVA" behind twelve plain files).
_EXTRA_BEHIND = re.compile(r"[\s._-]+(?:ova|ona|oad)(?![a-z0-9])", re.IGNORECASE)
# What leads a count behind a title: its separators and an "Episode" word, no part of the run's prefix.
_COUNT_LEAD = re.compile(r"(?:[\s._-]+(?:episode|ep\.?))?[\s._-]*$", re.IGNORECASE)
# A number right behind a label, no dash between ("Season 01"): the label's number.
_LABEL_NUMBER = re.compile(r"[\s._]*\d")
_COUNT_BEHIND = re.compile(
    rf"(?<![a-z0-9])(?!{_CODEC}(?:v\d+)?(?=\.|$))(\d{{2,3}}|{_LONG_COUNT})(?:v\d+)?(?=\.|$)", re.IGNORECASE
)
_KEY_TOKEN = re.compile(
    r"(?<![a-z0-9])(?:s\d{1,2}(?:[\s._-]*e\d{1,3})?|e\d{2,3}|\d{1,2}x\d{2,3}|season[\s._-]*\d{1,2}"
    r"|ep(?:isode)?[\s._-]*\d{1,3})(?:v\d+)?(?![a-z0-9])",
    re.IGNORECASE,
)
# An episode token the name spells out: "S01E05", "S01E05E06", "S01E05v2", "1x05", "01E05", or a bare "E05".
# A parse with a season and episode but none of these came from a glued number, a year, or a resolution.
_EXPLICIT_KEY = re.compile(
    r"(?<![a-z0-9])(?:s\d{1,2}[\s._-]*e\d{1,3}(?:e\d{1,3})*|\d{1,2}e\d{2,3}|e\d{2,3}|\d{1,2}x\d{2,3})(?:v\d+)?"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
# A three- or four-digit count standing alone in the stem ("Show.101.v4", "Show - 0517"), and one standing alone
# inside a bracket anywhere in the name ("S10E01 (0337)"). In the quality tail, a number behind a lone h or x and a
# dot ("H.264") is a codec's and no count; in the head it counts ("Show.X.264").
_BARE_NUMBER = re.compile(r"(?<![a-z0-9])(\d{3,4})(?:v\d+)?(?![a-z0-9])", re.IGNORECASE)
_TAIL_NUMBER = re.compile(r"(?<![a-z0-9])(?<![^a-z0-9][hx]\.)(\d{3,4})(?:v\d+)?(?![a-z0-9])", re.IGNORECASE)
_TAGGED_NUMBER = re.compile(r"[\[(](\d{3,4})[\])]")
# A word of a title as written: letters and digits of any script, split at anything else.
_WORD = re.compile(r"[^\W_]+")
# A word that marks an extra, numbered or not ("PV 01", "NCOP2", "ED1a", "menus"):
# an opening or ending (creditless too), a preview, a menu, a commercial, a trailer, a teaser, a promo.
_EXTRAS_WORD = re.compile(
    r"^(?:nc(?:op|ed)|menu|preview|trailer|teaser|promo)(?:s|\d{1,2}[a-z]?)?$"
    r"|^(?:op|ed|pv|cm)(?:\d{1,2}[a-z]?)?$|^creditless$|^commercials?$"
)


class Stem(NamedTuple):
    """A name stripped of its extension, trailing tags, and `vN`s (repeatedly), and its highest `vN` (1 if none)."""

    text: str
    version: int


@dataclass(frozen=True, slots=True)
class RunMember:
    """One file's numbered-run membership, read purely from its name."""

    name: str
    prefix: str
    """The text before the release number: files sharing it form one run."""
    number: int
    tail_words: tuple[str, ...]
    """The folded words after the release number (empty when nothing follows)."""
    version: int
    """The `vN` after the number or trailing (1 when none): of two names sharing a number, the higher is the member."""
    title_numbers: tuple[int, ...] = ()
    """The numbers the entry's title holds in the name ("100" of "Show.100.Again.12")."""


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
    """The words an episode title is read from: a member's tail, the tail behind a `titled_count`, else the whole
    stem."""
    explicit_key: bool
    """Whether the name spells an episode token out (`_EXPLICIT_KEY`). Without one, a parsed key is derived."""
    bare_numbers: tuple[int, ...]
    """The three- and four-digit counts standing alone in the stem, in order."""
    tagged_numbers: tuple[int, ...]
    """The three- and four-digit counts standing alone inside a bracket of the name, in order."""
    titled_count: int | None = None
    """The count behind the entry's title that a word behind it made an episode title's number ("13" of
    "Show - 13 Rainy Days"), when no form counted the name. The placer judges it (`Placer.counted_by_title`)."""

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
    return NameRead(
        stem,
        member,
        member.tail_words if member is not None else folded_words(stem.text),
        explicit_key=_EXPLICIT_KEY.search(stem.text) is not None,
        bare_numbers=tuple(count.number for count in _bare_counts(stem.text)),
        tagged_numbers=tuple(int(found) for found in _TAGGED_NUMBER.findall(name)),
    )


def _head_of(text: str) -> str:
    """The stem before its quality tail: the first resolution, else the first source word, else all of it."""

    found = _RESOLUTION.search(text) or _SOURCE_WORD.search(text)
    return text if found is None else text[: found.start()]


def _group_blanked(text: str) -> str:
    """The text with its leading group tag blanked: a number inside the tag ("[ABC-1234]") is the group's, no count.

    A leading bracket holding a number alone ("[0802]") counts the file, and stays.
    """

    tag = _GROUP_TAG.match(text)
    if tag is None or tag.group(1).isdigit():
        return text
    return " " * tag.end() + text[tag.end() :]


class _BareCount(NamedTuple):
    """A count standing alone in the stem, and where it starts."""

    start: int
    number: int


def _bare_counts(text: str) -> list[_BareCount]:
    """Every count standing alone in the stem, less the group tag's and a codec's number in the quality tail."""

    head = _group_blanked(_head_of(text))
    tail = text[len(head) :]
    counts = [_BareCount(found.start(1), int(found[1])) for found in _BARE_NUMBER.finditer(head)]
    return counts + [_BareCount(len(head) + found.start(1), int(found[1])) for found in _TAIL_NUMBER.finditer(tail)]


def _middle_count(text: str) -> re.Match[str] | None:
    """The " - NN - " count's match, unless a second such segment makes the first a title's number."""

    found = _MIDDLE_NUMBER.match(text)
    if found is None or _MIDDLE_SEGMENT.search(text, found.end()) is not None:
        return None
    return found


def _trailing_count(text: str) -> re.Match[str] | None:
    """The trailing number's match, unless it names no count.

    A plain "Season" leading it makes it a season ("Show - Season 01"), as does an ordinal season's lone digit
    ("The Final Season 2", unlike "The Final Season 25" or "2nd Season 02"); a joiner before it makes it a
    range's second number ("Show - 12 & 13").
    """

    found = _TRAILING_NUMBER.match(text)
    if found is None or _JOINED.search(found.group(1)) is not None:
        return None
    season = _SEASON_TAIL.search(found.group(1))
    if season is not None and (season.group("ordinal") is None or len(found["number"]) == 1):
        return None
    return found


def _dotted_count(text: str) -> re.Match[str] | None:
    """The scene-dotted count's match: the head's first, other than a codec's number.

    None when a key stands anywhere, or when another count follows it in the head unless only the first is padded.
    """

    head = _head_of(text)
    found = _DOTTED_NUMBER.match(head)
    if found is None or _KEY_TOKEN.search(text) is not None:
        return None
    padded = found["number"].startswith("0")
    for later in _COUNT_BEHIND.finditer(head, found.end()):
        if not padded or later.group(1).startswith("0"):
            return None
    return found


def _stem_version(name: str) -> Stem:
    """The stem plus the highest trailing `vN` it shed (1 when none)."""

    stem = name.rsplit(".", 1)[0].replace("_", " ")
    version = 1
    while True:
        trimmed = _TRAILING_TAG.sub("", stem).rstrip(" .-")
        if (found := _TRAILING_VERSION.search(trimmed)) is not None:
            version = max(version, int(found.group(1)))
            trimmed = trimmed[: found.start()].rstrip(" .-")
        if trimmed == stem:
            return Stem(stem, version)
        stem = trimmed


_NUMBER_FORMS: tuple[Callable[[str], re.Match[str] | None], ...] = (
    _KEYED_NUMBER.match,
    _middle_count,
    _PACKED_NUMBER.match,
    _trailing_count,
    _dotted_count,
)


class _Count(NamedTuple):
    """A release number found in a stem: the text before it, the number, where it ends, and the "vN" behind it."""

    prefix: str
    number: int
    end: int
    version: int

    @classmethod
    def of(cls, found: re.Match[str], prefix: str) -> Self:
        """The count a form matched: its number in group "number", its "vN" in group "version" (1 when none)."""

        return cls(prefix, int(found["number"]), found.end(), int(found.groupdict().get("version") or 1))

    def member(self, name: str, stem: Stem) -> RunMember:
        """The run membership the count gives the name.

        The separators around the number are dropped, so "show_-_07v2_[bd 1080p].mkv" reads as ("show", 7).
        """

        tail = folded_words(stem.text[self.end :])
        return RunMember(name, self.prefix.rstrip(" .-"), self.number, tail, max(stem.version, self.version))


def _member_of(name: str, stem: Stem) -> RunMember | None:
    """A name's run membership: the release's own number, read from the text alone, and what surrounds it.

    None when no form fits.
    """

    match = next((found for form in _NUMBER_FORMS if (found := form(stem.text)) is not None), None)
    return None if match is None else _Count.of(match, match.group(1)).member(name, stem)


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
    """The files sharing one prefix, in release-number order."""

    prefix: str
    """The text before the number that every member shares."""
    members: tuple[RunMember, ...]
    superseded: tuple[RunMember, ...]
    """The lower versions a member's higher `vN` replaced: duplicates once the run is placed."""

    @property
    def whole(self) -> tuple[RunMember, ...]:
        """The members plus the lower versions they superseded."""

        return (*self.members, *self.superseded)

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(member.number for member in self.members)

    def consecutive(self, width: int) -> bool:
        """Whether the run is exactly `width` consecutive numbers, wherever it starts."""

        return len(self.members) == width and is_consecutive(self.numbers)


def numbered_runs(members: Iterable[RunMember]) -> list[NumberedRun]:
    """Every numbered run among the members, grouped by prefix.

    Of several members sharing a number the highest `vN` stays and the rest are superseded (equal versions all stay).
    """

    by_prefix: dict[str, dict[int, list[RunMember]]] = {}
    for member in members:
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


# A title names a candidate when they share at least half of their combined words beyond the series title.
_MIN_TITLE_OVERLAP = 0.5
# A title read from a name's tail ends where a quality word begins: one holding a letter and a digit ("1080p",
# "x265"), a source word as the words split it ("WEB-DL" to "web" and "dl"), or an edition word. A bare number
# does not end it ("... Man 1").
_QUALITY_WORDS = frozenset(
    {
        "bluray",
        "blu",
        "ray",
        "bd",
        "bdrip",
        "bdremux",
        "web",
        "webdl",
        "dl",
        "webrip",
        "hdtv",
        "hdrip",
        "dvd",
        "dvdrip",
        "remux",
        "uhd",
        "repack",
        "proper",
    }
)
# The quality words that only follow another ("Blu-Ray", "WEB-DL" split): a word behind a count opens no quality
# tail with one ("Show - 05 Ray of Hope").
_SECOND_HALVES = frozenset({"ray", "dl"})
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


def _ascii(text: str) -> str:
    """The text with its accents folded away and every character outside ASCII dropped."""

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _ascii_digits(text: str) -> str:
    """The text with each decimal digit of any script (a fullwidth one too) as its ASCII digit, and "½" kept."""

    return "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in text)


def _kept(words: Sequence[str]) -> list[bool]:
    """Which words to keep: a count label beside a number says nothing the number does not."""

    return [
        word not in _COUNT_LABELS
        or not any(words[near].isdigit() for near in (index - 1, index + 1) if 0 <= near < len(words))
        for index, word in enumerate(words)
    ]


def folded_words(text: str) -> tuple[str, ...]:
    """The words of a title or name in order: case and accents folded, brackets dropped, seasons counted plainly."""

    words = [_count_word(word) or word for word in _raw_words(_ascii(_BRACKETED.sub(" ", text)))]
    return tuple(compress(words, _kept(words)))


# A name's own season tag: "S1", "Season 1", "1st Season", "First Season".
_SEASON_TAG = re.compile(
    rf"(?<![a-z0-9])(?:s\d{{1,2}}|season[\s._-]*\d{{1,2}}|{_ORDINAL_LEAD}season)(?![a-z0-9])", re.IGNORECASE
)


def whole_words(text: str) -> tuple[str, ...]:
    """A counted name's words as it is read whole against the titles: `folded_words` less its own season tag.

    "Show S1 - 28" reads (show, 28): a name's season tag and count name no episode, "1.28" included.
    """

    return folded_words(_SEASON_TAG.sub(" ", text))


# A bare number closing the text before a name's count, right behind a word: "Show 2 - 05".
_BARE_SEASON = re.compile(r"(?<=[^\W\d_])[\s.]+(\d{1,2})$")
# A season number beside a zero-padded episode number, however the name sets the two apart: "Show - 2 - 05",
# "Show.2.05", "Show-2-05", "Show 2nd - 05", "Show 2 Ep 05". Sonarr reads the padded number as the episode. An
# unpadded pair ("Show - 2 - 1 =") can be an episode title's numbers.
_PADDED_PAIR = re.compile(r"\b(\d{1,2})(?:st|[nr]d|th)?[\s.-]+(?:(?:ep|episode)[\s.-]*)?(0\d{1,3})\b", re.IGNORECASE)


def written_keys(read: NameRead) -> tuple[EpisodeKey, ...]:
    """The seasons and episodes a name writes in plain numbers, outside any key token like "S02E05".

    Each season beside a zero-padded episode, however the name sets the two apart (`_PADDED_PAIR`), and a member's
    bare number right behind a word with its count ("Show 2 - 5"), unless that number is a season tag's. Either can be
    the series title's own number: "Show No. 8 - 05" writes S08E05.
    """

    padded = tuple(EpisodeKey(int(season), int(episode)) for season, episode in _PADDED_PAIR.findall(read.stem.text))
    if read.member is None or (found := _BARE_SEASON.search(_SEASON_TAG.sub(" ", read.member.prefix))) is None:
        return padded
    return (EpisodeKey(int(found.group(1)), read.member.number), *padded)


def _blanked(text: str) -> str:
    """The text with each bracketed group (a group's tag, a CRC) blanked out in place: no part of a title."""

    return _BRACKETED.sub(lambda found: " " * len(found.group()), text)


class _Word(NamedTuple):
    """A word of a title or a name as a title is matched by it: its text, its span, and whether it is matched."""

    text: str
    start: int
    end: int
    kept: bool
    """False for a count label beside a number ("2nd Season"): dropped from the match, part of the span."""


# The connective a title writes as "and" or "&", and a name as either or neither.
_CONNECTIVE = "and"


def _title_words(text: str) -> tuple[_Word, ...]:
    """The words of a title, or of a name's text, as a title is matched: `folded_words` less the "and".

    Digits stay as written: "009" is the title's in "Show 009" and no "9" of a "Show 9"; "Show - 002" opens with no
    "Show 2". A name spells a title's "&" as "and", "&" or nothing ("Salt and Pepper - 01" opens with "Salt &
    Pepper"), so the connective is matched as the "&" is: not at all.
    """

    words = [
        (word if word.isdigit() else _count_word(word) or word, found.start(), found.end())
        for found in _WORD.finditer(text)
        for word in _raw_words(_ascii(found.group()))
        if word != _CONNECTIVE
    ]
    kept = _kept([word for word, _start, _end in words])
    return tuple(_Word(word, start, end, keep) for (word, start, end), keep in zip(words, kept, strict=True))


def _word_set(text: str) -> frozenset[str]:
    """The distinct words of a title or name."""

    return frozenset(folded_words(text))


class TitleGround(NamedTuple):
    """The title words an entry's names are read against."""

    series_words: frozenset[str]
    """The series title's words: an episode title names a file past them."""
    entry_words: frozenset[str]
    """The series and AniList titles' words: an extras word among them is a title word, not an extra."""
    titles: tuple[tuple[str, ...], ...]
    """Each title's words (`_title_words`), with and without its bracketed part, longest first: a name opening
    with one carries the title, and the numbers in it are the title's (`read_names`). Words that only count a
    season ("2nd Season" left of "[Show] 2nd Season") are no title."""
    numbers: frozenset[int]
    """The numbers a title writes in digits ("009", "100", "2011"): a count read off one of them inside the title
    is the title's, not a count. A season a title counts in words ("IV", "2nd") is not among them."""

    @classmethod
    def of(cls, names: EntryNames) -> Self:
        """The series title's words, those plus every AniList title's, each title's words in order, and its digits."""

        every = (names.series, *names.anilist)
        series = _word_set(names.series)
        titles = dict.fromkeys(
            words
            for title in every
            for text in (title, _BRACKETED.sub(" ", title))
            if (words := tuple(word.text for word in _title_words(text) if word.kept))
            and not all(word.isdigit() for word in words)
        )
        return cls(
            series,
            series.union(*(_word_set(title) for title in names.anilist)),
            tuple(sorted(titles, key=len, reverse=True)),
            frozenset(int(word) for title in every for word in _raw_words(_ascii(title)) if word.isdigit()),
        )

    def title_span(self, text: str) -> tuple[int, int] | None:
        """Where the title the text opens with starts and ends, the longest of several; None when it opens with none.

        A title stands whole, or shortened behind one of its numbers ("Big Show 0083" for "Big Show 0083: Last
        Stand", never "Big Show" for it), and never through half of one ("Show 1" of "Show: 1.0 Subtitle" is no
        "Show 1.11"). Its number alone is no title ("12" of "12: Show"). A title shortened in front stands through
        one of its numbers and a word ("Show 0083" for it, never "Last Stand" nor "0083" alone).
        """

        words = _title_words(text)
        kept = [index for index, word in enumerate(words) if word.kept]
        if not kept:
            return None
        first = words[kept[0]].text
        end = 0
        for whole in self.titles:
            for lead in range(len(whole)):
                if whole[lead] != first:
                    continue
                title = whole[lead:]
                matched = next(
                    (
                        count
                        for count, (index, want) in enumerate(zip(kept, title, strict=False))
                        if words[index].text != want
                    ),
                    min(len(kept), len(title)),
                )
                digits = [word.isdigit() for word in title[:matched]]
                mixed = any(digits) and not all(digits)
                if matched < len(title) and (not mixed or (digits[-1] and title[matched].isdigit())):
                    continue
                if lead and not mixed:
                    continue
                # A label hanging off the title's own number ("2nd Season" for "II") is the title's too, unless a
                # number follows it directly ("Season 01"): that number is the label's, and no count.
                last = kept[matched - 1] + 1
                if last < len(words) and self._label_of_title(words, last, text):
                    last += 1
                end = max(end, last)
        return None if end == 0 else (words[kept[0]].start, words[end - 1].end)

    @staticmethod
    def _label_of_title(words: Sequence[_Word], index: int, text: str) -> bool:
        """Whether the dropped label at `index` hangs off the title's number ending right before it."""

        return (
            not words[index].kept
            and words[index - 1].text.isdigit()
            and _LABEL_NUMBER.match(text, words[index].end) is None
        )


def _count_ends(text: str, end: int) -> bool:
    """Whether the count ending at `end` ends there (`_COUNT_END`), past any release tags: no title word follows."""

    while (found := _COUNT_END.match(text, end)) is not None:
        if found["word"] is None:
            return True
        word = found["word"].lower()
        if is_quality_word(word) and word not in _SECOND_HALVES:
            return True
        tag = _RELEASE_TAG.match(text, found.start("word"))
        if tag is None:
            return False
        end = tag.end()
    return False


def _span_member(name: str, stem: Stem, found: re.Match[str]) -> RunMember | None:
    """The count behind the title's span (`_SPAN_COUNT`) as the name's run membership, else None.

    A word behind that count makes it an episode title's number (`_count_ends`), a number standing alone behind it
    with only separators between makes it a range's or a decimal's first number ("Show 01-02", "Show - 21.5"), a
    key behind it says more than it does ("Show.101.S03"), and an OVA tag behind it marks an extra
    (`_EXTRA_BEHIND`): none of them counts.
    """

    text = stem.text
    if (
        not _count_ends(text, found.end())
        or _KEY_TOKEN.search(text, found.end()) is not None
        or _NUMBER_BEHIND.match(text, found.end()) is not None
        or _EXTRA_BEHIND.match(text, found.end()) is not None
    ):
        return None
    return _Count.of(found, _COUNT_LEAD.sub("", text[: found.start("number")])).member(name, stem)


def _tails_a_title(text: str, member: RunMember, end: int) -> bool:
    """Whether the member's number ends the stem behind words past the title ("[0802] Show - Squad 66" read 66)."""

    trailing = _TRAILING_NUMBER.match(text)
    return (
        trailing is not None
        and not text[len(member.prefix) : trailing.start("number")].strip(" ._-")
        and len(member.prefix) > end
        and _LABEL_LAST.search(member.prefix) is None
        and _WORD.search(text, end, len(member.prefix)) is not None
    )


def _titled(name: str, read: NameRead, ground: TitleGround) -> NameRead:
    """The read with the entry's title told apart, when the name opens with one of `TitleGround.titles`.

    The title's numbers leave `bare_numbers` for the member's `title_numbers`. A name counted by none of its numbers,
    or by one of the title's own, counts by the first count behind the title ("Show.009.10" reads 10), unless words
    follow it (then it is the `titled_count`). Without one, a title digit set off by a dash or a label still counts
    ("Show - 2" under "Show 2"). A number trailing words past the title, in a name counted before it, is no count.
    """

    text = read.stem.text
    blanked = _blanked(text)
    span = ground.title_span(blanked)
    if span is None:
        return read
    start, end = span
    counts = _bare_counts(text)
    before = any(count.start < start for count in counts)
    read = read._replace(bare_numbers=tuple(count.number for count in counts if not start <= count.start < end))
    found = _SPAN_COUNT.match(blanked, end)
    counted = None if found is None else _span_member(name, read.stem, found)
    member = read.member
    if member is None:
        member = counted
    elif len(member.prefix) < end and member.number in ground.numbers:
        if counted is not None:
            member = counted
        elif _DASH_BEHIND.match(text, len(member.prefix)) is None and _LABEL_LAST.search(member.prefix) is None:
            member = None
    elif before and _tails_a_title(text, member, end):
        member = None
    if member is None:
        if found is not None and not _count_ends(text, found.end()):
            tail = folded_words(text[found.end() :])
            return read._replace(member=None, title_words=tail, titled_count=int(found["number"]))
        return read._replace(member=None, title_words=folded_words(text))
    numbers = tuple(int(word) for word in _raw_words(_ascii_digits(blanked[start:end])) if word.isdigit())
    return read._replace(member=replace(member, title_numbers=numbers), title_words=member.tail_words)


def read_names(names: Iterable[str], ground: TitleGround) -> dict[str, NameRead]:
    """Read a pack's names: each alone, then against the entry's titles (`_titled`), then its lone-S set.

    Names ending in "S1", "S2", ... behind one prefix whose numbers count up from 1 are a set of specials
    ("Show S1".."Show S3" beside "Show - 01".."Show - 11"), each a run member of the prefix with the "S". One
    "Show S1" alone, or an "S2" repeated over extras, says which season and stays unnumbered.
    """

    reads = {name: _titled(name, read_name(name), ground) for name in names}
    sets: dict[str, dict[int, list[str]]] = {}
    for name, read in reads.items():
        if read.member is None and (found := _LONE_S.match(read.stem.text)) is not None:
            sets.setdefault(found.group(1), {}).setdefault(int(found.group(2)), []).append(name)
    for prefix, by_number in sets.items():
        numbers = sorted(by_number)
        if len(numbers) < 2 or numbers[0] != 1 or not is_consecutive(numbers):
            continue
        for number, members in by_number.items():
            for name in members:
                member = RunMember(name, prefix, number, (), reads[name].stem.version)
                reads[name] = reads[name]._replace(member=member, title_words=())
    return reads


def is_quality_word(word: str) -> bool:
    """Whether the word is quality junk: one of `_QUALITY_WORDS`, or letters and digits mixed ("1080p", "x265")."""

    return word in _QUALITY_WORDS or (any(c.isalpha() for c in word) and any(c.isdigit() for c in word))


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
    """Every candidate scored against the entry's AniList titles, with the series title's words dropped from both.

    Built once over every candidate. The questions take index subsets, and each answer depends only on
    the subset's own scores, so one naming serves every tie-break and refusal over the same candidates.
    """

    leftovers: tuple[_Leftover, ...]
    scored: tuple[_Scored, ...]

    @classmethod
    def of(cls, candidates: Sequence[str], names: EntryNames) -> Self | None:
        """Score the candidates, or None when nothing names them: no ground, no titles, or the series as a title."""

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

    The unique best sharing at least `_MIN_TITLE_OVERLAP` of the combined words beyond the series title wins,
    unless it shares only the opening words of every title it scores best against. A refused winner never
    lets the runner-up win.
    """

    naming = Naming.of(candidates, names)
    return None if naming is None else naming.sole_among(range(len(candidates)))
