"""The series' episode titles as words, which episode a name's words are titled as, and the placer's queries on them.

Reading a torrent's names against them belongs in `placer._names`, and reading a name's own numbers in
`release_names`. No I/O.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from .release_names import NameRead, folded_words, is_quality_word, written_keys
from .seadex_types import EpisodeKey, SonarrEpisode

# An episode title names a file from this many words past the series title and the labels below.
_MIN_TITLE_WORDS = 2
# The words that label a title without being one: "Part 2" and "OVA 1" name nothing.
_TITLE_LABELS = frozenset({"part", "episode", "ep", "special", "sp", "ova", "oad", "movie", "recap", "vol", "chapter"})


def is_distinct_title(words: Iterable[str]) -> bool:
    """Whether the words say enough to name an episode: `_MIN_TITLE_WORDS` that are neither a number nor a label."""

    return sum(not word.isdigit() and word not in _TITLE_LABELS for word in words) >= _MIN_TITLE_WORDS


def spells_title(words: Iterable[str]) -> bool:
    """Whether a name's words spell an episode title: `is_distinct_title` with the quality junk left out."""

    return is_distinct_title(word for word in words if not is_quality_word(word))


def names_alone(words: Iterable[str]) -> bool:
    """Whether the words name an episode when a name carries nothing else ("Room 12", "12 x 12").

    `_MIN_TITLE_WORDS` that are no label, numbers among them: "Part 2" names nothing.
    """

    return sum(word not in _TITLE_LABELS for word in words) >= _MIN_TITLE_WORDS


def opens_title(words: tuple[str, ...], title: tuple[str, ...]) -> bool:
    """Whether the title's words open the words, and the words end there or go on with quality junk."""

    return words[: len(title)] == title and _ends_title(words, len(title))


def _ends_title(words: tuple[str, ...], count: int) -> bool:
    """Whether the words end after the first `count`, or go on with quality junk."""

    return len(words) == count or (len(words) > count and is_quality_word(words[count]))


@dataclass(frozen=True, slots=True, kw_only=True)
class TitledEpisode:
    """One series episode's title, in one of its forms."""

    ep_id: int
    words: tuple[str, ...]
    """The title's folded words."""
    beyond: tuple[str, ...]
    """The title's words that are not words of the series title."""
    distinct: bool
    """Whether `beyond` is specific enough to place a file by."""
    alone: bool
    """Whether `beyond` names the episode when a name carries nothing else ("Room 12"): a number counts then."""
    part: bool
    """Whether this form is only a part a colon or a dash sets off, not the whole title."""
    spelled_id: int | None
    """The episode whose season and number `beyond` is, when it is those two numbers alone ("1.2" is S01E02)."""


@dataclass(frozen=True, slots=True)
class _TitleForm:
    """One form of an episode title: the whole title, or a part a colon or a dash sets off."""

    text: str
    part: bool


# A hyphen or dash between spaces: "Part One - The Return".
_SPACED_DASH = re.compile(r"\s[-\u2012-\u2015]\s")
# A segment wrapped in dashes: "The Long Road -Part Zero-".
_WRAPPED = re.compile(r"(?:^|(?<=\s))-([^\s-](?:[^-]*[^\s-])?)-(?=\s|$)")


def _parts(*texts: str) -> tuple[_TitleForm, ...]:
    """The texts with words, as title parts; none when fewer than two have words."""

    parts = tuple(_TitleForm(text, part=True) for text in texts if text.strip())
    return parts if len(parts) > 1 else ()


def _title_forms(title: str) -> tuple[_TitleForm, ...]:
    """The whole title and its parts: around its last colon, its last spaced dash, and a dash-wrapped segment.

    A name often carries only one part. A form with "&" comes again with it spelled "and".
    """

    head, _colon, subtitle = title.rpartition(":")
    forms = [_TitleForm(title, part=False), *_parts(head, subtitle)]
    if dashes := list(_SPACED_DASH.finditer(title)):
        forms += _parts(title[: dashes[-1].start()], title[dashes[-1].end() :])
    if wrapped := _WRAPPED.search(title):
        forms += _parts(title[: wrapped.start()], wrapped[1], title[wrapped.end() :])
    forms += [_TitleForm(form.text.replace("&", " and "), form.part) for form in forms if "&" in form.text]
    return tuple(forms)


def _sole(hits: Iterable[TitledEpisode]) -> TitledEpisode | None:
    """The one episode a whole title matches. When no whole title matches, the one a title part matches. Else None.

    Two matches count as none: a recap repeating a title, a two-part title, or a title's text before the colon
    that is another episode's subtitle.
    """

    whole: dict[int, TitledEpisode] = {}
    parts: dict[int, TitledEpisode] = {}
    for hit in hits:
        (parts if hit.part else whole).setdefault(hit.ep_id, hit)
    named = whole or parts
    return next(iter(named.values())) if len(named) == 1 else None


def _grouped[K](pairs: Iterable[tuple[K, TitledEpisode]]) -> Mapping[K, tuple[TitledEpisode, ...]]:
    """The title forms grouped by key, each group in the order given, read-only."""

    groups: dict[K, list[TitledEpisode]] = {}
    for key, ep in pairs:
        groups.setdefault(key, []).append(ep)
    return MappingProxyType({key: tuple(eps) for key, eps in groups.items()})


@dataclass(frozen=True, slots=True)
class TitleLead:
    """What the episode titles say about a name's words: a run member's tail, else the whole name."""

    titled: TitledEpisode | None
    """The one episode the words are titled as (`EpisodeTitles.leading` or `exactly`), whether or not other files
    carry it."""
    leaders: frozenset[int]
    """Every episode whose title, specific or not, starts the words: the evidence a run is judged by."""
    words: tuple[str, ...]
    """The words read. Two files carrying the same words are titled as nothing."""


@dataclass(frozen=True, slots=True)
class EpisodeTitles:
    """The series' episode titles as words: specific titles can place a file, and every title is run evidence."""

    series_words: frozenset[str]
    """The series title's words."""
    by_first: Mapping[tuple[str, ...], tuple[TitledEpisode, ...]]
    """Every title form by its first word (a one-word tuple, so an empty name finds none), in series order."""
    alone_by_beyond: Mapping[tuple[str, ...], tuple[TitledEpisode, ...]]
    """The forms that name their episode alone, by their words beyond the series title, in series order."""
    forms: tuple[TitledEpisode, ...]
    """Every title form, in series order."""
    id_by_key: Mapping[EpisodeKey, int]
    """Each series episode by its season and number."""

    # EpisodeIndex.titles keeps one view per index and word set, built from these three arguments and nothing else.
    @classmethod
    def of(
        cls, by_id: Mapping[int, SonarrEpisode], id_by_key: Mapping[EpisodeKey, int], series_words: frozenset[str]
    ) -> Self:
        """Fold every form of every episode title into words, noting the words beyond the series title."""

        titled: list[TitledEpisode] = []
        for ep_id, ep in by_id.items():
            for form in _title_forms(ep.title):
                words = folded_words(form.text)
                beyond = tuple(word for word in words if word not in series_words)
                distinct = is_distinct_title(beyond)
                # A title part too short to place a file ("Chapter 4: Home") is not run evidence either.
                if words and (distinct or not form.part):
                    titled.append(
                        TitledEpisode(
                            ep_id=ep_id,
                            words=words,
                            beyond=beyond,
                            distinct=distinct,
                            alone=names_alone(beyond),
                            part=form.part,
                            spelled_id=_spelled(id_by_key, beyond),
                        )
                    )
        return cls(
            series_words=series_words,
            by_first=_grouped((ep.words[:1], ep) for ep in titled),
            alone_by_beyond=_grouped((ep.beyond, ep) for ep in titled if ep.alone),
            forms=tuple(titled),
            id_by_key=id_by_key,
        )

    def leading(self, words: tuple[str, ...]) -> TitleLead:
        """Match a run member's tail: the one specific title that starts it, and every title that starts it.

        Quality tags may follow the specific title.
        """

        led = [ep for ep in self.by_first.get(words[:1], ()) if words[: len(ep.words)] == ep.words]
        titled = _sole(ep for ep in led if ep.distinct and opens_title(words, ep.words))
        return TitleLead(titled, frozenset(ep.ep_id for ep in led), words)

    def exactly(self, words: tuple[str, ...], keyed: Collection[int]) -> TitledEpisode | None:
        """The one episode whose title names it alone and equals a name's words, the series words dropped from both.

        `keyed` are the episodes the name's numbers are a key for (`keyed_by`). A title that is just the season and
        number of another of them ("1.2" against S01E02) is that key over again, not a title: it names nothing.
        """

        beyond = tuple(word for word in words if word not in self.series_words)
        equal = _sole(self.alone_by_beyond.get(beyond, ()))
        if equal is not None and equal.spelled_id != equal.ep_id and equal.spelled_id in keyed:
            return None
        return equal

    def keyed_by(self, read: NameRead, resolved: Collection[int]) -> frozenset[int]:
        """The episodes a name's numbers are a key for: Sonarr's read of it (`resolved`), and the keys it writes.

        "Show 2 - 05" and "Show.2.05" are keys for S02E05 (`written_keys`). "Show No. 8 - 05" counts as a key for
        S08E05 too, which is harmless: no title can spell it, since a title's series words are dropped before it
        spells a key.
        """

        written = (self.id_by_key.get(key) for key in written_keys(read))
        return frozenset((*resolved, *(ep_id for ep_id in written if ep_id is not None)))

    def nearly(self, ep_id: int, words: tuple[str, ...]) -> bool:
        """Whether the words open with the episode's title a word off at most: one swapped, added or dropped.

        The words the two share must still spell a title (`spells_title`): a short title confirms no near miss.
        Quality junk may follow the title, as under `leading`.
        """

        for ep in self.forms:
            if ep.ep_id != ep_id or not ep.distinct:
                continue
            for count in (len(ep.words) - 1, len(ep.words), len(ep.words) + 1):
                if _ends_title(words, count):
                    shared = _one_word_apart(words[:count], ep.words)
                    if shared is not None and spells_title(shared):
                        return True
        return False


def _spelled(id_by_key: Mapping[EpisodeKey, int], beyond: tuple[str, ...]) -> int | None:
    """The episode whose season and number the words are, when they are two numbers alone, else None."""

    if len(beyond) != 2 or not all(word.isdigit() for word in beyond):
        return None
    return id_by_key.get(EpisodeKey(int(beyond[0]), int(beyond[1])))


def _one_word_apart(words: tuple[str, ...], title: tuple[str, ...]) -> tuple[str, ...] | None:
    """The words the two share, in order, when the words are the title a word off at most, else None.

    A word off: one swapped for another, one added, or one dropped.
    """

    if len(words) == len(title):
        apart = [i for i, (word, titled) in enumerate(zip(words, title, strict=True)) if word != titled]
        return tuple(word for i, word in enumerate(title) if i not in apart) if len(apart) <= 1 else None
    longer, shorter = (words, title) if len(words) > len(title) else (title, words)
    if len(longer) != len(shorter) + 1:
        return None
    return shorter if any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer))) else None
