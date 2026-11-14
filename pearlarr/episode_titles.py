"""The series' episode titles as words, which episode a name's words are titled as, and the placer's queries on them.

Reading a torrent's names against them belongs in `placer._names`, and reading a name's numbers in
`file_names`. No I/O.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from .arr_models import SonarrEpisode
from .file_names import NameParts, folded_words, is_quality_word, written_keys
from .releases import EpisodeKey

# An episode title names a file from this many words past the series' name and the labels below.
_MIN_TITLE_WORDS = 2
# The words that label an episode title without being one: "Part 2" and "OVA 1" name nothing.
_TITLE_LABELS = frozenset({"part", "episode", "ep", "special", "sp", "ova", "oad", "movie", "recap", "vol", "chapter"})


def is_distinctive_title(words: Iterable[str]) -> bool:
    """Whether the words say enough to name an episode: `_MIN_TITLE_WORDS` that are neither a number nor a label."""

    return sum(not word.isdigit() and word not in _TITLE_LABELS for word in words) >= _MIN_TITLE_WORDS


def spells_title(words: Iterable[str]) -> bool:
    """Whether a name's words spell an episode title: `is_distinctive_title` with the quality junk left out."""

    return is_distinctive_title(word for word in words if not is_quality_word(word))


def names_alone(words: Iterable[str]) -> bool:
    """Whether the words name an episode when a name carries nothing else ("Room 12", "12 x 12").

    `_MIN_TITLE_WORDS` that are no label, numbers among them: "Part 2" names nothing.
    """

    return sum(word not in _TITLE_LABELS for word in words) >= _MIN_TITLE_WORDS


def opens_title(words: tuple[str, ...], title: tuple[str, ...]) -> bool:
    """Whether the episode title's words open the words, and the words end there or go on with quality junk."""

    return words[: len(title)] == title and _ends_title(words, len(title))


def _ends_title(words: tuple[str, ...], count: int) -> bool:
    """Whether the words end after the first `count`, or go on with quality junk."""

    return len(words) == count or (len(words) > count and is_quality_word(words[count]))


@dataclass(frozen=True, slots=True, kw_only=True)
class TitledEpisode:
    """An episode title of the series, in one of its forms."""

    ep_id: int
    words: tuple[str, ...]
    """The episode title's folded words."""
    beyond: tuple[str, ...]
    """The episode title's words that are not words of the series' name."""
    distinct: bool
    """Whether `beyond` is specific enough to place a file by."""
    alone: bool
    """Whether `beyond` names the episode when a name carries nothing else ("Room 12"): a number counts then."""
    part: bool
    """Whether this form is only a part a colon or a dash sets off, not the whole episode title."""
    spelled_id: int | None
    """The episode whose season and number `beyond` is, when it is those two numbers alone ("1.2" is S01E02)."""


@dataclass(frozen=True, slots=True)
class _TitleForm:
    """One form of an episode title: the whole episode title, or a part a colon or a dash sets off."""

    text: str
    part: bool


# A hyphen or dash between spaces: "Part One - The Return".
_SPACED_DASH = re.compile(r"\s[-\u2012-\u2015]\s")
# A segment wrapped in dashes: "The Long Road -Part Zero-".
_WRAPPED = re.compile(r"(?:^|(?<=\s))-([^\s-](?:[^-]*[^\s-])?)-(?=\s|$)")


def _parts(*texts: str) -> tuple[_TitleForm, ...]:
    """The texts with words, as episode title parts. None when fewer than two have words."""

    parts = tuple(_TitleForm(text, part=True) for text in texts if text.strip())
    return parts if len(parts) > 1 else ()


def _title_forms(title: str) -> tuple[_TitleForm, ...]:
    """The whole episode title and its parts: around its last colon, its last spaced dash, and a dash-wrapped segment.

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
    """The one episode a whole episode title matches, else the one an episode title part matches, else None.

    Two matches count as none: a recap repeating an episode title, a two-part episode title, or an episode title's
    text before the colon that is another episode's subtitle.
    """

    whole: dict[int, TitledEpisode] = {}
    parts: dict[int, TitledEpisode] = {}
    for hit in hits:
        (parts if hit.part else whole).setdefault(hit.ep_id, hit)
    named = whole or parts
    return next(iter(named.values())) if len(named) == 1 else None


def _grouped[K](pairs: Iterable[tuple[K, TitledEpisode]]) -> Mapping[K, tuple[TitledEpisode, ...]]:
    """The episode title forms grouped by key, each group in the order given, read-only."""

    groups: dict[K, list[TitledEpisode]] = {}
    for key, ep in pairs:
        groups.setdefault(key, []).append(ep)
    return MappingProxyType({key: tuple(eps) for key, eps in groups.items()})


@dataclass(frozen=True, slots=True)
class TitleLead:
    """What the episode titles say about a name's words: the words after its name number, else the whole name."""

    title: TitledEpisode | None
    """The one episode the words are titled as (`EpisodeTitles.leading_title` or `exact_match`), whether or not other
    files carry it."""
    leading_title_ids: frozenset[int]
    """Every episode whose episode title, specific or not, starts the words: what a numbered sequence is judged by."""
    words: tuple[str, ...]
    """The words read. Two files carrying the same words are titled as nothing."""


@dataclass(frozen=True, slots=True)
class EpisodeTitles:
    """The series' episode titles as words: specific ones can place a file, and a numbered sequence is judged by all."""

    series_words: frozenset[str]
    """The words of the series' name."""
    by_first: Mapping[tuple[str, ...], tuple[TitledEpisode, ...]]
    """Every episode title form by its first word (a one-word tuple, so an empty name finds none), in series order."""
    alone_by_beyond: Mapping[tuple[str, ...], tuple[TitledEpisode, ...]]
    """The forms that name their episode alone, by their words beyond the series' name, in series order."""
    forms: tuple[TitledEpisode, ...]
    """Every episode title form, in series order."""
    id_by_key: Mapping[EpisodeKey, int]
    """Each series episode by its season and number."""

    # EpisodeIndex.titles keeps one view per index and word set, built from these three arguments and nothing else.
    @classmethod
    def of(
        cls, by_id: Mapping[int, SonarrEpisode], id_by_key: Mapping[EpisodeKey, int], series_words: frozenset[str]
    ) -> Self:
        """Convert every form of every episode title to words, noting the words beyond the series' name."""

        titled: list[TitledEpisode] = []
        for ep_id, ep in by_id.items():
            for form in _title_forms(ep.title):
                words = folded_words(form.text)
                beyond = tuple(word for word in words if word not in series_words)
                distinct = is_distinctive_title(beyond)
                # An episode title part too short to place a file ("Chapter 4: Home") does not judge a numbered
                # sequence either.
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

    def leading_title(self, words: tuple[str, ...]) -> TitleLead:
        """Match the words after a name number: the one specific episode title starting them, and every one that does.

        Quality tags may follow the specific episode title.
        """

        led = [ep for ep in self.by_first.get(words[:1], ()) if words[: len(ep.words)] == ep.words]
        titled = _sole(ep for ep in led if ep.distinct and opens_title(words, ep.words))
        return TitleLead(titled, frozenset(ep.ep_id for ep in led), words)

    def exact_match(self, words: tuple[str, ...], numbered: Collection[int]) -> TitledEpisode | None:
        """The one episode whose episode title names it alone and equals a name's words, series words dropped from both.

        `keyed` are the episodes the name's numbers point at (`episodes_numbered_by`). An episode title that is just the
        season and number of another of them ("1.2" against S01E02) is that season and episode over again, not an
        episode title: it names nothing.
        """

        beyond = tuple(word for word in words if word not in self.series_words)
        equal = _sole(self.alone_by_beyond.get(beyond, ()))
        if equal is not None and equal.spelled_id != equal.ep_id and equal.spelled_id in numbered:
            return None
        return equal

    def episodes_numbered_by(self, parts: NameParts, resolved: Collection[int]) -> frozenset[int]:
        """The episodes a name's numbers point at: Sonarr's parse of it (`resolved`), and the ones it writes.

        "Show 2 - 05" and "Show.2.05" point at S02E05 (`written_keys`). "Show No. 8 - 05" points at S08E05 too, which
        is harmless: no episode title can spell it, since an episode title's series words are dropped before it spells
        a season and episode.
        """

        written = (self.id_by_key.get(key) for key in written_keys(parts))
        return frozenset((*resolved, *(ep_id for ep_id in written if ep_id is not None)))

    def near_matches(self, ep_id: int, words: tuple[str, ...]) -> bool:
        """Whether the words open with the episode title of `ep_id` a word off at most: one swapped, added or dropped.

        The words the two share must still spell an episode title (`spells_title`): a short episode title confirms no
        near miss. Quality junk may follow the episode title, as under `leading_title`.
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
    """The words the two share, in order, when the words are the episode title a word off at most, else None.

    A word off: one swapped for another, one added, or one dropped.
    """

    if len(words) == len(title):
        apart = [i for i, (word, titled) in enumerate(zip(words, title, strict=True)) if word != titled]
        return tuple(word for i, word in enumerate(title) if i not in apart) if len(apart) <= 1 else None
    longer, shorter = (words, title) if len(words) > len(title) else (title, words)
    if len(longer) != len(shorter) + 1:
        return None
    return shorter if any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer))) else None
