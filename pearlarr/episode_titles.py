"""Read a torrent's file names against the series' episode titles: which episode each name is titled as.

No I/O. Reading a name's own numbers belongs in `release_names`, and placing files in `placer`.
"""

import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import NamedTuple, Self

from .parse_reading import Reading
from .placement_types import EpisodeIndex, PlacementBatch, TargetScope
from .release_names import (
    NameRead,
    NumberedRun,
    TitleGround,
    folded_words,
    is_extras_name,
    is_quality_word,
    read_names,
    whole_words,
    written_keys,
)
from .seadex_types import EpisodeKey

# Episode titles pick a run, or refuse one, only when at least this many members carry them.
MIN_TITLE_HITS = 2
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


class _TitledEpisode(NamedTuple):
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


class _TitleForm(NamedTuple):
    """One form of an episode title: the whole title, or a part a colon or a dash sets off."""

    text: str
    part: bool


# A hyphen or dash between spaces: "Part One - The Return".
_SPACED_DASH = re.compile(r"\s[-\u2012-\u2015]\s")
# A segment wrapped in dashes: "The Long Road -Part Zero-".
_WRAPPED = re.compile(r"(?:^|(?<=\s))-([^\s-](?:[^-]*[^\s-])?)-(?=\s|$)")


def _parts(*texts: str) -> tuple[_TitleForm, ...]:
    """The texts with words, as title parts; none when fewer than two have words."""

    parts = tuple(_TitleForm(text, True) for text in texts if text.strip())
    return parts if len(parts) > 1 else ()


def _title_forms(title: str) -> tuple[_TitleForm, ...]:
    """The whole title and its parts: around its last colon, its last spaced dash, and a dash-wrapped segment.

    A name often carries only one part. A form with "&" comes again with it spelled "and".
    """

    head, _colon, subtitle = title.rpartition(":")
    forms = [_TitleForm(title, False), *_parts(head, subtitle)]
    if dashes := list(_SPACED_DASH.finditer(title)):
        forms += _parts(title[: dashes[-1].start()], title[dashes[-1].end() :])
    if wrapped := _WRAPPED.search(title):
        forms += _parts(title[: wrapped.start()], wrapped[1], title[wrapped.end() :])
    forms += [_TitleForm(form.text.replace("&", " and "), form.part) for form in forms if "&" in form.text]
    return tuple(forms)


def _sole(hits: Iterable[_TitledEpisode]) -> _TitledEpisode | None:
    """The one episode a whole title matches. When no whole title matches, the one a title part matches. Else None.

    Two matches count as none: a recap repeating a title, a two-part title, or a title's text before the colon
    that is another episode's subtitle.
    """

    whole: dict[int, _TitledEpisode] = {}
    parts: dict[int, _TitledEpisode] = {}
    for hit in hits:
        (parts if hit.part else whole).setdefault(hit.ep_id, hit)
    named = whole or parts
    return next(iter(named.values())) if len(named) == 1 else None


def _grouped[K](pairs: Iterable[tuple[K, _TitledEpisode]]) -> dict[K, tuple[_TitledEpisode, ...]]:
    """The title forms grouped by key, each group in the order given."""

    groups: dict[K, list[_TitledEpisode]] = {}
    for key, ep in pairs:
        groups.setdefault(key, []).append(ep)
    return {key: tuple(eps) for key, eps in groups.items()}


class _TitleLead(NamedTuple):
    """What the episode titles say about a name's words: a run member's tail, else the whole name."""

    titled: _TitledEpisode | None
    """The one episode the words are titled as (`EpisodeTitles.leading` or `exactly`), whether or not other files
    carry it."""
    leaders: frozenset[int]
    """Every episode whose title, specific or not, starts the words: the evidence a run is judged by."""
    words: tuple[str, ...]
    """The words read. Two files carrying the same words are titled as nothing."""


class EpisodeTitles(NamedTuple):
    """The series' episode titles as words: specific titles can place a file, and every title is run evidence."""

    series_words: frozenset[str]
    """The series title's words."""
    by_first: Mapping[tuple[str, ...], tuple[_TitledEpisode, ...]]
    """Every title form by its first word (a one-word tuple, so an empty name finds none), in series order."""
    alone_by_beyond: Mapping[tuple[str, ...], tuple[_TitledEpisode, ...]]
    """The forms that name their episode alone, by their words beyond the series title, in series order."""
    forms: tuple[_TitledEpisode, ...]
    """Every title form, in series order."""
    id_by_key: Mapping[EpisodeKey, int]
    """Each series episode by its season and number."""

    @classmethod
    def of(cls, series: EpisodeIndex, ground: TitleGround) -> Self:
        """Fold every form of every episode title into words, noting the words beyond the series title."""

        titled: list[_TitledEpisode] = []
        for ep_id, ep in series.by_id.items():
            for form in _title_forms(ep.title):
                words = folded_words(form.text)
                beyond = tuple(word for word in words if word not in ground.series_words)
                distinct = is_distinct_title(beyond)
                # A title part too short to place a file ("Chapter 4: Home") is not run evidence either.
                if words and (distinct or not form.part):
                    alone = names_alone(beyond)
                    titled.append(
                        _TitledEpisode(ep_id, words, beyond, distinct, alone, form.part, _spelled(series, beyond))
                    )
        return cls(
            ground.series_words,
            _grouped((ep.words[:1], ep) for ep in titled),
            _grouped((ep.beyond, ep) for ep in titled if ep.alone),
            tuple(titled),
            series.id_by_key,
        )

    def leading(self, words: tuple[str, ...]) -> _TitleLead:
        """Match a run member's tail: the one specific title that starts it, and every title that starts it.

        Quality tags may follow the specific title.
        """

        led = [ep for ep in self.by_first.get(words[:1], ()) if words[: len(ep.words)] == ep.words]
        titled = _sole(ep for ep in led if ep.distinct and opens_title(words, ep.words))
        return _TitleLead(titled, frozenset(ep.ep_id for ep in led), words)

    def exactly(self, words: tuple[str, ...], keyed: Collection[int]) -> _TitledEpisode | None:
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


def _spelled(series: EpisodeIndex, beyond: tuple[str, ...]) -> int | None:
    """The episode whose season and number the words are, when they are two numbers alone, else None."""

    if len(beyond) != 2 or not all(word.isdigit() for word in beyond):
        return None
    return series.id_by_key.get(EpisodeKey(int(beyond[0]), int(beyond[1])))


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


class TitleEvidence(NamedTuple):
    """How many run members' tails start with only window titles, and how many with only titles outside it."""

    inside: int
    outside: int

    @property
    def selects(self) -> bool:
        """Whether at least `MIN_TITLE_HITS` members carry window titles, and more than carry outside titles."""

        return self.inside >= MIN_TITLE_HITS and self.inside > self.outside

    @property
    def vetoes(self) -> bool:
        """Whether at least `MIN_TITLE_HITS` members carry outside titles and none carries a window title."""

        return self.outside >= MIN_TITLE_HITS and not self.inside


class _NameTitle(NamedTuple):
    """One name read against the episode titles."""

    episode: int | None
    """The episode the name is titled as: its specific title, carried by no other file of the torrent."""
    leaders: frozenset[int]
    """Every episode whose title starts the name's tail: run evidence, read for run members only."""


def _title_lead(episodes: EpisodeTitles, read: NameRead, resolved: Collection[int]) -> _TitleLead:
    """Read one name against the episode titles: the tail behind its count, else the whole name.

    A counted name is read whole too, its season tag left out: "Show - 17 Steps 2" is titled "17 Steps 2". A name
    that each reading titles as a different episode is titled as neither. `resolved` are the episodes Sonarr read
    for the file (`EpisodeTitles.keyed_by`).
    """

    words = read.title_words
    keyed = episodes.keyed_by(read, resolved)
    if read.member is None and read.titled_count is None:
        return _TitleLead(episodes.exactly(words, keyed), frozenset(), words)
    lead = episodes.leading(words)
    if lead.titled is None or read.titled_count is not None:
        whole = whole_words(read.stem.text)
        if (equal := episodes.exactly(whole, keyed)) is not None:
            if lead.titled is None:
                return _TitleLead(equal, lead.leaders, whole)
            if lead.titled.ep_id != equal.ep_id:
                return lead._replace(titled=None)
    return lead


class TorrentNames(NamedTuple):
    """Every file name of the torrent, read once: stem, run membership, episode title, and whether it is an extra."""

    reads: Mapping[str, NameRead]
    """Each name's read."""
    titles: Mapping[str, _NameTitle]
    """Each name read against the episode titles."""
    extras: frozenset[str]
    """The extras: set aside, and left out of every count."""
    episodes: EpisodeTitles
    """The episode titles the names were read against."""

    @classmethod
    def of(cls, batch: PlacementBatch, scope: TargetScope, readings: Mapping[str, Reading]) -> Self:
        """Read every name, seeded and gone ones included.

        Two versions of one file carrying the same title still count as titled as it. A file whose own key Sonarr
        matched (`Reading.keyed`) is an episode, whatever extras word its name carries.
        """

        ground = TitleGround.of(scope.names)
        episodes = EpisodeTitles.of(scope.series, ground)
        reads = read_names(batch.torrent_names, ground)
        leads = {name: _title_lead(episodes, read, readings[name].resolved) for name, read in reads.items()}
        carriers = Counter(words for _file, words in {(reads[name].file, lead.words) for name, lead in leads.items()})
        titles: dict[str, _NameTitle] = {}
        extras: set[str] = set()
        for name, lead in leads.items():
            titled = lead.titled
            episode = titled.ep_id if titled is not None and carriers[lead.words] == 1 else None
            titles[name] = _NameTitle(episode, lead.leaders)
            # An extras word that belongs to the entry's titles or the name's episode title marks no extra.
            shield = ground.entry_words | frozenset(titled.words if titled is not None else ())
            if not readings[name].keyed and is_extras_name(name, shield):
                extras.add(name)
        return cls(reads, titles, frozenset(extras), episodes)

    @property
    def counted(self) -> tuple[str, ...]:
        """Every name except the extras: the names the evidence checks read."""

        return tuple(name for name in self.reads if name not in self.extras)

    def episode(self, name: str) -> int | None:
        """The episode the name is titled as, else None."""

        return self.titles[name].episode

    def contradicts(self, name: str, ids: Sequence[int]) -> bool:
        """Whether the name is titled as an episode that is not among `ids`."""

        return (ep_id := self.episode(name)) is not None and ep_id not in ids

    def confirms(self, name: str, ids: Sequence[int]) -> bool:
        """Whether the episode the name is titled as is among `ids`."""

        return (ep_id := self.episode(name)) is not None and ep_id in ids

    def evidence(self, run: NumberedRun, window_ids: frozenset[int]) -> TitleEvidence:
        """Count the members whose tail starts with only window titles, and those with only outside titles.

        A tail matching titles on both sides counts for neither.
        """

        inside = outside = 0
        for member in run.members:
            sides = {ep_id in window_ids for ep_id in self.titles[member.name].leaders}
            if sides == {True}:
                inside += 1
            elif sides == {False}:
                outside += 1
        return TitleEvidence(inside, outside)
