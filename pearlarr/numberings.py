"""The numbering step: where an absolute-only file falls, once every way of counting agrees.

A release group that names a file "Show - 07" counted the series' episodes some way: from the first episode of the
whole series (`Numbering.FROM_SERIES_START`), from the first episode of the title's season (`SEASON_NUMBERS`), from the
first episode the title covers (`FROM_WINDOW_START`), straight through every title the torrent is listed under
(`ACROSS_WINDOWS`), or by TVDB's absolute numbers (`TVDB_ABSOLUTE`). The passes place such a file when Sonarr's parse
of its number lands inside the window or a numbered sequence carries it. For an absolute-only file they leave, the
step reads the number under every numbering that fits the torrent's windows and proposes its row only when the
numberings that fit all name the same row. A torrent whose windows don't match AniList's episode counts, a number
repeated under one parsed series name, a series row that doesn't carry its number, two numberings that disagree, or
a parsed series name the run doesn't know each leave the file with its verdict. Nothing the step proposes outranks a
pass: `WindowedPlacements.with_proposals` places a proposal only into a gap the windows left.

The step runs once per torrent, after the windows merge, on the grabbing title's series only, and only while
Sonarr numbers that series as aired, where the series' regular rows are the aired rows the groups count.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from types import MappingProxyType
from typing import Self

from .arr_models import ParsedFileInfo, SonarrEpisode
from .pending_import import TitleWindow
from .placement_types import EpisodeIndex, GrabbingTitle, PlacementBatch
from .placer import NameFacts
from .releases import season_episode_key


class Numbering(Enum):
    """One way a release group could have counted a series' episodes: what the number k in a file name means."""

    FROM_SERIES_START = "series row"
    """The series' k-th regular row, in season and episode order: the group counted every regular episode from the
    first season on. Fits a torrent only together with `TVDB_ABSOLUTE`."""
    TVDB_ABSOLUTE = "absolute"
    """The row TVDB numbers k. It only blocks: it never places alone, since a group counting its own way may land
    where TVDB's number happens to agree."""
    FROM_WINDOW_START = "window row"
    """The window's k-th row: the group counted the title's episodes from one. A window of nothing but specials
    counts its specials."""
    SEASON_NUMBERS = "window season"
    """Episode k of the window's first regular season: the group used Sonarr's own episode numbers."""
    ACROSS_WINDOWS = "union row"
    """The k-th regular row across every window of the torrent, two or more windows only: the group counted a torrent
    listed under several titles straight through."""


_PLACING = frozenset(
    {Numbering.FROM_SERIES_START, Numbering.FROM_WINDOW_START, Numbering.SEASON_NUMBERS, Numbering.ACROSS_WINDOWS}
)
"""The numberings whose agreement places a file. `TVDB_ABSOLUTE` only blocks."""
_SERIES_WIDE = frozenset({Numbering.FROM_SERIES_START, Numbering.TVDB_ABSOLUTE})
"""The numberings that count TVDB's whole series: its regular rows from the first season on, and its absolutes."""
_PER_WINDOW = (Numbering.FROM_WINDOW_START, Numbering.SEASON_NUMBERS)
"""The numberings read through one window of the torrent."""


@dataclass(frozen=True, slots=True)
class NumberingProposals:
    """What the step proposes for a torrent: each absolute-only file's ids where every fitting numbering agrees.

    A file not named keeps its verdict.
    """

    ids: Mapping[str, tuple[int, ...]]
    """Normalized name -> the proposed episode ids, one per number of the name, in the name's order."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "ids", MappingProxyType(dict(self.ids)))


NO_PROPOSALS = NumberingProposals({})
"""The step proposing nothing."""


def _name_key(title: str) -> str:
    """The name lowercased, `&` read as "and", every other run of non-alphanumerics one space.

    `file_names._raw_words` splits a name into words the same way, casefolded and without reading `&` as "and". This
    key stays its own so the name keys below never drift with `file_names`'.
    """

    folded = re.sub(r"[^a-z0-9]+", " ", title.lower().replace("&", " and "))
    return " ".join(folded.split())


def _raw_name_key(title: str) -> str:
    """The name lowercased with the " - " and ": " separators converted to a space, punctuation kept.

    Two parsed series names that differ only in how Sonarr joined their parts ("Show - Arc", "Show: Arc") are one
    name.
    """

    return " ".join(title.lower().replace(" - ", " ").replace(": ", " ").split())


_SEP_TAIL = re.compile(r"[\s\-:~\)\]\}]+$")
"""The separators a name can end in once its number is cut off."""
_MARKS = re.compile(r"[!+'’.?]+$")
"""The run of marks a name ends in, the only punctuation `_marked_name_key` keeps."""


def _marked_name_key(title: str) -> str:
    """`_name_key` plus the name's trailing marks, after a bar.

    The marks tell K-ON! from K-ON!!, New Game! from New Game!!, Needless from Needless+ and Dog Days from Dog
    Days', which the normalization merges. A typographic apostrophe counts as the plain one.
    """

    core = _SEP_TAIL.sub("", title.strip())
    found = _MARKS.search(core)
    marks = found.group(0).replace("’", "'") if found else ""
    return f"{_name_key(core)}|{marks}"


# `file_names.folded_words` isn't reused for these keys: it converts roman numerals, ordinals and accents and drops
# bracketed text, which would merge names the partition and the `known_titles` gate keep apart.


def _increasing(numbers: Sequence[int | None]) -> bool:
    """Whether every number is known and they strictly increase. An empty run does."""

    known = [n for n in numbers if n is not None]
    return len(known) == len(numbers) and all(later > earlier for earlier, later in pairwise(known))


def _ordered(episodes: Iterable[SonarrEpisode]) -> list[SonarrEpisode]:
    """The episodes in season and episode order."""

    return sorted(episodes, key=lambda ep: season_episode_key(ep.season_number, ep.episode_number))


def _is_regular(ep: SonarrEpisode) -> bool:
    """Whether the episode is in a season above 0: a row a group counts."""

    return ep.season_number is not None and ep.season_number > 0


def _regular(episodes: Iterable[SonarrEpisode]) -> list[SonarrEpisode]:
    """The episodes of a season above 0, in season and episode order: the rows a group counts."""

    return _ordered(filter(_is_regular, episodes))


def _absolutes_increase(episodes: Iterable[SonarrEpisode]) -> bool:
    """Whether every episode carries an absolute number and they strictly increase. No episodes at all pass."""

    return _increasing([ep.absolute_episode_number for ep in episodes])


@dataclass(frozen=True, slots=True, eq=False)
class _Window:
    """One window of the torrent on the grabbing title's series, with what the gates read off it.

    Compared by identity: a fit is keyed by the window it reads through.
    """

    ids: frozenset[int]
    """The window's ids the series index knows."""
    regular_ids: tuple[int, ...]
    """The window's regular rows (season above 0), in season and episode order."""
    specials: tuple[int, ...]
    """The window's specials (season 0), in episode order."""
    whole: bool
    """Whether the window matches AniList's episode count: the count is unknown, or it is the regular rows' or
    the whole window's."""
    monotone: bool
    """Whether the regular rows all carry absolutes that strictly increase. A window with no regular rows passes:
    its specials are read in order."""
    season_rows: Mapping[int, int]
    """Episode number -> id over the regular rows of the window's first regular season."""
    titles: frozenset[str]
    """`_name_key` of each AniList name of the title, the blank left out."""
    marked: frozenset[str]
    """`_marked_name_key` of each AniList name of the title whose `_name_key` isn't blank."""

    @classmethod
    def of(cls, window: TitleWindow, index: EpisodeIndex) -> Self:
        """The window's ids as `index` knows them, in season and episode order, and the facts read off them."""

        by_id = index.by_id
        rows = _ordered(by_id[ep_id] for ep_id in window.ordered_episode_ids if ep_id in by_id)
        regular_ids = [ep for ep in rows if _is_regular(ep)]
        first_season = regular_ids[0].season_number if regular_ids else None
        count = window.episode_count
        titles = window.names.anilist
        return cls(
            ids=frozenset(ep.id for ep in rows),
            regular_ids=tuple(ep.id for ep in regular_ids),
            specials=tuple(ep.id for ep in rows if ep.season_number == 0),
            whole=count is None or count in (len(regular_ids), len(rows)),
            monotone=_absolutes_increase(regular_ids),
            season_rows={
                ep.episode_number: ep.id
                for ep in regular_ids
                if ep.season_number == first_season and ep.episode_number is not None
            },
            titles=frozenset(_name_key(title) for title in titles) - {""},
            marked=frozenset(_marked_name_key(title) for title in titles if _name_key(title)),
        )

    def row(self, k: int) -> int | None:
        """`FROM_WINDOW_START`: the k-th regular row, or the k-th special when the window has no regular rows."""

        rows = self.regular_ids or self.specials
        return rows[k - 1] if 1 <= k <= len(rows) else None

    def season_row(self, k: int) -> int | None:
        """`SEASON_NUMBERS`: episode k of the window's first regular season."""

        return self.season_rows.get(k)


@dataclass(frozen=True, slots=True)
class _WideFit:
    """A numbering that fits the torrent and reads through no one window.

    `FROM_SERIES_START` and `TVDB_ABSOLUTE` read over the whole series, `ACROSS_WINDOWS` over the torrent's windows
    together.
    """

    numbering: Numbering


@dataclass(frozen=True, slots=True)
class _WindowFit:
    """A `FROM_WINDOW_START` or `SEASON_NUMBERS` that fits the torrent, read through one window."""

    numbering: Numbering
    window: _Window


type _Fit = _WideFit | _WindowFit
"""A numbering that fits the torrent's numbers, with the window it reads through when it reads through one."""


@dataclass(frozen=True, slots=True)
class _AbsoluteOnlyFile:
    """A file of the pool: its numbers and the series name Sonarr parsed from its file name."""

    name: str
    numbers: tuple[int, ...]
    """The absolute numbers the parse carries, in the name's order."""
    title: str
    """The parsed series name, "" when Sonarr parsed none."""

    @property
    def marked(self) -> str:
        """The parsed series name's `_marked_name_key`, the partition's key."""

        return _marked_name_key(self.title)


def _is_absolute_only(info: ParsedFileInfo) -> bool:
    """Whether the parse names the file by number alone.

    Absolute numbers and no episode numbers, no season above 0 (an absolute-numbered name reports season 0), and
    neither a special nor a season pack.
    """

    return (
        bool(info.absolute_episode_numbers)
        and not info.episode_numbers
        and info.season_number in (None, 0)
        and not info.special
        and not info.full_season
    )


@dataclass(frozen=True, slots=True)
class _Pool:
    """The torrent's absolute-only files, the numberings that fit them, and what the repeat and partition gates read.

    Built by `NumberingStep._pool`, once per torrent: a file is read under the pool's fits, or under its parsed
    series name's part.
    """

    files: tuple[_AbsoluteOnlyFile, ...]
    repeated: frozenset[int]
    """The numbers more than one file carries."""
    fits: tuple[_Fit, ...]
    """The numberings under which every number of the pool reads inside the torrent's windows together. Empty when
    the pool repeats a number and carries more than one parsed series name (`_raw_name_key`): that may be two seasons
    each numbered from one, so no file is read over the whole torrent, and each is read under the one window its name
    names, or gets no proposal. In a pool of one name a repeated number leaves out only the files carrying it."""
    parts: Mapping[str, tuple[_Fit, ...]]
    """`_marked_name_key` of a parsed series name -> the numberings under which every number of the name's files reads
    inside the one window that name names, for the names that name exactly one window and repeat no number among
    themselves."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", MappingProxyType(dict(self.parts)))

    def carries_repeat(self, file: _AbsoluteOnlyFile) -> bool:
        """Whether one of the file's numbers is one another file carries too."""

        return any(k in self.repeated for k in file.numbers)


def _window_numberings_disagree(rows_by_fit: Mapping[_Fit, tuple[int | None, ...]]) -> bool:
    """Whether neither series-wide numbering fits and the ones that do read the file to different rows.

    `ACROSS_WINDOWS` counts among them: like the window numberings it counts the torrent's rows from one, not
    TVDB's.
    """

    return all(fit.numbering not in _SERIES_WIDE for fit in rows_by_fit) and len(set(rows_by_fit.values())) > 1


def _rows(ids: Iterable[int | None]) -> tuple[int, ...] | None:
    """The ids when every number read to a row, else None."""

    read = tuple(ids)
    rows = tuple(ep_id for ep_id in read if ep_id is not None)
    return rows if len(rows) == len(read) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class NumberingStep:
    """Reads a torrent's absolute-only files against its windows on the grabbing title's series.

    Built by `of` from the grabbing title and the torrent's window order, and `propose` proposes ids for one batch.
    """

    series: EpisodeIndex
    """The grabbing title's whole series."""
    regular_ids: tuple[int, ...]
    """The series' regular rows, in season and episode order: what `FROM_SERIES_START` counts."""
    windows: tuple[_Window, ...]
    """The torrent's windows on the series, in window order. A window on another series is left out."""
    union_regular_ids: tuple[int, ...]
    """The regular rows of every window together, in season and episode order: what `ACROSS_WINDOWS` counts."""
    union_monotone: bool
    """Whether the union's regular rows all carry absolutes that strictly increase."""
    known_titles: frozenset[str]
    """`_name_key` of the series' name, of each alternate name, and of each window's AniList names."""
    aired_order: bool
    """Whether Sonarr numbers the series as aired. Off it the step proposes nothing: the regular rows are then
    not the rows a group counts."""

    @classmethod
    def of(cls, grabbing_title: GrabbingTitle, windows: Iterable[TitleWindow]) -> Self:
        """The step for one torrent: `windows` are the torrent's (`WindowOrder.windows`), any series."""

        index = grabbing_title.series_index
        own = tuple(_Window.of(window, index) for window in windows if window.series_id == grabbing_title.series_id)
        regular_ids = _regular(index.by_id.values())
        union = frozenset(ep_id for window in own for ep_id in window.ids)
        union_regular_ids = [ep for ep in regular_ids if ep.id in union]
        known = {_name_key(title) for title in (grabbing_title.title_names.series, *grabbing_title.alternate_titles)}
        known.update(title for window in own for title in window.titles)
        return cls(
            series=index,
            regular_ids=tuple(ep.id for ep in regular_ids),
            windows=own,
            union_regular_ids=tuple(ep.id for ep in union_regular_ids),
            union_monotone=_absolutes_increase(union_regular_ids),
            known_titles=frozenset(known - {""}),
            aired_order=grabbing_title.aired_order,
        )

    def propose(self, batch: PlacementBatch, facts: Mapping[str, NameFacts]) -> NumberingProposals:
        """The ids proposed for each absolute-only file of `batch` where every fitting numbering agrees.

        `facts` are the grabbing title's window's (`PlacerResult.facts`): they say which names are extras or parsed
        by their title's number, which keeps those out of the pool. Nothing is proposed off the aired numbering,
        while a parse of the torrent is unknown, with no window on the series, or when a window doesn't match
        AniList's count. Two files proposed overlapping ids are both dropped: one id, one file.
        """

        if not self.aired_order or not self.windows or not batch.all_parses_read:
            return NO_PROPOSALS
        # A window whose rows aren't what AniList counts for the title (Panty & Stocking: 13 double episodes on
        # AniList, 26 rows in Sonarr) isn't the thing the group counted, so no position in it means what it seems.
        if any(not window.whole for window in self.windows):
            return NO_PROPOSALS
        pool = self._pool(_absolute_only_files(batch, facts))
        proposals = {file.name: ids for file in pool.files if (ids := self._propose(file, pool)) is not None}
        use_counts = Counter(ep_id for ids in proposals.values() for ep_id in ids)
        return NumberingProposals(
            {name: ids for name, ids in proposals.items() if all(use_counts[i] == 1 for i in ids)}
        )

    def _pool(self, files: Sequence[_AbsoluteOnlyFile]) -> _Pool:
        """The pool over `files`, fitted against the torrent's windows, each name's part against its one window."""

        counts = Counter(k for file in files for k in file.numbers)
        repeated = frozenset(k for k, count in counts.items() if count > 1)
        one_title = len({_raw_name_key(file.title) for file in files}) == 1
        numbers = tuple(sorted(counts))
        # A marked name two windows share (two titles both named "K-ON!") names neither.
        owners = Counter(marked for window in self.windows for marked in window.marked)
        sole = {marked: window for window in self.windows for marked in window.marked if owners[marked] == 1}
        by_marked: dict[str, list[int]] = {}
        for file in files:
            by_marked.setdefault(file.marked, []).extend(file.numbers)
        parts: dict[str, tuple[_Fit, ...]] = {}
        for marked, carried in by_marked.items():
            window = sole.get(marked)
            if window is not None and len(set(carried)) == len(carried):
                parts[marked] = self._fits(tuple(sorted(carried)), (window,))
        return _Pool(
            files=tuple(files),
            repeated=repeated,
            fits=self._fits(numbers, self.windows) if one_title or not repeated else (),
            parts=parts,
        )

    def _propose(self, file: _AbsoluteOnlyFile, pool: _Pool) -> tuple[int, ...] | None:
        """The file's ids where every fitting numbering agrees, else None: the gates, in order."""

        rows_by_fit = self._rows_by_fit(file, pool.fits)
        # The name partition runs when the torrent repeats a number, when no numbering fits it, or when only
        # window numberings fit and they disagree: a torrent of two titles (K-ON! and K-ON!!) reads each name's
        # files under the one window that name names.
        if pool.repeated or not rows_by_fit or _window_numberings_disagree(rows_by_fit):
            fits = pool.parts.get(file.marked)
            if fits is not None:
                rows_by_fit = self._rows_by_fit(file, fits)
            elif pool.carries_repeat(file):
                return None
            if not rows_by_fit:
                return None
        gated = {fit: ids for fit, ids in rows_by_fit.items() if self._passes_window_gate(fit)}
        if not gated:
            return None
        distinct: set[tuple[int, ...]] = set()
        for ids in gated.values():
            rows = _rows(ids)
            if rows is None:
                # Only `FROM_SERIES_START` reads a fitting number to no row: the k-th regular row doesn't carry
                # absolute k.
                return None
            distinct.add(rows)
        if len(distinct) != 1 or not any(fit.numbering in _PLACING for fit in gated):
            return None
        # A parsed series name the run doesn't know (a recut's, another series') says the group may have counted
        # something other than this series' episodes, whatever the numbers agree on.
        if _name_key(file.title) not in self.known_titles:
            return None
        return distinct.pop()

    def _rows_by_fit(self, file: _AbsoluteOnlyFile, fits: Iterable[_Fit]) -> dict[_Fit, tuple[int | None, ...]]:
        """The file's numbers read under each of `fits`, in the name's order."""

        return {fit: tuple(self._row_of(fit, k) for k in file.numbers) for fit in fits}

    def _fits(self, numbers: Sequence[int], windows: Sequence[_Window]) -> tuple[_Fit, ...]:
        """The numberings under which every number reads to a row inside `windows`.

        `FROM_SERIES_START` and `TVDB_ABSOLUTE` fit together when TVDB's row for every number lies inside the
        windows: a group counting the series from one lands on TVDB's numbers wherever TVDB counts straight, and
        where it doesn't, the position check on each file tells. `FROM_WINDOW_START` and `SEASON_NUMBERS` fit per
        window. `ACROSS_WINDOWS` fits two or more windows when every number has a union row.
        """

        union = frozenset(ep_id for window in windows for ep_id in window.ids)
        fits: list[_Fit] = []
        if all((ep_id := self.series.id_by_absolute.get(k)) is not None and ep_id in union for k in numbers):
            fits.append(_WideFit(Numbering.FROM_SERIES_START))
            fits.append(_WideFit(Numbering.TVDB_ABSOLUTE))
        for window in windows:
            for numbering in _PER_WINDOW:
                fit = _WindowFit(numbering, window)
                if all((ep_id := self._row_of(fit, k)) is not None and ep_id in window.ids for k in numbers):
                    fits.append(fit)
        if len(windows) > 1 and all(self._union_row(k) is not None for k in numbers):
            fits.append(_WideFit(Numbering.ACROSS_WINDOWS))
        return tuple(fits)

    def _row_of(self, fit: _Fit, k: int) -> int | None:
        """The row number k names under `fit`, else None."""

        match fit:
            case _WindowFit(numbering=Numbering.FROM_WINDOW_START, window=window):
                return window.row(k)
            case _WindowFit(window=window):
                return window.season_row(k)
            case _WideFit(numbering=Numbering.FROM_SERIES_START):
                return self._series_row(k)
            case _WideFit(numbering=Numbering.TVDB_ABSOLUTE):
                return self.series.id_by_absolute.get(k)
            case _WideFit():
                return self._union_row(k)

    def _series_row(self, k: int) -> int | None:
        """The series' k-th regular row, when it carries absolute k.

        A row numbered otherwise means TVDB counts the series differently from its rows (a special in the count, a
        shuffled order), so the k-th row isn't what a group counting from one meant: the file gets no proposal.
        """

        if not 1 <= k <= len(self.regular_ids):
            return None
        ep_id = self.regular_ids[k - 1]
        return ep_id if self.series.absolute_of.get(ep_id) == k else None

    def _union_row(self, k: int) -> int | None:
        """The k-th regular row across the torrent's windows."""

        return self.union_regular_ids[k - 1] if 1 <= k <= len(self.union_regular_ids) else None

    def _passes_window_gate(self, fit: _Fit) -> bool:
        """Whether a window numbering's window, or the union, counts its rows in order.

        Rows whose absolutes don't increase are a season Sonarr orders differently from how it aired, so counting
        them from one lands wrong. The series-wide numberings aren't gated here: the position check is theirs.
        """

        if isinstance(fit, _WindowFit):
            return fit.window.monotone
        return fit.numbering is not Numbering.ACROSS_WINDOWS or self.union_monotone


def _absolute_only_files(batch: PlacementBatch, facts: Mapping[str, NameFacts]) -> list[_AbsoluteOnlyFile]:
    """The pool's files: every absolute-only file of the torrent, names in the file map and gone names included.

    `facts` must cover every name of the torrent (`PlacerResult.facts` does).
    """

    pool: list[_AbsoluteOnlyFile] = []
    for name in batch.all_names:
        info = batch.parsed.get(name)
        fact = facts[name]
        if info is not None and _is_absolute_only(info) and not fact.extra and not fact.parsed_by_title_number:
            pool.append(_AbsoluteOnlyFile(name, info.absolute_episode_numbers, info.series_title))
    return pool
