"""The numberings reader: where a file named by number alone falls, once every way of counting agrees.

A release group that names a file "Show - 07" counted the series' episodes some way: from the first episode of the
whole series (`Numbering.SERIES_ROW`), from the first episode of the entry's season (`WINDOW_SEASON`), from the
first episode the entry covers (`WINDOW_ROW`), straight through every entry the torrent is listed under
(`UNION_ROW`), or by TVDB's absolute numbers (`ABSOLUTE`). The passes place such a file when Sonarr's reading of its
number lands inside the window or a run carries it. For a lone-number file they leave, the reader reads the number
under every numbering that fits the torrent's windows and proposes its row only when the numberings that fit all
name the same row. A torrent whose windows don't match AniList's episode counts, a number repeated under one title,
a series row that doesn't carry its number, two numberings that disagree, or a lifted series title the run doesn't
know each refuse the file, which keeps its verdict. Nothing the reader proposes outranks a pass: `WindowedAssignment.filled`
places a proposal only into a gap the windows left.

The reader runs once per torrent, after the windows merge, on the grabbing entry's series only, and only while
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
    first season on. Fits a torrent only together with `ABSOLUTE`."""
    TVDB_ABSOLUTE = "absolute"
    """The row TVDB numbers k. A veto only: it never places alone, since a group counting its own way may land
    where TVDB's number happens to agree."""
    FROM_WINDOW_START = "window row"
    """The window's k-th row: the group counted the entry's episodes from one. A window of nothing but specials
    counts its specials."""
    SEASON_NUMBERS = "window season"
    """Episode k of the window's first regular season: the group used Sonarr's own episode numbers."""
    ACROSS_WINDOWS = "union row"
    """The k-th regular row across every window of the torrent, two or more windows only: the group counted a batch
    listed under several entries straight through."""


_PLACING = frozenset(
    {Numbering.FROM_SERIES_START, Numbering.FROM_WINDOW_START, Numbering.SEASON_NUMBERS, Numbering.ACROSS_WINDOWS}
)
"""The numberings whose agreement places a file. `ABSOLUTE` only vetoes."""
_SERIES_WIDE = frozenset({Numbering.FROM_SERIES_START, Numbering.TVDB_ABSOLUTE})
"""The numberings that count TVDB's whole series: its regular rows from the first season on, and its absolutes."""
_PER_WINDOW = (Numbering.FROM_WINDOW_START, Numbering.SEASON_NUMBERS)
"""The numberings read through one window of the torrent."""


@dataclass(frozen=True, slots=True)
class NumberingProposals:
    """What the reader proposes for a torrent: each lone-number file's ids where every fitting numbering agrees.

    A file not named keeps its verdict.
    """

    ids: Mapping[str, tuple[int, ...]]
    """Normalized name -> the proposed episode ids, one per number of the name, in the name's order."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "ids", MappingProxyType(dict(self.ids)))


NO_PROPOSALS = NumberingProposals({})
"""The reader proposing nothing."""


def _norm_title(title: str) -> str:
    """The title lowercased, `&` read as "and", every other run of non-alphanumerics one space.

    `release_names._raw_words` splits a name into words the same way, casefolded and without the `&` reading. This
    key stays its own so the title keys below never drift with the name reader's.
    """

    folded = re.sub(r"[^a-z0-9]+", " ", title.lower().replace("&", " and "))
    return " ".join(folded.split())


def _raw_title(title: str) -> str:
    """The title lowercased with the " - " and ": " separators folded to a space, punctuation kept.

    Two lifted titles that differ only in how Sonarr joined their parts ("Show - Arc", "Show: Arc") are one title.
    """

    return " ".join(title.lower().replace(" - ", " ").replace(": ", " ").split())


_SEP_TAIL = re.compile(r"[\s\-:~\)\]\}]+$")
"""The separators a title can end in once its number is cut off."""
_MARKS = re.compile(r"[!+'’.?]+$")
"""The run of marks a title ends in, the only punctuation `_marked_title` keeps."""


def _marked_title(title: str) -> str:
    """`_norm_title` plus the title's trailing marks, after a bar.

    The marks tell K-ON! from K-ON!!, New Game! from New Game!!, Needless from Needless+ and Dog Days from Dog
    Days', which the normalization folds together. A typographic apostrophe counts as the plain one.
    """

    core = _SEP_TAIL.sub("", title.strip())
    found = _MARKS.search(core)
    marks = found.group(0).replace("’", "'") if found else ""
    return f"{_norm_title(core)}|{marks}"


# `release_names.folded_words` isn't reused for these keys: it folds roman numerals, ordinals and accents and drops
# bracketed text, which would merge titles the partition and the known-title gate keep apart.


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
    """One window of the torrent on the grabbing entry's series, with what the gates read off it.

    Compared by identity: a reading is keyed by the window it was read through.
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
    """`_norm_title` of each AniList title of the entry, the blank left out."""
    marked: frozenset[str]
    """`_marked_title` of each AniList title of the entry whose `_norm_title` isn't blank."""

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
            titles=frozenset(_norm_title(title) for title in titles) - {""},
            marked=frozenset(_marked_title(title) for title in titles if _norm_title(title)),
        )

    def row(self, k: int) -> int | None:
        """`WINDOW_ROW`: the k-th regular row, or the k-th special when the window has no regular rows."""

        rows = self.regular_ids or self.specials
        return rows[k - 1] if 1 <= k <= len(rows) else None

    def season_row(self, k: int) -> int | None:
        """`WINDOW_SEASON`: episode k of the window's first regular season."""

        return self.season_rows.get(k)


@dataclass(frozen=True, slots=True)
class _WideFit:
    """A numbering that fits the torrent and reads through no one window.

    `SERIES_ROW` and `ABSOLUTE` read over the whole series, `UNION_ROW` over the torrent's windows together.
    """

    numbering: Numbering


@dataclass(frozen=True, slots=True)
class _WindowFit:
    """A `WINDOW_ROW` or `WINDOW_SEASON` that fits the torrent, read through one window."""

    numbering: Numbering
    window: _Window


type _Fit = _WideFit | _WindowFit
"""A numbering that fits the torrent's numbers, with the window it reads through when it reads through one."""


@dataclass(frozen=True, slots=True)
class _AbsoluteOnlyFile:
    """A file of the pool: its numbers and the series title Sonarr lifted from its name."""

    name: str
    numbers: tuple[int, ...]
    """The absolute numbers the parse carries, in the name's order."""
    title: str
    """The lifted series title, "" when Sonarr lifted none."""

    @property
    def marked(self) -> str:
        """The lifted title's `_marked_title`, the partition's key."""

        return _marked_title(self.title)


def _lone_number(info: ParsedFileInfo) -> bool:
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
    """The torrent's lone-number files, the numberings that fit them, and what the repeat and partition gates read.

    Built by `NumberingReader._pool`, once per torrent: a file is read under the pool's fits, or under its title's
    part.
    """

    files: tuple[_AbsoluteOnlyFile, ...]
    repeated: frozenset[int]
    """The numbers more than one file carries."""
    fits: tuple[_Fit, ...]
    """The numberings under which every number of the pool reads inside the torrent's windows together. Empty when
    the pool repeats a number and carries more than one lifted title (`_raw_title`): that may be two seasons each
    numbered from one, so no file is read over the whole torrent, and each is read under the one window its title
    names, or refused. In a pool of one title a repeated number refuses only the files carrying it."""
    parts: Mapping[str, tuple[_Fit, ...]]
    """`_marked_title` of a lifted title -> the numberings under which every number of the title's files reads
    inside the one window that title names, for the titles that name exactly one window and repeat no number among
    themselves."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", MappingProxyType(dict(self.parts)))

    def carries_repeat(self, file: _AbsoluteOnlyFile) -> bool:
        """Whether one of the file's numbers is one another file carries too."""

        return any(k in self.repeated for k in file.numbers)


def _window_numberings_disagree(readings: Mapping[_Fit, tuple[int | None, ...]]) -> bool:
    """Whether neither series-wide numbering fits and the ones that do read the file to different rows.

    `UNION_ROW` counts among them: like the window numberings it counts the torrent's own rows from one, not
    TVDB's.
    """

    return all(fit.numbering not in _SERIES_WIDE for fit in readings) and len(set(readings.values())) > 1


def _rows(ids: Iterable[int | None]) -> tuple[int, ...] | None:
    """The ids when every number read to a row, else None."""

    read = tuple(ids)
    rows = tuple(ep_id for ep_id in read if ep_id is not None)
    return rows if len(rows) == len(read) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class NumberingStep:
    """Reads a torrent's lone-number files against its windows on the grabbing entry's series.

    Built by `of` from the grab's scope and the torrent's window order; `read` proposes ids for one batch.
    """

    series: EpisodeIndex
    """The grabbing entry's whole series."""
    regular_ids: tuple[int, ...]
    """The series' regular rows, in season and episode order: what `SERIES_ROW` counts."""
    windows: tuple[_Window, ...]
    """The torrent's windows on the series, in window order. A window on another series is left out."""
    union_regs: tuple[int, ...]
    """The regular rows of every window together, in season and episode order: what `UNION_ROW` counts."""
    union_monotone: bool
    """Whether the union's regular rows all carry absolutes that strictly increase."""
    known_titles: frozenset[str]
    """`_norm_title` of the series title, of each alternate title, and of each window's AniList titles."""
    aired_order: bool
    """Whether Sonarr numbers the series as aired. Off it the reader proposes nothing: the regular rows are then
    not the rows a group counts."""

    @classmethod
    def of(cls, scope: GrabbingTitle, windows: Iterable[TitleWindow]) -> Self:
        """The reader for one torrent: `windows` are the torrent's (`WindowOrder.windows`), any series."""

        index = scope.series_index
        own = tuple(_Window.of(window, index) for window in windows if window.series_id == scope.series_id)
        regular_ids = _regular(index.by_id.values())
        union = frozenset(ep_id for window in own for ep_id in window.ids)
        union_regs = [ep for ep in regular_ids if ep.id in union]
        known = {_norm_title(title) for title in (scope.names.series, *scope.alternate_titles)}
        known.update(title for window in own for title in window.titles)
        return cls(
            series=index,
            regular_ids=tuple(ep.id for ep in regular_ids),
            windows=own,
            union_regs=tuple(ep.id for ep in union_regs),
            union_monotone=_absolutes_increase(union_regs),
            known_titles=frozenset(known - {""}),
            aired_order=scope.aired_order,
        )

    def read(self, batch: PlacementBatch, facts: Mapping[str, NameFacts]) -> NumberingProposals:
        """The ids proposed for each lone-number file of `batch` where every fitting numbering agrees.

        `facts` are the grabbing entry's own window's (`PlacerResult.facts`): they say which names are extras or
        counted by their own title, which keeps those out of the pool. Nothing is proposed off the aired numbering,
        while a parse of the torrent is unknown, with no window on the series, or when a window doesn't match
        AniList's count. Two files proposed overlapping ids are both dropped: one id, one file.
        """

        if not self.aired_order or not self.windows or not batch.all_parses_known:
            return NO_PROPOSALS
        # A window whose rows aren't what AniList counts for the entry (Panty & Stocking: 13 double episodes on
        # AniList, 26 rows in Sonarr) isn't the thing the group counted, so no position in it means what it seems.
        if any(not window.whole for window in self.windows):
            return NO_PROPOSALS
        pool = self._pool(_lone_number_files(batch, facts))
        proposals = {file.name: ids for file in pool.files if (ids := self._propose(file, pool)) is not None}
        held = Counter(ep_id for ids in proposals.values() for ep_id in ids)
        return NumberingProposals({name: ids for name, ids in proposals.items() if all(held[i] == 1 for i in ids)})

    def _pool(self, files: Sequence[_AbsoluteOnlyFile]) -> _Pool:
        """The pool over `files`, fitted against the torrent's windows, each title's part against its one window."""

        counts = Counter(k for file in files for k in file.numbers)
        repeated = frozenset(k for k, count in counts.items() if count > 1)
        one_title = len({_raw_title(file.title) for file in files}) == 1
        numbers = tuple(sorted(counts))
        # A marked title two windows share (two entries both titled "K-ON!") names neither.
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

        readings = self._readings(file, pool.fits)
        # The title partition runs when the torrent repeats a number, when no numbering fits it, or when only
        # window numberings fit and they disagree: a torrent of two titles (K-ON! and K-ON!!) reads each title's
        # files under the one window that title names.
        if pool.repeated or not readings or _window_numberings_disagree(readings):
            fits = pool.parts.get(file.marked)
            if fits is not None:
                readings = self._readings(file, fits)
            elif pool.carries_repeat(file):
                return None
            if not readings:
                return None
        gated = {fit: ids for fit, ids in readings.items() if self._passes_window_gate(fit)}
        if not gated:
            return None
        distinct: set[tuple[int, ...]] = set()
        for ids in gated.values():
            rows = _rows(ids)
            if rows is None:
                # Only `SERIES_ROW` reads a fitting number to no row: the k-th regular row doesn't carry absolute k.
                return None
            distinct.add(rows)
        if len(distinct) != 1 or not any(fit.numbering in _PLACING for fit in gated):
            return None
        # A lifted title the run doesn't know (a recut's, another show's) says the group may have counted something
        # other than this series' episodes, whatever the numbers agree on.
        if _norm_title(file.title) not in self.known_titles:
            return None
        return distinct.pop()

    def _readings(self, file: _AbsoluteOnlyFile, fits: Iterable[_Fit]) -> dict[_Fit, tuple[int | None, ...]]:
        """The file's numbers read under each of `fits`, in the name's order."""

        return {fit: tuple(self._read(fit, k) for k in file.numbers) for fit in fits}

    def _fits(self, numbers: Sequence[int], windows: Sequence[_Window]) -> tuple[_Fit, ...]:
        """The numberings under which every number reads to a row inside `windows`.

        `SERIES_ROW` and `ABSOLUTE` fit together when TVDB's row for every number lies inside the windows: a group
        counting the series from one lands on TVDB's numbers wherever TVDB counts straight, and where it doesn't,
        the position check on each file tells. `WINDOW_ROW` and `WINDOW_SEASON` fit per window. `UNION_ROW` fits
        two or more windows when every number has a union row.
        """

        union = frozenset(ep_id for window in windows for ep_id in window.ids)
        fits: list[_Fit] = []
        if all((ep_id := self.series.id_by_absolute.get(k)) is not None and ep_id in union for k in numbers):
            fits.append(_WideFit(Numbering.FROM_SERIES_START))
            fits.append(_WideFit(Numbering.TVDB_ABSOLUTE))
        for window in windows:
            for numbering in _PER_WINDOW:
                fit = _WindowFit(numbering, window)
                if all((ep_id := self._read(fit, k)) is not None and ep_id in window.ids for k in numbers):
                    fits.append(fit)
        if len(windows) > 1 and all(self._union_row(k) is not None for k in numbers):
            fits.append(_WideFit(Numbering.ACROSS_WINDOWS))
        return tuple(fits)

    def _read(self, fit: _Fit, k: int) -> int | None:
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
        shuffled order), so the k-th row isn't what a group counting from one meant: the file is refused.
        """

        if not 1 <= k <= len(self.regular_ids):
            return None
        ep_id = self.regular_ids[k - 1]
        return ep_id if self.series.absolute_of.get(ep_id) == k else None

    def _union_row(self, k: int) -> int | None:
        """The k-th regular row across the torrent's windows."""

        return self.union_regs[k - 1] if 1 <= k <= len(self.union_regs) else None

    def _passes_window_gate(self, fit: _Fit) -> bool:
        """Whether a window numbering's window, or the union, counts its rows in order.

        Rows whose absolutes don't increase are a season Sonarr orders differently from how it aired, so counting
        them from one lands wrong. The series-wide numberings aren't gated here: the position check is theirs.
        """

        if isinstance(fit, _WindowFit):
            return fit.window.monotone
        return fit.numbering is not Numbering.ACROSS_WINDOWS or self.union_monotone


def _lone_number_files(batch: PlacementBatch, facts: Mapping[str, NameFacts]) -> list[_AbsoluteOnlyFile]:
    """The pool's files: every file of the torrent named by number alone, seeded and gone names included.

    `facts` must cover every name of the torrent (`PlacerResult.facts` does).
    """

    pool: list[_AbsoluteOnlyFile] = []
    for name in batch.all_names:
        info = batch.parsed.get(name)
        fact = facts[name]
        if info is not None and _lone_number(info) and not fact.extra and not fact.title_counted:
            pool.append(_AbsoluteOnlyFile(name, info.absolute_episode_numbers, info.series_title))
    return pool
