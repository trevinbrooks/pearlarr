"""The numberings reader: where a file named by number alone falls, once every way of counting agrees.

A release group that names a file "Show - 07" counted the series' episodes some way: from the first episode of the
whole series (`Numbering.SERIES_ROW`), from the first episode of the entry's season (`WINDOW_SEASON`), from the
first episode the entry covers (`WINDOW_ROW`), straight through every entry the torrent is listed under
(`UNION_ROW`), or by TVDB's absolute numbers (`ABSOLUTE`). The passes place such a file when Sonarr's reading of its
number lands inside the window or a run carries it. For a lone-number file they leave, the reader reads the number
under every numbering that fits the torrent's windows and proposes its row only when the numberings that fit all
name the same row. A torrent whose windows don't match AniList's episode counts, a repeated number, a numbering that
counts past its rows, two numberings that disagree, or a lifted series title the run doesn't know each refuse the
file, which keeps its verdict. Nothing the reader proposes outranks a pass: `WindowedAssignment.filled` places a
proposal only into a gap the windows left.

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
from typing import TYPE_CHECKING, Self

from .manual_import import EntryWindow
from .placement_types import EpisodeIndex, PlacementBatch
from .placer import NameFacts
from .seadex_types import ParsedFileInfo, season_episode_key

if TYPE_CHECKING:
    # Annotation-only: grab_placement imports this module at runtime (cycle).
    from .grab_placement import SeedScope


class Numbering(Enum):
    """One way a release group could have counted a series' episodes: what the number k in a file name means."""

    SERIES_ROW = "series row"
    """The series' k-th regular row, in season and episode order: the group counted every regular episode from the
    first season on. Fits a torrent only together with `ABSOLUTE`."""
    ABSOLUTE = "absolute"
    """The row TVDB numbers k. A veto only: it never places alone, since a group counting its own way may land
    where TVDB's number happens to agree."""
    WINDOW_ROW = "window row"
    """The window's k-th row: the group counted the entry's episodes from one. A window of nothing but specials
    counts its specials."""
    WINDOW_SEASON = "window season"
    """Episode k of the window's first regular season: the group used Sonarr's own episode numbers."""
    UNION_ROW = "union row"
    """The k-th regular row across every window of the torrent, two or more windows only: the group counted a batch
    listed under several entries straight through."""


_PLACING = frozenset({Numbering.SERIES_ROW, Numbering.WINDOW_ROW, Numbering.WINDOW_SEASON, Numbering.UNION_ROW})
"""The numberings whose agreement places a file. `ABSOLUTE` only vetoes."""
_SERIES_WIDE = frozenset({Numbering.SERIES_ROW, Numbering.ABSOLUTE})
"""The numberings that count the whole series, read through no window."""
_RELATIVE = (Numbering.WINDOW_ROW, Numbering.WINDOW_SEASON)
"""The numberings read through one window."""


@dataclass(frozen=True, slots=True)
class NumberingReading:
    """What the reader proposes for a torrent: each lone-number file's ids where every fitting numbering agrees.

    A file not named keeps its verdict.
    """

    ids: Mapping[str, tuple[int, ...]]
    """Normalized name -> the proposed episode ids, one per number of the name, in the name's order."""

    def __post_init__(self) -> None:
        # Detach from the caller's dict, then wrap read-only.
        object.__setattr__(self, "ids", MappingProxyType(dict(self.ids)))


NO_PROPOSALS = NumberingReading({})
"""The reader proposing nothing."""


def _norm_title(title: str) -> str:
    """The title lowercased, `&` read as "and", every other run of non-alphanumerics one space."""

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
    """Whether every number is known and they strictly increase."""

    known = [n for n in numbers if n is not None]
    return len(known) == len(numbers) and all(later > earlier for earlier, later in pairwise(known))


@dataclass(frozen=True, slots=True, eq=False)
class _Window:
    """One window of the torrent on the grabbing entry's series, with what the gates read off it.

    Compared by identity: a reading is keyed by the window it was read through.
    """

    ids: frozenset[int]
    """The window's ids the series index knows."""
    regs: tuple[int, ...]
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
    def of(cls, window: EntryWindow, index: EpisodeIndex) -> Self:
        """The window's ids as `index` knows them, in season and episode order, and the facts read off them."""

        by_id = index.by_id
        rows = sorted(
            (by_id[ep_id] for ep_id in window.ordered_episode_ids if ep_id in by_id),
            key=lambda ep: season_episode_key(ep.season_number, ep.episode_number),
        )
        regs = [ep for ep in rows if ep.season_number is not None and ep.season_number > 0]
        first_season = regs[0].season_number if regs else None
        count = window.episode_count
        titles = window.names.anilist
        return cls(
            ids=frozenset(ep.id for ep in rows),
            regs=tuple(ep.id for ep in regs),
            specials=tuple(ep.id for ep in rows if ep.season_number == 0),
            whole=count is None or count in (len(regs), len(rows)),
            monotone=not regs or _increasing([ep.absolute_episode_number for ep in regs]),
            season_rows={
                ep.episode_number: ep.id
                for ep in regs
                if ep.season_number == first_season and ep.episode_number is not None
            },
            titles=frozenset(_norm_title(title) for title in titles) - {""},
            marked=frozenset(_marked_title(title) for title in titles if _norm_title(title)),
        )

    def row(self, k: int) -> int | None:
        """`WINDOW_ROW`: the k-th regular row, or the k-th special when the window has no regular rows."""

        rows = self.regs or self.specials
        return rows[k - 1] if 1 <= k <= len(rows) else None

    def season_row(self, k: int) -> int | None:
        """`WINDOW_SEASON`: episode k of the window's first regular season."""

        return self.season_rows.get(k)


@dataclass(frozen=True, slots=True)
class _Fit:
    """A numbering that fits the torrent's numbers, read through one window when it is a relative one."""

    numbering: Numbering
    window: _Window | None
    """The window a `WINDOW_ROW` or `WINDOW_SEASON` reads through; None for the series-wide numberings."""


@dataclass(frozen=True, slots=True)
class _LoneNumberFile:
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
class _Part:
    """One title's files of a repeating or ambiguous torrent, read again under the one window that title names."""

    window: _Window
    numbers: tuple[int, ...]
    """The part's numbers, each carried once, sorted."""


@dataclass(frozen=True, slots=True)
class _Pool:
    """The torrent's lone-number files and what the repeat and partition gates read over them."""

    files: tuple[_LoneNumberFile, ...]
    numbers: tuple[int, ...]
    """Every distinct number the pool carries, sorted."""
    repeated: frozenset[int]
    """The numbers more than one file carries."""
    one_title: bool
    """Whether the pool's lifted titles are one title (`_raw_title`). A repeated number under one title refuses only
    the files carrying it; under several titles, which may be two seasons each numbered from one, no file is read
    over the whole torrent: each is read under the one window its title names, or refused."""
    parts: Mapping[str, _Part]
    """`_marked_title` of a lifted title -> the one window that title names and the title's numbers, for the titles
    that name exactly one window and repeat no number among themselves."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", MappingProxyType(dict(self.parts)))

    @classmethod
    def of(cls, files: Sequence[_LoneNumberFile], windows: Sequence[_Window]) -> Self:
        """The pool over `files`, its parts routed to `windows`."""

        counts = Counter(k for file in files for k in file.numbers)
        # A marked title two windows share (two entries both titled "K-ON!") names neither.
        owners = Counter(marked for window in windows for marked in window.marked)
        by_marked: dict[str, list[_LoneNumberFile]] = {}
        for file in files:
            by_marked.setdefault(file.marked, []).append(file)
        parts: dict[str, _Part] = {}
        for marked, group in by_marked.items():
            hits = [window for window in windows if owners[marked] == 1 and marked in window.marked]
            numbers = [k for file in group for k in file.numbers]
            if len(hits) == 1 and len(set(numbers)) == len(numbers):
                parts[marked] = _Part(hits[0], tuple(sorted(numbers)))
        return cls(
            files=tuple(files),
            numbers=tuple(sorted(counts)),
            repeated=frozenset(k for k, count in counts.items() if count > 1),
            one_title=len({_raw_title(file.title) for file in files}) == 1,
            parts=parts,
        )


def _relative_disagreement(readings: Mapping[_Fit, tuple[int | None, ...]]) -> bool:
    """Whether only relative numberings fit and they read the file to different rows."""

    return all(fit.numbering not in _SERIES_WIDE for fit in readings) and len(set(readings.values())) > 1


def _rows(ids: Iterable[int | None]) -> tuple[int, ...] | None:
    """The ids when every number read to a row, else None."""

    read = tuple(ids)
    rows = tuple(ep_id for ep_id in read if ep_id is not None)
    return rows if len(rows) == len(read) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class NumberingReader:
    """Reads a torrent's lone-number files against its windows on the grabbing entry's series.

    Built by `of` from the grab's scope and the torrent's window order; `read` proposes ids for one batch.
    """

    series: EpisodeIndex
    """The grabbing entry's whole series."""
    regs: tuple[int, ...]
    """The series' regular rows, in season and episode order: what `SERIES_ROW` counts."""
    windows: tuple[_Window, ...]
    """The torrent's windows on the series, in window order. A window on another series is left out."""
    union_regs: tuple[int, ...]
    """The regular rows of every window together, in season and episode order: what `UNION_ROW` counts."""
    union_monotone: bool
    """Whether the union's regular rows all carry absolutes that strictly increase."""
    known_titles: frozenset[str]
    """`_norm_title` of the series title, of each alternate title, and of each window's AniList titles."""
    aired_numbering: bool
    """Whether Sonarr numbers the series as aired. Off it the reader proposes nothing: the regular rows are then
    not the rows a group counts."""

    @classmethod
    def of(cls, scope: "SeedScope", windows: Iterable[EntryWindow]) -> Self:
        """The reader for one torrent: `windows` are the torrent's (`WindowOrder.windows`), any series."""

        index = scope.series
        own = tuple(_Window.of(window, index) for window in windows if window.series_id == scope.series_id)
        regs = sorted(
            (ep for ep in index.by_id.values() if ep.season_number is not None and ep.season_number > 0),
            key=lambda ep: season_episode_key(ep.season_number, ep.episode_number),
        )
        union = frozenset(ep_id for window in own for ep_id in window.ids)
        union_regs = [ep for ep in regs if ep.id in union]
        known = {_norm_title(title) for title in (scope.names.series, *scope.alternate_titles)}
        known.update(title for window in own for title in window.titles)
        return cls(
            series=index,
            regs=tuple(ep.id for ep in regs),
            windows=own,
            union_regs=tuple(ep.id for ep in union_regs),
            union_monotone=bool(union_regs) and _increasing([ep.absolute_episode_number for ep in union_regs]),
            known_titles=frozenset(known - {""}),
            aired_numbering=scope.aired_numbering,
        )

    def read(self, batch: PlacementBatch, facts: Mapping[str, NameFacts]) -> NumberingReading:
        """The ids proposed for each lone-number file of `batch` where every fitting numbering agrees.

        `facts` are the grabbing entry's own window's (`name_facts`): they say which names are extras or counted by
        their own title, which keeps those out of the pool. Nothing is proposed off the aired numbering, while a
        parse of the torrent is unknown, with no window on the series, or when a window doesn't match AniList's
        count. Two files proposed overlapping ids are both dropped: one id, one file.
        """

        if not self.aired_numbering or not self.windows or not batch.all_parses_known:
            return NO_PROPOSALS
        if any(not window.whole for window in self.windows):
            return NO_PROPOSALS
        pool = _Pool.of(_lone_number_files(batch, facts), self.windows)
        proposals = {file.name: ids for file in pool.files if (ids := self._propose(file, pool)) is not None}
        held = Counter(ep_id for ids in proposals.values() for ep_id in ids)
        return NumberingReading({name: ids for name, ids in proposals.items() if all(held[i] == 1 for i in ids)})

    def _propose(self, file: _LoneNumberFile, pool: _Pool) -> tuple[int, ...] | None:
        """The file's ids where every fitting numbering agrees, else None: the gates, in order."""

        readings = self._whole_readings(file, pool)
        # The title partition runs when the torrent repeats a number, when no numbering fits it, or when only
        # relative numberings fit and they disagree: a torrent of two titles (K-ON! and K-ON!!) reads each title's
        # files under the one window that title names.
        if pool.repeated or not readings or _relative_disagreement(readings):
            part = pool.parts.get(file.marked)
            if part is not None:
                readings = self._readings(file, part.numbers, (part.window,))
            elif pool.repeated and (not pool.one_title or any(k in pool.repeated for k in file.numbers)):
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
        if _norm_title(file.title) not in self.known_titles:
            return None
        return distinct.pop()

    def _whole_readings(self, file: _LoneNumberFile, pool: _Pool) -> dict[_Fit, tuple[int | None, ...]]:
        """The file read under every numbering that fits the whole torrent's numbers.

        Nothing fits a torrent that repeats a number across more than one lifted title: the partition alone can
        read it.
        """

        if pool.repeated and not pool.one_title:
            return {}
        return self._readings(file, pool.numbers, self.windows)

    def _readings(
        self, file: _LoneNumberFile, numbers: Sequence[int], windows: Sequence[_Window]
    ) -> dict[_Fit, tuple[int | None, ...]]:
        """The file read under every numbering that reads each of `numbers` inside `windows`."""

        return {fit: tuple(self._read(fit, k) for k in file.numbers) for fit in self._fits(numbers, windows)}

    def _fits(self, numbers: Sequence[int], windows: Sequence[_Window]) -> list[_Fit]:
        """The numberings under which every number reads to a row inside `windows`.

        `SERIES_ROW` and `ABSOLUTE` fit together when TVDB's row for every number lies inside the windows: a group
        counting the series from one lands on TVDB's numbers wherever TVDB counts straight, and where it doesn't,
        the position check on each file tells. `WINDOW_ROW` and `WINDOW_SEASON` fit per window. `UNION_ROW` fits
        two or more windows when every number has a union row.
        """

        union = frozenset(ep_id for window in windows for ep_id in window.ids)
        fits: list[_Fit] = []
        if all((ep_id := self.series.id_by_absolute.get(k)) is not None and ep_id in union for k in numbers):
            fits.append(_Fit(Numbering.SERIES_ROW, None))
            fits.append(_Fit(Numbering.ABSOLUTE, None))
        for window in windows:
            for numbering in _RELATIVE:
                fit = _Fit(numbering, window)
                if all((ep_id := self._read(fit, k)) is not None and ep_id in window.ids for k in numbers):
                    fits.append(fit)
        if len(windows) > 1 and all(self._union_row(k) is not None for k in numbers):
            fits.append(_Fit(Numbering.UNION_ROW, None))
        return fits

    def _read(self, fit: _Fit, k: int) -> int | None:
        """The row number k names under `fit`, else None."""

        match fit.numbering:
            case Numbering.SERIES_ROW:
                return self._series_row(k)
            case Numbering.ABSOLUTE:
                return self.series.id_by_absolute.get(k)
            case Numbering.WINDOW_ROW:
                assert fit.window is not None
                return fit.window.row(k)
            case Numbering.WINDOW_SEASON:
                assert fit.window is not None
                return fit.window.season_row(k)
            case Numbering.UNION_ROW:
                return self._union_row(k)

    def _series_row(self, k: int) -> int | None:
        """The series' k-th regular row, when it carries absolute k.

        A row numbered otherwise means TVDB counts the series differently from its rows (a special in the count, a
        shuffled order), so the k-th row isn't what a group counting from one meant: the file is refused.
        """

        if not 1 <= k <= len(self.regs):
            return None
        ep_id = self.regs[k - 1]
        return ep_id if self.series.absolute_of.get(ep_id) == k else None

    def _union_row(self, k: int) -> int | None:
        """The k-th regular row across the torrent's windows."""

        return self.union_regs[k - 1] if 1 <= k <= len(self.union_regs) else None

    def _passes_window_gate(self, fit: _Fit) -> bool:
        """Whether a relative numbering's window, or the union, counts its rows in order.

        Rows whose absolutes don't increase are a season Sonarr orders differently from how it aired, so counting
        them from one lands wrong. The series-wide numberings aren't gated here: the position check is theirs.
        """

        if fit.numbering in _RELATIVE:
            assert fit.window is not None
            return fit.window.monotone
        return fit.numbering is not Numbering.UNION_ROW or self.union_monotone


def _lone_number_files(batch: PlacementBatch, facts: Mapping[str, NameFacts]) -> list[_LoneNumberFile]:
    """The pool: every file of the torrent named by number alone, seeded and gone names included.

    `facts` must cover every name of the torrent (`name_facts` does).
    """

    pool: list[_LoneNumberFile] = []
    for name in batch.torrent_names:
        info = batch.parsed.get(name)
        fact = facts[name]
        if info is not None and _lone_number(info) and not fact.extra and not fact.title_counted:
            pool.append(_LoneNumberFile(name, info.absolute_episode_numbers, info.series_title))
    return pool
