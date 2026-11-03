"""A torrent's file names read against the series' episode titles: which episode each name is titled as.

`TorrentNames` reads every name once, and `Placer.start` is its one reader. The passes read `TitleEvidence` and
`MIN_TITLE_HITS`. No I/O.
"""

from collections import Counter
from collections.abc import Collection, Mapping
from typing import NamedTuple, Self

from ..episode_titles import EpisodeTitles, TitleLead
from ..parse_reading import Reading
from ..placement_types import PlacementBatch, TargetScope
from ..release_names import NameRead, TitleGround, is_extras_name, read_names, whole_words

# Episode titles pick a run, or refuse one, only when at least this many members carry them.
MIN_TITLE_HITS = 2


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


class NameTitle(NamedTuple):
    """One name read on its own and against the episode titles."""

    read: NameRead
    """The name read on its own: stem, run membership, title words, numbers."""
    titled_as: int | None
    """The episode the name is titled as: its specific title, carried by no other file of the torrent, else None."""
    leaders: frozenset[int]
    """Every episode whose title starts the name's tail: run evidence, read for run members only."""
    extra: bool
    """Whether the name is an extra: set aside, and left out of every count."""


def _title_lead(episodes: EpisodeTitles, read: NameRead, resolved: Collection[int]) -> TitleLead:
    """Read one name against the episode titles: the tail behind its count, else the whole name.

    A counted name is read whole too, its season tag left out: "Show - 17 Steps 2" is titled "17 Steps 2". A name
    that each reading titles as a different episode is titled as neither. `resolved` are the episodes Sonarr read
    for the file (`EpisodeTitles.keyed_by`).
    """

    words = read.title_words
    keyed = episodes.keyed_by(read, resolved)
    if read.member is None and read.titled_count is None:
        return TitleLead(episodes.exactly(words, keyed), frozenset(), words)
    lead = episodes.leading(words)
    if lead.titled is None or read.titled_count is not None:
        whole = whole_words(read.stem.text)
        if (equal := episodes.exactly(whole, keyed)) is not None:
            if lead.titled is None:
                return TitleLead(equal, lead.leaders, whole)
            if lead.titled.ep_id != equal.ep_id:
                return lead._replace(titled=None)
    return lead


class TorrentNames(NamedTuple):
    """Every file name of the torrent, read once: stem, run membership, episode title, and whether it is an extra."""

    by_name: Mapping[str, NameTitle]
    """Each name read on its own and against the episode titles, in `PlacementBatch.torrent_names` order."""
    episode_titles: EpisodeTitles
    """The episode titles the names were read against."""

    @classmethod
    def of(cls, batch: PlacementBatch, scope: TargetScope, readings: Mapping[str, Reading]) -> Self:
        """Read every name, seeded and gone ones included.

        Two versions of one file carrying the same title still count as titled as it. A file whose own key Sonarr
        matched (`Reading.keyed`) is an episode, whatever extras word its name carries.
        """

        ground = TitleGround.of(scope.names)
        episode_titles = EpisodeTitles.of(scope.series.by_id, scope.series.id_by_key, ground.series_words)
        reads = read_names(batch.torrent_names, ground)
        leads = {name: _title_lead(episode_titles, read, readings[name].resolved) for name, read in reads.items()}
        carriers = Counter(words for _file, words in {(reads[name].file, lead.words) for name, lead in leads.items()})
        by_name: dict[str, NameTitle] = {}
        for name, lead in leads.items():
            titled = lead.titled
            titled_as = titled.ep_id if titled is not None and carriers[lead.words] == 1 else None
            # An extras word that belongs to the entry's titles or the name's episode title marks no extra.
            shield = ground.entry_words | frozenset(titled.words if titled is not None else ())
            extra = not readings[name].keyed and is_extras_name(name, shield)
            by_name[name] = NameTitle(reads[name], titled_as, lead.leaders, extra)
        return cls(by_name, episode_titles)
