"""A torrent's file names read against the series' episode titles: which episode's title each name carries.

`TorrentNames` reads every name once, and `PlacerState.start` is its one caller. No I/O.
"""

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Self

from ..episode_titles import EpisodeTitles, TitleLead
from ..file_names import NameParts, TitleNameWords, is_extras_name, split_names, whole_words
from ..parsed_episodes import ParsedEpisodes
from ..placement_types import PlacementBatch, PlacementTarget


@dataclass(frozen=True, slots=True)
class NameTitle:
    """One name read on its own and against the episode titles."""

    read: NameParts
    """The name read on its own: stem, sequence membership, title words, numbers."""
    episode_by_title: int | None
    """The episode whose distinctive title the name carries, when no other file of the torrent carries it, else
    None."""
    leading_title_ids: frozenset[int]
    """Every episode whose title starts the words after the name's number (`EpisodeTitleMatches`), read for sequence
    members only."""
    extra: bool
    """Whether the name is an extra: set aside, and left out of every count."""


def _title_lead(episodes: EpisodeTitles, read: NameParts, resolved: Collection[int]) -> TitleLead:
    """Read one name against the episode titles: the words after its number, else the whole name.

    A numbered name is read whole too, its season tag left out: "Show - 17 Steps 2" carries the title "17 Steps 2".
    When the words after the number and the whole name match different episodes, the name carries neither title.
    `resolved` are the episodes Sonarr parsed the file as (`EpisodeTitles.episodes_numbered_by`).
    """

    words = read.title_words
    numbered = episodes.episodes_numbered_by(read, resolved)
    if read.member is None and read.episode_title_number is None:
        return TitleLead(episodes.exact_match(words, numbered), frozenset(), words)
    lead = episodes.leading_title(words)
    if lead.title is None or read.episode_title_number is not None:
        whole = whole_words(read.stem.text)
        if (equal := episodes.exact_match(whole, numbered)) is not None:
            if lead.title is None:
                return TitleLead(equal, lead.leading_title_ids, whole)
            if lead.title.ep_id != equal.ep_id:
                return replace(lead, title=None)
    return lead


@dataclass(frozen=True, slots=True)
class TorrentNames:
    """Every name of the torrent, read once: stem, sequence membership, episode title, and whether it is an extra."""

    by_name: Mapping[str, NameTitle]
    """Each name read on its own and against the episode titles, in `PlacementBatch.all_names` order."""
    episode_titles: EpisodeTitles
    """The episode titles the names were read against."""

    @classmethod
    def of(cls, batch: PlacementBatch, target: PlacementTarget, parsed_by_name: Mapping[str, ParsedEpisodes]) -> Self:
        """Read every name, file map and gone ones included.

        Two versions of one file carrying the same title still count as carrying it. A file whose parsed season and
        episode Sonarr's match agrees with (`ParsedEpisodes.key_confirmed`) is an episode, whatever extras word its
        name carries.
        """

        ground = TitleNameWords.of(target.title_names)
        episode_titles = target.index.titles(ground.series_words)
        reads = split_names(batch.all_names, ground)
        leads = {name: _title_lead(episode_titles, read, parsed_by_name[name].ids) for name, read in reads.items()}
        carriers = Counter(
            words for _file, words in {(reads[name].identity, lead.words) for name, lead in leads.items()}
        )
        by_name: dict[str, NameTitle] = {}
        for name, lead in leads.items():
            titled = lead.title
            episode_by_title = titled.ep_id if titled is not None and carriers[lead.words] == 1 else None
            # An extras word that belongs to the title's names or the name's episode title marks no extra.
            shield = ground.all_name_words | frozenset(titled.words if titled is not None else ())
            extra = not parsed_by_name[name].key_confirmed and is_extras_name(name, shield)
            by_name[name] = NameTitle(reads[name], episode_by_title, lead.leading_title_ids, extra)
        return cls(by_name, episode_titles)
