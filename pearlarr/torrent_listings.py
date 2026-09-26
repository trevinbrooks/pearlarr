"""Reads every SeaDex entry of a series once per run and folds them into torrent listings and size identities."""

from collections.abc import Mapping

from .listing_fold import EntryListing, fold_listings
from .mappings import MappingEntry
from .placement_types import ListingsRead
from .seadex_gateway import SeaDexMiss, SeaDexSource
from .sonarr_episodes import SonarrEpisodes


class SeriesListings:
    """Reads every SeaDex entry of a series into a `ListingsRead`, once per series per run.

    The caller passes the series' AniList ids, with the ignore list already applied. Prefetched entries come from
    the gateway's cache, and anything else costs a request.
    """

    def __init__(self, seadex: SeaDexSource, episodes: SonarrEpisodes, ignore_tags: frozenset[str]) -> None:
        self._seadex = seadex
        self._episodes = episodes
        self._ignore_tags = ignore_tags
        self._reads: dict[int, ListingsRead | None] = {}

    def reset(self) -> None:
        """Forget this run's reads. Called at run start, alongside the episode cache reset."""

        self._reads = {}

    def read(self, series_id: int, entries: Mapping[int, MappingEntry]) -> ListingsRead | None:
        """The series' listings (see `fold_listings`), or None when SeaDex is unreachable.

        Only the first call in a run reads anything. Later calls return the same result, whatever `entries` they
        pass. An entry with an ambiguous AniDB id is skipped (its own run reports the error), and the identities
        still count as read.
        """

        if series_id not in self._reads:
            self._reads[series_id] = self._read(series_id, entries)
        return self._reads[series_id]

    def _read(self, series_id: int, entries: Mapping[int, MappingEntry]) -> ListingsRead | None:
        """Read each entry's record, window, and special aliases, then fold them."""

        listed: list[EntryListing] = []
        for al_id, mapping in entries.items():
            record = self._seadex.entry(al_id)
            if isinstance(record, SeaDexMiss):
                if record is SeaDexMiss.OUTAGE:
                    return None
                continue
            try:
                ep_list = self._episodes.get_ep_list(series_id, al_id, mapping)
            except ValueError:
                continue
            window = None if ep_list is None else frozenset(ep.id for ep in ep_list if ep.id)
            listed.append(EntryListing(record, window, mapping.special_aliasing))
        return fold_listings(listed, self._ignore_tags)
