"""The listing a numbered specials pack is judged by: the specials every entry of one series lists it under."""

from collections.abc import Iterable, Mapping

from .mappings import MappingEntry
from .placement_types import TorrentListings
from .seadex_gateway import SeaDexMiss, SeaDexSource
from .seadex_types import normalized_infohash
from .sonarr_episodes import SonarrEpisodes


class SeriesListings:
    """Reads, per torrent, the ids of every window of every entry of a series listing it (`TargetScope.listed`).

    The entries are the series' resolved ids as the run resolves them everywhere (the ignore list applied). A
    prefetched id is served from the gateway's cache, and one the run never prefetched costs a request.
    """

    def __init__(self, seadex: SeaDexSource, episodes: SonarrEpisodes) -> None:
        self._seadex = seadex
        self._episodes = episodes

    def read(self, series_id: int, entries: Mapping[int, MappingEntry], hashes: Iterable[str]) -> TorrentListings:
        """Each hash to the union of the windows listing it, None where a listing entry's record or window is unread.

        A SeaDex outage leaves every hash unread, an unread series list the hashes that entry lists. An entry
        whose window cannot be resolved (an ambiguous AniDB id) lists nothing, as one listing none of the hashes does.
        """

        wanted = frozenset(hashes)
        if not wanted:
            return {}
        listed: dict[str, frozenset[int] | None] = {infohash: frozenset() for infohash in wanted}
        for al_id, mapping in entries.items():
            record = self._seadex.entry(al_id)
            if isinstance(record, SeaDexMiss):
                if record is SeaDexMiss.OUTAGE:
                    return dict.fromkeys(wanted)
                continue
            here = wanted & {normalized_infohash(torrent.infohash) for torrent in record.torrents}
            if not here:
                continue
            try:
                ep_list = self._episodes.get_ep_list(series_id, al_id, mapping)
            except ValueError:
                continue
            window = None if ep_list is None else frozenset(ep.id for ep in ep_list if ep.id)
            for infohash in here:
                union = listed[infohash]
                listed[infohash] = None if window is None or union is None else union | window
        return listed
