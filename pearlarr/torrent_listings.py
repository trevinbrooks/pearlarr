"""Reads every SeaDex entry of a series once per run and folds them into torrent listings and size identities."""

from collections.abc import Callable

from .listing_fold import EntryListing, fold_listings
from .mappings import ResolvedMappings
from .pending_import import TitleNames, TitleWindow
from .placement_types import FetchedListings
from .seadex_gateway import SeaDexMiss, SeaDexSource
from .sonarr_episodes import SonarrEpisodes


class SeriesListings:
    """Reads every SeaDex entry of a series into a `ListingsRead`, once per series per run.

    The caller passes the series' resolved mappings, the ignored entries set apart: an ignored entry still lists
    its own files, and its window is flagged so `ordered_windows` runs it after the kept windows of the same kind
    (season or specials).
    Prefetched entries come from the gateway's cache. Anything else, an ignored entry among them, costs a request.

    Each collaborator serves `_read`: `seadex` gives each entry's record, `episodes` its window and AniList episode
    count, `ignore_tags` the torrents the size identities skip, and `names` the entry's titles for its window (the
    title ladder the entry's own run uses). All four hold for the run, so they come in at construction.
    """

    def __init__(
        self,
        seadex: SeaDexSource,
        episodes: SonarrEpisodes,
        ignore_tags: frozenset[str],
        names: Callable[[int], TitleNames],
    ) -> None:
        self._seadex = seadex
        self._episodes = episodes
        self._ignore_tags = ignore_tags
        self._names = names
        self._reads: dict[int, FetchedListings | None] = {}

    def reset(self) -> None:
        """Forget this run's reads. Called at run start, alongside the episode cache reset."""

        self._reads = {}

    def read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings | None:
        """The series' listings (see `fold_listings`), or None when SeaDex is unreachable.

        Only the first call in a run reads anything. Later calls return the same result, whatever `resolved` they
        pass. An entry with an ambiguous AniDB id is skipped (its own run reports the error), and the identities
        still count as read.
        """

        if series_id not in self._reads:
            self._reads[series_id] = self._read(series_id, resolved)
        return self._reads[series_id]

    def _read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings | None:
        """Read each entry's record, window, and special aliases, then fold them.

        An entry whose episode list comes back empty keeps a window with no ids.
        """

        listed: list[EntryListing] = []
        for al_id, mapping in resolved.every_entry.items():
            record = self._seadex.entry(al_id)
            if isinstance(record, SeaDexMiss):
                if record is SeaDexMiss.UNREACHABLE:
                    return None
                continue
            try:
                ep_list = self._episodes.mapping_episodes(series_id, al_id, mapping)
            except ValueError:
                continue
            window = (
                None
                if ep_list is None
                else TitleWindow.of(
                    al_id=al_id,
                    series_id=series_id,
                    ep_list=ep_list,
                    names=self._names(al_id),
                    ignored=al_id in resolved.ignored,
                    episode_count=self._episodes.episode_count(al_id),
                )
            )
            listed.append(EntryListing(record, window, mapping.special_aliasing))
        return fold_listings(listed, self._ignore_tags)
