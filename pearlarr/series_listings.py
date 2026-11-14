"""Reads every SeaDex entry of a series once per run and turns them into torrent listings and episodes by size."""

from collections.abc import Callable

from .listing_builder import ListedTitle, build_listings
from .mappings import ResolvedMappings
from .pending_import import TitleNames, TitleWindow
from .placement_types import FetchedListings
from .seadex_gateway import SeaDexNoEntry, SeaDexSource
from .sonarr_episodes import SonarrEpisodes


class SeriesListings:
    """Reads every SeaDex entry of a series into a `FetchedListings`, once per series per run.

    The caller passes the series' resolved mappings, the ignored titles set apart: an ignored title's entry still
    lists its files, and its window is flagged so `ordered_windows` puts it after the kept windows of the same kind
    (season or specials).
    Prefetched entries come from the gateway's cache. Anything else, an ignored title's entry among them, costs a
    request.

    Each collaborator serves `_read`: `seadex` gives each title's entry, `episodes` its window and AniList episode
    count, `ignore_tags` the torrents the episodes by size leave out, and `names` the title's names for its window
    (the ones the title's check uses). All four hold for the run, so they come in at construction.
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
        """Forget this run's listings. Called at run start, alongside the episode cache reset."""

        self._reads = {}

    def read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings | None:
        """The series' listings (see `build_listings`), or None when SeaDex is unreachable.

        Only the first call in a run reads anything. Later calls return the same result, whatever `resolved` they
        pass. A title with an ambiguous AniDB id is left out (its check reports the error), and the episodes by
        size still count as read.
        """

        if series_id not in self._reads:
            self._reads[series_id] = self._read(series_id, resolved)
        return self._reads[series_id]

    def _read(self, series_id: int, resolved: ResolvedMappings) -> FetchedListings | None:
        """Read each title's entry, window, and special aliases, then build the listings from them.

        A title whose episode list comes back empty keeps a window with no ids.
        """

        listed: list[ListedTitle] = []
        for anilist_id, mapping in resolved.kept_and_ignored.items():
            entry = self._seadex.entry(anilist_id)
            if isinstance(entry, SeaDexNoEntry):
                if entry is SeaDexNoEntry.UNREACHABLE:
                    return None
                continue
            try:
                title_episodes = self._episodes.mapping_episodes(series_id, anilist_id, mapping)
            except ValueError:
                continue
            window = (
                None
                if title_episodes is None
                else TitleWindow.of(
                    anilist_id=anilist_id,
                    series_id=series_id,
                    episodes=title_episodes,
                    names=self._names(anilist_id),
                    ignored=anilist_id in resolved.ignored,
                    episode_count=self._episodes.episode_count(anilist_id),
                )
            )
            listed.append(ListedTitle(entry, window, mapping.special_aliasing))
        return build_listings(listed, self._ignore_tags)
