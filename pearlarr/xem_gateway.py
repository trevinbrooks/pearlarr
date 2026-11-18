"""TheXEM scene mapping gateway: which series' mappings to read again, and the stored mappings per series."""

from collections.abc import Sequence
from datetime import datetime, timedelta

from .cache import AbstractCacheStore
from .progress import ProgressCallback
from .scene_numbers import CachedSceneMappings, SceneMapping
from .stamps import parse_stamp_or_none, stamp_of
from .tvdb_orders import SonarrSeriesOrders
from .xem_client import AbstractXemClient, XemNoReply

FRESH_DAYS = 7
"""How long a series' stored mappings stay fresh.

TheXEM adds mappings as new episodes are released and corrects old ones, and its reply doesn't say whether the
series is still airing. So every series is read again as often as a series that may still change has its TVDB
orders read (`skyhook_gateway.ACTIVE_FRESH_DAYS`), which keeps its mappings no staler than the episodes they number.
"""


class XemGateway:
    """The scene mapping cache over the TheXEM client: reads due series again, serves stored mappings at any age.

    Only a series Sonarr numbers by another TVDB order is ever asked for: Sonarr holds the scene numbers of the
    rest.
    """

    def __init__(self, *, client: AbstractXemClient, cache_store: AbstractCacheStore) -> None:
        """Bind the client and the cache (network-free). Stored mappings read are kept for the run."""

        self._client = client
        self._cache = cache_store
        self._stored: dict[int, CachedSceneMappings | None] = {}

    @property
    def unreachable(self) -> bool:
        """True once TheXEM has been marked unreachable for this run."""

        return self._client.unreachable

    def prefetch(
        self,
        series: Sequence[SonarrSeriesOrders],
        *,
        dry_run: bool,
        progress: ProgressCallback | None = None,
    ) -> int:
        """Read the mappings of every due series in another order, one at a time. Returns how many were written.

        Each series' mappings are saved as they are written, unless `dry_run`, so an interrupted run keeps what it
        read. A series whose reply can't be used keeps what is stored for it, and so does every series left when
        TheXEM is unreachable. Such a series is due again on the next run.
        """

        now = datetime.now()
        due = [facts.tvdb_id for facts in series if facts.in_another_order and self._due(facts.tvdb_id, now)]
        written = 0
        for done, tvdb_id in enumerate(due, start=1):
            reply = self._client.mappings(tvdb_id)
            if reply is XemNoReply.UNREACHABLE:
                break
            if reply is not XemNoReply.UNUSABLE:
                stored = CachedSceneMappings(fresh_until=stamp_of(now + timedelta(days=FRESH_DAYS)), mappings=reply)
                self._cache.put_scene_mappings(tvdb_id, stored)
                self._stored[tvdb_id] = stored
                self._cache.save(dry_run=dry_run)
                written += 1
            if progress is not None:
                progress.progress(done / len(due), f"{done}/{len(due)}")
        return written

    def scene_mappings(self, tvdb_id: int) -> tuple[SceneMapping, ...] | None:
        """The series' stored mappings at any age, or None when they have never been read.

        Empty for a series TheXEM has no mappings for.
        """

        stored = self._stored_mappings(tvdb_id)
        return None if stored is None else stored.mappings

    def _stored_mappings(self, tvdb_id: int) -> CachedSceneMappings | None:
        """The series' stored record, read from the cache once per run."""

        if tvdb_id not in self._stored:
            self._stored[tvdb_id] = self._cache.get_scene_mappings(tvdb_id)
        return self._stored[tvdb_id]

    def _due(self, tvdb_id: int, now: datetime) -> bool:
        """Whether the series needs a read: nothing is stored for it, or what is stored is past its `fresh_until`."""

        stored = self._stored_mappings(tvdb_id)
        if stored is None:
            return True
        fresh_until = parse_stamp_or_none(stored.fresh_until)
        return fresh_until is None or fresh_until <= now
