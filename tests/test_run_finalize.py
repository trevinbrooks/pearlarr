"""Guards the single end-of-run finalize site.

When ``max_torrents_to_add`` is reached mid-run, ``_grab`` returns a pure bool
(it no longer finalizes); ``run_sync`` breaks the per-item scan and runs the ONE
post-loop ``_finalize_run`` site - the same site the normal end-of-run path
reaches. These pin both halves of that hoist so a future change can't silently
double-finalize or skip the blocking/import pass on the cap-reached break.
"""

from types import SimpleNamespace
from typing import Any
from unittest import mock

from seadexarr.modules.config import Arr
from seadexarr.modules.reporter import RunContext
from seadexarr.modules.seadex_arr import GrabRequest, SeaDexArr

from .builders import make_bare_instance, make_config, make_logger


def _item(title: str) -> SimpleNamespace:
    """A minimal Arr item exposing only what the run loop reads."""

    return SimpleNamespace(title=title, monitored=True, id=1)


class TestCapReachedFinalizesOnce:
    """A mid-run cap stops the scan and finalizes exactly once, at the single site."""

    def test_cap_reached_breaks_loop_and_finalizes_once(self) -> None:
        finalize = mock.MagicMock()
        strategy: Any = mock.MagicMock()
        strategy.get_items.return_value = [_item("A"), _item("B")]
        strategy.item_anilist_ids.return_value = {1: object()}
        strategy.warms_episodes = False
        strategy.prefetch_episodes.return_value = 0
        strategy.pending_import_series_id.return_value = None
        # Cap reached on the first id: process_al_id returns True (stop the run).
        strategy.process_al_id.return_value = True

        anilist = mock.MagicMock()
        anilist.prefetch.return_value = 0
        seadex = mock.MagicMock()
        seadex.prefetch.return_value = 0

        engine = make_bare_instance(
            SeaDexArr,
            qbit=None,
            logger=make_logger(),
            _config=make_config(),
            _arr_config=mock.MagicMock(),
            _anilist=anilist,
            _seadex=seadex,
            _reporter=mock.MagicMock(),
            _filter=mock.MagicMock(),  # begin_run binds it; a stub absorbs the no-op
            _finalize_run=finalize,
        )

        result = engine.run_sync(strategy, arr=Arr.SONARR, item_id=None, dry_run=True)

        assert result is True
        # The cap stopped the scan after the first id: the second item is never reached.
        strategy.process_al_id.assert_called_once()
        # ...and the single post-loop finalize ran exactly once (reads ctx.arr now).
        finalize.assert_called_once_with()


class TestGrabReturnsPureBool:
    """_grab signals cap-reached as a bool and never finalizes itself."""

    def test_grab_at_cap_returns_true_without_finalizing(self) -> None:
        finalize = mock.MagicMock()
        notifier = mock.MagicMock()
        notifier.enabled = False

        engine = make_bare_instance(
            SeaDexArr,
            qbit=None,
            logger=make_logger(),
            _config=make_config(max_torrents_to_add=1),
            _ctx=RunContext(arr=Arr.SONARR),
            _anilist=mock.MagicMock(),
            _notifier=notifier,
            _reporter=mock.MagicMock(),
            add_torrent=mock.MagicMock(return_value=(1, [])),
            _finalize_run=finalize,
        )
        engine._ctx.torrents_added = 1  # already at the cap of 1

        req = GrabRequest(
            al_id=1,
            item_title="Show",
            anilist_title="Show",
            sd_url="https://seadex.example/1",
            seadex_dict={},
            torrent_hashes=[],
            cache_details={},
            release_group=None,
        )

        assert engine._grab(req) is True
        finalize.assert_not_called()
