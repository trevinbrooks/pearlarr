"""Unit tests for the grab "produce" side (:class:`GrabPipeline`).

Pin the add path - ``_add_one_url`` registering durable :class:`PendingImport`
records, ``add_torrent``'s cap bookkeeping, and ``_grab`` returning a pure
cap-reached bool (it never finalizes; the engine owns the single finalize site).
Built bare (``object.__new__`` via ``make_bare_instance``) so no live qBittorrent
login happens; the client ``add`` is faked by ``FakeTorrents``.
"""

from unittest import mock

from seadexarr.modules.config import Arr
from seadexarr.modules.grab_pipeline import GrabPipeline, GrabRequest
from seadexarr.modules.manual_import import ImportWaitMode
from seadexarr.modules.reporter import RunContext

from .builders import (
    CLIENT_SENTINEL,
    AddOutcome,
    FakeTorrents,
    make_grab_pipeline,
    one_release_dict,
    pending_import,
)


def _pipeline(
    *,
    torrents: FakeTorrents,
    mode: ImportWaitMode = ImportWaitMode.BLOCKING,
    qbit: object = CLIENT_SENTINEL,
    dry_run: bool = False,
    **config: object,
) -> GrabPipeline:
    """A bare ``GrabPipeline`` wired for the add path (a non-preview blocking run)."""

    return make_grab_pipeline(
        _torrents=torrents,
        qbit=qbit,
        _ctx=RunContext(arr=Arr.SONARR, dry_run=dry_run, import_wait_mode=mode),
        **config,
    )


def _pending(pipeline: GrabPipeline) -> dict:
    """The pipeline's durable per-arr pending store (what the engine reads back)."""

    return pipeline.cache_store.get_pending(Arr.SONARR)


class TestGrabReturnsPureBool:
    """_grab signals cap-reached as a bool and never finalizes itself.

    GrabPipeline holds no reference back to the engine, so "without finalizing" is
    now a structural property - the pipeline can't reach ``_finalize_run`` at all;
    the test pins the cap-reached return value the engine's single finalize site
    keys off.
    """

    def test_grab_at_cap_returns_true(self) -> None:
        pipeline = make_grab_pipeline(
            qbit=None,
            max_torrents_to_add=1,
            add_torrent=mock.MagicMock(return_value=(1, [])),
        )
        pipeline._ctx.torrents_added = 1  # already at the cap of 1

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

        assert pipeline._grab(req) is True


class TestAddOneUrlRegistersPending:
    """_add_one_url registers a PendingImport for a fresh AND an already-present
    torrent, but records a grab / counts toward the cap only for a fresh add."""

    def test_already_added_registers_pending_import(self) -> None:
        # The recommended release is already in qBittorrent (a prior run grabbed
        # it, still downloading): register it for the monitor, but don't count it
        # as a this-run grab.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        seeds = {"h1": pending_import(infohash="h1", series_id=7)}

        n_added, results = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert set(_pending(pipeline)) == {"h1"}
        assert [p.infohash for p in pipeline._ctx.pending_imports] == ["h1"]
        assert n_added == 0
        assert pipeline._ctx.torrents_added == 0
        assert pipeline._ctx.stats.added == []
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]

    def test_added_registers_and_counts(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ADDED, "Show-NAN0")})
        pipeline = _pipeline(torrents=torrents)
        seeds = {"h1": pending_import(infohash="h1")}

        n_added, _ = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert set(_pending(pipeline)) == {"h1"}
        assert [p.infohash for p in pipeline._ctx.pending_imports] == ["h1"]
        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1
        assert len(pipeline._ctx.stats.added) == 1

    def test_already_added_does_not_count_toward_cap(self) -> None:
        # One already-present + one fresh, cap 1: only the fresh add counts.
        torrents = FakeTorrents(
            {
                "already": (AddOutcome.ALREADY_ADDED, "old"),
                "fresh": (AddOutcome.ADDED, "new"),
            }
        )
        pipeline = _pipeline(torrents=torrents, max_torrents_to_add=1)
        seadex_dict = {
            **one_release_dict(srg="OLD", infohash="already", url="https://nyaa.si/view/1"),
            **one_release_dict(srg="NEW", infohash="fresh", url="https://nyaa.si/view/2"),
        }
        seeds = {
            "already": pending_import(infohash="already"),
            "fresh": pending_import(infohash="fresh"),
        }

        n_added, _ = pipeline.add_torrent(seadex_dict, pending_seeds=seeds)

        assert n_added == 1
        assert pipeline._ctx.torrents_added == 1

    def test_no_seed_does_not_register(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents)

        pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds={},
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == []

    def test_off_mode_does_not_register(self) -> None:
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents, mode=ImportWaitMode.OFF)
        seeds = {"h1": pending_import(infohash="h1")}

        pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == []

    def test_preview_does_not_register_but_returns_outcome(self) -> None:
        # No client -> preview: nothing persisted, but the outcome still surfaces.
        torrents = FakeTorrents({"h1": (AddOutcome.ALREADY_ADDED, "x")})
        pipeline = _pipeline(torrents=torrents, qbit=None)
        seeds = {"h1": pending_import(infohash="h1")}

        _, results = pipeline.add_torrent(
            one_release_dict(srg="NAN0", infohash="h1"),
            pending_seeds=seeds,
        )

        assert _pending(pipeline) == {}
        assert pipeline._ctx.pending_imports == []
        assert [r.outcome for r in results] == [AddOutcome.ALREADY_ADDED]
