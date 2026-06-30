# pyright: strict
"""Characterization tests for ``SeadexReleaseFilter.build`` (entry -> release dict).

This is the SeaDexGateway logic (the engine reaches it via ``get_seadex_dict``):
tracker filtering, the want_best / prefer_dual_audio narrowing, the is_public
computation, and the public_only per-group private-url drop.
"""

from datetime import datetime

import pytest
from seadex import EntryRecord, Tag, TorrentRecord, Tracker

from seadexarr.modules import seadex_filter

from .builders import make_entry_record, make_release_filter, rg_group


def _torrent(
    *,
    release_group: str,
    url: str,
    tracker: Tracker,
    tags: frozenset[Tag] = frozenset(),
    is_best: bool = False,
    is_dual_audio: bool = False,
    infohash: str | None = "hash",
) -> TorrentRecord:
    """A real ``seadex.TorrentRecord`` carrying only the fields ``build`` reads.

    The library type is a frozen ``msgspec.Struct`` (no ``make_torrent_record``
    builder exists), so this defaults the boilerplate fields and exposes the
    handful ``build`` varies; ``files`` stays empty since no test exercises them.
    """

    stamp = datetime(2026, 1, 1)
    return TorrentRecord(
        collection_id="col",
        collection_name="col-name",
        created_at=stamp,
        is_dual_audio=is_dual_audio,
        files=(),
        id="t1",
        infohash=infohash,
        is_best=is_best,
        release_group=release_group,
        tags=tags,
        tracker=tracker,
        updated_at=stamp,
        url=url,
        size=0,
    )


def _entry(*torrents: TorrentRecord) -> EntryRecord:
    """A real ``EntryRecord`` wrapping the given torrents (replaces the duck-typed fake)."""

    return make_entry_record(torrents=torrents)


class TestGetSeadexDict:
    def test_filters_out_unselected_trackers(self) -> None:
        filt = make_release_filter(trackers={"nyaa"}, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="u1", tracker=Tracker.NYAA),
            _torrent(release_group="B", url="u2", tracker=Tracker.ANIMETOSHO),
        )
        assert set(filt.build(entry)) == {"A"}

    def test_ignore_tags_match_is_case_insensitive(self) -> None:
        # The seadex Tag is a str-enum whose value is canonical-case ("Dolby Vision");
        # a natural-case config rule ("dolby vision") must still filter that release.
        filt = make_release_filter(
            ignore_tags=["dolby vision"],
            trackers={"nyaa"},
            want_best=False,
            prefer_dual_audio=False,
        )
        entry = _entry(
            _torrent(release_group="HDR", url="u1", tracker=Tracker.NYAA, tags=frozenset({Tag.DOLBY_VISION})),
            _torrent(release_group="SDR", url="u2", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"SDR"}

    def test_want_best_narrows_to_best(self) -> None:
        filt = make_release_filter(want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="u1", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="Rest", url="u2", tracker=Tracker.NYAA, is_best=False),
        )
        assert set(filt.build(entry)) == {"Best"}

    def test_prefer_dual_audio_narrows_when_present(self) -> None:
        filt = make_release_filter(want_best=False, prefer_dual_audio=True)
        entry = _entry(
            _torrent(release_group="Dual", url="u1", tracker=Tracker.NYAA, is_dual_audio=True),
            _torrent(release_group="Single", url="u2", tracker=Tracker.NYAA, is_dual_audio=False),
        )
        assert set(filt.build(entry)) == {"Dual"}

    def test_prefer_non_dual_when_flag_false(self) -> None:
        filt = make_release_filter(want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Dual", url="u1", tracker=Tracker.NYAA, is_dual_audio=True),
            _torrent(release_group="Single", url="u2", tracker=Tracker.NYAA, is_dual_audio=False),
        )
        assert set(filt.build(entry)) == {"Single"}

    def test_is_public_false_for_private_tracker_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even when the tracker claims public, a name in PRIVATE_TRACKERS is not public.
        # NYAA.is_public() is True, so pin "nyaa" into PRIVATE_TRACKERS to exercise the
        # second conjunct (a real private tracker is_public()==False already, which would
        # short-circuit before the membership check ever ran).
        monkeypatch.setattr(seadex_filter, "PRIVATE_TRACKERS", {"nyaa"})
        filt = make_release_filter(public_only=False, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="u1", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert result["A"].urls["u1"].is_public is False

    def test_public_only_drops_private_url_when_public_exists(self) -> None:
        filt = make_release_filter(public_only=True, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="pub", tracker=Tracker.NYAA),
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES),
        )
        result = filt.build(entry)
        assert set(result["A"].urls) == {"pub"}

    def test_public_only_keeps_private_only_group(self) -> None:
        # A group with no public option is kept here; it's only dropped later in
        # reduce_overlapping_downloads if the Arr already has a match.
        filt = make_release_filter(public_only=True, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES),
        )
        result = filt.build(entry)
        assert set(result["A"].urls) == {"priv"}


class TestInteractivePick:
    def test_tolerates_non_numeric_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A non-numeric pick (a typo) must not crash the entry with an uncaught
        # ValueError; the bad token is skipped with a warning, so nothing is selected.
        filt = make_release_filter()
        seadex_dict = {"GroupA": rg_group({})}
        sd_entry = make_entry_record()

        def fake_input(prompt: str = "") -> str:
            del prompt
            return "x"

        monkeypatch.setattr("builtins.input", fake_input)
        result = filt.interactive_pick(seadex_dict, sd_entry)
        assert result == {}
