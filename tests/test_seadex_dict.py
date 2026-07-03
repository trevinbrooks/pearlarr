# pyright: strict
"""Characterization tests for ``SeadexReleaseFilter.build`` (entry -> release dict).

This is the SeaDexGateway logic (the engine reaches it via ``get_seadex_dict``):
tracker filtering, the want_best / prefer_dual_audio narrowing, the is_public
computation, and the public_only per-group private-url drop.
"""

import logging
from datetime import datetime

import pytest
from seadex import EntryRecord, Tag, TorrentRecord, Tracker

from seadexarr.modules import seadex_filter

from .builders import make_entry_record, make_release_filter, rg_group
from .fakes import CaptureHandler


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
        filt = make_release_filter(private_releases="allow", want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="u1", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert result["A"].urls["u1"].is_public is False

    def test_public_only_drops_private_url_when_public_exists(self) -> None:
        filt = make_release_filter(private_releases="warn", want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="pub", tracker=Tracker.NYAA),
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES),
        )
        result = filt.build(entry)
        assert set(result["A"].urls) == {"pub"}

    def test_public_only_keeps_private_only_group(self) -> None:
        # A group with no public option is kept here; it's only dropped later in
        # reduce_overlapping_downloads if the Arr already has a match.
        filt = make_release_filter(private_releases="warn", want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES),
        )
        result = filt.build(entry)
        assert set(result["A"].urls) == {"priv"}


class TestPrivateFallback:
    """``private_releases: fallback`` - the public-alternative pool added by ``build``."""

    def test_private_only_preferred_adds_public_fallback(self) -> None:
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="Alt", url="pub", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        # The private preferred pick stays (for the already-have check); the best
        # public alternative rides along, marked as the fallback.
        assert set(result) == {"Best", "Alt"}
        assert result["Best"].urls["priv"].is_fallback is False
        assert result["Alt"].urls["pub"].is_fallback is True

    def test_no_fallback_added_when_preferred_has_public(self) -> None:
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="pub", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="Alt", url="pub2", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"Best"}

    def test_mixed_preferred_adds_fallback_for_private_only_group(self) -> None:
        # A private-only preferred group triggers the pool even when another
        # preferred group is public (the gate is per-group, not per-entry); the
        # public preferred pick is NOT marked as a fallback.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="PrivBest", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="PubBest", url="pub", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="Alt", url="pub2", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert set(result) == {"PrivBest", "PubBest", "Alt"}
        assert result["PubBest"].urls["pub"].is_fallback is False
        assert result["Alt"].urls["pub2"].is_fallback is True

    def test_same_group_public_copy_survives_as_fallback(self) -> None:
        # One group with a private preferred url and a public non-best url: the
        # pool re-adds the group's public copy, then the per-group drop removes
        # the private url, leaving a public group marked as the fallback.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="A", url="pub", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert set(result) == {"A"}
        assert set(result["A"].urls) == {"pub"}
        assert result["A"].urls["pub"].is_fallback is True

    def test_no_fallback_added_when_nothing_public(self) -> None:
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
        )
        result = filt.build(entry)
        assert set(result) == {"Best"}
        assert result["Best"].urls["priv"].is_fallback is False

    def test_warn_mode_does_not_add_fallback(self) -> None:
        # The default: a private-only preferred pick stays alone, so the planner
        # warns and holds the title exactly as before.
        filt = make_release_filter(private_releases="warn", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="Alt", url="pub", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"Best"}

    def test_fallback_pool_applies_the_preference_cascade(self) -> None:
        # The public pool is narrowed by the same want_best/audio preferences.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=True)
        entry = _entry(
            _torrent(release_group="Best", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True, is_dual_audio=True),
            _torrent(release_group="AltDual", url="pub1", tracker=Tracker.NYAA, is_dual_audio=True),
            _torrent(release_group="AltSingle", url="pub2", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"Best", "AltDual"}


class TestInteractivePick:
    def test_tolerates_non_numeric_input(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A non-numeric pick (a typo) must not crash the entry with an uncaught
        # ValueError; the bad token is skipped with a warning, so nothing is selected.
        filt = make_release_filter()
        seadex_dict = {"GroupA": rg_group({})}
        sd_entry = make_entry_record()

        def fake_input(prompt: str = "") -> str:
            del prompt
            return "x"

        monkeypatch.setattr("builtins.input", fake_input)
        handler = CaptureHandler()
        filt.logger.addHandler(handler)
        filt.logger.setLevel(logging.WARNING)
        try:
            result = filt.interactive_pick(seadex_dict, sd_entry)
        finally:
            filt.logger.removeHandler(handler)
        assert result == {}
        # The prompt rows are printed, not logged, so they stay visible even at
        # log_level WARNING (a demoted INFO row would vanish and leave input() blind).
        assert "[0]: GroupA" in capsys.readouterr().out
        # Only genuine problems warn (LogCounter tallies WARNINGs into the run's
        # issues summary): the invalid token, then the resulting empty pick.
        warnings = [r.getMessage() for r in handler.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 2
        assert "invalid selection" in warnings[0]
        assert "No valid selection" in warnings[1]
