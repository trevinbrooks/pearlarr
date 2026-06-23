"""Characterization tests for ``get_seadex_dict`` (SeaDex entry -> release dict).

This is the SeaDexGateway logic extracted in Phase 3: tracker filtering, the
want_best / prefer_dual_audio narrowing, the is_public computation, and the
public_only per-group private-url drop.
"""

from typing import Any

from tests.builders import FakeEntry, FakeTorrent, FakeTracker, make_arr


def _entry(*torrents: FakeTorrent) -> Any:
    # Returns Any so the duck-typed FakeEntry satisfies get_seadex_dict's
    # EntryRecord parameter without a per-call cast.
    return FakeEntry(list(torrents))


class TestGetSeadexDict:
    def test_filters_out_unselected_trackers(self) -> None:
        arr = make_arr(trackers={"nyaa"}, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            FakeTorrent(release_group="A", url="u1", tracker=FakeTracker("Nyaa", True)),
            FakeTorrent(release_group="B", url="u2", tracker=FakeTracker("AnimeTosho", True)),
        )
        assert set(arr.get_seadex_dict(entry)) == {"A"}

    def test_want_best_narrows_to_best(self) -> None:
        arr = make_arr(want_best=True, prefer_dual_audio=False)
        entry = _entry(
            FakeTorrent(release_group="Best", url="u1", tracker=FakeTracker("Nyaa", True), is_best=True),
            FakeTorrent(release_group="Rest", url="u2", tracker=FakeTracker("Nyaa", True), is_best=False),
        )
        assert set(arr.get_seadex_dict(entry)) == {"Best"}

    def test_prefer_dual_audio_narrows_when_present(self) -> None:
        arr = make_arr(want_best=False, prefer_dual_audio=True)
        entry = _entry(
            FakeTorrent(release_group="Dual", url="u1", tracker=FakeTracker("Nyaa", True), is_dual_audio=True),
            FakeTorrent(release_group="Single", url="u2", tracker=FakeTracker("Nyaa", True), is_dual_audio=False),
        )
        assert set(arr.get_seadex_dict(entry)) == {"Dual"}

    def test_prefer_non_dual_when_flag_false(self) -> None:
        arr = make_arr(want_best=False, prefer_dual_audio=False)
        entry = _entry(
            FakeTorrent(release_group="Dual", url="u1", tracker=FakeTracker("Nyaa", True), is_dual_audio=True),
            FakeTorrent(release_group="Single", url="u2", tracker=FakeTracker("Nyaa", True), is_dual_audio=False),
        )
        assert set(arr.get_seadex_dict(entry)) == {"Single"}

    def test_is_public_false_for_private_tracker_name(self) -> None:
        # Even when the tracker claims public, a name in PRIVATE_TRACKERS is not public
        arr = make_arr(public_only=False, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            FakeTorrent(release_group="A", url="u1", tracker=FakeTracker("AB", True)),
        )
        result = arr.get_seadex_dict(entry)
        assert result["A"].urls["u1"].is_public is False

    def test_public_only_drops_private_url_when_public_exists(self) -> None:
        arr = make_arr(public_only=True, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            FakeTorrent(release_group="A", url="pub", tracker=FakeTracker("Nyaa", True)),
            FakeTorrent(release_group="A", url="priv", tracker=FakeTracker("AB", False)),
        )
        result = arr.get_seadex_dict(entry)
        assert set(result["A"].urls) == {"pub"}

    def test_public_only_keeps_private_only_group(self) -> None:
        # A group with no public option is kept here; it's only dropped later in
        # reduce_overlapping_downloads if the Arr already has a match.
        arr = make_arr(public_only=True, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            FakeTorrent(release_group="A", url="priv", tracker=FakeTracker("AB", False)),
        )
        result = arr.get_seadex_dict(entry)
        assert set(result["A"].urls) == {"priv"}
