# pyright: strict
"""Characterization tests for `SeadexReleaseFilter.build` (entry -> release dict).

This is the SeadexReleaseFilter logic (the arr run reaches it via `get_seadex_dict`):
tracker filtering, the prefer_dual_audio / want_best narrowing, the is_public
computation, and the per-group private-release drop.
"""

from collections import Counter
from datetime import datetime

import pytest
from seadex import EntryRecord, Tag, TorrentRecord, Tracker

from pearlarr import seadex_filter
from pearlarr.arr_run_services import ArrRunServices
from pearlarr.config import Arr
from pearlarr.mappings import TitleMapping
from pearlarr.output import Diagnostic, Severity, install_hub
from pearlarr.radarr_strategy import RadarrStrategy

from .builders import (
    FakeCacheStore,
    FakeSeaDexSource,
    make_config,
    make_entry_record,
    make_release_filter,
    make_run_deps,
    make_torrent_record,
    rg_group,
)
from .fakes import FakeRadarrClient, RecordingHub


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
    """A real `seadex.TorrentRecord` carrying only the fields `build` reads.

    The library type is a frozen `msgspec.Struct` (no `make_torrent_record`
    builder exists), so this defaults the boilerplate fields and exposes the
    handful `build` varies. `files` stays empty since no test exercises them.
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
    """A real `EntryRecord` wrapping the given torrents (replaces the duck-typed fake)."""

    return make_entry_record(torrents=torrents)


class TestGetSeadexDict:
    """`make_release_filter`'s `build` narrows candidates by tracker, tags, want_best, and dual-audio.

    A private release is dropped only when a public copy of the same group survives.
    """

    def test_filters_out_unselected_trackers(self) -> None:
        filt = make_release_filter(trackers={"nyaa"}, want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="u1", tracker=Tracker.NYAA),
            _torrent(release_group="B", url="u2", tracker=Tracker.ANIMETOSHO),
        )
        assert set(filt.build(entry)) == {"A"}

    def test_ignore_tags_match_is_case_insensitive(self) -> None:
        # The seadex Tag is a str-enum whose value is canonical-case ("Dolby Vision").
        # A natural-case config rule ("dolby vision") must still filter that release.
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

    def test_dual_alternative_outranks_single_audio_best(self) -> None:
        filt = make_release_filter(want_best=True, prefer_dual_audio=True)
        entry = _entry(
            _torrent(release_group="Best", url="u1", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="DualAlt", url="u2", tracker=Tracker.NYAA, is_dual_audio=True),
        )
        assert set(filt.build(entry)) == {"DualAlt"}

    def test_single_alternative_outranks_dual_best_when_flag_false(self) -> None:
        filt = make_release_filter(want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="DualBest", url="u1", tracker=Tracker.NYAA, is_best=True, is_dual_audio=True),
            _torrent(release_group="SingleAlt", url="u2", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"SingleAlt"}

    def test_best_breaks_ties_within_preferred_audio(self) -> None:
        filt = make_release_filter(want_best=True, prefer_dual_audio=True)
        entry = _entry(
            _torrent(release_group="DualBest", url="u1", tracker=Tracker.NYAA, is_best=True, is_dual_audio=True),
            _torrent(release_group="DualAlt", url="u2", tracker=Tracker.NYAA, is_dual_audio=True),
            _torrent(release_group="Single", url="u3", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"DualBest"}

    def test_takes_best_when_no_preferred_audio(self) -> None:
        filt = make_release_filter(want_best=True, prefer_dual_audio=True)
        entry = _entry(
            _torrent(release_group="Best", url="u1", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="Rest", url="u2", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"Best"}

    def test_is_public_false_for_private_tracker_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even when the tracker claims public, a name in PRIVATE_TRACKERS is not public.
        # NYAA.is_public() is True, so pin "nyaa" into PRIVATE_TRACKERS to exercise the
        # second conjunct (a real private tracker is_public()==False already, which would
        # short-circuit before the membership check ever ran).
        monkeypatch.setattr(seadex_filter, "PRIVATE_TRACKERS", {"nyaa"})
        filt = make_release_filter(want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="u1", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert result["A"].releases["u1"].is_public is False

    def test_drops_private_release_when_public_exists(self) -> None:
        filt = make_release_filter(private_releases="warn", want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="pub", tracker=Tracker.NYAA),
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES),
        )
        result = filt.build(entry)
        assert set(result["A"].releases) == {"pub"}

    def test_keeps_private_only_group(self) -> None:
        # A group with no public option is kept here. It's only dropped later in
        # reduce_overlapping_grabs if the Arr already owns it.
        filt = make_release_filter(private_releases="warn", want_best=False, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES),
        )
        result = filt.build(entry)
        assert set(result["A"].releases) == {"priv"}


class TestPrivateFallback:
    """`private_releases: fallback` - the public fallback pool added by `build`."""

    def test_private_only_preferred_adds_public_fallback(self) -> None:
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="Alt", url="pub", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        # The private preferred release stays (for the already-have check). The best
        # public release is added beside it, marked as the fallback.
        assert set(result) == {"Best", "Alt"}
        assert result["Best"].releases["priv"].is_fallback is False
        assert result["Alt"].releases["pub"].is_fallback is True

    def test_no_fallback_added_when_preferred_has_public(self) -> None:
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="pub", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="Alt", url="pub2", tracker=Tracker.NYAA),
        )
        assert set(filt.build(entry)) == {"Best"}

    def test_mixed_preferred_adds_fallback_for_private_only_group(self) -> None:
        # A private-only preferred group triggers the pool even when another
        # preferred group is public (the gate is per-group, not per-entry). The
        # public preferred release is NOT marked as a fallback.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="PrivBest", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="PubBest", url="pub", tracker=Tracker.NYAA, is_best=True),
            _torrent(release_group="Alt", url="pub2", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert set(result) == {"PrivBest", "PubBest", "Alt"}
        assert result["PubBest"].releases["pub"].is_fallback is False
        assert result["Alt"].releases["pub2"].is_fallback is True

    def test_same_group_public_copy_survives_as_fallback(self) -> None:
        # One group with a private preferred release and a public non-best release: the
        # pool re-adds the group's public copy, then the per-group drop removes
        # the private release, leaving a public group marked as the fallback.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="A", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
            _torrent(release_group="A", url="pub", tracker=Tracker.NYAA),
        )
        result = filt.build(entry)
        assert set(result) == {"A"}
        assert set(result["A"].releases) == {"pub"}
        assert result["A"].releases["pub"].is_fallback is True

    def test_no_fallback_added_when_nothing_public(self) -> None:
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = _entry(
            _torrent(release_group="Best", url="priv", tracker=Tracker.ANIMEBYTES, is_best=True),
        )
        result = filt.build(entry)
        assert set(result) == {"Best"}
        assert result["Best"].releases["priv"].is_fallback is False

    def test_warn_mode_does_not_add_fallback(self) -> None:
        # The default: a private-only preferred release stays alone, so the planner
        # warns and skips it exactly as before.
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

    def test_same_group_private_release_with_uncovered_files_survives_the_drop(self) -> None:
        # Group A: a private dual S1+S2 torrent (preferred) + a public S1-only copy
        # re-added as the fallback. The coverage-aware drop must NOT delete the
        # torrent - its S2 file isn't covered by the group's public releases.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=True)
        entry = make_entry_record(
            torrents=(
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.ANIMEBYTES,
                    url="priv",
                    infohash=None,
                    file_names=("A - S01E01.mkv", "A - S02E01.mkv"),
                    is_dual_audio=True,
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.NYAA,
                    url="pub",
                    file_names=("A.S01E01.web.mkv",),
                    is_best=False,
                ),
            ),
        )
        result = filt.build(entry)
        assert set(result) == {"A"}
        assert set(result["A"].releases) == {"priv", "pub"}
        assert result["A"].releases["pub"].is_fallback is True

    def test_same_group_cross_seeded_private_copy_still_dropped(self) -> None:
        # Identical filenames on both trackers (a cross-seed): the private copy is
        # fully covered by the group's public release, so it's dropped as before.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = make_entry_record(
            torrents=(
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.ANIMEBYTES,
                    url="priv",
                    infohash=None,
                    file_names=("A - S01E01.mkv",),
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.NYAA,
                    url="pub",
                    file_names=("A - S01E01.mkv",),
                    is_best=False,
                ),
            ),
        )
        result = filt.build(entry)
        assert set(result["A"].releases) == {"pub"}

    def test_same_group_blind_public_copy_skips_the_search(self) -> None:
        # Group A: a private release with KNOWN files + a public copy whose fileset
        # SeaDex doesn't know (empty). The blind public copy counts as covering
        # its group - mirroring the private side's per-group gate - so the search
        # is skipped and group B's fallback stays out of the dict.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = make_entry_record(
            torrents=(
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.ANIMEBYTES,
                    url="priv",
                    infohash=None,
                    file_names=("A - S01E01.mkv",),
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.NYAA,
                    url="pub",
                    file_names=(),
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="B",
                    tracker=Tracker.NYAA,
                    url="b_pub",
                    file_names=("B.S01E01.mkv",),
                    is_best=False,
                ),
            ),
        )
        result = filt.build(entry)
        assert set(result) == {"A"}
        # The coverage-aware per-group drop still keeps the uncovered private release.
        assert set(result["A"].releases) == {"priv", "pub"}

    def test_exactly_covered_private_files_skip_the_search(self) -> None:
        # MUTATION PIN (needs_fallback): the coverage check `<= public_file_names`
        # flipped to `<` treats an EXACTLY-equal fileset as uncovered and runs the
        # fallback search. Priv's files equal Pub's file-for-file: no search, so
        # the non-best Alt never enters the dict.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = make_entry_record(
            torrents=(
                make_torrent_record(
                    release_group="Priv",
                    tracker=Tracker.ANIMEBYTES,
                    url="priv",
                    infohash=None,
                    file_names=("Show - S01E01.mkv",),
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="Pub",
                    tracker=Tracker.NYAA,
                    url="pub",
                    infohash="c" * 40,
                    file_names=("Show - S01E01.mkv",),
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="Alt",
                    tracker=Tracker.NYAA,
                    url="alt",
                    infohash="d" * 40,
                    file_names=("Show.S01E01.web.mkv",),
                    is_best=False,
                ),
            ),
        )
        result = filt.build(entry)
        assert set(result) == {"Priv", "Pub"}

    def test_private_first_iteration_still_registers_group_public(self) -> None:
        # MUTATION PIN (group_has_public accumulation): `prior or is_pub` degraded
        # (e.g. to `and`) loses the group's public flag when a PRIVATE candidate
        # iterates first. Group A: blind private then public - A is covered
        # per-group, so no search runs and Alt stays out.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = make_entry_record(
            torrents=(
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.ANIMEBYTES,
                    url="priv",
                    infohash=None,
                    file_names=(),  # blind private -> the per-group gate decides
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.NYAA,
                    url="pub",
                    infohash="c" * 40,
                    file_names=("A - S01E01.mkv",),
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="Alt",
                    tracker=Tracker.NYAA,
                    url="alt",
                    infohash="d" * 40,
                    file_names=("Alt.S01E01.mkv",),
                    is_best=False,
                ),
            ),
        )
        result = filt.build(entry)
        assert set(result) == {"A"}

    def test_mixed_group_private_files_uncovered_triggers_the_search(self) -> None:
        # Group A has a public S1 AND a private S2 among the preferred releases: the
        # gate is per-candidate file coverage (not per-group), so the private S2
        # still triggers the search and group B's public S2 release enters as
        # the fallback - while the coverage-aware drop keeps A's private S2 release.
        filt = make_release_filter(private_releases="fallback", want_best=True, prefer_dual_audio=False)
        entry = make_entry_record(
            torrents=(
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.NYAA,
                    url="a_pub",
                    infohash="c" * 40,
                    file_names=("A - S01E01.mkv",),
                    file_size=1000,
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="A",
                    tracker=Tracker.ANIMEBYTES,
                    url="a_priv",
                    infohash=None,
                    file_names=("A - S02E01.mkv",),
                    file_size=2000,
                    is_best=True,
                ),
                make_torrent_record(
                    release_group="B",
                    tracker=Tracker.NYAA,
                    url="b_pub",
                    infohash="d" * 40,
                    file_names=("B.S02E01.mkv",),
                    file_size=3000,
                    is_best=False,
                ),
            ),
        )
        result = filt.build(entry)
        assert set(result) == {"A", "B"}
        assert set(result["A"].releases) == {"a_pub", "a_priv"}
        assert result["B"].releases["b_pub"].is_fallback is True


class TestInteractivePick:
    """The interactive numbered-pick prompt supports comma-separated multi-select and tolerates bad input.

    All-invalid input grabs nothing for the title and caches no result.
    """

    def test_comma_separated_multi_pick_keeps_each_selection(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # MUTATION PIN: the comma-separated selection path was never exercised -
        # a mutated split/parse collapses "0, 2" to nothing. Both picked groups
        # (with the space-padded token) must survive, the unpicked one dropped.
        filt = make_release_filter()
        seadex_dict = {"GroupA": rg_group({}), "GroupB": rg_group({}), "GroupC": rg_group({})}

        def fake_input(prompt: str = "") -> str:
            del prompt
            return "0, 2"

        monkeypatch.setattr("builtins.input", fake_input)
        result = filt.interactive_pick(seadex_dict, make_entry_record())
        assert list(result) == ["GroupA", "GroupC"]
        # The picker prints its rows straight to the terminal (that contract is pinned
        # by test_tolerates_non_numeric_input). Here we only drain them so the prompt
        # stays off the terminal under `-s`.
        capsys.readouterr()

    def test_tolerates_non_numeric_input(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A non-numeric pick (a typo) must not crash the title with an uncaught
        # ValueError. The bad token is skipped with a warning, so nothing is selected.
        filt = make_release_filter()
        seadex_dict = {"GroupA": rg_group({})}
        sd_entry = make_entry_record()

        def fake_input(prompt: str = "") -> str:
            del prompt
            return "x"

        monkeypatch.setattr("builtins.input", fake_input)
        recording = RecordingHub()
        install_hub(recording.hub)  # conftest teardown restores the default
        result = filt.interactive_pick(seadex_dict, sd_entry)
        assert result == {}
        # The prompt rows are printed, not logged, so they stay visible even at
        # log_level WARNING (a demoted INFO row would vanish and leave input() blind).
        assert "[0]: GroupA" in capsys.readouterr().out
        # Only genuine problems warn (the hub's SeverityCounter tallies WARNINGs
        # into the run's issues summary): the invalid token, then the empty pick.
        warnings = [d.message for d in recording.of_type(Diagnostic) if d.severity is Severity.WARNING]
        assert len(warnings) == 2
        assert "invalid selection" in warnings[0]
        assert "No valid selection" in warnings[1]

    def test_seadex_notes_with_markup_like_text_do_not_crash(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # SeaDex notes are third-party text: a bracketed fragment that parses as
        # broken Rich markup (a closing tag that was never opened) used to raise
        # MarkupError through console.print and abort the whole arr run. The rows
        # must print literally instead.
        filt = make_release_filter()
        seadex_dict = {"GroupA": rg_group({}), "GroupB": rg_group({})}
        sd_entry = make_entry_record(notes="[/Kaleido] mux is preferred [b")

        def fake_input(prompt: str = "") -> str:
            del prompt
            return ""  # blank = keep all

        monkeypatch.setattr("builtins.input", fake_input)
        result = filt.interactive_pick(seadex_dict, sd_entry)

        assert set(result) == {"GroupA", "GroupB"}
        assert "[/Kaleido] mux is preferred [b" in capsys.readouterr().out

    def test_all_invalid_selection_skips_the_title_without_caching(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # An interactive pick where EVERY token is invalid returns {} - the
        # strategy must then move on WITHOUT caching the title. Caching would
        # stamp the entry's updated_at, so the fumbled title would be treated as
        # done and silently suppressed until SeaDex next updates the entry.
        class _MovieItem:
            def __init__(self) -> None:
                self.id = 1
                self.title = "Movie"
                self.imdbId: str | None = None
                self.monitored = True
                self.tmdbId = 550

        anilist_id = 99
        entry = make_entry_record(
            anilist_id=anilist_id,
            torrents=(
                make_torrent_record(release_group="A", tracker=Tracker.NYAA, url="a", infohash="a" * 40),
                make_torrent_record(release_group="B", tracker=Tracker.NYAA, url="b", infohash="b" * 40),
            ),
        )
        cache = FakeCacheStore()
        config = make_config(url="http://sonarr", api_key="key", interactive=True, sleep_time=0)
        deps = make_run_deps(config=config, cache_store=cache, seadex=FakeSeaDexSource({anilist_id: entry}))
        services = ArrRunServices(deps, Arr.RADARR)
        strat = RadarrStrategy(deps, services, radarr_client=FakeRadarrClient())
        # Serve the title from the gateway's in-memory cache so no AniList query runs.
        deps.anilist.answers[anilist_id] = {"data": {"Media": {"title": {"english": "Movie", "romaji": None}}}}

        def fake_input(prompt: str = "") -> str:
            del prompt
            return "42"  # out of range -> every token invalid -> empty pick

        monkeypatch.setattr("builtins.input", fake_input)
        strat.check_title(_MovieItem(), anilist_id, TitleMapping(anilist_id=anilist_id))
        capsys.readouterr()  # drain the picker's terminal rows

        # Nothing persisted: the title must resurface (and re-prompt) next run.
        assert cache.get_cached_title(Arr.RADARR, anilist_id) is None


def test_file_size_cover_merges_torrents_with_elementwise_max() -> None:
    """MUTATION PIN: `|` not `+`, and not a set union either.

    Two torrents carrying one copy of a size each cannot cover a torrent that
    genuinely carries two - only ONE torrent with both can. A `+` merge would
    fake that coverage and prune a real release; a set union would deflate a real
    two-copy torrent to one and offer a fallback nobody needs.
    """

    cover = seadex_filter.file_size_cover
    one_each = [[100], [100]]
    assert cover(one_each)[100] == 1
    assert not Counter([100, 100]) <= cover(one_each)
    assert cover([[100, 100]])[100] == 2
    # Coverage genuinely split across two torrents still covers.
    assert Counter([100, 200]) <= cover([[100], [200]])
