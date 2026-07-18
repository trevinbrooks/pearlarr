# pyright: strict
"""Real-API-fixture tests for the resolved-mapping manual import.

These pin the behavior the *old* code got wrong, using JSON captured verbatim
from a live Sonarr (`tests/fixtures/sonarr/`). The headline failure that
motivated the rewrite: a specials/alias release Sonarr can't match to a series
(`Yamada-kun and the Seven Witches` vs `... (2015)`) returns an empty
series-*matched* `episodes` array, so the import silently mapped nothing. The
fix reads the series-*agnostic* `parsedEpisodeInfo` and assigns it into OUR
resolved episode set - identity comes from the same mapping the add flow
already trusts; Sonarr's title match only informs, in-set (`matched_episodes`).

The pure `assign_episode_ids` tests encode the three cases the user raised
(correctly-named specials, mis-numbered specials, multi-season "To Love-Ru"); the
end-to-end test drives the real Yamada fixtures through `import_completed`.
"""

import json
from pathlib import Path

from pearlarr.config import AppConfig
from pearlarr.manual_import import (
    ImportProgress,
    ImportReadiness,
    normalize_basename,
)
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import (
    CommandResource,
    ManualImportCandidate,
    MatchedEpisode,
    ParsedFileInfo,
    QualityDefinition,
    QualitySource,
    SonarrEpisode,
)
from pearlarr.sonarr_import_plan import (
    CandidateFile,
    ContentPaths,
    EpisodeAssignment,
    ParsedQuality,
    QueueVerdict,
    assign_episode_ids,
    classify_queue,
    manual_import_in_flight,
    parse_se_from_filename,
    quality_axes_from_model,
    resolve_quality,
)

from .builders import (
    FakeCacheStore,
    make_config,
    make_sonarr_mapper,
    make_sonarr_sync,
    pending_import,
)
from .fakes import FakeSonarrClient

_FIXTURES = Path(__file__).parent / "fixtures" / "sonarr"


def load_fixture[T](name: str, _shape: type[T] | None = None) -> T:
    """Parse one captured Sonarr response, typed by the call site's annotation.

    `_shape` is unused at runtime; it gives `T` a second occurrence so pyright
    does not flag the otherwise return-only TypeVar (reportInvalidTypeVarUse). The
    raw JSON shape (`Any`) is narrowed by the consuming boundary models.
    """

    data: T = json.loads((_FIXTURES / name).read_text())
    return data


def _load_definitions() -> list[QualityDefinition]:
    """The captured quality-definition list, validated as the client boundary does."""

    raw: list[dict[str, object]] = load_fixture("qualitydefinitions.json")
    return [QualityDefinition.model_validate(d) for d in raw]


def _pinfo(
    *,
    season: int | None = None,
    episodes: tuple[int, ...] = (),
    absolutes: tuple[int, ...] = (),
    matched: tuple[tuple[int, int], ...] = (),
    full_season: bool = False,
    offline: bool = False,
) -> ParsedFileInfo:
    """Shorthand ParsedFileInfo for the pure-assignment tests."""

    return ParsedFileInfo(
        season_number=season,
        episode_numbers=episodes,
        absolute_episode_numbers=absolutes,
        matched_episodes=tuple(
            MatchedEpisode(season_number=matched_season, episode_number=episode) for matched_season, episode in matched
        ),
        full_season=full_season,
        offline=offline,
    )


# --------------------------------------------------------------------------- #
# Quality resolution - the (source, resolution) match, on real bodies
# --------------------------------------------------------------------------- #
class TestQualityResolution:
    """The quality fix's load-bearing claims.

    Quality is matched by the structured `(source, resolution)` pair. The
    candidate-read test runs on a verbatim live-Sonarr capture; the
    qualitydefinition list is a hand-authored STAND-IN (`qualitydefinitions.json`)
    mirroring real Sonarr - the live `/api/v3/qualitydefinition` capture is owed
    (the user's instance sits behind an auth proxy). Dropping a real capture in
    place of the stand-in re-runs these against reality unchanged.
    """

    def test_qualitydefinition_fixture_has_the_shape_the_matcher_needs(self) -> None:
        # CONTRACT, not validation: the matcher keys on (source, resolution), so
        # every definition must carry both. This guards the stand-in (and any real
        # capture swapped in for it) - it does NOT by itself prove the live
        # instance serializes the fields; that capture is owed to the user.
        defs = _load_definitions()
        assert defs
        for definition in defs:
            quality = definition.quality
            assert quality is not None
            assert isinstance(quality.resolution, int)
            assert isinstance(quality.source, str)
            if quality.name != "Unknown":
                assert QualitySource.parse(quality.source) is not None

    def test_bd_remux_resolves_against_full_def_list(self) -> None:
        # The original failure: a 1080p BD remux. Sonarr parses it as
        # (blurayRaw, 1080); against the full definition list that must resolve to
        # the "Bluray-1080p Remux" definition (valid id+name) - never omitted.
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            _load_definitions(),
            candidate_model=None,
        )
        quality = model.quality
        assert quality is not None
        assert quality.name == "Bluray-1080p Remux"
        assert quality.source == "blurayRaw"
        assert quality.resolution == 1080

    def test_structured_read_on_real_manualimport_candidate(self) -> None:
        # quality_axes_from_model reads (source, resolution) off a candidate
        # captured verbatim from a live Sonarr - proving the read works on real
        # output, not just hand-written dicts.
        raw: list[dict[str, object]] = load_fixture("manualimport_yamada.json")
        candidates = [ManualImportCandidate.model_validate(c) for c in raw]
        dvd = next(
            c
            for c in candidates
            if c.quality is not None and c.quality.quality is not None and c.quality.quality.name == "DVD"
        )
        assert quality_axes_from_model(dvd.quality) == ParsedQuality(
            source=QualitySource.DVD,
            resolution=480,
        )


# --------------------------------------------------------------------------- #
# ParsedFileInfo - the series-agnostic field, on real bodies
# --------------------------------------------------------------------------- #
class TestParsedFileInfoFromRealBodies:
    """The load-bearing claim: `parsedEpisodeInfo` populates even when `episodes` (series-matched) is empty."""

    def test_yamada_special_has_season_episode_despite_no_series_match(self) -> None:
        body: dict[str, object] = load_fixture("parse_yamada_s00e01.json")
        # The OLD code read this (series-matched) array and got nothing:
        assert body["episodes"] == []

        info = ParsedFileInfo.model_validate(body)
        assert info.season_number == 0
        assert info.episode_numbers == (1,)
        assert info.absolute_episode_numbers == ()

    def test_absolute_numbered_file_reports_absolute_not_season_episode(self) -> None:
        body: dict[str, object] = load_fixture("parse_toloveru_abs14.json")
        info = ParsedFileInfo.model_validate(body)
        assert info.episode_numbers == ()
        assert info.absolute_episode_numbers == (14,)

    def test_missing_parsed_info_is_all_empty(self) -> None:
        info = ParsedFileInfo.model_validate({})
        assert info == ParsedFileInfo()

    def test_full_season_flag_reads_through(self) -> None:
        info = ParsedFileInfo.model_validate({"parsedEpisodeInfo": {"fullSeason": True}})
        assert info.full_season is True

    def test_junk_matched_entry_poisons_the_whole_array(self) -> None:
        # One malformed episodes[] entry folds the WHOLE array to () - dropping
        # just the bad one would shorten a span into a partial placement.
        body: dict[str, object] = {
            "episodes": [
                {"seasonNumber": 1, "episodeNumber": 1, "id": 501},
                {"seasonNumber": 1, "id": 502},
            ],
        }
        info = ParsedFileInfo.model_validate(body)
        assert info.matched_episodes == ()


# --------------------------------------------------------------------------- #
# parse_se_from_filename - the offline SxxExx fallback
# --------------------------------------------------------------------------- #
class TestParseSeFromFilename:
    """`parse_se_from_filename` extracts an offline SxxExx pattern, never guessing a bare absolute number."""

    def test_sxxexx_extracted(self) -> None:
        info = parse_se_from_filename("Show.Name.S00E05.480p.mkv")
        assert info is not None
        assert info.season_number == 0
        assert info.episode_numbers == (5,)
        # Marked offline: the regex is blind to absolutes, so the positional
        # leg's duplicate tell must treat this stand-in as unknown.
        assert info.offline is True

    def test_dash_separated_sxxexx(self) -> None:
        info = parse_se_from_filename("Show - S2E3 [1080p].mkv")
        assert info is not None
        assert (info.season_number, info.episode_numbers) == (2, (3,))

    def test_bare_absolute_number_is_not_guessed(self) -> None:
        # "01" alone is NOT an SxxExx - left to Sonarr's parse / the absolute leg,
        # never guessed as S?E01 here.
        assert parse_se_from_filename("Show - 01 [1080p].mkv") is None


# --------------------------------------------------------------------------- #
# assign_episode_ids - the three cases the user raised, plus guards
# --------------------------------------------------------------------------- #
class TestAssignExactSeason:
    """Leg 1: a correctly-named file Sonarr just couldn't match to the series."""

    def test_yamada_specials_assigned_by_exact_season_episode(self) -> None:
        # Resolved set is the entry's S00 episodes (ids 8030..8032); the two files
        # carry S00E01 / S00E02 and land on 8030 / 8031.
        files = ["s00e01.mkv", "s00e02.mkv"]
        parsed = {
            "s00e01.mkv": _pinfo(season=0, episodes=(1,)),
            "s00e02.mkv": _pinfo(season=0, episodes=(2,)),
        }
        ep_id_map = {(0, 1): 8030, (0, 2): 8031, (0, 3): 8032, (1, 1): 8033}

        result = assign_episode_ids(files, parsed, [8030, 8031, 8032], ep_id_map)

        assert result == EpisodeAssignment(
            assigned={"s00e01.mkv": [8030], "s00e02.mkv": [8031]},
            skipped=[],
        )

    def test_exact_parse_outside_resolved_set_is_skipped(self) -> None:
        # File parses to S01E01 (id 8033) but the resolved set is only S00 -> never
        # imported (the over-grab guard: identity must land INSIDE our set).
        parsed = {"x.mkv": _pinfo(season=1, episodes=(1,))}
        ep_id_map = {(0, 1): 8030, (1, 1): 8033}

        result = assign_episode_ids(["x.mkv"], parsed, [8030], ep_id_map)

        assert result.assigned == {}
        assert result.skipped == ["x.mkv"]

    def test_empty_resolved_set_places_correctly_named_specials(self) -> None:
        # The stuck-record case: NO resolved set (an empty ordered_episode_ids, e.g.
        # a record whose grab-time specials resolution found nothing). The exact leg
        # falls back to the live series map, so a correctly-named file lands on its
        # real episode instead of sticking forever.
        files = ["s00e01.mkv", "s00e02.mkv"]
        parsed = {
            "s00e01.mkv": _pinfo(season=0, episodes=(1,)),
            "s00e02.mkv": _pinfo(season=0, episodes=(2,)),
        }
        ep_id_map = {(0, 1): 8030, (0, 2): 8031, (0, 3): 8032}

        result = assign_episode_ids(files, parsed, [], ep_id_map)

        assert result == EpisodeAssignment(
            assigned={"s00e01.mkv": [8030], "s00e02.mkv": [8031]},
            skipped=[],
        )


class TestAssignAbsolute:
    """Leg 2: absolute-number index onto the resolved set."""

    def test_mis_numbered_specials_map_positionally(self) -> None:
        # The user's case: files on disk are "01".."05" but are really S00E05..E09.
        # The release numbers never decide identity - they only ORDER the files onto
        # the resolved set, so "01" -> the first resolved episode (8034 = S00E05).
        files = [f"{n:02d}.mkv" for n in range(1, 6)]
        parsed = {name: _pinfo(absolutes=(i + 1,)) for i, name in enumerate(files)}
        resolved = [8034, 8035, 8036, 8037, 8038]  # S00E05..E09 ids

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.skipped == []
        assert result.assigned == {
            "01.mkv": [8034],
            "02.mkv": [8035],
            "03.mkv": [8036],
            "04.mkv": [8037],
            "05.mkv": [8038],
        }

    def test_continuous_absolute_batch_spans_seasons(self) -> None:
        # A continuous absolute batch (1..4) maps cleanly onto a season-sorted
        # multi-season resolved set - this is the only multi-season pack we trust.
        files = ["e1.mkv", "e2.mkv", "e3.mkv", "e4.mkv"]
        parsed = {f"e{i}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 5)}
        resolved = [501, 502, 601, 602]  # S05E01-02, S06E01-02

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.assigned == {
            "e1.mkv": [501],
            "e2.mkv": [502],
            "e3.mkv": [601],
            "e4.mkv": [602],
        }

    def test_no_signal_file_refuses_the_positional_leg(self) -> None:
        # A file whose parse yields nothing could be a hiccuped real episode;
        # the every-file check refuses the whole leg (skip + warn, retried).
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": _pinfo(absolutes=(1,)),
            "b.mkv": _pinfo(absolutes=(2,)),
            "menu.mkv": _pinfo(),  # a 200 /parse with null parsedEpisodeInfo
        }

        result = assign_episode_ids(["a.mkv", "b.mkv", "menu.mkv"], parsed, [501, 502], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv", "menu.mkv"]

    def test_overlord_absolute_ova_pack_maps_onto_resolved_set(self) -> None:
        # releases.moe/101083 ("Overlord II - Ple Ple Pleiades 2"): 13 OVA files
        # named "- 01".."- 13", all parsed season 0 / absolute-only. The add flow
        # resolves this entry (anibridge tvdb_mappings {0: [(16, 28)]}) to the 13
        # season-0 episodes S00E16..E28 (live ids 2090..2102), so the absolute leg
        # places each file onto its season-sorted id (count-matched 13:13, no-dup) -
        # "- 01" -> S00E16, "- 13" -> S00E28. No grab-time change needed.
        files = [f"{n:02d}.mkv" for n in range(1, 14)]
        parsed = {name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}
        resolved = list(range(2090, 2103))  # S00E16..E28 ids, season order

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.skipped == []
        assert result.assigned == {f"{n:02d}.mkv": [2089 + n] for n in range(1, 14)}


class TestAssignMatchedPairs:
    """Leg 1's matched-pairs fallback: Sonarr's series-matched `(season, episode)` for absolute-only names."""

    def test_multi_entry_batch_places_exactly_inside_the_set(self) -> None:
        # The real Thighs incident: a batch spanning two AL entries + a special,
        # record covering Cour 2 - in-set files place exactly, the rest skip.
        files = ["ep-11.mkv", "ep-12.mkv", "ep-13.mkv", "sp-17.5.mkv"]
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep-11.mkv": _pinfo(season=0, absolutes=(11,), matched=((1, 11),)),
            "ep-12.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            "ep-13.mkv": _pinfo(season=0, absolutes=(13,), matched=((1, 13),)),
            # The 17.5 special: S00E01 in the NAME, so no matched fallback needed.
            "sp-17.5.mkv": _pinfo(season=0, episodes=(1,), matched=((0, 1),)),
        }
        ep_id_map = {(1, 11): 2585, (1, 12): 2586, (1, 13): 2587, (0, 1): 2574}

        result = assign_episode_ids(files, parsed, [2586, 2587], ep_id_map)

        assert result.assigned == {"ep-12.mkv": [2586], "ep-13.mkv": [2587]}
        assert sorted(result.skipped) == ["ep-11.mkv", "sp-17.5.mkv"]

    def test_name_parsed_pair_beats_matched_pair(self) -> None:
        # A name that carries its own (season, episode) never defers to
        # Sonarr's matched resolution.
        parsed = {"x.mkv": _pinfo(season=2, episodes=(5,), matched=((9, 9),))}
        ep_id_map = {(2, 5): 400, (9, 9): 999}

        result = assign_episode_ids(["x.mkv"], parsed, [400, 999], ep_id_map)

        assert result.assigned == {"x.mkv": [400]}

    def test_matched_pairs_never_apply_unscoped(self) -> None:
        # With NO resolved set, the live-map fallback trusts a name-parsed pair
        # only - Sonarr's series match must not decide identity on its own.
        parsed = {"x.mkv": _pinfo(season=0, absolutes=(3,), matched=((1, 3),))}

        result = assign_episode_ids(["x.mkv"], parsed, [], {(1, 3): 300})

        assert result.assigned == {}
        assert result.skipped == ["x.mkv"]

    def test_partially_in_set_matched_span_is_skipped(self) -> None:
        # A matched span reaching outside the resolved set is refused whole -
        # same half-import posture as the name-parsed leg.
        parsed = {"span.mkv": _pinfo(season=0, matched=((1, 1), (1, 3)))}
        ep_id_map = {(1, 1): 501, (1, 3): 503}

        result = assign_episode_ids(["span.mkv"], parsed, [501, 502], ep_id_map)

        assert result.assigned == {}
        assert result.skipped == ["span.mkv"]

    def test_out_of_set_match_does_not_veto_the_single_file_fallback(self) -> None:
        # One numberless file, one leftover id: OUR resolution places it even
        # when Sonarr's title match claims an out-of-set episode.
        parsed = {"only.mkv": _pinfo(matched=((1, 5),))}

        result = assign_episode_ids(["only.mkv"], parsed, [900], {(1, 5): 555})

        assert result.assigned == {"only.mkv": [900]}
        assert result.skipped == []

    def test_wrong_series_matched_id_is_refused(self) -> None:
        # Sonarr matched some OTHER series whose numbers coincide with ours:
        # its episode id disagrees with our map, so the claim is refused.
        info = ParsedFileInfo(
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=999),),
        )

        result = assign_episode_ids(["x.mkv"], {"x.mkv": info}, [501, 502], {(1, 1): 501})

        assert result.assigned == {}
        assert result.skipped == ["x.mkv"]

    def test_agreeing_matched_id_places(self) -> None:
        # The same claim with Sonarr's id AGREEING with our map places normally.
        info = ParsedFileInfo(
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=501),),
        )

        result = assign_episode_ids(["x.mkv"], {"x.mkv": info}, [501, 502], {(1, 1): 501})

        assert result.assigned == {"x.mkv": [501]}

    def test_duplicate_matched_pairs_collapse_to_one_claim(self) -> None:
        # Junk wire duplicates of the same pair are one claim, not a veto.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1),
            ),
        )

        result = assign_episode_ids(["x.mkv"], {"x.mkv": info}, [501, 502], {(1, 1): 501})

        assert result.assigned == {"x.mkv": [501]}

    def test_mixed_id_duplicate_claims_place_once(self) -> None:
        # (s,e,None) and (s,e,id) survive the triple dedup as two claims; the
        # wire list still carries the episode id once. Two resolved ids keep
        # the degenerate arm out, so this pins leg 1 itself.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
            ),
        )

        result = assign_episode_ids(["x.mkv"], {"x.mkv": info}, [501, 502], {(1, 1): 501})

        assert result.assigned == {"x.mkv": [501]}

    def test_wrong_id_match_cannot_veto_the_single_file_fallback(self) -> None:
        # A disagreeing-id match refuses the CLAIM, but with one numberless
        # file and one leftover id the degenerate fallback still places the
        # only possible way (same posture as the out-of-set variant above).
        info = ParsedFileInfo(
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=999),),
        )

        result = assign_episode_ids(["only.mkv"], {"only.mkv": info}, [501], {(1, 1): 501})

        assert result.assigned == {"only.mkv": [501]}

    def test_full_season_parse_never_borrows_matched_pairs(self) -> None:
        # Sonarr matches a bare "S01" extras file to EVERY season episode; one
        # junk file must not swallow the entry while the real files place.
        parsed: dict[str, ParsedFileInfo | None] = {
            "extras.mkv": _pinfo(matched=((1, 1), (1, 2)), full_season=True),
            "ep-01.mkv": _pinfo(season=0, absolutes=(1,), matched=((1, 1),)),
            "ep-02.mkv": _pinfo(season=0, absolutes=(2,), matched=((1, 2),)),
        }
        ep_id_map = {(1, 1): 501, (1, 2): 502}

        result = assign_episode_ids(["extras.mkv", "ep-01.mkv", "ep-02.mkv"], parsed, [501, 502], ep_id_map)

        assert result.assigned == {"ep-01.mkv": [501], "ep-02.mkv": [502]}
        assert result.skipped == ["extras.mkv"]

    def test_wide_matched_span_is_refused(self) -> None:
        # A 4-episode matched span exceeds what one file plausibly holds (the
        # season-pack shape without the fullSeason flag), so it never borrows.
        parsed = {"pack.mkv": _pinfo(matched=((1, 1), (1, 2), (1, 3), (1, 4)))}
        ep_id_map = {(1, n): 500 + n for n in range(1, 5)}

        result = assign_episode_ids(["pack.mkv"], parsed, [501, 502, 503, 504], ep_id_map)

        assert result.assigned == {}
        assert result.skipped == ["pack.mkv"]

    def test_triple_episode_matched_span_places(self) -> None:
        # The cap boundary: a triple-episode file's span is still a per-file claim.
        parsed = {"triple.mkv": _pinfo(matched=((1, 1), (1, 2), (1, 3)))}
        ep_id_map = {(1, 1): 501, (1, 2): 502, (1, 3): 503}

        result = assign_episode_ids(["triple.mkv"], parsed, [501, 502, 503], ep_id_map)

        assert result.assigned == {"triple.mkv": [501, 502, 503]}

    def test_junk_duplicates_beyond_the_cap_still_collapse_and_place(self) -> None:
        # The cap counts DISTINCT claims: four wire duplicates of one pair are
        # one claim, not a season-pack shape.
        parsed = {"x.mkv": _pinfo(matched=((1, 1), (1, 1), (1, 1), (1, 1)))}

        result = assign_episode_ids(["x.mkv"], parsed, [501], {(1, 1): 501})

        assert result.assigned == {"x.mkv": [501]}

    def test_mixed_id_duplicate_of_a_triple_span_still_places(self) -> None:
        # The cap counts distinct (season, episode) pairs, so an id-bearing
        # junk duplicate of one pair can't inflate a triple past it.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
                MatchedEpisode(season_number=1, episode_number=2),
                MatchedEpisode(season_number=1, episode_number=3),
            ),
        )
        ep_id_map = {(1, 1): 501, (1, 2): 502, (1, 3): 503}

        result = assign_episode_ids(["x.mkv"], {"x.mkv": info}, [501, 502, 503], ep_id_map)

        assert result.assigned == {"x.mkv": [501, 502, 503]}

    def test_partially_resolved_double_absolute_never_half_imports(self) -> None:
        # A "12-13" file whose match resolved only E12 (absolute 13 beyond
        # Sonarr's mapping): the borrowed span doesn't cover the absolutes,
        # so placing the resolved half is refused.
        parsed = {"d.mkv": _pinfo(season=0, absolutes=(12, 13), matched=((1, 12),))}

        result = assign_episode_ids(["d.mkv"], parsed, [2586, 2587], {(1, 12): 2586})

        assert result.assigned == {}
        assert result.skipped == ["d.mkv"]

    def test_fully_resolved_double_absolute_places_both(self) -> None:
        # The same file with BOTH pairs resolved places as a two-episode file.
        parsed = {"d.mkv": _pinfo(season=0, absolutes=(12, 13), matched=((1, 12), (1, 13)))}
        ep_id_map = {(1, 12): 2586, (1, 13): 2587}

        result = assign_episode_ids(["d.mkv"], parsed, [2586, 2587], ep_id_map)

        assert result.assigned == {"d.mkv": [2586, 2587]}
        assert result.skipped == []

    def test_matched_span_never_half_imports_via_the_single_file_fallback(self) -> None:
        # A file Sonarr says spans E01+E02 must not import as E01 alone via
        # the degenerate arm - cardinality evidence is honored even where
        # identity evidence is not (restored 1fc1d5e pin).
        parsed = {"span.mkv": _pinfo(matched=((1, 1), (1, 2)))}

        result = assign_episode_ids(["span.mkv"], parsed, [501], {(1, 1): 501, (1, 2): 502})

        assert result.assigned == {}
        assert result.skipped == ["span.mkv"]

    def test_full_season_file_never_takes_the_spare_id(self) -> None:
        # Leg 1 quarantines the season-pack shape; the degenerate arm must
        # not hand it the one spare id either.
        parsed: dict[str, ParsedFileInfo | None] = {
            "extras-s01.mkv": _pinfo(matched=((1, 1), (1, 2), (1, 3), (1, 4)), full_season=True),
            "e01.mkv": _pinfo(season=1, episodes=(1,)),
            "e02.mkv": _pinfo(season=1, episodes=(2,)),
            "e03.mkv": _pinfo(season=1, episodes=(3,)),
        }
        files = ["extras-s01.mkv", "e01.mkv", "e02.mkv", "e03.mkv"]
        ep_id_map = {(1, n): 500 + n for n in range(1, 5)}

        result = assign_episode_ids(files, parsed, [501, 502, 503, 504], ep_id_map)

        assert result.assigned == {"e01.mkv": [501], "e02.mkv": [502], "e03.mkv": [503]}
        assert result.skipped == ["extras-s01.mkv"]


class TestAssignGuards:
    """Leg 3: refuse to guess - skip + warn instead."""

    def test_toloveru_per_title_restart_is_refused(self) -> None:
        # One torrent spanning two sub-series whose numbering BOTH restart at 1:
        # the shared absolutes are the tell of a season-boundary scramble, so the
        # whole absolute leg is refused rather than mis-assigned.
        main = {f"main-{i:02d}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 4)}
        dark = {f"dark-{i:02d}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 4)}
        parsed = {**main, **dark}
        files = list(parsed)
        resolved = [501, 502, 503, 601, 602, 603]

        result = assign_episode_ids(files, parsed, resolved, {})

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(files)


def _cand(basename: str) -> CandidateFile:
    return CandidateFile(
        basename=basename,
        path=f"/dl/{basename}",
        quality=None,
        is_sample=False,
        is_already_imported=False,
    )


class TestAssignScopeGate:
    """CB3: allow_unscoped must key off the FULL resolved set, not the post-seed remainder."""

    def test_explicit_allow_unscoped_false_keeps_scope_on_empty_set(self) -> None:
        # An empty resolved set but allow_unscoped pinned False (what the mapper passes
        # for a fully-seeded record): a correctly-named but out-of-scope file is
        # skipped, NOT placed on the live map.
        parsed = {"x.mkv": _pinfo(season=1, episodes=(1,))}
        ep_id_map = {(1, 1): 8033}

        result = assign_episode_ids(["x.mkv"], parsed, [], ep_id_map, allow_unscoped=False)

        assert result.assigned == {}
        assert result.skipped == ["x.mkv"]

    def test_fully_seeded_record_skips_out_of_scope_on_disk_leftover(self) -> None:
        # A fully-seeded record (every resolved episode already seeded) whose batch
        # folder also holds an OUT-OF-SCOPE file (a season-2 file in a season-1 grab).
        # The leftover must be skipped - not imported via the allow_unscoped fallback -
        # and the grab-time seed map must stay un-contaminated by the self-heal.
        seed_name = "Show - 01 [1080p].mkv"
        leftover_name = "Show - S02E01 [1080p].mkv"
        pending = pending_import(
            file_episode_map={seed_name: [101]},
            episode_ids=[101],
            ordered_episode_ids=[101],
            seadex_files=[seed_name],
        )
        sonarr = FakeSonarrClient(parse_episode_info_fn=lambda _f: _pinfo(season=2, episodes=(1,)))
        mapper = make_sonarr_mapper(sonarr=sonarr)

        candidates = {
            normalize_basename(seed_name): _cand(seed_name),
            normalize_basename(leftover_name): _cand(leftover_name),
        }
        ep_id_map = {(1, 1): 101, (2, 1): 999}  # 999 is OUTSIDE the resolved {101}

        merged, skipped = mapper.assign(pending, candidates, ep_id_map)

        placed_ids = {i for ids in merged.values() for i in ids}
        assert 999 not in placed_ids
        assert normalize_basename(leftover_name) in skipped
        assert pending.file_episode_map == {seed_name: [101]}

    def test_count_mismatch_skips(self) -> None:
        # Two absolute files but three resolved ids -> not a clean 1:1 -> skip both.
        parsed = {"a.mkv": _pinfo(absolutes=(1,)), "b.mkv": _pinfo(absolutes=(2,))}

        result = assign_episode_ids(["a.mkv", "b.mkv"], parsed, [1, 2, 3], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv"]

    def test_hiccuped_episode_parse_refuses_the_leg(self) -> None:
        # A None-parse file that is really an EPISODE (parse hiccup, no SxxExx
        # fallback) refuses the leg; the next poll re-parses (misses uncached).
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": _pinfo(absolutes=(1,)),
            "b.mkv": _pinfo(absolutes=(2,)),
            "c.mkv": None,
        }

        result = assign_episode_ids(["a.mkv", "b.mkv", "c.mkv"], parsed, [1, 2, 3], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv", "c.mkv"]

    def test_multi_absolute_file_vetoes_the_leg(self) -> None:
        # A file spanning two absolutes ("01-02") can't be placed positionally,
        # so the leg stays refused.
        parsed: dict[str, ParsedFileInfo | None] = {
            "span.mkv": _pinfo(absolutes=(1, 2)),
            "c.mkv": _pinfo(absolutes=(3,)),
        }

        result = assign_episode_ids(["span.mkv", "c.mkv"], parsed, [1, 2, 3], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == ["c.mkv", "span.mkv"]

    def test_out_of_set_absolute_cannot_fill_in_for_a_hiccuped_episode(self) -> None:
        # Reviewer-reproduced hazard: an out-of-entry sibling (absolute 11,
        # matched out of set) must not fill the count for a hiccuped real E12.
        parsed: dict[str, ParsedFileInfo | None] = {
            "s-11.mkv": _pinfo(season=0, absolutes=(11,), matched=((1, 11),)),
            "e-12.mkv": None,
        }
        ep_id_map = {(1, 11): 2585, (1, 12): 2586}

        result = assign_episode_ids(["s-11.mkv", "e-12.mkv"], parsed, [2586], ep_id_map)

        assert result.assigned == {}
        assert sorted(result.skipped) == ["e-12.mkv", "s-11.mkv"]

    def test_v2_duplicate_of_a_placed_file_is_refused(self) -> None:
        # Leg 1 places "- 12" via its matched pair; the v2 shares absolute 12,
        # so the BATCH-wide duplicate tell refuses the positional leg for it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        ep_id_map = {(1, 12): 2586, (1, 13): 2587}

        result = assign_episode_ids(["e-12.mkv", "e-12v2.mkv"], parsed, [2586, 2587], ep_id_map)

        assert result.assigned == {"e-12.mkv": [2586]}
        assert result.skipped == ["e-12v2.mkv"]

    def test_seeded_sharer_still_vetoes_the_positional_leg(self) -> None:
        # The v1 was placed on an EARLIER poll (seeded, not in ordered_files);
        # its parse still reaches the duplicate tell, so the v2 stays refused.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(["e-12v2.mkv"], parsed, [2587], {(1, 12): 2586})

        assert result.assigned == {}
        assert result.skipped == ["e-12v2.mkv"]

    def test_blipped_batch_parse_refuses_the_positional_leg(self) -> None:
        # A tell-only parse the caller couldn't get may be hiding a duplicate:
        # the leg fails CLOSED, like a hiccuped leftover already does.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": None,
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(["e-12v2.mkv"], parsed, [2587], {(1, 12): 2586})

        assert result.assigned == {}
        assert result.skipped == ["e-12v2.mkv"]

    def test_offline_fallback_parse_refuses_the_positional_leg(self) -> None:
        # The offline SxxExx stand-in knows nothing about absolutes: a
        # dual-numbered seeded sharer must not launder its lost "12" into a
        # known parse and unlock the leg.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-s01e12.mkv": _pinfo(season=1, episodes=(12,), offline=True),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(["e-12v2.mkv"], parsed, [2587], {(1, 12): 2586})

        assert result.assigned == {}
        assert result.skipped == ["e-12v2.mkv"]

    def test_junk_duplicate_absolute_within_one_parse_does_not_veto(self) -> None:
        # One parse repeating its own absolute ((12, 12)) is wire junk, not a
        # restart tell - the unrelated leftover still places.
        parsed: dict[str, ParsedFileInfo | None] = {
            "seeded-12.mkv": _pinfo(season=0, absolutes=(12, 12)),
            "left-13.mkv": _pinfo(season=0, absolutes=(13,)),
        }

        result = assign_episode_ids(["left-13.mkv"], parsed, [507], {})

        assert result.assigned == {"left-13.mkv": [507]}
        assert result.skipped == []

    def test_multi_absolute_seeded_sharer_still_vetoes(self) -> None:
        # A seeded "12-13" span file shares absolute 12 with the leftover v2:
        # every absolute of every parse is counted, so the duplicate shows.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12-13.mkv": _pinfo(season=0, absolutes=(12, 13)),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(["e-12v2.mkv"], parsed, [2588], {(1, 12): 2586})

        assert result.assigned == {}
        assert result.skipped == ["e-12v2.mkv"]

    def test_placed_sharer_still_vetoes_on_the_next_poll(self) -> None:
        # Poll 1 places the v1 and self-heals it onto the record; poll 2 must
        # not let the now-seeded v1 hide the shared absolute from the tell.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {
            v1: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        sonarr = FakeSonarrClient(parse_episode_info_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}
        ep_id_map = {(1, 12): 2586}

        first, _ = mapper.assign(pending, candidates, ep_id_map)
        second, skipped = mapper.assign(pending, candidates, ep_id_map)

        assert first[normalize_basename(v1)] == [2586]
        assert normalize_basename(v2) not in second
        assert normalize_basename(v2) in skipped

    def test_seeded_sharer_parse_blip_fails_closed(self) -> None:
        # A LATER RUN (fresh parse cache): the seeded v1's /parse blips to
        # None, so the tell's input is incomplete - the v2 must stay refused,
        # not slide onto the other episode.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),))}
        sonarr = FakeSonarrClient(parse_episode_info_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}

        merged, skipped = mapper.assign(pending, candidates, {(1, 12): 2586})

        assert normalize_basename(v2) not in merged
        assert normalize_basename(v2) in skipped

    def test_seeded_dual_numbered_sharer_offline_fallback_fails_closed(self) -> None:
        # The seeded v1 is dual-numbered; its /parse blips and the offline
        # SxxExx fallback loses the absolute - the tell must treat that
        # stand-in as unknown, not let the v2 slide onto the spare id.
        v1, v2 = "Show - S01E12 - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),))}
        sonarr = FakeSonarrClient(parse_episode_info_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}

        merged, skipped = mapper.assign(pending, candidates, {(1, 12): 2586, (1, 13): 2587})

        assert normalize_basename(v2) not in merged
        assert normalize_basename(v2) in skipped

    def test_moved_out_seeded_sharer_still_vetoes(self) -> None:
        # The seeded v1 already imported and MOVED OUT of the folder; its
        # name still parses (Sonarr's /parse is name-based), so the tell must
        # keep seeing absolute 12 and refuse the v2 the spare id.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {
            v1: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        sonarr = FakeSonarrClient(parse_episode_info_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2999],
            ordered_episode_ids=[2586, 2999],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(v2): _cand(v2)}  # v1 is gone from disk

        merged, skipped = mapper.assign(pending, candidates, {(1, 12): 2586})

        assert normalize_basename(v2) not in merged
        assert normalize_basename(v2) in skipped

    def test_none_parse_v2_never_rides_the_single_file_fallback(self) -> None:
        # The blip lands on the v2 itself: no parse at all is no evidence, so
        # the spare id stays open rather than going to a likely duplicate.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {v1: _pinfo(season=0, absolutes=(12,), matched=((1, 12),))}
        sonarr = FakeSonarrClient(parse_episode_info_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}

        merged, skipped = mapper.assign(pending, candidates, {(1, 12): 2586})

        assert normalize_basename(v2) not in merged
        assert normalize_basename(v2) in skipped

    def test_empty_resolved_set_skips_absolute_only_files(self) -> None:
        # With NO resolved set, the absolute leg has nothing to index into, so an
        # absolute-only pack (Overlord-style "- 01".."- 03") is left for the user
        # rather than guessed - absolute numbers are never trusted to decide identity
        # on their own (the To Love-Ru safety posture).
        files = [f"{n:02d}.mkv" for n in range(1, 4)]
        parsed = {name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = assign_episode_ids(files, parsed, [], {})

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(files)

    def test_single_numberless_file_single_target_is_placed(self) -> None:
        # Degenerate positional: one leftover file, one leftover episode, and
        # Sonarr SAW the name and found no number -> it's that one (the
        # single-file fallback, resolved-set form).
        result = assign_episode_ids(["only.mkv"], {"only.mkv": ParsedFileInfo()}, [900], {})

        assert result.assigned == {"only.mkv": [900]}
        assert result.skipped == []

    def test_single_none_parse_single_target_is_refused(self) -> None:
        # FLIPPED by review 2026-07-18: a None parse is no evidence at all (a
        # blipped v2's absolute may be hiding behind it), so refuse and let
        # the next poll decide - an unparseable name comes back as an
        # all-empty parse, not None, and still places above.
        result = assign_episode_ids(["only.mkv"], {"only.mkv": None}, [900], {})

        assert result.assigned == {}
        assert result.skipped == ["only.mkv"]

    def test_mixed_exact_then_leftover_absolute(self) -> None:
        # One file names its season (placed by leg 1); the remaining absolute file
        # maps onto the one leftover id.
        parsed = {
            "s01e01.mkv": _pinfo(season=1, episodes=(1,)),
            "extra.mkv": _pinfo(absolutes=(2,)),
        }
        ep_id_map = {(1, 1): 8033}

        result = assign_episode_ids(
            ["s01e01.mkv", "extra.mkv"],
            parsed,
            [8033, 8044],
            ep_id_map,
        )

        assert result.assigned == {"s01e01.mkv": [8033], "extra.mkv": [8044]}
        assert result.skipped == []


# --------------------------------------------------------------------------- #
# classify_queue on the real captured queue
# --------------------------------------------------------------------------- #
class TestClassifyRealQueue:
    """The real queue had a paused download (wait) + two importBlocked (step in)."""

    @staticmethod
    def _states_by_download() -> dict[str, list[str]]:
        body: dict[str, list[dict[str, object]]] = load_fixture("queue.json")
        states: dict[str, list[str]] = {}
        for rec in body["records"]:
            state = rec.get("trackedDownloadState", "")
            download_id = rec["downloadId"]
            states.setdefault(download_id if isinstance(download_id, str) else "", []).append(
                state if isinstance(state, str) else "",
            )
        return states

    def test_import_blocked_steps_in(self) -> None:
        states = self._states_by_download()
        yamada = states["1111111111111111111111111111111111111111"]
        assert classify_queue(yamada) is QueueVerdict.STEP_IN

    def test_paused_download_waits(self) -> None:
        states = self._states_by_download()
        paused = states["B7640FF13A2ADCA981B821D03CEBD1B569798459"]
        assert classify_queue(paused) is QueueVerdict.WAIT


# --------------------------------------------------------------------------- #
# PendingImport round-trip carries the new resolved set (with back-compat)
# --------------------------------------------------------------------------- #
class TestPendingImportOrderedIds:
    """`ordered_episode_ids` round-trips through JSON; a legacy record missing the key rehydrates to an empty list."""

    def test_round_trip_preserves_ordered_episode_ids(self) -> None:
        rec = pending_import(ordered_episode_ids=[8030, 8031, 8032])
        from pearlarr.manual_import import PendingImport

        again = PendingImport.from_json(rec.to_json())
        assert again.ordered_episode_ids == [8030, 8031, 8032]
        assert again == rec

    def test_legacy_record_without_ordered_ids_rehydrates_empty(self) -> None:
        from pearlarr.manual_import import PendingImport

        raw = pending_import().to_json()
        del raw["ordered_episode_ids"]
        assert PendingImport.from_json(raw).ordered_episode_ids == []


# --------------------------------------------------------------------------- #
# CommandResource.model_validate on the real captured /api/v3/command list
# --------------------------------------------------------------------------- #
# The capture is the bug-2 evidence: stacked ManualImport commands sharing one
# downloadId (a duplicate-import loop), plus a folder import with no downloadId
# and a non-ManualImport command. Scrubbed for the public fixture (infohash +
# server path root), matching the rest of tests/fixtures/sonarr/.
_SAO_DOWNLOAD_ID = "3333333333333333333333333333333333333333"


class TestCommandResourceFixture:
    """CommandResource.model_validate parses name / status / message / body.files."""

    @staticmethod
    def _commands() -> list[CommandResource]:
        raw: list[dict[str, object]] = load_fixture("command_list.json")
        return [CommandResource.model_validate(c) for c in raw]

    def test_started_manual_import_parses_message_and_files(self) -> None:
        started = next(c for c in self._commands() if c.name == "ManualImport" and c.status == "started")
        assert started.message == "Processing file 4 of 8"
        assert started.files  # body.files were parsed
        first = started.files[0]
        assert first.download_id == _SAO_DOWNLOAD_ID
        assert first.series_id == 169
        assert first.episode_ids == (6605,)

    def test_completed_manual_import_parses(self) -> None:
        completed = next(c for c in self._commands() if c.status == "completed")
        assert completed.name == "ManualImport"
        assert completed.message == "Manually imported 10 files"
        assert completed.result == "successful"

    def test_folder_import_has_no_download_id(self) -> None:
        # The Tensei Vodes season-pack import is folder-based: its files carry a
        # folderName + path but NO downloadId, so the guard must fall back to path.
        folder = next(c for c in self._commands() if c.files and c.files[0].series_id == 153)
        assert folder.files[0].download_id is None
        assert "Vodes" in (folder.files[0].path or "")

    def test_non_manual_import_command_parsed_without_files(self) -> None:
        proc = next(c for c in self._commands() if c.name == "ProcessMonitoredDownloads")
        assert proc.files == ()


class TestManualImportInFlightFixture:
    """manual_import_in_flight reads the real command list to close the loop."""

    @staticmethod
    def _commands() -> list[CommandResource]:
        raw: list[dict[str, object]] = load_fixture("command_list.json")
        return [CommandResource.model_validate(c) for c in raw]

    def test_matching_download_id_is_in_flight(self) -> None:
        # The SAO download has a started + queued ManualImport sharing its
        # downloadId -> a fresh import for it would stack a duplicate.
        assert manual_import_in_flight(
            self._commands(),
            _SAO_DOWNLOAD_ID,
            ContentPaths(raw="/downloads", sonarr_visible="/downloads"),
            set(),
        )

    def test_unrelated_download_id_is_not_in_flight(self) -> None:
        # A different infohash with no path/episode overlap -> proceed.
        assert not manual_import_in_flight(
            self._commands(),
            "ffffffffffffffffffffffffffffffffffffffff",
            ContentPaths(raw="/nowhere", sonarr_visible="/nowhere"),
            set(),
        )

    def test_folder_import_matches_by_episode_id(self) -> None:
        # The Vodes folder import carries no downloadId; episode 5645 is ours.
        assert manual_import_in_flight(
            self._commands(),
            "no-such-hash",
            ContentPaths(raw="/nowhere", sonarr_visible="/nowhere"),
            {5645},
        )


# --------------------------------------------------------------------------- #
# End-to-end: the real Yamada failure now imports to the resolved S00 ids
# --------------------------------------------------------------------------- #
def _yamada_parse_side_effect(raw_base: str) -> ParsedFileInfo | None:
    """Replay the captured /parse bodies for the two Yamada specials by basename."""

    if "S00E01" in raw_base:
        body: dict[str, object] = load_fixture("parse_yamada_s00e01.json")
        return ParsedFileInfo.model_validate(body)
    if "S00E02" in raw_base:
        body = load_fixture("parse_yamada_s00e02.json")
        return ParsedFileInfo.model_validate(body)
    return None


def _yamada_strat(config: AppConfig | None = None) -> tuple[SonarrSync, FakeSonarrClient, list[str]]:
    """The real Yamada fixtures wired into a bare SonarrSync + its scripted fake.

    Returns the strategy, its scripted `FakeSonarrClient` (replaying the captured
    episode list / manual-import candidates / per-file parse), and the on-disk
    basenames. `config` overrides the default (e.g. to flip `imports.mode`).
    """

    episodes_raw: list[dict[str, object]] = load_fixture("episodes_213_yamada.json")
    episodes = [SonarrEpisode.model_validate(e) for e in episodes_raw]
    candidates_raw: list[dict[str, object]] = load_fixture("manualimport_yamada.json")
    candidates = [ManualImportCandidate.model_validate(c) for c in candidates_raw]
    seadex_files = [c.path.rsplit("/", 1)[-1] for c in candidates if c.path]

    sonarr = FakeSonarrClient(
        queue=[],  # not tracked -> STEP_IN
        episodes=episodes,
        candidates=candidates,
        parse_episode_info_fn=_yamada_parse_side_effect,
        refresh_count=7,
        command_status=CommandResource(status="completed"),
        quality_defs=[],
        languages=[],
        execute_command_id=99,
    )

    strat = make_sonarr_sync(
        sonarr=sonarr,
        config=config or make_config(),
        cache_store=FakeCacheStore(),
    )
    return strat, sonarr, seadex_files


class TestYamadaEndToEnd:
    """Drive import_completed with the real fixtures for the failing queue item."""

    def test_specials_import_to_resolved_episode_ids(self) -> None:
        strat, sonarr, seadex_files = _yamada_strat()

        # Resolved set = the entry's S00 episodes (8030, 8031, 8032); the torrent
        # only carries E01/E02, so only those two get placed.
        pending = pending_import(
            infohash="1111111111111111111111111111111111111111",
            series_id=213,
            title="Yamada-kun and the Seven Witches",
            release_group="Headpatter",
            file_episode_map={},  # the real grab-time failure: nothing seeded
            episode_ids=[],
            ordered_episode_ids=[8030, 8031, 8032],
            seadex_files=seadex_files,
        )

        probe = strat.import_completed(pending, "/downloads/yamada")

        # The command was issued (copy is async -> RETRY + command_issued).
        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert len(sonarr.execute_calls) == 1
        # The configured import mode is threaded onto the execute command (default
        # "auto"; "move" deletes the source files, so a wrong mode must not be silent).
        assert sonarr.execute_calls[0][1] == "auto"

        files = sonarr.execute_calls[0][0]
        assigned = {f.episodeIds[0]: f for f in files}
        assert set(assigned) == {8030, 8031}
        assert all(f.seriesId == 213 for f in files)

    def test_import_mode_propagates_from_config(self) -> None:
        # imports.mode flows through to manual_import_execute - a regression that
        # hardcoded/ignored it (e.g. "move" -> source-file deletion) would be invisible
        # without this. Flip the config and assert the configured mode reaches Sonarr.
        strat, sonarr, seadex_files = _yamada_strat(make_config(import_mode="move"))

        pending = pending_import(
            infohash="1111111111111111111111111111111111111111",
            series_id=213,
            title="Yamada-kun and the Seven Witches",
            release_group="Headpatter",
            file_episode_map={},
            episode_ids=[],
            ordered_episode_ids=[8030, 8031, 8032],
            seadex_files=seadex_files,
        )

        strat.import_completed(pending, "/downloads/yamada")

        assert len(sonarr.execute_calls) == 1
        assert sonarr.execute_calls[0][1] == "move"

    def test_import_completed_probe_carries_seed_complete_counts(self) -> None:
        # A complete seed map -> the probe carries the determinate "files inserted"
        # counts (none landed yet here -> 0 / N), pinned to the seed set.
        strat, _sonarr, seadex_files = _yamada_strat()
        ep_map = {name: [8030 + i] for i, name in enumerate(seadex_files)}
        pending = pending_import(
            infohash="2222222222222222222222222222222222222222",
            series_id=213,
            release_group="Headpatter",
            file_episode_map=ep_map,
            episode_ids=[],
            ordered_episode_ids=[v[0] for v in ep_map.values()],
            seadex_files=seadex_files,
        )

        probe = strat.import_completed(pending, "/downloads/yamada")

        assert probe.target_count == len(seadex_files)
        assert probe.imported_count == 0

    def test_import_progress_is_read_only_and_counts_seed_targets(self) -> None:
        # The Tier-2 fast poll: a determinate count over the seed targets, reading
        # ONLY the episode files - never the refresh / queue / execute pipeline.
        strat, sonarr, seadex_files = _yamada_strat()
        ep_map = {name: [8030 + i] for i, name in enumerate(seadex_files)}
        pending = pending_import(
            infohash="4444444444444444444444444444444444444444",
            series_id=213,
            release_group="Headpatter",
            file_episode_map=ep_map,
            episode_ids=[],
            ordered_episode_ids=[v[0] for v in ep_map.values()],
            seadex_files=seadex_files,
        )

        progress = strat.import_progress(pending)

        assert progress.determinate is True
        assert progress.total == len(seadex_files)
        assert progress.done == 0  # no episode holds a recommended file yet
        assert sonarr.episodes_calls  # the one read it does make
        assert sonarr.execute_calls == []
        assert sonarr.refresh_calls == 0
        assert sonarr.queue_calls == 0

    def test_import_progress_indeterminate_when_seed_map_incomplete(self) -> None:
        # No (or partial) seed map -> indeterminate zero, and it never even fetches:
        # the importing row stays a spinner, promotion is left to the heavy poll.
        strat, sonarr, seadex_files = _yamada_strat()
        pending = pending_import(
            infohash="3333333333333333333333333333333333333333",
            series_id=213,
            release_group="Headpatter",
            file_episode_map={},  # the real grab-time gap
            episode_ids=[],
            ordered_episode_ids=[8030, 8031, 8032],
            seadex_files=seadex_files,
        )

        progress = strat.import_progress(pending)

        assert progress == ImportProgress(0, 0, determinate=False)
        assert sonarr.episodes_calls == []
        assert sonarr.execute_calls == []

    def test_specials_import_with_empty_resolved_set(self) -> None:
        # THE headline regression: the ACTUAL on-disk stuck record is pre-fix - EMPTY
        # everything (no ordered_episode_ids, no seed map). Before the fix this fell
        # to the legacy path, mapped nothing (Sonarr's series-matched episodes are
        # empty), and retried forever. Now the empty-set exact fallback places the
        # two specials onto the live series episodes, so it imports with no re-grab.
        strat, sonarr, seadex_files = _yamada_strat()

        pending = pending_import(
            infohash="1111111111111111111111111111111111111111",
            series_id=213,
            title="Yamada and the Seven Witches (OVA)",
            release_group="Headpatter",
            file_episode_map={},
            episode_ids=[],
            ordered_episode_ids=[],  # the pre-fix stuck record
            seadex_files=seadex_files,
        )

        probe = strat.import_completed(pending, "/downloads/yamada")

        assert probe.readiness is ImportReadiness.RETRY
        assert probe.command_issued is True
        assert len(sonarr.execute_calls) == 1
        assert sonarr.execute_calls[0][1] == "auto"

        files = sonarr.execute_calls[0][0]
        assigned = {f.episodeIds[0]: f for f in files}
        assert set(assigned) == {8030, 8031}
        assert all(f.seriesId == 213 for f in files)
