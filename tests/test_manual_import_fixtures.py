# pyright: strict
"""Real-API-fixture tests for the resolved-mapping manual import.

These pin the behavior the *old* code got wrong, using JSON captured verbatim
from a live Sonarr (`tests/fixtures/sonarr/`). The headline failure that
motivated the rewrite: a specials/alias release Sonarr can't match to a series
(its title carries a year suffix the release lacks) returns an empty
series-*matched* `episodes` array, so the import silently mapped nothing. The
fix reads the series-*agnostic* `parsedEpisodeInfo` and assigns it into OUR
resolved episode set - identity comes from the same mapping the add flow
already trusts. Sonarr's title match only informs, in-set (`matched_episodes`).

The pure `assign_episode_ids` tests encode the cases raised during triage
(correctly-named specials, mis-numbered specials, a multi-season pack) plus one
class per placement pass. The end-to-end test drives the real fixtures through
`import_completed`.
"""

import json
from pathlib import Path
from typing import ClassVar

from pearlarr.config import AppConfig
from pearlarr.manual_import import (
    AttemptKind,
    ImportProgress,
    PendingImport,
    normalize_basename,
)
from pearlarr.seadex_sonarr import SonarrSync
from pearlarr.seadex_types import (
    CommandResource,
    EpisodeKey,
    ManualImportCandidate,
    MatchedEpisode,
    ParsedFileInfo,
    QualityDefinition,
    QualitySource,
    QueueRecord,
    SonarrEpisode,
)
from pearlarr.sonarr_import_plan import (
    CandidateFile,
    ContentPaths,
    DownloadMatch,
    EpisodeAssignment,
    EpisodeIndex,
    ParsedQuality,
    PendingSeedContext,
    Placement,
    PlacementBatch,
    PlacementVerdict,
    QueueVerdict,
    SeedFile,
    SeedRelease,
    SeedScope,
    TargetScope,
    assign_episode_ids,
    build_pending_seed,
    classify_queue,
    episode_index,
    manual_import_in_flight,
    parse_se_from_filename,
    quality_axes_from_model,
    resolve_quality,
    started_disk_commands,
)

from .builders import (
    FakeCacheStore,
    make_config,
    make_sonarr_mapper,
    make_sonarr_sync,
    pending_import,
    sonarr_ep,
    url_item,
)
from .fakes import FakeSonarrClient

_FIXTURES = Path(__file__).parent / "fixtures" / "sonarr"


def load_fixture[T](name: str, _shape: type[T] | None = None) -> T:
    """Parse one captured Sonarr response, typed by the call site's annotation.

    `_shape` is unused at runtime. It gives `T` a second occurrence so pyright
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


def _verdicts(result: EpisodeAssignment) -> dict[str, PlacementVerdict]:
    """One assignment keyed name -> verdict, for the tests that pin the classification."""

    return {p.name: p.verdict for p in result.placements}


# --------------------------------------------------------------------------- #
# Quality resolution - the (source, resolution) match, on real bodies
# --------------------------------------------------------------------------- #
class TestQualityResolution:
    """The quality fix's load-bearing claims.

    Quality is matched by the structured `(source, resolution)` pair. The
    candidate-read test runs on a verbatim live-Sonarr capture. The
    qualitydefinition list is a hand-authored STAND-IN (`qualitydefinitions.json`)
    mirroring real Sonarr - the live `/api/v3/qualitydefinition` capture is owed
    (your instance sits behind an auth proxy). Dropping a real capture in
    place of the stand-in re-runs these against reality unchanged.
    """

    def test_qualitydefinition_fixture_has_the_shape_the_matcher_needs(self) -> None:
        # CONTRACT, not validation: the matcher keys on (source, resolution), so
        # every definition must carry both. This guards the stand-in (and any real
        # capture swapped in for it) - it does NOT by itself prove the live
        # instance serializes the fields. That capture is still owed.
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
        # (blurayRaw, 1080). Matched against the full definition list, that pair
        # must resolve to the "Bluray-1080p Remux" definition (valid id+name) -
        # never omitted.
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

    def test_special_has_season_episode_despite_no_series_match(self) -> None:
        body: dict[str, object] = load_fixture("parse_yamada_s00e01.json")
        # The OLD code read this (series-matched) array and got nothing:
        assert body["episodes"] == []

        info = ParsedFileInfo.model_validate(body)
        assert info.season_number == 0
        assert info.episode_numbers == (1,)
        assert info.absolute_episode_numbers == ()

    def test_absolute_numbered_file_reports_absolute_not_season_episode(self) -> None:
        body: dict[str, object] = load_fixture("parse_glimmerzu_abs14.json")
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
# assign_episode_ids - the three cases raised during triage, plus guards
# --------------------------------------------------------------------------- #
class TestAssignExactSeason:
    """Leg 1: a correctly-named file Sonarr just couldn't match to the series."""

    def test_specials_assigned_by_exact_season_episode(self) -> None:
        # Resolved set is the entry's S00 episodes (ids 8030..8032). The two files
        # carry S00E01 / S00E02 and land on 8030 / 8031.
        files = ["s00e01.mkv", "s00e02.mkv"]
        parsed = {
            "s00e01.mkv": _pinfo(season=0, episodes=(1,)),
            "s00e02.mkv": _pinfo(season=0, episodes=(2,)),
        }
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(0, 2): 8031, EpisodeKey(0, 3): 8032, EpisodeKey(1, 1): 8033}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([8030, 8031, 8032], ep_id_map))

        assert result.assigned == {"s00e01.mkv": [8030], "s00e02.mkv": [8031]}
        assert (result.skipped, result.excluded) == ((), ())

    def test_exact_parse_outside_resolved_set_is_foreign(self) -> None:
        # File parses to S01E01 (id 8033) but the resolved set is only S00 -> never
        # imported (the over-grab guard: identity must land INSIDE our set). The map
        # knows the key, so the reading is COMPLETE and proves the file another slice's.
        parsed = {"x.mkv": _pinfo(season=1, episodes=(1,))}
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(1, 1): 8033}

        result = assign_episode_ids(PlacementBatch(["x.mkv"], parsed), TargetScope([8030], ep_id_map))

        assert result.assigned == {}
        assert result.skipped == ()
        assert _verdicts(result) == {"x.mkv": PlacementVerdict.FOREIGN}

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
        ep_id_map = {EpisodeKey(0, 1): 8030, EpisodeKey(0, 2): 8031, EpisodeKey(0, 3): 8032}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([], ep_id_map))

        assert result.assigned == {"s00e01.mkv": [8030], "s00e02.mkv": [8031]}
        assert (result.skipped, result.excluded) == ((), ())


class TestAssignAbsolute:
    """Leg 2: absolute-number index onto the resolved set."""

    def test_mis_numbered_specials_map_positionally(self) -> None:
        # The user's case: files on disk are "01".."05" but are really S00E05..E09.
        # The release numbers never decide identity - they only ORDER the files onto
        # the resolved set, so "01" -> the first resolved episode (8034 = S00E05).
        files = [f"{n:02d}.mkv" for n in range(1, 6)]
        parsed = {name: _pinfo(absolutes=(i + 1,)) for i, name in enumerate(files)}
        resolved = [8034, 8035, 8036, 8037, 8038]  # S00E05..E09 ids

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope(resolved, {}))

        assert result.skipped == ()
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

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope(resolved, {}))

        assert result.assigned == {
            "e1.mkv": [501],
            "e2.mkv": [502],
            "e3.mkv": [601],
            "e4.mkv": [602],
        }

    def test_no_signal_file_refuses_the_positional_leg(self) -> None:
        # A file whose parse yields nothing could be a hiccuped real episode.
        # The every-file check refuses the whole leg (skip + warn, retried).
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": _pinfo(absolutes=(1,)),
            "b.mkv": _pinfo(absolutes=(2,)),
            "menu.mkv": _pinfo(),  # a 200 /parse with null parsedEpisodeInfo
        }

        result = assign_episode_ids(PlacementBatch(["a.mkv", "b.mkv", "menu.mkv"], parsed), TargetScope([501, 502], {}))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv", "menu.mkv"]

    def test_absolute_ova_pack_maps_onto_resolved_set(self) -> None:
        # releases.moe/101083: 13 OVA files
        # named "- 01".."- 13", all parsed season 0 / absolute-only. The add flow
        # resolves this entry (anibridge tvdb_mappings {0: [(16, 28)]}) to the 13
        # season-0 episodes S00E16..E28 (live ids 2090..2102), so the absolute leg
        # places each file onto its season-sorted id (count-matched 13:13, no-dup) -
        # "- 01" -> S00E16, "- 13" -> S00E28. No grab-time change needed.
        files = [f"{n:02d}.mkv" for n in range(1, 14)]
        parsed = {name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}
        resolved = list(range(2090, 2103))  # S00E16..E28 ids, season order

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope(resolved, {}))

        assert result.skipped == ()
        assert result.assigned == {f"{n:02d}.mkv": [2089 + n] for n in range(1, 14)}


class TestAssignMatchedPairs:
    """Leg 1's matched-pairs fallback: Sonarr's series-matched `(season, episode)` for absolute-only names."""

    def test_multi_entry_batch_places_exactly_inside_the_set(self) -> None:
        # A batch spanning two entries plus a special, the record covering the second only:
        # in-set files place exactly and the two the map resolves OUTSIDE are another slice's.
        files = ["ep-11.mkv", "ep-12.mkv", "ep-13.mkv", "sp-17.5.mkv"]
        parsed: dict[str, ParsedFileInfo | None] = {
            "ep-11.mkv": _pinfo(season=0, absolutes=(11,), matched=((1, 11),)),
            "ep-12.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            "ep-13.mkv": _pinfo(season=0, absolutes=(13,), matched=((1, 13),)),
            # The 17.5 special: S00E01 in the NAME, so no matched fallback needed.
            "sp-17.5.mkv": _pinfo(season=0, episodes=(1,), matched=((0, 1),)),
        }
        ep_id_map = {EpisodeKey(1, 11): 2585, EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587, EpisodeKey(0, 1): 2574}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([2586, 2587], ep_id_map))

        assert result.assigned == {"ep-12.mkv": [2586], "ep-13.mkv": [2587]}
        assert result.skipped == ()
        assert {n: v for n, v in _verdicts(result).items() if v.excluded} == {
            "ep-11.mkv": PlacementVerdict.FOREIGN,
            "sp-17.5.mkv": PlacementVerdict.FOREIGN,
        }

    def test_name_parsed_pair_beats_matched_pair(self) -> None:
        # A name that carries its own (season, episode) never defers to
        # Sonarr's matched resolution.
        parsed = {"x.mkv": _pinfo(season=2, episodes=(5,), matched=((9, 9),))}
        ep_id_map = {EpisodeKey(2, 5): 400, EpisodeKey(9, 9): 999}

        result = assign_episode_ids(PlacementBatch(["x.mkv"], parsed), TargetScope([400, 999], ep_id_map))

        assert result.assigned == {"x.mkv": [400]}

    def test_matched_pairs_never_apply_unscoped(self) -> None:
        # With NO resolved set, the live-map fallback trusts a name-parsed pair
        # only - Sonarr's series match must not decide identity on its own.
        parsed = {"x.mkv": _pinfo(season=0, absolutes=(3,), matched=((1, 3),))}

        result = assign_episode_ids(PlacementBatch(["x.mkv"], parsed), TargetScope([], {EpisodeKey(1, 3): 300}))

        assert result.assigned == {}
        assert result.skipped == ("x.mkv",)

    def test_partially_in_set_matched_span_is_skipped(self) -> None:
        # A matched span reaching outside the resolved set is refused whole -
        # same half-import posture as the name-parsed leg.
        parsed = {"span.mkv": _pinfo(season=0, matched=((1, 1), (1, 3)))}
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 3): 503}

        result = assign_episode_ids(PlacementBatch(["span.mkv"], parsed), TargetScope([501, 502], ep_id_map))

        assert result.assigned == {}
        assert result.skipped == ("span.mkv",)

    def test_out_of_set_match_does_not_veto_the_single_file_fallback(self) -> None:
        # One numberless file, one leftover id: OUR resolution places it even
        # when Sonarr's title match claims an out-of-set episode.
        parsed = {"only.mkv": _pinfo(matched=((1, 5),))}

        result = assign_episode_ids(PlacementBatch(["only.mkv"], parsed), TargetScope([900], {EpisodeKey(1, 5): 555}))

        assert result.assigned == {"only.mkv": [900]}
        assert result.skipped == ()

    def test_wrong_series_matched_id_is_refused(self) -> None:
        # Sonarr matched some OTHER series whose numbers coincide with ours:
        # its episode id disagrees with our map, so the claim is refused.
        info = ParsedFileInfo(
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=999),),
        )

        result = assign_episode_ids(
            PlacementBatch(["x.mkv"], {"x.mkv": info}), TargetScope([501, 502], {EpisodeKey(1, 1): 501})
        )

        assert result.assigned == {}
        assert result.skipped == ("x.mkv",)

    def test_agreeing_matched_id_places(self) -> None:
        # The same claim with Sonarr's id AGREEING with our map places normally.
        info = ParsedFileInfo(
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=501),),
        )

        result = assign_episode_ids(
            PlacementBatch(["x.mkv"], {"x.mkv": info}), TargetScope([501, 502], {EpisodeKey(1, 1): 501})
        )

        assert result.assigned == {"x.mkv": [501]}

    def test_duplicate_matched_pairs_collapse_to_one_claim(self) -> None:
        # Junk wire duplicates of the same pair are one claim, not a veto.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1),
            ),
        )

        result = assign_episode_ids(
            PlacementBatch(["x.mkv"], {"x.mkv": info}), TargetScope([501, 502], {EpisodeKey(1, 1): 501})
        )

        assert result.assigned == {"x.mkv": [501]}

    def test_mixed_id_duplicate_claims_place_once(self) -> None:
        # (s,e,None) and (s,e,id) survive the triple dedup as two claims. The
        # wire list still carries the episode id once. Two resolved ids keep
        # the degenerate arm out, so this pins leg 1 itself.
        info = ParsedFileInfo(
            matched_episodes=(
                MatchedEpisode(season_number=1, episode_number=1),
                MatchedEpisode(season_number=1, episode_number=1, id=501),
            ),
        )

        result = assign_episode_ids(
            PlacementBatch(["x.mkv"], {"x.mkv": info}), TargetScope([501, 502], {EpisodeKey(1, 1): 501})
        )

        assert result.assigned == {"x.mkv": [501]}

    def test_wrong_id_match_cannot_veto_the_single_file_fallback(self) -> None:
        # A disagreeing-id match refuses the CLAIM, but with one numberless
        # file and one leftover id the degenerate fallback still places the
        # only possible way (same posture as the out-of-set variant above).
        info = ParsedFileInfo(
            matched_episodes=(MatchedEpisode(season_number=1, episode_number=1, id=999),),
        )

        result = assign_episode_ids(
            PlacementBatch(["only.mkv"], {"only.mkv": info}), TargetScope([501], {EpisodeKey(1, 1): 501})
        )

        assert result.assigned == {"only.mkv": [501]}

    def test_full_season_parse_never_borrows_matched_pairs(self) -> None:
        # Sonarr matches a bare "S01" extras file to EVERY season episode. One
        # junk file must not swallow the entry while the real files place.
        parsed: dict[str, ParsedFileInfo | None] = {
            "extras.mkv": _pinfo(matched=((1, 1), (1, 2)), full_season=True),
            "ep-01.mkv": _pinfo(season=0, absolutes=(1,), matched=((1, 1),)),
            "ep-02.mkv": _pinfo(season=0, absolutes=(2,), matched=((1, 2),)),
        }
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502}

        result = assign_episode_ids(
            PlacementBatch(["extras.mkv", "ep-01.mkv", "ep-02.mkv"], parsed), TargetScope([501, 502], ep_id_map)
        )

        assert result.assigned == {"ep-01.mkv": [501], "ep-02.mkv": [502]}
        assert result.skipped == ("extras.mkv",)

    def test_wide_matched_span_is_refused(self) -> None:
        # A 4-episode matched span exceeds what one file plausibly holds (the
        # season-pack shape without the fullSeason flag), so it never borrows.
        parsed = {"pack.mkv": _pinfo(matched=((1, 1), (1, 2), (1, 3), (1, 4)))}
        ep_id_map = {EpisodeKey(1, n): 500 + n for n in range(1, 5)}

        result = assign_episode_ids(PlacementBatch(["pack.mkv"], parsed), TargetScope([501, 502, 503, 504], ep_id_map))

        assert result.assigned == {}
        assert result.skipped == ("pack.mkv",)

    def test_triple_episode_matched_span_places(self) -> None:
        # The cap boundary: a triple-episode file's span is still a per-file claim.
        parsed = {"triple.mkv": _pinfo(matched=((1, 1), (1, 2), (1, 3)))}
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502, EpisodeKey(1, 3): 503}

        result = assign_episode_ids(PlacementBatch(["triple.mkv"], parsed), TargetScope([501, 502, 503], ep_id_map))

        assert result.assigned == {"triple.mkv": [501, 502, 503]}

    def test_junk_duplicates_beyond_the_cap_still_collapse_and_place(self) -> None:
        # The cap counts DISTINCT claims: four wire duplicates of one pair are
        # one claim, not a season-pack shape.
        parsed = {"x.mkv": _pinfo(matched=((1, 1), (1, 1), (1, 1), (1, 1)))}

        result = assign_episode_ids(PlacementBatch(["x.mkv"], parsed), TargetScope([501], {EpisodeKey(1, 1): 501}))

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
        ep_id_map = {EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502, EpisodeKey(1, 3): 503}

        result = assign_episode_ids(PlacementBatch(["x.mkv"], {"x.mkv": info}), TargetScope([501, 502, 503], ep_id_map))

        assert result.assigned == {"x.mkv": [501, 502, 503]}

    def test_partially_resolved_double_absolute_never_half_imports(self) -> None:
        # A "12-13" file whose match resolved only E12 (absolute 13 beyond
        # Sonarr's mapping): the borrowed span doesn't cover the absolutes,
        # so placing the resolved half is refused.
        parsed = {"d.mkv": _pinfo(season=0, absolutes=(12, 13), matched=((1, 12),))}

        result = assign_episode_ids(
            PlacementBatch(["d.mkv"], parsed), TargetScope([2586, 2587], {EpisodeKey(1, 12): 2586})
        )

        assert result.assigned == {}
        assert result.skipped == ("d.mkv",)

    def test_fully_resolved_double_absolute_places_both(self) -> None:
        # The same file with BOTH pairs resolved places as a two-episode file.
        parsed = {"d.mkv": _pinfo(season=0, absolutes=(12, 13), matched=((1, 12), (1, 13)))}
        ep_id_map = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}

        result = assign_episode_ids(PlacementBatch(["d.mkv"], parsed), TargetScope([2586, 2587], ep_id_map))

        assert result.assigned == {"d.mkv": [2586, 2587]}
        assert result.skipped == ()

    def test_matched_span_never_half_imports_via_the_single_file_fallback(self) -> None:
        # A file Sonarr says spans E01+E02 must not import as E01 alone via
        # the degenerate arm - cardinality evidence is honored even where
        # identity evidence is not (restored 1fc1d5e pin).
        parsed = {"span.mkv": _pinfo(matched=((1, 1), (1, 2)))}

        result = assign_episode_ids(
            PlacementBatch(["span.mkv"], parsed), TargetScope([501], {EpisodeKey(1, 1): 501, EpisodeKey(1, 2): 502})
        )

        assert result.assigned == {}
        assert result.skipped == ("span.mkv",)

    def test_full_season_file_never_takes_the_spare_id(self) -> None:
        # Leg 1 quarantines the season-pack shape. The degenerate arm must
        # not hand it the one spare id either.
        parsed: dict[str, ParsedFileInfo | None] = {
            "extras-s01.mkv": _pinfo(matched=((1, 1), (1, 2), (1, 3), (1, 4)), full_season=True),
            "e01.mkv": _pinfo(season=1, episodes=(1,)),
            "e02.mkv": _pinfo(season=1, episodes=(2,)),
            "e03.mkv": _pinfo(season=1, episodes=(3,)),
        }
        files = ["extras-s01.mkv", "e01.mkv", "e02.mkv", "e03.mkv"]
        ep_id_map = {EpisodeKey(1, n): 500 + n for n in range(1, 5)}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([501, 502, 503, 504], ep_id_map))

        assert result.assigned == {"e01.mkv": [501], "e02.mkv": [502], "e03.mkv": [503]}
        assert result.skipped == ("extras-s01.mkv",)


class TestAssignGuards:
    """Leg 3: refuse to guess - skip + warn instead."""

    def test_glimmerzu_per_title_restart_is_refused(self) -> None:
        # One torrent spanning two sub-series whose numbering BOTH restart at 1:
        # the shared absolutes are the tell of a season-boundary scramble, so the
        # whole absolute leg is refused rather than mis-assigned.
        main = {f"main-{i:02d}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 4)}
        eclipse = {f"eclipse-{i:02d}.mkv": _pinfo(absolutes=(i,)) for i in range(1, 4)}
        parsed = {**main, **eclipse}
        files = list(parsed)
        resolved = [501, 502, 503, 601, 602, 603]

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope(resolved, {}))

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


class TestDuplicateEvidence:
    """A collision is a duplicate only when the id's holder reads there too. Otherwise it is a skip to report."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}
    _KEYED = "show - S01E12 [1080p].mkv"

    @classmethod
    def _place(cls, seeded_parse: ParsedFileInfo | None) -> EpisodeAssignment:
        # The seed holds 2586 through "seeded.mkv"; the keyed on-disk file resolves there by name.
        parsed = {"seeded.mkv": seeded_parse, cls._KEYED: _pinfo(season=1, episodes=(12,))}
        return assign_episode_ids(
            PlacementBatch([cls._KEYED], parsed), TargetScope([2586, 2587], cls._MAP, used=frozenset({2586}))
        )

    def test_a_seeded_holder_reading_the_same_episode_proves_the_duplicate(self) -> None:
        result = self._place(_pinfo(season=1, episodes=(12,)))

        assert _verdicts(result) == {self._KEYED: PlacementVerdict.DUPLICATE}

    def test_a_positionally_seeded_holder_leaves_a_skip_the_caller_reports(self) -> None:
        # The seed zipped a numberless file onto 2586. A file naming that episode outright disagrees
        # with it, and a disagreement is reported, never persisted as an exclusion.
        result = self._place(_pinfo())

        assert _verdicts(result) == {self._KEYED: PlacementVerdict.SKIPPED}
        assert result.excluded == ()


class TestAssignScopeGate:
    """CB3: the scope gate must key off the FULL resolved set, not the post-seed remainder."""

    def test_fully_seeded_scope_never_unlocks_the_live_map(self) -> None:
        # A fully seeded record (every resolved id used) keeps scope enforced: a
        # correctly-named but out-of-scope file is refused, NOT placed on the
        # live map. Only an EMPTY resolved set means "no scope at all".
        parsed = {"x.mkv": _pinfo(season=1, episodes=(1,))}
        ep_id_map = {EpisodeKey(1, 1): 8033}

        result = assign_episode_ids(
            PlacementBatch(["x.mkv"], parsed),
            TargetScope([8044], ep_id_map, used=frozenset({8044})),
        )

        assert result.assigned == {}
        assert _verdicts(result) == {"x.mkv": PlacementVerdict.FOREIGN}

    def test_fully_seeded_record_skips_out_of_scope_on_disk_leftover(self) -> None:
        # A fully-seeded record (every resolved episode already seeded) whose batch
        # folder also holds an OUT-OF-SCOPE file (a season-2 file in a season-1 grab).
        # The leftover must be skipped - not imported via the unscoped fallback -
        # and the grab-time seed map must stay un-contaminated by the placements.
        seed_name = "Show - 01 [1080p].mkv"
        leftover_name = "Show - S02E01 [1080p].mkv"
        pending = pending_import(
            file_episode_map={seed_name: [101]},
            episode_ids=[101],
            ordered_episode_ids=[101],
            seadex_files=[seed_name],
        )
        sonarr = FakeSonarrClient(parse_fn=lambda _f: _pinfo(season=2, episodes=(1,)))
        mapper = make_sonarr_mapper(sonarr=sonarr)

        candidates = {
            normalize_basename(seed_name): _cand(seed_name),
            normalize_basename(leftover_name): _cand(leftover_name),
        }
        ep_id_map = {EpisodeKey(1, 1): 101, EpisodeKey(2, 1): 999}  # 999 is OUTSIDE the resolved {101}

        result = mapper.assign(pending, candidates, ep_id_map)

        placed_ids = {i for ids in result.assigned.values() for i in ids}
        assert 999 not in placed_ids
        # The map resolves it to 999, outside the record's set, so it is another slice's.
        assert result.excluded == (Placement(normalize_basename(leftover_name), (), PlacementVerdict.FOREIGN),)
        assert result.placed == {}
        assert pending.file_episode_map == {seed_name: [101]}

    def test_count_mismatch_skips(self) -> None:
        # Two absolute files but three resolved ids -> not a clean 1:1 -> skip both.
        parsed = {"a.mkv": _pinfo(absolutes=(1,)), "b.mkv": _pinfo(absolutes=(2,))}

        result = assign_episode_ids(PlacementBatch(["a.mkv", "b.mkv"], parsed), TargetScope([1, 2, 3], {}))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv"]

    def test_hiccuped_episode_parse_refuses_the_leg(self) -> None:
        # A None-parse file that is really an EPISODE (parse hiccup, no SxxExx
        # fallback) refuses the leg. The next poll re-parses (misses uncached).
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": _pinfo(absolutes=(1,)),
            "b.mkv": _pinfo(absolutes=(2,)),
            "c.mkv": None,
        }

        result = assign_episode_ids(PlacementBatch(["a.mkv", "b.mkv", "c.mkv"], parsed), TargetScope([1, 2, 3], {}))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["a.mkv", "b.mkv", "c.mkv"]

    def test_multi_absolute_file_vetoes_the_leg(self) -> None:
        # A file spanning two absolutes ("01-02") can't be placed positionally,
        # so the leg stays refused.
        parsed: dict[str, ParsedFileInfo | None] = {
            "span.mkv": _pinfo(absolutes=(1, 2)),
            "c.mkv": _pinfo(absolutes=(3,)),
        }

        result = assign_episode_ids(PlacementBatch(["span.mkv", "c.mkv"], parsed), TargetScope([1, 2, 3], {}))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["c.mkv", "span.mkv"]

    def test_out_of_set_absolute_cannot_fill_in_for_a_hiccuped_episode(self) -> None:
        # Reviewer-reproduced hazard: an out-of-entry sibling (absolute 11,
        # matched out of set) must not fill the count for a hiccuped real E12.
        parsed: dict[str, ParsedFileInfo | None] = {
            "s-11.mkv": _pinfo(season=0, absolutes=(11,), matched=((1, 11),)),
            "e-12.mkv": None,
        }
        ep_id_map = {EpisodeKey(1, 11): 2585, EpisodeKey(1, 12): 2586}

        result = assign_episode_ids(PlacementBatch(["s-11.mkv", "e-12.mkv"], parsed), TargetScope([2586], ep_id_map))

        assert result.assigned == {}
        # The sibling resolves to 2585, outside the set, so it is excluded rather than counted.
        assert _verdicts(result) == {"s-11.mkv": PlacementVerdict.FOREIGN, "e-12.mkv": PlacementVerdict.SKIPPED}

    def test_v2_duplicate_of_a_placed_file_is_refused(self) -> None:
        # Leg 1 places "- 12" via its matched pair. The v2 shares absolute 12,
        # so the BATCH-wide duplicate tell refuses the positional leg for it.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        ep_id_map = {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587}

        result = assign_episode_ids(
            PlacementBatch(["e-12.mkv", "e-12v2.mkv"], parsed), TargetScope([2586, 2587], ep_id_map)
        )

        assert result.assigned == {"e-12.mkv": [2586]}
        assert _verdicts(result)["e-12v2.mkv"] == PlacementVerdict.DUPLICATE

    def test_seeded_sharer_still_vetoes_the_positional_leg(self) -> None:
        # The v1 was placed on an EARLIER poll (seeded, not in ordered_files). Its parse
        # still reaches the duplicate tell, so the v2 stays refused. The map is unserved for
        # the pair, so nothing resolves and the COUNT leg is the only thing that can decide.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        batch = PlacementBatch(["e-12v2.mkv"], parsed)

        result = assign_episode_ids(batch, TargetScope([2587], {}))

        assert result.assigned == {}
        assert _verdicts(result) == {"e-12v2.mkv": PlacementVerdict.SKIPPED}
        # The control: drop the sharer and the very same file takes the spare id.
        alone = assign_episode_ids(
            PlacementBatch(["e-12v2.mkv"], {"e-12v2.mkv": parsed["e-12v2.mkv"]}), TargetScope([2587], {})
        )
        assert alone.assigned == {"e-12v2.mkv": [2587]}

    def test_blipped_batch_parse_refuses_the_positional_leg(self) -> None:
        # A tell-only parse the caller couldn't get may be hiding a duplicate:
        # the leg fails CLOSED, like a hiccuped leftover already does.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12.mkv": None,
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(PlacementBatch(["e-12v2.mkv"], parsed), TargetScope([2587], {}))

        assert result.assigned == {}
        assert _verdicts(result) == {"e-12v2.mkv": PlacementVerdict.SKIPPED}

    def test_offline_fallback_parse_refuses_the_positional_leg(self) -> None:
        # The offline SxxExx stand-in knows nothing about absolutes: a
        # dual-numbered seeded sharer must not launder its lost "12" into a
        # known parse and unlock the leg.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-s01e12.mkv": _pinfo(season=1, episodes=(12,), offline=True),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(PlacementBatch(["e-12v2.mkv"], parsed), TargetScope([2587], {}))

        assert result.assigned == {}
        assert _verdicts(result) == {"e-12v2.mkv": PlacementVerdict.SKIPPED}

    def test_junk_duplicate_absolute_within_one_parse_does_not_veto(self) -> None:
        # One parse repeating its own absolute ((12, 12)) is wire junk, not a
        # restart tell - the unrelated leftover still places.
        parsed: dict[str, ParsedFileInfo | None] = {
            "seeded-12.mkv": _pinfo(season=0, absolutes=(12, 12)),
            "left-13.mkv": _pinfo(season=0, absolutes=(13,)),
        }

        result = assign_episode_ids(PlacementBatch(["left-13.mkv"], parsed), TargetScope([507], {}))

        assert result.assigned == {"left-13.mkv": [507]}
        assert result.skipped == ()

    def test_multi_absolute_seeded_sharer_still_vetoes(self) -> None:
        # A seeded "12-13" span file shares absolute 12 with the leftover v2:
        # every absolute of every parse is counted, so the duplicate shows.
        parsed: dict[str, ParsedFileInfo | None] = {
            "e-12-13.mkv": _pinfo(season=0, absolutes=(12, 13)),
            "e-12v2.mkv": _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }

        result = assign_episode_ids(PlacementBatch(["e-12v2.mkv"], parsed), TargetScope([2588], {}))

        assert result.assigned == {}
        assert _verdicts(result) == {"e-12v2.mkv": PlacementVerdict.SKIPPED}

    def test_assign_returns_placements_without_touching_the_record(self) -> None:
        # The mapper reports its fresh placements for the caller to persist;
        # the frozen record's map is never mutated behind it.
        name = "Show - S01E01 [1080p].mkv"
        sonarr = FakeSonarrClient(parse_fn=lambda _f: _pinfo(season=1, episodes=(1,)))
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(file_episode_map={}, episode_ids=[], ordered_episode_ids=[101], seadex_files=[name])
        candidates = {normalize_basename(name): _cand(name)}

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 1): 101})

        assert result.placed == result.assigned == {normalize_basename(name): [101]}
        assert pending.file_episode_map == {}

    def test_placed_excludes_the_seeded_entries(self) -> None:
        # `placed` is this poll's fresh work alone: a seeded entry rides
        # `assigned` only, so the seam never re-persists what the record holds.
        seed_name, leftover_name = "Show - S01E01 [1080p].mkv", "Show - S01E02 [1080p].mkv"
        parses = {
            seed_name: _pinfo(season=1, episodes=(1,)),
            leftover_name: _pinfo(season=1, episodes=(2,)),
        }
        sonarr = FakeSonarrClient(parse_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={seed_name: [101]},
            episode_ids=[],
            ordered_episode_ids=[101, 102],
            seadex_files=[seed_name, leftover_name],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (seed_name, leftover_name)}

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 1): 101, EpisodeKey(1, 2): 102})

        assert result.placed == {normalize_basename(leftover_name): [102]}
        assert result.assigned == {
            normalize_basename(seed_name): [101],
            normalize_basename(leftover_name): [102],
        }

    def test_placed_sharer_still_vetoes_on_the_next_poll(self) -> None:
        # Poll 1 places the v1 and the record seam folds it onto the record.
        # Poll 2 must not let the now-seeded v1 hide the shared absolute from
        # the tell.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {
            v1: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        sonarr = FakeSonarrClient(parse_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}
        ep_id_map = {EpisodeKey(1, 12): 2586}

        first = mapper.assign(pending, candidates, ep_id_map)
        second = mapper.assign(pending.with_placements(first.placed), candidates, ep_id_map)

        assert first.placed == {normalize_basename(v1): [2586]}
        assert normalize_basename(v2) not in second.assigned
        assert second.excluded == (Placement(normalize_basename(v2), (), PlacementVerdict.DUPLICATE),)
        assert pending.file_episode_map == {}

    def test_seeded_sharer_parse_blip_fails_closed(self) -> None:
        # A LATER RUN (fresh parse cache): the seeded v1's /parse blips to
        # None, so the tell's input is incomplete: the v2 must stay refused,
        # not slide onto the other episode. Nothing proves it a duplicate
        # either (the v1 read nothing), so it is a skip, re-asked next poll.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),))}
        sonarr = FakeSonarrClient(parse_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 12): 2586})

        assert normalize_basename(v2) not in result.assigned
        assert result.skipped == (normalize_basename(v2),)
        assert not result.settled

    def test_seeded_dual_numbered_sharer_offline_fallback_fails_closed(self) -> None:
        # The seeded v1 is dual-numbered. Its /parse blips and the offline
        # SxxExx fallback loses the absolute - the tell must treat that
        # stand-in as unknown, not let the v2 slide onto the spare id.
        v1, v2 = "Show - S01E12 - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),))}
        sonarr = FakeSonarrClient(parse_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 12): 2586, EpisodeKey(1, 13): 2587})

        assert normalize_basename(v2) not in result.assigned
        assert result.excluded == (Placement(normalize_basename(v2), (), PlacementVerdict.DUPLICATE),)

    def test_moved_out_seeded_sharer_still_vetoes(self) -> None:
        # The seeded v1 already imported and MOVED OUT of the folder. Its
        # name still parses (Sonarr's /parse is name-based), so the tell must
        # keep seeing absolute 12 and refuse the v2 the spare id.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {
            v1: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
            v2: _pinfo(season=0, absolutes=(12,), matched=((1, 12),)),
        }
        sonarr = FakeSonarrClient(parse_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2999],
            ordered_episode_ids=[2586, 2999],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(v2): _cand(v2)}  # v1 is gone from disk

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 12): 2586})

        assert normalize_basename(v2) not in result.assigned
        assert result.excluded == (Placement(normalize_basename(v2), (), PlacementVerdict.DUPLICATE),)

    def test_none_parse_v2_never_rides_the_single_file_fallback(self) -> None:
        # The blip lands on the v2 itself: no parse at all is no evidence, so
        # the spare id stays open rather than going to a likely duplicate.
        v1, v2 = "Show - 12 [1080p].mkv", "Show - 12v2 [1080p].mkv"
        parses = {v1: _pinfo(season=0, absolutes=(12,), matched=((1, 12),))}
        sonarr = FakeSonarrClient(parse_fn=parses.get)
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={v1: [2586]},
            episode_ids=[2586, 2587],
            ordered_episode_ids=[2586, 2587],
            seadex_files=[v1, v2],
        )
        candidates = {normalize_basename(name): _cand(name) for name in (v1, v2)}

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 12): 2586})

        assert normalize_basename(v2) not in result.assigned
        assert normalize_basename(v2) in result.skipped

    def test_empty_resolved_set_skips_absolute_only_files(self) -> None:
        # With NO resolved set, the absolute leg has nothing to index into, so an
        # absolute-only pack ("- 01".."- 03") is left for manual
        # placement rather than guessed - absolute numbers are never trusted to
        # decide identity on their own (the To Glimmer-Zu safety posture).
        files = [f"{n:02d}.mkv" for n in range(1, 4)]
        parsed = {name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(files)}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([], {}))

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(files)

    def test_single_numberless_file_single_target_is_placed(self) -> None:
        # Degenerate positional: one leftover file, one leftover episode, and
        # Sonarr SAW the name and found no number -> it's that one (the
        # single-file fallback, resolved-set form).
        result = assign_episode_ids(
            PlacementBatch(["only.mkv"], {"only.mkv": ParsedFileInfo()}), TargetScope([900], {})
        )

        assert result.assigned == {"only.mkv": [900]}
        assert result.skipped == ()

    def test_single_none_parse_single_target_is_refused(self) -> None:
        # A None parse is no evidence at all (a blipped v2's absolute may be
        # hiding behind it), so refuse and let the next poll decide - an
        # unparseable name comes back as an all-empty parse, not None, and
        # still places above.
        result = assign_episode_ids(PlacementBatch(["only.mkv"], {"only.mkv": None}), TargetScope([900], {}))

        assert result.assigned == {}
        assert result.skipped == ("only.mkv",)

    def test_mixed_exact_then_leftover_absolute(self) -> None:
        # One file names its season (placed by leg 1). The remaining absolute file
        # maps onto the one leftover id.
        parsed = {
            "s01e01.mkv": _pinfo(season=1, episodes=(1,)),
            "extra.mkv": _pinfo(absolutes=(2,)),
        }
        ep_id_map = {EpisodeKey(1, 1): 8033}

        result = assign_episode_ids(
            PlacementBatch(["s01e01.mkv", "extra.mkv"], parsed),
            TargetScope([8033, 8044], ep_id_map),
        )

        assert result.assigned == {"s01e01.mkv": [8033], "extra.mkv": [8044]}
        assert result.skipped == ()


class TestAssignDuplicateLeaves:
    """One basename in two folders collapses in the basename-keyed pool.

    Only one physical file can ever import, so the unmatched warning must
    follow the map: a placed name is never also reported skipped, and an
    unplaced one is reported once.
    """

    def test_placed_duplicate_leaf_is_not_reported_skipped(self) -> None:
        # The second occurrence of a placed name defers off the used-set and
        # used to land in skipped - the warning named a file that imported.
        name = "Show - 01 [1080p].mkv"
        sonarr = FakeSonarrClient(parse_fn=lambda _f: _pinfo(season=1, episodes=(1,)))
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={},
            episode_ids=[101],
            ordered_episode_ids=[101],
            seadex_files=[name, name],
        )
        candidates = {normalize_basename(name): _cand(name)}

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 1): 101})

        assert result.assigned == {normalize_basename(name): [101]}
        assert result.placed == {normalize_basename(name): [101]}
        assert result.skipped == ()

    def test_unplaced_duplicate_leaf_is_reported_once(self) -> None:
        # Both occurrences of an unplaceable duplicate refuse - the warning
        # names the leaf once, not once per folder.
        name = "Extra.mkv"
        sonarr = FakeSonarrClient(parse_fn=lambda _f: _pinfo())
        mapper = make_sonarr_mapper(sonarr=sonarr)
        pending = pending_import(
            file_episode_map={},
            episode_ids=[101, 102],
            ordered_episode_ids=[101, 102],
            seadex_files=[name, name],
        )
        candidates = {normalize_basename(name): _cand(name)}

        result = mapper.assign(pending, candidates, {})

        assert result.assigned == {}
        assert result.skipped == (normalize_basename(name),)


class TestAssignSettled:
    """`settled`: a skip is a verdict only when every parse was served and the episode index was."""

    @staticmethod
    def _numberless_pair() -> tuple[PendingImport, dict[str, CandidateFile]]:
        names = ("Movie Part 1.mkv", "Movie Part 2.mkv")
        pending = pending_import(
            file_episode_map={},
            episode_ids=[],
            ordered_episode_ids=[101],
            seadex_files=list(names),
        )
        return pending, {normalize_basename(name): _cand(name) for name in names}

    def test_a_parse_miss_leaves_the_skip_tentative(self) -> None:
        pending, candidates = self._numberless_pair()
        mapper = make_sonarr_mapper(sonarr=FakeSonarrClient(parse_fn=lambda _f: None))

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 1): 101})

        assert sorted(result.skipped) == sorted(candidates)
        assert result.settled is False

    def test_an_empty_episode_index_leaves_the_skip_tentative(self) -> None:
        # A failed episode fetch serves an empty index: the exact leg could not have matched anything.
        pending, candidates = self._numberless_pair()
        mapper = make_sonarr_mapper(sonarr=FakeSonarrClient(parse_fn=lambda _f: _pinfo()))

        result = mapper.assign(pending, candidates, {})

        assert sorted(result.skipped) == sorted(candidates)
        assert result.settled is False

    def test_served_parses_over_a_served_index_settle_the_skip(self) -> None:
        pending, candidates = self._numberless_pair()
        mapper = make_sonarr_mapper(sonarr=FakeSonarrClient(parse_fn=lambda _f: _pinfo()))

        result = mapper.assign(pending, candidates, {EpisodeKey(1, 1): 101})

        assert sorted(result.skipped) == sorted(candidates)
        assert result.settled is True

    def test_a_fully_seeded_batch_is_settled_without_a_parse(self) -> None:
        # Nothing left to place means nothing to parse, so the fake's default parse miss is never asked.
        name = "Show - 01 [1080p].mkv"
        pending = pending_import(file_episode_map={name: [101]}, episode_ids=[101], ordered_episode_ids=[101])
        mapper = make_sonarr_mapper(sonarr=FakeSonarrClient())

        result = mapper.assign(pending, {normalize_basename(name): _cand(name)}, {EpisodeKey(1, 1): 101})

        assert result.skipped == ()
        assert result.settled is True


class TestAssignBogusKeyDowngrade:
    """A name key that exists nowhere in the series is noise, not identity.

    The downgrade only ever feeds the 1:1 single-file fallback - a key that
    resolves ANYWHERE in the series map stays real evidence.
    """

    def test_movie_year_bogus_key_places_the_sole_resolved_episode(self) -> None:
        # "Chronicle.2020" parses S20E20 - a key the series doesn't have. One
        # file, one resolved id: the parse artifact downgrades to numberless.
        parsed = {"movie.mkv": _pinfo(season=20, episodes=(20,))}
        ep_id_map = {EpisodeKey(1, 1): 501}

        result = assign_episode_ids(PlacementBatch(["movie.mkv"], parsed), TargetScope([900], ep_id_map))

        assert result.assigned == {"movie.mkv": [900]}
        assert result.skipped == ()

    def test_resolving_key_is_never_downgraded(self) -> None:
        # The same key EXISTS in the series (resolving outside our set): that is real
        # evidence, so the out-of-set refusal stands and names the file another slice's.
        parsed = {"movie.mkv": _pinfo(season=20, episodes=(20,))}
        ep_id_map = {EpisodeKey(20, 20): 555}

        result = assign_episode_ids(PlacementBatch(["movie.mkv"], parsed), TargetScope([900], ep_id_map))

        assert result.assigned == {}
        assert _verdicts(result) == {"movie.mkv": PlacementVerdict.FOREIGN}

    def test_partially_real_multi_key_is_refused(self) -> None:
        # One of the two parsed keys resolves in the series, so the signal is
        # not provably bogus - the whole claim stays a refusal.
        parsed = {"d.mkv": _pinfo(season=1, episodes=(5, 99))}
        ep_id_map = {EpisodeKey(1, 5): 505}

        result = assign_episode_ids(PlacementBatch(["d.mkv"], parsed), TargetScope([900], ep_id_map))

        assert result.assigned == {}
        assert result.skipped == ("d.mkv",)

    def test_bogus_key_with_absolutes_is_not_downgraded(self) -> None:
        # Absolute numbers are real signal even when the SxxEyy key is bogus,
        # and the multi-absolute span keeps leg 2 refused too.
        parsed = {"movie.mkv": _pinfo(season=20, episodes=(20,), absolutes=(20, 21))}
        ep_id_map = {EpisodeKey(1, 1): 501}

        result = assign_episode_ids(PlacementBatch(["movie.mkv"], parsed), TargetScope([900], ep_id_map))

        assert result.assigned == {}
        assert result.skipped == ("movie.mkv",)

    def test_bogus_key_with_full_season_match_is_refused(self) -> None:
        # A full-season parse means the file plausibly holds MANY episodes -
        # cardinality evidence still vetoes the downgraded placement.
        parsed = {"movie.mkv": _pinfo(season=20, episodes=(20,), full_season=True)}
        ep_id_map = {EpisodeKey(1, 1): 501}

        result = assign_episode_ids(PlacementBatch(["movie.mkv"], parsed), TargetScope([900], ep_id_map))

        assert result.assigned == {}
        assert result.skipped == ("movie.mkv",)

    def test_bogus_key_with_single_matched_pair_still_places(self) -> None:
        # The heal-mode special shape: the name parses S02E00 (nonexistent) and
        # Sonarr matched one pair - a single pair never vetoes the fallback.
        parsed = {"sp.mkv": _pinfo(season=2, episodes=(0,), matched=((1, 5),))}
        ep_id_map = {EpisodeKey(1, 5): 555}

        result = assign_episode_ids(PlacementBatch(["sp.mkv"], parsed), TargetScope([900], ep_id_map))

        assert result.assigned == {"sp.mkv": [900]}
        assert result.skipped == ()


class TestResolvedIds:
    """`resolved_ids`: the ordered set when the record carries one, else the seeds' ids sorted."""

    def test_the_ordered_set_wins(self) -> None:
        pending = pending_import(file_episode_map={"a.mkv": [7]}, episode_ids=[9], ordered_episode_ids=[3, 1, 2])

        assert pending.resolved_ids() == [3, 1, 2]

    def test_an_older_record_falls_back_to_its_seeds_sorted(self) -> None:
        pending = pending_import(
            file_episode_map={"a.mkv": [7, 0], "b.mkv": [5]},
            episode_ids=[9, 7],
            ordered_episode_ids=[],
        )

        assert pending.resolved_ids() == [5, 7, 9]

    def test_a_record_with_no_targets_reads_empty(self) -> None:
        pending = pending_import(file_episode_map={}, episode_ids=[], ordered_episode_ids=[])

        assert pending.resolved_ids() == []


class TestPlacementBatchParsesKnown:
    """`all_parses_known`: the settled hinge on the parse leg. Any miss, transport or offline, unsettles the batch."""

    def test_a_transport_miss_is_unknown(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"a.mkv": _pinfo(), "b.mkv": None}
        assert PlacementBatch(["a.mkv", "b.mkv"], parsed).all_parses_known is False

    def test_an_offline_stand_in_is_unknown(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"a.mkv": _pinfo(season=1, episodes=(1,), offline=True)}
        assert PlacementBatch(["a.mkv"], parsed).all_parses_known is False

    def test_served_parses_are_known(self) -> None:
        # A numberless answer from Sonarr is a real answer: known, even though it places nothing.
        parsed: dict[str, ParsedFileInfo | None] = {"a.mkv": _pinfo(), "b.mkv": _pinfo(season=1, episodes=(1,))}
        assert PlacementBatch(["a.mkv", "b.mkv"], parsed).all_parses_known is True

    def test_an_empty_batch_is_known(self) -> None:
        # A fully seeded record parses nothing, and nothing is missing.
        assert PlacementBatch([], {}).all_parses_known is True


class TestAssignNumberlessZip:
    """The pristine numberless N:N zip - order is the only signal left."""

    def test_numberless_batch_zips_in_name_order(self) -> None:
        # Three numberless files, three leftover ids: name order maps onto
        # airing order regardless of the on-disk listing order.
        files = ["sp2.mkv", "sp1.mkv", "sp3.mkv"]
        parsed = {name: _pinfo() for name in files}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([901, 902, 903], {}))

        assert result.assigned == {"sp1.mkv": [901], "sp2.mkv": [902], "sp3.mkv": [903]}
        assert result.skipped == ()

    def test_zip_orders_digits_naturally(self) -> None:
        # "sp10" sorts after "sp2" - lexical order would hand sp10 the second id.
        files = ["sp1.mkv", "sp2.mkv", "sp10.mkv"]
        parsed = {name: _pinfo() for name in files}

        result = assign_episode_ids(PlacementBatch(files, parsed), TargetScope([901, 902, 903], {}))

        assert result.assigned == {"sp1.mkv": [901], "sp2.mkv": [902], "sp10.mkv": [903]}

    def test_mixed_batch_never_zips(self) -> None:
        # One file placed by leg 1 makes the parse map bigger than the
        # leftovers: the numberless extras must not fill episode slots.
        parsed = {
            "e01.mkv": _pinfo(season=1, episodes=(1,)),
            "op.mkv": _pinfo(),
            "ed.mkv": _pinfo(),
        }
        ep_id_map = {EpisodeKey(1, 1): 501}

        result = assign_episode_ids(
            PlacementBatch(["e01.mkv", "op.mkv", "ed.mkv"], parsed), TargetScope([501, 502, 503], ep_id_map)
        )

        assert result.assigned == {"e01.mkv": [501]}
        assert sorted(result.skipped) == ["ed.mkv", "op.mkv"]

    def test_seeded_sibling_parse_kills_the_zip(self) -> None:
        # A parse for a file NOT in the batch (seeded or moved out) proves a
        # prior placement - the pristine gate refuses the whole zip.
        parsed = {
            "seeded.mkv": _pinfo(),
            "sp1.mkv": _pinfo(),
            "sp2.mkv": _pinfo(),
        }

        result = assign_episode_ids(PlacementBatch(["sp1.mkv", "sp2.mkv"], parsed), TargetScope([901, 902], {}))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["sp1.mkv", "sp2.mkv"]

    def test_count_mismatch_refuses_both_ways(self) -> None:
        # Two files onto one id, and one file onto two ids: neither the zip
        # nor the degenerate arm ever places off a non-1:1 count.
        two_files = {"sp1.mkv": _pinfo(), "sp2.mkv": _pinfo()}
        one_file: dict[str, ParsedFileInfo | None] = {"sp1.mkv": _pinfo()}

        surplus_files = assign_episode_ids(PlacementBatch(["sp1.mkv", "sp2.mkv"], two_files), TargetScope([901], {}))
        surplus_ids = assign_episode_ids(PlacementBatch(["sp1.mkv"], one_file), TargetScope([901, 902], {}))

        assert surplus_files.assigned == {}
        assert sorted(surplus_files.skipped) == ["sp1.mkv", "sp2.mkv"]
        assert surplus_ids.assigned == {}
        assert surplus_ids.skipped == ("sp1.mkv",)

    def test_none_parse_refuses_the_zip(self) -> None:
        # A parse the caller couldn't get is no evidence - fail closed.
        parsed: dict[str, ParsedFileInfo | None] = {
            "sp1.mkv": _pinfo(),
            "sp2.mkv": _pinfo(),
            "sp3.mkv": None,
        }

        result = assign_episode_ids(
            PlacementBatch(["sp1.mkv", "sp2.mkv", "sp3.mkv"], parsed), TargetScope([901, 902, 903], {})
        )

        assert result.assigned == {}
        assert sorted(result.skipped) == ["sp1.mkv", "sp2.mkv", "sp3.mkv"]

    def test_offline_parse_refuses_the_zip(self) -> None:
        # The offline regex stand-in is blind to what the real parse would
        # have seen, so it never counts as a real numberless parse.
        parsed: dict[str, ParsedFileInfo | None] = {
            "sp1.mkv": _pinfo(),
            "sp2.mkv": _pinfo(),
            "sp3.mkv": _pinfo(offline=True),
        }

        result = assign_episode_ids(
            PlacementBatch(["sp1.mkv", "sp2.mkv", "sp3.mkv"], parsed), TargetScope([901, 902, 903], {})
        )

        assert result.assigned == {}
        assert sorted(result.skipped) == ["sp1.mkv", "sp2.mkv", "sp3.mkv"]

    def test_bogus_key_member_refuses_the_zip(self) -> None:
        # The bogus-key downgrade is 1:1-only: two movies can share one bogus
        # key, so a bogus-keyed member keeps the whole batch refused.
        parsed = {
            "sp1.mkv": _pinfo(),
            "sp2.mkv": _pinfo(),
            "movie.mkv": _pinfo(season=20, episodes=(20,)),
        }

        result = assign_episode_ids(
            PlacementBatch(["sp1.mkv", "sp2.mkv", "movie.mkv"], parsed), TargetScope([901, 902, 903], {})
        )

        assert result.assigned == {}
        assert sorted(result.skipped) == ["movie.mkv", "sp1.mkv", "sp2.mkv"]


class TestReleaseNumberForms:
    """Which name shapes carry a release number, read through the run pass that consumes them.

    A `1..N` run over a 3-wide window places. A shape carrying no number forms
    no run at all, so nothing indexes the window.
    """

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
    }

    @classmethod
    def _place(cls, names: list[str], *, blocked: bool = False) -> EpisodeAssignment:
        """Run three blind names against a 3-wide specials window.

        `blocked` adds a fourth parse so the numberless zip can never place a
        refused run instead, leaving the run pass the only thing under test.
        """

        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in names}
        if blocked:
            parsed["gone.mkv"] = _pinfo()
        return assign_episode_ids(PlacementBatch(names, parsed), TargetScope([501, 502, 503], cls._MAP))

    @staticmethod
    def _numbered(template: str) -> list[str]:
        return [template.format(n=n) for n in (1, 2, 3)]

    def test_the_middle_form_reads_the_number_between_dashes(self) -> None:
        names = self._numbered("show - 0{n} - title [tag].mkv")

        assert self._place(names).assigned == {names[0]: [501], names[1]: [502], names[2]: [503]}

    def test_the_trailing_form_survives_a_version_suffix(self) -> None:
        names = self._numbered("show 0{n}v2 [tag].mkv")

        assert self._place(names).assigned == {names[0]: [501], names[1]: [502], names[2]: [503]}

    def test_a_batch_mixing_both_forms_is_one_run(self) -> None:
        # Both forms drop the separator before the number, so all three read prefix "show"
        # and index one window. Name order matches run order here, so the verdict is the tell.
        names = ["show - 01 - title [tag].mkv", "show - 02.mkv", "show - 03 - other.mkv"]

        result = self._place(names)

        assert result.assigned == {names[0]: [501], names[1]: [502], names[2]: [503]}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.RELEASE_RUN}

    def test_underscores_read_as_spaces(self) -> None:
        names = self._numbered("show_-_0{n}_[bd].mkv")

        assert self._place(names).assigned == {names[0]: [501], names[1]: [502], names[2]: [503]}

    def test_nested_trailing_tags_come_off_to_a_fixpoint(self) -> None:
        # One strip would leave "[a] (b)" behind and the trailing form would miss the number.
        names = self._numbered("show - 0{n} [a] (b) [c].mkv")

        result = self._place(names)

        assert result.assigned == {names[0]: [501], names[1]: [502], names[2]: [503]}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.RELEASE_RUN}

    def test_a_four_digit_year_is_no_release_number(self) -> None:
        names = ["show 2019.mkv", "show 2020.mkv", "show 2021.mkv"]

        assert sorted(self._place(names, blocked=True).skipped) == sorted(names)

    def test_digits_only_inside_a_tail_tag_are_no_release_number(self) -> None:
        # The tag strip takes the whole bracket off, so a CRC's digits never become a count.
        names = [f"show part {word} [ABCD123{n}].mkv" for n, word in enumerate(("one", "two", "three"), 1)]

        assert sorted(self._place(names, blocked=True).skipped) == sorted(names)

    def test_a_leading_number_is_no_release_number(self) -> None:
        # Only the middle and trailing forms exist: a number opening the name is not one.
        names = [f"0{n} - show.mkv" for n in (1, 2, 3)]

        assert sorted(self._place(names, blocked=True).skipped) == sorted(names)

    def test_a_numbered_extras_run_is_no_run(self) -> None:
        # Previews count previews: a "PV 01..03" beside three unreadable specials must not take their window.
        names = self._numbered("show - PV 0{n} [tag].mkv")

        assert sorted(self._place(names, blocked=True).skipped) == sorted(names)


class TestAssignReleaseRun:
    """Pass A: the batch's one `1..N` run indexes a one-season window when Sonarr's reading is incoherent."""

    _WINDOW: ClassVar[list[int]] = [501, 502, 503]
    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(0, 3): 503,
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
    }
    _RUN: ClassVar[list[str]] = [f"sp - 0{i} [grp].mkv" for i in (1, 2, 3)]

    @classmethod
    def _scope(cls) -> TargetScope:
        return TargetScope(cls._WINDOW, cls._MAP)

    @classmethod
    def _place(
        cls,
        parsed: dict[str, ParsedFileInfo | None],
        *,
        scope: TargetScope | None = None,
        to_place: list[str] | None = None,
    ) -> EpisodeAssignment:
        return assign_episode_ids(PlacementBatch(to_place or list(parsed), parsed), scope or cls._scope())

    @classmethod
    def _ran(cls) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
        return {name: ((501 + i,), PlacementVerdict.RELEASE_RUN) for i, name in enumerate(cls._RUN)}

    @classmethod
    def _by_name(cls, result: EpisodeAssignment) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
        return {p.name: (p.ids, p.verdict) for p in result.placements}

    def test_a_reading_of_nothing_lets_the_run_index_the_window(self) -> None:
        # Sonarr read no number from any member, so the release's own 1..N is all there is.
        assert self._by_name(self._place({name: _pinfo() for name in self._RUN})) == self._ran()

    def test_matched_pairs_outside_the_window_do_not_stand(self) -> None:
        # Every member matched the same out-of-window episode: incoherent, so the run wins.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo(matched=((1, 1),)) for name in self._RUN}

        assert self._by_name(self._place(parsed)) == self._ran()

    def test_a_members_own_key_outside_a_specials_window_is_overridden(self) -> None:
        # D1'': a TVDB-shifted special names another season. Over a season-0 window the run stands.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed[self._RUN[0]] = _pinfo(season=1, episodes=(1,))

        assert self._by_name(self._place(parsed)) == self._ran()

    def test_bogus_keys_do_not_stand(self) -> None:
        # Keys that exist nowhere in the series are parse artifacts, never a reading.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo(season=20, episodes=(20,)) for name in self._RUN}

        assert self._by_name(self._place(parsed)) == self._ran()

    def test_a_coherent_permuted_reading_stands(self) -> None:
        # D2: every member reads one distinct id inside the window, so Sonarr's
        # reading decides placement even though it permutes the run's order.
        parsed: dict[str, ParsedFileInfo | None] = {
            self._RUN[0]: _pinfo(season=0, episodes=(3,)),
            self._RUN[1]: _pinfo(season=0, episodes=(1,)),
            self._RUN[2]: _pinfo(season=0, episodes=(2,)),
        }

        assert self._by_name(self._place(parsed)) == {
            self._RUN[0]: ((503,), PlacementVerdict.EXACT),
            self._RUN[1]: ((501,), PlacementVerdict.EXACT),
            self._RUN[2]: ((502,), PlacementVerdict.EXACT),
        }

    def test_two_runs_of_one_width_refuse_and_suppress_the_numbered_run(self) -> None:
        # Which run owns the window is unknowable, and the later blind pass must not guess either.
        names = [f"{prefix} - 0{i} [grp].mkv" for prefix in ("a", "b") for i in (1, 2, 3)]
        result = self._place({name: _pinfo() for name in names})

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(names)

    def test_a_width_one_window_refuses(self) -> None:
        # A one-id window is no run: the degenerate single-file leg is what places it.
        result = self._place({self._RUN[0]: _pinfo()}, scope=TargetScope([501], self._MAP))

        assert self._by_name(result) == {self._RUN[0]: ((501,), PlacementVerdict.SINGLE)}

    def test_a_gappy_window_refuses(self) -> None:
        # Episodes 1, 2, 4 are not a run's worth of consecutive slots.
        gappy = {EpisodeKey(0, 1): 501, EpisodeKey(0, 2): 502, EpisodeKey(0, 4): 504, EpisodeKey(1, 1): 601}
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed["pack.mkv"] = _pinfo(season=1, episodes=(1,))

        result = self._place(parsed, scope=TargetScope([601, 501, 502, 504], gappy))

        assert result.assigned == {"pack.mkv": [601]}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_two_season_window_refuses(self) -> None:
        # The extra parse keeps the numberless zip out, so the refusal is what is pinned.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed["gone.mkv"] = _pinfo()

        result = self._place(parsed, scope=TargetScope([501, 502, 601], self._MAP), to_place=self._RUN)

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_member_whose_lone_absolute_disagrees_leaves_the_run(self) -> None:
        # Its "02" was not the release's count, so no full-width run is left to fit.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed[self._RUN[1]] = _pinfo(absolutes=(9,))

        result = self._place(parsed)

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_an_empty_series_map_refuses(self) -> None:
        # D8: with no map there is no window to index, and the extra parse keeps the zip out.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed["gone.mkv"] = _pinfo()

        result = self._place(parsed, scope=TargetScope(self._WINDOW, {}), to_place=self._RUN)

        assert result.assigned == {}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_an_unknown_parse_holds_the_members(self) -> None:
        # D10: one unreadable name anywhere in the batch, and no pass may place the run.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed["gone.mkv"] = None

        result = self._place(parsed, to_place=self._RUN)

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.HELD}

    def test_a_members_own_key_in_another_regular_season_stands_the_run_down(self) -> None:
        # D1'': over a REGULAR-season window a member naming another season is
        # evidence the torrent is mislisted, so nothing is indexed onto it.
        regular = {EpisodeKey(2, 1): 701, EpisodeKey(2, 2): 702, EpisodeKey(2, 3): 703, EpisodeKey(1, 1): 601}
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed[self._RUN[0]] = _pinfo(season=1, episodes=(1,))

        result = self._place(parsed, scope=TargetScope([701, 702, 703], regular))

        assert result.assigned == {}
        assert self._by_name(result)[self._RUN[0]] == ((), PlacementVerdict.FOREIGN)
        assert sorted(result.skipped) == sorted(self._RUN[1:])

    def test_a_non_member_reading_inside_the_window_refuses_the_run(self) -> None:
        # Another file owns one of the slots, so the run does not own the window whole.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed["extra.mkv"] = _pinfo(season=0, episodes=(2,))

        result = self._place(parsed)

        assert result.assigned == {"extra.mkv": [502]}
        assert sorted(result.skipped) == sorted(self._RUN)

    def test_a_member_sonarr_matched_to_two_episodes_stays_in_the_run(self) -> None:
        # Sonarr's scene map reading one member as a double episode is the incoherence
        # the run overrides (measured: every such pack was N files for N episodes).
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN}
        parsed[self._RUN[1]] = _pinfo(matched=((0, 2), (0, 3)))

        result = self._place(parsed)

        assert result.assigned == dict(zip(self._RUN, ([501], [502], [503]), strict=True))
        assert {p.verdict for p in result.placements} == {PlacementVerdict.RELEASE_RUN}

    def test_a_name_the_parses_never_covered_holds_like_a_miss(self) -> None:
        # A batch whose parses skip a name to place is as unknown as one carrying a None.
        parsed: dict[str, ParsedFileInfo | None] = {name: _pinfo() for name in self._RUN[:2]}

        result = self._place(parsed, to_place=self._RUN)

        assert result.assigned == {}
        assert {p.verdict for p in result.placements} == {PlacementVerdict.HELD}


class TestAssignNumberedRun:
    """Pass F: one `1..N` run indexes a contiguous one-season window among the files Sonarr read nothing from."""

    _WINDOW: ClassVar[list[int]] = [10370, 10371, 10372]
    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 21): 10370,
        EpisodeKey(0, 22): 10371,
        EpisodeKey(0, 23): 10372,
        EpisodeKey(0, 25): 10374,
        EpisodeKey(1, 1): 10384,
    }
    _MAIN: ClassVar[str] = "show s01e01 [bd].mkv"

    @staticmethod
    def _run(stem: str, numbers: range) -> list[str]:
        return [f"[grp] show {stem}{i} [bd 1080p x264 10bit flac].mkv" for i in numbers]

    @classmethod
    def _batch(cls, run: list[str]) -> dict[str, ParsedFileInfo | None]:
        """The season-pack file plus a blind run, in file order."""

        parsed: dict[str, ParsedFileInfo | None] = {cls._MAIN: _pinfo(season=1, episodes=(1,))}
        parsed.update({name: _pinfo() for name in run})
        return parsed

    @classmethod
    def _place(
        cls,
        parsed: dict[str, ParsedFileInfo | None],
        scope: TargetScope,
        to_place: list[str] | None = None,
    ) -> EpisodeAssignment:
        return assign_episode_ids(PlacementBatch(to_place or list(parsed), parsed), scope)

    @classmethod
    def _scope(cls) -> TargetScope:
        return TargetScope([10384, *cls._WINDOW], cls._MAP)

    def test_a_run_beside_a_season_pack_places(self) -> None:
        # The mixed batch the ordered zip refuses: the pack takes its own key and the
        # run indexes what is left, which the two-season window keeps pass A out of.
        run = self._run("extra ", range(1, 4))
        result = self._place(self._batch(run), self._scope())

        assert result.assigned == {self._MAIN: [10384], **{name: [10370 + i] for i, name in enumerate(run)}}
        assert (result.skipped, result.excluded) == ((), ())
        assert {p.verdict for p in result.placements if p.name in run} == {PlacementVerdict.NUMBERED_RUN}

    def test_a_season_zero_absolute_run_still_counts_as_blind(self) -> None:
        # Absolutes alone resolve nothing, so a season-0 run reads as blind and indexes the window.
        main = "[grp] show s2 - 01 [bd].mkv"
        run = [f"[grp] ova - 0{i} [bd].mkv" for i in (1, 2, 3)]
        parsed: dict[str, ParsedFileInfo | None] = {main: _pinfo(season=1, episodes=(1,), absolutes=(1,))}
        parsed.update({name: _pinfo(season=0, absolutes=(i + 1,)) for i, name in enumerate(run)})

        result = self._place(parsed, self._scope())

        assert [list(p.ids) for p in result.placements] == [[10384], [10370], [10371], [10372]]

    def test_two_runs_of_one_width_refuse(self) -> None:
        run = self._run("a ", range(1, 4)) + self._run("b ", range(1, 4))
        result = self._place(self._batch(run), self._scope())

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)

    def test_a_run_not_starting_at_one_refuses(self) -> None:
        # Only a 1..N run indexes a window: a 2..4 run says nothing about where it starts.
        run = self._run("extra ", range(2, 5))
        result = self._place(self._batch(run), self._scope())

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)

    def test_a_run_wider_than_the_window_refuses(self) -> None:
        run = self._run("extra ", range(1, 6))
        result = self._place(self._batch(run), self._scope())

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)

    def test_a_gappy_window_refuses(self) -> None:
        run = self._run("extra ", range(1, 4))
        result = self._place(self._batch(run), TargetScope([10384, 10370, 10371, 10374], self._MAP))

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)

    def test_an_unknown_parse_refuses(self) -> None:
        # D10 again on the blind pass: an unreadable name anywhere holds every count leg closed.
        run = self._run("extra ", range(1, 4))
        parsed = self._batch(run)
        parsed["gone.mkv"] = None

        result = self._place(parsed, self._scope(), [self._MAIN, *run])

        assert result.assigned == {self._MAIN: [10384]}
        assert sorted(result.skipped) == sorted(run)


class TestPlacementClassification:
    """What `finish()` calls the leftovers, and what a leftover's classification does NOT buy it earlier."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(1, 1): 601,
    }

    def test_a_file_bound_for_foreign_still_counts_against_the_ordered_zip(self) -> None:
        # Exclusion is decided LAST, so a foreign leaf is an open leftover while the
        # count legs run: the numberless pair beside it never zips.
        parsed: dict[str, ParsedFileInfo | None] = {
            "one.mkv": _pinfo(),
            "two.mkv": _pinfo(),
            "far.mkv": _pinfo(season=1, episodes=(1,)),
        }

        result = assign_episode_ids(PlacementBatch(list(parsed), parsed), TargetScope([501, 502], self._MAP))

        assert result.assigned == {}
        assert sorted(result.skipped) == ["one.mkv", "two.mkv"]
        assert _verdicts(result)["far.mkv"] == PlacementVerdict.FOREIGN

    def test_a_second_file_on_a_placed_id_is_a_duplicate(self) -> None:
        # Both read the same episode: the first places, the second is knowably never imported.
        parsed: dict[str, ParsedFileInfo | None] = {
            "a.mkv": _pinfo(season=0, episodes=(1,)),
            "b.mkv": _pinfo(season=0, episodes=(1,)),
        }

        result = assign_episode_ids(PlacementBatch(list(parsed), parsed), TargetScope([501, 502], self._MAP))

        assert _verdicts(result) == {"a.mkv": PlacementVerdict.EXACT, "b.mkv": PlacementVerdict.DUPLICATE}

    def test_a_reading_that_resolves_nowhere_stays_countable(self) -> None:
        # D11: nothing proved it another slice's, so the absolute leg may still place it.
        parsed: dict[str, ParsedFileInfo | None] = {"x.mkv": _pinfo(season=9, episodes=(9,), absolutes=(9,))}

        result = assign_episode_ids(PlacementBatch(["x.mkv"], parsed), TargetScope([501], self._MAP))

        assert _verdicts(result) == {"x.mkv": PlacementVerdict.ABSOLUTE}

    def test_the_bogus_key_single_arm_refuses_on_an_empty_map(self) -> None:
        # D8: over an unserved map every key "misses", so no key may be called bogus.
        parsed: dict[str, ParsedFileInfo | None] = {"movie.mkv": _pinfo(season=20, episodes=(20,))}

        result = assign_episode_ids(PlacementBatch(["movie.mkv"], parsed), TargetScope([501], {}))

        assert _verdicts(result) == {"movie.mkv": PlacementVerdict.SKIPPED}


class TestSeedEqualsMapper:
    """One batch, two entry points: the grab-time seed and the import-time mapper agree file for file."""

    _MAP: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(0, 1): 501,
        EpisodeKey(0, 2): 502,
        EpisodeKey(1, 1): 601,
    }
    _NAMES: ClassVar[list[str]] = ["Show - S00E01 [1080p].mkv", "Show - S00E02 [1080p].mkv", "Show - S01E01 [BD].mkv"]

    @classmethod
    def _parses(cls) -> dict[str, ParsedFileInfo | None]:
        return {
            cls._NAMES[0]: _pinfo(season=0, episodes=(1,)),
            cls._NAMES[1]: _pinfo(season=0, episodes=(2,)),
            cls._NAMES[2]: _pinfo(season=1, episodes=(1,)),
        }

    @classmethod
    def _index(cls) -> EpisodeIndex:
        return episode_index([sonarr_ep(0, 1, ep_id=501), sonarr_ep(0, 2, ep_id=502)])

    def test_seed_scope_targets_the_entrys_ids_over_the_series_map(self) -> None:
        scope = SeedScope(self._index(), self._MAP)

        assert scope.target() == TargetScope([501, 502], self._MAP)

    def test_the_seed_and_the_mapper_place_and_exclude_alike(self) -> None:
        parses = self._parses()
        index = self._index()
        release = SeedRelease(
            release_group="grp",
            url_item=url_item(url="u", infohash="h"),
            infohash="h",
            files=tuple(SeedFile(name, parses[name]) for name in self._NAMES),
        )

        seed = build_pending_seed(
            release,
            SeedScope(index, self._MAP),
            PendingSeedContext(al_id=1, series_id=2, title="t", added_at="2026-01-01 00:00:00"),
        )
        mapper = make_sonarr_mapper(sonarr=FakeSonarrClient(parse_fn=parses.get))
        live = mapper.assign(
            pending_import(
                file_episode_map={},
                episode_ids=[],
                ordered_episode_ids=list(index.by_id),
                seadex_files=self._NAMES,
            ),
            {normalize_basename(name): _cand(name) for name in self._NAMES},
            self._MAP,
        )

        assert seed.file_episode_map == live.assigned
        assert seed.excluded_files == [p.name for p in live.excluded]
        # And it is a real placement, not two empty maps agreeing.
        assert seed.file_episode_map == {
            normalize_basename(self._NAMES[0]): [501],
            normalize_basename(self._NAMES[1]): [502],
        }
        assert seed.excluded_files == [normalize_basename(self._NAMES[2])]


# --------------------------------------------------------------------------- #
# classify_queue on the real captured queue
# --------------------------------------------------------------------------- #
class TestClassifyRealQueue:
    """The real queue had a paused download (wait) + two importBlocked (step in)."""

    @staticmethod
    def _records_by_download() -> dict[str, list[QueueRecord]]:
        body: dict[str, list[dict[str, object]]] = load_fixture("queue.json")
        records: dict[str, list[QueueRecord]] = {}
        for rec in body["records"]:
            record = QueueRecord.model_validate(rec)
            records.setdefault(record.download_id or "", []).append(record)
        return records

    def test_import_blocked_steps_in(self) -> None:
        records = self._records_by_download()
        unmatched = records["1111111111111111111111111111111111111111"]
        assert classify_queue(unmatched) is QueueVerdict.STEP_IN

    def test_paused_download_waits(self) -> None:
        records = self._records_by_download()
        paused = records["B7640FF13A2ADCA981B821D03CEBD1B569798459"]
        assert classify_queue(paused) is QueueVerdict.WAIT


# --------------------------------------------------------------------------- #
# PendingImport round-trip carries the new resolved set (with back-compat)
# --------------------------------------------------------------------------- #
class TestPendingImportOrderedIds:
    """`ordered_episode_ids` round-trips through JSON.

    A legacy record missing the key rehydrates to an empty list.
    """

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
    """Both command-list guards read the real captured list to close the loop."""

    @staticmethod
    def _commands() -> list[CommandResource]:
        raw: list[dict[str, object]] = load_fixture("command_list.json")
        return [CommandResource.model_validate(c) for c in raw]

    def test_matching_download_id_is_in_flight(self) -> None:
        # The SAO download has a started + queued ManualImport sharing its
        # downloadId -> a fresh import for it would stack a duplicate.
        assert manual_import_in_flight(
            self._commands(),
            DownloadMatch(_SAO_DOWNLOAD_ID, ContentPaths(raw="/downloads", sonarr_visible="/downloads"), set()),
        )

    def test_unrelated_download_id_is_not_in_flight(self) -> None:
        # A different infohash with no path/episode overlap -> proceed.
        assert not manual_import_in_flight(
            self._commands(),
            DownloadMatch(
                "ffffffffffffffffffffffffffffffffffffffff",
                ContentPaths(raw="/nowhere", sonarr_visible="/nowhere"),
                set(),
            ),
        )

    def test_folder_import_matches_by_episode_id(self) -> None:
        # The Vodes folder import carries no downloadId. Episode 5645 is ours.
        assert manual_import_in_flight(
            self._commands(),
            DownloadMatch("no-such-hash", ContentPaths(raw="/nowhere", sonarr_visible="/nowhere"), {5645}),
        )

    def test_disk_guard_defers_only_on_the_started_command(self) -> None:
        # The capture's one STARTED command (a ManualImport) defers; with it
        # gone, the queued remainder - including the parked
        # ProcessMonitoredDownloads - must not (queued never defers).
        commands = self._commands()
        assert started_disk_commands(commands)
        queued_only = [c for c in commands if c.status != "started"]
        assert not started_disk_commands(queued_only)


# --------------------------------------------------------------------------- #
# End-to-end: the captured specials failure now imports to the resolved S00 ids
# --------------------------------------------------------------------------- #
def _specials_parse_side_effect(raw_base: str) -> ParsedFileInfo | None:
    """Replay the captured /parse bodies for the two specials by basename."""

    if "S00E01" in raw_base:
        body: dict[str, object] = load_fixture("parse_yamada_s00e01.json")
        return ParsedFileInfo.model_validate(body)
    if "S00E02" in raw_base:
        body = load_fixture("parse_yamada_s00e02.json")
        return ParsedFileInfo.model_validate(body)
    return None


def _specials_strat(config: AppConfig | None = None) -> tuple[SonarrSync, FakeSonarrClient, list[str]]:
    """The captured specials fixtures wired into a bare SonarrSync + its scripted fake.

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
        parse_fn=_specials_parse_side_effect,
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


class TestCapturedSpecialsEndToEnd:
    """Drive import_completed with the real fixtures for the failing queue item."""

    def test_specials_import_to_resolved_episode_ids(self) -> None:
        strat, sonarr, seadex_files = _specials_strat()

        # Resolved set = the entry's S00 episodes (8030, 8031, 8032). The torrent
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

        probe = strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        # The command was issued, and the copy is async, so nothing is present yet.
        assert probe.files_present is False
        assert probe.command_issued is True
        assert len(sonarr.execute_calls) == 1
        # The configured import mode is threaded onto the execute command (default
        # "auto"). Selecting "move" deletes the source files, so a wrong mode must
        # not be silent.
        assert sonarr.execute_calls[0][1] == "auto"

        files = sonarr.execute_calls[0][0]
        assigned = {f.episodeIds[0]: f for f in files}
        assert set(assigned) == {8030, 8031}
        assert all(f.seriesId == 213 for f in files)

    def test_import_mode_propagates_from_config(self) -> None:
        # imports.mode flows through to manual_import_execute - a regression that
        # hardcoded/ignored it (e.g. "move" -> source-file deletion) would be invisible
        # without this. Flip the config and assert the configured mode reaches Sonarr.
        strat, sonarr, seadex_files = _specials_strat(make_config(import_mode="move"))

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

        strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert len(sonarr.execute_calls) == 1
        assert sonarr.execute_calls[0][1] == "move"

    def test_import_completed_probe_carries_seed_complete_counts(self) -> None:
        # A complete seed map -> the probe carries the determinate "files inserted"
        # counts (none landed yet here -> 0 / N), pinned to the seed set.
        strat, _sonarr, seadex_files = _specials_strat()
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

        probe = strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert probe.target_count == len(seadex_files)
        assert probe.imported_count == 0

    def test_import_progress_is_read_only_and_counts_seed_targets(self) -> None:
        # The Tier-2 fast poll: a determinate count over the seed targets, reading
        # ONLY the episode files - never the refresh / queue / execute pipeline.
        strat, sonarr, seadex_files = _specials_strat()
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
        strat, sonarr, seadex_files = _specials_strat()
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

    def test_import_progress_indeterminate_for_legacy_flat_record(self) -> None:
        # A legacy flat record (episode_ids only, no SeaDex file list): targets
        # exist but there is nothing to measure completeness against, so the
        # row stays indeterminate rather than trusting a listless record.
        strat, sonarr, _seadex_files = _specials_strat()
        pending = pending_import(
            infohash="6666666666666666666666666666666666666666",
            series_id=213,
            release_group="Headpatter",
            file_episode_map={},
            episode_ids=[8030],
            ordered_episode_ids=[],
            seadex_files=[],
        )

        progress = strat.import_progress(pending)

        assert progress == ImportProgress(0, 0, determinate=False)
        assert sonarr.execute_calls == []

    def test_excluded_files_make_the_heavy_counts_determinate_but_never_promote(self) -> None:
        # A pack carrying another slice's files: map + excluded account for
        # every file, so the heavy poll's probe carries determinate counts over
        # OUR slice. Tier-2 stays strict-indeterminate: it can PROMOTE (a
        # drop), and a grab-time exclusion must never decide one.
        strat, _sonarr, seadex_files = _specials_strat()
        pending = pending_import(
            infohash="5555555555555555555555555555555555555555",
            series_id=213,
            release_group="Headpatter",
            file_episode_map={seadex_files[0]: [8030]},
            episode_ids=[],
            ordered_episode_ids=[8030],
            seadex_files=seadex_files,
            excluded_files=[normalize_basename(name) for name in seadex_files[1:]],
        )

        progress = strat.import_progress(pending)
        probe = strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert progress == ImportProgress(0, 0, determinate=False)
        assert (probe.imported_count, probe.target_count) == (0, 1)

    def test_specials_import_with_empty_resolved_set(self) -> None:
        # THE headline regression: the ACTUAL on-disk stuck record is pre-fix - EMPTY
        # everything (no ordered_episode_ids, no seed map). Before the fix this fell
        # to the legacy path, mapped nothing (Sonarr's series-matched episodes are
        # empty), and retried forever. Now the empty-set exact fallback places the
        # two specials onto the live series episodes, so it imports with no re-grab.
        strat, sonarr, seadex_files = _specials_strat()

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

        probe = strat.import_completed(pending, "/downloads/yamada", AttemptKind.POLL)

        assert probe.files_present is False
        assert probe.command_issued is True
        assert len(sonarr.execute_calls) == 1
        assert sonarr.execute_calls[0][1] == "auto"

        files = sonarr.execute_calls[0][0]
        assigned = {f.episodeIds[0]: f for f in files}
        assert set(assigned) == {8030, 8031}
        assert all(f.seriesId == 213 for f in files)
