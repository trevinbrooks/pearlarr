# pyright: strict
"""Borrowed-pair placement and the foreign classification through `assign_episode_ids`.

The passes themselves are pinned by the fixture-driven `test_manual_import_fixtures`, and the grab-time
path through `place_release` by `test_grab_placement` and `test_pending_seeds`.
"""

from typing import ClassVar

from pearlarr.placement_types import PlacementBatch, PlacementVerdict, TargetScope
from pearlarr.placer import assign_episode_ids
from pearlarr.seadex_types import EpisodeKey, ParsedFileInfo

from .builders import parsed_info, series_index


def _verdicts(
    parsed: dict[str, ParsedFileInfo | None],
    scope: TargetScope,
) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
    """Run one batch through `assign_episode_ids`, keyed name -> (ids, verdict)."""

    result = assign_episode_ids(PlacementBatch(list(parsed), parsed), scope)
    return {p.name: (p.ids, p.verdict) for p in result.placements}


class TestBorrowedPairPlacement:
    """The reading limits every pass shares: borrow cap, all-or-nothing resolution, duplicate collapse.

    Sonarr's series-matched pairs are borrowed only by a name carrying no
    numbers of its own, and only inside the record's set.
    """

    _SCOPE: ClassVar[TargetScope] = TargetScope(
        [11, 12, 13], series_index({EpisodeKey(1, 1): 11, EpisodeKey(1, 2): 12, EpisodeKey(1, 3): 13})
    )

    def test_a_pair_inside_the_entry_places_exactly(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(matched=((1, 1), (1, 2)))}

        assert _verdicts(parsed, self._SCOPE) == {"span.mkv": ((11, 12), PlacementVerdict.EXACT)}

    def test_a_pair_resolving_nowhere_in_the_series_is_skipped(self) -> None:
        # D11: the series map knows nothing of it, so it is possibly ours, never proved another slice's.
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(matched=((9, 9),))}

        assert _verdicts(parsed, self._SCOPE) == {"span.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_partially_resolving_span_refuses_the_whole_file(self) -> None:
        # One pair in the map, one out: placing the resolved half would half-import a multi-episode file.
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(matched=((1, 1), (9, 9)))}

        assert _verdicts(parsed, self._SCOPE) == {"span.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_full_season_match_never_borrows(self) -> None:
        # D6: a bare "S01" extra matches the WHOLE season, so none of its pairs may place.
        parsed: dict[str, ParsedFileInfo | None] = {"pack.mkv": parsed_info(matched=((1, 1), (1, 2)), full_season=True)}

        assert _verdicts(parsed, self._SCOPE) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_full_season_name_key_is_refused_too(self) -> None:
        # D6 again, on the other claim source: the flag vetoes the name's own key, not just the borrow.
        parsed: dict[str, ParsedFileInfo | None] = {"pack.mkv": parsed_info(season=1, episodes=(1,), full_season=True)}

        assert _verdicts(parsed, self._SCOPE) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_wide_span_the_name_claims_itself_places_whole(self) -> None:
        # The cap is a BORROW limit. An explicit "E01-E04" range resolving every key inside the
        # set is a complete reading of the name's own claim, so width never refuses it.
        scope = TargetScope([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed: dict[str, ParsedFileInfo | None] = {"span.mkv": parsed_info(season=1, episodes=(1, 2, 3, 4))}

        assert _verdicts(parsed, scope) == {"span.mkv": ((11, 12, 13, 14), PlacementVerdict.EXACT)}

    def test_a_borrowed_span_just_over_the_cap_is_refused(self) -> None:
        # Four distinct pairs a NUMBERLESS name only borrowed is the season-pack shape sans flag.
        scope = TargetScope([11, 12, 13, 14], series_index({EpisodeKey(1, e): 10 + e for e in range(1, 5)}))
        parsed: dict[str, ParsedFileInfo | None] = {"pack.mkv": parsed_info(matched=tuple((1, e) for e in range(1, 5)))}

        assert _verdicts(parsed, scope) == {"pack.mkv": ((), PlacementVerdict.SKIPPED)}

    def test_a_triple_span_still_places(self) -> None:
        # The cap boundary: three episodes is still one file's claim.
        parsed: dict[str, ParsedFileInfo | None] = {"triple.mkv": parsed_info(matched=((1, 1), (1, 2), (1, 3)))}

        assert _verdicts(parsed, self._SCOPE) == {"triple.mkv": ((11, 12, 13), PlacementVerdict.EXACT)}

    def test_duplicate_pairs_collapse_to_one_claim(self) -> None:
        # Junk wire repeats are one claim, so the file places rather than reading as a wide span.
        parsed: dict[str, ParsedFileInfo | None] = {"one.mkv": parsed_info(matched=((1, 1), (1, 1)))}

        assert _verdicts(parsed, self._SCOPE) == {"one.mkv": ((11,), PlacementVerdict.EXACT)}


class TestForeignClassification:
    """D11: `FOREIGN` needs a COMPLETE reading landing entirely outside the record's set.

    Anything less (a partial span, a veto, no reading at all) is possibly ours
    and stays `SKIPPED`, so the count legs may still place it.
    """

    _MAP: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(3, 1): 101, EpisodeKey(3, 12): 112, EpisodeKey(3, 13): 113}

    def test_a_clean_reading_fully_outside_is_another_slice(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"other.mkv": parsed_info(season=3, episodes=(13,))}

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "other.mkv": ((), PlacementVerdict.FOREIGN)
        }

    def test_a_reading_inside_the_entry_is_ours(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {"mine.mkv": parsed_info(season=3, episodes=(1,))}

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "mine.mkv": ((101,), PlacementVerdict.EXACT)
        }

    def test_a_partially_resolving_span_stays_possibly_ours(self) -> None:
        # A boundary double-episode with one pair off the map may be partly ours.
        parsed: dict[str, ParsedFileInfo | None] = {"d.mkv": parsed_info(season=3, episodes=(12, 99))}

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "d.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_a_vetoed_full_season_reading_is_still_ours(self) -> None:
        # Sonarr reads a bare "S0X" as the whole season: a missing episode token,
        # not several episodes, so the one leftover id takes the file.
        parsed: dict[str, ParsedFileInfo | None] = {
            "pack.mkv": parsed_info(season=2, episodes=(1,), full_season=True),
        }

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "pack.mkv": ((101,), PlacementVerdict.SINGLE)
        }

    def test_a_vetoed_wide_span_stays_possibly_ours(self) -> None:
        parsed: dict[str, ParsedFileInfo | None] = {
            "pack.mkv": parsed_info(matched=tuple((9, n) for n in range(1, 11)))
        }

        assert _verdicts(parsed, TargetScope([101], series_index(self._MAP))) == {
            "pack.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_no_reading_at_all_stays_possibly_ours(self) -> None:
        # Two leftover ids keep the degenerate single-file arm out, so the classification is what is pinned.
        parsed: dict[str, ParsedFileInfo | None] = {"blank.mkv": parsed_info()}

        assert _verdicts(parsed, TargetScope([101, 112], series_index(self._MAP))) == {
            "blank.mkv": ((), PlacementVerdict.SKIPPED)
        }

    def test_an_empty_series_map_refuses_the_verdict(self) -> None:
        # D8: every key misses an unserved map, so nothing may be called another slice's.
        parsed: dict[str, ParsedFileInfo | None] = {"other.mkv": parsed_info(season=3, episodes=(13,))}

        assert _verdicts(parsed, TargetScope([101, 112], series_index({}))) == {
            "other.mkv": ((), PlacementVerdict.SKIPPED)
        }
