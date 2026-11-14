# pyright: strict
"""The other-orders pass: files the aired order left unplaced, placed where another TVDB order reads them."""

from typing import ClassVar

from pearlarr.arr_models import ParsedFileInfo
from pearlarr.pending_import import TitleNames
from pearlarr.placement_types import NO_EVIDENCE, BatchPlacements, PlacementTarget, PlacementVerdict, SeadexEvidence
from pearlarr.releases import EpisodeKey, OrderOnlyKey
from pearlarr.tvdb_orders import OrderRow, OrdersByName
from pearlarr.window_placement import WindowedPlacements, place_across_windows

from .builders import (
    batch,
    blind,
    by_name,
    final_windows,
    merged_of,
    parsed_info,
    parsed_specials,
    place,
    series_index,
    title_window,
    torrent_listing,
    under_window,
    verdicts_of,
)


def _order(*rows: tuple[int, int, int]) -> tuple[OrderRow, ...]:
    """An order's rows from `(season, episode, tvdb_id)`. `series_index` gives each episode its id as its TVDB id."""

    return tuple(OrderRow(season, episode, tvdb_id, None) for season, episode, tvdb_id in rows)


class TestInterleavedSpecialByOrder:
    """A season-episode sequence numbered past its season's end, the way a DVD order counts the special in.

    `TestInterleavedSpecial` pins how the aired order reads it under the season's window: from the special on
    the keys are rejected. An order that counts the special explains every key, so the misnumbered members go where
    it reads them, the special's window placing the one it holds.
    """

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
        EpisodeKey(1, 3): 603,
        EpisodeKey(0, 1): 501,
        EpisodeKey(2, 1): 701,
        EpisodeKey(2, 2): 702,
    }
    _ABSOLUTES: ClassVar[dict[int, int]] = {601: 1, 602: 2, 501: 3, 603: 4, 701: 5, 702: 6}
    _DVD: ClassVar[tuple[OrderRow, ...]] = _order((1, 1, 601), (1, 2, 602), (1, 3, 501), (1, 4, 603), (2, 1, 701))
    """The DVD order: the special is its third episode, the aired third its fourth."""

    _RUN: ClassVar[list[str]] = [f"[grp] Show - S01E0{n}.mkv" for n in (1, 2, 3, 4)]
    _PARSED: ClassVar[dict[str, ParsedFileInfo | None]] = {
        _RUN[0]: parsed_info(season=1, episodes=(1,), matched=((1, 1),)),
        _RUN[1]: parsed_info(season=1, episodes=(2,), matched=((1, 2),)),
        _RUN[2]: parsed_info(season=1, episodes=(3,), matched=((1, 3),)),
        _RUN[3]: parsed_info(season=1, episodes=(4,)),
    }
    _SEASON: ClassVar[list[int]] = [601, 602, 603]

    @classmethod
    def _target(
        cls,
        resolved: list[int],
        orders: OrdersByName,
        *,
        listed: tuple[int, ...] | None = None,
        absolutes: dict[int, int] | None = None,
        titles: dict[int, str] | None = None,
    ) -> PlacementTarget:
        index = series_index(
            cls._ID_BY_KEY, absolutes=cls._ABSOLUTES if absolutes is None else absolutes, titles=titles, orders=orders
        )
        listing = NO_EVIDENCE if listed is None else SeadexEvidence(torrent_listing(title_window(1, listed)))
        return PlacementTarget(resolved, index, title_names=TitleNames("Show"), seadex=listing)

    def _place(self, orders: OrdersByName, listed: tuple[int, ...] | None = None) -> BatchPlacements:
        """The sequence under the season's window alone."""

        return place(self._PARSED, self._target(self._SEASON, orders, listed=listed))

    @classmethod
    def _renamed(cls, at: int, name: str) -> dict[str, ParsedFileInfo | None]:
        """The sequence's parses with the member at `at` named `name`."""

        return {name if member == cls._RUN[at] else member: info for member, info in cls._PARSED.items()}

    def test_the_final_unplaced_members_go_where_the_order_reads_them(self) -> None:
        windows = (self._target(self._SEASON, {"dvd": self._DVD}), self._target([501], {"dvd": self._DVD}))

        result = place_across_windows(batch(self._PARSED), windows)
        own = result.seen_by(0, {0})

        assert by_name(merged_of(result)) == {
            self._RUN[0]: ((601,), PlacementVerdict.EXACT),
            self._RUN[1]: ((602,), PlacementVerdict.EXACT),
            self._RUN[2]: ((501,), PlacementVerdict.OTHER_ORDER),
            self._RUN[3]: ((603,), PlacementVerdict.OTHER_ORDER),
        }
        assert final_windows(result)[self._RUN[2]] == 1
        assert not own.misnumbered
        assert own.hand_import_verdict is None

    def test_a_file_read_outside_the_window_is_left_for_the_window_that_holds_it(self) -> None:
        result = self._place({"dvd": self._DVD})

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(result)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_a_final_unplaced_member_read_onto_a_used_episode_of_the_window_stays_final_unplaced(self) -> None:
        # Another file is already placed on the aired third episode, which the order reads "S01E04" as.
        target = self._target(self._SEASON, {"dvd": self._DVD}).using([603])

        result = place(self._PARSED, target)

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(result)[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)

    def test_a_file_left_for_an_episode_no_window_holds_stays_its_titles_hand_import(self) -> None:
        # The second window is the next season's, so no window holds the special.
        windows = (self._target(self._SEASON, {"dvd": self._DVD}), self._target([701], {"dvd": self._DVD}))

        result = place_across_windows(batch(self._PARSED), windows)
        own = result.seen_by(0, {0})

        assert by_name(under_window(result, 1))[self._RUN[2]] == ((), PlacementVerdict.OUTSIDE_WINDOW)
        assert by_name(merged_of(result))[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert own.placements.hand_import_files == (self._RUN[2],)
        assert by_name(result.seen_by(1, {1}).placements)[self._RUN[2]] == ((), PlacementVerdict.ANOTHER_TITLES)
        assert own.hand_import_verdict is PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW
        assert not own.misnumbered

    def test_a_file_outside_the_window_read_outside_the_window_keeps_its_verdict(self) -> None:
        # Spelled out and matched as the next season's first episode, which the order reads as its second: outside
        # the window either way.
        beyond = "[grp] Show - S02E01.mkv"
        parsed = {**self._PARSED, beyond: parsed_info(season=2, episodes=(1,), matched=((2, 1),))}
        order = _order((1, 1, 601), (1, 2, 602), (1, 3, 501), (1, 4, 603), (2, 1, 702))

        result = place(parsed, self._target(self._SEASON, {"dvd": order}))

        assert by_name(result)[beyond] == ((), PlacementVerdict.OUTSIDE_WINDOW)
        assert by_name(result)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    _SPLIT: ClassVar[tuple[OrderRow, ...]] = _order((3, 1, 9001), (3, 2, 9002), (3, 3, 601), (3, 4, 602), (3, 5, 603))
    """An order that numbers the season as its third: it opens with a two-part episode Sonarr doesn't list, then
    the aired episodes follow."""

    _THIRD: ClassVar[list[str]] = [f"[grp] Show - S03E0{n}.mkv" for n in (1, 2, 3, 4, 5)]
    _THIRD_PARSED: ClassVar[dict[str, ParsedFileInfo | None]] = {
        name: parsed_info(season=3, episodes=(n,)) for n, name in enumerate(_THIRD, start=1)
    }

    def test_a_key_the_order_gives_to_no_episode_of_sonarrs_is_left_for_a_hand_import(self) -> None:
        result = place(self._THIRD_PARSED, self._target(self._SEASON, {"dvd": self._SPLIT}))

        assert by_name(result) == {
            self._THIRD[0]: ((), PlacementVerdict.ORDER_ONLY),
            self._THIRD[1]: ((), PlacementVerdict.ORDER_ONLY),
            self._THIRD[2]: ((601,), PlacementVerdict.OTHER_ORDER),
            self._THIRD[3]: ((602,), PlacementVerdict.OTHER_ORDER),
            self._THIRD[4]: ((603,), PlacementVerdict.OTHER_ORDER),
        }
        assert result.order_only == {
            self._THIRD[0]: OrderOnlyKey("dvd", EpisodeKey(3, 1)),
            self._THIRD[1]: OrderOnlyKey("dvd", EpisodeKey(3, 2)),
        }
        assert result.hand_import_files == (self._THIRD[0], self._THIRD[1])

    def test_a_file_spanning_two_keys_only_the_order_has_is_left_under_the_first(self) -> None:
        # The two halves in one file.
        pair = "[grp] Show - S03E01-02.mkv"
        parsed = {pair: parsed_info(season=3, episodes=(1, 2)), **self._THIRD_PARSED}
        del parsed[self._THIRD[0]], parsed[self._THIRD[1]]

        result = place(parsed, self._target(self._SEASON, {"dvd": self._SPLIT}))

        assert result.order_only == {pair: OrderOnlyKey("dvd", EpisodeKey(3, 1))}

    def test_keys_only_the_order_has_stay_unplaced_when_it_reads_no_file_onto_an_episode(self) -> None:
        halves = {name: self._THIRD_PARSED[name] for name in self._THIRD[:2]}

        result = place(halves, self._target(self._SEASON, {"dvd": self._SPLIT}))

        assert by_name(result) == dict.fromkeys(self._THIRD[:2], ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_key_two_orders_give_to_no_episode_of_sonarrs_stays_unplaced(self) -> None:
        orders = {"dvd": self._SPLIT, "altdvd": self._SPLIT}

        result = place(self._THIRD_PARSED, self._target(self._SEASON, orders))

        assert by_name(result)[self._THIRD[0]] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert by_name(result)[self._THIRD[2]] == ((601,), PlacementVerdict.OTHER_ORDER)
        assert result.order_only == {}

    def test_a_key_one_order_reads_onto_an_episode_and_another_gives_to_none_of_sonarrs_is_left(self) -> None:
        # The second order opens its third season on the first aired episode and gives its next two numbers to
        # episodes Sonarr doesn't list, so the orders differ on "S03E01" and "S03E03".
        whole = _order((3, 1, 601), (3, 2, 9002), (3, 3, 9003), (3, 4, 602), (3, 5, 603))

        result = place(self._THIRD_PARSED, self._target(self._SEASON, {"dvd": self._SPLIT, "altdvd": whole}))

        assert by_name(result)[self._THIRD[0]] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert by_name(result)[self._THIRD[2]] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert by_name(result)[self._THIRD[3]] == ((602,), PlacementVerdict.OTHER_ORDER)

    def test_a_key_only_the_order_has_is_left_for_a_hand_import_though_it_reads_the_rest_as_aired(self) -> None:
        # No special in the absolutes, so the aired keys place three and find no episode for "S01E04". The order
        # numbers the season as aired and has a fourth episode Sonarr doesn't list.
        longer = _order((1, 1, 601), (1, 2, 602), (1, 3, 603), (1, 4, 9001))
        target = self._target(self._SEASON, {"dvd": longer}, absolutes={601: 1, 602: 2, 603: 3})

        result = place(self._PARSED, target)

        assert result.order_only == {self._RUN[3]: OrderOnlyKey("dvd", EpisodeKey(1, 4))}

    def test_a_final_unplaced_member_on_a_key_only_the_order_has_stays_final_unplaced(self) -> None:
        # The order's third episode is one Sonarr doesn't list, and its fourth the aired third.
        order = _order((1, 1, 601), (1, 2, 602), (1, 3, 9001), (1, 4, 603))

        result = self._place({"dvd": order})

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(result)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_an_order_only_half_outranks_a_window_that_only_found_no_episode_for_it(self) -> None:
        orderless = self._target([701], {})
        windows = (orderless, self._target(self._SEASON, {"dvd": self._SPLIT}))

        alone = place(self._THIRD_PARSED, orderless)
        result = place_across_windows(batch(self._THIRD_PARSED), windows)

        assert by_name(alone)[self._THIRD[0]] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert merged_of(result).order_only[self._THIRD[0]] == OrderOnlyKey("dvd", EpisodeKey(3, 1))
        assert by_name(merged_of(result))[self._THIRD[2]] == ((601,), PlacementVerdict.OTHER_ORDER)
        # It's the hand import of the title whose order read it, and another title's for the one that found no
        # episode for it.
        assert result.seen_by(1, {1}).placements.hand_import_files == (self._THIRD[0], self._THIRD[1])
        assert by_name(result.seen_by(0, {0}).placements)[self._THIRD[0]] == ((), PlacementVerdict.ANOTHER_TITLES)

    def test_a_window_proving_an_order_only_file_a_duplicate_outranks_the_orders_verdict(self) -> None:
        # A second cut of the first half, with the same season and episode. Another series has a third season, and
        # its window places the first cut on that season's first episode, so the second cut is its duplicate there.
        uncut = "[grp] Show - S03E01 (Uncut).mkv"
        parsed = {**self._THIRD_PARSED, uncut: self._THIRD_PARSED[self._THIRD[0]]}
        other = PlacementTarget([801], series_index({EpisodeKey(3, 1): 801}), title_names=TitleNames("Show"))

        result = place_across_windows(batch(parsed), (self._target(self._SEASON, {"dvd": self._SPLIT}), other))

        assert by_name(under_window(result, 0))[uncut] == ((), PlacementVerdict.ORDER_ONLY)
        assert by_name(merged_of(result))[uncut] == ((), PlacementVerdict.DUPLICATE)

    _SHIFTED: ClassVar[tuple[OrderRow, ...]] = _order((1, 1, 501), (1, 2, 601), (1, 3, 602), (1, 4, 603), (2, 1, 701))
    """An order that counts the special first: it reads "S01E01" as the special."""

    def test_an_order_placing_a_placed_member_on_another_episode_places_nothing(self) -> None:
        # The aired key placed "S01E01" as episode 1.
        result = self._place({"dvd": self._SHIFTED})

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(result)[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)

    def test_an_order_placing_a_member_in_the_file_map_on_another_episode_places_nothing(self) -> None:
        # An earlier grab stored the first two members on their aired episodes, and only the last two are placed.
        file_map = {self._RUN[0]: (601,), self._RUN[1]: (602,)}
        target = self._target(self._SEASON, {"dvd": self._SHIFTED})

        result = place(self._PARSED, target, self._RUN[2:], file_map=file_map)

        assert by_name(result) == dict.fromkeys(self._RUN[2:], ((), PlacementVerdict.MISNUMBERED))

    def test_an_order_placing_two_files_onto_one_episode_explains_nothing(self) -> None:
        # A second cut of the fourth file, with the same season and episode: the order reads both as the aired third
        # episode.
        uncut = "[grp] Show - S01E04 (Uncut).mkv"
        parsed = {**self._PARSED, uncut: self._PARSED[self._RUN[3]]}

        result = place(parsed, self._target(self._SEASON, {"dvd": self._DVD}))

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(result)[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)

    def test_an_order_moving_a_file_an_earlier_window_placed_places_nothing_under_a_later_one(self) -> None:
        # No special in the absolutes, so the aired keys place three and find no episode for "S01E04". The order
        # swaps the first two episodes and reads "S01E04" as the special. The later window holds the special and one
        # more episode, so two ids are unused and only the order could place the unplaced file.
        swapped = _order((1, 1, 602), (1, 2, 601), (1, 3, 603), (1, 4, 501))
        plain = {601: 1, 602: 2, 603: 3}
        windows = (
            self._target(self._SEASON, {"dvd": swapped}, absolutes=plain),
            self._target([501, 701], {"dvd": swapped}, absolutes=plain),
        )

        result = place_across_windows(batch(self._PARSED), windows)

        assert merged_of(result).placed == {self._RUN[0]: (601,), self._RUN[1]: (602,), self._RUN[2]: (603,)}
        assert by_name(merged_of(result))[self._RUN[3]] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    def test_a_torrent_the_aired_order_reads_whole_is_never_read_another_way(self) -> None:
        # Every key is an aired one. The order reads "S00E01" as the aired third episode, which the torrent leaves
        # unused.
        special = "[grp] Show - S00E01.mkv"
        parsed = {
            self._RUN[0]: self._PARSED[self._RUN[0]],
            self._RUN[1]: self._PARSED[self._RUN[1]],
            special: parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        order = _order((1, 1, 601), (1, 2, 602), (0, 1, 603), (0, 2, 501))
        listed = (*self._SEASON, 501)
        windows = tuple(self._target(ids, {"dvd": order}, listed=listed) for ids in (self._SEASON, [501]))

        result = place_across_windows(batch(parsed), windows)

        assert by_name(merged_of(result))[special] == ((501,), PlacementVerdict.EXACT)

    _NEXT: ClassVar[list[str]] = [f"[grp] Show - S02E0{n}.mkv" for n in (1, 2)]
    """Two files spelled out and matched as the next season's episodes, which a listing of the season leaves out."""
    _NEXT_PARSED: ClassVar[dict[str, ParsedFileInfo | None]] = {
        name: parsed_info(season=2, episodes=(n,), matched=((2, n),)) for n, name in enumerate(_NEXT, start=1)
    }

    def _place_next(self, order: tuple[OrderRow, ...], to_place: list[str] | None = None) -> BatchPlacements:
        """`_NEXT` under the season's window, listed under the season alone."""

        target = self._target(self._SEASON, {"dvd": order}, listed=tuple(self._SEASON))
        return place(self._NEXT_PARSED, target, to_place)

    def test_aired_keys_naming_episodes_no_listed_title_covers_go_where_the_order_reads_them(self) -> None:
        # The order numbers the season's last two episodes as the next season's.
        result = self._place_next(_order((1, 1, 601), (2, 1, 602), (2, 2, 603)))

        assert by_name(result) == {
            self._NEXT[0]: ((602,), PlacementVerdict.OTHER_ORDER),
            self._NEXT[1]: ((603,), PlacementVerdict.OTHER_ORDER),
        }

    def test_a_counted_special_no_title_lists_leaves_the_season_placed_by_the_order(self) -> None:
        result = self._place({"dvd": self._DVD}, listed=(601, 602, 603))

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(result)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_a_later_window_placing_a_file_left_for_the_orders_episode_off_it_is_dropped(self) -> None:
        # The second window holds four other specials, and its sequence-over-parse pass takes the third file by
        # position.
        specials = {EpisodeKey(0, n): 500 + n for n in (2, 3, 4, 5)}
        index = series_index(self._ID_BY_KEY | specials, absolutes=self._ABSOLUTES, orders={"dvd": self._DVD})
        windows = tuple(
            PlacementTarget(ids, index, title_names=TitleNames("Show")) for ids in (self._SEASON, [502, 503, 504, 505])
        )

        result = place_across_windows(batch(self._PARSED), windows)

        assert by_name(under_window(result, 1))[self._RUN[2]] == ((504,), PlacementVerdict.SEQUENCE_OVER_PARSE)
        assert by_name(merged_of(result))[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)

    def test_a_provisional_verdict_for_another_orders_episode_outranks_a_window_proving_the_file_a_duplicate(
        self,
    ) -> None:
        # Another series' window holds its own third episode, which an earlier grab stored a file on, so "S01E03"
        # is that file's duplicate there.
        stored = "[grp] Other Show - 03.mkv"
        parsed = {**self._PARSED, stored: parsed_info(absolutes=(3,), matched=((1, 3),))}
        other = PlacementTarget([803], series_index({EpisodeKey(1, 3): 803}), title_names=TitleNames("Other Show"))
        windows = (self._target(self._SEASON, {"dvd": self._DVD}), other)

        result = place_across_windows(batch(parsed, self._RUN, file_map={stored: (803,)}), windows)

        assert by_name(under_window(result, 1))[self._RUN[2]] == ((), PlacementVerdict.DUPLICATE)
        assert by_name(merged_of(result))[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)

    def test_a_later_window_placing_a_file_left_for_the_orders_episode_on_it_by_its_title_keeps_it(self) -> None:
        # The third file carries the special's title, so the special's window places it by title before the order
        # is asked. The title contradicts the first two keys, which the order reads as the aired order does: the
        # season's window places them.
        titled = "[grp] Show - S01E03 - Lost Lantern.mkv"
        parsed = {**self._PARSED, titled: self._PARSED[self._RUN[2]]}
        del parsed[self._RUN[2]]
        index = series_index(
            self._ID_BY_KEY, absolutes=self._ABSOLUTES, titles={501: "Lost Lantern"}, orders={"dvd": self._DVD}
        )
        windows = tuple(PlacementTarget(ids, index, title_names=TitleNames("Show")) for ids in (self._SEASON, [501]))

        result = place_across_windows(batch(parsed), windows)

        assert by_name(under_window(result, 0))[titled] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(under_window(result, 0))[self._RUN[0]] == ((601,), PlacementVerdict.OTHER_ORDER)
        assert by_name(merged_of(result))[titled] == ((501,), PlacementVerdict.EPISODE_TITLE)

    def test_a_key_an_order_reads_onto_an_episode_is_still_no_number_to_the_single_file_pass(self) -> None:
        # No special in the absolutes, so the aired keys place three and find no episode for "S01E04". The order
        # reads it as the next season's first episode, which no window holds.
        run_on = _order((1, 1, 601), (1, 2, 602), (1, 3, 603), (1, 4, 701))
        plain = {601: 1, 602: 2, 603: 3}
        windows = (
            self._target(self._SEASON, {"dvd": run_on}, absolutes=plain),
            self._target([501], {"dvd": run_on}, absolutes=plain),
        )

        result = place_across_windows(batch(self._PARSED), windows)

        assert by_name(merged_of(result))[self._RUN[3]] == ((501,), PlacementVerdict.SINGLE)

    _PARTIAL: ClassVar[tuple[OrderRow, ...]] = _order((1, 1, 601), (1, 2, 602), (1, 3, 501), (2, 1, 701))
    """The DVD order without its "S01E04"."""

    def test_an_order_missing_a_key_places_nothing_without_a_number_or_title(self) -> None:
        # No name carries a number or a title.
        result = self._place({"dvd": self._PARTIAL})

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(result)[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)

    def test_a_final_unplaced_members_title_lets_an_order_missing_a_key_place_it(self) -> None:
        # The third file carries the special's title, and the order reads it as the special.
        titled = "[grp] Show - S01E03 - Lost Lantern.mkv"
        target = self._target(self._SEASON, {"dvd": self._PARTIAL}, titles={501: "Lost Lantern"})

        result = place(self._renamed(2, titled), target)

        assert by_name(result)[titled] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(result)[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)

    def test_a_placed_members_title_does_not_let_an_order_missing_a_key_place_anything(self) -> None:
        # The first file carries its aired episode's title, and the aired key placed it there.
        titled = "[grp] Show - S01E01 - Festival Night.mkv"
        target = self._target(self._SEASON, {"dvd": self._PARTIAL}, titles={601: "Festival Night"})

        result = place(self._renamed(0, titled), target)

        assert by_name(result)[titled] == ((601,), PlacementVerdict.EXACT)
        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)

    def test_an_order_missing_an_already_placed_members_key_reads_no_file_apart_from_another_order(self) -> None:
        # The third file's bracketed number names the special, which would let an order missing a key place it. The
        # second order has no "S01E01" and reads "S01E04" as the next season's first episode. The first two members
        # are already placed each way: placed by their aired keys, stored by an earlier grab, placed by an earlier
        # window.
        counted = "[grp] Show - S01E03 (0003).mkv"
        parsed = self._renamed(2, counted)
        file_map = {self._RUN[0]: (601,), self._RUN[1]: (602,)}
        headless = _order((1, 2, 602), (1, 3, 501), (1, 4, 701))
        orders = {"dvd": self._DVD, "alternate": headless}
        target = self._target(self._SEASON, orders)
        windows = tuple(self._target(ids, orders) for ids in ([601, 602], [603]))

        placed = place(parsed, target)
        stored = place(parsed, target, [counted, self._RUN[3]], file_map=file_map)
        windowed = place_across_windows(batch(parsed), windows)

        assert by_name(placed)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)
        assert by_name(stored)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)
        assert by_name(merged_of(windowed))[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    _TITLED: ClassVar[str] = "[grp] Show - S01E03 - Festival Night.mkv"
    """The third file under the aired second episode's title, which contradicts the first two keys."""

    @classmethod
    def _contradicted(cls, order: tuple[OrderRow, ...]) -> WindowedPlacements:
        """The sequence with `_TITLED` third, under the season's window with `order` and no special in the absolutes."""

        plain = {601: 1, 602: 2, 603: 3}
        target = cls._target(cls._SEASON, {"dvd": order}, absolutes=plain, titles={602: "Festival Night"})
        return place_across_windows(batch(cls._renamed(2, cls._TITLED)), (target,))

    def test_a_contradicted_member_read_outside_the_window_is_left_for_the_window_that_holds_it(self) -> None:
        # The order counts the special second, so it reads the third file as the episode its title names.
        second = _order((1, 1, 601), (1, 2, 501), (1, 3, 602), (1, 4, 603))

        result = self._contradicted(second)

        assert by_name(merged_of(result))[self._RUN[1]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(merged_of(result))[self._TITLED] == ((602,), PlacementVerdict.OTHER_ORDER)

    def test_a_title_contradiction_is_the_hand_import_verdict_when_both_kinds_are_left(self) -> None:
        # The order counts the special second and has no "S01E01": it leaves the first file contradicted and the
        # second for the special's window. The third file's title lets the order place it.
        no_first = _order((1, 2, 501), (1, 3, 602), (1, 4, 603))

        result = self._contradicted(no_first)

        assert by_name(merged_of(result))[self._RUN[0]] == ((), PlacementVerdict.EPISODE_TITLES_CONTRADICT)
        assert by_name(merged_of(result))[self._RUN[1]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert result.seen_by(0, {0}).hand_import_verdict is PlacementVerdict.EPISODE_TITLES_CONTRADICT

    def test_two_orders_placing_the_members_apart_leave_them_until_the_listing_picks_one(self) -> None:
        # The second order leaves the special out and runs on into the next season: "S01E03" is the aired third
        # episode under it and "S01E04" the next season's first.
        run_on = _order((1, 1, 601), (1, 2, 602), (1, 3, 603), (1, 4, 701))
        orders = {"dvd": self._DVD, "alternate": run_on}

        unlisted = self._place(orders)
        listed = self._place(orders, listed=(601, 602, 603, 501))

        assert by_name(unlisted)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(unlisted)[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(listed)[self._RUN[2]] == ((), PlacementVerdict.OTHER_ORDER_OUTSIDE_WINDOW)
        assert by_name(listed)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_an_order_placing_a_misnumbered_file_as_aired_places_it_nowhere(self) -> None:
        # The order counts the special last: it reads "S01E03" as the aired third episode, as the aired order did.
        appended = _order((1, 1, 601), (1, 2, 602), (1, 3, 603), (1, 4, 501), (2, 1, 701))
        windows = (self._target(self._SEASON, {"dvd": appended}), self._target([501], {"dvd": appended}))

        result = place_across_windows(batch(self._PARSED), windows)

        assert by_name(merged_of(result))[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(merged_of(result))[self._RUN[3]] == ((), PlacementVerdict.MISNUMBERED)
        assert under_window(result, 1).placed == {}

    def test_a_file_sized_as_another_episode_is_left_alone_when_the_order_reads_it_as_aired(self) -> None:
        # "S01E01" has the size of the single-file release SeaDex lists for another episode, and the order reads
        # it as its spelled-out season and episode do. Away: that episode is the next season's first, which the
        # season's window doesn't hold, so the file is outside the window there. With the file map: an earlier grab
        # stored "S01E02" on that episode, so no episode is found for the file. Either way the order still places
        # "S01E04".
        index = self._target(self._SEASON, {"dvd": self._DVD}).index

        def season(listed: tuple[int, ...], sized_as: int) -> PlacementTarget:
            listing = SeadexEvidence(torrent_listing(title_window(1, listed)), {self._RUN[0]: sized_as})
            return PlacementTarget(self._SEASON, index, title_names=TitleNames("Show"), seadex=listing)

        away = by_name(place(self._PARSED, season((*self._SEASON, 701), 701)))
        rest = [self._RUN[0], *self._RUN[2:]]
        with_file_map = by_name(
            place(self._PARSED, season((*self._SEASON,), 602), rest, file_map={self._RUN[1]: (602,)})
        )

        assert away[self._RUN[0]] == ((), PlacementVerdict.OUTSIDE_WINDOW)
        assert with_file_map[self._RUN[0]] == ((), PlacementVerdict.NO_EPISODE_FOUND)
        assert away[self._RUN[3]] == with_file_map[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_an_aired_special_outside_the_listing_is_no_half_to_an_order_numbering_the_rest_as_aired(self) -> None:
        # The order gives "S00E01" to an episode Sonarr doesn't list and numbers the season as aired.
        special = "[grp] Show - S00E01.mkv"
        parsed = {
            **{name: self._PARSED[name] for name in self._RUN[:3]},
            special: parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        order = _order((1, 1, 601), (1, 2, 602), (1, 3, 603), (0, 1, 9001))

        result = place(parsed, self._target(self._SEASON, {"dvd": order}, listed=tuple(self._SEASON)))

        assert by_name(result)[special] == ((), PlacementVerdict.OUTSIDE_WINDOW)

    def test_an_order_placing_a_misnumbered_file_as_aired_names_no_half_either(self) -> None:
        # The order counts the special last, as above, and gives "S03E01" to an episode Sonarr doesn't list.
        half = "[grp] Show - S03E01.mkv"
        parsed = {**self._PARSED, half: parsed_info(season=3, episodes=(1,))}
        appended = _order((1, 1, 601), (1, 2, 602), (1, 3, 603), (1, 4, 501), (2, 1, 701), (3, 1, 9001))

        result = place(parsed, self._target(self._SEASON, {"dvd": appended}))

        assert by_name(result)[half] == ((), PlacementVerdict.NO_EPISODE_FOUND)

    _LANTERN: ClassVar[str] = "[grp] Show - S01E03 - Lost Lantern.mkv"
    """The third file under the special's title."""

    def _place_short(self, order: tuple[OrderRow, ...]) -> BatchPlacements:
        """The first two members and `_LANTERN`, whose title contradicts them, under the season's window.

        The torrent ends at the third file, so no number runs past the season's end.
        """

        parsed = {**{name: self._PARSED[name] for name in self._RUN[:2]}, self._LANTERN: self._PARSED[self._RUN[2]]}
        return place(parsed, self._target(self._SEASON, {"dvd": order}, titles={501: "Lost Lantern"}))

    def test_a_title_naming_another_episode_than_its_key_is_a_file_the_aired_order_fails_on(self) -> None:
        # The order reads the third file as the special its title names.
        result = self._place_short(self._DVD)

        assert by_name(result) == {
            self._RUN[0]: ((601,), PlacementVerdict.OTHER_ORDER),
            self._RUN[1]: ((602,), PlacementVerdict.OTHER_ORDER),
            self._LANTERN: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_a_key_sonarr_doesnt_list_answers_no_contradicted_file_and_no_title_that_names_an_episode(self) -> None:
        # The order reads "S01E02" as aired and gives the other two keys to episodes Sonarr doesn't list: the
        # contradicted "S01E01", and "S01E03", whose title names the special.
        result = self._place_short(_order((1, 1, 9001), (1, 2, 602), (1, 3, 9002)))

        assert by_name(result) == {
            self._RUN[0]: ((), PlacementVerdict.EPISODE_TITLES_CONTRADICT),
            self._RUN[1]: ((), PlacementVerdict.EPISODE_TITLES_CONTRADICT),
            self._LANTERN: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_an_order_moving_a_placed_member_still_reads_a_final_unplaced_one_apart_from_another_order(self) -> None:
        # The second order swaps the first two episodes, which the aired keys placed. It reads "S01E03" as the next
        # season's first episode, where the DVD order reads the special, and "S01E04" as the DVD order does.
        swapped = _order((1, 1, 602), (1, 2, 601), (1, 3, 701), (1, 4, 603))

        result = self._place({"dvd": self._DVD, "alternate": swapped})

        assert by_name(result)[self._RUN[2]] == ((), PlacementVerdict.MISNUMBERED)
        assert by_name(result)[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_an_order_leaving_one_unlisted_file_outside_the_listing_explains_nothing(self) -> None:
        # The order numbers the season's second episode as "S02E01" and the next season's second episode as aired,
        # so "S02E02" stays unlisted.
        result = self._place_next(_order((1, 1, 601), (2, 1, 602), (2, 2, 702)))

        assert verdicts_of(result) == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_an_order_moving_an_unlisted_file_onto_another_unlisted_episode_explains_nothing(self) -> None:
        # The order reads "S02E02" as the next season's first episode: off its aired episode, and still unlisted.
        result = self._place_next(_order((1, 1, 601), (2, 1, 602), (2, 2, 701)))

        assert verdicts_of(result) == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_an_unlisted_file_the_batch_does_not_place_still_keeps_the_order_from_explaining(self) -> None:
        # Only "S02E01" is placed. The order leaves the torrent's "S02E02" on its aired episode, as above.
        result = self._place_next(_order((1, 1, 601), (2, 1, 602), (2, 2, 702)), self._NEXT[:1])

        assert verdicts_of(result) == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_a_file_spanning_two_keys_is_in_the_window_or_the_listing_only_when_both_episodes_are(self) -> None:
        # One file spelled out and matched as the next season's first two episodes, and one as the special. The
        # order reads the special and half of the pair onto the season, and the pair's other half as aired.
        pair, special = "[grp] Show - S02E01-E02.mkv", "[grp] Show - S00E01.mkv"
        parsed = {
            pair: parsed_info(season=2, episodes=(1, 2), matched=((2, 1), (2, 2))),
            special: parsed_info(season=0, episodes=(1,), matched=((0, 1),)),
        }
        half = _order((1, 1, 601), (2, 1, 602), (2, 2, 702), (0, 1, 603))

        result = place(parsed, self._target(self._SEASON, {"dvd": half}, listed=tuple(self._SEASON)))

        assert verdicts_of(result) == {PlacementVerdict.OUTSIDE_WINDOW}

    def test_a_file_an_earlier_window_placed_is_not_outside_a_later_one(self) -> None:
        # The first window holds two episodes and places their files. Under the special's window those two sit
        # outside it, and "S01E03" is the one file left for the order to bring in.
        parsed = {name: self._PARSED[name] for name in self._RUN[:3]}
        windows = (self._target([601, 602], {"dvd": self._DVD}), self._target([501], {"dvd": self._DVD}))

        result = place_across_windows(batch(parsed), windows)

        assert by_name(merged_of(result))[self._RUN[2]] == ((501,), PlacementVerdict.OTHER_ORDER)

    def test_orders_placing_apart_a_file_an_earlier_window_placed_still_place_the_rest(self) -> None:
        # The next season's window comes first and places "S02E01" by its aired key. The second order reads that
        # file as the next season's second episode and the sequence as the DVD order does. No title lists the
        # special.
        beyond = "[grp] Show - S02E01.mkv"
        parsed = {**self._PARSED, beyond: parsed_info(season=2, episodes=(1,), matched=((2, 1),))}
        later = _order((1, 1, 601), (1, 2, 602), (1, 3, 501), (1, 4, 603), (2, 1, 702))
        orders = {"dvd": self._DVD, "alternate": later}
        listed = (701, *self._SEASON)
        windows = tuple(self._target(ids, orders, listed=listed) for ids in ([701], self._SEASON))

        result = place_across_windows(batch(parsed), windows)

        assert by_name(merged_of(result))[beyond] == ((701,), PlacementVerdict.EXACT)
        assert by_name(merged_of(result))[self._RUN[3]] == ((603,), PlacementVerdict.OTHER_ORDER)

    def test_a_file_spanning_a_key_the_aired_list_lacks_is_one_the_aired_order_cant_read(self) -> None:
        # One file spelled out as the third and fourth episodes, where the season ends at the third. No absolute
        # number counts the special in, and the window holds the special too.
        double = "[grp] Show - S01E03-E04.mkv"
        parsed = {
            **{name: self._PARSED[name] for name in self._RUN[:2]},
            double: parsed_info(season=1, episodes=(3, 4), matched=((1, 3),)),
        }
        target = self._target([*self._SEASON, 501], {"dvd": self._DVD}, absolutes={601: 1, 602: 2, 603: 3})

        result = place(parsed, target)

        assert by_name(result)[double] == ((501, 603), PlacementVerdict.OTHER_ORDER)

    def test_a_lower_version_of_a_number_is_no_second_file_on_the_orders_episode(self) -> None:
        # The torrent holds "S01E04" in two versions, which the order reads onto one episode.
        newer = "[grp] Show - S01E04v2.mkv"
        parsed = {**self._PARSED, newer: self._PARSED[self._RUN[3]]}

        result = place(parsed, self._target(self._SEASON, {"dvd": self._DVD}))

        assert by_name(result)[newer] == ((603,), PlacementVerdict.OTHER_ORDER)
        assert by_name(result)[self._RUN[3]] == ((), PlacementVerdict.NO_EPISODE_FOUND)


class TestMisnumberedSpecialsByOrder:
    """A batch of specials as wide as its listing and numbered off it, under an order that renumbers the specials."""

    _LISTED: ClassVar[tuple[int, ...]] = (502, 504, 506)

    @classmethod
    def _place(cls, order: tuple[OrderRow, ...]) -> BatchPlacements:
        """The batch numbered 1, 2 and 3 under the listed specials' window, with `order` as the DVD order."""

        index = series_index({EpisodeKey(0, n): 500 + n for n in range(1, 7)}, orders={"dvd": order})
        listing = SeadexEvidence(torrent_listing(title_window(1, cls._LISTED)))
        target = PlacementTarget(list(cls._LISTED), index, title_names=TitleNames("Show"), seadex=listing)
        return place(parsed_specials(1, 2, 3), target)

    def test_an_order_placing_a_special_off_the_listing_leaves_the_torrent_final_unplaced(self) -> None:
        # The order's specials run one behind the aired ones, so it reads the batch's "S00E02" as the third special.
        behind = _order(*((0, n, 500 + n + 1) for n in range(1, 6)))

        result = self._place(behind)

        assert verdicts_of(result) == {PlacementVerdict.MISNUMBERED}

    def test_an_order_placing_every_special_onto_the_listing_places_the_torrent(self) -> None:
        # The order numbers the three listed specials 1, 2 and 3, as the batch does.
        onto = _order(*((0, n, ep_id) for n, ep_id in enumerate(self._LISTED, start=1)))

        result = self._place(onto)

        assert by_name(result) == {
            name: ((ep_id,), PlacementVerdict.OTHER_ORDER)
            for name, ep_id in zip(parsed_specials(1, 2, 3), self._LISTED, strict=True)
        }


class TestKeyWithANumberByOrder:
    """A key whose bracketed number names another episode: the number is right about the order the key follows."""

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {
        EpisodeKey(1, 1): 601,
        EpisodeKey(1, 2): 602,
        EpisodeKey(2, 1): 701,
        EpisodeKey(2, 2): 702,
    }
    _ABSOLUTES: ClassVar[dict[int, int]] = {601: 1, 602: 2, 701: 3, 702: 4}
    _TITLES: ClassVar[dict[int, str]] = {602: "Festival Night", 701: "Sun and Moon", 702: "New Dawn"}
    _NAME: ClassVar[str] = "[grp] Show S02E01 (0004) (WEB 1080p) [ABCD1234].mkv"
    """Spelled out as the aired fourth episode's neighbour, numbered as the fourth episode itself."""

    _STREAMING: ClassVar[tuple[OrderRow, ...]] = _order((1, 1, 601), (1, 2, 602), (2, 1, 702), (2, 2, 701))
    """An order whose second season opens with the episode the number names."""

    _SPELLED_PARSE: ClassVar[ParsedFileInfo] = parsed_info(season=2, episodes=(1,), matched=((2, 1),))
    """Sonarr's parse of the name: the key, matched as the second season's first episode."""

    @classmethod
    def _target(cls, orders: OrdersByName) -> PlacementTarget:
        """The second season's window."""

        index = series_index(cls._ID_BY_KEY, absolutes=cls._ABSOLUTES, titles=cls._TITLES, orders=orders)
        return PlacementTarget([701, 702], index, title_names=TitleNames("Show"))

    @classmethod
    def _place(cls, orders: OrdersByName, name: str = _NAME) -> BatchPlacements:
        return place({name: cls._SPELLED_PARSE}, cls._target(orders))

    def test_the_number_names_the_order_the_key_follows(self) -> None:
        # The torrent also carries a special that neither Sonarr nor the order numbers, so the order doesn't number
        # every key and only the bracketed number lets it place anything.
        special = "[grp] Show S00E01 (WEB 1080p) [ABCD1234].mkv"
        parsed = {self._NAME: self._SPELLED_PARSE, special: parsed_info(season=0, episodes=(1,))}

        result = place(parsed, self._target({"alttwo": self._STREAMING}))

        assert by_name(result) == {
            self._NAME: ((702,), PlacementVerdict.OTHER_ORDER),
            special: ((), PlacementVerdict.NO_EPISODE_FOUND),
        }

    def test_an_order_the_number_contradicts_places_nothing_though_it_numbers_every_key(self) -> None:
        # The order's second season opens an episode early: it reads "S02E01" as the first season's last episode,
        # which is neither the aired order's episode nor the episode the number names.
        early = _order((1, 1, 601), (2, 1, 602), (2, 2, 701), (2, 3, 702))

        result = self._place({"dvd": early})

        assert by_name(result) == {self._NAME: ((), PlacementVerdict.MISNUMBERED)}

    def test_a_title_naming_a_third_episode_keeps_the_number_from_placing_the_file(self) -> None:
        # The number picks the streaming order, but the name carries the title of an episode of the first season.
        titled = "[grp] Show S02E01 (0004) - Festival Night (WEB 1080p) [ABCD1234].mkv"

        result = self._place({"alttwo": self._STREAMING}, name=titled)

        assert by_name(result)[titled] == ((), PlacementVerdict.MISNUMBERED)


class TestNameNumberSequenceByOrder:
    """A name-number Blu-ray torrent one file longer than its season: the DVD order splits the opener in two.

    The season airs 12 episodes, the torrent numbers 13 (`Show Ending S1 - 01` to `- 13`) and Sonarr parses nothing
    from them, so the pairing passes leave every file `NO_EPISODE_FOUND`. The DVD order numbers the season as its
    seventh: two halves Sonarr doesn't list, then the aired second episode on. The alternate order numbers it as its
    thirteenth with the merged opener first, so its thirteenth key is the next season's first episode.
    """

    _ID_BY_KEY: ClassVar[dict[EpisodeKey, int]] = {EpisodeKey(4, n): 400 + n for n in range(1, 13)} | {
        EpisodeKey(5, 1): 501
    }
    _SEASON: ClassVar[list[int]] = [400 + n for n in range(1, 13)]
    _DVD: ClassVar[tuple[OrderRow, ...]] = _order(
        (7, 1, 9001), (7, 2, 9002), *((7, n + 1, 400 + n) for n in range(2, 13)), (7, 14, 501), (7, 21, 401)
    )
    _ALT: ClassVar[tuple[OrderRow, ...]] = _order(
        *((13, n, 400 + n) for n in range(1, 13)), (13, 13, 501), (13, 20, 9001), (13, 21, 9002)
    )
    _RUN: ClassVar[list[str]] = [f"[grp] Show Ending S1 - {n:02d} [ABCD1234].mkv" for n in range(1, 14)]
    _NAMES: ClassVar[TitleNames] = TitleNames("Show", ("Show Ending",))
    _SECOND: ClassVar[list[str]] = [f"[grp] Show Ending S2 - {n:02d} [ABCD1234].mkv" for n in range(1, 8)]
    """A second sequence the same name of the title points at."""

    @classmethod
    def _target(
        cls, orders: OrdersByName, names: TitleNames = _NAMES, *, titles: dict[int, str] | None = None
    ) -> PlacementTarget:
        return PlacementTarget(
            cls._SEASON, series_index(cls._ID_BY_KEY, titles=titles, orders=orders), title_names=names
        )

    def _by_order(self) -> dict[str, tuple[tuple[int, ...], PlacementVerdict]]:
        return {
            **{self._RUN[n - 1]: ((), PlacementVerdict.ORDER_ONLY) for n in (1, 2)},
            **{self._RUN[n - 1]: ((400 + n - 1,), PlacementVerdict.OTHER_ORDER) for n in range(3, 14)},
        }

    def test_the_order_placing_the_whole_sequence_onto_the_window_places_it(self) -> None:
        result = place(blind(self._RUN), self._target({"dvd": self._DVD}))

        assert by_name(result) == self._by_order()
        assert result.order_only == {
            self._RUN[0]: OrderOnlyKey("dvd", EpisodeKey(7, 1)),
            self._RUN[1]: OrderOnlyKey("dvd", EpisodeKey(7, 2)),
        }

    def test_an_order_placing_a_member_outside_the_window_places_nothing(self) -> None:
        result = place(blind(self._RUN), self._target({"alternate": self._ALT}))

        assert by_name(result) == dict.fromkeys(self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_the_order_that_holds_the_sequence_wins_over_the_one_placing_past_the_window(self) -> None:
        result = place(blind(self._RUN), self._target({"dvd": self._DVD, "alternate": self._ALT}))

        assert by_name(result) == self._by_order()

    def test_a_sequence_no_longer_than_the_window_is_left_to_the_pairing_passes(self) -> None:
        # Twelve files for the twelve episodes: the torrent has no "- 01", and another file is placed on the aired
        # first episode, so no pairing pass places them either.
        rest = self._RUN[1:]

        result = place(blind(rest), self._target({"dvd": self._DVD}).using([401]))

        assert by_name(result) == dict.fromkeys(rest, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_sequence_longer_than_the_orders_season_is_left_alone(self) -> None:
        # Fourteen files, and the order's season has no fourteenth episode.
        longer = [*self._RUN, "[grp] Show Ending S1 - 14 [ABCD1234].mkv"]
        ends_at_13 = tuple(row for row in self._DVD if row.tvdb_id != 501)

        result = place(blind(longer), self._target({"dvd": ends_at_13}))

        assert by_name(result) == dict.fromkeys(longer, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_an_order_numbering_the_season_as_aired_explains_nothing(self) -> None:
        # The order's season is the aired one plus a thirteenth episode Sonarr doesn't list. It numbers the window
        # as the aired order does.
        as_aired = _order(*((4, n, 400 + n) for n in range(1, 13)), (4, 13, 9001))

        result = place(blind(self._RUN), self._target({"dvd": as_aired}))

        assert by_name(result) == dict.fromkeys(self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_an_order_splitting_the_season_in_two_reads_no_sequence(self) -> None:
        halves = _order(*((7, n, 400 + n) for n in range(1, 7)), *((8, n - 6, 400 + n) for n in range(7, 13)))

        result = place(blind(self._RUN), self._target({"dvd": halves}))

        assert by_name(result) == dict.fromkeys(self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_file_in_the_file_map_the_order_reads_outside_the_window_keeps_it_from_explaining_the_rest(self) -> None:
        # A sibling's grab stored "- 01" on the aired first episode. The order reads it as a half Sonarr doesn't
        # list, and reads no member onto that episode.
        rest = self._RUN[1:]

        result = place(blind(self._RUN), self._target({"dvd": self._DVD}), rest, file_map={self._RUN[0]: (401,)})

        assert by_name(result) == dict.fromkeys(rest, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_an_order_placing_a_member_onto_a_used_episode_explains_nothing(self) -> None:
        # Another file is placed on the aired fifth episode, which the order reads "- 06" as.
        result = place(blind(self._RUN), self._target({"dvd": self._DVD}).using([405]))

        assert by_name(result) == dict.fromkeys(self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_member_the_batch_does_not_place_is_passed_over(self) -> None:
        rest = [name for name in self._RUN if name != self._RUN[4]]

        result = place(blind(self._RUN), self._target({"dvd": self._DVD}), rest)

        assert by_name(result) == {name: placed for name, placed in self._by_order().items() if name in rest}

    def test_a_member_already_read_outside_the_window_keeps_the_order_from_explaining_the_rest(self) -> None:
        # Sonarr matched "- 13" as the next season's first episode, and another file is placed on the aired first
        # episode, so no pairing pass places the other twelve.
        parsed = {**blind(self._RUN), self._RUN[12]: parsed_info(absolutes=(13,), matched=((5, 1),))}

        result = place(parsed, self._target({"dvd": self._DVD}).using([401]))

        assert by_name(result) == {
            **dict.fromkeys(self._RUN[:12], ((), PlacementVerdict.NO_EPISODE_FOUND)),
            self._RUN[12]: ((), PlacementVerdict.OUTSIDE_WINDOW),
        }

    def test_two_members_tied_at_one_number_keep_the_order_from_explaining_the_sequence(self) -> None:
        # A second release of "- 05" under another hash: the order reads both as the aired fourth episode.
        names = [*self._RUN, "[grp] Show Ending S1 - 05 [EFGH5678].mkv"]

        result = place(blind(names), self._target({"dvd": self._DVD}))

        assert by_name(result) == dict.fromkeys(names, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_members_title_naming_another_episode_keeps_the_order_from_explaining_the_sequence(self) -> None:
        # The last file carries the next season's first title, and the order reads it as the aired twelfth episode.
        names = [*self._RUN[:12], "[grp] Show Ending S1 - 13 - New Dawn [ABCD1234].mkv"]

        result = place(blind(names), self._target({"dvd": self._DVD}, titles={501: "New Dawn"}))

        assert by_name(result) == dict.fromkeys(names, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_an_order_whose_season_misses_a_window_episode_reads_no_sequence(self) -> None:
        # The order lists the two halves and never the merged opener.
        no_opener = tuple(row for row in self._DVD if row.tvdb_id != 401)

        result = place(blind(self._RUN), self._target({"dvd": no_opener}))

        assert by_name(result) == dict.fromkeys(self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_season_episode_sequence_is_never_read_by_position(self) -> None:
        # Thirteen files spelled out into the first season, which neither Sonarr nor the order numbers.
        spelled_names = [f"[grp] Show Ending - S01E{n:02d} [ABCD1234].mkv" for n in range(1, 14)]
        parsed = {name: parsed_info(season=1, episodes=(n,)) for n, name in enumerate(spelled_names, start=1)}

        result = place(parsed, self._target({"dvd": self._DVD}))

        assert by_name(result) == dict.fromkeys(spelled_names, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_members_two_orders_read_apart_stay_unplaced(self) -> None:
        # The second order swaps the season's last two episodes.
        swap = {411: 412, 412: 411}
        swapped = _order(*((row.season, row.episode, swap.get(row.tvdb_id, row.tvdb_id)) for row in self._DVD))

        result = place(blind(self._RUN), self._target({"dvd": self._DVD, "altdvd": swapped}))

        apart = dict.fromkeys(self._RUN[11:], ((), PlacementVerdict.NO_EPISODE_FOUND))
        assert by_name(result) == {**self._by_order(), **apart}

    def test_a_sequence_no_title_names_is_left_alone(self) -> None:
        result = place(blind(self._RUN), self._target({"dvd": self._DVD}, TitleNames("Show", ("Show Beginning",))))

        assert by_name(result) == dict.fromkeys(self._RUN, ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_two_named_sequences_tie_and_neither_is_placed(self) -> None:
        result = place(blind([*self._RUN, *self._SECOND]), self._target({"dvd": self._DVD}))

        assert by_name(result) == dict.fromkeys((*self._RUN, *self._SECOND), ((), PlacementVerdict.NO_EPISODE_FOUND))

    def test_a_second_named_sequence_the_batch_does_not_place_is_no_tie(self) -> None:
        result = place(blind([*self._RUN, *self._SECOND]), self._target({"dvd": self._DVD}), self._RUN)

        assert by_name(result) == self._by_order()
