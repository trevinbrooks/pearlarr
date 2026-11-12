# pyright: strict
# pyright: reportPrivateUsage=false
# The title keys are the module's own; the pins on them read the private functions.
"""The numberings reader over real torrents: their names, Sonarr's parses, and the series rows as Sonarr holds them.

Each series is cut to the rows the read touches, with its real ids and absolute numbers. A case no real torrent
has is marked "Constructed".
"""

from collections.abc import Iterable, Mapping, Sequence

import pytest

from pearlarr.manual_import import EntryNames, EntryWindow
from pearlarr.numberings import NumberingReader, _marked_title, _norm_title, _raw_title
from pearlarr.placement_types import EpisodeIndex, SeedScope, TargetScope, episode_index
from pearlarr.placer import assign_episode_ids
from pearlarr.seadex_types import EpisodeKey, ParsedFileInfo

from .builders import batch, entry_window, parsed_info, seed_scope, series_index

type Row = tuple[int, int, int, int | None]
"""A Sonarr row: season, episode, id, absolute number."""


def _run(season: int, ids: Iterable[int], *, episode: int = 1, absolute: int | None = None) -> list[Row]:
    """Consecutive episodes of `season` from `episode` on, one per id, their absolutes counting up from `absolute`."""

    return [(season, episode + n, ep_id, None if absolute is None else absolute + n) for n, ep_id in enumerate(ids)]


def _series(*groups: Iterable[Row]) -> EpisodeIndex:
    rows = [row for group in groups for row in group]
    return series_index(
        {EpisodeKey(season, episode): ep_id for season, episode, ep_id, _ in rows},
        absolutes={ep_id: absolute for _, _, ep_id, absolute in rows if absolute is not None},
    )


def _ids(index: EpisodeIndex, season: int) -> tuple[int, ...]:
    """The ids of `season` in episode order."""

    return tuple(ep_id for key, ep_id in sorted(index.id_by_key.items()) if key.season == season)


def _window(al_id: int, ids: Iterable[int], count: int | None, series: str, *anilist: str) -> EntryWindow:
    return entry_window(al_id, ids, names=EntryNames(series, anilist), episode_count=count)


def _scope(
    index: EpisodeIndex, own: EntryWindow, *, alternate_titles: tuple[str, ...] = (), aired_numbering: bool = True
) -> SeedScope:
    """The grab scope of `own`'s entry over `index`."""

    return seed_scope(
        episode_index([index.by_id[ep_id] for ep_id in own.ordered_episode_ids]),
        index,
        al_id=own.al_id,
        names=own.names,
        episode_count=own.episode_count,
        alternate_titles=alternate_titles,
        aired_numbering=aired_numbering,
    )


def _lone(number: int, title: str, *numbers: int, matched: tuple[tuple[int, int], ...] = ()) -> ParsedFileInfo:
    """Sonarr's parse of a name carrying `number` (and `numbers`) alone: season 0, the absolutes, the lifted title."""

    return parsed_info(season=0, absolutes=(number, *numbers), matched=matched, series_title=title)


def _read(
    scope: SeedScope, windows: Sequence[EntryWindow], parsed: Mapping[str, ParsedFileInfo | None]
) -> dict[str, tuple[int, ...]]:
    """What the reader proposes for `parsed` under `windows`, the facts read under the own window as `place_release` does."""

    torrent = batch(parsed)
    own = next(window for window in windows if window.al_id == scope.al_id)
    facts = assign_episode_ids(torrent, TargetScope(own.ordered_episode_ids, scope.series, names=own.names)).facts
    return dict(NumberingReader.of(scope, windows).read(torrent, facts).ids)


# Stray Cats Overrun! (Sonarr 664): seven specials without absolutes, a season of 13. Tsundere's BD batch (AnimeBytes
# 124088) is listed under the season's entry alone; its names carry a bare number Sonarr matched to nothing.
_STRAY_CATS = _series(_run(0, range(32762, 32769)), _run(1, range(32769, 32782), absolute=1))
_STRAY_CATS_SEASON = _window(7590, _ids(_STRAY_CATS, 1), 13, "Stray Cats Overrun!", "Mayoi Neko Overrun!")
_STRAY_CATS_SPECIALS = _window(8599, _ids(_STRAY_CATS, 0)[:6], 6, "Stray Cats Overrun!", "Mayoi Neko Overrun! Specials")


def _stray_cat(number: int) -> str:
    return f"[Tsundere] Mayoi Neko Overrun! - {number:02d} [BDRip h264 1920x1080 FLAC][746CE979].mkv"


_STRAY_CATS_FILES = {_stray_cat(n): _lone(n, "Mayoi Neko Overrun!") for n in range(1, 13)}


class TestEveryNumberingAgrees:
    """A lone-number file is proposed where every numbering that fits the torrent reads it."""

    def test_a_season_window_counted_from_one_places_its_files(self) -> None:
        # The series row, the absolute, the window row and the window season all read 07 as S01E07.
        proposed = _read(_scope(_STRAY_CATS, _STRAY_CATS_SEASON), (_STRAY_CATS_SEASON,), _STRAY_CATS_FILES)

        assert proposed == {_stray_cat(n): (32768 + n,) for n in range(1, 13)}

    def test_a_window_of_specials_alone_counts_its_specials(self) -> None:
        # My Hero Academia (Sonarr 254): the Ikinokore OVA entry (AniList 122349) is specials 8 and 9, count 2.
        # HorribleSubs' single reads 01; the series row and the absolute read it as S01E01, outside the window, so
        # they're dropped, and the window row alone places it.
        series = _series(_run(0, range(9253, 9257), episode=7), _run(1, range(9271, 9274), absolute=1))
        window = _window(
            122349,
            (9254, 9255),
            2,
            "My Hero Academia",
            "My Hero Academia: Make It! Do-or-Die Survival Training",
            "Boku no Hero Academia: Ikinokore! Kesshi no Survival Kunren",
        )
        name = "[HorribleSubs] Boku no Hero Academia - Ikinokore! Kesshi no Survival Kunren - 01 [1080p].mkv"
        parsed = {name: _lone(1, "Boku no Hero Academia - Ikinokore! Kesshi no Survival Kunren")}

        proposed = _read(_scope(series, window, alternate_titles=("Boku no Hero Academia",)), (window,), parsed)

        assert proposed == {name: (9254,)}

    def test_a_window_row_counts_the_rows_the_window_holds(self) -> None:
        # Lupin III (Sonarr 437): the Part IV entry (AniList 20906) is 24 rows of season 4, which Sonarr numbers
        # 1-5, 7-22 and 24-26. scoot's 24 files count from one, so 06 is the window's sixth row, S04E07. Sonarr
        # matched them into season 1, whose rows carry the absolutes, so the series row and the absolute fall
        # outside the window and are dropped.
        season_four = [
            *_run(4, range(21900, 21905), absolute=229),
            *_run(4, range(21906, 21922), episode=7, absolute=235),
        ]
        season_four += _run(4, range(21923, 21926), episode=24, absolute=252)
        series = _series(_run(1, range(21672, 21696), absolute=1), season_four)
        window = _window(20906, _ids(series, 4), 24, "Lupin III", "Lupin the 3rd Part IV", "Lupin III: PART IV")
        names = {n: f"[scoot] Lupin the Third - {n:02d} [7AA4746D].mkv" for n in (1, 6, 24)}
        parsed = {names[n]: _lone(n, "Lupin the Third", matched=((1, n),)) for n in names}

        proposed = _read(_scope(series, window, alternate_titles=("Lupin the Third",)), (window,), parsed)

        assert proposed == {names[1]: (21900,), names[6]: (21906,), names[24]: (21925,)}

    def test_a_single_of_a_second_season_counted_from_one(self) -> None:
        # Sengoku Youko (Sonarr 1496): the second season's entry (AniList 176660) is 22 rows, absolutes 14-35.
        # Erai-raws' single 13 reads S02E13 under the window row and the window season; the absolute for 13 lies
        # in season 1, outside the window, so the series-wide numberings don't fit.
        series = _sengoku_youko()
        window = _sengoku_youko_window(series)
        name = "[Erai-raws] Sengoku Youko - Senma Konton-hen - 13 [1080p][Multiple Subtitle][8DA9E2B9].mkv"

        proposed = _read(_scope(series, window), (window,), {name: _lone(13, "Sengoku Youko - Senma Konton-hen")})

        assert proposed == {name: (51765,)}

    def test_two_windows_counted_straight_through_agree_with_the_series_row(self) -> None:
        # Godannar (Sonarr 529): two seasons of 13, listed under both entries. The files count 01-26, so each
        # window's own numberings count past it; the union row and the series row read 14 as S02E01.
        series = _series(_run(1, range(28079, 28092), absolute=1), _run(2, range(28092, 28105), absolute=14))
        first = _window(
            1103, _ids(series, 1), 13, "Godannar", "Marriage of God & Soul Godannar!!", "Shinkon Gattai Godannar"
        )
        second = _window(
            1104,
            _ids(series, 2),
            13,
            "Godannar",
            "Marriage of God & Soul Godannar!! (2004)",
            "Shinkon Gattai Godannar!! SECOND SEASON",
        )
        names = {n: f"Shinkon Gattai Godannar!! - {n:02d}.mkv" for n in range(1, 26)}
        names[26] = "Shinkon Gattai Godannar!! - 26 END.mkv"
        parsed: dict[str, ParsedFileInfo | None] = {names[n]: _lone(n, "Shinkon Gattai Godannar!!") for n in names}
        parsed["Shinkon Gattai Godannar!! - OP 1.mkv"] = parsed_info()

        proposed = _read(_scope(series, first), (first, second), parsed)

        assert proposed == {names[n]: (28078 + n,) for n in range(1, 27)}

    def test_a_number_past_the_window_is_read_by_the_series_row_and_the_absolute(self) -> None:
        # MF GHOST (Sonarr 1352): sam's BD batch of the second season counts 13-24, the series' numbers. Sonarr
        # matched 13 to S03E13 (another slice's). The 13th regular row carries absolute 13: S02E01.
        series = _series(
            _run(0, [49356]),
            _run(1, range(49357, 49369), absolute=1),
            _run(2, range(49369, 49381), absolute=13),
            _run(3, range(49381, 49394), absolute=25),
            _run(4, [49394], absolute=38),
        )
        window = _window(171642, _ids(series, 2), 12, "MF GHOST", "MF GHOST Season 2", "MF Ghost 2nd Season")
        names = {n: f"[sam] MF Ghost - {n} [BD 1080p FLAC] [68FDDB91].mkv" for n in range(13, 25)}
        parsed = {names[13]: _lone(13, "MF Ghost", matched=((3, 13),))}
        parsed.update({names[n]: _lone(n, "MF Ghost", matched=((2, n - 12),)) for n in range(14, 25)})

        proposed = _read(_scope(series, window), (window,), parsed)

        assert proposed == {names[n]: (49356 + n,) for n in range(13, 25)}

    def test_an_alternate_title_of_the_series_is_a_known_title(self) -> None:
        # Blue Exorcist (Sonarr 86): UltraRemux's Kyoto Saga names its files "Ao no Exorcist", Sonarr's alternate
        # title for the series, and counts the season from one.
        series = _series(_run(1, range(8805, 8830), absolute=1), _run(2, range(8830, 8842), absolute=26))
        window = _window(
            21861,
            _ids(series, 2),
            12,
            "Blue Exorcist",
            "Blue Exorcist: Kyoto Saga",
            "Ao no Exorcist: Kyoto Fujouou-hen",
        )
        name = "Ao no Exorcist - 01 [1080p - AVC - [PCM, ja][FLAC, en]].mkv"

        proposed = _read(
            _scope(series, window, alternate_titles=("Ao no Exorcist",)),
            (window,),
            {name: _lone(1, "Ao no Exorcist", matched=((1, 1),))},
        )

        assert proposed == {name: (8830,)}

    def test_a_two_number_name_is_read_onto_two_ids(self) -> None:
        # Gintama (Sonarr 39): CBT's DVD batch opens with 001-002 in one file, then 003 on.
        series, window, parsed = _gintama()

        proposed = _read(_scope(series, window), (window,), parsed)

        assert proposed[_gintama_name("001-002")] == (4604, 4605)
        assert proposed[_gintama_name("201")] == (4804,)

    def test_each_title_of_a_two_title_torrent_is_read_under_the_window_it_names(self) -> None:
        # New Game! (Sonarr 207): ToshY's batch (nyaa 1705729) holds New Game! 01-12, New Game!! 01-12 and an
        # OVA, listed under both seasons' entries. Every number repeats, so the files are read by title: the
        # trailing marks tell the two titles and the two windows apart. The OVA's title names no window.
        series = _series(
            _run(0, [13960], absolute=13),
            _run(1, range(13961, 13973), absolute=1),
            _run(2, range(13973, 13985), absolute=14),
        )
        first = _window(21455, _ids(series, 1), 12, "New Game!", "NEW GAME!")
        second = _window(98292, _ids(series, 2), 12, "New Game!", "NEW GAME!!")
        parsed = {f"[ToshY] New Game! - {n:02d}.mkv": _lone(n, "New Game!", matched=((1, n),)) for n in range(1, 13)}
        parsed.update(
            {f"[ToshY] New Game!! - {n:02d}.mkv": _lone(n, "New Game!!", matched=((2, n),)) for n in range(1, 13)}
        )
        parsed["[ToshY] New Game! OVA - 01.mkv"] = _lone(1, "New Game! OVA")
        alternates = ("New Game!!", "New Game! S2", "New Game!! S2", "New Game!")

        proposed = _read(_scope(series, first, alternate_titles=alternates), (first, second), parsed)

        assert proposed == {
            **{f"[ToshY] New Game! - {n:02d}.mkv": (13960 + n,) for n in range(1, 13)},
            **{f"[ToshY] New Game!! - {n:02d}.mkv": (13972 + n,) for n in range(1, 13)},
        }


class TestTheGatesRefuse:
    """Each gate leaves the file to its verdict, and refuses nothing a passing file would have."""

    def test_two_numberings_that_disagree_refuse_the_file(self) -> None:
        # Sengoku Youko's 14 and 22: the absolute for 14 is S02E01, inside the window, so the series row reads 14
        # as S02E01 while the window row reads it as S02E14.
        series = _sengoku_youko()
        window = _sengoku_youko_window(series)
        parsed = {
            f"[Erai-raws] Sengoku Youko - Senma Konton-hen - {n} [1080p][Multiple Subtitle][35EB3627].mkv": _lone(
                n, "Sengoku Youko - Senma Konton-hen"
            )
            for n in (14, 22)
        }

        assert _read(_scope(series, window), (window,), parsed) == {}

    def test_a_window_off_anilists_count_refuses_the_torrent(self) -> None:
        # Panty & Stocking (Sonarr 74): AniList counts the season as 13 double-episodes, Sonarr as 26 rows.
        # Underwater's 01v2 reads S01E01 every way, and is still refused: the window isn't the entry AniList counts.
        series = _series(
            _run(0, [8353]), _run(1, range(8354, 8380), absolute=1), _run(2, range(8380, 8410), absolute=27)
        )
        window = _window(
            8795, _ids(series, 1), 13, "Panty & Stocking with Garterbelt", "Panty & Stocking with Garterbelt"
        )
        name = "[Underwater] Panty and Stocking with Garterbelt 01v2 - Excretion Without Honor and Humanity - Death Race 2010 (BD 720p) [32949214].mkv"
        parsed = {name: _lone(1, "Panty and Stocking with Garterbelt", matched=((1, 1),))}

        assert _read(_scope(series, window), (window,), parsed) == {}

    def test_a_repeated_number_under_one_title_refuses_the_files_carrying_it(self) -> None:
        # K Project (Sonarr 101): Beatrice-kBaraka's Return of Kings carries 13 twice (a Director's Cut and a v2),
        # and a CM extra without a number. The two 13s are refused; 01-12 place.
        series = _series(
            _run(0, [9645], absolute=14),
            _run(1, range(9656, 9669), absolute=1),
            _run(2, range(9669, 9682), absolute=15),
        )
        window = _window(20913, _ids(series, 2), 13, "K Project", "K: RETURN OF KINGS")
        names = {
            n: f"[Beatrice-kBaraka] K Return of Kings {n:02d} [BDRip 1920x1080 x264 FLAC] [D1B8CFC5].mkv"
            for n in range(1, 13)
        }
        cut = "[Beatrice-kBaraka] K Return of Kings 13 Director's Cut [BDRip 1920x1080 x264 FLAC] [4E3BA8EB].mkv"
        v2 = "[Beatrice-kBaraka] K Return of Kings 13v2 [BDRip 1920x1080 x264 FLAC] [93026D00].mkv"
        parsed: dict[str, ParsedFileInfo | None] = {
            names[n]: _lone(n, "K Return of Kings", matched=((2, n),)) for n in range(1, 13)
        }
        parsed[cut] = _lone(13, "K Return of Kings", matched=((2, 13),))
        parsed[v2] = _lone(13, "K Return of Kings", matched=((2, 13),))
        parsed["[Beatrice-Raws] K Return of Kings (CM) [BDRip 1920x1080 x264 FLAC].mkv"] = parsed_info()

        proposed = _read(_scope(series, window, alternate_titles=("K - Return of Kings",)), (window,), parsed)

        assert proposed == {names[n]: (9668 + n,) for n in range(1, 13)}

    def test_a_two_number_names_numbers_count_toward_repeats(self) -> None:
        # Constructed: Gintama's batch with a 002 single added beside the 001-002 file. 2 repeats, and both files
        # are refused.
        series, window, parsed = _gintama()
        parsed[_gintama_name("002")] = _lone(2, "Gintama", matched=((1, 2),))

        proposed = _read(_scope(series, window), (window,), parsed)

        assert _gintama_name("001-002") not in proposed
        assert _gintama_name("002") not in proposed
        assert proposed[_gintama_name("003")] == (4606,)

    def test_a_row_numbered_otherwise_than_its_position_refuses_the_series_row(self) -> None:
        # Wandering Son (Sonarr 700): the two specials carry absolutes 11 and 12, so S01E11 carries 13. aro's BD
        # batch counts 01-12 and holds 10 twice (a 10+11 cut and a 10). The series row and the absolute fit the
        # torrent (every absolute lies in the two windows). 01-09 place; the 10s are a repeat; 11 reads to a row
        # carrying 13; 12 has no twelfth regular row.
        series, windows, parsed = _wandering_son()

        proposed = _read(_scope(series, windows[0]), windows, parsed)

        assert proposed == {_hourou_musuko(f"{n:02d}"): (33585 + n,) for n in range(1, 10)}

    def test_without_the_repeat_the_tenth_file_lands_on_the_tenth_row_which_is_wrong(self) -> None:
        # Constructed: the same batch without its 10+11 cut. The series row and the absolute agree on S01E10 for
        # 10, and by content the file is S01E11 (the cut is S01E10 and the special). Nothing guards this shape,
        # which no real torrent has: this pins the known gap so a change to it shows.
        series, windows, parsed = _wandering_son()
        del parsed[_hourou_musuko("10+11")]

        proposed = _read(_scope(series, windows[0]), windows, parsed)

        assert proposed[_hourou_musuko("10")] == (33595,)

    def test_a_shuffled_season_refuses_the_files_whose_rows_carry_another_absolute(self) -> None:
        # Cowboy Bebop (Sonarr 15): TVDB's absolutes follow the Japanese broadcast order, so S01E01 carries 2 and
        # S01E13 carries 1; 19-26 carry their own numbers. The window isn't monotone, so the window numberings
        # are dropped; the series row refuses 01 and reads 19 as S01E19, where the absolute agrees.
        absolutes = (2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 1, 4, 5, 6, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26)
        rows = [(1, n, 2733 + n, absolute) for n, absolute in enumerate(absolutes, 1)]
        series = _series(_run(0, range(2730, 2734)), rows)
        window = _window(1, _ids(series, 1), 26, "Cowboy Bebop", "Cowboy Bebop")
        parsed = {
            "[JySzE] Cowboy Bebop - 01 [v3].mkv": _lone(1, "Cowboy Bebop", matched=((1, 13),)),
            "[JySzE] Cowboy Bebop - 19 [v3].mkv": _lone(19, "Cowboy Bebop", matched=((1, 19),)),
        }

        proposed = _read(_scope(series, window), (window,), parsed)

        assert proposed == {"[JySzE] Cowboy Bebop - 19 [v3].mkv": (2752,)}

    @pytest.mark.parametrize(
        "title",
        [
            pytest.param("ReZero kara Hajimeru Isekai Seikatsu - Shin Henshuu-ban", id="a recut's title the run lacks"),
            pytest.param("", id="no title lifted (constructed)"),
        ],
    )
    def test_a_lifted_title_the_run_does_not_know_refuses_the_file(self, title: str) -> None:
        # Re:ZERO (Sonarr 197): OZR's Shin Henshuu-ban recut counts from one under the first season's entry. Every
        # numbering agrees on S01E01, but the lifted title is neither the series', an alternate's, nor AniList's.
        series = _series(_run(1, range(13590, 13615), absolute=1))
        window = _window(
            21355,
            _ids(series, 1),
            25,
            "Re: ZERO, Starting Life in Another World",
            "Re:ZERO -Starting Life in Another World-",
            "Re:Zero kara Hajimeru Isekai Seikatsu",
        )
        name = "[OZR] ReZero kara Hajimeru Isekai Seikatsu - Shin Henshuu-ban 01 [BDRip 1920x1080 HEVC FLAC].mkv"
        scope = _scope(series, window, alternate_titles=("ReZero kara Hajimeru Isekai Seikatsu",))

        assert _read(scope, (window,), {name: _lone(1, title)}) == {}

    def test_two_files_proposed_one_id_are_both_dropped(self) -> None:
        # Constructed: two titles of one torrent read under two windows that share their first regular season's
        # numbers.
        series = _series(_run(1, range(601, 604), absolute=1), _run(2, range(701, 704), absolute=4))
        first = _window(1, _ids(series, 1), 3, "Show", "Show!")
        second = _window(2, _ids(series, 1)[:1], 1, "Show", "Show!!")
        parsed = {"show! - 01.mkv": _lone(1, "Show!"), "show!! - 01.mkv": _lone(1, "Show!!")}

        assert _read(_scope(series, first), (first, second), parsed) == {}


class TestTheReaderStandsDown:
    """When the reader proposes nothing at all, and which names it never reads."""

    def test_an_unknown_parse_stands_the_reader_down(self) -> None:
        # Tsundere's nyaa batch (1050453) carries six SP files too; one not yet parsed holds the url, and the
        # reader proposes nothing until it is.
        parsed: dict[str, ParsedFileInfo | None] = {**_STRAY_CATS_FILES}
        parsed["[Tsundere] Mayoi Neko Overrun! - SP1 [BDRip h264 1920x1080 FLAC][B66672E2].mkv"] = None
        windows = (_STRAY_CATS_SEASON, _STRAY_CATS_SPECIALS)

        assert _read(_scope(_STRAY_CATS, _STRAY_CATS_SEASON), windows, parsed) == {}

    def test_off_the_aired_numbering_the_reader_proposes_nothing(self) -> None:
        # Constructed: Stray Cats' real batch under a numbering its series doesn't have (Sonarr numbers it as aired).
        scope = _scope(_STRAY_CATS, _STRAY_CATS_SEASON, aired_numbering=False)

        assert _read(scope, (_STRAY_CATS_SEASON,), _STRAY_CATS_FILES) == {}

    def test_a_window_on_another_series_is_left_out(self) -> None:
        # Constructed: a claim's window on Sonarr series 8 lists ids the grabbing series lacks and a count no window
        # of it matches. Left out, it gates nothing.
        elsewhere = entry_window(9, (81, 82, 83), series_id=8, episode_count=1)
        windows = (_STRAY_CATS_SEASON, elsewhere)

        proposed = _read(_scope(_STRAY_CATS, _STRAY_CATS_SEASON), windows, _STRAY_CATS_FILES)

        assert proposed == {_stray_cat(n): (32768 + n,) for n in range(1, 13)}

    def test_an_extra_and_a_title_counted_name_are_not_in_the_pool(self) -> None:
        # Constructed: an NCOP Sonarr read a number off, and a name whose number is its own title's (no title
        # confirms "03 Days of Summer" and the series has no absolute 3 among the window's rows), beside the
        # batch's 01. Neither repeats 01's number nor is proposed; 01 places.
        series = _series(_run(1, range(601, 604), absolute=1))
        window = _window(1, _ids(series, 1), 3, "Show")
        parsed = {
            "show - 01 [grp].mkv": _lone(1, "Show"),
            "[grp] show - NCOP1 [bd].mkv": _lone(1, "Show"),
            "show - 03 Days of Summer [grp].mkv": _lone(3, "Show", matched=((1, 3),)),
        }

        proposed = _read(_scope(series, window), (window,), parsed)

        assert proposed == {"show - 01 [grp].mkv": (601,)}


class TestTitleKeys:
    """The three title keys, on the titles they were measured against."""

    @pytest.mark.parametrize(
        ("title", "key"),
        [
            ("K-ON!", "k on"),
            ("K-ON!!", "k on"),
            ("Panty & Stocking with Garterbelt", "panty and stocking with garterbelt"),
            ("Re:ZERO -Starting Life in Another World-", "re zero starting life in another world"),
            ("", ""),
        ],
    )
    def test_norm_title_folds_case_punctuation_and_the_ampersand(self, title: str, key: str) -> None:
        assert _norm_title(title) == key

    @pytest.mark.parametrize(
        ("title", "key"),
        [
            ("Sengoku Youko - Senma Konton-hen", "sengoku youko senma konton-hen"),
            ("Sengoku Youko: Senma Konton-hen", "sengoku youko senma konton-hen"),
            ("K-ON!!", "k-on!!"),
        ],
    )
    def test_raw_title_folds_the_two_separators_and_keeps_the_marks(self, title: str, key: str) -> None:
        assert _raw_title(title) == key

    @pytest.mark.parametrize(
        ("title", "key"),
        [
            ("K-ON!", "k on|!"),
            ("K-ON!!", "k on|!!"),
            ("New Game!", "new game|!"),
            ("New Game!!", "new game|!!"),
            ("Needless+", "needless|+"),
            ("Needless", "needless|"),
            ("Dog Days'", "dog days|'"),
            ("Dog Days’", "dog days|'"),
            ("Kaguya-sama? - ", "kaguya sama|?"),
        ],
    )
    def test_marked_title_keeps_the_trailing_marks(self, title: str, key: str) -> None:
        assert _marked_title(title) == key


def _sengoku_youko() -> EpisodeIndex:
    """Sengoku Youko (Sonarr 1496): two specials, 13 then 22 regular rows, absolutes 1-35."""

    return _series(
        _run(0, [51738, 51739]), _run(1, range(51740, 51753), absolute=1), _run(2, range(51753, 51775), absolute=14)
    )


def _sengoku_youko_window(series: EpisodeIndex) -> EntryWindow:
    return _window(
        176660,
        _ids(series, 2),
        22,
        "Sengoku Youko",
        "Sengoku Youko: The Thousandfold Chaos Arc",
        "Sengoku Youko: Senma Konton-hen",
    )


def _gintama_name(number: str) -> str:
    return f"[CBT] Gintama {number} [DVDrip 760x576 x264 FLAC].mkv"


def _gintama() -> tuple[EpisodeIndex, EntryWindow, dict[str, ParsedFileInfo | None]]:
    """Gintama (Sonarr 39): the 201 rows AniList 918 covers, and CBT's 200 files, 001-002 in one."""

    series = _series(
        _run(1, range(4604, 4653), absolute=1),
        _run(2, range(4653, 4703), absolute=50),
        _run(3, range(4703, 4754), absolute=100),
        _run(4, range(4754, 4805), absolute=151),
    )
    window = _window(
        918, tuple(ep_id for season in (1, 2, 3, 4) for ep_id in _ids(series, season)), 201, "Gintama", "Gintama"
    )
    parsed: dict[str, ParsedFileInfo | None] = {
        _gintama_name("001-002"): _lone(1, "Gintama", 2, matched=((1, 1), (1, 2)))
    }
    for n in range(3, 202):
        key = series.key_by_id[window.ordered_episode_ids[n - 1]]
        parsed[_gintama_name(f"{n:03d}")] = _lone(n, "Gintama", matched=((key.season, key.episode),))
    return series, window, parsed


def _hourou_musuko(number: str) -> str:
    return f"[aro] Hourou Musuko - {number} [BD 1080p HEVC FLAC].mkv"


def _wandering_son() -> tuple[EpisodeIndex, tuple[EntryWindow, EntryWindow], dict[str, ParsedFileInfo | None]]:
    """Wandering Son (Sonarr 700) and aro's BD batch, listed under the season's entry and the specials'."""

    series = _series(
        _run(0, [33584, 33585], absolute=11), _run(1, range(33586, 33596), absolute=1), [(1, 11, 33596, 13)]
    )
    season = _window(8426, _ids(series, 1), 11, "Wandering Son", "Wandering Son", "Hourou Musuko")
    specials = _window(10935, _ids(series, 0), 2, "Wandering Son", "Wandering Son Specials", "Hourou Musuko Specials")
    numbers = [f"{n:02d}" for n in range(1, 13)]
    numbers.insert(10, "10+11")
    parsed: dict[str, ParsedFileInfo | None] = {
        _hourou_musuko(number): _lone(int(number[:2]), "Hourou Musuko") for number in numbers
    }
    return series, (season, specials), parsed
