# pyright: strict
"""Reading file names without Sonarr: the episode range a name spells, and which dotted number is its count."""

from typing import ClassVar

import pytest

from pearlarr.manual_import import EntryNames
from pearlarr.release_names import RangeKey, RunMember, TitleGround, range_key, read_name, read_names

_NO_TITLES = TitleGround.of(EntryNames())
# "100" in fullwidth digits.
_FULLWIDTH_100 = "\uff11\uff10\uff10"


def _read(name: str, series: str = "", anilist: tuple[str, ...] = ()) -> tuple[int | None, tuple[int, ...]]:
    """The member number (None when unnumbered) and the bare numbers of one name read under the entry's titles."""

    read = read_names([name], TitleGround.of(EntryNames(series, anilist)))[name]
    return read.member.number if read.member is not None else None, read.bare_numbers


class TestDottedNumber:
    """The scene-dotted count: the number before the quality words, never one in the title or a codec."""

    @pytest.mark.parametrize(
        "name",
        [
            "Zap.Pilot.100.01.1080p.BluRay.x264-grp.mkv",
            "Wisp.100.Lantern.Keepers.of.the.Moor.01.1080p.WEB.H264-grp.mkv",
            "Steel.Armor.Titan.0083.Comet.Archive.01.1080p.BluRay.x264-grp.mkv",
            "Steel.Armor.Titan.0080.Duel.in.the.Valley.E01.1080p.BluRay.mkv",
            "Tinman.009.01.DVDRip.x264-grp.mkv",
            "Show.2024.BluRay.1080p.FLAC5.1.H.264.REMUX-grp.mkv",
            "Zap.Pilot.100.S02.Ep01.1080p.WEB.H.264-grp.mkv",
        ],
    )
    def test_a_title_or_codec_number_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        "name",
        [
            "Quillfox.101.v4.480p.DVD.FLAC2.0.x264-grp.mkv",
            "Quillfox.101.v4.1080p.BluRay.10-bit.x265-grp.mkv",
        ],
    )
    def test_the_count_before_the_quality_words_is_read(self, name: str) -> None:
        member = read_name(name).member

        assert member is not None
        assert (member.number, member.version) == (101, 4)

    @pytest.mark.parametrize(
        "name",
        [
            "Quillfox.101.v4.480p.DVD.Vol.26.Dual-Audio.FLAC2.0.Hi10P.x264-grp.mkv",
            "Quillfox.101.v4.480p.DVD.Remux.10.bit.FLAC2.0.x264-grp.mkv",
            "Quillfox.101.v4.480p.DVD.Dual-Audio.FLAC.2.0.10.bit.x264-grp.mkv",
            "Quillfox.101.v4.480p.DVD.23.976fps.Dual-Audio.FLAC2.0.Hi10P.x264-grp.mkv",
        ],
    )
    def test_a_number_in_the_quality_tail_is_no_later_count(self, name: str) -> None:
        member = read_name(name).member

        assert member is not None
        assert (member.number, member.version) == (101, 4)

    @pytest.mark.parametrize(
        "name",
        [
            "Quillfox.101.The.100.Year.Plan.1080p.WEB-grp.mkv",
            "Quillfox.101.Disc.02.480p.DVD.x264-grp.mkv",
            "Quillfox.101.S03.480p.DVD.x264-grp.mkv",
            "Quillfox.101.480p.DVD.x264-grp.S01.mkv",
        ],
    )
    def test_a_count_with_another_count_or_a_key_behind_it_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    def test_a_number_glued_to_a_hyphen_is_no_trailing_count(self) -> None:
        member = read_name("Quillfox.101.480p.DVD.x264-grp-25.mkv").member

        assert member is not None
        assert member.number == 101

    @pytest.mark.parametrize(
        "name",
        [
            "[grp] Star Drums! - 12 & 13 [1080p].mkv",
            "[grp] Star Drums! - 12+13 [1080p].mkv",
            "[grp] Star Drums! - 12, 13 [1080p].mkv",
            "[grp] Star Drums! - 12 and 13 [1080p].mkv",
            "[grp] Star Drums! - 12 to 13 [1080p].mkv",
            "[grp] Star Drums! - 12~13 [1080p].mkv",
            "Star.Drums.12&13.1080p.WEB-grp.mkv",
        ],
    )
    def test_a_number_a_joiner_ties_to_the_one_before_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None
        assert _read(name, "Star Drums!")[0] is None

    def test_a_lone_season_token_is_no_trailing_count(self) -> None:
        assert read_name("[grp] Show S01 (BD 1080p x265 FLAC).mkv").member is None

    @pytest.mark.parametrize(
        ("name", "prefix"),
        [
            ("Cinder.001.1080p.BluRay.x264-grp.mkv", "Cinder"),
            ("Comet.X.001.1080p.WEB.H.264-grp.mkv", "Comet.X"),
            ("Ba-Ko-Ru.GX.001.720p.DVD.x264-grp.mkv", "Ba-Ko-Ru.GX"),
        ],
    )
    def test_a_title_ending_in_a_codec_letter_still_counts(self, name: str, prefix: str) -> None:
        member = read_name(name).member

        assert member is not None
        assert (member.prefix, member.number) == (prefix, 1)

    @pytest.mark.parametrize(
        "name",
        [
            "Show.H.264.1080p.WEB-grp.mkv",
            "Show.x.265.WEB-DL.1080p-grp.mkv",
            "Comet.X.264.1080p.WEB.H.264-grp.mkv",
        ],
    )
    def test_a_codec_number_behind_a_lone_letter_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Quillfox.101.x.264.480p.DVD-grp.mkv", 101),
            ("Show.018.H.264.1080p.WEB-grp.mkv", 18),
        ],
    )
    def test_a_codec_number_behind_the_count_is_no_later_count(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Show.H.264.101.1080p.WEB-grp.mkv", 101),
            ("Show.x.265.101.The.Title.1080p.WEB-grp.mkv", 101),
            ("Comet.X.264.001.1080p.WEB-grp.mkv", 1),
        ],
    )
    def test_a_codec_number_before_the_count_is_skipped(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Friend.Match.007.I.ll.Take.20.Million.Coins.Worth.of.Your.Life.1080p.WEB.H.264-grp.mkv", 7),
            ("Quillfox.001.The.100.Year.Plan.1080p.WEB-grp.mkv", 1),
            ("Lazy.Lazy.Meadow.033.401.Unauthorized.1080p.WEB-grp.mkv", 33),
            ("Pre.ZERO.050.Rilla.X.1000.X.Finders.Bar.1080p.WEB-grp.mkv", 50),
            ("Dark.Thistle.023.Vol.26.1080p.WEB.H.264-grp.mkv", 23),
        ],
    )
    def test_a_padded_count_stands_before_an_unpadded_title_number(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number

    @pytest.mark.parametrize(
        "name",
        [
            "Solid.Steel.Alarm.036.Episode.000.1080p.WEB-grp.mkv",
            "Rubi.ga.Hunt.025.RubiHunt.Theater.01.Taro.and.Hana.1080p.WEB-grp.mkv",
            "Wisp.100.Lantern.Keepers.of.the.Moor.05.1080p.WEB-grp.mkv",
            "Dark.Thistle.123.Vol.26.1080p.WEB.H.264-grp.mkv",
        ],
    )
    def test_a_padded_number_behind_the_count_or_an_unpadded_count_is_refused(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Pocketbeast.1069.1080p.WEB.H.264-grp.mkv", 1069),
            ("Show.1999.001.1080p.WEB-grp.mkv", 1),
            ("Seeker.x.Seeker.2011.067.1080p.WEB-grp.mkv", 67),
            ("1000.mile.Road.001.1080p.WEB-grp.mkv", 1),
        ],
    )
    def test_a_count_past_the_thousandth_episode_is_read_but_a_year_is_not(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number


class TestSeasonTail:
    """A name ending in "Season N": a season token when plain, the count behind an ordinal or a hyphen."""

    @pytest.mark.parametrize(
        "name",
        [
            "[BD Menu 01] Silver Harbor Lights - Season 01 [2021].mkv",
            "[Ending 01] Blueleaf - Season 01 [2022].mkv",
            "Quillfox.101.480p.DVD.x264-QvXbR.Season.3.mkv",
            # an ordinal season's lone digit numbers the season's part, not an episode
            "[grp] Tetsu no Kyodai - The Final Season 2 (1080p).mkv",
            "[grp] Spikeball!! 2nd Season 2.mkv",
        ],
    )
    def test_a_plain_season_number_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("[Tanuma] Mikan Shitara Pudding Datta Ken 2nd Season 25 [BDRip 1920x1080 x265 10bit FLAC].mkv", 25),
            ("[grp] Kataribanashi Series Second Season - 01 [1080p].mkv", 1),
            ("[grp] Show Season 2 - 05 [1080p].mkv", 5),
            ("[grp] Show Second Season 25 [1080p].mkv", 25),
            ("[grp] Storm over Hills The Final Season 25 [1080p].mkv", 25),
            ("[grp] Show Last Season 07.mkv", 7),
            ("[grp] Tetsu no Kyodai The Final Season 02 (1080p).mkv", 2),
            ("[grp] Tetsu no Kyodai The Final Season - 2 (1080p).mkv", 2),
        ],
    )
    def test_the_number_behind_an_ordinal_or_a_hyphen_is_the_count(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number


class TestReadNames:
    """A pack's lone-S set: "S1", "S2", "S3" behind one prefix count its specials; anything less says a season."""

    _EPISODES: ClassVar[list[str]] = [f"[grp] Show - {n:02d} (BD 1080p).mkv" for n in (1, 2, 3, 4)]

    @staticmethod
    def _numbers(names: list[str]) -> dict[str, int | None]:
        return {
            name: read.member.number if read.member else None for name, read in read_names(names, _NO_TITLES).items()
        }

    def test_a_set_counting_from_one_numbers_each_special_under_the_s_prefix(self) -> None:
        specials = [f"[grp] Show S{n} (BD 1080p).mkv" for n in (1, 2, 3)]
        reads = read_names(self._EPISODES + specials, _NO_TITLES)

        assert [reads[name].member for name in specials] == [RunMember("[grp] Show S", n, (), 1) for n in (1, 2, 3)]
        assert [reads[name].member for name in self._EPISODES] == [read_name(name).member for name in self._EPISODES]

    def test_a_higher_version_in_the_set_keeps_its_version(self) -> None:
        reads = read_names(["[grp] Show S1.mkv", "[grp] Show S1v2.mkv", "[grp] Show S2.mkv"], _NO_TITLES)

        assert [(read.member.number, read.member.version) for read in reads.values() if read.member] == [
            (1, 1),
            (1, 2),
            (2, 1),
        ]

    @pytest.mark.parametrize(
        "names",
        [
            ["[grp] Show S01 (BD 1080p).mkv"],
            ["[grp] Show S2.mkv", "[grp] Show S3.mkv"],
            ["[grp] Show S2 (Creditless OP).mkv", "[grp] Show S2 (Creditless ED).mkv"],
            ["Kidzone USA OP04 - Tom S3 [Bryn].mkv", "Kidzone USA OP06 - Tom S4 [Bryn].mkv"],
        ],
    )
    def test_a_lone_s_alone_repeated_or_not_from_one_stays_unnumbered(self, names: list[str]) -> None:
        assert self._numbers(self._EPISODES + names) == {**self._numbers(self._EPISODES), **dict.fromkeys(names)}


class TestMiddleNumber:
    """The " - NN - " count between a title and an episode title."""

    def test_a_second_middle_segment_is_a_title_number_and_no_count(self) -> None:
        assert read_name("COG x CLAN - 277 - 10 - 1 = [WEBDL-1080p].mkv").member is None


class TestPaddedCounts:
    """A count of four digits: padded behind a zero, or past the thousandth episode."""

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Sea Atlas - 0802 - An Angry Rider The Secret of Garmo 66 [WEBDL-1080p].mkv", 802),
            ("Sea Atlas - 1001 - The Title [WEBDL-1080p].mkv", 1001),
            ("[grp] Pocketbeast - 1069 [1080p].mkv", 1069),
        ],
    )
    def test_a_padded_middle_or_trailing_count_is_read(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number


class TestBareNumbers:
    """Every count standing alone in a name, less a codec's number in the quality tail."""

    @pytest.mark.parametrize(
        ("name", "numbers"),
        [
            ("Zappu.Pairotto.Hyaku.S02.Ep05.1080p.WEB.H.264-grp.mkv", ()),
            ("Comet.X.264.1080p.WEB.H.264-grp.mkv", (264,)),
            ("Quillfox.101.The.100.Year.Plan.1080p.WEB-grp.mkv", (101, 100)),
            ("Zap Pilot 100 Season 2 Episode 05 [1080p].mkv", (100,)),
            ("Show (2019) Season 2 Episode 05 [1080p].mkv", (2019,)),
            ("Quillfox.101.v4.480p.DVD.FLAC2.0.x264-grp.mkv", (101,)),
            ("[grp] Show - 046 - The 17 Years Old Truth [1080p].mkv", (46,)),
        ],
    )
    def test_every_standalone_count_is_bare_but_a_tail_codecs(self, name: str, numbers: tuple[int, ...]) -> None:
        assert read_name(name).bare_numbers == numbers


class TestRangeKey:
    """`range_key` reads a tight `SxxEyy-zz` range in any letter case, whether or not it counts up."""

    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("show s01e05-07 [1080p].mkv", RangeKey(1, 5, 7)),
            ("show s01e05-e07.mkv", RangeKey(1, 5, 7)),
            ("Show.S01E05-06v2.mkv", RangeKey(1, 5, 6)),
            ("Show.S01E05-06.1080p.mkv", RangeKey(1, 5, 6)),
            ("show s01e07-05.mkv", RangeKey(1, 7, 5)),
        ],
    )
    def test_a_tight_range_is_read(self, name: str, key: RangeKey) -> None:
        assert range_key(name) == key

    @pytest.mark.parametrize(
        "name",
        [
            "show s01e05 - 06 - title.mkv",
            "show s01e05-720p.mkv",
            "show s01e05-10bit.mkv",
            "show s01e05-576i.mkv",
            "show s01e05-06v2a.mkv",
            "show s01e05-6.mkv",
            "show s01e01-2.0.mkv",
            "shows01e05-06.mkv",
            "show s01e05.mkv",
        ],
    )
    def test_a_name_that_spells_no_range_reads_none(self, name: str) -> None:
        assert range_key(name) is None


class TestDecimalTail:
    """A trailing number behind a digit and a dot is a decimal's fraction, not a count; a "v2.1" is version 2."""

    @pytest.mark.parametrize(
        "name",
        [
            "[Vorlin] Shiku - 21.5.mkv",
            "Hollow Rebirth - 7.5 [1080p].mkv",
            "[XQZB] Kasumi - 11.5 (OVA) (BD 1080p) [4D2A81C0].mkv",
            "Star.Cousins.S01EP11.Sealed.Allies.1080p.Remux.AVC.FLAC2.0.mkv",
            "[Cordelia-Raws] Meet A Star (Director's Cut Edition) PV ver.1.1 [BDRip 1920x1080 x264 FLAC].mkv",
        ],
    )
    def test_a_fraction_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("[KyC] Tsukumo 12v2.1 (BD 720p Hi10 x264 AC3+FLAC) [Dual-Audio] [1B94F2D7].mkv", 12),
            ("[bro] Hagaki - 06 v2.1 [BD 1080p x264 FLAC].mkv", 6),
        ],
    )
    def test_a_minor_version_counts_as_its_major(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert (member.number, member.version) == (number, 2)


class TestTitledRead:
    """The entry's titles tell their own numbers apart from the count: a name opening with one counts past it."""

    @pytest.mark.parametrize(
        ("name", "series", "anilist", "number", "bare"),
        [
            (
                "Wisp.100.Lantern.Keepers.of.the.Moor.12.1080p.WEB.H264-grp.mkv",
                "Wisp 100: Lantern Keepers of the Moor",
                (),
                12,
                (),
            ),
            ("Zap.Pilot.100.001.1080p.WEB.H.264-grp.mkv", "Zap Pilot 100", (), 1, (1,)),
            ("Zap.Pilot.100.01.1080p.BluRay.x264-grp.mkv", "Zap Pilot 100", (), 1, ()),
            ("Tinman.009.001.1080p.mkv", "Tinman 009", (), 1, (1,)),
            ("Zap Pilot 100 II - 05 (OVA) - Title.mkv", "Zap Pilot 100", ("Zap Pilot 100 II",), 5, ()),
            ("Zap Pilot 100 2nd Season - 05 (OVA) - Title.mkv", "Zap Pilot 100", ("Zap Pilot 100 II",), 5, ()),
            ("[grp] Fisty - 05 (OVA) - Title.mkv", "Fisty (2018)", (), 5, ()),
            ("Seeker x Seeker (2011) - 05 (OVA) - Title.mkv", "Seeker x Seeker (2011)", (), 5, (2011,)),
            ("[grp] Brass;Door - 25 (OVA) - Hollow Wanderlust [ABCD1234].mkv", "Brass;Door", (), 25, ()),
            ("Show - Episode 12 (OVA) - Title.mkv", "Show", (), 12, ()),
            ("Show - 001.10.bit.mkv", "Show", (), 1, (1,)),
            ("Show - 01.10.bit.mkv", "Show", (), 1, ()),
            ("[grp] Chef Prince! - 14 END [1080p].mkv", "Chef Prince!", (), 14, ()),
            ("[grp] Was That a Penguin? - 13 END [1080p].mkv", "Was That a Penguin?", (), 13, ()),
            # a bracket behind the count, nested or not, a source word or a quality word there end it like the stem
            ("Drawing Sensei - 01 [1920x1080 - AVC - [PCM, ja]].mkv", "Drawing Sensei", (), 1, ()),
            ("[grp] Osaka Spirit - 50 (Osaka Spirit: Jill (OVA)) [1080p].mkv", "Osaka Spirit", (), 50, ()),
            ("Tinman.009.01.DVDRip.x264-grp.mkv", "Tinman 009", (), 1, ()),
            ("Zap.Pilot.100.018.PROPER.1080p.WEB-grp.mkv", "Zap Pilot 100", (), 18, (18,)),
            ("Zap.Pilot.100.018.Hi10P.1080p.WEB-grp.mkv", "Zap Pilot 100", (), 18, (18,)),
            # a form counted by the title's own digits: nothing was counted, the count behind the title is
            ("Tinman.009.10.1080p.WEB.H.264-grp.mkv", "Tinman 009", (), 10, ()),
            (
                "Steel.Armor.Titan.0083.Comet.Archive.10.1080p.mkv",
                "Steel Armor Titan 0083: Comet Archive",
                (),
                10,
                (),
            ),
            (
                "[grp] Sanjouhan Meiwa Taisei - 11 END (BD 1080p).mkv",
                "The Futon Galaxy",
                ("Sanjouhan Meiwa Taisei",),
                11,
                (),
            ),
        ],
    )
    def test_the_first_count_behind_the_title_is_the_count(
        self, name: str, series: str, anilist: tuple[str, ...], number: int, bare: tuple[int, ...]
    ) -> None:
        assert _read(name, series, anilist) == (number, bare)

    @pytest.mark.parametrize(
        ("name", "series", "anilist", "number"),
        [
            # the label hangs off the title's ordinal: the count follows it
            (
                "[grp] Kimi no Chef Academy 4th Season - 25 END [1080p].mkv",
                "My Chef Academy",
                ("Kimi no Chef Academy 4th Season",),
                25,
            ),
            # a number right behind the label is the label's: a season, no count
            ("[BD Menu 01] Silver Harbor Lights - Season 01 [2021].mkv", "Silver Harbor Lights", (), None),
            ("[PV 03] Kasane - The Hidden Parts - Season 01 [2023].mkv", "Kasane: The Hidden Parts", (), None),
            ("Warden II Season 02.mkv", "Warden", ("Warden II",), None),
            ("Zap Pilot 100 Season 02.mkv", "Zap Pilot 100", (), None),
        ],
    )
    def test_a_season_label_behind_the_title(
        self, name: str, series: str, anilist: tuple[str, ...], number: int | None
    ) -> None:
        assert _read(name, series, anilist)[0] == number

    @pytest.mark.parametrize(
        ("name", "series"),
        [
            (
                "[grp] Hat and Gloves with Scarf 06v2 - Les Voleurs (BD 720p).mkv",
                "Hat & Gloves with Scarf",
            ),
            (
                "[grp] Hat & Gloves with Scarf 06v2 - Les Voleurs (BD 720p).mkv",
                "Hat and Gloves with Scarf",
            ),
        ],
    )
    def test_an_ampersand_and_the_word_and_agree_between_a_name_and_its_title(self, name: str, series: str) -> None:
        member = read_names([name], TitleGround.of(EntryNames(series)))[name].member

        assert member is not None
        assert (member.number, member.tail_words) == (6, ("les", "voleurs"))

    def test_the_title_check_keeps_the_prefix_a_form_would_read(self) -> None:
        ground = TitleGround.of(EntryNames("Seeker x Seeker (2011)"))
        names = ["Seeker x Seeker (2011) - 04 - Title.mkv", "Seeker x Seeker (2011) - 05 (OVA) - Title.mkv"]

        assert {read.member.prefix for read in read_names(names, ground).values() if read.member} == {
            "Seeker x Seeker (2011)"
        }

    @pytest.mark.parametrize(
        ("name", "series", "anilist"),
        [
            ("Wisp.100.Lantern.Keepers.of.the.Moor.12.1080p.WEB.H264-grp.mkv", "", ()),
            ("Wisp.100.Lantern.Keepers.of.the.Moor.12.1080p.WEB.H264-grp.mkv", "Wispish 100", ()),
            ("Steel Armor Titan - 05 - Title.mkv", "Steel Armor Titan 0083: Comet Archive", ()),
            ("74 - 05 (OVA) - Title.mkv", "74: Seventy Four", ("74 SEVENTY-FOUR",)),
            (
                "[ZM] Halcyon 1.11 - We Are (Not) Alone (BD Remux 1080p).mkv",
                "Dawn Genesis Halcyon",
                ("Halcyon: 1.0 We Are (Not) Alone",),
            ),
            ("Zappu.Pairotto.Hyaku.S02.Ep05.1080p.WEB.H.264-grp.mkv", "Zap Pilot 100", ("Zap Pilot 100 II",)),
            ("Show.101.The.12.Realms.1080p.WEB-grp.mkv", "The 12 Realms", ()),
            # the title's number alone, or its words past the number alone, open nothing
            ("0083.10.1080p.BluRay.x264-GRP.mkv", "Steel Armor Titan 0083: Comet Archive", ()),
            ("Comet.Archive.10.1080p.BluRay.x264-GRP.mkv", "Steel Armor Titan 0083: Comet Archive", ()),
            # a name with no word opens no title
            ("&.mkv", "Steel Armor Titan", ()),
        ],
    )
    def test_a_title_the_name_does_not_open_with_leaves_the_read_alone(
        self, name: str, series: str, anilist: tuple[str, ...]
    ) -> None:
        plain = read_name(name)

        assert _read(name, series, anilist) == (plain.member.number if plain.member else None, plain.bare_numbers)

    @pytest.mark.parametrize(
        ("name", "series", "number", "numbers"),
        [
            ("Steel.Armor.Titan.0083.10.1080p.mkv", "Steel Armor Titan 0083: Comet Archive", 10, (83,)),
            ("Steel.Armor.Titan.0083.Comet.10.1080p.mkv", "Steel Armor Titan 0083: Comet Archive", 10, (83,)),
            # shortened in front too, through a number and a word
            ("Titan.0083.10.1080p.BluRay.x264-GRP.mkv", "Steel Armor Titan 0083: Comet Archive", 10, (83,)),
            ("Wisp.100.05.1080p.WEB.H264-grp.mkv", "Wisp 100: Lantern Keepers of the Moor", 5, (100,)),
            ("[grp] Wisp 100 - 05 (OVA) - Title.mkv", "Wisp 100: Lantern Keepers of the Moor", 5, (100,)),
        ],
    )
    def test_a_title_shortened_behind_its_number_still_opens_the_name(
        self, name: str, series: str, number: int, numbers: tuple[int, ...]
    ) -> None:
        member = read_names([name], TitleGround.of(EntryNames(series)))[name].member

        assert member is not None
        assert (member.number, member.title_numbers) == (number, numbers)

    @pytest.mark.parametrize(
        ("name", "series", "anilist", "number"),
        [
            ("[grp] Kimi no Chef Academy - 2 [1080p].mkv", "My Chef Academy", ("Kimi no Chef Academy 2",), 2),
            ("[grp] My Chef Academy - 2 (1080p).mkv", "My Chef Academy", ("My Chef Academy Season 2",), 2),
            ("My Chef Academy Episode 2.mkv", "My Chef Academy", ("My Chef Academy Season 2",), 2),
            ("[grp] Kimi no Chef Academy - 2 [1080p].mkv", "Kimi no Chef Academy 2", (), 2),
            ("[grp] Drawman - 3 [1080p].mkv", "Drawman.", ("Drawman. 3",), 3),
            # the count behind the title outranks the set-off one
            ("Kimi no Chef Academy - 2 - 05.mkv", "My Chef Academy", ("Kimi no Chef Academy 2",), 5),
            # the title alone, its number written as the title writes it: no count
            ("[grp] Kimi no Chef Academy 2 [1080p].mkv", "My Chef Academy", ("Kimi no Chef Academy 2",), None),
            ("[grp] Kimi no Chef Academy 2 [1080p].mkv", "Kimi no Chef Academy 2", (), None),
            ("[grp] Drawman. 3 [1080p].mkv", "Drawman.", ("Drawman. 3",), None),
            ("Hex Academy 0 (2021) [v3].mkv", "Hex Academy 0", (), None),
        ],
    )
    def test_a_count_equal_to_the_titles_season_digit_stands_when_set_off(
        self, name: str, series: str, anilist: tuple[str, ...], number: int | None
    ) -> None:
        assert _read(name, series, anilist)[0] == number

    @pytest.mark.parametrize(
        ("name", "series"),
        [
            ("Sea Atlas - 20 Years' Worth of Wishes.mkv", "Sea Atlas"),
            ("Pixie Saga - 24-Hour Marathon Relay Dash.mkv", "Pixie Saga"),
            ("Pocketbeast - 10,000,000 Reasons.mkv", "Pocketbeast"),
            ("Pixie.Saga.100.Years.Voyage.05.1080p.mkv", "Pixie Saga"),
            ("Show.101.The.12.Realms.1080p.WEB-grp.mkv", "Show"),
            ("[grp] Show - 12 TV Station Blues [1080p].mkv", "Show"),
            ("[grp] Show - 12 Raw Deal [1080p].mkv", "Show"),
        ],
    )
    def test_a_word_behind_the_count_makes_it_an_episode_titles_number(self, name: str, series: str) -> None:
        for anilist in ((), (series,), (series.upper(), series)):
            assert _read(name, series, anilist)[0] is None
        assert _read(name)[0] is None

    @pytest.mark.parametrize(
        ("name", "series", "anilist", "number"),
        [
            (
                "[Kiyomaru'sGirls] Modan-kun ga Tooru - 06v2 TV (1978) [BD AVC OPUS] [3AFD5410].mkv",
                "Modan-kun ga Tooru",
                ("Modan-kun: Here Comes Mister Retro", "Modan-kun ga Tooru"),
                6,
            ),
            (
                "[Kiyomaru'sGirls] Modan-kun ga Tooru - 29 TV (1978) [BD AVC OPUS] [7C19E0A2].mkv",
                "Modan-kun ga Tooru",
                (),
                29,
            ),
            ("[grp] Show - 12 MULTi [1080p].mkv", "Show", (), 12),
            ("Show - 12 Dual-Audio.mkv", "Show", (), 12),
            ("Sea Atlas - 1075 HEVC.mkv", "Sea Atlas", ("Sea Atlas",), 1075),
            (
                "Wisp.100.Lantern.Keepers.of.the.Moor.12.MULTi.1080p.mkv",
                "Wisp 100: Lantern Keepers of the Moor",
                (),
                12,
            ),
            ("Zap.Pilot.100.01.HEVC.1080p.mkv", "Zap Pilot 100", (), 1),
            (
                "[A-D] Q Dawn of Queens 13 Director's Cut [BDRip 1920x1080 x264 FLAC] [2E7B90C1].mkv",
                "Q Project",
                ("Q: Dawn of Queens", "Q Dawn of Queens"),
                13,
            ),
            ("[grp] Sea Atlas - 05 Uncensored [1080p].mkv", "Sea Atlas", (), 5),
            ("[grp] Show - 05 CR WEB-DL 1080p.mkv", "Show", (), 5),
            ("[grp] Show - 05 Uncensored Version [1080p].mkv", "Show", (), 5),
            ("[grp] Show - 05 VOSTFR Hardsub [1080p].mkv", "Show", (), 5),
            # the tag runs to the count's end or not at all: a title's word behind it, and the own special's mark
            ("[grp] Show - 12 TV Station Blues [1080p].mkv", "Show", (), None),
            ("[Brid] Brass;Door - 23 β - Open the Hidden Path [A81C39E5].mkv", "Brass;Door", (), None),
            ("[grp] BX8 the Eternity - Episode 01 Preview [1080p].mkv", "BX8 the Eternity", (), None),
        ],
    )
    def test_a_release_tag_behind_the_count_is_no_titles_word(
        self, name: str, series: str, anilist: tuple[str, ...], number: int | None
    ) -> None:
        assert _read(name, series, anilist)[0] == number

    @pytest.mark.parametrize(
        "name",
        [
            "[Fete Rider] SoLid Charge! - 13 OVA [BD 1080p][6E03B7D9].mkv",
            "[grp] SoLid Charge! - 13 ONA [1080p].mkv",
            "SoLid.Charge.13.OAD.1080p.BluRay.x264-grp.mkv",
        ],
    )
    def test_an_ova_tag_behind_the_count_marks_an_extra_with_no_count(self, name: str) -> None:
        read = read_names([name], TitleGround.of(EntryNames("Magic Boy Lyrical Pip", ("SoLid Charge!",))))[name]

        assert (read.member, read.titled_count) == (None, None)

    @pytest.mark.parametrize(
        "name",
        [
            "[CBX] Silvertail 01-02 [DVDrip 720x480 x264 FLAC].mkv",
            "Show - 21.5.mkv",
            "Show - 01.5.mkv",
            "Show - 001.5.mkv",
            "Show - 12 2011.mkv",
            "Show - 02 - 05 - Title.mkv",
            "Show.101.S03.480p.DVD.x264-grp.mkv",
            "Show.10.bit.1080p.mkv",
            "Show.60.fps.1080p.mkv",
            "Comet.X.264.1080p.WEB-grp.mkv",
            "Show 2nd Season - Season 02.mkv",
        ],
    )
    def test_a_refusal_the_title_does_not_explain_stands(self, name: str) -> None:
        assert _read(name, "Show", ("Silvertail", "Comet X"))[0] is None

    def test_a_read_with_a_number_already_keeps_it(self) -> None:
        # "002" is the count, and no "Pixie Saga 2".
        assert _read("[Fenwick] Pixie Saga - 002.mkv", "Pixie Saga", ("Pixie Saga 2",)) == (2, (2,))
        assert _read("Show 12 - 05.mkv", "Show") == (5, ())
        # the count past the title equals the title's number: counted past it, it stands
        assert _read("Tinman 009 - 009.mkv", "Tinman 009") == (9, (9,))
        # a season the title counts in words is no digit of the title's: the form's count stands
        assert _read("[grp] Meet A Star - 4 [1080p].mkv", "Meet A Star", ("Meet A Star IV",)) == (4, ())

    @pytest.mark.parametrize(
        ("name", "series", "numbers"),
        [
            (
                "Wisp.100.Lantern.Keepers.of.the.Moor.12.1080p.WEB.H264-grp.mkv",
                "Wisp 100: Lantern Keepers of the Moor",
                (100,),
            ),
            ("[grp] Zap Pilot 100 - 05 (BD 1080p).mkv", "Zap Pilot 100", (100,)),
            ("Seeker.x.Seeker.2011.05.1080p.BluRay.x264-grp.mkv", "Seeker x Seeker (2011)", (2011,)),
            ("[Zog42] Seeker x Seeker (2011) - 05 (BD 1080p).mkv", "Seeker x Seeker (2011)", ()),
            ("[grp] Zap Pilot 100 - 05 (BD 1080p).mkv", "Zap Pilot", ()),
            ("[grp] Show - 05 (BD 1080p).mkv", "Show", ()),
        ],
    )
    def test_the_titles_numbers_in_the_name_ride_on_the_member(
        self, name: str, series: str, numbers: tuple[int, ...]
    ) -> None:
        member = read_names([name], TitleGround.of(EntryNames(series)))[name].member

        assert member is not None
        assert member.title_numbers == numbers

    def test_a_title_with_no_digits_leaves_every_count(self) -> None:
        assert TitleGround.of(EntryNames("Show", ("Warden II", "Meet A Star IV"))).numbers == frozenset()
        assert TitleGround.of(EntryNames("Tinman 009", ("Seeker x Seeker (2011)",))).numbers == {9, 2011}

    def test_the_titles_numbers_leave_the_bare_numbers_only(self) -> None:
        assert _read("Tinman 009 - 009.mkv", "Tinman 009") == (9, (9,))
        assert _read("Zap Pilot 100 - 05.mkv", "Zap Pilot 100") == (5, ())
        assert _read("Zap Pilot 100 - 05.mkv") == (5, (100,))
        # a count before the title ("[0802]" of a leading absolute) is no title's: it stays
        assert _read("[0802] Sea Atlas - Garmo 66.mkv", "Sea Atlas") == (None, (802,))
        assert _read("[101] Quillfox - Title.mkv", "Quillfox") == (None, (101,))
        # a group tag's number ("[XYZ-2223]") counts nothing: the count behind a season word stays the member's
        assert _read("[XYZ-2223] Hana to Kaze S2 - 05 [1080p].mkv", "Hana to Kaze") == (5, ())
        # a title read short, with words between it and the count
        assert _read("Wisp.100.Lantern.Keepers.of.the.Moor.12.1080p.WEB.H264-grp.mkv", "Wisp 100") == (None, ())

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("[0802] Sea Atlas - Garmo 66.mkv", None),
            ("[0167] Sea Atlas - 100 Against 1.mkv", None),
            # a label's number, a count behind a two-digit tag, and a count with nothing counted before it stand
            ("[0802] Sea Atlas - Episode 66.mkv", 66),
            ("[44] Sea Atlas - 17 Times 2.mkv", 2),
            ("Sea Atlas - Garmo 66.mkv", 66),
        ],
    )
    def test_a_trailing_number_behind_words_past_the_title_is_the_titles_when_a_count_leads(
        self, name: str, number: int | None
    ) -> None:
        assert _read(name, "Sea Atlas")[0] == number

    def test_a_number_ending_the_members_tail_leaves_the_member(self) -> None:
        assert _read("[0802] Show Film - 05 - Red 2.mkv", "Show") == (5, (802,))

    def test_a_quality_tail_ahead_of_the_title_drops_just_the_titles_number(self) -> None:
        assert _read("[Grp][1080p][x.264] Show 100 - 05.mkv", "Show 100") == (5, ())

    def test_a_quality_words_second_half_behind_the_count_opens_no_quality_tail(self) -> None:
        name = "[grp] Show - 05 Ray of Hope [1080p].mkv"
        read = read_names([name], TitleGround.of(EntryNames("Show")))[name]

        assert (read.member, read.titled_count) == (None, 5)

    def test_fullwidth_digits_in_the_title_are_the_titles_numbers(self) -> None:
        name = f"[grp] Show {_FULLWIDTH_100} - 05 [1080p].mkv"
        member = read_names([name], TitleGround.of(EntryNames("Show 100")))[name].member

        assert member is not None
        assert (member.number, member.title_numbers) == (5, (100,))

    @pytest.mark.parametrize(
        ("name", "series", "count", "words"),
        [
            ("Show - 13 Days of Summer.mkv", "Show", 13, ("days", "of", "summer")),
            (
                "Cinder.206.The.Lost.Letter.Arrives.The.Story.from.110.Years.Ago.480p.DVD.x264-grp.mkv",
                "Cinder",
                206,
                (
                    "the",
                    "lost",
                    "letter",
                    "arrives",
                    "the",
                    "story",
                    "from",
                    "110",
                    "years",
                    "ago",
                    "480p",
                    "dvd",
                    "x264",
                    "grp",
                ),
            ),
            ("Pocketbeast - 10,000,000 Reasons.mkv", "Pocketbeast", 10, ("0", "0", "reasons")),
            ("Sea Atlas - 20 Years' Worth of Wishes.mkv", "Sea Atlas", 20, ("years", "worth", "of", "wishes")),
            # counted, or unnumbered with no count behind the title: no titled count, and the words as before
            ("[grp] Show - 12 MULTi [1080p].mkv", "Show", None, ("multi",)),
            ("[101] Quillfox - Title.mkv", "Quillfox", None, ("quillfox", "title")),
        ],
    )
    def test_a_count_a_word_follows_is_kept_as_the_titled_count_with_the_words_behind_it(
        self, name: str, series: str, count: int | None, words: tuple[str, ...]
    ) -> None:
        read = read_names([name], TitleGround.of(EntryNames(series)))[name]

        assert (read.titled_count, read.title_words) == (count, words)

    def test_the_titles_are_matched_longest_first_with_and_without_their_bracket(self) -> None:
        ground = TitleGround.of(EntryNames("Seeker x Seeker (2011)", ("Seeker x Seeker", "SEEKER×SEEKER (2011)")))

        assert ground.titles == (
            ("seeker", "x", "seeker", "2011"),
            ("seeker", "x", "seeker"),
            ("seeker", "seeker", "2011"),
            ("seeker", "seeker"),
        )
        assert TitleGround.of(EntryNames()).titles == ()
        # words that only count a season are no title: "2nd Season" left of its bracket would open any "2.01"
        assert TitleGround.of(EntryNames("[Ame no Hi]", ("[Ame no Hi] 2nd Season",))).titles == (
            ("ame", "no", "hi", "2"),
            ("ame", "no", "hi"),
        )
        assert _read("2.01 Kyoto Blade.mkv", "[Ame no Hi]", ("[Ame no Hi] 2nd Season",)) == (None, ())
