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
            "Mob.Psycho.100.01.1080p.BluRay.x264-grp.mkv",
            "Zom.100.Bucket.List.of.the.Dead.01.1080p.WEB.H264-grp.mkv",
            "Mobile.Suit.Gundam.0083.Stardust.Memory.01.1080p.BluRay.x264-grp.mkv",
            "Mobile.Suit.Gundam.0080.War.in.the.Pocket.E01.1080p.BluRay.mkv",
            "Cyborg.009.01.DVDRip.x264-grp.mkv",
            "Show.2024.BluRay.1080p.FLAC5.1.H.264.REMUX-grp.mkv",
            "Mob.Psycho.100.S02.Ep01.1080p.WEB.H.264-grp.mkv",
        ],
    )
    def test_a_title_or_codec_number_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        "name",
        [
            "Naruto.101.v4.480p.DVD.FLAC2.0.x264-grp.mkv",
            "Naruto.101.v4.1080p.BluRay.10-bit.x265-grp.mkv",
        ],
    )
    def test_the_count_before_the_quality_words_is_read(self, name: str) -> None:
        member = read_name(name).member

        assert member is not None
        assert (member.number, member.version) == (101, 4)

    @pytest.mark.parametrize(
        "name",
        [
            "Naruto.101.v4.480p.DVD.Vol.26.Dual-Audio.FLAC2.0.Hi10P.x264-grp.mkv",
            "Naruto.101.v4.480p.DVD.Remux.10.bit.FLAC2.0.x264-grp.mkv",
            "Naruto.101.v4.480p.DVD.Dual-Audio.FLAC.2.0.10.bit.x264-grp.mkv",
            "Naruto.101.v4.480p.DVD.23.976fps.Dual-Audio.FLAC2.0.Hi10P.x264-grp.mkv",
        ],
    )
    def test_a_number_in_the_quality_tail_is_no_later_count(self, name: str) -> None:
        member = read_name(name).member

        assert member is not None
        assert (member.number, member.version) == (101, 4)

    @pytest.mark.parametrize(
        "name",
        [
            "Naruto.101.The.100.Year.Plan.1080p.WEB-grp.mkv",
            "Naruto.101.Disc.02.480p.DVD.x264-grp.mkv",
            "Naruto.101.S03.480p.DVD.x264-grp.mkv",
            "Naruto.101.480p.DVD.x264-grp.S01.mkv",
        ],
    )
    def test_a_count_with_another_count_or_a_key_behind_it_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    def test_a_number_glued_to_a_hyphen_is_no_trailing_count(self) -> None:
        member = read_name("Naruto.101.480p.DVD.x264-grp-25.mkv").member

        assert member is not None
        assert member.number == 101

    @pytest.mark.parametrize(
        "name",
        [
            "[grp] Angel Beats! - 12 & 13 [1080p].mkv",
            "[grp] Angel Beats! - 12+13 [1080p].mkv",
            "[grp] Angel Beats! - 12, 13 [1080p].mkv",
            "[grp] Angel Beats! - 12 and 13 [1080p].mkv",
            "[grp] Angel Beats! - 12 to 13 [1080p].mkv",
            "[grp] Angel Beats! - 12~13 [1080p].mkv",
            "Angel.Beats.12&13.1080p.WEB-grp.mkv",
        ],
    )
    def test_a_number_a_joiner_ties_to_the_one_before_is_no_count(self, name: str) -> None:
        # A double episode's second number ends the name; counting by it places the file one episode late.
        assert read_name(name).member is None
        assert _read(name, "Angel Beats!")[0] is None

    def test_a_lone_season_token_is_no_trailing_count(self) -> None:
        assert read_name("[grp] Show S01 (BD 1080p x265 FLAC).mkv").member is None

    @pytest.mark.parametrize(
        ("name", "prefix"),
        [
            ("Bleach.001.1080p.BluRay.x264-grp.mkv", "Bleach"),
            ("Sonic.X.001.1080p.WEB.H.264-grp.mkv", "Sonic.X"),
            ("Yu-Gi-Oh.GX.001.720p.DVD.x264-grp.mkv", "Yu-Gi-Oh.GX"),
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
            "Sonic.X.264.1080p.WEB.H.264-grp.mkv",
        ],
    )
    def test_a_codec_number_behind_a_lone_letter_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Naruto.101.x.264.480p.DVD-grp.mkv", 101),
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
            ("Sonic.X.264.001.1080p.WEB-grp.mkv", 1),
        ],
    )
    def test_a_codec_number_before_the_count_is_skipped(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Tomodachi.Game.007.I.ll.Take.20.Million.Yen.Worth.of.Your.Life.1080p.WEB.H.264-grp.mkv", 7),
            ("Naruto.001.The.100.Year.Plan.1080p.WEB-grp.mkv", 1),
            ("Non.Non.Biyori.033.401.Unauthorized.1080p.WEB-grp.mkv", 33),
            ("Re.ZERO.050.Killua.X.1000.X.Hunters.Bar.1080p.WEB-grp.mkv", 50),
            ("Black.Clover.023.Vol.26.1080p.WEB.H.264-grp.mkv", 23),
        ],
    )
    def test_a_padded_count_stands_before_an_unpadded_title_number(self, name: str, number: int) -> None:
        member = read_name(name).member

        assert member is not None
        assert member.number == number

    @pytest.mark.parametrize(
        "name",
        [
            "Full.Metal.Panic.036.Episode.000.1080p.WEB-grp.mkv",
            "Akame.ga.Kill.025.AkaKill.Theater.01.Tatsumi.and.Leone.1080p.WEB-grp.mkv",
            "Zom.100.Bucket.List.of.the.Dead.05.1080p.WEB-grp.mkv",
            "Black.Clover.123.Vol.26.1080p.WEB.H.264-grp.mkv",
        ],
    )
    def test_a_padded_number_behind_the_count_or_an_unpadded_count_is_refused(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("Pokemon.1069.1080p.WEB.H.264-grp.mkv", 1069),
            ("Show.1999.001.1080p.WEB-grp.mkv", 1),
            ("Hunter.x.Hunter.2011.067.1080p.WEB-grp.mkv", 67),
            ("1000.nen.Joou.001.1080p.WEB-grp.mkv", 1),
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
            "[BD Menu 01] To Your Eternity - Season 01 [2021].mkv",
            "[Ending 01] Aoashi - Season 01 [2022].mkv",
            "Naruto.101.480p.DVD.x264-JySzE.Season.3.mkv",
            # an ordinal season's lone digit numbers the season's part, not an episode
            "[grp] Shingeki no Kyojin - The Final Season 2 (1080p).mkv",
            "[grp] Haikyuu!! 2nd Season 2.mkv",
        ],
    )
    def test_a_plain_season_number_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("[Asakura] Tensei Shitara Slime Datta Ken 2nd Season 25 [BDRip 1920x1080 x265 10bit FLAC].mkv", 25),
            ("[grp] Monogatari Series Second Season - 01 [1080p].mkv", 1),
            ("[grp] Show Season 2 - 05 [1080p].mkv", 5),
            ("[grp] Show Second Season 25 [1080p].mkv", 25),
            ("[grp] Attack on Titan The Final Season 25 [1080p].mkv", 25),
            ("[grp] Show Last Season 07.mkv", 7),
            ("[grp] Shingeki no Kyojin The Final Season 02 (1080p).mkv", 2),
            ("[grp] Shingeki no Kyojin The Final Season - 2 (1080p).mkv", 2),
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

        assert [reads[name].member for name in specials] == [
            RunMember(name, "[grp] Show S", n, (), 1) for n, name in enumerate(specials, 1)
        ]
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
            ["Toonami USA OP04 - Tom S3 [Arian].mkv", "Toonami USA OP06 - Tom S4 [Arian].mkv"],
        ],
    )
    def test_a_lone_s_alone_repeated_or_not_from_one_stays_unnumbered(self, names: list[str]) -> None:
        assert self._numbers(self._EPISODES + names) == {**self._numbers(self._EPISODES), **dict.fromkeys(names)}


class TestMiddleNumber:
    """The " - NN - " count between a title and an episode title."""

    def test_a_second_middle_segment_is_a_title_number_and_no_count(self) -> None:
        assert read_name("SPY x FAMILY - 277 - 10 - 1 = [WEBDL-1080p].mkv").member is None


class TestPaddedCounts:
    """A count of four digits: padded behind a zero, or past the thousandth episode."""

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("One Piece - 0802 - An Angry Sanji The Secret of Germa 66 [WEBDL-1080p].mkv", 802),
            ("One Piece - 1001 - The Title [WEBDL-1080p].mkv", 1001),
            ("[grp] Pokemon - 1069 [1080p].mkv", 1069),
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
            ("Mobu.Saiko.Hyaku.S02.Ep05.1080p.WEB.H.264-grp.mkv", ()),
            ("Sonic.X.264.1080p.WEB.H.264-grp.mkv", (264,)),
            ("Naruto.101.The.100.Year.Plan.1080p.WEB-grp.mkv", (101, 100)),
            ("Mob Psycho 100 Season 2 Episode 05 [1080p].mkv", (100,)),
            ("Show (2019) Season 2 Episode 05 [1080p].mkv", (2019,)),
            ("Naruto.101.v4.480p.DVD.FLAC2.0.x264-grp.mkv", (101,)),
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
            "[Sristi] Shiki - 21.5.mkv",
            "Mushoku Tensei - 7.5 [1080p].mkv",
            "[MTBB] Hyouka - 11.5 (OVA) (BD 1080p) [9199C7FF].mkv",
            "Space.Brothers.S01EP11.Trapped.Rivals.1080p.Remux.AVC.FLAC2.0.mkv",
            "[Beatrice-Raws] Date A Live (Director's Cut Edition) PV ver.1.1 [BDRip 1920x1080 x264 FLAC].mkv",
        ],
    )
    def test_a_fraction_is_no_count(self, name: str) -> None:
        assert read_name(name).member is None

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("[CyC] Mushishi 12v2.1 (BD 720p Hi10 x264 AC3+FLAC) [Dual-Audio] [52786A01].mkv", 12),
            ("[aro] Shigofumi - 06 v2.1 [BD 1080p x264 FLAC].mkv", 6),
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
                "Zom.100.Bucket.List.of.the.Dead.12.1080p.WEB.H264-grp.mkv",
                "Zom 100: Bucket List of the Dead",
                (),
                12,
                (),
            ),
            ("Mob.Psycho.100.001.1080p.WEB.H.264-grp.mkv", "Mob Psycho 100", (), 1, (1,)),
            ("Mob.Psycho.100.01.1080p.BluRay.x264-grp.mkv", "Mob Psycho 100", (), 1, ()),
            ("Cyborg.009.001.1080p.mkv", "Cyborg 009", (), 1, (1,)),
            ("Mob Psycho 100 II - 05 (OVA) - Title.mkv", "Mob Psycho 100", ("Mob Psycho 100 II",), 5, ()),
            ("Mob Psycho 100 2nd Season - 05 (OVA) - Title.mkv", "Mob Psycho 100", ("Mob Psycho 100 II",), 5, ()),
            ("[grp] Baki - 05 (OVA) - Title.mkv", "Baki (2018)", (), 5, ()),
            ("Hunter x Hunter (2011) - 05 (OVA) - Title.mkv", "Hunter x Hunter (2011)", (), 5, (2011,)),
            ("[grp] Steins;Gate - 25 (OVA) - Egoistic Poriomania [ABCD1234].mkv", "Steins;Gate", (), 25, ()),
            ("Show - Episode 12 (OVA) - Title.mkv", "Show", (), 12, ()),
            ("Show - 001.10.bit.mkv", "Show", (), 1, (1,)),
            ("Show - 01.10.bit.mkv", "Show", (), 1, ()),
            ("[grp] Maid Sama! - 14 END [1080p].mkv", "Maid Sama!", (), 14, ()),
            ("[grp] Is This a Zombie? - 13 END [1080p].mkv", "Is This a Zombie?", (), 13, ()),
            # a bracket behind the count, nested or not, a source word or a quality word there end it like the stem
            ("Eromanga Sensei - 01 [1920x1080 - AVC - [PCM, ja]].mkv", "Eromanga Sensei", (), 1, ()),
            ("[grp] Tokyo Ghoul - 50 (Tokyo Ghoul: Jack (OVA)) [1080p].mkv", "Tokyo Ghoul", (), 50, ()),
            ("Cyborg.009.01.DVDRip.x264-grp.mkv", "Cyborg 009", (), 1, ()),
            ("Mob.Psycho.100.018.PROPER.1080p.WEB-grp.mkv", "Mob Psycho 100", (), 18, (18,)),
            ("Mob.Psycho.100.018.Hi10P.1080p.WEB-grp.mkv", "Mob Psycho 100", (), 18, (18,)),
            # a form counted by the title's own digits: nothing was counted, the count behind the title is
            ("Cyborg.009.10.1080p.WEB.H.264-grp.mkv", "Cyborg 009", (), 10, ()),
            (
                "Mobile.Suit.Gundam.0083.Stardust.Memory.10.1080p.mkv",
                "Mobile Suit Gundam 0083: Stardust Memory",
                (),
                10,
                (),
            ),
            (
                "[grp] Yojouhan Shinwa Taikei - 11 END (BD 1080p).mkv",
                "The Tatami Galaxy",
                ("Yojouhan Shinwa Taikei",),
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
                "[grp] Boku no Hero Academia 4th Season - 25 END [1080p].mkv",
                "My Hero Academia",
                ("Boku no Hero Academia 4th Season",),
                25,
            ),
            # a number right behind the label is the label's: a season, no count
            ("[BD Menu 01] To Your Eternity - Season 01 [2021].mkv", "To Your Eternity", (), None),
            ("[PV 03] Horimiya - The Missing Pieces - Season 01 [2023].mkv", "Horimiya: The Missing Pieces", (), None),
            ("Overlord II Season 02.mkv", "Overlord", ("Overlord II",), None),
            ("Mob Psycho 100 Season 02.mkv", "Mob Psycho 100", (), None),
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
                "[grp] Panty and Stocking with Garterbelt 06v2 - Les Diaboliques (BD 720p).mkv",
                "Panty & Stocking with Garterbelt",
            ),
            (
                "[grp] Panty & Stocking with Garterbelt 06v2 - Les Diaboliques (BD 720p).mkv",
                "Panty and Stocking with Garterbelt",
            ),
        ],
    )
    def test_an_ampersand_and_the_word_and_agree_between_a_name_and_its_title(self, name: str, series: str) -> None:
        member = read_names([name], TitleGround.of(EntryNames(series)))[name].member

        assert member is not None
        assert (member.number, member.tail_words) == (6, ("les", "diaboliques"))

    def test_the_title_check_keeps_the_prefix_a_form_would_read(self) -> None:
        ground = TitleGround.of(EntryNames("Hunter x Hunter (2011)"))
        names = ["Hunter x Hunter (2011) - 04 - Title.mkv", "Hunter x Hunter (2011) - 05 (OVA) - Title.mkv"]

        assert {read.member.prefix for read in read_names(names, ground).values() if read.member} == {
            "Hunter x Hunter (2011)"
        }

    @pytest.mark.parametrize(
        ("name", "series", "anilist"),
        [
            ("Zom.100.Bucket.List.of.the.Dead.12.1080p.WEB.H264-grp.mkv", "", ()),
            ("Zom.100.Bucket.List.of.the.Dead.12.1080p.WEB.H264-grp.mkv", "Zombie 100", ()),
            ("Mobile Suit Gundam - 05 - Title.mkv", "Mobile Suit Gundam 0083: Stardust Memory", ()),
            ("86 - 05 (OVA) - Title.mkv", "86: Eighty Six", ("86 EIGHTY-SIX",)),
            (
                "[QM] Evangelion 1.11 - You Are (Not) Alone (BD Remux 1080p).mkv",
                "Neon Genesis Evangelion",
                ("Evangelion: 1.0 You Are (Not) Alone",),
            ),
            ("Mobu.Saiko.Hyaku.S02.Ep05.1080p.WEB.H.264-grp.mkv", "Mob Psycho 100", ("Mob Psycho 100 II",)),
            ("Show.101.The.12.Kingdoms.1080p.WEB-grp.mkv", "The 12 Kingdoms", ()),
            # the title's number alone, or its words past the number alone, open nothing
            ("0083.10.1080p.BluRay.x264-GRP.mkv", "Mobile Suit Gundam 0083: Stardust Memory", ()),
            ("Stardust.Memory.10.1080p.BluRay.x264-GRP.mkv", "Mobile Suit Gundam 0083: Stardust Memory", ()),
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
            ("Mobile.Suit.Gundam.0083.10.1080p.mkv", "Mobile Suit Gundam 0083: Stardust Memory", 10, (83,)),
            ("Mobile.Suit.Gundam.0083.Stardust.10.1080p.mkv", "Mobile Suit Gundam 0083: Stardust Memory", 10, (83,)),
            # shortened in front too, through a number and a word
            ("Gundam.0083.10.1080p.BluRay.x264-GRP.mkv", "Mobile Suit Gundam 0083: Stardust Memory", 10, (83,)),
            ("Zom.100.05.1080p.WEB.H264-grp.mkv", "Zom 100: Bucket List of the Dead", 5, (100,)),
            ("[grp] Zom 100 - 05 (OVA) - Title.mkv", "Zom 100: Bucket List of the Dead", 5, (100,)),
        ],
    )
    def test_a_title_shortened_behind_its_number_still_opens_the_name(
        self, name: str, series: str, number: int, numbers: tuple[int, ...]
    ) -> None:
        # A name drops a subtitle behind the title's number, never the number: the count follows what it kept.
        member = read_names([name], TitleGround.of(EntryNames(series)))[name].member

        assert member is not None
        assert (member.number, member.title_numbers) == (number, numbers)

    @pytest.mark.parametrize(
        ("name", "series", "anilist", "number"),
        [
            ("[grp] Boku no Hero Academia - 2 [1080p].mkv", "My Hero Academia", ("Boku no Hero Academia 2",), 2),
            ("[grp] My Hero Academia - 2 (1080p).mkv", "My Hero Academia", ("My Hero Academia Season 2",), 2),
            ("My Hero Academia Episode 2.mkv", "My Hero Academia", ("My Hero Academia Season 2",), 2),
            ("[grp] Boku no Hero Academia - 2 [1080p].mkv", "Boku no Hero Academia 2", (), 2),
            ("[grp] Bakuman - 3 [1080p].mkv", "Bakuman.", ("Bakuman. 3",), 3),
            # the count behind the title outranks the set-off one
            ("Boku no Hero Academia - 2 - 05.mkv", "My Hero Academia", ("Boku no Hero Academia 2",), 5),
            # the title alone, its number written as the title writes it: no count
            ("[grp] Boku no Hero Academia 2 [1080p].mkv", "My Hero Academia", ("Boku no Hero Academia 2",), None),
            ("[grp] Boku no Hero Academia 2 [1080p].mkv", "Boku no Hero Academia 2", (), None),
            ("[grp] Bakuman. 3 [1080p].mkv", "Bakuman.", ("Bakuman. 3",), None),
            ("Jujutsu Kaisen 0 (2021) [v3].mkv", "Jujutsu Kaisen 0", (), None),
        ],
    )
    def test_a_count_equal_to_the_titles_season_digit_stands_when_set_off(
        self, name: str, series: str, anilist: tuple[str, ...], number: int | None
    ) -> None:
        # Under the second season's entry "Show - 2" and "Show Episode 2" count its second episode, as they
        # would under any other title; "Show 2" alone is the title.
        assert _read(name, series, anilist)[0] == number

    @pytest.mark.parametrize(
        ("name", "series"),
        [
            ("One Piece - 20 Years' Worth of Prayers.mkv", "One Piece"),
            ("Fairy Tail - 24-Hour Endurance Road Race.mkv", "Fairy Tail"),
            ("Pokemon - 10,000,000 Reasons.mkv", "Pokemon"),
            ("Fairy.Tail.100.Years.Quest.05.1080p.mkv", "Fairy Tail"),
            ("Show.101.The.12.Kingdoms.1080p.WEB-grp.mkv", "Show"),
            ("[grp] Show - 12 TV Station Blues [1080p].mkv", "Show"),
            ("[grp] Show - 12 Raw Deal [1080p].mkv", "Show"),
        ],
    )
    def test_a_word_behind_the_count_makes_it_an_episode_titles_number(self, name: str, series: str) -> None:
        # The episode title opens with the number; the placer's title match needs the name unnumbered. Any word
        # refuses that is no release tag, and a tag only passes the count on to the word behind it.
        for anilist in ((), (series,), (series.upper(), series)):
            assert _read(name, series, anilist)[0] is None
        assert _read(name)[0] is None

    @pytest.mark.parametrize(
        ("name", "series", "anilist", "number"),
        [
            (
                "[Ranmaru'sBoys] Haikara-san ga Tooru - 06v2 TV (1978) [BD AVC OPUS] [1ECB9380].mkv",
                "Haikara-san ga Tooru",
                ("Haikara-san: Here Comes Miss Modern", "Haikara-san ga Tooru"),
                6,
            ),
            (
                "[Ranmaru'sBoys] Haikara-san ga Tooru - 29 TV (1978) [BD AVC OPUS] [542B81D3].mkv",
                "Haikara-san ga Tooru",
                (),
                29,
            ),
            ("[grp] Show - 12 MULTi [1080p].mkv", "Show", (), 12),
            ("Show - 12 Dual-Audio.mkv", "Show", (), 12),
            ("One Piece - 1075 HEVC.mkv", "One Piece", ("One Piece",), 1075),
            ("Zom.100.Bucket.List.of.the.Dead.12.MULTi.1080p.mkv", "Zom 100: Bucket List of the Dead", (), 12),
            ("Mob.Psycho.100.01.HEVC.1080p.mkv", "Mob Psycho 100", (), 1),
            (
                "[A-C] K Return of Kings 13 Director's Cut [BDRip 1920x1080 x264 FLAC] [9D545133].mkv",
                "K Project",
                ("K: Return of Kings", "K Return of Kings"),
                13,
            ),
            ("[grp] One Piece - 05 Uncensored [1080p].mkv", "One Piece", (), 5),
            ("[grp] Show - 05 CR WEB-DL 1080p.mkv", "Show", (), 5),
            ("[grp] Show - 05 Uncensored Version [1080p].mkv", "Show", (), 5),
            ("[grp] Show - 05 VOSTFR Hardsub [1080p].mkv", "Show", (), 5),
            # the tag runs to the count's end or not at all: a title's word behind it, and the own special's mark
            ("[grp] Show - 12 TV Station Blues [1080p].mkv", "Show", (), None),
            ("[Arid] Steins;Gate - 23 β - Open the Missing Link [F4B7708A].mkv", "Steins;Gate", (), None),
            ("[grp] SK8 the Infinity - Episode 01 Preview [1080p].mkv", "SK8 the Infinity", (), None),
        ],
    )
    def test_a_release_tag_behind_the_count_is_no_titles_word(
        self, name: str, series: str, anilist: tuple[str, ...], number: int | None
    ) -> None:
        # "TV (1978)", "MULTi", "Director's Cut" say what the file is, not which episode: the count stands.
        assert _read(name, series, anilist)[0] == number

    @pytest.mark.parametrize(
        "name",
        [
            "[Fete Rider] ViVid Strike! - 13 OVA [BD 1080p][F8BB58F0].mkv",
            "[grp] ViVid Strike! - 13 ONA [1080p].mkv",
            "ViVid.Strike.13.OAD.1080p.BluRay.x264-grp.mkv",
        ],
    )
    def test_an_ova_tag_behind_the_count_marks_an_extra_with_no_count(self, name: str) -> None:
        # The OVA after twelve plain files joins no run of theirs, and the tag is no title's word.
        read = read_names([name], TitleGround.of(EntryNames("Magical Girl Lyrical Nanoha", ("ViVid Strike!",))))[name]

        assert (read.member, read.titled_count) == (None, None)

    @pytest.mark.parametrize(
        "name",
        [
            "[CBT] Gintama 01-02 [DVDrip 720x480 x264 FLAC].mkv",
            "Show - 21.5.mkv",
            "Show - 01.5.mkv",
            "Show - 001.5.mkv",
            "Show - 12 2011.mkv",
            "Show - 02 - 05 - Title.mkv",
            "Show.101.S03.480p.DVD.x264-grp.mkv",
            "Show.10.bit.1080p.mkv",
            "Show.60.fps.1080p.mkv",
            "Sonic.X.264.1080p.WEB-grp.mkv",
            "Show 2nd Season - Season 02.mkv",
        ],
    )
    def test_a_refusal_the_title_does_not_explain_stands(self, name: str) -> None:
        assert _read(name, "Show", ("Gintama", "Sonic X"))[0] is None

    def test_a_read_with_a_number_already_keeps_it(self) -> None:
        # "002" is the count and no "Fairy Tail 2": a digit word matches a title's only as written.
        assert _read("[Placebo] Fairy Tail - 002.mkv", "Fairy Tail", ("Fairy Tail 2",)) == (2, (2,))
        assert _read("Show 12 - 05.mkv", "Show") == (5, ())
        # the count past the title equals the title's number: counted past it, it stands
        assert _read("Cyborg 009 - 009.mkv", "Cyborg 009") == (9, (9,))
        # a season the title counts in words is no digit of the title's: the form's count stands
        assert _read("[grp] Date A Live - 4 [1080p].mkv", "Date A Live", ("Date A Live IV",)) == (4, ())

    @pytest.mark.parametrize(
        ("name", "series", "numbers"),
        [
            ("Zom.100.Bucket.List.of.the.Dead.12.1080p.WEB.H264-grp.mkv", "Zom 100: Bucket List of the Dead", (100,)),
            ("[grp] Mob Psycho 100 - 05 (BD 1080p).mkv", "Mob Psycho 100", (100,)),
            ("Hunter.x.Hunter.2011.05.1080p.BluRay.x264-grp.mkv", "Hunter x Hunter (2011)", (2011,)),
            ("[Pog42] Hunter x Hunter (2011) - 05 (BD 1080p).mkv", "Hunter x Hunter (2011)", ()),
            ("[grp] Mob Psycho 100 - 05 (BD 1080p).mkv", "Mob Psycho", ()),
            ("[grp] Show - 05 (BD 1080p).mkv", "Show", ()),
        ],
    )
    def test_the_titles_numbers_in_the_name_ride_on_the_member(
        self, name: str, series: str, numbers: tuple[int, ...]
    ) -> None:
        # Bracketed digits are no title's, neither a group tag's ("[Pog42]") nor the name's own "(2011)".
        member = read_names([name], TitleGround.of(EntryNames(series)))[name].member

        assert member is not None
        assert member.title_numbers == numbers

    def test_a_title_with_no_digits_leaves_every_count(self) -> None:
        assert TitleGround.of(EntryNames("Show", ("Overlord II", "Date A Live IV"))).numbers == frozenset()
        assert TitleGround.of(EntryNames("Cyborg 009", ("Hunter x Hunter (2011)",))).numbers == {9, 2011}

    def test_the_titles_numbers_leave_the_bare_numbers_only(self) -> None:
        assert _read("Cyborg 009 - 009.mkv", "Cyborg 009") == (9, (9,))
        assert _read("Mob Psycho 100 - 05.mkv", "Mob Psycho 100") == (5, ())
        assert _read("Mob Psycho 100 - 05.mkv") == (5, (100,))
        # a count before the title ("[0802]" of a leading absolute) is no title's: it stays
        assert _read("[0802] One Piece - Germa 66.mkv", "One Piece") == (None, (802,))
        assert _read("[101] Naruto - Title.mkv", "Naruto") == (None, (101,))
        # a group tag's number ("[SCP-2223]") counts nothing: the count behind a season word stays the member's
        assert _read("[SCP-2223] Koi to Uso S2 - 05 [1080p].mkv", "Koi to Uso") == (5, ())
        # The title confirms its 100 without reading the count: no form counts ".Bucket" and the check adds none.
        assert _read("Zom.100.Bucket.List.of.the.Dead.12.1080p.WEB.H264-grp.mkv", "Zom 100") == (None, ())

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("[0802] One Piece - Germa 66.mkv", None),
            ("[0167] One Piece - 100 Against 1.mkv", None),
            # a label's number, a count behind a two-digit tag, and a count with nothing counted before it stand
            ("[0802] One Piece - Episode 66.mkv", 66),
            ("[44] One Piece - 17 Times 2.mkv", 2),
            ("One Piece - Germa 66.mkv", 66),
        ],
    )
    def test_a_trailing_number_behind_words_past_the_title_is_the_titles_when_a_count_leads(
        self, name: str, number: int | None
    ) -> None:
        # "[0802]" counted the file already: the 66 ending "Germa 66" is the episode title's.
        assert _read(name, "One Piece")[0] == number

    def test_a_number_ending_the_members_tail_leaves_the_member(self) -> None:
        # Only the member's own number ending the name tails a title; the "2" here ends its episode title.
        assert _read("[0802] Show Film - 05 - Red 2.mkv", "Show") == (5, (802,))

    def test_a_quality_tail_ahead_of_the_title_drops_just_the_titles_number(self) -> None:
        assert _read("[Grp][1080p][x.264] Show 100 - 05.mkv", "Show 100") == (5, ())

    def test_a_quality_words_second_half_behind_the_count_opens_no_quality_tail(self) -> None:
        # "Ray" ends "Blu-Ray" as the words split it, never a tag of its own: the count opens an episode title.
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
                "Bleach.206.The.Past.Chapter.Begins.The.Truth.from.110.Years.Ago.480p.DVD.x264-grp.mkv",
                "Bleach",
                206,
                (
                    "the",
                    "past",
                    "chapter",
                    "begins",
                    "the",
                    "truth",
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
            ("Pokemon - 10,000,000 Reasons.mkv", "Pokemon", 10, ("0", "0", "reasons")),
            ("One Piece - 20 Years' Worth of Prayers.mkv", "One Piece", 20, ("years", "worth", "of", "prayers")),
            # counted, or unnumbered with no count behind the title: no titled count, and the words as before
            ("[grp] Show - 12 MULTi [1080p].mkv", "Show", None, ("multi",)),
            ("[101] Naruto - Title.mkv", "Naruto", None, ("naruto", "title")),
        ],
    )
    def test_a_count_a_word_follows_is_kept_as_the_titled_count_with_the_words_behind_it(
        self, name: str, series: str, count: int | None, words: tuple[str, ...]
    ) -> None:
        # The placer reads the words both ways: behind the count, and the whole name as a title.
        read = read_names([name], TitleGround.of(EntryNames(series)))[name]

        assert (read.titled_count, read.title_words) == (count, words)

    def test_the_titles_are_matched_longest_first_with_and_without_their_bracket(self) -> None:
        ground = TitleGround.of(EntryNames("Hunter x Hunter (2011)", ("Hunter x Hunter", "HUNTER×HUNTER (2011)")))

        assert ground.titles == (
            ("hunter", "x", "hunter", "2011"),
            ("hunter", "x", "hunter"),
            ("hunter", "hunter", "2011"),
            ("hunter", "hunter"),
        )
        assert TitleGround.of(EntryNames()).titles == ()
        # words that only count a season are no title: "2nd Season" left of its bracket would open any "2.01"
        assert TitleGround.of(EntryNames("[Oshi no Ko]", ("[Oshi no Ko] 2nd Season",))).titles == (
            ("oshi", "no", "ko", "2"),
            ("oshi", "no", "ko"),
        )
        assert _read("2.01 Tokyo Blade.mkv", "[Oshi no Ko]", ("[Oshi no Ko] 2nd Season",)) == (None, ())
