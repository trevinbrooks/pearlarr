# pyright: strict
"""Tests for what the notifier posts: the wait summary, the wait webhook's report, the grab embed, and a failed post.

Discord and the wait webhook are served over respx, so each test reads what the POST carried. A failed post warns
and is reported unsent, never raising into the grab or the run. A notification that raises past the notifier is
shown in `test_arr_run_finish`, and the payload's own shape and Discord's limits in `test_discord`.
"""

import json
import time
from collections.abc import Sequence
from typing import NotRequired, TypedDict, cast

import httpx
import pytest
import respx
from seadex import EntryRecord, Tag, Tracker

from pearlarr.config import Arr
from pearlarr.discord import COLOR_DEFERRED, COLOR_FAILED, COLOR_GRAB, COLOR_SUCCESS, DiscordEmbed, EmbedField
from pearlarr.notify import GrabNotice, Notifier
from pearlarr.output import Severity
from pearlarr.releases import SeadexDict
from pearlarr.torrent_service import AddStatus, GrabResult
from pearlarr.wait_types import Outcome
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

from .builders import (
    files_left_for_hand_import,
    make_entry_record,
    make_release,
    order_only_hand_import_files,
    rg_group,
)
from .fakes import FakeClock, diagnostic_messages, install_recording_hub
from .http_mock import requests_sent

_DISCORD = "https://discord.example/api/webhooks/1/secret-token"
"""The Discord webhook the notifier posts to. Its path is the credential, so no warning may show it."""

_HOOK = "https://hook.example"
"""The wait webhook."""


@pytest.fixture
def discord(respx_mock: respx.MockRouter) -> respx.Route:
    """Discord's webhook, answering each POST with the 204 it sends for a delivered message."""

    return respx_mock.post(_DISCORD).respond(204)


@pytest.fixture
def hook(respx_mock: respx.MockRouter) -> respx.Route:
    """The wait webhook, answering each POST with a 200."""

    return respx_mock.post(_HOOK).respond(200)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """A fake clock in place of `time.monotonic` and `time.sleep`.

    The notifier paces its pushes and waits out a 429 on `time` itself, with no clock handed in.
    """

    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake.monotonic)
    monkeypatch.setattr(time, "sleep", fake.sleep)
    return fake


class _SentField(TypedDict):
    """One field row as Discord gets it."""

    name: str
    value: str
    inline: bool


class _SentEmbed(TypedDict):
    """One embed as Discord gets it, as far as these tests read it."""

    author: dict[str, str]
    title: str
    color: int
    fields: list[_SentField]
    description: NotRequired[str]
    url: NotRequired[str]
    thumbnail: NotRequired[dict[str, str]]
    image: NotRequired[dict[str, str]]


def _embed_posted(discord: respx.Route) -> DiscordEmbed:
    """The embed of the one POST Discord got, read back into the form the notifier builds it in.

    A key the payload leaves out reads back as unset.
    """

    [request] = requests_sent(discord.calls)
    [sent] = cast("list[_SentEmbed]", json.loads(request.content)["embeds"])
    return DiscordEmbed(
        author_name=sent["author"]["name"],
        author_icon_url=sent["author"].get("icon_url"),
        title=sent["title"],
        color=sent["color"],
        url=sent.get("url"),
        description=sent.get("description", ""),
        fields=tuple(EmbedField(field["name"], field["value"], field["inline"]) for field in sent["fields"]),
        thumb_url=sent.get("thumbnail", {}).get("url"),
        image_url=sent.get("image", {}).get("url"),
    )


def _notifier() -> Notifier:
    """A notifier posting to Discord alone."""

    return Notifier(discord_url=_DISCORD, web=httpx.Client())


def _result() -> WaitResult:
    return WaitResult(
        (
            WaitOutcomeRow("Frieren", Outcome.IMPORTED),
            WaitOutcomeRow("Apothecary Diaries", Outcome.IMPORTED),
            WaitOutcomeRow("Spy x Family", Outcome.DOWNLOAD_TIMED_OUT),
            WaitOutcomeRow("Bleach TYBW", Outcome.DOWNLOAD_ERRORED),
        ),
        elapsed_s=4264,
    )


def test_push_wait_summary_no_url_is_noop() -> None:
    notifier = Notifier(discord_url=None, webhook_url=None, web=httpx.Client())

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is False


def test_push_wait_summary_empty_result_is_noop() -> None:
    notifier = Notifier(
        discord_url="https://discord",
        webhook_url="https://hook",
        web=httpx.Client(),
    )

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=WaitResult((), 0.0)) is False


def test_push_wait_summary_posts_to_webhook(hook: respx.Route) -> None:
    notifier = Notifier(discord_url=None, webhook_url=_HOOK, web=httpx.Client())

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is True
    assert json.loads(hook.calls.last.request.content) == {
        "arr": "sonarr",
        "imported": 2,
        "left": 1,
        "failed": 1,
        "elapsed_s": 4264,
        "rows": [
            {"label": "Frieren", "outcome": "IMPORTED", "word": "imported"},
            {"label": "Apothecary Diaries", "outcome": "IMPORTED", "word": "imported"},
            {"label": "Spy x Family", "outcome": "DOWNLOAD_TIMED_OUT", "word": "timed out"},
            {"label": "Bleach TYBW", "outcome": "DOWNLOAD_ERRORED", "word": "errored"},
        ],
    }


def test_webhook_rows_name_every_outcome(hook: respx.Route) -> None:
    notifier = Notifier(discord_url=None, webhook_url=_HOOK, web=httpx.Client())
    result = WaitResult(tuple(WaitOutcomeRow("Demo", outcome) for outcome in Outcome), elapsed_s=1)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    payload = cast("dict[str, list[dict[str, str]]]", json.loads(hook.calls.last.request.content))
    assert [row["outcome"] for row in payload["rows"]] == [
        "IMPORTED",
        "GONE",
        "DOWNLOAD_ERRORED",
        "DOWNLOAD_TIMED_OUT",
        "NO_CONTENT_PATH",
        "STILL_IMPORTING",
        "SONARR_BUSY",
        "NOT_READY",
        "NOTHING_PLACED",
        "PARTIAL",
        "ATTEMPT_FAILED",
        "QBITTORRENT_UNREACHABLE",
        "IMPORT_IN_PROGRESS",
        "AWAITING_IMPORT",
        "STILL_DOWNLOADING",
    ]


def test_push_wait_summary_webhook_failure_warns_and_returns_false(hook: respx.Route) -> None:
    # The exception is never put in the warning: its str holds the webhook URL, which is the credential. The config
    # key points at the fix.
    hook.mock(side_effect=httpx.ConnectError)
    notifier = Notifier(discord_url=None, webhook_url=_HOOK, web=httpx.Client())
    recording = install_recording_hub()

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is False
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Wait-report webhook POST failed (ConnectError) - check notifications.wait_webhook_url"
    ]


def test_push_wait_summary_goes_to_both_discord_and_the_webhook(discord: respx.Route, hook: respx.Route) -> None:
    notifier = Notifier(discord_url=_DISCORD, webhook_url=_HOOK, web=httpx.Client())

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is True
    assert (discord.call_count, hook.call_count) == (1, 1)


def test_push_wait_summary_builds_discord_embed(discord: respx.Route) -> None:
    assert _notifier().push_wait_summary(arr=Arr.RADARR, result=_result()) is True

    embed = _embed_posted(discord)
    assert embed.title == "Radarr wait complete"
    assert embed.description == "2 imported · 1 left · 1 failed · 1h 11m 04s"
    # Imported rows list the name alone. Left and failed rows carry the outcome's detail.
    assert embed.fields == (
        EmbedField(name="Imported (2)", value="Frieren\nApothecary Diaries"),
        EmbedField(name="Left for the next run (1)", value="Spy x Family - download timed out; left for the next run"),
        EmbedField(name="Failed (1)", value="Bleach TYBW - download errored; left for the next run"),
    )


@pytest.mark.parametrize(
    ("outcomes", "color"),
    [
        pytest.param((Outcome.IMPORTED,), COLOR_SUCCESS, id="all-imported-green"),
        pytest.param((Outcome.STILL_IMPORTING,), COLOR_DEFERRED, id="one-left-orange"),
        pytest.param((Outcome.STILL_IMPORTING, Outcome.DOWNLOAD_ERRORED), COLOR_FAILED, id="one-failed-red"),
    ],
)
def test_push_wait_summary_is_colored_by_its_worst_outcome(
    discord: respx.Route, outcomes: tuple[Outcome, ...], color: int
) -> None:
    result = WaitResult(tuple(WaitOutcomeRow("Frieren", outcome) for outcome in outcomes), elapsed_s=60)

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert _embed_posted(discord).color == color


def test_push_wait_summary_says_a_gone_download_has_its_title_checked_again(discord: respx.Route) -> None:
    row = WaitOutcomeRow("Demo Batch", Outcome.GONE)

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult((row,), elapsed_s=15)) is True
    [field] = _embed_posted(discord).fields
    assert field.value == "Demo Batch - gone from qBittorrent; title checked again next run"


@pytest.mark.parametrize(
    ("outcome", "value"),
    [
        pytest.param(
            Outcome.NOTHING_PLACED,
            "Demo Batch - no file was placed on an episode; import by hand in Sonarr\n`Part_1.mkv`\n`Part_2.mkv`",
            id="nothing-placed",
        ),
        pytest.param(
            Outcome.PARTIAL,
            "Demo Batch - imported in part; files left for hand import in Sonarr\n`Part_1.mkv`\n`Part_2.mkv`",
            id="partial",
        ),
    ],
)
def test_push_wait_summary_lists_the_files_left_for_hand_import_under_their_row(
    discord: respx.Route, outcome: Outcome, value: str
) -> None:
    # One backticked line per name: Discord strips leading whitespace and reads `_` as markdown.
    left = files_left_for_hand_import("Part_1.mkv", "Part_2.mkv")
    row = WaitOutcomeRow("Demo Batch", outcome, left_for_hand_import=left)

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult((row,), elapsed_s=15)) is True
    [field] = _embed_posted(discord).fields
    assert field == EmbedField(name="Left for the next run (1)", value=value)


def test_push_wait_summary_says_why_order_only_files_are_left_for_hand_import(discord: respx.Route) -> None:
    row = WaitOutcomeRow(
        "Show Ending · grp · S04 E01-E12", Outcome.PARTIAL, left_for_hand_import=order_only_hand_import_files()
    )

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult((row,), elapsed_s=15)) is True
    [field] = _embed_posted(discord).fields
    assert field.value == (
        "Show Ending · grp · S04 E01-E12 - S04 E01 is still missing in Sonarr.\n"
        "This torrent is numbered by the DVD order, and 2 of its files are DVD episodes "
        "that aren't in Sonarr's list, so Pearlarr can't tell where they belong:\n"
        "DVD S07E01: `[grp] Show Ending S1 - 01 [ABCD1234].mkv`\n"
        "DVD S07E02: `[grp] Show Ending S1 - 02 [ABCD1234].mkv`"
    )


def _nothing_placed(label: str, name: str) -> WaitOutcomeRow:
    """A one-file torrent of `label` whose file no episode took, so it is left for a hand import."""

    return WaitOutcomeRow(label, Outcome.NOTHING_PLACED, left_for_hand_import=files_left_for_hand_import(name))


def test_push_wait_summary_says_a_releases_torrents_once_with_their_files_in_order(discord: respx.Route) -> None:
    # A release of a torrent per episode, every one left for a hand import, in the order the downloads finished.
    finished = (
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 05 [1080p HEVC AAC][F94C53E6].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 11 [1080p HEVC AAC][E52DF113].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 01 [1080p HEVC AAC][DC4D7B5D].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 09 [1080p HEVC AAC][1395FE2A].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 03 [1080p HEVC AAC][3F92B6D1].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 10 [1080p HEVC AAC][A7C46E90].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 02 [1080p HEVC AAC][8A41C0E7].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 07 [1080p HEVC AAC][E6A0247C].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 04 [1080p HEVC AAC][C07E5A28].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 08 [1080p HEVC AAC][29D8F31B].mkv",
        "[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 06 [1080p HEVC AAC][5B13D9AF].mkv",
    )
    rows = (
        WaitOutcomeRow("Shakugan no Shana · Doki · S01 E01-E24", Outcome.IMPORTED),
        *(_nothing_placed("Though I Am an Inept Villainess · Chihiro · S01 E01-E11", name) for name in finished),
    )

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult(rows, elapsed_s=872)) is True
    assert _embed_posted(discord).fields == (
        EmbedField("Imported (1)", "Shakugan no Shana · Doki · S01 E01-E24"),
        EmbedField(
            "Left for the next run (11)",
            "Though I Am an Inept Villainess · Chihiro · S01 E01-E11 · 11 torrents"
            " - no file was placed on an episode; import by hand in Sonarr\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 01 [1080p HEVC AAC][DC4D7B5D].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 02 [1080p HEVC AAC][8A41C0E7].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 03 [1080p HEVC AAC][3F92B6D1].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 04 [1080p HEVC AAC][C07E5A28].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 05 [1080p HEVC AAC][F94C53E6].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 06 [1080p HEVC AAC][5B13D9AF].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 07 [1080p HEVC AAC][E6A0247C].mkv`",
        ),
        # The list runs on under no name, so Discord shows no header between the two halves.
        EmbedField(
            "",
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 08 [1080p HEVC AAC][29D8F31B].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 09 [1080p HEVC AAC][1395FE2A].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 10 [1080p HEVC AAC][A7C46E90].mkv`\n"
            "`[Chihiro] Futsutsuka na Akujo dewa Gozaimasu ga- Suuguu Chouso Torikae Den - 11 [1080p HEVC AAC][E52DF113].mkv`",
        ),
    )


def test_push_wait_summary_says_a_releases_torrents_apart_when_their_outcomes_differ(discord: respx.Route) -> None:
    # The entry of two stands where the first of them did, ahead of the torrent that timed out.
    label = "Frieren · SubsPlease · S01 E01-E28"
    rows = (
        _nothing_placed(label, "[SubsPlease] Sousou no Frieren - 02 (1080p) [3E7D1A42].mkv"),
        WaitOutcomeRow(label, Outcome.DOWNLOAD_TIMED_OUT),
        _nothing_placed(label, "[SubsPlease] Sousou no Frieren - 01 (1080p) [C1F05B97].mkv"),
    )

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult(rows, elapsed_s=15)) is True
    [field] = _embed_posted(discord).fields
    assert field.value == (
        "Frieren · SubsPlease · S01 E01-E28 · 2 torrents - no file was placed on an episode; import by hand in Sonarr\n"
        "`[SubsPlease] Sousou no Frieren - 01 (1080p) [C1F05B97].mkv`\n"
        "`[SubsPlease] Sousou no Frieren - 02 (1080p) [3E7D1A42].mkv`\n"
        "Frieren · SubsPlease · S01 E01-E28 - download timed out; left for the next run"
    )


def _backlog(outcome: Outcome, numbers: range) -> tuple[WaitOutcomeRow, ...]:
    """A row with `outcome` for each numbered title, as a big wait's backlog has."""

    return tuple(WaitOutcomeRow(f"Show {n:03d} · SubsPlease · S01 E01-E12", outcome) for n in numbers)


@pytest.mark.parametrize(
    ("rows", "names", "last_lines"),
    [
        pytest.param(
            _backlog(Outcome.IMPORTED, range(1, 81)),
            ["Imported (80)", ""],
            ["Show 056 · SubsPlease · S01 E01-E12", "… +24 more"],
            id="between-torrents",
        ),
        pytest.param(
            (
                WaitOutcomeRow(
                    "Gintama · Okay-Subs · S01 E01-E49",
                    Outcome.NOTHING_PLACED,
                    left_for_hand_import=files_left_for_hand_import(
                        *(f"[Okay-Subs] Gintama - {n:02d} (BD 1080p HEVC FLAC 2.0).mkv" for n in range(1, 50))
                    ),
                ),
                WaitOutcomeRow("Dandadan · SubsPlease · S01 E01-E12", Outcome.DOWNLOAD_TIMED_OUT),
                WaitOutcomeRow("Frieren · SubsPlease · S01 E01-E28", Outcome.DOWNLOAD_TIMED_OUT),
            ),
            ["Left for the next run (3)", ""],
            # The second field has room for file 34 but not for the count after it, so the count takes its place.
            ["`[Okay-Subs] Gintama - 33 (BD 1080p HEVC FLAC 2.0).mkv`", "… +16 more files · +2 more"],
            id="in-a-torrents-files",
        ),
    ],
)
def test_push_wait_summary_ends_a_section_past_two_fields_on_a_whole_line_and_a_count_of_the_rest(
    discord: respx.Route, rows: tuple[WaitOutcomeRow, ...], names: list[str], last_lines: list[str]
) -> None:
    # A field holds 1024 characters.
    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult(rows, elapsed_s=15)) is True
    fields = _embed_posted(discord).fields
    assert [field.name for field in fields] == names
    assert fields[-1].value.splitlines()[-2:] == last_lines


def test_push_wait_summary_fits_three_long_sections_in_the_embeds_total(discord: respx.Route) -> None:
    # Two full fields for each of three sections pass the 6000 characters Discord allows an embed, and the payload
    # would drop the last field with its count. The last section has only the room the first two left.
    rows = (
        *_backlog(Outcome.IMPORTED, range(1, 81)),
        *_backlog(Outcome.DOWNLOAD_TIMED_OUT, range(101, 141)),
        *_backlog(Outcome.GONE, range(201, 241)),
    )

    assert _notifier().push_wait_summary(arr=Arr.SONARR, result=WaitResult(rows, elapsed_s=15)) is True
    fields = _embed_posted(discord).fields
    names = ["Imported (80)", "", "Left for the next run (40)", "", "Failed (40)", ""]
    assert [field.name for field in fields] == names
    assert fields[-1].value.splitlines()[-1] == "… +20 more"


def _notice(
    *,
    arr: Arr = Arr.SONARR,
    arr_title: str = "Show",
    title_name: str = "Show",
    entry: EntryRecord | None = None,
    thumb_url: str | None = None,
    banner_url: str | None = None,
    replaced_groups: tuple[str, ...] = (),
    seadex_dict: SeadexDict | None = None,
    results: Sequence[GrabResult] = (),
    failed_groups: frozenset[str] = frozenset(),
    coverage: str = "",
) -> GrabNotice:
    """One grab notice with everything defaulted to the quiet minimum."""

    return GrabNotice(
        arr=arr,
        arr_title=arr_title,
        title_name=title_name,
        entry=entry if entry is not None else make_entry_record(),
        thumb_url=thumb_url,
        banner_url=banner_url,
        replaced_groups=replaced_groups,
        seadex_dict=seadex_dict or {},
        results=results,
        failed_groups=failed_groups,
        coverage=coverage,
    )


def _post_grab(discord: respx.Route, notice: GrabNotice) -> DiscordEmbed:
    """Push `notice` as a grab and read back the embed Discord got."""

    assert _notifier().push_grab(notice) is True
    return _embed_posted(discord)


def test_push_grab_builds_linked_embed(discord: respx.Route) -> None:
    embed = _post_grab(
        discord,
        _notice(
            arr_title="Sousou no Frieren",
            title_name="Frieren: Beyond Journey's End",
            entry=make_entry_record(url="https://releases.moe/154587/"),
            thumb_url="https://img.anili.st/cover.png",
            banner_url="https://img.anili.st/banner.png",
        ),
    )

    # The author line names the event, with the arr's logo beside it. The
    # title is the AniList name linking to the entry page.
    assert embed.author_name == "Sonarr · SeaDex grab"
    assert embed.author_icon_url == "https://raw.githubusercontent.com/Sonarr/Sonarr/develop/Logo/512.png"
    assert embed.title == "Frieren: Beyond Journey's End"
    assert embed.url == "https://releases.moe/154587/"
    assert embed.color == COLOR_GRAB
    assert embed.thumb_url == "https://img.anili.st/cover.png"
    assert embed.image_url == "https://img.anili.st/banner.png"
    # A genuinely different Arr series name becomes the muted subtext byline, going in
    # the trailing nameless notes field (an empty field name renders as no header at all).
    assert embed.description == ""
    assert embed.fields == (EmbedField(name="", value="-# Sousou no Frieren"),)


def test_push_grab_radarr_wears_the_radarr_logo(discord: respx.Route) -> None:
    embed = _post_grab(discord, _notice(arr=Arr.RADARR))

    assert embed.author_name == "Radarr · SeaDex grab"
    assert embed.author_icon_url == "https://raw.githubusercontent.com/Radarr/Radarr/develop/Logo/512.png"


class TestGrabNotes:
    """The Notes field stack: subtitle dedupe, notes blockquote, comparisons, caveat."""

    def _notes(self, discord: respx.Route, notice: GrabNotice) -> str:
        fields = _post_grab(discord, notice).fields
        return fields[-1].value if fields and fields[-1].name == "" else ""

    def test_near_duplicate_titles_render_no_subtitle(self, discord: respx.Route) -> None:
        # Apostrophe style and case are not information: Sonarr's ASCII name
        # vs AniList's typographic one must not produce a duplicate byline.
        notes = self._notes(
            discord,
            _notice(arr_title="frieren: beyond journey's end", title_name="Frieren: Beyond Journey’s End"),
        )

        assert notes == ""

    def test_season_suffix_title_renders_no_subtitle(self, discord: respx.Route) -> None:
        # The usual per-season AniList title merely extends the series name,
        # so a byline would repeat the embed title directly above the notes.
        notes = self._notes(
            discord,
            _notice(arr_title="Frieren: Beyond Journey's End", title_name="Frieren: Beyond Journey's End Season 2"),
        )

        assert notes == ""

    def test_shared_prefix_short_of_a_word_keeps_the_subtitle(self, discord: respx.Route) -> None:
        # A prefix that splits a word is not a season suffix: the Arr series name
        # names a different series and stays informative.
        notes = self._notes(discord, _notice(arr_title="Show", title_name="Showtime Season 1"))

        assert notes == "-# Show"

    def test_subtitle_escapes_markdown(self, discord: respx.Route) -> None:
        notes = self._notes(discord, _notice(arr_title="K-ON! *Special*"))

        assert notes == "-# K-ON! \\*Special\\*"

    def test_notes_render_as_one_blockquote(self, discord: respx.Route) -> None:
        # Blank lines drop so the notes stay one quote block.
        notes = self._notes(
            discord,
            _notice(entry=make_entry_record(notes="PMR is the JPN BD Remux\n\nLostYears has the dub")),
        )

        assert notes == "> PMR is the JPN BD Remux\n> LostYears has the dub"

    def test_long_notes_truncate_on_a_word_boundary(self, discord: respx.Route) -> None:
        notes = self._notes(
            discord,
            _notice(entry=make_entry_record(notes="alpha " * 80 + "OMEGA")),
        )

        assert notes.endswith("alpha …")
        assert "OMEGA" not in notes

    def test_cut_landing_on_whitespace_keeps_the_whole_word(self, discord: respx.Route) -> None:
        # 400 chars of complete 5-char words: the cut lands on the separator, so
        # no word is sacrificed.
        notes = self._notes(
            discord,
            _notice(entry=make_entry_record(notes="word " * 81)),
        )

        assert notes == "> " + " ".join(["word"] * 80) + " …"

    @pytest.mark.parametrize(
        ("comparisons", "links"),
        [
            pytest.param(("https://slow.pics/c/one",), "[Comparison](https://slow.pics/c/one)", id="one-unnumbered"),
            pytest.param(
                ("https://slow.pics/c/1", "https://slow.pics/c/2"),
                "[Comparison 1](https://slow.pics/c/1) · [Comparison 2](https://slow.pics/c/2)",
                id="several-numbered",
            ),
            # The lib's lax parse can yield an empty segment, sorted first. It gets no broken link and no number.
            pytest.param(
                ("", "https://slow.pics/c/only"), "[Comparison](https://slow.pics/c/only)", id="blank-dropped"
            ),
        ],
    )
    def test_comparison_links(self, discord: respx.Route, comparisons: tuple[str, ...], links: str) -> None:
        notes = self._notes(discord, _notice(entry=make_entry_record(comparisons=comparisons)))

        assert notes == links

    def test_incomplete_entry_renders_caveat(self, discord: respx.Route) -> None:
        notes = self._notes(
            discord,
            _notice(entry=make_entry_record(notes="Best available", is_incomplete=True)),
        )

        # Pieces stack line by line, no blank spacers.
        assert notes == "> Best available\n*Entry marked incomplete on SeaDex*"


class TestGrabLayout:
    """The grab layout: single-group grabs go in the description, several stack fields."""

    def test_single_group_goes_in_the_description(self, discord: respx.Route) -> None:
        seadex_dict = {
            "PMR": rg_group(
                {
                    "https://nyaa.si/view/1": make_release(
                        url="https://nyaa.si/view/1",
                        tracker=Tracker.NYAA,
                        download=True,
                        size=[536870912, 536870912],
                        files=["e1.mkv", "e2.mkv"],
                        is_dual_audio=True,
                        is_fallback=True,
                        upgrade=True,
                    ),
                },
                tags=frozenset({Tag.VFR, Tag.HDR}),
            ),
        }

        embed = _post_grab(
            discord,
            _notice(
                replaced_groups=("Erai_raws",),
                seadex_dict=seadex_dict,
                results=[GrabResult(AddStatus.ADDED, "PMR release", "PMR")],
                coverage="S01 E01-E12",
            ),
        )

        # The single group's grab hoists into the description (bold label, group as a code
        # span, then the release line + tags subtext). Episodes and the replaced
        # groups pair up as side-by-side inline fields (the caller passes them clean).
        assert embed.description == (
            "**Grabbed · `PMR`**\n"
            "[Nyaa](https://nyaa.si/view/1) · 1.0 GB · 2 files · dual audio · fallback · upgrade\n"
            "-# HDR · VFR"
        )
        assert embed.fields == (
            EmbedField(name="Episodes", value="S01 E01-E12", inline=True),
            EmbedField(name="Replacing", value="Erai_raws", inline=True),
        )

    def test_multiple_groups_keep_the_field_stack(self, discord: respx.Route) -> None:
        # "Grabbed" only when the client actually added something. A release already
        # mid-download says so. A contained add failure (no result) reads
        # "Failed". A group whose releases were never attempted reads "Skipped".
        seadex_dict = {
            srg: rg_group({f"https://nyaa.si/view/{i}": make_release(url=f"https://nyaa.si/view/{i}", download=True)})
            for i, srg in enumerate(("Fresh", "InFlight", "Errored", "Held"))
        }

        embed = _post_grab(
            discord,
            _notice(
                seadex_dict=seadex_dict,
                results=[
                    GrabResult(AddStatus.ADDED, None, "Fresh"),
                    GrabResult(AddStatus.ALREADY_IN_QBIT, None, "InFlight"),
                ],
                failed_groups=frozenset({"Errored"}),
                replaced_groups=("Erai_raws",),
            ),
        )

        # No description hoist with several groups, and "Replacing" appears
        # exactly once, after the group stack.
        assert embed.description == ""
        assert [f.name for f in embed.fields] == [
            "Grabbed · Fresh",
            "Already in qBittorrent · InFlight",
            "Failed · Errored",
            "Skipped · Held",
            "Replacing",
        ]

    def test_unflagged_group_contributes_nothing(self, discord: respx.Route) -> None:
        seadex_dict = {"Quiet": rg_group({"https://nyaa.si/view/9": make_release(url="https://nyaa.si/view/9")})}

        embed = _post_grab(discord, _notice(seadex_dict=seadex_dict))

        assert embed.description == ""
        assert embed.fields == ()

    @pytest.mark.parametrize(
        ("groups", "value"),
        [
            pytest.param(3, "Group00, Group01, Group02", id="3-all-listed"),
            pytest.param(12, "Group00, Group01, Group02, … +9 more", id="past-3-counted"),
        ],
    )
    def test_replacing_collapses_beyond_cap(self, discord: respx.Route, groups: int, value: str) -> None:
        embed = _post_grab(discord, _notice(replaced_groups=tuple(f"Group{i:02d}" for i in range(groups))))

        [replacing] = embed.fields
        assert replacing.value == value


@pytest.mark.usefixtures("discord")
def test_pushes_are_paced_only_within_a_burst(clock: FakeClock) -> None:
    # A single or final push never pays a trailing sleep. Only a burst's later pushes wait out the rest of the 1s
    # spacing.
    notifier = _notifier()

    assert notifier.push_grab(_notice()) is True
    assert clock.sleeps == []

    clock.tick(0.25)  # a burst: the next push comes 0.25s later
    assert notifier.push_grab(_notice()) is True
    assert clock.sleeps == [0.75]  # topped up to the 1s spacing

    clock.tick(5.0)  # a slow follow-up needs no pacing
    assert notifier.push_grab(_notice()) is True
    assert clock.sleeps == [0.75]


def _limited(retry_after: str) -> httpx.Response:
    """Discord's 429 for a throttled webhook, asking for a wait of `retry_after`."""

    return httpx.Response(429, headers={"Retry-After": retry_after})


def _push_grab(notifier: Notifier) -> bool:
    """One grab push with the quiet minimum of embed data, for the tests of what a failed post does."""

    return notifier.push_grab(_notice())


class TestAFailedDiscordPost:
    """A post Discord refuses or never gets warns and is reported unsent, never raising into the grab or the run.

    A 429 is Discord throttling a healthy webhook: a Retry-After of up to 5s is waited out and the post tried once
    more, and a notification that still can't go is dropped with a warning that doesn't blame the config. A 4xx
    otherwise means the webhook itself is bad, so one warning turns Discord off for the rest of the run. Anything else
    warns on each push.
    """

    @pytest.mark.parametrize(
        ("limited", "waited"),
        [
            pytest.param(httpx.Response(429), 1.0, id="no-header"),
            pytest.param(_limited("5"), 5.0, id="at-the-5s-cap"),
            pytest.param(_limited("0.5"), 0.5, id="fractional"),
            # The header's other form, a date, isn't read, so the default is waited.
            pytest.param(_limited("Fri, 31 Dec 1999 23:59:59 GMT"), 1.0, id="http-date"),
        ],
    )
    def test_a_short_rate_limit_is_waited_out_and_delivered(
        self, discord: respx.Route, clock: FakeClock, limited: httpx.Response, waited: float
    ) -> None:
        discord.mock(side_effect=[limited, httpx.Response(204)])
        recording = install_recording_hub()

        assert _push_grab(_notifier()) is True

        assert discord.call_count == 2
        assert clock.sleeps == [waited]
        assert diagnostic_messages(recording, Severity.WARNING) == []

    @pytest.mark.parametrize(
        ("answers", "posts", "waited"),
        [
            pytest.param([_limited("1"), _limited("1")], 2, [1.0], id="limited-again-on-the-retry"),
            pytest.param([_limited("6")], 1, [], id="retry-after-past-the-cap"),
        ],
    )
    def test_a_rate_limit_that_cannot_be_waited_out_drops_the_notification(
        self, discord: respx.Route, clock: FakeClock, answers: list[httpx.Response], posts: int, waited: list[float]
    ) -> None:
        discord.mock(side_effect=answers)
        recording = install_recording_hub()

        assert _push_grab(_notifier()) is False

        assert discord.call_count == posts
        assert clock.sleeps == waited
        assert diagnostic_messages(recording, Severity.WARNING) == [
            "Discord notification failed (HTTP 429) - rate limited by Discord - this notification was dropped",
        ]

    @pytest.mark.parametrize(
        ("answer", "warning"),
        [
            pytest.param(
                httpx.ConnectError,
                "Discord notification failed (ConnectError) - check notifications.discord_url",
                id="unreachable",
            ),
            pytest.param(
                httpx.Response(500),
                "Discord notification failed (HTTP 500) - check notifications.discord_url",
                id="server-error",
            ),
            pytest.param(
                _limited("6"),
                "Discord notification failed (HTTP 429) - rate limited by Discord - this notification was dropped",
                id="rate-limited",
            ),
        ],
    )
    @pytest.mark.usefixtures("clock")
    def test_a_failure_that_may_pass_warns_on_each_push_and_keeps_discord_on(
        self, discord: respx.Route, answer: httpx.Response | type[httpx.ConnectError], warning: str
    ) -> None:
        discord.mock(side_effect=[answer, answer])
        notifier = _notifier()
        recording = install_recording_hub()

        assert (_push_grab(notifier), _push_grab(notifier)) == (False, False)

        assert discord.call_count == 2
        assert diagnostic_messages(recording, Severity.WARNING) == [warning, warning]

    def test_a_deleted_webhook_turns_discord_off_for_the_run(self, discord: respx.Route) -> None:
        # Discord answers a deleted webhook with a 404. Warning on every grab after it would say nothing new.
        discord.respond(404)
        notifier = _notifier()
        recording = install_recording_hub()

        assert (_push_grab(notifier), _push_grab(notifier)) == (False, False)

        assert discord.call_count == 1
        assert diagnostic_messages(recording, Severity.WARNING) == [
            (
                "Discord notification failed (HTTP 404) - disabling Discord notifications for this run - "
                "check notifications.discord_url"
            ),
        ]
