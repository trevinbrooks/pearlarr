# pyright: strict
"""Tests for the Notifier pushes: the grab embed and the wait-complete summary.

``Notifier.push_wait_summary`` posts the wait-pass outcome (colored by its
worst outcome class) to Discord and/or a generic webhook; ``push_grab`` posts
the per-title grab embed. Both are best-effort, so these pin the happy paths,
the no-url no-op, and the containment invariant: a notification failure warns
and returns False, it must never abort a grab or the end-of-run cache save.
"""

import json
import time
from collections.abc import Sequence
from typing import cast

import httpx
import pytest
import respx
from seadex import EntryRecord, Tag, Tracker

from pearlarr.modules import notify
from pearlarr.modules.config import Arr
from pearlarr.modules.discord import (
    COLOR_DEFERRED,
    COLOR_FAILED,
    COLOR_GRAB,
    COLOR_SUCCESS,
    DiscordEmbed,
    EmbedField,
)
from pearlarr.modules.manual_import import Outcome
from pearlarr.modules.notify import GrabNotice, Notifier
from pearlarr.modules.output import Severity
from pearlarr.modules.seadex_types import SeadexDict
from pearlarr.modules.torrents import AddOutcome, ReleaseOutcome
from pearlarr.modules.wait_view import WaitOutcomeRow, WaitResult

from .builders import SEP, make_entry_record, rg_group, url_item
from .fakes import diagnostic_messages, install_recording_hub


@pytest.fixture
def pushes(monkeypatch: pytest.MonkeyPatch) -> list[DiscordEmbed]:
    """Route ``notify.discord_push`` into a recording list (no network)."""

    recorded: list[DiscordEmbed] = []

    def fake_discord_push(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
        del url, client
        recorded.append(embed)

    monkeypatch.setattr(notify, "discord_push", fake_discord_push)
    return recorded


def _http_error(status: int, *, retry_after: str | None = None) -> httpx.HTTPStatusError:
    """An ``HTTPStatusError`` carrying a response, as ``raise_for_status`` raises it.

    The url (in the message and on the request, as httpx builds both) stands in
    for the webhook credential: the containment tests assert it never reaches a
    warning message. ``retry_after`` sets the Retry-After header the 429
    handling parses.
    """

    url = "https://discord.example/api/webhooks/1/secret-token"
    request = httpx.Request("POST", url)
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError(f"error '{status}' for url '{url}'", request=request, response=response)


class _SequencedPush:
    """``discord_push`` stand-in: raises the queued errors in order, then succeeds."""

    def __init__(self, *errors: httpx.HTTPStatusError) -> None:
        self.errors = list(errors)
        self.posts: list[str] = []

    def __call__(self, *, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
        del embed, client
        self.posts.append(url)
        if self.errors:
            raise self.errors.pop(0)


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Route ``time.sleep`` (pacing + the 429 Retry-After) into a recording list."""

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


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


@respx.mock
def test_push_wait_summary_posts_to_webhook() -> None:
    route = respx.post("https://hook.example").respond(json={})
    notifier = Notifier(
        discord_url=None,
        webhook_url="https://hook.example",
        web=httpx.Client(),
    )

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=_result()) is True
    payload = cast("dict[str, object]", json.loads(route.calls.last.request.content))
    assert payload["imported"] == 2
    assert payload["failed"] == 1


@respx.mock
def test_push_wait_summary_webhook_failure_warns_and_returns_false() -> None:
    # The generic-webhook twin of the Discord containment: a request failure is
    # warned about and swallowed, never propagated to the finalize tail.
    respx.post("https://hook.example").mock(side_effect=httpx.ConnectError("webhook down"))
    notifier = Notifier(discord_url=None, webhook_url="https://hook.example", web=httpx.Client())

    recording = install_recording_hub()
    posted = notifier.push_wait_summary(arr=Arr.SONARR, result=_result())  # must not raise

    assert posted is False
    [warning] = diagnostic_messages(recording, Severity.WARNING)
    # The exception is never interpolated: its str embeds the webhook URL,
    # which IS the credential. The config key points the user at the fix.
    assert warning == "Wait-report webhook POST failed (ConnectError) - check notifications.wait_webhook_url"
    assert "hook.example" not in warning


def test_push_wait_summary_builds_discord_embed(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    assert notifier.push_wait_summary(arr=Arr.RADARR, result=_result()) is True
    embed = pushes[0]
    assert embed.title == "Radarr wait complete"
    assert f"2 imported{SEP}1 left{SEP}1 failed" in embed.description
    names = [field.name for field in embed.fields]
    assert names == ["Imported (2)", "Left for a later run (1)", "Failed (1)"]
    assert "Frieren" in embed.fields[0].value
    # Deferred/failed rows carry the outcome detail; a failure colors the embed red.
    assert embed.fields[2].value == "Bleach TYBW — download errored; left pending"
    assert embed.color == COLOR_FAILED


def test_push_wait_summary_all_imported_is_green(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert pushes[0].color == COLOR_SUCCESS


def test_push_wait_summary_deferred_only_is_orange(pushes: list[DiscordEmbed]) -> None:
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.STILL_IMPORTING),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert pushes[0].color == COLOR_DEFERRED


def test_pushes_are_paced_only_within_a_burst(
    monkeypatch: pytest.MonkeyPatch,
    pushes: list[DiscordEmbed],
) -> None:
    # Pacing lives in the Notifier (discord_push is a pure POST), so a single or
    # final push never pays a trailing sleep; only a burst's later pushes wait
    # out the remainder of the 1s spacing.
    del pushes  # the pushes fixture supplies the no-network discord_push
    sleeps: list[float] = []
    now = {"t": 100.0}

    def fake_monotonic() -> float:
        return now["t"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=60)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert sleeps == []  # the first (and possibly only) push never sleeps

    now["t"] += 0.25  # a burst: the next push arrives 0.25s later
    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert sleeps == [0.75]  # topped up to the 1s spacing

    now["t"] += 5.0  # a slow follow-up needs no pacing
    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    assert sleeps == [0.75]


def _notice(
    *,
    arr: Arr = Arr.SONARR,
    arr_title: str = "Show",
    al_title: str = "Show",
    entry: EntryRecord | None = None,
    thumb_url: str | None = None,
    banner_url: str | None = None,
    release_group: list[str | None] | None = None,
    seadex_dict: SeadexDict | None = None,
    results: Sequence[ReleaseOutcome] = (),
    failed_groups: frozenset[str] = frozenset(),
    coverage: str = "",
) -> GrabNotice:
    """One grab notice with everything defaulted to the quiet minimum."""

    return GrabNotice(
        arr=arr,
        arr_title=arr_title,
        al_title=al_title,
        entry=entry if entry is not None else make_entry_record(),
        thumb_url=thumb_url,
        banner_url=banner_url,
        release_group=release_group,
        seadex_dict=seadex_dict or {},
        results=results,
        failed_groups=failed_groups,
        coverage=coverage,
    )


def _grab_notifier() -> Notifier:
    return Notifier(discord_url="https://discord.example", web=httpx.Client())


def test_push_grab_builds_linked_embed(pushes: list[DiscordEmbed]) -> None:
    posted = _grab_notifier().push_grab(
        _notice(
            arr_title="Sousou no Frieren",
            al_title="Frieren: Beyond Journey's End",
            entry=make_entry_record(url="https://releases.moe/154587/"),
            thumb_url="https://img.anili.st/cover.png",
            banner_url="https://img.anili.st/banner.png",
        ),
    )

    assert posted is True
    embed = pushes[0]
    # The author line names the event, with the arr's own logo beside it; the
    # title is the AniList title linking to the entry page.
    assert embed.author_name == "Sonarr · SeaDex grab"
    assert embed.author_icon_url == "https://raw.githubusercontent.com/Sonarr/Sonarr/develop/Logo/512.png"
    assert embed.title == "Frieren: Beyond Journey's End"
    assert embed.url == "https://releases.moe/154587/"
    assert embed.color == COLOR_GRAB
    assert embed.thumb_url == "https://img.anili.st/cover.png"
    assert embed.image_url == "https://img.anili.st/banner.png"
    # A genuinely different Arr title becomes the muted subtext byline, riding
    # the trailing nameless notes field (an empty field name renders as no
    # header at all - verified live against the webhook API).
    assert embed.description == ""
    assert embed.fields == (EmbedField(name="", value="-# Sousou no Frieren"),)


def test_push_grab_radarr_wears_the_radarr_logo(pushes: list[DiscordEmbed]) -> None:
    assert _grab_notifier().push_grab(_notice(arr=Arr.RADARR)) is True

    embed = pushes[0]
    assert embed.author_name == "Radarr · SeaDex grab"
    assert embed.author_icon_url == "https://raw.githubusercontent.com/Radarr/Radarr/develop/Logo/512.png"


class TestGrabNotes:
    """The Notes field stack: subtitle dedupe, notes blockquote, comparisons, caveat."""

    def _notes(self, pushes: list[DiscordEmbed], notice: GrabNotice) -> str:
        assert _grab_notifier().push_grab(notice) is True
        fields = pushes[-1].fields
        return fields[-1].value if fields and fields[-1].name == "" else ""

    def test_near_duplicate_titles_render_no_subtitle(self, pushes: list[DiscordEmbed]) -> None:
        # Apostrophe style and case are not information: Sonarr's ASCII title
        # vs AniList's typographic one must not produce a duplicate byline.
        notes = self._notes(
            pushes,
            _notice(arr_title="frieren: beyond journey's end", al_title="Frieren: Beyond Journey’s End"),
        )

        assert notes == ""

    def test_subtitle_escapes_markdown(self, pushes: list[DiscordEmbed]) -> None:
        notes = self._notes(pushes, _notice(arr_title="K-ON! *Special*", al_title="K-ON!"))

        assert notes == "-# K-ON! \\*Special\\*"

    def test_notes_render_as_one_blockquote(self, pushes: list[DiscordEmbed]) -> None:
        # Blank lines drop so the notes stay one quote block.
        notes = self._notes(
            pushes,
            _notice(entry=make_entry_record(notes="PMR is the JPN BD Remux\n\nLostYears has the dub")),
        )

        assert notes == "> PMR is the JPN BD Remux\n> LostYears has the dub"

    def test_long_notes_truncate_on_a_word_boundary(self, pushes: list[DiscordEmbed]) -> None:
        notes = self._notes(
            pushes,
            _notice(entry=make_entry_record(notes="alpha " * 80 + "OMEGA")),
        )

        assert notes.endswith("alpha …")
        assert "OMEGA" not in notes

    def test_cut_landing_on_whitespace_keeps_the_whole_word(self, pushes: list[DiscordEmbed]) -> None:
        # 400 chars of complete 5-char words: the cut lands ON the separator, so
        # no word is sacrificed.
        notes = self._notes(
            pushes,
            _notice(entry=make_entry_record(notes="word " * 81)),
        )

        assert notes == "> " + " ".join(["word"] * 80) + " …"

    def test_comparison_links(self, pushes: list[DiscordEmbed]) -> None:
        # A single link needs no number; several are numbered on one line.
        single = self._notes(
            pushes,
            _notice(entry=make_entry_record(comparisons=("https://slow.pics/c/one",))),
        )
        multi = self._notes(
            pushes,
            _notice(entry=make_entry_record(comparisons=("https://slow.pics/c/1", "https://slow.pics/c/2"))),
        )

        assert single == "[Comparison](https://slow.pics/c/one)"
        assert multi == "[Comparison 1](https://slow.pics/c/1) · [Comparison 2](https://slow.pics/c/2)"

    def test_blank_comparison_segments_are_dropped(self, pushes: list[DiscordEmbed]) -> None:
        # The lib's lax parse can yield an empty segment (sorted first); it must
        # not render a broken [Comparison N]() link - nor inflate the numbering.
        notes = self._notes(
            pushes,
            _notice(entry=make_entry_record(comparisons=("", "https://slow.pics/c/only"))),
        )

        assert notes == "[Comparison](https://slow.pics/c/only)"

    def test_incomplete_entry_renders_caveat(self, pushes: list[DiscordEmbed]) -> None:
        notes = self._notes(
            pushes,
            _notice(entry=make_entry_record(notes="Best available", is_incomplete=True)),
        )

        # Pieces stack line by line, no blank spacers.
        assert notes == "> Best available\n*Entry marked incomplete on SeaDex*"


class TestGrabLayout:
    """The pick layout: single-group grabs ride the description, several stack fields."""

    def _embed(self, pushes: list[DiscordEmbed], notice: GrabNotice) -> DiscordEmbed:
        assert _grab_notifier().push_grab(notice) is True
        return pushes[0]

    def test_single_group_rides_the_description(self, pushes: list[DiscordEmbed]) -> None:
        seadex_dict = {
            "PMR": rg_group(
                {
                    "https://nyaa.si/view/1": url_item(
                        url="https://nyaa.si/view/1",
                        tracker=Tracker.NYAA,
                        download=True,
                        size=[536870912, 536870912],
                        files=["e1.mkv", "e2.mkv"],
                        is_dual_audio=True,
                        is_fallback=True,
                        size_mismatch=True,
                    ),
                },
                tags=frozenset({Tag.VFR, Tag.HDR}),
            ),
        }

        embed = self._embed(
            pushes,
            _notice(
                release_group=["Erai_raws", None, ""],
                seadex_dict=seadex_dict,
                results=[ReleaseOutcome(AddOutcome.ADDED, "PMR release", "PMR")],
                coverage="S01 E01-E12",
            ),
        )

        # The lone pick hoists into the description (bold label, group as a code
        # span, then the release line + tags subtext); episodes and the replaced
        # groups pair up as side-by-side inline fields; empty/None groups drop.
        assert embed.description == (
            "**Grabbed · `PMR`**\n"
            "[Nyaa](https://nyaa.si/view/1) · 1.0 GB · 2 files · dual audio · fallback · upgrade\n"
            "-# HDR · VFR"
        )
        assert embed.fields == (
            EmbedField(name="Episodes", value="S01 E01-E12", inline=True),
            EmbedField(name="Replacing", value="Erai_raws", inline=True),
        )

    def test_multiple_groups_keep_the_field_stack(self, pushes: list[DiscordEmbed]) -> None:
        # "Grabbed" only when the client actually added something; a pick already
        # mid-download says so; a contained add failure (no outcome) reads
        # "Failed"; a group whose releases were never attempted reads "Skipped".
        seadex_dict = {
            srg: rg_group({f"https://nyaa.si/view/{i}": url_item(url=f"https://nyaa.si/view/{i}", download=True)})
            for i, srg in enumerate(("Fresh", "InFlight", "Errored", "Held"))
        }

        embed = self._embed(
            pushes,
            _notice(
                seadex_dict=seadex_dict,
                results=[
                    ReleaseOutcome(AddOutcome.ADDED, None, "Fresh"),
                    ReleaseOutcome(AddOutcome.ALREADY_ADDED, None, "InFlight"),
                ],
                failed_groups=frozenset({"Errored"}),
                release_group=["Erai_raws"],
            ),
        )

        # No description hoist with several groups - and "Replacing" appears
        # exactly once, after the group stack.
        assert embed.description == ""
        assert [f.name for f in embed.fields] == [
            "Grabbed · Fresh",
            "Already downloading · InFlight",
            "Failed · Errored",
            "Skipped · Held",
            "Replacing",
        ]

    def test_unflagged_group_contributes_nothing(self, pushes: list[DiscordEmbed]) -> None:
        seadex_dict = {"Quiet": rg_group({"https://nyaa.si/view/9": url_item(url="https://nyaa.si/view/9")})}

        embed = self._embed(pushes, _notice(seadex_dict=seadex_dict))

        assert embed.description == ""
        assert embed.fields == ()

    def test_replacing_collapses_beyond_cap(self, pushes: list[DiscordEmbed]) -> None:
        embed = self._embed(pushes, _notice(release_group=[f"Group{i:02d}" for i in range(12)]))

        [replacing] = embed.fields
        assert replacing.value == "Group00, Group01, Group02, … +9 more"


def _push_grab(notifier: Notifier) -> bool:
    """Drive one grab push with minimal embed data (the containment tests' focus)."""

    return notifier.push_grab(_notice())


def test_push_grab_discord_failure_warns_and_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # The invariant the engine relies on: a Discord failure is contained in
    # _push (warn, False) - it must never abort the grab that triggered it.
    def raising_discord_push(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
        del url, embed, client
        raise httpx.ConnectError("discord down")

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    recording = install_recording_hub()
    posted = _push_grab(notifier)  # must not raise

    assert posted is False
    [warning] = diagnostic_messages(recording, Severity.WARNING)
    # The exception type only - an httpx exception's str embeds the webhook
    # URL (the credential) - and the config key to check.
    assert warning == "Discord notification failed (ConnectError) - check notifications.discord_url"
    assert "discord.example" not in warning


def test_push_grab_4xx_disables_discord_for_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 4xx (e.g. a deleted webhook's 404) is permanent: one actionable warning,
    # then Discord pushes are disabled - no warn-per-grab retry storm.
    posts: list[str] = []

    def raising_discord_push(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
        del embed, client
        posts.append(url)
        raise _http_error(404)

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    recording = install_recording_hub()
    assert _push_grab(notifier) is False
    assert _push_grab(notifier) is False

    assert posts == ["https://discord.example"]  # the dead webhook is POSTed once
    assert notifier.enabled is False
    warnings = diagnostic_messages(recording, Severity.WARNING)
    assert warnings == [
        "Discord notification failed (HTTP 404) - disabling Discord notifications "
        "for this run; check notifications.discord_url",
    ]
    assert all("secret-token" not in message for message in warnings)


def test_push_grab_429_retry_delivers_after_default_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 429 without a Retry-After header sleeps the 1s default then retries
    # once: the notification is DELIVERED, so the caller sees True and nothing
    # warns - a rate-limited push that made it is not a failure.
    push = _SequencedPush(_http_error(429))
    monkeypatch.setattr(notify, "discord_push", push)
    sleeps = _record_sleeps(monkeypatch)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    recording = install_recording_hub()
    assert _push_grab(notifier) is True

    assert push.posts == ["https://discord.example"] * 2  # original + retry
    assert sleeps == [1.0]  # missing header -> the 1s default
    assert diagnostic_messages(recording, Severity.WARNING) == []


@pytest.mark.parametrize(
    ("header", "expected_delay"),
    [("3", 3.0), ("0.5", 0.5), ("soon", 1.0)],
)
def test_push_grab_429_honors_retry_after(
    header: str,
    expected_delay: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Discord's Retry-After (an int or float seconds string) is slept out
    # before the retry; an unparseable value falls back to the 1s default.
    push = _SequencedPush(_http_error(429, retry_after=header))
    monkeypatch.setattr(notify, "discord_push", push)
    sleeps = _record_sleeps(monkeypatch)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    assert _push_grab(notifier) is True
    assert sleeps == [expected_delay]


def test_push_grab_429_retry_exhausted_drops_and_stays_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 429 whose retry 429s again drops THIS notification with one truthful
    # warning - never an optimistic "later notifications" claim. The webhook is
    # healthy (throttled, not dead), so pushes stay enabled, the config is not
    # blamed, and a later push still attempts (and drops the same way).
    posts: list[str] = []

    def always_429(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
        del embed, client
        posts.append(url)
        raise _http_error(429, retry_after="0.1")

    monkeypatch.setattr(notify, "discord_push", always_429)
    _record_sleeps(monkeypatch)  # neutralize the retry delay + the 1s pacing
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    recording = install_recording_hub()
    assert _push_grab(notifier) is False
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Discord notification failed (HTTP 429) - rate limited by Discord; this notification was dropped",
    ]
    assert _push_grab(notifier) is False  # a later push still attempts

    assert notifier.enabled is True  # NOT disabled - later pushes still go out
    assert posts == ["https://discord.example"] * 4  # (original + retry) per push
    warnings = diagnostic_messages(recording, Severity.WARNING)
    assert len(warnings) == 2  # one truthful warning per dropped push
    assert all("discord_url" not in message for message in warnings)  # config isn't at fault
    assert all("secret-token" not in message for message in warnings)


def test_push_grab_429_long_retry_after_skips_the_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Retry-After above the 5s cap isn't worth blocking the run for: no
    # sleep, no retry - the push is dropped immediately with the same truthful
    # warning, and pushes stay enabled for the rest of the run.
    push = _SequencedPush(_http_error(429, retry_after="30"))
    monkeypatch.setattr(notify, "discord_push", push)
    sleeps = _record_sleeps(monkeypatch)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    recording = install_recording_hub()
    assert _push_grab(notifier) is False

    assert push.posts == ["https://discord.example"]  # no retry POST
    assert sleeps == []  # the 30s was NOT slept
    assert notifier.enabled is True
    assert diagnostic_messages(recording, Severity.WARNING) == [
        "Discord notification failed (HTTP 429) - rate limited by Discord; this notification was dropped",
    ]


def test_push_grab_5xx_stays_per_push(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 5xx is transient (Discord hiccup): keep trying - and warning - per push.
    def raising_discord_push(*, url: str, embed: DiscordEmbed, client: httpx.Client) -> None:
        del url, embed, client
        raise _http_error(500)

    def no_sleep(seconds: float) -> None:
        del seconds  # the second push would otherwise pay the real 1s pacing

    monkeypatch.setattr(notify, "discord_push", raising_discord_push)
    monkeypatch.setattr(time, "sleep", no_sleep)
    notifier = Notifier(discord_url="https://discord.example", web=httpx.Client())

    recording = install_recording_hub()
    assert _push_grab(notifier) is False
    assert _push_grab(notifier) is False

    assert notifier.enabled is True
    assert (
        diagnostic_messages(recording, Severity.WARNING)
        == ["Discord notification failed (HTTP 500) - check notifications.discord_url"] * 2
    )
