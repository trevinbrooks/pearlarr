# pyright: strict
"""Tests for the Discord embed boundary and the grab-embed fields.

``DiscordEmbed.to_payload`` is the single JSON-shaped seam - these pin the
omitted-when-unset optional keys and the clamping to Discord's hard limits -
and ``Notifier.build_fields`` shapes a grab's fields (markdown tracker links,
the one-line tag list, the release-group fallback).
"""

from typing import cast

from seadex import Tag, Tracker

from seadexarr.modules.config import Arr
from seadexarr.modules.discord import PROJECT_URL, DiscordEmbed, EmbedField
from seadexarr.modules.notify import Notifier

from .builders import make_logger, rg_group, url_item


def _notifier() -> Notifier:
    return Notifier(discord_url=None, webhook_url=None, logger=make_logger())


def test_to_payload_omits_unset_optional_keys() -> None:
    embed = DiscordEmbed(author_name="Frieren", title="Sousou no Frieren", color=0x123456)

    payload = embed.to_payload()

    assert payload["title"] == "Sousou no Frieren"
    assert payload["color"] == 0x123456
    assert payload["author"] == {"name": "Frieren", "url": PROJECT_URL}
    assert "url" not in payload
    assert "description" not in payload
    assert "thumbnail" not in payload


def test_to_payload_carries_optional_keys_and_stamp() -> None:
    embed = DiscordEmbed(
        author_name="Frieren",
        title="Sousou no Frieren",
        color=1,
        url="https://releases.moe/1",
        description="all imported",
        fields=(EmbedField(name="n", value="v"),),
        thumb_url="https://img.anili.st/cover.png",
    )

    payload = embed.to_payload()

    assert payload["url"] == "https://releases.moe/1"
    assert payload["description"] == "all imported"
    assert payload["thumbnail"] == {"url": "https://img.anili.st/cover.png"}
    assert payload["fields"] == [{"name": "n", "value": "v"}]
    assert "timestamp" in payload
    footer = cast("dict[str, str]", payload["footer"])
    assert footer["text"].startswith("SeaDexArr v")


def test_to_payload_clamps_item_limits() -> None:
    embed = DiscordEmbed(
        author_name="a" * 300,
        title="t" * 300,
        color=1,
        description="d" * 5000,
        fields=(EmbedField(name="n" * 300, value="x" * 2000),),
    )

    payload = embed.to_payload()

    title = cast("str", payload["title"])
    assert len(title) == 256
    assert title.endswith("…")
    author = cast("dict[str, str]", payload["author"])
    assert len(author["name"]) == 256
    assert len(cast("str", payload["description"])) == 4096
    fields = cast("list[dict[str, str]]", payload["fields"])
    assert len(fields[0]["name"]) == 256
    assert len(fields[0]["value"]) == 1024
    assert fields[0]["value"].endswith("…")


def test_to_payload_caps_field_count() -> None:
    embed = DiscordEmbed(
        author_name="a",
        title="t",
        color=1,
        fields=tuple(EmbedField(name=f"f{i}", value="v") for i in range(30)),
    )

    fields = cast("list[dict[str, str]]", embed.to_payload()["fields"])

    assert len(fields) == 25


def test_to_payload_drops_fields_over_embed_total() -> None:
    embed = DiscordEmbed(
        author_name="a",
        title="t",
        color=1,
        fields=tuple(EmbedField(name=f"f{i}", value="x" * 1024) for i in range(10)),
    )

    payload = embed.to_payload()

    # Trailing fields are shed so the whole embed stays under Discord's 6000
    # total (title + author + footer + all field names/values).
    fields = cast("list[dict[str, str]]", payload["fields"])
    assert 0 < len(fields) < 10
    author = cast("dict[str, str]", payload["author"])
    footer = cast("dict[str, str]", payload["footer"])
    total = (
        len(cast("str", payload["title"]))
        + len(author["name"])
        + len(footer["text"])
        + sum(len(f["name"]) + len(f["value"]) for f in fields)
    )
    assert total <= 6000


def test_build_fields_tracker_links_and_tags() -> None:
    seadex_dict = {
        "SubsPlease": rg_group(
            {
                "https://nyaa.si/view/1": url_item(url="https://nyaa.si/view/1", tracker=Tracker.NYAA, download=True),
                "https://skipped.example": url_item(url="https://skipped.example", download=False),
            },
            tags=frozenset({Tag.VFR, Tag.HDR}),
        ),
        "NotFlagged": rg_group({"https://nyaa.si/view/2": url_item(url="https://nyaa.si/view/2", download=False)}),
    }

    fields = _notifier().build_fields(arr=Arr.SONARR, release_group=["OldGroup"], seadex_dict=seadex_dict)

    # A group with nothing flagged for download contributes no field.
    assert fields == [
        EmbedField(name="Sonarr Release:", value="OldGroup"),
        EmbedField(name="SeaDex recommendation: SubsPlease", value="Tags: HDR, VFR\n[Nyaa](https://nyaa.si/view/1)"),
    ]


def test_build_fields_release_group_falls_back_to_none() -> None:
    fields = _notifier().build_fields(arr=Arr.RADARR, release_group=[None, ""], seadex_dict={})

    assert fields == [EmbedField(name="Radarr Release:", value="None")]
