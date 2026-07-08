# pyright: strict
"""Tests for the Discord embed boundary.

``DiscordEmbed.to_payload`` is the single JSON-shaped seam - these pin the
omitted-when-unset optional keys, the field ``inline`` flag, and the clamping
to Discord's hard limits - and ``discord_push`` is the pure POST that ships it
(wire shape, errors propagate: containment lives in the Notifier).
"""

import json
from typing import cast

import httpx
import pytest
import respx

from seadexarr.modules.discord import (
    PROJECT_URL,
    DiscordEmbed,
    EmbedField,
    discord_push,
)
from seadexarr.modules.seadex_types import Json


def _str_at(payload: dict[str, Json], key: str) -> str:
    """The payload's string value at ``key``, narrowed for the length asserts."""

    value = payload[key]
    assert isinstance(value, str)
    return value


def _obj_at(payload: dict[str, Json], key: str) -> dict[str, str]:
    """The payload's one-level string object at ``key`` (author/footer/thumbnail)."""

    value = payload[key]
    assert isinstance(value, dict)
    return cast("dict[str, str]", value)


def _fields_of(payload: dict[str, Json]) -> list[dict[str, Json]]:
    """The payload's fields list, narrowed to its row objects."""

    value = payload["fields"]
    assert isinstance(value, list)
    return cast("list[dict[str, Json]]", value)


def _name_value(field: dict[str, Json]) -> tuple[str, str]:
    """A field row's name/value pair, narrowed for the length asserts."""

    name, value = field["name"], field["value"]
    assert isinstance(name, str)
    assert isinstance(value, str)
    return name, value


def test_to_payload_omits_unset_optional_keys() -> None:
    embed = DiscordEmbed(author_name="Frieren", title="Sousou no Frieren", color=0x123456)

    payload = embed.to_payload()

    assert payload["title"] == "Sousou no Frieren"
    assert payload["color"] == 0x123456
    assert payload["author"] == {"name": "Frieren", "url": PROJECT_URL}
    assert "url" not in payload
    assert "description" not in payload
    assert "thumbnail" not in payload
    assert "image" not in payload


def test_to_payload_carries_optional_keys_and_stamp() -> None:
    embed = DiscordEmbed(
        author_name="Frieren",
        title="Sousou no Frieren",
        color=1,
        url="https://releases.moe/1",
        description="all imported",
        fields=(EmbedField(name="n", value="v"), EmbedField(name="i", value="w", inline=True)),
        thumb_url="https://img.anili.st/cover.png",
        image_url="https://img.anili.st/banner.png",
        author_icon_url="https://releases.moe/favicon.png",
    )

    payload = embed.to_payload()

    assert payload["url"] == "https://releases.moe/1"
    assert payload["description"] == "all imported"
    assert payload["thumbnail"] == {"url": "https://img.anili.st/cover.png"}
    assert payload["image"] == {"url": "https://img.anili.st/banner.png"}
    assert payload["author"] == {
        "name": "Frieren",
        "url": PROJECT_URL,
        "icon_url": "https://releases.moe/favicon.png",
    }
    # The inline flag rides each field row to the wire.
    assert payload["fields"] == [
        {"name": "n", "value": "v", "inline": False},
        {"name": "i", "value": "w", "inline": True},
    ]
    assert "timestamp" in payload
    assert _obj_at(payload, "footer")["text"].startswith("SeaDexArr v")


def test_to_payload_clamps_item_limits() -> None:
    embed = DiscordEmbed(
        author_name="a" * 300,
        title="t" * 300,
        color=1,
        description="d" * 5000,
        fields=(EmbedField(name="n" * 300, value="x" * 2000),),
    )

    payload = embed.to_payload()

    title = _str_at(payload, "title")
    assert len(title) == 256
    assert title.endswith("…")
    assert len(_obj_at(payload, "author")["name"]) == 256
    assert len(_str_at(payload, "description")) == 4096
    name, value = _name_value(_fields_of(payload)[0])
    assert len(name) == 256
    assert len(value) == 1024
    assert value.endswith("…")


def test_to_payload_caps_field_count() -> None:
    embed = DiscordEmbed(
        author_name="a",
        title="t",
        color=1,
        fields=tuple(EmbedField(name=f"f{i}", value="v") for i in range(30)),
    )

    fields = _fields_of(embed.to_payload())

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
    fields = _fields_of(payload)
    assert 0 < len(fields) < 10
    total = (
        len(_str_at(payload, "title"))
        + len(_obj_at(payload, "author")["name"])
        + len(_obj_at(payload, "footer")["text"])
        + sum(len(f[0]) + len(f[1]) for f in map(_name_value, fields))
    )
    assert total <= 6000


def test_to_payload_counts_description_in_embed_total() -> None:
    # The description is part of Discord's 6000 total: a fat one (the notes
    # blockquote can be long) must eat field budget, not overflow the embed.
    embed = DiscordEmbed(
        author_name="a",
        title="t",
        color=1,
        description="d" * 4000,
        fields=tuple(EmbedField(name=f"f{i}", value="x" * 1024) for i in range(10)),
    )

    fields = _fields_of(embed.to_payload())

    # 4000 + author/title/footer leaves room for exactly one ~1027-char field.
    assert len(fields) == 1


@respx.mock
def test_discord_push_posts_the_wrapped_embed() -> None:
    route = respx.post("https://discord.example/hook").respond(json={})
    embed = DiscordEmbed(author_name="Frieren", title="Sousou no Frieren", color=1)

    discord_push(url="https://discord.example/hook", embed=embed, client=httpx.Client())

    body = cast("dict[str, Json]", json.loads(route.calls.last.request.content))
    # The wire shape: the SeaDexArr display identity riding beside the embed.
    assert body["username"] == "SeaDexArr"
    assert _str_at(body, "avatar_url").startswith("https://")
    embeds = body["embeds"]
    assert isinstance(embeds, list)
    (sent,) = cast("list[dict[str, Json]]", embeds)
    expected = embed.to_payload()
    # The timestamp is stamped at send time; everything else matches the boundary.
    assert {k: v for k, v in sent.items() if k != "timestamp"} == {
        k: v for k, v in expected.items() if k != "timestamp"
    }
    assert "timestamp" in sent


@respx.mock
def test_discord_push_propagates_http_errors() -> None:
    # A pure POST: an HTTP error status surfaces to the caller - the
    # warn-and-swallow containment lives in Notifier._push, not here.
    respx.post("https://discord.example/hook").respond(status_code=400)
    embed = DiscordEmbed(author_name="a", title="t", color=1)

    with pytest.raises(httpx.HTTPStatusError):
        discord_push(url="https://discord.example/hook", embed=embed, client=httpx.Client())
