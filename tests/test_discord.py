# pyright: strict
"""Tests for the Discord embed boundary and the grab-embed fields.

``DiscordEmbed.to_payload`` is the single JSON-shaped seam - these pin the
omitted-when-unset optional keys and the clamping to Discord's hard limits -
``discord_push`` is the pure POST that ships it (wire shape + timeout, errors
propagate: containment lives in the Notifier), and ``Notifier.build_fields``
shapes a grab's fields (markdown tracker links, the one-line tag list, the
release-group fallback).
"""

from typing import cast

import pytest
import requests
from seadex import Tag, Tracker

from seadexarr.modules.config import Arr
from seadexarr.modules.discord import (
    DISCORD_TIMEOUT_S,
    PROJECT_URL,
    DiscordEmbed,
    EmbedField,
    discord_push,
)
from seadexarr.modules.notify import Notifier
from seadexarr.modules.seadex_types import Json

from .builders import make_logger, rg_group, url_item


def _notifier() -> Notifier:
    return Notifier(discord_url=None, webhook_url=None, logger=make_logger())


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


def _fields_of(payload: dict[str, Json]) -> list[dict[str, str]]:
    """The payload's fields list, narrowed to its name/value rows."""

    value = payload["fields"]
    assert isinstance(value, list)
    return cast("list[dict[str, str]]", value)


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
    fields = _fields_of(payload)
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
        + sum(len(f["name"]) + len(f["value"]) for f in fields)
    )
    assert total <= 6000


class _FakeResponse:
    """A minimal response scripting ``raise_for_status`` (all discord_push reads)."""

    def __init__(self, error: requests.HTTPError | None = None) -> None:
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


def test_discord_push_posts_the_wrapped_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[tuple[str, dict[str, list[dict[str, Json]]], tuple[int, int]]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, list[dict[str, Json]]],
        timeout: tuple[int, int],
    ) -> _FakeResponse:
        posts.append((url, json, timeout))
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    embed = DiscordEmbed(author_name="Frieren", title="Sousou no Frieren", color=1)

    discord_push(url="https://discord.example/hook", embed=embed)

    url, body, timeout = posts[0]
    assert url == "https://discord.example/hook"
    assert timeout == DISCORD_TIMEOUT_S  # the hung-webhook bound is actually passed
    (sent,) = body["embeds"]  # the wire shape: {"embeds": [to_payload()]}
    expected = embed.to_payload()
    # The timestamp is stamped at send time; everything else matches the boundary.
    assert {k: v for k, v in sent.items() if k != "timestamp"} == {
        k: v for k, v in expected.items() if k != "timestamp"
    }
    assert "timestamp" in sent


def test_discord_push_propagates_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pure POST: an HTTP error status surfaces to the caller - the
    # warn-and-swallow containment lives in Notifier._push, not here.
    def fake_post(
        url: str,
        *,
        json: dict[str, list[dict[str, Json]]],
        timeout: tuple[int, int],
    ) -> _FakeResponse:
        del url, json, timeout
        return _FakeResponse(error=requests.HTTPError("400 Bad Request"))

    monkeypatch.setattr(requests, "post", fake_post)
    embed = DiscordEmbed(author_name="a", title="t", color=1)

    with pytest.raises(requests.HTTPError):
        discord_push(url="https://discord.example/hook", embed=embed)


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
