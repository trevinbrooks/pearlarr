"""Discord webhook boundary: the typed embed and the POST that ships it.

`DiscordEmbed` carries a notification as typed data; `to_payload` is
the single JSON-shaped boundary, where Discord's documented hard limits are
enforced so an oversized notification degrades to a truncated embed instead of
a 400 from the webhook.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple

import httpx

from .paths import PROJECT_URL
from .seadex_types import Json
from .. import __version__

# The SeaDex logo, sent as the webhook's avatar so every notification posts
# under a consistent Pearlarr identity regardless of the webhook's own setup.
_SEADEX_ICON = "https://cdn.discordapp.com/icons/771790175591333909/529b324695b60f65abb7aaa31c114c7b.webp?size=2048"

# Embed accent colors (the strip Discord renders along the embed's left edge).
# Grab amber follows the *arr ecosystem convention (grabbed = amber "in
# flight"; green = imported) in the flat-UI shade matching its siblings below.
COLOR_GRAB = 0xF1C40F
COLOR_SUCCESS = 0x2ECC71
COLOR_DEFERRED = 0xE67E22
COLOR_FAILED = 0xE74C3C

# Discord's documented embed limits, enforced at the payload boundary. The
# total spans title + description + author name + footer + all field names and
# values, so trailing fields are dropped once the running sum would exceed it.
_MAX_FIELDS = 25
_MAX_NAME_LEN = 256
_MAX_VALUE_LEN = 1024
_MAX_DESCRIPTION_LEN = 4096
_MAX_TOTAL_LEN = 6000


class EmbedField(NamedTuple):
    """One Discord embed field (name/value), typed until the payload boundary.

    `inline` fields render side by side (up to three per row); the grab
    embed pairs its short metadata fields, everything else stays full-width.
    """

    name: str
    value: str
    inline: bool = False


def _clamp(text: str, limit: int) -> str:
    """Truncate to a Discord limit with a visible ellipsis."""

    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class DiscordEmbed:
    """One webhook embed, held as typed data until `to_payload`.

    `url` makes the title a link (the SeaDex entry page for a grab),
    `thumb_url` is the AniList cover and `image_url` the wide AniList
    banner; each is omitted from the payload when unset rather than sent as a
    null.
    """

    author_name: str
    title: str
    color: int
    url: str | None = None
    description: str = ""
    fields: tuple[EmbedField, ...] = ()
    thumb_url: str | None = None
    image_url: str | None = None
    author_icon_url: str | None = None

    def to_payload(self) -> dict[str, Json]:
        """The `embeds[0]` object Discord expects, clamped to its hard limits."""

        author = _clamp(self.author_name, _MAX_NAME_LEN)
        title = _clamp(self.title, _MAX_NAME_LEN)
        description = _clamp(self.description, _MAX_DESCRIPTION_LEN) if self.description else ""
        footer = f"Pearlarr v{__version__}"

        # Keep fields while the embed total stays under the limit; a grab embed
        # has one field per release group, so a fat entry sheds trailing groups
        # instead of 400ing the webhook.
        total = len(author) + len(title) + len(description) + len(footer)
        fields: list[Json] = []
        for f in self.fields[:_MAX_FIELDS]:
            name = _clamp(f.name, _MAX_NAME_LEN)
            value = _clamp(f.value, _MAX_VALUE_LEN)
            if total + len(name) + len(value) > _MAX_TOTAL_LEN:
                break
            total += len(name) + len(value)
            fields.append({"name": name, "value": value, "inline": f.inline})

        author_obj: dict[str, Json] = {"name": author, "url": PROJECT_URL}
        if self.author_icon_url:
            author_obj["icon_url"] = self.author_icon_url
        embed: dict[str, Json] = {
            "author": author_obj,
            "title": title,
            "color": self.color,
            "fields": fields,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "footer": {"text": footer},
        }
        if description:
            embed["description"] = description
        if self.url:
            embed["url"] = self.url
        if self.thumb_url:
            embed["thumbnail"] = {"url": self.thumb_url}
        if self.image_url:
            embed["image"] = {"url": self.image_url}
        return embed


def discord_push(url: str, embed: DiscordEmbed, *, client: httpx.Client) -> None:
    """POST one embed to the webhook - a pure POST, pacing lives in the Notifier.

    The payload overrides the webhook's display identity (Pearlarr name +
    SeaDex avatar) so posts look the same on any webhook. Raises
    `httpx.HTTPError` (incl. HTTP error statuses) so the caller's containment
    decides; a webhook failure must never abort a grab. The hung-webhook bound
    is the shared web client's default timeout.
    """

    payload = {
        "username": "Pearlarr",
        "avatar_url": _SEADEX_ICON,
        "embeds": [embed.to_payload()],
    }
    response = client.post(url, json=payload)
    response.raise_for_status()
