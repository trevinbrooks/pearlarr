"""Discord webhook boundary: the typed embed and the POST that ships it.

:class:`DiscordEmbed` carries a notification as typed data; ``to_payload`` is
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

# Embed accent colors (the strip Discord renders along the embed's left edge).
COLOR_GRAB = 0x3498DB
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
    """One Discord embed field (name/value), typed until the payload boundary."""

    name: str
    value: str


def _clamp(text: str, limit: int) -> str:
    """Truncate to a Discord limit with a visible ellipsis."""

    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class DiscordEmbed:
    """One webhook embed, held as typed data until :meth:`to_payload`.

    ``url`` makes the title a link (the SeaDex entry page for a grab) and
    ``thumb_url`` is the AniList cover; both are omitted from the payload when
    unset rather than sent as nulls.
    """

    author_name: str
    title: str
    color: int
    url: str | None = None
    description: str = ""
    fields: tuple[EmbedField, ...] = ()
    thumb_url: str | None = None

    def to_payload(self) -> dict[str, Json]:
        """The ``embeds[0]`` object Discord expects, clamped to its hard limits."""

        author = _clamp(self.author_name, _MAX_NAME_LEN)
        title = _clamp(self.title, _MAX_NAME_LEN)
        description = _clamp(self.description, _MAX_DESCRIPTION_LEN) if self.description else ""
        footer = f"SeaDexArr v{__version__}"

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
            fields.append({"name": name, "value": value})

        embed: dict[str, Json] = {
            "author": {"name": author, "url": PROJECT_URL},
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
        return embed


def discord_push(url: str, embed: DiscordEmbed, *, client: httpx.Client) -> None:
    """POST one embed to the webhook - a pure POST, pacing lives in the Notifier.

    Raises ``httpx.HTTPError`` (incl. HTTP error statuses) so the caller's
    containment decides; a webhook failure must never abort a grab. The
    hung-webhook bound is the shared web client's default timeout.
    """

    response = client.post(url, json={"embeds": [embed.to_payload()]})
    response.raise_for_status()
