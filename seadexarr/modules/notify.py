"""Discord notifier: build the embed fields and post the grab notification.

``Notifier`` owns the Discord webhook - building the per-release embed fields
from a shaped ``seadex_dict`` and pushing the message. It's gated on the
configured webhook url; with none, it's a no-op.
"""

from .discord import discord_push
from .seadex_types import EmbedField, SeadexDict


class Notifier:
    """Builds Discord embed fields and posts grab notifications."""

    def __init__(self, *, discord_url: str | None) -> None:
        """Configure the notifier.

        Args:
            discord_url (str | None): Webhook url, or None to disable.
        """

        self.discord_url = discord_url

    @property
    def enabled(self) -> bool:
        """True when a Discord webhook is configured."""

        return self.discord_url is not None

    def build_fields(
        self,
        *,
        arr: str,
        release_group: str | list[str | None] | None,
        seadex_dict: SeadexDict,
    ) -> list[dict[str, str]]:
        """Build the Discord embed fields for a grab.

        The first field names the Arr's current release group; one field per
        SeaDex release group then lists its tags and the URLs flagged for
        download. Fields are assembled as typed :class:`EmbedField`s and
        serialized to the plain ``{"name", "value"}`` dicts the webhook expects
        at the return boundary.

        Args:
            arr (str): Type of arr instance
            release_group (str | list[str | None] | None): Arr release group(s)
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        fields: list[EmbedField] = []

        # The first field should be the Arr group. If it's empty, mention it's
        # missing. Each branch allocates a fresh list, so the caller's
        # release_group is never mutated (no defensive copy needed).
        if not release_group:
            names = ["None"]
        elif isinstance(release_group, str):
            names = [release_group]
        else:
            # The list can statically carry None entries (the Arr release dict
            # keys typed str | None); drop them so the joined value is str-only.
            names = [group for group in release_group if group is not None]

        fields.append(
            EmbedField(
                name=f"{arr.capitalize()} Release:",
                value="\n".join(names),
            ),
        )

        # SeaDex options with links
        for srg, srg_item in seadex_dict.items():

            # URLs flagged for download in this group, in one pass
            urls_to_download = [
                url
                for url, u in srg_item.urls.items()
                if u.download
            ]

            if urls_to_download:

                # Include any tags in the string
                discord_value = ""
                tags = srg_item.tags
                if len(tags) > 0:
                    discord_value += "Tags:\n"
                    discord_value += "\n".join(tags)
                    discord_value += "\n\n"

                # And include URLs for files we're downloading
                discord_value += "Links:\n"
                discord_value += "\n".join(urls_to_download)

                fields.append(
                    EmbedField(
                        name=f"SeaDex recommendation: {srg}",
                        value=f"{discord_value}",
                    ),
                )

        return [f.to_dict() for f in fields]

    def push(
        self,
        *,
        arr_title: str,
        al_title: str,
        seadex_url: str,
        fields: list[dict[str, str]],
        thumb_url: str | None,
    ) -> bool:
        """Post a grab notification to the configured Discord webhook.

        A no-op (returns False) when no webhook is configured; the caller also
        gates on having actually grabbed something and not being a preview.

        Args:
            arr_title (str): Title as in the Arr instance
            al_title (str): Title as in AniList
            seadex_url (str): URL to the SeaDex page
            fields (list): Embed fields from :meth:`build_fields`
            thumb_url (str | None): AniList cover thumbnail URL
        """

        if self.discord_url is None:
            return False

        return discord_push(
            url=self.discord_url,
            arr_title=arr_title,
            al_title=al_title,
            seadex_url=seadex_url,
            fields=fields,
            thumb_url=thumb_url,
        )
