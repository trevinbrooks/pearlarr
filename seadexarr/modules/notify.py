"""Discord notifier: build the embed fields and post the grab notification.

``Notifier`` owns the Discord webhook - building the per-release embed fields
from a shaped ``seadex_dict`` and pushing the message - plus a wait-complete
summary push to Discord and/or a generic outbound webhook. It's gated on a
configured url; with none, every push is a no-op.
"""

import logging

import requests

from .discord import discord_push
from .log import LogFormatter
from .manual_import import OutcomeCategory
from .seadex_types import EmbedField, SeadexDict
from .wait_view import WaitResult

# Cap how many titles a single notification field lists before collapsing the
# remainder into a "… +N more" line, so a big carried-over backlog can't blow
# past Discord's per-field limit.
_MAX_FIELD_TITLES = 25


class Notifier:
    """Builds Discord embed fields and posts grab + wait-complete notifications."""

    def __init__(
        self,
        *,
        discord_url: str | None,
        webhook_url: str | None = None,
        logger: logging.Logger,
    ) -> None:
        """Configure the notifier.

        Args:
            discord_url (str | None): Discord webhook url, or None to disable.
            webhook_url (str | None): A generic outbound webhook POSTed the
                wait-complete report JSON (ntfy/gotify/Home-Assistant), or None.
            logger (logging.Logger): For the failed-push warnings.
        """

        self.discord_url = discord_url
        self.webhook_url = webhook_url
        self.logger = logger

    @property
    def enabled(self) -> bool:
        """True when a Discord webhook is configured."""

        return self.discord_url is not None

    def push_wait_summary(self, *, arr: str, result: WaitResult) -> bool:
        """Post the wait-pass outcome to Discord and/or the generic webhook.

        A no-op (returns False) when nothing waited or no url is configured; the
        caller already gates on ``wait_notify`` and swallows any error, so this
        can never abort the end-of-run cache save.

        Args:
            arr (str): Which Arr the wait pass ran for (for the title).
            result (WaitResult): The terminal outcomes + elapsed time.
        """

        if result.waited == 0:
            return False
        elapsed = LogFormatter.format_elapsed(result.elapsed_s)
        posted = self.push(
            arr_title=f"SeaDexArr - {arr.capitalize()} wait complete",
            al_title=(f"{result.imported} imported - {result.left} left - {result.failed} failed  ({elapsed})"),
            seadex_url="",
            fields=self._wait_fields(result),
            thumb_url=None,
        )
        if self.webhook_url is not None:
            posted = self._post_webhook(arr, result) or posted
        return posted

    @staticmethod
    def _wait_fields(result: WaitResult) -> list[dict[str, str]]:
        """One embed field per outcome class (imported / left / failed), if any.

        Fields are typed :class:`EmbedField`s, serialized to the plain
        ``{"name", "value"}`` dicts at the return boundary (as in
        :meth:`build_fields`).
        """

        sections = (
            (OutcomeCategory.SUCCESS, "Imported"),
            (OutcomeCategory.DEFERRED, "Left for a later run"),
            (OutcomeCategory.FAILED, "Failed"),
        )
        fields: list[EmbedField] = []
        for category, name in sections:
            labels = [r.label for r in result.rows if r.outcome.category is category]
            if not labels:
                continue
            shown = labels[:_MAX_FIELD_TITLES]
            value = "\n".join(shown)
            if len(labels) > _MAX_FIELD_TITLES:
                value += f"\n… +{len(labels) - _MAX_FIELD_TITLES} more"
            fields.append(EmbedField(name=name, value=value))
        return [f.to_dict() for f in fields]

    def _post_webhook(self, arr: str, result: WaitResult) -> bool:
        """POST the report JSON to the generic webhook; warn-and-swallow request errors."""

        if self.webhook_url is None:
            return False
        payload = {
            "arr": str(arr),
            "imported": result.imported,
            "left": result.left,
            "failed": result.failed,
            "elapsed_s": result.elapsed_s,
            "rows": [{"label": r.label, "outcome": r.outcome.name, "word": r.outcome.word} for r in result.rows],
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=10)
        except requests.RequestException as exc:
            self.logger.warning(f"Wait-report webhook POST failed: {exc}")
            return False
        return True

    def build_fields(
        self,
        *,
        arr: str,
        release_group: list[str | None] | None,
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
            release_group (list[str | None] | None): Arr release group(s)
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        fields: list[EmbedField] = []

        # The first field names the Arr's current release group(s); fall back to
        # "None" when there isn't one. Each branch allocates a fresh list, so the
        # caller's release_group is never mutated (no defensive copy needed).
        if not release_group:
            names = ["None"]
        else:
            # Both Arrs pass the release-dict keys (str | None); drop the blank /
            # None entries, falling back to "None" if that empties the list.
            names = [group for group in release_group if group] or ["None"]

        fields.append(
            EmbedField(
                name=f"{arr.capitalize()} Release:",
                value="\n".join(names),
            ),
        )

        # SeaDex options with links
        for srg, srg_item in seadex_dict.items():
            # URLs flagged for download in this group, in one pass
            urls_to_download = [url for url, u in srg_item.urls.items() if u.download]

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

        A no-op (returns False) when no webhook is configured. A request failure
        is contained here (warn, return False): a notification failure must never
        abort a grab or skip the cache-update tail.

        Args:
            arr_title (str): Title as in the Arr instance
            al_title (str): Title as in AniList
            seadex_url (str): URL to the SeaDex page
            fields (list): Embed fields from :meth:`build_fields`
            thumb_url (str | None): AniList cover thumbnail URL
        """

        if self.discord_url is None:
            return False

        try:
            return discord_push(
                url=self.discord_url,
                arr_title=arr_title,
                al_title=al_title,
                seadex_url=seadex_url,
                fields=fields,
                thumb_url=thumb_url,
            )
        except requests.RequestException as exc:
            self.logger.warning(f"Discord push failed: {exc}")
            return False
