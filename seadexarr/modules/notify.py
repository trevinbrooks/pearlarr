"""Discord notifier: build the typed embeds and post the notifications.

``Notifier`` owns the Discord webhook - building the per-release grab embed
from a shaped ``seadex_dict`` and the wait-complete summary embed (plus a
generic outbound webhook POST of the wait report). It's gated on a configured
url; with none, every push is a no-op.
"""

import logging
import time
from collections.abc import Sequence

import requests

from .config import Arr
from .discord import (
    COLOR_DEFERRED,
    COLOR_FAILED,
    COLOR_GRAB,
    COLOR_SUCCESS,
    DiscordEmbed,
    EmbedField,
    discord_push,
)
from .log import format_elapsed
from .manual_import import OutcomeCategory
from .seadex_types import SeadexDict
from .wait_view import WaitResult

# Cap how many titles a single notification field lists before collapsing the
# remainder into a "… +N more" line, keeping a big carried-over backlog readable;
# the payload boundary's char clamps are the hard limit guarantee.
_MAX_FIELD_TITLES = 25

# Minimum spacing between consecutive Discord pushes (webhook rate limiting).
_PUSH_SPACING_S = 1.0


def _failure_detail(exc: requests.RequestException) -> str:
    """Describe a request failure WITHOUT interpolating the exception.

    A requests exception's str embeds the request URL - for a webhook that URL
    IS the credential - so only the HTTP status (when a response exists) or the
    exception type name is reported.
    """

    if exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _wait_color(result: WaitResult) -> int:
    """The summary accent: red if anything failed, orange if deferred, else green."""

    if result.failed > 0:
        return COLOR_FAILED
    if result.left > 0:
        return COLOR_DEFERRED
    return COLOR_SUCCESS


class Notifier:
    """Builds Discord embeds and posts grab + wait-complete notifications."""

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
        # Monotonic instant of the last Discord POST, for burst pacing.
        self._last_push: float | None = None

    @property
    def enabled(self) -> bool:
        """True when a Discord webhook is configured (and not disabled by a dead-webhook 4xx)."""

        return self.discord_url is not None

    def push_grab(
        self,
        *,
        arr_title: str,
        al_title: str,
        seadex_url: str,
        fields: Sequence[EmbedField],
        thumb_url: str | None,
    ) -> bool:
        """Post a grab notification: the AniList title linking to the SeaDex entry.

        Args:
            arr_title (str): Title as in the Arr instance (the author line)
            al_title (str): Title as in AniList (the embed title)
            seadex_url (str): URL to the SeaDex page (the title link)
            fields (Sequence[EmbedField]): Embed fields from :meth:`build_fields`
            thumb_url (str | None): AniList cover thumbnail URL
        """

        return self._push(
            DiscordEmbed(
                author_name=arr_title,
                title=al_title,
                color=COLOR_GRAB,
                url=seadex_url or None,
                fields=tuple(fields),
                thumb_url=thumb_url,
            ),
        )

    def push_wait_summary(self, *, arr: Arr, result: WaitResult) -> bool:
        """Post the wait-pass outcome to Discord and/or the generic webhook.

        A no-op (returns False) when nothing waited or no url is configured; the
        caller already gates on ``wait_notify`` and swallows any error, so this
        can never abort the end-of-run cache save.

        Args:
            arr (Arr): Which Arr the wait pass ran for (for the title).
            result (WaitResult): The terminal outcomes + elapsed time.
        """

        if result.waited == 0:
            return False
        elapsed = format_elapsed(result.elapsed_s)
        posted = self._push(
            DiscordEmbed(
                author_name="SeaDexArr",
                title=f"{arr.capitalize()} wait complete",
                color=_wait_color(result),
                description=f"{result.imported} imported · {result.left} left · {result.failed} failed · {elapsed}",
                fields=self._wait_fields(result),
            ),
        )
        if self.webhook_url is not None:
            posted = self._post_webhook(arr, result, self.webhook_url) or posted
        return posted

    @staticmethod
    def _wait_fields(result: WaitResult) -> tuple[EmbedField, ...]:
        """One counted field per outcome class (imported / left / failed), if any.

        Deferred and failed rows carry the outcome's human detail (the reason
        the torrent didn't land); imported rows list just the title.
        """

        sections = (
            (OutcomeCategory.SUCCESS, "Imported"),
            (OutcomeCategory.DEFERRED, "Left for a later run"),
            (OutcomeCategory.FAILED, "Failed"),
        )
        fields: list[EmbedField] = []
        for category, name in sections:
            rows = [r for r in result.rows if r.outcome.category is category]
            if not rows:
                continue
            lines = [
                r.label if category is OutcomeCategory.SUCCESS else f"{r.label} — {r.outcome.detail}"
                for r in rows[:_MAX_FIELD_TITLES]
            ]
            value = "\n".join(lines)
            if len(rows) > _MAX_FIELD_TITLES:
                value += f"\n… +{len(rows) - _MAX_FIELD_TITLES} more"
            fields.append(EmbedField(name=f"{name} ({len(rows)})", value=value))
        return tuple(fields)

    def _post_webhook(self, arr: Arr, result: WaitResult, url: str) -> bool:
        """POST the report JSON to the generic webhook; warn-and-swallow request errors."""

        payload = {
            "arr": str(arr),
            "imported": result.imported,
            "left": result.left,
            "failed": result.failed,
            "elapsed_s": result.elapsed_s,
            "rows": [{"label": r.label, "outcome": r.outcome.name, "word": r.outcome.word} for r in result.rows],
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except requests.RequestException as exc:
            self.logger.warning(
                f"Wait-report webhook POST failed ({_failure_detail(exc)}) - check notifications.wait_webhook_url",
            )
            return False
        return True

    def build_fields(
        self,
        *,
        arr: Arr,
        release_group: list[str | None] | None,
        seadex_dict: SeadexDict,
    ) -> list[EmbedField]:
        """Build the Discord embed fields for a grab.

        The first field names the Arr's current release group; one field per
        SeaDex release group then lists its tags and, as ``[Tracker](url)``
        markdown links, the releases flagged for download.

        Args:
            arr (Arr): Type of arr instance
            release_group (list[str | None] | None): Arr release group(s)
            seadex_dict (dict): Dictionary of SeaDex releases
        """

        fields: list[EmbedField] = []

        # First field: the Arr's current release group(s), blanks/None dropped,
        # falling back to "None" when nothing is left.
        names = [group for group in (release_group or []) if group] or ["None"]

        fields.append(
            EmbedField(
                name=f"{arr.capitalize()} Release:",
                value="\n".join(names),
            ),
        )

        # One field per SeaDex group with links flagged for download; the link
        # text is the tracker the release comes from.
        for srg, srg_item in seadex_dict.items():
            links = [f"[{u.tracker.value}]({url})" for url, u in srg_item.urls.items() if u.download]
            if not links:
                continue
            value = ""
            if srg_item.tags:
                value += "Tags: " + ", ".join(sorted(str(tag) for tag in srg_item.tags)) + "\n"
            value += "\n".join(links)
            fields.append(EmbedField(name=f"SeaDex recommendation: {srg}", value=value))

        return fields

    def _pace(self) -> None:
        """Keep burst pushes >= 1s apart and stamp this push's instant.

        Sleeps only when a prior push happened under the spacing ago, so a
        single (or final) push never pays a trailing dead second.
        """

        now = time.monotonic()
        if self._last_push is not None:
            remaining = _PUSH_SPACING_S - (now - self._last_push)
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        self._last_push = now

    def _push(self, embed: DiscordEmbed) -> bool:
        """Post one embed to the configured Discord webhook.

        A no-op (returns False) when no webhook is configured. A request failure
        is contained here (warn, return False): a notification failure must never
        abort a grab or skip the cache-update tail. A 4xx means the webhook
        itself is bad (e.g. deleted -> 404), so Discord pushes are disabled for
        the rest of the run instead of warning once per grab - EXCEPT a 429,
        which is Discord throttling a healthy webhook (a burst can outrun the 1s
        pacing), so it stays per-push like the transient 5xx / connection errors.
        """

        if self.discord_url is None:
            return False

        self._pace()
        try:
            discord_push(url=self.discord_url, embed=embed)
        except requests.RequestException as exc:
            detail = _failure_detail(exc)
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                # Rate-limited, not a dead webhook: this push is dropped, later
                # ones still go out. Don't point at the config - it's fine.
                self.logger.warning(
                    f"Discord notification failed ({detail}) - rate limited by Discord; "
                    f"later notifications will still be sent",
                )
            elif status is not None and 400 <= status < 500:
                self.discord_url = None
                self.logger.warning(
                    f"Discord notification failed ({detail}) - disabling Discord notifications "
                    f"for this run; check notifications.discord_url",
                )
            else:
                self.logger.warning(f"Discord notification failed ({detail}) - check notifications.discord_url")
            return False
        return True
