"""Discord notifier: build the typed embeds and post the notifications.

``Notifier`` owns the Discord webhook - building the grab embed from a
:class:`GrabNotice` and the wait-complete summary embed (plus a generic
outbound webhook POST of the wait report). It's gated on a configured url;
with none, every push is a no-op.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import httpx
from seadex import EntryRecord

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
from .log import count_noun, format_elapsed, human_bytes
from .manual_import import OutcomeCategory
from .output import hub_warn
from .seadex_types import SeadexDict, SeadexUrlItem
from .torrents import AddOutcome, ReleaseOutcome
from .wait_view import WaitResult

# Cap how many titles a single notification field lists before collapsing the
# remainder into a "… +N more" line, keeping a big carried-over backlog readable;
# the payload boundary's char clamps are the hard limit guarantee.
_MAX_FIELD_TITLES = 25

# Arr logos for the grab embed's author line: the icon marks which arr the
# grab ran for (the webhook's own avatar already carries the SeaDex identity).
_SONARR_ICON = "https://raw.githubusercontent.com/Sonarr/Sonarr/develop/Logo/512.png"
_RADARR_ICON = "https://raw.githubusercontent.com/Radarr/Radarr/develop/Logo/512.png"

# Cap the comma-separated "Replacing" list so its inline field stays one short
# row; the remainder collapses into a "… +N more" tail.
_MAX_REPLACING = 3

# Taste limit for the SeaDex notes blockquote (the 1024 field-value clamp is
# the hard backstop); truncation lands on a word boundary.
_MAX_NOTES_LEN = 400

# Minimum spacing between consecutive Discord pushes (webhook rate limiting).
_PUSH_SPACING_S = 1.0

# Longest 429 Retry-After worth blocking the run for; anything above skips the
# retry and drops the notification instead.
_MAX_RETRY_AFTER_S = 5.0


def _retry_after_seconds(exc: httpx.HTTPError) -> float:
    """The 429's Retry-After header in seconds (int or float string), default 1.0."""

    header = exc.response.headers.get("Retry-After") if isinstance(exc, httpx.HTTPStatusError) else None
    if header is None:
        return 1.0
    try:
        return max(float(header), 0.0)
    except ValueError:
        return 1.0


def _failure_detail(exc: httpx.HTTPError) -> str:
    """Describe a request failure WITHOUT interpolating the exception.

    An httpx exception's str embeds the request URL - for a webhook that URL
    IS the credential - so only the HTTP status (a status error is the one
    kind carrying a response) or the exception type name is reported.
    """

    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _wait_color(result: WaitResult) -> int:
    """The summary accent: red if anything failed, orange if deferred, else green."""

    if result.failed > 0:
        return COLOR_FAILED
    if result.left > 0:
        return COLOR_DEFERRED
    return COLOR_SUCCESS


@dataclass(frozen=True, slots=True)
class GrabNotice:
    """Everything one grab notification renders, resolved by the grab pipeline.

    ``entry`` is the SeaDex entry whole (url / notes / comparisons / incomplete
    flag). ``results`` are the torrent-client add outcomes, so the embed can
    label a group whose releases were already in the client accordingly rather
    than claiming a fresh grab.
    """

    arr: Arr
    arr_title: str
    al_title: str
    entry: EntryRecord
    thumb_url: str | None
    banner_url: str | None
    release_group: list[str | None] | None
    seadex_dict: SeadexDict
    results: Sequence[ReleaseOutcome]
    # Groups whose add failed at the client (contained; retried next run) - a
    # failed add produces no outcome, so the label needs the explicit note.
    failed_groups: frozenset[str]
    coverage: str


def _md_escape(text: str) -> str:
    """Backslash-escape Discord markdown so free-form text renders literally.

    The backslash escapes first so the escapes just added aren't re-escaped.
    """

    for ch in "\\`*_~|>[]":
        text = text.replace(ch, "\\" + ch)
    return text


def _titles_match(a: str, b: str) -> bool:
    """True when two titles differ only by case or apostrophe style.

    The Arr and AniList titles are near-duplicates for most entries; the embed
    shows the Arr's own title only when it adds information.
    """

    def norm(title: str) -> str:
        return title.replace("’", "'").replace("‘", "'").casefold().strip()

    return norm(a) == norm(b)


def _notes_block(notes: str) -> str:
    """The SeaDex entry notes as one blockquote — the "why this pick" line.

    Notes are plain text; they are NOT markdown-escaped (a stray emphasis is
    cosmetic, mangled escape backslashes are worse). Blank lines drop so the
    quote stays one block; overlong notes truncate on a word boundary.
    """

    text = notes.strip()
    if not text:
        return ""
    if len(text) > _MAX_NOTES_LEN:
        cut = text[:_MAX_NOTES_LEN]
        if not text[_MAX_NOTES_LEN].isspace() and not cut[-1].isspace():
            # A word straddles the cut: drop the partial word (a whitespace-free
            # prefix has nothing to split and stays whole).
            cut = cut.rsplit(maxsplit=1)[0]
        text = cut.rstrip() + " …"
    return "\n".join("> " + line for line in text.splitlines() if line.strip())


def _grab_notes(notice: GrabNotice) -> str:
    """The trailing notes stack: subtitle, entry notes, comparison links, caveats.

    The Arr's own title appears as a muted ``-#`` subtext byline only when it
    differs from the AniList one; each piece is omitted when it has nothing to
    say.
    """

    parts: list[str] = []
    if not _titles_match(notice.arr_title, notice.al_title):
        parts.append(f"-# {_md_escape(notice.arr_title)}")
    if notes := _notes_block(notice.entry.notes):
        parts.append(notes)
    # The lib's lax comparison parse can yield an empty segment; a blank url
    # would render a broken [Comparison N]() link, so filter falsy first.
    if comparisons := tuple(c for c in notice.entry.comparisons if c):
        single = len(comparisons) == 1
        parts.append(
            " · ".join(f"[Comparison{'' if single else f' {i}'}]({url})" for i, url in enumerate(comparisons, start=1)),
        )
    if notice.entry.is_incomplete:
        parts.append("*Entry marked incomplete on SeaDex*")
    return "\n".join(parts)


def _release_line(item: SeadexUrlItem) -> str:
    """One release's line: the tracker link plus size/audio/provenance markers."""

    parts = [f"[{item.tracker.value}]({item.url})"]
    if item.size:
        parts.append(human_bytes(sum(item.size)))
    if item.files:
        parts.append(count_noun(len(item.files), "file"))
    if item.is_dual_audio:
        parts.append("dual audio")
    if item.is_fallback:
        parts.append("fallback")
    if item.size_mismatch:
        parts.append("upgrade")
    return " · ".join(parts)


class _GroupBlock(NamedTuple):
    """One group's rendered pick: the outcome label, the group name, the release lines."""

    label: str
    group: str
    body: str


def _group_blocks(notice: GrabNotice) -> list[_GroupBlock]:
    """One block per group with anything flagged for download.

    The label reflects what actually happened at the torrent client: "Grabbed"
    when anything was added, "Already downloading" when every acted release was
    already there, "Failed" when an add errored (contained; retried next run),
    "Skipped" when none of its releases were attempted. Tags trail the release
    lines as muted subtext.
    """

    outcome_by_group = {r.group: r.outcome for r in notice.results}
    blocks: list[_GroupBlock] = []
    for srg, srg_item in notice.seadex_dict.items():
        release_lines = [_release_line(u) for u in srg_item.urls.values() if u.download]
        if not release_lines:
            continue
        if srg_item.tags:
            release_lines.append("-# " + " · ".join(sorted(str(tag) for tag in srg_item.tags)))

        add_outcome = outcome_by_group.get(srg)
        if add_outcome is AddOutcome.ADDED:
            label = "Grabbed"
        elif add_outcome is AddOutcome.ALREADY_ADDED:
            label = "Already downloading"
        elif srg in notice.failed_groups:
            label = "Failed"
        else:
            label = "Skipped"
        blocks.append(_GroupBlock(label, srg, "\n".join(release_lines)))
    return blocks


def _meta_fields(notice: GrabNotice) -> list[EmbedField]:
    """The short metadata pair, rendered side by side: coverage + replaced groups."""

    fields: list[EmbedField] = []
    if notice.coverage:
        fields.append(EmbedField(name="Episodes", value=notice.coverage, inline=True))
    if current := [group for group in (notice.release_group or []) if group]:
        replaced = current[:_MAX_REPLACING]
        if len(current) > _MAX_REPLACING:
            replaced.append(f"… +{len(current) - _MAX_REPLACING} more")
        fields.append(EmbedField(name="Replacing", value=", ".join(replaced), inline=True))
    return fields


def _grab_embed(notice: GrabNotice) -> DiscordEmbed:
    """The grab embed: the AniList title/art framing the outcome-labelled pick(s).

    A single-group grab (the common case) reads as one compact card: its pick
    rides the description directly under the title, markdown intact. Several
    groups each keep a full-width field. Either way the episodes/replacing
    pair follows side by side and the notes stack trails headerless.
    """

    blocks = _group_blocks(notice)
    description = ""
    fields: list[EmbedField] = []
    if len(blocks) == 1:
        [block] = blocks
        description = f"**{block.label} · `{block.group}`**\n{block.body}"
    else:
        fields += (EmbedField(name=f"{b.label} · {b.group}", value=b.body) for b in blocks)
    fields += _meta_fields(notice)
    if notes := _grab_notes(notice):
        fields.append(EmbedField(name="", value=notes))
    return DiscordEmbed(
        author_name=f"{notice.arr.capitalize()} · SeaDex grab",
        title=notice.al_title,
        color=COLOR_GRAB,
        url=notice.entry.url or None,
        description=description,
        fields=tuple(fields),
        thumb_url=notice.thumb_url,
        image_url=notice.banner_url,
        author_icon_url=_SONARR_ICON if notice.arr is Arr.SONARR else _RADARR_ICON,
    )


class Notifier:
    """Builds Discord embeds and posts grab + wait-complete notifications."""

    def __init__(
        self,
        *,
        discord_url: str | None,
        webhook_url: str | None = None,
        web: httpx.Client,
    ) -> None:
        """Configure the notifier.

        Args:
            discord_url (str | None): Discord webhook url, or None to disable.
            webhook_url (str | None): A generic outbound webhook POSTed the
                wait-complete report JSON (ntfy/gotify/Home-Assistant), or None.
            web (httpx.Client): The shared web client every POST rides.
        """

        self.discord_url = discord_url
        self.webhook_url = webhook_url
        self.web = web
        # Monotonic instant of the last Discord POST, for burst pacing.
        self._last_push: float | None = None

    @property
    def enabled(self) -> bool:
        """True when a Discord webhook is configured (and not disabled by a dead-webhook 4xx)."""

        return self.discord_url is not None

    def push_grab(self, notice: GrabNotice) -> bool:
        """Post a grab notification: the AniList title linking to the SeaDex entry.

        The author line names the event ("Sonarr · SeaDex grab") under the
        arr's own logo; a single-group grab carries its pick in the
        description, a multi-group grab stacks one field per group; the
        AniList art frames it (cover thumbnail + wide banner).

        Args:
            notice (GrabNotice): The grab's resolved notification payload.
        """

        return self._push(_grab_embed(notice))

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
                author_name="Pearlarr",
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
            self.web.post(url, json=payload, timeout=10)
        except httpx.HTTPError as exc:
            hub_warn(f"Wait-report webhook POST failed ({_failure_detail(exc)}) - check notifications.wait_webhook_url")
            return False
        return True

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
        pacing): a short Retry-After is slept out and the push retried once, and
        only a push dropped for real warns, per push, without disabling anything.
        """

        if self.discord_url is None:
            return False

        self._pace()
        try:
            discord_push(url=self.discord_url, embed=embed, client=self.web)
        except httpx.HTTPError as exc:
            detail = _failure_detail(exc)
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            if status == 429:
                return self._retry_rate_limited(url=self.discord_url, embed=embed, exc=exc)
            if status is not None and 400 <= status < 500:
                self.discord_url = None
                hub_warn(
                    f"Discord notification failed ({detail}) - disabling Discord notifications "
                    f"for this run; check notifications.discord_url"
                )
            else:
                hub_warn(f"Discord notification failed ({detail}) - check notifications.discord_url")
            return False
        return True

    def _retry_rate_limited(self, *, url: str, embed: DiscordEmbed, exc: httpx.HTTPError) -> bool:
        """Honor a 429's Retry-After with one retry, or drop the push truthfully.

        A 429 is Discord throttling a healthy webhook, not a dead one, so pushes
        stay enabled and the config is never blamed. A short Retry-After
        (<= 5s) is slept out and the push retried once; a longer one - or a
        retry that fails too - drops THIS notification and says so.
        """

        delay = _retry_after_seconds(exc)
        if delay <= _MAX_RETRY_AFTER_S:
            time.sleep(delay)
            try:
                discord_push(url=url, embed=embed, client=self.web)
            except httpx.HTTPError as retry_exc:
                exc = retry_exc
            else:
                return True
        hub_warn(
            f"Discord notification failed ({_failure_detail(exc)}) - rate limited by Discord; "
            f"this notification was dropped"
        )
        return False
