"""Regenerate the README's grab-notification screenshot, or post one for real.

Maintainer tooling for iterating on the embed layout:

    uv run python scripts/sample_grab_post.py                        # rewrites docs/assets/example_post.png
    uv run python scripts/sample_grab_post.py <discord-webhook-url>  # throwaway post to real Discord

Capture mode needs no Discord account and no interaction: it intercepts the
exact webhook JSON `push_grab` builds, renders it in Discohook's message
viewer (a Discord-faithful renderer) in headless Chromium, and screenshots
the embed card. One-time setup: `uv run playwright install chromium`. It
renders the single-group layout the README shows (the common case: one
compact card for PMR, the SeaDex best).

A real post stays the rendering ground truth - use it when the embed adopts
markdown Discord only just shipped. It flags BOTH real groups on the entry
(PMR plus LostYears, the alternative): two groups make `push_grab` render
per-group full-width FIELDS instead of the description - the layout whose
`-#` subtext rendering the throwaway post verifies.

Every value is real: the SeaDex entry (notes, comparison links) and its
torrents (files, sizes, dual-audio flag) are fetched live. The AniList art
URLs and both titles were verified against the live AniList GraphQL API and
Skyhook.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from playwright.sync_api import sync_playwright
from seadex import SeaDexEntry, Tracker

from pearlarr.config import Arr
from pearlarr.json_narrow import is_json_obj
from pearlarr.notify import GrabNotice, Notifier
from pearlarr.seadex_types import SeadexReleaseGroupItem, SeadexUrlItem
from pearlarr.torrents import AddOutcome, ReleaseOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pearlarr.seadex_types import Json

# The Frieren entry (AniList 154587): PMR is the SeaDex best, LostYears the alternative.
_ENTRY_ID = 154587
_BEST_GROUP = "PMR"
_ALT_GROUP = "LostYears"

_PNG = Path(__file__).resolve().parents[1] / "docs" / "assets" / "example_post.png"

# Discohook's iframe-friendly renderer: ?data= is url-safe base64 QueryData.
_VIEWER = "https://discohook.app/viewer"


def _notice(groups: Sequence[str]) -> GrabNotice:
    """The sample grab notice, restricted to `groups` (each one grabbed cleanly)."""

    entry = SeaDexEntry().from_id(_ENTRY_ID)
    picks = [t for t in entry.torrents if t.tracker is Tracker.NYAA and t.release_group in groups]
    seadex_dict = {
        t.release_group: SeadexReleaseGroupItem(
            urls={
                t.url: SeadexUrlItem(
                    url=t.url,
                    files=[f.name for f in t.files],
                    size=[f.size for f in t.files],
                    tracker=t.tracker,
                    is_dual_audio=t.is_dual_audio,
                    download=True,
                ),
            },
            tags=frozenset(t.tags),
        )
        for t in picks
    }
    return GrabNotice(
        arr=Arr.SONARR,
        arr_title="Frieren: Beyond Journey's End",  # Sonarr's title (verified via Skyhook)
        al_title="Frieren: Beyond Journey’s End",  # AniList english title (what the gateway returns)
        entry=entry,
        # AniList cover + banner for 154587, verified live.
        thumb_url="https://s4.anilist.co/file/anilistcdn/media/anime/cover/medium/bx154587-qQTzQnEJJ3oB.jpg",
        banner_url="https://s4.anilist.co/file/anilistcdn/media/anime/banner/154587-ivXNJ23SM1xB.jpg",
        replaced_groups=("Erai-raws",),
        seadex_dict=seadex_dict,
        results=[ReleaseOutcome(AddOutcome.ADDED, None, group) for group in groups],
        failed_groups=frozenset(),
        coverage="S01 E01-E28",  # what a full-season Sonarr grab computes
    )


def _grab_payload(notice: GrabNotice) -> dict[str, Json]:
    """The exact webhook JSON `push_grab` would POST, captured instead of sent."""

    bodies: list[bytes] = []

    def record(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(204)

    notifier = Notifier(
        discord_url="https://capture.invalid/webhook",
        web=httpx.Client(transport=httpx.MockTransport(record)),
    )
    if not notifier.push_grab(notice):
        raise RuntimeError("push_grab did not post to the capture transport")
    [body] = bodies
    payload: object = json.loads(body)
    if not is_json_obj(payload):
        raise TypeError("webhook payload is not a JSON object")
    return payload


def _viewer_url(payload: dict[str, Json]) -> str:
    """The Discohook viewer link rendering `payload`, dark-themed, header hidden."""

    query: Json = {"messages": [{"data": payload}]}
    data = base64.urlsafe_b64encode(json.dumps(query).encode("utf-8")).decode("ascii").rstrip("=")
    return f"{_VIEWER}?data={data}&theme=dark&header=false"


def _capture(notice: GrabNotice) -> int:
    """Render the embed headlessly and screenshot its card over the README PNG."""

    url = _viewer_url(_grab_payload(notice))
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1280, "height": 2400},
            device_scale_factor=2,  # the README displays at half size - keep it crisp
        )
        page.goto(url, wait_until="networkidle")
        # The card renders at Discohook's 520px cap. The README's width="520"
        # displays it 1:1, so keep the two in step.
        page.wait_for_function("() => [...document.images].every((img) => img.complete && img.naturalWidth > 0)")
        page.locator('div[style*="border-left-color"]').screenshot(path=str(_PNG))
        browser.close()
    sys.stdout.write(f"wrote {_PNG}\n")
    return 0


def _post(webhook: str, notice: GrabNotice) -> int:
    """Fire the notice at a real Discord webhook."""

    notifier = Notifier(discord_url=webhook, web=httpx.Client(timeout=30))
    ok = notifier.push_grab(notice)
    sys.stdout.write(f"posted: {ok}\n")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) == 1:
        return _capture(_notice([_BEST_GROUP]))
    if len(sys.argv) == 2:
        return _post(sys.argv[1], _notice([_BEST_GROUP, _ALT_GROUP]))
    sys.stderr.write("usage: sample_grab_post.py [discord-webhook-url]\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
