"""Fire one representative grab notification at a Discord webhook.

Maintainer tooling for iterating on the embed layout:

    uv run python scripts/sample_grab_post.py <discord-webhook-url>

Screenshot the resulting embed (crop to the embed only) and replace
docs/assets/example_post.png with it.

Every value is real: the SeaDex entry (notes, comparison links) and its PMR
torrent (files, sizes, dual-audio flag) are fetched live; the AniList art URLs
and both titles were verified against the live AniList GraphQL API and Skyhook.
"""

from __future__ import annotations

import sys

import httpx
from seadex import SeaDexEntry, Tracker

from pearlarr.config import Arr
from pearlarr.notify import GrabNotice, Notifier
from pearlarr.seadex_types import SeadexReleaseGroupItem, SeadexUrlItem
from pearlarr.torrents import AddOutcome, ReleaseOutcome


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: sample_grab_post.py <discord-webhook-url>\n")
        return 2
    webhook = sys.argv[1]

    # Both real groups on the Frieren entry (AniList 154587), fetched live: PMR
    # (the SeaDex best) and LostYears (the alternative). TWO flagged groups make
    # push_grab render per-group full-width FIELDS instead of the description -
    # the layout whose `-#` subtext rendering the throwaway post verifies.
    entry = SeaDexEntry().from_id(154587)
    picks = [t for t in entry.torrents if t.tracker is Tracker.NYAA]
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

    notifier = Notifier(discord_url=webhook, web=httpx.Client(timeout=30))
    ok = notifier.push_grab(
        GrabNotice(
            arr=Arr.SONARR,
            arr_title="Frieren: Beyond Journey's End",  # Sonarr's title (verified via Skyhook)
            al_title="Frieren: Beyond Journey’s End",  # AniList english title (what the gateway returns)
            entry=entry,
            # AniList cover + banner for 154587, verified live.
            thumb_url="https://s4.anilist.co/file/anilistcdn/media/anime/cover/medium/bx154587-qQTzQnEJJ3oB.jpg",
            banner_url="https://s4.anilist.co/file/anilistcdn/media/anime/banner/154587-ivXNJ23SM1xB.jpg",
            release_group=["Erai-raws"],
            seadex_dict=seadex_dict,
            results=[
                ReleaseOutcome(AddOutcome.ADDED, None, "PMR"),
                ReleaseOutcome(AddOutcome.ADDED, None, "LostYears"),
            ],
            failed_groups=frozenset(),
            coverage="S01 E01-E28",  # what a full-season Sonarr grab computes
        ),
    )
    sys.stdout.write(f"posted: {ok}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
