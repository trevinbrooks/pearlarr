"""Decides which listed or on-disk files could be an episode. The seed, the import, and the parse share these rules."""

import os
from collections.abc import Iterator, Sequence

from .manual_import import path_leaf

TORRENT_FILENAMES_TO_SKIP = [
    "NCED",
    "NCOP",
    "Creditless Ending",
    "Creditless Opening",
    "Creditless ED",
    "Creditless OP",
]

# File extensions that never map to an episode (subtitles, fonts, chapters,
# metadata, images, samples, ...). We skip these before querying Sonarr so we
# don't waste a round-trip on them. This is deliberately a deny-list rather than
# an allow-list of video extensions: the cost of missing one here is a single
# harmless API call (Sonarr just returns no episode), whereas an allow-list that
# omits an unusual container would silently drop a real episode.
NON_VIDEO_EXTENSIONS = {
    ".ass",
    ".srt",
    ".ssa",
    ".sub",
    ".idx",
    ".sup",
    ".vtt",
    # Matroska SUBTITLES (often SxxExx-named, so they'd parse). The video .mkv is kept.
    ".mks",
    ".nfo",
    ".txt",
    ".md",
    ".sfv",
    ".xml",
    ".json",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".torrent",
    ".url",
    ".rar",
    ".zip",
    ".7z",
    # Bundled manga volumes and books.
    ".cbz",
    ".cbr",
    ".cb7",
    ".pdf",
    ".epub",
    # Audio tracks + rip sidecars (OSTs, EAC logs/cuesheets) bundled in releases:
    # never an episode. `.mka` is Matroska audio (the video `.mkv` is kept).
    ".flac",
    ".mka",
    ".wav",
    ".aac",
    ".ac3",
    ".dts",
    ".dtshd",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".wv",
    ".tak",
    ".ape",
    ".cue",
    ".log",
    ".m3u8",
    # Checksum sidecars: Sonarr's manual-import scan never offers them, so a
    # seeded one sticks a record on "intended file missing" until the deadline.
    ".blake3",
    ".md5",
    ".sha2",
    ".sha256",
}


def is_video_candidate(basename: str) -> bool:
    """Whether a filename is an importable video (not a sub, font, or creditless OP/ED).

    The skip rules live only here, so the seed, the import, and the parse agree on what could be an episode.
    """

    if any(skip in basename for skip in TORRENT_FILENAMES_TO_SKIP):
        return False
    return os.path.splitext(basename)[1].lower() not in NON_VIDEO_EXTENSIONS


def video_file_entries(files: Sequence[str]) -> Iterator[tuple[int, str]]:
    """Yield `(index, basename)` for each importable video file in `files`, the index keying a size list."""

    for idx, name in enumerate(files):
        base = path_leaf(name)
        if is_video_candidate(base):
            yield idx, base
