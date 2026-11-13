"""Path and file name helpers: the basename normalizer every match uses, path leaves, and download path mapping."""

import posixpath
import unicodedata
from collections.abc import Sequence

from .arr_models import RemotePathMapping


def normalize_name(name: str) -> str:
    """Normalize a filename leaf for cross-source matching.

    Args:
        name: A filename (basename or full path - only the text is folded).

    Returns:
        The name stripped and case-folded, composed (NFC) before and after the casefold, so two spellings of
        one name fold alike and a folded key folds to itself.
    """

    folded = unicodedata.normalize("NFC", name).strip().casefold()
    # NFC before the casefold keeps two spellings of one name alike: casefold turns U+0345 (the Greek iota below)
    # into a letter where it sits, so a mark after it moves onto that letter. NFC after it because casefold can
    # expand one letter into several (ß to ss, İ to i plus a dot) and leave the marks after that letter uncomposed
    # or out of order, and then folding a stored key a second time would change it.
    return unicodedata.normalize("NFC", folded)


def to_forward_slashes(path: str) -> str:
    r"""Fold `\` to `/` for cross-platform comparison."""

    return path.replace("\\", "/")


def _path_segments(path: str) -> list[str]:
    """Split a path into non-empty segments for the boundary-aware compare (separators folded)."""

    return [segment for segment in to_forward_slashes(path).split("/") if segment]


def translate_download_path(
    content_path: str,
    mappings: Sequence[RemotePathMapping],
    qbit_host: str | None,
) -> str:
    """Map a download-client path into Sonarr's filesystem view.

    Longest-`remotePath`-prefix match is the PRIMARY rule. Host equality (our
    `qbit_host`, folded here) only tiebreaks equally-long prefixes, and host
    inequality never excludes a mapping: Sonarr's `host` is the download-client
    host as SONARR configured it, routinely a different string from our
    qBittorrent host (localhost vs container name vs IP). Matching is per path
    segment, so it is separator-boundary-aware (`/downloads` never matches
    `/downloads-x/f`), tolerant of trailing slashes and Windows backslashes on
    either side, and case-insensitive, while the suffix keeps its ORIGINAL case
    (POSIX targets are case-sensitive). No match returns the path untranslated
    (the same-filesystem no-op). `content_path` is qBittorrent's, a folder or a
    single file.
    """

    content_segments = _path_segments(content_path)
    folded = [segment.casefold() for segment in content_segments]
    target_host = qbit_host.casefold() if qbit_host else None

    best_rank: tuple[int, bool] | None = None
    best_local = ""
    for mapping in mappings:
        if not mapping.remote_path or not mapping.local_path:
            continue
        remote = [segment.casefold() for segment in _path_segments(mapping.remote_path)]
        if not remote or folded[: len(remote)] != remote:
            continue
        host_matches = target_host is not None and (mapping.host or "").casefold() == target_host
        rank = (len(remote), host_matches)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_local = mapping.local_path

    if best_rank is None:
        return content_path
    base = best_local.rstrip("/\\") or "/"
    suffix = content_segments[best_rank[0] :]
    if not suffix:
        return base
    joined = "/".join(suffix)
    return f"/{joined}" if base == "/" else f"{base}/{joined}"


def file_name(name: str) -> str:
    """A path's leaf with case and unicode preserved, for parser and display use."""

    # posixpath on purpose: the separators are already `/`, and on Windows `os.path` reads a leaf whose second
    # character is a colon (`K: Return of Kings - 01.mkv`) as a drive and cuts those two characters off.
    return posixpath.basename(to_forward_slashes(name).rstrip("/"))
