# pyright: strict
"""Direct tests for the path and name helpers: the basename and group normalizers, path leaves, and download path mapping."""

from pearlarr.arr_models import RemotePathMapping
from pearlarr.file_paths import file_name, normalize_name, normalized_leaf, translate_download_path
from pearlarr.releases import normalize_group, normalize_rg


class TestNormalize:
    """`normalize_basename`/`normalize_group` casefold and strip for stable comparison.

    Interior dashes in a group name are never stripped, only wrapping ones.
    """

    def test_nfc_nfd_match(self) -> None:
        # Same text, NFC (composed) vs NFD (decomposed) "é". Both fold equal.
        nfc = "Café - 01.mkv"
        nfd = "Café - 01.mkv"
        assert normalize_name(nfc) == normalize_name(nfd)

    def test_strips_and_casefolds(self) -> None:
        assert normalize_name("  Show - 01.MKV  ") == "show - 01.mkv"

    def test_folding_a_folded_key_gives_the_same_key(self) -> None:
        # MUTATION PIN: casefold expands ß to "ss" and leaves the mark after it hanging off the second s,
        # uncomposed, until the NFC after the casefold composes it. A stored record folds its keys again on
        # every load, so without that NFC the second pass would re-key it.
        name = "Der Weiß\u0323e Ritter - 01.mkv"
        once = normalize_name(name)
        assert once == "der weis\u1e63e ritter - 01.mkv"
        assert normalize_name(once) == once

    def test_canonically_equal_names_fold_alike_when_casefold_turns_a_mark_into_a_letter(self) -> None:
        # MUTATION PIN: casefold turns U+0345 (the Greek iota below) into a letter, so the NFC before it is what
        # makes these two spellings of one name fold alike. With only the NFC after it, they fold apart.
        assert normalize_name("\u1fb4 - 01.mkv") == normalize_name("\u03b1\u0345\u0301 - 01.mkv")

    def test_windows_paths_and_trailing_separators_still_fold_to_a_leaf(self) -> None:
        # MUTATION PIN: a Windows arr's paths reach pearlarr verbatim, where posixpath.basename
        # sees no separator at all and would key the whole path.
        assert normalized_leaf("C:\\downloads\\Show\\Show - 01.mkv") == "show - 01.mkv"
        # And a directory entry never folds to the empty leaf, which every other
        # unnamed thing would then collide with.
        assert normalized_leaf("Show/NC/") == "nc"

    def test_path_leaf_folds_separators_but_preserves_the_name(self) -> None:
        # The parser-input twin of normalized_leaf: same separator folding, but
        # case/unicode intact - Sonarr's /parse and the video gate must see the
        # real filename, never a whole Windows path or a folded one.
        assert file_name("C:\\downloads\\Show\\Show - 01 [1080p].MKV") == "Show - 01 [1080p].MKV"
        assert file_name("Show/NC/") == "NC"

    def test_path_leaf_keeps_a_leaf_that_looks_like_a_drive(self) -> None:
        # MUTATION PIN: catches a file_name on os.path only on the Windows CI leg, where os.path is ntpath and
        # reads the leading "K:" as a drive, so the grab's key for the file never matches the one Sonarr's path
        # gives.
        assert file_name("K: Return of Kings - 01.mkv") == "K: Return of Kings - 01.mkv"

    def test_group_casefold(self) -> None:
        assert normalize_group("SubGroup") == normalize_group("subgroup")

    def test_group_strip_only_removes_wrapping_dashes(self) -> None:
        # MUTATION PIN: strip("-") widened to a multi-char strip set would eat
        # the X off an X-edged group. Only wrapping dashes (and whitespace) go,
        # and interior dashes always stay.
        assert normalize_group("Xrays-") == "xrays"
        assert normalize_group("X-Raws") == "x-raws"

    def test_group_dash_wrapped_agrees_with_planner(self) -> None:
        # normalize_group is the single source of truth normalize_rg delegates to.
        # A dash-wrapped group must compare equal on both ends or a release the
        # planner grabbed gets re-imported over by the overwrite guard.
        assert normalize_group("-Aergia-") == "aergia"
        assert normalize_group("-Aergia-") == normalize_rg("-Aergia-")
        assert normalize_group("Aergia") == normalize_rg("-Aergia-")


def _mapping(remote: str, local: str, *, host: str | None = None) -> RemotePathMapping:
    """One remote path mapping from the raw API field names."""

    return RemotePathMapping.model_validate({"host": host, "remotePath": remote, "localPath": local})


class TestTranslateDownloadPath:
    """The remote-path translation behind the folder-scan fallback."""

    def test_no_mappings_is_a_no_op(self) -> None:
        assert translate_download_path("/d/folder", [], "qbit") == "/d/folder"

    def test_prefix_translates_and_suffix_survives(self) -> None:
        # The live incident mapping: trailing slash on both stored paths.
        mappings = [_mapping("/home/u/torrents/4k-tv/", "/remote/torrents/4k-tv/")]
        assert (
            translate_download_path("/home/u/torrents/4k-tv/Show S01", mappings, None)
            == "/remote/torrents/4k-tv/Show S01"
        )

    def test_exact_match_translates_to_local_root(self) -> None:
        mappings = [_mapping("/downloads", "/data")]
        assert translate_download_path("/downloads", mappings, None) == "/data"

    def test_separator_boundary_is_respected(self) -> None:
        # /downloads must NOT prefix-match /downloads-x/f.
        mappings = [_mapping("/downloads", "/data")]
        assert translate_download_path("/downloads-x/f", mappings, None) == "/downloads-x/f"

    def test_trailing_slash_tolerated_on_either_side(self) -> None:
        assert translate_download_path("/d/f", [_mapping("/d/", "/l")], None) == "/l/f"
        assert translate_download_path("/d/f", [_mapping("/d", "/l/")], None) == "/l/f"

    def test_suffix_case_is_preserved(self) -> None:
        # Compare case-insensitively but never fold the suffix: POSIX targets
        # are case-sensitive.
        mappings = [_mapping("/Downloads", "/data")]
        assert translate_download_path("/downloads/Show S01/Ep.MKV", mappings, None) == "/data/Show S01/Ep.MKV"

    def test_windows_backslash_remote_path(self) -> None:
        mappings = [_mapping("C:\\torrents\\", "/data/torrents")]
        assert translate_download_path("C:\\torrents\\Show\\ep.mkv", mappings, None) == "/data/torrents/Show/ep.mkv"

    def test_longest_prefix_wins(self) -> None:
        mappings = [
            _mapping("/d", "/short"),
            _mapping("/d/tv", "/long"),
        ]
        assert translate_download_path("/d/tv/Show", mappings, None) == "/long/Show"

    def test_host_equality_tiebreaks_equal_prefixes(self) -> None:
        mappings = [
            _mapping("/d", "/other-client", host="other"),
            _mapping("/d", "/ours", host="qbit.local"),
        ]
        assert translate_download_path("/d/Show", mappings, "QBIT.LOCAL") == "/ours/Show"

    def test_host_mismatch_never_excludes(self) -> None:
        # Sonarr's host is the download-client host as SONARR knows it -
        # routinely a different string from our qBittorrent host.
        mappings = [_mapping("/d", "/data", host="sonarr-view-of-qbit")]
        assert translate_download_path("/d/Show", mappings, "localhost") == "/data/Show"

    def test_longer_prefix_beats_host_match(self) -> None:
        mappings = [
            _mapping("/d", "/host-matched", host="qbit"),
            _mapping("/d/tv", "/longer", host="other"),
        ]
        assert translate_download_path("/d/tv/Show", mappings, "qbit") == "/longer/Show"

    def test_mapping_missing_either_path_is_skipped(self) -> None:
        mappings = [_mapping("", "/data"), _mapping("/d", "")]
        assert translate_download_path("/d/Show", mappings, None) == "/d/Show"

    def test_single_file_content_path_translates(self) -> None:
        # A single-FILE torrent's content_path is the file itself.
        mappings = [_mapping("/d/", "/data/")]
        assert translate_download_path("/d/Show - 01.mkv", mappings, None) == "/data/Show - 01.mkv"
