# pyright: strict
"""Which listed or folder files could be an episode, the rule the file map, the import and the parse share."""

import pytest

from pearlarr.video_files import is_video_candidate


class TestIsVideoCandidate:
    """Checksum and sidecar files never enter the candidate pool.

    A seeded checksum file is never offered among Sonarr's ManualImport candidates,
    so it would stick a pending import on "intended file missing" until the deadline.
    All of these extensions were observed in real SeaDex releases.
    """

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("[Marin] Show - S01 [BD 1080p].blake3", id="blake3"),
            pytest.param("checksum.md5", id="md5"),
            pytest.param("checksum.sha2", id="sha2"),
            pytest.param("checksum.sha256", id="sha256"),
            pytest.param("[Dae] Show - S01E01.mks", id="mks"),
            pytest.param("Animate Benefits - 01.tif", id="tif"),
            pytest.param("playlist.m3u8", id="m3u8"),
        ],
    )
    def test_checksum_and_sidecar_extensions_rejected(self, name: str) -> None:
        assert not is_video_candidate(name)

    # Manga volumes and books sit beside the video in some releases.
    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("[X] Show - Volume 01.cbz", id="cbz"),
            pytest.param("[X] Show - Volume 02.cbr", id="cbr"),
            pytest.param("[X] Show - Volume 03.cb7", id="cb7"),
            pytest.param("[X] Show - Guidebook.pdf", id="pdf"),
            pytest.param("[X] Show - Novel 01.epub", id="epub"),
        ],
    )
    def test_bundled_reading_matter_rejected(self, name: str) -> None:
        assert not is_video_candidate(name)

    def test_an_upper_case_extension_is_rejected_too(self) -> None:
        # A file keeps the case its maker gave its name.
        assert not is_video_candidate("ARIALUNI.TTF")

    @pytest.mark.parametrize(
        "name", [pytest.param("[X] Show - S01E01.mkv", id="mkv"), pytest.param("[X] Show - S01E01.mp4", id="mp4")]
    )
    def test_video_containers_kept(self, name: str) -> None:
        assert is_video_candidate(name)
