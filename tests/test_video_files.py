# pyright: strict
"""Which listed or folder files could be an episode, the rule the file map, the import and the parse share."""

import pytest

from pearlarr.video_files import is_video_candidate


class TestIsVideoCandidate:
    """A video file could be an episode unless it is a creditless opening or ending. No other file could."""

    @pytest.mark.parametrize(
        ("name", "candidate"),
        [
            pytest.param("[X] Show - S01E01.mkv", True, id="mkv"),
            pytest.param("[X] Show - S01E01.mp4", True, id="mp4"),
            pytest.param("[X] Show - NCOP 01.mkv", False, id="creditless"),
            # Checksum and sidecar files, all seen in real SeaDex releases. A seeded checksum file is never offered
            # among Sonarr's ManualImport candidates, so it would stick a pending import on "intended file missing"
            # until the deadline.
            pytest.param("[Marin] Show - S01 [BD 1080p].blake3", False, id="blake3"),
            pytest.param("checksum.md5", False, id="md5"),
            pytest.param("checksum.sha2", False, id="sha2"),
            pytest.param("checksum.sha256", False, id="sha256"),
            pytest.param("[Dae] Show - S01E01.mks", False, id="mks"),
            pytest.param("Animate Benefits - 01.tif", False, id="tif"),
            pytest.param("playlist.m3u8", False, id="m3u8"),
            # Manga volumes and books sit beside the video in some releases.
            pytest.param("[X] Show - Volume 01.cbz", False, id="cbz"),
            pytest.param("[X] Show - Volume 02.cbr", False, id="cbr"),
            pytest.param("[X] Show - Volume 03.cb7", False, id="cb7"),
            pytest.param("[X] Show - Guidebook.pdf", False, id="pdf"),
            pytest.param("[X] Show - Novel 01.epub", False, id="epub"),
            # A file keeps the case its maker gave its name.
            pytest.param("ARIALUNI.TTF", False, id="an-upper-case-extension"),
        ],
    )
    def test_only_a_video_file_could_be_an_episode(self, name: str, candidate: bool) -> None:
        assert is_video_candidate(name) == candidate
