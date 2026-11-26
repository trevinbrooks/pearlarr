# pyright: strict
"""Tests for what a Sonarr import posts for each file.

Each file goes with the pending import's series, episodes, group and hash, the quality Sonarr or the file name says,
and the languages the release's audio calls for. They also cover a sample, a failed fetch, a rejected post, and the
warning of a default quality Sonarr doesn't define.
"""

import unicodedata

import pytest

from pearlarr.arr_models import Language, Quality, QualityDefinition, QualityModel
from pearlarr.output import Severity
from pearlarr.sonarr_strategy import SonarrStrategy
from pearlarr.wait_types import AttemptKind

from .builders import manual_candidate, pending_import
from .fakes import FakeSonarrClient, RecordingHub, diagnostic_messages, install_recording_hub
from .sonarr_import_builders import make_sonarr_for_import, posted_imports


class TestImportCompletedPayload:
    """An import posts each file it finds on disk with the pending import's series, episodes, group and hash.

    The episodes come from pearlarr's file map, never Sonarr's parse, and the languages from the release's audio.
    A file Sonarr calls a sample is not sent.
    """

    def test_the_posted_file_carries_the_pending_imports_series_episodes_group_and_hash(self) -> None:
        # Sonarr's candidate for the file names no series or episodes of its own: they come from the pending import.
        pending = pending_import(
            series_id=7,
            release_group="SubGroup",
            infohash="HASH",
            file_episode_map={"Show - 01 [1080p].mkv": [101]},
        )
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate("/downloads/Show - 01 [1080p].mkv")])

        result = strat.import_completed(pending, "/downloads/Show", AttemptKind.ORDINARY)

        # The copy runs after the post, so the attempt has issued the command and is not done yet.
        assert (result.all_done, result.command_issued) == (False, True)
        [(files, _mode)] = sonarr.execute_calls
        assert [(f.path, f.seriesId, f.episodeIds, f.releaseGroup, f.downloadId) for f in files] == [
            ("/downloads/Show - 01 [1080p].mkv", 7, [101], "SubGroup", "HASH")
        ]

    def test_a_file_holding_two_episodes_is_posted_for_both(self) -> None:
        # A double-length premiere ships as one file that holds two episodes.
        name = "Show - 01-02 [1080p].mkv"
        pending = pending_import(file_episode_map={name: [101, 102]}, seadex_files=[name])
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate(f"/d/{name}")])

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == [[(f"/d/{name}", [101, 102])]]

    def test_a_decomposed_disk_name_still_matches_its_composed_file_map_name(self) -> None:
        # The file map holds the composed e-acute. A macOS mount hands the name back decomposed.
        name = "Café - 01 [1080p].mkv"
        composed, decomposed = unicodedata.normalize("NFC", name), unicodedata.normalize("NFD", name)
        pending = pending_import(file_episode_map={composed: [101]}, seadex_files=[composed])
        strat, sonarr = make_sonarr_for_import(candidates=[manual_candidate(f"/d/{decomposed}")])

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert posted_imports(sonarr) == [[(f"/d/{decomposed}", [101])]]

    def test_sample_candidate_is_skipped(self) -> None:
        pending = pending_import(
            file_episode_map={
                "Show - 01 [1080p].mkv": [101],
                "Show - 01 [1080p].sample.mkv": [101],
            },
            seadex_files=["Show - 01 [1080p].mkv", "Show - 01 [1080p].sample.mkv"],
        )
        good = manual_candidate("/d/Show - 01 [1080p].mkv")
        sample = manual_candidate(
            "/d/Show - 01 [1080p].sample.mkv",
            rejections=[{"reason": "Sample"}],
        )
        strat, sonarr = make_sonarr_for_import(candidates=[good, sample])

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        # The good file's import command was issued.
        # The sample is never sent.
        assert result.all_done is False
        assert result.command_issued is True
        paths = [f.path for f in sonarr.execute_calls[0][0]]
        assert paths == ["/d/Show - 01 [1080p].mkv"]

    def test_languages_follow_dual_audio_flag(self) -> None:
        pending = pending_import(is_dual_audio=True)
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        languages = [Language(id=1, name="English"), Language(id=2, name="French"), Language(id=8, name="Japanese")]
        strat, sonarr = make_sonarr_for_import(
            candidates=[candidate],
            languages=languages,
            config_overrides={"import_languages_dual": ["Japanese", "French"]},
        )

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        assert result.all_done is False
        assert result.command_issued is True
        names = [lang.name for lang in sonarr.execute_calls[0][0][0].languages]
        assert names == ["Japanese", "French"]


class TestPostedQuality:
    """The posted quality is Sonarr's parse, else the file name's, and Unknown with a warning when neither says one."""

    @pytest.mark.parametrize(
        ("sonarr_parsed", "posted"),
        [
            pytest.param(
                {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
                "Bluray-1080p",
                id="sonarrs-parse-wins-over-the-file-name",
            ),
            pytest.param(
                {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0},
                "WEBDL-1080p",
                id="the-file-name-fills-an-unknown",
            ),
        ],
    )
    def test_the_posted_quality_is_sonarrs_parse_else_the_file_names(
        self, sonarr_parsed: dict[str, object], posted: str
    ) -> None:
        name = "Show - 01 [1080p][WEB-DL].mkv"
        pending = pending_import(file_episode_map={name: [101]}, seadex_files=[name])
        qualities = (
            {"id": 3, "name": "WEBDL-1080p", "source": "web"},
            {"id": 7, "name": "Bluray-1080p", "source": "bluray"},
        )
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}", quality={"quality": sonarr_parsed})],
            quality_defs=[QualityDefinition.model_validate({"quality": {**q, "resolution": 1080}}) for q in qualities],
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        quality = sonarr.execute_calls[0][0][0].quality
        assert quality is not None
        assert quality.quality is not None
        assert quality.quality.name == posted

    def test_a_file_of_no_known_quality_is_posted_as_unknown_with_a_warning(self) -> None:
        # Neither Sonarr's parse nor the file name says a quality, and no default is set.
        recording = install_recording_hub()
        name = "Show - 01.mkv"
        pending = pending_import(file_episode_map={name: [101]}, seadex_files=[name])
        unknown = {"id": 0, "name": "Unknown", "source": "unknown", "resolution": 0}
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate(f"/d/{name}", quality={"quality": unknown})]
        )

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        ((files, _),) = sonarr.execute_calls
        assert [(file.path, file.episodeIds, file.quality) for file in files] == [
            (f"/d/{name}", [101], QualityModel(quality=Quality(id=0, name="Unknown", source="unknown", resolution=0)))
        ]
        assert diagnostic_messages(recording, Severity.WARNING) == [
            "/d: could not confidently resolve quality for Show - 01.mkv - importing as Unknown (re-grab risk)"
        ]


class TestDefaultQualityWarning:
    """An unmatched `imports.default_quality` warns once per run, in the executor that uses it.

    `quality_parts_from_name` stays pure (its silent-empty return is pinned
    elsewhere). The executor is where the configured name meets the run's real
    Sonarr definitions, and it runs once per FILE, hence the once-per-run guard.
    """

    @staticmethod
    def _strat_with_default(name: str | None) -> tuple[SonarrStrategy, FakeSonarrClient, RecordingHub]:
        # The candidate + definitions resolve cleanly, so the only quality-related
        # warning in play is the unmatched-default one.
        quality = {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}}
        overrides: dict[str, list[str] | str] = {} if name is None else {"import_default_quality": name}
        strat, sonarr = make_sonarr_for_import(
            candidates=[manual_candidate("/d/Show - 01 [1080p].mkv", quality=quality)],
            quality_defs=[QualityDefinition.model_validate(quality)],
            config_overrides=overrides,
        )
        return strat, sonarr, install_recording_hub()

    @staticmethod
    def _default_quality_warnings(recording: RecordingHub) -> list[str]:
        return [message for message in diagnostic_messages(recording, Severity.WARNING) if "default_quality" in message]

    def test_unmatched_name_warns_once_per_run(self) -> None:
        pending = pending_import()
        strat, sonarr, recording = self._strat_with_default("Blueray-1080p")

        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)

        # Both polls ran the auto import (rebuilt the payload), yet the typo warned once.
        assert len(sonarr.execute_calls) == 2
        assert self._default_quality_warnings(recording) == [
            "imports.default_quality 'Blueray-1080p' matches no Sonarr quality definition - ignoring it",
        ]

        # The next arr run warns of it again.
        strat.get_items()
        strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert len(self._default_quality_warnings(recording)) == 2

    @pytest.mark.parametrize("name", [pytest.param("WEBDL-1080p", id="matched"), pytest.param(None, id="unset")])
    def test_a_matched_or_unset_name_does_not_warn(self, name: str | None) -> None:
        strat, sonarr, recording = self._strat_with_default(name)

        strat.import_completed(pending_import(), "/d", AttemptKind.ORDINARY)

        assert len(sonarr.execute_calls) == 1
        assert self._default_quality_warnings(recording) == []


class TestImportCompletedFailures:
    """A failed read of the download's files or a rejected post leaves the attempt waiting for the next one."""

    def test_a_failed_candidate_fetch_waits_and_calls_no_file_missing(self) -> None:
        # A fetch that failed says nothing of the folder, where an empty answer says the file isn't there.
        recording = install_recording_hub()
        strat, sonarr = make_sonarr_for_import(candidates=None)

        result = strat.import_completed(pending_import(), "/d", AttemptKind.FINAL)

        assert (result.all_done, sonarr.execute_calls) == (False, [])
        assert diagnostic_messages(recording, Severity.WARNING) == []

    def test_failed_execute_retries(self) -> None:
        # Sonarr rejected the import command (busy / locked) -> retry, not give up.
        pending = pending_import()
        candidate = manual_candidate("/d/Show - 01 [1080p].mkv")
        strat, sonarr = make_sonarr_for_import(candidates=[candidate], cmd_id=None)

        result = strat.import_completed(pending, "/d", AttemptKind.ORDINARY)
        assert result.all_done is False
        assert result.command_issued is False
        # The execute WAS attempted (Sonarr rejected it -> command_issued False).
        # A regression returning early without even trying would still set False here.
        assert len(sonarr.execute_calls) == 1
