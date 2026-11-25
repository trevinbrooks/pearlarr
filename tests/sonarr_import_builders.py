"""Builders for the Sonarr import tests: a strategy over a scripted Sonarr, its rescans, commands and history."""

from datetime import timedelta

from pearlarr.arr_categories import ArrCategoryResolver
from pearlarr.arr_models import (
    CommandResource,
    HistoryPage,
    Language,
    ManualImportCandidate,
    QualityDefinition,
    QueueRecord,
    SonarrEpisode,
)
from pearlarr.sonarr_strategy import SonarrStrategy

from .builders import PENDING_IMPORT_ADDED, FakeCacheStore, FakeClock, make_config, make_sonarr_sync, sonarr_ep
from .fakes import NEUTRAL_RESCAN, HookedSonarrClient

_RESCAN_NAME = "RefreshMonitoredDownloads"
RESCAN_AFTER_ADD = CommandResource(
    name=_RESCAN_NAME, status="completed", started=PENDING_IMPORT_ADDED + timedelta(days=1)
)
RESCAN_BEFORE_ADD = CommandResource(
    name=_RESCAN_NAME, status="completed", started=PENDING_IMPORT_ADDED - timedelta(days=1)
)


FILELESS_EPISODES = [sonarr_ep(1, number, ep_id=100 + number, episode_file_id=0) for number in range(1, 10)]
"""Episodes 101 to 109 of one season, none with a file."""


def make_sonarr_for_import(
    *,
    candidates: list[ManualImportCandidate] | None,
    queue: list[QueueRecord] | None = None,
    episodes: list[SonarrEpisode] | None = None,
    quality_defs: list[QualityDefinition] | None = None,
    languages: list[Language] | None = None,
    commands: list[CommandResource] | None = None,
    command_status: CommandResource | None = None,
    command_status_script: list[CommandResource] | None = None,
    cmd_id: int | None = 42,
    config_overrides: dict[str, list[str] | str] | None = None,
    cache_store: FakeCacheStore | None = None,
    clock: FakeClock | None = None,
    categories: ArrCategoryResolver | None = None,
) -> tuple[SonarrStrategy, HookedSonarrClient]:
    """A bare `SonarrStrategy` plus its scripted `self.sonarr` `HookedSonarrClient`.

    The fake returns the given queue records, current episodes, ManualImport
    candidates, quality definitions, languages and an execute command id -
    everything `import_completed` reaches over the network - and records the two
    import commands so a test asserts on recorded state. `episodes` defaults to
    `FILELESS_EPISODES`, so the file map episodes have NO file yet (they need importing)
    and the done-check never short-circuits. Pass episodes carrying a file to exercise the
    "already imported" / never-overwrite paths. `queue` defaults to empty (Sonarr
    isn't tracking the download, so the strategy runs an auto import). `commands` lists the
    commands beside the fake's neutral rescan, so no list given here reads as a
    restart. The fake's refresh / command-status defaults resolve immediately so
    the rescan never waits.
    """

    sonarr = HookedSonarrClient(
        queue=queue,
        episodes=FILELESS_EPISODES if episodes is None else episodes,
        commands=None if commands is None else [NEUTRAL_RESCAN, *commands],
        command_status=command_status,
        command_status_script=command_status_script,
        candidates=candidates,
        quality_defs=quality_defs,
        languages=languages,
        execute_command_id=cmd_id,
    )
    overrides: dict[str, list[str] | str] = config_overrides or {}
    strat = make_sonarr_sync(
        sonarr=sonarr,
        config=make_config(**overrides),
        cache_store=cache_store or FakeCacheStore(),
        clock=clock,
        categories=categories,
    )
    return strat, sonarr


def started_process_pass() -> CommandResource:
    """Sonarr's own ProcessMonitoredDownloads command, executing right now."""

    return CommandResource.model_validate({"name": "ProcessMonitoredDownloads", "status": "started"})


def inflight_manual_import(infohash: str, *, status: str = "started", command_id: int = 0) -> CommandResource:
    """A ManualImport command whose one file carries `infohash` as downloadId."""

    return CommandResource.model_validate(
        {
            "id": command_id,
            "name": "ManualImport",
            "status": status,
            "body": {"files": [{"downloadId": infohash, "episodeIds": [101]}]},
        },
    )


def closed_history() -> HistoryPage:
    """A history page whose newest relevant event is a prior full import."""

    return HistoryPage.model_validate(
        {
            "records": [
                {"id": 3, "eventType": "downloadFolderImported", "date": "2026-06-20T06:15:30Z"},
                {"id": 1, "eventType": "grabbed", "date": "2026-06-19T00:00:00Z"},
            ],
        },
    )
