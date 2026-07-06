1.0.0 (Unreleased)
==================

- **Breaking:** the config schema is now nested groups (``sonarr:``, ``radarr:``, ``qbittorrent:``, ``seadex:``, ``imports:``, ``notifications:``, ``mappings:``, ``advanced:``) with strict validation - unknown or flat top-level keys are rejected, and constrained fields (``imports.mode``, ``advanced.log_level``, ``seadex.private_releases``) reject typos at load. ``advanced.log_level: ERROR`` is now honored as a real level for the first time (previously the logger silently treated it as INFO). Rewrite an existing ``config.yml`` against the new layout (``seadexarr config init`` writes a commented template); e.g. ``sonarr_url`` is now ``sonarr.url``, ``qbit_info`` is ``qbittorrent.*``, ``torrent_tags`` is ``qbittorrent.tags``
- Store config, caches and logs together in one OS-standard data directory (via ``platformdirs``); override with ``SEADEX_ARR_DATA_DIR`` or the global ``--data-dir`` flag, and run ``seadexarr paths`` to print the resolved locations. **Breaking:** the default location moved off the install directory (e.g. ``~/.local/share/seadexarr`` on Linux, ``~/Library/Application Support/seadexarr`` on macOS); move an existing ``config.yml``/cache into the new directory, or set ``SEADEX_ARR_DATA_DIR``. Logs now follow the data directory instead of the working directory
- The persistent cache moved from ``cache.json`` to a SQLite ``cache.db``, with a one-time automatic migration on first run (the legacy file is kept alongside as ``cache.json.migrated``); the parsed ID-mapping sources are likewise cached in a ``mappings.db``, so warm runs skip the re-parse
- Add wait-for-completion and Sonarr manual import (``imports.wait_mode``: off/deferred/blocking/hybrid): optionally wait for qBittorrent to finish grabbed torrents, let Sonarr import them, and step in with a manual import when it can't - with optional completion notifications (``notifications.wait_notify``)
- Add ``notifications.wait_webhook_url``: a generic outbound webhook (e.g. ntfy/gotify/Home Assistant) for the wait-pass summary, alongside the Discord webhook
- Add ``imports.post_import_category``: once a torrent's import is verified complete (so a non-off ``imports.wait_mode`` is required), move it to this qBittorrent category (created if missing), e.g. to hand finished torrents different seeding rules
- Private releases are never grabbed: prefer public URLs per release group; if the only option for a release the Arr is missing is private, log a warning and skip the title (leaving it uncached so it's retried) instead of grabbing a private release
- **Breaking:** ``seadex.public_only`` is replaced by ``seadex.private_releases`` (``warn`` / ``fallback``): ``warn`` (the default) warns and skips when the preferred release is only available privately, and ``fallback`` grabs the entry's best public alternative instead, warning only when no public alternative can be found
- In ``fallback`` mode a public substitute never replaces a copy of the preferred private release you already own: when the Arr holds it at a stale size (SeaDex's record changed), the title now warns and holds every run - the summary tip names both ways out (update it from its tracker, or delete the stale files) - instead of grabbing the fallback over it. A preferred public release still supersedes as before
- Titles satisfied by a public fallback are remembered in the cache (a ``fallback_satisfied`` marker); switching ``seadex.private_releases`` back to ``warn`` re-checks them and resurfaces the private-only warning, instead of leaving them cache-skipped until SeaDex updates the entry
- Where multiple preferred release groups cover the exact same files, only download one (preferring a public release), rather than grabbing them all
- Add ``seadex.ignore_anilist_ids``, which allows you to skip specific AniList IDs from being processed
- Add ``qbittorrent.tags``, which allows you to tag torrents as added to qBittorrent
- Add ``seadex.ignore_tags``, which allows you to filter out SeaDex releases by tag
- Use AniBridge mappings to mop up missed Sonarr/Radarr titles
- Discord notifications are now rich embeds with colors, links and a version footer, and the console shows a live cockpit during startup and the import-wait pass (spinners, ticking timers, files-imported progress)
- **Breaking:** Python 3.13+ is now required (3.12 dropped); CI tests 3.13 and 3.14, and the Docker image runs 3.14. Runtime dependencies bumped to latest (notably typer 0.26, rich 15, qbittorrent-api 2026.6, requests 2.34)
- **Breaking (scripting API):** the run machinery is split into a ``RunServices`` per-id hub (which the strategies hold) and a run loop: build ``RunDeps.build`` → ``RunServices(deps, arr)`` → ``RunLoop(deps, services)`` and call ``run_sync`` without the ``arr`` kwarg; close resources via ``deps.close()`` (``SeaDexArr.close`` is gone). The old ``SeaDexArr`` class is now ``RunLoop`` (module ``seadex_arr`` → ``run_loop``). See the README scripting example
- New CLI surface: ``run single`` gains ``--dry-run`` (simulate a run without grabbing torrents, writing the cache, or notifying), ``--movie-id``/``--series-id`` for single-title runs (by TMDB/TVDB ID, implying the matching module), and ``--import-wait-mode`` to override the configured ``imports.wait_mode`` for that run; new ``cache stats`` and ``cache check`` commands report cache health and integrity
- Failed CLI commands now exit non-zero (previously always 0), and a missing cache/backup file is reported as a one-line message instead of a traceback
- ``cache backup`` writes the snapshot via a temp file, so a failed backup can never replace or delete a previous good ``cache.backup.db``; ``cache restore`` now copies the backup into place instead of consuming it, so a restore is repeatable
- ``config init`` refuses to overwrite an existing ``config.yml`` unless ``--force`` is passed
- Detect Sonarr/Radarr-side file changes between runs (``advanced.detect_arr_activity``, on by default): each pass polls the arr's history once and re-checks just the entries whose files were imported or deleted arr-side since the last pass, so a quality upgrade or manual grab under an unchanged SeaDex entry is re-evaluated without waiting for SeaDex to update it. The first scan covers the last 30 days, and a coverage gap (stopped longer than that) re-checks everything once
- The scheduled-run cadence is now a config field, ``schedule.interval_hours`` (default 6), re-read each cycle so an edit takes effect without a restart; the ``SCHEDULE_TIME`` env var is deprecated but still wins when set
- An invalid ``SCHEDULE_TIME`` falls back to the configured ``schedule.interval_hours`` with a report instead of crashing the scheduler, and ``run single`` with no module selected prints a usage hint and fails

0.9.0 (2025-09-13)
==================

- Include PlexAniBridge-Mappings to mop up some missed titles
- Update cache if version/config changes
- Add option to ignore SeaDex update time
- Don't recreate cache on config change
- Add a number of useful CLI commands
- Include option to just check torrents by hash
- Update cache if no suitable releases found
- Check file sizes, for different versions of releases etc.

0.8.1 (2025-09-05)
==================

- Fix cache not updating with versions

0.8.0 (2025-09-05)
==================

- If we're ignoring Radarr movies in Sonarr, also check the cache
- Do a more proper check for episodes in Sonarr
- Ensure docker-compose run also uses cache
- Fix crash if AniList ID isn't already in cache
- Include AniList name in cache
- Sort cache by AniList ID
- Revert removing trackers

0.7.0 (2025-08-24)
==================

- Create a cache to avoid checked entries that haven't updated
- Fix grabbing multiple releases when there's a mismatch in episode parsing
- Removed trackers that aren't used by SeaDex
- Add support for RuTracker

0.6.0 (2025-08-13)
==================

- Take Ja-only releases if ``prefer_dual_audio`` is False
- Catch crash if SeaDex is unreachable
- Cleanup dictionaries
- Search specifically in qBittorrent by torrent hash, to speed up hash
  checks
- Skip adding downloads to torrent client if download flag not set
- Fix bug where maximum number of torrents added was not respected

0.5.0 (2025-08-08)
==================

- Fix UTF-8 encoding warning in log
- Move to episode-based filtering of torrents for Sonarr
- Include SeaDex tags in log and Discord messages
- Add options to ignore unmonitored series/movies
- Save logs to file
- Ensure SCHEDULE_TIME is brought in as a float
- Fix Discord messages not getting pushed if no torrent client selected
- Fix adding too many torrents in one go

0.4.1 (2025-07-31)
==================

- Add PyYAML to pyproject.toml
- Added support for AnimeTosho

0.4.0 (2025-07-31)
==================

- Added ignore_movies_in_radarr for SeaDexSonarr, which will skip movies flagged as Specials in Sonarr that already
  exist in Radarr
- More robust config when parameters added
- Build Docker for every main update
- Fix crash when AniDB mapping contains no text
- Use IMDb for finding AniList mappings as well
- Initial support for other trackers
- Better handle Discord notifications and log messages when torrent already in client
- Map potentially weird episodes SeaDexSonarr if in Season 0 (Specials)

0.3.0 (2025-07-30)
==================

- Added scheduling in Docker mode

0.2.0 (2025-07-30)
==================

- Add Docker support
- Move to config files, to make the call simpler
- Fix crash if torrent in list but not already downloaded

0.1.0 (2025-07-22)
==================

- Add support for Radarr

0.0.3 (2025-07-22)
==================

- Rename from seadex_sonarr to seadexarr, in preparation for Radarr support
- Add interactive mode, for selecting when multiple "best" options are found
- Add support for adding torrents to qBittorrent

0.0.2 (2025-07-13)
==================

- Improved Discord messaging
- Catch the case where we don't find any suitable SeaDex releases
- Include potentially weird offset mappings via AniDB lists
- Add a rest time to not hit AniList rate limiting

0.0.1 (2025-07-12)
==================

- Initial release
