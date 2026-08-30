# Changelog

All notable, user-observable changes to Pearlarr are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pearlarr is a fork of [bbtufty/seadexarr](https://github.com/bbtufty/seadexarr). Everything up to and including 0.9.0 is inherited upstream history.

## [Unreleased]

## [1.2.0] - 2026-08-30

### Upgrade notes

- **Config files are now migrated on disk automatically.** The first run rewrites an old-schema `config.yml` at the current layout, keeping the previous file as `config.yml.bak`. The rewrite regenerates the file from the annotated template with your values filled in: hand-written comments are not carried over and survive only in `config.yml.bak` - until a later schema migration overwrites that backup in turn. A dry run never rewrites.
- **Downgrading needs the backup.** The rewrite stamps the file with a schema version older Pearlarr releases refuse to load; to go back, restore `config.yml.bak` over `config.yml` first.

### Added

- Runs that do not hold the blocking wait (deferred wait mode, and every Radarr run with waiting on) now end with a visible check pass over carried-over downloads. It imports what is ready under the same rules as the blocking wait, and the run summary counts each remaining download by its actual state (queued, downloaded, or imported) instead of assuming queued.

### Changed

- An old-schema config file is now migrated in place at the start of a run, keeping the previous file as `config.yml.bak`, instead of warning on every run. `pearlarr config migrate` remains for migrating without a run.
- An omitted `torrent_category` or `post_import_category` now adopts the matching category of that arr's qBittorrent download client (its grab and imported categories respectively); an explicit value still wins, and an explicit blank (`""`) keeps no category at all.
- The post-import category move now runs before the queue cleanup, and when the effective post-import category takes the torrent out of the arr's watched download category the explicit queue removal is skipped: the leftover entry clears on its own, without Sonarr recording a permanent "download ignored" history row. Setups without an effective post-import category keep the explicit removal under `imports.remove_from_queue`.
- Post-import cleanup now retries transient failures within the run and keeps the download's record on give-up so the next run finishes it, instead of losing the category move or queue removal to a one-shot attempt.

### Fixed

- Overwrite-protection evidence is now kept once per entry instead of copied into each grabbed torrent's record, so torrents of one entry seeded across different runs can no longer disagree at import time about which on-disk files to protect.
- A successful import no longer warns `Could not remove the finished download from Sonarr's queue (status code 500)` when Sonarr never matched the download's title to a series. Dismissing such a queue entry always fails inside Sonarr and records nothing, so Pearlarr now leaves it alone; the entry clears on its own once the imported torrent leaves Sonarr's watched category.
- A Sonarr queue outage during the wait no longer risks a mistimed manual import. A failed queue read now retries on later polls and defers the download at the deadline.
- A torrent re-added to qBittorrent no longer resets its pending-import age on every run, so stale pending records expire on schedule (`imports.pending_max_age_days`).
- A queue entry Sonarr already dismissed no longer draws a removal warning. It now counts quietly as done.
- A deferred post-import cleanup now re-verifies the import before touching the torrent or the queue. If the imported files vanished since the import was verified, the download is re-imported instead of cleaned up; only a torrent gone from qBittorrent is still cleaned up unverified.
- A transient category lookup failure during cleanup can no longer drop a download's record with both the category move and the queue removal silently skipped. Both effects now share one lookup.

## [1.1.0] - 2026-07-29

### Changed

- The wait-and-import subsystem is now on by default: `imports.wait_mode` defaults to `hybrid` instead of `off`. Set `imports.wait_mode: off` to keep it disabled; a config that already sets a mode is unaffected. `pearlarr config migrate` calls out the flip for configs that relied on the old default.
- `imports.post_import_category` is now set per arr, as `sonarr.post_import_category` and `radarr.post_import_category`, next to each arr's grab-time `torrent_category`. `pearlarr config migrate` moves an existing value onto both arrs, keeping the old behavior; until then it is applied in memory at every load.

### Fixed

- Discord grab notifications no longer repeat the series title above the notes when the entry title merely extends it with a season suffix (for example a grab titled `Show Season 2` no longer carries a `Show` byline). A genuinely different series title still shows.

- A release listed with different folder layouts on two trackers (one nesting extras in a folder the other lists flat) is now recognized as one release. Listings are compared by file size rather than by listing path, so fallback mode (`seadex.private_releases: fallback`) no longer re-grabs a batch the library already holds, a cross-seeded copy of a release is no longer grabbed twice, and a listing SeaDex named differently from its twin no longer reports a release the library has as needing action.
- An owned file whose release-group tag a rename scheme stripped now counts as the recommended release when its size matches the listed file exactly (movies: when every untagged file on disk belongs to the listing). Such files previously read as foreign, so a re-check could re-download releases the library already held - including from a different recommended release than the one on disk. Zero or unreadable sizes never match, so a failed zero-byte copy is still repaired.
- A pack import no longer replaces on-disk files that belong to another of the entry's recommended releases, or files a rename left untagged that the grab decision had already identified as that release. The never-overwrite guard only knew the groups Pearlarr itself had grabbed, so a pack import could overwrite a recommended copy that predated Pearlarr. Protection needs size evidence: a release the library holds at an outdated size on any of its episodes is still replaced, and a listing that hides its file sizes neither protects a copy nor condemns one. The opt-in `seadex.use_torrent_hash_to_filter` mode compares no sizes and keeps its grabbed-groups-only guard.
- A same-group size upgrade Sonarr refuses to import (typically over custom-format score) is now imported by Pearlarr over the old files. The old copy carries the same release group as the new one, so the import read it as already done and the run reported imported with the outdated files still in place - and the queue entry removed. Old and new files are now told apart by size against the release's current listing.

## [1.0.9] - 2026-07-28

### Fixed

- Pearlarr again steps in for a download Sonarr refuses to import (for example a recommended release whose custom-format score is below the existing files'). The disk-command deferral added in 1.0.8 was checked right after Pearlarr's own queue rescan - and completing a rescan is exactly what starts Sonarr's next monitored-download pass, so the check deferred on every poll and the download sat blocked until the wait gave up. The rescan now waits for that pass to finish before deciding.
- The import wait no longer abandons a download mid-copy while its own import command is still running, and time spent queued behind another Pearlarr import no longer counts against `imports.ready_timeout`. Imports run one at a time in Sonarr, so with several finishing together the later ones could burn their whole ready window waiting their turn and walk away just as (or even before) their copy ran - reported as `import not ready` when a copy was in fact in flight. The exemption is capped at six ready windows, and waiting on Sonarr's own or a foreign command still counts, so even a wedged command ends in the bounded walk-away rather than an indefinite wait.
- A torrent that carries more than one entry's episodes (a multi-cour pack imported as separate slices) now shows a real files-inserted bar, and each landing file extends the ready deadline as it does for any other pack. Files belonging to the other entry made the whole record count as unmapped, leaving the wait row an indeterminate spinner whose deadline could cut the import off mid-copy.

## [1.0.8] - 2026-07-21

### Fixed

- `imports.ready_timeout` now counts from the last file seen landing instead of the moment the download finished, so a season pack Sonarr imports file-by-file is no longer abandoned mid-copy. Previously the wait could report `0 imported · 1 left` while a healthy import was minutes from done - the episodes then landed anyway, contradicting the notification.
- Pearlarr no longer steps in while Sonarr is executing an import pass or another disk command (a rename or move sweep). Sonarr holds a manual import POSTed meanwhile and replays it verbatim once the blocker finishes, re-copying every file its own pass had placed in between.
- A torrent imported across several passes (some files by Sonarr, the rest by Pearlarr) no longer lingers in Sonarr's queue afterwards, where completed-download handling could one day re-import it. Sonarr clears an entry itself only when a single import covers the grab's full episode count; once every import using the torrent has completed, Pearlarr now removes the leftover queue entry: Sonarr records the download as manually ignored, and the torrent keeps seeding. New `imports.remove_from_queue` (default on) turns this off.

## [1.0.7] - 2026-07-21

### Fixed

- Manual imports now register against Sonarr's download tracking (the command's `downloadId` was sent lowercase; Sonarr keys tracked downloads by the uppercased hash), so Sonarr sees the queue item as handled. Previously its completed-download handling would later re-import the same torrent - deleting the just-imported file and copying an identical one over it.
- A completed download whose Sonarr import attempt failed (queue state `importPending` with a warning - typically the file not yet visible on Sonarr's mount) is now imported by Pearlarr on the next poll instead of waited on for the full `imports.ready_timeout`. Sonarr does not reliably retry such downloads, so the wait only delayed the import by the whole timeout.
- Wait rows, warnings, and completion notifications now name each torrent's own episodes (for example `Show · Group · S02 E06`). Per-episode grabs from one release previously all rendered the identical `Show · Group` label, so a notification could not say which episodes imported and which were left for a later run.

## [1.0.6] - 2026-07-18

### Changed

- `imports.post_import_category` now also applies to Radarr grabs, moving a movie's torrent once Radarr has imported it. Previously the setting was Sonarr-only.

### Fixed

- A torrent that SeaDex lists on both a Sonarr entry and a Radarr movie is no longer moved out of Radarr's qBittorrent category before Radarr imports it. Radarr filters its queue by category, so the early move made the movie vanish from Radarr and never import.
- Torrents that SeaDex lists on several AniList entries (multi-cour batches) now import every entry's episodes; previously one entry's tracking record silently replaced the other's and that slice was never imported.
- `imports.post_import_category` is applied only after ALL entries sharing a torrent have imported, so cleanup scripts keyed on the category can no longer delete data another entry still needs.
- Absolute-numbered batches now import even when the grab-time mapping is missing and the folder holds files outside the entry (a sibling cour, a special, an NC extra). Files place by Sonarr's own series-matched resolution - scoped to the entry and cross-checked against episode ids - instead of an all-or-nothing count a single stray file could veto. Ambiguous names are refused and retried, never guessed.
- The folder-scan fallback no longer warns "will retry" on every poll of a download it is about to handle: the by-id scan fails quietly, the dated history note explains the fallback, and a warning fires only for a genuinely stuck download.
- Warnings and notes emitted while a series is processing now indent with the entry listing instead of breaking it at the left margin.
- Movies whose year parses as an episode number ("Chronicle.2020" read as S20E20) now import to the entry's sole resolved episode when the parsed episode provably does not exist in the series.
- All-numberless specials batches ("Special 1".."Special N") now import in order when the file count exactly matches the entry's remaining episodes; any numbered, seeded, or ambiguous file keeps the whole-batch refusal.
- A bare-season extra ("Show S05 Ending.mkv") that Sonarr matches to a whole season is no longer pre-assigned every episode at grab time - the shape that could import one OP/ED video as twelve episodes. Grab-time assignment now follows import-time's limits and leaves ambiguity for import to refuse.
- Full-season parses are now refused at grab time by Sonarr's own `fullSeason` flag, closing the gap where a season of three or fewer episodes slipped under the span cap.
- Checksum and sidecar files (`.blake3`, `.md5`, `.sha256`, `.mks` subtitle tracks, `.tif`, `.m3u8`) are no longer treated as importable videos, so releases shipping them no longer stick on "intended file missing" warnings.
- A torrent carrying the same filename in two folders no longer warns that the file "could not be matched to an episode" after one copy imported; the warning reports only names that truly went unplaced, once each.

## [1.0.5] - 2026-07-18

### Changed

- Repeated identical Sonarr/Radarr connection warnings now coalesce: the first failure warns as before, repeats drop to the debug log with a running count, a "still failing - attempt N" reminder fires about every `imports.digest_interval` seconds, and a note reports when the call recovers. Import-wait polls (manual-import scans, the download-history probe, the import-time filename parse) also no longer retry inside each call - the poll loop itself is the retry - so a failing poll cycle emits one connection line instead of four and skips several seconds of backoff sleep.

### Fixed

- Re-grabs of a release Sonarr has previously imported (or failed/ignored) now import instead of deferring "for a later run" forever. Sonarr hides such a re-added download from its queue and its manual-import scan returns HTTP 500 on every poll (an unpatched Sonarr bug: the tracked download's import state is never re-initialized), so Pearlarr could neither import nor ever finish - the pending record was eventually dropped silently. Pearlarr now detects the state from Sonarr's download history and imports the files by scanning the download folder directly (translated through Sonarr's remote path mappings), copying rather than moving so the torrent keeps seeding.
- A run during a SeaDex outage no longer advances the Arr-activity checkpoint, so file changes it detected but could not act on (every lookup is skipped during an outage) are re-detected and handled by the next healthy run instead of being silently consumed. Previously such changes were only picked up if some later, unrelated event touched the same series.

## [1.0.4] - 2026-07-14

### Changed

- Releases are now cut end to end by CI: merging the release PR records the README media, tags the version, publishes to PyPI and GHCR, and assembles the GitHub release. `scripts/release.sh` is gone - maintainers run the "Release prepare" workflow instead (see CONTRIBUTING "Releasing").
- The README's media moved to the separate [pearlarr-assets](https://github.com/trevinbrooks/pearlarr-assets) repository, so cloning Pearlarr never downloads a media byte, and each PyPI version's page now keeps its own release's media forever: the package readme pins its images to that repository's immutable `vX.Y.Z` tag at build time, while the GitHub README follows its `main` branch.

## [1.0.3] - 2026-07-13

### Fixed

- The README's screenshot and demo GIF now render on the PyPI project page: they are served from the repository's `assets` branch instead of release-asset downloads, whose forced `application/octet-stream` content type PyPI's image proxy refuses.

## [1.0.2] - 2026-07-13

### Changed

- The README's screenshot and demo GIF are served as GitHub release assets instead of repository files, so the images always show the latest release.

## [1.0.1] - 2026-07-13

### Changed

- `advanced.sleep_time` now defaults to `0` (was `2`), disabling the inter-query rate-limit sleep out of the box and enabling the concurrent episode-fetch fast path by default, so runs are noticeably faster.
  Set a positive value to pace API queries again. Existing configs that already set `sleep_time` explicitly are unaffected.

## [1.0.0] - 2026-07-12

The first release of the fork.

### Upgrade notes

Coming from upstream 0.9.x:

- **The project is renamed** from SeaDexArr to Pearlarr: the package, the `pearlarr` command, the data directory, and the `PEARLARR_*` environment variables all follow (the `seadexarr` PyPI name stays with upstream).
- **The config format changed wholesale**, from flat keys to nested groups with strict validation.
  Rewrite your `config.yml` against the new layout (`pearlarr config init` writes a commented starter, and [docs/configuration.md](docs/configuration.md) documents every key): `sonarr_url` is now `sonarr.url`, `qbit_info` is `qbittorrent.*`, `torrent_tags` is `qbittorrent.tags`, `seadex.public_only` is replaced by `seadex.private_releases`.
  Unknown or misspelled keys now fail at load instead of being ignored.
- **The data location moved** off the install directory to one OS-standard data directory (`~/.local/share/pearlarr` on Linux, `~/Library/Application Support/pearlarr` on macOS).
  `pearlarr config init` writes the new config there. Set `PEARLARR_DATA_DIR` to relocate everything.
  Logs follow the data directory instead of the working directory. `pearlarr paths` prints every resolved location.
- **The cache is not carried over**: the SQLite `cache.db` replaces `cache.json`, and the old file is not read.
  The first run re-evaluates the library from scratch - cheap in the default matching mode, which checks what the arrs already have on disk. See [docs/deployment.md](docs/deployment.md#migrating-from-upstream-seadexarr) before enabling hash matching.
- **Python 3.13 or newer is required** (3.12 support dropped). The Docker image runs 3.14.
- **`SCHEDULE_TIME` is deprecated** in favor of `schedule.interval_hours` (on bare metal a still-set `SCHEDULE_TIME` wins, with a deprecation warning. The Docker image never reads it - set `PEARLARR_CRON`).

### Added

- Wait-for-completion and Sonarr manual import (`imports.wait_mode`: `off`/`deferred`/`blocking`/`hybrid`): Pearlarr waits for qBittorrent to finish grabbed torrents, lets Sonarr import them, and steps in with a series-pinned manual import when Sonarr can't.
  Downloads that outlast a run are carried as pending imports and picked up by a later run.
- `notifications.wait_webhook_url`: a generic outbound webhook (ntfy, gotify, Home Assistant, ...) for the wait-pass summary, alongside the Discord webhook, with `notifications.wait_notify` controlling the push.
- `imports.post_import_category`: move a torrent to a different qBittorrent category (created if missing) once its import is verified complete, e.g. to give finished torrents different seeding rules.
- Arr-side activity detection (`advanced.detect_arr_activity`, on by default): each run polls the arr's history and re-checks titles whose files were imported or deleted arr-side since the last run, so a quality upgrade or manual grab is re-evaluated without waiting for SeaDex to update.
  The first scan covers the last 30 days. A coverage gap re-checks everything once.
- AniBridge mappings as the primary ID/episode mapping source, mopping up titles the other sources miss.
- `seadex.ignore_anilist_ids` (skip specific AniList IDs), `seadex.ignore_tags` (filter releases by SeaDex tag), and `qbittorrent.tags` (tag every added torrent).
- Discord notifications are rich embeds with colors, links, and a version footer.
- The console shows a live cockpit during startup and the import-wait pass (spinners, ticking timers, files-imported progress).
- `advanced.log_format` picks the console surface (`auto`/`rich`/`plain`/`json`). `json` writes one JSON object per event to stdout - a versioned machine interface with a generated event catalog ([docs/output.md](docs/output.md)).
- Rotated logs are dated per-run backups, kept for `advanced.log_retention_days` days, replacing the fixed ten-file cascade.
- New CLI surface:
  - `run single --dry-run` (simulate without grabbing, caching, or notifying), `--movie-id`/`--series-id` (single-title runs by TMDB/TVDB ID), and `--import-wait-mode`/`--log-level` per-run overrides.
  - `config validate` and `config show` (effective config with secrets redacted, safe to paste into a bug report).
  - `cache stats` and `cache check`.
  - `pearlarr --version`, `-h` everywhere. A bare group command prints its help.
  - `replay FILE` (or `-` for stdin) re-renders a captured `log_format: json` / `--json` stream back into the readable text log grammar, for reading a docker-captured log after the fact.
- `--json` on every subcommand (`paths`, `config init`/`validate`/`migrate`/`show`, `cache backup`/`restore`/`remove`/`stats`/`check`) emitting the same `schema_version` 1 envelope as run logs.
- Every config key can be set by environment variable as `PEARLARR_<GROUP>__<KEY>` (for example `PEARLARR_SONARR__URL`). Values are parsed as YAML and an environment override beats the file.
- The scheduled-run cadence is a config field, `schedule.interval_hours` (default 6), re-read each cycle so an edit takes effect without a restart.
- Config schema versioning: `config_version` stamps the file, and a file from an older Pearlarr is migrated automatically in memory at every load (a nested-format file still saying `seadex.public_only` or `private_releases: allow` keeps loading, with a warning naming the fold).
  `pearlarr config migrate` rewrites the file itself at the current schema, keeping the previous file as `config.yml.bak`.

### Changed

- **Private releases are never grabbed** (SeaDex carries no download link for them, and no private-tracker auth is supported).
  `seadex.private_releases` decides what happens when a title's preferred release is private-only: `warn` (default) warns and leaves the title uncached so it is re-checked every run. `fallback` grabs the entry's best public alternative, warning only when none exists.
  Titles satisfied by a fallback are remembered. Switching back to `warn` re-checks them and resurfaces the warning.
- Where multiple preferred release groups cover exactly the same files, only one is downloaded (preferring a public release) instead of all of them.
- `advanced.max_torrents_to_add` defaults to `10` instead of unlimited, so a first run against a large library doesn't flood qBittorrent - later runs pick up where the cap stopped.
  Preview runs ignore the cap and always report the whole library. Set the key to `0` to remove the cap.
- Only configured arrs run: a Sonarr-only (or Radarr-only) config skips the other arr with a ledger note instead of failing every cycle. Explicitly selecting an unconfigured arr fails with a one-line error naming the missing keys. A half-configured arr (URL without API key, or the reverse) is warned about by name.
- `run single` with no selection flags runs every configured arr, mirroring scheduled mode (previously it printed a usage hint and failed).
- Failed CLI commands exit non-zero (previously always 0). A malformed config, missing backup, unreachable arr, or rejected API key is reported as a clean one-line error naming the config keys to check, instead of a traceback.
- `cache backup` writes via a temp file so a failed backup can never destroy the previous good one. `cache restore` copies instead of consuming the backup, so a restore is repeatable.
- `config init` refuses to overwrite an existing `config.yml` unless `--force` is passed.
- The Python import surface is internal as of 1.0.0: the supported interfaces are the CLI, the config schema, the JSON event stream, and the notification payloads.
- Runtime dependencies bumped to current majors (typer 0.26, rich 15, qbittorrent-api 2026.6).

### Deprecated

- `SCHEDULE_TIME` (Docker-era env var): set `schedule.interval_hours` (bare metal) or `PEARLARR_CRON` (Docker, which ignores `SCHEDULE_TIME`) instead. An invalid value now falls back to the configured cadence with a report instead of crashing the scheduler.

### Removed

- `seadex.public_only` (replaced by `seadex.private_releases`, see Upgrade notes).
- Python 3.12 support.

### Fixed

- `advanced.log_level` is honored everywhere: `ERROR` is a real level (the logger previously treated it as `INFO`), and CLI runs no longer force `INFO` regardless of config.
- In `fallback` mode, a public substitute never replaces a copy of the preferred private release you already own: when your private copy is stale-sized, Pearlarr warns and holds the title (the summary names both ways out) instead of overwriting it with the fallback.

## Inherited from upstream

The releases below are [bbtufty/seadexarr](https://github.com/bbtufty/seadexarr) history, reproduced as shipped.

## [0.9.0] - 2025-09-13

- Include PlexAniBridge-Mappings to mop up some missed titles
- Update cache if version/config changes
- Add option to ignore SeaDex update time
- Don't recreate cache on config change
- Add a number of useful CLI commands
- Include option to just check torrents by hash
- Update cache if no suitable releases found
- Check file sizes, for different versions of releases etc.

## [0.8.1] - 2025-09-05

- Fix cache not updating with versions

## [0.8.0] - 2025-09-05

- If we're ignoring Radarr movies in Sonarr, also check the cache
- Do a more proper check for episodes in Sonarr
- Ensure docker-compose run also uses cache
- Fix crash if AniList ID isn't already in cache
- Include AniList name in cache
- Sort cache by AniList ID
- Revert removing trackers

## [0.7.0] - 2025-08-24

- Create a cache to avoid checked entries that haven't updated
- Fix grabbing multiple releases when there's a mismatch in episode parsing
- Removed trackers that aren't used by SeaDex
- Add support for RuTracker

## [0.6.0] - 2025-08-13

- Take Ja-only releases if `prefer_dual_audio` is False
- Catch crash if SeaDex is unreachable
- Cleanup dictionaries
- Search specifically in qBittorrent by torrent hash, to speed up hash checks
- Skip adding downloads to torrent client if download flag not set
- Fix bug where maximum number of torrents added was not respected

## [0.5.0] - 2025-08-08

- Fix UTF-8 encoding warning in log
- Move to episode-based filtering of torrents for Sonarr
- Include SeaDex tags in log and Discord messages
- Add options to ignore unmonitored series/movies
- Save logs to file
- Ensure SCHEDULE_TIME is brought in as a float
- Fix Discord messages not getting pushed if no torrent client selected
- Fix adding too many torrents in one go

## [0.4.1] - 2025-07-31

- Add PyYAML to pyproject.toml
- Added support for AnimeTosho

## [0.4.0] - 2025-07-31

- Added ignore_movies_in_radarr for SeaDexSonarr, which will skip movies flagged as Specials in Sonarr that already exist in Radarr
- More robust config when parameters added
- Build Docker for every main update
- Fix crash when AniDB mapping contains no text
- Use IMDb for finding AniList mappings as well
- Initial support for other trackers
- Better handle Discord notifications and log messages when torrent already in client
- Map potentially weird episodes SeaDexSonarr if in Season 0 (Specials)

## [0.3.0] - 2025-07-30

- Added scheduling in Docker mode

## [0.2.0] - 2025-07-30

- Add Docker support
- Move to config files, to make the call simpler
- Fix crash if torrent in list but not already downloaded

## [0.1.0] - 2025-07-22

- Add support for Radarr

## [0.0.3] - 2025-07-22

- Rename from seadex_sonarr to seadexarr, in preparation for Radarr support
- Add interactive mode, for selecting when multiple "best" options are found
- Add support for adding torrents to qBittorrent

## [0.0.2] - 2025-07-13

- Improved Discord messaging
- Catch the case where we don't find any suitable SeaDex releases
- Include potentially weird offset mappings via AniDB lists
- Add a rest time to not hit AniList rate limiting

## [0.0.1] - 2025-07-12

- Initial release
