# SeaDexArr

[![](https://img.shields.io/pypi/v/seadexarr.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/seadexarr/)
[![](https://img.shields.io/pypi/pyversions/seadexarr.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/seadexarr/)
[![Actions](https://img.shields.io/github/actions/workflow/status/trevinbrooks/seadexarr/build.yaml?branch=main&style=flat-square)](https://github.com/trevinbrooks/seadexarr/actions)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg?label=License&style=flat-square)](LICENSE)

SeaDexArr keeps the anime in your [Sonarr](https://sonarr.tv) and [Radarr](https://radarr.video)
libraries at the bar set by [SeaDex](https://releases.moe), the curated index of the best anime
releases. On a schedule, it maps your library to SeaDex's entries, picks the best release you
don't already have, grabs it through qBittorrent, and can notify you on Discord:

![A SeaDexArr grab notification in Discord](example_post.png)

## How it works

Each run walks your library title by title:

1. **Map** — Sonarr series (by TVDB/IMDb ID) and Radarr movies (by TMDB/IMDb ID) are resolved to
   AniList IDs, the key SeaDex indexes by.
   [PlexAniBridge-Mappings](https://github.com/eliasbenb/PlexAniBridge-Mappings) is the primary
   source, with [Kometa Anime-IDs](https://github.com/Kometa-Team/Anime-IDs) and the
   [AniDB anime-lists](https://github.com/Anime-Lists/anime-lists) filling gaps (and covering
   specials).
2. **Check** — the title's SeaDex entry is fetched, unless the cache shows it unchanged since the
   last pass (see [When a title is checked again](#when-a-title-is-checked-again)).
3. **Select** — the entry's torrents are cut down to the preferred release(s): tag and tracker
   filters, SeaDex's "best" marks, your audio preference (see
   [How a release is chosen](#how-a-release-is-chosen)).
4. **Compare** — the picks are matched against what the arr already has (release groups, episodes,
   file sizes), so only missing or outdated releases go further.
5. **Grab** — the release is added to qBittorrent with your category and tags, a Discord
   notification goes out, and the result is cached.
6. **Import** (optional, Sonarr) — SeaDexArr waits for the downloads to finish and shepherds them
   into Sonarr, stepping in with a manual import when Sonarr can't place the files itself (see
   [Waiting for imports](#waiting-for-imports-sonarr)).

Runs are incremental and safe to repeat: results are cached in SQLite, a config without
qBittorrent credentials runs as a preview (nothing is grabbed), and `run single --dry-run`
simulates a full run with no side effects at all.

## Quick start

### Docker (recommended)

Copy [docker-compose.example.yml](docker-compose.example.yml) to `docker-compose.yml` (or fold the
`seadexarr` service into an existing stack), create the config directory yourself, and bring it
up:

```
mkdir -p ./config
docker compose up -d seadexarr
```

Creating `./config` first matters: a missing bind-mount directory is auto-created by Docker
root-owned, and the container (uid 1000 by default) then stops immediately with a message naming
the fix (`chown` the host directory, or point `user:`/PUID/PGID at its owner — see
[Permissions](#permissions)).

On the very first boot the container writes a starter `config.yml` into the `/config` mount and
exits — it can't do anything useful until the file is filled in. With `restart: unless-stopped`
Docker restarts it; the template now exists, so the container stays up but each run just logs that
neither arr is configured. Fill in `./config/config.yml`, then:

```
docker compose restart seadexarr
docker compose logs -f seadexarr
```

One-off commands run through the same service — everything after the service name is passed to
the `seadexarr` CLI:

```
docker compose run --rm seadexarr run single --sonarr
docker compose run --rm seadexarr cache stats
```

The image also carries a healthcheck (`seadexarr paths`), so `docker ps` shows the container
healthy once the CLI and its data directory resolve — purely informational; nothing auto-restarts
on unhealthy.

#### Scheduling

Inside the container, cron owns the cadence:
[supercronic](https://github.com/aptible/supercronic) runs `seadexarr run single` on the
`SEADEXARR_CRON` schedule (default `0 */6 * * *`, i.e. every 6 hours), evaluated in the `TZ`
timezone (default UTC). `SEADEXARR_RUN_ON_START=true` (the default) also runs one catch-up pass
when the container starts; set it to `false` to wait for the first cron fire instead. The config's
`schedule.interval_hours` does not apply in the container — it only drives the bare-metal
`seadexarr run scheduled` loop.

Overlapping runs are double-guarded: supercronic won't start a run while the previous one is still
going, and SeaDexArr holds its own cross-process file lock besides. If you enable a blocking
`imports.wait_mode`, consider raising `stop_grace_period` (see the compose example) so a
`docker stop` doesn't SIGKILL a mid-import run after Docker's default 10 seconds.

#### Permissions

The compose file's `user: "${PUID:-1000}:${PGID:-1000}"` line sets the uid/gid the container runs
as. PUID and PGID are compose *interpolation* variables — set them in an `.env` file next to the
compose file or in the shell; they are **not** `environment:` keys (an `environment:` entry is
silently ignored for this). Match them to the owner of the `./config` host directory so the
container can write its config, caches and logs there.

The starter config is written mode 600, which lines up when your host editor runs as the same uid
the container uses; if you loosen it (e.g. `chmod 644`), SeaDexArr warns on every run that the
file — which holds API keys — is readable by other users.

### pip

SeaDexArr needs Python 3.13 or newer:

```
pip install seadexarr
```

Write the starter config, fill it in, then run:

```
seadexarr config init
seadexarr paths          # shows where config.yml landed
seadexarr run single     # one pass, or just `seadexarr` for the scheduled loop
```

Or install the cutting edge from source:

```
git clone https://github.com/trevinbrooks/seadexarr.git
cd seadexarr
pip install -e .
```

## The data directory

SeaDexArr keeps everything — `config.yml`, the caches (`cache.db`, `mappings.db`) and `logs/` —
in a single data directory. By default this is the OS-standard per-user location:

| OS      | Default data directory                               |
|---------|------------------------------------------------------|
| Linux   | `~/.local/share/seadexarr` (honors `$XDG_DATA_HOME`) |
| macOS   | `~/Library/Application Support/seadexarr`            |
| Windows | `%LOCALAPPDATA%\seadexarr`                           |

Override the location with the `SEADEXARR_DATA_DIR` environment variable or the global
`--data-dir` flag (the flag wins). Run `seadexarr paths` to print the resolved locations. The
Docker image sets `SEADEXARR_DATA_DIR=/config`, so the `/config` volume mount holds the whole data
directory.

## CLI

Every command answers `--help` (or `-h`), and `seadexarr --version` prints the installed version.
Commands exit non-zero when they fail (a failed run, backup, invalid config, or refused restore),
so they compose with `&&`, cron, and health checks. Only the arrs you've configured (`url` +
`api_key`) are run: a Sonarr-only or Radarr-only setup just skips the other one. Tab completion
for your shell is available via `seadexarr --install-completion`.

In Docker, run commands through your compose service: `docker compose run --rm seadexarr <args>`.

### `seadexarr run`

`run scheduled` — the default when you invoke bare `seadexarr` — loops forever, running every
configured arr each `schedule.interval_hours` (default 6, re-read each cycle so a config edit
takes effect without a restart). It's the bare-metal fallback scheduler; the Docker image
schedules via supercronic + `SEADEXARR_CRON` instead (see [Scheduling](#scheduling)).

`run single` runs once and exits. On its own it runs every configured arr;
`--sonarr` / `--radarr` narrow it to one module. To run a single title, pass `--series-id` with a
series' TVDB ID or `--movie-id` with a movie's TMDB ID — either flag implies the matching module,
so `run single --movie-id 12345` is a complete command.

`run single` also accepts:

- `--dry-run` — simulate the run: no grabs, no cache writes, no notifications
- `--import-wait-mode` — override the configured `imports.wait_mode` (off/deferred/blocking/hybrid)
  for this run (see [Waiting for imports](#waiting-for-imports-sonarr))

Both run commands accept `--log-level`, which overrides the configured `advanced.log_level` for
that invocation (handy for a one-off `--log-level debug` run).

### `seadexarr config`

- `config init` — write the starter `config.yml` to the data directory. An existing file is never
  overwritten unless you pass `--force`
- `config validate` — check the file parses and validates (listing exactly which keys are wrong
  otherwise) and report what a run would use: whether each arr is configured, and whether
  qBittorrent credentials are set (without them, runs are previews — nothing is grabbed)
- `config show` — print the effective configuration with every default applied and secrets
  redacted (API keys, passwords, usernames, webhook URLs, everything in the free-form
  `qbittorrent.options` block, and any login embedded in a URL), so it's safe to paste into a bug
  report

### `seadexarr paths`

Prints the resolved data directory and the files within it (config, caches, logs). Useful for
confirming where SeaDexArr is reading and writing, especially with a custom `--data-dir` or
`SEADEXARR_DATA_DIR`.

### `seadexarr cache`

- `cache backup` — back up `cache.db` to `cache.backup.db`, using the SQLite online-backup API so
  the snapshot is consistent even mid-write
- `cache restore` — replace `cache.db` with a copy of `cache.backup.db` (the backup is kept, so a
  restore is repeatable)
- `cache remove` — delete `cache.db`. Useful after config changes (see
  [When a title is checked again](#when-a-title-is-checked-again))
- `cache stats` — print cache health (per-block row counts and on-disk size)
- `cache check` — run a SQLite integrity check on the cache database

Commands that modify the cache refuse to run while a SeaDexArr run is active in the same data
directory, so they can't clobber an in-flight database.

## How a release is chosen

A SeaDex entry usually lists several torrents. SeaDexArr cuts them down in order:

1. **Tags** — torrents carrying any tag in `seadex.ignore_tags` are dropped.
2. **Trackers** — torrents from trackers outside `seadex.trackers` are dropped (the default allows
   every tracker).
3. **Best** — if `seadex.want_best` is on and at least one surviving torrent is marked "best" on
   SeaDex, only those stay.
4. **Audio** — if `seadex.prefer_dual_audio` is on, dual-audio torrents are preferred; if it's
   off, Ja-only torrents are preferred instead. Like the "best" cut, this only applies when it
   leaves at least one torrent.

Private releases are never grabbed — SeaDex carries no downloadable link for them, and no
private-tracker auth is supported. When the release this lands on is only available on private
trackers, `seadex.private_releases` sets the policy:

- `warn` (the default) warns and leaves the title uncached, so it's re-checked every run until a
  public release appears.
- `fallback` grabs the entry's best *public* alternative instead (the same best/dual-audio cuts
  applied to the public torrents only), warning only when no public release covers those files.
  A fallback never replaces a copy of the preferred private release you already own: if the arr
  holds it at a stale size (SeaDex's record changed, e.g. the group patched it), the title warns
  and holds instead of grabbing the substitute — update the release from its private tracker, or
  delete the stale files to let the fallback stand in. A preferred public release still supersedes
  as usual. Titles satisfied by a fallback are remembered; switching back to `warn` re-checks them
  and resurfaces the private-only warning.

Where several preferred release groups cover exactly the same files, only one is downloaded
(preferring a public release). Everything that survives is grabbed, capped by
`advanced.max_torrents_to_add`; with `advanced.interactive` on, SeaDexArr prompts you to choose
whenever several options match.

Downloads are currently supported from **Nyaa**, **AnimeTosho** and **RuTracker**. A winning
release on another public tracker (e.g. AniDex) is skipped with a warning and kept out of the
cached result, so it isn't treated as grabbed and is re-considered once support lands or your
config changes.

### Matching against your library

Two modes control how SeaDexArr decides you already have a release:

**Against existing files** (default; `seadex.use_torrent_hash_to_filter: false`): SeaDex picks are
matched to the release groups Sonarr/Radarr report on disk; for Sonarr, filenames are also parsed
to check coverage against individual episodes. File sizes are compared too, to catch release
groups putting out revised or higher-quality versions of the same release.

**Against torrent hashes** (`seadex.use_torrent_hash_to_filter: true`): SeaDex picks are matched
to the torrent hashes SeaDexArr itself grabbed before (from its cache). Updated releases are
reliably re-grabbed, but there are sharp edges: with an existing library, everything looks
un-grabbed on the first pass and gets downloaded again; and the mode is blind to what the arr
actually has on disk, so releases obtained out-of-band (e.g. a private release grabbed directly
from its tracker) are invisible to it and the owned-file protections of the default mode don't
apply.

### When a title is checked again

- **SeaDex updates**: a processed title is cached and skipped until SeaDex updates its entry. Set
  `seadex.ignore_seadex_update_times: true` to re-check everything regardless.
- **Arr-side changes** (`advanced.detect_arr_activity`, on by default): each run polls the arr's
  history once and re-checks just the titles whose files were imported or deleted arr-side since
  the last pass — so a quality upgrade or manual grab under an unchanged SeaDex entry is
  re-evaluated without waiting for SeaDex. The first scan covers the last 30 days; a coverage gap
  (SeaDexArr stopped for longer than that) re-checks everything once.
- **Config changes**: the cache is *not* invalidated when the config changes. After editing
  settings that affect release selection (trackers, tags, audio preference, …), run
  `seadexarr cache remove` so the next pass re-evaluates everything.

## Waiting for imports (Sonarr)

SeaDexArr can optionally wait for grabbed torrents to finish downloading and then shepherd them
into Sonarr (Sonarr only; on Radarr runs this is a no-op). After a torrent is added, SeaDexArr
waits for qBittorrent to finish it, then asks Sonarr to rescan and watches its queue: it lets
Sonarr import the files itself, and only steps in with a series-pinned manual import (using
SeaDexArr's own authoritative episode mapping) when Sonarr can't auto-import them or isn't
tracking the download.

The feature is controlled by `imports.wait_mode`:

- `off` — disabled (default): no waiting, no pending records, no manual import
- `deferred` — never block; import already-complete downloads on a later run
- `blocking` — block at the end of the run until downloads complete, then import
- `hybrid` — recommended: a deferred reconcile at the start of the run plus a blocking pass at
  the end

On a terminal, the wait pass renders a live progress view (download bars, speeds, a
files-imported bar); piped or under Docker/cron the log shows the wait's start line, each
download's outcome as it finishes, and the closing tally. The remaining `imports.*` keys
(timeouts, poll cadence, import mode, languages) are described in
[the config reference](#import-settings). To get a push notification when a wait pass
finishes, see [Notifications](#notifications).

## Notifications

With `notifications.discord_url` set, every grab posts a rich embed (the screenshot at the top):
per release group, what was grabbed — or already downloading, failed, or skipped — with tracker
links, sizes and audio markers; the episodes covered; the release groups being replaced; and the
SeaDex entry's notes and comparison links, framed by the title's AniList art. If a wait pass ran,
a summary embed follows: how many imports landed, were left for a later run, or failed, and why.

`notifications.wait_webhook_url` sends that wait summary as a plain JSON POST instead — for ntfy,
gotify, Home Assistant, or anything else that accepts a webhook. Both URLs can be set at once;
`notifications.wait_notify` controls the wait-complete ping and defaults to on whenever either
webhook is configured.

## Configuration

The config file is nested YAML: settings live in nine groups (`sonarr`, `radarr`, `qbittorrent`,
`seadex`, `imports`, `notifications`, `schedule`, `mappings`, `advanced`), referred to as
`group.key` below. The config is validated on load — an unknown or misspelled key fails with an
error naming it rather than being silently ignored — and keys left blank fall back to their
defaults. `seadexarr config init` writes a commented starter file, and `seadexarr config show`
prints the effective result.

### Sonarr/Radarr settings

- `sonarr.url` — URL for Sonarr. Required when running the Sonarr module
- `sonarr.api_key` — API key for Sonarr (Settings → General → API Key). Required when running the
  Sonarr module
- `sonarr.verify_ssl` — verify the TLS certificate on an HTTPS Sonarr. Defaults to true; set
  false only as a last resort for a self-signed instance whose CA can't be trusted via the OS
  store (see below)
- `sonarr.ignore_unmonitored` — skip series unmonitored in Sonarr (or whose episodes are all
  unmonitored). Defaults to false
- `sonarr.torrent_category` — qBittorrent category for Sonarr-added torrents. Defaults to blank,
  which sets no category
- `sonarr.ignore_movies_in_radarr` — skip movie specials found via Sonarr when they already exist
  as movies in Radarr. Defaults to false

The `radarr` group takes the same keys (minus `ignore_movies_in_radarr`): `radarr.url`,
`radarr.api_key`, `radarr.verify_ssl`, `radarr.ignore_unmonitored`, and `radarr.torrent_category`.

#### TLS and custom certificate authorities

SeaDexArr verifies TLS against the **operating-system trust store** (via
[truststore](https://truststore.readthedocs.io)), not a bundled CA list. A corporate or homelab CA
installed on the host — `update-ca-certificates` on Debian/Ubuntu, Keychain on macOS, certmgr on
Windows — is picked up automatically. In a container the OS store is typically bare beyond the
public CAs, so mount your CA and point OpenSSL at it instead of disabling verification:

```yaml
environment:
  - SSL_CERT_FILE=/config/ca.pem
```

`verify_ssl: false` skips verification entirely for that arr and should be the last resort.

### qBittorrent settings

- `qbittorrent.host` / `qbittorrent.username` / `qbittorrent.password` — connection details for
  qBittorrent. All three are needed to add torrents; leave any blank and SeaDexArr runs in
  preview mode (nothing is grabbed)
- `qbittorrent.tags` — tags applied to every added torrent. Defaults to blank (no tags)
- `qbittorrent.options` — extra keyword arguments for the qBittorrent client, e.g.
  `{VERIFY_WEBUI_CERTIFICATE: false}` for a self-signed WebUI. Defaults to empty

### SeaDex filters

- `seadex.private_releases` — policy when the preferred release is only available on private
  trackers: `warn` (default) or `fallback`. See
  [How a release is chosen](#how-a-release-is-chosen)
- `seadex.prefer_dual_audio` — prefer releases tagged dual audio, when any exist; if false,
  prefer Ja-only releases instead. Defaults to true
- `seadex.want_best` — prefer releases tagged "best" on SeaDex, when any exist. Defaults to true
- `seadex.ignore_tags` — SeaDex tags to filter out, e.g. `Dolby Vision`, `Misplaced Special`,
  `Deband Required`. Defaults to empty
- `seadex.trackers` — restrict to a list of trackers. Defaults to blank, which allows all public
  and private trackers regardless of `seadex.private_releases` — private filtering happens later
  in the selection, so the `warn`/`fallback` policies can still see (and report on) private-only
  releases. Names are matched case-insensitively. SeaDex carries torrents from:
  - Public: Nyaa, AnimeTosho, RuTracker (all three grabbable), AniDex, and `Other` (SeaDex's
    catch-all bucket for public releases not on a named tracker)
  - Private: AB, BeyondHD, PassThePopcorn, BroadcastTheNet, HDBits, Blutopia, Aither, and
    `OtherPrivate` (the private catch-all)
- `seadex.ignore_anilist_ids` — AniList IDs to never process (a list of integer IDs). Defaults to
  empty
- `seadex.ignore_seadex_update_times` — re-check entries even when SeaDex hasn't updated them
  since the cached pass. Defaults to false
- `seadex.use_torrent_hash_to_filter` — decide "already have it" by torrent hash (true) instead
  of by release group and files (false). Defaults to false. See
  [Matching against your library](#matching-against-your-library)

### Import settings

These control the wait-for-completion and Sonarr manual-import feature (see
[Waiting for imports](#waiting-for-imports-sonarr); Sonarr only).

- `imports.wait_mode` — `off` (disabled), `deferred`, `blocking`, or `hybrid`. Defaults to off
- `imports.wait_timeout` — seconds to wait per torrent for qBittorrent to finish. Defaults to 3600
- `imports.ready_timeout` — seconds to then wait for Sonarr to rescan and import. Defaults to 600
- `imports.poll_interval` — seconds between polls of qBittorrent and the Sonarr queue. Defaults
  to 30
- `imports.progress_poll_interval` — seconds between cheap re-reads for the wait screen's
  files-imported bar and live download telemetry. 0 disables it, and values at or above
  `imports.poll_interval` behave the same. Defaults to 5
- `imports.mode` — Sonarr import mode: `auto`, `move`, or `copy`. Defaults to auto
- `imports.post_import_category` — qBittorrent category to move a torrent to once its import is
  verified complete (e.g. to hand finished torrents different seeding rules). The category is
  created if it doesn't exist; the move happens when SeaDexArr verifies the import, so it needs a
  non-off wait mode. Note that with qBittorrent's Automatic Torrent Management enabled, changing
  category can relocate the torrent's data to the new category's save path. Defaults to blank,
  which leaves the add-time category in place
- `imports.default_quality` — fallback quality name (e.g. `Bluray-2160p`) for manual imports,
  useful on a 4K instance. Defaults to blank
- `imports.languages_dual` — languages applied to imported dual-audio releases. Defaults to
  `[Japanese, English]`
- `imports.languages_single` — languages applied to imported single-audio releases. Defaults to
  `[Japanese]`
- `imports.pending_max_age_days` — drop pending-import records older than this many days.
  Defaults to 14
- `imports.digest_interval` — target seconds between "still waiting" digest lines when the
  rich console is forced (`advanced.log_format: rich`) on a terminal that can't render the
  live view. Defaults to 300

### Notification settings

- `notifications.discord_url` — Discord webhook URL for grab and wait-summary embeds (set one up
  following [this guide](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks)).
  Defaults to blank, which disables the Discord integration
- `notifications.wait_webhook_url` — generic outbound webhook (ntfy/gotify/Home Assistant) POSTed
  the wait summary as JSON. Defaults to blank
- `notifications.wait_notify` — push a notification when a wait pass completes. Defaults to on
  whenever either webhook above is set

### Schedule settings

- `schedule.interval_hours` — hours between cycles of the bare-metal `run scheduled` loop,
  re-read each cycle. The Docker image ignores it (`SEADEXARR_CRON` owns the container cadence).
  Defaults to 6

### Mapping settings

The general user should leave all three blank (auto-download):

- `mappings.anime_mappings` — custom Kometa-style anime ID mappings, or `false` to disable the
  source entirely. Defaults to blank, which downloads the Kometa mappings
- `mappings.anidb_mappings` — `false` to disable the AniDB mappings (used for specials).
  Defaults to blank, which downloads them
- `mappings.anibridge_mappings` — custom AniBridge mappings, or `false` to disable the source
  entirely. Defaults to blank, which downloads the PlexAniBridge mappings (the primary source)

### Advanced settings

- `advanced.sleep_time` — seconds to wait after each API query, to avoid rate limits. 0 disables
  the sleep. Defaults to 2
- `advanced.cache_time` — days to cache the downloaded mapping sources (they don't change often).
  Defaults to 1
- `advanced.interactive` — when several torrent options match, prompt to choose one (or several)
  instead of grabbing everything. Defaults to false
- `advanced.max_torrents_to_add` — cap the number of torrents added in one run. Defaults to
  blank (unlimited)
- `advanced.detect_arr_activity` — re-check titles whose files Sonarr/Radarr changed since the
  last pass (see [When a title is checked again](#when-a-title-is-checked-again)). Opt out if
  you deliberately replace releases arr-side and don't want SeaDexArr re-evaluating them.
  Defaults to true
- `advanced.log_level` — `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` (case-insensitive).
  Defaults to INFO
- `advanced.log_format` — console output format. `auto` (the default) renders the rich styled
  console on a terminal and structured log lines (timestamp, level, context, message) when
  piped or under Docker; `rich`, `plain` and `json` force a renderer. `plain` output matches
  the log file line for line, except for forensic file-only diagnostics that only the file
  keeps; `json` emits one JSON object per event. `plain` and `json` also disable the live
  progress views (expected). The log file always uses the structured format

## Scripting

The CLI is the primary interface — a one-off run of everything configured is just
`seadexarr run single`, and non-zero exit codes make the commands scriptable as-is. The same
composition is available programmatically: build the shared collaborators with `RunDeps.build`,
wrap them in a `RunServices` hub, inject both into the `RunLoop` plus an arr strategy, and drive
`run_sync`:

```python
from seadexarr import RunDeps, RunLoop, RunServices, SonarrSync, setup_logger
from seadexarr.modules.config import AppConfig, Arr
from seadexarr.modules.mappings import MappingResolver, MappingSources
from seadexarr.modules.paths import ensure_data_dir, resolve_paths
from seadexarr.modules.web_client import make_web_client

paths = resolve_paths()
ensure_data_dir(paths)
config = AppConfig.load(paths.config)
logger = setup_logger(config.advanced.log_level, paths.log_dir)

# One shared client for all non-arr web traffic (mapping downloads, trackers, webhooks).
web = make_web_client()
try:
    # Downloads/parses the ID-mapping sources once; shared across Arr runs.
    mappings = MappingResolver(
        cache_time=config.advanced.cache_time,
        ignore_anilist_ids=config.seadex.ignore_anilist_ids,
        sources=MappingSources(
            anime=config.mappings.anime_mappings,
            anidb=config.mappings.anidb_mappings,
            anibridge=config.mappings.anibridge_mappings,
        ),
        web=web,
        mappings_db=paths.mappings_db,
        logger=logger,
    )
    try:
        deps = RunDeps.build(
            Arr.SONARR,
            paths.cache,
            logger=logger,
            mappings=mappings,
            app_config=config,
            web=web,
        )
        try:
            services = RunServices(deps, Arr.SONARR)
            runner = RunLoop(deps, services)
            runner.run_sync(
                SonarrSync(deps, services),
                item_id=None,
                dry_run=False,
            )
        finally:
            deps.close()
    finally:
        mappings.close()
finally:
    web.close()
```

A Radarr run is the same shape with `RadarrSync` and `Arr.RADARR`. If no config file exists yet,
`AppConfig.load` writes the starter template to the data directory (run `seadexarr paths` to find
it) and raises `FileNotFoundError`; fill it in and run again.

## Roadmap

- Support for other torrent clients (currently qBittorrent)
- Download support for more trackers (currently Nyaa, AnimeTosho and RuTracker)

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development
setup (uv) and the quality gate every change runs through (ruff, the strict type checkers, and
pytest).

## Acknowledgements

SeaDexArr was originally created by [bbtufty](https://github.com/bbtufty). Release data comes from
the [SeaDex](https://releases.moe) project, and ID mappings from
[PlexAniBridge-Mappings](https://github.com/eliasbenb/PlexAniBridge-Mappings),
[Kometa Anime-IDs](https://github.com/Kometa-Team/Anime-IDs) and
[Anime-Lists](https://github.com/Anime-Lists/anime-lists).

SeaDexArr is licensed under the [GPL-3.0-or-later](LICENSE).
