# Pearlarr

[![PyPI](https://img.shields.io/pypi/v/pearlarr.svg?label=PyPI&style=flat-square)](https://pypi.org/project/pearlarr/)
[![Python](https://img.shields.io/pypi/pyversions/pearlarr.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/project/pearlarr/)
[![Actions](https://img.shields.io/github/actions/workflow/status/trevinbrooks/pearlarr/build.yaml?branch=main&style=flat-square)](https://github.com/trevinbrooks/pearlarr/actions)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg?label=License&style=flat-square)](LICENSE)

Pearlarr automatically grabs the releases [SeaDex](https://releases.moe) recommends for the anime in your [Sonarr](https://sonarr.tv) and [Radarr](https://radarr.video) libraries.

SeaDex is an index of the highest quality releases for a given anime.
On a schedule, Pearlarr maps your library to SeaDex's entries, compares each recommendation against what you already have, grabs anything missing or outdated through qBittorrent, and can shepherd the download into Sonarr and notify you on Discord:

<img src="https://raw.githubusercontent.com/trevinbrooks/pearlarr/main/docs/assets/example_post.png" width="520" alt="A Pearlarr grab notification in Discord: the release, its tracker, size, and audio, the episodes covered, the group being replaced, and SeaDex's notes">

## How it works

Each run walks your library title by title:

1. **Map** - Sonarr series and Radarr movies are resolved to AniList IDs (the key SeaDex indexes by) through three public ID-mapping sources.
2. **Check** - the title's SeaDex entry is fetched, unless the cache shows nothing relevant changed since the last run.
3. **Select** - the entry's torrents are cut down to the preferred release: your tag and tracker filters, SeaDex's "best" marks, your audio preference.
4. **Compare** - the picks are matched against what the arr already has (release groups, episodes, file sizes), so only missing or outdated releases go further.
5. **Grab** - the release is added to qBittorrent with your category and tags, a Discord notification goes out, and the result is cached.
6. **Import** (optional, Sonarr) - Pearlarr waits for the downloads to finish and shepherds them into Sonarr, stepping in with a manual import when Sonarr can't place the files itself.

Here is that walk on a three-title library - one title already has SeaDex's pick, two get grabbed, and the wait pass drives both imports home:

![A Pearlarr run: the boot steps connect to Sonarr and qBittorrent, three series are checked - Cowboy Bebop already has SeaDex's pick, while Frieren and Fullmetal Alchemist: Brotherhood are grabbed - and the wait pass shows live download progress until both imports land in Sonarr](https://raw.githubusercontent.com/trevinbrooks/pearlarr/main/docs/assets/demo_run.gif)

Runs are incremental and safe to repeat: results live in a SQLite cache, a title is re-checked only when SeaDex or your arr changed something, and an interrupted run never corrupts state.

## Install

> Coming from upstream SeaDexArr? The config format and data locations changed - see [migrating from upstream](docs/deployment.md#migrating-from-upstream-seadexarr) first.

### Docker Compose (recommended)

Copy [docker-compose.example.yml](docker-compose.example.yml) to `docker-compose.yml` (or fold the `pearlarr` service into an existing stack), create the config directory yourself, and bring it up:

```console
$ mkdir -p ./config
$ docker compose up -d pearlarr
```

On first boot the container writes a starter `config.yml` into `./config` and restarts; each run fails with a one-line error until the file is filled in.
Fill it in, then:

```console
$ docker compose restart pearlarr
$ docker compose logs -f pearlarr
```

The container schedules its own runs; set `PEARLARR_CRON` to change the cadence.
One-off commands run through the same service:

```console
$ docker compose run --rm pearlarr run single --sonarr
$ docker compose run --rm pearlarr cache stats
```

Everything operational lives in [docs/deployment.md](docs/deployment.md): permissions and PUID/PGID, timezones, custom CAs, image tags, stopping safely, backups, upgrades.

### uv / pipx / pip

Pearlarr needs Python 3.13 or newer; [uv](https://docs.astral.sh/uv/) or pipx give it its own environment (and uv fetches a matching Python if your system's is too old):

```console
$ uv tool install pearlarr    # or: pipx install pearlarr
```

Plain `pip install pearlarr` works too, as does `pip install -e .` from a clone for the cutting edge.

## First run

The step-by-step version of this section, with expected output, is [docs/getting-started.md](docs/getting-started.md).

```console
$ pearlarr config init
$ pearlarr paths          # shows where config.yml landed
```

Fill in just your Sonarr and/or Radarr connection first, then run one pass:

```console
$ pearlarr run single
```

Without qBittorrent credentials, every run is a **preview**: Pearlarr evaluates your whole library and reports everything it *would* grab, but grabs nothing and records nothing.
That is the recommended way to check a new setup - read the preview's summary, adjust the config, repeat.
When the preview picks what you'd pick, add `qbittorrent.host`, `username`, and `password`, and the same command grabs for real.
(`run single --dry-run` simulates a run with no side effects even after credentials are set.)

Bare `pearlarr` runs the scheduled loop, one cycle every `schedule.interval_hours`; under Docker the container's cron owns the cadence instead.

## Configuration

`config.yml` is nested YAML in nine groups:

- `sonarr` / `radarr` - connection details and per-arr behavior (unmonitored handling, torrent category).
- `qbittorrent` - WebUI credentials (blank = preview mode), tags, extra client options.
- `seadex` - how a release is chosen: tracker and tag filters, best/dual-audio preference, the private-release policy.
- `imports` - the wait-for-completion and Sonarr manual-import pass.
- `notifications` - the Discord webhook and a generic JSON webhook.
- `schedule` - the bare-metal loop's cadence.
- `mappings` - override or disable the ID-mapping sources.
- `advanced` - request pacing, cache lifetime, arr-activity detection, log level and format.

Every key, with its default, allowed values, and description, is in [docs/configuration.md](docs/configuration.md) - generated from the source, so it is always current.
The starter config carries the same documentation as comments, and a `$schema` line gives editors completion and validation.
The config is validated on load: an unknown or misspelled key fails with an error naming it rather than being silently ignored.

`pearlarr config validate` checks the file and reports what a run would use; `pearlarr config show` prints the effective configuration with secrets redacted - safe to paste into a bug report.

## Scope and limitations

- **Downloads come from public trackers only**, currently Nyaa, AnimeTosho, and RuTracker.
  A winning release on another public tracker is skipped with a warning and re-considered once support lands.
- **Private releases are never grabbed** - SeaDex carries no download link for them, and no private-tracker auth is supported.
  `seadex.private_releases` decides what happens when a title's preferred release is private-only.
- **qBittorrent is the only download client**; Usenet is out of scope.
  More clients and trackers are on the roadmap.
- **The supported interfaces** are the CLI, the config schema, the JSON event stream, and the notification payloads.
  Every Python import path is internal and may change without notice.
- **Support is best-effort by a single maintainer.**
  [SECURITY.md](SECURITY.md) states what is promised - notably that logs and `config show` output never contain secrets, so they are safe to paste.

## Compatibility

| | Supported |
| --- | --- |
| Sonarr | 4.x (the v3 API) |
| Radarr | 5.x (the v3 API) |
| qBittorrent | 4.1 or newer, WebUI enabled |
| Python | 3.13+ (the Docker image ships its own 3.14) |
| OS | Linux, macOS, Windows (CI covers Linux and Windows); Docker images for amd64 and arm64 |

## Documentation

- [docs/getting-started.md](docs/getting-started.md) - the first-sync walkthrough, install to verified grab.
- [docs/configuration.md](docs/configuration.md) - every setting: defaults, allowed values, semantics.
- [docs/cli.md](docs/cli.md) - every command and option, with the exit codes.
- [docs/deployment.md](docs/deployment.md) - Docker in depth, scheduling, backups, upgrades, uninstalling.
- [docs/output.md](docs/output.md) - console, logs, the JSON event stream, and webhook payloads.
- [docs/troubleshooting.md](docs/troubleshooting.md) - symptoms, the messages behind them, and the fixes.
- [docs/architecture.md](docs/architecture.md) - how Pearlarr is put together, and every external host it talks to.
- [CHANGELOG.md](CHANGELOG.md) - user-observable changes, with upgrade notes.
- [CONTRIBUTING.md](CONTRIBUTING.md) - dev setup, the quality gate, task playbooks.
- [SECURITY.md](SECURITY.md) - threat model, the redaction guarantee, reporting.

## Acknowledgements

Pearlarr is a fork of [SeaDexArr](https://github.com/bbtufty/seadexarr), originally created by [bbtufty](https://github.com/bbtufty).
Release data comes from the [SeaDex](https://releases.moe) project, which Pearlarr is not affiliated with; ID mappings come from [AniBridge Mappings](https://github.com/anibridge/anibridge-mappings), [Kometa Anime-IDs](https://github.com/Kometa-Team/Anime-IDs), and [Anime-Lists](https://github.com/Anime-Lists/anime-lists).

Pearlarr is licensed under the [GPL-3.0-or-later](LICENSE).
