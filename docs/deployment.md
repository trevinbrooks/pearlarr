# Deployment

Running Pearlarr in production: the Docker image in detail, bare-metal scheduling, what lives in the data directory, backups, upgrades, what an interruption does, and how to uninstall cleanly.
For the ten-minute first run, start with the README's [Install](../README.md#install) and [First run](../README.md#first-run) sections instead.

## The Docker image

`ghcr.io/trevinbrooks/pearlarr` is a multi-arch image (linux/amd64 and linux/arm64) on a `python:3.14-slim` base.
It runs as the unprivileged `pearlarr` user (uid 1000) unless the compose `user:` key says otherwise, and sets `PEARLARR_DATA_DIR=/config`, so the single `/config` mount holds the config, caches and logs together.

| Tag | Meaning |
| --- | --- |
| `vX.Y.Z` | A pinned release - the recommended choice. |
| `latest` | The most recently pushed release tag. |
| `main` | Every push to `main` - the cutting edge, may break. |

For stronger pinning, reference the image by digest (`ghcr.io/trevinbrooks/pearlarr@sha256:...`).
Images are published with build provenance attestations, so you can verify an image was built by this repository's workflow before running it:

```console
$ gh attestation verify oci://ghcr.io/trevinbrooks/pearlarr:latest --owner trevinbrooks
```

### What the entrypoint does

Started with arguments, the container runs them as a one-off CLI invocation and exits - that is what `docker compose run --rm pearlarr <args>` uses.
Started bare (the normal service), it walks four steps:

1. **Writability check** - if `/config` is not writable by the container's uid, it stops immediately with the message `ERROR: /config is not writable by uid 1000 (gid 1000).` and the fix (chown the host directory, or point `user:`/PUID/PGID at its owner).
   A bind-mount directory auto-created by Docker is root-owned, which is why the quick start says to `mkdir -p ./config` yourself.
2. **First boot** - if `/config/config.yml` does not exist, `pearlarr config init` writes the starter template and the container exits so you can fill it in.
   With `restart: unless-stopped` Docker restarts it; the file now exists, so the container stays up (each run fails with a one-line error until the file is filled in).
3. **Catch-up run** - with `PEARLARR_RUN_ON_START=true` (the default), one `pearlarr run single` runs at container start.
   Its failure does not stop the container; the exit code is logged and the schedule proceeds.
4. **Schedule** - [supercronic](https://github.com/aptible/supercronic) becomes PID 1 and runs `pearlarr run single` on the `PEARLARR_CRON` schedule.

### Scheduling and TZ

`PEARLARR_CRON` (default `0 */6 * * *`, every six hours) is a standard five-field cron expression, evaluated in the `TZ` timezone (default UTC; `tzdata` is installed in the image).
The config's `schedule.interval_hours` does **not** apply in the container - it only drives the bare-metal `pearlarr run scheduled` loop.

Overlapping runs are double-guarded: supercronic will not start a run while the previous one is still going, and Pearlarr holds its own cross-process file lock besides (see [Interruption semantics](#interruption-semantics)).

### Permissions

The compose example's `user: "${PUID:-1000}:${PGID:-1000}"` line sets the uid/gid the container runs as.
PUID and PGID are compose *interpolation* variables - set them in an `.env` file next to the compose file or in the shell.
They are **not** `environment:` keys; an `environment:` entry is silently ignored for this.
Match them to the owner of the `./config` host directory so the container can write its config, caches and logs there.

The starter config is written with mode `600`, which works when your host editor runs as the same uid as the container.
If you loosen it (say `chmod 644`), Pearlarr warns on every run that the file - which holds API keys - is readable by other users.

### Custom certificate authorities

Pearlarr verifies TLS against the operating-system trust store (via [truststore](https://truststore.readthedocs.io)), not a bundled CA list.
On a bare-metal host, a corporate or homelab CA installed system-wide - `update-ca-certificates` on Debian/Ubuntu, Keychain on macOS, certmgr on Windows - is picked up automatically.
In the container the OS store carries only the public CAs, so mount your CA and point OpenSSL at it instead of disabling verification:

```yaml
environment:
  - SSL_CERT_FILE=/config/ca.pem
```

`sonarr.verify_ssl: false` (and its `radarr` twin) skips verification entirely for that arr and should be the last resort.

### Stopping, and `stop_grace_period`

`docker stop` sends SIGTERM to supercronic, which stops scheduling and waits for an in-flight run to finish; Docker escalates to SIGKILL after `stop_grace_period`, default ten seconds.
Ten seconds is fine for the default configuration, but a blocking `imports.wait_mode` can legitimately hold a run for up to `imports.wait_timeout` (default one hour) - raise the grace period so a routine `docker stop` does not SIGKILL a mid-import run:

```yaml
stop_grace_period: 1h
```

See [Interruption semantics](#interruption-semantics) for what a killed run leaves behind.

### Healthcheck

The image's healthcheck runs `pearlarr paths` every five minutes: the container reports healthy once the CLI starts and its data directory resolves.
It is purely informational - Docker does not restart anything on unhealthy, and a failing *run* does not make the container unhealthy.

## Bare-metal scheduling

Two options:

- **The built-in loop** - `pearlarr run scheduled` (or bare `pearlarr`) runs every configured arr each `schedule.interval_hours` (default 6), re-reading the config each cycle so edits take effect without a restart.
  Run it under a process supervisor (systemd service, launchd, etc.); it exits 0 on SIGTERM, so ordinary service stops are clean.
- **Your own scheduler** - cron or a systemd timer invoking `pearlarr run single`.
  Failed runs exit non-zero, so cron mail and `OnFailure=` hooks work as expected.

If two invocations pointing at the same data directory ever overlap, the second warns `Another Pearlarr run is active in <data_dir>; skipping this run.` and exits non-zero (in scheduled mode, the loop retries next cycle).
Running multiple instances *intentionally* is fine - give each its own data directory (`--data-dir` or `PEARLARR_DATA_DIR`), which also gives each its own lock.

## The data directory

Everything Pearlarr reads and writes lives under one directory.
The `--data-dir` flag wins over the `PEARLARR_DATA_DIR` environment variable (which the Docker image sets to `/config`), which wins over the OS-standard per-user default; `pearlarr paths` prints the resolved locations.

| OS | Default data directory |
| --- | --- |
| Linux | `~/.local/share/pearlarr` (honors `$XDG_DATA_HOME`) |
| macOS | `~/Library/Application Support/pearlarr` |
| Windows | `%LOCALAPPDATA%\pearlarr` |

What is in it, and what deleting each piece costs:

| File | What it holds | If you delete it |
| --- | --- | --- |
| `config.yml` | Your settings and credentials - the only file you author. | Gone is gone: `pearlarr config init` writes a fresh starter, but you refill it yourself. Keep it backed up. |
| `cache.db` | Run state: which entries were processed, grab history and torrent hashes, pending imports awaiting a wait pass, the arr-activity checkpoint. | Safe but costly: the next run re-evaluates the whole library, pending imports are forgotten (Sonarr still imports what it can on its own), and with `seadex.use_torrent_hash_to_filter: true` everything looks un-grabbed and gets re-downloaded. |
| `cache.backup.db` | The snapshot written by `pearlarr cache backup`. | You lose the rollback point; take a new backup. |
| `mappings.db` | The parsed ID-mapping sources. | Cheap: rebuilt from the mapping downloads on the next run. |
| `logs/` | `Pearlarr.log` plus rotated `.log.1` through `.log.9`. | Disposable. |
| `.pearlarr.lock` | The single-instance run lock. | Leave it alone - it is an empty coordination file, recreated as needed. |

## Backup and restore

Two files matter: `config.yml` and `cache.db`.
The mapping cache is regenerable; the logs are disposable.

- `pearlarr cache backup` snapshots `cache.db` to `cache.backup.db` using SQLite's online-backup API, so the copy is consistent even if taken mid-write; it writes through a temp file, so a failed backup can never destroy the previous good one.
- `pearlarr cache restore` replaces `cache.db` with a copy of the backup; the backup is kept, so a restore is repeatable.
- Both refuse to touch the cache while a run is active in the same data directory: `Another Pearlarr run is active in <data_dir>; refusing to modify the cache.`

Copying the whole data directory while Pearlarr is not running is also a complete backup.

## Upgrading

1. Read the "Upgrade notes" for the new version in [CHANGELOG.md](../CHANGELOG.md) - config-key and cache changes are always called out there.
2. Take a `pearlarr cache backup` and a copy of `config.yml`.
3. Upgrade: `docker compose pull && docker compose up -d` on Docker, `pip install --upgrade pearlarr` otherwise.

The cache schema is versioned and migrates itself at load; no manual step.

Downgrading is not supported: an older Pearlarr may not open a newer cache schema.
To roll back anyway, reinstall the older version and restore the cache backup taken with it (`pearlarr cache restore`), along with the matching `config.yml`.

## Interruption semantics

Pearlarr is safe to interrupt - Ctrl-C, `docker stop`, a crash, a power cut:

- **Decisions commit once per arr, at the end of its run.**
  A kill mid-run rolls the staged writes back, so nothing is half-recorded; the next run re-evaluates that arr from the last completed state.
- **Torrents already handed to qBittorrent stay there.**
  The next run recognizes them by hash ("already in qBittorrent") and does not add duplicates.
- **`cache.db` cannot be half-written.**
  SQLite journaling covers normal operation, and the very first write of a fresh cache builds the database in memory and promotes it to disk with an atomic rename.
- **The run lock cannot wedge.**
  It is an OS-level advisory lock, released by the kernel when the process dies - a crash never blocks future runs.

An interrupted wait pass strands nothing either: downloads keep going in qBittorrent, and the next run - re-evaluating the title, since its outcome was never committed - recognizes the torrent as already added and, with a non-`off` `imports.wait_mode`, re-registers it for the wait pass.

## Uninstalling

1. Note the data directory first: `pearlarr paths` prints it (in Docker it is your `./config` mount).
2. Remove the program - `docker compose down` and delete the image, or `pip uninstall pearlarr`.
3. Delete the data directory.

What remains, deliberately, because deleting it may not be what you want:

- **Torrents keep seeding in qBittorrent.**
  If you set `sonarr.torrent_category` / `radarr.torrent_category` or `qbittorrent.tags`, Pearlarr's torrents are filterable by that category or tag; without them there is no marker distinguishing its adds.
- **Imported files stay in your library** - they are Sonarr's/Radarr's now.
- **The Discord webhook** you created is untouched; delete it in Discord if nothing else uses it.

## Migrating from upstream SeaDexArr

Pearlarr 1.0 is a renamed fork of [bbtufty/seadexarr](https://github.com/bbtufty/seadexarr) with a changed config format; coming from upstream 0.9.x:

- **The image and package moved**: `ghcr.io/trevinbrooks/pearlarr` replaces `ghcr.io/bbtufty/seadexarr`, the PyPI package is `pearlarr`, the command is `pearlarr`, and the environment variables are `PEARLARR_*`.
- **Rewrite the config**: the format changed wholesale from flat keys to nested groups, and unknown keys now fail at load instead of being ignored.
  Move your old `config.yml` aside first - `config init` refuses to overwrite an existing file, and following its `--force` hint would destroy the values you are about to transfer.
  Then run `pearlarr config init` for a commented starter and transfer your values - for example `sonarr_url` is now `sonarr.url`, `qbit_info` is the `qbittorrent` group, `torrent_tags` is `qbittorrent.tags`, and `public_only` is replaced by the `seadex.private_releases` policy.
  [docs/configuration.md](configuration.md) documents every key.
- **The data directory is new**: upstream kept files beside the install; Pearlarr uses one OS-standard directory (see [above](#the-data-directory)).
  In Docker, mount `/config` as before - the layout inside it is Pearlarr's own.
- **The cache is not carried over**: the old `cache.json` is not read, and Pearlarr starts fresh.
  That is cheap in the default matching mode - the first run re-evaluates the library against what the arrs already have on disk and downloads nothing it finds there.
  Only with `seadex.use_torrent_hash_to_filter: true` does a fresh cache mean re-grabbing, since hash matching only knows what Pearlarr itself added.
- **`SCHEDULE_TIME` is deprecated**: set `schedule.interval_hours` (bare metal) or `PEARLARR_CRON` (Docker) instead.
  On bare metal a still-set `SCHEDULE_TIME` wins for now, with a deprecation warning; the container never reads it.
- **Python 3.13 or newer** is required for a pip install; the Docker image brings its own interpreter.
