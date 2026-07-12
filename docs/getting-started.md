# Getting started

This walkthrough takes you from nothing to a verified first sync: install, connect one arr, preview what Pearlarr would grab, then let it grab for real.
Budget ten minutes, plus however long the first download takes.

You will need:

- A running [Sonarr](https://sonarr.tv) 4.x or [Radarr](https://radarr.video) 5.x with some anime in its library, and its API key (in the arr's UI: Settings → General → Security).
- qBittorrent with the WebUI enabled - not needed until step 6; everything before that runs without it.
- Python 3.13 or newer (or Docker - see below).

The steps use Sonarr; Radarr works the same with `radarr.*` keys.

## 1. Install

```console
$ uv tool install pearlarr    # or: pipx install pearlarr
```

Running under Docker instead?
Follow the compose setup in the [README](../README.md#docker-compose-recommended) first; every command below then runs as `docker compose run --rm pearlarr <command>`, and the container writes the starter config for you (skip step 2).

## 2. Create the config

```console
$ pearlarr config init
Wrote a starter config to /home/you/.local/share/pearlarr/config.yml
```

The path varies by OS; `pearlarr paths` prints where everything lives:

```console
$ pearlarr paths
data_dir:    /home/you/.local/share/pearlarr
config:      /home/you/.local/share/pearlarr/config.yml
cache:       /home/you/.local/share/pearlarr/cache.db
mappings_db: /home/you/.local/share/pearlarr/mappings.db
logs:        /home/you/.local/share/pearlarr/logs
```

## 3. Connect Sonarr

Open `config.yml` in your editor.
Every key is documented in place, and the `$schema` line at the top gives most editors completion and validation as you type.

Fill in just the connection for now - two keys:

```yaml
sonarr:
  url: http://localhost:8989
  api_key: your-sonarr-api-key
```

Leave everything else blank; blank keys take the built-in defaults.

## 4. Validate

```console
$ pearlarr config validate
OK: /home/you/.local/share/pearlarr/config.yml is valid
  sonarr:      configured
  radarr:      not configured (runs will skip it)
  qbittorrent: not configured (preview mode - nothing is grabbed)
```

That last line is the point of the next step: with no qBittorrent credentials, runs are previews.
A typo'd or unknown key fails here with an error naming it, so a clean `OK` means the file really is what a run will use.

## 5. Preview run

```console
$ pearlarr run single
```

Watch it work, top to bottom:

- A **boot ledger** first: one line per startup step (reading the config, connecting to Sonarr, downloading the ID mappings, fetching your library).
  The first run downloads and parses the mapping sources, so it is the slowest; later runs reuse them.
- Then a **block per library title**: the SeaDex entry it resolved to, what your library already has, and what Pearlarr would do about it.
- Finally a **summary scoreboard**: how many titles were checked, what it *would add*, what is already up to date, and what needs your attention - each "would add" and "needs action" line names the exact release.
  The summary is marked `DRY RUN - qBittorrent not configured; nothing grabbed`.

Nothing was grabbed and nothing was recorded, so you can run this as often as you like.
This is the tuning loop: read the preview, adjust the release choices in the `seadex` group (tracker and tag filters, audio preference - see [configuration.md](configuration.md#seadex)), and run it again until the picks are what you would pick.

## 6. Add qBittorrent and grab for real

Fill in the qBittorrent WebUI credentials:

```yaml
qbittorrent:
  host: http://localhost:8080
  username: admin
  password: your-password
```

Then the same command grabs for real:

```console
$ pearlarr run single
```

The blocks now read "adding recommended release" instead of "would add", and the results are cached - handled titles are skipped on later runs until SeaDex or your library changes.
(`pearlarr run single --dry-run` still simulates with credentials set, if you want one more rehearsal.)

## 7. Confirm the grab

The summary's "added" lines name each grabbed release; you'll find torrents with those names in qBittorrent, downloading into your usual arr flow: Sonarr sees the finished download and imports it like any other.
Set `sonarr.torrent_category` if you want Pearlarr's grabs grouped under their own qBittorrent category, and `notifications.discord_url` if you want each grab posted to Discord.

## 8. Keep it running

One command a night is all Pearlarr needs:

- **Bare metal**: bare `pearlarr` runs the scheduled loop, one full run every `schedule.interval_hours`; wire `pearlarr run single` into cron or a systemd timer if you prefer your own scheduler.
- **Docker**: the container schedules itself; set `PEARLARR_CRON` to change the cadence ([deployment.md](deployment.md#scheduling-and-tz)).

From here:

- Every setting, with defaults and allowed values: [configuration.md](configuration.md)
- Sonarr users: the `imports` group can wait for downloads and shepherd stuck imports into Sonarr automatically ([configuration.md](configuration.md#imports))
- Something looks off: [troubleshooting.md](troubleshooting.md)
