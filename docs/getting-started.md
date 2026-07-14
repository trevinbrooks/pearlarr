# Getting started

By the end of this page Pearlarr will have grabbed its first release for real, and you will have watched it land in qBittorrent.
Everything before that is a preview you can rerun freely - nothing is grabbed until qBittorrent joins in step 6.
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
Follow the [Docker Compose](../README.md#docker-compose) setup in the README first; every command below then runs as `docker compose run --rm pearlarr <command>`, and the container writes the starter config into the mounted `config` directory for you (skip step 2, and edit that file on the host).
One catch: inside the container `localhost` is Pearlarr itself, so point `sonarr.url` and `qbittorrent.host` at the compose service name (`http://sonarr:8989`) or the host's IP, never `localhost`.

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
A typo'd or unknown key fails here with an error naming it, so a clean `OK` means the file itself is sound - but validation never touches the network, so a wrong key or dead URL passes here and surfaces only on the first run.

## 5. Run a preview

```console
$ pearlarr run single
```

Watch it work, top to bottom:

- A **boot ledger** first: one line per startup step (reading the config, refreshing the ID mappings, fetching your library, fetching the SeaDex entries).
  The first run downloads and parses the mapping sources, so it is the slowest; later runs reuse them.
- Then a **block per library title**: the SeaDex entry it resolved to, what your library already has, and what Pearlarr would do about it.
- Finally a **summary scoreboard**: how many titles were checked, what it *would grab*, what is already up to date, and what needs your attention - each "added" line names its release; each "needs action" line names the title and why Pearlarr stopped.
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

The blocks now read "adding SeaDex's recommended release" instead of "would add", and the results are cached - handled titles are skipped on later runs until SeaDex or your library changes.
A real run adds at most `advanced.max_torrents_to_add` torrents (default 10) - the preview ignored that cap, so on a large library later runs pick up where the first stopped.
(`pearlarr run single --dry-run` still simulates with credentials set, if you want one more rehearsal.)

## 7. Confirm the grab

The summary's "added" lines name each grabbed release; you will find torrents with those names in qBittorrent.
From there the usual arr flow takes over: Sonarr sees the finished download and imports it like any other.
Set `sonarr.torrent_category` if you want Pearlarr's grabs grouped under their own qBittorrent category, and `notifications.discord_url` if you want each grab posted to Discord.

## 8. Keep it running

One run every few hours is all Pearlarr needs:

- **Bare metal**: bare `pearlarr` runs the scheduled loop, one cycle every `schedule.interval_hours`; keep it alive under a process supervisor, or wire `pearlarr run single` into cron or a systemd timer if you prefer your own scheduler ([deployment.md](deployment.md#bare-metal-scheduling)).
- **Docker**: the container schedules itself; set `PEARLARR_CRON` to change the cadence ([deployment.md](deployment.md#scheduling-and-tz)).

From here:

- Every setting, with defaults and allowed values: [configuration.md](configuration.md)
- Sonarr users: the `imports` group can wait for downloads and shepherd stuck imports into Sonarr automatically ([configuration.md](configuration.md#imports))
- Something looks off: [troubleshooting.md](troubleshooting.md)
