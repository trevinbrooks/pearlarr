# Getting started

This guide takes you from a fresh install to grabbing your first release.
Pearlarr only grabs once qBittorrent is connected in step 6, so every run before then is a dry run.
It shows exactly what would be grabbed without touching anything, letting you rerun and refine your setup as often as you like.

You will need:

- A running [Sonarr](https://sonarr.tv) 4.x or [Radarr](https://radarr.video) 5.x with some anime in its library, and its API key (in the arr's UI: Settings → General → Security).
- qBittorrent with the WebUI enabled (not needed until step 6).
- Python 3.13 or newer (or Docker, see below).

The steps use Sonarr. Radarr works the same with `radarr.*` keys.

## 1. Install

```console
$ uv tool install pearlarr    # or: pipx install pearlarr
```

Running under Docker instead?
Follow the [Docker Compose](../README.md#docker-compose) setup in the README first. Every command below then runs as `docker compose run --rm pearlarr <command>`, and the container writes the starter config into the mounted `config` directory for you (skip step 2, and edit that file on the host).
One catch: inside the container `localhost` is Pearlarr itself, so point `sonarr.url` and `qbittorrent.host` at the compose service name (`http://sonarr:8989`) or the host's IP, never `localhost`.

## 2. Create the config

```console
$ pearlarr config init
Wrote the config template to /home/you/.local/share/pearlarr/config.yml
```

The path varies by OS. `pearlarr paths` prints where everything lives:

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
Every key is documented in the file itself, and the `$schema` line at the top gives most editors completion and validation as you type.

For now, fill in just the two connection keys:

```yaml
sonarr:
  url: http://localhost:8989
  api_key: your-sonarr-api-key
```

Leave everything else blank. Blank keys take the built-in defaults.

## 4. Validate

```console
$ pearlarr config validate
OK: /home/you/.local/share/pearlarr/config.yml is valid
  sonarr:      configured
  radarr:      not configured (it is not run)
  qbittorrent: not configured (dry runs only - nothing is grabbed)
```

That last line is the point of the next step: with no qBittorrent credentials, runs are dry runs.
A typo'd or unknown key fails here with an error naming it, so a clean `OK` means the file itself is sound.
Validation never touches the network, though, so a wrong API key or dead URL passes here and shows up only on the first run.

## 5. Do a dry run

```console
$ pearlarr run single
```

Watch it work, top to bottom:

- The **boot steps** first: one line per step (reading the config, refreshing the ID mappings, fetching your library, fetching the SeaDex entries).
  The first run downloads and parses the mapping sources, so it is the slowest. Later runs reuse them.
- Then a **block per title**: the SeaDex entry it resolved to, what your library already has, and what Pearlarr would do about it.
- Finally the **summary**: how many titles were checked, what it *would grab*, what is already up to date, and what needs your attention.
  Each block under "grabbed" names its release, and each block under "needs action" names the title and why Pearlarr stopped.
  The summary is marked `DRY RUN - qBittorrent not configured; nothing grabbed`.

Nothing was grabbed and nothing was cached, so you can run this as often as you like.
This is the tuning loop: read the dry run's output, adjust the release choices in the `seadex` group (tracker and tag filters, audio preference, all in [configuration.md](configuration.md#seadex)), and run it again until it chooses what you would choose.

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

The blocks now read "grabbing" and "grabbed" instead of "would grab", and the results are cached, so later runs don't check those titles again until SeaDex or your library changes.
A real run adds at most `advanced.max_torrents_to_add` torrents (default 10) and holds the rest, so on a large library later runs grab the titles the first run held. The dry run ignored that cap.
(`pearlarr run single --dry-run` still simulates with credentials set, if you want one more rehearsal.)

## 7. Confirm the grab

Under "grabbed", the summary names each release Pearlarr grabbed. You will find torrents with those names in qBittorrent.
From there the usual arr flow takes over: Sonarr sees the finished download and imports it like any other.
Set `sonarr.torrent_category` if you want Pearlarr's grabs grouped under their own qBittorrent category (Sonarr's qBittorrent client doesn't watch that category, so Pearlarr imports those downloads itself), and `notifications.discord_url` if you want each grab posted to Discord.

## 8. Keep it running

One run every few hours is all Pearlarr needs:

- **Bare metal**: bare `pearlarr` runs the scheduled loop, one run every `schedule.interval_hours`. Keep it alive under a process supervisor, or wire `pearlarr run single` into cron or a systemd timer if you prefer your own scheduler ([deployment.md](deployment.md#bare-metal-scheduling)).
- **Docker**: the container schedules itself. Set `PEARLARR_CRON` to change the cadence ([deployment.md](deployment.md#scheduling-and-tz)).

From here:

- Every setting, with defaults and allowed values: [configuration.md](configuration.md)
- Sonarr users: the `imports` group waits for downloads and gets them imported into Sonarr - on by default, `imports.wait_mode: off` disables it ([configuration.md](configuration.md#imports))
- Something looks off: [troubleshooting.md](troubleshooting.md)
