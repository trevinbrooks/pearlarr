# Troubleshooting

Each section opens with the message it is about, quoted from the running code (`...` stands for the changing parts); find yours with your pager's search.
A test greps every quoted fragment against the source tree, so the quotes cannot silently go stale.

| Symptom | Section |
| --- | --- |
| Runs report releases but never download anything | [Nothing is ever grabbed](#nothing-is-ever-grabbed) |
| A run or cache command refuses to start | [Another run is active](#another-run-is-active) |
| Pearlarr exits immediately with a permissions error | [The data directory is not writable](#the-data-directory-is-not-writable) |
| The config file fails to load | [The config is missing or invalid](#the-config-is-missing-or-invalid) |
| Every run warns about an older config schema | [Old config schema](#old-config-schema) |
| The config refuses to load after a downgrade | [Config from a newer Pearlarr](#config-from-a-newer-pearlarr) |
| The cache refuses to load after a downgrade | [Cache from a newer Pearlarr](#cache-from-a-newer-pearlarr) |
| Every run skips both arrs | [No arr is configured](#no-arr-is-configured) |
| Sonarr or Radarr is unreachable or rejects the key | [The arr connection fails](#the-arr-connection-fails) |
| qBittorrent is unreachable | [The qBittorrent connection fails](#the-qbittorrent-connection-fails) |
| The whole run aborts before scanning any title | [The ID-mapping sources cannot be downloaded](#the-id-mapping-sources-cannot-be-downloaded) |
| Titles keep showing up as needing action | [Private-only releases](#private-only-releases) |
| Releases are skipped over their tracker | [Tracker skips](#tracker-skips) |
| A tracker filter seems ignored | [Unknown tracker names](#unknown-tracker-names) |
| A warning about file permissions on the config | [The config file is world-readable](#the-config-file-is-world-readable) |
| The first run takes ages, or a run re-checks everything | [Slow or unusually busy runs](#slow-or-unusually-busy-runs) |
| Wondering whether cache.db can be deleted | [Deleting the cache](#deleting-the-cache) |
| Looking for the log files | [Where the logs live](#where-the-logs-live) |

## Nothing is ever grabbed

```text
not configured (preview mode - nothing is grabbed)
qBittorrent not configured; nothing grabbed
```

Without qBittorrent credentials every run is a preview: the whole library is evaluated and reported - the run summary is marked `DRY RUN` - but nothing is downloaded and nothing is recorded.
This is a feature while you tune the config - see [getting-started.md](getting-started.md#5-run-a-preview) - and the fix is filling in `qbittorrent.host`, `qbittorrent.username`, and `qbittorrent.password`.
`pearlarr config validate` shows which mode you are in.

## Another run is active

```text
Another Pearlarr run is active in ... - skipping this run
Another Pearlarr run is active in ... - refusing to modify the cache
```

Pearlarr takes a lock in the data directory so overlapping runs cannot corrupt the cache; the cache commands take the same lock.
Wait for the running pass to finish (a wait-for-completion pass can hold it for a while), or find the other `pearlarr` process if you did not expect one.
A skipped scheduled cycle retries on the next one; a skipped `run single` exits 1 so your own scheduler can tell.

## The data directory is not writable

```text
Cannot write to the data directory ... - fix its permissions, or point
```

The data directory (config, caches, and logs) must be writable by the user running Pearlarr.
Fix the directory's ownership or permissions, or point `--data-dir` / `PEARLARR_DATA_DIR` somewhere writable.
Under Docker this is almost always a PUID/PGID mismatch with the mounted `./config` directory - see [deployment.md](deployment.md#permissions).

## The config is missing or invalid

```text
No config file at ... - a starter template was written - fill it in and re-run
Invalid configuration in ...
Unreadable YAML in ...
```

A run with no config writes the starter template and stops; fill it in and run again.
A validation failure lists each offending key with what is wrong - unknown or misspelled keys fail loudly rather than being silently ignored.
`pearlarr config validate` runs the same checks without starting a run, and [configuration.md](configuration.md) documents every key.

## Old config schema

```text
Config file ... uses an older config schema - migrated in memory ... run pearlarr config migrate to update the file (a backup is kept)
```

Harmless: the file was written for an older Pearlarr, and every load brings it forward in memory, so runs behave as if the file were current.
The parenthesized part of the warning names each key or value that was folded, if any.
`pearlarr config migrate` rewrites the file at the current schema and the warning stops; the previous file is kept beside it as `config.yml.bak`.
See [configuration.md](configuration.md#the-config-schema-version) for what the rewrite does and does not preserve.

## Config from a newer Pearlarr

```text
the file was written for a newer Pearlarr (schema version ... - upgrade Pearlarr
```

The `config_version` in the file is higher than this Pearlarr understands - almost always a downgraded install reading a config a newer version wrote (or migrated).
Upgrade Pearlarr back, restore the pre-migration `config.yml.bak` if you have one, or lower `config_version` by hand and fix whatever the load then rejects.

## Cache from a newer Pearlarr

```text
Cache database at ... uses schema v..., newer than this pearlarr understands ... - it was written by a newer release - upgrade pearlarr
```

The cache's schema version is higher than this Pearlarr understands - almost always a downgraded install reading a `cache.db` a newer version wrote.
Upgrade Pearlarr back, restore a cache backup taken with the older version (`pearlarr cache restore`), or move `cache.db` aside to start a fresh cache - the next run rebuilds it (see [Deleting the cache](#deleting-the-cache) for what that costs).

## No arr is configured

```text
Neither sonarr nor radarr is configured - set sonarr.url and sonarr.api_key, or
... is set but ... is not - skipping
```

A run needs at least one arr's `url` AND `api_key`; setting only one of the pair reads as "not configured" and the arr is skipped, with a warning naming the missing key.
`pearlarr config validate` shows both arrs' status and names any half-configured gap.

## The arr connection fails

```text
Could not reach ... at ...
... rejected the API key
```

"Could not reach" is a network problem: check the `url` (scheme, host, port, and any URL base your reverse proxy adds), that the arr is actually running, and that nothing between them blocks the connection.
"Rejected the API key" means the arr answered: re-copy the key from the arr's UI (Settings → General → Security) into `sonarr.api_key` / `radarr.api_key`.
A failed arr aborts only that arr's run; the other arr still runs.

## The qBittorrent connection fails

```text
qBittorrent connection failed - check qbittorrent.host, qbittorrent.username, and
```

Pearlarr talks to qBittorrent's WebUI: it must be enabled (qBittorrent → Options → Web UI), reachable at `qbittorrent.host` (include the port), and the credentials must match.
If the WebUI has "Bypass authentication for clients on localhost" enabled, username and password can be anything non-blank.

## The ID-mapping sources cannot be downloaded

```text
Could not download the id-mapping sources (...) - check your network connection
Could not fetch/parse the id-mapping sources - skipping this run
```

Pearlarr downloads the ID-mapping sources at the start of a run and cannot resolve any title without them, so a fetch or parse failure skips the whole run rather than one title.
It is almost always a first run with no network, or a proxy or firewall between Pearlarr and GitHub - the sources live on `github.com`/`objects.githubusercontent.com` and `raw.githubusercontent.com` (see [architecture.md](architecture.md#external-hosts)).
Fix the connection or allowlist those hosts and the run succeeds; a scheduled cycle retries on its own, and once a source is cached (`advanced.cache_time` days) a later network blip falls back to the cached copy instead of skipping.

## Private-only releases

```text
private-only release; private releases not supported
private-only release; no public alternative covers these files
Tip: manually grab private releases or set private_releases: fallback to
```

SeaDex's preferred release for these titles exists only on a private tracker, and Pearlarr never grabs private releases - SeaDex carries no download link for them, and no private-tracker auth is supported.
Your choices, via `seadex.private_releases` ([configuration.md](configuration.md#seadex)): grab the release yourself from the private tracker (the summary links the SeaDex entry), or set the policy to `fallback` so a public alternative is grabbed instead where one exists.
Titles with no public alternative stay in the summary's "needs action" list and are re-checked every run until one appears.

## Tracker skips

```text
(tracker ... not yet supported)
(tracker ... not in your selected list)
tracker not yet supported; grab manually
```

"Not yet supported" means the winning release lives on a public tracker Pearlarr cannot parse download links from yet (currently supported: Nyaa, AnimeTosho, and RuTracker); the title is re-considered once support lands, or grab it manually meanwhile.
"Not in your selected list" is your own `seadex.trackers` filter doing its job - add the tracker to the list if you want releases from it.

## Unknown tracker names

```text
Unknown seadex.trackers value(s) ignored by matching:
```

A name in `seadex.trackers` matched no tracker SeaDex uses, so it filters nothing - usually a typo.
The warning lists the known names (matching is case-insensitive); the generated table in [configuration.md](configuration.md#seadex) has them too.

## The config file is world-readable

```text
is readable by other users and holds API keys -
```

`config.yml` holds API keys and passwords in plain text, so Pearlarr warns when other users on the machine can read it.
The warning names the exact `chmod` command to run; `pearlarr config init` creates the file owner-only to begin with.

## Slow or unusually busy runs

```text
history gap - rechecking all entries
Matching settings changed - rechecking cached entries
```

Four situations make a run noticeably slower, all expected:

- **The first run** downloads and parses the ID-mapping sources and evaluates the whole library from scratch.
  On a large library this takes minutes; later runs reuse the parsed mappings and the cache, and typically finish in well under a minute when little changed.
- **A long gap since the last run** (or a restored cache) exceeds the arr-activity lookback, so change detection cannot vouch for the interval and every cached title is re-checked once - that is the "history gap" note above.
- **A changed matching preference** (anything in the `seadex` group, or `imports.languages_*`) makes the next full run re-check every cached title against the new rules once - the "Matching settings changed" note above; see [configuration.md](configuration.md#configuration-changes-and-the-cache).
- **A SeaDex or mapping-source hiccup** makes affected titles count as unchecked; they are simply retried next run.

Runs are also deliberately paced (`advanced.sleep_time`) to be a polite API citizen, so "slow" is partly by design.

## Deleting the cache

`cache.db` is regenerable: deleting it loses no media, and the next run rebuilds it.
But it holds the "already handled" memory, pending-import state, and grab history - so the next run re-evaluates the whole library, may re-notify about things it already told you about, and forgets in-flight imports.
Prefer `pearlarr cache remove` over deleting the file (it takes the run lock and cleans up the WAL/SHM sidecars).
Config edits never require it: a changed matching preference is detected and re-checked automatically, keeping all of the state above - see [configuration.md](configuration.md#configuration-changes-and-the-cache).
Removing the cache is a factory reset for when the database itself is suspect (a failed `pearlarr cache check`, a restore gone wrong), not routine maintenance.

## Where the logs live

Every run writes `Pearlarr.log` under the data directory's `logs/` folder; `pearlarr paths` prints the exact location, and the per-OS defaults are listed in [deployment.md](deployment.md#the-data-directory).
Previous runs are kept as dated `Pearlarr.log.<timestamp>` backups for `advanced.log_retention_days` days - see [output.md](output.md#the-log-file).
Logs are safe to paste into an issue: they never contain API keys, passwords, or webhook URLs ([SECURITY.md](../SECURITY.md#the-redaction-guarantee)).
