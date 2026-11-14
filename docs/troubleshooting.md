# Troubleshooting

Each section below is about an error or warning Pearlarr can print, with what it means and how to fix it.[^verified]
Find yours in the table, or search your pager for the message text.
In the section headings, `...` stands for the parts of a message that change between runs.

| Symptom | Section |
| --- | --- |
| Runs report releases but never download anything | [Nothing is ever grabbed](#nothing-is-ever-grabbed) |
| A run or cache command won't start | [Another run is active](#another-run-is-active) |
| Pearlarr exits immediately with a permissions error | [The data directory is not writable](#the-data-directory-is-not-writable) |
| The config file fails to load | [The config is missing or invalid](#the-config-is-missing-or-invalid) |
| A run reports an older config schema | [Old config schema](#old-config-schema) |
| The config won't load after a downgrade | [Config from a newer Pearlarr](#config-from-a-newer-pearlarr) |
| The cache won't load after a downgrade | [Cache from a newer Pearlarr](#cache-from-a-newer-pearlarr) |
| Neither arr is run | [No arr is configured](#no-arr-is-configured) |
| Sonarr or Radarr is unreachable or rejects the key | [The arr connection fails](#the-arr-connection-fails) |
| qBittorrent is unreachable | [The qBittorrent connection fails](#the-qbittorrent-connection-fails) |
| The whole run aborts before scanning any title | [The ID-mapping sources cannot be downloaded](#the-id-mapping-sources-cannot-be-downloaded) |
| Titles keep showing up as needing action | [Private-only releases](#private-only-releases) |
| A title names a torrent Pearlarr never added | [A torrent Pearlarr never added](#a-torrent-pearlarr-never-added) |
| Releases are skipped over their tracker | [Tracker skips](#tracker-skips) |
| A tracker filter seems ignored | [Unknown tracker names](#unknown-tracker-names) |
| A warning about file permissions on the config | [The config file is world-readable](#the-config-file-is-world-readable) |
| The first run takes ages, or a run re-checks everything | [Slow or unusually busy runs](#slow-or-unusually-busy-runs) |
| Wondering whether cache.db can be deleted | [Deleting the cache](#deleting-the-cache) |
| Looking for the log files | [Where the logs live](#where-the-logs-live) |

## Nothing is ever grabbed

```text
not configured (dry runs only - nothing is grabbed)
qBittorrent not configured; nothing grabbed
```

Without qBittorrent credentials every run is a dry run: the whole library is evaluated and reported, the run summary is marked `DRY RUN`, and nothing is grabbed or cached.
This is a feature while you tune the config (see [getting-started.md](getting-started.md#5-do-a-dry-run)), and the fix is filling in `qbittorrent.host`, `qbittorrent.username`, and `qbittorrent.password`.
`pearlarr config validate` shows which mode you are in.

## Another run is active

```text
Another Pearlarr run is active in ... - skipping this run
Another Pearlarr run is active in ... - not modifying ...
```

Pearlarr takes a lock in the data directory so overlapping runs cannot corrupt the cache. The cache commands and `pearlarr config migrate` take the same lock, so they don't modify the cache or the config while a run is live.
Wait for the other run to finish (its wait pass can keep it going for a while), or find the other `pearlarr` process if you did not expect one.
A skipped scheduled run tries again at the next scheduled time. A skipped `run single` exits 1 so your own scheduler can tell.

## The data directory is not writable

```text
Cannot write to the data directory ... - fix its permissions, or point
```

The data directory (config, caches, and logs) must be writable by the account running Pearlarr.
Fix the directory's ownership or permissions, or point `--data-dir` / `PEARLARR_DATA_DIR` somewhere writable.
Under Docker this is almost always a PUID/PGID mismatch with the mounted `./config` directory (see [deployment.md](deployment.md#permissions)).

## The config is missing or invalid

```text
No config file at ... - a template was written - fill it in and re-run
Invalid configuration in ...
Unreadable YAML in ...
```

A run with no config writes a template and stops. Fill it in and run again.
A validation failure lists each offending key with what is wrong: unknown or misspelled keys fail loudly rather than being silently ignored.
`pearlarr config validate` runs the same checks without starting a run, and [configuration.md](configuration.md) documents every key.

## Old config schema

```text
Config migrated to the current schema - previous file saved as ...
Config ... uses an older schema and could not be rewritten in place - ... run pearlarr config migrate to update the file (a backup is kept)
```

The file was written for an older Pearlarr. Every load brings it forward in memory, and a run then rewrites the file itself at the current schema: the first line, with the previous file kept beside it as `config.yml.bak` and the changes written to the log file.
The warning appears instead only when the file could not be rewritten - typically a read-only config file or directory, or a config bind-mounted into the container as a single file, which cannot be replaced atomically (mount its directory instead). The run still behaves as if the file were current; fix write access, or run `pearlarr config migrate` to see exactly why the rewrite fails.
See [configuration.md](configuration.md#the-config-schema-version) for what the rewrite does and does not preserve.

## Config from a newer Pearlarr

```text
the file was written for a newer Pearlarr (schema version ... - upgrade Pearlarr
```

The `config_version` in the file is higher than this Pearlarr understands, almost always a downgraded install reading a config a newer version wrote (or migrated).
Upgrade Pearlarr back, restore the pre-migration `config.yml.bak` if you have one, or lower `config_version` by hand and fix whatever the load then rejects.

## Cache from a newer Pearlarr

```text
Cache database at ... uses schema v..., newer than this pearlarr understands ... - it was written by a newer release - upgrade pearlarr
```

The cache's schema version is higher than this Pearlarr understands, almost always a downgraded install reading a `cache.db` a newer version wrote.
Upgrade Pearlarr back, restore a cache backup taken with the older version (`pearlarr cache restore`), or move `cache.db` aside to start a fresh cache, which the next run rebuilds (see [Deleting the cache](#deleting-the-cache) for what that costs).

## No arr is configured

```text
Neither sonarr nor radarr is configured - set sonarr.url and sonarr.api_key, or
... is set but ... is not - ... is not run
```

A run needs at least one arr's `url` AND `api_key`. Setting only one of the pair reads as "not configured" and the arr is not run, with a warning naming the missing key.
`pearlarr config validate` shows both arrs' status and names any half-configured gap.

## The arr connection fails

```text
Could not reach ... at ...
... rejected the API key
```

"Could not reach" is a network problem: check the `url` (scheme, host, port, and any URL base your reverse proxy adds), that the arr is actually running, and that nothing between them blocks the connection.
"Rejected the API key" means the arr answered: re-copy the key from the arr's UI (Settings → General → Security) into `sonarr.api_key` / `radarr.api_key`.
A failed arr aborts only that arr's run. The other arr still runs.

## The qBittorrent connection fails

```text
qBittorrent connection failed - check qbittorrent.host, qbittorrent.username, and
```

Pearlarr talks to qBittorrent's WebUI: it must be enabled (qBittorrent → Options → Web UI), reachable at `qbittorrent.host` (include the port), and the credentials must match.
If the WebUI has "Bypass authentication for clients on localhost" enabled, username and password can be anything non-blank.

## The ID-mapping sources cannot be downloaded

```text
Could not download the mapping sources (...) - check your network connection
Could not fetch/parse the mapping sources - skipping this run
```

Pearlarr downloads the ID-mapping sources at the start of a run and cannot resolve any title without them, so a fetch or parse failure skips the whole run rather than one title.
It is almost always a first run with no network, or a proxy or firewall between Pearlarr and GitHub.
The sources live on `github.com`/`objects.githubusercontent.com` and `raw.githubusercontent.com` (see [architecture.md](architecture.md#external-hosts)).
Fix the connection or allowlist those hosts and the run succeeds. A scheduled run tries again on its own, and once a source is cached (`mappings.refresh_days` days) a later network blip uses the cached copy instead of stopping the run.

## Private-only releases

```text
private-only release; private releases not supported
private-only release; no fallback covers these files
Tip: manually grab private releases or set private_releases: fallback to
```

SeaDex's preferred release for these titles exists only on a private tracker, and Pearlarr never grabs private releases: SeaDex carries no download link for them, and no private-tracker auth is supported.
Your choices, via `seadex.private_releases` ([configuration.md](configuration.md#seadex)): grab the release yourself from the private tracker (the summary links the SeaDex entry), or set the policy to `fallback` so a public release is grabbed instead where one exists.
Titles that no public release covers stay in the summary's "needs action" list and are re-checked every run until one does.

## A read the placement needs failed

The run summary lists a title under "needs action" with "a read the placement needs failed; will retry next run". A read the grab-time placement needed did not answer: Sonarr's parse of one of that release's file names, the episode list of a series the torrent is already in qBittorrent for, or the SeaDex listing or episode list of a title the torrent is listed under. Without the SeaDex listings nothing of the release is placed, since another title may have some of its files. A release not yet in qBittorrent is not grabbed. One already in qBittorrent for another title of the show stays there, and this title waits to import from it. The title is not cached, so it is re-checked on the next run, and the row stops once the read answers. A row that repeats every run means the same read fails every time, such as Sonarr's parse of one file name. The log names the file and the debug log the release. Fix the cause in Sonarr, or download the release and import it by hand.

## A release whose numbers do not match its SeaDex listing

The run summary lists a title under "needs action" with "numbering does not match its SeaDex listing; import it by hand", and the release's "skipped" line names the mismatch. The release can be a torrent of specials whose `S00Exx` numbers follow another numbering than the specials SeaDex lists it for, such as an older TVDB numbering or TMDB's specials order: it has as many files as the listed specials, and at least one of its numbers is not a listed special. Importing such a torrent by its own numbers would put files on the wrong episodes. When the AniBridge mappings pair its numbers one to one with the listed specials, Pearlarr grabs the release and imports each file onto its paired special, and no row appears. If an older version already imported part of the release by its own numbers, onto specials the pairing disagrees with, the pairing is not used for the rest. The release gets this row instead, and Pearlarr does not grab it, when one of its numbers has no pairing, the pairings do not match the listed specials one to one, a pairing lands on a special another title of the show also maps, the titles listing the release pair against different TMDB shows, a file's title or size points at a different special than its pairing, or another Sonarr series SeaDex recommends the release for places one of its files by its number. Download it and import it by hand, matching each file to its episode by content. The row stops once every special of the title has one of the release's files at its listed size, or a file of another preferred release covering that special. A special the release has no file for keeps the row.

It can also be a release of any season whose file names disagree with their own numbers: a count beside an episode key that names another episode, a key Sonarr worked out from bare digits that its match doesn't back, a bare number Sonarr reads as a season and episode that is another episode's absolute number, a number followed by words that name no episode, or a torrent numbered past its season's end when the season's absolute numbering counts a special. When another TVDB order of the series accounts for the torrent's numbers on the episodes it is listed for, such as a DVD order that counts a special mid-season, Pearlarr grabs the release and imports its files by that order, and no row appears. Otherwise Pearlarr can't tell which number is right, so it does not grab the release. Download it and import it by hand, matching each file to its episode by content. The row stops once every episode of the title has one of the release's files at its listed size, or a file of another preferred release covering that episode.

## A release whose episode titles contradict its numbers

The run summary lists a title under "needs action" with "episode titles contradict its numbers; import it by hand", and the release's "skipped" line names the contradiction. Some of the release's file names carry episode titles, and those titles name other episodes than the files' numbers do, so the files with no title can't be placed by their numbers either. None of the release's files could be placed on the title's episodes, and Pearlarr does not grab it: importing it by its numbers would put files on the wrong episodes. Download it and import it by hand, matching each file to its episode by its title or its content. The row stops once every episode of the title has one of the release's files at its listed size, or a file of another preferred release covering that episode. An episode the release has no file for keeps the row.

When some of the release's files are placed by their titles, Pearlarr grabs the release and imports those, and no row appears. The rest are left for a hand import: the import's line counts them, the wait notification and the log name them, and the download stays tracked, named again on every run, until a hand import fills its episodes. Removing the download from qBittorrent ends its pending import. Pearlarr then checks the title again, and grabs the release again only if the files it placed are no longer in your library.

## A release whose files place under another title

The run summary lists a title under "needs action" with "its files place under another title SeaDex lists it with; any it could not place need a hand import", and the release's "skipped" line counts the files, such as "12 files are another title's, 6 could not be placed". SeaDex lists the release under more than one title of the show, such as a season and its specials, and Pearlarr places its files under every one of them. None of them place on this title's episodes and some place on another's, so the release is that title's to grab, and Pearlarr does not grab it for this one. The other title's grab imports its files onto that title's episodes, unless that title is in `ignore_anilist_ids`: an ignored title places only the files no kept title of its kind (a season's or the specials') places, and it is never grabbed for, so nothing imports them. The count of another title's files also covers files that title's mapping read but left for a hand import. That title's own row reports them, or no row does when it is ignored, so check them by hand too. The files counted as not placed are the ones this title may be missing: download the release and import them by hand onto this title's episodes, matching each file to its episode by content. The row ends once one of this title's episodes has a file of the release at its listed size, whether a hand import put it there or the other title's import did, which happens when the two titles share episodes. When the line counts no unplaced files and no import of the other title's can reach this title's episodes, because that title is ignored or the two share none, the row stays every run while this title is checked.

## A file named by number alone that was not imported

The import warns "N files could not be placed on an episode and were not imported" beside the files it did import, or reports the download as "unplaced" ("no file was placed on an episode; import by hand in Sonarr") when none were, and the log names files such as "Show - 07.mkv": a bare number, no season or episode key, which Sonarr's parse put on no episode of the title, or on one outside it. Pearlarr places such a file by working out what the release group counted: the series' episodes from the first, the title's episodes from one, Sonarr's own episode numbers for the title's season, every title the release is listed under counted straight through, and TVDB's absolute numbers. When every way of counting that fits the release names the same episode, the file is imported there, so a second season's release numbered from one, or from the series' first episode, imports either way, and a release listed under a title of nothing but specials counts them from one. Only Sonarr's aired episode order is read this way. A release numbered from one that fits a season of another TVDB order instead, such as a DVD order that splits a double-length opener in two, is placed by that order (see [Files from a DVD-order torrent that Sonarr doesn't list](#files-from-a-dvd-order-torrent-that-sonarr-doesnt-list)). A file is left for a hand import when a title the release is listed under has an episode count on AniList that does not match the episodes Sonarr has for it, when the release repeats a number under one title (two titles each numbered from one, K-ON! and K-ON!! in one release, are read apart when each names one listed title), when two ways of counting that both fit name different episodes (a second season's release numbered 14 to 22, say, where 14 can be the season's fourteenth episode or the series'), when the series title in the file name is not the series', one of its alternate titles in Sonarr, or an AniList title of a listed entry, or when an episode's absolute number in Sonarr is not its position in the series. Download the release and import those files by hand, matching each to its episode by content.

## Files from a DVD-order torrent that Sonarr doesn't list

```text
... still missing in Sonarr
... has a file that isn't from a trusted group
this torrent is numbered by the ... order, and ... of its files are ... episodes
that aren't in Sonarr's list, so Pearlarr can't tell where they belong
```

An import's row ends with an episode that is "still missing in Sonarr", or one that "has a file that isn't from a trusted group". Under it are a `reason` row and the files it is about, each behind an episode number such as `DVD S07E01`. The wait notification and the log say the same in sentences.

TVDB keeps several episode orders for a series: the aired order Sonarr lists, and often a DVD order or others. Another order can number an episode Sonarr doesn't list, such as each half of an opener that aired as one double-length episode. When a torrent is numbered by such an order, Pearlarr imports the files whose numbers stand for an episode Sonarr lists and leaves the rest in the download folder. Nothing in TVDB's data says which aired episode a half belongs to, so Pearlarr never places one.

When every episode of the title has a file from a trusted group once the rest are imported, the import ends there and says nothing about the files it left. The warning shows while an episode is empty or has some other file, since that is where a left file may belong. The download stays tracked and the warning repeats on every run, with no age limit. To end it:

- You import one of the named files onto that episode by hand in Sonarr. That is your call: half of a double-length episode is still only half of it. The next run sees the episode filled and ends the pending import.
- You remove the download from qBittorrent. The files already imported stay in your library. Pearlarr then checks the title again, and grabs the release again only if those files are no longer there.

When more than one TVDB order accounts for a torrent, Pearlarr can't always tell which one such a file follows. That file doesn't keep the pending import open. The import counts it in its "could not be placed on an episode" warning, unless its number is an episode of another title in Sonarr's list, which the import doesn't mention.

## A torrent Pearlarr never added

```text
already in qBittorrent
release already in qBittorrent
Could not read qBittorrent's torrents (...) - this run may grab a release that another torrent already has
```

The first two lines are a title's status when qBittorrent already has its release. That can be a torrent Pearlarr never added: SeaDex's own torrent that someone else added, or one that has exactly the files of the preferred release, by name and size, under another hash, such as Sonarr's grab of the release from another tracker. Pearlarr treats that torrent as its own grab instead of adding SeaDex's torrent into the same download folder.

A torrent under another hash counts when it is in the arr's download category (`torrent_category`), downloads every one of its files, is not in an error state, and has not finished. When Pearlarr imports for Sonarr (an `imports.wait_mode` other than `off`), a finished one counts too, as does one in `post_import_category`. If the torrent is removed from qBittorrent before its files are imported, a later run grabs SeaDex's release.

The warning means qBittorrent could not be read when Pearlarr looked through those categories, so for the rest of that run a release is grabbed even when another torrent has its files. If such a grab went into a folder another torrent is using, remove it from qBittorrent and keep its files, which the other torrent shares.

## Tracker skips

```text
(tracker ... not yet supported)
(tracker ... not in your selected list)
tracker not yet supported; grab manually
```

"Not yet supported" means the winning release lives on a public tracker Pearlarr cannot parse download links from yet (currently supported: Nyaa, AnimeTosho, and RuTracker). The title is re-considered once support lands, or grab it manually meanwhile.
"Not in your selected list" is your own `seadex.trackers` filter doing its job: add the tracker to the list if you want releases from it.

## Unknown tracker names

```text
Unknown seadex.trackers value(s) ignored:
```

A name in `seadex.trackers` names no tracker SeaDex uses, so it filters nothing, usually a typo.
The warning lists the known names (matching is case-insensitive). The generated table in [configuration.md](configuration.md#seadex) has them too.

## The config file is world-readable

```text
is readable by other users and holds API keys -
```

`config.yml` contains API keys and passwords in plain text, so Pearlarr warns when other users on the machine can read it.
The warning names the exact `chmod` command to run. `pearlarr config init` creates the file owner-only to begin with.

## Slow or unusually busy runs

```text
history gap - checking every title again
Selection settings changed - checking cached titles again
```

Four situations make a run noticeably slower, all expected:

- **The first run** downloads and parses the ID-mapping sources, fetches the TVDB episode orders of each Sonarr series it checks from Sonarr's metadata service (SkyHook), and evaluates the whole library from scratch.
  On a large library this takes minutes. Later runs reuse the parsed mappings and the cache, and typically finish in well under a minute when little changed.
- **A long gap since the last run** (or a restored cache) exceeds the arr-activity lookback, so change detection can't cover the interval and every cached title is re-checked once (the "history gap" note above).
- **A changed selection setting** (anything in the `seadex` group, or `imports.languages_*`), or an upgrade whose release notes say it re-checks cached titles, makes the next full run re-check every cached title once (the "Selection settings changed" note above). See [configuration.md](configuration.md#configuration-changes-and-the-cache).
- **A SeaDex, SkyHook, AniList, or mapping-source hiccup** makes affected titles count as unchecked. They are simply retried next run. A SkyHook hiccup (`SkyHook failed ... - re-checking cached titles until it answers`) leaves unchecked the titles of a Sonarr series whose TVDB episode orders were never fetched. Like a SeaDex hiccup, it also stops the run from marking its work done: the next run re-reads the same Sonarr activity, and an unfinished "Selection settings changed" re-check runs again. An AniList hiccup (`AniList ..., skipping further lookups this run`) leaves unchecked the titles of a Sonarr series when Pearlarr has no stored AniList data, air dates included, for a title it looked up. Unlike a SkyHook hiccup, it doesn't stop the run from marking its work done.

Runs are also deliberately paced (`advanced.sleep_time`) to be a polite API citizen, so "slow" is partly by design.

## Deleting the cache

`cache.db` is regenerable: deleting it loses no media, and the next run rebuilds it.
But it stores which titles were already checked and the pending imports, so the next run re-evaluates the whole library, may re-notify about things it already told you about, and forgets in-flight imports.
Prefer `pearlarr cache remove` over deleting the file (it takes the run lock and cleans up the WAL/SHM sidecars).
Config edits never require it: a changed selection setting is detected and re-checked automatically, keeping all of the state above (see [configuration.md](configuration.md#configuration-changes-and-the-cache)).
Removing the cache is a factory reset for when the database itself is suspect (a failed `pearlarr cache check`, a restore gone wrong), not routine maintenance.

## Where the logs live

Every run writes `Pearlarr.log` under the data directory's `logs/` folder. `pearlarr paths` prints the exact location, and the per-OS defaults are listed in [deployment.md](deployment.md#the-data-directory).
Previous runs are kept as dated `Pearlarr.log.<timestamp>` backups for `advanced.log_retention_days` days (see [output.md](output.md#the-log-file)).
Logs are safe to paste into an issue: they never contain API keys, passwords, or webhook URLs ([SECURITY.md](../SECURITY.md#the-redaction-guarantee)).

[^verified]: Every quoted message is checked against Pearlarr's source by a test, so this page can't silently fall out of date.
