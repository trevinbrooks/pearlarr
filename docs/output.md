# Output

Pearlarr writes four kinds of output: the console, an always-on log file, a JSON event stream, and the notification webhooks.
This page explains what each is for and which ones are stable interfaces you can build on.

The machine-readable contract is the **JSON event stream** and the **webhook payloads**.
The rich and plain console output and the text log grammar are for humans: their wording and layout change without notice.
Do not parse them.

## Console formats

`advanced.log_format` chooses the console format (see [configuration.md](configuration.md#advanced)):

- `auto` (the default) resolves to `rich` on a terminal and to `plain` when stdout is piped or redirected, which is what `docker logs` sees.
- `rich` is the live console: the boot step lines, a block per title during the scan, and a live wait region, with color and progress bars.
- `plain` writes one line per event in the same grammar as the log file.
- `json` writes one JSON object per event to stdout, the [event catalog](#event-catalog) below.

The choice only affects stdout.
The log file is always written, and always in the text grammar, whatever the console shows.

## The log file

Every run appends to `Pearlarr.log` under the data directory's `logs/` folder (`pearlarr paths` prints the location. Per-OS defaults are in [deployment.md](deployment.md#the-data-directory)).

Each line is `<timestamp> LEVEL [scope path] message key=value ...`, one line per event. A rendered traceback is the only multi-line form.
The grammar is stable enough to read and grep, but it is **not a parsing API**: wording, scope paths, and `key=value` tails may change in any release.
Automation belongs on the JSON stream or the webhooks.

At the start of each run, scheduled ones included, the previous file rotates to a dated backup, `Pearlarr.log.<run's timestamp>`.
Backups older than `advanced.log_retention_days` are deleted once the run's configuration loads.

## Levels

`advanced.log_level` sets the minimum level written to the console, the file, and the JSON stream.

| Level | Meaning |
| --- | --- |
| `DEBUG` | Implementation tracing for maintainers. The redaction guarantee still applies |
| `INFO` | The facts and outcomes of the run |
| `WARNING` | Something went wrong but the run continues. The message names the fix |
| `ERROR` | The run, or one title, failed. The message names the fix |
| `CRITICAL` | Accepted as a threshold, but Pearlarr itself never logs above ERROR. Only a dependency's adopted log records carry it |

## The JSON event stream

With `advanced.log_format: json`, Pearlarr writes one JSON object per line to stdout, a machine-readable twin of the run's output, meant for log shippers and scripts.
The log file stays in the text grammar.

Every object carries the envelope keys, in stable order:

| Key | Meaning |
| --- | --- |
| `schema_version` | The version of these keys (see the stability policy below) |
| `time` | Local time with its UTC offset, ISO 8601, seconds precision |
| `event` | The event name. The catalog below enumerates them |
| `level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `message` | The human-readable line (informational, not an interface) |
| `component` | The subsystem it came from (`run`, `scan`, `title`, `cli`, ...). Every event has it. The text log shows it as the line's `[scope path]` when the event carries no `path` |

Per-event fields follow the envelope.
Three additions appear where they apply: a `diagnostic` carries `origin` (and sometimes `during` and `exc`), events inside a nested scope carry their scope path as `path`, and `arr_run_summary` carries the `needs_action_records` and `grabbed_records` arrays.
Absent facts are omitted, never `null`.

The stream honors `advanced.log_level`, with one carve-out: an `arr_run_summary` with `needs_action_records` or errors is admitted at `WARNING`, so a warnings-only stream still ends each arr run with its summary.

The command events at the end of the catalog (`paths_shown`, `starter_config_written`, `effective_config_shown`, and the `config_*` and `cache_*` events) appear only in a subcommand's own `--json` stream (`pearlarr cache stats --json`, `pearlarr config show --json`, ...), never in a run's stream. `advanced.log_format` does not affect them, only the per-command `--json` flag does.

### Stability policy

`schema_version` is currently `1`.
It only covers the keys every line carries, the ones in the table above, such as `time`, `event` and `level`.
Events and their own fields are not covered. Any release can add one, so ignore what you do not recognize.
Any release can also rename or remove one, or change what it means, and says so under "Upgrade notes" in the [CHANGELOG](../CHANGELOG.md).

### Replaying a capture

`pearlarr replay` turns a captured JSON stream back into the readable text grammar, for reading a docker-captured or archived log after the fact: `pearlarr replay events.jsonl`, or `docker logs pearlarr | pearlarr replay -` (`-` reads stdin).
Each envelope prints as one `<timestamp> LEVEL [scope path] message key=value` line, the same shape the log file uses.
Because the renderer knows only the envelope, not each event, lines from a newer Pearlarr still render best-effort, and non-JSON lines (docker interleaves stderr) are skipped with a count on stderr.

## Event catalog

<!-- gen:json-events - GENERATED by scripts/gen_docs.py from pearlarr/output/events.py + line_renderers.py. Do not edit between the markers. Regenerate: uv run python scripts/gen_docs.py -->
Specimens are generated by running one representative event of every kind
through the real JSON serializer, so serialized names, enum values, and
null-vs-absent behavior are the wire truth. The stream writes each object
on a single line. Specimens are pretty-printed here, with `time` normalized.

### `pearlarr_started`

The process banner: the installed version and the resolved data directory. The first event of every invocation.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "pearlarr_started",
  "level": "INFO",
  "message": "Pearlarr started",
  "component": "run",
  "version": "v1.0.0",
  "data_dir": "/home/user/.local/share/pearlarr"
}
```

### `scheduled_run_started`

Scheduled mode only: a new run begins (`number` counts from 1).

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "scheduled_run_started",
  "level": "INFO",
  "message": "scheduled run started",
  "component": "run",
  "number": 1
}
```

### `boot_step_finished`

One boot step finished. `outcome` is `ok`, `warned`, or `failed`, with an optional `detail`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "boot_step_finished",
  "level": "INFO",
  "message": "done",
  "component": "boot",
  "path": "Connecting to Sonarr",
  "outcome": "ok",
  "detail": "42 series",
  "elapsed_s": 1.2
}
```

### `boot_ready`

The boot finished. The scan begins.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "boot_ready",
  "level": "INFO",
  "message": "ready",
  "component": "boot",
  "elapsed_s": 3.5
}
```

### `scan_started`

The per-arr scan opened: which arr, and how many library items it will walk.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "scan_started",
  "level": "INFO",
  "message": "starting",
  "component": "scan",
  "arr": "sonarr",
  "total": 42
}
```

### `item_started`

The scan moved to the next library item (`index` of `total`, `name` its name).

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "item_started",
  "level": "INFO",
  "message": "item",
  "component": "scan",
  "arr": "sonarr",
  "index": 3,
  "total": 42,
  "name": "Sousou no Frieren"
}
```

### `diagnostic`

A problem or notice at any level. `origin` names the subsystem. A diagnostic raised outside any scope carries `during`, the scope that was open then, and one with a captured traceback carries it in `exc`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "diagnostic",
  "level": "WARNING",
  "message": "AniList rate limited - waiting 30s before retrying",
  "component": "anilist",
  "origin": "anilist",
  "during": "[3/42] Sousou no Frieren"
}
```

### `scope_opened`

A nested scope opened (`kind` + `serial` identify it). Events inside carry the scope path as `path` instead of the id.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "scope_opened",
  "level": "INFO",
  "message": "scope opened",
  "component": "scope",
  "kind": "title",
  "serial": 2,
  "label": "Sousou no Frieren"
}
```

### `title_header`

A title's block opened. `message` is the title's state word.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "title_header",
  "level": "INFO",
  "message": "checking",
  "component": "title",
  "path": "sonarr › [3/42] Sousou no Frieren › title",
  "title": "Sousou no Frieren",
  "al_id": 154587,
  "coverage": "S01",
  "link": "https://seadex.moe/154587/"
}
```

### `title_detail`

A labeled line inside a title's block. `message` is `label: value`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "title_detail",
  "level": "INFO",
  "message": "status: missing episodes",
  "component": "title",
  "path": "sonarr › [3/42] Sousou no Frieren › title",
  "note": "S01E27-E28"
}
```

### `title_row`

A one-line result for a title: `message` is the state word, `title` the title's name.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "title_row",
  "level": "INFO",
  "message": "unchanged",
  "component": "title",
  "path": "sonarr › [3/42] Sousou no Frieren › title",
  "title": "Sousou no Frieren"
}
```

### `release_skipped`

A release was skipped at the grab. `reason` is `private_only`, `unsupported_tracker`, `tracker_not_selected`, `misnumbered`, `episode_titles_contradict`, `placed_elsewhere`, or `placement_input_unread`. On `placed_elsewhere`, `another_titles` counts the release's files placed on another title's episodes and `unplaced` the rest, which nothing placed.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "release_skipped",
  "level": "WARNING",
  "message": "release skipped",
  "component": "title",
  "path": "sonarr › [3/42] Sousou no Frieren › title",
  "group": "Vodes",
  "tracker": "AnimeBytes",
  "reason": "private_only",
  "link": "https://seadex.moe/108/"
}
```

### `grab_failed`

Grabbing a release failed: qBittorrent did not take the torrent. The title is retried next run.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "grab_failed",
  "level": "WARNING",
  "message": "grab failed",
  "component": "title",
  "path": "sonarr › [3/42] Sousou no Frieren › title",
  "group": "SubsPlease",
  "link": "https://nyaa.si/view/1734567",
  "error": "qBittorrent rejected the torrent"
}
```

### `grab_action`

The grab decision for a title. `message` says which: grabbing, would grab (a dry run), or already in qBittorrent.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "grab_action",
  "level": "INFO",
  "message": "grabbing release",
  "component": "title",
  "path": "sonarr › [3/42] Sousou no Frieren › title",
  "waiting_to_import": true,
  "groups": "SubsPlease [best]",
  "grabbed": "[SubsPlease] Sousou no Frieren"
}
```

### `scope_closed`

The matching close of a `scope_opened` (anything nested deeper closes with it).

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "scope_closed",
  "level": "INFO",
  "message": "scope closed",
  "component": "scope",
  "kind": "title",
  "serial": 2
}
```

### `cap_reached`

The `advanced.max_torrents_to_add` cap was reached. The run keeps checking the remaining titles without grabbing, so a torrent already in qBittorrent is still tied to every title that lists it, and the held grabs are made on later runs.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "cap_reached",
  "level": "INFO",
  "message": "torrent cap reached",
  "component": "run",
  "cap": 25
}
```

### `scan_finished`

The per-arr scan closed (a boundary event, the summary carries the facts).

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "scan_finished",
  "level": "INFO",
  "message": "scan finished",
  "component": "scan",
  "arr": "sonarr"
}
```

### `arr_run_summary`

The arr run's summary: the tally counters (`held_by_cap` only when the run held titles past the cap), plus `needs_action_records` and `grabbed_records` arrays mirroring the summary's per-title lines.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "arr_run_summary",
  "level": "INFO",
  "message": "arr run complete",
  "component": "summary",
  "arr": "sonarr",
  "checked": 42,
  "needs_action": 1,
  "grabbed": 1,
  "imported": 1,
  "up_to_date": 31,
  "unchanged": 6,
  "no_mappings": 1,
  "no_seadex_entry": 2,
  "no_releases": 1,
  "warnings": 1,
  "errors": 0,
  "elapsed_s": 63.4,
  "tip": "private_only_no_fallback",
  "needs_action_records": [
    {
      "title": "Made in Abyss",
      "coverage": "S02",
      "group": "Vodes",
      "reason": "private-only release; no fallback covers these files",
      "cause": "private_only_no_fallback",
      "link": "https://seadex.moe/108/"
    }
  ],
  "grabbed_records": [
    {
      "title": "Sousou no Frieren",
      "coverage": "S01",
      "group": "SubsPlease",
      "torrent": "[SubsPlease] Sousou no Frieren",
      "link": "https://seadex.moe/154587/"
    }
  ]
}
```

### `wait_started`

A wait pass opened on `total` pending imports. `wait_pass` is `blocking` (waits for the downloads to finish, then imports) or `deferred` (one poll of the pending imports from earlier runs).

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "wait_started",
  "level": "INFO",
  "message": "waiting",
  "component": "wait",
  "path": "sonarr › wait",
  "wait_pass": "blocking",
  "total": 1
}
```

### `wait_row_finished`

The wait pass finished one row: a pending import got its outcome for this run. `message` is the outcome word. `hand_import_files` names the files left for a hand import, on the `unplaced` and `partial` outcomes. On `partial`, a file numbered by another TVDB order as an episode Sonarr doesn't list is named behind that number, as `DVD S07E01: name`, and when every file is, `detail` says which episodes are still unfilled and why the files are left for a hand import.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "wait_row_finished",
  "level": "INFO",
  "message": "imported",
  "component": "wait",
  "path": "sonarr › wait",
  "torrent": "[SubsPlease] Sousou no Frieren",
  "files": 28,
  "waited_s": 412.0
}
```

### `wait_finished`

The wait pass closed, with its `imported`, `in_progress`, `left` and `failed` counts.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "wait_finished",
  "level": "INFO",
  "message": "complete",
  "component": "wait",
  "path": "sonarr › wait",
  "wait_pass": "blocking",
  "imported": 1,
  "in_progress": 0,
  "left": 0,
  "failed": 0,
  "elapsed_s": 430.0
}
```

### `arr_run_finished`

The per-arr run closed (a boundary event).

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "arr_run_finished",
  "level": "INFO",
  "message": "arr run finished",
  "component": "run",
  "arr": "sonarr"
}
```

### `next_run_scheduled`

Scheduled mode only: when the next run fires.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "next_run_scheduled",
  "level": "INFO",
  "message": "next run scheduled",
  "component": "run",
  "at": "2026-01-02T00:00:00+00:00"
}
```

### `paths_shown`

The `paths` command: the resolved data directory and the files within it.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "paths_shown",
  "level": "INFO",
  "message": "resolved paths",
  "component": "cli",
  "data_dir": "/home/user/.local/share/pearlarr",
  "config": "/home/user/.local/share/pearlarr/config.yml",
  "cache": "/home/user/.local/share/pearlarr/cache.db",
  "mappings_db": "/home/user/.local/share/pearlarr/mappings.db",
  "log_dir": "/home/user/.local/share/pearlarr/logs"
}
```

### `starter_config_written`

`config init` wrote the template to `path`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "starter_config_written",
  "level": "INFO",
  "message": "starter config written",
  "component": "cli",
  "config_path": "/home/user/.local/share/pearlarr/config.yml"
}
```

### `config_validated`

`config validate` succeeded. `migrated` flags an in-memory schema migration (with `migration_notes`), and the `*_missing_keys` arrays and `qbit_configured` say what a run would use.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "config_validated",
  "level": "INFO",
  "message": "config valid",
  "component": "cli",
  "config_path": "/home/user/.local/share/pearlarr/config.yml",
  "migrated": false,
  "sonarr_missing_keys": [],
  "radarr_missing_keys": [
    "radarr.api_key"
  ],
  "qbit_configured": false
}
```

### `config_up_to_date`

`config migrate` had nothing to do: the file is already at the current schema.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "config_up_to_date",
  "level": "INFO",
  "message": "config up to date",
  "component": "cli",
  "config_path": "/home/user/.local/share/pearlarr/config.yml"
}
```

### `config_migrated`

`config migrate` rewrote the file at the current schema. `backup_path` is the saved previous file and `notes` lists what changed.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "config_migrated",
  "level": "INFO",
  "message": "config migrated",
  "component": "cli",
  "config_path": "/home/user/.local/share/pearlarr/config.yml",
  "backup_path": "/home/user/.local/share/pearlarr/config.yml.bak",
  "notes": [
    "replaced seadex.public_only with seadex.private_releases"
  ]
}
```

### `effective_config_shown`

`config show`: the effective config (defaults applied) as `config`, with secrets already redacted.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "effective_config_shown",
  "level": "INFO",
  "message": "effective config",
  "component": "cli",
  "config_path": "/home/user/.local/share/pearlarr/config.yml",
  "config": {
    "sonarr": {
      "url": "http://sonarr:8989",
      "api_key": "REDACTED"
    }
  }
}
```

### `cache_backed_up`

`cache backup` wrote a fresh snapshot to `backup_path`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "cache_backed_up",
  "level": "INFO",
  "message": "cache backed up",
  "component": "cli",
  "backup_path": "/home/user/.local/share/pearlarr/cache.backup.db"
}
```

### `cache_restored`

`cache restore` restored the database from `backup_path`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "cache_restored",
  "level": "INFO",
  "message": "cache restored",
  "component": "cli",
  "backup_path": "/home/user/.local/share/pearlarr/cache.backup.db"
}
```

### `cache_removed`

`cache remove` deleted the database at `path`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "cache_removed",
  "level": "INFO",
  "message": "cache removed",
  "component": "cli",
  "cache_path": "/home/user/.local/share/pearlarr/cache.db"
}
```

### `cache_stats_reported`

`cache stats`: per-block row counts and the database file's `size_bytes`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "cache_stats_reported",
  "level": "INFO",
  "message": "cache stats",
  "component": "cli",
  "entries": 128,
  "anilist_meta": 96,
  "sonarr_parse": 412,
  "pending_imports": 0,
  "guard_facts": 0,
  "tvdb_orders": 90,
  "size_bytes": 1179648
}
```

### `cache_integrity_reported`

`cache check` passed: the SQLite integrity-check `result`.

```json
{
  "schema_version": 1,
  "time": "2026-01-01T18:00:00+00:00",
  "event": "cache_integrity_reported",
  "level": "INFO",
  "message": "cache integrity",
  "component": "cli",
  "result": "ok"
}
```

### Events that never reach the JSON stream

- `BootStepStarted` - for the live console only. `boot_step_finished` carries the step's facts.
- `BootStepProgressed` - progress for the live console only.
- `BootStepSlow` - a notice for the console and the text log. The JSON stream sees the step finish.
- `WaitProgress` - the wait pass's progress at each poll. On the JSON stream, wait progress is the `wait_started` / `wait_row_finished` / `wait_finished` sequence.
<!-- /gen:json-events -->

## Webhooks

Both webhook URLs are secrets under the [redaction guarantee](../SECURITY.md#the-redaction-guarantee): they are never logged, and a delivery failure never fails the run.

### Discord grab notifications

With `notifications.discord_url` set, every title that adds at least one torrent posts one message, never on a dry run.
The payload is Discord's standard webhook format: username `Pearlarr`, one embed carrying the release, its tracker, size, and audio, the episodes covered, the group being replaced, and SeaDex's notes (the README shows one rendered).
When `notifications.wait_notify` is on, the wait pass posts a summary the same way when it finishes.

Delivery semantics: posts are paced at least one second apart. A `429` is retried once when Discord's `Retry-After` is five seconds or less, and dropped otherwise. Any other `4xx` disables Discord for the rest of the run (the URL is likely wrong). A `5xx` or a network error drops that one notification with a warning.

### The wait-summary webhook

With `notifications.wait_webhook_url` set, the end of each wait pass POSTs one JSON summary, a generic payload for ntfy, gotify, Home Assistant, and similar receivers.
(Setting the URL turns `notifications.wait_notify` on. An explicit `wait_notify: false` wins and silences it.)

```json
{
  "arr": "sonarr",
  "imported": 2,
  "left": 1,
  "failed": 0,
  "elapsed_s": 431.7,
  "rows": [
    {
      "label": "[SubsPlease] Sousou no Frieren",
      "outcome": "IMPORTED",
      "word": "imported"
    }
  ]
}
```

`rows` has one object per watched torrent. `outcome` is the row's outcome name and `word` its short display form. The names are `IMPORTED`, `GONE`, `DOWNLOAD_ERRORED`, `DOWNLOAD_TIMED_OUT`, `NO_CONTENT_PATH`, `STILL_IMPORTING`, `SONARR_BUSY`, `NOT_READY`, `NOTHING_PLACED`, `PARTIAL`, `ATTEMPT_FAILED`, `QBITTORRENT_UNREACHABLE`, `IMPORT_IN_PROGRESS`, `AWAITING_IMPORT` and `STILL_DOWNLOADING`.
Delivery semantics: one POST per wait pass (none when nothing was waited on), a ten-second timeout, no retry, and the response status is not checked: fire and forget.
The payload follows the same stability policy as the JSON event stream.
