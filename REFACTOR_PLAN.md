# SeaDexArr Refactor Plan

**Goal:** decompose the `SeaDexArr` god class (`seadexarr/modules/seadex_arr.py`)
into a slim orchestrator plus focused, independently-testable collaborators —
**without changing behaviour or the CLI surface**.

**Branch:** `public-only-release-filtering`
**Status:** Phase 0 ✅ done · Phase 1 ✅ done · Phase 2 ✅ done · Phases 3–6 pending
**Gates (every phase):** `pyrefly check` = 0 errors · `ruff check` = clean ·
`pytest` = all pass · package import smoke · (user-side) one `--dry-run` run.
Native unions, no `Optional`, no suppressions.

---

## 1. Verdict

`SeaDexArr` was a god class because it fused **~10 responsibilities** and threaded
**run-scoped mutable state through `self`**, and the subclasses used inheritance as
a shared-state bag (reaching into ~18 base attributes). The leaf modules
(`torrent`, `anilist`, `anibridge`, `discord`, `log`) are already clean seams, and
there is a real template-method spine (`run_sync` + 4 hooks) to build on. There
were **zero tests**, so the refactor is sequenced test-first.

Chosen approach: **extract collaborators (composition), keep thin Arr subclasses,
introduce an explicit `RunContext`.**

## 2. Diagnosis

| # | Responsibility | Representative methods | ~LOC |
|---|---|---|---|
| 1 | Config load/verify/normalize | `__init__` config block, `verify_config` | ~160 |
| 2 | Persistent cache (schema owner) | `setup_cache`, `save_cache`, `update_cache`, `get_cached_*`, `check_al_id_in_cache`, `save_json` | ~180 |
| 3 | ID-mapping resolution | `get_anime/anidb/anibridge_mappings`, `get_external_mappings`, `get_mappings_from_*`, `get_anilist_ids`, the module-global parse memo | ~520 |
| 4 | AniList client + meta cache | `get_anilist_title`, `prefetch_anilist`, `load/save_anilist_cache`, `_anilist_meta_is_fresh`, `al_cache` | ~200 |
| 5 | SeaDex fetch + shape | `get_seadex_entry`, `get_seadex_dict`, `filter_seadex_interactive` | ~140 |
| 6 | **Download-decision engine (the heart)** | `filter_seadex_downloads`, `filter_by_release_group`, `filter_by_torrent_hash`, `reduce_overlapping_downloads`, `get_any_to_download` + pure helpers | ~450 |
| 7 | qBittorrent adapter | `add_torrent`, `add_torrent_to_qbit`, `_is_preview` | ~200 |
| 8 | Coverage formatting (pure) | `format_episode_ranges/coverage`, `coverage_string`, `episodes_from_ep_list` | ~120 |
| 9 | Presentation + stats | 15 × `log_*`, `log_run_summary`, `_fresh_stats` | ~640 |
| 10 | Run orchestration | `run_sync`, `_al_id_prologue`, `_cached_entry_skip`, `_grab_and_cache`, hooks | ~300 |

**Cross-cutting smell:** per-run mutable state on `self` — `current_title/url/coverage`,
`public_only_skipped/groups`, `torrents_added`, `stats`, `_ep_list_cache`,
`_log_counts_at_start`. The decision engine *mutates* `self.public_only_skipped` and
*logs* via `self.log_fmt`; `add_torrent` mutates `stats`, reads `current_*`, enforces
`max_torrents_to_add`, and logs. That tangle is why nothing was independently testable.

**Subclasses:** Sonarr (~981 lines, episode-heavy) and Radarr (~279 lines, lean)
implement 4 abstract hooks (`_get_all_items`, `_filter_to_single_item`,
`_item_anilist_ids`, `_process_al_id`) and reach into ~18 base attributes.

## 3. Decisions

1. **Composition, not mixins.** Mixins would spread the surface across files while
   keeping the coupling.
2. **Introduce `RunContext`** — a per-run state object. Collaborators stop mutating
   shared `self`; they read/return data. This is what unlocks testing.
3. **The decision engine returns a `PlanResult`**, not mutate `self` and log.
4. **Keep `SeaDexSonarr` / `SeaDexRadarr` class names + `(config, cache, logger)`
   constructors + `.run()/.close()`.** Preserves the entire CLI/entry surface at
   zero churn; subclasses shrink to thin hook-adapters over an Arr REST client.
5. **Inheritance → strategy is optional/last.** Once the base is a slim orchestrator,
   two thin subclasses implementing 4 hooks is idiomatic; strategy buys little for
   real entry-point churn.
6. **Preserve two intentional optimizations exactly:** the module-global mapping
   parse memo (`_PARSED_MAPPING_CACHE`, shared across both arr instances) and
   **per-arr-instance cache ownership** (Radarr runs + saves `cache.json`, then
   Sonarr is constructed and re-reads it — they hand off through the *file*, not
   shared memory). Do **not** share an in-memory `CacheStore` across arrs.

## 4. Target module layout

New files live beside the leaf utilities under `seadexarr/modules/` (no directory
reshuffle, so `from .x import Y` patterns are unchanged).

| Module | Class / contents | Status |
|---|---|---|
| `coverage.py` | episode-range / coverage formatting (pure functions) | ✅ created (Phase 1) |
| `planner.py` | release-matching helpers now; `DownloadPlanner` + `PlanResult` in Phase 4 | ✅ created (Phase 1, helpers only) |
| `config.py` | `AppConfig` — typed settings + YAML load + template-sync | ✅ created (Phase 2) |
| `cache.py` | `CacheStore` — owns the cache schema | ✅ created (Phase 2) |
| `mappings.py` | `MappingResolver` — 3 sources → AniList ids; owns the parse memo | Phase 3 |
| `anilist_gateway.py` | `AniListGateway` — `al_cache` + meta TTL + prefetch + titles/thumbs | Phase 3 |
| `seadex_gateway.py` | `SeaDexGateway` — fetch entry → normalized `seadex_dict` | Phase 3 |
| `torrents.py` | `TorrentService` — qBittorrent adapter | Phase 3 |
| `notify.py` | `Notifier` — Discord fields + push | Phase 3 |
| `reporter.py` | `RunReporter` (all `log_*` + stats) + `RunContext` | Phase 4 |
| `sync.py` | `SeaDexSync(ABC)` — `run_sync` template + the 3 shared helpers | Phase 5 |
| `sonarr.py` / `radarr.py` | `SonarrClient`/`RadarrClient` + thin `SeaDex*` adapter | Phase 5 |

Net target: base class **3213 → ~300**, Sonarr **981 → ~280**, Radarr **279 → ~130**.

## 5. Dependency graph

The orchestrator is the only thing that knows everything; collaborators mostly
depend on `AppConfig` plus one peer.

```
                         AppConfig  (leaf)
                            │
        ┌──────────┬────────┼──────────┬───────────┬──────────┐
   CacheStore  MappingRes  SeaDexGw  TorrentSvc  Notifier   RunReporter
        │          │          │          │           │       (+ RunContext)
   AniListGw ──────┘          │       torrent.py  discord.py    log.py
        │                  DownloadPlanner ── coverage.py
        │                     │
        └─────────────────────┴──────────── SeaDexSync (orchestrator)
                                                 ▲
                                    ┌────────────┴────────────┐
                              SeaDexSonarr               SeaDexRadarr
                              (SonarrClient)             (RadarrClient)
```

## 6. Deep design — `RunContext` and `PlanResult`

These two value types are the linchpin of the decoupling. They replace "mutate
`self` and log from deep in the call stack" with "return data the orchestrator
applies." Shapes below are the **target** for Phases 4; the Phase 0 tests will be
adapted to assert on these instead of on mutated `self` + return tuples.

### 6.1 `RunContext` (replaces the run-scoped `self.*` fields)

Created fresh at the top of `run_sync` (replacing `reset_run_stats`), threaded
explicitly into the collaborators that need it (`TorrentService.add(ctx, ...)`,
`RunReporter.*(ctx, ...)`).

```python
@dataclass
class RunStats:
    checked: int = 0
    added: list[GrabRecord] = field(default_factory=list)
    up_to_date: int = 0
    cached: int = 0
    no_seadex_entry: int = 0
    no_releases: int = 0
    no_mappings: int = 0
    needs_action: list[NeedsActionRecord] = field(default_factory=list)
    unmonitored: int = 0

@dataclass
class RunContext:
    arr: str                              # "sonarr" | "radarr" (was threaded as a param)
    dry_run: bool
    stats: RunStats = field(default_factory=RunStats)
    torrents_added: int = 0
    # current entry being processed (for attribution in grabs/summary)
    current_title: str | None = None
    current_url: str | None = None
    current_coverage: str | None = None
    # run clock + logger-counter snapshot for the end-of-run summary
    started_monotonic: float | None = None
    log_counts_at_start: dict[int, int] = field(default_factory=dict)
```

Notes:
- `GrabRecord` / `NeedsActionRecord` are small dataclasses replacing the ad-hoc
  `{"title","coverage","url","name","group"}` / `{"title","reason",...}` dicts.
- `public_only_skipped` / `public_only_groups` are **per-title transients**, so they
  leave `self` entirely and live on `PlanResult` (§6.2); the orchestrator reads them
  per id and appends a `NeedsActionRecord` when nothing was grabbed.
- `_ep_list_cache` is Sonarr-specific scratch and belongs to the Sonarr adapter, not
  the generic `RunContext`.

### 6.2 `PlanResult` (the decision engine's output)

`DownloadPlanner.plan(...)` becomes near-pure: it consumes the shaped `seadex_dict`,
the Arr's current release info, optional episode list, and the cached hashes, and
returns a `PlanResult`. It no longer touches `self.log_fmt`, `self.public_only_skipped`,
or `self.stats`.

```python
@dataclass
class SkipNotice:
    groups: list[str]               # release-group name(s) skipped
    reason: str                     # e.g. "private-only (public_only on)"
    level: int = logging.WARNING

@dataclass
class PlanResult:
    seadex_dict: dict               # annotated in place with per-url download flags
    torrent_hashes: list[str]       # unique hashes to remember in the cache record
    public_only_skipped: bool       # a release was skipped purely for being private
    public_only_groups: list[str]   # which groups (for the run summary)
    skip_notices: list[SkipNotice]  # what to log — orchestrator hands these to RunReporter
```

Orchestrator usage (replacing the in-engine logging/mutation):
```python
result = planner.plan(seadex_dict, arr_release_dict=rd, ep_list=eps,
                      cached_hashes=cache.torrent_hashes(arr, al_id))
for notice in result.skip_notices:
    reporter.log_skip(notice)
# ... add torrents from result.seadex_dict, then per-title public_only handling
```

`public_only` / `interactive` / `use_torrent_hash_to_filter` come from `AppConfig`
injected into the planner; `reduce_overlapping_downloads`'s private-only skip becomes
a `SkipNotice` appended to the result instead of an inline `log_fmt.detail` call.

## 7. Migration phases

Each phase is independently shippable and ends at the gates in the header.

- **Phase 0 — Safety net.** ✅ pytest config + characterization tests pinning current
  behaviour: pure helpers, coverage formatting, `get_seadex_dict`, decision engine.
  50 tests under `tests/`.
- **Phase 1 — Pure leaves.** ✅ relocated coverage formatters → `coverage.py` and
  release-matching helpers → `planner.py`; `SeaDexArr` keeps thin delegating methods
  and imports the helpers from their new homes. Behaviour-preserving.
- **Phase 2 — Value/config.** `AppConfig`, then `CacheStore` (preserve per-instance
  ownership + in-memory-mutate / save-at-checkpoint + preview gating).
- **Phase 3 — I/O gateways.** `MappingResolver` (preserve module memo; return
  dropped-ignored-ids instead of logging) → `AniListGateway` (drop the
  `get_anilist_title` → `current_title` side-effect) → `SeaDexGateway` →
  `TorrentService` (return results+notices, no stats/`current_*` mutation) → `Notifier`.
- **Phase 4 — Core + presentation.** `DownloadPlanner` returning `PlanResult`; then
  `RunReporter` + `RunContext`. Untie the `add_torrent`/`_grab_and_cache` knot behind
  explicit return values. Adapt the Phase 0 tests to assert on `PlanResult`.
- **Phase 5 — Slim subclasses.** Extract `SonarrClient`/`RadarrClient`; reduce
  `SeaDexSonarr`/`SeaDexRadarr` to the 4 hooks. Fix `ignore_movies_in_radarr` to
  depend on a `RadarrClient`/movie-id set rather than nesting a `SeaDexRadarr`.
  **Collapse the config to a single representation (decided 2026-06-22).** Today
  there are three: the raw `data` dict, `AppConfig`'s per-access `@property` views,
  and 16 flat *mirror* attributes copied in `SeaDexArr.__init__`
  (`self.public_only = self._config.public_only`, …) — verified pure mirrors (never
  reassigned at runtime; config never written post-load), i.e. transitional
  scaffolding like the `self.config`/`self.cache` aliases. Target: `AppConfig`
  normalizes **once** into real typed fields (in `__post_init__` so the in-memory
  `_cfg(**data)` tests still construct without a file; consider `frozen=True` to
  enforce the post-load immutability — `load` must finish the template-sync before
  constructing), then delete the 16 mirrors **and** the raw `self.config` alias and
  read `self._config.X` everywhere (~40+ sites across `seadex_arr.py` + the
  subclasses — folded in here because slimming the adapters rewrites those sites
  anyway). Keep `data` only until the arr-specific keys (`sonarr_url`, profiles, …)
  also get typed accessors.
- **Phase 6 — Optional polish.** inheritance → strategy; inject one shared
  `MappingResolver` from the CLI; optional `core/` subpackage.

## 8. Behaviour-preservation invariants (landmines)

- **No tests existed** → Phase 0 is the net; do not skip it.
- **`get_anilist_title` sets `current_title`** as a side-effect — easy to lose when
  extracting AniList.
- **Per-instance cache file handoff** in `run_scheduled` — don't share in-memory cache
  across arrs.
- **`add_torrent` / `_grab_and_cache`** are the densest coupling — extract last, behind
  `RunContext` + `PlanResult`.
- **The `arr` string** ("sonarr"/"radarr") becomes a property of the adapter/`RunContext`,
  not a passed-around param.
- **Module-global mapping memo** (`_PARSED_MAPPING_CACHE`) is a cross-instance
  optimization — don't regress it.
- **The config-setting mirrors double as a cache** — `trackers` and
  `ignore_anilist_ids` rebuild a `set` on every property access; the 16 `self.*`
  mirrors hide that today by reading each once into a field. When Phase 5 removes the
  mirrors and reads through `self._config`, make those normalized values parse-once
  (typed fields / `cached_property`) or the per-access rebuild reappears inside the
  hot loops in `get_seadex_dict` / `filter_*` (lines ~598/891/1620).

## 9. Public surface that must not change

- `SeaDexSonarr(config, cache, logger).run(tvdb_id=..., dry_run=...) / .close()`
- `SeaDexRadarr(config, cache, logger).run(tmdb_id=..., dry_run=...) / .close()`
- Re-exports from `seadexarr/__init__.py` and `seadexarr/modules/__init__.py`
  (`SeaDexSonarr`, `SeaDexRadarr`, `seadexarr_cli`, `setup_logger`)
- `cli.py` flow (`run_scheduled`, `run_single`) — instantiates the two classes directly.

## 10. Progress log

- **2026-06-22 — Phase 0 done.** Added `[tool.pytest.ini_options]` and `tests/`:
  `builders.py` (bare-instance factory + dict/fake builders), `test_release_matching.py`,
  `test_coverage.py`, `test_seadex_dict.py`, `test_download_planner.py`. 50 tests, all
  gates green.
- **2026-06-22 — Phase 1 done.** Created `coverage.py` and `planner.py` (verbatim
  relocation of the pure helpers). `seadex_arr.py` imports them (`from . import coverage
  as _coverage`, `from .planner import ...`); the three coverage methods became thin
  delegators; `seadex_sonarr.py` imports `get_episode_keys` from `.planner`. Tests
  repointed to the new modules. 50 tests, all gates green.
- **2026-06-22 — Phase 2 done.** Extracted `config.py` (`AppConfig`: file lifecycle —
  copy-template / parse / key-order sync — plus `checksum()` and typed, normalized
  settings; now owns `PUBLIC_TRACKERS` / `PRIVATE_TRACKERS`) and `cache.py`
  (`CacheStore`: schema + version/checksum reconcile + freshness check + records +
  preview-gated `save`; now owns `UPDATED_AT_STR_FORMAT` and `save_json`).
  `SeaDexArr.__init__` builds `self._config = AppConfig.load(...)` and
  `self.cache_store = CacheStore.load(...)`, sourcing every scalar setting from the
  typed props. **Two transitional aliases kept for zero-churn / behaviour-parity:**
  `self.config` stays bound to the raw dict (subclasses still read `self.config.get`)
  and `self.cache` stays bound to `cache_store.data` (the not-yet-extracted direct
  reads — `anilist_meta`, `anilist_entries`, `sonarr_parse_cache` — are untouched;
  they migrate in Phases 3–5). The cache read/write methods became thin delegators.
  `seadex_sonarr` / `tests/builders` now import the relocated constants from their new
  homes. Typing `qbit_info` as `dict | None` surfaced a latent None-unsafety; guarded
  with an inlined `qbit_info is not None and ...` (behaviour-identical for every real
  config — the None path is unreachable after template-sync). `seadex_arr.py`
  3020 → 2776. Added `tests/test_config.py` + `tests/test_cache.py` (22 tests).
  72 tests, all gates green.
