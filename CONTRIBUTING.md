# Contributing

## Setup

Pearlarr is developed with [uv](https://docs.astral.sh/uv/). Python 3.13 is the
floor; development and the Docker image run 3.14, and CI
(`.github/workflows/build.yaml`) tests both.

```console
uv sync --group dev
uv run pre-commit install
```

This creates `.venv` with the runtime dependencies plus every tool below, pinned
by `uv.lock`. The lockfile is tracked: commit lockfile churn together with the
change that caused it.

The acceptance test for this document is the ten-minute rule: a fresh clone
following it verbatim reaches a green quality gate within ten minutes.

## Quality gate

Every change must pass the full chain, in order, with zero findings:

```console
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pyright
uv run pyrefly check
uv run pytest -q
```

CI (`.github/workflows/build.yaml`) runs the same gates on Python 3.13 and 3.14,
plus a total-coverage floor and a changed-line coverage gate (diff-cover, 90%).
Docs get their own CI job: codespell (`[tool.codespell]` in `pyproject.toml`),
markdownlint (`.markdownlint-cli2.yaml` - generated files included on purpose),
and offline lychee for local links and anchors (`lychee.toml`); external URLs
are checked weekly by `links.yaml`. Dependabot watches the GitHub Actions and
pip ecosystems weekly. Run one test file with
`uv run pytest tests/test_planner.py -q`.

### Suppressions

Production code carries zero suppressions: no `# type: ignore`, no `# noqa`, no
inline pyright directives — a PR adding one will be asked to remove it and fix
the underlying type instead. Tests are held to the same bar: every test module
opens with a `# pyright: strict` header. The one sanctioned escape hatch is a
module-level `# pyright: reportPrivateUsage=false` in a white-box test that
legitimately reads a collaborator's private members, and it must carry a
justification comment.

## Tests

- Tests are strict-typed and use recording fakes (`tests/fakes.py`) rather than
  `MagicMock`: a fake records the calls it received and returns typed values, so
  a drifted signature is a type error, not a silently green test.
- Shared object builders live in `tests/builders.py`; prefer extending an
  existing fake or builder over an ad-hoc double.

### Verifying against live services

The recording fakes are the supported development loop. When you need a live
check, point a config at real Sonarr/Radarr instances and leave the
`qbittorrent` credentials blank: that is preview mode — every decision is made
and reported, nothing is grabbed, nothing is cached — and it is the safe live
harness. Never paste real API keys, hostnames, or webhook URLs into issues,
docs, code, or screenshots; placeholders only.

## Running locally

```console
uv run pearlarr config init          # write the starter config.yml
uv run pearlarr paths                # show where config/caches/logs resolve
uv run pearlarr run single --sonarr  # one Sonarr pass
```

Everything lives in one data directory (`pearlarr paths` prints it).

## Documentation

One authored home per fact: enumerable facts (config keys, defaults, allowed
values, env vars) are authored once, in code, and generated into every other
surface. Never edit a generated file or island — edit the source and run:

```console
uv run python scripts/gen_docs.py
```

Generated artifacts (each carries a `GENERATED` banner; CI and pre-commit fail
on drift):

| Artifact | Source |
| --- | --- |
| `pearlarr/config_sample.yml` | attribute docstrings in `pearlarr/config.py` |
| `schemas/config.schema.json` | the same config models |
| `docs/configuration.md` (islands between `gen:` markers) | config models + `pearlarr/env_registry.py` |
| `CONTRIBUTING.md` (the env-var island below) | `pearlarr/env_registry.py` |
| `docs/cli.md` | the typer app in `pearlarr/cli.py` |
| `docs/output.md` (the event-catalog island) | the output events run through the real JSON serializer |

### Drift map

Change X, also do Y — most rows are enforced mechanically, listed anyway so you
can predict CI:

| You changed | Also do |
| --- | --- |
| A config field, its docstring, default, or constraint | `uv run python scripts/gen_docs.py` |
| An enum a config field references | Docstring every member, then regenerate |
| An environment variable | Register it in `pearlarr/env_registry.py`, then regenerate |
| A CLI command, option, or help string | Regenerate (`docs/cli.md` is generated) |
| An output event type or its JSON fields | Regenerate; a new event also needs a specimen + description in `scripts/gen_docs.py`, and an additive-only check against `docs/output.md`'s stability policy |
| A config key or cache schema (breaking) | An "Upgrade notes" entry in `CHANGELOG.md` |
| A user-visible behavior | A `CHANGELOG.md` entry |
| A user-facing message | Update any docs that quote it (the `docs/troubleshooting.md` anchors fail the suite mechanically) |
| Install or quickstart steps in the README | Re-run them verbatim, end to end |

### Writing rules

- Docstrings are Google style. Types never appear in docstrings (the checkers
  own types); no Sphinx roles; identifiers in single backticks.
- Modules, packages, and classes always carry a docstring. Functions and
  methods are documented unless the full contract is legible from name +
  signature; when in doubt, one line.
- A documented field gets a per-field attribute docstring on the field itself —
  never an `Attributes:` section, a field-enumerating paragraph in the class
  docstring, or a `#` comment carrying the field's contract. The class
  docstring keeps only class-level posture (what the class is, how it reads or
  fails as a whole).
- A docstring states the current contract. How the code came to be this way —
  ports, replaced designs, fixed bugs — belongs in the changelog or the
  architecture design notes, not in code.
- Config-model attribute docstrings are a compiled dialect — plain text plus
  single backticks only — because they render into YAML comments, markdown
  tables, and JSON Schema descriptions. State meaning, interactions, and
  blank/`None` semantics; never defaults, types, or allowed-value lists (the
  generator injects those).
- Comments state a constraint the code cannot show, in 1–2 lines. Change
  provenance (PR numbers, phase tags, dates) never appears in code — that is
  git's job. A workaround for an external defect names the defect and the
  version observed against, so it can be retested and removed.
- `# Invariant:` is a reserved comment prefix for load-bearing invariants at
  their enforcement sites.
- WARNING/ERROR messages read `what - cause - next action`, name the offending
  value and the exact config key/flag/command, sentence case, ASCII `" - "`
  connector, no markdown in terminal strings.
- Docs ride the same PR as the change they describe, never a follow-up.

## Task playbooks

### Add a config key, end to end

1. Add the field to its group model in `pearlarr/config.py`, with an
   attribute docstring in the compiled dialect. Constrain it in the type
   (`Literal`, an enum, `ge=`) so a bad value fails at load; new enum members
   get their own docstrings.
2. `uv run python scripts/gen_docs.py` — the sample, schema, and reference
   tables update themselves.
3. Read the key where the behavior lives; add tests.
4. If the key interacts with others, extend the group's prose island in
   `docs/configuration.md` (outside the `gen:` markers).
5. Add a `CHANGELOG.md` entry (plus "Upgrade notes" if breaking).

### Add a tracker

1. Add the display name to the right `*_TRACKER_NAMES` tuple in
   `pearlarr/config.py` (a parity test pins the set against the seadex
   library's `Tracker` enum).
2. If its releases carry a parseable download page, add a parser to the table
   in `pearlarr/torrents.py` (`PARSEABLE_TRACKERS` derives from it) —
   with tests; trackers without a parser are skipped with a warning.
3. `uv run python scripts/gen_docs.py`.

## Commits

`type(scope): imperative summary` at ≤72 characters (`feat(planner): …`,
`fix(cache): …`); the body says why. Summaries must be meaningful without
private context — no session or chat references.

## Releasing

Releases are cut locally with [`scripts/release.sh`](scripts/release.sh): it
runs under your own GitHub identity, so branch protection's required checks
fire normally - a bot token would not trigger them. Versions follow
[Semantic Versioning](https://semver.org); every user-observable change lands
with a `CHANGELOG.md` entry under `## [Unreleased]` (see the drift map above),
and the release script promotes that section into the new version's notes.

1. Land everything the release should carry on `main`; verify "Upgrade notes"
   covers every config/cache change.
2. From an up-to-date `main`, `scripts/release.sh prepare X.Y.Z` bumps the
   version (`uv version`, which also re-locks `uv.lock`), regenerates the
   version-pinned schema URLs (`scripts/gen_docs.py`), dates the CHANGELOG
   section, and opens the release PR. Review it, let the required checks
   pass, and merge.
3. Pull the merged `main`, then `scripts/release.sh publish X.Y.Z` re-records
   the README assets at the release version, tags `vX.Y.Z` (which triggers
   the PyPI and GHCR publish workflows), builds the GitHub release from the
   CHANGELOG section as a draft, attaches the assets, and publishes it -
   drafted so `releases/latest` never points at a release without its
   assets. A publish interrupted partway is safe to rerun.
4. Verify the PyPI package and `ghcr.io` image land; smoke the Docker quick
   start from the README on a clean host, verbatim.

The README's screenshot and demo GIF are GitHub release assets, not tracked
files: gitignored under `docs/assets/` and served from
`releases/latest/download/<name>`, so the README always shows the latest
release's images and the repository carries no binaries. Both bake the
installed version into their pixels (the GIF's boot title, the embed's
footer), which is why every publish re-records them; the recorders need
`vhs`, `ffmpeg`, playwright's chromium (`uv run playwright install
chromium`), and network. If a re-record can't run, publish falls back to the
previous release's copy with a warning - the image stays alive but its baked
version lags. To iterate on the demo or the embed layout by hand:

```console
scripts/demo/record.sh                       # writes scripts/demo/demo_run.gif
uv run python scripts/sample_grab_post.py    # writes docs/assets/example_post.png
```

Publishing is tokenless: `publish_pypi.yaml` authenticates to PyPI via Trusted
Publishing (OIDC, the `pypi` environment), and the Docker workflow pushes to
GHCR with the workflow's own identity - no long-lived secrets to rotate.

## Environment variables

<!-- gen:env-vars - GENERATED by scripts/gen_docs.py from pearlarr/env_registry.py; do not edit between the markers; regenerate: uv run python scripts/gen_docs.py -->
| Variable | Read by | Meaning |
| --- | --- | --- |
| `PEARLARR_DATA_DIR` | application | Override the data directory; the global `--data-dir` flag wins over it. |
| `PEARLARR_<GROUP>__<KEY>` | application | Override any config key by its double-underscore path; the value is parsed as YAML. See docs/configuration.md. |
| `PEARLARR_CRON` | Docker entrypoint | Cron schedule for the container's recurring runs. |
| `PEARLARR_RUN_ON_START` | Docker entrypoint | Whether the container starts with a catch-up run, before the cron cadence takes over. |
<!-- /gen:env-vars -->

Names use the `PEARLARR_` prefix with `__` as the nesting delimiter: the
config-override layer reads `PEARLARR_SONARR__URL` into `sonarr.url` (see
[docs/configuration.md](docs/configuration.md#overriding-config-keys)), so new
operational names must stay unambiguous under that split - a delimiter-less
`PEARLARR_*` name is reserved for them and never read as config. Register new
variables in `pearlarr/env_registry.py`.

## License

Contributions are accepted under the repository license (GPL-3.0-or-later),
which covers the documentation as well as the code.
