# Releasing

Pearlarr releases are cut locally with [`scripts/release.sh`](scripts/release.sh). It runs
under your own GitHub identity, so branch protection's required checks fire normally - a bot
token would not trigger them. Versions follow [Semantic Versioning](https://semver.org), and
every user-observable change is recorded in [CHANGELOG.md](CHANGELOG.md).

## During normal work

Add each user-observable change to the `## [Unreleased]` section of the CHANGELOG in the same
PR that makes the change, under the usual headings (`Added`, `Changed`, `Fixed`, and so on).
The release script promotes whatever is there into the new version's notes.

## Cutting a release

First land your changes on `main`, each with its `## [Unreleased]` CHANGELOG entry.

If the demo behavior changed, re-record the README assets - otherwise the previous release's
assets are carried forward, so skip this:

```console
scripts/demo/record.sh                       # then: cp its GIF to docs/assets/demo_run.gif
uv run python scripts/sample_grab_post.py    # writes docs/assets/example_post.png
```

Then, from an up-to-date `main`, prepare the release PR:

```console
scripts/release.sh prepare 1.2.3
```

That bumps the version (`uv version`, which also re-locks `uv.lock`), regenerates the
version-pinned schema URLs (`scripts/gen_docs.py`), dates the CHANGELOG section, and opens a
PR. Review it, let the required checks pass, and merge.

Finally, pull the merged `main` and publish:

```console
git switch main && git pull
scripts/release.sh publish 1.2.3
```

That tags `v1.2.3` (which triggers the PyPI and GHCR publish workflows), creates the GitHub
release from the CHANGELOG section, and uploads the README assets to it.

## Assets

The README's screenshot and demo GIF are GitHub release assets, not tracked in the repo. They
are gitignored under `docs/assets/` and served from `releases/latest/download/<name>`, so the
`latest` URL always points at the newest release. `publish` attaches them to every release to
keep that URL valid; re-record only when the demo actually changes.
