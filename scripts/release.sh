#!/usr/bin/env bash
# Cut a Pearlarr release. Runs under your own identity (so branch protection's
# required checks fire normally) - CONTRIBUTING.md "Releasing" is the full runbook.
#
#   scripts/release.sh prepare X.Y.Z   # bump + re-lock + regen docs + roll the
#                                        CHANGELOG, then open the release PR to review
#   scripts/release.sh publish X.Y.Z   # after the PR merges: re-record the README
#                                        assets, tag vX.Y.Z, create the GitHub release
#
# X.Y.Z is the bare version (no leading v). Both publish workflows fire on the tag.
set -euo pipefail

die() { echo "release: $*" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel)"

# The CHANGELOG body between "## [<name>]" and the next "## [" (name is
# "Unreleased" or a bare version; the version header also carries " - <date>").
changelog_section() {
  awk -v s="$1" '
    $0 ~ "^## \\[" s "\\]" { inside = 1; next }
    inside && /^## \[/ { exit }
    inside { print }
  ' CHANGELOG.md
}

# Promote "## [Unreleased]" to a dated "## [X.Y.Z]" section, leaving Unreleased empty.
roll_changelog() {
  awk -v v="$1" -v d="$2" '
    !done && $0 == "## [Unreleased]" {
      print; print ""; print "## [" v "] - " d; done = 1; next
    }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
}

# Fallback when a re-record can't run: keep/fetch the previous release's copy so
# the release and the assets branch still get media; its baked version lags until
# a real re-record.
carry_forward() {
  echo "release: WARNING - $2; docs/assets/$1 will show the previous release's baked version" >&2
  [ -f "docs/assets/$1" ] && return 0
  gh release download --pattern "$1" --dir docs/assets --clobber \
    || die "no docs/assets/$1 and none on the latest release; record one first (see CONTRIBUTING.md \"Releasing\")"
}

# The README assets bake the installed version into their pixels (the GIF's boot
# title, the embed's footer), so publish re-records both at the release version.
regen_assets() {
  mkdir -p docs/assets   # gitignored + untracked, so absent on a fresh clone
  uv sync --group dev --quiet   # a stale venv would bake the previous version
  if scripts/demo/record.sh; then
    cp scripts/demo/demo_run.gif docs/assets/demo_run.gif
  else
    carry_forward demo_run.gif "demo re-record failed (vhs + ffmpeg + network?)"
  fi
  if ! uv run python scripts/sample_grab_post.py; then
    carry_forward example_post.png "screenshot capture failed (playwright chromium + network?)"
  fi
}

# The README hot-links the media from the orphan `assets` branch: GitHub's
# release-asset CDN forces application/octet-stream, which PyPI's image proxy
# refuses to render, while raw.githubusercontent.com serves real image types.
# Built with plumbing (no checkout) and force-pushed, so git keeps one copy.
push_assets_branch() {
  local tag=$1 blob_gif blob_png blob_readme tree commit
  blob_gif=$(git hash-object -w docs/assets/demo_run.gif)
  blob_png=$(git hash-object -w docs/assets/example_post.png)
  blob_readme=$(git hash-object -w --stdin <<'EOF'
# Pearlarr README media

Binary assets the main README hot-links via `raw.githubusercontent.com`
(GitHub's release-asset CDN forces `application/octet-stream`, which PyPI's
image proxy refuses to render). Force-pushed as a fresh orphan commit by
`scripts/release.sh publish` on every release - do not edit by hand.
EOF
  )
  tree=$(printf '100644 blob %s\tREADME.md\n100644 blob %s\tdemo_run.gif\n100644 blob %s\texample_post.png\n' \
    "$blob_readme" "$blob_gif" "$blob_png" | git mktree)
  commit=$(git commit-tree "$tree" -m "chore(release): README media for $tag")
  git push --force origin "${commit}:refs/heads/assets"
}

cmd_prepare() {
  local version=$1 branch="release-$1" date
  [ "$(git branch --show-current)" = "main" ] || die "run prepare from main"
  [ -z "$(git status --porcelain)" ] || die "working tree is dirty; commit or stash first"
  git fetch --quiet origin main
  [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "main is behind origin/main; pull first"
  git rev-parse -q --verify "refs/tags/v$version" >/dev/null 2>&1 && die "tag v$version already exists"
  changelog_section Unreleased | grep '[^[:space:]]' >/dev/null \
    || die "CHANGELOG '## [Unreleased]' is empty; add release notes there first"

  date=$(date +%Y-%m-%d)
  git switch -c "$branch"
  uv version "$version"               # bumps pyproject.toml and re-locks uv.lock
  uv run python scripts/gen_docs.py   # regenerates the version-pinned schema URLs (G8)
  roll_changelog "$version" "$date"
  git add -A
  git commit -m "chore(release): $version"
  git push -u origin "$branch"
  gh pr create --base main --head "$branch" --title "chore(release): $version" \
    --body "Release $version. Bumps the version, re-locks, regenerates the schema URLs, and rolls the CHANGELOG. After this merges, run \`scripts/release.sh publish $version\` to tag and publish."
  echo
  echo "Release PR opened. Review it, let the required checks pass, then merge (squash or rebase)."
  echo "Then run:  scripts/release.sh publish $version"
}

cmd_publish() {
  local version=$1 tag="v$1" notes draft
  [ "$(git branch --show-current)" = "main" ] || die "switch to main first"
  git fetch --quiet origin main
  [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "main is not level with origin/main; pull first"
  [ "$(uv version --short)" = "$version" ] \
    || die "pyproject version is $(uv version --short), not $version; is the release PR merged and pulled?"
  changelog_section "$version" | grep '[^[:space:]]' >/dev/null || die "CHANGELOG has no '## [$version]' section"
  draft=$(gh release view "$tag" --json isDraft --jq .isDraft 2>/dev/null || echo absent)
  [ "$draft" = "false" ] && die "release $tag is already published"

  regen_assets
  notes=$(mktemp)
  changelog_section "$version" | sed '/./,$!d' > "$notes"   # drop leading blank lines

  # Tag, then assemble the release as a draft and publish only once the assets are
  # attached, so `releases/latest` never points at a release without them. Each
  # step tolerates its own leftovers, so an interrupted publish is safe to rerun.
  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1; then
    echo "release: tag $tag already exists; resuming"
  else
    git tag -a "$tag" -m "Pearlarr $version"
  fi
  git push origin "refs/tags/$tag"
  if [ "$draft" = "absent" ]; then
    gh release create "$tag" --draft --title "Pearlarr $version" --notes-file "$notes"
  fi
  gh release upload "$tag" docs/assets/demo_run.gif docs/assets/example_post.png --clobber
  gh release edit "$tag" --draft=false
  push_assets_branch "$tag"
  rm -f "$notes"
  echo "Published $tag: PyPI + GHCR workflows are running; GitHub release created; README media refreshed on the assets branch."
}

main() {
  [ $# -eq 2 ] || die "usage: scripts/release.sh {prepare|publish} X.Y.Z"
  local cmd=$1 version=$2
  [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must be bare semver X.Y.Z (no leading v)"
  case "$cmd" in
    prepare) cmd_prepare "$version" ;;
    publish) cmd_publish "$version" ;;
    *) die "usage: scripts/release.sh {prepare|publish} X.Y.Z" ;;
  esac
}

main "$@"
