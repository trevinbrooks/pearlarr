#!/usr/bin/env bash
# CHANGELOG.md surgery shared by the release workflows.
#
#   scripts/changelog.sh section <name>       # print the body of "## [<name>]"
#   scripts/changelog.sh roll <version> <date># promote Unreleased to a dated section
#
# <name> is "Unreleased" or a bare version. The version header carries " - <date>".
set -euo pipefail

die() { echo "changelog: $*" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel)"

# The body between "## [<name>]" and the next "## [", leading blank lines dropped.
cmd_section() {
  awk -v s="$1" '
    $0 ~ "^## \\[" s "\\]" { inside = 1; next }
    inside && /^## \[/ { exit }
    inside { print }
  ' CHANGELOG.md | sed '/./,$!d'
}

# Insert a dated "## [<version>]" header under "## [Unreleased]", leaving it empty.
cmd_roll() {
  awk -v v="$1" -v d="$2" '
    !done && $0 == "## [Unreleased]" {
      print; print ""; print "## [" v "] - " d; done = 1; next
    }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
}

case "${1:-}" in
  section) [ $# -eq 2 ] || die "usage: changelog.sh section <name>"; cmd_section "$2" ;;
  roll)    [ $# -eq 3 ] || die "usage: changelog.sh roll <version> <date>"; cmd_roll "$2" "$3" ;;
  *)       die "usage: changelog.sh {section <name>|roll <version> <date>}" ;;
esac
