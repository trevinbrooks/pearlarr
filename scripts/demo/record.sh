#!/usr/bin/env bash
# Re-record docs/assets/demo_run.gif in one command (maintainer tooling).
#
# Pipeline: fixtures -> fresh demo data dir -> mock Sonarr/qBittorrent -> vhs -> ffmpeg.
# VHS ignores `Set Framerate` for GIF output (the raw gif comes out 25fps). The ffmpeg
# 10fps pass is what keeps the GitHub iOS app's inline renderer happy - it decodes
# every frame to a raw bitmap and gives up past ~600 frames (1432 broke, 573 works).
#
# Needs: vhs + ffmpeg (brew), the repo venv, and network (SeaDex is queried live by
# both the mocks and the real CLI run being recorded).
set -euo pipefail

HARNESS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HARNESS/../.." && pwd)"
PY="$REPO/.venv/bin/python"
# Rendered in the gif's boot header ("Data directory: ...") - keep it presentable,
# and keep it in sync with the export line in demo.tape.
DATA_DIR="$HOME/.local/share/pearlarr"
RAW="$HARNESS/demo.gif"
FINAL="$HARNESS/demo_run.gif"
MAX_FRAMES=600

command -v vhs >/dev/null || { echo "vhs not installed (brew install vhs)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not installed (brew install ffmpeg)" >&2; exit 1; }
[ -x "$PY" ] || { echo "repo venv missing: $PY (run: uv sync)" >&2; exit 1; }

# The fixtures are committed. Regenerate only if lost (needs network).
if [ ! -f "$HARNESS/fixtures/series_demo.json" ]; then
  echo "== regenerating fixtures"
  "$PY" "$HARNESS/make_fixtures.py"
fi

# Fresh demo data dir. The marker keeps rm -rf from ever eating a real data dir.
if [ -e "$DATA_DIR" ] && [ ! -f "$DATA_DIR/.pearlarr-demo" ]; then
  echo "$DATA_DIR exists but has no .pearlarr-demo marker - not demo-owned, move it aside" >&2
  exit 1
fi
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"
touch "$DATA_DIR/.pearlarr-demo"
install -m 600 "$HARNESS/config.yml" "$DATA_DIR/config.yml"

echo "== starting mock Sonarr (:8989) + qBittorrent (:8080)"
"$PY" "$HARNESS/demo_mocks.py" >"$HARNESS/mocks.log" 2>&1 &
MOCKS=$!
# The || true matters: set -e applies inside the trap, so wait's 143 (the mocks'
# SIGTERM status) would otherwise become the whole script's exit code.
trap 'kill "$MOCKS" 2>/dev/null; wait "$MOCKS" 2>/dev/null || true' EXIT
ready=
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8080/api/v2/app/webapiVersion >/dev/null 2>&1 &&
     curl -fsS http://127.0.0.1:8989/api/v3/qualitydefinition >/dev/null 2>&1; then
    ready=1
    break
  fi
  kill -0 "$MOCKS" 2>/dev/null || break
  sleep 1
done
if [ -z "$ready" ]; then
  echo "mocks failed to start; mocks.log tail:" >&2
  tail -20 "$HARNESS/mocks.log" >&2
  exit 1
fi

echo "== recording (~70s of terminal time)"
(cd "$HARNESS" && vhs demo.tape)

echo "== post-processing to 10fps"
ffmpeg -y -loglevel error -i "$RAW" \
  -vf "fps=10,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  "$FINAL"

frames=$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of csv=p=0 "$FINAL")
if [ "$frames" -gt "$MAX_FRAMES" ]; then
  echo "FAIL: $FINAL has $frames frames (> $MAX_FRAMES) - the GitHub iOS app will show '?'" >&2
  exit 1
fi
echo "== done: $FINAL ($frames frames, $(du -h "$FINAL" | cut -f1 | tr -d ' '))"
# The GIF is never committed to main (gitignored). The Release workflow runs this
# script itself and publishes the result - run by hand only to iterate.
