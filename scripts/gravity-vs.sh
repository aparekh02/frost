#!/bin/bash
# "With vs without the gravity containerization": render the SAME game three ways and compare.
#   1. containerized GPU bridge  — a sandboxed capsule rendering through frostd (the real thing)
#   2. direct Metal, no container — same hardware render in-process, no broker/sandbox (overhead baseline)
#   3. CPU only, no GPU           — software rasterizer (no acceleration baseline)
# Usage: scripts/gravity-vs.sh [frames] [balls] [width] [height]
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -q
F=./target/release/frostd
FRAMES=${1:-1200}; BALLS=${2:-16}; W=${3:-854}; H=${4:-480}

tmp=$(mktemp -d)
cap() { cat > "$tmp/g.toml" <<EOF
name = "gravity"
exec = "$(pwd)/target/release/gravity"
args = ["--frames", "$FRAMES", "--balls", "$BALLS", "--width", "$W", "--height", "$H", "--spin", "--depth", "$1"]
[cpu]
class = "performance"
[gpu]
arena_mb = 64
priority = 50
EOF
}
fps() { python3 - <<'PY'
import sys, json
for l in sys.stdin:
    i = l.find("RESULT ")
    if i < 0: continue
    d = json.loads(l[i+7:]).get("gravity")
    if d: print("%7.0f fps  (gpu %.0fus/frame)" % (d["fps"], d["gpu_render"]["p50_us"]))
PY
}

echo "### same game, ${W}x${H}, ${BALLS} balls, ${FRAMES} frames — with vs without the bridge"
echo
printf 'containerized bridge, depth 1 : '; cap 1; $F run --quiet "$tmp/g.toml" 2>/dev/null | fps
printf 'containerized bridge, depth 8 : '; cap 8; $F run --quiet "$tmp/g.toml" 2>/dev/null | fps
$F bench-gravity --frames "$FRAMES" --balls "$BALLS" --width "$W" --height "$H"
rm -rf "$tmp"
