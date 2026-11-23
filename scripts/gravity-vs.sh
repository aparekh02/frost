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
cat > "$tmp/gravity.toml" <<EOF
name = "gravity"
exec = "$(pwd)/target/release/gravity"
args = ["--frames", "$FRAMES", "--balls", "$BALLS", "--width", "$W", "--height", "$H", "--spin"]
[cpu]
class = "performance"
[gpu]
arena_mb = 32
priority = 50
EOF

echo "### same game, ${W}x${H}, ${BALLS} balls, ${FRAMES} frames — with vs without the bridge"
echo
printf 'containerized GPU bridge   : '
$F run --quiet "$tmp/gravity.toml" 2>/dev/null | python3 - <<'PY'
import sys, json
for l in sys.stdin:
    i = l.find("RESULT ")
    if i < 0: continue
    d = json.loads(l[i+7:]).get("gravity")
    if d:
        fl = d["frame_latency"]
        print("%7.0f fps  frame p50 %.0fus p99 %.0fus  gpu %.0fus" % (d["fps"], fl["p50_us"], fl["p99_us"], d["gpu_render"]["p50_us"]))
PY
$F bench-gravity --frames "$FRAMES" --balls "$BALLS" --width "$W" --height "$H"
rm -rf "$tmp"
