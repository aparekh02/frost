#!/bin/bash
# Reproduce Frost Bridge's headline measurements. Usage: scripts/bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -q
F=./target/release/frostd
echo "### platform";                $F info
echo; echo "### direct Metal baseline (no broker)"; $F bench-direct --iters 3000
$F run capsules/gpu-latency.toml 2>/dev/null | grep -E "round trip|pickup|pipelined" | grep -v RESULT | sed 's/^\[[^]]*\] *//'
echo; echo "### crypto + inference";           $F run capsules/crypto-hash.toml capsules/infer-mlp.toml 2>/dev/null | grep -E "sha256|mlp|tok/s" | grep -v RESULT | sed 's/^\[[^]]*\] *//'
echo; echo "### RT jitter under 14 CPU hogs";  $F run capsules/rt-jitter.toml 2>/dev/null | grep lateness | grep -v RESULT | sed 's/^\[[^]]*\] *//'
for mode in --no-slicing ""; do
  echo; echo "### high-prio latency next to a GPU flood ${mode:-(slicing on)}"
  $F run $mode capsules/lat-under-load.toml capsules/hash-flood.toml 2>/dev/null | grep -E "round trip .spin|sha256" | grep -v RESULT | sed 's/^\[[^]]*\] *//'
done
echo; echo "### gravity: a game rendered on the GPU through the bridge (hardware render pipeline)"
$F run capsules/gravity.toml 2>/dev/null | grep -E "gravity:|gpu render" | grep -v RESULT | sed 's/^\[[^]]*\] *//'
echo; echo "### gravity: hardware rasterizer vs compute rasterizer, 4000 sprites @ 1280x720"
for m in render compute; do
  tmp=$(mktemp -d); cat > "$tmp/g.toml" <<EOF
name = "g"
exec = "$(pwd)/target/release/gravity"
args = ["--mode", "$m", "--frames", "400", "--balls", "4000", "--width", "1280", "--height", "720"]
[cpu]
class = "performance"
[gpu]
arena_mb = 32
priority = 50
EOF
  printf '%-8s ' "$m:"; $F run "$tmp/g.toml" 2>/dev/null | grep -E "gravity:" | grep -v RESULT | sed 's/^\[[^]]*\] *//'
  rm -rf "$tmp"
done
