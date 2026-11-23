# Frost Bridge

Native macOS capsules for low-latency Rust, with brokered zero-copy access to the Apple
Silicon GPU. It's built for local AI inference and GPU/CPU crypto workloads where Docker's
Linux VM (no Metal, separate memory pool) is a non-starter.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the design, trust model and measurements.

## Quick start

```bash
cargo build --release
./target/release/frostd info                          # GPU, cores, working set, kernels
./target/release/frostd run capsules/probe.toml       # what can a capsule do? (almost nothing)
./target/release/frostd run capsules/crypto-hash.toml capsules/infer-mlp.toml
./target/release/frostd run capsules/lat-under-load.toml capsules/hash-flood.toml   # priority slicing
./target/release/frostd run capsules/gravity.toml     # a game rendered on the GPU through the bridge
./target/release/frostd run capsules/gravity-demo.toml && open /tmp/gravity-frames   # viewable PPM frames
scripts/bench.sh                                      # reproduce all headline numbers
```

> The `gravity` capsule is the interactive answer to "run a game on the GPU." It is **not**
> a Linux container — Linux on macOS runs in a VM with no Metal access, which is the reason
> Frost Bridge exists. It is a native, Seatbelt-sandboxed capsule whose frames are drawn by
> the GPU's **hardware render pipeline** (vertex+fragment shaders) into a texture that aliases
> a zero-copy unified-memory framebuffer — ~30× faster than the compute rasterizer at scale
> (`--mode compute`). See ARCHITECTURE.md § "Gravity: the GPU graphics plane".

Long-lived daemon:

```bash
./target/release/frostd serve &                       # socket: /private/tmp/frost-$UID/frostd.sock (0600)
./target/release/frostd ctl launch capsules/infer-mlp.toml
./target/release/frostd ctl ps
./target/release/frostd ctl kill infer-mlp
./target/release/frostd ctl shutdown
```

Useful flags for `run`/`serve`: `--rt` (real-time dispatcher thread), `--slice-us N`,
`--no-slicing`, `--keepwarm-us N`, `--idle-spin-us N`, `--no-watchdog`, `--json`, `--quiet`.
Use `frostd profile <capsule.toml>` to print the generated Seatbelt profile.

## Writing a capsule

```toml
# capsules/mine.toml
name = "mine"
exec = "../target/release/mine"
[cpu]
class = "performance"            # or "efficiency" (E-cores only)
realtime = { period_us = 1000, computation_us = 300, constraint_us = 900 }
[memory]
limit_mb = 512
[gpu]
arena_mb = 64                    # zero-copy GPU-visible memory
priority = 50
gpu_ms_per_sec = 500
```

```rust
let mut fb = fb_client::Frost::connect()?;          // uses fds handed down by frostd
fb.enter_realtime()?;                               // optional: RT band for this thread
let x = fb.alloc::<f32>(n)?;  let y = fb.alloc::<f32>(n)?;
fb.view_mut::<f32>(x).fill(1.0);                    // CPU writes, GPU reads: same pages
let c = fb.run(fb.job("saxpy_f32").arg(x).arg(y).grid(n as u32).param(n as u32).param_f32(2.0))?;
println!("gpu {} ns", c.gpu_ns);
```

Built-in kernels: `vadd_f32`, `saxpy_f32`, `gemv_f32`, `sha256_batch`, `noop`,
`gravity_render` (a compute-shader rasterizer), plus `draw_triangles` — the hardware
render pipeline (vertex+fragment) for the graphics plane. With
`gpu.custom_kernels = true`, `fb.register_kernel(name, msl_source, entry)` adds your own.

## Tests

```bash
cargo test --release --workspace -- --test-threads=1     # 24 tests, ~20 s, needs an Apple Silicon GPU
```

Latest run (Apple M4, macOS 27.0.1, Rust 1.98.1): **24 passed, 0 failed, 1 ignored**
(the ignored one is the persistent-kernel experiment, kept as a documented negative result).

| Suite | Tests | What it proves |
|---|---|---|
| `fb-core` unit | 6/6 ✅ | SPSC ring ordering over 2M items, full/empty, **corrupted peer indices contained**, ABI layout stable, manifest validation, cross-thread `os_sync` wake |
| `fb-capsule` unit | 2/2 ✅ | Seatbelt profile is deny-by-default; a sandboxed shell can't read `/etc/hosts` or write outside its root |
| `fb-gpu` integration | 7/7 ✅ | zero-copy dispatch on a file-backed `mmap`; SHA-256 (lengths 0–200 B) and GEMV match the CPU; out-of-window/overflow/bad-kernel rejected; custom MSL kernels; **split jobs byte-identical to unsplit**; dropped uncommitted batches don't abort |
| `workloads` end-to-end | 10/10 ✅ | real `frostd` + real sandboxed capsules (details below) |
| `fb-client` doctest | 1/1 ✅ | API example compiles |

End-to-end results from that run:

| E2E test | Result |
|---|---|
| `isolation_and_core_placement` | FS read/write outside root, list dirs, network, fork, exec: **all denied**; efficiency capsule at QoS 9 on E-cores only |
| `adversarial_capsule_is_contained` | header/huge/wrapping/unaligned buffer refs → `OutOfBounds`, bogus kernel → `BadKernel`, unauthorised MSL registration refused, capsule still works |
| `hostile_capsules_do_not_starve_others` | ring-corrupting capsule killed; 100,000 garbage jobs rejected; victim p50 **147 µs** (same as idle) |
| `memory_limit_kills_capsule` | killed by watchdog just above its limit |
| `gpu_workloads_are_correct` | 0 SHA-256 mismatches; MLP rel. error < 1e-3 |
| `gpu_budget_is_enforced` | 100 ms/s budget → < 16% GPU busy (≈ 9.8% typical) |
| `priority_slicing_protects_latency` | high-prio p90 under a GPU flood: **6,823 µs → 152 µs** |
| `gravity_renders_frames` | sandboxed game renders on the GPU (both render + compute paths); deterministic (checksum reproducible), **> 120 fps** |
| `gravity_hardware_render_beats_compute` | the hardware rasterizer spends less GPU time per frame than the compute software rasterizer |
| `realtime_policy_bounds_jitter` | p99 wake lateness under 14 CPU hogs: **14,620 µs → 3.5 µs**, 0 misses |

## Benchmark results

Reproduce with `scripts/bench.sh` (Apple M4: 4P + 6E cores, 10-core GPU, 16 GB).

| Metric | Result |
|---|---|
| Capsule → daemon job pickup | 2.3 µs p50 · 3.6 µs p99 |
| GPU noop round trip, brokered vs direct Metal | 145 µs vs 144 µs p50 (Metal's submit→complete floor; the kernel itself takes 6 µs) |
| Pipelined dispatch throughput | 455,000 jobs/s (2.2 µs/job) |
| SHA-256, 262,144 × 64 B messages | 173–178 MH/s GPU vs 4.5 MH/s one CPU thread, all digests verified |
| MLP inference, 8 layers × 1024², per token | 587 µs p50 · ~1,625 tok/s · rel. err 1.4e-6 (CPU one thread: 3.9 ms) |
| Wake lateness, 500 µs period, 14 CPU hogs | timeshare p99 4.9–50 ms → **RT p99 4.8 µs**, 0 misses |
| High-prio GPU latency next to a 97%-busy flood | p90 5,744 → 141 µs · p99 6,886 → 228 µs; flood keeps ~83% throughput |
| GPU budget 100 ms/s | 9.8% GPU busy |
| Capsule startup | ~14 ms sandboxed (warm) · 2.4 ms unsandboxed |

Known limits and negative results (no fine-grained CPU↔GPU coherence for persistent kernels,
jetsam limits need root, ~2.5 ms p99.9 GPU tail) are in
[ARCHITECTURE.md](ARCHITECTURE.md#things-that-didnt-work-documented-negative-results) and tracked
as GitHub issues.
