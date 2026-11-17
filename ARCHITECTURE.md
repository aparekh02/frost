# Frost Bridge — Architecture

A native macOS runtime that runs latency-critical Rust workloads in isolated **capsules**
and gives them brokered, zero-copy access to the Apple Silicon GPU and unified memory.

## Why not Docker?

On macOS, Docker (and Apple's `container`) run Linux inside a VM. The VM has **no Metal
access**, its memory is a separate guest pool rather than the GPU-visible unified memory,
and every syscall/IPC crosses a hypervisor. For local inference and GPU crypto kernels
that's disqualifying. Frost Bridge isolates with the primitives the XNU kernel *does*
provide natively, and puts one trusted process, `frostd`, in charge of the GPU.

| Concern            | Docker on macOS               | Frost Bridge capsule                                       |
|--------------------|-------------------------------|------------------------------------------------------------|
| GPU                | none (VM)                     | Metal, brokered by `frostd`, zero-copy unified memory       |
| Filesystem/network | Linux namespaces in a VM      | Seatbelt (SBPL) deny-by-default profile                     |
| CPU placement      | VM vCPUs                      | QoS class → P-cores vs E-cores (verified: E-only = CPUs 0–5)|
| CPU real-time      | n/a                           | Mach `THREAD_TIME_CONSTRAINT_POLICY` (p99 wake 4.8 µs)      |
| Memory limit       | cgroup in VM                  | phys_footprint watchdog (2 ms) + jetsam attr (root only)    |
| GPU time/memory    | n/a                           | per-capsule token bucket + arena quota vs. working set      |
| Startup            | container + VM                | ~14 ms sandboxed, ~2.4 ms unsandboxed                       |

## Components

```
                         ┌──────────────────────── frostd (trusted) ───────────────────────┐
 frostd ctl ──unix sock──▶ serve loop ── launch ──▶ fb-capsule::spawn                         │
                         │                          (posix_spawn SETEXEC: QoS, pgroup,       │
                         │                           CLOEXEC_DEFAULT, rlimits, sandbox-exec)  │
                         │                                                                   │
                         │  ┌ dispatcher thread (QoS UI / optional RT) ────────────────────┐ │
                         │  │ for lane in lanes by priority:                               │ │
                         │  │   pop Jobs ─▶ validate ─▶ [split if yielding] ─▶ encode      │ │
                         │  │   per-capsule MTLCommandQueue ─▶ commit ─▶ poll ─▶ Completion│ │
                         │  │ token bucket · in-flight cap · backpressure · keep-warm      │ │
                         │  └──────────────────────────────────────────────────────────────┘ │
                         │  watchdog (footprint)  control thread/capsule  reaper (wait4)     │
                         └───────────▲──────────────────────────▲────────────────────────────┘
                                     │ MTLBuffer(bytesNoCopy)    │ JSON lines (socketpair fd)
                     ┌───────────────┴──────────────┐            │
  unified memory ──▶ │ arena (file-backed, unlinked) │◀── mmap ── capsule (sandboxed, no GPU
                     │ header · submit ring · compl. │            entitlement, no FS, no net)
                     │ ring · data region            │            links fb-client
                     └───────────────────────────────┘
```

| Crate        | Role |
|--------------|------|
| `fb-core`    | ABI (arena header, `Job`, `Completion`), SPSC rings in shared memory, arenas, manifest, scheduling (RT, QoS, rusage), `os_sync_wait_on_address` doorbells |
| `fb-gpu`     | Metal: device, per-capsule queues, kernel registry (built-in MSL + custom), zero-copy wrap, validated batch encoding, job splitting |
| `fb-capsule` | Seatbelt profile generation and process spawning |
| `fb-client`  | What a workload links: connect from inherited fds, bump-allocate GPU memory, submit/wait |
| `frostd`     | Supervisor + broker: `run`, `serve`, `ctl`, `info`, `profile`, `bench-direct` |
| `workloads`  | Example capsules + end-to-end tests |

## Data plane: shared arena

Each capsule gets one file in the run dir. It's created, `ftruncate`d, mapped by the daemon,
**unlinked** and then handed to the child only as an fd (`FROST_ARENA_FD`). The
sandboxed child needs no filesystem rights to it. The daemon wraps the whole mapping as a
single `MTLBuffer` (`newBufferWithBytesNoCopy`, `StorageModeShared`), so the CPU in both
processes and the GPU all touch the same physical pages.

```
@0        ArenaHeader     magic, ABI version, offsets, doorbells, closed flag
@128      submit ring     Job[256]        capsule → daemon   (168-byte slots)
@…        complete ring   Completion[256] daemon → capsule   (40-byte slots)
@16 KiB   data region     capsule-owned tensors / messages (bump allocated)
```

Rings are lock-free SPSC with head and tail on **separate 128-byte lines** (Apple Silicon's
cache-line size). The hot path is a single slot write plus a release store, with no syscalls.
Idle sides can park with `os_sync_wait_on_address(SHARED)`. Wakes are only issued when a
waiter flag is set, so spinning clients never pay for a syscall.

A job is `{kernel id, grid, threadgroup, ≤6 BufRef(offset,len), 8×u32 params}`. Buffers are
bound with `setBuffer:offset:` into the capsule's own arena buffer, and the params are bound with `setBytes`.

## Trust boundary

The capsule is untrusted and can write *anything* into its arena, including ring indices.
The daemon therefore:

* validates every `BufRef` against the **data window** of that capsule's arena: no overflow,
  16-byte alignment, and no access to the header or rings
* rejects unknown kernel ids, oversize grids and illegal threadgroups, and returns `Status` codes
* treats peer-written ring indices defensively (`wrapping_sub`, capacity bounds). An impossible
  `tail` gets the capsule **killed** ("protocol: corrupted submit ring")
* bounds per-tick work per lane, *counting rejected jobs and chain waits*, so garbage
  floods can't monopolise the dispatcher
* applies **backpressure** instead of growing memory when a capsule doesn't drain completions
* ends encoders on dropped batches (an all-rejected batch used to abort the daemon; found by
  the e2e suite)
* custom MSL kernels are **opt-in per manifest** (`gpu.custom_kernels`). They are a real trust
  boundary: all arenas live in the daemon's GPU address space, so a hostile kernel could
  address another capsule's arena. Only enable custom kernels for code you trust. Built-ins
  bounds-check by their count params.

## Scheduling

**CPU.** Core cluster is chosen by QoS at spawn (`posix_spawnattr_set_qos_class_np`,
which, unlike setting it before `exec`, survives the exec). `efficiency` → QoS BACKGROUND,
measured confined to the 6 E-cores. `performance` → USER_INTERACTIVE. Hot threads call
`Frost::enter_realtime()`, which applies the manifest's declared
`{period, computation, constraint}` as a Mach time-constraint policy.

**GPU.** Each capsule has its own `MTLCommandQueue`. The dispatcher admits work in priority
order, subject to:

* **Token bucket**: `gpu_ms_per_sec` with a 100 ms burst window, debited by measured
  `GPUEndTime − GPUStartTime`
* **In-flight cap**: 8 command buffers normally, or 1 while yielding
* **Preemption by slicing**: Metal can't preempt a running kernel, so while a higher-priority
  capsule is active (submitted within 20 ms), splittable jobs from lower-priority capsules
  (`vadd`, `saxpy`, `sha256_batch`, `gemv`) are cut into chunks sized from an EWMA of
  ns/item to about `slice_us` (default 100 µs). Completion is emitted only after the final chunk.
  Results are byte-identical to unsplit dispatch (tested).
* **Batching**: up to 64 ready jobs per command buffer, and `CHAIN` jobs share one buffer
  (e.g. all 16 layers of an MLP token)
* **Keep-warm** (optional, `--keepwarm-us`): tiny periodic dispatch to hold GPU clocks

## Measured (Apple M4, 4P+6E, 16 GB, macOS 27)

| Metric | Result |
|---|---|
| Capsule → daemon pickup | **2.3 µs p50, 3.6 µs p99** |
| GPU noop round trip, brokered | 145 µs p50, the same as direct Metal (144 µs). Metal's submit→complete path is the floor; the kernel itself runs in 6 µs |
| Pipelined dispatch throughput | **455 K jobs/s** (2.2 µs/job) |
| SHA-256, 262 K × 64 B | **173–178 MH/s** GPU vs 4.5 MH/s one CPU thread (all digests verified) |
| MLP 8×1024², per token | **587 µs p50, about 1,625 tok/s**, rel err 1.4e-6 (CPU 1-thread: 3.9 ms) |
| Wake lateness, 500 µs period, 14 hogs | timeshare p99 4.9–50 ms (178–1,862 misses) → **RT p99 4.8 µs, 0 misses** |
| High-prio GPU latency next to a 97%-busy flood | p90 **5,744 → 141 µs**, p99 **6,886 → 228 µs** with slicing; flood keeps ~83% of throughput |
| GPU budget 100 ms/s | 9.8% GPU busy observed |
| Memory limit 128 MiB | killed at 128.8 MiB peak |
| Startup | ~14 ms sandboxed (warm), 2.4 ms unsandboxed |

## Things that didn't work (documented negative results)

* **GPU-resident persistent kernel / mailbox.** A long-running kernel polling a CPU-written
  word in shared memory *never* observed the CPU's stores, with relaxed atomics, fences,
  RMW atomics, `coherent(device)` or `volatile`. GPU stores took 360–950 ms to surface.
  Apple GPUs expose no fine-grained CPU↔GPU coherence through public Metal, so the
  command-buffer path (about 100 µs floor) is the real minimum. See `fb-gpu/tests/persistent.rs`.
* **Kernel jetsam limit** (`posix_spawnattr_setjetsam_ext`) is ignored for an unprivileged
  spawner, so the footprint watchdog is the enforcing mechanism. It may apply when frostd runs
  as a root launchd daemon (untested).
* **Keep-warm** gave a modest p50/p99 gain but doesn't remove the roughly 2.5 ms p99.9 tail,
  which looks like display/WindowServer contention on the shared GPU.

## Limits and next steps

* `sandbox-exec` is deprecated (but still shipped and used by Apple tooling). The
  replacement path is an App Sandbox / XPC-service design with a signed helper.
* Hard per-capsule GPU memory isolation needs either trusted kernels only (current) or
  one broker process per trust domain (separate GPU VA spaces).
* Splitting covers built-ins. Custom kernels could declare a split spec (per-arg item
  stride + count param) at registration.
* No sharing of compiled SBPL profiles yet. Caching would cut most of the 14 ms startup.
