# fb-cuda — a CUDA-compatible runtime for Apple Silicon

Run CUDA-style classical kernels **and** CUDA-Q-style quantum kernels directly on an Apple
Silicon GPU (tested on **Apple M4**), through Metal.

## Read this first: what this is, and what it cannot be

**There is no NVIDIA GPU in an Apple Silicon Mac.** CUDA is NVIDIA's proprietary stack
whose kernel driver binds to NVIDIA hardware over a closed interface. A *real* NVIDIA CUDA
driver therefore **cannot run on an M-series chip** — there is no device for it to talk to.
That is a hardware fact, not a limitation of this code, and no software anywhere changes it.

`fb-cuda` is a **compatibility layer**, in the spirit of
[ZLUDA](https://github.com/vosen/ZLUDA) / [SCALE](https://docs.scale-lang.com/) on other
non-NVIDIA hardware, but targeting Apple Silicon + Metal:

* it implements a useful **subset of the CUDA Driver API** (`cuInit`, `cuMemAlloc`,
  `cuLaunchKernel`, …) and **Runtime API** (`cudaMalloc`, `cudaMemcpy`, …),
* it dispatches the actual compute to the Mac's GPU via **Metal** (the same path the rest of
  Frost Bridge uses), and
* it adds a **CUDA-Q-style quantum backend**: a GPU statevector simulator.

It will *not* run an unmodified NVIDIA binary or arbitrary `.cubin`/PTX. Kernels are authored
in a **CUDA-C subset** that is source-translated to Metal Shading Language (see below).

## Why Metal maps onto CUDA so cleanly here

| CUDA concept | This runtime |
|---|---|
| `CUdeviceptr` | address of a `StorageModeShared` Metal buffer — unified memory, so H↔D copies are a real `memcpy` (zero-copy), matching Frost Bridge's arena model |
| `<<<grid, block>>>` | `dispatchThreadgroups:threadsPerThreadgroup:` (exact 1:1) |
| `threadIdx/blockIdx/blockDim/gridDim` | `thread_position_in_threadgroup` / `threadgroup_position_in_grid` / `threads_per_threadgroup` / `threadgroups_per_grid` |
| `__shared__` / `__syncthreads()` | `threadgroup` / `threadgroup_barrier` |
| PTX/cubin JIT | CUDA-C-subset → MSL translation + `newLibraryWithSource` |

## Classical example (Rust API)

```rust
use fb_cuda::{Arg, CudaRuntime};

let rt = CudaRuntime::new()?;
let module = rt.load_cuda_source(r#"
    extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];
    }
"#)?;
let saxpy = module.function("saxpy")?;

let n = 1 << 20;
let (dx, dy) = (rt.malloc(n * 4)?, rt.malloc(n * 4)?);
rt.copy_to_device(dx, &vec![1.0f32; n])?;
rt.copy_to_device(dy, &vec![2.0f32; n])?;

let block = 256;
saxpy.launch_sync([(n as u32).div_ceil(block), 1, 1], [block, 1, 1],
    &[Arg::scalar(2.0f32), Arg::Ptr(dx), Arg::Ptr(dy), Arg::scalar(n as i32)])?;

let mut out = vec![0f32; n];
rt.copy_from_device(&mut out, dy)?;   // all 4.0
```

## Quantum example (CUDA-Q style)

```rust
use fb_cuda::{CudaRuntime, quantum::QuantumState};

let rt = CudaRuntime::new()?;
let mut q = QuantumState::new(&rt, 3)?;   // |000>
q.h(0)?; q.cx(0, 1)?; q.cx(1, 2)?;        // GHZ state on the GPU
let counts = q.sample(10_000, 42)?;       // ~5000 on |000>, ~5000 on |111>
```

Gates: `x y z h s sdg t tdg rx ry rz phase` and controlled `cx cy cz cphase`, plus
`apply_1q` / `apply_controlled_1q` for an arbitrary 2×2. Readout: `amplitudes`,
`probabilities`, `expect_z`, `sample`. Each gate is one GPU compute dispatch over
`2^(n-1)` amplitude pairs; a circuit is flushed as a single serial command buffer.

## C ABI / drop-in library

```
cargo build -p fb-cuda --release      # -> target/release/libfb_cuda.dylib
```

exports `cuInit`, `cuDeviceGet(Count|Name)`, `cuCtxCreate_v2`, `cuCtxSynchronize`,
`cuModuleLoadData`/`GetFunction`/`Unload`, `cuMemAlloc_v2`/`MemFree_v2`,
`cuMemcpyHtoD_v2`/`DtoH_v2`, `cuLaunchKernel`, `cuGetErrorString`, and Runtime-API
`cudaMalloc`/`cudaFree`/`cudaMemcpy`/`cudaDeviceSynchronize`/`cudaGetDeviceCount`.

`cuModuleLoadData`'s image is a **null-terminated CUDA-C-subset source string** (translated
to MSL and compiled), not a cubin. Launch via the Driver API's `cuLaunchKernel`; the Runtime
`kernel<<<...>>>(...)` syntax needs `nvcc` host stubs that are not emulated.

## CUDA-C subset supported by the translator

Supported: `__global__` entry points; `threadIdx/blockIdx/blockDim/gridDim`; pointer and
scalar kernel params; `__device__` helpers with value/scalar args; `__shared__`;
`__syncthreads()`; and the `f`-suffixed math builtins (`expf`, `sqrtf`, `fmaxf`, …).

Out of scope (hand-write MSL and use `CudaRuntime::load_msl` instead): textures/surfaces,
warp-shuffle intrinsics, device `atomicAdd` (Metal atomics differ), `__device__` helpers
taking pointers (MSL needs an explicit address space), and dynamic parallelism.

## Measured (Apple M4, 10-core GPU, macOS 27)

| Workload | Result |
|---|---|
| SAXPY, 16M f32, GPU | ~2.4 ms, result verified vs CPU |
| GHZ(3) sampling, 10k shots | ~50/50 on \|000⟩ / \|111⟩ |
| QFT, **22 qubits** (4.19M amplitudes, 253 gates) | evolved on GPU in ~108 ms |
| Test suite | 16 tests (classical + quantum + C ABI) pass on-device |

SAXPY is bandwidth-bound, so its CPU/GPU ratio is modest; compute-dense kernels (GEMM,
quantum evolution) are where the GPU pulls ahead.

## Tests

```
cargo test -p fb-cuda          # classical.rs, quantum.rs, abi.rs — all run on the GPU
cargo run -p fb-cuda --release --example saxpy
cargo run -p fb-cuda --release --example quantum_ghz
```
