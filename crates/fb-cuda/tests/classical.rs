//! Classical CUDA kernels, authored in the CUDA-C subset, translated to MSL and run on the
//! Apple GPU. Each result is checked against a CPU reference.

use fb_cuda::{Arg, CudaRuntime};

const KERNELS: &str = r#"
extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Naive C[MxN] = A[MxK] * B[KxN], one thread per output element (2D grid).
extern "C" __global__ void gemm(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// Exercises the f-suffixed math-builtin rewrite (expf -> exp) and the ternary operator.
extern "C" __global__ void elu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : (expf(v) - 1.0f);
    }
}

// Exercises __shared__ and __syncthreads: per-block reduction.
extern "C" __global__ void block_sum(const float* x, float* out, int n) {
    __shared__ float tmp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    tmp[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] += tmp[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = tmp[0];
}
"#;

fn ceil_div(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b) as u32
}

#[test]
fn vadd_matches_cpu() {
    let rt = CudaRuntime::new().unwrap();
    assert!(rt.unified_memory(), "Apple Silicon should report unified memory");
    println!("device: {}", rt.device_name());
    let m = rt.load_cuda_source(KERNELS).unwrap();
    let f = m.function("vadd").unwrap();

    let n = 1_000_003usize; // deliberately not a multiple of the block size
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();

    let da = rt.malloc(n * 4).unwrap();
    let db = rt.malloc(n * 4).unwrap();
    let dc = rt.malloc(n * 4).unwrap();
    rt.copy_to_device(da, &a).unwrap();
    rt.copy_to_device(db, &b).unwrap();

    let block = 256;
    f.launch_sync(
        [ceil_div(n, block), 1, 1],
        [block as u32, 1, 1],
        &[Arg::Ptr(da), Arg::Ptr(db), Arg::Ptr(dc), Arg::scalar(n as i32)],
    )
    .unwrap();

    let mut c = vec![0f32; n];
    rt.copy_from_device(&mut c, dc).unwrap();
    for i in 0..n {
        assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
    }
    rt.free(da).unwrap();
    rt.free(db).unwrap();
    rt.free(dc).unwrap();
}

#[test]
fn saxpy_matches_cpu() {
    let rt = CudaRuntime::new().unwrap();
    let m = rt.load_cuda_source(KERNELS).unwrap();
    let f = m.function("saxpy").unwrap();

    let n = 50_000usize;
    let alpha = 2.5f32;
    let x: Vec<f32> = (0..n).map(|i| (i % 97) as f32).collect();
    let mut y: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let expected: Vec<f32> = (0..n).map(|i| alpha * x[i] + y[i]).collect();

    let dx = rt.malloc(n * 4).unwrap();
    let dy = rt.malloc(n * 4).unwrap();
    rt.copy_to_device(dx, &x).unwrap();
    rt.copy_to_device(dy, &y).unwrap();

    f.launch_sync(
        [ceil_div(n, 256), 1, 1],
        [256, 1, 1],
        &[Arg::scalar(alpha), Arg::Ptr(dx), Arg::Ptr(dy), Arg::scalar(n as i32)],
    )
    .unwrap();

    rt.copy_from_device(&mut y, dy).unwrap();
    for i in 0..n {
        assert!((y[i] - expected[i]).abs() < 1e-3, "mismatch at {i}: {} vs {}", y[i], expected[i]);
    }
}

#[test]
fn gemm_matches_cpu() {
    let rt = CudaRuntime::new().unwrap();
    let m = rt.load_cuda_source(KERNELS).unwrap();
    let f = m.function("gemm").unwrap();

    let (mm, nn, kk) = (64usize, 48usize, 80usize);
    let a: Vec<f32> = (0..mm * kk).map(|i| ((i * 7 % 11) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..kk * nn).map(|i| ((i * 5 % 9) as f32) * 0.2).collect();
    let mut c = vec![0f32; mm * nn];

    // CPU reference.
    let mut expected = vec![0f32; mm * nn];
    for r in 0..mm {
        for col in 0..nn {
            let mut acc = 0f32;
            for k in 0..kk {
                acc += a[r * kk + k] * b[k * nn + col];
            }
            expected[r * nn + col] = acc;
        }
    }

    let da = rt.malloc(a.len() * 4).unwrap();
    let db = rt.malloc(b.len() * 4).unwrap();
    let dc = rt.malloc(c.len() * 4).unwrap();
    rt.copy_to_device(da, &a).unwrap();
    rt.copy_to_device(db, &b).unwrap();

    let block = [16u32, 16, 1];
    let grid = [ceil_div(nn, 16), ceil_div(mm, 16), 1];
    f.launch_sync(
        grid,
        block,
        &[
            Arg::Ptr(da),
            Arg::Ptr(db),
            Arg::Ptr(dc),
            Arg::scalar(mm as i32),
            Arg::scalar(nn as i32),
            Arg::scalar(kk as i32),
        ],
    )
    .unwrap();

    rt.copy_from_device(&mut c, dc).unwrap();
    for i in 0..c.len() {
        assert!((c[i] - expected[i]).abs() < 1e-2, "gemm mismatch at {i}: {} vs {}", c[i], expected[i]);
    }
}

#[test]
fn elu_uses_translated_math() {
    let rt = CudaRuntime::new().unwrap();
    let m = rt.load_cuda_source(KERNELS).unwrap();
    let f = m.function("elu").unwrap();

    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 100.0).collect();
    let dx = rt.malloc(n * 4).unwrap();
    let dy = rt.malloc(n * 4).unwrap();
    rt.copy_to_device(dx, &x).unwrap();

    f.launch_sync(
        [ceil_div(n, 256), 1, 1],
        [256, 1, 1],
        &[Arg::Ptr(dx), Arg::Ptr(dy), Arg::scalar(n as i32)],
    )
    .unwrap();

    let mut y = vec![0f32; n];
    rt.copy_from_device(&mut y, dy).unwrap();
    for i in 0..n {
        let want = if x[i] > 0.0 { x[i] } else { x[i].exp() - 1.0 };
        assert!((y[i] - want).abs() < 1e-4, "elu mismatch at {i}: {} vs {}", y[i], want);
    }
}

#[test]
fn block_sum_shared_memory() {
    let rt = CudaRuntime::new().unwrap();
    let m = rt.load_cuda_source(KERNELS).unwrap();
    let f = m.function("block_sum").unwrap();

    let n = 100_000usize;
    let x: Vec<f32> = (0..n).map(|_| 1.0f32).collect(); // sum should be exactly n
    let blocks = ceil_div(n, 256) as usize;

    let dx = rt.malloc(n * 4).unwrap();
    let dout = rt.malloc(blocks * 4).unwrap();
    rt.copy_to_device(dx, &x).unwrap();

    f.launch_sync(
        [blocks as u32, 1, 1],
        [256, 1, 1],
        &[Arg::Ptr(dx), Arg::Ptr(dout), Arg::scalar(n as i32)],
    )
    .unwrap();

    let mut partial = vec![0f32; blocks];
    rt.copy_from_device(&mut partial, dout).unwrap();
    let total: f32 = partial.iter().sum();
    assert_eq!(total, n as f32, "block reduction sum mismatch");
}
