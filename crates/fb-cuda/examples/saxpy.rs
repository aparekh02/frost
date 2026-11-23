//! Classical CUDA on Apple Silicon: a SAXPY kernel authored in the CUDA-C subset, compiled
//! to Metal and run on the GPU, timed against a single-threaded CPU loop.
//!
//!     cargo run -p fb-cuda --release --example saxpy

use fb_cuda::{Arg, CudaRuntime};
use std::time::Instant;

const SRC: &str = r#"
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

fn main() -> anyhow::Result<()> {
    let rt = CudaRuntime::new()?;
    println!("device: {}  (unified memory: {})", rt.device_name(), rt.unified_memory());

    let module = rt.load_cuda_source(SRC)?;
    let saxpy = module.function("saxpy")?;

    let n = 16_000_000usize;
    let a = 2.0f32;
    let x = vec![1.0f32; n];
    let y = vec![2.0f32; n];

    let dx = rt.malloc(n * 4)?;
    let dy = rt.malloc(n * 4)?;
    rt.copy_to_device(dx, &x)?;
    rt.copy_to_device(dy, &y)?;

    let block = 256u32;
    let grid = (n as u32).div_ceil(block);

    // Warm up (first dispatch pays pipeline/clock costs).
    saxpy.launch_sync([grid, 1, 1], [block, 1, 1], &[Arg::scalar(a), Arg::Ptr(dx), Arg::Ptr(dy), Arg::scalar(n as i32)])?;
    rt.copy_to_device(dy, &y)?;

    let t = Instant::now();
    saxpy.launch_sync([grid, 1, 1], [block, 1, 1], &[Arg::scalar(a), Arg::Ptr(dx), Arg::Ptr(dy), Arg::scalar(n as i32)])?;
    let gpu = t.elapsed();

    let mut out = vec![0f32; n];
    rt.copy_from_device(&mut out, dy)?;

    let mut cy = y.clone();
    let t = Instant::now();
    for i in 0..n {
        cy[i] = a * x[i] + cy[i];
    }
    let cpu = t.elapsed();

    assert!(out.iter().all(|&v| (v - 4.0).abs() < 1e-4));
    println!("SAXPY n={n}: GPU {gpu:?}, CPU(1 thread) {cpu:?}  ->  {:.1}x", cpu.as_secs_f64() / gpu.as_secs_f64());
    println!("verified: y[0]={} (expected 4.0)", out[0]);
    Ok(())
}
