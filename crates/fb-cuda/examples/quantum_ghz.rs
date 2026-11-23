//! CUDA-Q-style quantum kernel on Apple Silicon: build a GHZ state and a QFT, with the
//! statevector evolution running on the GPU.
//!
//!     cargo run -p fb-cuda --release --example quantum_ghz

use fb_cuda::quantum::QuantumState;
use fb_cuda::CudaRuntime;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let rt = CudaRuntime::new()?;
    println!("quantum backend on: {}\n", rt.device_name());

    // --- GHZ state on 3 qubits: (|000> + |111>)/sqrt(2) ---
    let mut q = QuantumState::new(&rt, 3)?;
    q.h(0)?;
    q.cx(0, 1)?;
    q.cx(1, 2)?;
    println!("GHZ(3) measurement counts over 10,000 shots:");
    for (i, c) in q.sample(10_000, 42)?.iter().enumerate() {
        if *c > 0 {
            println!("  |{i:03b}> : {c}");
        }
    }

    // --- QFT on a larger register, timed ---
    let n = 22u32; // 2^22 = ~4.2M amplitudes evolving on the GPU
    let mut big = QuantumState::new(&rt, n)?;
    let t = Instant::now();
    for j in 0..n {
        big.h(j)?;
        for k in (j + 1)..n {
            let angle = std::f32::consts::PI / (1u32 << (k - j)) as f32;
            big.cphase(k, j, angle)?;
        }
    }
    big.flush()?; // force all GPU dispatches to complete
    let elapsed = t.elapsed();
    let gates = (n * (n + 1) / 2) as usize;
    println!(
        "\nQFT on {n} qubits ({} amplitudes, {gates} gates) evolved on GPU in {elapsed:?}",
        1usize << n
    );
    Ok(())
}
