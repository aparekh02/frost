//! Batch SHA-256 on the GPU through frostd, verified against the CPU (sha2 crate).
use fb_client::Frost;
use fb_core::time::now_ns;
use sha2::{Digest, Sha256};

fn main() -> anyhow::Result<()> {
    let n: usize = workloads::arg("--n", 1 << 18);
    let len: usize = workloads::arg("--len", 64);
    let rounds: usize = workloads::arg("--rounds", 10);
    let mut fb = Frost::connect()?;
    let stride = len.div_ceil(16) * 16;
    let msgs = fb.alloc::<u8>(n * stride)?;
    let out = fb.alloc::<u8>(n * 32)?;
    {
        let m = fb.view_mut::<u8>(msgs);
        let mut x = 0x9e3779b97f4a7c15u64;
        for b in m.iter_mut() {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *b = x as u8;
        }
    }
    // warm-up (pipeline + page tables + GPU clocks)
    fb.run(fb.job("sha256_batch").arg(msgs).arg(out).grid(n as u32).param(n as u32).param(len as u32).param(stride as u32))?;
    let t0 = now_ns();
    let mut gpu_ns = 0;
    for _ in 0..rounds {
        let c = fb.run(fb.job("sha256_batch").arg(msgs).arg(out).grid(n as u32).param(n as u32).param(len as u32).param(stride as u32))?;
        gpu_ns += c.gpu_ns;
    }
    let wall = now_ns() - t0;
    let gpu_mhs = (n * rounds) as f64 / (wall as f64 / 1e9) / 1e6;

    // CPU reference: verify every digest, and time single-thread throughput.
    let m = fb.view_mut::<u8>(msgs);
    let o = fb.view_mut::<u8>(out);
    let t1 = now_ns();
    let mut bad = 0usize;
    for i in 0..n {
        let d = Sha256::digest(&m[i * stride..i * stride + len]);
        if d.as_slice() != &o[i * 32..i * 32 + 32] {
            bad += 1;
        }
    }
    let cpu = now_ns() - t1;
    let cpu_mhs = n as f64 / (cpu as f64 / 1e9) / 1e6;
    println!("sha256 x{n} ({len} B msgs): GPU {gpu_mhs:.1} MH/s (gpu busy {:.1}%), CPU 1-thread {cpu_mhs:.1} MH/s, mismatches={bad}",
        100.0 * gpu_ns as f64 / wall as f64);
    workloads::result(serde_json::json!({"crypto_hash": {"n": n, "len": len, "gpu_mhs": gpu_mhs, "cpu1_mhs": cpu_mhs, "mismatches": bad, "speedup": gpu_mhs / cpu_mhs, "gpu_busy": gpu_ns as f64 / wall as f64}}));
    anyhow::ensure!(bad == 0, "GPU digests mismatch");
    Ok(())
}
