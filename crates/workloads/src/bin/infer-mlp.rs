//! Toy inference: an L-layer MLP (GEMV + custom ReLU kernel) per token, each token one
//! chained command buffer, weights resident in the zero-copy arena. Verified vs CPU.
use fb_client::Frost;
use fb_core::time::{now_ns, Summary};

const RELU: &str = r#"
#include <metal_stdlib>
kernel void relu(device float* x [[buffer(0)]], constant uint* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) x[i] = metal::max(x[i], 0.0f);
}"#;

fn main() -> anyhow::Result<()> {
    let d: usize = workloads::arg("--dim", 1024);
    let layers: usize = workloads::arg("--layers", 8);
    let tokens: usize = workloads::arg("--tokens", 200);
    let mut fb = Frost::connect()?;
    let relu = fb.register_kernel("relu", RELU, "relu")?;
    let ws: Vec<_> = (0..layers).map(|_| fb.alloc::<f32>(d * d)).collect::<Result<_, _>>()?;
    let acts: Vec<_> = (0..=layers).map(|_| fb.alloc::<f32>(d)).collect::<Result<_, _>>()?;
    let mut seed = 1u32;
    let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5 };
    for w in &ws { for v in fb.view_mut::<f32>(*w) { *v = rnd() * 0.06; } }
    let mut lat = Vec::with_capacity(tokens);
    let mut gpu = Vec::with_capacity(tokens);
    for t in 0..tokens + 5 {
        for (i, v) in fb.view_mut::<f32>(acts[0]).iter_mut().enumerate() { *v = ((i + t) % 17) as f32 * 0.1 - 0.8; }
        let mut chain = Vec::new();
        for l in 0..layers {
            chain.push(fb.job("gemv_f32").arg(ws[l]).arg(acts[l]).arg(acts[l + 1]).grid((d * 32) as u32).threadgroup(256).param(d as u32).param(d as u32));
            anyhow::ensure!(fb.kernel_id("relu") == Some(relu));
            chain.push(fb.job("relu").arg(acts[l + 1]).grid(d as u32).param(d as u32));
        }
        let t0 = now_ns();
        let seqs = fb.submit_chain(chain);
        let c = fb.wait(*seqs.last().unwrap())?;
        for s in &seqs[..seqs.len() - 1] { fb.wait(*s)?; }
        if t >= 5 { lat.push(now_ns() - t0); gpu.push(c.gpu_ns); }
    }
    // CPU reference for the last token
    let mut x: Vec<f32> = fb.view_mut::<f32>(acts[0]).to_vec();
    let tc = now_ns();
    for l in 0..layers {
        let w = fb.view_mut::<f32>(ws[l]);
        x = (0..d).map(|r| w[r * d..(r + 1) * d].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>().max(0.0)).collect();
    }
    let cpu_ns = now_ns() - tc;
    let y = fb.view_mut::<f32>(acts[layers]);
    let max_err = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let s = Summary::of(&mut lat);
    let g = Summary::of(&mut gpu);
    let tps = 1e9 / s.mean;
    println!("mlp {layers}x{d}x{d}: per-token {s}");
    println!("   gpu time/token {g}; {tps:.0} tok/s; cpu 1-thread {:.0} us/token; rel err {:.2e}", cpu_ns as f64 / 1e3, max_err / scale);
    workloads::result(serde_json::json!({"infer_mlp": {"dim": d, "layers": layers, "token_latency": workloads::summary_json(&s),
        "tok_per_s": tps, "cpu1_us_per_token": cpu_ns as f64 / 1e3, "rel_err": max_err / scale}}));
    anyhow::ensure!(max_err / scale < 1e-3, "GPU/CPU mismatch");
    Ok(())
}
