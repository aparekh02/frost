//! Experiment: can a persistent GPU kernel poll a CPU-written mailbox in unified memory?
//! Tries several MSL coherence strategies and reports which (if any) observe CPU writes.
use fb_core::abi::{BufRef, Job};
use fb_core::time::{now_ns, Summary};
use fb_gpu::{Gpu, Window};
use std::sync::atomic::{AtomicU32, Ordering};

fn src(variant: &str) -> String {
    let (qual, load) = match variant {
        "relaxed" => ("device atomic_uint*", "atomic_load_explicit(&mbox[0], memory_order_relaxed)"),
        "fence" => ("device atomic_uint*", "(atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device), atomic_load_explicit(&mbox[0], memory_order_relaxed))"),
        "rmw" => ("device atomic_uint*", "atomic_fetch_add_explicit(&mbox[0], 0u, memory_order_relaxed)"),
        "coherent" => ("coherent(device) device atomic_uint*", "atomic_load_explicit(&mbox[0], memory_order_relaxed)"),
        "volatile" => ("volatile device atomic_uint*", "atomic_load_explicit(&mbox[0], memory_order_relaxed)"),
        _ => unreachable!(),
    };
    format!(r#"
#include <metal_stdlib>
using namespace metal;
kernel void resident({qual} mbox [[buffer(0)]], constant uint* p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    if (gid != 0) return;
    uint last = 0; uint spins = 0;
    while (spins < p[0]) {{
        uint s = {load};
        if (s != last) {{
            if (s == 0xFFFFFFFFu) break;
            last = s;
            atomic_store_explicit(&mbox[1], s * 2u + 1u, memory_order_relaxed);
            spins = 0;
        }} else {{ spins++; }}
        atomic_fetch_add_explicit(&mbox[2], 1u, memory_order_relaxed);
    }}
}}"#)
}

fn try_variant(g: &Gpu, v: &str, iters: u32) -> Option<Vec<u64>> {
    let id = match g.register(&format!("resident_{v}"), &src(v), "resident") {
        Ok(id) => id,
        Err(e) => { println!("[{v}] compile failed: {}", e.to_string().lines().next().unwrap_or("")); return None; }
    };
    let p = std::env::temp_dir().join(format!("fb-resident-{}-{v}", std::process::id()));
    let a = fb_core::shm::Arena::create(&p, 1 << 16).unwrap();
    std::fs::remove_file(&p).ok();
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    let q = g.new_queue().unwrap();
    let mut j = Job { kernel: id, n_args: 1, grid: [1, 1, 1], tg: [1, 1, 1], ..Default::default() };
    j.args[0] = BufRef { off: 0, len: 64 };
    j.params[0] = 20_000_000;
    let mut b = g.batch_on(&q).unwrap();
    b.encode(&buf, Window { off: 0, len: a.len() as u64 }, &j).unwrap();
    let inflight = b.commit();
    let mb = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const AtomicU32, 3) };
    let t0 = now_ns();
    while mb[2].load(Ordering::Relaxed) == 0 && now_ns() - t0 < 2_000_000_000 { std::hint::spin_loop() }
    println!("[{v}] heartbeat visible mid-kernel: {} after {:.1} ms", mb[2].load(Ordering::Relaxed) != 0, (now_ns() - t0) as f64 / 1e6);
    let mut lat = Vec::new();
    let mut ok = true;
    for s in 1..=iters {
        let t = now_ns();
        mb[0].store(s, Ordering::Release);
        loop {
            if mb[1].load(Ordering::Acquire) == s * 2 + 1 { lat.push(now_ns() - t); break; }
            if now_ns() - t > 20_000_000 { ok = false; break; }
            std::hint::spin_loop();
        }
        if !ok { break; }
    }
    mb[0].store(0xFFFF_FFFF, Ordering::Release);
    let d = inflight.wait();
    println!("[{v}] observed={ok} answered={} gpu_ok={} heartbeat={}", lat.len(), d.ok, mb[2].load(Ordering::Relaxed));
    if ok { Some(lat) } else { None }
}

/// Run with `cargo test -p fb-gpu --release --test persistent -- --ignored --nocapture`.
/// Finding (M4, macOS 27): CPU stores are never observed by a running kernel and GPU
/// stores take hundreds of ms to surface, so a GPU-resident mailbox is not viable.
#[test]
#[ignore]
fn persistent_kernel_mailbox() {
    let g = Gpu::new().unwrap();
    let mut best: Option<(String, Vec<u64>)> = None;
    for v in ["relaxed", "fence", "rmw", "coherent", "volatile"] {
        if let Some(l) = try_variant(&g, v, 20000) {
            if best.is_none() { best = Some((v.to_string(), l)); }
        }
    }
    if let Some((v, mut l)) = best {
        println!("resident GPU mailbox round trip [{v}]: {}", Summary::of(&mut l));
    } else {
        println!("no variant observed CPU writes");
    }
}
