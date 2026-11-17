//! Round-trip latency of a no-op GPU job through frostd (submit ring → Metal → complete ring).
use fb_client::{Frost, WaitMode};
use fb_core::time::{now_ns, Summary};

fn main() -> anyhow::Result<()> {
    let iters: usize = workloads::arg("--iters", 3000);
    let mut fb = Frost::connect()?;
    std::thread::sleep(std::time::Duration::from_millis(workloads::arg("--delay-ms", 0)));
    if workloads::flag("--rt") {
        fb.enter_realtime()?;
    }
    let out = fb.alloc::<u32>(4)?;
    let mut results = serde_json::Map::new();
    for (label, mode) in [("spin", WaitMode::Spin), ("park", WaitMode::Park { spin_ns: 20_000 })] {
        fb.wait_mode = mode;
        let mut rt = Vec::with_capacity(iters);
        let mut daemon_to_seen = Vec::with_capacity(iters);
        let mut gpu = Vec::with_capacity(iters);
        for i in 0..iters + 100 {
            let t0 = now_ns();
            let c = fb.run(fb.job("noop").arg(out).grid(1).param(i as u32))?;
            let t1 = now_ns();
            assert_eq!(fb.view_mut::<u32>(out)[0], i as u32 + 1);
            if i >= 100 {
                rt.push(t1 - t0);
                daemon_to_seen.push(c.t_seen_ns.saturating_sub(t0));
                gpu.push(c.gpu_ns);
            }
        }
        let s = Summary::of(&mut rt);
        let q = Summary::of(&mut daemon_to_seen);
        let g = Summary::of(&mut gpu);
        println!("brokered noop round trip ({label}): {s}");
        println!("   submit→daemon pickup         : {q}");
        println!("   gpu execution                : {g}");
        results.insert(label.into(), serde_json::json!({
            "round_trip": workloads::summary_json(&s), "pickup": workloads::summary_json(&q), "gpu": workloads::summary_json(&g)
        }));
    }
    // Pipelined throughput: keep up to `depth` jobs outstanding.
    fb.wait_mode = WaitMode::Spin;
    let total: u64 = workloads::arg("--pipeline", 20000);
    let depth: u64 = workloads::arg("--depth", 128);
    let t0 = now_ns();
    let mut issued = 0u64;
    let mut done = 0u64;
    let mut next_wait = 1u64;
    let base = fb.job("noop").arg(out).grid(1);
    let first_seq = fb.submit(base) ; issued += 1;
    let _ = first_seq;
    while done < total {
        while issued < total && issued - done < depth {
            fb.submit(base);
            issued += 1;
        }
        fb.wait(next_wait + (first_seq - 1))?;
        next_wait += 1;
        done += 1;
    }
    let el = now_ns() - t0;
    let jps = total as f64 / (el as f64 / 1e9);
    println!("pipelined noop throughput (depth {depth}): {:.0} jobs/s ({:.2} us/job)", jps, el as f64 / total as f64 / 1e3);
    results.insert("pipeline_jobs_per_s".into(), serde_json::json!(jps));
    workloads::result(serde_json::json!({"gpu_latency": results}));
    Ok(())
}
