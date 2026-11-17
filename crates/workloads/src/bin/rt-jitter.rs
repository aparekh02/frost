//! Periodic-deadline jitter under CPU contention, timeshare vs real-time policy.
//! This is the CPU-side primitive for latency-critical Rust (e.g. order/crypto loops).
use fb_core::time::{ns_to_ticks, ticks, ticks_to_ns, Summary};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

extern "C" {
    fn mach_wait_until(deadline: u64) -> i32;
}

fn measure(period_ns: u64, iters: usize) -> Vec<u64> {
    let mut late = Vec::with_capacity(iters);
    let p = ns_to_ticks(period_ns);
    let mut deadline = ticks() + p;
    for _ in 0..iters {
        unsafe { mach_wait_until(deadline) };
        let now = ticks();
        late.push(ticks_to_ns(now.saturating_sub(deadline)));
        // ~20us of "work" (e.g. sign/verify, book update)
        let t = now + ns_to_ticks(20_000);
        let mut x = 1u64;
        while ticks() < t { x = x.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(1); }
        std::hint::black_box(x);
        deadline += p;
    }
    late
}

fn main() -> anyhow::Result<()> {
    let period_us: u64 = workloads::arg("--period-us", 500);
    let iters: usize = workloads::arg("--iters", 4000);
    let stress: usize = workloads::arg("--stress", 0);
    let stop = Arc::new(AtomicBool::new(false));
    let hogs: Vec<_> = (0..stress).map(|_| {
        let s = stop.clone();
        std::thread::spawn(move || {
            let _ = fb_core::sched::set_thread_qos(fb_core::sched::Qos::UserInteractive);
            let mut x = 0u64;
            while !s.load(Ordering::Relaxed) { x = x.wrapping_add(1); std::hint::black_box(x); }
        })
    }).collect();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut out = serde_json::Map::new();
    for mode in ["timeshare", "realtime"] {
        let res = std::thread::scope(|sc| sc.spawn(|| {
            let _ = fb_core::sched::set_thread_qos(fb_core::sched::Qos::UserInteractive);
            if mode == "realtime" {
                fb_core::sched::set_realtime(period_us * 1000, 100_000, period_us * 1000 * 9 / 10).expect("rt");
            }
            measure(period_us * 1000, iters)
        }).join().unwrap());
        let mut late = res;
        let miss = late.iter().filter(|&&l| l > period_us * 1000 / 2).count();
        let s = Summary::of(&mut late);
        println!("{mode:>9} wake lateness (period {period_us}us, {stress} hogs): {s}  misses(>P/2)={miss}");
        out.insert(mode.into(), serde_json::json!({"lateness": workloads::summary_json(&s), "misses": miss}));
    }
    stop.store(true, Ordering::Relaxed);
    for h in hogs { h.join().unwrap(); }
    workloads::result(serde_json::json!({"rt_jitter": out, "stress": stress}));
    Ok(())
}
