//! Allocates and touches memory until it reaches --mb (to exercise the watchdog).
//!
//! `--rate-mbps R` (R > 0) paces allocation to ~R MiB/s. At a bounded rate the measured
//! peak reflects the *watchdog's* overshoot (one ~2 ms poll interval), not raw memory
//! bandwidth — this is what issue #6's "within 5% at 1 GB/s" acceptance measures. With no
//! flag the historical fixed 300 us/MiB pace is kept (a near-flat-out allocator).
fn main() {
    let mb: usize = workloads::arg("--mb", 2048);
    let rate_mbps: usize = workloads::arg("--rate-mbps", 0);
    let per_mib_us: u64 = if rate_mbps > 0 {
        1_000_000 / rate_mbps as u64
    } else {
        300
    };
    let mut v: Vec<Vec<u8>> = Vec::new();
    for i in 0..mb {
        v.push(vec![(i & 0xff) as u8 | 1; 1 << 20]);
        if i % 64 == 0 {
            println!("allocated {i} MiB");
        }
        std::thread::sleep(std::time::Duration::from_micros(per_mib_us));
    }
    workloads::result(serde_json::json!({"mem_hog_survived": mb}));
}
