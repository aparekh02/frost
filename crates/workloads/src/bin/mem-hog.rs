//! Allocates and touches memory until it reaches --mb (to exercise the watchdog).
fn main() {
    let mb: usize = workloads::arg("--mb", 2048);
    let mut v: Vec<Vec<u8>> = Vec::new();
    for i in 0..mb {
        v.push(vec![(i & 0xff) as u8 | 1; 1 << 20]);
        if i % 64 == 0 { println!("allocated {i} MiB"); }
        std::thread::sleep(std::time::Duration::from_micros(300));
    }
    workloads::result(serde_json::json!({"mem_hog_survived": mb}));
}
