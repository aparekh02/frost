//! Shared helpers for example capsule workloads.
pub fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}
pub fn flag(name: &str) -> bool {
    std::env::args().any(|x| x == name)
}
/// Emit a machine-readable result line (the daemon forwards these even with --quiet).
pub fn result(v: serde_json::Value) {
    println!("RESULT {v}");
}
pub fn us(ns: u64) -> f64 {
    ns as f64 / 1000.0
}
pub fn summary_json(s: &fb_core::time::Summary) -> serde_json::Value {
    serde_json::json!({"n": s.n, "min_us": us(s.min), "p50_us": us(s.p50), "p90_us": us(s.p90), "p99_us": us(s.p99), "p999_us": us(s.p999), "max_us": us(s.max)})
}
