//! mach_absolute_time helpers (on Apple Silicon one tick = 125/3 ns, i.e. 24 MHz).

use std::sync::OnceLock;

#[repr(C)]
#[derive(Default)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

fn timebase() -> (u64, u64) {
    static TB: OnceLock<(u64, u64)> = OnceLock::new();
    *TB.get_or_init(|| {
        let mut i = MachTimebaseInfo::default();
        unsafe { mach_timebase_info(&mut i) };
        (i.numer as u64, i.denom as u64)
    })
}

#[inline]
pub fn ticks() -> u64 {
    unsafe { mach_absolute_time() }
}

#[inline]
pub fn ticks_to_ns(t: u64) -> u64 {
    let (n, d) = timebase();
    ((t as u128 * n as u128) / d as u128) as u64
}

#[inline]
pub fn ns_to_ticks(ns: u64) -> u64 {
    let (n, d) = timebase();
    ((ns as u128 * d as u128) / n as u128) as u64
}

/// Monotonic nanoseconds, comparable across processes on the same machine.
#[inline]
pub fn now_ns() -> u64 {
    ticks_to_ns(ticks())
}

/// Simple latency summary over a sample of nanosecond values.
#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub n: usize,
    pub min: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
    pub max: u64,
    pub mean: f64,
}

impl Summary {
    pub fn of(samples: &mut [u64]) -> Summary {
        assert!(!samples.is_empty());
        samples.sort_unstable();
        let n = samples.len();
        let q = |p: f64| samples[(((n - 1) as f64) * p).round() as usize];
        Summary {
            n,
            min: samples[0],
            p50: q(0.50),
            p90: q(0.90),
            p99: q(0.99),
            p999: q(0.999),
            max: samples[n - 1],
            mean: samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64,
        }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let us = |v: u64| v as f64 / 1000.0;
        write!(
            f,
            "n={} min={:.2}us p50={:.2}us p90={:.2}us p99={:.2}us p99.9={:.2}us max={:.2}us",
            self.n,
            us(self.min),
            us(self.p50),
            us(self.p90),
            us(self.p99),
            us(self.p999),
            us(self.max)
        )
    }
}
