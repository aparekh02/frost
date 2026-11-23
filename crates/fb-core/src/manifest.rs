//! Capsule manifest (`capsule.toml`).
//!
//! ```toml
//! name = "hasher"
//! exec = "target/release/crypto-hash"
//! args = ["--batches", "64"]
//!
//! [cpu]
//! class = "performance"          # performance | efficiency | default
//! max_cpu_seconds = 120
//! realtime = { period_us = 1000, computation_us = 300, constraint_us = 900 }
//!
//! [memory]
//! limit_mb = 512                 # phys_footprint watchdog (SIGKILL on breach)
//! wire_arena = true              # mlock the shared GPU arena
//!
//! [gpu]
//! arena_mb = 64                  # zero-copy unified-memory arena shared with the GPU
//! priority = 10                  # higher is served first by the GPU scheduler
//! gpu_ms_per_sec = 800           # GPU time budget (token bucket)
//! max_percent = 25               # OR: cap GPU time at 25% (overrides gpu_ms_per_sec)
//! max_inflight = 4               # cap command buffers in flight (instantaneous occupancy)
//! custom_kernels = false         # allow runtime MSL registration (trust boundary!)
//!
//! [sandbox]
//! enabled = true
//! network = false
//! read  = []                     # extra read-only paths
//! write = []                     # extra read-write paths (capsule root is always rw)
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub name: String,
    pub exec: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cpu: Cpu,
    #[serde(default)]
    pub memory: Memory,
    #[serde(default)]
    pub gpu: Gpu,
    #[serde(default)]
    pub sandbox: Sandbox,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoreClass {
    /// P-cores preferred (QoS USER_INTERACTIVE).
    Performance,
    /// E-cores only (QoS BACKGROUND).
    Efficiency,
    #[default]
    Default,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Realtime {
    pub period_us: u64,
    pub computation_us: u64,
    pub constraint_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Cpu {
    #[serde(default)]
    pub class: CoreClass,
    pub max_cpu_seconds: Option<u64>,
    pub realtime: Option<Realtime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    #[serde(default = "d_limit")]
    pub limit_mb: u64,
    #[serde(default = "d_true")]
    pub wire_arena: bool,
}
impl Default for Memory {
    fn default() -> Self {
        Memory { limit_mb: d_limit(), wire_arena: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    #[serde(default = "d_arena")]
    pub arena_mb: u64,
    #[serde(default)]
    pub priority: i32,
    /// GPU time budget as a token bucket (ms of GPU time per wall-clock second).
    #[serde(default = "d_gpu_budget")]
    pub gpu_ms_per_sec: u64,
    /// Friendlier cap: at most this percent of the GPU's time. When set it overrides
    /// `gpu_ms_per_sec` (25 => the capsule may never exceed 25% GPU utilization).
    #[serde(default)]
    pub max_percent: Option<u32>,
    /// Cap on command buffers this capsule may have in flight at once — bounds how much of
    /// the GPU it can occupy *instantaneously*, even inside its time budget.
    #[serde(default = "d_inflight")]
    pub max_inflight: u32,
    #[serde(default)]
    pub custom_kernels: bool,
}
impl Default for Gpu {
    fn default() -> Self {
        Gpu {
            arena_mb: d_arena(),
            priority: 0,
            gpu_ms_per_sec: d_gpu_budget(),
            max_percent: None,
            max_inflight: d_inflight(),
            custom_kernels: false,
        }
    }
}
impl Gpu {
    /// Effective GPU-time budget (ms/s): `max_percent` wins over `gpu_ms_per_sec` when set.
    /// 1% of a second == 10 ms of GPU time per second.
    pub fn budget_ms_per_sec(&self) -> u64 {
        match self.max_percent {
            Some(p) => (p.min(100) as u64) * 10,
            None => self.gpu_ms_per_sec,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sandbox {
    #[serde(default = "d_true")]
    pub enabled: bool,
    #[serde(default)]
    pub network: bool,
    #[serde(default)]
    pub read: Vec<String>,
    #[serde(default)]
    pub write: Vec<String>,
}
impl Default for Sandbox {
    fn default() -> Self {
        Sandbox { enabled: true, network: false, read: vec![], write: vec![] }
    }
}

fn d_true() -> bool {
    true
}
fn d_limit() -> u64 {
    1024
}
fn d_arena() -> u64 {
    16
}
fn d_gpu_budget() -> u64 {
    1000
}
fn d_inflight() -> u32 {
    8
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Manifest> {
        let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&s).with_context(|| format!("parse {}", path.display()))
    }
    pub fn parse(s: &str) -> Result<Manifest> {
        let m: Manifest = toml::from_str(s)?;
        anyhow::ensure!(
            !m.name.is_empty() && m.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "capsule name must be [A-Za-z0-9_-]+"
        );
        anyhow::ensure!(m.gpu.arena_mb >= 1 && m.gpu.arena_mb <= 8192, "gpu.arena_mb out of range");
        anyhow::ensure!(m.gpu.max_percent.is_none_or(|p| p >= 1 && p <= 100), "gpu.max_percent must be 1..=100");
        anyhow::ensure!(m.gpu.max_inflight >= 1 && m.gpu.max_inflight <= 64, "gpu.max_inflight must be 1..=64");
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_full_and_defaults() {
        let m = Manifest::parse(
            r#"
            name = "x"
            exec = "/bin/echo"
            [cpu]
            class = "efficiency"
            realtime = { period_us = 1000, computation_us = 200, constraint_us = 800 }
            [gpu]
            arena_mb = 8
            priority = 3
        "#,
        )
        .unwrap();
        assert_eq!(m.cpu.class, CoreClass::Efficiency);
        assert_eq!(m.gpu.priority, 3);
        assert_eq!(m.gpu.max_inflight, 8); // default
        assert_eq!(m.gpu.budget_ms_per_sec(), 1000); // default = 100%
        assert!(m.sandbox.enabled && !m.sandbox.network);
        // max_percent overrides gpu_ms_per_sec and is validated.
        let capped = Manifest::parse("name='c'\nexec='x'\n[gpu]\nmax_percent=25\nmax_inflight=2").unwrap();
        assert_eq!(capped.gpu.budget_ms_per_sec(), 250);
        assert_eq!(capped.gpu.max_inflight, 2);
        assert!(Manifest::parse("name='c'\nexec='x'\n[gpu]\nmax_percent=0").is_err());
        assert!(Manifest::parse("name='c'\nexec='x'\n[gpu]\nmax_inflight=0").is_err());
        assert!(Manifest::parse("name='bad name'\nexec='x'").is_err());
        assert!(Manifest::parse("name='a'\nexec='x'\nbogus=1").is_err());
    }
}
