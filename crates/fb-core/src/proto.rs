//! Control plane: newline-delimited JSON over a socketpair inherited by the capsule.
//! Only setup and rare operations go here — the hot path is the shared-memory rings.

use crate::manifest::Realtime;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};

/// Env vars the daemon sets for a capsule.
pub const ENV_CTL_FD: &str = "FROST_CTL_FD";
pub const ENV_ARENA_FD: &str = "FROST_ARENA_FD";
pub const ENV_CAPSULE: &str = "FROST_CAPSULE";
pub const ENV_ROOT: &str = "FROST_ROOT";
pub const ENV_BELL_FD: &str = "FROST_BELL_FD";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum ToDaemon {
    Hello { pid: i32 },
    RegisterKernel { name: String, source: String, entry: String },
    Stats,
    Bye,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum ToCapsule {
    Welcome {
        capsule: String,
        gpu_name: String,
        kernels: BTreeMap<String, u32>,
        realtime: Option<Realtime>,
        data_off: u64,
        data_len: u64,
    },
    KernelRegistered { name: String, id: u32 },
    /// `throttled` is in milliseconds.
    Stats { jobs: u64, gpu_ns: u64, throttled: u64, footprint: u64 },
    Error { msg: String },
}

pub fn send<W: Write, T: Serialize>(w: &mut W, msg: &T) -> std::io::Result<()> {
    let mut s = serde_json::to_vec(msg)?;
    s.push(b'\n');
    w.write_all(&s)?;
    w.flush()
}

pub fn recv<R: BufRead, T: for<'de> Deserialize<'de>>(r: &mut R) -> std::io::Result<Option<T>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    serde_json::from_str(&line).map(Some).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
