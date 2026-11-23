//! Binary ABI shared by `frostd` and capsules. Everything here is `repr(C)` and lives
//! in the shared arena, so changing it requires bumping [`ABI_VERSION`].
//!
//! ```text
//! arena (file-backed, MAP_SHARED, wrapped by one MTLBuffer in the daemon)
//! ┌───────────────┬──────────────────┬────────────────────┬──────────────────────────┐
//! │ ArenaHeader   │ submit ring      │ complete ring      │ data region (16K aligned)│
//! │ @0            │ Job[SUBMIT_CAP]  │ Completion[..]     │ capsule-owned buffers    │
//! └───────────────┴──────────────────┴────────────────────┴──────────────────────────┘
//! ```

use crate::ring::{ring_bytes, CACHE_LINE};
use std::sync::atomic::AtomicU32;

pub const ARENA_MAGIC: u64 = 0x4652_4f53_5442_5247; // "FROSTBRG"
pub const ABI_VERSION: u32 = 1;

pub const SUBMIT_CAP: usize = 256;
pub const COMPLETE_CAP: usize = 256;
pub const MAX_ARGS: usize = 6;
pub const MAX_PARAMS: usize = 8;

/// A reference to a byte range inside the arena (offset from arena start).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufRef {
    pub off: u64,
    pub len: u64,
}

/// One GPU dispatch request written by the capsule into the submit ring.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Job {
    pub seq: u64,
    pub kernel: u32,
    pub n_args: u16,
    pub flags: u16,
    /// Total threads (not threadgroups) in each dimension.
    pub grid: [u32; 3],
    /// Threads per threadgroup; 0 = let the daemon pick.
    pub tg: [u32; 3],
    /// Bound at buffer indices 0..n_args.
    pub args: [BufRef; MAX_ARGS],
    /// Inline constants, bound with setBytes at buffer index `n_args`.
    pub params: [u32; MAX_PARAMS],
}

/// More jobs follow that should be encoded into the same command buffer.
pub const JOB_FLAG_CHAIN: u16 = 1 << 0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Completion {
    pub seq: u64,
    pub status: u32,
    pub kernel: u32,
    /// GPU execution time of the command buffer that carried this job.
    pub gpu_ns: u64,
    /// Daemon observed the job (mach absolute time, ns).
    pub t_seen_ns: u64,
    /// Daemon observed GPU completion (mach absolute time, ns).
    pub t_done_ns: u64,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    BadKernel = 1,
    OutOfBounds = 2,
    Quota = 3,
    GpuError = 4,
    BadArgs = 5,
}

impl Status {
    pub fn from_u32(v: u32) -> Status {
        match v {
            0 => Status::Ok,
            1 => Status::BadKernel,
            2 => Status::OutOfBounds,
            3 => Status::Quota,
            4 => Status::GpuError,
            _ => Status::BadArgs,
        }
    }
}

/// Built-in kernel ids (see fb-gpu/src/kernels.metal). Custom kernels get ids >= 1000.
pub mod kernel {
    pub const VADD_F32: u32 = 1;
    pub const GEMV_F32: u32 = 2;
    pub const SHA256_BATCH: u32 = 3;
    pub const NOOP: u32 = 4;
    pub const SAXPY_F32: u32 = 5;
    /// Rasterize a frame into an RGBA8 framebuffer (the "gravity" interactive plane).
    pub const GRAVITY_RENDER: u32 = 6;
    pub const FIRST_CUSTOM: u32 = 1000;
}

#[repr(C, align(128))]
pub struct ArenaHeader {
    pub magic: u64,
    pub abi_version: u32,
    pub _r0: u32,
    pub arena_len: u64,
    pub submit_off: u64,
    pub complete_off: u64,
    pub data_off: u64,
    pub data_len: u64,
    /// Bumped by the capsule after pushing jobs (lets an idle daemon sleep on it).
    pub submit_bell: AtomicU32,
    /// Non-zero while the daemon is parked on `submit_bell`.
    pub submit_waiters: AtomicU32,
    /// Bumped by the daemon after pushing completions.
    pub complete_bell: AtomicU32,
    /// Non-zero while a capsule thread is parked on `complete_bell`; the daemon only
    /// issues a wake syscall when this is set, so spinning clients pay nothing.
    pub complete_waiters: AtomicU32,
    /// Set by the daemon when it shuts the capsule's GPU channel down.
    pub closed: AtomicU32,
}

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub submit_off: usize,
    pub complete_off: usize,
    pub data_off: usize,
    pub arena_len: usize,
}

impl Layout {
    /// Compute the layout for an arena whose *data region* is at least `data_len` bytes.
    pub fn for_data(data_len: usize) -> Layout {
        let page = crate::page_size();
        let submit_off = crate::round_up(std::mem::size_of::<ArenaHeader>(), CACHE_LINE);
        let complete_off = crate::round_up(submit_off + ring_bytes::<Job>(SUBMIT_CAP), CACHE_LINE);
        let data_off =
            crate::round_up(complete_off + ring_bytes::<Completion>(COMPLETE_CAP), page);
        let arena_len = crate::round_up(data_off + data_len, page);
        Layout { submit_off, complete_off, data_off, arena_len }
    }
}

/// Daemon-wide doorbell page shared (read-write) with every capsule. Its only power is
/// waking an idle dispatcher, so a misbehaving capsule can at worst cause spurious wakeups.
#[repr(C, align(128))]
pub struct BellPage {
    pub bell: AtomicU32,
    _pad: [u8; 124],
    /// Non-zero while the dispatcher is parked.
    pub waiters: AtomicU32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_are_stable() {
        assert_eq!(std::mem::size_of::<BufRef>(), 16);
        assert_eq!(std::mem::size_of::<Job>(), 8 + 4 + 2 + 2 + 12 + 12 + 16 * MAX_ARGS + 4 * MAX_PARAMS);
        assert_eq!(std::mem::size_of::<Completion>(), 40);
        let l = Layout::for_data(1 << 20);
        assert_eq!(l.data_off % crate::page_size(), 0);
        assert!(l.arena_len >= l.data_off + (1 << 20));
    }
}
