//! fb-core: shared, dependency-light building blocks for Frost Bridge.
//!
//! * [`shm`]      – file-backed shared memory arenas (mmap MAP_SHARED, page aligned)
//! * [`ring`]     – lock-free single-producer/single-consumer rings that live *inside* an arena
//! * [`abi`]      – the binary layout shared by daemon and capsule (arena header, Job, Completion)
//! * [`proto`]    – control-plane messages (JSON lines over an inherited socketpair)
//! * [`manifest`] – capsule specification (TOML)
//! * [`sched`]    – macOS thread/CPU/memory controls (RT policy, QoS, mlock, footprint)
//! * [`time`]     – mach_absolute_time helpers

pub mod abi;
pub mod manifest;
pub mod proto;
pub mod ring;
pub mod sched;
pub mod shm;
pub mod time;
pub mod wait;

/// Page size on Apple Silicon is 16 KiB; we query it anyway.
pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

pub fn round_up(v: usize, align: usize) -> usize {
    v.div_ceil(align) * align
}
