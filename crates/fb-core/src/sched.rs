//! macOS scheduling and memory controls.
//!
//! Apple Silicon does not expose hard CPU affinity (THREAD_AFFINITY_POLICY is a no-op on
//! arm64). The levers that *do* work, unprivileged, are:
//!
//! * **QoS class** – USER_INTERACTIVE/USER_INITIATED threads are eligible for P-cores;
//!   BACKGROUND threads are confined to E-cores. This is how we pick a core cluster.
//! * **Real-time time-constraint policy** – the same mechanism CoreAudio uses; the thread
//!   gets a fixed-priority band above all timeshare threads as long as it honours its
//!   declared computation budget.
//! * **Wired memory** – `mlock` keeps pages resident (no compression / swap).
//! * **Footprint accounting** – `proc_pid_rusage` gives phys_footprint, the number the
//!   kernel's memory-pressure system (jetsam) itself uses.

use anyhow::{bail, Result};

type MachPort = u32;
type KernReturn = i32;

const THREAD_TIME_CONSTRAINT_POLICY: u32 = 2;
const THREAD_TIME_CONSTRAINT_POLICY_COUNT: u32 = 4;
const THREAD_PRECEDENCE_POLICY: u32 = 3;

#[repr(C)]
struct ThreadTimeConstraintPolicy {
    period: u32,
    computation: u32,
    constraint: u32,
    preemptible: i32,
}

extern "C" {
    fn mach_thread_self() -> MachPort;
    fn mach_task_self() -> MachPort;
    fn mach_port_deallocate(task: MachPort, name: MachPort) -> KernReturn;
    fn thread_policy_set(thread: MachPort, flavor: u32, info: *const i32, count: u32) -> KernReturn;
    fn pthread_set_qos_class_self_np(qos: u32, relpri: i32) -> i32;
    fn pthread_cpu_number_np(cpu: *mut usize) -> i32;
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV2) -> i32;
}

/// QoS classes from <sys/qos.h>.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qos {
    UserInteractive = 0x21,
    UserInitiated = 0x19,
    Default = 0x15,
    Utility = 0x11,
    Background = 0x09,
}

pub fn set_thread_qos(q: Qos) -> Result<()> {
    let r = unsafe { pthread_set_qos_class_self_np(q as u32, 0) };
    if r != 0 {
        bail!("pthread_set_qos_class_self_np({q:?}) = {r}");
    }
    Ok(())
}

/// Put the calling thread into the real-time band.
///
/// * `period_ns`      – nominal interval between wakeups (0 = aperiodic)
/// * `computation_ns` – CPU time needed in each period
/// * `constraint_ns`  – max wall time from start to finish of the computation
///
/// The kernel demotes threads that overrun `computation` persistently, so be honest.
pub fn set_realtime(period_ns: u64, computation_ns: u64, constraint_ns: u64) -> Result<()> {
    use crate::time::ns_to_ticks;
    let pol = ThreadTimeConstraintPolicy {
        period: ns_to_ticks(period_ns) as u32,
        computation: ns_to_ticks(computation_ns) as u32,
        constraint: ns_to_ticks(constraint_ns) as u32,
        preemptible: 1,
    };
    unsafe {
        let th = mach_thread_self();
        let kr = thread_policy_set(
            th,
            THREAD_TIME_CONSTRAINT_POLICY,
            &pol as *const _ as *const i32,
            THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        );
        mach_port_deallocate(mach_task_self(), th);
        if kr != 0 {
            bail!("thread_policy_set(TIME_CONSTRAINT) kern_return={kr}");
        }
    }
    Ok(())
}

/// Raise (or lower) the thread's importance within its band.
pub fn set_precedence(importance: i32) -> Result<()> {
    unsafe {
        let th = mach_thread_self();
        let kr = thread_policy_set(th, THREAD_PRECEDENCE_POLICY, &importance, 1);
        mach_port_deallocate(mach_task_self(), th);
        if kr != 0 {
            bail!("thread_policy_set(PRECEDENCE) kern_return={kr}");
        }
    }
    Ok(())
}

/// Logical CPU the calling thread is currently running on.
pub fn current_cpu() -> usize {
    let mut c = 0usize;
    unsafe { pthread_cpu_number_np(&mut c) };
    c
}

/// Wire a region of memory.
pub fn mlock(ptr: *const u8, len: usize) -> Result<()> {
    if unsafe { libc::mlock(ptr as *const _, len) } != 0 {
        bail!("mlock: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

pub fn set_rlimit(resource: libc::c_int, soft: u64, hard: u64) -> Result<()> {
    let rl = libc::rlimit { rlim_cur: soft as libc::rlim_t, rlim_max: hard as libc::rlim_t };
    if unsafe { libc::setrlimit(resource, &rl) } != 0 {
        bail!("setrlimit({resource}): {}", std::io::Error::last_os_error());
    }
    Ok(())
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct RusageInfoV2 {
    pub uuid: [u8; 16],
    pub user_time: u64,
    pub system_time: u64,
    pub pkg_idle_wkups: u64,
    pub interrupt_wkups: u64,
    pub pageins: u64,
    pub wired_size: u64,
    pub resident_size: u64,
    pub phys_footprint: u64,
    pub proc_start_abstime: u64,
    pub proc_exit_abstime: u64,
    pub child_user_time: u64,
    pub child_system_time: u64,
    pub child_pkg_idle_wkups: u64,
    pub child_interrupt_wkups: u64,
    pub child_pageins: u64,
    pub child_elapsed_abstime: u64,
    pub diskio_bytesread: u64,
    pub diskio_byteswritten: u64,
}

const RUSAGE_INFO_V2: i32 = 2;

pub fn rusage(pid: i32) -> Result<RusageInfoV2> {
    let mut r = RusageInfoV2::default();
    if unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V2, &mut r) } != 0 {
        bail!("proc_pid_rusage({pid}): {}", std::io::Error::last_os_error());
    }
    Ok(r)
}

/// Physical footprint in bytes (what jetsam and Activity Monitor's "Memory" column use).
pub fn phys_footprint(pid: i32) -> Result<u64> {
    Ok(rusage(pid)?.phys_footprint)
}

/// CPU time (user+system) in ns for a process.
pub fn cpu_time_ns(pid: i32) -> Result<u64> {
    let r = rusage(pid)?;
    // On Apple Silicon these are mach absolute time units.
    Ok(crate::time::ticks_to_ns(r.user_time + r.system_time))
}

/// Topology from sysctl: (performance cores, efficiency cores).
pub fn core_topology() -> (u32, u32) {
    fn sysctl_u32(name: &str) -> u32 {
        let c = std::ffi::CString::new(name).unwrap();
        let mut v: u32 = 0;
        let mut sz = std::mem::size_of::<u32>();
        let r = unsafe {
            libc::sysctlbyname(c.as_ptr(), &mut v as *mut _ as *mut _, &mut sz, std::ptr::null_mut(), 0)
        };
        if r == 0 {
            v
        } else {
            0
        }
    }
    (sysctl_u32("hw.perflevel0.physicalcpu"), sysctl_u32("hw.perflevel1.physicalcpu"))
}
