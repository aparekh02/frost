//! Cross-process futex-style waiting via `os_sync_wait_on_address` (macOS 14.4+).
//! Used so a capsule can park while its GPU job runs instead of burning a core.

use std::sync::atomic::{AtomicU32, Ordering};

const OS_SYNC_WAIT_ON_ADDRESS_SHARED: u32 = 1;
const OS_SYNC_WAKE_BY_ADDRESS_SHARED: u32 = 1;
const OS_CLOCK_MACH_ABSOLUTE_TIME: u32 = 32;

extern "C" {
    fn os_sync_wait_on_address_with_timeout(
        addr: *mut libc::c_void,
        value: u64,
        size: usize,
        flags: u32,
        clockid: u32,
        timeout_ns: u64,
    ) -> libc::c_int;
    fn os_sync_wake_by_address_all(addr: *mut libc::c_void, size: usize, flags: u32) -> libc::c_int;
}

/// Block while `*a == expected`, for at most `timeout_ns`. Spurious wakeups possible.
pub fn wait(a: &AtomicU32, expected: u32, timeout_ns: u64) {
    if a.load(Ordering::Acquire) != expected {
        return;
    }
    unsafe {
        os_sync_wait_on_address_with_timeout(
            a.as_ptr() as *mut _,
            expected as u64,
            4,
            OS_SYNC_WAIT_ON_ADDRESS_SHARED,
            OS_CLOCK_MACH_ABSOLUTE_TIME,
            timeout_ns.max(1),
        );
    }
}

pub fn wake_all(a: &AtomicU32) {
    unsafe {
        os_sync_wake_by_address_all(a.as_ptr() as *mut _, 4, OS_SYNC_WAKE_BY_ADDRESS_SHARED);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    fn wake_unblocks_waiter() {
        let a = Arc::new(AtomicU32::new(0));
        let b = a.clone();
        let t = std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            while b.load(Ordering::Acquire) == 0 {
                wait(&b, 0, 5_000_000_000);
            }
            t0.elapsed()
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        a.store(1, Ordering::Release);
        wake_all(&a);
        let el = t.join().unwrap();
        assert!(el < std::time::Duration::from_secs(1), "{el:?}");
    }
}
