//! File-backed shared memory.
//!
//! The daemon creates the backing file, `ftruncate`s it and maps it `MAP_SHARED`.
//! The fd is then inherited by the capsule process (no filesystem rights needed in
//! the sandbox) and mapped again there. Because the mapping is page aligned and a
//! multiple of the page size, Metal can wrap it zero-copy with
//! `newBufferWithBytesNoCopy`, so CPU (in either process) and GPU all see the same
//! physical pages of unified memory.

use anyhow::{bail, Context, Result};
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;

pub struct Arena {
    ptr: *mut u8,
    len: usize,
    fd: OwnedFd,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Create (or truncate) a backing file and map it. `len` is rounded up to the page size.
    pub fn create(path: &Path, len: usize) -> Result<Arena> {
        let len = crate::round_up(len.max(1), crate::page_size());
        let c = CString::new(path.as_os_str().as_encoded_bytes())?;
        let fd = unsafe {
            libc::open(
                c.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            bail!("open {}: {}", path.display(), std::io::Error::last_os_error());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            bail!("ftruncate: {}", std::io::Error::last_os_error());
        }
        Self::map(fd, len)
    }

    /// Map an already-open (e.g. inherited) fd. Length is taken from fstat.
    pub fn from_fd(fd: RawFd) -> Result<Arena> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            bail!("fstat arena fd {fd}: {}", std::io::Error::last_os_error());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        Self::map(fd, st.st_size as usize)
    }

    fn map(fd: OwnedFd, len: usize) -> Result<Arena> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error()).context("mmap arena");
        }
        Ok(Arena { ptr: ptr as *mut u8, len, fd })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Wire the pages so they never page out / compress (needs memory headroom,
    /// subject to RLIMIT_MEMLOCK).
    pub fn lock(&self) -> Result<()> {
        if unsafe { libc::mlock(self.ptr as *const _, self.len) } != 0 {
            bail!("mlock: {}", std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Touch every page so first-use page faults don't land on the hot path.
    pub fn prefault(&self) {
        let ps = crate::page_size();
        let mut off = 0;
        while off < self.len {
            unsafe {
                let p = self.ptr.add(off);
                std::ptr::write_volatile(p, std::ptr::read_volatile(p));
            }
            off += ps;
        }
    }

    /// Byte slice view of `[off, off+len)`; bounds checked. Shared memory is mutable by
    /// construction (other processes and the GPU write it), hence `&self -> &mut`.
    #[allow(clippy::mut_from_ref)]
    pub fn slice(&self, off: usize, len: usize) -> Option<&mut [u8]> {
        if off.checked_add(len)? > self.len {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), len) })
    }

    /// Typed view; panics on misalignment or out-of-bounds.
    #[allow(clippy::mut_from_ref)]
    pub fn typed<T: Copy>(&self, off: usize, count: usize) -> &mut [T] {
        let bytes = count * std::mem::size_of::<T>();
        assert!(off + bytes <= self.len, "typed view out of bounds");
        assert!(off.is_multiple_of(std::mem::align_of::<T>()), "typed view misaligned");
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut T, count) }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}

/// Clear FD_CLOEXEC so a spawned child inherits this fd.
pub fn set_inheritable(fd: RawFd, inherit: bool) -> Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        bail!("fcntl F_GETFD: {}", std::io::Error::last_os_error());
    }
    let nf = if inherit { flags & !libc::FD_CLOEXEC } else { flags | libc::FD_CLOEXEC };
    if unsafe { libc::fcntl(fd, libc::F_SETFD, nf) } < 0 {
        bail!("fcntl F_SETFD: {}", std::io::Error::last_os_error());
    }
    Ok(())
}
