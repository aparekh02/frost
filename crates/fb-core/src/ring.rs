//! Lock-free SPSC ring living in shared memory.
//!
//! Layout (all offsets relative to the ring base, which must be 128-byte aligned —
//! Apple Silicon cache lines are 128 bytes, so head and tail get their own line to
//! avoid false sharing between the producer and consumer cores):
//!
//! ```text
//!   [0   ..128)  tail  (written by producer)
//!   [128 ..256)  head  (written by consumer)
//!   [256 ..384)  capacity, slot size, magic (immutable after init)
//!   [384 ..   )  slots[capacity]
//! ```
//!
//! Capacity must be a power of two. Indices are free-running u64s.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

pub const CACHE_LINE: usize = 128;
const RING_MAGIC: u64 = 0x4642_5249_4e47_0001; // "FBRING" v1

#[repr(C, align(128))]
struct Line(AtomicU64);

#[repr(C)]
struct RingHeader {
    tail: Line,
    head: Line,
    meta: RingMeta,
}

#[repr(C, align(128))]
struct RingMeta {
    capacity: u64,
    slot_size: u64,
    magic: u64,
}

pub const fn ring_bytes<T>(capacity: usize) -> usize {
    std::mem::size_of::<RingHeader>() + capacity * std::mem::size_of::<T>()
}

/// A handle onto a ring in shared memory. Use [`Producer`]/[`Consumer`] to get the
/// role-specific API; the type system can't stop two processes from both producing,
/// so the protocol (daemon consumes submit / produces complete) is the contract.
pub struct Ring<T: Copy> {
    hdr: *const RingHeader,
    slots: *mut T,
    mask: u64,
    _p: PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for Ring<T> {}

impl<T: Copy> Ring<T> {
    /// Initialise a fresh ring at `base`.
    ///
    /// # Safety
    /// `base` must be 128-byte aligned and valid for `ring_bytes::<T>(capacity)` bytes.
    pub unsafe fn init(base: *mut u8, capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        assert_eq!(base as usize % CACHE_LINE, 0);
        let hdr = base as *mut RingHeader;
        std::ptr::write_bytes(base, 0, std::mem::size_of::<RingHeader>());
        (*hdr).meta.capacity = capacity as u64;
        (*hdr).meta.slot_size = std::mem::size_of::<T>() as u64;
        std::sync::atomic::fence(Ordering::Release);
        (*hdr).meta.magic = RING_MAGIC;
        Self::attach(base).expect("fresh ring must attach")
    }

    /// Attach to a ring that another process initialised.
    ///
    /// # Safety
    /// `base` must point at a ring created by [`Ring::init`] that stays mapped.
    pub unsafe fn attach(base: *mut u8) -> Option<Self> {
        let hdr = base as *const RingHeader;
        if (*hdr).meta.magic != RING_MAGIC || (*hdr).meta.slot_size != std::mem::size_of::<T>() as u64 {
            return None;
        }
        let cap = (*hdr).meta.capacity;
        Some(Ring {
            hdr,
            slots: base.add(std::mem::size_of::<RingHeader>()) as *mut T,
            mask: cap - 1,
            _p: PhantomData,
        })
    }

    pub fn capacity(&self) -> usize {
        (self.mask + 1) as usize
    }
    fn tail(&self) -> &AtomicU64 {
        unsafe { &(*self.hdr).tail.0 }
    }
    fn head(&self) -> &AtomicU64 {
        unsafe { &(*self.hdr).head.0 }
    }

    pub fn producer(self) -> Producer<T> {
        let tail = self.tail().load(Ordering::Relaxed);
        let head = self.head().load(Ordering::Acquire);
        Producer { r: self, tail, head_cache: head }
    }
    pub fn consumer(self) -> Consumer<T> {
        let head = self.head().load(Ordering::Relaxed);
        let tail = self.tail().load(Ordering::Acquire);
        Consumer { r: self, head, tail_cache: tail }
    }
}

pub struct Producer<T: Copy> {
    r: Ring<T>,
    tail: u64,
    head_cache: u64,
}

impl<T: Copy> Producer<T> {
    /// Non-blocking push. Returns the item back if the ring is full.
    #[inline]
    pub fn push(&mut self, v: T) -> Result<(), T> {
        // wrapping_sub + `>=`: if the (untrusted) peer corrupts `head`, the ring simply
        // looks full instead of letting us overwrite unread slots or panic.
        let cap = self.r.mask + 1;
        if self.tail.wrapping_sub(self.head_cache) >= cap {
            self.head_cache = self.r.head().load(Ordering::Acquire);
            if self.tail.wrapping_sub(self.head_cache) >= cap {
                return Err(v);
            }
        }
        unsafe { self.r.slots.add((self.tail & self.r.mask) as usize).write_volatile(v) };
        self.tail += 1;
        self.r.tail().store(self.tail, Ordering::Release);
        Ok(())
    }

    pub fn len(&self) -> usize {
        (self.tail.wrapping_sub(self.r.head().load(Ordering::Acquire)) as usize).min(self.r.capacity())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Consumer<T: Copy> {
    r: Ring<T>,
    head: u64,
    tail_cache: u64,
}

impl<T: Copy> Consumer<T> {
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.head == self.tail_cache {
            self.tail_cache = self.r.tail().load(Ordering::Acquire);
            if self.head == self.tail_cache {
                return None;
            }
        }
        // A peer-corrupted tail (behind us, or more than a ring ahead) yields nothing.
        if self.tail_cache.wrapping_sub(self.head) > self.r.mask + 1 {
            self.tail_cache = self.head;
            return None;
        }
        let v = unsafe { self.r.slots.add((self.head & self.r.mask) as usize).read_volatile() };
        self.head += 1;
        self.r.head().store(self.head, Ordering::Release);
        Some(v)
    }

    /// Number of items currently visible.
    pub fn available(&mut self) -> usize {
        self.tail_cache = self.r.tail().load(Ordering::Acquire);
        let n = self.tail_cache.wrapping_sub(self.head);
        if n > self.r.mask + 1 {
            self.tail_cache = self.head;
            return 0;
        }
        n as usize
    }

    /// True if the peer has written an impossible tail index.
    pub fn corrupt(&self) -> bool {
        self.r.tail().load(Ordering::Acquire).wrapping_sub(self.head) > self.r.mask + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(128))]
    struct Buf([u8; 4096 * 16]);

    #[test]
    fn spsc_threads_in_order() {
        let mut buf = Box::new(Buf([0; 4096 * 16]));
        let base = buf.0.as_mut_ptr();
        let r = unsafe { Ring::<u64>::init(base, 1024) };
        let mut p = r.producer();
        let addr = base as usize;
        const N: u64 = 2_000_000;
        let t = std::thread::spawn(move || {
            let mut c = unsafe { Ring::<u64>::attach(addr as *mut u8).unwrap() }.consumer();
            let mut expect = 0;
            while expect < N {
                if let Some(v) = c.pop() {
                    assert_eq!(v, expect);
                    expect += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
        for i in 0..N {
            let mut v = i;
            while let Err(back) = p.push(v) {
                v = back;
                std::hint::spin_loop();
            }
        }
        t.join().unwrap();
        drop(buf);
    }

    #[test]
    fn corrupted_indices_are_contained() {
        let mut buf = Box::new(Buf([0; 4096 * 16]));
        let base = buf.0.as_mut_ptr();
        let r = unsafe { Ring::<u32>::init(base, 8) };
        let mut c = r.consumer();
        let mut p = unsafe { Ring::<u32>::attach(base).unwrap() }.producer();
        p.push(1).unwrap();
        // peer scribbles tail far ahead / behind
        let hdr = base as *mut RingHeader;
        unsafe { (*hdr).tail.0.store(u64::MAX - 3, Ordering::Release) };
        assert!(c.corrupt());
        assert_eq!(c.available(), 0);
        assert_eq!(c.pop(), None);
        // producer facing a corrupted head: reports full, never panics
        unsafe { (*hdr).head.0.store(1 << 40, Ordering::Release) };
        let mut p2 = unsafe { Ring::<u32>::attach(base).unwrap() }.producer();
        for _ in 0..20 { let _ = p2.push(7); }
        assert!(p2.len() <= 8);
    }

    #[test]
    fn full_and_empty() {
        let mut buf = Box::new(Buf([0; 4096 * 16]));
        let r = unsafe { Ring::<u32>::init(buf.0.as_mut_ptr(), 4) };
        let mut p = r.producer();
        for i in 0..4 {
            p.push(i).unwrap();
        }
        assert_eq!(p.push(9), Err(9));
        let mut c = unsafe { Ring::<u32>::attach(buf.0.as_mut_ptr()).unwrap() }.consumer();
        assert_eq!(c.pop(), Some(0));
        p.push(4).unwrap();
        assert_eq!((1..=4).map(|_| c.pop().unwrap()).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(c.pop(), None);
    }
}
