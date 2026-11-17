//! fb-client: the library a capsule links to talk to `frostd`.
//!
//! ```no_run
//! let mut fb = fb_client::Frost::connect()?;
//! let x = fb.alloc::<f32>(1024)?;
//! fb.view_mut::<f32>(x).fill(1.0);
//! let c = fb.run(fb.job("saxpy_f32").arg(x).arg(x).grid(1024).param(1024).param_f32(2.0))?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Hot path cost: one 168-byte store into the submit ring + one atomic increment.
//! No syscalls unless the client chooses to park ([`WaitMode::Park`]).

use anyhow::{anyhow, bail, Context, Result};
use fb_core::abi::{ArenaHeader, BellPage, BufRef, Completion, Job, Status, JOB_FLAG_CHAIN, MAX_ARGS, MAX_PARAMS};
use fb_core::manifest::Realtime;
use fb_core::proto::{self, ToCapsule, ToDaemon};
use fb_core::ring::{Consumer, Producer, Ring};
use fb_core::shm::Arena;
use std::collections::{BTreeMap, HashMap};
use std::io::BufReader;
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitMode {
    /// Busy-poll the completion ring. Lowest latency.
    Spin,
    /// Spin briefly, then sleep on the completion doorbell (os_sync_wait_on_address).
    Park { spin_ns: u64 },
}

pub struct Frost {
    ctl_r: BufReader<UnixStream>,
    ctl_w: UnixStream,
    arena: Arena,
    bell: Option<Arena>,
    submit: Producer<Job>,
    complete: Consumer<Completion>,
    data_off: u64,
    data_len: u64,
    cursor: u64,
    seq: u64,
    kernels: BTreeMap<String, u32>,
    /// Completions that arrived while waiting for a different seq.
    early: HashMap<u64, Completion>,
    pub capsule: String,
    pub gpu_name: String,
    pub realtime: Option<Realtime>,
    pub wait_mode: WaitMode,
}

impl Frost {
    /// Connect using the fds frostd handed down (FROST_CTL_FD / FROST_ARENA_FD).
    pub fn connect() -> Result<Frost> {
        let fd = |k: &str| -> Result<i32> {
            std::env::var(k).with_context(|| format!("{k} not set — not running under frostd?"))?.parse().context(k.to_string())
        };
        let ctl = unsafe { UnixStream::from_raw_fd(fd(proto::ENV_CTL_FD)?) };
        let arena = Arena::from_fd(fd(proto::ENV_ARENA_FD)?)?;
        let bell = match fd(proto::ENV_BELL_FD) {
            Ok(b) => Some(Arena::from_fd(b)?),
            Err(_) => None,
        };
        let mut ctl_w = ctl.try_clone()?;
        let mut ctl_r = BufReader::new(ctl);
        proto::send(&mut ctl_w, &ToDaemon::Hello { pid: std::process::id() as i32 })?;
        let welcome: ToCapsule = proto::recv(&mut ctl_r)?.ok_or_else(|| anyhow!("daemon closed"))?;
        let ToCapsule::Welcome { capsule, gpu_name, kernels, realtime, data_off, data_len } = welcome else {
            bail!("unexpected first message: {welcome:?}");
        };
        let hdr = unsafe { &*(arena.as_ptr() as *const ArenaHeader) };
        anyhow::ensure!(hdr.magic == fb_core::abi::ARENA_MAGIC, "bad arena magic");
        anyhow::ensure!(hdr.abi_version == fb_core::abi::ABI_VERSION, "ABI mismatch");
        let base = arena.as_ptr();
        let submit = unsafe { Ring::<Job>::attach(base.add(hdr.submit_off as usize)) }
            .ok_or_else(|| anyhow!("submit ring attach"))?
            .producer();
        let complete = unsafe { Ring::<Completion>::attach(base.add(hdr.complete_off as usize)) }
            .ok_or_else(|| anyhow!("complete ring attach"))?
            .consumer();
        Ok(Frost {
            ctl_r,
            ctl_w,
            arena,
            bell,
            submit,
            complete,
            data_off,
            data_len,
            cursor: data_off,
            seq: 0,
            kernels,
            early: HashMap::new(),
            capsule,
            gpu_name,
            realtime,
            wait_mode: WaitMode::Spin,
        })
    }

    fn hdr(&self) -> &ArenaHeader {
        unsafe { &*(self.arena.as_ptr() as *const ArenaHeader) }
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }
    pub fn data_capacity(&self) -> u64 {
        self.data_len
    }
    pub fn data_remaining(&self) -> u64 {
        self.data_off + self.data_len - self.cursor
    }

    /// Put the calling thread into the RT band using the manifest's declared budget.
    pub fn enter_realtime(&self) -> Result<()> {
        let rt = self.realtime.ok_or_else(|| anyhow!("manifest has no cpu.realtime"))?;
        fb_core::sched::set_realtime(rt.period_us * 1000, rt.computation_us * 1000, rt.constraint_us * 1000)
    }

    /// Bump-allocate `count` elements of T in the GPU-visible arena (256-byte aligned).
    pub fn alloc<T>(&mut self, count: usize) -> Result<BufRef> {
        let bytes = (count * std::mem::size_of::<T>()) as u64;
        let off = (self.cursor + 255) & !255;
        if off + bytes > self.data_off + self.data_len {
            bail!("arena exhausted: need {bytes} B, {} B left", self.data_remaining());
        }
        self.cursor = off + bytes;
        Ok(BufRef { off, len: bytes })
    }

    /// Release every allocation (bump allocator reset).
    pub fn reset_alloc(&mut self) {
        self.cursor = self.data_off;
    }

    #[allow(clippy::mut_from_ref)]
    pub fn view_mut<T: Copy>(&self, r: BufRef) -> &mut [T] {
        self.arena.typed::<T>(r.off as usize, r.len as usize / std::mem::size_of::<T>())
    }

    pub fn kernel_id(&self, name: &str) -> Option<u32> {
        self.kernels.get(name).copied()
    }
    pub fn kernels(&self) -> &BTreeMap<String, u32> {
        &self.kernels
    }

    /// Ask the daemon to compile a custom kernel (requires gpu.custom_kernels = true).
    pub fn register_kernel(&mut self, name: &str, source: &str, entry: &str) -> Result<u32> {
        proto::send(
            &mut self.ctl_w,
            &ToDaemon::RegisterKernel { name: name.into(), source: source.into(), entry: entry.into() },
        )?;
        match proto::recv::<_, ToCapsule>(&mut self.ctl_r)? {
            Some(ToCapsule::KernelRegistered { id, name }) => {
                self.kernels.insert(name, id);
                Ok(id)
            }
            Some(ToCapsule::Error { msg }) => bail!("register_kernel: {msg}"),
            other => bail!("register_kernel: unexpected {other:?}"),
        }
    }

    pub fn stats(&mut self) -> Result<ToCapsule> {
        proto::send(&mut self.ctl_w, &ToDaemon::Stats)?;
        proto::recv(&mut self.ctl_r)?.ok_or_else(|| anyhow!("daemon closed"))
    }

    pub fn job(&self, kernel: &str) -> JobBuilder {
        JobBuilder { j: Job { kernel: self.kernel_id(kernel).unwrap_or(u32::MAX), ..Default::default() }, np: 0 }
    }

    /// Enqueue a job; returns its sequence number. Spins if the ring is full.
    #[inline]
    pub fn submit(&mut self, b: JobBuilder) -> u64 {
        self.submit_raw(b.j, false)
    }

    /// Enqueue several jobs that the daemon will encode into one command buffer.
    pub fn submit_chain(&mut self, jobs: Vec<JobBuilder>) -> Vec<u64> {
        let n = jobs.len();
        jobs.into_iter().enumerate().map(|(i, b)| self.submit_raw(b.j, i + 1 < n)).collect()
    }

    fn submit_raw(&mut self, mut j: Job, chain: bool) -> u64 {
        self.seq += 1;
        j.seq = self.seq;
        if chain {
            j.flags |= JOB_FLAG_CHAIN;
        }
        let mut v = j;
        while let Err(back) = self.submit.push(v) {
            v = back;
            std::hint::spin_loop();
        }
        // Doorbell is only for an idle daemon; it never blocks the client.
        if let Some(b) = &self.bell {
            let b = unsafe { &*(b.as_ptr() as *const BellPage) };
            b.bell.fetch_add(1, Ordering::Release);
            if b.waiters.load(Ordering::Acquire) != 0 {
                fb_core::wait::wake_all(&b.bell);
            }
        }
        self.seq
    }

    /// Wait for completion of `seq`.
    pub fn wait(&mut self, seq: u64) -> Result<Completion> {
        if let Some(c) = self.early.remove(&seq) {
            return check(c);
        }
        let t0 = fb_core::time::now_ns();
        loop {
            if let Some(c) = self.complete.pop() {
                if c.seq == seq {
                    return check(c);
                }
                self.early.insert(c.seq, c);
                continue;
            }
            if self.hdr().closed.load(Ordering::Acquire) != 0 {
                bail!("daemon closed the GPU channel");
            }
            match self.wait_mode {
                WaitMode::Spin => std::hint::spin_loop(),
                WaitMode::Park { spin_ns } => {
                    if fb_core::time::now_ns() - t0 < spin_ns {
                        std::hint::spin_loop();
                        continue;
                    }
                    // Raw deref: the header lives as long as the arena mapping.
                    let h = unsafe { &*(self.arena.as_ptr() as *const ArenaHeader) };
                    let bell = h.complete_bell.load(Ordering::Acquire);
                    h.complete_waiters.fetch_add(1, Ordering::AcqRel);
                    if self.complete.available() == 0 {
                        fb_core::wait::wait(&h.complete_bell, bell, 50_000_000);
                    }
                    h.complete_waiters.fetch_sub(1, Ordering::AcqRel);
                }
            }
        }
    }

    /// Submit and wait.
    #[inline]
    pub fn run(&mut self, b: JobBuilder) -> Result<Completion> {
        let s = self.submit(b);
        self.wait(s)
    }
}

fn check(c: Completion) -> Result<Completion> {
    match Status::from_u32(c.status) {
        Status::Ok => Ok(c),
        s => Err(anyhow!("job {} failed: {s:?}", c.seq)),
    }
}

impl Drop for Frost {
    fn drop(&mut self) {
        let _ = proto::send(&mut self.ctl_w, &ToDaemon::Bye);
    }
}

#[derive(Clone, Copy)]
pub struct JobBuilder {
    j: Job,
    np: usize,
}

impl JobBuilder {
    pub fn arg(mut self, r: BufRef) -> Self {
        assert!((self.j.n_args as usize) < MAX_ARGS, "too many args");
        self.j.args[self.j.n_args as usize] = r;
        self.j.n_args += 1;
        self
    }
    pub fn param(mut self, v: u32) -> Self {
        assert!(self.np < MAX_PARAMS);
        self.j.params[self.np] = v;
        self.np += 1;
        self
    }
    pub fn param_f32(self, v: f32) -> Self {
        self.param(v.to_bits())
    }
    pub fn grid(mut self, x: u32) -> Self {
        self.j.grid = [x, 1, 1];
        self
    }
    pub fn grid3(mut self, g: [u32; 3]) -> Self {
        self.j.grid = g;
        self
    }
    pub fn threadgroup(mut self, x: u32) -> Self {
        self.j.tg = [x, 1, 1];
        self
    }
    pub fn raw(&self) -> &Job {
        &self.j
    }
}
