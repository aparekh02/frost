//! The frostd supervisor: owns the GPU, launches capsules, runs the dispatcher.
//!
//! Threads:
//! * **dispatcher** (1, QoS user-interactive, optionally RT): polls every capsule's submit
//!   ring in priority order, encodes jobs into per-capsule command queues, polls in-flight
//!   command buffers and writes completions. Spins while busy, parks on the bell page
//!   when idle.
//! * **control** (1 per capsule): JSON-lines socket (hello, kernel registration, stats).
//! * **reaper** (1 per capsule): waits for exit, records status.
//! * **watchdog** (1): phys_footprint enforcement.
//! * **log pumps** (2 per capsule): prefix capsule stdout/stderr.

use anyhow::{bail, Context, Result};
use fb_core::abi::{self, ArenaHeader, BellPage, Completion, Job, Layout, Status, JOB_FLAG_CHAIN};
use fb_core::manifest::Manifest;
use fb_core::proto::{self, ToCapsule, ToDaemon};
use fb_core::ring::{Consumer, Producer, Ring};
use fb_core::shm::Arena;
use fb_gpu::{Gpu, GpuBuffer, GpuQueue, InFlight, RenderTarget, Window};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Config {
    pub run_dir: PathBuf,
    /// Keep spinning this long after the last activity before parking.
    pub idle_spin_us: u64,
    /// Max jobs encoded into one command buffer.
    pub max_batch: usize,
    /// Put the dispatcher thread in the real-time band.
    pub realtime_dispatcher: bool,
    /// Fraction of the GPU's recommended working set that capsule arenas may use.
    pub working_set_fraction: f64,
    pub quiet: bool,
    /// Poll phys_footprint and SIGKILL capsules over their limit (backs up jetsam).
    pub watchdog: bool,
    /// While capsules are attached, submit a tiny GPU pulse this often (0 = off) so the
    /// GPU doesn't drop into a low power state between sparse latency-critical jobs.
    pub keepwarm_us: u64,
    /// Preemption by slicing: while a higher-priority capsule is active, splittable jobs
    /// from lower-priority capsules are cut into chunks of about this much GPU time.
    pub slice_us: u64,
    /// A capsule counts as "active" for this long after its last submission.
    pub active_window_us: u64,
}

impl Default for Config {
    fn default() -> Self {
        let uid = unsafe { libc::getuid() };
        Config {
            run_dir: PathBuf::from(format!("/private/tmp/frost-{uid}")),
            idle_spin_us: 2000,
            max_batch: 64,
            realtime_dispatcher: false,
            working_set_fraction: 0.6,
            quiet: false,
            watchdog: true,
            keepwarm_us: 0,
            slice_us: 100,
            active_window_us: 20_000,
        }
    }
}

#[derive(Default, Debug, Serialize, Clone)]
pub struct Report {
    pub name: String,
    pub pid: i32,
    /// PID of the broker process whose GPU address space served this capsule. Capsules with
    /// different `broker_pid` are in different GPU VA spaces (separate trust domains).
    pub broker_pid: i32,
    pub exit: Option<i32>,
    pub signal: Option<i32>,
    pub killed_for: Option<String>,
    pub jobs: u64,
    pub rejected: u64,
    pub gpu_ms: f64,
    /// Live GPU utilization over the last window (percent of the whole device).
    pub gpu_pct: f64,
    /// The capsule's GPU time cap, as a percent of the device (100 = uncapped).
    pub gpu_cap_pct: f64,
    /// GPU memory this capsule may use (its arena) in MiB — a hard per-capsule cap.
    pub gpu_mem_mb: f64,
    pub throttled_ms: f64,
    pub peak_footprint_mb: f64,
    pub cpu_ms: f64,
    pub wall_ms: f64,
    /// Spawn → capsule's Hello on the control socket (sandbox + dyld + connect).
    pub startup_ms: Option<f64>,
    pub alive: bool,
}

/// State shared between the dispatcher, control, reaper and watchdog threads.
pub struct Slot {
    pub name: String,
    pub manifest: Manifest,
    pub pid: AtomicI32,
    pub alive: AtomicBool,
    pub jobs: AtomicU64,
    pub rejected: AtomicU64,
    pub gpu_ns: AtomicU64,
    pub throttled: AtomicU64,
    pub peak_footprint: AtomicU64,
    pub cpu_ns: AtomicU64,
    pub hello_ns: AtomicU64,
    /// Live GPU utilization in per-mille of the device (0..1000), last window.
    pub util_pm: AtomicU64,
    pub killed_for: Mutex<Option<String>>,
    pub exit: Mutex<Option<std::process::ExitStatus>>,
    pub started: std::time::Instant,
    pub ended: Mutex<Option<std::time::Instant>>,
    arena: Arena,
    layout: Layout,
}

impl Slot {
    fn hdr(&self) -> &ArenaHeader {
        unsafe { &*(self.arena.as_ptr() as *const ArenaHeader) }
    }
    pub fn report(&self) -> Report {
        let st = *self.exit.lock().unwrap();
        use std::os::unix::process::ExitStatusExt;
        let end = self.ended.lock().unwrap().unwrap_or_else(std::time::Instant::now);
        Report {
            name: self.name.clone(),
            pid: self.pid.load(Ordering::Relaxed),
            broker_pid: std::process::id() as i32,
            exit: st.and_then(|s| s.code()),
            signal: st.and_then(|s| s.signal()),
            killed_for: self.killed_for.lock().unwrap().clone(),
            jobs: self.jobs.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            gpu_ms: self.gpu_ns.load(Ordering::Relaxed) as f64 / 1e6,
            gpu_pct: self.util_pm.load(Ordering::Relaxed) as f64 / 10.0,
            gpu_cap_pct: self.manifest.gpu.budget_ms_per_sec() as f64 / 10.0,
            gpu_mem_mb: self.manifest.gpu.arena_mb as f64,
            throttled_ms: self.throttled.load(Ordering::Relaxed) as f64 / 1e6,
            peak_footprint_mb: self.peak_footprint.load(Ordering::Relaxed) as f64 / 1048576.0,
            cpu_ms: self.cpu_ns.load(Ordering::Relaxed) as f64 / 1e6,
            wall_ms: end.duration_since(self.started).as_secs_f64() * 1e3,
            startup_ms: match self.hello_ns.load(Ordering::Relaxed) {
                0 => None,
                n => Some(n as f64 / 1e6),
            },
            alive: self.alive.load(Ordering::Relaxed),
        }
    }
}

/// Dispatcher-private view of a capsule.
// Field order matters for Drop: the GPU buffer and rings must go before `slot`, which
// owns the arena mapping they point into.
struct Lane {
    submit: Consumer<Job>,
    complete: Producer<Completion>,
    buf: GpuBuffer,
    queue: GpuQueue,
    window: Window,
    inflight: VecDeque<Pending>,
    /// Completions that could not be pushed because the ring was full.
    backlog: VecDeque<Completion>,
    tokens_ns: i64,
    budget_ns_per_s: i64,
    /// Bucket depth: how much GPU time may be spent in a burst.
    burst_ns: i64,
    throttled_since: Option<u64>,
    /// Chunks of split jobs waiting to be issued: (job, is_final_chunk, items).
    deferred: VecDeque<(Job, bool, u32)>,
    /// Observed GPU ns per work item, per kernel (EWMA) — sizes future slices.
    ns_per_item: std::collections::HashMap<u32, f64>,
    last_submit_ns: u64,
    last_refill: u64,
    priority: i32,
    /// Max command buffers this capsule may have on the GPU at once (occupancy cap).
    max_inflight: usize,
    /// Rolling GPU-utilization window: busy ns accumulated since `util_win_start`.
    util_win_start: u64,
    util_busy_ns: u64,
    /// Cached render target for the graphics plane (reused across frames of one size).
    rt: Option<RenderTarget>,
    slot: Arc<Slot>,
}

struct Pending {
    cb: InFlight,
    /// (seq, kernel, emits_completion, items) — non-final chunks of a split job don't.
    jobs: Vec<(u64, u32, bool, u32)>,
    t_seen: u64,
}

fn work_items(j: &Job) -> u32 {
    use abi::kernel::*;
    match j.kernel {
        VADD_F32 | SAXPY_F32 | SHA256_BATCH | GEMV_F32 => j.params[0],
        _ => 0,
    }
}

pub struct Daemon {
    pub gpu: Arc<Gpu>,
    cfg: Config,
    bell: Arc<Arena>,
    lanes_tx: mpsc::Sender<Lane>,
    slots: Arc<Mutex<Vec<Arc<Slot>>>>,
    arena_bytes: AtomicU64,
    stop: Arc<AtomicBool>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Daemon {
    pub fn start(cfg: Config) -> Result<Arc<Daemon>> {
        std::fs::create_dir_all(&cfg.run_dir).with_context(|| format!("mkdir {}", cfg.run_dir.display()))?;
        let gpu = Arc::new(Gpu::new()?);
        let bell_path = cfg.run_dir.join(format!("bell-{}", std::process::id()));
        let bell = Arc::new(Arena::create(&bell_path, fb_core::page_size())?);
        std::fs::remove_file(&bell_path).ok(); // fd-only from now on
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let d = Arc::new(Daemon {
            gpu: gpu.clone(),
            cfg: cfg.clone(),
            bell: bell.clone(),
            lanes_tx: tx,
            slots: Arc::new(Mutex::new(Vec::new())),
            arena_bytes: AtomicU64::new(0),
            stop: stop.clone(),
            threads: Mutex::new(Vec::new()),
        });
        let disp = {
            let gpu = gpu.clone();
            let cfg = cfg.clone();
            std::thread::Builder::new()
                .name("fb-dispatch".into())
                .spawn(move || dispatcher(gpu, cfg, bell, rx, stop))?
        };
        let wd = {
            let slots = d.slots.clone();
            let stop = d.stop.clone();
            let enforce = cfg.watchdog;
            std::thread::Builder::new().name("fb-watchdog".into()).spawn(move || watchdog(slots, stop, enforce))?
        };
        d.threads.lock().unwrap().extend([disp, wd]);
        Ok(d)
    }

    pub fn slots(&self) -> Vec<Arc<Slot>> {
        self.slots.lock().unwrap().clone()
    }

    /// Launch a capsule from a manifest. `base` resolves a relative `exec`.
    pub fn launch(self: &Arc<Self>, m: Manifest, base: &Path) -> Result<Arc<Slot>> {
        let exec = {
            let p = Path::new(&m.exec);
            let p = if p.is_absolute() { p.to_path_buf() } else { base.join(p) };
            std::fs::canonicalize(&p).with_context(|| format!("exec {} not found", p.display()))?
        };
        // ---- memory quota (unified memory: arenas compete with everything else)
        let layout = Layout::for_data((m.gpu.arena_mb as usize) << 20);
        let cap = (self.gpu.recommended_working_set() as f64 * self.cfg.working_set_fraction) as u64;
        let now = self.arena_bytes.fetch_add(layout.arena_len as u64, Ordering::AcqRel);
        if now + layout.arena_len as u64 > cap {
            self.arena_bytes.fetch_sub(layout.arena_len as u64, Ordering::AcqRel);
            bail!(
                "capsule {}: arena {} MiB would exceed GPU arena quota ({} of {} MiB used)",
                m.name,
                layout.arena_len >> 20,
                now >> 20,
                cap >> 20
            );
        }
        let dir = self.cfg.run_dir.join(format!("{}-{}", m.name, fb_core::time::ticks()));
        let root = dir.join("root");
        std::fs::create_dir_all(&root)?;

        // ---- arena: create, lay out, unlink (fd-only), wire, prefault
        let apath = dir.join("arena");
        let arena = Arena::create(&apath, layout.arena_len)?;
        std::fs::remove_file(&apath).ok();
        unsafe {
            let base = arena.as_ptr();
            std::ptr::write_bytes(base, 0, layout.data_off);
            let h = &mut *(base as *mut ArenaHeader);
            h.abi_version = abi::ABI_VERSION;
            h.arena_len = layout.arena_len as u64;
            h.submit_off = layout.submit_off as u64;
            h.complete_off = layout.complete_off as u64;
            h.data_off = layout.data_off as u64;
            h.data_len = (layout.arena_len - layout.data_off) as u64;
            Ring::<Job>::init(base.add(layout.submit_off), abi::SUBMIT_CAP);
            Ring::<Completion>::init(base.add(layout.complete_off), abi::COMPLETE_CAP);
            std::sync::atomic::fence(Ordering::Release);
            h.magic = abi::ARENA_MAGIC;
        }
        arena.prefault();
        if m.memory.wire_arena {
            if let Err(e) = arena.lock() {
                self.log(&m.name, &format!("warning: could not wire arena: {e}"));
            }
        }
        let gbuf = unsafe { self.gpu.wrap(arena.as_ptr(), arena.len())? };
        let queue = self.gpu.new_queue()?;

        // ---- control socket
        let (ours, theirs) = UnixStream::pair()?;

        let slot = Arc::new(Slot {
            name: m.name.clone(),
            manifest: m.clone(),
            pid: AtomicI32::new(0),
            alive: AtomicBool::new(true),
            jobs: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            gpu_ns: AtomicU64::new(0),
            throttled: AtomicU64::new(0),
            peak_footprint: AtomicU64::new(0),
            cpu_ns: AtomicU64::new(0),
            hello_ns: AtomicU64::new(0),
            util_pm: AtomicU64::new(0),
            killed_for: Mutex::new(None),
            exit: Mutex::new(None),
            started: std::time::Instant::now(),
            ended: Mutex::new(None),
            arena,
            layout,
        });

        let env = vec![
            (proto::ENV_CTL_FD.to_string(), theirs.as_raw_fd().to_string()),
            (proto::ENV_ARENA_FD.to_string(), slot.arena.fd().to_string()),
            (proto::ENV_BELL_FD.to_string(), self.bell.fd().to_string()),
            (proto::ENV_CAPSULE.to_string(), m.name.clone()),
            (proto::ENV_ROOT.to_string(), root.display().to_string()),
        ];
        let mut child = fb_capsule::spawn(fb_capsule::SpawnSpec {
            manifest: &m,
            exec,
            root: root.clone(),
            inherit: vec![theirs.as_raw_fd(), slot.arena.fd(), self.bell.fd()],
            env,
            stdout: Stdio::piped(),
            stderr: Stdio::piped(),
        })?;
        drop(theirs);
        let pid = child.id() as i32;
        slot.pid.store(pid, Ordering::Release);
        self.log(&m.name, &format!("started pid={pid} qos={:?} arena={}MiB prio={} gpu_cap={}% max_inflight={}", fb_capsule::qos_for(m.cpu.class), layout.arena_len >> 20, m.gpu.priority, m.gpu.budget_ms_per_sec() / 10, m.gpu.max_inflight));

        // ---- threads: log pumps, reaper, control
        let mut th = Vec::new();
        for (stream, tag) in [
            (child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), "out"),
            (child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), "err"),
        ] {
            if let Some(s) = stream {
                let name = m.name.clone();
                let quiet = self.cfg.quiet;
                th.push(std::thread::spawn(move || {
                    for line in BufReader::new(s).lines().map_while(Result::ok) {
                        if !quiet || line.starts_with("RESULT ") {
                            if tag == "err" {
                                eprintln!("[{name}!] {line}");
                            } else {
                                println!("[{name}] {line}");
                            }
                        }
                    }
                }));
            }
        }
        {
            let slot = slot.clone();
            let d = self.clone();
            th.push(std::thread::spawn(move || {
                // wait4 (not Child::wait) so we get the child's rusage at reap time.
                let _keep = child;
                let mut status = 0;
                let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
                let r = loop {
                    let r = unsafe { libc::wait4(pid, &mut status, 0, &mut ru) };
                    if r < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    break r;
                };
                let tv = |t: libc::timeval| t.tv_sec as u64 * 1_000_000_000 + t.tv_usec as u64 * 1000;
                slot.cpu_ns.store(tv(ru.ru_utime) + tv(ru.ru_stime), Ordering::Relaxed);
                use std::os::unix::process::ExitStatusExt;
                let st: std::io::Result<std::process::ExitStatus> = if r == pid {
                    Ok(std::process::ExitStatus::from_raw(status))
                } else {
                    Err(std::io::Error::last_os_error())
                };
                slot.alive.store(false, Ordering::Release);
                *slot.ended.lock().unwrap() = Some(std::time::Instant::now());
                if let Ok(s) = st {
                    *slot.exit.lock().unwrap() = Some(s);
                    d.log(&slot.name, &format!("exited: {s}"));
                }
                fb_capsule::kill_group(pid, libc::SIGKILL);
                d.arena_bytes.fetch_sub(slot.layout.arena_len as u64, Ordering::AcqRel);
            }));
        }
        {
            let slot = slot.clone();
            let gpu = self.gpu.clone();
            th.push(std::thread::spawn(move || control(ours, slot, gpu)));
        }
        self.threads.lock().unwrap().extend(th);

        let lane = Lane {
            slot: slot.clone(),
            submit: unsafe { Ring::<Job>::attach(slot.arena.as_ptr().add(layout.submit_off)).unwrap() }.consumer(),
            complete: unsafe { Ring::<Completion>::attach(slot.arena.as_ptr().add(layout.complete_off)).unwrap() }
                .producer(),
            buf: gbuf,
            queue,
            window: Window { off: layout.data_off as u64, len: (layout.arena_len - layout.data_off) as u64 },
            inflight: VecDeque::new(),
            backlog: VecDeque::new(),
            // 100 ms burst window: a capsule can never run more than budget/10 ahead.
            // `max_percent` (if set) wins over `gpu_ms_per_sec`.
            tokens_ns: (m.gpu.budget_ms_per_sec() * 100_000) as i64,
            budget_ns_per_s: (m.gpu.budget_ms_per_sec() * 1_000_000) as i64,
            burst_ns: (m.gpu.budget_ms_per_sec() * 100_000) as i64,
            throttled_since: None,
            deferred: VecDeque::new(),
            ns_per_item: std::collections::HashMap::new(),
            last_submit_ns: 0,
            last_refill: fb_core::time::now_ns(),
            max_inflight: m.gpu.max_inflight as usize,
            util_win_start: fb_core::time::now_ns(),
            util_busy_ns: 0,
            priority: m.gpu.priority,
            rt: None,
        };
        self.slots.lock().unwrap().push(slot.clone());
        self.lanes_tx.send(lane).ok();
        // Kick the dispatcher in case it is parked.
        let b = unsafe { &*(self.bell.as_ptr() as *const BellPage) };
        b.bell.fetch_add(1, Ordering::Release);
        fb_core::wait::wake_all(&b.bell);
        Ok(slot)
    }

    /// Block until every launched capsule has exited.
    pub fn wait_all(&self) {
        loop {
            if self.slots().iter().all(|s| !s.alive.load(Ordering::Acquire)) {
                std::thread::sleep(Duration::from_millis(20)); // let log pumps drain
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// SIGKILL a capsule by name. Returns false if no live capsule has that name.
    pub fn kill(&self, name: &str) -> bool {
        let mut hit = false;
        for s in self.slots() {
            if s.name == name && s.alive.load(Ordering::Acquire) {
                s.killed_for.lock().unwrap().get_or_insert_with(|| "operator".into());
                fb_capsule::kill_group(s.pid.load(Ordering::Relaxed), libc::SIGKILL);
                hit = true;
            }
        }
        hit
    }

    pub fn shutdown(&self) {
        for s in self.slots() {
            if s.alive.load(Ordering::Acquire) {
                fb_capsule::kill_group(s.pid.load(Ordering::Relaxed), libc::SIGKILL);
            }
        }
        self.stop.store(true, Ordering::Release);
        let b = unsafe { &*(self.bell.as_ptr() as *const BellPage) };
        b.bell.fetch_add(1, Ordering::Release);
        fb_core::wait::wake_all(&b.bell);
        for t in self.threads.lock().unwrap().drain(..) {
            let _ = t.join();
        }
    }

    fn log(&self, name: &str, msg: &str) {
        if !self.cfg.quiet {
            eprintln!("frostd: [{name}] {msg}");
        }
    }
}

// ------------------------------------------------------------------------------------
// control plane

fn control(sock: UnixStream, slot: Arc<Slot>, gpu: Arc<Gpu>) {
    let mut w = match sock.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut r = BufReader::new(sock);
    loop {
        let msg: ToDaemon = match proto::recv(&mut r) {
            Ok(Some(m)) => m,
            _ => break,
        };
        let reply = match msg {
            ToDaemon::Hello { pid } => {
                slot.hello_ns.store(slot.started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if pid != slot.pid.load(Ordering::Acquire) {
                    // sandbox-exec execs in place, so the pid must match.
                    ToCapsule::Error { msg: format!("pid mismatch {pid}") }
                } else {
                    ToCapsule::Welcome {
                        capsule: slot.name.clone(),
                        gpu_name: gpu.name(),
                        kernels: gpu.kernel_names(),
                        realtime: slot.manifest.cpu.realtime,
                        data_off: slot.layout.data_off as u64,
                        data_len: (slot.layout.arena_len - slot.layout.data_off) as u64,
                    }
                }
            }
            ToDaemon::RegisterKernel { name, source, entry } => {
                if !slot.manifest.gpu.custom_kernels {
                    ToCapsule::Error { msg: "custom kernels not permitted by manifest".into() }
                } else {
                    let scoped = format!("{}.{}", slot.name, name);
                    match gpu.register(&scoped, &source, &entry) {
                        Ok(id) => ToCapsule::KernelRegistered { name, id },
                        Err(e) => ToCapsule::Error { msg: format!("{e:#}") },
                    }
                }
            }
            ToDaemon::Stats => ToCapsule::Stats {
                jobs: slot.jobs.load(Ordering::Relaxed),
                gpu_ns: slot.gpu_ns.load(Ordering::Relaxed),
                throttled: slot.throttled.load(Ordering::Relaxed) / 1_000_000,
                footprint: fb_core::sched::phys_footprint(slot.pid.load(Ordering::Relaxed)).unwrap_or(0),
            },
            ToDaemon::Bye => break,
        };
        if proto::send(&mut w, &reply).is_err() {
            break;
        }
    }
}

// ------------------------------------------------------------------------------------
// watchdog

fn watchdog(slots: Arc<Mutex<Vec<Arc<Slot>>>>, stop: Arc<AtomicBool>, enforce: bool) {
    while !stop.load(Ordering::Acquire) {
        for s in slots.lock().unwrap().iter() {
            if !s.alive.load(Ordering::Acquire) {
                continue;
            }
            let pid = s.pid.load(Ordering::Relaxed);
            let limit = s.manifest.memory.limit_mb << 20;
            match fb_capsule::check_memory(pid, limit) {
                fb_capsule::Verdict::Ok { footprint } => {
                    s.peak_footprint.fetch_max(footprint, Ordering::Relaxed);
                }
                fb_capsule::Verdict::OverLimit { footprint, .. } if !enforce => {
                    s.peak_footprint.fetch_max(footprint, Ordering::Relaxed);
                }
                fb_capsule::Verdict::OverLimit { footprint, limit } => {
                    s.peak_footprint.fetch_max(footprint, Ordering::Relaxed);
                    *s.killed_for.lock().unwrap() =
                        Some(format!("memory: footprint {} MiB > limit {} MiB", footprint >> 20, limit >> 20));
                    fb_capsule::kill_group(pid, libc::SIGKILL);
                }
                fb_capsule::Verdict::Gone => {}
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

// ------------------------------------------------------------------------------------
// dispatcher

fn dispatcher(gpu: Arc<Gpu>, cfg: Config, bell: Arc<Arena>, rx: mpsc::Receiver<Lane>, stop: Arc<AtomicBool>) {
    let _ = fb_core::sched::set_thread_qos(fb_core::sched::Qos::UserInteractive);
    if cfg.realtime_dispatcher {
        // 1 ms period, up to 500 us of CPU, done within 900 us.
        if let Err(e) = fb_core::sched::set_realtime(1_000_000, 500_000, 900_000) {
            eprintln!("frostd: RT dispatcher unavailable: {e}");
        }
    }
    let bell_arena = bell.clone();
    let bell = unsafe { &*(bell.as_ptr() as *const BellPage) };
    let mut lanes: Vec<Lane> = Vec::new();
    // Keep-warm pulse: noop into a scratch word past the bell header.
    let warm = if cfg.keepwarm_us > 0 {
        let buf = unsafe { gpu.wrap(bell_arena.as_ptr(), bell_arena.len()) }.ok();
        let q = gpu.new_queue().ok();
        buf.zip(q)
    } else {
        None
    };
    let mut warm_inflight: Option<InFlight> = None;
    let mut last_warm = 0u64;
    let warm_job = {
        let mut j = Job { kernel: abi::kernel::NOOP, n_args: 1, grid: [1, 1, 1], ..Default::default() };
        j.args[0] = abi::BufRef { off: 1024, len: 16 };
        j
    };
    let mut last_activity = fb_core::time::now_ns();
    let idle_spin_ns = cfg.idle_spin_us * 1000;

    while !stop.load(Ordering::Acquire) {
        let mut added = false;
        while let Ok(l) = rx.try_recv() {
            lanes.push(l);
            added = true;
        }
        if added {
            // Higher priority first; stable so equal priorities keep arrival order.
            lanes.sort_by_key(|l| std::cmp::Reverse(l.priority));
        }
        let mut progress = false;
        let now = fb_core::time::now_ns();

        // 1) retire completed command buffers
        for l in lanes.iter_mut() {
            // Publish a live GPU-utilization figure per capsule (~250 ms window) so a
            // container's actual share of the device is observable in `ctl ps`.
            let win = now.saturating_sub(l.util_win_start);
            if win >= 250_000_000 {
                l.slot.util_pm.store((l.util_busy_ns as u128 * 1000 / win as u128) as u64, Ordering::Relaxed);
                l.util_busy_ns = 0;
                l.util_win_start = now;
            }
            while l.inflight.front().is_some_and(|p| p.cb.is_done()) {
                let p = l.inflight.pop_front().unwrap();
                let done = p.cb.wait(); // already complete: returns immediately
                let t_done = fb_core::time::now_ns();
                l.tokens_ns -= done.gpu_ns as i64;
                l.util_busy_ns += done.gpu_ns;
                l.slot.gpu_ns.fetch_add(done.gpu_ns, Ordering::Relaxed);
                let status = if done.ok { Status::Ok } else { Status::GpuError } as u32;
                if let [(_, k, _, items)] = p.jobs[..] {
                    if items > 0 && done.ok {
                        let npi = done.gpu_ns as f64 / items as f64;
                        let e = l.ns_per_item.entry(k).or_insert(npi);
                        *e = 0.7 * *e + 0.3 * npi;
                    }
                }
                for (seq, kernel, emit, _) in p.jobs {
                    if !emit {
                        continue;
                    }
                    l.slot.jobs.fetch_add(1, Ordering::Relaxed);
                    l.backlog.push_back(Completion {
                        seq,
                        status,
                        kernel,
                        gpu_ns: done.gpu_ns,
                        t_seen_ns: p.t_seen,
                        t_done_ns: t_done,
                    });
                }
                progress = true;
            }
            flush_completions(l);
        }

        // 2) admit new work in priority order
        let active_cut = now.saturating_sub(cfg.active_window_us * 1000);
        for l in lanes.iter_mut() {
            if l.submit.available() > 0 {
                l.last_submit_ns = now;
            }
        }
        let hot_prio = lanes
            .iter()
            .filter(|l| l.last_submit_ns > active_cut && l.slot.alive.load(Ordering::Relaxed))
            .map(|l| l.priority)
            .max()
            .unwrap_or(i32::MIN);
        for l in lanes.iter_mut() {
            if !l.slot.alive.load(Ordering::Acquire) {
                while l.submit.pop().is_some() {} // drop work from dead capsules
                l.deferred.clear();
                continue;
            }
            // token bucket refill
            let dt = now.saturating_sub(l.last_refill) as i128;
            l.last_refill = now;
            l.tokens_ns = (l.tokens_ns as i128 + dt * l.budget_ns_per_s as i128 / 1_000_000_000)
                .min(l.burst_ns as i128) as i64;
            if l.submit.corrupt() {
                // Capsule scribbled impossible indices into its ring: protocol violation.
                l.slot.killed_for.lock().unwrap().get_or_insert_with(|| "protocol: corrupted submit ring".into());
                fb_capsule::kill_group(l.slot.pid.load(Ordering::Relaxed), libc::SIGKILL);
                l.slot.alive.store(false, Ordering::Release);
                continue;
            }
            if l.deferred.is_empty() && l.submit.available() == 0 {
                continue;
            }
            // Backpressure: a capsule that isn't draining completions gets no new work
            // admitted (its submit ring fills and it blocks), so daemon memory stays bounded.
            if l.backlog.len() >= abi::COMPLETE_CAP {
                continue;
            }
            if l.tokens_ns <= 0 {
                l.throttled_since.get_or_insert(now);
                continue;
            }
            if let Some(t) = l.throttled_since.take() {
                l.slot.throttled.fetch_add(now - t, Ordering::Relaxed);
            }
            // A higher-priority capsule is active: this lane yields. One slice in flight.
            let yielding = cfg.slice_us > 0 && hot_prio > l.priority;
            let depth = if yielding { 1 } else { l.max_inflight };
            if l.inflight.len() >= depth {
                continue;
            }
            let mut batch = match gpu.batch_on(&l.queue) {
                Ok(b) => Some(b),
                Err(_) => continue,
            };
            let mut jobs = Vec::new();
            let t_seen = fb_core::time::now_ns();
            // Bound the work done for one lane per tick, counting *rejected* jobs and
            // chain waits too, so a hostile capsule can't monopolise the dispatcher.
            let mut popped = 0usize;
            let mut chain_wait_ns = 0u64;
            loop {
                popped += 1;
                if popped > cfg.max_batch * 2 {
                    break;
                }
                // Source: leftover chunks first (they precede anything newer), then ring.
                let (j, emit, items) = if let Some(c) = l.deferred.pop_front() {
                    c
                } else if let Some(j) = l.submit.pop() {
                    if yielding {
                        let npi = l.ns_per_item.get(&j.kernel).copied().unwrap_or(50.0).max(0.01);
                        let max_items = ((cfg.slice_us * 1000) as f64 / npi).clamp(64.0, u32::MAX as f64) as u32;
                        if let Some(parts) = fb_gpu::split_job(&j, max_items) {
                            let n = parts.len();
                            for (i, c) in parts.into_iter().enumerate() {
                                let it = work_items(&c);
                                l.deferred.push_back((c, i + 1 == n, it));
                            }
                            continue;
                        }
                    }
                    (j, true, work_items(&j))
                } else {
                    break;
                };
                // Graphics plane: a hardware-rasterized frame is not a compute job. Flush any
                // compute work already batched, draw the frame on its own command buffer, and
                // end this lane's tick.
                if j.kernel == abi::kernel::DRAW_TRIANGLES {
                    if let Some(b) = batch.take() {
                        if !jobs.is_empty() {
                            l.inflight.push_back(Pending {
                                cb: b.commit(),
                                jobs: std::mem::take(&mut jobs),
                                t_seen,
                            });
                        }
                        // otherwise `b` drops here; its empty encoder is ended by Drop.
                    }
                    render_frame(gpu.as_ref(), l, &j, t_seen);
                    break;
                }
                match batch.as_mut().unwrap().encode(&l.buf, l.window, &j) {
                    Ok(()) => jobs.push((j.seq, j.kernel, emit, items)),
                    Err(st) => {
                        // A rejected chunk fails the whole job; drop its remaining chunks.
                        l.deferred.retain(|(d, _, _)| d.seq != j.seq);
                        l.slot.rejected.fetch_add(1, Ordering::Relaxed);
                        l.backlog.push_back(Completion {
                            seq: j.seq,
                            status: st as u32,
                            kernel: j.kernel,
                            t_seen_ns: t_seen,
                            t_done_ns: fb_core::time::now_ns(),
                            ..Default::default()
                        });
                    }
                }
                if yielding && !jobs.is_empty() {
                    break; // one unit per command buffer while yielding
                }
                let chained = j.flags & JOB_FLAG_CHAIN != 0;
                if jobs.len() >= cfg.max_batch && !chained {
                    break;
                }
                if chained && l.deferred.is_empty() && chain_wait_ns < 200_000 {
                    // The rest of the chain is being written right now; wait for it
                    // (bounded per tick).
                    let t0 = fb_core::time::now_ns();
                    while l.submit.available() == 0 && fb_core::time::now_ns() - t0 < 200_000 - chain_wait_ns {
                        std::hint::spin_loop();
                    }
                    chain_wait_ns += fb_core::time::now_ns() - t0;
                }
            }
            if let Some(b) = batch {
                if !jobs.is_empty() {
                    l.inflight.push_back(Pending { cb: b.commit(), jobs, t_seen });
                }
            }
            flush_completions(l);
            progress = true;
        }

        // 3) retire dead lanes
        lanes.retain(|l| {
            let keep = l.slot.alive.load(Ordering::Acquire) || !l.inflight.is_empty();
            if !keep {
                l.slot.hdr().closed.store(1, Ordering::Release);
                fb_core::wait::wake_all(&l.slot.hdr().complete_bell);
            }
            keep
        });

        // 3b) keep-warm pulse
        if let Some((wbuf, wq)) = &warm {
            if !lanes.is_empty() && now - last_warm > cfg.keepwarm_us * 1000
                && warm_inflight.as_ref().is_none_or(|f| f.is_done()) {
                    if let Ok(mut b) = gpu.batch_on(wq) {
                        let _ = b.encode(wbuf, Window { off: 1024, len: 1024 }, &warm_job);
                        warm_inflight = Some(b.commit());
                    }
                    last_warm = now;
                }
        }

        // 4) idle policy
        if progress {
            last_activity = fb_core::time::now_ns();
            continue;
        }
        let busy = lanes.iter().any(|l| !l.inflight.is_empty() || !l.deferred.is_empty());
        let warming = warm.is_some() && !lanes.is_empty();
        if busy || warming || fb_core::time::now_ns() - last_activity < idle_spin_ns {
            std::hint::spin_loop();
            continue;
        }
        let seen = bell.bell.load(Ordering::Acquire);
        bell.waiters.store(1, Ordering::SeqCst);
        // Re-check after publishing the waiter flag to avoid a lost wakeup.
        let pending = lanes.iter_mut().any(|l| l.submit.available() > 0) || !rx_is_empty(&rx);
        if !pending {
            fb_core::wait::wait(&bell.bell, seen, 5_000_000);
        }
        bell.waiters.store(0, Ordering::SeqCst);
        last_activity = fb_core::time::now_ns();
    }
}

fn rx_is_empty(_rx: &mpsc::Receiver<Lane>) -> bool {
    // New lanes always ring the bell after sending, so no lost wakeup here.
    true
}

/// Validate and submit one hardware-rasterized frame (kernel `DRAW_TRIANGLES`). The
/// framebuffer and vertex buffer both live in the capsule's arena, so every reference is
/// bounds-checked against the lane's data window before the GPU touches it.
fn render_frame(gpu: &Gpu, l: &mut Lane, j: &Job, t_seen: u64) {
    fn reject(l: &mut Lane, j: &Job, t_seen: u64, st: Status) {
        l.slot.rejected.fetch_add(1, Ordering::Relaxed);
        l.backlog.push_back(Completion {
            seq: j.seq,
            status: st as u32,
            kernel: j.kernel,
            t_seen_ns: t_seen,
            t_done_ns: fb_core::time::now_ns(),
            ..Default::default()
        });
    }
    let (w, h, vcount) = (j.params[0] as usize, j.params[1] as usize, j.params[2] as usize);
    if j.n_args < 2 || w == 0 || h == 0 || w > 16384 || h > 16384 || vcount > (1 << 22) {
        return reject(l, j, t_seen, Status::BadArgs);
    }
    let win = l.window;
    let (a0, a1) = (j.args[0], j.args[1]);
    let stride = gpu.fb_stride_bytes(w) as u64;
    let fb_bytes = h as u64 * stride;
    let v_bytes = vcount as u64 * fb_gpu::VERTEX_BYTES as u64;
    let in_win = |off: u64, bytes: u64| -> bool {
        off.checked_add(bytes).is_some_and(|end| off >= win.off && end <= win.off + win.len)
    };
    // The framebuffer offset must satisfy Metal's linear-texture alignment (>= 256, which
    // the arena bump allocator already guarantees for 256-aligned allocations).
    if a0.off % 256 != 0 || !in_win(a0.off, fb_bytes) || a1.off % 16 != 0 || !in_win(a1.off, v_bytes) {
        return reject(l, j, t_seen, Status::OutOfBounds);
    }
    let stale = !matches!(&l.rt, Some(rt) if rt.off == a0.off && rt.w == w as u32 && rt.h == h as u32);
    if stale {
        match gpu.make_target(&l.buf, a0.off as usize, w, h) {
            Ok(t) => l.rt = Some(t),
            Err(_) => return reject(l, j, t_seen, Status::GpuError),
        }
    }
    let target = l.rt.as_ref().unwrap();
    match gpu.render(&l.queue, target, &l.buf, a1.off as usize, vcount) {
        Ok(cb) => l.inflight.push_back(Pending { cb, jobs: vec![(j.seq, j.kernel, true, 0)], t_seen }),
        Err(_) => reject(l, j, t_seen, Status::GpuError),
    }
}

fn flush_completions(l: &mut Lane) {
    let mut pushed = false;
    while let Some(c) = l.backlog.front().copied() {
        if l.complete.push(c).is_err() {
            break;
        }
        l.backlog.pop_front();
        pushed = true;
    }
    if pushed {
        let h = l.slot.hdr();
        h.complete_bell.fetch_add(1, Ordering::Release);
        if h.complete_waiters.load(Ordering::Acquire) != 0 {
            fb_core::wait::wake_all(&h.complete_bell);
        }
    }
}
