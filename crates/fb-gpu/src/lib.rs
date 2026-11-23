//! fb-gpu: thin, latency-oriented Metal layer used by `frostd`.
//!
//! * one `MTLDevice` + one high-priority command queue per daemon
//! * a kernel registry (built-ins compiled from MSL at start, plus optional custom MSL)
//! * zero-copy wrapping of shared arenas (`newBufferWithBytesNoCopy`, StorageModeShared)
//! * a [`Batch`] that encodes many [`Job`]s into one command buffer, validating every
//!   buffer reference against the owning arena's data region first.

use anyhow::{anyhow, bail, Result};
use fb_core::abi::{kernel, Job, Status, MAX_ARGS, MAX_PARAMS};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::RwLock;

pub const BUILTIN_SRC: &str = include_str!("kernels.metal");

type Dev = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type Pipe = ProtocolObject<dyn MTLComputePipelineState>;
type CmdBuf = ProtocolObject<dyn MTLCommandBuffer>;
type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub struct Kernel {
    pub name: String,
    pipe: Retained<Pipe>,
    pub max_threads: usize,
    pub simd_width: usize,
}

pub struct Gpu {
    dev: Retained<Dev>,
    queue: Retained<Queue>,
    kernels: RwLock<HashMap<u32, Kernel>>,
    names: RwLock<BTreeMap<String, u32>>,
    next_id: std::sync::atomic::AtomicU32,
}

// Metal device, queue and pipeline objects are documented thread-safe.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let dev = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("no Metal device"))?;
        let queue = dev
            .newCommandQueueWithMaxCommandBufferCount(64)
            .ok_or_else(|| anyhow!("newCommandQueue failed"))?;
        let g = Gpu {
            dev,
            queue,
            kernels: RwLock::new(HashMap::new()),
            names: RwLock::new(BTreeMap::new()),
            next_id: std::sync::atomic::AtomicU32::new(kernel::FIRST_CUSTOM),
        };
        for (id, entry) in [
            (kernel::VADD_F32, "fb_vadd_f32"),
            (kernel::GEMV_F32, "fb_gemv_f32"),
            (kernel::SHA256_BATCH, "fb_sha256_batch"),
            (kernel::NOOP, "fb_noop"),
            (kernel::SAXPY_F32, "fb_saxpy_f32"),
            (kernel::GRAVITY_RENDER, "fb_gravity_render"),
        ] {
            g.install(id, entry.trim_start_matches("fb_"), BUILTIN_SRC, entry)?;
        }
        Ok(g)
    }

    pub fn name(&self) -> String {
        self.dev.name().to_string()
    }
    pub fn unified_memory(&self) -> bool {
        self.dev.hasUnifiedMemory()
    }
    pub fn recommended_working_set(&self) -> u64 {
        self.dev.recommendedMaxWorkingSetSize()
    }
    pub fn allocated(&self) -> usize {
        self.dev.currentAllocatedSize()
    }
    pub fn kernel_names(&self) -> BTreeMap<String, u32> {
        self.names.read().unwrap().clone()
    }

    fn install(&self, id: u32, name: &str, src: &str, entry: &str) -> Result<()> {
        let lib = self
            .dev
            .newLibraryWithSource_options_error(&NSString::from_str(src), None)
            .map_err(|e| anyhow!("MSL compile failed: {}", e.localizedDescription()))?;
        let f = lib
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| anyhow!("entry point {entry} not found"))?;
        let pipe = self
            .dev
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| anyhow!("pipeline: {}", e.localizedDescription()))?;
        let k = Kernel {
            name: name.to_string(),
            max_threads: pipe.maxTotalThreadsPerThreadgroup(),
            simd_width: pipe.threadExecutionWidth(),
            pipe,
        };
        self.kernels.write().unwrap().insert(id, k);
        self.names.write().unwrap().insert(name.to_string(), id);
        Ok(())
    }

    /// Compile and register a custom kernel; returns its id.
    pub fn register(&self, name: &str, src: &str, entry: &str) -> Result<u32> {
        if let Some(&id) = self.names.read().unwrap().get(name) {
            if id >= kernel::FIRST_CUSTOM {
                bail!("kernel name {name} already registered as {id}");
            }
            bail!("kernel name {name} shadows a built-in");
        }
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.install(id, name, src, entry)?;
        Ok(id)
    }

    /// Wrap page-aligned shared memory as a zero-copy MTLBuffer. The caller keeps the
    /// memory mapped for at least as long as the returned buffer lives.
    ///
    /// # Safety
    /// `ptr..ptr+len` must be a live, page-aligned mapping.
    pub unsafe fn wrap(&self, ptr: *mut u8, len: usize) -> Result<GpuBuffer> {
        let page = fb_core::page_size();
        if !(ptr as usize).is_multiple_of(page) || !len.is_multiple_of(page) {
            bail!("wrap: pointer/length must be page aligned ({page})");
        }
        let buf = self
            .dev
            .newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(ptr as *mut c_void).unwrap(),
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .ok_or_else(|| anyhow!("newBufferWithBytesNoCopy failed"))?;
        Ok(GpuBuffer { buf, len, base: ptr as usize })
    }

    /// Ordinary daemon-owned shared buffer (used by in-process benchmarks).
    pub fn alloc(&self, len: usize) -> Result<GpuBuffer> {
        let buf = self
            .dev
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBuffer failed"))?;
        let base = buf.contents().as_ptr() as usize;
        Ok(GpuBuffer { buf, len, base })
    }

    /// A dedicated command queue. Each capsule gets its own so a long-running job in
    /// one capsule never sits in front of another capsule's work in a FIFO.
    pub fn new_queue(&self) -> Result<GpuQueue> {
        let q = self
            .dev
            .newCommandQueueWithMaxCommandBufferCount(64)
            .ok_or_else(|| anyhow!("newCommandQueue failed"))?;
        Ok(GpuQueue(q))
    }

    /// Batch on the device's default queue.
    pub fn batch(&self) -> Result<Batch<'_>> {
        self.batch_raw(&self.queue)
    }

    pub fn batch_on<'g>(&'g self, q: &GpuQueue) -> Result<Batch<'g>> {
        self.batch_raw(&q.0)
    }

    fn batch_raw<'g>(&'g self, queue: &Queue) -> Result<Batch<'g>> {
        // Unretained references: the daemon guarantees arenas outlive in-flight work,
        // and it saves retain/release traffic per dispatch.
        let cb = queue
            .commandBufferWithUnretainedReferences()
            .ok_or_else(|| anyhow!("commandBuffer failed"))?;
        let enc = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or_else(|| anyhow!("computeCommandEncoder failed"))?;
        Ok(Batch { gpu: self, cb, enc: Some(enc), jobs: 0 })
    }
}

pub struct GpuQueue(Retained<Queue>);
unsafe impl Send for GpuQueue {}
unsafe impl Sync for GpuQueue {}

pub struct GpuBuffer {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
    base: usize,
}
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    pub fn as_ptr(&self) -> *mut u8 {
        self.base as *mut u8
    }
    pub fn gpu_address(&self) -> u64 {
        self.buf.gpuAddress()
    }
}

/// Allowed window inside a buffer that jobs may reference (the arena data region).
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub off: u64,
    pub len: u64,
}

pub struct Batch<'g> {
    gpu: &'g Gpu,
    cb: Retained<CmdBuf>,
    enc: Option<Retained<Enc>>,
    jobs: usize,
}

impl<'g> Batch<'g> {
    pub fn len(&self) -> usize {
        self.jobs
    }
    pub fn is_empty(&self) -> bool {
        self.jobs == 0
    }

    /// Validate and encode one job. Validation failures do not poison the batch.
    pub fn encode(&mut self, buf: &GpuBuffer, win: Window, job: &Job) -> Result<(), Status> {
        let kernels = self.gpu.kernels.read().unwrap();
        let k = kernels.get(&job.kernel).ok_or(Status::BadKernel)?;
        let n = job.n_args as usize;
        if n > MAX_ARGS {
            return Err(Status::BadArgs);
        }
        for a in &job.args[..n] {
            let end = a.off.checked_add(a.len).ok_or(Status::OutOfBounds)?;
            if a.off < win.off || end > win.off + win.len || a.off % 16 != 0 {
                return Err(Status::OutOfBounds);
            }
        }
        let grid = job.grid.map(|v| v.max(1) as usize);
        let total: usize = grid.iter().product();
        if total == 0 || total > (1 << 30) {
            return Err(Status::BadArgs);
        }
        let tg = if job.tg[0] == 0 {
            [k.max_threads.min(256).min(grid[0].next_power_of_two().max(k.simd_width)), 1, 1]
        } else {
            let t = job.tg.map(|v| v.max(1) as usize);
            if t.iter().product::<usize>() > k.max_threads {
                return Err(Status::BadArgs);
            }
            t
        };
        let enc = self.enc.as_ref().unwrap();
        enc.setComputePipelineState(&k.pipe);
        unsafe {
            for (i, a) in job.args[..n].iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(&buf.buf), a.off as usize, i);
            }
            enc.setBytes_length_atIndex(
                NonNull::new(job.params.as_ptr() as *mut c_void).unwrap(),
                4 * MAX_PARAMS,
                n,
            );
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: grid[0], height: grid[1], depth: grid[2] },
            MTLSize { width: tg[0], height: tg[1], depth: tg[2] },
        );
        self.jobs += 1;
        Ok(())
    }

    /// End encoding and submit to the GPU.
    pub fn commit(mut self) -> InFlight {
        if let Some(enc) = self.enc.take() {
            enc.endEncoding();
        }
        self.cb.commit();
        InFlight { cb: self.cb.clone() }
    }
}

impl Drop for Batch<'_> {
    /// A batch dropped without commit (e.g. every job was rejected) must still end its
    /// encoder, or Metal asserts and aborts the whole daemon.
    fn drop(&mut self) {
        if let Some(enc) = self.enc.take() {
            enc.endEncoding();
        }
    }
}

pub struct InFlight {
    cb: Retained<CmdBuf>,
}
unsafe impl Send for InFlight {}

pub struct Done {
    pub ok: bool,
    pub gpu_ns: u64,
    pub error: Option<String>,
}

impl InFlight {
    /// Busy-poll for completion (lowest latency; burns a core).
    pub fn spin(self) -> Done {
        loop {
            let s = self.cb.status();
            if s == MTLCommandBufferStatus::Completed || s == MTLCommandBufferStatus::Error {
                return self.done();
            }
            std::hint::spin_loop();
        }
    }
    /// Block in the driver until completion.
    pub fn wait(self) -> Done {
        self.cb.waitUntilCompleted();
        self.done()
    }
    pub fn is_done(&self) -> bool {
        matches!(self.cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error)
    }
    fn done(self) -> Done {
        let ok = self.cb.status() == MTLCommandBufferStatus::Completed;
        let gpu_ns = ((self.cb.GPUEndTime() - self.cb.GPUStartTime()).max(0.0) * 1e9) as u64;
        let error = self.cb.error().map(|e| e.localizedDescription().to_string());
        Done { ok, gpu_ns, error }
    }
}

/// Split a dispatch of a *splittable* built-in kernel into chunks of at most
/// `max_items` work items, rewriting buffer offsets and counts so each chunk is an
/// independent, valid dispatch. This lets the scheduler bound how long a low-priority
/// capsule can occupy the GPU (Metal cannot preempt a running kernel).
/// Returns `None` if the kernel isn't splittable or splitting would misalign a buffer.
pub fn split_job(job: &Job, max_items: u32) -> Option<Vec<Job>> {
    // (items, threads per item, per-arg byte stride per item, index of count param)
    let (items, tpi, strides): (u32, u32, [u64; MAX_ARGS]) = match job.kernel {
        kernel::VADD_F32 => (job.params[0], 1, [4, 4, 4, 0, 0, 0]),
        kernel::SAXPY_F32 => (job.params[0], 1, [4, 4, 0, 0, 0, 0]),
        kernel::SHA256_BATCH => (job.params[0], 1, [job.params[2] as u64, 32, 0, 0, 0, 0]),
        kernel::GEMV_F32 => (job.params[0], 32, [job.params[1] as u64 * 4, 0, 4, 0, 0, 0]),
        _ => return None,
    };
    let max_items = max_items.max(16) / 16 * 16;
    if items <= max_items || job.grid[1] > 1 || job.grid[2] > 1 || job.grid[0] != items * tpi {
        return None;
    }
    let mut out = Vec::new();
    let mut base = 0u32;
    while base < items {
        let n = (items - base).min(max_items);
        let mut c = *job;
        c.params[0] = n;
        c.grid = [n * tpi, 1, 1];
        for i in 0..job.n_args as usize {
            let s = strides[i];
            if s == 0 {
                continue; // broadcast argument (e.g. GEMV x)
            }
            let off = job.args[i].off + base as u64 * s;
            if !off.is_multiple_of(16) {
                return None;
            }
            c.args[i] = fb_core::abi::BufRef { off, len: n as u64 * s };
        }
        out.push(c);
        base += n;
    }
    Some(out)
}
