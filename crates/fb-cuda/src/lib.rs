//! # fb-cuda — a CUDA-compatible runtime for Apple Silicon
//!
//! **There is no NVIDIA GPU in an Apple Silicon Mac.** CUDA is NVIDIA's proprietary
//! stack whose kernel driver binds to NVIDIA hardware; it physically cannot run on an
//! M-series chip. This crate is therefore *not* the NVIDIA CUDA driver. It is a
//! **compatibility layer**: it implements a useful subset of the CUDA Driver API, the
//! CUDA Runtime API and a CUDA-Q-style quantum kernel API, and dispatches the actual
//! compute to the Mac's GPU through **Metal** (the same path `fb-gpu`/`frostd` use).
//!
//! The approach mirrors ZLUDA / SCALE on other non-NVIDIA hardware:
//!
//! * `CUdeviceptr` is the address of a `StorageModeShared` Metal buffer. Because Apple
//!   Silicon has unified memory, host↔device copies are real `memcpy`s over the same
//!   physical pages — zero-copy, matching Frost Bridge's arena model.
//! * CUDA kernels are authored in a **CUDA-C subset** (see [`translate`]) that is
//!   source-translated to Metal Shading Language and compiled at load time with
//!   `newLibraryWithSource`.
//! * CUDA's `<<<grid, block>>>` maps exactly onto Metal's
//!   `dispatchThreadgroups:threadsPerThreadgroup:`; `threadIdx/blockIdx/blockDim/gridDim`
//!   map onto Metal's thread-position attributes.
//!
//! Two front ends sit on top of [`CudaRuntime`]:
//! * the ergonomic Rust API in this module (used by the tests and examples), and
//! * the C ABI in [`driver`] (`cuInit`, `cuLaunchKernel`, `cudaMalloc`, …), which lets a
//!   `libfb_cuda.dylib` stand in for `libcuda`/`libcudart` for programs using that subset.
//!
//! The quantum (CUDA-Q) front end is in [`quantum`].

use anyhow::{anyhow, bail, Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

pub mod driver;
pub mod quantum;
pub mod translate;

type Dev = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type Pipe = ProtocolObject<dyn MTLComputePipelineState>;
type Buf = ProtocolObject<dyn MTLBuffer>;
type CmdBuf = ProtocolObject<dyn MTLCommandBuffer>;

/// A CUDA device pointer. It is literally the host-visible address of a shared Metal
/// buffer (unified memory), so the host can also read/write it directly.
pub type DevicePtr = u64;

/// One device allocation, tracked so a `DevicePtr` can be resolved back to its buffer.
struct Allocation {
    base: u64,
    len: usize,
    #[allow(dead_code)]
    buf: Retained<Buf>,
}

/// An argument to a kernel launch.
#[derive(Clone, Debug)]
pub enum Arg {
    /// A device pointer (bound with `setBuffer:offset:atIndex:`).
    Ptr(DevicePtr),
    /// Raw scalar bytes (bound with `setBytes:length:atIndex:`).
    Scalar(Vec<u8>),
}

impl Arg {
    /// Convenience: a scalar argument from any `Copy` plain-old-data value.
    pub fn scalar<T: Copy>(v: T) -> Arg {
        let bytes = unsafe {
            std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>())
        };
        Arg::Scalar(bytes.to_vec())
    }
}

/// The CUDA runtime: one Metal device + command queue, an allocation table, and the set
/// of command buffers submitted but not yet synchronized (CUDA's implicit stream model).
pub struct CudaRuntime {
    dev: Retained<Dev>,
    queue: Retained<Queue>,
    allocs: Mutex<Vec<Allocation>>,
    pending: Mutex<Vec<Retained<CmdBuf>>>,
}

// Metal device/queue/buffer objects are documented thread-safe.
unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    /// Initialize the runtime on the system default Metal device (CUDA's `cuInit`).
    pub fn new() -> Result<Arc<CudaRuntime>> {
        let dev = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("no Metal device"))?;
        let queue = dev
            .newCommandQueueWithMaxCommandBufferCount(64)
            .ok_or_else(|| anyhow!("newCommandQueue failed"))?;
        Ok(Arc::new(CudaRuntime {
            dev,
            queue,
            allocs: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
        }))
    }

    /// The device name, e.g. "Apple M4".
    pub fn device_name(&self) -> String {
        self.dev.name().to_string()
    }

    /// Whether the device has unified memory (always true on Apple Silicon).
    pub fn unified_memory(&self) -> bool {
        self.dev.hasUnifiedMemory()
    }

    /// `cudaMalloc`: allocate `len` bytes of device memory, returning a device pointer.
    pub fn malloc(&self, len: usize) -> Result<DevicePtr> {
        let len = len.max(1);
        let buf = self
            .dev
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBuffer({len}) failed"))?;
        let base = buf.contents().as_ptr() as u64;
        self.allocs.lock().unwrap().push(Allocation { base, len, buf });
        Ok(base)
    }

    /// `cudaFree`: release a device allocation.
    pub fn free(&self, ptr: DevicePtr) -> Result<()> {
        let mut a = self.allocs.lock().unwrap();
        let before = a.len();
        a.retain(|al| al.base != ptr);
        if a.len() == before {
            bail!("free: unknown device pointer {ptr:#x}");
        }
        Ok(())
    }

    /// Resolve a device pointer to its owning buffer and the offset within it.
    fn resolve(&self, ptr: DevicePtr) -> Result<(Retained<Buf>, usize)> {
        let a = self.allocs.lock().unwrap();
        for al in a.iter() {
            if ptr >= al.base && ptr < al.base + al.len as u64 {
                return Ok((al.buf.clone(), (ptr - al.base) as usize));
            }
        }
        bail!("device pointer {ptr:#x} is not inside any allocation")
    }

    /// `cudaMemcpy(HostToDevice)`. Unified memory ⇒ a plain `memcpy`.
    ///
    /// # Safety
    /// `dst .. dst+len` must lie inside one live allocation.
    pub unsafe fn memcpy_htod(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        self.check_range(dst, src.len())?;
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        Ok(())
    }

    /// `cudaMemcpy(DeviceToHost)`.
    ///
    /// # Safety
    /// `src .. src+dst.len()` must lie inside one live allocation.
    pub unsafe fn memcpy_dtoh(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        self.check_range(src, dst.len())?;
        std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len());
        Ok(())
    }

    /// `cudaMemcpy(DeviceToDevice)`.
    pub fn memcpy_dtod(&self, dst: DevicePtr, src: DevicePtr, len: usize) -> Result<()> {
        self.check_range(dst, len)?;
        self.check_range(src, len)?;
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
        Ok(())
    }

    /// Typed host→device helper.
    pub fn copy_to_device<T: Copy>(&self, dst: DevicePtr, src: &[T]) -> Result<()> {
        let bytes =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) };
        unsafe { self.memcpy_htod(dst, bytes) }
    }

    /// Typed device→host helper.
    pub fn copy_from_device<T: Copy>(&self, dst: &mut [T], src: DevicePtr) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, std::mem::size_of_val(dst))
        };
        unsafe { self.memcpy_dtoh(bytes, src) }
    }

    fn check_range(&self, ptr: DevicePtr, len: usize) -> Result<()> {
        let a = self.allocs.lock().unwrap();
        for al in a.iter() {
            if ptr >= al.base && ptr.checked_add(len as u64).is_some_and(|e| e <= al.base + al.len as u64)
            {
                return Ok(());
            }
        }
        bail!("range [{ptr:#x}, +{len}) escapes every allocation")
    }

    /// Compile a module from **CUDA-C subset** source (translated to MSL). Equivalent to
    /// `cuModuleLoadData` with source input.
    pub fn load_cuda_source(self: &Arc<Self>, cuda_src: &str) -> Result<Arc<Module>> {
        let translated = translate::cuda_to_msl(cuda_src).context("CUDA→MSL translation")?;
        self.load_msl(&translated.msl, translated.functions)
    }

    /// Compile a module from Metal Shading Language directly, with a caller-supplied
    /// signature for each kernel. Mostly used internally and by the quantum backend.
    pub fn load_msl(
        self: &Arc<Self>,
        msl: &str,
        functions: Vec<KernelSig>,
    ) -> Result<Arc<Module>> {
        let lib = self
            .dev
            .newLibraryWithSource_options_error(&NSString::from_str(msl), None)
            .map_err(|e| anyhow!("MSL compile failed: {}\n--- source ---\n{msl}", e.localizedDescription()))?;
        let mut pipes = HashMap::new();
        for sig in &functions {
            let f = lib
                .newFunctionWithName(&NSString::from_str(&sig.name))
                .ok_or_else(|| anyhow!("entry point `{}` not found after compile", sig.name))?;
            let pipe = self
                .dev
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow!("pipeline `{}`: {}", sig.name, e.localizedDescription()))?;
            pipes.insert(sig.name.clone(), (pipe, sig.clone()));
        }
        Ok(Arc::new(Module { rt: self.clone(), pipes }))
    }

    /// Launch `func` with CUDA `<<<grid, block>>>` semantics and the given arguments.
    /// The launch is enqueued (not awaited); call [`CudaRuntime::synchronize`] before
    /// reading results back — exactly like CUDA's default stream.
    pub fn launch(&self, func: &Function, grid: [u32; 3], block: [u32; 3], args: &[Arg]) -> Result<()> {
        let sig = &func.sig;
        if args.len() != sig.params.len() {
            bail!("`{}` expects {} args, got {}", sig.name, sig.params.len(), args.len());
        }
        for (i, (a, p)) in args.iter().zip(&sig.params).enumerate() {
            match (a, p.is_pointer) {
                (Arg::Ptr(_), true) | (Arg::Scalar(_), false) => {}
                _ => bail!("arg {i} ({}) has the wrong kind for `{}`", p.name, sig.name),
            }
        }
        let gx = grid[0].max(1) as usize;
        let gy = grid[1].max(1) as usize;
        let gz = grid[2].max(1) as usize;
        let bx = block[0].max(1) as usize;
        let by = block[1].max(1) as usize;
        let bz = block[2].max(1) as usize;
        let tg_total = bx * by * bz;
        if tg_total == 0 || tg_total > func.max_threads {
            bail!(
                "block {bx}x{by}x{bz} = {tg_total} threads exceeds device max {} for `{}`",
                func.max_threads,
                sig.name
            );
        }

        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer failed"))?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder failed"))?;
        enc.setComputePipelineState(&func.pipe);

        // Keep scalar byte buffers alive until encoding is done.
        for (i, arg) in args.iter().enumerate() {
            match arg {
                Arg::Ptr(ptr) => {
                    let (buf, off) = self.resolve(*ptr)?;
                    unsafe { enc.setBuffer_offset_atIndex(Some(&buf), off, i) };
                }
                Arg::Scalar(bytes) => unsafe {
                    enc.setBytes_length_atIndex(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len(),
                        i,
                    );
                },
            }
        }

        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: gx, height: gy, depth: gz },
            MTLSize { width: bx, height: by, depth: bz },
        );
        enc.endEncoding();
        cb.commit();
        self.pending.lock().unwrap().push(cb);
        Ok(())
    }

    /// Encode a sequence of single-buffer + inline-bytes dispatches of one pipeline into a
    /// single command buffer with a **serial** compute encoder, then wait. Metal inserts a
    /// memory barrier between serial dispatches, so each op sees the previous op's writes —
    /// exactly the dependency chain a quantum circuit needs. Used by [`quantum`].
    pub(crate) fn dispatch_serial(
        &self,
        func: &Function,
        state: DevicePtr,
        ops: &[(Vec<u8>, usize)],
    ) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let (buf, off) = self.resolve(state)?;
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer failed"))?;
        let enc = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
            .ok_or_else(|| anyhow!("computeCommandEncoder failed"))?;
        enc.setComputePipelineState(&func.pipe);
        unsafe { enc.setBuffer_offset_atIndex(Some(&buf), off, 0) };
        for (params, total) in ops {
            unsafe {
                enc.setBytes_length_atIndex(
                    NonNull::new(params.as_ptr() as *mut c_void).unwrap(),
                    params.len(),
                    1,
                );
            }
            let tg = (*total).min(func.max_threads).max(1);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: *total, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            let msg = cb
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown GPU error".into());
            bail!("quantum gate batch failed: {msg}");
        }
        Ok(())
    }

    /// `cudaDeviceSynchronize` / `cuCtxSynchronize`: wait for every enqueued launch and
    /// surface the first GPU error, if any.
    pub fn synchronize(&self) -> Result<()> {
        let pending: Vec<_> = std::mem::take(&mut *self.pending.lock().unwrap());
        for cb in &pending {
            cb.waitUntilCompleted();
        }
        for cb in &pending {
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown GPU error".into());
                bail!("kernel execution failed: {msg}");
            }
        }
        Ok(())
    }
}

/// The signature of one kernel: its name and ordered parameters.
#[derive(Clone, Debug)]
pub struct KernelSig {
    pub name: String,
    pub params: Vec<Param>,
}

/// One kernel parameter.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    /// Pointer parameters are bound as buffers; scalars are bound inline with `setBytes`.
    pub is_pointer: bool,
    /// Size in bytes of a scalar argument's value (8 for pointers). Used by the C ABI's
    /// `cuLaunchKernel` to copy the right number of bytes out of `kernelParams`.
    pub byte_size: usize,
}

/// A compiled module (a Metal library + its pipelines), analogous to `CUmodule`.
pub struct Module {
    rt: Arc<CudaRuntime>,
    pipes: HashMap<String, (Retained<Pipe>, KernelSig)>,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// Look up a kernel by name (`cuModuleGetFunction`).
    pub fn function(&self, name: &str) -> Result<Function> {
        let (pipe, sig) = self
            .pipes
            .get(name)
            .ok_or_else(|| anyhow!("no kernel named `{name}` in module"))?;
        Ok(Function {
            rt: self.rt.clone(),
            pipe: pipe.clone(),
            max_threads: pipe.maxTotalThreadsPerThreadgroup(),
            sig: sig.clone(),
        })
    }

    /// Names of all kernels in this module.
    pub fn kernel_names(&self) -> Vec<String> {
        self.pipes.keys().cloned().collect()
    }
}

/// A handle to a launchable kernel, analogous to `CUfunction`.
pub struct Function {
    rt: Arc<CudaRuntime>,
    pipe: Retained<Pipe>,
    pub max_threads: usize,
    sig: KernelSig,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Launch this kernel and then synchronize (convenience for the common case).
    pub fn launch_sync(&self, grid: [u32; 3], block: [u32; 3], args: &[Arg]) -> Result<()> {
        self.rt.launch(self, grid, block, args)?;
        self.rt.synchronize()
    }

    pub fn signature(&self) -> &KernelSig {
        &self.sig
    }
}
