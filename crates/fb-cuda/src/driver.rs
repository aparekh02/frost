//! C ABI: a subset of the CUDA **Driver API** (`cu*`) and **Runtime API** (`cuda*`).
//!
//! Building this crate as a `cdylib` produces `libfb_cuda.dylib`, whose exported symbols
//! match these CUDA entry points. A program (or a `dlopen`-based shim) that uses only this
//! subset can therefore be pointed at it instead of NVIDIA's `libcuda`/`libcudart` and run
//! on the Mac's GPU.
//!
//! Scope and honesty:
//! * `cuModuleLoadData` takes a **null-terminated CUDA-C-subset source string** as its
//!   image (not a cubin/PTX binary); it is translated to MSL and compiled. This is the
//!   kernel-input path chosen for this runtime.
//! * The CUDA **Runtime** launch syntax `kernel<<<grid,block>>>(...)` relies on host-side
//!   stubs that `nvcc` emits (`__cudaRegisterFunction`); we do not emulate those. Launch
//!   through the Driver API's [`cuLaunchKernel`] (or [`fbCudaLaunch`]) instead.
//! * Streams are accepted and ignored (single implicit stream); `cuCtxSynchronize` /
//!   `cudaDeviceSynchronize` drain all work.

use crate::{Arg, CudaRuntime, Function, Module};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

// ---- CUDA result codes ----------------------------------------------------------------
pub const CUDA_SUCCESS: i32 = 0;
pub const CUDA_ERROR_INVALID_VALUE: i32 = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
pub const CUDA_ERROR_INVALID_SOURCE: i32 = 300;
pub const CUDA_ERROR_INVALID_HANDLE: i32 = 400;
pub const CUDA_ERROR_LAUNCH_FAILED: i32 = 719;

// cudaMemcpyKind
const MEMCPY_HTOD: i32 = 1;
const MEMCPY_DTOH: i32 = 2;
const MEMCPY_DTOD: i32 = 3;

#[derive(Default)]
struct Globals {
    rt: Option<Arc<CudaRuntime>>,
    modules: HashMap<u64, Arc<Module>>,
    funcs: HashMap<u64, Function>,
    next: u64,
}

fn state() -> &'static Mutex<Globals> {
    static S: OnceLock<Mutex<Globals>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Globals::default()))
}

/// Run `f`, converting a Rust error or panic into a CUDA result code.
fn guard<F: FnOnce() -> Result<(), i32>>(f: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => CUDA_SUCCESS,
        Ok(Err(code)) => code,
        Err(_) => CUDA_ERROR_LAUNCH_FAILED,
    }
}

fn next_handle(g: &mut Globals) -> u64 {
    g.next += 1;
    g.next
}

// =======================================================================================
// Driver API
// =======================================================================================

/// `cuInit` — initialize the driver on the default Metal device.
#[no_mangle]
pub extern "C" fn cuInit(_flags: u32) -> i32 {
    guard(|| {
        let mut g = state().lock().unwrap();
        if g.rt.is_none() {
            g.rt = Some(CudaRuntime::new().map_err(|_| CUDA_ERROR_NOT_INITIALIZED)?);
        }
        Ok(())
    })
}

/// `cuDeviceGetCount` — always 1 (the single Apple GPU).
#[no_mangle]
pub extern "C" fn cuDeviceGetCount(count: *mut i32) -> i32 {
    guard(|| {
        if count.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        unsafe { *count = 1 };
        Ok(())
    })
}

/// `cuDeviceGet` — ordinal 0 is the only device.
#[no_mangle]
pub extern "C" fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32 {
    guard(|| {
        if device.is_null() || ordinal != 0 {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        unsafe { *device = 0 };
        Ok(())
    })
}

/// `cuDeviceGetName` — the Metal device name (e.g. "Apple M4").
#[no_mangle]
pub extern "C" fn cuDeviceGetName(name: *mut c_char, len: i32, _dev: i32) -> i32 {
    guard(|| {
        if name.is_null() || len <= 0 {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let s = rt.device_name();
        let bytes = s.as_bytes();
        let n = (len as usize - 1).min(bytes.len());
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), name as *mut u8, n);
            *name.add(n) = 0;
        }
        Ok(())
    })
}

/// `cuCtxCreate` — contexts are not modeled; returns a non-null opaque handle.
#[no_mangle]
pub extern "C" fn cuCtxCreate_v2(pctx: *mut *mut c_void, _flags: u32, _dev: i32) -> i32 {
    guard(|| {
        if pctx.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        unsafe { *pctx = 1 as *mut c_void };
        Ok(())
    })
}

/// `cuCtxDestroy` — no-op.
#[no_mangle]
pub extern "C" fn cuCtxDestroy_v2(_ctx: *mut c_void) -> i32 {
    CUDA_SUCCESS
}

/// `cuCtxSynchronize` — wait for all enqueued work.
#[no_mangle]
pub extern "C" fn cuCtxSynchronize() -> i32 {
    guard(|| {
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        rt.synchronize().map_err(|_| CUDA_ERROR_LAUNCH_FAILED)
    })
}

/// `cuModuleLoadData` — `image` is a null-terminated CUDA-C-subset source string.
#[no_mangle]
pub extern "C" fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32 {
    guard(|| {
        if module.is_null() || image.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        let src = unsafe { CStr::from_ptr(image as *const c_char) }
            .to_str()
            .map_err(|_| CUDA_ERROR_INVALID_SOURCE)?;
        let mut g = state().lock().unwrap();
        let rt = g.rt.clone().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let m = rt.load_cuda_source(src).map_err(|_| CUDA_ERROR_INVALID_SOURCE)?;
        let h = next_handle(&mut g);
        g.modules.insert(h, m);
        unsafe { *module = h as *mut c_void };
        Ok(())
    })
}

/// `cuModuleUnload`.
#[no_mangle]
pub extern "C" fn cuModuleUnload(module: *mut c_void) -> i32 {
    guard(|| {
        let mut g = state().lock().unwrap();
        g.modules.remove(&(module as u64));
        Ok(())
    })
}

/// `cuModuleGetFunction`.
#[no_mangle]
pub extern "C" fn cuModuleGetFunction(
    func: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> i32 {
    guard(|| {
        if func.is_null() || name.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        let name = unsafe { CStr::from_ptr(name) }.to_str().map_err(|_| CUDA_ERROR_INVALID_VALUE)?;
        let mut g = state().lock().unwrap();
        let m = g.modules.get(&(module as u64)).cloned().ok_or(CUDA_ERROR_INVALID_HANDLE)?;
        let f = m.function(name).map_err(|_| CUDA_ERROR_INVALID_HANDLE)?;
        let h = next_handle(&mut g);
        g.funcs.insert(h, f);
        unsafe { *func = h as *mut c_void };
        Ok(())
    })
}

/// `cuMemAlloc`.
#[no_mangle]
pub extern "C" fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32 {
    guard(|| {
        if dptr.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let p = rt.malloc(bytesize).map_err(|_| CUDA_ERROR_OUT_OF_MEMORY)?;
        unsafe { *dptr = p };
        Ok(())
    })
}

/// `cuMemFree`.
#[no_mangle]
pub extern "C" fn cuMemFree_v2(dptr: u64) -> i32 {
    guard(|| {
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        rt.free(dptr).map_err(|_| CUDA_ERROR_INVALID_VALUE)
    })
}

/// `cuMemcpyHtoD`.
#[no_mangle]
pub extern "C" fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, n: usize) -> i32 {
    guard(|| {
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let s = unsafe { std::slice::from_raw_parts(src as *const u8, n) };
        unsafe { rt.memcpy_htod(dst, s) }.map_err(|_| CUDA_ERROR_INVALID_VALUE)
    })
}

/// `cuMemcpyDtoH`.
#[no_mangle]
pub extern "C" fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, n: usize) -> i32 {
    guard(|| {
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let d = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, n) };
        unsafe { rt.memcpy_dtoh(d, src) }.map_err(|_| CUDA_ERROR_INVALID_VALUE)
    })
}

/// `cuLaunchKernel` — the real CUDA Driver API launch entry point.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn cuLaunchKernel(
    f: *mut c_void,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    _shared_mem: u32,
    _stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> i32 {
    guard(|| {
        let g = state().lock().unwrap();
        let rt = g.rt.clone().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let func = g.funcs.get(&(f as u64)).ok_or(CUDA_ERROR_INVALID_HANDLE)?;
        let args = build_args(func, kernel_params)?;
        rt.launch(func, [grid_x, grid_y, grid_z], [block_x, block_y, block_z], &args)
            .map_err(|_| CUDA_ERROR_LAUNCH_FAILED)
    })
}

/// Read each argument out of CUDA's `kernelParams` (array of pointers to the values).
fn build_args(func: &Function, kernel_params: *mut *mut c_void) -> Result<Vec<Arg>, i32> {
    let sig = func.signature();
    if sig.params.is_empty() {
        return Ok(Vec::new());
    }
    if kernel_params.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let mut args = Vec::with_capacity(sig.params.len());
    for (i, p) in sig.params.iter().enumerate() {
        let slot = unsafe { *kernel_params.add(i) };
        if slot.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        if p.is_pointer {
            let dptr = unsafe { *(slot as *const u64) };
            args.push(Arg::Ptr(dptr));
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(slot as *const u8, p.byte_size) };
            args.push(Arg::Scalar(bytes.to_vec()));
        }
    }
    Ok(args)
}

// =======================================================================================
// Runtime API
// =======================================================================================

/// `cudaGetDeviceCount`.
#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut i32) -> i32 {
    // Runtime API auto-initializes.
    if cuInit(0) != CUDA_SUCCESS {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    cuDeviceGetCount(count)
}

/// `cudaMalloc`.
#[no_mangle]
pub extern "C" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32 {
    guard(|| {
        if dev_ptr.is_null() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        ensure_init()?;
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        let p = rt.malloc(size).map_err(|_| CUDA_ERROR_OUT_OF_MEMORY)?;
        unsafe { *dev_ptr = p as *mut c_void };
        Ok(())
    })
}

/// `cudaFree`.
#[no_mangle]
pub extern "C" fn cudaFree(ptr: *mut c_void) -> i32 {
    guard(|| {
        if ptr.is_null() {
            return Ok(());
        }
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        rt.free(ptr as u64).map_err(|_| CUDA_ERROR_INVALID_VALUE)
    })
}

/// `cudaMemcpy`.
#[no_mangle]
pub extern "C" fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32 {
    guard(|| {
        let g = state().lock().unwrap();
        let rt = g.rt.as_ref().ok_or(CUDA_ERROR_NOT_INITIALIZED)?;
        match kind {
            MEMCPY_HTOD => {
                let s = unsafe { std::slice::from_raw_parts(src as *const u8, count) };
                unsafe { rt.memcpy_htod(dst as u64, s) }.map_err(|_| CUDA_ERROR_INVALID_VALUE)
            }
            MEMCPY_DTOH => {
                let d = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, count) };
                unsafe { rt.memcpy_dtoh(d, src as u64) }.map_err(|_| CUDA_ERROR_INVALID_VALUE)
            }
            MEMCPY_DTOD => {
                rt.memcpy_dtod(dst as u64, src as u64, count).map_err(|_| CUDA_ERROR_INVALID_VALUE)
            }
            0 => {
                // HostToHost
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, count) };
                Ok(())
            }
            _ => Err(CUDA_ERROR_INVALID_VALUE),
        }
    })
}

/// `cudaDeviceSynchronize`.
#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> i32 {
    cuCtxSynchronize()
}

fn ensure_init() -> Result<(), i32> {
    let mut g = state().lock().unwrap();
    if g.rt.is_none() {
        g.rt = Some(CudaRuntime::new().map_err(|_| CUDA_ERROR_NOT_INITIALIZED)?);
    }
    Ok(())
}

// ---- error strings --------------------------------------------------------------------

/// `cuGetErrorString`.
#[no_mangle]
pub extern "C" fn cuGetErrorString(error: i32, p_str: *mut *const c_char) -> i32 {
    if p_str.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *p_str = error_cstr(error).as_ptr() as *const c_char };
    CUDA_SUCCESS
}

/// `cudaGetErrorString` (returns the string directly, like the Runtime API).
#[no_mangle]
pub extern "C" fn cudaGetErrorString(error: i32) -> *const c_char {
    error_cstr(error).as_ptr() as *const c_char
}

fn error_cstr(code: i32) -> &'static [u8] {
    match code {
        CUDA_SUCCESS => b"no error\0",
        CUDA_ERROR_INVALID_VALUE => b"invalid argument\0",
        CUDA_ERROR_OUT_OF_MEMORY => b"out of memory\0",
        CUDA_ERROR_NOT_INITIALIZED => b"driver not initialized\0",
        CUDA_ERROR_INVALID_SOURCE => b"device kernel image is invalid\0",
        CUDA_ERROR_INVALID_HANDLE => b"invalid resource handle\0",
        CUDA_ERROR_LAUNCH_FAILED => b"unspecified launch failure\0",
        _ => b"unknown error\0",
    }
}
