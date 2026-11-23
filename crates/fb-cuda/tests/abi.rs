//! Drive the C ABI (CUDA Driver API subset) exactly as a C program would, verifying the
//! whole path: cuInit -> cuModuleLoadData -> cuModuleGetFunction -> cuMemAlloc ->
//! cuMemcpyHtoD -> cuLaunchKernel -> cuCtxSynchronize -> cuMemcpyDtoH.

use fb_cuda::driver::*;
use std::ffi::{c_void, CString};
use std::ptr;

const SRC: &str = concat!(
    "extern \"C\" __global__ void vadd(const float* a, const float* b, float* c, int n){\n",
    "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n",
    "  if (i < n) c[i] = a[i] + b[i];\n",
    "}\0" // null-terminated for cuModuleLoadData
);

#[test]
fn driver_api_vadd_end_to_end() {
    unsafe {
        assert_eq!(cuInit(0), CUDA_SUCCESS);

        let mut count = 0i32;
        assert_eq!(cuDeviceGetCount(&mut count), CUDA_SUCCESS);
        assert_eq!(count, 1);

        let mut dev = -1i32;
        assert_eq!(cuDeviceGet(&mut dev, 0), CUDA_SUCCESS);

        let mut name = [0u8; 64];
        assert_eq!(cuDeviceGetName(name.as_mut_ptr() as *mut i8, 64, dev), CUDA_SUCCESS);
        let dev_name = std::ffi::CStr::from_ptr(name.as_ptr() as *const i8).to_string_lossy();
        println!("driver reports device: {dev_name}");
        assert!(!dev_name.is_empty());

        let mut ctx: *mut c_void = ptr::null_mut();
        assert_eq!(cuCtxCreate_v2(&mut ctx, 0, dev), CUDA_SUCCESS);

        let mut module: *mut c_void = ptr::null_mut();
        assert_eq!(
            cuModuleLoadData(&mut module, SRC.as_ptr() as *const c_void),
            CUDA_SUCCESS,
            "module load (CUDA->MSL compile) failed"
        );

        let fname = CString::new("vadd").unwrap();
        let mut func: *mut c_void = ptr::null_mut();
        assert_eq!(cuModuleGetFunction(&mut func, module, fname.as_ptr()), CUDA_SUCCESS);

        let n: i32 = 2048;
        let bytes = n as usize * 4;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();

        let (mut da, mut db, mut dc) = (0u64, 0u64, 0u64);
        assert_eq!(cuMemAlloc_v2(&mut da, bytes), CUDA_SUCCESS);
        assert_eq!(cuMemAlloc_v2(&mut db, bytes), CUDA_SUCCESS);
        assert_eq!(cuMemAlloc_v2(&mut dc, bytes), CUDA_SUCCESS);

        assert_eq!(cuMemcpyHtoD_v2(da, a.as_ptr() as *const c_void, bytes), CUDA_SUCCESS);
        assert_eq!(cuMemcpyHtoD_v2(db, b.as_ptr() as *const c_void, bytes), CUDA_SUCCESS);

        // kernelParams: array of pointers to each argument's value (CUDA convention).
        let mut p_da = da;
        let mut p_db = db;
        let mut p_dc = dc;
        let mut p_n = n;
        let mut params: [*mut c_void; 4] = [
            &mut p_da as *mut u64 as *mut c_void,
            &mut p_db as *mut u64 as *mut c_void,
            &mut p_dc as *mut u64 as *mut c_void,
            &mut p_n as *mut i32 as *mut c_void,
        ];

        let block = 256u32;
        let grid = (n as u32).div_ceil(block);
        assert_eq!(
            cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()),
            CUDA_SUCCESS,
            "launch failed"
        );
        assert_eq!(cuCtxSynchronize(), CUDA_SUCCESS);

        let mut c = vec![0f32; n as usize];
        assert_eq!(cuMemcpyDtoH_v2(c.as_mut_ptr() as *mut c_void, dc, bytes), CUDA_SUCCESS);

        for i in 0..n as usize {
            assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
        }

        assert_eq!(cuMemFree_v2(da), CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(db), CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(dc), CUDA_SUCCESS);
        assert_eq!(cuModuleUnload(module), CUDA_SUCCESS);
    }
}

#[test]
fn runtime_api_malloc_memcpy() {
    let mut dptr: *mut c_void = ptr::null_mut();
    assert_eq!(cudaMalloc(&mut dptr, 256), CUDA_SUCCESS);
    let src = [7u8; 256];
    let mut dst = [0u8; 256];
    assert_eq!(cudaMemcpy(dptr, src.as_ptr() as *const c_void, 256, 1), CUDA_SUCCESS); // H2D
    assert_eq!(cudaMemcpy(dst.as_mut_ptr() as *mut c_void, dptr, 256, 2), CUDA_SUCCESS); // D2H
    assert_eq!(dst, src);
    assert_eq!(cudaFree(dptr), CUDA_SUCCESS);
}
