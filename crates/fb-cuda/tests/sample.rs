//! Issue #9, milestone 2: "run an unmodified CUDA sample."
//!
//! This drives the canonical NVIDIA CUDA SDK `vectorAdd` sample through the CUDA Driver
//! API subset, with the device kernel pasted **verbatim** from the toolkit sample
//! (samples/0_Introduction/vectorAdd/vectorAdd.cu). Nothing in the kernel string is
//! altered; the host side is the sample's own cuMemAlloc -> cuMemcpyHtoD -> launch ->
//! cuMemcpyDtoH sequence, expressed against our libcudart/libcuda-compatible shim.
//!
//! It proves a real, unmodified CUDA kernel compiles (CUDA-C subset -> MSL) and runs
//! correctly on the Apple GPU via Metal.

use fb_cuda::driver::*;
use std::ffi::{c_void, CString};
use std::ptr;

/// Verbatim device code from the NVIDIA `vectorAdd` sample (the `+ 0.0f` is theirs).
/// Only the trailing NUL is added, because cuModuleLoadData takes a C string image.
const VECTORADD_CU: &str = concat!(
    "/**\n",
    " * CUDA Kernel Device code\n",
    " *\n",
    " * Computes the vector addition of A and B into C. The 3 vectors have the same\n",
    " * number of elements numElements.\n",
    " */\n",
    "extern \"C\" __global__ void vectorAdd(const float *A, const float *B,\n",
    "                                      float *C, int numElements) {\n",
    "  int i = blockDim.x * blockIdx.x + threadIdx.x;\n",
    "\n",
    "  if (i < numElements) {\n",
    "    C[i] = A[i] + B[i] + 0.0f;\n",
    "  }\n",
    "}\n\0"
);

#[test]
fn cuda_sdk_vectoradd_sample_unmodified() {
    {
        // --- host setup, as in the sample's main() ---
        assert_eq!(cuInit(0), CUDA_SUCCESS);
        let mut dev = -1i32;
        assert_eq!(cuDeviceGet(&mut dev, 0), CUDA_SUCCESS);
        let mut ctx: *mut c_void = ptr::null_mut();
        assert_eq!(cuCtxCreate_v2(&mut ctx, 0, dev), CUDA_SUCCESS);

        let mut module: *mut c_void = ptr::null_mut();
        assert_eq!(
            cuModuleLoadData(&mut module, VECTORADD_CU.as_ptr() as *const c_void),
            CUDA_SUCCESS,
            "compiling the unmodified NVIDIA vectorAdd sample (CUDA-C -> MSL) failed"
        );
        let fname = CString::new("vectorAdd").unwrap();
        let mut func: *mut c_void = ptr::null_mut();
        assert_eq!(cuModuleGetFunction(&mut func, module, fname.as_ptr()), CUDA_SUCCESS);

        // The sample uses 50,000 elements.
        let num_elements: i32 = 50_000;
        let size = num_elements as usize * std::mem::size_of::<f32>();

        // Host input vectors, initialized exactly as the sample does (rand()/RAND_MAX in
        // the original; here deterministic values so we can assert exact equality).
        let h_a: Vec<f32> = (0..num_elements).map(|i| (i as f32) * 0.5).collect();
        let h_b: Vec<f32> = (0..num_elements).map(|i| (i as f32) * 0.25).collect();

        let (mut d_a, mut d_b, mut d_c) = (0u64, 0u64, 0u64);
        assert_eq!(cuMemAlloc_v2(&mut d_a, size), CUDA_SUCCESS);
        assert_eq!(cuMemAlloc_v2(&mut d_b, size), CUDA_SUCCESS);
        assert_eq!(cuMemAlloc_v2(&mut d_c, size), CUDA_SUCCESS);

        assert_eq!(cuMemcpyHtoD_v2(d_a, h_a.as_ptr() as *const c_void, size), CUDA_SUCCESS);
        assert_eq!(cuMemcpyHtoD_v2(d_b, h_b.as_ptr() as *const c_void, size), CUDA_SUCCESS);

        // Launch config straight from the sample:
        //   threadsPerBlock = 256; blocksPerGrid = (numElements + tpb - 1) / tpb;
        let threads_per_block: u32 = 256;
        let blocks_per_grid: u32 =
            (num_elements as u32 + threads_per_block - 1) / threads_per_block;

        // kernelParams: &A, &B, &C, &numElements  (pointers to the argument values).
        let mut p_a = d_a;
        let mut p_b = d_b;
        let mut p_c = d_c;
        let mut p_n = num_elements;
        let mut params: [*mut c_void; 4] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];

        assert_eq!(
            cuLaunchKernel(
                func,
                blocks_per_grid, 1, 1,
                threads_per_block, 1, 1,
                0,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ),
            CUDA_SUCCESS,
            "launching the unmodified vectorAdd sample failed"
        );
        assert_eq!(cuCtxSynchronize(), CUDA_SUCCESS);

        let mut h_c = vec![0f32; num_elements as usize];
        assert_eq!(
            cuMemcpyDtoH_v2(h_c.as_mut_ptr() as *mut c_void, d_c, size),
            CUDA_SUCCESS
        );

        // Verification loop, exactly as the sample: fabs(A+B - C) <= 1e-5.
        for i in 0..num_elements as usize {
            let expected = h_a[i] + h_b[i];
            assert!(
                (expected - h_c[i]).abs() <= 1e-5,
                "Result verification failed at element {i}: {} vs {}",
                h_c[i], expected
            );
        }
        println!("Test PASSED: unmodified NVIDIA vectorAdd sample ran on the Apple GPU via Metal");

        assert_eq!(cuMemFree_v2(d_a), CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(d_b), CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(d_c), CUDA_SUCCESS);
        assert_eq!(cuModuleUnload(module), CUDA_SUCCESS);
        assert_eq!(cuCtxDestroy_v2(ctx), CUDA_SUCCESS);
    }
}
