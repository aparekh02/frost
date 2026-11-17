use fb_core::abi::{kernel, BufRef, Job, Status};
use fb_core::shm::Arena;
use fb_gpu::{Gpu, Window};
use sha2::{Digest, Sha256};

fn arena(len: usize) -> (Arena, std::path::PathBuf) {
    let p = std::env::temp_dir().join(format!("fbgpu-test-{}-{}", std::process::id(), fb_core::time::ticks()));
    (Arena::create(&p, len).unwrap(), p)
}

fn job(k: u32, args: &[(u64, u64)], grid: u32, params: &[u32]) -> Job {
    let mut j = Job { kernel: k, n_args: args.len() as u16, grid: [grid, 1, 1], ..Default::default() };
    for (i, &(off, len)) in args.iter().enumerate() {
        j.args[i] = BufRef { off, len };
    }
    j.params[..params.len()].copy_from_slice(params);
    j
}

#[test]
fn vadd_zero_copy_on_file_backed_arena() {
    let g = Gpu::new().unwrap();
    assert!(g.unified_memory());
    let (a, p) = arena(1 << 20);
    let n = 10_000usize;
    let (oa, ob, oc) = (0u64, 65536u64, 131072u64);
    for (i, v) in a.typed::<f32>(oa as usize, n).iter_mut().enumerate() { *v = i as f32; }
    for (i, v) in a.typed::<f32>(ob as usize, n).iter_mut().enumerate() { *v = 2.0 * i as f32; }
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    let win = Window { off: 0, len: a.len() as u64 };
    let mut b = g.batch().unwrap();
    let bytes = (n * 4) as u64;
    b.encode(&buf, win, &job(kernel::VADD_F32, &[(oa, bytes), (ob, bytes), (oc, bytes)], n as u32, &[n as u32])).unwrap();
    let d = b.commit().wait();
    assert!(d.ok, "{:?}", d.error);
    let c = a.typed::<f32>(oc as usize, n);
    for i in 0..n { assert_eq!(c[i], 3.0 * i as f32); }
    std::fs::remove_file(p).ok();
}

#[test]
fn sha256_matches_cpu() {
    let g = Gpu::new().unwrap();
    let (a, p) = arena(4 << 20);
    for &len in &[0usize, 3, 55, 56, 64, 100, 200] {
        let n = 1000usize;
        let stride = (len.div_ceil(16) * 16).max(16);
        let msgs = a.slice(0, n * stride).unwrap();
        for i in 0..n { for j in 0..len { msgs[i * stride + j] = (i * 31 + j * 7) as u8; } }
        let out_off = 2 << 20;
        let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
        let mut b = g.batch().unwrap();
        b.encode(&buf, Window { off: 0, len: a.len() as u64 },
            &job(kernel::SHA256_BATCH, &[(0, (n * stride) as u64), (out_off, (n * 32) as u64)], n as u32,
                 &[n as u32, len as u32, stride as u32])).unwrap();
        assert!(b.commit().wait().ok);
        let out = a.slice(out_off as usize, n * 32).unwrap();
        for i in 0..n {
            let want = Sha256::digest(&msgs[i * stride..i * stride + len]);
            assert_eq!(&out[i * 32..i * 32 + 32], want.as_slice(), "len={len} i={i}");
        }
    }
    std::fs::remove_file(p).ok();
}

#[test]
fn gemv_matches_cpu() {
    let g = Gpu::new().unwrap();
    let (rows, cols) = (513usize, 1027usize);
    let (a, p) = arena(8 << 20);
    let off_x = fb_core::round_up(rows * cols * 4, 16);
    let off_y = fb_core::round_up(off_x + cols * 4, 16);
    let m = a.typed::<f32>(0, rows * cols);
    for (i, v) in m.iter_mut().enumerate() { *v = ((i % 13) as f32 - 6.0) * 0.01; }
    let x = a.typed::<f32>(off_x, cols);
    for (i, v) in x.iter_mut().enumerate() { *v = ((i % 7) as f32) * 0.5; }
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    let mut b = g.batch().unwrap();
    let mut j = job(kernel::GEMV_F32, &[(0, (rows * cols * 4) as u64), (off_x as u64, (cols * 4) as u64), (off_y as u64, (rows * 4) as u64)],
        (rows * 32) as u32, &[rows as u32, cols as u32]);
    j.tg = [256, 1, 1];
    b.encode(&buf, Window { off: 0, len: a.len() as u64 }, &j).unwrap();
    assert!(b.commit().wait().ok);
    let y = a.typed::<f32>(off_y, rows);
    for r in 0..rows {
        let want: f32 = (0..cols).map(|c| m[r * cols + c] * x[c]).sum();
        assert!((y[r] - want).abs() < 1e-3 * want.abs().max(1.0), "row {r}: {} vs {}", y[r], want);
    }
    std::fs::remove_file(p).ok();
}

#[test]
fn rejects_out_of_window_and_bad_kernel() {
    let g = Gpu::new().unwrap();
    let (a, p) = arena(1 << 20);
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    let win = Window { off: 65536, len: 65536 };
    let mut b = g.batch().unwrap();
    assert_eq!(b.encode(&buf, win, &job(kernel::NOOP, &[(0, 16)], 1, &[])).unwrap_err(), Status::OutOfBounds);
    assert_eq!(b.encode(&buf, win, &job(kernel::NOOP, &[(65536 + 65530, 16)], 1, &[])).unwrap_err(), Status::OutOfBounds);
    assert_eq!(b.encode(&buf, win, &job(kernel::NOOP, &[(u64::MAX - 4, 16)], 1, &[])).unwrap_err(), Status::OutOfBounds);
    assert_eq!(b.encode(&buf, win, &job(999, &[(65536, 16)], 1, &[])).unwrap_err(), Status::BadKernel);
    b.encode(&buf, win, &job(kernel::NOOP, &[(65536, 16)], 1, &[41])).unwrap();
    assert!(b.commit().wait().ok);
    assert_eq!(a.typed::<u32>(65536, 1)[0], 42);
    std::fs::remove_file(p).ok();
}

#[test]
fn custom_kernel_registration() {
    let g = Gpu::new().unwrap();
    let id = g.register("triple", r#"
        #include <metal_stdlib>
        kernel void triple(device uint* o [[buffer(0)]], constant uint* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if (i < p[0]) o[i] = i * 3;
        }"#, "triple").unwrap();
    assert!(id >= kernel::FIRST_CUSTOM);
    assert!(g.register("triple", "x", "y").is_err());
    assert!(g.register("broken", "this is not msl", "broken").is_err());
    let (a, p) = arena(1 << 16);
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    let mut b = g.batch().unwrap();
    b.encode(&buf, Window { off: 0, len: a.len() as u64 }, &job(id, &[(0, 4096)], 1024, &[1024])).unwrap();
    assert!(b.commit().wait().ok);
    assert_eq!(a.typed::<u32>(0, 1024)[1023], 3069);
    std::fs::remove_file(p).ok();
}

#[test]
fn split_sha256_and_gemv_match_unsplit() {
    let g = Gpu::new().unwrap();
    let (a, p) = arena(8 << 20);
    let win = Window { off: 0, len: a.len() as u64 };
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    // sha256: 5000 msgs of 40 B, stride 48; outputs at 1 MiB (whole) and 2 MiB (split)
    let n = 5000u32;
    for (i, b) in a.slice(0, (n * 48) as usize).unwrap().iter_mut().enumerate() { *b = (i * 7) as u8; }
    let whole = job(kernel::SHA256_BATCH, &[(0, (n * 48) as u64), (1 << 20, (n * 32) as u64)], n, &[n, 40, 48]);
    let mut parts_src = whole;
    parts_src.args[1].off = 2 << 20;
    let parts = fb_gpu::split_job(&parts_src, 1024).unwrap();
    assert_eq!(parts.len(), 5);
    let mut b = g.batch().unwrap();
    b.encode(&buf, win, &whole).unwrap();
    for pj in &parts { b.encode(&buf, win, pj).unwrap(); }
    assert!(b.commit().wait().ok);
    assert_eq!(a.slice(1 << 20, (n * 32) as usize).unwrap().to_vec(), a.slice(2 << 20, (n * 32) as usize).unwrap().to_vec());
    // gemv 300x64 split by rows
    let (rows, cols) = (300u32, 64u32);
    let m_off = 3 << 20; let x_off = 4 << 20; let y1 = 5 << 20; let y2 = 6 << 20;
    for (i, v) in a.typed::<f32>(m_off, (rows * cols) as usize).iter_mut().enumerate() { *v = (i % 5) as f32; }
    for v in a.typed::<f32>(x_off, cols as usize).iter_mut() { *v = 0.5; }
    let mk = |y: usize| { let mut j = job(kernel::GEMV_F32, &[(m_off as u64, (rows * cols * 4) as u64), (x_off as u64, (cols * 4) as u64), (y as u64, (rows * 4) as u64)], rows * 32, &[rows, cols]); j.tg = [64, 1, 1]; j };
    let parts = fb_gpu::split_job(&mk(y2), 64).unwrap();
    assert_eq!(parts.len(), 5);
    let mut b = g.batch().unwrap();
    b.encode(&buf, win, &mk(y1)).unwrap();
    for pj in &parts { b.encode(&buf, win, pj).unwrap(); }
    assert!(b.commit().wait().ok);
    assert_eq!(a.typed::<f32>(y1, rows as usize).to_vec(), a.typed::<f32>(y2, rows as usize).to_vec());
    assert!(fb_gpu::split_job(&job(kernel::NOOP, &[(0, 16)], 1, &[]), 1).is_none());
    std::fs::remove_file(p).ok();
}

#[test]
fn dropping_uncommitted_batch_is_safe() {
    let g = Gpu::new().unwrap();
    let (a, p) = arena(1 << 16);
    let buf = unsafe { g.wrap(a.as_ptr(), a.len()).unwrap() };
    {
        let mut b = g.batch().unwrap();
        assert!(b.encode(&buf, Window { off: 0, len: 16 }, &job(kernel::NOOP, &[(4096, 16)], 1, &[])).is_err());
    } // dropped without commit: must not abort
    {
        let mut b = g.batch().unwrap();
        b.encode(&buf, Window { off: 0, len: a.len() as u64 }, &job(kernel::NOOP, &[(0, 16)], 1, &[1])).unwrap();
    } // encoded work, never committed
    let mut b = g.batch().unwrap();
    b.encode(&buf, Window { off: 0, len: a.len() as u64 }, &job(kernel::NOOP, &[(0, 16)], 1, &[5])).unwrap();
    assert!(b.commit().wait().ok);
    assert_eq!(a.typed::<u32>(0, 1)[0], 6);
    std::fs::remove_file(p).ok();
}
