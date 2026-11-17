//! Adversarial capsule: tries to make the GPU touch memory outside its arena,
//! use bogus kernels, and register kernels it is not allowed to.
use fb_client::Frost;
use fb_core::abi::BufRef;

fn main() -> anyhow::Result<()> {
    let mut fb = Frost::connect()?;
    if workloads::flag("--corrupt-ring") {
        // Scribble an impossible tail index into our own submit ring, then hang around.
        let hdr = unsafe { &*(fb.arena().as_ptr() as *const fb_core::abi::ArenaHeader) };
        let tail = unsafe { &*(fb.arena().as_ptr().add(hdr.submit_off as usize) as *const std::sync::atomic::AtomicU64) };
        tail.store(u64::MAX / 2, std::sync::atomic::Ordering::Release);
        std::thread::sleep(std::time::Duration::from_secs(10));
        workloads::result(serde_json::json!({"corrupt_survived": true}));
        return Ok(());
    }
    if workloads::flag("--garbage-flood") {
        // Valid ring, but a flood of chained, rejected jobs.
        let bad = BufRef { off: 0, len: 16 };
        let mut rejected = 0u64;
        for _ in 0..1000 {
            let mut seqs = vec![];
            for _ in 0..50 {
                seqs.extend(fb.submit_chain(vec![fb.job("noop").arg(bad).grid(1), fb.job("noop").arg(bad).grid(1)]));
            }
            for s in seqs {
                if fb.wait(s).is_err() { rejected += 1; }
            }
        }
        eprintln!("garbage rejected {rejected}");
        workloads::result(serde_json::json!({"garbage_done": true}));
        return Ok(());
    }
    let ok = fb.alloc::<u32>(4)?;
    let header = BufRef { off: 0, len: 4096 }; // arena header + rings: not in data window
    let huge = BufRef { off: ok.off, len: 1 << 40 };
    let wrap = BufRef { off: u64::MAX - 8, len: 64 };
    let unaligned = BufRef { off: ok.off + 4, len: 16 };
    let mut res = serde_json::Map::new();
    for (name, r) in [("header", header), ("huge", huge), ("wrap", wrap), ("unaligned", unaligned)] {
        let e = fb.run(fb.job("noop").arg(r).grid(1)).err().map(|e| e.to_string());
        res.insert(name.into(), serde_json::json!(e));
    }
    let bad_kernel = fb.run(fb.job("does_not_exist").arg(ok).grid(1)).err().map(|e| e.to_string());
    res.insert("bad_kernel".into(), serde_json::json!(bad_kernel));
    let reg = fb.register_kernel("x", "kernel void x(){}", "x").err().map(|e| e.to_string());
    res.insert("register".into(), serde_json::json!(reg));
    // still functional afterwards?
    fb.run(fb.job("noop").arg(ok).grid(1).param(7))?;
    res.insert("still_works".into(), serde_json::json!(fb.view_mut::<u32>(ok)[0] == 8));
    workloads::result(serde_json::json!({"evil": res}));
    Ok(())
}
