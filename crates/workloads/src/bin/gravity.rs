//! gravity — a tiny interactive game rendered on the GPU through frostd.
//!
//! This is the honest, buildable form of "a GPU bridge for running games" on macOS. A
//! containerized *Linux* game cannot use this bridge — Linux runs in a VM with no Metal
//! access, which is the whole reason Frost Bridge exists (see ARCHITECTURE.md). Instead the
//! game runs as a native **capsule**: Frost Bridge's own container primitive (Seatbelt
//! sandbox, no filesystem, no network, no GPU entitlement of its own). It reaches the GPU
//! only through the broker.
//!
//! Each frame the capsule:
//!   1. steps the ball physics on the CPU (cheap, deterministic),
//!   2. writes the ball state into a `scene` buffer in the shared arena,
//!   3. submits one `gravity_render` job — the GPU rasterizes the frame into a framebuffer
//!      that lives in the *same* unified-memory arena (zero copy, no readback),
//!   4. reads the finished frame back with a plain slice (it is already in its address space).
//!
//! Run headless it reports frames/s and per-frame GPU latency; with `--dump DIR` it writes
//! the frames as PPM images you can actually look at.
use fb_client::{Frost, WaitMode};
use fb_core::time::{now_ns, Summary};

/// Matches `struct Ball` in kernels.metal: 16 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Ball {
    pos: [f32; 2],
    radius: f32,
    rgba: u32,
}

/// Deterministic little LCG so a given seed renders a byte-identical sequence (tested).
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn pack(r: u8, g: u8, b: u8) -> u32 {
    0xff00_0000 | (b as u32) << 16 | (g as u32) << 8 | r as u32
}

fn main() -> anyhow::Result<()> {
    let w: usize = workloads::arg("--width", 854);
    let h: usize = workloads::arg("--height", 480);
    let n_balls: usize = workloads::arg("--balls", 12);
    let frames: usize = workloads::arg("--frames", 1200);
    let dump: Option<String> = {
        let v: String = workloads::arg("--dump", String::new());
        (!v.is_empty()).then_some(v)
    };
    let dump_every: usize = workloads::arg("--dump-every", 1);

    let mut fb = Frost::connect()?;
    fb.wait_mode = WaitMode::Park { spin_ns: 50_000 };
    if workloads::flag("--rt") {
        fb.enter_realtime()?;
    }
    eprintln!(
        "gravity: {w}x{h}, {n_balls} balls, {frames} frames on {} (arena {} MiB free)",
        fb.gpu_name,
        fb.data_remaining() >> 20
    );

    // Two arena buffers: the framebuffer (GPU writes, CPU reads) and the scene (CPU writes).
    let fbuf = fb.alloc::<u32>(w * h)?;
    let scene = fb.alloc::<Ball>(n_balls)?;

    // --- initial state (positions in pixel space, velocities in px/frame)
    let floor_y = h as f32 - h as f32 * 0.08;
    let palette = [
        (0xff, 0x6b, 0x6b), (0x4e, 0xcd, 0xc4), (0xff, 0xd9, 0x3d),
        (0xa8, 0x6b, 0xff), (0x6b, 0xcB, 0xff), (0xff, 0x9f, 0x43),
    ];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut pos = vec![[0f32; 2]; n_balls];
    let mut vel = vec![[0f32; 2]; n_balls];
    let mut rad = vec![0f32; n_balls];
    let mut col = vec![0u32; n_balls];
    for i in 0..n_balls {
        rad[i] = 14.0 + rng.next_f32() * 30.0;
        pos[i] = [rad[i] + rng.next_f32() * (w as f32 - 2.0 * rad[i]), rng.next_f32() * floor_y * 0.6];
        vel[i] = [(rng.next_f32() - 0.5) * 10.0, rng.next_f32() * 3.0];
        let (r, g, b) = palette[i % palette.len()];
        col[i] = pack(r, g, b);
    }

    const G: f32 = 0.45; // gravity, px/frame^2
    const REST: f32 = 0.82; // restitution

    if let Some(d) = &dump {
        std::fs::create_dir_all(d).ok();
    }

    let mut latency = Vec::with_capacity(frames);
    let mut gpu = Vec::with_capacity(frames);
    let mut checksum: u64 = 0;
    let t_start = now_ns();

    for frame in 0..frames {
        // 1) step physics
        for i in 0..n_balls {
            vel[i][1] += G;
            pos[i][0] += vel[i][0];
            pos[i][1] += vel[i][1];
            let r = rad[i];
            if pos[i][0] < r {
                pos[i][0] = r;
                vel[i][0] = -vel[i][0] * REST;
            } else if pos[i][0] > w as f32 - r {
                pos[i][0] = w as f32 - r;
                vel[i][0] = -vel[i][0] * REST;
            }
            if pos[i][1] > floor_y - r {
                pos[i][1] = floor_y - r;
                vel[i][1] = -vel[i][1] * REST;
                vel[i][0] *= 0.99; // friction on bounce
            }
        }
        // 2) publish the scene into the arena
        let sv = fb.view_mut::<Ball>(scene);
        for i in 0..n_balls {
            sv[i] = Ball { pos: pos[i], radius: rad[i], rgba: col[i] };
        }
        // 3) render this frame on the GPU
        let t0 = now_ns();
        let c = fb.run(
            fb.job("gravity_render")
                .arg(fbuf)
                .arg(scene)
                .grid3([w as u32, h as u32, 1])
                .param(w as u32)
                .param(h as u32)
                .param(n_balls as u32)
                .param(frame as u32),
        )?;
        let t1 = now_ns();
        latency.push(t1 - t0);
        gpu.push(c.gpu_ns);

        // 4) read the finished frame back (zero copy) — checksum + optional PPM dump
        let pixels = fb.view_mut::<u32>(fbuf);
        if frame + 1 == frames {
            checksum = fnv(pixels);
        }
        if let (Some(d), true) = (&dump, frame % dump_every == 0) {
            write_ppm(&format!("{d}/frame_{frame:05}.ppm"), pixels, w, h)?;
        }
    }

    let wall = now_ns() - t_start;
    let fps = frames as f64 / (wall as f64 / 1e9);
    let lat = Summary::of(&mut latency);
    let g = Summary::of(&mut gpu);
    println!("gravity: {fps:.0} fps  frame latency {lat}");
    println!("   gpu render time             : {g}");
    if let Some(d) = &dump {
        eprintln!("gravity: wrote PPM frames to {d}");
    }
    workloads::result(serde_json::json!({"gravity": {
        "fps": fps,
        "frames": frames,
        "width": w, "height": h, "balls": n_balls,
        "frame_latency": workloads::summary_json(&lat),
        "gpu_render": workloads::summary_json(&g),
        "checksum": checksum,
    }}));
    Ok(())
}

fn fnv(px: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &p in px {
        h = (h ^ p as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn write_ppm(path: &str, px: &[u32], w: usize, h: usize) -> anyhow::Result<()> {
    use std::io::Write;
    let mut out = Vec::with_capacity(w * h * 3 + 32);
    write!(out, "P6\n{w} {h}\n255\n")?;
    for &p in &px[..w * h] {
        out.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8]);
    }
    std::fs::write(path, out)?;
    Ok(())
}
