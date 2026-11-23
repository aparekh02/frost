//! gravity — a game rendered on the GPU through frostd.
//!
//! This is the honest form of "run a game on the GPU" on macOS. A containerized *Linux*
//! game cannot use this bridge: Linux runs in a VM with no Metal access, which is the whole
//! reason Frost Bridge exists (see ARCHITECTURE.md). The game instead runs as a native
//! **capsule** — Frost Bridge's own container primitive (Seatbelt sandbox: no filesystem,
//! no network, no GPU entitlement of its own) — reaching the GPU only through the broker.
//!
//! Two GPU paths, selected with `--mode`:
//!   * `render` (default): the **hardware graphics pipeline**. The capsule builds triangle
//!     geometry for the frame (a gradient background, a floor, and one anti-aliased disc
//!     "sprite" per ball) in the arena; frostd draws it with a vertex+fragment pipeline into
//!     a texture that *aliases* the arena framebuffer. The GPU's fixed-function rasterizer
//!     writes the pixels straight into unified memory, which the capsule then reads with no
//!     copy. This is the path a real 2D game targets.
//!   * `compute`: a compute-shader software rasterizer (`gravity_render`), kept for contrast.
//!
//! Each frame: step physics (CPU) → write geometry into the arena → submit one draw/render
//! job → read the finished frame back (zero copy). Headless it reports frames/s and per-frame
//! GPU time; `--dump DIR` writes the frames as PPM images.
use fb_client::{Frost, WaitMode};
use fb_core::time::{now_ns, Summary};

/// Matches `struct GVertex` in fb-gpu/src/render.metal: 48 bytes, 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct GVertex {
    color: [f32; 4],
    pos: [f32; 2],
    center: [f32; 2],
    radius: f32,
    _pad: [f32; 3],
}

/// Matches `struct Ball` in kernels.metal (compute path): 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Ball {
    pos: [f32; 2],
    radius: f32,
    rgba: u32,
}

struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Framebuffer row stride in pixels. Must match `Gpu::fb_stride_bytes` in the daemon: a
/// buffer-backed texture needs its rows aligned (>= 256 B) on Apple Silicon.
fn stride_px(w: usize) -> usize {
    ((w * 4 + 255) / 256 * 256) / 4
}

fn pack(r: u8, g: u8, b: u8) -> u32 {
    0xff00_0000 | (b as u32) << 16 | (g as u32) << 8 | r as u32
}

const PALETTE: [(u8, u8, u8); 6] = [
    (0xff, 0x6b, 0x6b), (0x4e, 0xcd, 0xc4), (0xff, 0xd9, 0x3d),
    (0xa8, 0x6b, 0xff), (0x6b, 0xcb, 0xff), (0xff, 0x9f, 0x43),
];

fn main() -> anyhow::Result<()> {
    let w: usize = workloads::arg("--width", 854);
    let h: usize = workloads::arg("--height", 480);
    let n_balls: usize = workloads::arg("--balls", 16);
    let frames: usize = workloads::arg("--frames", 1200);
    let render_mode = workloads::arg("--mode", "render".to_string()) != "compute";
    let dump: Option<String> = {
        let v: String = workloads::arg("--dump", String::new());
        (!v.is_empty()).then_some(v)
    };
    let dump_every: usize = workloads::arg("--dump-every", 1);

    let mut fb = Frost::connect()?;
    // --spin = busy-wait for each frame (lowest latency, keeps the GPU warm); default parks
    // (saves CPU between frames but lets the GPU downclock).
    fb.wait_mode = if workloads::flag("--spin") { WaitMode::Spin } else { WaitMode::Park { spin_ns: 50_000 } };
    if workloads::flag("--rt") {
        fb.enter_realtime()?;
    }
    let sp = if render_mode { stride_px(w) } else { w };
    // Pipeline depth: keep this many frames in flight at once (multi-buffered) so throughput
    // is bound by the GPU, not the per-frame submit→complete round trip. Dumping needs frames
    // read back in order, so it forces depth 1. Compute mode stays single-buffered.
    let mut depth: usize = if render_mode { workloads::arg("--depth", 1usize).max(1) } else { 1 };
    if dump.is_some() {
        depth = 1;
    }
    eprintln!(
        "gravity: {}, {w}x{h}, {n_balls} balls, {frames} frames, depth {depth} on {}",
        if render_mode { "hardware render pipeline" } else { "compute rasterizer" },
        fb.gpu_name
    );

    // One framebuffer + one vertex buffer per in-flight slot (GPU writes fb, CPU writes verts).
    let max_verts = 12 + 6 * n_balls; // background quad + floor quad + 6 per ball
    let fbufs: Vec<_> = (0..depth).map(|_| fb.alloc::<u32>(h * sp)).collect::<anyhow::Result<Vec<_>>>()?;
    let vbufs: Vec<_> = if render_mode {
        (0..depth).map(|_| fb.alloc::<GVertex>(max_verts)).collect::<anyhow::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let scene = if render_mode { None } else { Some(fb.alloc::<Ball>(n_balls)?) }; // compute only

    // --- initial state (pixels; velocities in px/frame)
    let floor_y = h as f32 - h as f32 * 0.08;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut pos = vec![[0f32; 2]; n_balls];
    let mut vel = vec![[0f32; 2]; n_balls];
    let mut rad = vec![0f32; n_balls];
    let mut rgb = vec![[0f32; 4]; n_balls];
    let mut rgba = vec![0u32; n_balls];
    for i in 0..n_balls {
        rad[i] = 12.0 + rng.f32() * 28.0;
        pos[i] = [rad[i] + rng.f32() * (w as f32 - 2.0 * rad[i]), rng.f32() * floor_y * 0.6];
        vel[i] = [(rng.f32() - 0.5) * 10.0, rng.f32() * 3.0];
        let (r, g, b) = PALETTE[i % PALETTE.len()];
        rgb[i] = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0];
        rgba[i] = pack(r, g, b);
    }

    const G: f32 = 0.45;
    const REST: f32 = 0.82;
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).ok();
    }

    let mut latency = Vec::with_capacity(frames);
    let mut gpu = Vec::with_capacity(frames);
    let mut checksum = 0u64;
    // In-flight frames: (seq, slot, submit time, frame index). Up to `depth` outstanding.
    let mut inflight: std::collections::VecDeque<(u64, usize, u64, usize)> = std::collections::VecDeque::new();
    let scene_buf = scene; // Option<BufRef>, Copy
    let t_start = now_ns();

    // Retire one in-flight frame: wait it, record timing, and (depth 1) read it back.
    let mut retire = |fb: &mut Frost, inflight: &mut std::collections::VecDeque<(u64, usize, u64, usize)>,
                      gpu: &mut Vec<u64>, latency: &mut Vec<u64>| -> anyhow::Result<()> {
        let (seq, slot, t_sub, fidx) = inflight.pop_front().unwrap();
        let c = fb.wait(seq)?;
        gpu.push(c.gpu_ns);
        if depth == 1 {
            latency.push(now_ns() - t_sub);
        }
        if let (Some(d), true) = (&dump, fidx % dump_every == 0) {
            let pixels = fb.view_mut::<u32>(fbufs[slot]);
            write_ppm(&format!("{d}/frame_{fidx:05}.ppm"), pixels, w, h, sp)?;
        }
        Ok(())
    };

    for frame in 0..frames {
        // 1) physics
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
                vel[i][0] *= 0.99;
            }
        }

        // 2) free this frame's slot by retiring the oldest in-flight frame if we're full
        if inflight.len() >= depth {
            retire(&mut fb, &mut inflight, &mut gpu, &mut latency)?;
        }
        let slot = frame % depth;

        // 3) build geometry / scene into this slot's buffers, 4) submit (don't wait yet)
        let t_sub = now_ns();
        let seq = if render_mode {
            let top = [0.05, 0.07, 0.12, 1.0];
            let bot = [0.10, 0.13, 0.22, 1.0];
            let floor = [0.18, 0.20, 0.26, 1.0];
            let v = fb.view_mut::<GVertex>(vbufs[slot]);
            let mut n = 0usize;
            quad_grad(v, &mut n, 0.0, 0.0, w as f32, h as f32, top, bot);
            quad(v, &mut n, 0.0, floor_y, w as f32, h as f32, floor);
            for i in 0..n_balls {
                sprite(v, &mut n, pos[i][0], pos[i][1], rad[i], rgb[i]);
            }
            fb.submit(
                fb.job("draw_triangles").arg(fbufs[slot]).arg(vbufs[slot])
                    .param(w as u32).param(h as u32).param(n as u32).param(sp as u32),
            )
        } else {
            let sv = fb.view_mut::<Ball>(scene_buf.unwrap());
            for i in 0..n_balls {
                sv[i] = Ball { pos: pos[i], radius: rad[i], rgba: rgba[i] };
            }
            fb.submit(
                fb.job("gravity_render").arg(fbufs[0]).arg(scene_buf.unwrap())
                    .grid3([w as u32, h as u32, 1])
                    .param(w as u32).param(h as u32).param(n_balls as u32).param(frame as u32),
            )
        };
        inflight.push_back((seq, slot, t_sub, frame));
    }
    // 5) drain remaining in-flight frames
    while !inflight.is_empty() {
        retire(&mut fb, &mut inflight, &mut gpu, &mut latency)?;
    }
    // checksum the final frame (zero-copy readback)
    if frames > 0 {
        let pixels = fb.view_mut::<u32>(fbufs[(frames - 1) % depth]);
        checksum = fnv(pixels, w, h, sp);
    }

    let wall = now_ns() - t_start;
    let fps = frames as f64 / (wall as f64 / 1e9);
    // Per-frame latency is only meaningful at depth 1 (pipelined frames overlap); at higher
    // depth the headline metric is throughput (fps).
    let lat = (!latency.is_empty()).then(|| Summary::of(&mut latency));
    let g = Summary::of(&mut gpu);
    match &lat {
        Some(l) => println!("gravity: {fps:.0} fps  frame latency {l}"),
        None => println!("gravity: {fps:.0} fps  (depth {depth}, pipelined — see fps)"),
    }
    println!("   gpu render time             : {g}");
    if let Some(d) = &dump {
        eprintln!("gravity: wrote PPM frames to {d}");
    }
    workloads::result(serde_json::json!({"gravity": {
        "mode": if render_mode { "render" } else { "compute" },
        "fps": fps,
        "frames": frames,
        "depth": depth,
        "width": w, "height": h, "balls": n_balls,
        "frame_latency": lat.as_ref().map(workloads::summary_json),
        "gpu_render": workloads::summary_json(&g),
        "checksum": checksum,
    }}));
    Ok(())
}

fn put(v: &mut [GVertex], n: &mut usize, pos: [f32; 2], center: [f32; 2], radius: f32, color: [f32; 4]) {
    v[*n] = GVertex { color, pos, center, radius, _pad: [0.0; 3] };
    *n += 1;
}
const Z: [f32; 2] = [0.0, 0.0];
fn quad(v: &mut [GVertex], n: &mut usize, x0: f32, y0: f32, x1: f32, y1: f32, c: [f32; 4]) {
    for p in [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]] {
        put(v, n, p, Z, 0.0, c);
    }
}
fn quad_grad(v: &mut [GVertex], n: &mut usize, x0: f32, y0: f32, x1: f32, y1: f32, top: [f32; 4], bot: [f32; 4]) {
    for (p, c) in [([x0, y0], top), ([x1, y0], top), ([x1, y1], bot), ([x0, y0], top), ([x1, y1], bot), ([x0, y1], bot)] {
        put(v, n, p, Z, 0.0, c);
    }
}
fn sprite(v: &mut [GVertex], n: &mut usize, cx: f32, cy: f32, r: f32, c: [f32; 4]) {
    let ctr = [cx, cy];
    let (x0, y0, x1, y1) = (cx - r, cy - r, cx + r, cy + r);
    for p in [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]] {
        put(v, n, p, ctr, r, c);
    }
}

fn fnv(px: &[u32], w: usize, h: usize, stride: usize) -> u64 {
    let mut hsh = 0xcbf29ce484222325u64;
    for y in 0..h {
        for x in 0..w {
            hsh = (hsh ^ px[y * stride + x] as u64).wrapping_mul(0x100000001b3);
        }
    }
    hsh
}

fn write_ppm(path: &str, px: &[u32], w: usize, h: usize, stride: usize) -> anyhow::Result<()> {
    use std::io::Write;
    let mut out = Vec::with_capacity(w * h * 3 + 32);
    write!(out, "P6\n{w} {h}\n255\n")?;
    for y in 0..h {
        for x in 0..w {
            let p = px[y * stride + x];
            out.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8]);
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}
