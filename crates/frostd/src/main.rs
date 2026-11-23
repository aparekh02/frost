//! frostd — Frost Bridge capsule supervisor and GPU broker.
//!
//! ```text
//! frostd info
//! frostd run [--rt] [--quiet] [--isolate] [--json] <capsule.toml>...
//!     --isolate: run each capsule in its own broker process (its own GPU address space)
//! frostd bench-direct [--iters N]
//! frostd bench-gravity [--frames N] [--balls N] [--width W] [--height H]  # GPU-direct + CPU baselines
//! frostd profile <capsule.toml>         # print the generated Seatbelt profile
//! frostd serve [--socket PATH] [run flags]   # long-lived daemon
//! frostd ctl [--socket PATH] launch <capsule.toml> | ps | kill <name> | shutdown
//! ```

mod daemon;

use anyhow::{bail, Context, Result};
use daemon::{Config, Daemon};
use fb_core::manifest::Manifest;
use std::path::Path;

fn main() {
    if let Err(e) = real_main() {
        eprintln!("frostd: error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        bail!("usage: frostd <info|run|bench-direct|profile> ...");
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "info" => info(),
        "run" => run(rest),
        "bench-direct" => bench_direct(rest),
        "bench-gravity" => bench_gravity(rest),
        "serve" => serve(rest),
        "ctl" => ctl(rest),
        "profile" => {
            let p = Path::new(rest.first().context("profile <capsule.toml>")?);
            let m = Manifest::load(p)?;
            let base = p.parent().unwrap_or(Path::new("."));
            let exec = std::fs::canonicalize(base.join(&m.exec)).unwrap_or_else(|_| base.join(&m.exec));
            print!("{}", fb_capsule::profile(&m, &exec, Path::new("/private/tmp/frost-root")));
            Ok(())
        }
        other => bail!("unknown command {other}"),
    }
}

fn info() -> Result<()> {
    let g = fb_gpu::Gpu::new()?;
    let (p, e) = fb_core::sched::core_topology();
    println!("gpu:              {}", g.name());
    println!("unified memory:   {}", g.unified_memory());
    println!("working set:      {} MiB recommended", g.recommended_working_set() >> 20);
    println!("cpu:              {p} performance + {e} efficiency cores");
    println!("page size:        {} B", fb_core::page_size());
    println!("kernels:          {:?}", g.kernel_names());
    Ok(())
}

fn default_socket() -> std::path::PathBuf {
    Config::default().run_dir.join("frostd.sock")
}

struct Flags {
    cfg: Config,
    json: bool,
    /// Put every capsule in its own trust domain (its own broker process).
    isolate: bool,
    socket: std::path::PathBuf,
    manifests: Vec<String>,
}

/// Parse shared daemon flags.
fn parse_flags(args: &[String]) -> Result<Flags> {
    let mut cfg = Config::default();
    let mut json = false;
    let mut isolate = false;
    let mut socket = default_socket();
    let mut manifests = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--socket" => socket = it.next().context("--socket PATH")?.into(),
            "--rt" => cfg.realtime_dispatcher = true,
            "--quiet" => cfg.quiet = true,
            "--json" => json = true,
            "--isolate" => isolate = true,
            "--no-watchdog" => cfg.watchdog = false,
            "--slice-us" => cfg.slice_us = it.next().context("--slice-us N")?.parse()?,
            "--no-slicing" => cfg.slice_us = 0,
            "--keepwarm-us" => cfg.keepwarm_us = it.next().context("--keepwarm-us N")?.parse()?,
            "--idle-spin-us" => cfg.idle_spin_us = it.next().context("--idle-spin-us N")?.parse()?,
            "--run-dir" => cfg.run_dir = it.next().context("--run-dir DIR")?.into(),
            m => manifests.push(m.to_string()),
        }
    }
    Ok(Flags { cfg, json, isolate, socket, manifests })
}

/// Flags to forward to a per-domain broker (`frostd serve`) subprocess.
fn passthrough_flags(cfg: &Config) -> Vec<String> {
    let mut v = vec!["--run-dir".to_string(), cfg.run_dir.display().to_string(), "--idle-spin-us".to_string(), cfg.idle_spin_us.to_string()];
    if cfg.realtime_dispatcher {
        v.push("--rt".into());
    }
    if cfg.quiet {
        v.push("--quiet".into());
    }
    if !cfg.watchdog {
        v.push("--no-watchdog".into());
    }
    if cfg.slice_us == 0 {
        v.push("--no-slicing".into());
    } else {
        v.push("--slice-us".into());
        v.push(cfg.slice_us.to_string());
    }
    if cfg.keepwarm_us > 0 {
        v.push("--keepwarm-us".into());
        v.push(cfg.keepwarm_us.to_string());
    }
    v
}

/// Send one JSON command to a broker's control socket and read its one-line reply.
fn broker_cmd(socket: &Path, req: &serde_json::Value) -> Result<serde_json::Value> {
    use std::io::{BufRead, BufReader, Write};
    let mut s = std::os::unix::net::UnixStream::connect(socket)
        .with_context(|| format!("connect broker {}", socket.display()))?;
    writeln!(s, "{req}")?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line)?;
    Ok(serde_json::from_str(&line)?)
}

fn load_and_launch(d: &std::sync::Arc<Daemon>, m: &str) -> Result<String> {
    let p = Path::new(m);
    let man = Manifest::load(p)?;
    let parent = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let base = std::fs::canonicalize(parent)?;
    let name = man.name.clone();
    d.launch(man, &base)?;
    Ok(name)
}

fn serve(args: &[String]) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    let f = parse_flags(args)?;
    let (cfg, socket) = (f.cfg, f.socket);
    let d = Daemon::start(cfg)?;
    let _ = std::fs::remove_file(&socket);
    let l = UnixListener::bind(&socket).with_context(|| format!("bind {}", socket.display()))?;
    // Owner-only: the socket can launch code with GPU access.
    std::fs::set_permissions(&socket, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    eprintln!("frostd: serving on {} (gpu: {})", socket.display(), d.gpu.name());
    for conn in l.incoming() {
        let Ok(conn) = conn else { continue };
        let mut w = conn.try_clone()?;
        let mut stop = false;
        for line in BufReader::new(conn).lines().map_while(Result::ok) {
            let req: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => serde_json::json!({"cmd": "", "err": e.to_string()}),
            };
            let resp = match req["cmd"].as_str().unwrap_or("") {
                "launch" => match load_and_launch(&d, req["manifest"].as_str().unwrap_or("")) {
                    Ok(name) => serde_json::json!({"ok": true, "name": name}),
                    Err(e) => serde_json::json!({"ok": false, "err": format!("{e:#}")}),
                },
                "ps" => serde_json::json!({"ok": true, "capsules": d.slots().iter().map(|s| s.report()).collect::<Vec<_>>()}),
                "kill" => serde_json::json!({"ok": d.kill(req["name"].as_str().unwrap_or(""))}),
                "shutdown" => { stop = true; serde_json::json!({"ok": true}) }
                _ => serde_json::json!({"ok": false, "err": "unknown cmd"}),
            };
            writeln!(w, "{resp}")?;
            if stop { break; }
        }
        if stop { break; }
    }
    d.shutdown();
    let _ = std::fs::remove_file(&socket);
    Ok(())
}

fn ctl(args: &[String]) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};
    let f = parse_flags(args)?;
    let (socket, pos) = (f.socket, f.manifests);
    let req = match pos.first().map(String::as_str) {
        Some("launch") => {
            let m = std::fs::canonicalize(pos.get(1).context("launch <capsule.toml>")?)?;
            serde_json::json!({"cmd": "launch", "manifest": m})
        }
        Some("kill") => serde_json::json!({"cmd": "kill", "name": pos.get(1).context("kill <name>")?}),
        Some(c @ ("ps" | "shutdown")) => serde_json::json!({"cmd": c}),
        _ => bail!("ctl launch <toml> | ps | kill <name> | shutdown"),
    };
    let mut s = std::os::unix::net::UnixStream::connect(&socket)
        .with_context(|| format!("connect {} (is `frostd serve` running?)", socket.display()))?;
    writeln!(s, "{req}")?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line)?;
    let v: serde_json::Value = serde_json::from_str(&line)?;
    if req["cmd"] == "ps" {
        println!("{:<16} {:>7} {:>9} {:>7} {:>9} {:>12} {:>9} {:>9}", "capsule", "pid", "state", "jobs", "gpu ms", "gpu now/cap", "gpu MiB", "peak MiB");
        for c in v["capsules"].as_array().into_iter().flatten() {
            let state = if c["alive"] == true { "up".to_string() } else { format!("exit {}", c["exit"].as_i64().map(|x| x.to_string()).unwrap_or_else(|| format!("sig{}", c["signal"]))) };
            let share = format!("{:.0}%/{:.0}%", c["gpu_pct"].as_f64().unwrap_or(0.0), c["gpu_cap_pct"].as_f64().unwrap_or(100.0));
            println!("{:<16} {:>7} {:>9} {:>7} {:>9.2} {:>12} {:>9.0} {:>9.1}", c["name"].as_str().unwrap_or(""), c["pid"], state, c["jobs"],
                c["gpu_ms"].as_f64().unwrap_or(0.0), share, c["gpu_mem_mb"].as_f64().unwrap_or(0.0), c["peak_footprint_mb"].as_f64().unwrap_or(0.0));
        }
    } else {
        println!("{v}");
    }
    anyhow::ensure!(v["ok"] == true, "request failed");
    Ok(())
}

fn run(args: &[String]) -> Result<()> {
    let f = parse_flags(args)?;
    if f.manifests.is_empty() {
        bail!("run: no capsule manifests given");
    }
    // Decide domains. More than one distinct trust domain (or --isolate) means each domain
    // runs in its own broker *process* — separate GPU address spaces, hardware-isolated.
    let domains: std::collections::BTreeSet<String> = f
        .manifests
        .iter()
        .filter_map(|m| Manifest::load(Path::new(m)).ok())
        .map(|man| if f.isolate { man.name } else { man.trust_domain() })
        .collect();
    if f.isolate || domains.len() > 1 {
        run_supervised(&f)
    } else {
        run_in_process(&f)
    }
}

/// Single shared broker in this process (the default; behaviour unchanged).
fn run_in_process(f: &Flags) -> Result<()> {
    let d = Daemon::start(f.cfg.clone())?;
    let mut failed = false;
    for m in &f.manifests {
        if let Err(e) = load_and_launch(&d, m) {
            eprintln!("frostd: launch {m}: {e:#}");
            failed = true;
        }
    }
    d.wait_all();
    let reports: Vec<serde_json::Value> =
        d.slots().iter().map(|s| serde_json::to_value(s.report()).unwrap()).collect();
    d.shutdown();
    print_reports(f.json, &reports);
    if failed {
        bail!("one or more capsules failed to launch");
    }
    Ok(())
}

/// One broker process per trust domain. Capsules in different domains therefore live in
/// different GPU virtual address spaces — a hostile custom kernel in one cannot address
/// another's GPU memory.
fn run_supervised(f: &Flags) -> Result<()> {
    use std::process::Command;
    use std::time::{Duration, Instant};
    std::fs::create_dir_all(&f.cfg.run_dir)?;
    let exe = std::env::current_exe().context("current_exe")?;
    let sup = std::process::id();

    // Resolve each manifest to (abspath, domain), reporting parse failures like the old path.
    let mut failed = false;
    let mut caps: Vec<(std::path::PathBuf, String)> = Vec::new();
    for m in &f.manifests {
        match Manifest::load(Path::new(m)) {
            Ok(man) => {
                let dom = if f.isolate { man.name.clone() } else { man.trust_domain() };
                match std::fs::canonicalize(m) {
                    Ok(p) => caps.push((p, dom)),
                    Err(e) => {
                        eprintln!("frostd: launch {m}: {e:#}");
                        failed = true;
                    }
                }
            }
            Err(e) => {
                eprintln!("frostd: launch {m}: {e:#}");
                failed = true;
            }
        }
    }

    // Spawn one broker (`frostd serve`) per distinct domain.
    struct Worker {
        sock: std::path::PathBuf,
        child: std::process::Child,
    }
    let mut workers: std::collections::BTreeMap<String, Worker> = std::collections::BTreeMap::new();
    for (_, dom) in &caps {
        if workers.contains_key(dom) {
            continue;
        }
        let sock = f.cfg.run_dir.join(format!("broker-{sup}-{dom}.sock"));
        let _ = std::fs::remove_file(&sock);
        let mut a = vec!["serve".to_string(), "--socket".to_string(), sock.display().to_string()];
        a.extend(passthrough_flags(&f.cfg));
        let child = Command::new(&exe).args(&a).spawn().with_context(|| format!("spawn broker for domain '{dom}'"))?;
        let t0 = Instant::now();
        while !sock.exists() {
            anyhow::ensure!(t0.elapsed() < Duration::from_secs(15), "broker for domain '{dom}' did not come up");
            std::thread::sleep(Duration::from_millis(10));
        }
        workers.insert(dom.clone(), Worker { sock, child });
    }
    eprintln!("frostd: supervising {} broker process(es) [{}] — separate GPU address spaces", workers.len(), workers.keys().cloned().collect::<Vec<_>>().join(", "));

    // Launch each capsule into its domain's broker.
    let mut launched = 0usize;
    for (path, dom) in &caps {
        let req = serde_json::json!({"cmd": "launch", "manifest": path.display().to_string()});
        match broker_cmd(&workers[dom].sock, &req) {
            Ok(r) if r["ok"] == true => launched += 1,
            Ok(r) => {
                eprintln!("frostd: launch {}: {}", path.display(), r["err"]);
                failed = true;
            }
            Err(e) => {
                eprintln!("frostd: launch {}: {e:#}", path.display());
                failed = true;
            }
        }
    }

    // Poll every broker until all launched capsules have exited, then collect reports.
    let reports = loop {
        let mut all: Vec<serde_json::Value> = Vec::new();
        for w in workers.values() {
            if let Ok(r) = broker_cmd(&w.sock, &serde_json::json!({"cmd": "ps"})) {
                if let Some(arr) = r["capsules"].as_array() {
                    all.extend(arr.iter().cloned());
                }
            }
        }
        if all.len() >= launched && all.iter().all(|r| r["alive"] != true) {
            break all;
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    // Shut the brokers down (joins their log pumps, flushing capsule RESULT lines) then print.
    for w in workers.values() {
        let _ = broker_cmd(&w.sock, &serde_json::json!({"cmd": "shutdown"}));
    }
    for (_, mut w) in workers {
        let _ = w.child.wait();
        let _ = std::fs::remove_file(&w.sock);
    }
    print_reports(f.json, &reports);
    if failed {
        bail!("one or more capsules failed to launch");
    }
    Ok(())
}

/// Print capsule reports (as JSON Values) either as a `REPORT` json line or a table.
fn print_reports(json: bool, reports: &[serde_json::Value]) {
    if json {
        println!("REPORT {}", serde_json::Value::Array(reports.to_vec()));
        return;
    }
    let g = |r: &serde_json::Value, k: &str| r[k].as_f64().unwrap_or(0.0);
    println!("\n{:<16} {:>7} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}  notes", "capsule", "exit", "jobs", "gpu ms", "cpu ms", "wall ms", "peak MiB", "broker", "start ms");
    for r in reports {
        let exit = match (r["exit"].as_i64(), r["signal"].as_i64()) {
            (Some(c), _) => c.to_string(),
            (_, Some(s)) => format!("sig{s}"),
            _ => "?".into(),
        };
        println!(
            "{:<16} {:>7} {:>6} {:>9.2} {:>9.1} {:>9.1} {:>9.1} {:>8} {:>8.2}  {}",
            r["name"].as_str().unwrap_or(""), exit, r["jobs"].as_u64().unwrap_or(0),
            g(r, "gpu_ms"), g(r, "cpu_ms"), g(r, "wall_ms"), g(r, "peak_footprint_mb"),
            r["broker_pid"].as_i64().unwrap_or(0),
            r["startup_ms"].as_f64().unwrap_or(f64::NAN),
            r["killed_for"].as_str().unwrap_or(""),
        );
    }
}

/// Baseline: dispatch latency when the workload owns Metal directly (no broker).
fn bench_direct(args: &[String]) -> Result<()> {
    let iters: usize = args.iter().position(|a| a == "--iters").and_then(|i| args.get(i + 1)).map(|v| v.parse()).transpose()?.unwrap_or(2000);
    let _ = fb_core::sched::set_thread_qos(fb_core::sched::Qos::UserInteractive);
    let g = fb_gpu::Gpu::new()?;
    let buf = g.alloc(1 << 16)?;
    let q = g.new_queue()?;
    let mut job = fb_core::abi::Job { kernel: fb_core::abi::kernel::NOOP, n_args: 1, grid: [1, 1, 1], ..Default::default() };
    job.args[0] = fb_core::abi::BufRef { off: 0, len: 16 };
    let win = fb_gpu::Window { off: 0, len: buf.len as u64 };
    for mode in ["spin", "wait"] {
        let mut samples = Vec::with_capacity(iters);
        for i in 0..iters + 100 {
            let t0 = fb_core::time::now_ns();
            let mut b = g.batch_on(&q)?;
            b.encode(&buf, win, &job).map_err(|s| anyhow::anyhow!("{s:?}"))?;
            let f = b.commit();
            let d = if mode == "spin" { f.spin() } else { f.wait() };
            anyhow::ensure!(d.ok);
            if i >= 100 {
                samples.push(fb_core::time::now_ns() - t0);
            }
        }
        println!("direct-metal noop round trip ({mode:>4}): {}", fb_core::time::Summary::of(&mut samples));
    }
    Ok(())
}

/// GVertex matching fb-gpu/src/render.metal (48 bytes).
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct GVertex {
    color: [f32; 4],
    pos: [f32; 2],
    center: [f32; 2],
    radius: f32,
    _pad: [f32; 3],
}

/// Baselines for "with vs without the gravity containerization": render the same gravity
/// scene (a) directly on the GPU in-process (no broker, no sandbox) and (b) on the CPU (no
/// GPU acceleration). Compare these against `frostd run capsules/gravity.toml`, which renders
/// it through the containerized bridge.
fn bench_gravity(args: &[String]) -> Result<()> {
    use fb_core::time::{now_ns, Summary};
    let argn = |name: &str, d: usize| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (w, h) = (argn("--width", 854), argn("--height", 480));
    let balls = argn("--balls", 16);
    let frames = argn("--frames", 1200);
    let cpu_frames = argn("--cpu-frames", 120).min(frames);
    let _ = fb_core::sched::set_thread_qos(fb_core::sched::Qos::UserInteractive);

    // --- shared initial scene (same seed/motion as the gravity capsule)
    let floor_y = h as f32 - h as f32 * 0.08;
    let palette = [[1.0, 0.42, 0.42f32], [0.3, 0.8, 0.77], [1.0, 0.85, 0.24], [0.66, 0.42, 1.0], [0.42, 0.8, 1.0], [1.0, 0.62, 0.26]];
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 40) as f32) / (1u64 << 24) as f32 };
    let mut pos0 = vec![[0f32; 2]; balls];
    let mut vel0 = vec![[0f32; 2]; balls];
    let mut rad = vec![0f32; balls];
    let mut col = vec![[0f32; 4]; balls];
    for i in 0..balls {
        rad[i] = 12.0 + rng() * 28.0;
        pos0[i] = [rad[i] + rng() * (w as f32 - 2.0 * rad[i]), rng() * floor_y * 0.6];
        vel0[i] = [(rng() - 0.5) * 10.0, rng() * 3.0];
        let c = palette[i % palette.len()];
        col[i] = [c[0], c[1], c[2], 1.0];
    }
    const G: f32 = 0.45;
    const REST: f32 = 0.82;
    let step = |pos: &mut [[f32; 2]], vel: &mut [[f32; 2]]| {
        for i in 0..balls {
            vel[i][1] += G;
            pos[i][0] += vel[i][0];
            pos[i][1] += vel[i][1];
            let r = rad[i];
            if pos[i][0] < r { pos[i][0] = r; vel[i][0] = -vel[i][0] * REST; }
            else if pos[i][0] > w as f32 - r { pos[i][0] = w as f32 - r; vel[i][0] = -vel[i][0] * REST; }
            if pos[i][1] > floor_y - r { pos[i][1] = floor_y - r; vel[i][1] = -vel[i][1] * REST; vel[i][0] *= 0.99; }
        }
    };

    // ---------- (a) direct GPU, no container: hardware render pipeline in-process ----------
    let g = fb_gpu::Gpu::new()?;
    let q = g.new_queue()?;
    let stride = g.fb_stride_bytes(w);
    let fb_buf = g.alloc(h * stride)?;
    let target = g.make_target(&fb_buf, 0, w, h)?;
    let max_verts = 12 + 6 * balls;
    let vbuf = g.alloc(max_verts * std::mem::size_of::<GVertex>())?;
    let verts = unsafe { std::slice::from_raw_parts_mut(vbuf.as_ptr() as *mut GVertex, max_verts) };
    let (mut pos, mut vel) = (pos0.clone(), vel0.clone());
    let z = [0.0f32, 0.0];
    let mut put = |v: &mut [GVertex], n: &mut usize, p: [f32; 2], c2: [f32; 2], r: f32, c: [f32; 4]| { v[*n] = GVertex { color: c, pos: p, center: c2, radius: r, _pad: [0.0; 3] }; *n += 1; };
    let (mut rt, mut gpu) = (Vec::with_capacity(frames), Vec::with_capacity(frames));
    let t_start = now_ns();
    for _ in 0..frames {
        step(&mut pos, &mut vel);
        let mut n = 0usize;
        let top = [0.05, 0.07, 0.12, 1.0]; let bot = [0.10, 0.13, 0.22, 1.0]; let fl = [0.18, 0.20, 0.26, 1.0];
        for (p, c) in [([0.0, 0.0], top), ([w as f32, 0.0], top), ([w as f32, h as f32], bot), ([0.0, 0.0], top), ([w as f32, h as f32], bot), ([0.0, h as f32], bot)] { put(verts, &mut n, p, z, 0.0, c); }
        for p in [[0.0, floor_y], [w as f32, floor_y], [w as f32, h as f32], [0.0, floor_y], [w as f32, h as f32], [0.0, h as f32]] { put(verts, &mut n, p, z, 0.0, fl); }
        for i in 0..balls {
            let (cx, cy, r) = (pos[i][0], pos[i][1], rad[i]);
            for p in [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy - r], [cx + r, cy + r], [cx - r, cy + r]] { put(verts, &mut n, p, [cx, cy], r, col[i]); }
        }
        let t0 = now_ns();
        let d = g.render(&q, &target, &vbuf, 0, n)?.spin();
        rt.push(now_ns() - t0);
        gpu.push(d.gpu_ns);
    }
    let fps = frames as f64 / ((now_ns() - t_start) as f64 / 1e9);
    println!("direct-metal, NO container : {fps:>7.0} fps  frame {}", Summary::of(&mut rt));
    println!("   gpu render time         : {}", Summary::of(&mut gpu));

    // ---------- (b) CPU only, no GPU acceleration: software rasterizer ----------
    let (mut pos, mut vel) = (pos0.clone(), vel0.clone());
    let mut fbc = vec![0u32; w * h];
    let mut rtc = Vec::with_capacity(cpu_frames);
    let t_start = now_ns();
    for _ in 0..cpu_frames {
        step(&mut pos, &mut vel);
        let t0 = now_ns();
        for y in 0..h {
            let v = y as f32 / h as f32;
            let base = if (y as f32) > floor_y { [0.18, 0.20, 0.26] } else { [0.05 + 0.05 * v, 0.07 + 0.06 * v, 0.12 + 0.10 * v] };
            for x in 0..w {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let mut c = base;
                for i in 0..balls {
                    let d = ((px - pos[i][0]).powi(2) + (py - pos[i][1]).powi(2)).sqrt();
                    let cov = (rad[i] - d + 0.5).clamp(0.0, 1.0);
                    if cov > 0.0 { for k in 0..3 { c[k] = c[k] * (1.0 - cov) + col[i][k] * cov; } }
                }
                fbc[y * w + x] = 0xff00_0000 | ((c[2] * 255.0) as u32) << 16 | ((c[1] * 255.0) as u32) << 8 | (c[0] * 255.0) as u32;
            }
        }
        rtc.push(now_ns() - t0);
    }
    std::hint::black_box(&fbc);
    let fpsc = cpu_frames as f64 / ((now_ns() - t_start) as f64 / 1e9);
    println!("cpu only,  NO gpu          : {fpsc:>7.0} fps  frame {}", Summary::of(&mut rtc));
    println!("\n(compare 'frostd run capsules/gravity.toml' for the containerized GPU bridge: same render, sandboxed + brokered)");
    Ok(())
}
