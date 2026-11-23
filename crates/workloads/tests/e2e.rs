//! End-to-end tests: real frostd, real sandboxed capsules, real GPU.
//! Run: cargo test --release -p workloads --test e2e -- --test-threads=1
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

fn bin(name: &str) -> PathBuf {
    // All bins share one target dir; CARGO_BIN_EXE_probe tells us where it is.
    let dir = PathBuf::from(env!("CARGO_BIN_EXE_probe")).parent().unwrap().to_path_buf();
    dir.join(name)
}

fn frostd() -> PathBuf {
    let p = bin("frostd");
    if !p.exists() {
        let release = p.to_string_lossy().contains("/release/");
        let mut c = Command::new(env!("CARGO"));
        c.args(["build", "-p", "frostd"]);
        if release { c.arg("--release"); }
        assert!(c.status().unwrap().success(), "building frostd failed");
    }
    p
}

struct Run { results: Vec<Value>, report: Vec<Value>, stdout: String }

impl Run {
    fn result(&self, key: &str) -> &Value {
        self.results.iter().find(|r| r.get(key).is_some()).map(|r| &r[key])
            .unwrap_or_else(|| panic!("no RESULT with {key}; stdout:\n{}", self.stdout))
    }
    fn report(&self, name: &str) -> &Value {
        self.report.iter().find(|r| r["name"] == name).unwrap_or_else(|| panic!("no report for {name}"))
    }
}

/// Write manifests (with absolute exec paths) to a temp dir and run frostd on them.
fn run(flags: &[&str], capsules: &[(&str, &str)]) -> Run {
    let dir = std::env::temp_dir().join(format!("fb-e2e-{}-{}", std::process::id(), fb_core::time::ticks()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut paths = vec![];
    for (exe, body) in capsules {
        let name = body.lines().find_map(|l| l.strip_prefix("name = ")).unwrap().trim_matches('"').to_string();
        let p = dir.join(format!("{name}.toml"));
        std::fs::write(&p, format!("exec = \"{}\"\n{body}", bin(exe).display())).unwrap();
        paths.push(p);
    }
    let out = Command::new(frostd()).arg("run").arg("--json").arg("--quiet").args(flags).args(&paths).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "frostd failed\nstdout:{stdout}\nstderr:{stderr}");
    let mut results = vec![];
    let mut report = vec![];
    for l in stdout.lines() {
        if let Some(i) = l.find("RESULT ") { results.push(serde_json::from_str(&l[i + 7..]).unwrap()); }
        if let Some(r) = l.strip_prefix("REPORT ") { report = serde_json::from_str::<Vec<Value>>(r).unwrap(); }
    }
    std::fs::remove_dir_all(&dir).ok();
    Run { results, report, stdout }
}

#[test]
fn isolation_and_core_placement() {
    let r = run(&[], &[("probe", "name = \"probe-e\"\n[cpu]\nclass = \"efficiency\"\n[gpu]\narena_mb = 1")]);
    // The probe's RESULT is a flat object tagged with "probe": true.
    let p = r.results.iter().find(|x| x.get("probe").is_some()).expect("no probe RESULT");
    for k in ["read_etc", "list_shared", "list_home", "write_tmp", "network", "fork", "exec"] {
        assert_eq!(p[k], false, "capsule was able to {k}");
    }
    assert_eq!(p["write_root"], true);
    assert_eq!(p["fds"], true);
    assert!(p["gpu"].is_string(), "sandboxed capsule could not reach the GPU broker");
    assert_eq!(p["qos"], 9, "efficiency capsule must run at QoS BACKGROUND");
    let (_, ecores) = fb_core::sched::core_topology();
    let cpus = p["cpus"].as_array().unwrap();
    assert!(cpus.len() as u32 <= ecores, "background capsule ran on {} cpus (> {ecores} E-cores): {cpus:?}", cpus.len());
}

#[test]
fn adversarial_capsule_is_contained() {
    let r = run(&[], &[("evil", "name = \"evil\"\n[gpu]\narena_mb = 1")]);
    let e = r.result("evil");
    for k in ["header", "huge", "wrap", "unaligned"] {
        assert!(e[k].as_str().unwrap().contains("OutOfBounds"), "{k}: {}", e[k]);
    }
    assert!(e["bad_kernel"].as_str().unwrap().contains("BadKernel"));
    assert!(e["register"].as_str().unwrap().contains("not permitted"));
    assert_eq!(e["still_works"], true);
    assert_eq!(r.report("evil")["rejected"], 5);
}

#[test]
fn memory_limit_kills_capsule() {
    let r = run(&[], &[("mem-hog", "name = \"hog\"\nargs = [\"--mb\", \"1024\"]\n[memory]\nlimit_mb = 96\n[gpu]\narena_mb = 1")]);
    let rep = r.report("hog");
    assert_eq!(rep["signal"], 9);
    assert!(rep["killed_for"].as_str().unwrap().contains("memory"));
    assert!(rep["peak_footprint_mb"].as_f64().unwrap() < 200.0, "watchdog too slow: {rep}");
    assert!(r.results.iter().all(|x| x.get("mem_hog_survived").is_none()));
}

#[test]
fn gpu_workloads_are_correct() {
    let r = run(&[], &[
        ("crypto-hash", "name = \"hash\"\nargs = [\"--n\", \"65536\", \"--len\", \"77\", \"--rounds\", \"3\"]\n[gpu]\narena_mb = 16"),
        ("infer-mlp", "name = \"mlp\"\nargs = [\"--dim\", \"512\", \"--layers\", \"4\", \"--tokens\", \"20\"]\n[gpu]\narena_mb = 16\ncustom_kernels = true"),
    ]);
    assert_eq!(r.result("crypto_hash")["mismatches"], 0);
    assert!(r.result("infer_mlp")["rel_err"].as_f64().unwrap() < 1e-3);
    assert_eq!(r.report("hash")["exit"], 0);
    assert_eq!(r.report("mlp")["exit"], 0);
}

#[test]
fn gravity_renders_frames() {
    // The "gravity" graphics plane: a sandboxed capsule renders a game on the GPU through
    // the broker, into a zero-copy framebuffer in its arena. Test both GPU paths.
    for mode in ["render", "compute"] {
        let caps = format!(
            "name = \"game\"\nargs = [\"--mode\", \"{mode}\", \"--frames\", \"200\", \"--balls\", \"12\", \"--width\", \"640\", \"--height\", \"360\"]\n[cpu]\nclass = \"performance\"\n[gpu]\narena_mb = 8\npriority = 50"
        );
        let a = run(&[], &[("gravity", &caps)]);
        let g = a.result("gravity");
        assert_eq!(a.report("game")["exit"], 0, "{mode}");
        assert_eq!(g["mode"], mode);
        assert_eq!(g["frames"], 200);
        // It actually drew something (not a blank buffer) ...
        assert_ne!(g["checksum"].as_u64().unwrap(), 0, "{mode} rendered nothing");
        // ... and rendering is deterministic: the same seed yields a byte-identical frame.
        let b = run(&[], &[("gravity", &caps)]);
        assert_eq!(g["checksum"], b.result("gravity")["checksum"], "{mode} not reproducible");
        let fps = g["fps"].as_f64().unwrap();
        let gpu_p50 = g["gpu_render"]["p50_us"].as_f64().unwrap();
        println!("gravity[{mode}]: {fps:.0} fps, gpu render p50 {gpu_p50:.0}us");
        assert!(fps > 120.0, "{mode}: only {fps:.0} fps");
        assert!(gpu_p50 < 5000.0, "{mode}: gpu render p50 {gpu_p50}us too slow");
    }
}

#[test]
fn gravity_hardware_render_beats_compute() {
    // The hardware rasterizer (vertex+fragment pipeline) should spend less GPU time per
    // frame than the compute software rasterizer for the same scene.
    let caps = |mode: &str| format!(
        "name = \"g\"\nargs = [\"--mode\", \"{mode}\", \"--frames\", \"400\", \"--balls\", \"24\"]\n[cpu]\nclass = \"performance\"\n[gpu]\narena_mb = 16\npriority = 50"
    );
    let r = run(&[], &[("gravity", &caps("render"))]).result("gravity")["gpu_render"]["p50_us"].as_f64().unwrap();
    let c = run(&[], &[("gravity", &caps("compute"))]).result("gravity")["gpu_render"]["p50_us"].as_f64().unwrap();
    println!("gpu render p50: hardware {r:.0}us vs compute {c:.0}us");
    assert!(r < c, "hardware render ({r:.0}us) should beat compute ({c:.0}us)");
}

#[test]
fn gpu_budget_is_enforced() {
    let r = run(&[], &[("crypto-hash", "name = \"budget\"\nargs = [\"--n\", \"1048576\", \"--len\", \"200\", \"--rounds\", \"15\"]\n[cpu]\nclass = \"performance\"\n[memory]\nlimit_mb = 512\n[gpu]\narena_mb = 300\ngpu_ms_per_sec = 100")]);
    let busy = r.result("crypto_hash")["gpu_busy"].as_f64().unwrap();
    assert!(busy < 0.16, "GPU busy {busy:.3} exceeds a 10% budget");
    assert!(r.report("budget")["throttled_ms"].as_f64().unwrap() > 100.0);
}

#[test]
fn gpu_percent_cap_contains_a_capsule() {
    // "I don't want it using all of the GPU": a capsule declaring max_percent = 25 that
    // tries to flood the device must be held to ~a quarter of it.
    let r = run(&[], &[("crypto-hash", "name = \"quarter\"\nargs = [\"--n\", \"1048576\", \"--len\", \"200\", \"--rounds\", \"30\"]\n[cpu]\nclass = \"performance\"\n[memory]\nlimit_mb = 512\n[gpu]\narena_mb = 300\nmax_percent = 25")]);
    let busy = r.result("crypto_hash")["gpu_busy"].as_f64().unwrap();
    println!("capsule capped at 25%: measured GPU busy {:.1}%", busy * 100.0);
    assert!(busy < 0.35, "max_percent=25 not enforced: {busy:.3} busy");
    let rep = r.report("quarter");
    assert_eq!(rep["gpu_cap_pct"], 25.0);
    assert_eq!(rep["gpu_mem_mb"], 300.0);
    assert!(rep["throttled_ms"].as_f64().unwrap() > 50.0, "a flooding capped capsule should be throttled");
}

#[test]
fn max_inflight_one_still_correct() {
    // Occupancy cap: a capsule limited to a single command buffer in flight must still
    // produce correct results (just with less pipelining).
    let r = run(&[], &[("crypto-hash", "name = \"serial\"\nargs = [\"--n\", \"65536\", \"--len\", \"77\", \"--rounds\", \"2\"]\n[gpu]\narena_mb = 16\nmax_inflight = 1")]);
    assert_eq!(r.result("crypto_hash")["mismatches"], 0);
    assert_eq!(r.report("serial")["exit"], 0);
}

#[test]
fn trust_domains_isolate_brokers() {
    // Capsules in different trust domains run in different broker *processes* — separate GPU
    // virtual address spaces, so a hostile custom kernel in one can't reach the other's GPU
    // memory. Same domain shares one broker (shared address space, fine-grained scheduling).
    let cap = |name: &str, dom: &str| format!(
        "name = \"{name}\"\nargs = [\"--n\", \"8192\", \"--len\", \"32\", \"--rounds\", \"1\"]\n[gpu]\narena_mb = 4\ntrust_domain = \"{dom}\""
    );
    let r = run(&[], &[("crypto-hash", &cap("dom-a", "alpha")), ("crypto-hash", &cap("dom-b", "beta"))]);
    assert_eq!(r.report("dom-a")["exit"], 0);
    assert_eq!(r.report("dom-b")["exit"], 0);
    let (pa, pb) = (r.report("dom-a")["broker_pid"].as_i64().unwrap(), r.report("dom-b")["broker_pid"].as_i64().unwrap());
    println!("broker pids: alpha={pa}, beta={pb}");
    assert_ne!(pa, pb, "different trust domains must be different broker processes");

    let r2 = run(&[], &[("crypto-hash", &cap("dom-c", "shared")), ("crypto-hash", &cap("dom-d", "shared"))]);
    assert_eq!(r2.report("dom-c")["broker_pid"], r2.report("dom-d")["broker_pid"], "same domain should share one broker");
}

#[test]
fn isolate_flag_gives_each_capsule_its_own_broker() {
    let cap = |name: &str| format!(
        "name = \"{name}\"\nargs = [\"--n\", \"8192\", \"--len\", \"32\", \"--rounds\", \"1\"]\n[gpu]\narena_mb = 4"
    );
    let r = run(&["--isolate"], &[("crypto-hash", &cap("iso-a")), ("crypto-hash", &cap("iso-b"))]);
    assert_eq!(r.report("iso-a")["exit"], 0);
    assert_eq!(r.report("iso-b")["exit"], 0);
    assert_ne!(r.report("iso-a")["broker_pid"], r.report("iso-b")["broker_pid"], "--isolate must give each capsule its own broker process");
}

#[test]
fn priority_slicing_protects_latency() {
    let lat = "name = \"lat\"\nargs = [\"--iters\", \"1500\", \"--delay-ms\", \"1800\", \"--pipeline\", \"2000\"]\n[cpu]\nclass = \"performance\"\n[gpu]\narena_mb = 1\npriority = 100";
    let flood = "name = \"flood\"\nargs = [\"--n\", \"1048576\", \"--len\", \"200\", \"--rounds\", \"400\"]\n[cpu]\nclass = \"efficiency\"\n[memory]\nlimit_mb = 512\n[gpu]\narena_mb = 300";
    let p90 = |r: &Run| r.result("gpu_latency")["spin"]["round_trip"]["p90_us"].as_f64().unwrap();
    let off = run(&["--no-slicing"], &[("gpu-latency", lat), ("crypto-hash", flood)]);
    let on = run(&[], &[("gpu-latency", lat), ("crypto-hash", flood)]);
    let (a, b) = (p90(&off), p90(&on));
    println!("p90 round trip under flood: no slicing {a:.0}us, slicing {b:.0}us");
    assert!(b < a / 3.0, "slicing should cut p90 at least 3x: {a:.0} -> {b:.0}");
    assert_eq!(on.result("crypto_hash")["mismatches"], 0, "sliced dispatch corrupted results");
}

#[test]
fn realtime_policy_bounds_jitter() {
    let r = run(&[], &[("rt-jitter", "name = \"rt\"\nargs = [\"--period-us\", \"500\", \"--iters\", \"2000\", \"--stress\", \"14\"]\n[cpu]\nclass = \"performance\"\n[gpu]\narena_mb = 1")]);
    let j = r.result("rt_jitter");
    let rt = j["realtime"]["lateness"]["p99_us"].as_f64().unwrap();
    let ts = j["timeshare"]["lateness"]["p99_us"].as_f64().unwrap();
    println!("p99 wake lateness under 14 hogs: timeshare {ts:.0}us, realtime {rt:.1}us");
    assert!(rt < 100.0, "RT p99 lateness {rt}us");
    assert_eq!(j["realtime"]["misses"], 0);
}

#[test]
fn hostile_capsules_do_not_starve_others() {
    let r = run(&[], &[
        ("evil", "name = \"corrupt\"\nargs = [\"--corrupt-ring\"]\n[gpu]\narena_mb = 1"),
        ("evil", "name = \"garbage\"\nargs = [\"--garbage-flood\"]\n[gpu]\narena_mb = 1\npriority = 200"),
        ("gpu-latency", "name = \"victim\"\nargs = [\"--iters\", \"1500\", \"--delay-ms\", \"300\", \"--pipeline\", \"2000\"]\n[cpu]\nclass = \"performance\"\n[gpu]\narena_mb = 1\npriority = 10"),
    ]);
    let c = r.report("corrupt");
    assert_eq!(c["signal"], 9, "corrupt capsule should be killed: {c}");
    assert!(c["killed_for"].as_str().unwrap().contains("corrupted"));
    let g = r.report("garbage");
    assert_eq!(g["exit"], 0, "garbage capsule just gets rejections + backpressure: {g}");
    assert_eq!(r.report("victim")["exit"], 0);
    let p50 = r.result("gpu_latency")["spin"]["round_trip"]["p50_us"].as_f64().unwrap();
    println!("victim p50 next to hostile capsules: {p50:.0}us; garbage rejected {}", g["rejected"]);
    assert!(p50 < 1000.0, "victim starved: p50 {p50}us");
}
