//! fb-capsule: process-level isolation for workloads on macOS.
//!
//! macOS has no namespaces or cgroups, so a "capsule" is assembled from the primitives
//! the kernel does offer:
//!
//! | concern        | mechanism                                                         |
//! |----------------|-------------------------------------------------------------------|
//! | filesystem/net | Seatbelt (SBPL) profile, deny-by-default, applied by sandbox-exec |
//! | GPU            | none granted — GPU work is brokered by frostd via shared arena    |
//! | CPU placement  | QoS class of the main thread (P-cores vs E-cores)                 |
//! | CPU time       | RLIMIT_CPU (SIGXCPU → kill)                                       |
//! | memory         | phys_footprint watchdog in the daemon (SIGKILL), RLIMIT_MEMLOCK   |
//! | lifecycle      | own process group; daemon kills the group on teardown             |
//! | IPC            | only the fds the daemon hands down (ctl socket + arena)           |

use anyhow::{bail, Context, Result};
use fb_core::manifest::{CoreClass, Manifest};
use fb_core::sched::Qos;
use std::ffi::CString;
use std::os::fd::RawFd;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

fn sbpl_str(p: &Path) -> String {
    // SBPL string literal: escape backslash and quote.
    let s = p.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{s}\"")
}

/// Resolve symlinks (/tmp → /private/tmp etc.) since Seatbelt matches real paths.
fn real(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Generate the Seatbelt profile for a capsule.
pub fn profile(m: &Manifest, exec: &Path, root: &Path) -> String {
    let exec = real(exec);
    let root = real(root);
    let mut s = String::new();
    s.push_str("(version 1)\n(deny default)\n");
    // Apple's baseline for BSD daemons (dyld shared cache, libSystem, timezone,
    // notification shm). Everything not granted below stays denied.
    s.push_str("(import \"bsd.sb\")\n");
    // --- process: may exec only its own binary; fork is denied.
    s.push_str(&format!("(allow process-exec (literal {}))\n", sbpl_str(&exec)));
    s.push_str(&format!("(allow file-read* (literal {}))\n", sbpl_str(&exec)));
    s.push_str("(allow file-read* (literal \"/dev/urandom\") (literal \"/dev/random\"))\n");
    s.push_str("(allow file-read* file-write-data (literal \"/dev/null\"))\n");
    s.push_str("(allow sysctl-read)\n");
    // --- the capsule's private root (read-write).
    s.push_str(&format!("(allow file-read* file-write* (subpath {}))\n", sbpl_str(&root)));
    for r in &m.sandbox.read {
        s.push_str(&format!("(allow file-read* (subpath {}))\n", sbpl_str(&real(Path::new(r)))));
    }
    for w in &m.sandbox.write {
        s.push_str(&format!("(allow file-read* file-write* (subpath {}))\n", sbpl_str(&real(Path::new(w)))));
    }
    if m.sandbox.network {
        s.push_str("(allow network* system-socket)\n(allow mach-lookup (global-name \"com.apple.dnssd.service\"))\n");
    }
    s
}

pub fn qos_for(class: CoreClass) -> Qos {
    match class {
        CoreClass::Performance => Qos::UserInteractive,
        CoreClass::Efficiency => Qos::Background,
        CoreClass::Default => Qos::Default,
    }
}

pub struct SpawnSpec<'a> {
    pub manifest: &'a Manifest,
    /// Absolute path of the executable.
    pub exec: PathBuf,
    /// Capsule root directory (cwd + only writable location).
    pub root: PathBuf,
    /// fds to hand down (they keep their numbers in the child).
    pub inherit: Vec<RawFd>,
    pub env: Vec<(String, String)>,
    pub stdout: Stdio,
    pub stderr: Stdio,
}

// Apple extensions to posix_spawn (<spawn.h>, <spawn_private.h>).
const POSIX_SPAWN_SETEXEC: libc::c_short = 0x0040;
const POSIX_SPAWN_CLOEXEC_DEFAULT: libc::c_short = 0x4000;
const POSIX_SPAWN_JETSAM_MEMLIMIT_ACTIVE_FATAL: libc::c_short = 0x04;
const POSIX_SPAWN_JETSAM_MEMLIMIT_INACTIVE_FATAL: libc::c_short = 0x08;

extern "C" {
    fn posix_spawnattr_set_qos_class_np(attr: *mut libc::posix_spawnattr_t, qos: u32) -> libc::c_int;
    fn posix_spawn_file_actions_addinherit_np(fa: *mut libc::posix_spawn_file_actions_t, fd: libc::c_int) -> libc::c_int;
    fn posix_spawnattr_setjetsam_ext(
        attr: *mut libc::posix_spawnattr_t,
        flags: libc::c_short,
        priority: libc::c_int,
        memlimit_active: libc::c_int,
        memlimit_inactive: libc::c_int,
    ) -> libc::c_int;
}

/// Everything the forked child needs, prepared (allocated) in the parent so the child
/// only performs async-signal-safe work before it is replaced.
struct ExecPlan {
    path: CString,
    _argv: Vec<CString>,
    argv_p: Vec<*const libc::c_char>,
    _envp: Vec<CString>,
    envp_p: Vec<*const libc::c_char>,
    attr: libc::posix_spawnattr_t,
    fa: libc::posix_spawn_file_actions_t,
    cpu_secs: Option<u64>,
    inherit: Vec<RawFd>,
    memlock: u64,
}
unsafe impl Send for ExecPlan {}
unsafe impl Sync for ExecPlan {}

/// Launch a capsule. Returns the child; its pid is also its process-group id.
///
/// Mechanism: std forks and wires stdio; in the child we set rlimits and then call
/// `posix_spawn(POSIX_SPAWN_SETEXEC)`, which *replaces* the child (like exec) but lets
/// us attach spawn attributes exec can't carry: QoS class (core cluster), a kernel
/// jetsam memory limit, and CLOEXEC_DEFAULT so only explicitly granted fds survive.
pub fn spawn(spec: SpawnSpec<'_>) -> Result<Child> {
    let m = spec.manifest;
    if !spec.exec.is_absolute() {
        bail!("exec must be absolute: {}", spec.exec.display());
    }
    std::fs::create_dir_all(&spec.root).with_context(|| format!("mkdir {}", spec.root.display()))?;

    let cs = |s: &str| CString::new(s).map_err(|_| anyhow::anyhow!("NUL in argument"));
    let exec_s = spec.exec.to_string_lossy().to_string();
    let (path, mut argv) = if m.sandbox.enabled {
        let prof = profile(m, &spec.exec, &spec.root);
        (cs(SANDBOX_EXEC)?, vec![cs("sandbox-exec")?, cs("-p")?, cs(&prof)?, cs(&exec_s)?])
    } else {
        (cs(&exec_s)?, vec![cs(&exec_s)?])
    };
    for a in &m.args {
        argv.push(cs(a)?);
    }
    let mut env: Vec<(String, String)> = vec![
        ("PATH".into(), "/usr/bin:/bin".into()),
        ("HOME".into(), spec.root.display().to_string()),
        ("TMPDIR".into(), spec.root.display().to_string()),
    ];
    env.extend(m.env.iter().map(|(k, v)| (k.clone(), v.clone())));
    env.extend(spec.env.iter().cloned());
    let envp: Vec<CString> = env.iter().map(|(k, v)| cs(&format!("{k}={v}"))).collect::<Result<_>>()?;

    let mut plan = ExecPlan {
        argv_p: argv.iter().map(|c| c.as_ptr()).chain(std::iter::once(std::ptr::null())).collect(),
        envp_p: envp.iter().map(|c| c.as_ptr()).chain(std::iter::once(std::ptr::null())).collect(),
        path,
        _argv: argv,
        _envp: envp,
        attr: std::ptr::null_mut(),
        fa: std::ptr::null_mut(),
        cpu_secs: m.cpu.max_cpu_seconds,
        inherit: spec.inherit.clone(),
        memlock: if m.memory.wire_arena { m.gpu.arena_mb * 2 * 1024 * 1024 } else { 0 },
    };
    unsafe {
        anyhow::ensure!(libc::posix_spawnattr_init(&mut plan.attr) == 0, "posix_spawnattr_init");
        anyhow::ensure!(libc::posix_spawn_file_actions_init(&mut plan.fa) == 0, "file_actions_init");
        libc::posix_spawnattr_setflags(&mut plan.attr, POSIX_SPAWN_SETEXEC | POSIX_SPAWN_CLOEXEC_DEFAULT);
        posix_spawnattr_set_qos_class_np(&mut plan.attr, qos_for(m.cpu.class) as u32);
        let mb = m.memory.limit_mb.min(i32::MAX as u64) as i32;
        posix_spawnattr_setjetsam_ext(
            &mut plan.attr,
            POSIX_SPAWN_JETSAM_MEMLIMIT_ACTIVE_FATAL | POSIX_SPAWN_JETSAM_MEMLIMIT_INACTIVE_FATAL,
            -1,
            mb,
            mb,
        );
        for fd in [0, 1, 2].into_iter().chain(spec.inherit.iter().copied()) {
            posix_spawn_file_actions_addinherit_np(&mut plan.fa, fd);
        }
    }

    let mut cmd = Command::new(SANDBOX_EXEC); // never actually exec'd: replaced in pre_exec
    cmd.current_dir(&spec.root).stdin(Stdio::null()).stdout(spec.stdout).stderr(spec.stderr);
    let plan = std::sync::Arc::new(plan);
    let p2 = plan.clone();
    unsafe {
        cmd.pre_exec(move || {
            let p = &*p2;
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            for &fd in &p.inherit {
                let fl = libc::fcntl(fd, libc::F_GETFD);
                if fl < 0 || libc::fcntl(fd, libc::F_SETFD, fl & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            if let Some(s) = p.cpu_secs {
                let rl = libc::rlimit { rlim_cur: s, rlim_max: s + 1 };
                libc::setrlimit(libc::RLIMIT_CPU, &rl);
            }
            let rl = libc::rlimit { rlim_cur: p.memlock, rlim_max: p.memlock };
            libc::setrlimit(libc::RLIMIT_MEMLOCK, &rl);
            let mut pid = 0;
            let r = libc::posix_spawn(
                &mut pid,
                p.path.as_ptr(),
                &p.fa,
                &p.attr,
                p.argv_p.as_ptr() as *const *mut libc::c_char,
                p.envp_p.as_ptr() as *const *mut libc::c_char,
            );
            // Only reached on failure.
            Err(std::io::Error::from_raw_os_error(r))
        });
    }
    let child = cmd.spawn().with_context(|| format!("spawn capsule {}", m.name));
    unsafe {
        let p = std::sync::Arc::as_ptr(&plan) as *mut ExecPlan;
        libc::posix_spawnattr_destroy(&mut (*p).attr);
        libc::posix_spawn_file_actions_destroy(&mut (*p).fa);
    }
    child
}

/// Kill a capsule's whole process group.
pub fn kill_group(pgid: i32, sig: i32) {
    unsafe {
        libc::killpg(pgid, sig);
    }
}

/// Outcome of the footprint watchdog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Ok { footprint: u64 },
    OverLimit { footprint: u64, limit: u64 },
    Gone,
}

pub fn check_memory(pid: i32, limit_bytes: u64) -> Verdict {
    match fb_core::sched::phys_footprint(pid) {
        Ok(f) if f > limit_bytes => Verdict::OverLimit { footprint: f, limit: limit_bytes },
        Ok(f) => Verdict::Ok { footprint: f },
        Err(_) => Verdict::Gone,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile_is_deny_default_and_scoped() {
        let m = Manifest::parse("name='t'\nexec='/bin/echo'").unwrap();
        let p = profile(&m, Path::new("/bin/echo"), Path::new("/tmp"));
        assert!(p.contains("(deny default)"));
        assert!(p.contains("(subpath \"/private/tmp\")"), "{p}");
        assert!(!p.contains("network*"));
    }

    #[test]
    fn sandboxed_echo_runs_and_cannot_write_outside_root() {
        let root = std::env::temp_dir().join(format!("fbcap-{}", std::process::id()));
        // bash builtins only: the sandbox denies fork.
        let m = Manifest::parse(
            "name='t'\nexec='/bin/bash'\nargs=['-c','echo hi > ok.txt; read x < /etc/hosts && echo READ_ETC; echo bad > /Users/Shared/fb-escape.txt; echo done']",
        )
        .unwrap();
        let out = spawn(SpawnSpec {
            manifest: &m,
            exec: "/bin/bash".into(),
            root: root.clone(),
            inherit: vec![],
            env: vec![],
            stdout: Stdio::piped(),
            stderr: Stdio::piped(),
        })
        .unwrap()
        .wait_with_output()
        .unwrap();
        let so = String::from_utf8_lossy(&out.stdout);
        let se = String::from_utf8_lossy(&out.stderr);
        assert!(so.contains("done"), "stdout={so} stderr={se}");
        assert_eq!(std::fs::read_to_string(root.join("ok.txt")).unwrap(), "hi\n");
        assert!(!Path::new("/Users/Shared/fb-escape.txt").exists(), "sandbox escape!");
        assert!(se.contains("Operation not permitted"), "stderr={se}");
        assert!(!so.contains("READ_ETC"), "sandbox allowed reading /etc/hosts");
        std::fs::remove_dir_all(root).ok();
    }
}
