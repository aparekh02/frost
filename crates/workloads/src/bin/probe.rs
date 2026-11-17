//! Isolation probe: reports what the capsule can and cannot do.
use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

extern "C" {
    fn qos_class_self() -> u32;
}

fn main() {
    let home_ssh = "/Users/Shared"; // world-readable dir outside the capsule root
    let can = |r: bool| if r { "ALLOWED" } else { "denied" };
    let read_etc = std::fs::read("/etc/hosts").is_ok();
    let list_shared = std::fs::read_dir(home_ssh).is_ok();
    let user_home = std::env::var("FROST_REAL_HOME").unwrap_or_else(|_| "/Users".into());
    let list_home = std::fs::read_dir(&user_home).is_ok();
    let write_tmp = std::fs::File::create("/private/tmp/fb-probe-escape").and_then(|mut f| f.write_all(b"x")).is_ok();
    let write_root = std::fs::write("inside.txt", b"ok").is_ok();
    let net = TcpStream::connect_timeout(&"1.1.1.1:443".parse().unwrap(), Duration::from_millis(500)).is_ok();
    let fork = unsafe {
        let p = libc::fork();
        if p == 0 {
            libc::_exit(0);
        }
        if p > 0 {
            libc::waitpid(p, std::ptr::null_mut(), 0);
        }
        p > 0
    };
    let exec = std::process::Command::new("/bin/ls").output().is_ok();
    let qos = unsafe { qos_class_self() };
    let mut cpus = std::collections::BTreeSet::new();
    let t0 = std::time::Instant::now();
    let mut x = 0u64;
    while t0.elapsed() < Duration::from_millis(200) {
        for i in 0..10000 { x = x.wrapping_mul(6364136223846793005).wrapping_add(i); }
        cpus.insert(fb_core::sched::current_cpu());
    }
    std::hint::black_box(x);
    let fds_ok = std::env::var("FROST_CTL_FD").is_ok();
    let gpu = fb_client::Frost::connect().map(|f| f.gpu_name.clone());
    println!("read /etc/hosts:   {}", can(read_etc));
    println!("list /Users/Shared:{}", can(list_shared));
    println!("write /private/tmp:{}", can(write_tmp));
    println!("write capsule root:{}", can(write_root));
    println!("network egress:    {}", can(net));
    println!("fork:              {}", can(fork));
    println!("exec /bin/ls:      {}", can(exec));
    workloads::result(serde_json::json!({
        "probe": true, "read_etc": read_etc, "list_shared": list_shared, "list_home": list_home,
        "write_tmp": write_tmp, "write_root": write_root, "network": net, "fork": fork, "exec": exec,
        "qos": qos, "cpus": cpus, "fds": fds_ok, "gpu": gpu.ok(),
    }));
}
