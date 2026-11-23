//! CUDA-Q-style quantum kernels: statevector evolution on the Apple GPU.

use fb_cuda::quantum::QuantumState;
use fb_cuda::CudaRuntime;

const EPS: f32 = 1e-5;

#[test]
fn x_gate_flips() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 1).unwrap();
    q.x(0).unwrap();
    let p = q.probabilities().unwrap();
    assert!((p[0] - 0.0).abs() < EPS, "P(0) should be 0, got {}", p[0]);
    assert!((p[1] - 1.0).abs() < EPS, "P(1) should be 1, got {}", p[1]);
}

#[test]
fn hadamard_superposition() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 1).unwrap();
    q.h(0).unwrap();
    let p = q.probabilities().unwrap();
    assert!((p[0] - 0.5).abs() < EPS);
    assert!((p[1] - 0.5).abs() < EPS);
    // <Z> of |+> is 0.
    assert!(q.expect_z(0).unwrap().abs() < EPS);
}

#[test]
fn bell_state() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 2).unwrap();
    q.h(0).unwrap();
    q.cx(0, 1).unwrap();
    let p = q.probabilities().unwrap();
    // (|00> + |11>)/sqrt(2): only indices 0 and 3 populated, each ~0.5.
    assert!((p[0] - 0.5).abs() < EPS, "{p:?}");
    assert!((p[1] - 0.0).abs() < EPS, "{p:?}");
    assert!((p[2] - 0.0).abs() < EPS, "{p:?}");
    assert!((p[3] - 0.5).abs() < EPS, "{p:?}");
}

#[test]
fn ghz_three_qubit() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 3).unwrap();
    q.h(0).unwrap();
    q.cx(0, 1).unwrap();
    q.cx(1, 2).unwrap();
    let p = q.probabilities().unwrap();
    // (|000> + |111>)/sqrt(2): indices 0 and 7.
    for (i, pi) in p.iter().enumerate() {
        let want = if i == 0 || i == 7 { 0.5 } else { 0.0 };
        assert!((pi - want).abs() < EPS, "P({i:03b}) = {pi}, want {want}");
    }
}

#[test]
fn rx_pi_is_x_up_to_phase() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 1).unwrap();
    q.rx(0, std::f32::consts::PI).unwrap();
    let p = q.probabilities().unwrap();
    assert!((p[1] - 1.0).abs() < 1e-4, "RX(pi)|0> should land on |1>, got {p:?}");
}

#[test]
fn expect_z_basis_states() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 1).unwrap();
    assert!((q.expect_z(0).unwrap() - 1.0).abs() < EPS); // |0> -> +1
    q.x(0).unwrap();
    assert!((q.expect_z(0).unwrap() + 1.0).abs() < EPS); // |1> -> -1
}

#[test]
fn sampling_matches_distribution() {
    let rt = CudaRuntime::new().unwrap();
    let mut q = QuantumState::new(&rt, 2).unwrap();
    q.h(0).unwrap();
    q.cx(0, 1).unwrap();
    let counts = q.sample(100_000, 0xC0FFEE).unwrap();
    // Only |00> and |11> occur; each ~50%.
    assert_eq!(counts[1], 0, "|01> should never occur in a Bell state");
    assert_eq!(counts[2], 0, "|10> should never occur in a Bell state");
    let f00 = counts[0] as f64 / 100_000.0;
    let f11 = counts[3] as f64 / 100_000.0;
    assert!((f00 - 0.5).abs() < 0.02, "P(00)~0.5, got {f00}");
    assert!((f11 - 0.5).abs() < 0.02, "P(11)~0.5, got {f11}");
}

#[test]
fn normalization_preserved_over_deep_circuit() {
    // A longer circuit that forces many GPU dispatches; the statevector must stay a unit
    // vector, which checks both correctness and that the serial-dispatch ordering holds.
    let rt = CudaRuntime::new().unwrap();
    let n = 6;
    let mut q = QuantumState::new(&rt, n).unwrap();
    for layer in 0..8 {
        for t in 0..n {
            q.h(t).unwrap();
            q.rz(t, 0.3 * (layer + 1) as f32).unwrap();
        }
        for c in 0..n - 1 {
            q.cx(c, c + 1).unwrap();
        }
    }
    let p = q.probabilities().unwrap();
    let norm: f32 = p.iter().sum();
    assert!((norm - 1.0).abs() < 1e-3, "statevector lost normalization: {norm}");
}

#[test]
fn qft_on_basis_state() {
    // QFT of |0...0> is the uniform superposition (all amplitudes 1/sqrt(2^n)).
    let rt = CudaRuntime::new().unwrap();
    let n = 4u32;
    let mut q = QuantumState::new(&rt, n).unwrap();
    qft(&mut q, n);
    let p = q.probabilities().unwrap();
    let want = 1.0f32 / (1u32 << n) as f32;
    for (i, pi) in p.iter().enumerate() {
        assert!((pi - want).abs() < 1e-4, "QFT uniform amplitude at {i}: {pi} vs {want}");
    }
}

fn qft(q: &mut QuantumState, n: u32) {
    for j in 0..n {
        q.h(j).unwrap();
        for k in (j + 1)..n {
            let angle = std::f32::consts::PI / (1u32 << (k - j)) as f32;
            q.cphase(k, j, angle).unwrap();
        }
    }
}
