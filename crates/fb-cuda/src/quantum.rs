//! A CUDA-Q-style quantum kernel backend: a **statevector simulator whose circuit
//! evolution runs on the Apple GPU** via Metal.
//!
//! An `n`-qubit state is `2^n` complex amplitudes held in a unified-memory device buffer.
//! Each gate is a Metal compute dispatch over `2^(n-1)` amplitude pairs (see
//! `quantum_kernels.metal`). Gates are accumulated and flushed as one serial command
//! buffer, so the whole circuit is one GPU submission. Measurement/expectation read the
//! amplitudes back over unified memory (no copy) and reduce on the CPU.
//!
//! This is the quantum analogue of the classical CUDA path: you build a circuit in the
//! host language and the heavy linear-algebra "kernel" executes on the GPU.
//!
//! ```no_run
//! use fb_cuda::{CudaRuntime, quantum::QuantumState};
//! let rt = CudaRuntime::new().unwrap();
//! let mut q = QuantumState::new(&rt, 2).unwrap();   // |00>
//! q.h(0);                                           // superposition
//! q.cx(0, 1);                                       // entangle -> Bell state
//! let p = q.probabilities().unwrap();               // ~[0.5, 0, 0, 0.5]
//! ```

use crate::{CudaRuntime, Function, KernelSig, Param};
use anyhow::{bail, Result};
use std::sync::Arc;

const GATE_SRC: &str = include_str!("quantum_kernels.metal");

/// Matches `struct GateP` in `quantum_kernels.metal` (repr-C, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct GateParams {
    target: u32,
    control: u32,
    has_control: u32,
    n: u32,
    m: [f32; 8],
}

/// A simulated quantum register backed by a GPU statevector.
pub struct QuantumState {
    rt: Arc<CudaRuntime>,
    func: Function,
    n: u32,
    dim: usize,
    state: crate::DevicePtr,
    pending: Vec<(Vec<u8>, usize)>,
}

impl QuantumState {
    /// Allocate an `n`-qubit register initialized to |0…0⟩. `n` must be in `1..=28`
    /// (2^28 complex amplitudes ≈ 2 GiB; raise if you have the memory).
    pub fn new(rt: &Arc<CudaRuntime>, n: u32) -> Result<QuantumState> {
        if !(1..=28).contains(&n) {
            bail!("qubit count {n} out of supported range 1..=28");
        }
        let module = rt.load_msl(
            GATE_SRC,
            vec![KernelSig {
                name: "qgate".into(),
                params: vec![
                    Param { name: "s".into(), is_pointer: true, byte_size: 8 },
                    Param { name: "p".into(), is_pointer: false, byte_size: std::mem::size_of::<GateParams>() },
                ],
            }],
        )?;
        let func = module.function("qgate")?;
        let dim = 1usize << n;
        let state = rt.malloc(dim * 8)?; // float2 == 8 bytes
        let mut q = QuantumState { rt: rt.clone(), func, n, dim, state, pending: Vec::new() };
        q.reset()?;
        Ok(q)
    }

    /// Reset to |0…0⟩.
    pub fn reset(&mut self) -> Result<()> {
        self.pending.clear();
        let zeros = vec![0f32; self.dim * 2];
        self.rt.copy_to_device(self.state, &zeros)?;
        // amplitude[0] = (1, 0)
        self.rt.copy_to_device(self.state, &[1.0f32, 0.0f32])?;
        Ok(())
    }

    pub fn num_qubits(&self) -> u32 {
        self.n
    }

    fn check(&self, q: u32) -> Result<()> {
        if q >= self.n {
            bail!("qubit index {q} out of range for {}-qubit register", self.n);
        }
        Ok(())
    }

    /// Apply an arbitrary single-qubit gate given as a row-major complex 2x2 matrix
    /// `[m00r,m00i, m01r,m01i, m10r,m10i, m11r,m11i]`.
    pub fn apply_1q(&mut self, target: u32, m: [f32; 8]) -> Result<()> {
        self.check(target)?;
        self.enqueue(GateParams { target, control: 0, has_control: 0, n: self.n, m });
        Ok(())
    }

    /// Apply a singly-controlled single-qubit gate.
    pub fn apply_controlled_1q(&mut self, control: u32, target: u32, m: [f32; 8]) -> Result<()> {
        self.check(control)?;
        self.check(target)?;
        if control == target {
            bail!("control and target must differ (both {target})");
        }
        self.enqueue(GateParams { target, control, has_control: 1, n: self.n, m });
        Ok(())
    }

    fn enqueue(&mut self, p: GateParams) {
        let bytes = unsafe {
            std::slice::from_raw_parts(&p as *const GateParams as *const u8, std::mem::size_of::<GateParams>())
        }
        .to_vec();
        self.pending.push((bytes, self.dim / 2));
    }

    /// Flush all pending gates to the GPU (called automatically before any read).
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let ops = std::mem::take(&mut self.pending);
        self.rt.dispatch_serial(&self.func, self.state, &ops)
    }

    // ---- standard gates -------------------------------------------------------------
    pub fn x(&mut self, q: u32) -> Result<()> { self.apply_1q(q, [0.,0., 1.,0., 1.,0., 0.,0.]) }
    pub fn y(&mut self, q: u32) -> Result<()> { self.apply_1q(q, [0.,0., 0.,-1., 0.,1., 0.,0.]) }
    pub fn z(&mut self, q: u32) -> Result<()> { self.apply_1q(q, [1.,0., 0.,0., 0.,0., -1.,0.]) }
    pub fn h(&mut self, q: u32) -> Result<()> {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        self.apply_1q(q, [s,0., s,0., s,0., -s,0.])
    }
    pub fn s(&mut self, q: u32) -> Result<()> { self.apply_1q(q, [1.,0., 0.,0., 0.,0., 0.,1.]) }
    pub fn sdg(&mut self, q: u32) -> Result<()> { self.apply_1q(q, [1.,0., 0.,0., 0.,0., 0.,-1.]) }
    pub fn t(&mut self, q: u32) -> Result<()> {
        let (c, s) = (std::f32::consts::FRAC_PI_4.cos(), std::f32::consts::FRAC_PI_4.sin());
        self.apply_1q(q, [1.,0., 0.,0., 0.,0., c, s])
    }
    pub fn tdg(&mut self, q: u32) -> Result<()> {
        let (c, s) = (std::f32::consts::FRAC_PI_4.cos(), std::f32::consts::FRAC_PI_4.sin());
        self.apply_1q(q, [1.,0., 0.,0., 0.,0., c, -s])
    }

    /// Rotation about X by `theta` radians.
    pub fn rx(&mut self, q: u32, theta: f32) -> Result<()> {
        let (c, s) = ((theta / 2.).cos(), (theta / 2.).sin());
        self.apply_1q(q, [c,0., 0.,-s, 0.,-s, c,0.])
    }
    /// Rotation about Y by `theta` radians.
    pub fn ry(&mut self, q: u32, theta: f32) -> Result<()> {
        let (c, s) = ((theta / 2.).cos(), (theta / 2.).sin());
        self.apply_1q(q, [c,0., -s,0., s,0., c,0.])
    }
    /// Rotation about Z by `theta` radians.
    pub fn rz(&mut self, q: u32, theta: f32) -> Result<()> {
        let (c, s) = ((theta / 2.).cos(), (theta / 2.).sin());
        self.apply_1q(q, [c,-s, 0.,0., 0.,0., c,s])
    }
    /// Phase gate: |1⟩ → e^{iλ}|1⟩.
    pub fn phase(&mut self, q: u32, lambda: f32) -> Result<()> {
        self.apply_1q(q, [1.,0., 0.,0., 0.,0., lambda.cos(), lambda.sin()])
    }

    // ---- controlled gates -----------------------------------------------------------
    /// Controlled-NOT.
    pub fn cx(&mut self, control: u32, target: u32) -> Result<()> {
        self.apply_controlled_1q(control, target, [0.,0., 1.,0., 1.,0., 0.,0.])
    }
    /// Controlled-Y.
    pub fn cy(&mut self, control: u32, target: u32) -> Result<()> {
        self.apply_controlled_1q(control, target, [0.,0., 0.,-1., 0.,1., 0.,0.])
    }
    /// Controlled-Z.
    pub fn cz(&mut self, control: u32, target: u32) -> Result<()> {
        self.apply_controlled_1q(control, target, [1.,0., 0.,0., 0.,0., -1.,0.])
    }
    /// Controlled phase: |11⟩ → e^{iλ}|11⟩ (the workhorse of the QFT).
    pub fn cphase(&mut self, control: u32, target: u32, lambda: f32) -> Result<()> {
        self.apply_controlled_1q(control, target, [1.,0., 0.,0., 0.,0., lambda.cos(), lambda.sin()])
    }

    // ---- readout --------------------------------------------------------------------
    /// The raw complex amplitudes `(re, im)` as a `Vec` of length `2^n`.
    pub fn amplitudes(&mut self) -> Result<Vec<(f32, f32)>> {
        self.flush()?;
        let mut raw = vec![0f32; self.dim * 2];
        self.rt.copy_from_device(&mut raw, self.state)?;
        Ok(raw.chunks_exact(2).map(|c| (c[0], c[1])).collect())
    }

    /// Measurement probabilities `|amp|^2`, length `2^n`.
    pub fn probabilities(&mut self) -> Result<Vec<f32>> {
        Ok(self.amplitudes()?.into_iter().map(|(re, im)| re * re + im * im).collect())
    }

    /// Expectation value ⟨Z⟩ of qubit `q` (+1 for |0⟩, −1 for |1⟩).
    pub fn expect_z(&mut self, q: u32) -> Result<f32> {
        self.check(q)?;
        let probs = self.probabilities()?;
        let mut acc = 0.0f32;
        for (idx, p) in probs.iter().enumerate() {
            let bit = (idx >> q) & 1;
            acc += if bit == 0 { *p } else { -*p };
        }
        Ok(acc)
    }

    /// Sample the register `shots` times, returning a count per basis state (index = the
    /// integer whose bit `q` is the measured value of qubit `q`). Deterministic given `seed`.
    pub fn sample(&mut self, shots: usize, seed: u64) -> Result<Vec<usize>> {
        let probs = self.probabilities()?;
        // Cumulative distribution.
        let mut cdf = Vec::with_capacity(probs.len());
        let mut acc = 0.0f64;
        for p in &probs {
            acc += *p as f64;
            cdf.push(acc);
        }
        let total = *cdf.last().unwrap_or(&1.0);
        let mut counts = vec![0usize; probs.len()];
        let mut rng = SplitMix64 { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) };
        for _ in 0..shots {
            let r = (rng.next_f64()) * total;
            // Binary search for the first cdf >= r.
            let idx = cdf.partition_point(|&c| c < r).min(probs.len() - 1);
            counts[idx] += 1;
        }
        Ok(counts)
    }

    /// Device pointer to the raw statevector (unified memory), for interop.
    pub fn statevector_ptr(&self) -> crate::DevicePtr {
        self.state
    }
}

impl Drop for QuantumState {
    fn drop(&mut self) {
        let _ = self.rt.free(self.state);
    }
}

/// Avoid pulling in a dependency: a small, well-mixed PRNG for deterministic sampling.
struct SplitMix64 {
    s: u64,
}
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
