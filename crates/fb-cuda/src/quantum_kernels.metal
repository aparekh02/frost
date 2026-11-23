// Quantum statevector gate kernels for fb-cuda's CUDA-Q-style backend.
// The statevector is an array of `float2` complex amplitudes of length 2^n. Applying a
// single-qubit (optionally singly-controlled) gate touches each amplitude pair exactly
// once, so one GPU thread handles one pair — an embarrassingly parallel O(2^n) sweep.
#include <metal_stdlib>
using namespace metal;

struct GateP {
    uint target;       // qubit the 2x2 matrix acts on
    uint control;      // control qubit (valid only if has_control != 0)
    uint has_control;  // 0 = unconditional, 1 = apply only where control bit == 1
    uint n;            // number of qubits
    float m[8];        // row-major complex 2x2: [m00r,m00i, m01r,m01i, m10r,m10i, m11r,m11i]
};

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Apply the 2x2 (optionally controlled) gate. Launch with 2^(n-1) threads.
kernel void qgate(device float2* s      [[buffer(0)]],
                  constant GateP& p     [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    uint half_states = 1u << (p.n - 1u);
    if (gid >= half_states) return;

    uint t = p.target;
    uint lower = gid & ((1u << t) - 1u);          // bits below the target
    uint upper = (gid >> t) << (t + 1u);          // bits above the target, shifted up
    uint i0 = upper | lower;                       // amplitude with target bit = 0
    uint i1 = i0 | (1u << t);                      // amplitude with target bit = 1

    if (p.has_control != 0u) {
        // i0 and i1 differ only in the target bit, and control != target, so the control
        // bit is identical for the pair; testing i0 is sufficient.
        if (((i0 >> p.control) & 1u) == 0u) return;
    }

    float2 a0 = s[i0];
    float2 a1 = s[i1];
    float2 m00 = float2(p.m[0], p.m[1]);
    float2 m01 = float2(p.m[2], p.m[3]);
    float2 m10 = float2(p.m[4], p.m[5]);
    float2 m11 = float2(p.m[6], p.m[7]);
    s[i0] = cmul(m00, a0) + cmul(m01, a1);
    s[i1] = cmul(m10, a0) + cmul(m11, a1);
}
