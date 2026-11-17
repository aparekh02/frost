// Frost Bridge built-in kernels. Compiled at daemon start with newLibraryWithSource.
// Convention: arena buffers are bound at indices 0..n_args-1, the inline Job.params
// block is bound with setBytes at index n_args.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- id 4: noop
kernel void fb_noop(device uint* out [[buffer(0)]],
                    constant uint* p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid == 0) { out[0] = p[0] + 1; }
}

// ---------------------------------------------------------------- id 1: vadd
struct NParams { uint n; };
kernel void fb_vadd_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* o       [[buffer(2)]],
                        constant NParams& p   [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) o[gid] = a[gid] + b[gid];
}

// ---------------------------------------------------------------- id 5: saxpy  y = a*x + y
struct SaxpyParams { uint n; float a; };
kernel void fb_saxpy_f32(device const float* x [[buffer(0)]],
                         device float* y       [[buffer(1)]],
                         constant SaxpyParams& p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) y[gid] = fma(p.a, x[gid], y[gid]);
}

// ---------------------------------------------------------------- id 2: gemv  y = A x
// One SIMD-group (32 lanes) per row; launch grid = rows*32, threadgroup = 32*k.
struct GemvParams { uint rows; uint cols; };
kernel void fb_gemv_f32(device const float* A [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* y       [[buffer(2)]],
                        constant GemvParams& p [[buffer(3)]],
                        uint gid  [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= p.rows) return;
    device const float4* r4 = (device const float4*)(A + (ulong)row * p.cols);
    device const float4* x4 = (device const float4*)x;
    uint c4 = p.cols / 4;
    float acc = 0.0f;
    for (uint i = lane; i < c4; i += 32) acc += dot(r4[i], x4[i]);
    for (uint i = c4 * 4 + lane; i < p.cols; i += 32) acc += A[(ulong)row * p.cols + i] * x[i];
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------- id 3: sha256 batch
// Hash `n` independent messages of `len` bytes each, laid out every `stride` bytes.
// Output: 32-byte big-endian digests, contiguous.
struct ShaParams { uint n; uint len; uint stride; };

constant uint K256[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

static inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

static inline uchar msg_byte(device const uchar* m, uint len, uint i) {
    if (i < len) return m[i];
    if (i == len) return 0x80;
    return 0;
}

kernel void fb_sha256_batch(device const uchar* in [[buffer(0)]],
                            device uchar* out      [[buffer(1)]],
                            constant ShaParams& p  [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    device const uchar* m = in + (ulong)gid * p.stride;
    uint len = p.len;
    uint nblocks = (len + 9 + 63) / 64;
    ulong bitlen = (ulong)len * 8;
    uint h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint w[64];
    for (uint b = 0; b < nblocks; ++b) {
        uint base = b * 64;
        bool last = (b == nblocks - 1);
        for (uint t = 0; t < 16; ++t) {
            uint v = 0;
            for (uint k = 0; k < 4; ++k) {
                uint i = base + t * 4 + k;
                uchar byte;
                if (last && t >= 14) {
                    uint shift = (7 - ((t - 14) * 4 + k)) * 8;
                    byte = (uchar)(bitlen >> shift);
                } else {
                    byte = msg_byte(m, len, i);
                }
                v = (v << 8) | byte;
            }
            w[t] = v;
        }
        for (uint t = 16; t < 64; ++t) {
            uint s0 = rotr(w[t-15], 7) ^ rotr(w[t-15], 18) ^ (w[t-15] >> 3);
            uint s1 = rotr(w[t-2], 17) ^ rotr(w[t-2], 19) ^ (w[t-2] >> 10);
            w[t] = w[t-16] + s0 + w[t-7] + s1;
        }
        uint a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (uint t = 0; t < 64; ++t) {
            uint S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint ch = (e & f) ^ (~e & g);
            uint t1 = hh + S1 + ch + K256[t] + w[t];
            uint S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint mj = (a & bb) ^ (a & c) ^ (bb & c);
            uint t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h[0]+=a; h[1]+=bb; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    device uchar* o = out + (ulong)gid * 32;
    for (uint i = 0; i < 8; ++i) {
        o[i*4+0] = (uchar)(h[i] >> 24); o[i*4+1] = (uchar)(h[i] >> 16);
        o[i*4+2] = (uchar)(h[i] >> 8);  o[i*4+3] = (uchar)(h[i]);
    }
}
