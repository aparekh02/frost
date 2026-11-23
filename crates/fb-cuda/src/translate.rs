//! CUDA-C (subset) → Metal Shading Language source translator.
//!
//! This is deliberately a **practical subset**, not a full CUDA compiler. It is enough to
//! author real AI/compute kernels (elementwise, SAXPY, GEMM, ReLU, reductions with shared
//! memory) and run them on the Apple GPU. What it handles:
//!
//! * `__global__` entry points, with `<<<grid, block>>>` launch semantics preserved.
//! * `threadIdx` / `blockIdx` / `blockDim` / `gridDim` (as `uint3`), mapped to Metal's
//!   `thread_position_in_threadgroup`, `threadgroup_position_in_grid`,
//!   `threads_per_threadgroup`, `threadgroups_per_grid`.
//! * Pointer kernel params → `device T* [[buffer(i)]]`; scalar params → `constant T& [[buffer(i)]]`.
//! * `__device__` helper functions (value/scalar args), `__shared__` → `threadgroup`,
//!   `__syncthreads()` → `threadgroup_barrier`.
//! * The `f`-suffixed math builtins (`expf`, `sqrtf`, `fmaxf`, …) → their MSL names.
//!
//! What is out of scope (documented, not silently mistranslated): texture/surface objects,
//! warp-shuffle intrinsics, device `atomicAdd` (Metal's atomics have a different API),
//! `__device__` helpers that take pointers (MSL requires an explicit address space), and
//! dynamic parallelism. Kernels using those should be hand-written in MSL and loaded with
//! [`crate::CudaRuntime::load_msl`].

use crate::{KernelSig, Param};
use anyhow::{bail, Result};

/// The result of translating a CUDA-C source string.
pub struct Translated {
    /// The generated Metal Shading Language source.
    pub msl: String,
    /// One signature per `__global__` kernel found.
    pub functions: Vec<KernelSig>,
}

/// Translate a CUDA-C subset source string to MSL.
pub fn cuda_to_msl(src: &str) -> Result<Translated> {
    // 1. Strip preprocessor lines MSL supplies itself or cannot use.
    let pre: String = src
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("#include") || t.starts_with("#pragma"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    // 2. Whole-source identifier rewrites (qualifiers + f-suffixed math builtins). These
    //    run before kernel extraction so helper functions and kernel bodies are both fixed.
    //    `__global__` is intentionally preserved for the extraction pass below.
    let mut s = pre;
    for (from, to) in [
        ("__restrict__", ""),
        ("__restrict", ""),
        ("__forceinline__", ""),
        ("__inline__", ""),
        ("__host__", ""),
        ("__device__", "inline"),
        ("__launch_bounds__", ""),
        ("__shared__", "threadgroup"),
    ] {
        s = replace_ident(&s, from, to);
    }
    s = s.replace("__syncthreads()", "threadgroup_barrier(mem_flags::mem_threadgroup)");
    s = s.replace("__syncwarp()", "simdgroup_barrier(mem_flags::mem_none)");
    for (from, to) in [
        ("expf", "exp"), ("__expf", "exp"), ("logf", "log"), ("__logf", "log"),
        ("log2f", "log2"), ("log10f", "log10"), ("sqrtf", "sqrt"), ("rsqrtf", "rsqrt"),
        ("sinf", "sin"), ("cosf", "cos"), ("tanf", "tan"), ("tanhf", "tanh"),
        ("asinf", "asin"), ("acosf", "acos"), ("atanf", "atan"), ("atan2f", "atan2"),
        ("powf", "pow"), ("fabsf", "fabs"), ("fminf", "fmin"), ("fmaxf", "fmax"),
        ("fmaf", "fma"), ("floorf", "floor"), ("ceilf", "ceil"), ("roundf", "round"),
        ("expf2", "exp2"), ("exp2f", "exp2"),
    ] {
        s = replace_ident(&s, from, to);
    }

    // 3. Extract and transform each __global__ kernel.
    let mut out = String::from(PRELUDE);
    let mut functions = Vec::new();
    let mut rest = s.as_str();
    loop {
        let Some(gidx) = find_ident(rest, "__global__") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..gidx]);
        let after = &rest[gidx + "__global__".len()..];

        // Find the parameter list's opening paren; the kernel name is the identifier
        // immediately before it.
        let Some(lparen) = after.find('(') else {
            bail!("`__global__` with no parameter list");
        };
        let header = after[..lparen].trim_end();
        let name = last_ident(header)
            .ok_or_else(|| anyhow::anyhow!("could not find a kernel name before `(`"))?;

        let params_end = matched(after, lparen, b'(', b')')?;
        let params_str = &after[lparen + 1..params_end];

        // Body: the next `{ ... }`.
        let body_region = &after[params_end + 1..];
        let Some(lbrace_rel) = body_region.find('{') else {
            bail!("kernel `{name}` has no body (prototype-only kernels aren't supported)");
        };
        let lbrace = params_end + 1 + lbrace_rel;
        let body_end = matched(after, lbrace, b'{', b'}')?;
        let body = &after[lbrace + 1..body_end];

        let (new_params, sig_params) = transform_params(params_str)?;
        functions.push(KernelSig { name: name.clone(), params: sig_params });

        let joined = if new_params.is_empty() {
            BUILTIN_PARAMS.trim_start_matches(", ").to_string()
        } else {
            format!("{new_params}{BUILTIN_PARAMS}")
        };
        out.push_str(&format!(
            "kernel void {name}(\n    {joined}\n) {{\n{BODY_PRELUDE}{body}}}\n"
        ));

        rest = &after[body_end + 1..];
    }

    Ok(Translated { msl: out, functions })
}

/// MSL preamble + CUDA-isms that have no keyword equivalent.
const PRELUDE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\n";

/// Builtin parameters appended to every kernel signature.
const BUILTIN_PARAMS: &str = ",\n    uint3 _cu_tid [[thread_position_in_threadgroup]],\
\n    uint3 _cu_ctaid [[threadgroup_position_in_grid]],\
\n    uint3 _cu_ntid [[threads_per_threadgroup]],\
\n    uint3 _cu_nctaid [[threadgroups_per_grid]]";

/// Aliases injected at the top of every kernel body so CUDA code compiles unchanged.
const BODY_PRELUDE: &str = "    uint3 threadIdx = _cu_tid; uint3 blockIdx = _cu_ctaid; \
uint3 blockDim = _cu_ntid; uint3 gridDim = _cu_nctaid;\n    (void)threadIdx; (void)blockIdx; \
(void)blockDim; (void)gridDim;\n";

/// Transform a CUDA parameter list into MSL params and record the signature.
fn transform_params(params: &str) -> Result<(String, Vec<Param>)> {
    let mut out_parts = Vec::new();
    let mut sig = Vec::new();
    let mut idx = 0usize;
    for raw in split_top_level(params) {
        let p = raw.trim();
        if p.is_empty() || p == "void" {
            continue;
        }
        let is_ptr = p.contains('*');
        let name = last_ident(p)
            .ok_or_else(|| anyhow::anyhow!("parameter `{p}` has no name"))?;
        let name_pos = p.rfind(&name).unwrap();
        let typ = p[..name_pos].trim();
        if typ.is_empty() {
            bail!("parameter `{p}` has no type");
        }
        let byte_size = if is_ptr { 8 } else { scalar_size(typ) };
        if is_ptr {
            out_parts.push(format!("device {typ} {name} [[buffer({idx})]]"));
        } else {
            out_parts.push(format!("constant {typ}& {name} [[buffer({idx})]]"));
        }
        sig.push(Param { name, is_pointer: is_ptr, byte_size });
        idx += 1;
    }
    Ok((out_parts.join(",\n    "), sig))
}

/// Best-effort byte size of a scalar C type string (for the C ABI launch path). The
/// macOS/Apple-Silicon LP64 model: `long`, `size_t`, pointers are 8 bytes.
fn scalar_size(typ: &str) -> usize {
    let t = typ.replace("const", "");
    let t = t.trim();
    if t.contains("double") {
        8
    } else if t.contains("long long") || t.contains("size_t") || t.contains("ptrdiff_t")
        || t.contains("int64") || t.contains("uint64") || t.contains("long")
    {
        8
    } else if t.contains("short") || t.contains("half") || t.contains("int16") || t.contains("uint16") {
        2
    } else if t.contains("char") || t.contains("bool") || t.contains("int8") || t.contains("uint8") {
        1
    } else {
        // int, unsigned, uint, float, int32, uint32, …
        4
    }
}

/// Split on top-level commas, respecting (), [], <> nesting.
fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    parts
}

/// Given a string ending just before `(`, return the trailing identifier (the name).
fn last_ident(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 && (bytes[end - 1] as char).is_whitespace() {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && is_ident_byte(bytes[start - 1]) {
        start -= 1;
    }
    if start == end {
        None
    } else {
        Some(s[start..end].to_string())
    }
}

/// Find `ident` as a whole word (not a substring of a larger identifier).
fn find_ident(s: &str, ident: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let n = ident.len();
    let mut i = 0;
    while i + n <= bytes.len() {
        if &s[i..i + n] == ident {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok = i + n == bytes.len() || !is_ident_byte(bytes[i + n]);
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Replace every whole-word occurrence of `from` with `to`.
fn replace_ident(s: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = find_ident(rest, from) {
        out.push_str(&rest[..i]);
        out.push_str(to);
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

/// From `open` at index `start` in `s`, return the index of the matching `close`.
fn matched(s: &str, start: usize, open: u8, close: u8) -> Result<usize> {
    let bytes = s.as_bytes();
    debug_assert_eq!(bytes[start], open);
    let mut depth = 0i32;
    let mut i = start;
    while i < bytes.len() {
        if bytes[i] == open {
            depth += 1;
        } else if bytes[i] == close {
            depth -= 1;
            if depth == 0 {
                return Ok(i);
            }
        }
        i += 1;
    }
    bail!("unbalanced `{}`/`{}`", open as char, close as char)
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
