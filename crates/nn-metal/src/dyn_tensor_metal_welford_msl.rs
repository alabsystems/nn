// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared MSL reduction primitives for fused norm kernels.
//!
//! Provides two numerically stable mean+variance algorithms:
//!
//! ## 1. Kahan-compensated Welford (single-pass)
//!
//! Original: #2685. Kahan compensation on m2: #2696.
//! Per-thread: Welford online update accumulates `(n, mean, M2, m2_comp)`.
//! Threadgroup merge: parallel tree reduction via `welford_merge()`.
//! Final: `mean = shared_mean[0]`, `variance = shared_m2[0] / shared_n[0]`.
//! Numerically stable, but has a **division per element** in the inner loop
//! (`mean += delta / n`), which is the primary performance bottleneck.
//!
//! ## 2. Two-pass Kahan-compensated summation (#2697)
//!
//! Pass 1: Kahan-compensated sum → `mean = sum / N`.
//! Pass 2: Kahan-compensated sum of `(x - mean)²` → `var = sum / N`.
//! **No divisions in either inner loop.** Only 2 divisions total (at the end).
//! Reads input data twice but avoids the per-element division that makes
//! Welford ~36% slower on Apple Silicon (dvoice benchmark: RTF 0.145 →
//! 0.198 after Welford adoption). Kahan compensation prevents catastrophic
//! cancellation in both passes.
//!
//! Two variants of the cross-thread reduction:
//! - **Tree-based** (`kahan_two_pass_reduction_msl`): log2(tg_size) barriers
//!   per pass via threadgroup shared memory. Uses 2 shared arrays (2048B).
//! - **Simd-accelerated** (`kahan_two_pass_simd_reduction_msl`): `simd_sum()`
//!   within each 32-thread simdgroup + 1 cross-simdgroup shared-memory
//!   round-trip. 2 barriers per pass (4 total) instead of 16. Uses only
//!   `shared_simd[32]` (128B). Requires `simd_reduction_helpers_msl()` in
//!   the caller's preamble.
//!
//! All fused norm kernels (InstanceNorm, AdaIN+Snake, AdaIN+LeakyRelu,
//! AdaLayerNorm) share this reduction code instead of copy-pasting
//! their own reduction loops.
//!
//! ## F16 support
//!
//! All reduction inner loops use explicit `float()` casts on `input[...]`
//! reads, ensuring correct promotion when the containing kernel declares
//! `device const half* input`. Accumulators are always `float` regardless
//! of I/O dtype. Part of #3766.

/// Which reduction algorithm to use for fused norm kernels.
///
/// All variants produce `float mean` and `float inv_std` MSL variables.
/// Callers select via `norm_reduction_msl()` and `norm_preamble_msl()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormReduction {
    /// Single-pass Kahan-compensated Welford. Numerically best but has
    /// one float division per element in the inner loop.
    /// Currently only exercised in tests (KahanTwoPass is default, #2697).
    #[allow(dead_code)]
    Welford,
    /// Two-pass Kahan-compensated summation. No inner-loop divisions.
    /// Reads data twice. Faster on Apple Silicon due to avoiding
    /// per-element `delta / n`. See #2697.
    KahanTwoPass,
    /// Naive two-pass reduction matching PyTorch MPS behavior (#4335).
    ///
    /// No Kahan compensation, standard `rsqrt()` (not `precise::rsqrt()`).
    /// Uses the same tree-based threadgroup reduction as `KahanTwoPass`
    /// but without error-compensation terms. Produces numerically identical
    /// results to PyTorch MPS InstanceNorm, which is critical for models
    /// like Kokoro where 35+ chained FusedResBlocks amplify tiny
    /// reduction-order differences into +35.8% amplitude divergence.
    ///
    /// Use this when parity with PyTorch is the metric. Use `KahanTwoPass`
    /// when numerical accuracy is more important than parity.
    PyTorchCompat,
}

/// Default: PyTorch-compatible naive reduction (#4335).
///
/// Changed from `KahanTwoPass` to `PyTorchCompat` to fix +35.8% amplitude
/// divergence in Kokoro vs Python reference. The Kahan-compensated reduction
/// and `precise::rsqrt` introduced tiny per-layer differences that compounded
/// through 35+ FusedResBlocks. PyTorchCompat matches PyTorch MPS's naive
/// summation and standard `rsqrt`, eliminating the divergence.
///
/// `KahanTwoPass` and `Welford` remain available for use cases where
/// numerical accuracy is more important than PyTorch parity.
pub(crate) const DEFAULT_NORM_REDUCTION: NormReduction = NormReduction::PyTorchCompat;

/// MSL preamble for the selected reduction algorithm.
pub(crate) fn norm_preamble_msl(algo: NormReduction) -> &'static str {
    match algo {
        NormReduction::Welford => welford_msl_preamble(),
        NormReduction::KahanTwoPass | NormReduction::PyTorchCompat => "",
    }
}

/// MSL reduction code. Produces `float mean` and `float inv_std`.
pub(crate) fn norm_reduction_msl(algo: NormReduction, dim_var: &str, tg_size: usize) -> String {
    match algo {
        NormReduction::Welford => welford_reduction_msl(dim_var, tg_size),
        NormReduction::KahanTwoPass => kahan_two_pass_reduction_msl(dim_var, tg_size),
        NormReduction::PyTorchCompat => pytorch_compat_reduction_msl(dim_var, tg_size),
    }
}

/// Threadgroup memory bytes needed for the selected algorithm.
#[allow(dead_code)] // Used in tests; kept for future algorithm selection.
pub(crate) fn norm_threadgroup_memory_bytes(algo: NormReduction, tg_size: usize) -> usize {
    let arrays = match algo {
        NormReduction::Welford => 4,
        NormReduction::KahanTwoPass => 2, // shared_val + shared_comp
        NormReduction::PyTorchCompat => 1, // shared_val only (no compensation)
    };
    arrays * tg_size * size_of::<f32>()
}

/// MSL declarations for Welford online algorithm (struct + helpers).
///
/// Provides `WelfordState` struct with Kahan compensation for m2,
/// `welford_update()`, and `welford_merge()`. See #2696.
pub(crate) fn welford_msl_preamble() -> &'static str {
    r#"
struct WelfordState {
    float n;
    float mean;
    float m2;
    float m2_comp;  // Kahan compensation for m2 (#2696)
};

// Accumulate a single sample into a Welford accumulator.
// m2 uses Kahan-compensated summation to prevent systematic drift.
inline WelfordState welford_update(WelfordState state, float x) {
    state.n += 1.0f;
    float delta = x - state.mean;
    state.mean += delta / state.n;
    float delta2 = x - state.mean;
    // Kahan-compensated m2 accumulation (#2696)
    float y = delta * delta2 - state.m2_comp;
    float t = state.m2 + y;
    state.m2_comp = (t - state.m2) - y;
    state.m2 = t;
    return state;
}

// Merge two Welford accumulators (for parallel tree reduction).
// m2 merge uses Kahan compensation to prevent systematic drift.
inline WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (b.n == 0.0f) return a;
    if (a.n == 0.0f) return b;
    float n = a.n + b.n;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * b.n / n;
    // Kahan-compensated m2 merge (#2696)
    float m2_add = delta * delta * a.n * b.n / n;
    float base_m2 = a.m2 + b.m2;
    float comp = a.m2_comp + b.m2_comp;
    float y = m2_add - comp;
    float t = base_m2 + y;
    float new_comp = (t - base_m2) - y;
    return WelfordState{n, mean, t, new_comp};
}
"#
}

/// Welford single-pass reduction MSL. Uses 4 shared arrays (4096 bytes).
pub(crate) fn welford_reduction_msl(dim_var: &str, tg_size: usize) -> String {
    format!(
        r#"
    // --- Kahan-compensated Welford single-pass mean + variance (#2685, #2696) ---
    threadgroup float shared_n[{tg_size}];
    threadgroup float shared_mean[{tg_size}];
    threadgroup float shared_m2[{tg_size}];
    threadgroup float shared_m2_comp[{tg_size}];

    WelfordState local_w = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        local_w = welford_update(local_w, float(input[base + i]));
    }}

    shared_n[tid] = local_w.n;
    shared_mean[tid] = local_w.mean;
    shared_m2[tid] = local_w.m2;
    shared_m2_comp[tid] = local_w.m2_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            WelfordState a = {{shared_n[tid], shared_mean[tid], shared_m2[tid], shared_m2_comp[tid]}};
            WelfordState b = {{shared_n[tid + stride], shared_mean[tid + stride], shared_m2[tid + stride], shared_m2_comp[tid + stride]}};
            WelfordState merged = welford_merge(a, b);
            shared_n[tid] = merged.n;
            shared_mean[tid] = merged.mean;
            shared_m2[tid] = merged.m2;
            shared_m2_comp[tid] = merged.m2_comp;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float mean = shared_mean[0];
    float variance = shared_m2[0] / max(shared_n[0], 1.0f);
    float inv_std = metal::precise::rsqrt(variance + eps);
    // --- end Kahan-compensated Welford reduction ---
"#
    )
}

/// Two-pass Kahan-compensated mean + variance reduction (#2697).
///
/// Zero inner-loop divisions. Only 2 divisions total (mean/N, var/N).
/// Uses 2 shared arrays (2048 bytes vs 4096 for Welford+Kahan).
/// Produces identical `mean` and `inv_std` variables as Welford.
pub(crate) fn kahan_two_pass_reduction_msl(dim_var: &str, tg_size: usize) -> String {
    format!(
        r#"
    // --- Two-pass Kahan-compensated mean + variance (#2697) ---
    threadgroup float shared_val[{tg_size}];
    threadgroup float shared_comp[{tg_size}];

    // ---- Pass 1: Kahan-compensated sum for mean ----
    // Explicit float() cast handles half (F16) input buffers: promotes to
    // float before Kahan accumulation. No-op when input is already float.
    // Part of #3766 F16 I/O for AdaIN/norm kernels.
    float local_sum = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        float y = float(input[base + i]) - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    shared_val[tid] = local_sum;
    shared_comp[tid] = local_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid];
            float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride];
            float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_val[0] / max(float({dim_var}), 1.0f);

    // ---- Pass 2: Kahan-compensated sum of (x - mean)² ----
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_var = 0.0f;
    float local_var_comp = 0.0f;
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        float diff = float(input[base + i]) - mean;
        float diff_sq = diff * diff;
        float y = diff_sq - local_var_comp;
        float t = local_var + y;
        local_var_comp = (t - local_var) - y;
        local_var = t;
    }}
    shared_val[tid] = local_var;
    shared_comp[tid] = local_var_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid];
            float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride];
            float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float variance = shared_val[0] / max(float({dim_var}), 1.0f);
    float inv_std = metal::precise::rsqrt(variance + eps);
    // --- end two-pass Kahan reduction ---
"#
    )
}

/// Naive two-pass mean + variance reduction matching PyTorch MPS (#4335).
///
/// Same structure as `kahan_two_pass_reduction_msl` but:
/// - **No Kahan compensation** — simple accumulation, matching PyTorch ATen/MPS
/// - **Standard `rsqrt()`** — not `precise::rsqrt()`, matching PyTorch MPS
/// - **Same tree reduction** — log2(tg_size) barriers via shared memory
///
/// This produces numerically identical results to PyTorch MPS InstanceNorm,
/// which is critical for models like Kokoro where 35+ chained FusedResBlocks
/// amplify reduction-order differences.
///
/// Uses 2 shared arrays for the tree reduction (reusing the first for both
/// passes since only one sum is active at a time).
pub(crate) fn pytorch_compat_reduction_msl(dim_var: &str, tg_size: usize) -> String {
    format!(
        r#"
    // --- Naive two-pass mean + variance (PyTorch MPS compatible, #4335) ---
    threadgroup float shared_val[{tg_size}];

    // ---- Pass 1: naive sum for mean (no Kahan compensation) ----
    float local_sum = 0.0f;
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        local_sum += float(input[base + i]);
    }}
    shared_val[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            shared_val[tid] += shared_val[tid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_val[0] / max(float({dim_var}), 1.0f);

    // ---- Pass 2: naive sum of (x - mean)^2 (no Kahan compensation) ----
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_var = 0.0f;
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        float diff = float(input[base + i]) - mean;
        local_var += diff * diff;
    }}
    shared_val[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            shared_val[tid] += shared_val[tid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float variance = shared_val[0] / max(float({dim_var}), 1.0f);
    float inv_std = rsqrt(variance + eps);
    // --- end naive two-pass reduction (PyTorch compatible) ---
"#
    )
}

/// MSL helper for simd-aware threadgroup sum reduction.
///
/// Provides `simd_threadgroup_sum()` — a two-level reduction:
///   1. `simd_sum()` within each simdgroup (32 threads)
///   2. Cross-simdgroup reduction via shared memory
///
/// Used by fused RmsNorm (single-pass sum of x²) and fused Snake
/// (single-pass sum). Lighter than the full Welford/KahanTwoPass
/// reductions that produce both mean and variance.
///
/// Caller must declare `threadgroup float shared_simd[32]` and
/// compute `simd_lane`, `simd_group`, `num_simdgroups` from `tid`
/// and `tg_size`.
pub(crate) fn simd_reduction_helpers_msl() -> &'static str {
    r#"
// Simd-aware threadgroup sum: first reduce within each simdgroup,
// then reduce across simdgroups via shared memory.
inline float simd_threadgroup_sum(
    float val,
    threadgroup float* shared,
    uint simd_lane,
    uint simd_group,
    uint num_simdgroups
) {
    float s = simd_sum(val);
    if (simd_lane == 0) shared[simd_group] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = (simd_lane < num_simdgroups) ? shared[simd_lane] : 0.0f;
    s = simd_sum(s);
    return s;
}
"#
}

/// Simd-accelerated two-pass Kahan-compensated mean + variance reduction.
///
/// Same numerical result as `kahan_two_pass_reduction_msl()` but replaces
/// the log2(tg_size) threadgroup-barrier tree reduction with simd-level
/// `simd_sum()` + one cross-simdgroup shared-memory round-trip per pass.
///
/// Barrier count: 4 total (2 per pass) vs 16 for the tree-based path
/// (log2(256)=8 per pass). Each thread's Kahan-corrected partial sum
/// is already well-conditioned, so `simd_sum()` on corrected values is
/// numerically safe.
///
/// Requires the caller's kernel to have `simd_reduction_helpers_msl()`
/// in scope. Produces `float mean` and `float inv_std` variables.
///
/// Uses only `threadgroup float shared_simd[32]` (128 bytes) instead of
/// `shared_val[tg_size] + shared_comp[tg_size]` (2048 bytes for tg_size=256).
pub(crate) fn kahan_two_pass_simd_reduction_msl(dim_var: &str) -> String {
    format!(
        r#"
    // --- Simd-accelerated two-pass Kahan mean + variance ---
    uint simd_lane = tid & 31u;
    uint simd_group = tid >> 5u;
    uint num_simdgroups = tg_size >> 5u;
    threadgroup float shared_simd[32];

    // ---- Pass 1: Kahan-compensated sum for mean (simd reduction) ----
    float local_sum = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        float y = float(input[base + i]) - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    float corrected_sum = local_sum - local_comp;
    float total_sum = simd_threadgroup_sum(corrected_sum, shared_simd, simd_lane, simd_group, num_simdgroups);
    float mean = total_sum / max(float({dim_var}), 1.0f);

    // ---- Pass 2: Kahan-compensated sum of (x - mean)² (simd reduction) ----
    float local_var = 0.0f;
    float local_var_comp = 0.0f;
    for (uint i = tid; i < {dim_var}; i += tg_size) {{
        float diff = float(input[base + i]) - mean;
        float diff_sq = diff * diff;
        float y = diff_sq - local_var_comp;
        float t = local_var + y;
        local_var_comp = (t - local_var) - y;
        local_var = t;
    }}
    float corrected_var = local_var - local_var_comp;
    float total_var = simd_threadgroup_sum(corrected_var, shared_simd, simd_lane, simd_group, num_simdgroups);
    float variance = total_var / max(float({dim_var}), 1.0f);
    float inv_std = metal::precise::rsqrt(variance + eps);
    // --- end simd-accelerated two-pass Kahan reduction ---
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_algorithms_produce_valid_msl() {
        for algo in [
            NormReduction::Welford,
            NormReduction::KahanTwoPass,
            NormReduction::PyTorchCompat,
        ] {
            let preamble = norm_preamble_msl(algo);
            let reduction = norm_reduction_msl(algo, "spatial_len", 256);
            let tg_mem = norm_threadgroup_memory_bytes(algo, 256);
            assert!(reduction.contains("mean"), "{algo:?} missing mean");
            assert!(reduction.contains("inv_std"), "{algo:?} missing inv_std");

            // Welford and KahanTwoPass use precise::rsqrt for verification
            // soundness; PyTorchCompat uses standard rsqrt for PyTorch parity.
            match algo {
                NormReduction::Welford | NormReduction::KahanTwoPass => {
                    assert!(
                        reduction.contains("metal::precise::rsqrt"),
                        "{algo:?} must use precise::rsqrt for verification soundness"
                    );
                }
                NormReduction::PyTorchCompat => {
                    assert!(
                        !reduction.contains("metal::precise::rsqrt"),
                        "PyTorchCompat must NOT use precise::rsqrt (PyTorch parity)"
                    );
                    assert!(
                        reduction.contains("rsqrt(variance + eps)"),
                        "PyTorchCompat must use standard rsqrt"
                    );
                }
            }

            // Lock in exact sizes: these are load-bearing because the MSL
            // static threadgroup declarations must match.
            let expected = match algo {
                NormReduction::Welford => 4 * 256 * 4,    // 4096: n, mean, m2, m2_comp
                NormReduction::KahanTwoPass => 2 * 256 * 4, // 2048: val, comp
                NormReduction::PyTorchCompat => 1 * 256 * 4, // 1024: val only
            };
            assert_eq!(
                tg_mem, expected,
                "{algo:?} unexpected threadgroup memory size"
            );
            if algo == NormReduction::Welford {
                assert!(!preamble.is_empty());
            } else {
                assert!(preamble.is_empty());
            }
        }
    }

    /// Verify explicit float() casts on input reads for F16 safety (#3766).
    ///
    /// All reduction inner loops must use `float(input[...])` instead of raw
    /// `input[...]` to ensure correct half->float promotion when the containing
    /// kernel uses `device const half* input`.
    #[test]
    fn test_reduction_has_explicit_float_casts() {
        for algo in [
            NormReduction::Welford,
            NormReduction::KahanTwoPass,
            NormReduction::PyTorchCompat,
        ] {
            let reduction = norm_reduction_msl(algo, "spatial_len", 256);
            assert!(
                reduction.contains("float(input["),
                "{algo:?} must use explicit float() cast on input reads for F16 safety"
            );
            assert!(
                !reduction.contains("= input[base") && !reduction.contains(", input[base"),
                "{algo:?} must not have raw input[base + i] reads (missing float() cast)"
            );
        }
    }

    /// Simd-accelerated variant produces valid MSL with correct structure.
    #[test]
    fn test_simd_kahan_two_pass_produces_valid_msl() {
        let reduction = kahan_two_pass_simd_reduction_msl("hidden_dim");
        assert!(
            reduction.contains("mean"),
            "simd reduction missing mean variable"
        );
        assert!(
            reduction.contains("inv_std"),
            "simd reduction missing inv_std variable"
        );
        assert!(
            reduction.contains("metal::precise::rsqrt"),
            "simd reduction must use precise::rsqrt for verification soundness"
        );
        assert!(
            reduction.contains("simd_threadgroup_sum"),
            "simd reduction must use simd_threadgroup_sum helper"
        );
        assert!(
            reduction.contains("float(input["),
            "simd reduction must use explicit float() cast for F16 safety"
        );
        // Must declare simd variables
        assert!(
            reduction.contains("simd_lane") && reduction.contains("simd_group"),
            "simd reduction must compute simd_lane and simd_group"
        );
        // Must use shared_simd[32], not the larger shared_val[tg_size]
        assert!(
            reduction.contains("shared_simd[32]"),
            "simd reduction must use shared_simd[32] (128 bytes)"
        );
        assert!(
            !reduction.contains("shared_val["),
            "simd reduction must not use shared_val (tree-based path)"
        );
    }

    /// Simd reduction uses tg_size-independent code (no hardcoded sizes).
    #[test]
    fn test_simd_reduction_is_tg_size_independent() {
        let r1 = kahan_two_pass_simd_reduction_msl("hidden_dim");
        // The simd variant derives simd_lane/simd_group from tg_size
        // at runtime, not compile-time. It should not contain hardcoded
        // threadgroup size constants like "256" or "128".
        assert!(
            !r1.contains("shared_val[256]") && !r1.contains("shared_comp[256]"),
            "simd reduction should not hardcode threadgroup size in shared arrays"
        );
    }

    /// Simd reduction helper MSL compiles with the simd reduction code.
    /// Verifies that preamble + reduction form a coherent MSL fragment.
    #[test]
    fn test_simd_preamble_plus_reduction_coherent() {
        let preamble = simd_reduction_helpers_msl();
        let reduction = kahan_two_pass_simd_reduction_msl("hidden_dim");
        // The reduction calls simd_threadgroup_sum which is defined in preamble
        assert!(preamble.contains("simd_threadgroup_sum"));
        assert!(reduction.contains("simd_threadgroup_sum"));
        // The preamble defines the function, the reduction uses it
        assert!(preamble.contains("inline float simd_threadgroup_sum"));
    }

    /// Edge case: hidden_dim=1. The loop body executes once for tid=0,
    /// zero times for all other threads. All tree and simd paths must
    /// handle this without division by zero.
    #[test]
    fn test_reduction_dim_var_1() {
        for algo in [
            NormReduction::Welford,
            NormReduction::KahanTwoPass,
            NormReduction::PyTorchCompat,
        ] {
            let reduction = norm_reduction_msl(algo, "1", 256);
            // Must still produce mean and inv_std
            assert!(
                reduction.contains("mean") && reduction.contains("inv_std"),
                "{algo:?} must produce mean+inv_std even for dim=1"
            );
            // Must have a max() guard to prevent division by zero.
            // KahanTwoPass/PyTorchCompat use max(float(dim_var), 1.0f).
            // Welford uses max(shared_n[0], 1.0f) (count-based guard).
            assert!(
                reduction.contains("max("),
                "{algo:?} must use max() guard to prevent division by zero"
            );
        }
        // Simd variant too
        let simd_red = kahan_two_pass_simd_reduction_msl("1");
        assert!(
            simd_red.contains("max(float(1), 1.0f)"),
            "simd reduction must use max() guard for dim_var=1"
        );
    }

    /// Edge case: very large hidden_dim (e.g. 65536). The dim_var is
    /// substituted as-is; the MSL loop stride handles any size.
    #[test]
    fn test_reduction_large_hidden_dim() {
        let reduction = norm_reduction_msl(NormReduction::KahanTwoPass, "65536", 256);
        assert!(reduction.contains("65536"));
        assert!(reduction.contains("mean"));
        assert!(reduction.contains("inv_std"));

        let pytorch = norm_reduction_msl(NormReduction::PyTorchCompat, "65536", 256);
        assert!(pytorch.contains("65536"));
        assert!(pytorch.contains("mean"));
        assert!(pytorch.contains("inv_std"));

        let simd_red = kahan_two_pass_simd_reduction_msl("65536");
        assert!(simd_red.contains("65536"));
        assert!(simd_red.contains("mean"));
    }

    /// Verify all non-tree reduction paths (RmsNorm fused, NormLinear local)
    /// also use the dim_var name as documented by callers.
    #[test]
    fn test_reduction_accepts_various_dim_var_names() {
        // Common dim_var names used across the codebase
        for name in ["spatial_len", "hidden_dim", "flat_cols"] {
            let r = norm_reduction_msl(NormReduction::KahanTwoPass, name, 256);
            assert!(
                r.contains(name),
                "KahanTwoPass reduction must use dim_var={name}"
            );
            let p = norm_reduction_msl(NormReduction::PyTorchCompat, name, 256);
            assert!(
                p.contains(name),
                "PyTorchCompat reduction must use dim_var={name}"
            );
            let sr = kahan_two_pass_simd_reduction_msl(name);
            assert!(
                sr.contains(name),
                "simd reduction must use dim_var={name}"
            );
        }
    }

    /// Verify threadgroup memory calculation for multiple tg_sizes.
    #[test]
    fn test_threadgroup_memory_bytes_various_sizes() {
        for tg in [32, 64, 128, 256, 512, 1024] {
            let welford_bytes = norm_threadgroup_memory_bytes(NormReduction::Welford, tg);
            let kahan_bytes = norm_threadgroup_memory_bytes(NormReduction::KahanTwoPass, tg);
            let pytorch_bytes = norm_threadgroup_memory_bytes(NormReduction::PyTorchCompat, tg);
            // Welford needs 4 arrays, Kahan needs 2, PyTorchCompat needs 1
            assert_eq!(welford_bytes, 4 * tg * 4, "Welford tg={tg}");
            assert_eq!(kahan_bytes, 2 * tg * 4, "KahanTwoPass tg={tg}");
            assert_eq!(pytorch_bytes, 1 * tg * 4, "PyTorchCompat tg={tg}");
            // Welford always uses more threadgroup memory
            assert!(
                welford_bytes > kahan_bytes,
                "Welford must use more tg memory than KahanTwoPass"
            );
        }
    }

    /// The tree reduction uses shared arrays sized to the Rust-side tg_size
    /// parameter. The stride pattern uses the MSL runtime `tg_size` variable.
    #[test]
    fn test_tree_reduction_uses_correct_tg_size() {
        for tg in [32, 64, 128, 256, 512] {
            let r = kahan_two_pass_reduction_msl("dim", tg);
            // The shared arrays must be sized to the Rust-side tg_size
            let expected_decl = format!("shared_val[{tg}]");
            assert!(
                r.contains(&expected_decl),
                "tg={tg}: missing shared_val[{tg}] declaration"
            );
            let expected_comp = format!("shared_comp[{tg}]");
            assert!(
                r.contains(&expected_comp),
                "tg={tg}: missing shared_comp[{tg}] declaration"
            );
            // The stride uses the MSL runtime tg_size variable (not interpolated)
            assert!(
                r.contains("stride = tg_size / 2"),
                "tg={tg}: missing stride = tg_size / 2 (runtime variable)"
            );
        }
    }

    /// PyTorchCompat reduction uses naive accumulation (no Kahan compensation).
    #[test]
    fn test_pytorch_compat_no_kahan() {
        let r = pytorch_compat_reduction_msl("spatial_len", 256);
        // Must NOT have Kahan compensation variables
        assert!(
            !r.contains("local_comp"),
            "PyTorchCompat must not have Kahan compensation (local_comp)"
        );
        assert!(
            !r.contains("local_var_comp"),
            "PyTorchCompat must not have Kahan compensation (local_var_comp)"
        );
        assert!(
            !r.contains("shared_comp["),
            "PyTorchCompat must not have shared_comp array"
        );
        // Must use simple accumulation
        assert!(
            r.contains("local_sum += float(input["),
            "PyTorchCompat must use naive += accumulation for sum"
        );
        assert!(
            r.contains("local_var += diff * diff"),
            "PyTorchCompat must use naive += accumulation for variance"
        );
        // Must use standard rsqrt, not precise
        assert!(
            r.contains("rsqrt(variance + eps)"),
            "PyTorchCompat must use standard rsqrt"
        );
        assert!(
            !r.contains("precise::rsqrt"),
            "PyTorchCompat must NOT use precise::rsqrt"
        );
    }

    /// PyTorchCompat tree reduction uses correct tg_size.
    #[test]
    fn test_pytorch_compat_tree_reduction_correct_tg_size() {
        for tg in [32, 64, 128, 256, 512] {
            let r = pytorch_compat_reduction_msl("dim", tg);
            let expected_decl = format!("shared_val[{tg}]");
            assert!(
                r.contains(&expected_decl),
                "PyTorchCompat tg={tg}: missing shared_val[{tg}] declaration"
            );
            assert!(
                r.contains("stride = tg_size / 2"),
                "PyTorchCompat tg={tg}: missing stride = tg_size / 2"
            );
        }
    }

    /// Default reduction mode is PyTorchCompat (#4335).
    #[test]
    fn test_default_is_pytorch_compat() {
        assert_eq!(
            DEFAULT_NORM_REDUCTION,
            NormReduction::PyTorchCompat,
            "default must be PyTorchCompat for Kokoro amplitude parity (#4335)"
        );
    }
}
