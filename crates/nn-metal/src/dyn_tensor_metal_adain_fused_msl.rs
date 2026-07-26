// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused AdaIN+Snake and AdaIN+LeakyRelu kernels (#2472).
//!
//! Uses two-pass Kahan-compensated mean+variance reduction (#2697).
//! Default algorithm selected via `DEFAULT_NORM_REDUCTION`.
//!
//! Input x is pre-reshaped to `[B*C, spatial]`. gamma/beta are `[B*C]`
//! (flattened from `[B, C, 1]`). alpha is `[C]` (per-channel, not per-batch).

use super::super::welford_msl;

/// MSL source for the fused AdaIN+Snake f32 kernel.
///
/// Buffers:
///   - 0: `input` — `[rows, spatial_len]` f32 (read-only)
///   - 1: `gamma` — `[rows]` f32 (read-only, per batch×channel)
///   - 2: `beta` — `[rows]` f32 (read-only, per batch×channel)
///   - 3: `alpha` — `[channels]` f32 (read-only, per-channel)
///   - 4: `output` — `[rows, spatial_len]` f32 (write-only)
///
/// Constants (set_bytes):
///   - 5: `spatial_len` — uint
///   - 6: `channels` — uint
///   - 7: `eps` — float
///
/// Dispatch: one threadgroup per row (B*C rows), 256 threads per threadgroup.
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision. Part of #2981 F16 Tier 2.
pub(super) fn fused_adain_snake_msl(scalar_type: &str, residual_gamma: bool) -> String {
    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);
    let reduction = welford_msl::norm_reduction_msl(algo, "spatial_len", 256);

    // Part of #3257: conditional affine formula based on residual_gamma.
    let affine_expr = if residual_gamma {
        "(1.0f + g) * normed + b"
    } else {
        "g * normed + b"
    };
    let affine_comment = if residual_gamma {
        "// y = (1 + g) * normed + b  (residual gamma)"
    } else {
        "// y = g * normed + b  (standard AdaIN)"
    };

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_adain_snake_{scalar_type}(
    device const {scalar_type}* input  [[buffer(0)]],
    device const {scalar_type}* gamma  [[buffer(1)]],
    device const {scalar_type}* beta   [[buffer(2)]],
    device const {scalar_type}* alpha  [[buffer(3)]],
    device {scalar_type}* output       [[buffer(4)]],
    constant uint& spatial_len [[buffer(5)]],
    constant uint& channels    [[buffer(6)]],
    constant float& eps        [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * spatial_len;

{reduction}

    // Load per-channel/per-batch params.
    float g = float(gamma[gid]);            // [B*C] indexed by row
    float b = float(beta[gid]);             // [B*C] indexed by row
    float a = max(float(alpha[gid % channels]), 1e-8f); // [C] indexed by channel, clamped (#2648)
    float inv_a = 1.0f / a;

    // InstanceNorm + Affine + Snake.
    // normed = (x - mean) * inv_std
    {affine_comment}
    // out = y + (1/a) * sin²(a * y)
    for (uint i = tid; i < spatial_len; i += tg_size) {{
        float normed = (float(input[base + i]) - mean) * inv_std;
        float y = {affine_expr};
        float sin_val = sin(a * y);
        output[base + i] = {scalar_type}(y + inv_a * sin_val * sin_val);
    }}
}}
"#
    )
}

/// MSL source for the fused AdaIN+LeakyRelu kernel.
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision. Part of #2981 F16 Tier 2.
pub(super) fn fused_adain_leaky_relu_msl(scalar_type: &str) -> String {
    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);
    let reduction = welford_msl::norm_reduction_msl(algo, "spatial_len", 256);

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_adain_leaky_relu_{scalar_type}(
    device const {scalar_type}* input  [[buffer(0)]],
    device const {scalar_type}* gamma  [[buffer(1)]],
    device const {scalar_type}* beta   [[buffer(2)]],
    device {scalar_type}* output       [[buffer(3)]],
    constant uint& spatial_len [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    constant float& slope      [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * spatial_len;

{reduction}

    // Load per-batch×channel params.
    float g = float(gamma[gid]);
    float b = float(beta[gid]);

    // InstanceNorm + Affine + LeakyRelu.
    // normed = (x - mean) * inv_std
    // y = (1 + g) * normed + b
    // out = y >= 0 ? y : slope * y
    for (uint i = tid; i < spatial_len; i += tg_size) {{
        float normed = (float(input[base + i]) - mean) * inv_std;
        float y = (1.0f + g) * normed + b;
        output[base + i] = {scalar_type}(y >= 0.0f ? y : slope * y);
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify F16 AdaIN+Snake MSL uses half I/O with float accumulators (#3782).
    ///
    /// When scalar_type is "half", the MSL must:
    /// - Declare `device const half* input` and `device half* output`
    /// - Use `float()` casts on all input reads (float accumulators)
    /// - Cast output back: `half(...)`
    #[test]
    fn test_adain_snake_f16_msl_has_half_io() {
        let msl = fused_adain_snake_msl("half", false);

        // I/O pointers are half
        assert!(
            msl.contains("device const half* input"),
            "F16 AdaIN+Snake MSL must declare `device const half* input`"
        );
        assert!(
            msl.contains("device half* output"),
            "F16 AdaIN+Snake MSL must declare `device half* output`"
        );
        // gamma/beta/alpha buffers are also half
        assert!(
            msl.contains("device const half* gamma"),
            "F16 AdaIN+Snake MSL must declare `device const half* gamma`"
        );
        assert!(
            msl.contains("device const half* beta"),
            "F16 AdaIN+Snake MSL must declare `device const half* beta`"
        );
        assert!(
            msl.contains("device const half* alpha"),
            "F16 AdaIN+Snake MSL must declare `device const half* alpha`"
        );
        // Accumulators use float() promotion
        assert!(
            msl.contains("float(input["),
            "F16 AdaIN+Snake MSL must use float() cast on input reads"
        );
        assert!(
            msl.contains("float(gamma["),
            "F16 AdaIN+Snake MSL must use float() cast on gamma reads"
        );
        assert!(
            msl.contains("float(alpha["),
            "F16 AdaIN+Snake MSL must use float() cast on alpha reads"
        );
        // Output cast back to half
        assert!(
            msl.contains("half("),
            "F16 AdaIN+Snake MSL must cast output back to half"
        );
        // Kernel function name encodes dtype
        assert!(
            msl.contains("fused_adain_snake_half"),
            "F16 AdaIN+Snake kernel function name must include 'half'"
        );
    }

    /// Verify F16 AdaIN+Snake MSL with residual_gamma variant (#3782, #3257).
    #[test]
    fn test_adain_snake_f16_msl_residual_gamma() {
        let msl = fused_adain_snake_msl("half", true);

        assert!(
            msl.contains("device const half* input"),
            "residual_gamma F16 variant must use half I/O"
        );
        assert!(
            msl.contains("(1.0f + g) * normed + b"),
            "residual_gamma variant must use (1+g)*normed+b formula"
        );
        assert!(
            msl.contains("half("),
            "residual_gamma F16 variant must cast output to half"
        );
    }

    /// Verify F32 AdaIN+Snake MSL uses float I/O (sanity check).
    #[test]
    fn test_adain_snake_f32_msl_has_float_io() {
        let msl = fused_adain_snake_msl("float", false);

        assert!(
            msl.contains("device const float* input"),
            "F32 AdaIN+Snake MSL must declare float input"
        );
        assert!(
            msl.contains("device float* output"),
            "F32 AdaIN+Snake MSL must declare float output"
        );
        assert!(
            msl.contains("fused_adain_snake_float"),
            "F32 kernel function name must include 'float'"
        );
        // float(input[...]) is a no-op for float but still present for uniformity
        assert!(
            msl.contains("float(input["),
            "F32 MSL still uses float() cast for uniformity"
        );
    }

    /// Verify F16 AdaIN+LeakyRelu MSL uses half I/O with float accumulators (#3782).
    #[test]
    fn test_adain_leaky_relu_f16_msl_has_half_io() {
        let msl = fused_adain_leaky_relu_msl("half");

        // I/O pointers are half
        assert!(
            msl.contains("device const half* input"),
            "F16 AdaIN+LeakyRelu MSL must declare `device const half* input`"
        );
        assert!(
            msl.contains("device half* output"),
            "F16 AdaIN+LeakyRelu MSL must declare `device half* output`"
        );
        // gamma/beta buffers are also half
        assert!(
            msl.contains("device const half* gamma"),
            "F16 AdaIN+LeakyRelu MSL must declare `device const half* gamma`"
        );
        assert!(
            msl.contains("device const half* beta"),
            "F16 AdaIN+LeakyRelu MSL must declare `device const half* beta`"
        );
        // Accumulators use float() promotion
        assert!(
            msl.contains("float(input["),
            "F16 AdaIN+LeakyRelu MSL must use float() cast on input reads"
        );
        assert!(
            msl.contains("float(gamma["),
            "F16 AdaIN+LeakyRelu MSL must use float() cast on gamma reads"
        );
        // Output cast back to half
        assert!(
            msl.contains("half("),
            "F16 AdaIN+LeakyRelu MSL must cast output back to half"
        );
        // Kernel function name encodes dtype
        assert!(
            msl.contains("fused_adain_leaky_relu_half"),
            "F16 AdaIN+LeakyRelu kernel function name must include 'half'"
        );
    }

    /// Verify F32 AdaIN+LeakyRelu MSL uses float I/O (sanity check).
    #[test]
    fn test_adain_leaky_relu_f32_msl_has_float_io() {
        let msl = fused_adain_leaky_relu_msl("float");

        assert!(
            msl.contains("device const float* input"),
            "F32 AdaIN+LeakyRelu MSL must declare float input"
        );
        assert!(
            msl.contains("device float* output"),
            "F32 AdaIN+LeakyRelu MSL must declare float output"
        );
        assert!(
            msl.contains("fused_adain_leaky_relu_float"),
            "F32 kernel function name must include 'float'"
        );
    }

    /// Verify no raw (uncast) input reads in F16 mode (#3766 invariant).
    ///
    /// Every `input[...]` read must be wrapped in `float(...)` for correct
    /// half->float promotion. Raw reads would lose precision in accumulators.
    #[test]
    fn test_f16_no_raw_input_reads() {
        for (label, msl) in [
            ("AdaIN+Snake", fused_adain_snake_msl("half", false)),
            (
                "AdaIN+Snake (residual)",
                fused_adain_snake_msl("half", true),
            ),
            ("AdaIN+LeakyRelu", fused_adain_leaky_relu_msl("half")),
        ] {
            // Every occurrence of `input[` must be preceded by `float(`
            // (no bare `= input[` or `, input[` patterns).
            assert!(
                !msl.contains("= input[base") && !msl.contains(", input[base"),
                "{label}: must not have raw input[base + i] reads (missing float() cast)"
            );
        }
    }
}
