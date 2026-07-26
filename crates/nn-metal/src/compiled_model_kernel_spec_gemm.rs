// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! GEMM-based `spec_*()` builders: LinearActivation, NormLinear.
//!
//! These builders generate MSL source at build time (via nn_dsl codegen)
//! and produce [`KernelSpec`]s for single-dispatch fused GEMM kernels.
//!
//! Part of #3503 D3 (KernelSpec unification).

use std::mem::size_of;

use nn_dsl::trace_compile::FusedNormKind;
use nn_dsl::GemmActivation;

use super::{KernelBinding, KernelSpec, SpecDispatchMode};
use super::norm::NORM_TG_SIZE;

// -------------------------------------------------------------------------
// LinearActivation
// -------------------------------------------------------------------------

/// Build a [`KernelSpec`] for fused Linear + Activation.
///
/// Routes to simdgroup-tiled GEMM when dimensions conform (all % 8,
/// M*N >= 16384, K >= 128), otherwise falls back to naive per-element kernel.
///
/// Naive buffer layout:
///   0: input `[..batch, in_features]` (Edge 0)
///   1: weight `[out_features, in_features]` (Weight "weight")
///   2: bias `[out_features]` (Weight "bias", if has_bias)
///   3/2: output (Output)
///
/// Simdgroup buffer layout:
///   0: input (Edge 0)
///   1: weight (Weight "weight")
///   2: bias (Weight "bias", if has_bias)
///   3/2: output (Output)
///
/// Part of #3503 D3.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spec_linear_activation(
    scalar_type: nn_dsl::ir::ScalarType,
    activation: &GemmActivation,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    input_shape: &[usize],
) -> Result<KernelSpec, String> {
    let batch_size: usize = input_shape.iter().rev().skip(1).product();
    let total_output = batch_size.checked_mul(out_features).ok_or_else(|| {
        format!("spec_linear_activation: output overflow ({batch_size} * {out_features})")
    })?;

    let elem_bytes = scalar_type.byte_size();
    let scalar_str = scalar_type.msl_str();

    let use_simdgroup =
        crate::dyn_tensor_metal::should_use_simdgroup(batch_size, in_features, out_features);

    let act_tag = activation_tag(activation);
    let kernel_name = if use_simdgroup {
        format!(
            "simd_la_{}_m{batch_size}_k{in_features}_n{out_features}_{act_tag}_b{}",
            scalar_str,
            u8::from(has_bias),
        )
    } else {
        format!(
            "la_{}_k{in_features}_n{out_features}_{act_tag}_b{}",
            scalar_str,
            u8::from(has_bias),
        )
    };

    let msl_source = if use_simdgroup {
        nn_dsl::emit_simdgroup_linear_activation_kernel(
            &kernel_name,
            scalar_type,
            in_features,
            out_features,
            batch_size,
            has_bias,
            activation,
        )
    } else {
        nn_dsl::emit_linear_activation_kernel(
            &kernel_name,
            scalar_type,
            in_features,
            out_features,
            has_bias,
            Some(activation),
            true, // include MSL prelude
        )
    }
    .map_err(|e| format!("spec_linear_activation: MSL codegen: {e}"))?;

    let param_count = if has_bias { 3 } else { 2 };

    let output_bytes = total_output.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_linear_activation: output bytes overflow ({total_output} * {elem_bytes})")
    })?;

    let (grid, threadgroup, tg_mem_bytes, dispatch_mode) = if use_simdgroup {
        let m_u32 = u32::try_from(batch_size)
            .map_err(|_| format!("spec_linear_activation: batch_size {batch_size} exceeds u32"))?;
        let n_u32 = u32::try_from(out_features)
            .map_err(|_| format!("spec_linear_activation: out_features {out_features} exceeds u32"))?;
        let is_half = elem_bytes == 2;
        // Threadgroup memory: As + Bs (element-sized) + tile_out (float).
        let tg_bytes: u64 = if is_half {
            2 * 32 * 33 * 2 + 32 * 33 * 4
        } else {
            3 * 32 * 33 * 4
        };
        (
            [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
            [32u32, 4, 1],
            tg_bytes,
            SpecDispatchMode::Threadgroups,
        )
    } else {
        let total_u32 = u32::try_from(total_output)
            .map_err(|_| format!("spec_linear_activation: total {total_output} exceeds u32"))?;
        let tg_size = 256u32;
        let num_tg = total_u32.div_ceil(tg_size);
        (
            [num_tg, 1, 1],
            [tg_size, 1, 1],
            0u64,
            SpecDispatchMode::Threads,
        )
    };

    let mut bindings = vec![
        (0, KernelBinding::Edge(0)),
        (1, KernelBinding::Weight("weight".into())),
    ];
    if has_bias {
        bindings.push((2, KernelBinding::Weight("bias".into())));
        bindings.push((3, KernelBinding::Output));
    } else {
        bindings.push((2, KernelBinding::Output));
    }

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid,
        threadgroup,
        dispatch_mode,
        threadgroup_memory_bytes: tg_mem_bytes,
        output_bytes,
        bindings,
        param_count,
        fast_math: false,
    })
}

// -------------------------------------------------------------------------
// NormLinear (scalar fallback path only — simdgroup path is 2-dispatch)
// -------------------------------------------------------------------------

/// Build a [`KernelSpec`] for fused Norm + Linear (scalar dot-product GEMM).
///
/// Single Metal dispatch: normalize → threadgroup memory → GEMM.
/// One threadgroup per input row. Dynamic threadgroup memory for
/// normalized values (`hidden_dim * sizeof(float)` bytes).
///
/// Routes to the scalar (single-dispatch) fallback when simdgroup dimensions
/// do NOT qualify. When they do qualify, the caller should use a 2-dispatch
/// MultiKernelSpec (norm-only → simdgroup GEMM) instead — that path is not
/// yet covered by KernelSpec.
///
/// Buffer layout depends on norm_kind and has_bias:
///
/// LayerNorm + bias:
///   0: input (Edge 0)
///   1: norm_weight (Weight "norm_weight")
///   2: norm_bias (Weight "norm_bias")
///   3: weight (Weight "weight")
///   4: bias (Weight "bias")
///   5: output (Output)
///   6: hidden_dim (Constant u32)
///   7: eps (Constant f32)
///   8: out_features (Constant u32)
///   9: flat_rows (Constant u32)
///
/// RmsNorm + bias:
///   0: input (Edge 0)
///   1: norm_weight (Weight "norm_weight")
///   2: weight (Weight "weight")
///   3: bias (Weight "bias")
///   4: output (Output)
///   5: hidden_dim (Constant u32)
///   6: eps (Constant f32)
///   7: out_features (Constant u32)
///   8: flat_rows (Constant u32)
///
/// Part of #3503 D3.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spec_norm_linear(
    scalar_type: nn_dsl::ir::ScalarType,
    norm_kind: FusedNormKind,
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<KernelSpec, String> {
    let flat_rows: usize = input_shape.iter().rev().skip(1).product();
    let total_output = flat_rows.checked_mul(out_features).ok_or_else(|| {
        format!("spec_norm_linear: output overflow ({flat_rows} * {out_features})")
    })?;

    if flat_rows == 0 || hidden_dim == 0 || out_features == 0 {
        return Err("spec_norm_linear: zero-size dimension".into());
    }

    // This builder only covers the scalar fallback path.
    // If simdgroup would be used, the caller should use the 2-dispatch path.
    if crate::dyn_tensor_metal::should_use_simdgroup(flat_rows, hidden_dim, out_features) {
        return Err(
            "spec_norm_linear: simdgroup path requires 2-dispatch MultiKernelSpec, \
             not covered by this single-dispatch builder"
                .into(),
        );
    }

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let norm_tag = match norm_kind {
        FusedNormKind::LayerNorm => "ln",
        FusedNormKind::RmsNorm => "rms",
        _ => {
            return Err(format!(
                "spec_norm_linear: unsupported norm kind {norm_kind:?}"
            ))
        }
    };
    let kernel_name = format!(
        "fused_norm_linear_{norm_tag}_{scalar_str}_b{}",
        u8::from(has_bias)
    );

    // Generate MSL source using the same generator as the executor.
    // We need to replicate the MSL generation here since it's a private fn
    // in the executor module. The MSL is self-contained and parametric.
    let msl_source = norm_linear_msl(scalar_str, norm_kind, has_bias)?;

    let has_norm_b = norm_kind == FusedNormKind::LayerNorm;
    let input_buf_count = match (has_norm_b, has_bias) {
        (true, true) => 5,
        (true, false) => 4,
        (false, true) => 4,
        (false, false) => 3,
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_norm_linear: flat_rows {flat_rows} exceeds u32"))?;
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| format!("spec_norm_linear: hidden_dim {hidden_dim} exceeds u32"))?;
    let out_features_u32 = u32::try_from(out_features)
        .map_err(|_| format!("spec_norm_linear: out_features {out_features} exceeds u32"))?;

    let tg_mem_bytes = hidden_dim
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| "spec_norm_linear: tg_mem bytes overflow".to_string())?
        as u64;

    let output_bytes = total_output.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_norm_linear: output bytes overflow ({total_output} * {elem_bytes})")
    })?;

    // Build bindings with dynamic buffer indices.
    let mut bindings = Vec::new();
    let mut idx: usize = 0;
    bindings.push((idx, KernelBinding::Edge(0)));
    idx += 1;
    bindings.push((idx, KernelBinding::Weight("norm_weight".into())));
    idx += 1;
    if has_norm_b {
        bindings.push((idx, KernelBinding::Weight("norm_bias".into())));
        idx += 1;
    }
    bindings.push((idx, KernelBinding::Weight("weight".into())));
    idx += 1;
    if has_bias {
        bindings.push((idx, KernelBinding::Weight("bias".into())));
        idx += 1;
    }
    bindings.push((idx, KernelBinding::Output));
    idx += 1;
    bindings.push((idx, KernelBinding::constant_u32(hidden_dim_u32)));
    idx += 1;
    bindings.push((idx, KernelBinding::constant_f32(eps)));
    idx += 1;
    bindings.push((idx, KernelBinding::constant_u32(out_features_u32)));
    idx += 1;
    bindings.push((idx, KernelBinding::constant_u32(flat_rows_u32)));

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [flat_rows_u32, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: tg_mem_bytes,
        output_bytes,
        bindings,
        param_count: input_buf_count,
        fast_math: false,
    })
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Short tag for [`GemmActivation`] in kernel names.
fn activation_tag(act: &GemmActivation) -> &'static str {
    match act {
        GemmActivation::Relu => "relu",
        GemmActivation::Gelu => "gelu",
        GemmActivation::GeluErf => "geluerf",
        GemmActivation::Sigmoid => "sig",
        GemmActivation::Silu => "silu",
        GemmActivation::Tanh => "tanh",
        _ => "unsupported_activation",
    }
}

/// Generate MSL source for the fused NormLinear kernel.
///
/// Replicates the MSL generation from `compiled_model_execute_native_norm_linear.rs`
/// so that KernelSpec builders can produce self-contained specs. The generated MSL
/// is identical to what the executor produces.
fn norm_linear_msl(
    scalar_type: &str,
    norm_kind: FusedNormKind,
    has_bias: bool,
) -> Result<String, String> {
    let tg = NORM_TG_SIZE as usize;

    // Cast helpers for half-precision I/O with float accumulators.
    let load = |var: &str| -> String {
        if scalar_type == "float" {
            var.to_string()
        } else {
            format!("float({var})")
        }
    };
    let store = |var: &str| -> String {
        if scalar_type == "float" {
            var.to_string()
        } else {
            format!("{scalar_type}({var})")
        }
    };

    let load_input = load("input[base + i]");
    let load_nw = load("norm_w[i]");
    let load_weight = load("weight[col * hidden_dim + k]");
    let store_out = store("dot");

    let (norm_params, phase1, phase2) = match norm_kind {
        FusedNormKind::LayerNorm => {
            let algo = crate::dyn_tensor_metal::welford_msl::DEFAULT_NORM_REDUCTION;
            let preamble = crate::dyn_tensor_metal::welford_msl::norm_preamble_msl(algo);
            let reduction =
                crate::dyn_tensor_metal::welford_msl::norm_reduction_msl(algo, "hidden_dim", tg);
            let load_nb = load("norm_b[i]");
            let norm_b_param =
                format!("    device const {scalar_type}* norm_b   [[buffer(2)]],\n");
            let p2 = format!(
                r#"    // Phase 2: normalize + affine → threadgroup memory
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = ({load_input} - mean) * inv_std;
        normed[i] = val * {load_nw} + {load_nb};
    }}"#
            );
            ((preamble.to_string(), norm_b_param), reduction, p2)
        }
        FusedNormKind::RmsNorm => {
            let rms_reduction = rms_reduction_msl(tg);
            let p2 = format!(
                r#"    // Phase 2: scale by inv_rms * weight → threadgroup memory
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        normed[i] = {load_input} * inv_rms * {load_nw};
    }}"#
            );
            ((String::new(), String::new()), rms_reduction, p2)
        }
        _ => {
            return Err(format!(
                "norm_linear_msl: unsupported norm kind {norm_kind:?}"
            ))
        }
    };

    let (preamble, norm_b_param) = norm_params;

    let has_norm_b = norm_kind == FusedNormKind::LayerNorm;
    let weight_idx = if has_norm_b { 3 } else { 2 };
    let (bias_param, bias_add, out_idx) = if has_bias {
        let bi = weight_idx + 1;
        let oi = bi + 1;
        let bp = format!("    device const {scalar_type}* bias      [[buffer({bi})]],\n");
        (bp, "        dot += float(bias[col]);\n".to_string(), oi)
    } else {
        (String::new(), String::new(), weight_idx + 1)
    };

    let hd_idx = out_idx + 1;
    let eps_idx = hd_idx + 1;
    let of_idx = eps_idx + 1;
    let fr_idx = of_idx + 1;

    let norm_tag = match norm_kind {
        FusedNormKind::LayerNorm => "ln",
        FusedNormKind::RmsNorm => "rms",
        _ => {
            return Err(format!(
                "norm_linear_msl tag: unsupported norm kind {norm_kind:?}"
            ))
        }
    };

    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_norm_linear_{norm_tag}_{scalar_type}_b{has_bias_u}(
    device const {scalar_type}* input    [[buffer(0)]],
    device const {scalar_type}* norm_w   [[buffer(1)]],
{norm_b_param}    device const {scalar_type}* weight   [[buffer({weight_idx})]],
{bias_param}    device {scalar_type}* output         [[buffer({out_idx})]],
    constant uint& hidden_dim            [[buffer({hd_idx})]],
    constant float& eps                  [[buffer({eps_idx})]],
    constant uint& out_features          [[buffer({of_idx})]],
    constant uint& flat_rows             [[buffer({fr_idx})]],
    threadgroup float* normed            [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    if (gid >= flat_rows) return;
    uint base = gid * hidden_dim;

{phase1}

{phase2}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: GEMM from threadgroup memory
    uint out_base = gid * out_features;
    for (uint col = tid; col < out_features; col += tg_size) {{
        float dot = 0;
        for (uint k = 0; k < hidden_dim; k++) {{
            dot += normed[k] * {load_weight};
        }}
{bias_add}        output[out_base + col] = {store_out};
    }}
}}"#,
        has_bias_u = u8::from(has_bias),
    ))
}

// -------------------------------------------------------------------------
// Int8Gemm (W8A16 dequantizing GEMM)
// -------------------------------------------------------------------------

/// Build a [`KernelSpec`] for INT8 W8A16 dequantizing matmul.
///
/// Uses the 32x32 simdgroup-tiled GEMM with on-the-fly INT8→half dequantization
/// in the weight tile load phase. F32 activations are demoted to half on load;
/// accumulation is F32; output is F32.
///
/// Buffer layout:
///   0: input `[..batch, in_features]` (Edge 0, F32)
///   1: weight_int8 `[out_features, in_features]` (Weight "weight_int8", U8)
///   2: scale `[out_features]` (Weight "scale", F32)
///   3: zero_point `[out_features]` (Weight "zero_point", I32)
///   4: bias `[out_features]` (Weight "bias", F32, if has_bias)
///   4/5: output (Output, F32)
///
/// Part of #3522.
pub(crate) fn spec_int8_matmul(
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    input_shape: &[usize],
) -> Result<KernelSpec, String> {
    let batch_size: usize = input_shape.iter().rev().skip(1).product();
    if batch_size == 0 || in_features == 0 || out_features == 0 {
        return Err("spec_int8_matmul: zero-size dimension".into());
    }

    let total_output = batch_size.checked_mul(out_features).ok_or_else(|| {
        format!("spec_int8_matmul: output overflow ({batch_size} * {out_features})")
    })?;

    let info = crate::compiled_model::int8_gemm_msl::Int8GemmInfo {
        m: batch_size,
        k: in_features,
        n: out_features,
        has_bias,
    };

    let msl_source = crate::compiled_model::int8_gemm_msl::generate_int8_gemm_msl(&info);
    let param_count = crate::compiled_model::int8_gemm_msl::int8_gemm_input_count(has_bias);

    let m_u32 = u32::try_from(batch_size)
        .map_err(|_| format!("spec_int8_matmul: batch_size {batch_size} exceeds u32"))?;
    let n_u32 = u32::try_from(out_features)
        .map_err(|_| format!("spec_int8_matmul: out_features {out_features} exceeds u32"))?;

    let tg_mem_bytes = crate::compiled_model::int8_gemm_msl::int8_gemm_threadgroup_bytes();

    let output_bytes = total_output
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| {
            format!("spec_int8_matmul: output bytes overflow ({total_output} * 4)")
        })?;

    let mut bindings = vec![
        (0, KernelBinding::Edge(0)),
        (1, KernelBinding::Weight("weight_int8".into())),
        (2, KernelBinding::Weight("scale".into())),
        (3, KernelBinding::Weight("zero_point".into())),
    ];
    if has_bias {
        bindings.push((4, KernelBinding::Weight("bias".into())));
        bindings.push((5, KernelBinding::Output));
    } else {
        bindings.push((4, KernelBinding::Output));
    }

    Ok(KernelSpec {
        // Fixed MSL function name — PipelineCache differentiates by MSL source hash
        // (different M/K/N constants produce different source).
        kernel_name: "int8_matmul_dequant".to_string(),
        msl_source,
        grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
        threadgroup: [32u32, 4, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: tg_mem_bytes,
        output_bytes,
        bindings,
        param_count,
        fast_math: false,
    })
}

/// RmsNorm reduction MSL: Kahan-compensated sum of x², then rsqrt.
fn rms_reduction_msl(tg_size: usize) -> String {
    format!(
        r#"
    // --- RmsNorm: Kahan-compensated sum of x² ---
    threadgroup float shared_val[{tg_size}];
    threadgroup float shared_comp[{tg_size}];

    float local_sum = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float v = float(input[base + i]);
        float sq = v * v;
        float y = sq - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    shared_val[tid] = local_sum;
    shared_comp[tid] = local_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {tg_size} / 2; stride > 0; stride >>= 1) {{
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
    float mean_sq = shared_val[0] / max(float(hidden_dim), 1.0f);
    float inv_rms = metal::precise::rsqrt(mean_sq + eps);"#
    )
}
