// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused NormLinear executor for `CompiledModel`.
//!
//! Executes `NativeOpKind::NormLinear` (LayerNorm/RmsNorm + Linear) in a
//! single Metal dispatch. Uses threadgroup memory to hold normalized values
//! between the reduction phase and GEMM phase.
//!
//! Part of #3089 (Norm+Linear peephole fusion).

use std::mem::size_of;

use nn_core::Result;
use nn_dsl::trace_compile::FusedNormKind;

use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::welford_msl;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::native_dispatch_err;
use super::CompiledModel;

#[path = "compiled_model_execute_native_norm_linear_simd.rs"]
mod simd;

/// Threadgroup size (matches fused LayerNorm kernel).
const TG_SIZE: u32 = 256;

/// Execute a `NativeOpKind::NormLinear` step.
///
/// Generates a fused Metal kernel that:
/// 1. Computes normalization statistics (mean+var for LayerNorm, RMS for RmsNorm)
/// 2. Normalizes (and applies affine for LayerNorm) into threadgroup memory
/// 3. Performs GEMM (matrix-vector multiply) from threadgroup memory
///
/// One threadgroup per input row, dynamic threadgroup memory for
/// normalized values (`hidden_dim * sizeof(float)` bytes).
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_norm_linear(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    norm_kind: FusedNormKind,
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
    out_features: usize,
    has_bias: bool,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    // Resolve graph input.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve weight buffers.
    let step_weights = &model.def.weight_buffers[step_idx];
    let norm_w = step_weights
        .get("norm_weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "NormLinear: missing 'norm_weight'".into()))?;

    let norm_b = match norm_kind {
        FusedNormKind::LayerNorm => Some(step_weights.get("norm_bias").ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                "NormLinear: missing 'norm_bias' for LayerNorm".into(),
            )
        })?),
        FusedNormKind::RmsNorm => None,
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };

    let weight = step_weights
        .get("weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "NormLinear: missing 'weight'".into()))?;
    let bias =
        if has_bias {
            Some(step_weights.get("bias").ok_or_else(|| {
                native_dispatch_err(step_idx, "NormLinear: missing 'bias'".into())
            })?)
        } else {
            None
        };

    // Compute dimensions.
    let flat_rows: usize = input_shape.iter().rev().skip(1).product();
    let total_output = flat_rows
        .checked_mul(out_features)
        .ok_or_else(|| native_dispatch_err(step_idx, "NormLinear: output size overflow".into()))?;

    if flat_rows == 0 || hidden_dim == 0 || out_features == 0 {
        return Err(native_dispatch_err(
            step_idx,
            "NormLinear: zero-size dimension".into(),
        ));
    }

    // Route: simdgroup-tiled when M=flat_rows, K=hidden_dim, N=out_features conform.
    // Two-dispatch path: norm-only → intermediate buffer → simdgroup GEMM.
    if crate::dyn_tensor_metal::should_use_simdgroup(flat_rows, hidden_dim, out_features) {
        return simd::execute_norm_linear_simdgroup(
            model,
            step_idx,
            &input_slice,
            norm_kind,
            eps,
            norm_w,
            norm_b,
            weight,
            bias,
            flat_rows,
            hidden_dim,
            out_features,
            cache,
        );
    }

    // Fallback: fused single-dispatch path (scalar dot-product GEMM).
    let norm_tag = match norm_kind {
        FusedNormKind::LayerNorm => "ln",
        FusedNormKind::RmsNorm => "rms",
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };
    let kernel_name = format!(
        "fused_norm_linear_{norm_tag}_{scalar_str}_b{}",
        u8::from(has_bias)
    );
    let msl_src = norm_linear_msl(scalar_str, norm_kind, has_bias, step_idx)?;

    // Compile pipeline (cached by kernel name).
    let input_buf_count = match (norm_kind, has_bias) {
        (FusedNormKind::LayerNorm, true) => 5, // input, norm_w, norm_b, weight, bias
        (FusedNormKind::LayerNorm, false) => 4, // input, norm_w, norm_b, weight
        (FusedNormKind::RmsNorm, true) => 4,   // input, norm_w, weight, bias
        (FusedNormKind::RmsNorm, false) => 3,  // input, norm_w, weight
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };
    let pipeline = KernelPipeline::from_msl(cache, &msl_src, &kernel_name, input_buf_count, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear pipeline: {e}")))?;

    // Allocate output buffer.
    let out_bytes = total_output
        .checked_mul(elem_bytes)
        .ok_or_else(|| native_dispatch_err(step_idx, "NormLinear: output bytes overflow".into()))?;
    let ctx = cache.context();
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear alloc: {e}")))?;

    // Encode constants.
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| native_dispatch_err(step_idx, "NormLinear: hidden_dim > u32".into()))?;
    let out_features_u32 = u32::try_from(out_features)
        .map_err(|_| native_dispatch_err(step_idx, "NormLinear: out_features > u32".into()))?;
    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| native_dispatch_err(step_idx, "NormLinear: flat_rows > u32".into()))?;

    // Dynamic threadgroup memory for normalized values.
    let tg_mem_bytes = hidden_dim
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| native_dispatch_err(step_idx, "NormLinear: tg_mem bytes overflow".into()))?
        as u64;

    // Encode and dispatch.
    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        // Bind input buffers — layout depends on norm_kind.
        let mut buf_idx: usize = 0;
        enc.set_buffer_with_offset(buf_idx, input_slice.buffer(), input_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, norm_w, 0);
        buf_idx += 1;
        if let Some(nb) = norm_b {
            enc.set_buffer_with_offset(buf_idx, nb, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, weight, 0);
        buf_idx += 1;
        if let Some(b) = bias {
            enc.set_buffer_with_offset(buf_idx, b, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, &out_buf, out_offset);
        buf_idx += 1;

        // Constants.
        enc.set_bytes(buf_idx, &hidden_dim_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &eps);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_features_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &flat_rows_u32);

        // Dynamic threadgroup memory at index 0.
        enc.set_threadgroup_memory_length(0, tg_mem_bytes);

        enc.encode_threadgroups(pipeline.pipeline(), [flat_rows_u32, 1, 1], [TG_SIZE, 1, 1])?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Generate MSL source for the fused NormLinear kernel.
///
/// LayerNorm variant (three-phase):
/// 1. Kahan two-pass mean+variance reduction → `mean`, `inv_std`
/// 2. Normalize + affine (weight+bias) → threadgroup memory
/// 3. GEMM (dot products from threadgroup memory)
///
/// RmsNorm variant (three-phase):
/// 1. Kahan sum of x² → `inv_rms = rsqrt(mean(x²) + eps)`
/// 2. Scale: `x * inv_rms * weight` → threadgroup memory
/// 3. GEMM (dot products from threadgroup memory)
fn norm_linear_msl(
    scalar_type: &str,
    norm_kind: FusedNormKind,
    has_bias: bool,
    step_idx: usize,
) -> Result<String> {
    let tg = TG_SIZE as usize;

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

    // Build buffer parameter list and Phase 2 body based on norm_kind.
    let (norm_params, phase1, phase2) = match norm_kind {
        FusedNormKind::LayerNorm => {
            let algo = welford_msl::DEFAULT_NORM_REDUCTION;
            let preamble = welford_msl::norm_preamble_msl(algo);
            let reduction = welford_msl::norm_reduction_msl(algo, "hidden_dim", tg);
            let load_nb = load("norm_b[i]");
            let norm_b_param = format!("    device const {scalar_type}* norm_b   [[buffer(2)]],\n");
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
            // RmsNorm: sum of x², no mean subtraction. Single-pass reduction.
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
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear MSL: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };

    let (preamble, norm_b_param) = norm_params;

    // Compute buffer indices dynamically based on which params are present.
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
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear MSL tag: unsupported norm kind {norm_kind:?}"),
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

/// Execute a `NativeOpKind::AddNormLinear` step.
///
/// Fuses residual-add + LayerNorm + Linear: `Linear(LN(a + b))`.
/// Same kernel structure as NormLinear but with two input buffers (a, b)
/// and an add phase before normalization.
///
/// Part of #4252.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_add_norm_linear(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
    out_features: usize,
    has_bias: bool,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    // Resolve two graph inputs: residual (a) and new value (b).
    let a_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let b_slice = model.resolve_input_slice(step_idx, 1, buffers)?;

    // Resolve weight buffers.
    let step_weights = &model.def.weight_buffers[step_idx];
    let norm_w = step_weights
        .get("norm_weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "AddNormLinear: missing 'norm_weight'".into()))?;
    let norm_b = step_weights
        .get("norm_bias")
        .ok_or_else(|| native_dispatch_err(step_idx, "AddNormLinear: missing 'norm_bias'".into()))?;
    let weight = step_weights
        .get("weight")
        .ok_or_else(|| native_dispatch_err(step_idx, "AddNormLinear: missing 'weight'".into()))?;
    let bias = if has_bias {
        Some(step_weights.get("bias").ok_or_else(|| {
            native_dispatch_err(step_idx, "AddNormLinear: missing 'bias'".into())
        })?)
    } else {
        None
    };

    // Compute dimensions.
    let flat_rows: usize = input_shape.iter().rev().skip(1).product();
    let total_output = flat_rows
        .checked_mul(out_features)
        .ok_or_else(|| native_dispatch_err(step_idx, "AddNormLinear: output size overflow".into()))?;

    if flat_rows == 0 || hidden_dim == 0 || out_features == 0 {
        return Err(native_dispatch_err(
            step_idx,
            "AddNormLinear: zero-size dimension".into(),
        ));
    }

    // Route: simdgroup-tiled when dimensions qualify.
    if crate::dyn_tensor_metal::should_use_simdgroup(flat_rows, hidden_dim, out_features) {
        return simd::execute_add_norm_linear_simdgroup(
            model,
            step_idx,
            &a_slice,
            &b_slice,
            eps,
            norm_w,
            norm_b,
            weight,
            bias,
            flat_rows,
            hidden_dim,
            out_features,
            cache,
        );
    }

    // Fallback: fused single-dispatch path (scalar dot-product GEMM).
    let kernel_name = format!(
        "fused_add_norm_linear_{scalar_str}_b{}",
        u8::from(has_bias)
    );
    let msl_src = add_norm_linear_msl(scalar_str, has_bias, step_idx)?;

    // input_a, input_b, norm_w, norm_b, weight, [bias,] output = 6 or 7
    let input_buf_count = if has_bias { 6 } else { 5 };
    let pipeline = KernelPipeline::from_msl(cache, &msl_src, &kernel_name, input_buf_count, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear pipeline: {e}")))?;

    let out_bytes = total_output
        .checked_mul(elem_bytes)
        .ok_or_else(|| native_dispatch_err(step_idx, "AddNormLinear: output bytes overflow".into()))?;
    let ctx = cache.context();
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear alloc: {e}")))?;

    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| native_dispatch_err(step_idx, "AddNormLinear: hidden_dim > u32".into()))?;
    let out_features_u32 = u32::try_from(out_features)
        .map_err(|_| native_dispatch_err(step_idx, "AddNormLinear: out_features > u32".into()))?;
    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| native_dispatch_err(step_idx, "AddNormLinear: flat_rows > u32".into()))?;

    let tg_mem_bytes = hidden_dim
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| native_dispatch_err(step_idx, "AddNormLinear: tg_mem overflow".into()))?
        as u64;

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        let mut buf_idx: usize = 0;
        enc.set_buffer_with_offset(buf_idx, a_slice.buffer(), a_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, b_slice.buffer(), b_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, norm_w, 0);
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, norm_b, 0);
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, weight, 0);
        buf_idx += 1;
        if let Some(b) = bias {
            enc.set_buffer_with_offset(buf_idx, b, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, &out_buf, out_offset);
        buf_idx += 1;

        enc.set_bytes(buf_idx, &hidden_dim_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &eps);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &out_features_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &flat_rows_u32);

        enc.set_threadgroup_memory_length(0, tg_mem_bytes);
        enc.encode_threadgroups(pipeline.pipeline(), [flat_rows_u32, 1, 1], [TG_SIZE, 1, 1])?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("AddNormLinear encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Generate MSL source for the fused AddNormLinear kernel.
///
/// Three-phase: add(a,b) → LayerNorm → GEMM from threadgroup memory.
fn add_norm_linear_msl(
    scalar_type: &str,
    has_bias: bool,
    _step_idx: usize,
) -> Result<String> {
    let tg = TG_SIZE as usize;

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

    let load_nw = load("norm_w[i]");
    let load_nb = load("norm_b[i]");
    let load_weight = load("weight[col * hidden_dim + k]");
    let store_out = store("dot");

    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);

    // Buffer indices: a(0), b(1), norm_w(2), norm_b(3), weight(4), [bias(5)], output(5 or 6)
    let (bias_param, bias_add, out_idx) = if has_bias {
        let bp = format!("    device const {scalar_type}* bias      [[buffer(5)]],\n");
        (bp, "        dot += float(bias[col]);\n".to_string(), 6)
    } else {
        (String::new(), String::new(), 5)
    };

    let hd_idx = out_idx + 1;
    let eps_idx = hd_idx + 1;
    let of_idx = eps_idx + 1;
    let fr_idx = of_idx + 1;

    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_add_norm_linear_{scalar_type}_b{has_bias_u}(
    device const {scalar_type}* input_a  [[buffer(0)]],
    device const {scalar_type}* input_b  [[buffer(1)]],
    device const {scalar_type}* norm_w   [[buffer(2)]],
    device const {scalar_type}* norm_b   [[buffer(3)]],
    device const {scalar_type}* weight   [[buffer(4)]],
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

    // Phase 0: compute added = input_a + input_b into threadgroup memory
    // (reuse normed array as scratch — overwritten in Phase 2).
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        normed[i] = float(input_a[base + i]) + float(input_b[base + i]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: LayerNorm reduction on the added values (in threadgroup).
    // Redefine input reads to use normed[i] instead of global input.
    #define NORM_INPUT(idx) normed[idx]
{reduction_add}

    // Phase 2: normalize + affine → threadgroup memory
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = (normed[i] - mean) * inv_std;
        normed[i] = val * {load_nw} + {load_nb};
    }}
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
        reduction_add = add_norm_reduction_msl(tg),
    ))
}

/// LayerNorm reduction MSL reading from threadgroup `normed[]` instead of global memory.
///
/// Kahan two-pass mean+variance from threadgroup data. Produces `mean` and `inv_std`.
fn add_norm_reduction_msl(tg_size: usize) -> String {
    format!(
        r#"
    // --- LayerNorm: Kahan-compensated mean+variance on normed[] (threadgroup) ---
    threadgroup float shared_val[{tg_size}];
    threadgroup float shared_comp[{tg_size}];

    // Pass 1: mean
    float local_sum = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float v = normed[i];
        float y = v - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    shared_val[tid] = local_sum;
    shared_comp[tid] = local_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = {tg_size} / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid]; float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride]; float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_val[0] / max(float(hidden_dim), 1.0f);

    // Pass 2: variance
    local_sum = 0.0f;
    local_comp = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float d = normed[i] - mean;
        float sq = d * d;
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
            float a_val = shared_val[tid]; float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride]; float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float var = shared_val[0] / max(float(hidden_dim), 1.0f);
    float inv_std = metal::precise::rsqrt(var + eps);"#
    )
}

/// RmsNorm reduction MSL: Kahan-compensated sum of x², then rsqrt.
///
/// Produces `float inv_rms` variable for Phase 2.
/// Uses same threadgroup arrays as the Kahan two-pass reduction
/// (`shared_val[TG_SIZE]`, `shared_comp[TG_SIZE]`).
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
