// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simdgroup-tiled NormLinear execution (two-dispatch path).
//!
//! When `should_use_simdgroup(flat_rows, hidden_dim, out_features)` is true,
//! NormLinear splits into two Metal dispatches:
//!   1. Norm-only kernel (Phase 1+2): normalization → intermediate global buffer
//!   2. Simdgroup GEMM (Phase 3): intermediate × weights → output
//!
//! The norm phase uses the same Kahan-compensated reduction as the fused path.
//! The GEMM phase uses `simdgroup_matrix<T, 8, 8>` with 32×32 output tiles
//! (4 SIMD groups of 32 threads), matching the standard matmul path.
//!
//! Part of #3292.

use nn_core::Result;
use nn_dsl::trace_compile::FusedNormKind;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::dispatch_plan::DispatchMode;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::native_dispatch_err;
use super::CompiledModel;

/// Threadgroup size for the norm-only dispatch (same as fused path).
const NORM_TG_SIZE: u32 = 256;

/// Execute NormLinear via two dispatches: norm-only → simdgroup GEMM.
///
/// Expects `should_use_simdgroup(flat_rows, hidden_dim, out_features)` is true.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_norm_linear_simdgroup(
    model: &CompiledModel,
    step_idx: usize,
    input_slice: &GpuSlice,
    norm_kind: FusedNormKind,
    eps: f32,
    norm_w: &MetalBuffer,
    norm_b: Option<&MetalBuffer>,
    weight: &MetalBuffer,
    bias: Option<&MetalBuffer>,
    flat_rows: usize,
    hidden_dim: usize,
    out_features: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();
    let is_half = elem_bytes == 2;

    let ctx = cache.context();

    // --- Dispatch 1: Norm-only kernel → intermediate buffer ---
    let norm_tag = match norm_kind {
        FusedNormKind::LayerNorm => "ln",
        FusedNormKind::RmsNorm => "rms",
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear simd: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };

    // Intermediate buffer stores normalized values in step scalar type.
    // Norm computes in f32, stores in scalar_type. Simdgroup GEMM loads
    // scalar_type and accumulates in f32 internally.
    let inter_elems = flat_rows.checked_mul(hidden_dim).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NormLinear simd: intermediate size overflow".into(),
        )
    })?;
    let inter_bytes = inter_elems.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NormLinear simd: intermediate bytes overflow".into(),
        )
    })?;
    let (inter_buf, inter_offset) = crate::arena::arena_alloc_or_create(ctx, inter_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear simd inter alloc: {e}")))?;

    let norm_kernel_name = format!("norm_only_{norm_tag}_{scalar_str}");
    let norm_msl = norm_only_msl(scalar_str, norm_kind, step_idx)?;

    let norm_buf_count = if norm_b.is_some() { 3 } else { 2 }; // input, norm_w, [norm_b,] output
    let norm_buf_count_with_out = norm_buf_count + 1;
    let norm_pipeline = KernelPipeline::from_msl(
        cache,
        &norm_msl,
        &norm_kernel_name,
        norm_buf_count_with_out,
        false,
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear simd norm pipeline: {e}")))?;

    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| native_dispatch_err(step_idx, "NormLinear simd: hidden_dim > u32".into()))?;
    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| native_dispatch_err(step_idx, "NormLinear simd: flat_rows > u32".into()))?;

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        let mut buf_idx: usize = 0;
        enc.set_buffer_with_offset(buf_idx, input_slice.buffer(), input_slice.byte_offset());
        buf_idx += 1;
        enc.set_buffer_with_offset(buf_idx, norm_w, 0);
        buf_idx += 1;
        if let Some(nb) = norm_b {
            enc.set_buffer_with_offset(buf_idx, nb, 0);
            buf_idx += 1;
        }
        enc.set_buffer_with_offset(buf_idx, &inter_buf, inter_offset);
        buf_idx += 1;

        enc.set_bytes(buf_idx, &hidden_dim_u32);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &eps);
        buf_idx += 1;
        enc.set_bytes(buf_idx, &flat_rows_u32);

        // No dynamic threadgroup memory needed — reduction arrays are inline.
        enc.encode_threadgroups(
            norm_pipeline.pipeline(),
            [flat_rows_u32, 1, 1],
            [NORM_TG_SIZE, 1, 1],
        )?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear simd norm encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    // --- Dispatch 2: Simdgroup GEMM from intermediate buffer ---
    let has_bias = bias.is_some();
    let gemm_kernel_name = format!(
        "simd_nl_{scalar_str}_m{flat_rows}_k{hidden_dim}_n{out_features}_b{}",
        u8::from(has_bias),
    );
    let gemm_msl = nn_dsl::emit_simdgroup_linear_standalone_kernel(
        &gemm_kernel_name,
        scalar_type,
        hidden_dim,
        out_features,
        flat_rows,
        has_bias,
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear simd GEMM codegen: {e}")))?;

    let gemm_buf_count = if has_bias { 3 } else { 2 }; // input, weight, [bias]
    let gemm_pipeline =
        KernelPipeline::from_msl(cache, &gemm_msl, &gemm_kernel_name, gemm_buf_count, false)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("NormLinear simd GEMM pipeline: {e}"))
            })?;

    // Allocate final output buffer.
    let total_output = flat_rows.checked_mul(out_features).ok_or_else(|| {
        native_dispatch_err(step_idx, "NormLinear simd: output size overflow".into())
    })?;
    let out_bytes = total_output.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(step_idx, "NormLinear simd: output bytes overflow".into())
    })?;
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear simd out alloc: {e}")))?;

    // Simdgroup dispatch: 32×32 tiles, 128 threads (4 SIMD groups).
    let m_u32 = flat_rows_u32;
    let n_u32 = u32::try_from(out_features)
        .map_err(|_| native_dispatch_err(step_idx, "NormLinear simd: out_features > u32".into()))?;

    let tg_bytes: u64 = if is_half {
        2 * 32 * 33 * 2 + 32 * 33 * 4 // As(half) + Bs(half) + tile_out(float)
    } else {
        3 * 32 * 33 * 4 // As(float) + Bs(float) + tile_out(float)
    };

    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
        threads: [32, 4, 1],
    }
    .plan()
    .map_err(|e| native_dispatch_err(step_idx, format!("NormLinear simd GEMM plan: {e}")))?
    .with_output_elems(total_output)
    .with_constants(vec![])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_bytes));

    let mut inputs: Vec<&MetalBuffer> = Vec::with_capacity(gemm_buf_count);
    inputs.push(&inter_buf);
    inputs.push(weight);
    if let Some(b) = bias {
        inputs.push(b);
    }
    let mut offsets = vec![inter_offset];
    offsets.resize(gemm_buf_count, 0);

    gemm_pipeline
        .dispatch_buffers_with_all_offsets(ctx, &inputs, &offsets, &out_buf, out_offset, &plan)
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("NormLinear simd GEMM dispatch: {e}"))
        })?;

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Execute AddNormLinear via two dispatches: add+norm-only → simdgroup GEMM.
///
/// Part of #4252.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_add_norm_linear_simdgroup(
    model: &CompiledModel,
    step_idx: usize,
    a_slice: &GpuSlice,
    b_slice: &GpuSlice,
    eps: f32,
    norm_w: &MetalBuffer,
    norm_b: &MetalBuffer,
    weight: &MetalBuffer,
    bias: Option<&MetalBuffer>,
    flat_rows: usize,
    hidden_dim: usize,
    out_features: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();
    let is_half = elem_bytes == 2;
    let ctx = cache.context();

    // --- Dispatch 1: Add+Norm-only kernel → intermediate buffer ---
    let inter_elems = flat_rows.checked_mul(hidden_dim).ok_or_else(|| {
        native_dispatch_err(step_idx, "AddNormLinear simd: intermediate size overflow".into())
    })?;
    let inter_bytes = inter_elems.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(step_idx, "AddNormLinear simd: intermediate bytes overflow".into())
    })?;
    let (inter_buf, inter_offset) = crate::arena::arena_alloc_or_create(ctx, inter_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear simd inter alloc: {e}")))?;

    let norm_kernel_name = format!("add_norm_only_ln_{scalar_str}");
    let norm_msl = add_norm_only_msl(scalar_str, step_idx)?;

    // Buffers: input_a, input_b, norm_w, norm_b, output = 5
    let norm_pipeline = KernelPipeline::from_msl(cache, &norm_msl, &norm_kernel_name, 5, false)
        .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear simd norm pipeline: {e}")))?;

    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| native_dispatch_err(step_idx, "AddNormLinear simd: hidden_dim > u32".into()))?;
    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| native_dispatch_err(step_idx, "AddNormLinear simd: flat_rows > u32".into()))?;

    // Threadgroup memory for the added values during norm reduction.
    let tg_mem_bytes = (hidden_dim * size_of::<f32>()) as u64;

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;
        enc.set_buffer_with_offset(0, a_slice.buffer(), a_slice.byte_offset());
        enc.set_buffer_with_offset(1, b_slice.buffer(), b_slice.byte_offset());
        enc.set_buffer_with_offset(2, norm_w, 0);
        enc.set_buffer_with_offset(3, norm_b, 0);
        enc.set_buffer_with_offset(4, &inter_buf, inter_offset);
        enc.set_bytes(5, &hidden_dim_u32);
        enc.set_bytes(6, &eps);
        enc.set_bytes(7, &flat_rows_u32);
        enc.set_threadgroup_memory_length(0, tg_mem_bytes);
        enc.encode_threadgroups(
            norm_pipeline.pipeline(),
            [flat_rows_u32, 1, 1],
            [NORM_TG_SIZE, 1, 1],
        )?;
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("AddNormLinear simd norm encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    // --- Dispatch 2: Simdgroup GEMM from intermediate buffer ---
    let has_bias = bias.is_some();
    let gemm_kernel_name = format!(
        "simd_anl_{scalar_str}_m{flat_rows}_k{hidden_dim}_n{out_features}_b{}",
        u8::from(has_bias),
    );
    let gemm_msl = nn_dsl::emit_simdgroup_linear_standalone_kernel(
        &gemm_kernel_name,
        scalar_type,
        hidden_dim,
        out_features,
        flat_rows,
        has_bias,
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear simd GEMM codegen: {e}")))?;

    let gemm_buf_count = if has_bias { 3 } else { 2 };
    let gemm_pipeline =
        KernelPipeline::from_msl(cache, &gemm_msl, &gemm_kernel_name, gemm_buf_count, false)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("AddNormLinear simd GEMM pipeline: {e}"))
            })?;

    let total_output = flat_rows.checked_mul(out_features).ok_or_else(|| {
        native_dispatch_err(step_idx, "AddNormLinear simd: output size overflow".into())
    })?;
    let out_bytes = total_output.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(step_idx, "AddNormLinear simd: output bytes overflow".into())
    })?;
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear simd out alloc: {e}")))?;

    let m_u32 = flat_rows_u32;
    let n_u32 = u32::try_from(out_features)
        .map_err(|_| native_dispatch_err(step_idx, "AddNormLinear simd: out_features > u32".into()))?;

    let tg_bytes: u64 = if is_half {
        2 * 32 * 33 * 2 + 32 * 33 * 4
    } else {
        3 * 32 * 33 * 4
    };

    let plan = DispatchMode::Grid3D {
        grid: [n_u32.div_ceil(32), m_u32.div_ceil(32), 1],
        threads: [32, 4, 1],
    }
    .plan()
    .map_err(|e| native_dispatch_err(step_idx, format!("AddNormLinear simd GEMM plan: {e}")))?
    .with_output_elems(total_output)
    .with_constants(vec![])
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_bytes));

    let mut inputs: Vec<&MetalBuffer> = Vec::with_capacity(gemm_buf_count);
    inputs.push(&inter_buf);
    inputs.push(weight);
    if let Some(b) = bias {
        inputs.push(b);
    }
    let mut offsets = vec![inter_offset];
    offsets.resize(gemm_buf_count, 0);

    gemm_pipeline
        .dispatch_buffers_with_all_offsets(ctx, &inputs, &offsets, &out_buf, out_offset, &plan)
        .map_err(|e| {
            native_dispatch_err(step_idx, format!("AddNormLinear simd GEMM dispatch: {e}"))
        })?;

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Generate MSL source for add+norm-only kernel (for simdgroup path).
///
/// Phase 0: add a+b into threadgroup scratch.
/// Phase 1: LayerNorm reduction from threadgroup data.
/// Phase 2: normalize+affine → output global buffer.
fn add_norm_only_msl(scalar_type: &str, step_idx: usize) -> Result<String> {
    let tg = NORM_TG_SIZE as usize;

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
    let _ = step_idx; // reserved for error context

    let reduction = super::add_norm_reduction_msl(tg);
    let store_val = store(&format!("val * {load_nw} + {load_nb}"));

    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void add_norm_only_ln_{scalar_type}(
    device const {scalar_type}* input_a  [[buffer(0)]],
    device const {scalar_type}* input_b  [[buffer(1)]],
    device const {scalar_type}* norm_w   [[buffer(2)]],
    device const {scalar_type}* norm_b   [[buffer(3)]],
    device {scalar_type}* output         [[buffer(4)]],
    constant uint& hidden_dim            [[buffer(5)]],
    constant float& eps                  [[buffer(6)]],
    constant uint& flat_rows             [[buffer(7)]],
    threadgroup float* normed            [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    if (gid >= flat_rows) return;
    uint base = gid * hidden_dim;

    // Phase 0: add a+b into threadgroup scratch
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        normed[i] = float(input_a[base + i]) + float(input_b[base + i]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

{reduction}

    // Phase 2: normalize + affine → output buffer
    uint out_base = gid * hidden_dim;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = (normed[i] - mean) * inv_std;
        output[out_base + i] = {store_val};
    }}
}}"#
    ))
}

/// Generate MSL source for norm-only kernel (Phase 1+2, no GEMM).
///
/// Writes normalized values to a global output buffer instead of
/// threadgroup memory. Used as the first dispatch in the simdgroup path.
fn norm_only_msl(scalar_type: &str, norm_kind: FusedNormKind, step_idx: usize) -> Result<String> {
    let tg = NORM_TG_SIZE as usize;

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

    let (norm_b_param, reduction, phase2) = match norm_kind {
        FusedNormKind::LayerNorm => {
            let algo = crate::dyn_tensor_metal::welford_msl::DEFAULT_NORM_REDUCTION;
            let preamble = crate::dyn_tensor_metal::welford_msl::norm_preamble_msl(algo);
            let red =
                crate::dyn_tensor_metal::welford_msl::norm_reduction_msl(algo, "hidden_dim", tg);
            let load_nb = load("norm_b[i]");
            let nb_param = format!("    device const {scalar_type}* norm_b   [[buffer(2)]],\n");
            let out_idx = 3;
            let p2 = format!(
                r#"    // Phase 2: normalize + affine → output buffer
    uint out_base = gid * hidden_dim;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = ({load_input} - mean) * inv_std;
        output[out_base + i] = {store_val};
    }}"#,
                store_val = store(&format!("val * {load_nw} + {load_nb}")),
            );
            ((preamble.to_string(), nb_param, out_idx), red, p2)
        }
        FusedNormKind::RmsNorm => {
            let red = super::rms_reduction_msl(tg);
            let out_idx = 2;
            let p2 = format!(
                r#"    // Phase 2: scale by inv_rms * weight → output buffer
    uint out_base = gid * hidden_dim;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        output[out_base + i] = {store_val};
    }}"#,
                store_val = store(&format!("{load_input} * inv_rms * {load_nw}")),
            );
            ((String::new(), String::new(), out_idx), red, p2)
        }
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear simd MSL: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };

    let (preamble, nb_param, out_idx) = norm_b_param;
    let hd_idx = out_idx + 1;
    let eps_idx = hd_idx + 1;
    let fr_idx = eps_idx + 1;

    let norm_tag = match norm_kind {
        FusedNormKind::LayerNorm => "ln",
        FusedNormKind::RmsNorm => "rms",
        _ => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NormLinear simd MSL tag: unsupported norm kind {norm_kind:?}"),
            ))
        }
    };

    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void norm_only_{norm_tag}_{scalar_type}(
    device const {scalar_type}* input    [[buffer(0)]],
    device const {scalar_type}* norm_w   [[buffer(1)]],
{nb_param}    device {scalar_type}* output         [[buffer({out_idx})]],
    constant uint& hidden_dim            [[buffer({hd_idx})]],
    constant float& eps                  [[buffer({eps_idx})]],
    constant uint& flat_rows             [[buffer({fr_idx})]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    if (gid >= flat_rows) return;
    uint base = gid * hidden_dim;

{reduction}

{phase2}
}}"#
    ))
}
