// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct Metal dispatch for FusedAdainSnake — bypasses DynTensor intermediaries.
//!
//! The standard path creates 4 DynTensors (x, gamma, beta, alpha) from
//! GpuSlice buffers, dispatches through the eager-path `gpu_adain_snake_fused`,
//! then extracts the output buffer back. Each DynTensor wrapping allocates an
//! `Arc<MetalTensorData>` + `Vec<usize>` shape.
//!
//! This direct path encodes the same MSL kernel (`fused_adain_snake_{scalar_type}`)
//! directly on the raw GpuSlice buffer/offset pairs. Zero DynTensor allocations,
//! zero `gpu_data()` extractions.
//!
//! Part of #4252 / #4449.

use nn_core::Result;
use nn_dsl::ir::ScalarType;

use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::welford_msl;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::super::CompiledModel;
use super::native_dispatch_err;

/// Threadgroup size — must match `dyn_tensor_metal_adain_fused.rs::TG_SIZE`.
const TG_SIZE: u32 = 256;

/// Execute `NativeOpKind::FusedAdainSnake` via direct Metal dispatch.
///
/// Dispatches the fused InstanceNorm + affine(gamma, beta) + Snake(alpha)
/// MSL kernel directly on GpuSlice buffer/offset pairs. Eliminates 4
/// DynTensor wrappings + 1 gpu_data extraction vs. the bridge path.
///
/// Returns the output `GpuSlice` (arena-allocated or fresh buffer).
pub(in super::super) fn execute_fused_adain_snake_direct(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);
    let elem_bytes = scalar_type.byte_size();
    let st_str = scalar_type.msl_str();

    let batch = input_shape[0];
    let spatial: usize = input_shape[2..].iter().product();

    if spatial == 0 {
        // Zero-length spatial dimension: return a zero-sized output slice.
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(cache.context(), 0).map_err(|e| {
                native_dispatch_err(step_idx, format!("FusedAdainSnake direct alloc (zero): {e}"))
            })?;
        return Ok(GpuSlice::from_ref(&out_buf, out_offset));
    }

    let flat_rows = batch
        .checked_mul(channels)
        .ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!("FusedAdainSnake direct: B*C overflow ({batch} * {channels})"),
            )
        })?;
    let total_elems = flat_rows.checked_mul(spatial).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!("FusedAdainSnake direct: total overflow ({flat_rows} * {spatial})"),
        )
    })?;

    // Validate eps.
    if !eps.is_finite() || eps <= 0.0 {
        return Err(native_dispatch_err(
            step_idx,
            format!("FusedAdainSnake direct: eps must be finite and positive, got {eps}"),
        ));
    }

    // Resolve GpuSlice inputs: x (0), gamma (1), beta (2).
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let gamma_slice = model.resolve_input_slice(step_idx, 1, buffers)?;
    let beta_slice = model.resolve_input_slice(step_idx, 2, buffers)?;

    // Alpha is a static weight: [C] per-channel Snake parameter.
    let weights = &model.def.weight_buffers[step_idx];
    let alpha_buf = weights.get("alpha").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "FusedAdainSnake direct: missing weight 'alpha'".into(),
        )
    })?;

    // Compile (or cache-hit) the MSL kernel.
    let kernel_name = format!("fused_adain_snake_{st_str}");
    let msl_src = generate_fused_adain_snake_msl(st_str);
    let pipeline = KernelPipeline::from_msl(
        cache,
        &msl_src,
        &kernel_name,
        4, // 4 input buffers: x, gamma, beta, alpha
        false,
    )
    .map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedAdainSnake direct pipeline: {e}"),
        )
    })?;

    // Allocate output buffer.
    let out_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!("FusedAdainSnake direct: output bytes overflow ({total_elems} * {elem_bytes})"),
        )
    })?;
    let (out_buf, out_offset) =
        crate::arena::arena_alloc_or_create(cache.context(), out_bytes).map_err(|e| {
            native_dispatch_err(step_idx, format!("FusedAdainSnake direct alloc: {e}"))
        })?;

    // Encode the dispatch directly on raw buffers.
    let spatial_u32 = crate::to_u32(spatial, "fused_adain_snake spatial")?;
    let channels_u32 = crate::to_u32(channels, "fused_adain_snake channels")?;
    let flat_rows_u32 = crate::to_u32(flat_rows, "fused_adain_snake flat_rows")?;

    let encode =
        |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
            let enc = batch.new_encoder()?;
            enc.set_buffer_with_offset(0, x_slice.buffer(), x_slice.byte_offset());
            enc.set_buffer_with_offset(1, gamma_slice.buffer(), gamma_slice.byte_offset());
            enc.set_buffer_with_offset(2, beta_slice.buffer(), beta_slice.byte_offset());
            enc.set_buffer_with_offset(3, alpha_buf, 0);
            enc.set_buffer_with_offset(4, &out_buf, out_offset);
            enc.set_bytes(5, &spatial_u32);
            enc.set_bytes(6, &channels_u32);
            enc.set_bytes(7, &eps);
            enc.encode_threadgroups(
                pipeline.pipeline(),
                [flat_rows_u32, 1, 1],
                [TG_SIZE, 1, 1],
            )?;
            enc.end_encoding();
            Ok(())
        };

    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch_cmd| encode(batch_cmd));
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("FusedAdainSnake direct encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Generate the MSL source for the fused AdaIN+Snake direct kernel.
///
/// Uses standard gamma convention: `y = gamma * normed + beta` (no residual).
/// Reuses the Welford reduction infrastructure for mean/variance computation.
///
/// This is functionally identical to `dyn_tensor_metal_adain_fused_msl::
/// fused_adain_snake_msl(scalar_type, false)`, but generated here to avoid
/// cross-module visibility dependencies. The MSL source is cached by
/// `KernelPipeline::from_msl` via content hash, so duplicated source strings
/// are automatically deduplicated at the pipeline level.
pub(crate) fn generate_fused_adain_snake_msl(scalar_type: &str) -> String {
    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);
    let reduction = welford_msl::norm_reduction_msl(algo, "spatial_len", TG_SIZE as usize);

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
    float a = max(float(alpha[gid % channels]), 1e-8f); // [C] indexed by channel, clamped
    float inv_a = 1.0f / a;

    // InstanceNorm + Affine + Snake.
    // normed = (x - mean) * inv_std
    // y = g * normed + b  (standard AdaIN)
    // out = y + (1/a) * sin^2(a * y)
    for (uint i = tid; i < spatial_len; i += tg_size) {{
        float normed = (float(input[base + i]) - mean) * inv_std;
        float y = g * normed + b;
        float sin_val = sin(a * y);
        output[base + i] = {scalar_type}(y + inv_a * sin_val * sin_val);
    }}
}}
"#
    )
}

/// Check whether the FusedAdainSnake direct path supports the given scalar type.
pub(crate) fn supports_scalar_type(st: ScalarType) -> bool {
    matches!(st, ScalarType::F32 | ScalarType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_adain_snake_direct_msl_f32_structure() {
        let msl = generate_fused_adain_snake_msl("float");

        // Single kernel entry point.
        let kernel_count = msl.matches("kernel void").count();
        assert_eq!(
            kernel_count, 1,
            "MSL must have exactly 1 kernel entry (single dispatch)"
        );
        assert!(
            msl.contains("kernel void fused_adain_snake_float"),
            "kernel name must encode F32 dtype"
        );

        // 5 buffer bindings: input, gamma, beta, alpha, output.
        for i in 0..5 {
            assert!(
                msl.contains(&format!("[[buffer({i})]]")),
                "MSL must have buffer({i}) binding"
            );
        }

        // 3 scalar constants: spatial_len, channels, eps.
        assert!(msl.contains("constant uint& spatial_len"), "missing spatial_len");
        assert!(msl.contains("constant uint& channels"), "missing channels");
        assert!(msl.contains("constant float& eps"), "missing eps");

        // AdaIN formula present.
        assert!(
            msl.contains("g * normed + b"),
            "MSL must compute g * normed + b (standard AdaIN)"
        );

        // Snake activation present.
        assert!(
            msl.contains("sin(a * y)"),
            "MSL must compute sin(a * y) for Snake"
        );
        assert!(
            msl.contains("inv_a * sin_val * sin_val"),
            "MSL must compute (1/a) * sin^2(a * y)"
        );

        // Alpha clamped.
        assert!(
            msl.contains("max(float(alpha["),
            "alpha must be clamped via max()"
        );
    }

    #[test]
    fn test_fused_adain_snake_direct_msl_f16_has_half_io() {
        let msl = generate_fused_adain_snake_msl("half");

        // I/O pointers are half.
        assert!(
            msl.contains("device const half* input"),
            "F16 MSL must declare half input"
        );
        assert!(
            msl.contains("device half* output"),
            "F16 MSL must declare half output"
        );
        // gamma/beta/alpha buffers are also half.
        assert!(msl.contains("device const half* gamma"), "missing half gamma");
        assert!(msl.contains("device const half* beta"), "missing half beta");
        assert!(msl.contains("device const half* alpha"), "missing half alpha");

        // Accumulators use float() promotion.
        assert!(
            msl.contains("float(input["),
            "F16 MSL must cast input reads to float"
        );
        assert!(
            msl.contains("float(gamma["),
            "F16 MSL must cast gamma reads to float"
        );
        assert!(
            msl.contains("float(alpha["),
            "F16 MSL must cast alpha reads to float"
        );

        // Output cast back to half.
        assert!(
            msl.contains("half("),
            "F16 MSL must cast output back to half"
        );

        // Kernel function name encodes dtype.
        assert!(
            msl.contains("fused_adain_snake_half"),
            "kernel name must include 'half'"
        );
    }

    #[test]
    fn test_fused_adain_snake_direct_msl_no_raw_f16_reads() {
        let msl = generate_fused_adain_snake_msl("half");
        // Every `input[` read must be within a `float()` cast.
        assert!(
            !msl.contains("= input[base") && !msl.contains(", input[base"),
            "F16 MSL must not have raw input reads (missing float() cast)"
        );
    }

    #[test]
    fn test_supports_scalar_type_f32() {
        assert!(supports_scalar_type(ScalarType::F32));
    }

    #[test]
    fn test_supports_scalar_type_f16() {
        assert!(supports_scalar_type(ScalarType::F16));
    }

    #[test]
    fn test_supports_scalar_type_bf16_unsupported() {
        assert!(!supports_scalar_type(ScalarType::BF16));
    }

    #[test]
    fn test_fused_adain_snake_direct_msl_includes_metal_header() {
        let msl = generate_fused_adain_snake_msl("float");
        assert!(
            msl.contains("#include <metal_stdlib>"),
            "MSL must include metal stdlib"
        );
        assert!(
            msl.contains("using namespace metal"),
            "MSL must use metal namespace"
        );
    }

    #[test]
    fn test_fused_adain_snake_direct_msl_threadgroup_dispatch() {
        let msl = generate_fused_adain_snake_msl("float");
        // Must use threadgroup_position_in_grid (one TG per B*C row).
        assert!(
            msl.contains("threadgroup_position_in_grid"),
            "MSL must use threadgroup-based dispatch"
        );
        assert!(
            msl.contains("thread_index_in_threadgroup"),
            "MSL must use thread_index_in_threadgroup for reduction"
        );
    }
}
