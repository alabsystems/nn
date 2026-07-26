// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct Metal dispatch for FusedSnakeInstanceNorm — bypasses DynTensor intermediaries.
//!
//! Fuses Snake activation + InstanceNorm into a single Metal dispatch:
//!   1. Snake activation: `y = x + (1/alpha) * sin(alpha * x)^2`
//!   2. InstanceNorm: per-channel mean/var on Snake output, then normalize.
//!
//! One threadgroup per `[B, C]` row. Each threadgroup:
//!   - Applies Snake element-wise while accumulating Welford mean/var
//!   - Normalizes the Snake output using the computed statistics
//!
//! Saves 1 Metal dispatch per Snake+InstanceNorm pair in the Kokoro generator.
//!
//! Part of #4264.

use nn_core::Result;
use nn_dsl::ir::ScalarType;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::super::CompiledModel;
use super::native_dispatch_err;

/// Threadgroup size — matches other norm kernels.
const TG_SIZE: u32 = 256;

/// Execute `NativeOpKind::FusedSnakeInstanceNorm` via direct Metal dispatch.
///
/// Dispatches the fused Snake + InstanceNorm MSL kernel directly on GpuSlice
/// buffer/offset pairs. Eliminates DynTensor wrappings vs. the bridge path.
///
/// Input: x `[B, C, T]`, alpha `[C]` (static weight).
/// Output: InstanceNorm(Snake(x)) with shape `[B, C, T]`.
///
/// Returns the output `GpuSlice` (arena-allocated or fresh buffer).
pub(in super::super) fn execute_fused_snake_instance_norm_direct(
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
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(cache.context(), 0).map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedSnakeInstanceNorm direct alloc (zero): {e}"),
                )
            })?;
        return Ok(GpuSlice::from_ref(&out_buf, out_offset));
    }

    let flat_rows = batch.checked_mul(channels).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!("FusedSnakeInstanceNorm direct: B*C overflow ({batch} * {channels})"),
        )
    })?;
    let total_elems = flat_rows.checked_mul(spatial).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!("FusedSnakeInstanceNorm direct: total overflow ({flat_rows} * {spatial})"),
        )
    })?;

    // Validate eps.
    if !eps.is_finite() || eps <= 0.0 {
        return Err(native_dispatch_err(
            step_idx,
            format!("FusedSnakeInstanceNorm direct: eps must be finite and positive, got {eps}"),
        ));
    }

    // Resolve GpuSlice input: x (0).
    let x_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Alpha is a static weight: [C] per-channel Snake parameter.
    let weights = &model.def.weight_buffers[step_idx];
    let alpha_buf = weights.get("alpha").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "FusedSnakeInstanceNorm direct: missing weight 'alpha'".into(),
        )
    })?;

    // Compile (or cache-hit) the MSL kernel.
    let kernel_name = format!("fused_snake_instance_norm_{st_str}");
    let msl_src = generate_fused_snake_instance_norm_msl(st_str);
    let pipeline = KernelPipeline::from_msl(
        cache,
        &msl_src,
        &kernel_name,
        2, // 2 input buffers: x, alpha
        false,
    )
    .map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("FusedSnakeInstanceNorm direct pipeline: {e}"),
        )
    })?;

    // Allocate output buffer.
    let out_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!(
                "FusedSnakeInstanceNorm direct: output bytes overflow ({total_elems} * {elem_bytes})"
            ),
        )
    })?;
    let (out_buf, out_offset) =
        crate::arena::arena_alloc_or_create(cache.context(), out_bytes).map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("FusedSnakeInstanceNorm direct alloc: {e}"),
            )
        })?;

    // Encode the dispatch directly on raw buffers.
    let spatial_u32 = crate::to_u32(spatial, "fused_snake_instance_norm spatial")?;
    let channels_u32 = crate::to_u32(channels, "fused_snake_instance_norm channels")?;
    let flat_rows_u32 = crate::to_u32(flat_rows, "fused_snake_instance_norm flat_rows")?;

    let encode =
        |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
            let enc = batch.new_encoder()?;
            enc.set_buffer_with_offset(0, x_slice.buffer(), x_slice.byte_offset());
            enc.set_buffer_with_offset(1, alpha_buf, 0);
            enc.set_buffer_with_offset(2, &out_buf, out_offset);
            enc.set_bytes(3, &spatial_u32);
            enc.set_bytes(4, &channels_u32);
            enc.set_bytes(5, &eps);
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
                format!("FusedSnakeInstanceNorm direct encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Generate the MSL source for the fused Snake + InstanceNorm kernel.
///
/// Three-pass design within a single kernel (#4335):
///   Pass 1: Apply Snake activation, write output, accumulate naive sum.
///   Pass 2: Read back Snake output, accumulate naive sum of (y - mean)^2.
///   Pass 3: Normalize the output in-place.
///
/// Uses naive (non-Kahan) accumulation and standard `rsqrt()` to match
/// PyTorch MPS InstanceNorm behavior. The previous Welford single-pass
/// design produced slightly different floating-point results that
/// compounded through 35+ chained FusedResBlocks into +35.8% amplitude
/// divergence vs PyTorch reference.
///
/// This avoids materializing an intermediate buffer for the Snake output.
pub(crate) fn generate_fused_snake_instance_norm_msl(scalar_type: &str) -> String {
    // Naive two-pass mean/variance matching PyTorch MPS (#4335).
    // Cannot use shared norm_reduction_msl because we interleave Snake
    // activation with the first sum pass.
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_snake_instance_norm_{scalar_type}(
    device const {scalar_type}* input  [[buffer(0)]],
    device const {scalar_type}* alpha  [[buffer(1)]],
    device {scalar_type}* output       [[buffer(2)]],
    constant uint& spatial_len [[buffer(3)]],
    constant uint& channels    [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * spatial_len;

    // Load per-channel alpha, clamped to avoid division by zero.
    float a = max(float(alpha[gid % channels]), 1e-8f);
    float inv_a = 1.0f / a;

    // --- Pass 1: Snake activation + naive sum for mean (PyTorch compatible, #4335) ---
    // Apply Snake: y = x + (1/a) * sin(a * x)^2
    // Accumulate naive sum (no Kahan compensation) to match PyTorch MPS.
    threadgroup float shared_val[{TG_SIZE}];
    float local_sum = 0.0f;

    for (uint i = tid; i < spatial_len; i += tg_size) {{
        float x = float(input[base + i]);
        float sin_val = sin(a * x);
        float y = x + inv_a * sin_val * sin_val;

        // Write Snake output to buffer (will be normalized in pass 3).
        output[base + i] = {scalar_type}(y);
        local_sum += y;
    }}

    // Threadgroup tree reduction for sum.
    shared_val[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            shared_val[tid] += shared_val[tid + stride];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_val[0] / max(float(spatial_len), 1.0f);

    // --- Pass 2: Naive sum of (y - mean)^2 for variance (PyTorch compatible) ---
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_var = 0.0f;
    for (uint i = tid; i < spatial_len; i += tg_size) {{
        float diff = float(output[base + i]) - mean;
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
    float variance = shared_val[0] / max(float(spatial_len), 1.0f);
    float inv_std = rsqrt(variance + eps);

    // --- Pass 3: Normalize in-place ---
    for (uint i = tid; i < spatial_len; i += tg_size) {{
        float y = float(output[base + i]);
        output[base + i] = {scalar_type}((y - mean) * inv_std);
    }}
}}
"#,
    )
}

/// Check whether the FusedSnakeInstanceNorm direct path supports the given scalar type.
pub(crate) fn supports_scalar_type(st: ScalarType) -> bool {
    matches!(st, ScalarType::F32 | ScalarType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_snake_instance_norm_msl_f32_structure() {
        let msl = generate_fused_snake_instance_norm_msl("float");

        // Single kernel entry point.
        let kernel_count = msl.matches("kernel void").count();
        assert_eq!(
            kernel_count, 1,
            "MSL must have exactly 1 kernel entry (single dispatch)"
        );
        assert!(
            msl.contains("kernel void fused_snake_instance_norm_float"),
            "kernel name must encode F32 dtype"
        );

        // 3 buffer bindings: input, alpha, output.
        for i in 0..3 {
            assert!(
                msl.contains(&format!("[[buffer({i})]]")),
                "MSL must have buffer({i}) binding"
            );
        }

        // 3 scalar constants: spatial_len, channels, eps.
        assert!(msl.contains("constant uint& spatial_len"), "missing spatial_len");
        assert!(msl.contains("constant uint& channels"), "missing channels");
        assert!(msl.contains("constant float& eps"), "missing eps");

        // Snake activation present.
        assert!(
            msl.contains("sin(a * x)"),
            "MSL must compute sin(a * x) for Snake"
        );
        assert!(
            msl.contains("inv_a * sin_val * sin_val"),
            "MSL must compute (1/a) * sin^2(a * x)"
        );

        // InstanceNorm normalization present.
        assert!(
            msl.contains("(y - mean) * inv_std"),
            "MSL must normalize with (y - mean) * inv_std"
        );

        // PyTorch-compatible naive reduction (#4335) — no Welford, no Kahan.
        assert!(
            msl.contains("local_sum += y"),
            "MSL must use naive sum accumulation (PyTorch compat)"
        );
        assert!(
            !msl.contains("local_m2"),
            "MSL must NOT use Welford M2 accumulator (PyTorch compat)"
        );
        assert!(
            !msl.contains("local_count"),
            "MSL must NOT use Welford count (PyTorch compat)"
        );
        assert!(
            msl.contains("rsqrt(variance + eps)"),
            "MSL must use standard rsqrt (PyTorch compat)"
        );
        assert!(
            !msl.contains("precise::rsqrt"),
            "MSL must NOT use precise::rsqrt (PyTorch compat)"
        );

        // Alpha clamped.
        assert!(
            msl.contains("max(float(alpha["),
            "alpha must be clamped via max()"
        );
    }

    #[test]
    fn test_fused_snake_instance_norm_msl_f16_has_half_io() {
        let msl = generate_fused_snake_instance_norm_msl("half");

        // I/O pointers are half.
        assert!(
            msl.contains("device const half* input"),
            "F16 MSL must declare half input"
        );
        assert!(
            msl.contains("device half* output"),
            "F16 MSL must declare half output"
        );
        assert!(msl.contains("device const half* alpha"), "missing half alpha");

        // Accumulators use float() promotion.
        assert!(
            msl.contains("float(input["),
            "F16 MSL must cast input reads to float"
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
            msl.contains("fused_snake_instance_norm_half"),
            "kernel name must include 'half'"
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
    fn test_fused_snake_instance_norm_msl_three_pass_design() {
        let msl = generate_fused_snake_instance_norm_msl("float");

        // Pass 1 writes Snake output + accumulates naive sum.
        assert!(
            msl.contains("output[base + i] = float(y)") || msl.contains("output[base + i] = float"),
            "Pass 1 must write Snake output for later passes"
        );
        assert!(
            msl.contains("local_sum += y"),
            "Pass 1 must accumulate naive sum for mean"
        );

        // Pass 2 reads back Snake output, accumulates (y-mean)^2.
        assert!(
            msl.contains("float(output[base + i]) - mean"),
            "Pass 2 must read back Snake output and compute diff from mean"
        );
        assert!(
            msl.contains("local_var += diff * diff"),
            "Pass 2 must accumulate naive variance"
        );

        // Pass 3 normalizes in-place.
        assert!(
            msl.contains("float y = float(output[base + i])"),
            "Pass 3 must read back from output buffer for normalization"
        );
    }
}
