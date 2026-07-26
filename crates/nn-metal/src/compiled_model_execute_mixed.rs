// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mixed-precision GEMM dispatch for autocast Phase 2.
//!
//! When a compiled Dispatch step has F16 weights and uses simdgroup GEMM
//! (pre-computed as `MixedGemmInfo` at build time), this module dispatches
//! the `simd_gemm_mixed` kernel directly instead of the IR-generated kernel.
//!
//! The mixed kernel reads F32 activations and F16 weights, using F32
//! accumulators for full precision output. This gives 2x weight bandwidth
//! savings on Apple Silicon without any intermediate overflow risk.
//!
//! Part of #3085 (per-op autocast Phase 2), #2981 (F16 pipeline).

use nn_core::{Result, TensorError};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::dispatch_plan::DispatchMode;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::super::mixed_gemm_msl;
use super::super::{CompiledModel, CompiledModelError, MixedGemmInfo};

/// Dispatch the `simd_gemm_mixed` kernel with explicit buffer references.
///
/// Used by `execute_mixed_dispatch` for Dispatch/NativeOp steps with
/// pre-computed `MixedGemmInfo`.
///
/// Takes F32 activation and F16 weight buffers directly — the caller resolves
/// buffer keys. Output is always F32.
///
/// Part of #3085, #2981.
pub(super) fn dispatch_mixed_gemm_raw(
    cache: &PipelineCache,
    info: &MixedGemmInfo,
    activation: &GpuSlice,
    weight: &MetalBuffer,
    bias: Option<&MetalBuffer>,
    step_idx: usize,
) -> Result<GpuSlice> {
    let ctx = cache.context();

    // Generate MSL for this specific M, K, N configuration.
    let msl = mixed_gemm_msl::generate_mixed_gemm_msl(info);
    let num_inputs = mixed_gemm_msl::mixed_gemm_input_count(info.has_bias);
    let pipeline = KernelPipeline::from_msl(cache, &msl, "simd_gemm_mixed", num_inputs, false)
        .map_err(|e| dispatch_err(step_idx, format!("mixed GEMM pipeline: {e}")))?;

    // Build input list: [activation, weight, optional bias].
    let mut in_bufs: Vec<&MetalBuffer> = vec![activation.buffer(), weight];
    let mut in_offsets: Vec<usize> = vec![activation.byte_offset(), 0];

    if let Some(bias_buf) = bias {
        in_bufs.push(bias_buf);
        in_offsets.push(0);
    }

    // Allocate output buffer (F32).
    let total_output = info
        .batch_count
        .checked_mul(info.m)
        .and_then(|v| v.checked_mul(info.n))
        .ok_or_else(|| {
            dispatch_err(
                step_idx,
                format!(
                    "mixed GEMM output overflow: {}×{}×{}",
                    info.batch_count, info.m, info.n
                ),
            )
        })?;
    let out_bytes = total_output
        .checked_mul(4)
        .ok_or_else(|| dispatch_err(step_idx, "mixed GEMM output bytes overflow".into()))?; // 4 bytes per f32
    let (out_buf, out_offset) = crate::arena::arena_alloc_or_create(ctx, out_bytes)
        .map_err(|e| dispatch_err(step_idx, format!("mixed GEMM output alloc: {e}")))?;

    // Dispatch parameters.
    let m_u32 = u32::try_from(info.m)
        .map_err(|_| dispatch_err(step_idx, "mixed GEMM: M exceeds u32".into()))?;
    let n_u32 = u32::try_from(info.n)
        .map_err(|_| dispatch_err(step_idx, "mixed GEMM: N exceeds u32".into()))?;
    let batch_u32 = u32::try_from(info.batch_count)
        .map_err(|_| dispatch_err(step_idx, "mixed GEMM: batch exceeds u32".into()))?;

    let grid_x = n_u32.div_ceil(32);
    let grid_y = m_u32.div_ceil(32);

    let tg_bytes = mixed_gemm_msl::mixed_gemm_threadgroup_bytes();

    let plan = DispatchMode::Grid3D {
        grid: [grid_x, grid_y, batch_u32],
        threads: [32, 4, 1],
    }
    .plan()
    .map_err(|e| dispatch_err(step_idx, format!("mixed GEMM plan: {e}")))?
    .with_output_elems(total_output)
    .with_constants(vec![]) // All constants embedded in MSL
    .with_use_threadgroups(true)
    .with_threadgroup_memory_bytes(Some(tg_bytes));

    // Encode into the lazy batch.
    crate::gpu_scope::get_or_create_batch()
        .map_err(|e| dispatch_err(step_idx, format!("mixed GEMM batch: {e}")))?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(
        |batch| -> std::result::Result<(), crate::error::MetalError> {
            let enc = batch.new_encoder()?;
            pipeline.encode_into(enc, &in_bufs, &in_offsets, &out_buf, out_offset, &plan)?;
            Ok(())
        },
    );
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(dispatch_err(step_idx, format!("mixed GEMM encode: {e}")));
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::new(out_buf, out_offset))
}

impl CompiledModel {
    /// Execute a Dispatch step using the mixed-precision simdgroup GEMM kernel.
    ///
    /// Resolves activation from graph edges (F32) and weight from
    /// `weight_buffers[step_idx]["weight"]` (F16), then delegates to
    /// [`dispatch_mixed_gemm_raw`].
    pub(super) fn execute_mixed_dispatch(
        &self,
        cache: &PipelineCache,
        info: &MixedGemmInfo,
        step_idx: usize,
        buffers: &[Option<GpuSlice>],
    ) -> Result<GpuSlice> {
        // Resolve activation input (first graph edge).
        let activation = self.resolve_input_slice(step_idx, 0, buffers)?;

        // Resolve weight from pre-uploaded F16 weight buffers.
        let step_weights = &self.def.weight_buffers[step_idx];
        let weight = step_weights
            .get("weight")
            .ok_or_else(|| dispatch_err(step_idx, "mixed GEMM: missing 'weight' buffer".into()))?;

        let bias = if info.has_bias {
            Some(step_weights.get("bias").ok_or_else(|| {
                dispatch_err(step_idx, "mixed GEMM: missing 'bias' buffer".into())
            })?)
        } else {
            None
        };

        dispatch_mixed_gemm_raw(cache, info, &activation, weight, bias, step_idx)
    }
}

fn dispatch_err(step_idx: usize, reason: String) -> TensorError {
    TensorError::from(CompiledModelError::DispatchFailed { step_idx, reason })
}
