// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv-with-output-stats and conv-with-precomputed-stats dispatch (#1815 Tier 2).
//!
//! Reduces FusedResBlock from 4 → 3 Metal dispatches by:
//! - Phase 1: conv kernel computes output stats in its epilogue (2 dispatches)
//! - Phase 2: conv kernel uses precomputed stats, skipping stats dispatch (1 dispatch)
//!
//! The with-stats conv kernel uses a Kahan-compensated Welford epilogue (#3309)
//! to compute per-channel mean + inv_std of its output, which the next phase
//! uses directly. Replaces the naive E[X²]-E[X]² formula that caused #3233.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::cache::PipelineCache;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;
use crate::MetalBuffer;

use super::MetalTensorData;

#[path = "dyn_tensor_metal_norm_conv_stats_msl.rs"]
mod stats_msl;

// -- Thread-local pipeline cache for FusedResBlock hot path --
//
// Each FusedResBlock dispatches 3 kernel pipelines (stats, conv_with_stats,
// conv_precomputed). The MSL source for each depends only on (activation_type,
// scalar_type) — 10 unique combinations total. Without caching, each dispatch
// regenerates the MSL source (~2-4KB format!), constructs a KernelSource (clone),
// hashes the full source string, and does an equality check on L1 cache hit.
// At 35 FusedResBlocks per Kokoro forward, this is ~105 multi-KB string ops.
//
// This cache stores resolved KernelPipeline handles after the first lookup,
// reducing subsequent lookups to a thread-local array index + Option check.
//
// Slots: 0-1 stats {float,half}, 2-5 conv_with_stats {leaky,snake}×{float,half},
//        6-9 precomputed {leaky,snake}×{float,half}.

const PIPE_SLOTS: usize = 10;

thread_local! {
    static NORM_CONV_PIPE_CACHE: RefCell<[Option<KernelPipeline>; PIPE_SLOTS]> =
        RefCell::new(Default::default());
    /// Thread-local flag: when `true`, FusedResBlock conv kernels use
    /// half-precision accumulators instead of float accumulators. Set via
    /// [`with_fast_half_scope`] at the `step_generate` level.
    static FAST_HALF_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Execute `f` with the fast-half accumulator flag set to `fast_half`.
///
/// Restores the previous value on exit (including on panic). The flag is
/// read by [`fast_half_active`] inside the fused conv dispatch path.
pub(in super::super) fn with_fast_half_scope<T>(fast_half: bool, f: impl FnOnce() -> T) -> T {
    struct Guard(bool);
    impl Drop for Guard {
        fn drop(&mut self) {
            FAST_HALF_ACTIVE.with(|c| c.set(self.0));
        }
    }
    let prev = FAST_HALF_ACTIVE.with(Cell::get);
    FAST_HALF_ACTIVE.with(|c| c.set(fast_half));
    let _guard = Guard(prev);
    f()
}

/// Query the current fast-half accumulator flag.
///
/// Returns `true` only when inside a [`with_fast_half_scope(true, ...)`] call.
pub(super) fn fast_half_active() -> bool {
    FAST_HALF_ACTIVE.with(Cell::get)
}

fn scalar_offset(scalar_type: &str) -> usize {
    if scalar_type == "half" {
        1
    } else {
        0
    }
}

/// Resolve a cached KernelPipeline, bypassing MSL generation on cache hit.
pub(super) fn resolve_cached_pipeline(
    cache: &PipelineCache,
    slot: usize,
    msl_gen: impl FnOnce() -> String,
    entry_point_gen: impl FnOnce() -> String,
    param_count: usize,
) -> std::result::Result<KernelPipeline, crate::error::MetalError> {
    let cached = NORM_CONV_PIPE_CACHE.with(|c| c.borrow()[slot].clone());
    if let Some(pipe) = cached {
        return Ok(pipe);
    }
    let msl = msl_gen();
    let entry = entry_point_gen();
    let pipe = KernelPipeline::from_msl(cache, &msl, &entry, param_count, false)?;
    NORM_CONV_PIPE_CACHE.with(|c| {
        c.borrow_mut()[slot] = Some(pipe.clone());
    });
    Ok(pipe)
}

/// Threadgroup size for the stats kernel (must match norm_conv_fused).
const STATS_TG_SIZE: usize = 256;

/// Threadgroup width for the fused Conv1d kernel (must match norm_conv_fused).
pub(super) const CONV_TG_X: usize = 64;

#[path = "dyn_tensor_metal_norm_conv_precomputed.rs"]
mod precomputed;

/// Precomputed channel statistics from a conv-with-stats dispatch.
///
/// Contains per-channel (b, c_out) mean + inv_std for the conv output,
/// used by the next FusedResBlock phase to skip the stats dispatch.
pub(in super::super) struct PrecomputedStats {
    pub(in super::super) buffer: MetalBuffer,
    pub(in super::super) offset: usize,
}

/// Internal activation binding for the conv dispatch.
pub(super) enum StatsActivation<'a> {
    LeakyRelu { slope: f32 },
    Snake { alpha_data: &'a MetalTensorData },
}

impl StatsActivation<'_> {
    fn with_stats_msl_source(&self, scalar_type: &str) -> String {
        match self {
            Self::LeakyRelu { .. } => {
                stats_msl::fused_norm_conv1d_leaky_relu_with_stats_msl(scalar_type)
            }
            Self::Snake { .. } => stats_msl::fused_norm_conv1d_snake_with_stats_msl(scalar_type),
        }
    }

    fn with_stats_kernel_name(&self, scalar_type: &str) -> String {
        match self {
            Self::LeakyRelu { .. } => {
                format!("fused_norm_conv1d_leaky_relu_with_stats_{scalar_type}")
            }
            Self::Snake { .. } => {
                format!("fused_norm_conv1d_snake_with_stats_{scalar_type}")
            }
        }
    }

    fn standard_msl_source(&self, scalar_type: &str) -> String {
        let fh = fast_half_active() && scalar_type == "half";
        match self {
            Self::LeakyRelu { .. } => {
                super::norm_conv_fused::leaky_relu_conv_msl(scalar_type, fh)
            }
            Self::Snake { .. } => super::norm_conv_fused::snake_conv_msl(scalar_type, fh),
        }
    }

    fn standard_kernel_name(&self, scalar_type: &str) -> String {
        let fh = fast_half_active() && scalar_type == "half";
        match self {
            Self::LeakyRelu { .. } => {
                super::norm_conv_fused::leaky_relu_conv_kernel_name(scalar_type, fh)
            }
            Self::Snake { .. } => {
                super::norm_conv_fused::snake_conv_kernel_name(scalar_type, fh)
            }
        }
    }

    fn input_buffer_count(&self) -> usize {
        match self {
            Self::LeakyRelu { .. } => 7,
            Self::Snake { .. } => 8,
        }
    }
}

impl super::MetalDynBackend {
    /// Fused NormActivConv1d + LeakyRelu with output stats epilogue.
    ///
    /// 2 Metal dispatches (stats + conv_with_stats). Returns output + precomputed
    /// stats for the next FusedResBlock phase.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gpu_norm_activ_conv1d_with_output_stats(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        slope: f64,
        padding: usize,
        dilation: usize,
        residual: Option<super::norm_conv_fused::ResidualParams<'_>>,
        next_phase_eps: f32,
    ) -> Result<(DynTensor, PrecomputedStats)> {
        let slope_f32 = slope as f32;
        if !slope_f32.is_finite() {
            return Err(TensorError::InvalidShape(format!(
                "gpu_norm_activ_conv1d_with_output_stats: slope must be finite, got {slope}"
            )));
        }
        Self::gpu_norm_activ_conv1d_with_output_stats_inner(
            x,
            gamma,
            beta,
            weight,
            bias,
            eps,
            StatsActivation::LeakyRelu { slope: slope_f32 },
            padding,
            dilation,
            residual,
            next_phase_eps,
        )
    }

    /// Fused NormActivConv1d + Snake with output stats epilogue.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gpu_norm_activ_conv1d_snake_with_output_stats(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        alpha: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        padding: usize,
        dilation: usize,
        residual: Option<super::norm_conv_fused::ResidualParams<'_>>,
        next_phase_eps: f32,
    ) -> Result<(DynTensor, PrecomputedStats)> {
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;
        Self::gpu_norm_activ_conv1d_with_output_stats_inner(
            x,
            gamma,
            beta,
            weight,
            bias,
            eps,
            StatsActivation::Snake { alpha_data },
            padding,
            dilation,
            residual,
            next_phase_eps,
        )
    }

    /// Shared inner for with-output-stats dispatch.
    ///
    /// Dispatches: stats(x) + conv_with_stats_epilogue(x, stats) → (output, next_stats).
    #[allow(clippy::too_many_arguments)]
    fn gpu_norm_activ_conv1d_with_output_stats_inner(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        activation: StatsActivation<'_>,
        padding: usize,
        dilation: usize,
        residual: Option<super::norm_conv_fused::ResidualParams<'_>>,
        next_phase_eps: f32,
    ) -> Result<(DynTensor, PrecomputedStats)> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() != 3 {
            return Err(TensorError::InvalidShape(
                "gpu_norm_activ_conv1d_with_output_stats: rank 3 input required".into(),
            ));
        }
        let (batch, in_channels, in_len) = (dims[0], dims[1], dims[2]);
        let w_dims = weight.dims();
        let (out_channels, kernel_size) = (w_dims[0], w_dims[2]);

        let effective_k = (kernel_size - 1) * dilation + 1;
        let padded = in_len + 2 * padding;
        if padded < effective_k {
            return Err(TensorError::InvalidShape(format!(
                "conv_with_stats: padded {padded} < effective_k {effective_k}"
            )));
        }
        let out_len = padded - effective_k + 1;

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "conv_with_stats: eps must be finite+positive, got {eps}"
            )));
        }
        if !next_phase_eps.is_finite() || next_phase_eps <= 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "conv_with_stats: next_phase_eps must be finite+positive, got {next_phase_eps}"
            )));
        }

        let flat_rows =
            batch
                .checked_mul(in_channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;
        let flat_out_rows =
            batch
                .checked_mul(out_channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let gamma_data = gamma.gpu_data::<MetalTensorData>()?;
        let beta_data = beta.gpu_data::<MetalTensorData>()?;
        let weight_data = weight.gpu_data::<MetalTensorData>()?;
        let bias_data = bias.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            // --- Dispatch 1: Compute input stats (cached pipeline) ---
            let stats_pipe = resolve_cached_pipeline(
                cache,
                scalar_offset(scalar_type),
                || super::norm_conv_fused::stats_kernel_msl_source(scalar_type),
                || format!("compute_channel_stats_{scalar_type}"),
                1,
            )
            .map_err(metal_err)?;

            let stats_bytes = flat_rows.checked_mul(2 * size_of::<f32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                }
            })?;
            let (stats_buf, stats_off) =
                crate::arena::arena_alloc_or_create(ctx, stats_bytes).map_err(metal_err)?;

            let spatial_u32 = crate::to_u32(in_len, "conv_stats spatial")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "conv_stats rows")?;

            crate::gpu_scope::get_or_create_batch()?;
            let enc_stats = |b: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = b.new_encoder()?;
                enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                enc.set_buffer_with_offset(1, &stats_buf, stats_off);
                enc.set_bytes(2, &spatial_u32);
                enc.set_bytes(3, &eps_f32);
                enc.encode_threadgroups(
                    stats_pipe.pipeline(), [flat_rows_u32, 1, 1], [STATS_TG_SIZE as u32, 1, 1],
                )?;
                enc.end_encoding();
                Ok(())
            };
            match crate::gpu_scope::encode_into_lazy_batch(|b| enc_stats(b)) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            // --- Dispatch 2: Conv with stats epilogue (cached pipeline) ---
            let conv_slot = match &activation {
                StatsActivation::LeakyRelu { .. } => 2,
                StatsActivation::Snake { .. } => 4,
            } + scalar_offset(scalar_type);
            let conv_param_count = activation.input_buffer_count();
            let conv_pipe = resolve_cached_pipeline(
                cache,
                conv_slot,
                || activation.with_stats_msl_source(scalar_type),
                || activation.with_stats_kernel_name(scalar_type),
                conv_param_count,
            )
            .map_err(metal_err)?;

            let out_shape = vec![batch, out_channels, out_len];
            let total_out = batch
                .checked_mul(out_channels)
                .and_then(|v| v.checked_mul(out_len))
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                })?;
            let out_bytes = total_out.checked_mul(elem_bytes).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                }
            })?;
            let (out_buf, out_off) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            // Allocate next-phase stats buffer.
            let next_stats_bytes =
                flat_out_rows
                    .checked_mul(2 * size_of::<f32>())
                    .ok_or_else(|| TensorError::DimensionOverflow {
                        dims: out_shape.clone(),
                    })?;
            let (next_stats_buf, next_stats_off) =
                crate::arena::arena_alloc_or_create(ctx, next_stats_bytes).map_err(metal_err)?;

            let grid_x = (out_len as u32).div_ceil(CONV_TG_X as u32);
            let grid_x_count_u32 = grid_x;

            // Allocate counter + partials for multi-TG case (Case 2).
            let counter_bytes = flat_out_rows.checked_mul(size_of::<u32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                }
            })?;
            let (counter_buf, counter_off) =
                crate::arena::arena_alloc_or_create(ctx, counter_bytes).map_err(metal_err)?;

            // 3 floats per TG per row: (n, mean, m2) for Welford partials (#3309).
            let partials_bytes = (grid_x as usize)
                .checked_mul(flat_out_rows)
                .and_then(|v| v.checked_mul(3 * size_of::<f32>()))
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                })?;
            let (partials_buf, partials_off) =
                crate::arena::arena_alloc_or_create(ctx, partials_bytes).map_err(metal_err)?;

            // Zero the counter buffer via blit fill — only needed for multi-TG
            // case (grid_x > 1). When grid_x == 1, the MSL epilogue takes the
            // Case 1 branch which writes final stats directly without reading
            // the counter. Skipping the blit avoids a compute→blit→compute
            // encoder switch in the command buffer. Part of #3765.
            if grid_x > 1 {
                let enc_zero = |b: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    b.blit_fill(&counter_buf, counter_off, counter_bytes, 0)
                };
                match crate::gpu_scope::encode_into_lazy_batch(|b| enc_zero(b)) {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(metal_err(e)),
                    Err(e) => return Err(e),
                }
            }

            let batch_u32 = crate::to_u32(batch, "conv_stats batch")?;
            let in_ch_u32 = crate::to_u32(in_channels, "conv_stats in_ch")?;
            let out_ch_u32 = crate::to_u32(out_channels, "conv_stats out_ch")?;
            let in_len_u32 = crate::to_u32(in_len, "conv_stats in_len")?;
            let out_len_u32 = crate::to_u32(out_len, "conv_stats out_len")?;
            let ks_u32 = crate::to_u32(kernel_size, "conv_stats ks")?;
            let pad_u32 = crate::to_u32(padding, "conv_stats pad")?;
            let dil_u32 = crate::to_u32(dilation, "conv_stats dil")?;
            let out_rows_u32 = crate::to_u32(flat_out_rows, "conv_stats out_rows")?;

            let (has_res_u32, res_scale_f32, res_data) = match &residual {
                Some(p) => {
                    let rd = p.residual.gpu_data::<MetalTensorData>()?;
                    (1u32, p.scale, Some(rd))
                }
                None => (0u32, 1.0f32, None),
            };

            let enc_conv = |b: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = b.new_encoder()?;
                enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                enc.set_buffer_with_offset(1, &stats_buf, stats_off);
                enc.set_buffer_with_offset(2, &gamma_data.buffer, gamma_data.byte_offset);
                enc.set_buffer_with_offset(3, &beta_data.buffer, beta_data.byte_offset);
                enc.set_buffer_with_offset(4, &weight_data.buffer, weight_data.byte_offset);
                enc.set_buffer_with_offset(5, &bias_data.buffer, bias_data.byte_offset);
                if let Some(rd) = res_data {
                    enc.set_buffer_with_offset(6, &rd.buffer, rd.byte_offset);
                } else {
                    enc.set_buffer_with_offset(6, &out_buf, out_off);
                }
                enc.set_buffer_with_offset(7, &out_buf, out_off);
                enc.set_bytes(8, &batch_u32);
                enc.set_bytes(9, &in_ch_u32);
                enc.set_bytes(10, &out_ch_u32);
                enc.set_bytes(11, &in_len_u32);
                enc.set_bytes(12, &out_len_u32);
                enc.set_bytes(13, &ks_u32);
                enc.set_bytes(14, &pad_u32);
                enc.set_bytes(15, &dil_u32);
                match &activation {
                    StatsActivation::LeakyRelu { slope } => {
                        enc.set_bytes(16, slope);
                        enc.set_bytes(17, &has_res_u32);
                        enc.set_bytes(18, &res_scale_f32);
                    }
                    StatsActivation::Snake { alpha_data } => {
                        enc.set_buffer_with_offset(16, &alpha_data.buffer, alpha_data.byte_offset);
                        enc.set_bytes(17, &has_res_u32);
                        enc.set_bytes(18, &res_scale_f32);
                    }
                }
                // Epilogue buffers (19-23).
                enc.set_buffer_with_offset(19, &next_stats_buf, next_stats_off);
                enc.set_buffer_with_offset(20, &counter_buf, counter_off);
                enc.set_buffer_with_offset(21, &partials_buf, partials_off);
                enc.set_bytes(22, &grid_x_count_u32);
                enc.set_bytes(23, &next_phase_eps);
                enc.encode_threadgroups(
                    conv_pipe.pipeline(),
                    [grid_x, out_rows_u32, 1],
                    [CONV_TG_X as u32, 1, 1],
                )?;
                enc.end_encoding();
                Ok(())
            };
            match crate::gpu_scope::encode_into_lazy_batch(|b| enc_conv(b)) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_off);
            let output =
                DynTensor::from_gpu_storage(out_shape, dtype, Arc::new(storage), Device::metal())?;
            let precomputed = PrecomputedStats {
                buffer: next_stats_buf,
                offset: next_stats_off,
            };
            Ok((output, precomputed))
        })
    }
}

/// Collect with-stats MSL sources for pre-compilation.
pub(super) fn collect_msl_sources() -> Vec<(&'static str, String)> {
    stats_msl::collect_msl_sources()
}
