// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused NormActivConv1d GPU kernels (#2780).
//!
//! Two-dispatch architecture:
//!   1. `compute_channel_stats` — per-channel mean + inv_std (tiny output)
//!   2. `fused_norm_conv1d_{leaky_relu,snake}` — inline norm + affine +
//!      activation during Conv1d accumulation (no intermediate tensor)
//!
//! Memory traffic savings: eliminates B×C_in×T f32 intermediate write+read.
//! For F0 shapes (B=1, C=512, T=100), saves ~200KB per NormActivConv1d phase.
//!
//! Optionally folds residual add + scale into dispatch 2 (used by
//! FusedResBlock phase 2 to eliminate 2 extra dispatches).
//!
//! Part of #2780: FusedAdainResBlock GPU NativeOp.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;

use super::MetalTensorData;

#[path = "dyn_tensor_metal_norm_conv_fused_msl.rs"]
mod msl;

/// Threadgroup size for the stats kernel (matches AdaIN kernels).
const STATS_TG_SIZE: usize = 256;

/// Threadgroup width for the fused Conv1d kernel.
const CONV_TG_X: usize = 64;

/// Optional residual parameters for the fused kernel.
pub(in super::super) struct ResidualParams<'a> {
    /// Residual tensor to add to conv output. Shape must match output.
    pub residual: &'a DynTensor,
    /// Scale factor applied after residual add (1/sqrt(2) for F0 blocks).
    pub scale: f32,
}

/// Internal activation binding for buffer(16) dispatch.
enum ActivationBinding<'a> {
    LeakyRelu { slope: f32 },
    Snake { alpha_data: &'a MetalTensorData },
}

impl ActivationBinding<'_> {
    fn msl_source(&self, scalar_type: &str, fast_half: bool) -> String {
        match self {
            Self::LeakyRelu { .. } => {
                msl::fused_norm_conv1d_leaky_relu_msl(scalar_type, fast_half)
            }
            Self::Snake { .. } => msl::fused_norm_conv1d_snake_msl(scalar_type, fast_half),
        }
    }

    fn kernel_name(&self, scalar_type: &str, fast_half: bool) -> String {
        if fast_half && scalar_type == "half" {
            match self {
                Self::LeakyRelu { .. } => {
                    "fused_norm_conv1d_leaky_relu_fast_half".to_owned()
                }
                Self::Snake { .. } => "fused_norm_conv1d_snake_fast_half".to_owned(),
            }
        } else {
            match self {
                Self::LeakyRelu { .. } => format!("fused_norm_conv1d_leaky_relu_{scalar_type}"),
                Self::Snake { .. } => format!("fused_norm_conv1d_snake_{scalar_type}"),
            }
        }
    }

    fn input_buffer_count(&self) -> usize {
        match self {
            Self::LeakyRelu { .. } => 7, // input, stats, gamma, beta, weight, bias, residual
            Self::Snake { .. } => 8,     // + alpha device buffer
        }
    }
}

impl super::MetalDynBackend {
    /// Fused NormActivConv1d with LeakyRelu activation.
    ///
    /// Two Metal dispatches (stats + fused conv). See module docs.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gpu_norm_activ_conv1d(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        slope: f64,
        padding: usize,
        dilation: usize,
        residual: Option<ResidualParams<'_>>,
    ) -> Result<DynTensor> {
        let slope_f32 = slope as f32;
        if !slope_f32.is_finite() {
            return Err(TensorError::InvalidShape(format!(
                "gpu_norm_activ_conv1d: slope must be finite, got {slope}"
            )));
        }
        Self::gpu_norm_activ_conv1d_inner(
            x,
            gamma,
            beta,
            weight,
            bias,
            eps,
            ActivationBinding::LeakyRelu { slope: slope_f32 },
            padding,
            dilation,
            residual,
        )
    }

    /// Fused NormActivConv1d with Snake activation.
    ///
    /// Two Metal dispatches (stats + fused conv). Same as LeakyRelu variant
    /// but buffer(16) binds per-channel `alpha [C_in]` as a device buffer
    /// instead of a scalar `slope`.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn gpu_norm_activ_conv1d_snake(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        alpha: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        padding: usize,
        dilation: usize,
        residual: Option<ResidualParams<'_>>,
    ) -> Result<DynTensor> {
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;
        Self::gpu_norm_activ_conv1d_inner(
            x,
            gamma,
            beta,
            weight,
            bias,
            eps,
            ActivationBinding::Snake { alpha_data },
            padding,
            dilation,
            residual,
        )
    }

    /// Shared implementation for fused NormActivConv1d dispatch.
    #[allow(clippy::too_many_arguments)]
    fn gpu_norm_activ_conv1d_inner(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
        activation: ActivationBinding<'_>,
        padding: usize,
        dilation: usize,
        residual: Option<ResidualParams<'_>>,
    ) -> Result<DynTensor> {
        let dtype = x.dtype();
        let st = ScalarType::try_from(dtype)
            .map_err(|_| TensorError::dtype_mismatch(DType::F32, dtype))?;
        let scalar_type = st.msl_str();
        let elem_bytes = st.byte_size();

        let dims = x.dims();
        if dims.len() != 3 {
            return Err(TensorError::InvalidShape(
                "gpu_norm_activ_conv1d requires rank 3 input [B, C_in, T]".into(),
            ));
        }
        let batch = dims[0];
        let in_channels = dims[1];
        let in_len = dims[2];

        if in_len == 0 {
            let w_dims = weight.dims();
            let out_channels = w_dims[0];
            return DynTensor::zeros(&[batch, out_channels, 0], dtype, &Device::metal());
        }

        let w_dims = weight.dims();
        if w_dims.len() != 3 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_norm_activ_conv1d: weight must be rank 3, got {w_dims:?}"
            )));
        }
        let out_channels = w_dims[0];
        let kernel_size = w_dims[2];

        let effective_k = (kernel_size - 1) * dilation + 1;
        let padded = in_len + 2 * padding;
        if padded < effective_k {
            return Err(TensorError::InvalidShape(format!(
                "gpu_norm_activ_conv1d: padded length {padded} < effective kernel {effective_k}"
            )));
        }
        let out_len = padded - effective_k + 1;

        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_norm_activ_conv1d: eps must be finite and positive, got {eps}"
            )));
        }

        let flat_rows =
            batch
                .checked_mul(in_channels)
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
            // --- Dispatch 1: Compute per-channel stats ---
            let stats_kernel_name = format!("compute_channel_stats_{scalar_type}");
            let stats_msl = msl::compute_channel_stats_msl(scalar_type);
            let stats_pipeline =
                KernelPipeline::from_msl(cache, &stats_msl, &stats_kernel_name, 1, false)
                    .map_err(metal_err)?;

            let stats_bytes = flat_rows.checked_mul(2 * size_of::<f32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                }
            })?;
            let (stats_buf, stats_offset) =
                crate::arena::arena_alloc_or_create(ctx, stats_bytes).map_err(metal_err)?;

            let spatial_u32 = crate::to_u32(in_len, "norm_conv stats spatial")?;
            let flat_rows_u32 = crate::to_u32(flat_rows, "norm_conv stats rows")?;
            let tg_size_u32 = STATS_TG_SIZE as u32;

            crate::gpu_scope::get_or_create_batch()?;
            let encode_stats =
                |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch_cmd.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &stats_buf, stats_offset);
                    enc.set_bytes(2, &spatial_u32);
                    enc.set_bytes(3, &eps_f32);
                    enc.encode_threadgroups(
                        stats_pipeline.pipeline(),
                        [flat_rows_u32, 1, 1],
                        [tg_size_u32, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };
            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|b| encode_stats(b));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            // --- Dispatch 2: Fused norm + activation + conv1d ---
            // Pipeline is specialized via function constants for kernel_size,
            // padding, and dilation (#3449). The Metal compiler unrolls the
            // conv inner loop and eliminates dead padding checks.
            // Fast-half accumulator selection: reads the thread-local flag
            // set by `with_fast_half_scope` at the step_generate level.
            // When active AND scalar_type is "half", selects the _fast_half
            // kernel variant with half-precision accumulators (~2x throughput).
            let fh = super::norm_conv_stats::fast_half_active() && scalar_type == "half";
            let conv_kernel_name = activation.kernel_name(scalar_type, fh);
            let conv_msl = activation.msl_source(scalar_type, fh);
            let kernel_size_u32 = crate::to_u32(kernel_size, "norm_conv kernel_size")?;
            let padding_u32 = crate::to_u32(padding, "norm_conv padding")?;
            let dilation_u32 = crate::to_u32(dilation, "norm_conv dilation")?;
            let fc = [(0u32, kernel_size_u32), (1, padding_u32), (2, dilation_u32)];
            let conv_pipeline = KernelPipeline::from_msl_specialized(
                cache,
                &conv_msl,
                &conv_kernel_name,
                activation.input_buffer_count(),
                false,
                &fc,
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
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let batch_u32 = crate::to_u32(batch, "norm_conv batch")?;
            let in_channels_u32 = crate::to_u32(in_channels, "norm_conv in_channels")?;
            let out_channels_u32 = crate::to_u32(out_channels, "norm_conv out_channels")?;
            let in_len_u32 = crate::to_u32(in_len, "norm_conv in_len")?;
            let out_len_u32 = crate::to_u32(out_len, "norm_conv out_len")?;

            let (has_residual_u32, residual_scale_f32, residual_data) = match &residual {
                Some(params) => {
                    let rdata = params.residual.gpu_data::<MetalTensorData>()?;
                    (1u32, params.scale, Some(rdata))
                }
                None => (0u32, 1.0f32, None),
            };

            let out_rows_u32 = crate::to_u32(batch * out_channels, "norm_conv out_rows")?;
            let grid_x = (out_len as u32).div_ceil(CONV_TG_X as u32);

            let encode_conv =
                |batch_cmd: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    let enc = batch_cmd.new_encoder()?;
                    enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                    enc.set_buffer_with_offset(1, &stats_buf, stats_offset);
                    enc.set_buffer_with_offset(2, &gamma_data.buffer, gamma_data.byte_offset);
                    enc.set_buffer_with_offset(3, &beta_data.buffer, beta_data.byte_offset);
                    enc.set_buffer_with_offset(4, &weight_data.buffer, weight_data.byte_offset);
                    enc.set_buffer_with_offset(5, &bias_data.buffer, bias_data.byte_offset);
                    if let Some(rdata) = residual_data {
                        enc.set_buffer_with_offset(6, &rdata.buffer, rdata.byte_offset);
                    } else {
                        enc.set_buffer_with_offset(6, &out_buf, out_offset);
                    }
                    enc.set_buffer_with_offset(7, &out_buf, out_offset);
                    enc.set_bytes(8, &batch_u32);
                    enc.set_bytes(9, &in_channels_u32);
                    enc.set_bytes(10, &out_channels_u32);
                    enc.set_bytes(11, &in_len_u32);
                    enc.set_bytes(12, &out_len_u32);
                    // kernel_size, padding, dilation are function constants —
                    // baked into the pipeline at creation time (#3449).
                    // Buffer 13: activation-specific binding.
                    match &activation {
                        ActivationBinding::LeakyRelu { slope } => {
                            enc.set_bytes(13, slope);
                            enc.set_bytes(14, &has_residual_u32);
                            enc.set_bytes(15, &residual_scale_f32);
                        }
                        ActivationBinding::Snake { alpha_data } => {
                            enc.set_buffer_with_offset(
                                13, &alpha_data.buffer, alpha_data.byte_offset,
                            );
                            enc.set_bytes(14, &has_residual_u32);
                            enc.set_bytes(15, &residual_scale_f32);
                        }
                    }
                    enc.encode_threadgroups(
                        conv_pipeline.pipeline(),
                        [grid_x, out_rows_u32, 1],
                        [CONV_TG_X as u32, 1, 1],
                    )?;
                    enc.end_encoding();
                    Ok(())
                };

            let scope_result = crate::gpu_scope::encode_into_lazy_batch(|b| encode_conv(b));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(out_shape, dtype, Arc::new(storage), Device::metal())
        })
    }
}

/// Re-export stats kernel MSL source for the conv-stats dispatch module (#1815 Tier 2)
/// and the fused dispatch executors (#4264).
pub(crate) fn stats_kernel_msl_source(scalar_type: &str) -> String {
    msl::compute_channel_stats_msl(scalar_type)
}

/// Re-export LeakyRelu conv MSL for the precomputed-stats path (#1815 Tier 2).
/// `fast_half`: when true and `scalar_type` is `"half"`, emits the fast-half
/// variant with half-precision accumulators for ~2x throughput.
pub(super) fn leaky_relu_conv_msl(scalar_type: &str, fast_half: bool) -> String {
    msl::fused_norm_conv1d_leaky_relu_msl(scalar_type, fast_half)
}

/// Re-export Snake conv MSL for the precomputed-stats path (#1815 Tier 2).
/// `fast_half`: when true and `scalar_type` is `"half"`, emits the fast-half
/// variant with half-precision accumulators for ~2x throughput.
pub(super) fn snake_conv_msl(scalar_type: &str, fast_half: bool) -> String {
    msl::fused_norm_conv1d_snake_msl(scalar_type, fast_half)
}

/// Kernel name for the LeakyRelu conv variant.
///
/// Returns the `_fast_half` kernel name when `fast_half` is true and
/// `scalar_type` is `"half"`, otherwise the standard `_{scalar_type}` name.
pub(super) fn leaky_relu_conv_kernel_name(scalar_type: &str, fast_half: bool) -> String {
    if fast_half && scalar_type == "half" {
        "fused_norm_conv1d_leaky_relu_fast_half".to_owned()
    } else {
        format!("fused_norm_conv1d_leaky_relu_{scalar_type}")
    }
}

/// Kernel name for the Snake conv variant.
///
/// Returns the `_fast_half` kernel name when `fast_half` is true and
/// `scalar_type` is `"half"`, otherwise the standard `_{scalar_type}` name.
pub(super) fn snake_conv_kernel_name(scalar_type: &str, fast_half: bool) -> String {
    if fast_half && scalar_type == "half" {
        "fused_norm_conv1d_snake_fast_half".to_owned()
    } else {
        format!("fused_norm_conv1d_snake_{scalar_type}")
    }
}

/// Collect unspecialized NormActivConv1d MSL sources for pre-compilation.
///
/// Only the standalone stats kernels are safe to ship in the build-time
/// metallib. The fused conv kernels are instantiated at runtime with Metal
/// function constants for `kernel_size`, `padding`, and `dilation`; loading
/// their unspecialized entry points from a metallib aborts pipeline creation
/// instead of helping (#3449).
pub(super) fn collect_msl_sources() -> Vec<(&'static str, String)> {
    vec![
        (
            "compute_channel_stats_float",
            msl::compute_channel_stats_msl("float"),
        ),
        (
            "compute_channel_stats_half",
            msl::compute_channel_stats_msl("half"),
        ),
    ]
}

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_conv_fused_tests.rs"]
mod tests;
