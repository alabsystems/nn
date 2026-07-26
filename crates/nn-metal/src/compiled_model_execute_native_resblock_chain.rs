// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! FusedResBlockChain executor: chains 2-4 FusedResBlocks in a single NativeOp.
//!
//! Each block in the chain executes the same logic as `execute_native_fused_resblock`
//! (2x NormActivConv1d + residual add), but the output of block i feeds directly
//! as input to block i+1 without going through the compiled plan's buffer machinery.
//!
//! This reduces dispatch count by eliminating N-1 inter-block transitions and
//! lets the lazy command buffer batch the entire chain more efficiently.
//!
//! Part of #4264.

use nn_core::Result;
use nn_dsl::{NormActivConv1dParams, NormActivation, ResBlockChainEntry, StyleBatchOffset};

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

/// Pre-computed weight lookup keys for a block+phase in the chain.
struct ChainPhaseKeys {
    alpha: String,
    conv_weight: String,
    conv_bias: String,
}

impl ChainPhaseKeys {
    fn new(block_idx: usize, phase: usize) -> Self {
        Self {
            alpha: format!("block{block_idx}_p{phase}_alpha"),
            conv_weight: format!("block{block_idx}_p{phase}_conv_weight"),
            conv_bias: format!("block{block_idx}_p{phase}_conv_bias"),
        }
    }
}

/// Execute a `NativeOpKind::FusedResBlockChain` step.
///
/// Chains N FusedResBlocks: for each block, runs 2x NormActivConv1d + residual
/// add. The output of block i becomes the input to block i+1.
///
/// All blocks share the batched style projection output for gamma/beta.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_native_fused_resblock_chain(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    blocks: &[ResBlockChainEntry],
    input_steps: &[usize],
    style_batch_offsets: &[StyleBatchOffset],
    first_shortcut_step: Option<usize>,
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    if blocks.is_empty() {
        return Err(native_dispatch_err(
            step_idx,
            "FusedResBlockChain: empty blocks list".into(),
        ));
    }
    if blocks.len() != style_batch_offsets.len() {
        return Err(native_dispatch_err(
            step_idx,
            format!(
                "FusedResBlockChain: blocks.len()={} != style_batch_offsets.len()={}",
                blocks.len(),
                style_batch_offsets.len()
            ),
        ));
    }
    if input_steps.len() < 2 {
        return Err(native_dispatch_err(
            step_idx,
            format!(
                "FusedResBlockChain: expected >=2 input_steps [x, style], got {}",
                input_steps.len()
            ),
        ));
    }

    let dtype = model.step_dtype(step_idx);

    // Resolve initial input x from buffers.
    let x_step = input_steps[0];
    let x_slice = buffers[x_step]
        .as_ref()
        .map(GpuSlice::alias)
        .ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!("FusedResBlockChain: input_steps[0]={x_step} has no buffer"),
            )
        })?;

    // Resolve batched style projection output (shared across all blocks).
    let style_step = input_steps[1];
    let style_slice = buffers[style_step]
        .as_ref()
        .map(GpuSlice::alias)
        .ok_or_else(|| {
            native_dispatch_err(
                step_idx,
                format!("FusedResBlockChain: input_steps[1]={style_step} has no buffer"),
            )
        })?;

    // Resolve optional first-block shortcut.
    let first_shortcut_tensor = if let Some(sc_step) = first_shortcut_step {
        let sc_slice = buffers[sc_step]
            .as_ref()
            .map(GpuSlice::alias)
            .ok_or_else(|| {
                native_dispatch_err(
                    step_idx,
                    format!("FusedResBlockChain: first_shortcut_step {sc_step} has no buffer"),
                )
            })?;
        Some(sc_slice)
    } else {
        None
    };

    let first_block = &blocks[0];
    let batch = first_block.phase1.input_shape[0];

    // Build initial x_tensor from input shape.
    let mut current_tensor =
        slice_to_dyn(&x_slice, &first_block.phase1.input_shape, dtype)?;

    // Process each block in the chain.
    for (block_idx, (block, sbo)) in blocks.iter().zip(style_batch_offsets.iter()).enumerate() {
        let channels1 = block.phase1.input_shape[1];
        let channels2 = block.phase2.input_shape[1];

        // Resolve gamma/beta from batched style projection output.
        let slice_bytes = style_slice.buffer().len() - style_slice.byte_offset();
        let total_out_dim = slice_bytes / (batch * dtype.size_bytes());
        let batch_tensor = slice_to_dyn(&style_slice, &[batch, total_out_dim], dtype)?;

        let mut off = sbo.offset;
        let g1_2d = batch_tensor
            .narrow(1, off, sbo.channels1)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] narrow g1: {e}"))
            })?;
        off += sbo.channels1;
        let b1_2d = batch_tensor
            .narrow(1, off, sbo.channels1)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] narrow b1: {e}"))
            })?;
        off += sbo.channels1;
        let g2_2d = batch_tensor
            .narrow(1, off, sbo.channels2)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] narrow g2: {e}"))
            })?;
        off += sbo.channels2;
        let b2_2d = batch_tensor
            .narrow(1, off, sbo.channels2)
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] narrow b2: {e}"))
            })?;

        // Reshape to [B, C, 1] for AdaIN compatibility (zero-copy).
        let gamma1 = g1_2d
            .reshape([batch, sbo.channels1, 1])
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] reshape g1: {e}"))
            })?;
        let beta1 = b1_2d
            .reshape([batch, sbo.channels1, 1])
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] reshape b1: {e}"))
            })?;
        let gamma2 = g2_2d
            .reshape([batch, sbo.channels2, 1])
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] reshape g2: {e}"))
            })?;
        let beta2 = b2_2d
            .reshape([batch, sbo.channels2, 1])
            .map_err(|e| {
                native_dispatch_err(step_idx, format!("chain[{block_idx}] reshape b2: {e}"))
            })?;

        // Determine residual: for block 0, use shortcut if provided; else identity.
        let residual_tensor = if block_idx == 0 {
            if let Some(ref sc_slice) = first_shortcut_tensor {
                let sc_shape = vec![batch, block.phase2.output_channels, block.phase1.input_shape[2]];
                slice_to_dyn(sc_slice, &sc_shape, dtype)?
            } else {
                current_tensor.clone()
            }
        } else {
            current_tensor.clone()
        };

        // Execute the two phases using the same 3-dispatch fast path as FusedResBlock.
        let block_output = execute_single_block(
            model,
            step_idx,
            block_idx,
            &current_tensor,
            &residual_tensor,
            &gamma1,
            &beta1,
            &gamma2,
            &beta2,
            &block.phase1,
            &block.phase2,
            block.residual_scale,
            channels1,
            channels2,
        )?;

        current_tensor = block_output;
    }

    dyn_to_slice(&current_tensor, step_idx, "FusedResBlockChain")
}

/// Execute a single ResBlock within the chain: 2x NormActivConv1d + residual add.
///
/// Uses the same conv-stats fusion paths as FusedResBlock (3-dispatch for
/// LeakyRelu, 3-dispatch for Snake).
#[allow(clippy::too_many_arguments)]
fn execute_single_block(
    model: &CompiledModel,
    step_idx: usize,
    block_idx: usize,
    x_tensor: &nn_core::dyn_tensor::DynTensor,
    residual_tensor: &nn_core::dyn_tensor::DynTensor,
    gamma1: &nn_core::dyn_tensor::DynTensor,
    beta1: &nn_core::dyn_tensor::DynTensor,
    gamma2: &nn_core::dyn_tensor::DynTensor,
    beta2: &nn_core::dyn_tensor::DynTensor,
    phase1: &NormActivConv1dParams,
    phase2: &NormActivConv1dParams,
    residual_scale: f32,
    channels1: usize,
    channels2: usize,
) -> Result<nn_core::dyn_tensor::DynTensor> {
    let dtype = x_tensor.dtype();
    let p1_keys = ChainPhaseKeys::new(block_idx, 1);
    let p2_keys = ChainPhaseKeys::new(block_idx, 2);

    let step_weights = &model.def.weight_buffers[step_idx];

    // --- LeakyRelu fast path: 3-dispatch fused stats+norm+conv ---
    if matches!(phase1.activation, NormActivation::LeakyRelu { .. })
        && matches!(phase2.activation, NormActivation::LeakyRelu { .. })
    {
        let slope1 = match phase1.activation {
            NormActivation::LeakyRelu { slope } => slope,
            _ => unreachable!(),
        };
        let slope2 = match phase2.activation {
            NormActivation::LeakyRelu { slope } => slope,
            _ => unreachable!(),
        };

        let p1_conv_w = load_chain_conv_weight(
            step_weights, &p1_keys.conv_weight, channels1, phase1, dtype, step_idx, block_idx,
        )?;
        let p1_conv_b = load_chain_conv_bias(
            step_weights, &p1_keys.conv_bias, phase1.output_channels, dtype, step_idx, block_idx,
        )?;

        let (phase1_output, precomputed_stats) =
            crate::dyn_tensor_metal::native_norm_activ_conv1d_with_output_stats(
                x_tensor,
                gamma1,
                beta1,
                &p1_conv_w,
                &p1_conv_b,
                f64::from(phase1.eps),
                f64::from(slope1),
                phase1.conv_padding,
                phase1.conv_dilation,
                None,
                phase2.eps,
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("chain[{block_idx}] p1 conv_with_stats: {e}"),
                )
            })?;

        let p2_conv_w = load_chain_conv_weight(
            step_weights, &p2_keys.conv_weight, channels2, phase2, dtype, step_idx, block_idx,
        )?;
        let p2_conv_b = load_chain_conv_bias(
            step_weights, &p2_keys.conv_bias, phase2.output_channels, dtype, step_idx, block_idx,
        )?;

        let residual_params = crate::dyn_tensor_metal::ResidualParams {
            residual: residual_tensor,
            scale: residual_scale,
        };
        let output = crate::dyn_tensor_metal::native_norm_activ_conv1d_with_precomputed_stats(
            &phase1_output,
            gamma2,
            beta2,
            &p2_conv_w,
            &p2_conv_b,
            f64::from(slope2),
            phase2.conv_padding,
            phase2.conv_dilation,
            Some(residual_params),
            &precomputed_stats,
        )
        .map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("chain[{block_idx}] p2 conv_precomputed: {e}"),
            )
        })?;

        return Ok(output);
    }

    // --- Snake fast path: 3-dispatch fused ---
    if matches!(phase1.activation, NormActivation::Snake)
        && matches!(phase2.activation, NormActivation::Snake)
    {
        let p1_alpha = load_chain_alpha(
            step_weights, &p1_keys.alpha, channels1, dtype, step_idx, block_idx,
        )?;
        let p1_conv_w = load_chain_conv_weight(
            step_weights, &p1_keys.conv_weight, channels1, phase1, dtype, step_idx, block_idx,
        )?;
        let p1_conv_b = load_chain_conv_bias(
            step_weights, &p1_keys.conv_bias, phase1.output_channels, dtype, step_idx, block_idx,
        )?;

        let (phase1_output, precomputed_stats) =
            crate::dyn_tensor_metal::native_norm_activ_conv1d_snake_with_output_stats(
                x_tensor,
                gamma1,
                beta1,
                &p1_alpha,
                &p1_conv_w,
                &p1_conv_b,
                f64::from(phase1.eps),
                phase1.conv_padding,
                phase1.conv_dilation,
                None,
                phase2.eps,
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("chain[{block_idx}] p1 Snake conv_with_stats: {e}"),
                )
            })?;

        let p2_alpha = load_chain_alpha(
            step_weights, &p2_keys.alpha, channels2, dtype, step_idx, block_idx,
        )?;
        let p2_conv_w = load_chain_conv_weight(
            step_weights, &p2_keys.conv_weight, channels2, phase2, dtype, step_idx, block_idx,
        )?;
        let p2_conv_b = load_chain_conv_bias(
            step_weights, &p2_keys.conv_bias, phase2.output_channels, dtype, step_idx, block_idx,
        )?;

        let residual_params = crate::dyn_tensor_metal::ResidualParams {
            residual: residual_tensor,
            scale: residual_scale,
        };
        let output =
            crate::dyn_tensor_metal::native_norm_activ_conv1d_snake_with_precomputed_stats(
                &phase1_output,
                gamma2,
                beta2,
                &p2_alpha,
                &p2_conv_w,
                &p2_conv_b,
                phase2.conv_padding,
                phase2.conv_dilation,
                Some(residual_params),
                &precomputed_stats,
            )
            .map_err(|e| {
                native_dispatch_err(
                    step_idx,
                    format!("chain[{block_idx}] p2 Snake conv_precomputed: {e}"),
                )
            })?;

        return Ok(output);
    }

    // Mixed activation fallback (should be rare in production chains).
    Err(native_dispatch_err(
        step_idx,
        format!(
            "FusedResBlockChain[{block_idx}]: mixed activation not supported in chain \
             (phase1={:?}, phase2={:?})",
            phase1.activation, phase2.activation
        ),
    ))
}

// --- Weight loading helpers ---

fn load_chain_conv_weight(
    step_weights: &std::collections::HashMap<String, crate::buffer::MetalBuffer>,
    key: &str,
    in_channels: usize,
    params: &NormActivConv1dParams,
    dtype: nn_core::DType,
    step_idx: usize,
    block_idx: usize,
) -> Result<nn_core::dyn_tensor::DynTensor> {
    let shape = vec![params.output_channels, in_channels, params.kernel_size];
    weight_to_dyn(
        step_weights,
        key,
        &shape,
        dtype,
        step_idx,
        &format!("chain[{block_idx}]"),
    )
}

fn load_chain_conv_bias(
    step_weights: &std::collections::HashMap<String, crate::buffer::MetalBuffer>,
    key: &str,
    out_channels: usize,
    dtype: nn_core::DType,
    step_idx: usize,
    block_idx: usize,
) -> Result<nn_core::dyn_tensor::DynTensor> {
    weight_to_dyn(
        step_weights,
        key,
        &[out_channels],
        dtype,
        step_idx,
        &format!("chain[{block_idx}]"),
    )
}

fn load_chain_alpha(
    step_weights: &std::collections::HashMap<String, crate::buffer::MetalBuffer>,
    key: &str,
    channels: usize,
    dtype: nn_core::DType,
    step_idx: usize,
    block_idx: usize,
) -> Result<nn_core::dyn_tensor::DynTensor> {
    weight_to_dyn(
        step_weights,
        key,
        &[1, channels, 1],
        dtype,
        step_idx,
        &format!("chain[{block_idx}]"),
    )
}
