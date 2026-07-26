// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Native operation execution for `CompiledModel`.
//!
//! Extracted from `compiled_model_execute.rs` to keep files under 450 lines.
//! Contains `execute_native_op` and per-variant helpers (LSTM sequence, etc.).
//!
//! Part of #2236 (LSTM sequence fusion in compiled plan).

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result};
use nn_dsl::NativeOpKind;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_slice::GpuSlice;

use super::helpers::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};
use super::{CompiledModel, CompiledModelError};

#[path = "compiled_model_execute_native_fused.rs"]
mod fused;

#[path = "compiled_model_execute_native_simple.rs"]
mod simple;

#[path = "compiled_model_execute_native_norm_linear.rs"]
mod norm_linear;

#[path = "compiled_model_execute_native_add_ln.rs"]
mod add_ln;

#[path = "compiled_model_execute_native_batched.rs"]
mod batched;
pub(super) use batched::clear_projection_temps;

#[path = "compiled_model_execute_native_bilstm.rs"]
mod bilstm;

#[path = "compiled_model_execute_native_moe.rs"]
mod moe;

#[path = "compiled_model_execute_native_conv_transpose1d_activation.rs"]
mod conv_transpose1d_activation;

#[path = "compiled_model_execute_native_norm_activ_conv_transpose1d.rs"]
mod norm_activ_conv_transpose1d;

#[path = "compiled_model_execute_native_fused_norm_conv.rs"]
mod fused_norm_conv;

#[path = "compiled_model_execute_native_resblock_chain.rs"]
mod resblock_chain;

impl CompiledModel {
    /// Execute a `NativeOp` step by dispatching to the appropriate
    /// pre-compiled Metal kernel implementation.
    pub(super) fn execute_native_op(
        &self,
        op: &NativeOpKind,
        step_idx: usize,
        buffers: &[Option<GpuSlice>],
        cache: &PipelineCache,
    ) -> Result<GpuSlice> {
        match op {
            NativeOpKind::LstmSequence {
                hidden_size,
                input_shape,
                h_shape,
                reverse,
            } => execute_native_lstm_sequence(
                self,
                step_idx,
                buffers,
                *hidden_size,
                input_shape,
                h_shape,
                *reverse,
            ),
            NativeOpKind::Cumsum { dim, input_shape } => {
                simple::execute_native_cumsum(self, step_idx, buffers, *dim, input_shape, cache)
            }
            NativeOpKind::InstanceNorm { eps, input_shape } => {
                simple::execute_native_instance_norm(self, step_idx, buffers, *eps, input_shape)
            }
            NativeOpKind::LayerNorm {
                eps,
                input_shape,
                hidden_dim,
            } => simple::execute_native_layer_norm(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *hidden_dim,
            ),
            NativeOpKind::AddLayerNorm {
                eps,
                input_shape,
                hidden_dim,
            } => add_ln::execute_native_add_layer_norm(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *hidden_dim,
            ),
            NativeOpKind::AdainSnake {
                eps,
                input_shape,
                channels,
                residual_gamma,
                ..
            } => fused::execute_native_adain_snake(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *channels,
                *residual_gamma,
            ),
            NativeOpKind::AdainLeakyRelu {
                eps,
                slope,
                input_shape,
                ..
            } => fused::execute_native_adain_leaky_relu(
                self,
                step_idx,
                buffers,
                *eps,
                *slope,
                input_shape,
            ),
            NativeOpKind::AdaLayerNorm {
                eps,
                input_shape,
                hidden_dim,
            } => fused::execute_native_ada_layer_norm(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *hidden_dim,
            ),
            NativeOpKind::FlashAttention {
                scale,
                causal,
                q_shape,
                k_shape,
                output_shape: _,
                input_layout,
            } => fused::execute_native_flash_attention(
                self,
                step_idx,
                buffers,
                *scale,
                *causal,
                q_shape,
                k_shape,
                *input_layout,
            ),
            NativeOpKind::MaxPool1d {
                kernel_size,
                stride,
                padding,
                input_shape,
            } => simple::execute_native_max_pool1d(
                self,
                step_idx,
                buffers,
                *kernel_size,
                *stride,
                *padding,
                input_shape,
            ),
            NativeOpKind::ConstantWeight { name, shape } => {
                simple::execute_native_constant_weight(self, step_idx, name, shape)
            }
            NativeOpKind::NormActivConv1d {
                activation,
                eps,
                conv_dilation,
                conv_padding,
                input_shape,
                output_channels,
                kernel_size,
                ..
            } => fused::execute_native_norm_activ_conv1d(
                self,
                step_idx,
                buffers,
                activation,
                *eps,
                *conv_dilation,
                *conv_padding,
                input_shape,
                *output_channels,
                *kernel_size,
            ),
            NativeOpKind::FusedResBlock {
                phase1,
                phase2,
                input_steps,
                residual_scale,
                style_proj,
                shortcut_step,
                pool_step,
                style_batch_offset,
            } => fused::execute_native_fused_resblock(
                self,
                step_idx,
                buffers,
                phase1,
                phase2,
                input_steps,
                *residual_scale,
                style_proj.as_ref(),
                *shortcut_step,
                *pool_step,
                style_batch_offset.as_ref(),
                cache,
            ),
            NativeOpKind::BatchedStyleProjection {
                style_dim,
                total_out,
                style_step,
                ..
            } => fused::execute_native_batched_style_projection(
                self,
                step_idx,
                buffers,
                *style_dim,
                *total_out,
                *style_step,
            ),
            NativeOpKind::LinearActivation {
                activation,
                in_features,
                out_features,
                has_bias,
                input_shape,
            } => simple::execute_native_linear_activation(
                self,
                step_idx,
                buffers,
                activation,
                *in_features,
                *out_features,
                *has_bias,
                input_shape,
                cache,
            ),
            NativeOpKind::NormLinear {
                norm_kind,
                eps,
                input_shape,
                hidden_dim,
                out_features,
                has_bias,
            } => norm_linear::execute_native_norm_linear(
                self,
                step_idx,
                buffers,
                *norm_kind,
                *eps,
                input_shape,
                *hidden_dim,
                *out_features,
                *has_bias,
                cache,
            ),
            NativeOpKind::AddNormLinear {
                eps,
                input_shape,
                hidden_dim,
                out_features,
                has_bias,
            } => norm_linear::execute_native_add_norm_linear(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *hidden_dim,
                *out_features,
                *has_bias,
                cache,
            ),
            NativeOpKind::BatchedLinearProjection {
                in_features,
                total_out_features,
                projection_sizes,
                has_bias,
                input_shape,
            } => batched::execute_native_batched_linear_projection(
                self,
                step_idx,
                buffers,
                *in_features,
                *total_out_features,
                projection_sizes,
                *has_bias,
                input_shape,
                cache,
            ),
            NativeOpKind::ProjectionSlice {
                source_step,
                dim,
                start,
                length,
                output_shape,
            } => batched::execute_native_projection_slice(
                step_idx,
                *source_step,
                *dim,
                *start,
                *length,
                output_shape,
            ),
            NativeOpKind::Conv1dGemm {
                input_shape,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                has_bias,
            } => simple::execute_native_conv1d_gemm(
                self,
                step_idx,
                buffers,
                input_shape,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *groups,
                *has_bias,
            ),
            NativeOpKind::ChannelsFirstLayerNorm {
                eps,
                input_shape,
                channels,
                leaky_relu_slope,
            } => simple::execute_native_channels_first_layer_norm(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *channels,
                *leaky_relu_slope,
            ),
            NativeOpKind::Int8Gemm {
                in_features,
                out_features,
                has_bias,
                input_shape,
            } => simple::execute_native_int8_gemm(
                self,
                step_idx,
                buffers,
                *in_features,
                *out_features,
                *has_bias,
                input_shape,
                cache,
            ),
            NativeOpKind::SiluMul { input_shape } => {
                // Try fused single-dispatch path first (no DynTensor bridge).
                // Falls back to 2-dispatch bridge path for unsupported dtypes.
                // Part of #3537.
                let scalar_type = self.step_scalar_type(step_idx);
                let direct = crate::native_op_direct::SiluMulDirect;
                if crate::native_op_direct::DirectDispatch::supports_scalar_type(
                    &direct,
                    scalar_type,
                ) {
                    let gate_slice = self.resolve_input_slice(step_idx, 0, buffers)?;
                    let up_slice = self.resolve_input_slice(step_idx, 1, buffers)?;
                    let num_elements: usize = input_shape.iter().product();
                    if num_elements > 0 {
                        let out_bytes = crate::native_op_direct::DirectDispatch::output_bytes(
                            &direct,
                            num_elements,
                            scalar_type,
                        );
                        let (out_buf, out_offset) =
                            crate::arena::arena_alloc_or_create(cache.context(), out_bytes)
                                .map_err(|e| {
                                    CompiledModelError::DispatchFailed {
                                        step_idx,
                                        reason: format!("SiluMul direct alloc: {e}"),
                                    }
                                })?;
                        let output = GpuSlice::new(out_buf.alias(), out_offset);
                        crate::native_op_direct::DirectDispatch::dispatch_direct(
                            &direct,
                            &[&gate_slice, &up_slice],
                            &output,
                            num_elements,
                            scalar_type,
                            cache,
                        )
                        .map_err(|e| CompiledModelError::DispatchFailed {
                            step_idx,
                            reason: format!("SiluMul direct: {e}"),
                        })?;
                        return Ok(GpuSlice::from_ref(&out_buf, out_offset));
                    }
                }
                simple::execute_native_silu_mul(self, step_idx, buffers, input_shape)
            }
            NativeOpKind::RotaryEmbedding {
                head_dim,
                input_shape,
            } => simple::execute_native_rope(
                self,
                step_idx,
                buffers,
                input_shape,
                *head_dim,
            ),
            NativeOpKind::BiLstmCat {
                hidden_size,
                input_shape,
                h_shape,
                fwd_lstm_step,
                rev_lstm_step,
            } => bilstm::execute_native_bilstm_cat(
                self,
                step_idx,
                buffers,
                *hidden_size,
                input_shape,
                h_shape,
                *fwd_lstm_step,
                *rev_lstm_step,
            ),
            NativeOpKind::MoeGating {
                num_experts,
                top_k,
                input_shape,
            } => moe::execute_native_moe_gating(
                self,
                step_idx,
                buffers,
                *num_experts,
                *top_k,
                input_shape,
            ),
            NativeOpKind::FusedAdainSnake {
                eps,
                input_shape,
                channels,
                ..
            } => fused::execute_native_fused_adain_snake(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *channels,
                cache,
            ),
            NativeOpKind::FusedInstanceNormMulAdd {
                eps,
                input_shape,
                channels,
                ..
            } => fused::execute_native_fused_instance_norm_mul_add(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *channels,
            ),
            NativeOpKind::FusedSnakeInstanceNorm {
                eps,
                input_shape,
                channels,
            } => fused::execute_native_fused_snake_instance_norm(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *channels,
                cache,
            ),
            NativeOpKind::FusedUpsampleConv1d {
                upsample_factor,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                input_shape,
            } => fused::execute_native_fused_upsample_conv1d(
                self,
                step_idx,
                buffers,
                *upsample_factor,
                *in_channels,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                input_shape,
                cache,
            ),
            NativeOpKind::FusedLayerNormLinear {
                eps,
                input_shape,
                hidden_dim,
                out_features,
                has_bias,
            } => norm_linear::execute_native_norm_linear(
                self,
                step_idx,
                buffers,
                nn_dsl::trace_compile::FusedNormKind::LayerNorm,
                *eps,
                input_shape,
                *hidden_dim,
                *out_features,
                *has_bias,
                cache,
            ),
            NativeOpKind::BatchNorm2d {
                eps,
                num_channels,
                input_shape,
                has_weight,
                has_bias,
            } => simple::execute_native_batch_norm_2d(
                self,
                step_idx,
                buffers,
                *eps,
                *num_channels,
                input_shape,
                *has_weight,
                *has_bias,
            ),
            NativeOpKind::FusedConv1dActivation {
                activation,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                has_bias,
                input_shape,
                pre_activation,
            } => fused::execute_native_conv1d_activation(
                self,
                step_idx,
                buffers,
                activation,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *groups,
                *has_bias,
                input_shape,
                *pre_activation,
                cache,
            ),
            NativeOpKind::FusedMulAdd { input_shape } => {
                execute_fused_elementwise_direct(
                    self,
                    step_idx,
                    buffers,
                    input_shape,
                    3, // a, b, c
                    &crate::native_op_direct::FusedMulAddDirect,
                    "FusedMulAdd",
                    cache,
                )
            }
            NativeOpKind::FusedSiGLU { input_shape } => {
                execute_fused_elementwise_direct(
                    self,
                    step_idx,
                    buffers,
                    input_shape,
                    1, // x
                    &crate::native_op_direct::FusedSiGLUDirect,
                    "FusedSiGLU",
                    cache,
                )
            }
            NativeOpKind::FusedGeGLU { input_shape } => {
                execute_fused_elementwise_direct(
                    self,
                    step_idx,
                    buffers,
                    input_shape,
                    2, // gate, up
                    &crate::native_op_direct::FusedGeGLUDirect,
                    "FusedGeGLU",
                    cache,
                )
            }
            NativeOpKind::FusedConv1dSnakeNorm {
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                has_bias,
                eps,
                input_shape,
            } => fused::execute_native_conv1d_snake_norm(
                self,
                step_idx,
                buffers,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *groups,
                *has_bias,
                *eps,
                input_shape,
                cache,
            ),
            NativeOpKind::FusedConv1dSnakeNormResBlock {
                phase1_out_channels,
                phase1_kernel_size,
                phase1_padding,
                phase1_dilation,
                phase1_has_bias,
                phase2_out_channels,
                phase2_kernel_size,
                phase2_padding,
                phase2_dilation,
                phase2_has_bias,
                eps,
                residual_scale,
                input_shape,
                x_step,
            } => fused::execute_native_conv1d_snake_norm_resblock(
                self,
                step_idx,
                buffers,
                *phase1_out_channels,
                *phase1_kernel_size,
                *phase1_padding,
                *phase1_dilation,
                *phase1_has_bias,
                *phase2_out_channels,
                *phase2_kernel_size,
                *phase2_padding,
                *phase2_dilation,
                *phase2_has_bias,
                *eps,
                *residual_scale,
                input_shape,
                *x_step,
                cache,
            ),
            NativeOpKind::FusedAddInstanceNormConv1x1 {
                eps,
                input_shape,
                in_channels,
                out_channels,
                has_bias,
            } => fused::execute_native_fused_add_instance_norm_conv1x1(
                self,
                step_idx,
                buffers,
                *eps,
                input_shape,
                *in_channels,
                *out_channels,
                *has_bias,
                cache,
            ),
            NativeOpKind::FusedConvTranspose1dActivation {
                activation,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
                has_bias,
                input_shape,
            } => conv_transpose1d_activation::execute_native_conv_transpose1d_activation(
                self,
                step_idx,
                buffers,
                activation,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *groups,
                *output_padding,
                *has_bias,
                input_shape,
                cache,
            ),
            NativeOpKind::NormActivConvTranspose1d {
                activation,
                eps,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
                output_channels,
                input_shape,
                ..
            } => norm_activ_conv_transpose1d::execute_native_norm_activ_conv_transpose1d(
                self,
                step_idx,
                buffers,
                activation,
                *eps,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *groups,
                *output_padding,
                *output_channels,
                input_shape,
                cache,
            ),
            NativeOpKind::FusedLinearLayerNorm {
                in_features,
                out_features,
                has_bias,
                eps,
                input_shape,
            } => fused_norm_conv::execute_fused_linear_layer_norm(
                self,
                step_idx,
                buffers,
                *in_features,
                *out_features,
                *has_bias,
                *eps,
                input_shape,
                cache,
            ),
            NativeOpKind::FusedInstanceNormConv1d {
                eps,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                has_bias,
                input_shape,
                ..
            } => fused_norm_conv::execute_fused_instance_norm_conv1d(
                self,
                step_idx,
                buffers,
                *eps,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *has_bias,
                input_shape,
                cache,
            ),
            NativeOpKind::FusedConv1dInstanceNorm {
                eps,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                has_bias,
                input_shape,
                ..
            } => fused_norm_conv::execute_fused_conv1d_instance_norm(
                self,
                step_idx,
                buffers,
                *eps,
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *has_bias,
                input_shape,
                cache,
            ),
            NativeOpKind::FusedResBlockChain {
                blocks,
                input_steps,
                style_batch_offsets,
                first_shortcut_step,
            } => resblock_chain::execute_native_fused_resblock_chain(
                self,
                step_idx,
                buffers,
                blocks,
                input_steps,
                style_batch_offsets,
                *first_shortcut_step,
                cache,
            ),
            _ => Err(CompiledModelError::DispatchFailed {
                step_idx,
                reason: "unsupported NativeOp variant".into(),
            }
            .into()),
        }
    }
}

/// Execute a fused elementwise NativeOp via the DirectDispatch path.
///
/// Shared helper for FusedMulAdd, FusedSiGLU, FusedGeGLU, and any future
/// simple elementwise fusions that use the DirectDispatch trait.
///
/// Resolves `num_inputs` graph inputs, allocates an output buffer, and
/// dispatches via the `DirectDispatch` implementation. Falls through to
/// the catch-all error if the scalar type is unsupported.
///
/// Part of #4252.
fn execute_fused_elementwise_direct(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    input_shape: &[usize],
    num_inputs: usize,
    direct: &dyn crate::native_op_direct::DirectDispatch,
    op_name: &str,
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let scalar_type = model.step_scalar_type(step_idx);

    if !crate::native_op_direct::DirectDispatch::supports_scalar_type(direct, scalar_type) {
        return Err(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("{op_name}: unsupported scalar type {scalar_type:?}"),
        }
        .into());
    }

    let num_elements: usize = input_shape.iter().product();
    if num_elements == 0 {
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(cache.context(), 0).map_err(|e| {
                CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!("{op_name} alloc (zero): {e}"),
                }
            })?;
        return Ok(GpuSlice::from_ref(&out_buf, out_offset));
    }

    // Resolve input slices.
    let mut input_slices = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        input_slices.push(model.resolve_input_slice(step_idx, i, buffers)?);
    }
    let input_refs: Vec<&GpuSlice> = input_slices.iter().collect();

    // Allocate output buffer.
    let out_bytes =
        crate::native_op_direct::DirectDispatch::output_bytes(direct, num_elements, scalar_type);
    let (out_buf, out_offset) =
        crate::arena::arena_alloc_or_create(cache.context(), out_bytes).map_err(|e| {
            CompiledModelError::DispatchFailed {
                step_idx,
                reason: format!("{op_name} alloc: {e}"),
            }
        })?;
    let output = GpuSlice::from_ref(&out_buf, out_offset);

    crate::native_op_direct::DirectDispatch::dispatch_direct(
        direct,
        &input_refs,
        &output,
        num_elements,
        scalar_type,
        cache,
    )
    .map_err(|e| {
        CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("{op_name} direct: {e}"),
        }
    })?;

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Execute a `NativeOpKind::LstmSequence` step.
///
/// Routes between two GPU execution paths:
///
/// **Precomputed path** (#2981, restored in #3491): When `weight_ih_t` is
/// available in step weights and GEMM alignment conditions are met, the input
/// projection `X @ W_ih.T + bias` is computed via a parallel simdgroup matmul
/// across all timesteps, then the lighter precomputed recurrence kernel handles
/// only the sequential `w_hh @ h` loop. 1.85x GPU speedup at D512.
///
/// **Fused path** (fallback): The original single-dispatch kernel that performs
/// both `w_ih @ x` and `w_hh @ h` per timestep with Kahan compensation.
///
/// Both paths discard `h_n` and `c_n` — the compiled model only tracks the
/// primary `[S, B, H]` output. See #2236.
fn execute_native_lstm_sequence(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    hidden_size: usize,
    input_shape: &[usize],
    h_shape: &[usize],
    reverse: bool,
) -> Result<GpuSlice> {
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let dtype = model.step_dtype(step_idx);
    let step_weights = &model.def.weight_buffers[step_idx];

    // Route: precomputed GEMM path if weight_ih_t available.
    // K (input_size) and N (4*hidden_size) must be 8-aligned for efficient
    // simdgroup tile loads. M (seq_len*batch) is not required to be aligned:
    // the simdgroup matmul kernel handles edge tiles with bounds-checked
    // loads (zeroed) and writes. This enables precomputed LSTM for all
    // production sequence lengths, not just multiples of 8.
    let has_weight_ih_t = step_weights.contains_key("weight_ih_t");
    let seq_len = input_shape[0];
    let batch_size = input_shape[1];
    let input_size = input_shape[2];
    let n = 4 * hidden_size;

    if has_weight_ih_t && input_size.is_multiple_of(8) && n.is_multiple_of(8) {
        return execute_precomputed_lstm(
            step_weights, &input_slice, dtype, step_idx, seq_len, batch_size,
            input_size, hidden_size, reverse,
        );
    }

    // Fused path: original single-dispatch kernel.
    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;

    let w_ih = weight_to_dyn(
        step_weights, "weight_ih", &[n, input_size], dtype, step_idx, "NativeOp LSTM",
    )?;
    let w_hh = weight_to_dyn(
        step_weights, "weight_hh", &[n, hidden_size], dtype, step_idx, "NativeOp LSTM",
    )?;
    let h0 = weight_to_dyn(
        step_weights, "h0", h_shape, dtype, step_idx, "NativeOp LSTM",
    )?;
    let c0 = weight_to_dyn(
        step_weights, "c0", h_shape, dtype, step_idx, "NativeOp LSTM",
    )?;
    let bias = load_combined_bias(step_weights, hidden_size, dtype, step_idx)?;

    // Skip weight validation (#2795): pre-uploaded weights are immutable.
    let dispatch_fn = if reverse {
        crate::dyn_tensor_metal::native_lstm_sequence_reverse
    } else {
        crate::dyn_tensor_metal::native_lstm_sequence
    };
    let (output, _h_n, _c_n) = dispatch_fn(
        &input_tensor, &w_ih, &w_hh, bias.as_ref(), &h0, &c0, hidden_size,
        true,
    )
    .ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!(
                "NativeOp LSTM: gpu_lstm_sequence{} returned None \
                 (hidden_size={hidden_size}, max=512)",
                if reverse { "_reverse" } else { "" }
            ),
        )
    })??;

    dyn_to_slice(&output, step_idx, "NativeOp LSTM")
}

/// Combine `bias_ih + bias_hh` or use single `bias` from step weights.
fn load_combined_bias(
    step_weights: &HashMap<String, MetalBuffer>,
    hidden_size: usize,
    dtype: DType,
    step_idx: usize,
) -> Result<Option<DynTensor>> {
    let has_bih = step_weights.contains_key("bias_ih");
    let has_bhh = step_weights.contains_key("bias_hh");
    let has_single = step_weights.contains_key("bias");
    if has_bih && has_bhh {
        let bih = weight_to_dyn(
            step_weights, "bias_ih", &[4 * hidden_size], dtype, step_idx, "NativeOp LSTM",
        )?;
        let bhh = weight_to_dyn(
            step_weights, "bias_hh", &[4 * hidden_size], dtype, step_idx, "NativeOp LSTM",
        )?;
        Ok(Some(bih.add(&bhh)?))
    } else if has_single {
        Ok(Some(weight_to_dyn(
            step_weights, "bias", &[4 * hidden_size], dtype, step_idx, "NativeOp LSTM",
        )?))
    } else {
        Ok(None)
    }
}

/// Precomputed LSTM path: simdgroup matmul for input projection + lighter
/// recurrence kernel. Part of #2981, restored in #3491.
///
/// Phase 1: `input_proj = input_2d @ weight_ih_t + bias` — parallel GEMM
/// across all timesteps via simdgroup matmul (encoded into lazy batch).
///
/// Phase 2: precomputed recurrence kernel — only does sequential `w_hh @ h`
/// loop, reading pre-projected gates from `input_proj`.
fn execute_precomputed_lstm(
    step_weights: &HashMap<String, MetalBuffer>,
    input_slice: &GpuSlice,
    dtype: DType,
    step_idx: usize,
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
    reverse: bool,
) -> Result<GpuSlice> {
    let m = seq_len * batch_size;
    let n = 4 * hidden_size;

    // Phase 1: input_proj = input_2d @ weight_ih_t
    // Input is [S, B, input_size] — treat as [S*B, input_size] (contiguous, no copy).
    let input_data = MetalTensorData::view(input_slice.buffer().alias(), input_slice.byte_offset());

    let weight_ih_t_buf = step_weights.get("weight_ih_t").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LSTM precomputed: missing weight 'weight_ih_t'".into(),
        )
    })?;
    let weight_ih_t_data = MetalTensorData::new(weight_ih_t_buf.alias());

    // Simdgroup matmul: [S*B, input_size] @ [input_size, 4*H] → [S*B, 4*H]
    let proj_data = crate::dyn_tensor_metal::encode_simdgroup_matmul_into_batch(
        &input_data,
        &weight_ih_t_data,
        m,
        input_size,
        n,
    )?;

    // Add bias if present: proj + bias (broadcast [4*H] → [S*B, 4*H]).
    let bias = load_combined_bias(step_weights, hidden_size, dtype, step_idx)?;
    let proj_tensor = DynTensor::from_gpu_storage(
        vec![m, n],
        dtype,
        Arc::new(proj_data),
        Device::metal(),
    )?;
    let proj_with_bias = match bias {
        Some(ref b) => proj_tensor.add(b)?,
        None => proj_tensor,
    };

    // Reshape to [S, B, 4*H] for the precomputed kernel (contiguous, no copy).
    let proj_3d = proj_with_bias.reshape([seq_len, batch_size, n])?;
    let proj_3d_data = proj_3d.gpu_data::<MetalTensorData>().map_err(|_| {
        native_dispatch_err(
            step_idx,
            "NativeOp LSTM precomputed: proj not GPU tensor".into(),
        )
    })?;

    // Phase 2: precomputed LSTM recurrence.
    let w_hh_buf = step_weights.get("weight_hh").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LSTM precomputed: missing 'weight_hh'".into(),
        )
    })?;
    let w_hh_data = MetalTensorData::new(w_hh_buf.alias());

    let h0_buf = step_weights.get("h0").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LSTM precomputed: missing 'h0'".into(),
        )
    })?;
    let h0_data = MetalTensorData::new(h0_buf.alias());

    let c0_buf = step_weights.get("c0").ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            "NativeOp LSTM precomputed: missing 'c0'".into(),
        )
    })?;
    let c0_data = MetalTensorData::new(c0_buf.alias());

    let (output, _h_n, _c_n) = crate::dyn_tensor_metal::dispatch_lstm_precomputed(
        proj_3d_data,
        &w_hh_data,
        &h0_data,
        &c0_data,
        seq_len,
        batch_size,
        hidden_size,
        reverse,
        false, // mixed: LSTM weights stay F32 (builder line 188). TODO(#3491): F16 w_hh upload
    )?;

    dyn_to_slice(&output, step_idx, "NativeOp LSTM precomputed")
}

// BiLstmCat execution extracted to compiled_model_execute_native_bilstm.rs
