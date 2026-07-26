// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Native op bridge functions for the compiled model executor.
//!
//! Each function here delegates to a `MetalDynBackend` method, providing
//! a `pub(crate)` entry point that `compiled_model_execute_native.rs` and
//! related modules can call without importing backend internals.
//!
//! Extracted from `dyn_tensor_metal.rs` to keep that file under the
//! 450-line limit.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;

use super::MetalDynBackend;

/// Call the fused LSTM sequence GPU kernel from the compiled model executor.
///
/// This is a `pub(crate)` bridge that allows `compiled_model_execute.rs` to
/// dispatch `NativeOpKind::LstmSequence` steps to the existing
/// `MetalDynBackend::gpu_lstm_sequence()` implementation.
///
/// When `skip_weight_validation` is `true`, the `any_non_finite()` checks on
/// weights and initial states are skipped. This eliminates 5 GPU flushes per
/// LSTM NativeOp in the compiled model path where weights are pre-uploaded
/// buffers that never change between forward passes (#2795).
///
/// Returns `(output, h_n, c_n)` as DynTensors.
pub(crate) fn native_lstm_sequence(
    input: &DynTensor,
    w_ih: &DynTensor,
    w_hh: &DynTensor,
    bias: Option<&DynTensor>,
    h0: &DynTensor,
    c0: &DynTensor,
    hidden_size: usize,
    skip_weight_validation: bool,
) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
    MetalDynBackend::gpu_lstm_sequence(
        input,
        w_ih,
        w_hh,
        bias,
        h0,
        c0,
        hidden_size,
        skip_weight_validation,
    )
}

/// Reverse-direction LSTM sequence for BiLSTM backward pass (#1815).
///
/// Same as [`native_lstm_sequence`] but processes timesteps in reverse order.
/// Eliminates the need for external `flip(dim=0)` dispatches around backward
/// LSTM in BiLSTM, saving ~192 Metal dispatches in Kokoro (45% of total).
///
/// Returns `(output, h_n, c_n)` where output[0] corresponds to input[seq_len-1].
pub(crate) fn native_lstm_sequence_reverse(
    input: &DynTensor,
    w_ih: &DynTensor,
    w_hh: &DynTensor,
    bias: Option<&DynTensor>,
    h0: &DynTensor,
    c0: &DynTensor,
    hidden_size: usize,
    skip_weight_validation: bool,
) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
    MetalDynBackend::gpu_lstm_sequence_reverse(
        input,
        w_ih,
        w_hh,
        bias,
        h0,
        c0,
        hidden_size,
        skip_weight_validation,
    )
}

/// Bridge for `NativeOpKind::Cumsum` in `compiled_model_execute.rs`.
///
/// Delegates to `MetalDynBackend::gpu_cumsum()` — Blelloch parallel prefix
/// sum (single-pass for axis <= 256, three-pass for axis <= 65536).
#[allow(dead_code)] // Cumsum native op wiring pending
pub(crate) fn native_cumsum(x: &DynTensor, dim: usize) -> Result<DynTensor> {
    MetalDynBackend::gpu_cumsum(x, dim)
}

/// Bridge for `NativeOpKind::InstanceNorm` in `compiled_model_execute.rs`.
///
/// Delegates to `MetalDynBackend::gpu_instance_norm_fused()` — single Metal
/// dispatch using threadgroup parallel reduction. Replaces the 7-dispatch IR
/// decomposition path. Part of #2472.
pub(crate) fn native_instance_norm(x: &DynTensor, eps: f64) -> Result<DynTensor> {
    MetalDynBackend::gpu_instance_norm_fused(x, eps)
}

/// Kahan-compensated InstanceNorm for `PrecisionTier::Strict` contexts.
///
/// Uses the decomposed 7-dispatch path with `PrecisionContract::Strict`,
/// which enables Kahan-compensated sum/mean reductions. Slower than the
/// fused single-dispatch kernel but eliminates the float32 rounding drift
/// that causes amplitude regressions in chained normalization layers.
/// Part of #2528.
pub(crate) fn native_instance_norm_precise(x: &DynTensor, eps: f64) -> Result<DynTensor> {
    MetalDynBackend::gpu_instance_norm(x, eps)
}

/// Bridge for `NativeOpKind::AdainSnake` in `compiled_model_execute_native.rs`.
///
/// Fused InstanceNorm + affine(gamma, beta) + Snake(alpha) in a single Metal
/// dispatch. Replaces ~20 dispatches per AdaIN+Snake call. Part of #2472.
///
/// `residual_gamma`: if true, `(1+g)*normed+b`; if false, `g*normed+b`. Part of #3257.
pub(crate) fn native_adain_snake(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    alpha: &DynTensor,
    eps: f64,
    residual_gamma: bool,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_adain_snake_fused(x, gamma, beta, alpha, eps, residual_gamma)
}

/// Kahan-precise AdaIN+Snake for `PrecisionTier::Strict` (#2546).
///
/// Routes to the IR-decomposed path (`gpu_adain_snake`) which builds a
/// `TensorBlockBuilder` graph with `PrecisionContract::bootstrap(Strict)`.
/// The `emit_reduce_kernel()` codegen uses Kahan-compensated summation for
/// the InstanceNorm mean/var reductions. Slower than the fused kernel but
/// eliminates f32 accumulation drift in chained InstanceNorm layers.
pub(crate) fn native_adain_snake_precise(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    alpha: &DynTensor,
    eps: f64,
    residual_gamma: bool,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_adain_snake(x, gamma, beta, alpha, eps, residual_gamma)
}

/// Bridge for `NativeOpKind::AdainLeakyRelu` in `compiled_model_execute_native.rs`.
///
/// Fused InstanceNorm + affine(gamma, beta) + LeakyRelu(slope) in a single
/// Metal dispatch. Replaces ~20 dispatches per AdaIN+LeakyRelu call. Part of #2472.
pub(crate) fn native_adain_leaky_relu(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    eps: f64,
    slope: f64,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_adain_leaky_relu_fused(x, gamma, beta, eps, slope)
}

/// Bridge for `NativeOpKind::AdaLayerNorm` in `compiled_model_execute_native.rs`.
///
/// Fused LayerNorm + adaptive affine `(1+gamma)*normed+beta` in a single Metal
/// dispatch. Replaces ~6-7 dispatches per AdaLayerNorm call. Part of #2482.
pub(crate) fn native_ada_layer_norm(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    norm_weight: &DynTensor,
    norm_bias: &DynTensor,
    eps: f64,
    time_steps: usize,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_ada_layer_norm_fused(
        x,
        gamma,
        beta,
        norm_weight,
        norm_bias,
        eps,
        time_steps,
    )
}

/// Bridge for `NativeOpKind::LayerNorm` in `compiled_model_execute_native.rs`.
///
/// GPU LayerNorm: `weight * ((x - mean) * inv_std) + bias`.
/// Uses the fused single-dispatch kernel (#2937), replacing the decomposed
/// 14-dispatch path. Saves ~364 dispatches across a Kokoro forward pass.
pub(crate) fn native_layer_norm(
    x: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_layer_norm_fused(x, weight, bias, eps)
}

/// Bridge for `NativeOpKind::ChannelsFirstLayerNorm` in `compiled_model_execute_native.rs`.
///
/// Channels-first LayerNorm: normalizes over dim 1 (channel dimension) of
/// `[B, C, T]`. Eliminates two Transpose(1,2) dispatches. Part of #3457.
/// When `leaky_relu_slope` is Some, the kernel fuses LeakyReLU after
/// normalization in a single dispatch.
pub(crate) fn native_channels_first_layer_norm_with_activation(
    x: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
    leaky_relu_slope: Option<f32>,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_channels_first_layer_norm_fused(x, weight, bias, eps, leaky_relu_slope)
}

/// Bridge for `NativeOpKind::AddLayerNorm` in `compiled_model_execute_native.rs`.
///
/// Fused residual-add + LayerNorm: `LN(a + b, weight, bias)`.
/// 1 dispatch instead of 2. Part of #1815 Tier 5 D2.
pub(crate) fn native_add_layer_norm(
    a: &DynTensor,
    b: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_add_layer_norm_fused(a, b, weight, bias, eps)
}

/// Bridge for `NativeOpKind::FlashAttention` in `compiled_model_execute_native.rs`.
///
/// Fused Flash Attention: `softmax(Q @ K^T * scale) @ V` in a single Metal
/// dispatch using online softmax. Avoids materializing O(S_q × S_kv).
/// Supports GQA and optional causal masking. Part of #2434.
pub(crate) fn native_flash_attention(
    q: &DynTensor,
    k: &DynTensor,
    v: &DynTensor,
    scale: f64,
    causal: bool,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_flash_attention(q, k, v, scale, causal)
}

/// Bridge for `NativeOpKind::FlashAttention` with SeqFirst layout.
///
/// Fused Flash Attention in `[B, S, H, D]` layout — eliminates Transpose
/// dispatches around attention. Part of #1815 Tier 5 D1.
pub(crate) fn native_flash_attention_seq_first(
    q: &DynTensor,
    k: &DynTensor,
    v: &DynTensor,
    scale: f64,
    causal: bool,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_flash_attention_seq_first(q, k, v, scale, causal)
}

/// Bridge for `NativeOpKind::MaxPool1d` in `compiled_model_execute_native.rs`.
///
/// Delegates to `DynTensor::max_pool1d()` which handles both CPU and GPU
/// tensors. For GPU tensors, the current implementation does a CPU roundtrip.
/// Part of #2295 (PyanNet speaker segmentation).
pub(crate) fn native_max_pool1d(
    x: &DynTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<DynTensor> {
    x.max_pool1d(kernel_size, stride, padding)
}

/// Bridge for fused NormActivConv1d GPU kernel (#2780).
///
/// Fuses InstanceNorm + LeakyReLU + Conv1d into two dispatches:
/// stats kernel + fused conv accumulation. Optionally folds residual add.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_norm_activ_conv1d(
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
) -> Result<DynTensor> {
    MetalDynBackend::gpu_norm_activ_conv1d(
        x, gamma, beta, weight, bias, eps, slope, padding, dilation, residual,
    )
}

/// Bridge for fused NormActivConv1d with Snake activation (#2780).
///
/// Same as [`native_norm_activ_conv1d`] but uses per-channel `alpha` device
/// buffer instead of scalar `slope`. Optionally folds residual add.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_norm_activ_conv1d_snake(
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
) -> Result<DynTensor> {
    MetalDynBackend::gpu_norm_activ_conv1d_snake(
        x, gamma, beta, alpha, weight, bias, eps, padding, dilation, residual,
    )
}

/// Bridge for fused NormActivConv1d + LeakyRelu with output stats epilogue (#1815 Tier 2).
///
/// 2 Metal dispatches (stats + conv_with_stats). Returns output + precomputed
/// stats for the next FusedResBlock phase, saving 1 dispatch per block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_norm_activ_conv1d_with_output_stats(
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
) -> Result<(DynTensor, super::PrecomputedStats)> {
    MetalDynBackend::gpu_norm_activ_conv1d_with_output_stats(
        x,
        gamma,
        beta,
        weight,
        bias,
        eps,
        slope,
        padding,
        dilation,
        residual,
        next_phase_eps,
    )
}

/// Bridge for fused NormActivConv1d + Snake with output stats epilogue (#1815 Tier 2).
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_norm_activ_conv1d_snake_with_output_stats(
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
) -> Result<(DynTensor, super::PrecomputedStats)> {
    MetalDynBackend::gpu_norm_activ_conv1d_snake_with_output_stats(
        x,
        gamma,
        beta,
        alpha,
        weight,
        bias,
        eps,
        padding,
        dilation,
        residual,
        next_phase_eps,
    )
}

/// Bridge for conv with precomputed stats (skip stats dispatch). LeakyRelu. (#1815 Tier 2).
///
/// 1 Metal dispatch: conv only, using stats from phase 1's epilogue.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_norm_activ_conv1d_with_precomputed_stats(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    slope: f64,
    padding: usize,
    dilation: usize,
    residual: Option<super::norm_conv_fused::ResidualParams<'_>>,
    precomputed: &super::PrecomputedStats,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_norm_activ_conv1d_with_precomputed_stats(
        x,
        gamma,
        beta,
        weight,
        bias,
        slope,
        padding,
        dilation,
        residual,
        precomputed,
    )
}

/// Bridge for conv with precomputed stats (skip stats dispatch). Snake. (#1815 Tier 2).
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_norm_activ_conv1d_snake_with_precomputed_stats(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    alpha: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    padding: usize,
    dilation: usize,
    residual: Option<super::norm_conv_fused::ResidualParams<'_>>,
    precomputed: &super::PrecomputedStats,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_norm_activ_conv1d_snake_with_precomputed_stats(
        x,
        gamma,
        beta,
        alpha,
        weight,
        bias,
        padding,
        dilation,
        residual,
        precomputed,
    )
}

/// Dispatch prefix-sum without flushing or reading back (#2911 Phase 2).
///
/// Returns the GPU-resident offsets buffer. Caller must synchronize before
/// calling [`read_prefix_sum_total`].
pub(crate) fn dispatch_prefix_sum_only(
    counts: &DynTensor,
    dim_size: usize,
) -> Result<crate::MetalBuffer> {
    MetalDynBackend::dispatch_prefix_sum_only(counts, dim_size)
}

/// Read the total repeats from a completed prefix-sum offsets buffer.
pub(crate) fn read_prefix_sum_total(
    offsets_buf: &crate::MetalBuffer,
    dim_size: usize,
) -> Result<usize> {
    MetalDynBackend::read_prefix_sum_total(offsets_buf, dim_size)
}

/// Bridge for GPU scatter using pre-computed offsets (#2911).
///
/// Scatter `x` along `dim` using an offsets buffer from
/// [`dispatch_prefix_sum_only`]. No additional flushes needed.
pub(crate) fn gpu_scatter_with_offsets(
    x: &DynTensor,
    dim: usize,
    offsets_buf: &crate::MetalBuffer,
    dim_size: usize,
    total_repeats: usize,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_scatter_with_offsets(x, dim, offsets_buf, dim_size, total_repeats)
}

/// Maximum counts length for GPU-native prefix sum (single threadgroup).
///
/// Re-exported from `dyn_tensor_metal_repeat_interleave_gpu.rs` for use
/// by `step_regulate` fallback guard (#2911).
pub(crate) const MAX_GPU_PREFIX_SUM: usize = super::repeat_interleave_gpu::MAX_GPU_PREFIX_SUM;

/// Fused polar-to-rectangular conversion in a single Metal dispatch (#2491).
///
/// Computes `real = magnitude * cos(phase)` and `imag = magnitude * sin(phase)`
/// using the Metal `sincos()` intrinsic. Replaces 4 dispatches in the iSTFT path.
pub(crate) fn gpu_polar_to_rect(
    magnitude: &DynTensor,
    phase: &DynTensor,
) -> Result<(DynTensor, DynTensor)> {
    MetalDynBackend::gpu_polar_to_rect(magnitude, phase)
}

/// Bridge for Conv1d GEMM NativeOp (#3390, #4264).
///
/// Routes to direct sliding-window Conv1d for Kokoro K=3 shapes (avoids
/// im2col buffer + blit dispatch), falling back to im2col + simdgroup GEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_conv1d_gemm(
    input: &DynTensor,
    kernel: &DynTensor,
    bias: Option<&DynTensor>,
    padding: usize,
    stride: usize,
    dilation: usize,
    out_shape: &[usize],
) -> Result<DynTensor> {
    // Route to direct sliding-window Conv1d for Kokoro K=3 shapes (#4264).
    // Avoids im2col buffer allocation + blit, saving 1 dispatch per Conv1d.
    let in_shape = input.dims();
    let k_shape = kernel.dims();
    let l_out = out_shape.last().copied().unwrap_or(0);
    if MetalDynBackend::should_use_direct_conv1d_k3(
        in_shape, k_shape, l_out, /*groups=*/ 1, stride, dilation, input.dtype(),
    ) {
        return MetalDynBackend::gpu_direct_conv1d_k3(input, kernel, bias, padding, out_shape);
    }
    MetalDynBackend::gpu_conv1d_gemm(input, kernel, bias, padding, stride, dilation, out_shape)
}

/// Bridge for generic Conv1d NativeOp (supports groups > 1, depthwise). (#3538).
///
/// Delegates to `MetalDynBackend::gpu_conv1d` which handles arbitrary groups
/// via the TensorBlockBuilder IR path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_conv1d(
    input: &DynTensor,
    kernel: &DynTensor,
    bias: Option<&DynTensor>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_conv1d(input, kernel, bias, padding, stride, dilation, groups)
}

/// Bridge for fused Upsample1d + Conv1d NativeOp (#4310).
///
/// Delegates to `MetalDynBackend::gpu_fused_upsample_conv1d` which dispatches
/// a single MSL kernel that reads `[B, C_in, T]` input and writes
/// `[B, C_out, T_out]` output, computing nearest-neighbor upsample inline
/// during the Conv1d accumulation. No intermediate upsampled buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_fused_upsample_conv1d(
    input: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    upsample_factor: usize,
    padding: usize,
    stride: usize,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_fused_upsample_conv1d(input, weight, bias, upsample_factor, padding, stride)
}

/// Call the fused BatchNorm2d GPU kernel from the compiled model executor.
///
/// Delegates to `MetalDynBackend::gpu_batch_norm_fused()` which uses a single
/// Metal dispatch for `(x - running_mean) / sqrt(running_var + eps) * weight + bias`.
/// Part of #4324.
pub(crate) fn native_batch_norm_2d(
    x: &DynTensor,
    running_mean: &DynTensor,
    running_var: &DynTensor,
    weight: Option<&DynTensor>,
    bias: Option<&DynTensor>,
    eps: f64,
) -> Result<DynTensor> {
    MetalDynBackend::gpu_batch_norm_fused(x, running_mean, running_var, weight, bias, eps)
}
