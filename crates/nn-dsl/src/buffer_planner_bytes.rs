// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Byte-size computation for buffer planning.
//!
//! Each [`CompiledStep`] variant has a deterministic output byte size
//! (or 0 for alias/runtime-dependent steps). These functions compute
//! that size, with overflow protection via `checked_mul`.

use crate::ir::ScalarType;
use crate::trace_compile::CompiledStep;

const F32_BYTES: usize = 4;
const F16_BYTES: usize = 2;

/// Compute `shape.product() * F32_BYTES` with overflow protection.
///
/// Returns 0 on overflow — the buffer planner treats this as "cannot
/// pre-plan; executor allocates at runtime."
fn checked_shape_bytes(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .and_then(|product| product.checked_mul(F32_BYTES))
        .unwrap_or(0)
}

/// Compute the output byte size for a single compiled step.
///
/// Returns 0 for steps that alias existing buffers (InputForward,
/// IdentityPassthrough, Passthrough).
pub(super) fn step_output_bytes(step: &CompiledStep) -> usize {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            kernel.output_shape().map(checked_shape_bytes).unwrap_or(0)
        }
        CompiledStep::ConstantValue { shape, .. } => checked_shape_bytes(shape),
        CompiledStep::NativeOp { op, .. } => native_op_output_bytes(op),
        // RuntimeOp has data-dependent output shape — cannot pre-plan.
        // The executor allocates the output buffer at runtime.
        CompiledStep::RuntimeOp { .. } => 0,
        // These variants alias an existing buffer -- no new allocation.
        CompiledStep::InputForward
        | CompiledStep::IdentityPassthrough
        | CompiledStep::Passthrough { .. }
        | CompiledStep::NarrowView { .. } => 0,
        // No catch-all: same-crate match on #[non_exhaustive] CompiledStep.
        // Adding a new variant without a planner arm causes a compile error,
        // preventing silent 0-byte pre-allocation for new step types.
    }
}

/// Compute the output byte size for a NativeOp step.
pub(super) fn native_op_output_bytes(op: &crate::trace_compile::NativeOpKind) -> usize {
    use crate::trace_compile::NativeOpKind;
    match op {
        // LSTM output shape: [seq_len, batch, hidden_size]
        NativeOpKind::LstmSequence {
            hidden_size,
            input_shape,
            ..
        } => {
            if input_shape.len() >= 2 {
                let seq_len = input_shape[0];
                let batch = input_shape[1];
                seq_len
                    .checked_mul(batch)
                    .and_then(|v| v.checked_mul(*hidden_size))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // Cumsum output shape == input shape.
        NativeOpKind::Cumsum { input_shape, .. } => checked_shape_bytes(input_shape),
        // Flash Attention output shape == Q shape [B, H_q, S_q, D].
        NativeOpKind::FlashAttention { output_shape, .. } => checked_shape_bytes(output_shape),
        // NormActivConv1d output shape: [B, C_out, T_out].
        // T_out = (T_in + 2*padding - dilation*(K-1) - 1) / 1 + 1
        //       = T_in + 2*padding - dilation*(K-1)      (stride=1)
        NativeOpKind::NormActivConv1d {
            input_shape,
            output_channels,
            kernel_size,
            conv_padding,
            conv_dilation,
            ..
        } => {
            if input_shape.len() >= 3 && *kernel_size > 0 {
                let batch = input_shape[0];
                let t_in = input_shape[2];
                let padded = t_in.checked_add(2_usize.saturating_mul(*conv_padding));
                let kernel_span = conv_dilation.checked_mul(kernel_size - 1);
                match (padded, kernel_span) {
                    (Some(p), Some(ks)) if p >= ks => {
                        let t_out = p - ks;
                        batch
                            .checked_mul(*output_channels)
                            .and_then(|v| v.checked_mul(t_out))
                            .and_then(|v| v.checked_mul(F32_BYTES))
                            .unwrap_or(0)
                    }
                    _ => 0,
                }
            } else {
                0
            }
        }
        // LayerNorm/InstanceNorm/AdaIN/SiluMul: output shape == input shape.
        NativeOpKind::LayerNorm { input_shape, .. }
        | NativeOpKind::AddLayerNorm { input_shape, .. }
        | NativeOpKind::InstanceNorm { input_shape, .. }
        | NativeOpKind::AdainSnake { input_shape, .. }
        | NativeOpKind::AdainLeakyRelu { input_shape, .. }
        | NativeOpKind::AdaLayerNorm { input_shape, .. }
        | NativeOpKind::ChannelsFirstLayerNorm { input_shape, .. }
        | NativeOpKind::SiluMul { input_shape, .. }
        | NativeOpKind::RotaryEmbedding { input_shape, .. }
        | NativeOpKind::MoeGating { input_shape, .. }
        | NativeOpKind::FusedAdainSnake { input_shape, .. }
        | NativeOpKind::FusedInstanceNormMulAdd { input_shape, .. }
        | NativeOpKind::FusedSnakeInstanceNorm { input_shape, .. }
        | NativeOpKind::FusedMulAdd { input_shape, .. }
        | NativeOpKind::FusedSiGLU { input_shape, .. }
        | NativeOpKind::FusedGeGLU { input_shape, .. }
        | NativeOpKind::BatchNorm2d { input_shape, .. } => checked_shape_bytes(input_shape),
        // FusedResBlock: residual add(x, phase2_output) preserves x shape.
        // Output shape == phase1.input_shape (the block input).
        NativeOpKind::FusedResBlock { phase1, .. } => checked_shape_bytes(&phase1.input_shape),
        // MaxPool1d output: [B, C, floor((L + 2*P - K) / S) + 1].
        NativeOpKind::MaxPool1d {
            kernel_size,
            stride,
            padding,
            input_shape,
        } => {
            if input_shape.len() >= 3 && *stride > 0 && *kernel_size > 0 {
                let batch = input_shape[0];
                let channels = input_shape[1];
                let length = input_shape[2];
                let padded = length.saturating_add(2_usize.saturating_mul(*padding));
                if padded >= *kernel_size {
                    let out_len = (padded - kernel_size) / stride + 1;
                    batch
                        .checked_mul(channels)
                        .and_then(|v| v.checked_mul(out_len))
                        .and_then(|v| v.checked_mul(F32_BYTES))
                        .unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            }
        }
        // LinearActivation: output shape = input shape with last dim replaced by out_features.
        NativeOpKind::LinearActivation {
            out_features,
            input_shape,
            ..
        } => {
            if input_shape.is_empty() {
                0
            } else {
                let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
                batch
                    .checked_mul(*out_features)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            }
        }
        // NormLinear: output shape = input shape with last dim replaced by out_features.
        NativeOpKind::NormLinear {
            out_features,
            input_shape,
            ..
        } => {
            if input_shape.is_empty() {
                0
            } else {
                let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
                batch
                    .checked_mul(*out_features)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            }
        }
        // AddNormLinear: same output shape as NormLinear (input shape with last dim
        // replaced by out_features). Part of #3351 T2.1.
        NativeOpKind::AddNormLinear {
            out_features,
            input_shape,
            ..
        } => {
            if input_shape.is_empty() {
                0
            } else {
                let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
                batch
                    .checked_mul(*out_features)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            }
        }
        // BatchedLinearProjection: step buffer holds the first projection's
        // narrow output [..batch, projection_sizes[0]]. The full matmul
        // intermediate is in a thread-local temp, not in the buffer plan.
        NativeOpKind::BatchedLinearProjection {
            projection_sizes,
            input_shape,
            ..
        } => {
            let first_out = projection_sizes.first().copied().unwrap_or(0);
            if input_shape.is_empty() || first_out == 0 {
                0
            } else {
                let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
                batch
                    .checked_mul(first_out)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            }
        }
        // ProjectionSlice: output shape is the narrowed slice.
        NativeOpKind::ProjectionSlice { output_shape, .. } => checked_shape_bytes(output_shape),
        // BatchedStyleProjection: output = [B, total_out] where B comes from
        // the style embedding input. We use batch=1 for buffer sizing since
        // the actual batch dim is resolved at runtime.
        NativeOpKind::BatchedStyleProjection { total_out, .. } => {
            // Conservative: allocate for batch=1. Runtime may expand.
            total_out.checked_mul(F32_BYTES).unwrap_or(0)
        }
        // ConstantWeight aliases a pre-uploaded buffer — no new allocation.
        NativeOpKind::ConstantWeight { .. } => 0,
        // Int8Gemm: output shape = input shape with last dim replaced by out_features.
        NativeOpKind::Int8Gemm {
            out_features,
            input_shape,
            ..
        } => {
            if input_shape.is_empty() {
                0
            } else {
                let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
                batch
                    .checked_mul(*out_features)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            }
        }
        // Conv1dGemm: output shape [B, out_channels, L_out].
        // L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        NativeOpKind::Conv1dGemm {
            input_shape,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            ..
        } => {
            if input_shape.len() >= 3 {
                let batch = input_shape[0];
                let l_in = input_shape[2];
                let effective_k = dilation * (kernel_size - 1) + 1;
                let l_out = (l_in + 2 * padding).saturating_sub(effective_k) / stride + 1;
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(l_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedUpsampleConv1d: output shape [B, out_channels, L_out].
        // L_up = T * upsample_factor. L_out = (L_up + 2*padding - kernel_size) / stride + 1.
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor,
            out_channels,
            kernel_size,
            stride,
            padding,
            input_shape,
            ..
        } => {
            if input_shape.len() >= 3 && *stride > 0 && *kernel_size > 0 {
                let batch = input_shape[0];
                let l_in = input_shape[2];
                let l_up = l_in.saturating_mul(*upsample_factor);
                let padded = l_up.saturating_add(2_usize.saturating_mul(*padding));
                let l_out = if padded >= *kernel_size {
                    (padded - kernel_size) / stride + 1
                } else {
                    0
                };
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(l_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedLayerNormLinear: output shape = input shape with last dim replaced
        // by out_features. Same as NormLinear. Part of #4252.
        NativeOpKind::FusedLayerNormLinear {
            out_features,
            input_shape,
            ..
        } => {
            if input_shape.is_empty() {
                0
            } else {
                let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
                batch
                    .checked_mul(*out_features)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            }
        }
        // BiLstmCat: output shape [seq_len, batch, 2 * hidden_size].
        NativeOpKind::BiLstmCat {
            hidden_size,
            input_shape,
            ..
        } => {
            if input_shape.len() >= 2 {
                let seq_len = input_shape[0];
                let batch = input_shape[1];
                seq_len
                    .checked_mul(batch)
                    .and_then(|v| v.checked_mul(2 * hidden_size))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedConv1dActivation: output shape = [B, out_channels, L_out].
        // Conv1d output length: (L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1.
        // Part of #4264.
        NativeOpKind::FusedConv1dActivation {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            input_shape,
            ..
        } => {
            if input_shape.len() >= 3 {
                let batch = input_shape[0];
                let l_in = input_shape[input_shape.len() - 1];
                let l_out = (l_in + 2 * padding)
                    .saturating_sub(dilation * (kernel_size - 1))
                    .saturating_sub(1)
                    / stride
                    + 1;
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(l_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedConv1dSnakeNorm: output shape = [B, out_channels, L_out].
        // Conv1d → Snake → InstanceNorm in a single NativeOp. Part of #4264.
        NativeOpKind::FusedConv1dSnakeNorm {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            input_shape,
            ..
        } => {
            if input_shape.len() >= 3 {
                let batch = input_shape[0];
                let l_in = input_shape[input_shape.len() - 1];
                let l_out = (l_in + 2 * padding)
                    .saturating_sub(dilation * (kernel_size - 1))
                    .saturating_sub(1)
                    / stride
                    + 1;
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(l_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedConv1dSnakeNormResBlock: output shape = input shape [B, C, L].
        // 2x (Conv1d → Snake → InstanceNorm) + residual add.
        // stride=1 in both phases, same-padding → output L = input L.
        // Output channels = phase2_out_channels (should equal input channels
        // for the residual add to be valid). Part of #4264.
        NativeOpKind::FusedConv1dSnakeNormResBlock {
            input_shape,
            phase2_out_channels,
            ..
        } => {
            if input_shape.len() >= 3 {
                let batch = input_shape[0];
                let l = input_shape[input_shape.len() - 1];
                batch
                    .checked_mul(*phase2_out_channels)
                    .and_then(|v| v.checked_mul(l))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedAddInstanceNormConv1x1: output shape = [B, out_channels, T].
        // Add + InstanceNorm + Conv1d(K=1) fused. T preserved (K=1, stride=1).
        // Part of #4264.
        NativeOpKind::FusedAddInstanceNormConv1x1 {
            input_shape,
            out_channels,
            ..
        } => {
            if input_shape.len() >= 3 {
                let batch = input_shape[0];
                let t = input_shape[2];
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(t))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedConvTranspose1dActivation: output shape = [B, out_channels, T_out].
        // T_out = (T_in - 1) * stride - 2*padding + dilation*(K-1) + output_padding + 1
        // Part of #4264.
        NativeOpKind::FusedConvTranspose1dActivation {
            input_shape,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            ..
        } => {
            if input_shape.len() >= 3 && *stride > 0 {
                let batch = input_shape[0];
                let t_in = input_shape[2];
                // ConvTranspose1d output length formula.
                let t_out = (t_in.saturating_sub(1))
                    .saturating_mul(*stride)
                    .saturating_add(dilation.saturating_mul(kernel_size.saturating_sub(1)))
                    .saturating_add(*output_padding)
                    .saturating_add(1)
                    .saturating_sub(2_usize.saturating_mul(*padding));
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(t_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // NormActivConvTranspose1d: InstanceNorm + activation + ConvTranspose1d.
        // Output shape = [B, output_channels, T_out].
        // T_out = (T_in - 1) * stride - 2*padding + dilation*(K-1) + output_padding + 1
        // Part of #4264.
        NativeOpKind::NormActivConvTranspose1d {
            input_shape,
            output_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            ..
        } => {
            if input_shape.len() >= 3 && *stride > 0 {
                let batch = input_shape[0];
                let t_in = input_shape[2];
                let t_out = (t_in.saturating_sub(1))
                    .saturating_mul(*stride)
                    .saturating_add(dilation.saturating_mul(kernel_size.saturating_sub(1)))
                    .saturating_add(*output_padding)
                    .saturating_add(1)
                    .saturating_sub(2_usize.saturating_mul(*padding));
                batch
                    .checked_mul(*output_channels)
                    .and_then(|v| v.checked_mul(t_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedInstanceNormConv1d: InstanceNorm(x) → Conv1d(result).
        // Output shape from Conv1d: [B, out_channels, L_out].
        // L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1
        // Part of #4264.
        NativeOpKind::FusedInstanceNormConv1d {
            input_shape,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            ..
        } => {
            if input_shape.len() >= 3 && *stride > 0 {
                let batch = input_shape[0];
                let l_in = input_shape[2];
                let l_out = l_in
                    .saturating_add(2_usize.saturating_mul(*padding))
                    .saturating_sub(dilation.saturating_mul(kernel_size.saturating_sub(1)))
                    .saturating_sub(1)
                    / *stride
                    + 1;
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(l_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedConv1dInstanceNorm: Conv1d(x) → InstanceNorm(result).
        // Output shape from InstanceNorm: same as Conv1d output = [B, out_channels, L_out].
        // L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1
        // Part of #4264.
        NativeOpKind::FusedConv1dInstanceNorm {
            input_shape,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            ..
        } => {
            if input_shape.len() >= 3 && *stride > 0 {
                let batch = input_shape[0];
                let l_in = input_shape[2];
                let l_out = l_in
                    .saturating_add(2_usize.saturating_mul(*padding))
                    .saturating_sub(dilation.saturating_mul(kernel_size.saturating_sub(1)))
                    .saturating_sub(1)
                    / *stride
                    + 1;
                batch
                    .checked_mul(*out_channels)
                    .and_then(|v| v.checked_mul(l_out))
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedLinearLayerNorm: Linear(x) → LayerNorm(result).
        // Output shape: [..batch, out_features]. LayerNorm preserves shape.
        // Part of #4264.
        NativeOpKind::FusedLinearLayerNorm {
            input_shape,
            out_features,
            ..
        } => {
            if !input_shape.is_empty() {
                let batch: usize = input_shape.iter().rev().skip(1).product::<usize>().max(1);
                batch
                    .checked_mul(*out_features)
                    .and_then(|v| v.checked_mul(F32_BYTES))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        // FusedResBlockChain: chains 2-4 FusedResBlocks.
        // Output shape = last block's input shape (residual add preserves shape).
        // Part of #4264.
        NativeOpKind::FusedResBlockChain { blocks, .. } => {
            if let Some(last) = blocks.last() {
                checked_shape_bytes(&last.phase1.input_shape)
            } else {
                0
            }
        } // No catch-all: same-crate match on #[non_exhaustive] NativeOpKind.
          // Adding a new variant without a planner arm causes a compile error,
          // preventing silent 0-byte pre-allocation for new NativeOp types.
    }
}

/// Compute output bytes for a step with an optional dtype override.
///
/// Uses the dtype's byte width (2 for F16/BF16, 4 for F32) for all
/// step types. NativeOps are included because mixed-precision executors
/// cast NativeOp output to the target dtype before storing.
pub(super) fn step_output_bytes_typed(step: &CompiledStep, dtype: Option<ScalarType>) -> usize {
    let elem_bytes = dtype.map(scalar_type_bytes).unwrap_or(F32_BYTES);
    match step {
        CompiledStep::Dispatch { kernel, .. } => kernel
            .output_shape()
            .map(|s| checked_shape_bytes_with_elem_size(s, elem_bytes))
            .unwrap_or(0),
        CompiledStep::ConstantValue { shape, .. } => {
            checked_shape_bytes_with_elem_size(shape, elem_bytes)
        }
        // Scale NativeOp sizes from F32 base to target dtype.
        // native_op_output_bytes returns numel * F32_BYTES; rescale to elem_bytes.
        CompiledStep::NativeOp { op, .. } => {
            let f32_bytes = native_op_output_bytes(op);
            f32_bytes / F32_BYTES * elem_bytes
        }
        CompiledStep::RuntimeOp { .. } => 0,
        CompiledStep::InputForward
        | CompiledStep::IdentityPassthrough
        | CompiledStep::Passthrough { .. }
        | CompiledStep::NarrowView { .. } => 0,
    }
}

/// Byte width for a ScalarType.
fn scalar_type_bytes(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::F16 | ScalarType::BF16 => F16_BYTES,
        ScalarType::F32 => F32_BYTES,
    }
}

/// Compute `shape.product() * elem_bytes` with overflow protection.
fn checked_shape_bytes_with_elem_size(shape: &[usize], elem_bytes: usize) -> usize {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .and_then(|product| product.checked_mul(elem_bytes))
        .unwrap_or(0)
}
