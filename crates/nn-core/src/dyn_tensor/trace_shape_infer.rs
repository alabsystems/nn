// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape inference for individual `TraceOp` variants.
//!
//! Given an operation, its input shapes, and its current (possibly stale) output
//! shape, infers the correct output shape using deterministic op rules (conv
//! formulas, broadcast, reshape, etc.).
//!
//! Extracted from `trace_shape_propagate.rs` to comply with the 500-line limit.
//! Called by `ComputationGraph::propagate_shapes()`.

use crate::dyn_tensor::trace::TraceOp;

/// Infer the output shape for an operation given its input shapes.
///
/// Returns the original `current_shape` for ops where shape cannot be
/// determined from inputs alone (data-dependent ops, constants, etc.).
pub(super) fn infer_output_shape(
    op: &TraceOp,
    input_shapes: &[Vec<usize>],
    current_shape: &[usize],
) -> Vec<usize> {
    match op {
        // Input and constant nodes keep their shapes (set by override or weights).
        TraceOp::Input
        | TraceOp::Constant { .. }
        | TraceOp::ConstantWeight { .. }
        | TraceOp::Arange { .. } => current_shape.to_vec(),

        // Unary element-wise: output shape = input shape.
        TraceOp::Relu
        | TraceOp::Gelu
        | TraceOp::GeluErf
        | TraceOp::Silu
        | TraceOp::Tanh
        | TraceOp::Sigmoid
        | TraceOp::Exp
        | TraceOp::Log
        | TraceOp::Sqrt
        | TraceOp::Sqr
        | TraceOp::Abs
        | TraceOp::Neg
        | TraceOp::Recip
        | TraceOp::Sin
        | TraceOp::Cos
        | TraceOp::Tan
        | TraceOp::Floor
        | TraceOp::Ceil
        | TraceOp::Round
        | TraceOp::Sign
        | TraceOp::Fract
        | TraceOp::Dropout
        | TraceOp::Softplus
        | TraceOp::Selu
        | TraceOp::Mish
        | TraceOp::HardSigmoid
        | TraceOp::HardSwish
        | TraceOp::Softsign => first_input_or_current(input_shapes, current_shape),

        // Activations with parameters (still element-wise).
        TraceOp::Elu { .. }
        | TraceOp::LeakyRelu { .. }
        | TraceOp::Celu { .. }
        | TraceOp::PRelu { .. }
        | TraceOp::Activation { .. } => first_input_or_current(input_shapes, current_shape),

        // Power: same shape as first input.
        TraceOp::Powf { .. } => first_input_or_current(input_shapes, current_shape),

        // Dtype conversion: same shape.
        TraceOp::ToDtype { .. } => first_input_or_current(input_shapes, current_shape),

        // Binary element-wise: broadcast shape of two inputs.
        TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Mul
        | TraceOp::Div
        | TraceOp::Maximum
        | TraceOp::Minimum
        | TraceOp::Atan2 => {
            if input_shapes.len() >= 2 {
                broadcast_shape(&input_shapes[0], &input_shapes[1])
            } else {
                first_input_or_current(input_shapes, current_shape)
            }
        }

        // Comparison ops: same shape as input (or broadcast).
        TraceOp::Compare { .. } => first_input_or_current(input_shapes, current_shape),
        TraceOp::CompareTensor { .. } => {
            if input_shapes.len() >= 2 {
                broadcast_shape(&input_shapes[0], &input_shapes[1])
            } else {
                first_input_or_current(input_shapes, current_shape)
            }
        }

        // WhereCond: broadcast of condition, true, false tensors.
        TraceOp::WhereCond => {
            if input_shapes.len() >= 3 {
                let bc = broadcast_shape(&input_shapes[0], &input_shapes[1]);
                broadcast_shape(&bc, &input_shapes[2])
            } else {
                first_input_or_current(input_shapes, current_shape)
            }
        }

        // MatMul: [..., M, K] x [..., K, N] -> [..., M, N]
        TraceOp::MatMul => {
            if input_shapes.len() >= 2 {
                infer_matmul_shape(&input_shapes[0], &input_shapes[1])
                    .unwrap_or_else(|| current_shape.to_vec())
            } else {
                current_shape.to_vec()
            }
        }

        // Linear: [..., in_features] -> [..., out_features]
        TraceOp::Linear { weight, .. } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if !input.is_empty() && w_shape.len() == 2 {
                    let mut out = input.clone();
                    *out.last_mut().unwrap() = w_shape[0]; // out_features
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // Conv1d: [B, C_in, L] -> [B, C_out, L_out]
        TraceOp::Conv1d {
            weight,
            padding,
            stride,
            dilation,
            ..
        } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if input.len() >= 3 && w_shape.len() >= 3 {
                    let out_channels = w_shape[0];
                    let kernel_size = w_shape[2];
                    let l_in = input[input.len() - 1];
                    let l_out =
                        conv1d_output_length(l_in, kernel_size, *stride, *padding, *dilation);
                    let mut out = input.clone();
                    out[input.len() - 2] = out_channels;
                    out[input.len() - 1] = l_out;
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // Conv2d: [B, C_in, H, W] -> [B, C_out, H_out, W_out]
        TraceOp::Conv2d {
            weight,
            padding,
            stride,
            dilation,
            ..
        } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if input.len() >= 4 && w_shape.len() >= 4 {
                    let out_channels = w_shape[0];
                    let h_in = input[input.len() - 2];
                    let w_in = input[input.len() - 1];
                    let h_out =
                        conv1d_output_length(h_in, w_shape[2], stride[0], padding[0], dilation[0]);
                    let w_out =
                        conv1d_output_length(w_in, w_shape[3], stride[1], padding[1], dilation[1]);
                    let mut out = input.clone();
                    out[input.len() - 3] = out_channels;
                    out[input.len() - 2] = h_out;
                    out[input.len() - 1] = w_out;
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // ConvTranspose1d: [B, C_in, L] -> [B, C_out, L_out]
        TraceOp::ConvTranspose1d {
            weight,
            padding,
            output_padding,
            stride,
            dilation,
            ..
        } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if input.len() >= 3 && w_shape.len() >= 3 {
                    let out_channels = w_shape[1]; // ConvTranspose: weight is [C_in, C_out, K]
                    let kernel_size = w_shape[2];
                    let l_in = input[input.len() - 1];
                    let l_out = conv_transpose1d_output_length(
                        l_in,
                        kernel_size,
                        *stride,
                        *padding,
                        *output_padding,
                        *dilation,
                    );
                    let mut out = input.clone();
                    out[input.len() - 2] = out_channels;
                    out[input.len() - 1] = l_out;
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // ConvTranspose2d: [B, C_in, H, W] -> [B, C_out, H_out, W_out]
        TraceOp::ConvTranspose2d {
            weight,
            padding,
            output_padding,
            stride,
            dilation,
            ..
        } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if input.len() >= 4 && w_shape.len() >= 4 {
                    let out_channels = w_shape[1];
                    let h_in = input[input.len() - 2];
                    let w_in = input[input.len() - 1];
                    let h_out = conv_transpose1d_output_length(
                        h_in,
                        w_shape[2],
                        stride[0],
                        padding[0],
                        output_padding[0],
                        dilation[0],
                    );
                    let w_out = conv_transpose1d_output_length(
                        w_in,
                        w_shape[3],
                        stride[1],
                        padding[1],
                        output_padding[1],
                        dilation[1],
                    );
                    let mut out = input.clone();
                    out[input.len() - 3] = out_channels;
                    out[input.len() - 2] = h_out;
                    out[input.len() - 1] = w_out;
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // Normalization ops: same shape as input.
        TraceOp::LayerNorm { .. }
        | TraceOp::RmsNorm { .. }
        | TraceOp::GroupNorm { .. }
        | TraceOp::InstanceNorm { .. }
        | TraceOp::BatchNorm { .. } => first_input_or_current(input_shapes, current_shape),

        // Softmax/LogSoftmax: same shape.
        TraceOp::Softmax { .. } | TraceOp::LogSoftmax { .. } => {
            first_input_or_current(input_shapes, current_shape)
        }

        // Reductions: shrink one dim (keepdim preserves rank).
        TraceOp::ReduceSum { dim, keepdim }
        | TraceOp::ReduceMean { dim, keepdim }
        | TraceOp::ReduceMax { dim, keepdim }
        | TraceOp::ReduceMin { dim, keepdim } => {
            if let Some(input) = input_shapes.first() {
                infer_reduction_shape(input, *dim, *keepdim)
            } else {
                current_shape.to_vec()
            }
        }

        // Reshape: use the target_shape, but handle -1 dimensions.
        TraceOp::Reshape { target_shape } => {
            if let Some(input) = input_shapes.first() {
                infer_reshape_shape(input, target_shape)
            } else {
                current_shape.to_vec()
            }
        }

        // Transpose: swap two dims.
        TraceOp::Transpose { dim0, dim1 } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if *dim0 < out.len() && *dim1 < out.len() {
                    out.swap(*dim0, *dim1);
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Permute: reorder dims.
        TraceOp::Permute { axes } => {
            if let Some(input) = input_shapes.first() {
                if axes.len() == input.len() {
                    axes.iter()
                        .map(|&a| input.get(a).copied().unwrap_or(0))
                        .collect()
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // Unsqueeze: insert a dim of size 1.
        TraceOp::Unsqueeze { dim } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                let d = if *dim > out.len() { out.len() } else { *dim };
                out.insert(d, 1);
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Squeeze: remove a dim of size 1.
        TraceOp::Squeeze { dim } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if *dim < out.len() && out[*dim] == 1 {
                    out.remove(*dim);
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Narrow: slice along a dim.
        TraceOp::Narrow { dim, start, length } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if *dim < out.len() {
                    let remaining = out[*dim].saturating_sub(*start);
                    out[*dim] = if *length == usize::MAX || *length > remaining {
                        remaining
                    } else {
                        *length
                    };
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Cat: concatenate along a dim.
        TraceOp::Cat { dim, num_inputs } => {
            if input_shapes.len() >= *num_inputs && *num_inputs > 0 {
                let mut out = input_shapes[0].clone();
                if *dim < out.len() {
                    let total: usize = input_shapes[..*num_inputs]
                        .iter()
                        .filter_map(|s| s.get(*dim))
                        .sum();
                    out[*dim] = total;
                }
                out
            } else {
                first_input_or_current(input_shapes, current_shape)
            }
        }

        // Expand: use the explicit target shape.
        TraceOp::Expand { target_shape } => target_shape.clone(),

        // Flip/Roll: same shape as input.
        TraceOp::Flip { .. } | TraceOp::Roll { .. } | TraceOp::Cumsum { .. } => {
            first_input_or_current(input_shapes, current_shape)
        }

        // Clamp: same shape as input.
        TraceOp::Clamp { .. } => first_input_or_current(input_shapes, current_shape),

        // Padding ops.
        TraceOp::ReflectionPad1d {
            pad_left,
            pad_right,
        } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if let Some(last) = out.last_mut() {
                    *last += pad_left + pad_right;
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        TraceOp::ReflectionPad2d {
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                let rank = out.len();
                if rank >= 2 {
                    out[rank - 1] += pad_left + pad_right;
                    out[rank - 2] += pad_top + pad_bottom;
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        TraceOp::ConstantPadNd { padding, .. } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                // padding is [left, right, ...] pairs from innermost dim outward
                let rank = out.len();
                for (pair_idx, chunk) in padding.chunks(2).enumerate() {
                    if chunk.len() == 2 {
                        let dim = rank.saturating_sub(1 + pair_idx);
                        if dim < rank {
                            out[dim] += chunk[0] + chunk[1];
                        }
                    }
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Upsample1d: multiply last dim by factor.
        TraceOp::Upsample1d { factor } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if let Some(last) = out.last_mut() {
                    *last *= factor;
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Upsample2d: multiply last two dims by scale factors.
        TraceOp::Upsample2d {
            scale_h, scale_w, ..
        } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                let rank = out.len();
                if rank >= 2 {
                    out[rank - 2] = (out[rank - 2] as f64 * scale_h).round() as usize;
                    out[rank - 1] = (out[rank - 1] as f64 * scale_w).round() as usize;
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // ResizeBilinear: set last two dims to target.
        TraceOp::ResizeBilinear { target_h, target_w } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                let rank = out.len();
                if rank >= 2 {
                    out[rank - 2] = *target_h;
                    out[rank - 1] = *target_w;
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Pooling ops.
        TraceOp::MaxPool1d {
            kernel_size,
            stride,
            padding,
        }
        | TraceOp::AvgPool1d {
            kernel_size,
            stride,
            padding,
        } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if let Some(last) = out.last_mut() {
                    *last = conv1d_output_length(*last, *kernel_size, *stride, *padding, 1);
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        TraceOp::AdaptiveAvgPool1d { output_size } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                if let Some(last) = out.last_mut() {
                    *last = *output_size;
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        TraceOp::AvgPool2d {
            kernel_size,
            stride,
            padding,
        }
        | TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                let rank = out.len();
                if rank >= 2 {
                    out[rank - 2] = conv1d_output_length(
                        out[rank - 2],
                        kernel_size[0],
                        stride[0],
                        padding[0],
                        1,
                    );
                    out[rank - 1] = conv1d_output_length(
                        out[rank - 1],
                        kernel_size[1],
                        stride[1],
                        padding[1],
                        1,
                    );
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        TraceOp::AdaptiveAvgPool2d { output_size } | TraceOp::AdaptiveMaxPool2d { output_size } => {
            if let Some(input) = input_shapes.first() {
                let mut out = input.clone();
                let rank = out.len();
                if rank >= 2 {
                    out[rank - 2] = output_size[0];
                    out[rank - 1] = output_size[1];
                }
                out
            } else {
                current_shape.to_vec()
            }
        }

        // Embedding: [seq_len] -> [seq_len, embed_dim]
        TraceOp::Embedding { weight } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if w_shape.len() == 2 {
                    let mut out = input.clone();
                    out.push(w_shape[1]);
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // PixelShuffle / PixelUnshuffle
        TraceOp::PixelShuffle { upscale_factor } => {
            if let Some(input) = input_shapes.first() {
                if input.len() >= 3 {
                    let r = *upscale_factor;
                    let mut out = input.clone();
                    let rank = out.len();
                    out[rank - 3] /= r * r;
                    out[rank - 2] *= r;
                    out[rank - 1] *= r;
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        TraceOp::PixelUnshuffle { downscale_factor } => {
            if let Some(input) = input_shapes.first() {
                if input.len() >= 3 {
                    let r = *downscale_factor;
                    let mut out = input.clone();
                    let rank = out.len();
                    out[rank - 3] *= r * r;
                    out[rank - 2] /= r;
                    out[rank - 1] /= r;
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // Triu/Tril: same shape as input.
        TraceOp::Triu { .. } | TraceOp::Tril { .. } => {
            first_input_or_current(input_shapes, current_shape)
        }

        // Kokoro fused ops: element-wise, preserve shape.
        TraceOp::KokoroFused(_) => first_input_or_current(input_shapes, current_shape),

        // SDPA: output shape = Q shape.
        TraceOp::Sdpa { .. } | TraceOp::SdpaCausal { .. } => {
            first_input_or_current(input_shapes, current_shape)
        }

        // Scatter/IndexAdd/etc: same shape as first input (target tensor).
        TraceOp::Scatter { .. }
        | TraceOp::ScatterAdd { .. }
        | TraceOp::IndexAdd { .. }
        | TraceOp::IndexPut { .. }
        | TraceOp::SliceSet { .. } => first_input_or_current(input_shapes, current_shape),

        // SwiGlu: same shape as input (gated feedforward).
        TraceOp::SwiGlu => first_input_or_current(input_shapes, current_shape),

        // RotaryEmbedding: same shape as input.
        TraceOp::RotaryEmbedding { .. } => first_input_or_current(input_shapes, current_shape),

        // QLinear: same as Linear.
        TraceOp::QLinear { weight, .. } => {
            if let Some(input) = input_shapes.first() {
                let w_shape = weight.shape();
                if !input.is_empty() && w_shape.len() == 2 {
                    let mut out = input.clone();
                    *out.last_mut().unwrap() = w_shape[0];
                    out
                } else {
                    current_shape.to_vec()
                }
            } else {
                current_shape.to_vec()
            }
        }

        // SegmentBoundary: same shape as input (marker only).
        TraceOp::SegmentBoundary { .. } => first_input_or_current(input_shapes, current_shape),

        // GridSample: same spatial dims as grid input.
        TraceOp::GridSample { .. } => current_shape.to_vec(),

        // For all other ops, keep the current (original) shape.
        // This is conservative and correct: if we can't infer the shape,
        // we don't change it.
        _ => current_shape.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Helper: first input shape or current fallback
// ---------------------------------------------------------------------------

/// Returns the first input shape if available, otherwise the current shape.
fn first_input_or_current(input_shapes: &[Vec<usize>], current_shape: &[usize]) -> Vec<usize> {
    if let Some(s) = input_shapes.first() {
        s.clone()
    } else {
        current_shape.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Convolution formulas
// ---------------------------------------------------------------------------

/// Conv1d output length: `floor((L_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)`
pub(super) fn conv1d_output_length(
    l_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    if stride == 0 {
        return l_in;
    }
    let numerator = l_in + 2 * padding;
    let kernel_extent = dilation * (kernel_size - 1) + 1;
    if numerator < kernel_extent {
        return 0;
    }
    (numerator - kernel_extent) / stride + 1
}

/// ConvTranspose1d output length.
pub(super) fn conv_transpose1d_output_length(
    l_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> usize {
    (l_in.saturating_sub(1)) * stride + dilation * (kernel_size - 1) + output_padding + 1
        - 2 * padding
}

// ---------------------------------------------------------------------------
// Broadcast / matmul / reduction / reshape helpers
// ---------------------------------------------------------------------------

/// Broadcast shape following NumPy rules (right-aligned).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![0usize; max_rank];
    for i in 0..max_rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        result[max_rank - 1 - i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            // Incompatible — return the larger (best effort).
            da.max(db)
        };
    }
    result
}

/// Infer matmul output shape.
fn infer_matmul_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let m = a[a.len() - 2];
    let n = b[b.len() - 1];
    // Batch dims: broadcast
    let batch_a = &a[..a.len() - 2];
    let batch_b = &b[..b.len() - 2];
    let batch = broadcast_shape(batch_a, batch_b);
    let mut result = batch;
    result.push(m);
    result.push(n);
    Some(result)
}

/// Infer reduction shape.
fn infer_reduction_shape(input: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    if dim >= input.len() {
        return input.to_vec();
    }
    if keepdim {
        let mut out = input.to_vec();
        out[dim] = 1;
        out
    } else {
        let mut out = input.to_vec();
        out.remove(dim);
        out
    }
}

/// Infer reshape shape, resolving -1 dimensions.
fn infer_reshape_shape(input: &[usize], target: &[usize]) -> Vec<usize> {
    let total_elements: usize = input.iter().product();
    if total_elements == 0 {
        return target.to_vec();
    }
    // Check for any zero dims (used as "infer from input").
    // In torch, 0 means "keep from input" and -1 means "infer".
    // But since target_shape is Vec<usize>, -1 wraps to usize::MAX.
    let has_infer = target.contains(&usize::MAX);
    if has_infer {
        let known_product: usize = target.iter().filter(|&&d| d != usize::MAX).product();
        if known_product == 0 {
            return target.to_vec();
        }
        let inferred = total_elements / known_product;
        target
            .iter()
            .map(|&d| if d == usize::MAX { inferred } else { d })
            .collect()
    } else {
        // target_shape is fully specified; use as-is.
        // But recompute if total elements differ (shape was captured at a different input size).
        let target_elements: usize = target.iter().product();
        if target_elements == total_elements {
            target.to_vec()
        } else {
            // Shapes don't match — try to adjust the largest dimension.
            current_shape_from_reshape(input, target, total_elements)
        }
    }
}

/// Try to adjust reshape target when total elements differ.
///
/// Finds dimensions that changed between the original input and target, and
/// scales them proportionally. Falls back to the original target shape.
fn current_shape_from_reshape(
    _input: &[usize],
    target: &[usize],
    total_elements: usize,
) -> Vec<usize> {
    // If there's exactly one dimension that doesn't divide evenly,
    // treat it as the "variable" dimension and recompute it.
    let fixed_product: usize = target.iter().product::<usize>().max(1);
    if fixed_product == 0 {
        return target.to_vec();
    }

    // Try: keep all target dims except one, and recompute that one.
    for i in 0..target.len() {
        let other_product: usize = target
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &d)| d)
            .product::<usize>()
            .max(1);
        if total_elements.is_multiple_of(other_product) {
            let inferred = total_elements / other_product;
            let mut out = target.to_vec();
            out[i] = inferred;
            return out;
        }
    }

    // Can't resolve — return original target.
    target.to_vec()
}

#[cfg(test)]
mod tests {
    use super::infer_output_shape;
    use crate::dyn_tensor::trace::TraceOp;

    #[test]
    fn infer_narrow_shape_clamps_open_ended_length_to_remaining_extent() {
        let shape = infer_output_shape(
            &TraceOp::Narrow {
                dim: 1,
                start: 11,
                length: usize::MAX,
            },
            &[vec![1, 22, 3001]],
            &[1, 11, 3001],
        );
        assert_eq!(shape, vec![1, 11, 3001]);
    }

    #[test]
    fn infer_narrow_shape_clamps_oversized_length_to_remaining_extent() {
        let shape = infer_output_shape(
            &TraceOp::Narrow {
                dim: 1,
                start: 11,
                length: (i64::MAX as usize).saturating_sub(11),
            },
            &[vec![1, 22, 3001]],
            &[1, 11, 3001],
        );
        assert_eq!(shape, vec![1, 11, 3001]);
    }
}
