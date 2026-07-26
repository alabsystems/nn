// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape inference for the tensor IR.
//!
//! Extracted from `tensor_ir_validate.rs` to stay under the 500-line limit.
//! Convolution shape inference lives in [`shape_conv`].

use super::super::{TensorIRError, TensorIRLayerError, TensorNode, TensorOpKind};

#[path = "tensor_ir_validate_shape_conv.rs"]
mod shape_conv;
use shape_conv::{
    conv1d_output_shape, conv2d_output_shape, conv_transpose1d_output_shape,
    conv_transpose2d_output_shape, pool2d_output_shape,
};

/// Compute the output shape of a tensor operation.
pub(crate) fn compute_output_shape(
    node: &TensorNode,
    all_nodes: &[TensorNode],
) -> Result<Vec<usize>, TensorIRError> {
    match &node.kind {
        TensorOpKind::Input { shape, .. } => Ok(shape.clone()),

        TensorOpKind::Reshape { target_shape, .. } => Ok(target_shape.clone()),

        TensorOpKind::AxisSelect { input, axis, .. } => {
            let input_shape = &all_nodes[input.index()].shape;
            let mut output_shape = input_shape.clone();
            output_shape.remove(*axis);
            if output_shape.is_empty() {
                output_shape.push(1);
            }
            Ok(output_shape)
        }

        TensorOpKind::Stack { inputs, axis } => {
            if inputs.is_empty() {
                return Err(TensorIRError::EmptyStack);
            }
            let mut output_shape = all_nodes[inputs[0].index()].shape.clone();
            output_shape.insert(*axis, inputs.len());
            Ok(output_shape)
        }

        TensorOpKind::Concat { inputs, axis } => {
            if inputs.len() < 2 {
                return Err(TensorIRLayerError::EmptyConcat.into());
            }
            let mut output_shape = all_nodes[inputs[0].index()].shape.clone();
            let concat_sum: usize = inputs
                .iter()
                .map(|id| all_nodes[id.index()].shape[*axis])
                .sum();
            output_shape[*axis] = concat_sum;
            Ok(output_shape)
        }

        TensorOpKind::Reduce {
            input,
            axis,
            keepdim,
            ..
        } => {
            let input_shape = &all_nodes[input.index()].shape;
            let mut output_shape = input_shape.clone();
            if *keepdim {
                // Replace the reduced axis with size 1.
                output_shape[*axis] = 1;
            } else {
                output_shape.remove(*axis);
                if output_shape.is_empty() {
                    // Reducing a 1-D tensor produces a scalar (0-D) — represent as [1]
                    output_shape.push(1);
                }
            }
            Ok(output_shape)
        }

        TensorOpKind::Elementwise { inputs, .. } => {
            // Output shape = first input shape. All inputs must have matching
            // shapes (enforced by validate_node). compute_output_shape is also
            // called during validation to check node.shape consistency, so
            // guard against empty inputs here.
            if inputs.is_empty() {
                return Err(TensorIRError::EmptyGraph);
            }
            let first_shape = &all_nodes[inputs[0].index()].shape;
            for (i, input_id) in inputs[1..].iter().enumerate() {
                let input_shape = &all_nodes[input_id.index()].shape;
                if input_shape != first_shape {
                    return Err(TensorIRError::ElementwiseShapeMismatch {
                        expected: first_shape.clone(),
                        found: input_shape.clone(),
                        index: i + 1,
                    });
                }
            }
            Ok(first_shape.clone())
        }

        TensorOpKind::Broadcast { target_shape, .. } => Ok(target_shape.clone()),

        TensorOpKind::Narrow {
            input,
            axis,
            length,
            ..
        } => {
            let input_shape = &all_nodes[input.index()].shape;
            let mut output_shape = input_shape.clone();
            output_shape[*axis] = *length;
            Ok(output_shape)
        }

        TensorOpKind::ZeroPad1d {
            input,
            pad_left,
            pad_right,
        } => {
            let input_shape = &all_nodes[input.index()].shape;
            let mut output_shape = input_shape.clone();
            let last = output_shape.len() - 1;
            // Overflow already checked by validate_zero_pad_1d
            output_shape[last] = input_shape[last] + pad_left + pad_right;
            Ok(output_shape)
        }

        // Shape-preserving: output shape = input shape (elementwise, normalization).
        TensorOpKind::Softmax { input, .. }
        | TensorOpKind::LogSoftmax { input, .. }
        | TensorOpKind::Sigmoid { input }
        | TensorOpKind::Silu { input }
        | TensorOpKind::Gelu { input }
        | TensorOpKind::GeluErf { input }
        | TensorOpKind::Relu { input }
        | TensorOpKind::LeakyRelu { input, .. }
        | TensorOpKind::Elu { input, .. }
        | TensorOpKind::Tanh { input }
        | TensorOpKind::Softplus { input }
        | TensorOpKind::Exp { input }
        | TensorOpKind::InstanceNorm1d { input, .. }
        | TensorOpKind::RmsNorm { input, .. }
        | TensorOpKind::AdaIN1d { input, .. }
        | TensorOpKind::LayerNorm { input, .. }
        | TensorOpKind::BatchNorm { input, .. } => Ok(all_nodes[input.index()].shape.clone()),

        // Binary ops: output shape = left input shape. Equality enforced by validate_node.
        TensorOpKind::BinaryAdd { left, .. } | TensorOpKind::BinaryMul { left, .. } => {
            Ok(all_nodes[left.index()].shape.clone())
        }

        TensorOpKind::Linear { input, weight, .. } => {
            // Linear: output shape = input shape with last dim replaced by out_features.
            let input_shape = &all_nodes[input.index()].shape;
            let weight_shape = &all_nodes[weight.index()].shape;
            let out_features = weight_shape[0];
            let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
            output_shape.push(out_features);
            Ok(output_shape)
        }

        TensorOpKind::MatMul {
            left,
            right,
            transpose_right,
            ..
        } => {
            // MatMul: [*, M, K] @ [*, K, N] -> [*, M, N]
            // transpose_right: [*, M, K] @ [*, N, K]^T -> [*, M, N]
            let left_shape = &all_nodes[left.index()].shape;
            let right_shape = &all_nodes[right.index()].shape;
            let m = left_shape[left_shape.len() - 2];
            let n = if *transpose_right {
                right_shape[right_shape.len() - 2]
            } else {
                right_shape[right_shape.len() - 1]
            };
            // Output: batch dims from left + [M, N]
            let mut output_shape = left_shape[..left_shape.len() - 2].to_vec();
            output_shape.push(m);
            output_shape.push(n);
            Ok(output_shape)
        }

        TensorOpKind::Conv1d {
            input,
            weight,
            stride,
            padding,
            dilation,
            ..
        } => {
            let input_node = all_nodes
                .get(input.index())
                .ok_or(TensorIRError::InvalidNodeRef(*input))?;
            let weight_node = all_nodes
                .get(weight.index())
                .ok_or(TensorIRError::InvalidNodeRef(*weight))?;
            conv1d_output_shape(
                &input_node.shape,
                &weight_node.shape,
                *stride,
                *padding,
                *dilation,
            )
        }

        TensorOpKind::Conv2d {
            input,
            weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            ..
        } => {
            let input_node = all_nodes
                .get(input.index())
                .ok_or(TensorIRError::InvalidNodeRef(*input))?;
            let weight_node = all_nodes
                .get(weight.index())
                .ok_or(TensorIRError::InvalidNodeRef(*weight))?;
            conv2d_output_shape(
                &input_node.shape,
                &weight_node.shape,
                *stride_h,
                *stride_w,
                *padding_h,
                *padding_w,
                *dilation_h,
                *dilation_w,
            )
        }

        TensorOpKind::ConvTranspose1d {
            input,
            weight,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
            ..
        } => {
            let input_node = all_nodes
                .get(input.index())
                .ok_or(TensorIRError::InvalidNodeRef(*input))?;
            let weight_node = all_nodes
                .get(weight.index())
                .ok_or(TensorIRError::InvalidNodeRef(*weight))?;
            conv_transpose1d_output_shape(
                &input_node.shape,
                &weight_node.shape,
                *stride,
                *padding,
                *dilation,
                *groups,
                *output_padding,
            )
        }

        TensorOpKind::ConvTranspose2d {
            input,
            weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            output_padding_h,
            output_padding_w,
            ..
        } => {
            let input_node = all_nodes
                .get(input.index())
                .ok_or(TensorIRError::InvalidNodeRef(*input))?;
            let weight_node = all_nodes
                .get(weight.index())
                .ok_or(TensorIRError::InvalidNodeRef(*weight))?;
            conv_transpose2d_output_shape(
                &input_node.shape,
                &weight_node.shape,
                *stride_h,
                *stride_w,
                *padding_h,
                *padding_w,
                *dilation_h,
                *dilation_w,
                *groups,
                *output_padding_h,
                *output_padding_w,
            )
        }
        TensorOpKind::AvgPool2d { input, params } | TensorOpKind::MaxPool2d { input, params } => {
            let input_shape = &all_nodes[input.index()].shape;
            pool2d_output_shape(
                input_shape,
                params.kernel_h,
                params.kernel_w,
                params.stride_h,
                params.stride_w,
                params.padding_h,
                params.padding_w,
            )
        }

        TensorOpKind::Embedding { input, weight, .. } => {
            // Embedding: indices shape [*] -> output shape [*, embedding_dim]
            let input_shape = &all_nodes[input.index()].shape;
            let weight_shape = &all_nodes[weight.index()].shape;
            let embedding_dim = weight_shape[1];
            let mut output_shape = input_shape.clone();
            output_shape.push(embedding_dim);
            Ok(output_shape)
        }
        TensorOpKind::Attention { q, v, .. } => {
            // Output: Q batch dims + [T, D_v]
            // Q shape: [*, T, D], V shape: [*, T_kv, D_v]
            let q_shape = &all_nodes[q.index()].shape;
            let v_shape = &all_nodes[v.index()].shape;
            let t = q_shape[q_shape.len() - 2];
            let d_v = v_shape[v_shape.len() - 1];
            let mut output_shape = q_shape[..q_shape.len() - 2].to_vec();
            output_shape.push(t);
            output_shape.push(d_v);
            Ok(output_shape)
        }

        TensorOpKind::Transpose { input, axes } => {
            // Output shape is the input shape with dimensions permuted by axes.
            let input_shape = &all_nodes[input.index()].shape;
            let output_shape: Vec<usize> = axes.iter().map(|&a| input_shape[a]).collect();
            Ok(output_shape)
        }

        TensorOpKind::Lstm {
            input,
            hidden_state,
            ..
        } => {
            // LSTM output shape: input's leading dims + hidden_size.
            //
            // Single-timestep: input [B, I], hidden [B, H] → output [B, H].
            // Sequence (different rank): input [S, B, I], hidden [B, H] → [S, B, H].
            // Sequence (same rank): input [S, B, I], hidden [1, B, H] → [S, B, H].
            //   (initial state has dim-0 = 1 for num_layers*num_directions = 1.)
            let h_shape = &all_nodes[hidden_state.index()].shape;
            let in_shape = &all_nodes[input.index()].shape;
            let hidden_size = h_shape[h_shape.len() - 1];
            if in_shape.len() > h_shape.len() {
                // Sequence case (rank differs): prepend leading dims from input.
                let extra = in_shape.len() - h_shape.len();
                let mut out = in_shape[..extra].to_vec();
                out.extend_from_slice(h_shape);
                Ok(out)
            } else {
                // Same rank: use input's leading dims + hidden_size.
                // Handles both [B, I] → [B, H] and [S, B, I] → [S, B, H].
                let mut out = in_shape[..in_shape.len() - 1].to_vec();
                out.push(hidden_size);
                Ok(out)
            }
        }

        TensorOpKind::GatedDeltaNet { q, v, .. } => {
            // Output o_t = scale * q_t @ S_t has shape [*, H, V].
            // Q shape: [*, H, K], V shape: [*, H, V].
            // Output: Q batch dims + [H, V_dim] where V_dim = V's last dim.
            let q_shape = &all_nodes[q.index()].shape;
            let v_shape = &all_nodes[v.index()].shape;
            let v_dim = v_shape[v_shape.len() - 1];
            // Output: [*, H, V] — same leading dims as Q, last dim from V.
            let mut output_shape = q_shape[..q_shape.len() - 1].to_vec();
            output_shape.push(v_dim);
            Ok(output_shape)
        }

        TensorOpKind::IndexSelect {
            input,
            indices,
            dim,
        } => {
            // IndexSelect replaces dimension `dim` with the number of indices.
            let input_shape = &all_nodes[input.index()].shape;
            let indices_shape = &all_nodes[indices.index()].shape;
            let num_indices = indices_shape.iter().product::<usize>();
            let mut output_shape = input_shape.clone();
            output_shape[*dim] = num_indices;
            Ok(output_shape)
        }

        TensorOpKind::Gather { indices, .. } => {
            // Gather output shape matches the indices tensor shape.
            Ok(all_nodes[indices.index()].shape.clone())
        }
    }
}
