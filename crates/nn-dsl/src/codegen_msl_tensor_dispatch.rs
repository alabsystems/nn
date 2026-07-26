// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-node dispatch step dispatch for `build_dispatch_plan`.
//!
//! Extracted from `codegen_msl_tensor.rs` (#827 Direction 1) to keep the
//! dispatch planner under 400 lines. Contains the full match over
//! `TensorOpKind` variants to build `DispatchStep`s.

use crate::ir::ScalarType;
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorOpKind};

use super::activation::build_unary_activation;
use super::{node_shape, ops, shape_total, DispatchStep, TensorMSLCodegenError};

/// Build a dispatch step for a single tensor node.
///
/// Returns `Ok(None)` for `Input` nodes (no dispatch needed).
/// Returns `Ok(Some(step))` for ops that produce a dispatch step.
/// Returns `Err` for unsupported or unexpanded ops.
pub(super) fn build_step_for_node(
    effective: &TensorKernelDef,
    node: &TensorNode,
    dtype: ScalarType,
) -> Result<Option<DispatchStep>, TensorMSLCodegenError> {
    match &node.kind {
        TensorOpKind::Input { .. } => Ok(None),

        TensorOpKind::Reshape { input, .. } => Ok(Some(DispatchStep::Reshape {
            input: *input,
            output: node.id,
        })),

        TensorOpKind::AxisSelect { input, axis, index } => {
            let input_shape = node_shape(effective, *input)?.to_vec();
            Ok(Some(DispatchStep::AxisSelect {
                kernel_name: format!("{}_axis_select_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                output: node.id,
                input_shape,
                axis: *axis,
                index: *index,
            }))
        }

        TensorOpKind::Stack { inputs, axis } => {
            let input_shape = node_shape(effective, inputs[0])?.to_vec();
            Ok(Some(DispatchStep::Stack {
                kernel_name: format!("{}_stack_n{}", effective.name, node.id.index()),
                dtype,
                inputs: inputs.clone(),
                output: node.id,
                input_shape,
                axis: *axis,
            }))
        }

        TensorOpKind::Concat { inputs, axis } => {
            let first_shape = node_shape(effective, inputs[0])?.to_vec();
            let input_axis_sizes: Vec<usize> = inputs
                .iter()
                .map(|id| node_shape(effective, *id).map(|s| s[*axis]))
                .collect::<Result<_, _>>()?;
            Ok(Some(DispatchStep::Concat {
                kernel_name: format!("{}_concat_n{}", effective.name, node.id.index()),
                dtype,
                inputs: inputs.clone(),
                output: node.id,
                first_input_shape: first_shape,
                input_axis_sizes,
                axis: *axis,
            }))
        }

        TensorOpKind::InstanceNorm1d { .. } => Err(TensorMSLCodegenError::UnexpandedNormOp {
            node_id: node.id,
            op_name: "InstanceNorm1d",
        }),

        TensorOpKind::Reduce {
            op,
            input,
            axis,
            keepdim,
        } => Ok(Some(ops::build_reduce_step(
            effective, node.id, op, input, *axis, *keepdim, dtype,
        )?)),

        TensorOpKind::Elementwise {
            kernel: scalar_k,
            inputs,
        } => {
            let total_elements = shape_total(&node.shape)?;
            let mut renamed = scalar_k.clone();
            renamed.name = format!("{}_{}_n{}", effective.name, scalar_k.name, node.id.index());
            Ok(Some(DispatchStep::Elementwise {
                kernel_name: format!("{}_kernel", renamed.name),
                scalar_kernel: renamed,
                inputs: inputs.clone(),
                output: node.id,
                total_elements,
            }))
        }

        TensorOpKind::Broadcast {
            input,
            target_shape,
            alignment,
        } => {
            let input_shape = node_shape(effective, *input)?.to_vec();
            // Identity broadcast: input already matches target shape.
            // Convert to zero-cost Reshape (buffer alias, no GPU dispatch).
            if input_shape == *target_shape {
                return Ok(Some(DispatchStep::Reshape {
                    input: *input,
                    output: node.id,
                }));
            }
            let total_elements = shape_total(target_shape)?;
            Ok(Some(DispatchStep::Broadcast {
                kernel_name: format!("{}_broadcast_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                output: node.id,
                input_shape,
                output_shape: target_shape.clone(),
                total_elements,
                alignment: *alignment,
            }))
        }

        TensorOpKind::Conv1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        } => Ok(Some(ops::build_conv1d_step(
            effective,
            node.id,
            &node.shape,
            input,
            weight,
            bias,
            *stride,
            *padding,
            *dilation,
            *groups,
            dtype,
        )?)),

        TensorOpKind::Conv2d {
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
        } => Ok(Some(ops::build_conv2d_step(
            effective,
            node.id,
            &node.shape,
            input,
            weight,
            bias,
            *stride_h,
            *stride_w,
            *padding_h,
            *padding_w,
            *dilation_h,
            *dilation_w,
            *groups,
            dtype,
        )?)),

        TensorOpKind::ConvTranspose1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
        } => Ok(Some(ops::build_conv_transpose_1d_step(
            effective,
            node.id,
            &node.shape,
            input,
            weight,
            bias,
            *stride,
            *padding,
            *dilation,
            *groups,
            *output_padding,
            dtype,
        )?)),

        TensorOpKind::ConvTranspose2d { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "ConvTranspose2d",
            reason: "MSL codegen not yet implemented — use runtime dispatch",
        }),

        TensorOpKind::BinaryAdd { left, right } => {
            let total_elements = shape_total(&node.shape)?;
            Ok(Some(DispatchStep::BinaryAdd {
                kernel_name: format!("{}_binary_add_n{}", effective.name, node.id.index()),
                dtype,
                left: *left,
                right: *right,
                output: node.id,
                total_elements,
                broadcast: None,
            }))
        }

        TensorOpKind::BinaryMul { left, right } => {
            let total_elements = shape_total(&node.shape)?;
            Ok(Some(DispatchStep::BinaryMul {
                kernel_name: format!("{}_binary_mul_n{}", effective.name, node.id.index()),
                dtype,
                left: *left,
                right: *right,
                output: node.id,
                total_elements,
                broadcast: None,
            }))
        }

        TensorOpKind::Transpose { input, axes } => {
            // Identity transpose: axes == [0, 1, 2, ...]. No reordering needed.
            // Convert to zero-cost Reshape (buffer alias, no GPU dispatch).
            let is_identity = axes.iter().enumerate().all(|(i, &a)| a == i);
            if is_identity {
                return Ok(Some(DispatchStep::Reshape {
                    input: *input,
                    output: node.id,
                }));
            }
            let input_shape = node_shape(effective, *input)?.to_vec();
            let total_elements = shape_total(&node.shape)?;
            Ok(Some(DispatchStep::Transpose {
                kernel_name: format!("{}_transpose_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                output: node.id,
                input_shape,
                axes: axes.clone(),
                total_elements,
            }))
        }

        TensorOpKind::Narrow {
            input,
            axis,
            start,
            length,
        } => {
            let input_shape = node_shape(effective, *input)?.to_vec();
            Ok(Some(DispatchStep::Narrow {
                kernel_name: format!("{}_narrow_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                output: node.id,
                input_shape,
                axis: *axis,
                start: *start,
                length: *length,
            }))
        }

        TensorOpKind::ZeroPad1d {
            input,
            pad_left,
            pad_right,
        } => Ok(Some(ops::build_zero_pad_step(
            effective, node.id, input, *pad_left, *pad_right, dtype,
        )?)),

        TensorOpKind::Sigmoid { input } => {
            build_unary_activation(effective, node, *input, "sigmoid", dtype)
        }
        // Silu is a verification-only fused node (enables ny's SwiGLU zonotope
        // tightening). Metal runtime uses the decomposed Sigmoid+BinaryMul or
        // the dedicated `silu_mul` native op, so direct MSL codegen is deferred.
        TensorOpKind::Silu { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "Silu",
            reason: "MSL codegen deferred — verification uses NY Layer::SiLU; \
                     runtime uses decomposed Sigmoid+BinaryMul or silu_mul native op",
        }),
        TensorOpKind::Gelu { input } => {
            build_unary_activation(effective, node, *input, "gelu", dtype)
        }
        TensorOpKind::GeluErf { input } => {
            build_unary_activation(effective, node, *input, "gelu_erf", dtype)
        }
        TensorOpKind::Relu { input } => {
            build_unary_activation(effective, node, *input, "relu", dtype)
        }
        TensorOpKind::LeakyRelu {
            input,
            negative_slope,
        } => {
            let total_elements = shape_total(&node.shape)?;
            Ok(Some(DispatchStep::LeakyRelu {
                kernel_name: format!("{}_leaky_relu_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                output: node.id,
                total_elements,
                negative_slope: *negative_slope,
            }))
        }
        TensorOpKind::Elu { input, alpha } => {
            let total_elements = shape_total(&node.shape)?;
            Ok(Some(DispatchStep::Elu {
                kernel_name: format!("{}_elu_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                output: node.id,
                total_elements,
                alpha: *alpha,
            }))
        }
        TensorOpKind::Tanh { input } => {
            build_unary_activation(effective, node, *input, "tanh", dtype)
        }

        TensorOpKind::Softplus { input } => {
            build_unary_activation(effective, node, *input, "softplus", dtype)
        }
        TensorOpKind::Exp { input } => {
            build_unary_activation(effective, node, *input, "exp", dtype)
        }

        TensorOpKind::Softmax { input, axis } => Ok(Some(ops::build_softmax_step(
            effective, node.id, input, *axis, dtype,
        )?)),

        TensorOpKind::Linear {
            input,
            weight,
            bias,
        } => Ok(Some(ops::build_linear_step(
            effective,
            node.id,
            &node.shape,
            input,
            weight,
            bias,
            dtype,
        )?)),

        TensorOpKind::MatMul {
            left,
            right,
            transpose_right,
            scale,
        } => Ok(Some(ops::build_matmul_step(
            effective,
            node.id,
            &node.shape,
            left,
            right,
            *transpose_right,
            *scale,
            dtype,
        )?)),

        // Norm ops: expanded to primitives by expand_norm_ops() above.
        TensorOpKind::RmsNorm { .. }
        | TensorOpKind::AdaIN1d { .. }
        | TensorOpKind::LayerNorm { .. } => Err(TensorMSLCodegenError::UnexpandedNormOp {
            node_id: node.id,
            op_name: "NormOp",
        }),

        // Attention(Standard): expanded by expand_norm_ops() above (#812).
        TensorOpKind::Attention {
            mask: crate::AttentionMask::Standard,
            ..
        } => Err(TensorMSLCodegenError::UnexpandedNormOp {
            node_id: node.id,
            op_name: "Attention(Standard)",
        }),
        // Attention(Causal): needs causal mask infrastructure (deferred).
        TensorOpKind::Attention { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "Attention(Causal)",
            reason: "causal masking requires causal softmax infrastructure — deferred",
        }),

        // LSTM: expanded to decomposed gates before dispatch planning (#2306).
        // If this arm is reached, the expansion pass was skipped.
        TensorOpKind::Lstm { .. } => Err(TensorMSLCodegenError::UnexpandedNormOp {
            node_id: node.id,
            op_name: "Lstm",
        }),

        // GatedDeltaNet: MSL deferred (#834), verification via decomposed MatMul+BinaryMul+BinaryAdd.
        TensorOpKind::GatedDeltaNet { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "GatedDeltaNet",
            reason:
                "MSL codegen deferred — verification uses decomposed MatMul+BinaryMul+BinaryAdd",
        }),

        // Pool2d: MSL codegen deferred — runtime uses CPU/GPU dispatch.
        TensorOpKind::AvgPool2d { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "AvgPool2d",
            reason: "MSL codegen deferred — use runtime dispatch",
        }),
        TensorOpKind::MaxPool2d { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "MaxPool2d",
            reason: "MSL codegen deferred — use runtime dispatch",
        }),

        // BatchNorm: verification uses NY native BatchNormLayer (#1045).
        // MSL codegen deferred — runtime uses decomposed sub/mul/add primitives.
        TensorOpKind::BatchNorm { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "BatchNorm",
            reason: "MSL codegen deferred — verification uses NY native BatchNormLayer",
        }),

        // LogSoftmax: trace_compile decomposes to softmax + log elementwise.
        // Direct MSL codegen deferred — the decomposed path handles it.
        TensorOpKind::LogSoftmax { .. } => Err(TensorMSLCodegenError::UnsupportedOp {
            op_name: "LogSoftmax",
            reason: "MSL codegen deferred — trace_compile decomposes to softmax + log",
        }),

        TensorOpKind::Embedding { input, weight } => {
            let embedding_dim = node_shape(effective, *weight)?[1];
            let num_indices = shape_total(node_shape(effective, *input)?)?;
            Ok(Some(DispatchStep::Embedding {
                kernel_name: format!("{}_embedding_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                weight: *weight,
                output: node.id,
                embedding_dim,
                num_indices,
                total_elements: shape_total(&node.shape)?,
            }))
        }

        TensorOpKind::IndexSelect {
            input,
            indices,
            dim,
        } => {
            let input_shape = node_shape(effective, *input)?.to_vec();
            let num_indices = shape_total(node_shape(effective, *indices)?)?;
            Ok(Some(DispatchStep::IndexSelect {
                kernel_name: format!("{}_index_select_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                indices: *indices,
                output: node.id,
                dim: *dim,
                input_shape,
                num_indices,
                total_elements: shape_total(&node.shape)?,
            }))
        }

        TensorOpKind::Gather {
            input,
            indices,
            dim,
        } => {
            let input_shape = node_shape(effective, *input)?.to_vec();
            Ok(Some(DispatchStep::Gather {
                kernel_name: format!("{}_gather_n{}", effective.name, node.id.index()),
                dtype,
                input: *input,
                indices: *indices,
                output: node.id,
                dim: *dim,
                input_shape,
                total_elements: shape_total(&node.shape)?,
            }))
        }
    }
}
