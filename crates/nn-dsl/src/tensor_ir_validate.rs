// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation and shape computation for the tensor IR.
//!
//! Extracted from `tensor_ir.rs` to keep that module under 500 lines.
//! This module owns:
//!
//! - [`TensorKernelDef::validate`]: full graph validation.
//! - [`compute_output_shape`]: shape inference per node.
//!
//! Per-operation validators live in submodules (Part of #619):
//! - `structural`: reshape, axis_select, stack, reduce, elementwise, broadcast.
//! - `layers`: InstanceNorm1d, Conv1d, RmsNorm, AdaIN1d.

use super::{
    TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

#[path = "tensor_ir_validate_structural.rs"]
mod structural;

#[path = "tensor_ir_validate_layers.rs"]
mod layers;

#[path = "tensor_ir_validate_layers_ext.rs"]
mod layers_ext;

#[path = "tensor_ir_validate_layers_conv.rs"]
mod layers_conv;

#[path = "tensor_ir_validate_layers_recurrent.rs"]
mod layers_recurrent;

#[path = "tensor_ir_validate_layers_pool.rs"]
mod layers_pool;

#[path = "tensor_ir_validate_shape.rs"]
mod shape;
use shape::compute_output_shape;

impl TensorKernelDef {
    /// Validate the tensor IR graph.
    ///
    /// Checks:
    /// - Node IDs match their array index.
    /// - All node references are in bounds and topologically ordered.
    /// - Reduce axes are within input shape bounds.
    /// - Elementwise param counts match tensor input counts.
    /// - Elementwise inputs have identical shapes.
    /// - Broadcast target shapes are compatible with input shapes.
    /// - No empty dimensions in shapes.
    #[must_use = "returns a Result that may contain an error"]
    pub fn validate(&self) -> Result<(), TensorIRError> {
        if self.nodes.is_empty() {
            return Err(TensorIRError::EmptyGraph);
        }

        // Check node ID invariant
        for (i, node) in self.nodes.iter().enumerate() {
            if node.id != TensorNodeId::new(i) {
                return Err(TensorIRError::MismatchedNodeId {
                    found: node.id,
                    expected_index: i,
                });
            }
        }

        // Validate each node
        for node in &self.nodes {
            self.validate_node(node)?;
        }

        // Validate output reference
        self.check_ref_bounds(self.output)?;

        Ok(())
    }

    fn validate_node(&self, node: &TensorNode) -> Result<(), TensorIRError> {
        let current = node.id;

        match &node.kind {
            TensorOpKind::Input { shape, .. } => validate_shape(shape)?,
            TensorOpKind::Reshape {
                input,
                target_shape,
            } => self.validate_reshape(current, *input, target_shape)?,
            TensorOpKind::AxisSelect { input, axis, index } => {
                self.validate_axis_select(current, *input, *axis, *index)?;
            }
            TensorOpKind::Stack { inputs, axis } => {
                self.validate_stack(current, inputs, *axis)?;
            }
            TensorOpKind::Concat { inputs, axis } => {
                self.validate_concat(current, inputs, *axis)?;
            }
            TensorOpKind::Reduce { input, axis, .. } => {
                self.validate_reduce(current, *input, *axis)?;
            }
            TensorOpKind::Elementwise { kernel, inputs } => {
                self.validate_elementwise(current, kernel, inputs)?;
            }
            TensorOpKind::Broadcast {
                input,
                target_shape,
                alignment,
            } => {
                self.validate_broadcast(current, *input, target_shape, *alignment)?;
            }
            TensorOpKind::InstanceNorm1d {
                input,
                eps,
                axis,
                gamma,
                beta,
            } => {
                self.validate_instance_norm(current, *input, *eps, *axis, *gamma, *beta)?;
            }
            TensorOpKind::Conv1d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => {
                self.validate_conv1d(
                    current, *input, *weight, *bias, *stride, *padding, *dilation, *groups,
                )?;
            }
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
            } => {
                self.validate_conv2d(
                    current,
                    *input,
                    *weight,
                    *bias,
                    *stride_h,
                    *stride_w,
                    *padding_h,
                    *padding_w,
                    *dilation_h,
                    *dilation_w,
                    *groups,
                )?;
            }
            TensorOpKind::RmsNorm {
                input,
                eps,
                axis,
                weight,
            } => {
                self.validate_rms_norm(current, *input, *eps, *axis, *weight)?;
            }
            TensorOpKind::AdaIN1d {
                input,
                eps,
                axis,
                style_gamma,
                style_beta,
            } => {
                self.validate_adain1d(current, *input, *eps, *axis, *style_gamma, *style_beta)?;
            }
            TensorOpKind::ConvTranspose1d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
            } => {
                self.validate_conv_transpose_1d(
                    current,
                    *input,
                    *weight,
                    *bias,
                    *stride,
                    *padding,
                    *dilation,
                    *groups,
                    *output_padding,
                )?;
            }
            TensorOpKind::ConvTranspose2d {
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
                output_padding_h,
                output_padding_w,
            } => {
                self.validate_conv_transpose_2d(
                    current,
                    *input,
                    *weight,
                    *bias,
                    *stride_h,
                    *stride_w,
                    *padding_h,
                    *padding_w,
                    *dilation_h,
                    *dilation_w,
                    *groups,
                    *output_padding_h,
                    *output_padding_w,
                )?;
            }
            TensorOpKind::Linear {
                input,
                weight,
                bias,
            } => {
                self.validate_linear(current, *input, *weight, *bias)?;
            }
            TensorOpKind::MatMul {
                left,
                right,
                transpose_right,
                scale,
            } => {
                self.validate_matmul(current, *left, *right, *transpose_right, *scale)?;
            }
            TensorOpKind::Narrow {
                input,
                axis,
                start,
                length,
            } => {
                self.validate_narrow(current, *input, *axis, *start, *length)?;
            }
            TensorOpKind::Sigmoid { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::Silu { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::Gelu { input } | TensorOpKind::GeluErf { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::Relu { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::LeakyRelu {
                input,
                negative_slope,
            } => {
                self.check_ref(current, *input)?;
                if !negative_slope.is_finite() {
                    return Err(TensorIRLayerError::LeakyReluSlopeInvalid {
                        value: *negative_slope,
                    }
                    .into());
                }
            }
            TensorOpKind::Elu { input, alpha } => {
                self.check_ref(current, *input)?;
                if !alpha.is_finite() {
                    return Err(TensorIRLayerError::EluAlphaInvalid { value: *alpha }.into());
                }
            }
            TensorOpKind::Tanh { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::Softplus { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::Exp { input } => {
                self.check_ref(current, *input)?;
            }
            TensorOpKind::BinaryAdd { left, right } => {
                self.validate_binary_add(current, *left, *right)?;
            }
            TensorOpKind::BinaryMul { left, right } => {
                self.validate_binary_mul(current, *left, *right)?;
            }
            TensorOpKind::Softmax { input, axis } | TensorOpKind::LogSoftmax { input, axis } => {
                self.validate_softmax(current, *input, *axis)?;
            }
            TensorOpKind::ZeroPad1d {
                input,
                pad_left,
                pad_right,
            } => {
                self.validate_zero_pad_1d(current, *input, *pad_left, *pad_right)?;
            }
            TensorOpKind::Embedding { input, weight } => {
                self.validate_embedding(current, *input, *weight)?;
            }
            TensorOpKind::LayerNorm {
                input,
                eps,
                axis,
                weight,
                bias,
            } => {
                self.validate_layer_norm(current, *input, *eps, *axis, *weight, *bias)?;
            }
            TensorOpKind::Attention { q, k, v, scale, .. } => {
                self.validate_attention(current, *q, *k, *v, *scale)?;
            }
            TensorOpKind::Transpose { input, axes } => {
                self.validate_transpose(current, *input, axes)?;
            }
            TensorOpKind::Lstm {
                input,
                hidden_state,
                cell_state,
                weight_ih,
                weight_hh,
                bias,
            } => {
                self.validate_lstm(
                    current,
                    *input,
                    *hidden_state,
                    *cell_state,
                    *weight_ih,
                    *weight_hh,
                    *bias,
                )?;
            }
            TensorOpKind::BatchNorm {
                input,
                running_mean,
                running_var,
                weight,
                bias,
                eps,
            } => {
                self.validate_batch_norm(
                    current,
                    *input,
                    *running_mean,
                    *running_var,
                    *weight,
                    *bias,
                    *eps,
                )?;
            }
            TensorOpKind::AvgPool2d { input, params }
            | TensorOpKind::MaxPool2d { input, params } => {
                self.validate_pool2d(
                    current,
                    *input,
                    params.kernel_h,
                    params.kernel_w,
                    params.stride_h,
                    params.stride_w,
                    params.padding_h,
                    params.padding_w,
                )?;
            }
            TensorOpKind::GatedDeltaNet {
                q,
                k,
                v,
                state,
                gate,
                beta,
                scale,
            } => {
                self.validate_gated_delta_net(current, *q, *k, *v, *state, *gate, *beta, *scale)?;
            }
            TensorOpKind::IndexSelect { input, indices, .. } => {
                self.check_ref(current, *input)?;
                self.check_ref(current, *indices)?;
            }
            TensorOpKind::Gather { input, indices, .. } => {
                self.check_ref(current, *input)?;
                self.check_ref(current, *indices)?;
            }
        }

        // Verify node shape consistency
        let expected_shape = compute_output_shape(node, &self.nodes)?;
        if node.shape != expected_shape {
            return Err(TensorIRError::IncompatibleBroadcast {
                input: expected_shape,
                target: node.shape.clone(),
            });
        }

        Ok(())
    }

    fn check_ref(&self, current: TensorNodeId, target: TensorNodeId) -> Result<(), TensorIRError> {
        if target.index() >= self.nodes.len() {
            return Err(TensorIRError::InvalidNodeRef(target));
        }
        if target.index() >= current.index() {
            return Err(TensorIRError::ForwardRef(current, target));
        }
        Ok(())
    }

    fn check_ref_bounds(&self, id: TensorNodeId) -> Result<(), TensorIRError> {
        if id.index() >= self.nodes.len() {
            return Err(TensorIRError::InvalidNodeRef(id));
        }
        Ok(())
    }
}

/// Validate that a shape has no zero-sized dimensions.
fn validate_shape(shape: &[usize]) -> Result<(), TensorIRError> {
    if shape.contains(&0) {
        return Err(TensorIRError::EmptyDimension(shape.to_vec()));
    }
    Ok(())
}
