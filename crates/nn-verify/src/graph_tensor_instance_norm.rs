// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! InstanceNorm1d tensor op → NY translation.
//!
//! Extracted from `graph_tensor.rs` to stay under the 500-line file limit.

use ny_propagate::layers::InstanceNorm1dLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, FiniteF32};

use super::norm_util::{extract_norm_param, validate_norm_shape_and_eps};
use super::{TensorNodeValue, TensorTranslationContext};

/// Translate an `InstanceNorm1d` tensor op to NY's native `InstanceNorm1dLayer`.
///
/// Supports both non-affine (gamma=None, beta=None → gamma=1, beta=0 defaults)
/// and affine (gamma=Some, beta=Some → per-channel scale/shift) modes.
///
/// For variable input: creates an `InstanceNorm1dLayer` with gamma/beta params.
/// For constant input: InstanceNorm of a constant tensor is 0 (or beta if affine).
///
/// NY's `InstanceNorm1dLayer` always normalizes over the last axis. This
/// function validates that `axis` matches the last dimension of the input shape;
/// non-last-axis normalization is rejected as unsupported.
pub(super) fn translate_instance_norm_1d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    gamma: Option<&TensorNodeId>,
    beta: Option<&TensorNodeId>,
    node_values: &[TensorNodeValue],
    all_nodes: &[nn_dsl::tensor_ir::TensorNode],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let (input_val, eps_val, input_shape) =
        validate_norm_shape_and_eps(node_values, all_nodes, input, eps, axis, "InstanceNorm1d")?;

    // Extract gamma/beta arrays if affine params are present.
    let num_channels = input_shape[input_shape.len() - 2];
    let affine_params = match (gamma, beta) {
        (Some(g), Some(b)) => {
            let gamma_arr =
                extract_norm_param(node_values, g, num_channels, "InstanceNorm1d", "gamma")?;
            let beta_arr =
                extract_norm_param(node_values, b, num_channels, "InstanceNorm1d", "beta")?;
            Some((gamma_arr, beta_arr))
        }
        // Affine requires both gamma AND beta; partial is treated as no affine.
        (None, _) | (_, None) => None,
    };

    match input_val {
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            let norm_mode = ctx.norm_mode;
            let layer = match affine_params {
                Some((gamma_arr, beta_arr)) => Layer::InstanceNorm1d(
                    InstanceNorm1dLayer::new(gamma_arr, beta_arr, eps_val)?
                        .with_forward_mode(norm_mode.forward_mode())
                        .with_crown_mode(norm_mode.crown_mode()),
                ),
                None => Layer::InstanceNorm1d(
                    InstanceNorm1dLayer::new_default(num_channels, eps_val)?
                        .with_forward_mode(norm_mode.forward_mode())
                        .with_crown_mode(norm_mode.crown_mode()),
                ),
            };
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(_) => {
            // InstanceNorm of a constant tensor: all values are identical per channel,
            // so (x - mean) = 0 for every element.
            // With affine: output = gamma * 0 + beta = beta.
            // Without affine: output = 0.
            // When beta is uniform (ConstantScalar binding), beta_arr[0]
            // is the correct scalar. Non-uniform beta (ConstantTensor)
            // requires per-element output which TensorNodeValue::Constant
            // cannot represent — reject this degenerate case.
            let result = match &affine_params {
                Some((_gamma, beta_arr)) => {
                    let first = beta_arr[0];
                    if !beta_arr.iter().all(|&v| v == first) {
                        return Err(VerifyError::UnsupportedOp(
                            "InstanceNorm1d constant-fold with non-uniform beta unsupported".into(),
                        ));
                    }
                    first
                }
                None => 0.0,
            };
            Ok(TensorNodeValue::Constant(FiniteF32::new(result)?))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "weight tensor cannot be used as InstanceNorm1d input".into(),
        )),
    }
}
