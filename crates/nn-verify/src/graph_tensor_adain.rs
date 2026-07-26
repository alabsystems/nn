// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdaIN1d tensor op → NY translation.
//!
//! Maps `TensorOpKind::AdaIN1d` to `Layer::AdaIN1d(AdaIN1dLayer)`.
//! AdaIN1d = InstanceNorm + style-conditioned affine (gamma, beta).

use ny_propagate::layers::{AdaIN1dLayer, InstanceNorm1dLayer};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, FiniteF32};

use super::norm_util::{extract_norm_param, validate_norm_shape_and_eps};
use super::{TensorNodeValue, TensorTranslationContext};

/// Translate an `AdaIN1d` tensor op to NY's native `AdaIN1dLayer`.
///
/// Style gamma/beta must be constant. Eps must be constant scalar.
/// NY's `AdaIN1dLayer` normalizes over the last axis only.
pub(super) fn translate_adain1d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    style_gamma: &TensorNodeId,
    style_beta: &TensorNodeId,
    node_values: &[TensorNodeValue],
    all_nodes: &[nn_dsl::tensor_ir::TensorNode],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let (input_val, eps_val, input_shape) =
        validate_norm_shape_and_eps(node_values, all_nodes, input, eps, axis, "AdaIN1d")?;

    let num_channels = input_shape[input_shape.len() - 2];
    let gamma_val = extract_norm_param(
        node_values,
        style_gamma,
        num_channels,
        "AdaIN1d",
        "style_gamma",
    )?;
    let beta_val = extract_norm_param(
        node_values,
        style_beta,
        num_channels,
        "AdaIN1d",
        "style_beta",
    )?;

    match input_val {
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            let norm_mode = ctx.norm_mode;
            let inner_norm = InstanceNorm1dLayer::new_default(num_channels, eps_val)?
                .with_forward_mode(norm_mode.forward_mode())
                .with_crown_mode(norm_mode.crown_mode());
            let layer = Layer::AdaIN1d(
                AdaIN1dLayer::new(inner_norm, gamma_val, beta_val)?
                    .with_forward_mode(norm_mode.forward_mode())
                    .with_crown_mode(norm_mode.crown_mode()),
            );
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(_) => {
            // AdaIN1d of a constant: InstanceNorm(constant) = 0, then
            // output = gamma * 0 + beta = beta.
            // When beta is uniform (ConstantScalar binding), beta_val[0]
            // is the correct scalar. Non-uniform beta (ConstantTensor)
            // requires per-element output which TensorNodeValue::Constant
            // cannot represent — reject this degenerate case.
            let first = beta_val[0];
            if !beta_val.iter().all(|&v| v == first) {
                return Err(VerifyError::UnsupportedOp(
                    "AdaIN1d constant-fold with non-uniform beta unsupported".into(),
                ));
            }
            Ok(TensorNodeValue::Constant(FiniteF32::new(first)?))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "weight tensor cannot be used as AdaIN1d input".into(),
        )),
    }
}
