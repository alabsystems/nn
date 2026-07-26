// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BatchNorm tensor op → NY `BatchNormLayer` translation.
//!
//! Maps `TensorOpKind::BatchNorm` (frozen inference-mode batch normalization)
//! to NY's native `BatchNormLayer`, which pre-computes:
//!   `scale = gamma / sqrt(var + eps)`
//!   `bias = beta - mean * scale`
//!
//! This replaces what would otherwise be a 6-op decomposition
//! (sub → add_eps → sqrt → recip → mul → add) with a single native layer,
//! yielding tighter IBP and CROWN bounds.
//!
//! Part of #1045 (NY layer utilization gap).

use ny_propagate::layers::BatchNormLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::TensorNodeValue;

/// Extract a 1-D `ArrayD<f32>` from a `WeightTensor` node, validating it
/// has exactly `expected_len` elements.
fn extract_1d_param(
    node_values: &[TensorNodeValue],
    id: &TensorNodeId,
    expected_len: usize,
    op_name: &str,
    param_name: &str,
) -> Result<ndarray::ArrayD<f32>, VerifyError> {
    match get_value(node_values, id.index(), &format!("{op_name} {param_name}"))? {
        TensorNodeValue::WeightTensor(arr) => {
            if arr.len() != expected_len {
                return Err(VerifyError::WeightValidation {
                    op: "BatchNorm",
                    reason: format!(
                        "{param_name} must have {expected_len} elements, got {}",
                        arr.len()
                    ),
                });
            }
            Ok(arr.clone())
        }
        other => Err(VerifyError::WeightValidation {
            op: "BatchNorm",
            reason: format!("{param_name} must be a WeightTensor, got {other:?}"),
        }),
    }
}

/// Extract a scalar epsilon value from a `Constant` or `WeightTensor` node.
fn extract_eps(node_values: &[TensorNodeValue], eps_id: &TensorNodeId) -> Result<f32, VerifyError> {
    match get_value(node_values, eps_id.index(), "BatchNorm eps")? {
        TensorNodeValue::Constant(val) => Ok(val.get()),
        TensorNodeValue::WeightTensor(arr) => {
            if arr.len() != 1 {
                return Err(VerifyError::WeightValidation {
                    op: "BatchNorm",
                    reason: format!("eps must be scalar, got shape {:?}", arr.shape()),
                });
            }
            Ok(arr[[0]])
        }
        other => Err(VerifyError::WeightValidation {
            op: "BatchNorm",
            reason: format!("eps must be Constant or scalar WeightTensor, got {other:?}"),
        }),
    }
}

/// Translate a `TensorOpKind::BatchNorm` node to NY's native `BatchNormLayer`.
///
/// All parameter nodes (running_mean, running_var, weight, bias) must be
/// `WeightTensor` — these are frozen running statistics and affine parameters.
/// The eps node must be a `Constant` scalar.
///
/// For variable input: creates a `BatchNormLayer` node in the NY graph.
/// For constant input: computes `gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]`
/// per channel. Returns a scalar constant if all channels agree.
pub(super) fn translate_batch_norm(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    running_mean: &TensorNodeId,
    running_var: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &TensorNodeId,
    eps_id: &TensorNodeId,
    node_values: &[TensorNodeValue],
    all_nodes: &[nn_dsl::tensor_ir::TensorNode],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "BatchNorm input")?;

    // Infer num_channels from the BN param length, not a hardcoded input axis.
    // The four BN params (running_mean/_var, weight, bias) are per-channel
    // rank-1 `[C]`, so running_mean's length is the authoritative channel
    // count. We then require a plausible input channel axis to match it (axis 0
    // = channels-first / no batch, axis 1 = batched NCHW) — exactly the axes
    // NY's runtime `detect_input_layout` considers, and mirroring nn-dsl's
    // validate_batch_norm. The old `input_shape[1]` heuristic wrongly read the
    // spatial dim as channels for valid channels-first rank-3 inputs like
    // [C,S,S] with `[C]` params, producing a false-positive rejection.
    //
    // Soundness: strictly a false-positive removal — we still reject non-rank-1
    // running_mean and any channel count that matches no candidate axis, so we
    // never accept a layout NY cannot resolve.
    let input_shape = &all_nodes[input.index()].shape;
    if input_shape.len() < 2 {
        return Err(VerifyError::UnsupportedOp(
            "BatchNorm input must have at least 2 dimensions".into(),
        ));
    }
    let mean_shape = &all_nodes[running_mean.index()].shape;
    if mean_shape.len() != 1 {
        return Err(VerifyError::UnsupportedOp(
            "BatchNorm running_mean must be rank-1 [C]".into(),
        ));
    }
    let num_channels = mean_shape[0];
    if input_shape[0] != num_channels && input_shape[1] != num_channels {
        return Err(VerifyError::UnsupportedOp(format!(
            "BatchNorm channel count {num_channels} matches no input axis (input shape {input_shape:?})"
        )));
    }

    let eps = extract_eps(node_values, eps_id)?;
    let gamma = extract_1d_param(node_values, weight, num_channels, "BatchNorm", "weight")?;
    let beta = extract_1d_param(node_values, bias, num_channels, "BatchNorm", "bias")?;
    let mean = extract_1d_param(
        node_values,
        running_mean,
        num_channels,
        "BatchNorm",
        "running_mean",
    )?;
    let var = extract_1d_param(
        node_values,
        running_var,
        num_channels,
        "BatchNorm",
        "running_var",
    )?;

    match input_val {
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            let layer = BatchNormLayer::new(&gamma, &beta, &mean, &var, eps).map_err(|e| {
                VerifyError::WeightValidation {
                    op: "BatchNorm",
                    reason: format!("NY layer creation: {e}"),
                }
            })?;
            add_unary_node(&node_name, Layer::BatchNorm(layer), input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(const_val) => {
            // Compute full BatchNorm formula per channel:
            //   output[c] = gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]
            let x = const_val.get();
            let first = gamma[[0]] * (x - mean[[0]]) / (var[[0]] + eps).sqrt() + beta[[0]];
            for c in 1..num_channels {
                let out_c = gamma[[c]] * (x - mean[[c]]) / (var[[c]] + eps).sqrt() + beta[[c]];
                if out_c != first {
                    return Err(VerifyError::UnsupportedOp(
                        "BatchNorm constant-fold: channels produce different values".into(),
                    ));
                }
            }
            super::checked_tensor_constant(first, "BatchNorm constant-fold")
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "weight tensor cannot be used as BatchNorm input".into(),
        )),
    }
}
