// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder for `TensorOpKind::Linear` — fully-connected layer: `y = x @ W^T + b`.
//!
//! Constructs a tensor-level IR graph for a standard linear (dense / FC) layer.
//! Weight is a fixed model parameter (shape `[out_features, in_features]`),
//! input is the bounded variable (shape `[*, in_features]`).
//!
//! Maps to NY's `LinearLayer::new(weight, bias)` for IBP/CROWN verification.
//!
//! Design doc: `designs/2026-03-01-matmul-linear-integration.md` (Direction 2).
//! Issue: #730.

use crate::tensor_ir::{
    TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a `TensorKernelDef` for a fully-connected linear layer.
///
/// Constructs a 3-node (or 4-node with bias) graph:
/// - `%0 = Input("data", [in_features])`
/// - `%1 = Input("weight", [out_features, in_features])`
/// - `%2 = Input("bias", [out_features])` (if `has_bias`)
/// - `%N = Linear { input: %0, weight: %1, bias: Some(%2) }`
///
/// # Parameters
///
/// - `name`: Kernel name for the generated code.
/// - `in_features`: Size of the input's last dimension (contracted axis).
/// - `out_features`: Size of the output's last dimension (weight rows).
/// - `has_bias`: Whether to include a bias addition.
///
/// # Errors
///
/// Returns `TensorIRLayerError::LinearInputScalar` if `in_features` is 0.
/// Returns `TensorIRLayerError::LinearWeightNotMatrix` if `out_features` is 0.
pub fn build_linear(
    name: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    if in_features == 0 {
        return Err(TensorIRLayerError::LinearInputScalar.into());
    }
    if out_features == 0 {
        return Err(TensorIRLayerError::LinearWeightNotMatrix {
            shape: vec![0, in_features],
        }
        .into());
    }

    let mut nodes = Vec::new();

    // %0: input data — shape [in_features] for the simplest single-vector case.
    // Higher-rank inputs (e.g. [batch, in_features]) are handled by validation
    // at graph construction time, but the builder creates the minimal graph.
    let data_shape = vec![in_features];
    nodes.push(TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: crate::input_names::DATA.into(),
            shape: data_shape.clone(),
        },
        data_shape,
    ));

    // %1: weight matrix — shape [out_features, in_features] (PyTorch convention).
    let weight_shape = vec![out_features, in_features];
    nodes.push(TensorNode::new(
        TensorNodeId::new(1),
        TensorOpKind::Input {
            name: "weight".into(),
            shape: weight_shape.clone(),
        },
        weight_shape,
    ));

    // %2 (optional): bias vector — shape [out_features].
    let bias_node = if has_bias {
        let bias_shape = vec![out_features];
        let bias_id = TensorNodeId::new(nodes.len());
        nodes.push(TensorNode::new(
            bias_id,
            TensorOpKind::Input {
                name: "bias".into(),
                shape: bias_shape.clone(),
            },
            bias_shape,
        ));
        Some(bias_id)
    } else {
        None
    };

    // Linear operation node — output shape [out_features].
    let output_shape = vec![out_features];
    let linear_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        linear_id,
        TensorOpKind::Linear {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_node,
        },
        output_shape,
    ));

    Ok(TensorKernelDef::new(name, nodes, linear_id))
}

/// Build a batched `TensorKernelDef` for a fully-connected linear layer.
///
/// Like `build_linear` but with an explicit batch dimension:
/// - `%0 = Input("data", [batch_size, in_features])`
/// - Output shape: `[batch_size, out_features]`
///
/// This matches the typical inference usage where input is `[B, in_features]`.
pub fn build_linear_batched(
    name: &str,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    if in_features == 0 {
        return Err(TensorIRLayerError::LinearInputScalar.into());
    }
    if out_features == 0 {
        return Err(TensorIRLayerError::LinearWeightNotMatrix {
            shape: vec![0, in_features],
        }
        .into());
    }
    if batch_size == 0 {
        return Err(TensorIRError::EmptyDimension(vec![0, in_features]));
    }

    let mut nodes = Vec::new();

    // %0: input data — shape [batch_size, in_features].
    let data_shape = vec![batch_size, in_features];
    nodes.push(TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: crate::input_names::DATA.into(),
            shape: data_shape.clone(),
        },
        data_shape,
    ));

    // %1: weight matrix — shape [out_features, in_features].
    let weight_shape = vec![out_features, in_features];
    nodes.push(TensorNode::new(
        TensorNodeId::new(1),
        TensorOpKind::Input {
            name: "weight".into(),
            shape: weight_shape.clone(),
        },
        weight_shape,
    ));

    // %2 (optional): bias vector — shape [out_features].
    let bias_node = if has_bias {
        let bias_shape = vec![out_features];
        let bias_id = TensorNodeId::new(nodes.len());
        nodes.push(TensorNode::new(
            bias_id,
            TensorOpKind::Input {
                name: "bias".into(),
                shape: bias_shape.clone(),
            },
            bias_shape,
        ));
        Some(bias_id)
    } else {
        None
    };

    // Linear operation node — output shape [batch_size, out_features].
    let output_shape = vec![batch_size, out_features];
    let linear_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        linear_id,
        TensorOpKind::Linear {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_node,
        },
        output_shape,
    ));

    Ok(TensorKernelDef::new(name, nodes, linear_id))
}

#[cfg(test)]
#[path = "linear_tests.rs"]
mod tests;
