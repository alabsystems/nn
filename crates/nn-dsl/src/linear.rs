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

#[cfg(kani)]
mod kani_proofs {
    //! Kani proof harnesses for the Linear builder.
    //!
    //! Same pattern as conv1d/conv_transpose_1d/causal_conv1d builder harnesses:
    //! prove no-panic, output shape positive, and output shape formula correct.

    use super::*;

    /// Prove `build_linear` never panics for bounded params.
    ///
    /// Domain: in_features in [0, 4], out_features in [0, 4].
    /// Reduced from [0, 8] and added unwind(16) for CBMC Vec heap
    /// unwinding tractability (#767 AC3).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn linear_build_no_panic() {
        let in_features: usize = kani::any();
        let out_features: usize = kani::any();
        let has_bias: bool = kani::any();

        kani::assume(in_features <= 4);
        kani::assume(out_features <= 4);

        // Must not panic — returns Err for invalid params.
        let _ = build_linear("kani_test", in_features, out_features, has_bias);
    }

    /// Prove `build_linear` output shape is [out_features] when it succeeds.
    ///
    /// Domain reduced from [1, 512] to [1, 8] for CBMC tractability (#767 AC3).
    #[kani::unwind(1)]
    #[kani::proof]
    fn linear_output_shape_correct() {
        let in_features: usize = kani::any();
        let out_features: usize = kani::any();
        let has_bias: bool = kani::any();

        kani::assume(in_features >= 1 && in_features <= 8);
        kani::assume(out_features >= 1 && out_features <= 8);

        let def = build_linear("kani_test", in_features, out_features, has_bias)
            .expect("valid params must succeed");
        let out_node = &def.nodes[def.output.index()];
        assert_eq!(out_node.shape.len(), 1, "output rank must be 1");
        assert_eq!(
            out_node.shape[0], out_features,
            "output dim must equal out_features"
        );
    }

    /// Prove `build_linear_batched` output shape is [batch, out_features].
    #[kani::unwind(1)]
    #[kani::proof]
    fn linear_batched_output_shape_correct() {
        let batch_size: usize = kani::any();
        let in_features: usize = kani::any();
        let out_features: usize = kani::any();
        let has_bias: bool = kani::any();

        kani::assume(batch_size >= 1 && batch_size <= 64);
        kani::assume(in_features >= 1 && in_features <= 256);
        kani::assume(out_features >= 1 && out_features <= 256);

        let def =
            build_linear_batched("kani_test", batch_size, in_features, out_features, has_bias)
                .expect("valid params must succeed");
        let out_node = &def.nodes[def.output.index()];
        assert_eq!(out_node.shape.len(), 2, "output rank must be 2");
        assert_eq!(out_node.shape[0], batch_size, "output batch dim must match");
        assert_eq!(
            out_node.shape[1], out_features,
            "output feature dim must match"
        );
    }
}

#[cfg(test)]
#[path = "linear_tests.rs"]
mod tests;
