// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RoPE native layer pattern matcher for tensor-level IR → NY translation.
//!
//! Detects the 10-node `rope_rotate` tensor pattern and collapses it into a single
//! `Layer::RoPE(RopeLayer)` when the frequency input is a constant scalar. This
//! produces tighter bounds than the decomposed path because `RopeLayer` uses f64
//! interval arithmetic with directed rounding for the rotation matrix, whereas
//! the decomposed path introduces relaxation at each intermediate layer.
//!
//! Pattern (from `build_rope_rotate_kernel`):
//! ```text
//! Node 0: x input [BH, S, D]
//! Node 1: freqs input [S, D/2]             ← must be ConstantScalar
//! Node 2: Reshape(x) → [BH, S, D/2, 2]
//! Node 3: AxisSelect(node2, axis=3, idx=0)  x_even [BH, S, D/2]
//! Node 4: AxisSelect(node2, axis=3, idx=1)  x_odd  [BH, S, D/2]
//! Node 5: Broadcast(freqs) → [BH, S, D/2]
//! Node 6: Elementwise(rope_cos, [3,4,5])    y_even
//! Node 7: Elementwise(rope_sin, [3,4,5])    y_odd
//! Node 8: Stack([6,7], axis=3) → [BH, S, D/2, 2]
//! Node 9: Reshape → [BH, S, D]
//! ```
//!
//! Part of #525.

use ny_propagate::layers::RopeLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::{TensorKernelDef, TensorOpKind};

use crate::error::VerifyError;
use crate::graph_tensor::TensorParamBinding;

/// Attempt to match the `rope_rotate` 10-node pattern and emit a native
/// `Layer::RoPE(RopeLayer)` graph.
///
/// Returns `Ok(Some(graph))` if the pattern is matched and the native path is
/// used. Returns `Ok(None)` to fall through to the decomposition path.
///
/// # When this fires
///
/// - Kernel name is exactly "rope_rotate"
/// - Exactly 10 nodes with the canonical structure
/// - Second input (freqs) is `ConstantScalar`
///
/// # When this does NOT fire (fallback to decomposition)
///
/// - Freqs is `Variable` (per-position verification with variable freqs)
/// - Kernel doesn't match the expected structure
/// - Non-finite frequency value
pub(crate) fn try_native_rope(
    kernel: &TensorKernelDef,
    input_bindings: &[TensorParamBinding],
) -> Result<Option<GraphNetwork>, VerifyError> {
    // Quick name check: bail early if not a rope_rotate kernel.
    if kernel.name != "rope_rotate" {
        return Ok(None);
    }

    // Must have exactly 10 nodes (the canonical rope_rotate pattern).
    if kernel.nodes.len() != 10 {
        return Ok(None);
    }

    // Must have exactly 2 inputs: x (Variable) and freqs (ConstantScalar).
    if input_bindings.len() != 2 {
        return Ok(None);
    }

    let freq_scalar = match &input_bindings[1] {
        TensorParamBinding::ConstantScalar(v) => *v,
        TensorParamBinding::Variable | TensorParamBinding::ConstantTensor(_) => return Ok(None),
    };

    if !freq_scalar.is_finite() {
        return Err(VerifyError::NonFiniteConstant {
            value: freq_scalar,
            context: "RoPE native path: freqs binding".to_string(),
        });
    }

    // Validate structural pattern: node 0 = Input(x), node 1 = Input(freqs).
    if !matches!(kernel.nodes[0].kind, TensorOpKind::Input { .. }) {
        return Ok(None);
    }
    if !matches!(kernel.nodes[1].kind, TensorOpKind::Input { .. }) {
        return Ok(None);
    }

    // Extract head_dim from x's input shape (last dimension).
    let x_shape = &kernel.nodes[0].shape;
    if x_shape.len() < 2 {
        return Ok(None);
    }
    let head_dim = *x_shape.last().unwrap_or(&0);
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Ok(None);
    }
    let num_pairs = head_dim / 2;

    // Validate remaining structural nodes match the pattern.
    // Node 2: Reshape
    if !matches!(kernel.nodes[2].kind, TensorOpKind::Reshape { .. }) {
        return Ok(None);
    }
    // Node 3: AxisSelect index=0
    if !matches!(
        kernel.nodes[3].kind,
        TensorOpKind::AxisSelect { index: 0, .. }
    ) {
        return Ok(None);
    }
    // Node 4: AxisSelect index=1
    if !matches!(
        kernel.nodes[4].kind,
        TensorOpKind::AxisSelect { index: 1, .. }
    ) {
        return Ok(None);
    }
    // Node 5: Broadcast
    if !matches!(kernel.nodes[5].kind, TensorOpKind::Broadcast { .. }) {
        return Ok(None);
    }
    // Node 6: Elementwise (rope_cos)
    // SAFETY: TensorOpKind is #[non_exhaustive]. Non-Elementwise nodes don't
    // match the RoPE pattern; false causes early return Ok(None) (conservative).
    let is_rope_cos = match &kernel.nodes[6].kind {
        TensorOpKind::Elementwise { kernel: k, .. } => k.name == "rope_cos",
        _ => false,
    };
    if !is_rope_cos {
        return Ok(None);
    }
    // Node 7: Elementwise (rope_sin)
    // SAFETY: Same as rope_cos above.
    let is_rope_sin = match &kernel.nodes[7].kind {
        TensorOpKind::Elementwise { kernel: k, .. } => k.name == "rope_sin",
        _ => false,
    };
    if !is_rope_sin {
        return Ok(None);
    }
    // Node 8: Stack
    if !matches!(kernel.nodes[8].kind, TensorOpKind::Stack { .. }) {
        return Ok(None);
    }
    // Node 9: Reshape (output)
    if !matches!(kernel.nodes[9].kind, TensorOpKind::Reshape { .. }) {
        return Ok(None);
    }

    // Pattern matched. Build a native RoPE graph.
    //
    // Since freqs is a constant scalar, all pairs get the same rotation angle.
    // cos_freqs and sin_freqs are uniform vectors of length num_pairs.
    let cos_val = freq_scalar.cos();
    let sin_val = freq_scalar.sin();

    if !cos_val.is_finite() || !sin_val.is_finite() {
        return Err(VerifyError::NonFiniteConstant {
            value: freq_scalar,
            context: format!(
                "RoPE native path: cos({freq_scalar})={cos_val} or sin({freq_scalar})={sin_val}"
            ),
        });
    }

    let cos_freqs = vec![cos_val; num_pairs];
    let sin_freqs = vec![sin_val; num_pairs];

    let rope_layer = RopeLayer::new(cos_freqs, sin_freqs)?;

    let mut graph = GraphNetwork::new();
    let output_name = "rope_out".to_string();

    // GraphNode::from_input connects from NETWORK_INPUT by default.
    graph.add_node(GraphNode::from_input(
        output_name.clone(),
        Layer::RoPE(rope_layer),
    ));
    graph.set_output(output_name);

    Ok(Some(graph))
}

#[cfg(test)]
#[path = "graph_tensor_rope_tests.rs"]
mod tests;
