// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gated DeltaNet tensor op translation to NY decomposed layers.
//!
//! NY has no native DeltaNet layer, so we decompose the single-step
//! Gated DeltaNet recurrence into primitive operations that NY CAN verify:
//!
//! ```text
//! decayed = gate * state                              // [H, K, V]
//! v_retrieved = k^T @ decayed                         // [H, V]
//! new_state = decayed + outer(k, beta*v) - outer(k, beta*v_retrieved)
//! output = scale * q @ new_state                      // [H, V]
//! ```
//!
//! The decomposition produces a graph of MulBinary, MatMul, Add — plus explicit
//! `Reshape` nodes that mirror `decompose_gated_delta_net()` in nn-dsl. The
//! reshapes are NOT optional: NY's `MatMulLayer` treats the leading dims as a
//! batch prefix that must match *exactly* between the two operands (it does not
//! auto-broadcast rank, see `parse_matmul_dims`). The recurrence needs
//! rank-changing reshapes — e.g. `k^T @ decayed` is `k[H,1,K] @ decayed[H,K,V]`,
//! and the outer products are `k[H,K,1] @ row[H,1,V]`. Without the reshape nodes,
//! `k[H,K]` (batch prefix `[]`) is matmul'd against `decayed[H,K,V]` (batch
//! prefix `[H]`), which fails with `Shape mismatch: expected [], got [H]`.
//!
//! Mirrors `gated_delta_net.rs` in nn-dsl. Part of #834.

use ny_propagate::layers::{AddLayer, MatMulLayer, MulBinaryLayer, ReshapeLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::{TensorNode, TensorNodeId};

use super::TensorNodeValue;
use crate::error::{StructuralError, VerifyError};
use crate::util::get_value;

/// Read a Variable input's declared shape from the kernel's tensor nodes.
fn input_shape<'a>(
    all_nodes: &'a [TensorNode],
    id: TensorNodeId,
    context: &str,
) -> Result<&'a [usize], VerifyError> {
    all_nodes
        .get(id.index())
        .map(|n| n.shape.as_slice())
        .ok_or_else(|| {
            VerifyError::Structural(StructuralError::ShapeConstraint {
                context: format!("GatedDeltaNet {context}: node id {} out of range", id.index()),
            })
        })
}

/// Emit a `Reshape` node and return its name. Reshapes are exact layout ops, so
/// they only correct shape and never loosen bounds.
fn add_reshape(
    graph: &mut GraphNetwork,
    name: String,
    input: String,
    target: &[usize],
) -> String {
    let dims: Vec<i64> = target.iter().map(|&d| d as i64).collect();
    graph.add_node(GraphNode::new(
        name.clone(),
        Layer::Reshape(ReshapeLayer::new(dims)),
        vec![input],
    ));
    name
}

/// Translate a `TensorOpKind::GatedDeltaNet` node to a NY decomposed graph.
///
/// All 6 inputs (q, k, v, state, gate, beta) must be `Variable` (graph nodes).
/// The decomposition mirrors `decompose_gated_delta_net()` in nn-dsl but builds
/// NY layer nodes instead of tensor IR nodes.
///
/// Reshapes are NOT transparent in NY: every rank-changing step emits an explicit
/// `Reshape` node so each MatMul's two operands share an identical `[H]` batch
/// prefix (`parse_matmul_dims` requires the leading batch dims to match exactly).
/// The NY layers produced are MulBinary, MatMul, Add, and Reshape.
///
/// `H`, `K`, `V` are derived from the kernel input shapes (`q=[H,K]`, `v=[H,V]`)
/// read from `all_nodes`; malformed/inconsistent shapes return a
/// `StructuralError` rather than passing vacuously.
pub(super) fn translate_gated_delta_net(
    node_id: TensorNodeId,
    q_id: TensorNodeId,
    k_id: TensorNodeId,
    v_id: TensorNodeId,
    state_id: TensorNodeId,
    gate_id: TensorNodeId,
    beta_id: TensorNodeId,
    scale: f32,
    all_nodes: &[TensorNode],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // Validate scale is finite and positive (matches DSL decomposition validation).
    if !scale.is_finite() || scale <= 0.0 {
        return Err(VerifyError::NonFiniteConstant {
            value: scale,
            context: format!("GatedDeltaNet scale must be finite and positive, got {scale}"),
        });
    }

    // --- Derive dimensions from input shapes: q=[H,K], v=[H,V], state=[H,K,V] ---
    let q_shape = input_shape(all_nodes, q_id, "q")?;
    let v_shape = input_shape(all_nodes, v_id, "v")?;
    if q_shape.len() != 2 || v_shape.len() != 2 {
        return Err(VerifyError::Structural(StructuralError::ShapeConstraint {
            context: format!(
                "GatedDeltaNet expects q rank-2 [H,K] and v rank-2 [H,V], got q={q_shape:?} v={v_shape:?}"
            ),
        }));
    }
    let (h, key_dim) = (q_shape[0], q_shape[1]);
    let value_dim = v_shape[1];
    if v_shape[0] != h {
        return Err(VerifyError::Structural(StructuralError::ShapeConstraint {
            context: format!(
                "GatedDeltaNet head dim mismatch: q has H={h} but v has H={}",
                v_shape[0]
            ),
        }));
    }
    let hv_shape = [h, value_dim];

    let prefix = format!("t{}_gdn", node_id.index());

    // --- Extract variable names for all inputs ---
    let q_name = extract_var(node_values, q_id, "GatedDeltaNet q")?;
    let k_name = extract_var(node_values, k_id, "GatedDeltaNet k")?;
    let v_name = extract_var(node_values, v_id, "GatedDeltaNet v")?;
    let state_name = extract_var(node_values, state_id, "GatedDeltaNet state")?;
    let gate_name = extract_var(node_values, gate_id, "GatedDeltaNet gate")?;
    let beta_name = extract_var(node_values, beta_id, "GatedDeltaNet beta")?;

    // Step 1: Decay — gate * state -> [H, K, V]
    // Broadcast is transparent in NY; gate [H,1,1] broadcasts to state [H,K,V].
    let decayed = format!("{prefix}_decay");
    graph.add_node(GraphNode::binary(
        decayed.clone(),
        Layer::MulBinary(MulBinaryLayer),
        gate_name,
        state_name,
    ));

    // Step 2: Retrieval — v_retrieved = k^T @ decayed
    // k [H, K] -> [H, 1, K], matmul with decayed [H, K, V] -> [H, 1, V], -> [H, V].
    // NY's MatMul requires matching batch prefixes, so the rank-change is an
    // explicit Reshape node (not transparent).
    let k_row = add_reshape(
        graph,
        format!("{prefix}_k_row"),
        k_name.clone(),
        &[h, 1, key_dim],
    );
    let vr_3d = format!("{prefix}_vr3");
    graph.add_node(GraphNode::binary(
        vr_3d.clone(),
        Layer::MatMul(MatMulLayer::new(false, None)),
        k_row,
        decayed.clone(),
    ));
    let v_retrieved = add_reshape(graph, format!("{prefix}_vr"), vr_3d, &hv_shape);

    // Step 3: Scale terms — beta * v and beta * v_retrieved (both [H, V])
    let beta_v = format!("{prefix}_beta_v");
    graph.add_node(GraphNode::binary(
        beta_v.clone(),
        Layer::MulBinary(MulBinaryLayer),
        beta_name.clone(),
        v_name,
    ));

    let beta_vr = format!("{prefix}_beta_vr");
    graph.add_node(GraphNode::binary(
        beta_vr.clone(),
        Layer::MulBinary(MulBinaryLayer),
        beta_name,
        v_retrieved,
    ));

    // Step 4: Outer products — rank-1 state updates
    // outer(k, x) = k_col [H, K, 1] @ x_row [H, 1, V] -> [H, K, V].
    let k_col = add_reshape(
        graph,
        format!("{prefix}_k_col"),
        k_name.clone(),
        &[h, key_dim, 1],
    );
    let bv_row = add_reshape(graph, format!("{prefix}_bv_row"), beta_v, &[h, 1, value_dim]);
    let bvr_row = add_reshape(
        graph,
        format!("{prefix}_bvr_row"),
        beta_vr,
        &[h, 1, value_dim],
    );

    let pos_update = format!("{prefix}_pos");
    graph.add_node(GraphNode::binary(
        pos_update.clone(),
        Layer::MatMul(MatMulLayer::new(false, None)),
        k_col.clone(),
        bv_row,
    ));

    // outer(k, beta_vr) with scale=-1.0 for subtraction
    let neg_update = format!("{prefix}_neg");
    graph.add_node(GraphNode::binary(
        neg_update.clone(),
        Layer::MatMul(MatMulLayer::new(false, Some(-1.0))),
        k_col,
        bvr_row,
    ));

    // Step 5: State update — new_state = decayed + pos_update + neg_update -> [H, K, V]
    let tmp = format!("{prefix}_tmp");
    graph.add_node(GraphNode::binary(
        tmp.clone(),
        Layer::Add(AddLayer),
        decayed,
        pos_update,
    ));

    let new_state = format!("{prefix}_state");
    graph.add_node(GraphNode::binary(
        new_state.clone(),
        Layer::Add(AddLayer),
        tmp,
        neg_update,
    ));

    // Step 6: Output — o = scale * q @ new_state
    // q [H, K] -> [H, 1, K] @ new_state [H, K, V] -> [H, 1, V] -> [H, V].
    let q_row = add_reshape(graph, format!("{prefix}_q_row"), q_name, &[h, 1, key_dim]);
    let o_3d = format!("{prefix}_o3");
    graph.add_node(GraphNode::binary(
        o_3d.clone(),
        Layer::MatMul(MatMulLayer::new(false, Some(scale))),
        q_row,
        new_state,
    ));
    let output = add_reshape(graph, format!("{prefix}_out"), o_3d, &hv_shape);

    Ok(TensorNodeValue::Variable(output))
}

/// Extract a variable name from a node value, requiring it to be `Variable`.
fn extract_var(
    node_values: &[TensorNodeValue],
    id: TensorNodeId,
    context: &str,
) -> Result<String, VerifyError> {
    match get_value(node_values, id.index(), context)? {
        TensorNodeValue::Variable(name) => Ok(name.clone()),
        other => Err(VerifyError::UnsupportedOp(format!(
            "{context} must be Variable, got {other:?}"
        ))),
    }
}

#[cfg(test)]
#[path = "graph_tensor_gated_delta_net_tests.rs"]
mod tests;
