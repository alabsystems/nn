// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Embedding lookup tensor op translation to NY constant-fold bounds.
//!
//! Embedding is a table lookup: `output[*, d] = weight[input[*], d]` where the
//! variable input contains integer row indices into the fixed weight table
//! `[V, D]`. For verification the table is a constant and the indices are the
//! bounded variable.
//!
//! Since *any* row of the table could be selected, the tightest sound,
//! index-agnostic bound on the output is the per-dimension box over all rows:
//!   `lo[d] = min(weight[0..V, d])`, `hi[d] = max(weight[0..V, d])`.
//! Every reachable row's value `weight[i, d]` lies in `[lo[d], hi[d]]`, so this
//! box CONTAINS every possible embedded value and is a sound over-approximation.
//!
//! Crucially, this bound is INDEPENDENT of the index tensor's interval: the
//! output box must not inherit (and must not shrink with) the indices' bounds,
//! because the table is an arbitrary (non-affine) function of the index. The
//! previous translation emitted `AddConstant(midpoint[D])` directly on the
//! `[*index_dims]` index tensor, which (1) failed IBP with `ShapeMismatch`
//! (broadcasting `[*index_dims]` against `[D]`) and (2) was unsound on the
//! accidental `index_dims == D` path, where it merely SHIFTED the index
//! interval instead of emitting the table's `[lo, hi]` spread.
//!
//! The corrected translation emits a small subgraph whose output bounds have
//! the declared shape `[*index_dims, D]` and, per dimension `d`, the fixed
//! interval `[lo[d], hi[d]]`, discarding the index interval entirely:
//!   1. `Reshape([-1])`         — flatten the indices to seed a valid graph edge.
//!   2. `Linear(zeros, midpoint)` — collapse to the per-position/per-dim midpoint
//!      `mid[d] = (lo[d]+hi[d])/2`. The zero weight makes the output a degenerate
//!      point INDEPENDENT of the index bounds (same idiom as the LSTM/attention
//!      constant injection).
//!   3. `Qdq(epsilon = global_half_spread)` — a sound additive `±epsilon`
//!      perturbation that widens the degenerate point into a box. `epsilon` is
//!      the GLOBAL half-spread `max_d (hi[d]-lo[d]) / 2`, so the resulting box
//!      `[mid[d]-epsilon, mid[d]+epsilon]` CONTAINS the tight per-dim box
//!      `[lo[d], hi[d]]` for every `d` (sound over-approximation; index-agnostic).
//!      For a degenerate table (all rows equal -> zero spread) this step is
//!      omitted and the exact constant point is emitted.
//!   4. `Reshape([*index_dims, D])` — restore the declared output shape.
//!
//! Soundness: every value-carrying weight is zero, so the output bounds do not
//! depend on the index interval; the emitted box is a fixed, index-independent
//! over-approximation that provably contains every row of the table.

use ny_propagate::layers::{LinearLayer, QdqPerturbationLayer, ReshapeLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::{Array1, Array2};

use crate::error::VerifyError;
use crate::util::get_value;

use super::TensorNodeValue;

/// Translate a `TensorOpKind::Embedding` node to NY graph nodes.
///
/// The weight table is a constant `[V, D]` matrix and the input indices are a
/// bounded variable. The emitted subgraph produces output bounds of the declared
/// shape `[*index_dims, D]` whose per-dim interval is the table's `[lo[d], hi[d]]`
/// spread, independent of the index interval. See the module docs for the exact
/// node sequence and the soundness argument.
pub(super) fn translate_embedding(
    node_id: TensorNodeId,
    input: TensorNodeId,
    weight: TensorNodeId,
    output_shape: &[usize],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // Input (indices) must be a Variable graph node.
    let input_name = match get_value(node_values, input.index(), "Embedding input")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Embedding input must be a variable tensor (indices), not a constant scalar".into(),
            ));
        }
        TensorNodeValue::WeightTensor(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Embedding input must be a variable tensor (indices), not a weight tensor".into(),
            ));
        }
    };

    // Weight (embedding table) must be a 2-D WeightTensor `[V, D]`.
    let weight_array = match get_value(node_values, weight.index(), "Embedding weight")? {
        TensorNodeValue::WeightTensor(arr) => {
            if arr.shape().len() != 2 {
                return Err(VerifyError::WeightValidation {
                    op: "Embedding",
                    reason: format!(
                        "weight must be 2-D [num_embeddings, embedding_dim], got {}-D",
                        arr.shape().len()
                    ),
                });
            }
            arr.clone()
        }
        TensorNodeValue::Variable(_) => {
            return Err(VerifyError::WeightValidation {
                op: "Embedding",
                reason: "weight must be a ConstantTensor binding, not a variable".into(),
            });
        }
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::WeightValidation {
                op: "Embedding",
                reason: "weight must be a ConstantTensor binding, not a constant scalar".into(),
            });
        }
    };

    let weight_shape = weight_array.shape();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    // An empty table (no rows / no columns) has no rows to bound; reject up front
    // so the per-dim min/max loop below can never produce NaN/inf extrema.
    if num_embeddings == 0 || embedding_dim == 0 {
        return Err(VerifyError::WeightValidation {
            op: "Embedding",
            reason: format!(
                "weight table must be non-empty [num_embeddings, embedding_dim], got \
                 [{num_embeddings}, {embedding_dim}]"
            ),
        });
    }

    // The declared output shape must be `[*index_dims, embedding_dim]`; its last
    // axis is the embedding dimension. Validate so a malformed spec is rejected
    // up front rather than producing a mis-shaped node.
    if output_shape.last() != Some(&embedding_dim) {
        return Err(VerifyError::WeightValidation {
            op: "Embedding",
            reason: format!(
                "declared output shape {output_shape:?} last axis must equal embedding_dim \
                 {embedding_dim}"
            ),
        });
    }
    let out_total: usize = output_shape.iter().product();
    if out_total == 0 {
        return Err(VerifyError::WeightValidation {
            op: "Embedding",
            reason: format!("declared output shape {output_shape:?} has a zero dimension"),
        });
    }
    // Number of index positions = product of all output axes except the last.
    let num_positions = out_total / embedding_dim;

    // Per-dimension extrema over all rows: lo[d] = min, hi[d] = max. Every
    // reachable row `weight[i, d]` lies in `[lo[d], hi[d]]`, so this box is the
    // tightest sound index-agnostic bound. We also track the global half-spread
    // used to widen the (index-independent) midpoint point into the box.
    let mut mid = vec![0.0f32; embedding_dim];
    let mut global_half_spread = 0.0f32;
    for d in 0..embedding_dim {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in 0..num_embeddings {
            let val = weight_array[[row, d]];
            if !val.is_finite() {
                return Err(VerifyError::WeightValidation {
                    op: "Embedding",
                    reason: format!("weight contains non-finite value at [{row}, {d}]"),
                });
            }
            if val < lo {
                lo = val;
            }
            if val > hi {
                hi = val;
            }
        }
        mid[d] = f32::midpoint(lo, hi);
        let half_spread = 0.5 * (hi - lo);
        if half_spread > global_half_spread {
            global_half_spread = half_spread;
        }
    }

    let base = format!("t{}", node_id.index());

    // 1. Flatten the indices to 1-D so the collapse Linear sees a known
    //    `in_features`. The reshape only seeds a valid graph edge; its values
    //    are discarded by the zero weight below. `in_features` here is the index
    //    tensor's total element count; it can differ from `num_positions` if the
    //    index tensor and the output share index dims but we always overwrite the
    //    values via the zero weight, so only its total matters for the edge.
    let flat_name = format!("{base}_idx_flat");
    graph.add_node(GraphNode::new(
        flat_name.clone(),
        Layer::Reshape(ReshapeLayer::new(vec![-1])),
        vec![input_name],
    ));

    // The flattened index length equals the index tensor's total element count.
    // For an embedding it is exactly `num_positions` (one integer per output
    // position), so the zero-weight Linear's `in_features` is `num_positions`.
    let in_features = num_positions;

    // 2. Collapse to the per-position/per-dim midpoint via a ZERO weight: the
    //    output is exactly `bias` (a degenerate point) regardless of the index
    //    bounds, so the result is fully decoupled from the index interval. The
    //    bias broadcasts the per-dim midpoint `mid[d]` across all positions in
    //    row-major `[*index_dims, D]` order.
    let mut midpoint_flat = Vec::with_capacity(out_total);
    for _pos in 0..num_positions {
        midpoint_flat.extend_from_slice(&mid);
    }
    let collapse = LinearLayer::new(
        Array2::zeros((out_total, in_features)),
        Some(Array1::from_vec(midpoint_flat)),
    )
    .map_err(|e| VerifyError::InternalTranslationError {
        context: format!("Embedding midpoint Linear construction failed: {e}"),
    })?;
    let mid_name = format!("{base}_mid");
    graph.add_node(GraphNode::new(
        mid_name.clone(),
        Layer::Linear(collapse),
        vec![flat_name],
    ));

    // 3. Widen the degenerate midpoint point into the table's box via a sound
    //    additive `±epsilon` perturbation. `epsilon` is the GLOBAL half-spread,
    //    so the resulting box `[mid[d]-epsilon, mid[d]+epsilon]` contains the
    //    tight per-dim box `[lo[d], hi[d]]` for every `d` (sound over-approx).
    //    The perturbation is independent of the input bounds, preserving the
    //    index-agnostic property. For a degenerate table (zero spread) we skip
    //    this and emit the exact constant point.
    let widened_name = if global_half_spread > 0.0 {
        // QdqPerturbation models a sound additive `x +/- epsilon` relaxation; we
        // reuse it here purely as an index-independent symmetric widening. Drive
        // `epsilon = global_half_spread` via `scale = 2 * global_half_spread`
        // (the layer uses `epsilon = next_up(scale * 0.5) >= global_half_spread`,
        // which only ever widens -> still sound), and set the saturation range
        // wide enough (zero_point 0, large symmetric qmin/qmax) that the tiny
        // midpoint point can never saturate.
        let scale = 2.0 * global_half_spread;
        // Make the saturation range as wide as possible (`zero_point = 0`,
        // `qmin/qmax = -/+f32::MAX`) so the tiny, index-independent midpoint
        // point can never saturate; if it ever did, Qdq fails CLOSED with an
        // error (never an unsound pass).
        let qdq = QdqPerturbationLayer::new(scale, 0.0, -f32::MAX, f32::MAX).map_err(|e| {
            VerifyError::InternalTranslationError {
                context: format!("Embedding spread perturbation construction failed: {e}"),
            }
        })?;
        let qdq_name = format!("{base}_box");
        graph.add_node(GraphNode::new(
            qdq_name.clone(),
            Layer::QdqPerturbation(qdq),
            vec![mid_name],
        ));
        qdq_name
    } else {
        mid_name
    };

    // 4. Restore the declared `[*index_dims, D]` output shape.
    let target: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
    let node_name = base;
    graph.add_node(GraphNode::new(
        node_name.clone(),
        Layer::Reshape(ReshapeLayer::new(target)),
        vec![widened_name],
    ));

    Ok(TensorNodeValue::Variable(node_name))
}
