// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MatMul tensor-level IR → NY `MatMulLayer` translation.
//!
//! Maps `TensorOpKind::MatMul` to `Layer::MatMul(MatMulLayer)`, a binary
//! operation where both inputs carry bounds (unlike Linear, which has fixed
//! weights). Used for attention score computation (`Q @ K^T / sqrt(d_k)`)
//! and attention-value multiplication (`attn_weights @ V`).
//!
//! NY's `MatMulLayer` uses McCormick bilinear relaxation for tight
//! IBP bounds when both operands are perturbed — essential for verifying
//! attention layers in dvoice's Kokoro and Whisper models.

use ny_propagate::layers::{ConcatLayer, LinearLayer, MatMulLayer, SliceLayer, TransposeLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::{ArrayD, IxDyn};

use super::TensorNodeValue;
use crate::error::VerifyError;
use crate::graph::{add_unary_node, FiniteF32};
use crate::util::get_value;

/// Translate a MatMul tensor operation to a NY graph node.
///
/// Both-Variable case: produces a `GraphNode::binary` with
/// `Layer::MatMul(MatMulLayer::new(transpose_right, scale))` using McCormick
/// bilinear relaxation. This is the primary attention path (Q, K, V).
///
/// Constant-folding: when both operands are scalar constants, folds to the
/// product. When one operand is `Constant(0.0)`, folds to zero (anything times
/// zero is zero). These cases arise in decomposed DeltaNet with zero-init state.
///
/// Variable × WeightTensor (2D): uses `LinearLayer` to propagate bounds through
/// the variable operand. `transpose_right` and `scale` are applied to the weight
/// matrix before constructing the LinearLayer. This enables mixed-binding
/// composition paths (e.g., DeltaNet with constant projection matrices).
///
/// WeightTensor × WeightTensor: eager constant fold via ndarray matmul.
pub(super) fn translate_matmul(
    node_id: TensorNodeId,
    left_id: TensorNodeId,
    right_id: TensorNodeId,
    transpose_right: bool,
    scale: Option<f32>,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let left = get_value(node_values, left_id.index(), "MatMul left")?;
    let right = get_value(node_values, right_id.index(), "MatMul right")?;
    let name = format!("t{}_matmul", node_id.index());

    match (left, right) {
        // Both variable: binary MatMul node with McCormick relaxation.
        (TensorNodeValue::Variable(lhs_name), TensorNodeValue::Variable(rhs_name)) => {
            let layer = Layer::MatMul(MatMulLayer::new(transpose_right, scale));
            graph.add_node(GraphNode::binary(
                name.clone(),
                layer,
                lhs_name.clone(),
                rhs_name.clone(),
            ));
            Ok(TensorNodeValue::Variable(name))
        }
        // Constant × Constant: scalar constant fold (e.g., gate*state=0.9*0.0=0.0).
        (TensorNodeValue::Constant(a), TensorNodeValue::Constant(b)) => {
            let product = a.get() * b.get();
            let scaled = match scale {
                Some(s) => product * s,
                None => product,
            };
            let finite = FiniteF32::new(scaled).map_err(|_| VerifyError::NonFiniteConstant {
                value: scaled,
                context: format!("MatMul constant fold {a:?} * {b:?}"),
            })?;
            Ok(TensorNodeValue::Constant(finite))
        }
        // Variable × Constant(0.0) or Constant(0.0) × Variable: fold to zero.
        // MatMul with a zero matrix produces a zero result regardless of the other operand.
        (TensorNodeValue::Variable(_), TensorNodeValue::Constant(c))
        | (TensorNodeValue::Constant(c), TensorNodeValue::Variable(_))
            if c.get() == 0.0 =>
        {
            Ok(TensorNodeValue::Constant(FiniteF32::new(0.0)?))
        }
        // Variable × WeightTensor: x @ W → LinearLayer path.
        // 2D: single LinearLayer(W^T_scaled) applied to x.
        // 3D+: batch decomposition via Slice→LinearLayer→Concat per batch element.
        (TensorNodeValue::Variable(var_name), TensorNodeValue::WeightTensor(w)) => {
            if w.ndim() == 2 {
                let w2d = to_array2(w, "MatMul right WeightTensor")?;
                let w_lin = if transpose_right {
                    w2d
                } else {
                    w2d.t().to_owned()
                };
                let w_scaled = apply_scale(w_lin, scale);
                let linear = LinearLayer::new(w_scaled, None).map_err(|e| {
                    VerifyError::WeightValidation {
                        op: "MatMul",
                        reason: format!("Variable×WeightTensor LinearLayer failed: {e}"),
                    }
                })?;
                add_unary_node(&name, Layer::Linear(linear), var_name, graph);
            } else {
                batch_linear_decompose(&name, var_name, w, transpose_right, scale, false, graph)?;
            }
            Ok(TensorNodeValue::Variable(name))
        }
        // WeightTensor × Variable: W @ x → LinearLayer path.
        // 2D: single LinearLayer(W_scaled) applied to x.
        // 3D+: batch decomposition via Slice→LinearLayer→Concat per batch element.
        (TensorNodeValue::WeightTensor(w), TensorNodeValue::Variable(var_name))
            if !transpose_right =>
        {
            if w.ndim() == 2 {
                // W[M,K] @ x[K,N] → [M,N]. A `LinearLayer` applies its weight to
                // the LAST axis of the input (computing x @ W^T), so feeding the
                // [K,N] variable directly would yield [N,M] (a transposed,
                // wrong-shaped result). Mirror the 3D `weight_is_left` path:
                // Transpose x → [N,K], LinearLayer(W[M,K]) → [N,M], Transpose →
                // [M,N]. The transpose is a pure layout op and IBP-exact.
                let w2d = to_array2(w, "MatMul left WeightTensor")?;
                let w_scaled = apply_scale(w2d, scale);
                let linear = LinearLayer::new(w_scaled, None).map_err(|e| {
                    VerifyError::WeightValidation {
                        op: "MatMul",
                        reason: format!("WeightTensor×Variable LinearLayer failed: {e}"),
                    }
                })?;
                let txp_pre = format!("{name}_txp_pre");
                add_unary_node(
                    &txp_pre,
                    Layer::Transpose(TransposeLayer::batched_transpose()),
                    var_name,
                    graph,
                );
                let lin_name = format!("{name}_lin");
                add_unary_node(&lin_name, Layer::Linear(linear), &txp_pre, graph);
                graph.add_node(GraphNode::new(
                    name.clone(),
                    Layer::Transpose(TransposeLayer::batched_transpose()),
                    vec![lin_name],
                ));
            } else {
                batch_linear_decompose(&name, var_name, w, transpose_right, scale, true, graph)?;
            }
            Ok(TensorNodeValue::Variable(name))
        }
        // WeightTensor × WeightTensor: eager constant fold via ndarray matmul.
        // 2D: direct ndarray dot. 3D+: batch matmul over leading dimensions.
        (TensorNodeValue::WeightTensor(lhs), TensorNodeValue::WeightTensor(rhs)) => {
            let product = batch_matmul_fold(lhs, rhs, transpose_right, scale, node_id)?;
            Ok(TensorNodeValue::WeightTensor(product))
        }
        _ => Err(VerifyError::UnsupportedOp(format!(
            "MatMul unsupported operand combination; got left={left:?}, right={right:?}"
        ))),
    }
}

/// Convert a dynamic-dimension array to 2D, required for LinearLayer construction.
fn to_array2(arr: &ArrayD<f32>, context: &str) -> Result<ndarray::Array2<f32>, VerifyError> {
    if arr.ndim() != 2 {
        return Err(VerifyError::WeightValidation {
            op: "MatMul",
            reason: format!(
                "{context} must be 2-D for LinearLayer, got {}-D",
                arr.ndim()
            ),
        });
    }
    arr.clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("{context} Array2 conversion: {e}"),
        })
}

/// Apply optional scale factor to a 2D weight matrix.
fn apply_scale(mut w: ndarray::Array2<f32>, scale: Option<f32>) -> ndarray::Array2<f32> {
    if let Some(s) = scale {
        w.mapv_inplace(|v| v * s);
    }
    w
}

/// Batch matmul constant fold for WeightTensor×WeightTensor of any rank.
///
/// 2D: direct `dot`. 3D+: iterates over leading batch dimensions, performing
/// 2D matmul per batch element and assembling the result.
fn batch_matmul_fold(
    lhs: &ArrayD<f32>,
    rhs: &ArrayD<f32>,
    transpose_right: bool,
    scale: Option<f32>,
    node_id: TensorNodeId,
) -> Result<ArrayD<f32>, VerifyError> {
    if lhs.ndim() < 2 || rhs.ndim() < 2 {
        return Err(VerifyError::WeightValidation {
            op: "MatMul",
            reason: format!(
                "WeightTensor fold requires at least 2-D operands, got {}-D and {}-D",
                lhs.ndim(),
                rhs.ndim()
            ),
        });
    }
    if lhs.ndim() == 2 && rhs.ndim() == 2 {
        let l2d = to_array2(lhs, "MatMul left WeightTensor")?;
        let r2d = to_array2(rhs, "MatMul right WeightTensor")?;
        let mut product = if transpose_right {
            l2d.dot(&r2d.t())
        } else {
            l2d.dot(&r2d)
        };
        if let Some(s) = scale {
            product.mapv_inplace(|v| v * s);
        }
        let dyn_product = product.into_dyn();
        validate_finite_array(&dyn_product, node_id)?;
        return Ok(dyn_product);
    }
    // 3D+: batch dimensions must match. Last two dims are the matmul dims.
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    let l_batch = &l_shape[..l_shape.len() - 2];
    let r_batch = &r_shape[..r_shape.len() - 2];
    if l_batch != r_batch {
        return Err(VerifyError::WeightValidation {
            op: "MatMul",
            reason: format!("WeightTensor fold batch dims mismatch: {l_batch:?} vs {r_batch:?}"),
        });
    }
    let batch_size: usize = l_batch
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| VerifyError::DimensionOverflow {
            op: "MatMul",
            context: format!("batch dims overflow: {l_batch:?}"),
        })?;
    let m = l_shape[l_shape.len() - 2];
    let k_l = l_shape[l_shape.len() - 1];
    let (r_m, r_n) = if transpose_right {
        (r_shape[r_shape.len() - 1], r_shape[r_shape.len() - 2])
    } else {
        (r_shape[r_shape.len() - 2], r_shape[r_shape.len() - 1])
    };
    if k_l != r_m {
        return Err(VerifyError::WeightValidation {
            op: "MatMul",
            reason: format!("WeightTensor fold inner dim mismatch: {k_l} vs {r_m}"),
        });
    }
    let n = r_n;
    // Flatten batch dims, do per-element 2D matmul, then reshape back.
    let l_flat =
        lhs.to_shape((batch_size, m, k_l))
            .map_err(|e| VerifyError::InternalTranslationError {
                context: format!("MatMul batch reshape left: {e}"),
            })?;
    let r_2d_rows = r_shape[r_shape.len() - 2];
    let r_2d_cols = r_shape[r_shape.len() - 1];
    let r_flat = rhs
        .to_shape((batch_size, r_2d_rows, r_2d_cols))
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("MatMul batch reshape right: {e}"),
        })?;
    let mut result = ndarray::Array3::zeros((batch_size, m, n));
    for b in 0..batch_size {
        let l_slice = l_flat.slice(ndarray::s![b, .., ..]);
        let r_slice = r_flat.slice(ndarray::s![b, .., ..]);
        let prod = if transpose_right {
            l_slice.dot(&r_slice.t())
        } else {
            l_slice.dot(&r_slice)
        };
        result.slice_mut(ndarray::s![b, .., ..]).assign(&prod);
    }
    if let Some(s) = scale {
        result.mapv_inplace(|v| v * s);
    }
    let mut out_shape = l_batch.to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out = result
        .into_shape_with_order(IxDyn(&out_shape))
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("MatMul batch reshape output: {e}"),
        })?;
    validate_finite_array(&out, node_id)?;
    Ok(out)
}

/// Validate all elements of an ndarray are finite.
fn validate_finite_array(arr: &ArrayD<f32>, node_id: TensorNodeId) -> Result<(), VerifyError> {
    for &val in arr.iter() {
        if !val.is_finite() {
            return Err(VerifyError::NonFiniteConstant {
                value: val,
                context: format!("MatMul WeightTensor fold t{}", node_id.index()),
            });
        }
    }
    Ok(())
}

/// Decompose a 3D+ WeightTensor×Variable matmul into per-batch LinearLayers.
///
/// Builds: Slice(axis=0, b, b+1) → LinearLayer(W[b]) → Concat(axis=0) for each
/// batch element b. This enables NY bound propagation through batched
/// matmul with constant weights (e.g., decomposed GatedDeltaNet k_row×decayed).
///
/// `weight_is_left`: true if layout is W @ x, false if layout is x @ W.
fn batch_linear_decompose(
    output_name: &str,
    var_name: &str,
    weight: &ArrayD<f32>,
    transpose_right: bool,
    scale: Option<f32>,
    weight_is_left: bool,
    graph: &mut GraphNetwork,
) -> Result<(), VerifyError> {
    let ndim = weight.ndim();
    if ndim < 3 {
        return Err(VerifyError::WeightValidation {
            op: "MatMul",
            reason: format!("batch_linear_decompose requires 3D+ weight, got {ndim}-D"),
        });
    }
    let batch = weight.shape()[0];
    if batch == 0 {
        return Err(VerifyError::WeightValidation {
            op: "MatMul",
            reason: "batch_linear_decompose: zero batch dim".into(),
        });
    }
    // Slice variable along batch dim (axis=0).
    // For each batch b: extract W[b] (2D), build LinearLayer, apply to sliced var.
    //
    // For weight_is_left (W @ x): LinearLayer computes W @ input where the last
    // dim of input is in_features. When x is [K,V], last dim V ≠ K. We insert
    // Transpose before LinearLayer to swap the last two dims, then transpose back.
    let mut slice_names = Vec::with_capacity(batch);
    for b in 0..batch {
        let slice_name = format!("{output_name}_slice{b}");
        let slice_layer = SliceLayer::new(0, b, b + 1);
        graph.add_node(GraphNode::new(
            slice_name.clone(),
            Layer::Slice(slice_layer),
            vec![var_name.to_string()],
        ));
        let w_b = weight.slice(ndarray::s![b, .., ..]).to_owned();
        let w2d = w_b.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            VerifyError::InternalTranslationError {
                context: format!("batch_linear_decompose W[{b}] to 2D: {e}"),
            }
        })?;
        let lin_input_name;
        let w_lin;
        if weight_is_left {
            // W[M,K] @ x[..., K, V]: transpose x to [..., V, K], apply
            // LinearLayer(W[M,K]) with in=K, out=M → [..., V, M], transpose → [..., M, V].
            let txp_name = format!("{output_name}_txp_pre{b}");
            graph.add_node(GraphNode::new(
                txp_name.clone(),
                Layer::Transpose(TransposeLayer::batched_transpose()),
                vec![slice_name],
            ));
            lin_input_name = txp_name;
            w_lin = w2d;
        } else {
            lin_input_name = slice_name;
            w_lin = if transpose_right {
                w2d
            } else {
                w2d.t().to_owned()
            };
        }
        let w_scaled = apply_scale(w_lin, scale);
        let linear =
            LinearLayer::new(w_scaled, None).map_err(|e| VerifyError::WeightValidation {
                op: "MatMul",
                reason: format!("batch_linear_decompose LinearLayer[{b}]: {e}"),
            })?;
        let lin_name = format!("{output_name}_lin{b}");
        add_unary_node(&lin_name, Layer::Linear(linear), &lin_input_name, graph);
        let out_name = if weight_is_left {
            let txp_post = format!("{output_name}_txp_post{b}");
            graph.add_node(GraphNode::new(
                txp_post.clone(),
                Layer::Transpose(TransposeLayer::batched_transpose()),
                vec![lin_name],
            ));
            txp_post
        } else {
            lin_name
        };
        slice_names.push(out_name);
    }
    let concat_layer = ConcatLayer::new(0);
    graph.add_node(GraphNode::new(
        output_name.to_string(),
        Layer::Concat(concat_layer),
        slice_names,
    ));
    Ok(())
}

#[cfg(test)]
#[path = "graph_tensor_matmul_tests.rs"]
mod tests;
