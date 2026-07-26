// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm tensor op → NY translation.
//!
//! Maps `TensorOpKind::LayerNorm` to `Layer::LayerNorm(LayerNormLayer)`.
//! Follows the same pattern as `graph_tensor_rms_norm.rs`.

use ny_propagate::layers::LayerNormLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, FiniteF32};

use super::norm_util::{extract_norm_param, validate_norm_shape_and_eps};
use super::{TensorNodeValue, TensorTranslationContext};

/// Translate a `LayerNorm` tensor op to NY's native `LayerNormLayer`.
///
/// Weight (gamma) and bias (beta) must be constant. Eps must be constant scalar.
/// NY's `LayerNormLayer` normalizes over the last axis only.
pub(super) fn translate_layer_norm(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    weight: &TensorNodeId,
    bias: &TensorNodeId,
    node_values: &[TensorNodeValue],
    all_nodes: &[nn_dsl::tensor_ir::TensorNode],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let (input_val, eps_val, input_shape) =
        validate_norm_shape_and_eps(node_values, all_nodes, input, eps, axis, "LayerNorm")?;

    let hidden_size = input_shape[input_shape.len() - 1];
    let gamma_val = extract_norm_param(node_values, weight, hidden_size, "LayerNorm", "weight")?;
    let beta_val = extract_norm_param(node_values, bias, hidden_size, "LayerNorm", "bias")?;

    match input_val {
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            let norm_mode = ctx.norm_mode;
            let layer = Layer::LayerNorm(
                LayerNormLayer::new(gamma_val, beta_val, eps_val)?
                    .with_forward_mode(norm_mode.forward_mode())
                    .with_crown_mode(norm_mode.crown_mode()),
            );
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(_c) => {
            // LayerNorm of a constant tensor where all values equal c:
            //   mean = c, variance = 0
            //   normalized = (c - c) / sqrt(0 + eps) = 0
            //   output = 0 * gamma + beta = beta
            // When beta is uniform (ConstantScalar binding), beta_val[0]
            // is the correct scalar. Non-uniform beta (ConstantTensor)
            // requires per-element output which TensorNodeValue::Constant
            // cannot represent — reject this degenerate case.
            let first = beta_val[0];
            if !beta_val.iter().all(|&v| v == first) {
                return Err(VerifyError::UnsupportedOp(
                    "LayerNorm constant-fold with non-uniform beta unsupported".into(),
                ));
            }
            Ok(TensorNodeValue::Constant(FiniteF32::new(first)?))
        }
        TensorNodeValue::WeightTensor(input_arr) => {
            // Constant-fold path: the input is a fully-known constant tensor
            // (e.g. the KV branch of a cross-attention block bound to a
            // ConstantTensor). LayerNorm of a known constant is exactly
            // computable, so we evaluate it deterministically at translation
            // time and emit the resulting constant tensor — strictly more
            // information than rejecting the graph, and no over/under-
            // approximation is introduced. This mirrors the trusted Linear
            // constant-fold (graph_tensor_linear.rs:108-137).
            //
            // The math reproduces NY's `LayerNormLayer::eval` exactly
            // (Standard mode, which is what the Variable arm constructs):
            //   mean = mean(row), var = mean((row-mean)^2)  [biased, /n]
            //   std  = sqrt(var + eps)
            //   out_i = gamma_i * (x_i - mean) / std + beta_i
            // Accumulation is done in f64 to match `eval` (#3325). `eps` is
            // floored to NORM_MIN_EPS exactly as `LayerNormLayer::new` does
            // (validate_norm_eps), guaranteeing std > 0 so the divide is never
            // 0/0 even for a zero-variance (all-equal) row.
            const NORM_MIN_EPS: f64 = 1e-12;
            let eps64 = (eps_val as f64).max(NORM_MIN_EPS);

            // The constant input's last axis must match the normalized size
            // (gamma/beta are length `hidden_size`). Reject cleanly on a
            // mismatch rather than risking an out-of-bounds panic.
            let last_len = *input_arr.shape().last().unwrap_or(&0);
            if last_len != hidden_size {
                return Err(VerifyError::UnsupportedOp(format!(
                    "LayerNorm constant-fold: input last-axis {last_len} != normalized size {hidden_size}"
                )));
            }

            let mut result = input_arr.clone();
            let last_axis = ndarray::Axis(result.ndim() - 1);
            for mut row in result.lanes_mut(last_axis) {
                let n = row.len() as f64;
                let mean = row.iter().map(|&x| x as f64).sum::<f64>() / n;
                let var = row.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
                let std = (var + eps64).sqrt();
                for (i, x) in row.iter_mut().enumerate() {
                    let g = gamma_val[i] as f64;
                    let b = beta_val[i] as f64;
                    *x = (g * (*x as f64 - mean) / std + b) as f32;
                }
            }

            // Reject non-finite outputs, exactly as the Linear fold does.
            for &val in result.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("LayerNorm constant-fold t{}", node_id.index()),
                    });
                }
            }

            Ok(TensorNodeValue::WeightTensor(result.into_dyn()))
        }
    }
}
