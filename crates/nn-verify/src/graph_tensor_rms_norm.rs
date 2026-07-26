// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RmsNorm tensor op → NY translation.
//!
//! Maps `TensorOpKind::RmsNorm` to `Layer::RmsNorm(RmsNormLayer)`.
//! Follows the same pattern as `graph_tensor_instance_norm.rs`.

use ny_propagate::layers::RmsNormLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, FiniteF32};

use super::norm_util::{extract_norm_param, validate_norm_shape_and_eps};
use super::{TensorNodeValue, TensorTranslationContext};

/// Translate an `RmsNorm` tensor op to NY's native `RmsNormLayer`.
///
/// Weight (gamma) must be constant. Eps must be constant scalar.
/// NY's `RmsNormLayer` normalizes over the last axis only.
pub(super) fn translate_rms_norm(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    weight: &TensorNodeId,
    node_values: &[TensorNodeValue],
    all_nodes: &[nn_dsl::tensor_ir::TensorNode],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let (input_val, eps_val, input_shape) =
        validate_norm_shape_and_eps(node_values, all_nodes, input, eps, axis, "RmsNorm")?;

    let hidden_size = input_shape[input_shape.len() - 1];
    let weight_val = extract_norm_param(node_values, weight, hidden_size, "RmsNorm", "weight")?;

    match input_val {
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            let norm_mode = ctx.norm_mode;
            let layer = Layer::RmsNorm(
                RmsNormLayer::new(weight_val, eps_val)?
                    .with_forward_mode(norm_mode.forward_mode())
                    .with_crown_mode(norm_mode.crown_mode()),
            );
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(c) => {
            // RmsNorm of a constant tensor where all values equal c:
            //   RMS = sqrt(c² + eps)
            //   output = (c / RMS) * weight = c / sqrt(c² + eps) * weight
            // When weight is uniform (ConstantScalar binding), weight_val[0]
            // is the correct scalar. Non-uniform weight (ConstantTensor)
            // requires per-element output which TensorNodeValue::Constant
            // cannot represent — reject this degenerate case.
            let first_w = weight_val[0];
            if !weight_val.iter().all(|&v| v == first_w) {
                return Err(VerifyError::UnsupportedOp(
                    "RmsNorm constant-fold with non-uniform weight unsupported".into(),
                ));
            }
            let c_val = c.get();
            let rms = (c_val * c_val + eps_val).sqrt();
            let result = if rms == 0.0 {
                0.0
            } else {
                (c_val / rms) * first_w
            };
            Ok(TensorNodeValue::Constant(FiniteF32::new(result)?))
        }
        TensorNodeValue::WeightTensor(input_arr) => {
            // Constant-fold path: the input is a fully-known constant tensor
            // (e.g. dec_input bound to a ConstantTensor feeding a self-attn
            // block's RmsNorm). RmsNorm of a known constant is exactly
            // computable, so we evaluate it deterministically at translation
            // time and emit the resulting constant tensor — strictly more
            // information than rejecting the graph, with no over/under-
            // approximation. This mirrors the trusted Linear constant-fold
            // (graph_tensor_linear.rs:108-137) and the scalar RmsNorm path above.
            //
            // The math reproduces NY's `RmsNormLayer::eval` exactly:
            //   mean_sq = mean(row^2)
            //   rms     = sqrt(mean_sq + eps)
            //   out_i   = weight_i * x_i / rms
            // Accumulation is done in f64 to match `eval` (#3325). `eps` is
            // floored to NORM_MIN_EPS exactly as `RmsNormLayer::new` does
            // (validate_norm_eps), guaranteeing rms > 0 so the divide is never
            // 0/0 even for an all-zero row.
            const NORM_MIN_EPS: f64 = 1e-12;
            let eps64 = (eps_val as f64).max(NORM_MIN_EPS);

            // The constant input's last axis must match the normalized size
            // (weight is length `hidden_size`). Reject cleanly on a mismatch
            // rather than risking an out-of-bounds panic.
            let last_len = *input_arr.shape().last().unwrap_or(&0);
            if last_len != hidden_size {
                return Err(VerifyError::UnsupportedOp(format!(
                    "RmsNorm constant-fold: input last-axis {last_len} != normalized size {hidden_size}"
                )));
            }

            let mut result = input_arr.clone();
            let last_axis = ndarray::Axis(result.ndim() - 1);
            for mut row in result.lanes_mut(last_axis) {
                let n = row.len() as f64;
                let mean_sq = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n;
                let rms = (mean_sq + eps64).sqrt();
                for (i, x) in row.iter_mut().enumerate() {
                    let w = weight_val[i] as f64;
                    *x = (w * (*x as f64) / rms) as f32;
                }
            }

            // Reject non-finite outputs, exactly as the Linear fold does.
            for &val in result.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("RmsNorm constant-fold t{}", node_id.index()),
                    });
                }
            }

            Ok(TensorNodeValue::WeightTensor(result.into_dyn()))
        }
    }
}
