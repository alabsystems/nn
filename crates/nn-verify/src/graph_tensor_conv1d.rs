// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d tensor-level IR → NY `Conv1dLayer` translation.
//!
//! Extracted from `graph_tensor.rs` following the module-per-op pattern
//! (see `graph_tensor_instance_norm.rs`, `graph_tensor_structural.rs`).
//!
//! Maps `TensorOpKind::Conv1d` to `Layer::Conv1d(Conv1dLayer)`, extracting
//! weight and bias tensors from `ConstantTensor` bindings.
//!
//! Dilation and groups are passed directly to NY's `Conv1dLayer`
//! via `with_input_length_full`, which supports them natively.
//! The `expand_dilated_kernel` helper is retained for backward compatibility
//! with existing compose tests that use it directly.

use ny_propagate::layers::Conv1dLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::Array1;

use super::{TensorNodeValue, TensorTranslationContext};
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

/// Expand a dilated kernel to an equivalent standard kernel with zero-insertion.
///
/// A Conv1d with dilation `d` and kernel size `k` is equivalent to a
/// Conv1d with dilation 1 and kernel size `d*(k-1)+1`, where the expanded
/// kernel has zeros at non-dilation positions.
///
/// # Soundness
/// The expansion is exact: `Conv1d(x, kernel, dilation=d) == Conv1d(x, expanded, dilation=1)`
/// for all inputs x. IBP/CROWN bounds through the expanded kernel are identical (not just
/// sound) to native dilated conv — zero entries contribute 0 to both W+/W- and CROWN backward.
#[cfg(test)]
fn expand_dilated_kernel(
    kernel: &ndarray::ArrayD<f32>, // [out_ch, in_ch, k]
    dilation: usize,
) -> ndarray::ArrayD<f32> {
    if dilation <= 1 {
        return kernel.clone();
    }
    let shape = kernel.shape();
    let (out_ch, in_ch, k) = (shape[0], shape[1], shape[2]);
    let expanded_k = dilation * (k - 1) + 1;
    let mut expanded = ndarray::ArrayD::zeros(ndarray::IxDyn(&[out_ch, in_ch, expanded_k]));
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            for i in 0..k {
                expanded[[oc, ic, i * dilation]] = kernel[[oc, ic, i]];
            }
        }
    }
    expanded
}

/// Translate a Conv1d tensor operation to a NY graph node.
///
/// The input data must be a `Variable` (the tensor being verified).
/// Weight and bias must be `WeightTensor` (fixed model parameters).
///
/// Creates a `Layer::Conv1d(Conv1dLayer)` node with the weight kernel,
/// optional bias, stride, padding, and input_length set for CROWN
/// backward propagation.
pub(super) fn translate_conv1d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // Input must be a Variable (data tensor being verified).
    let input_name = match get_value(node_values, input.index(), "Conv1d input")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Conv1d input must be a variable tensor, not a constant scalar".into(),
            ));
        }
        TensorNodeValue::WeightTensor(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Conv1d input must be a variable tensor, not a weight tensor".into(),
            ));
        }
    };

    // Weight must be a WeightTensor (constant kernel parameters).
    let raw_kernel = match get_value(node_values, weight.index(), "Conv1d weight")? {
        TensorNodeValue::WeightTensor(arr) => arr.clone(),
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "Conv1d",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // NY Conv1dLayer natively supports dilation and groups via
    // with_input_length_full — no kernel expansion workaround needed.
    let kernel_array = raw_kernel;

    // Bias extraction (optional).
    let bias_array = if let Some(bias_id) = bias {
        match get_value(node_values, bias_id.index(), "Conv1d bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                // Convert from ArrayD to Array1 for NY's API.
                let flat: Vec<f32> = arr.iter().copied().collect();
                Some(Array1::from_vec(flat))
            }
            _ => {
                return Err(VerifyError::WeightValidation {
                    op: "Conv1d",
                    reason: "bias must be a ConstantTensor binding".into(),
                });
            }
        }
    } else {
        None
    };

    // Get input spatial length for CROWN backward propagation (bounds-checked).
    let input_node =
        ctx.all_nodes
            .get(input.index())
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: format!(
                    "Conv1d input node index {} out of bounds (len {})",
                    input.index(),
                    ctx.all_nodes.len()
                ),
            })?;
    let in_length =
        *input_node
            .shape
            .last()
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: "Conv1d input shape is empty".into(),
            })?;

    // Build NY Conv1dLayer with dilation, groups, and input_length.
    let conv_layer = Conv1dLayer::with_input_length_full(
        kernel_array,
        bias_array,
        stride,
        padding,
        dilation,
        groups,
        in_length,
    )
    .map_err(|e| VerifyError::UnsupportedOp(format!("Conv1dLayer construction failed: {e}")))?;

    let node_name = format!("t{}", node_id.index());
    let layer = Layer::Conv1d(conv_layer);
    add_unary_node(&node_name, layer, &input_name, graph);

    Ok(TensorNodeValue::Variable(node_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_expand_dilated_kernel_passthrough_dilation_1() {
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expanded = expand_dilated_kernel(&kernel, 1);
        assert_eq!(expanded.shape(), &[2, 1, 3]);
        assert_eq!(expanded, kernel);
    }

    #[test]
    fn test_expand_dilated_kernel_dilation_2() {
        // kernel [1, 1, 3] = [1.0, 2.0, 3.0]
        // expanded [1, 1, 5] = [1.0, 0.0, 2.0, 0.0, 3.0]
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expanded = expand_dilated_kernel(&kernel, 2);
        assert_eq!(expanded.shape(), &[1, 1, 5]);
        assert_eq!(expanded[[0, 0, 0]], 1.0);
        assert_eq!(expanded[[0, 0, 1]], 0.0);
        assert_eq!(expanded[[0, 0, 2]], 2.0);
        assert_eq!(expanded[[0, 0, 3]], 0.0);
        assert_eq!(expanded[[0, 0, 4]], 3.0);
    }

    #[test]
    fn test_expand_dilated_kernel_dilation_8_dvoice() {
        // dvoice DConv: k=3, dilation=8 -> expanded k = 8*(3-1)+1 = 17
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![0.5, -0.3, 0.8]).unwrap();
        let expanded = expand_dilated_kernel(&kernel, 8);
        assert_eq!(expanded.shape(), &[1, 1, 17]);
        // Original positions at 0, 8, 16
        assert_eq!(expanded[[0, 0, 0]], 0.5);
        assert_eq!(expanded[[0, 0, 8]], -0.3);
        assert_eq!(expanded[[0, 0, 16]], 0.8);
        // All other positions are zero
        for i in 0..17 {
            if i != 0 && i != 8 && i != 16 {
                assert_eq!(expanded[[0, 0, i]], 0.0, "position {i} should be zero");
            }
        }
    }

    #[test]
    fn test_expand_dilated_kernel_multi_channel() {
        // [2, 2, 2] kernel, dilation=3 -> expanded [2, 2, 4]
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let expanded = expand_dilated_kernel(&kernel, 3);
        assert_eq!(expanded.shape(), &[2, 2, 4]); // 3*(2-1)+1 = 4
                                                  // oc=0, ic=0: [1.0, 0, 0, 2.0]
        assert_eq!(expanded[[0, 0, 0]], 1.0);
        assert_eq!(expanded[[0, 0, 1]], 0.0);
        assert_eq!(expanded[[0, 0, 2]], 0.0);
        assert_eq!(expanded[[0, 0, 3]], 2.0);
        // oc=0, ic=1: [3.0, 0, 0, 4.0]
        assert_eq!(expanded[[0, 1, 0]], 3.0);
        assert_eq!(expanded[[0, 1, 3]], 4.0);
        // oc=1, ic=0: [5.0, 0, 0, 6.0]
        assert_eq!(expanded[[1, 0, 0]], 5.0);
        assert_eq!(expanded[[1, 0, 3]], 6.0);
        // oc=1, ic=1: [7.0, 0, 0, 8.0]
        assert_eq!(expanded[[1, 1, 0]], 7.0);
        assert_eq!(expanded[[1, 1, 3]], 8.0);
    }
}
