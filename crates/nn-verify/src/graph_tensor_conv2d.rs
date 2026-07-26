// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d tensor-level IR → NY `Conv2dLayer` translation.
//!
//! Maps `TensorOpKind::Conv2d` to `Layer::Conv2d(Conv2dLayer)`, extracting
//! weight and bias tensors from `ConstantTensor` bindings.
//!
//! Dilation support uses kernel expansion (same approach as Conv1d):
//! a dilated 2D kernel is expanded to an equivalent standard kernel with
//! zeros at non-dilation positions.

use ny_propagate::layers::Conv2dLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::{Array1, ArrayD, IxDyn};

use super::{TensorNodeValue, TensorTranslationContext};
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

/// Expand a dilated 2D kernel to an equivalent standard kernel with zero-insertion.
///
/// A Conv2d with dilation `(dh, dw)` and kernel size `(kh, kw)` is equivalent to a
/// Conv2d with dilation `(1, 1)` and kernel size `(dh*(kh-1)+1, dw*(kw-1)+1)`, where
/// the expanded kernel has zeros at non-dilation positions.
///
/// # Soundness
/// The expansion is exact: `Conv2d(x, kernel, dilation=(dh,dw)) ==
/// Conv2d(x, expanded, dilation=(1,1))` for all inputs x.
fn expand_dilated_kernel_2d(
    kernel: &ArrayD<f32>, // [out_ch, in_ch, kh, kw]
    dilation_h: usize,
    dilation_w: usize,
) -> ArrayD<f32> {
    if dilation_h <= 1 && dilation_w <= 1 {
        return kernel.clone();
    }
    let shape = kernel.shape();
    let (out_ch, in_ch, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
    let expanded_kh = dilation_h * (kh - 1) + 1;
    let expanded_kw = dilation_w * (kw - 1) + 1;
    let mut expanded = ArrayD::zeros(IxDyn(&[out_ch, in_ch, expanded_kh, expanded_kw]));
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            for ih in 0..kh {
                for iw in 0..kw {
                    expanded[[oc, ic, ih * dilation_h, iw * dilation_w]] = kernel[[oc, ic, ih, iw]];
                }
            }
        }
    }
    expanded
}

/// Exact constant-fold of a Conv2d on a fully-constant input.
///
/// Computes the cross-correlation (PyTorch/NY Conv2d convention) of a constant
/// input tensor with a constant kernel, exactly mirroring the IBP-forward
/// arithmetic in NY's `conv2d_ibp_forward` (zero-padding outside the input,
/// strided + dilated taps): for each output position,
/// `out[oc,oh,ow] = bias[oc] + Σ_{ic,ki,kj} input[ic, oh*sh+ki*dh-ph, ow*sw+kj*dw-pw] * kernel[oc,ic,ki,kj]`.
///
/// # Soundness
/// The input, kernel and bias are all known finite constants, so the result is
/// a single deterministic value per output element — `lower == upper`. This is
/// the exact value NY would propagate for a degenerate (point) input interval;
/// it introduces no interval/relaxation and cannot exclude any reachable value.
/// Caller passes the dilation-expanded kernel with `dilation = (1, 1)` so this
/// matches the variable path, which feeds the same expanded kernel to NY.
#[allow(clippy::too_many_arguments)]
fn constant_fold_conv2d(
    input: &ArrayD<f32>, // [C, H, W] or [B, C, H, W]
    kernel: &ArrayD<f32>, // [out_c, in_c, kh, kw]
    bias: Option<&Array1<f32>>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<ArrayD<f32>, VerifyError> {
    let in_shape = input.shape();
    // Accept [C, H, W] (3-D) or [B, C, H, W] (4-D); fold over the leading batch.
    let (batch, in_c, in_h, in_w) = match in_shape.len() {
        3 => (1usize, in_shape[0], in_shape[1], in_shape[2]),
        4 => (in_shape[0], in_shape[1], in_shape[2], in_shape[3]),
        other => {
            return Err(VerifyError::UnsupportedOp(format!(
                "Conv2d constant-fold: input must be 3-D [C,H,W] or 4-D [B,C,H,W], got {other}-D"
            )));
        }
    };

    let k_shape = kernel.shape();
    let (out_c, ker_in_c, kh, kw) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    if ker_in_c != in_c {
        return Err(VerifyError::UnsupportedOp(format!(
            "Conv2d constant-fold: kernel in_channels {ker_in_c} != input channels {in_c}"
        )));
    }

    let padded_h = in_h + 2 * padding_h;
    let padded_w = in_w + 2 * padding_w;
    if padded_h < kh || padded_w < kw {
        return Err(VerifyError::UnsupportedOp(format!(
            "Conv2d constant-fold: kernel ({kh},{kw}) larger than padded input ({padded_h},{padded_w})"
        )));
    }
    let out_h = (padded_h - kh) / stride_h + 1;
    let out_w = (padded_w - kw) / stride_w + 1;

    // Reshape input to a [B, C, H, W] view for uniform indexing.
    let input4 = input
        .view()
        .into_shape_with_order(IxDyn(&[batch, in_c, in_h, in_w]))
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("Conv2d constant-fold input reshape: {e}"),
        })?;

    let mut out = ArrayD::<f32>::zeros(IxDyn(&[batch, out_c, out_h, out_w]));
    for b in 0..batch {
        for oc in 0..out_c {
            let bias_oc = bias.map_or(0.0, |arr| arr[oc]);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_oc;
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            let ih = (oh * stride_h + ki) as isize - padding_h as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kj in 0..kw {
                                let iw = (ow * stride_w + kj) as isize - padding_w as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                acc += input4[[b, ic, ih as usize, iw as usize]]
                                    * kernel[[oc, ic, ki, kj]];
                            }
                        }
                    }
                    out[[b, oc, oh, ow]] = acc;
                }
            }
        }
    }

    // Validate finiteness (checked_constant pattern): a constant fold must
    // never emit a NaN/Inf that a downstream op would silently propagate.
    for &val in out.iter() {
        if !val.is_finite() {
            return Err(VerifyError::NonFiniteConstant {
                value: val,
                context: "Conv2d constant-fold".into(),
            });
        }
    }

    // Drop the synthetic leading batch dim for 3-D inputs so the output shape
    // matches the variable path ([out_c, out_h, out_w]).
    if in_shape.len() == 3 {
        out.into_shape_with_order(IxDyn(&[out_c, out_h, out_w]))
            .map_err(|e| VerifyError::InternalTranslationError {
                context: format!("Conv2d constant-fold output reshape: {e}"),
            })
    } else {
        Ok(out)
    }
}

/// Translate a Conv2d tensor operation to a NY graph node.
///
/// The input data must be a `Variable` (the tensor being verified).
/// Weight and bias must be `WeightTensor` (fixed model parameters).
///
/// Creates a `Layer::Conv2d(Conv2dLayer)` node with the weight kernel,
/// optional bias, stride, padding, and input spatial dims set for CROWN
/// backward propagation.
pub(super) fn translate_conv2d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // NY Conv2dLayer does not support groups.
    // Silently dropping groups would produce unsound bounds.
    if groups != 1 {
        return Err(VerifyError::UnsupportedOp(format!(
            "Conv2d groups={groups} not supported by NY Conv2dLayer (only groups=1)"
        )));
    }

    // Weight must be a WeightTensor (constant kernel parameters).
    let raw_kernel = match get_value(node_values, weight.index(), "Conv2d weight")? {
        TensorNodeValue::WeightTensor(arr) => arr.clone(),
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "Conv2d",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // Dilated Conv2d workaround: expand the kernel with zero-insertion so
    // NY's Conv2dLayer (which only supports dilation=1) produces
    // identical output.
    let kernel_array = if dilation_h > 1 || dilation_w > 1 {
        expand_dilated_kernel_2d(&raw_kernel, dilation_h, dilation_w)
    } else {
        raw_kernel
    };

    // Bias extraction (optional).
    let bias_array = if let Some(bias_id) = bias {
        match get_value(node_values, bias_id.index(), "Conv2d bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                // Convert from ArrayD to Array1 for NY's API.
                let flat: Vec<f32> = arr.iter().copied().collect();
                Some(Array1::from_vec(flat))
            }
            _ => {
                return Err(VerifyError::WeightValidation {
                    op: "Conv2d",
                    reason: "bias must be a ConstantTensor binding".into(),
                });
            }
        }
    } else {
        None
    };

    // Branch on input kind.
    //
    // Constant-fold path: a fully-constant input (e.g. a fixed coarse feature
    // map projected by a 1x1 conv) is convolved eagerly into a constant output.
    // This is EXACT (input, kernel and bias are known finite constants), so the
    // result is a deterministic WeightTensor with lower == upper — no interval
    // or relaxation is introduced. Mirrors the Linear constant-fold
    // (graph_tensor_linear.rs:108-137).
    let input_name = match get_value(node_values, input.index(), "Conv2d input")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Conv2d input must be a variable tensor, not a constant scalar".into(),
            ));
        }
        TensorNodeValue::WeightTensor(input_arr) => {
            let folded = constant_fold_conv2d(
                input_arr,
                &kernel_array,
                bias_array.as_ref(),
                stride_h,
                stride_w,
                padding_h,
                padding_w,
            )?;
            return Ok(TensorNodeValue::WeightTensor(folded));
        }
    };

    // Get input spatial dimensions for CROWN backward propagation.
    let input_node =
        ctx.all_nodes
            .get(input.index())
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: format!(
                    "Conv2d input node index {} out of bounds (len {})",
                    input.index(),
                    ctx.all_nodes.len()
                ),
            })?;
    let input_shape = &input_node.shape;
    // Input shape is [C, H, W] or [B, C, H, W]; take last two dims.
    if input_shape.len() < 2 {
        return Err(VerifyError::UnsupportedOp(
            "Conv2d input shape must have at least 2 dimensions (H, W)".into(),
        ));
    }
    let in_height = input_shape[input_shape.len() - 2];
    let in_width = input_shape[input_shape.len() - 1];

    // Build NY Conv2dLayer with input_shape set.
    let conv_layer = Conv2dLayer::with_input_shape(
        kernel_array,
        bias_array,
        (stride_h, stride_w),
        (padding_h, padding_w),
        in_height,
        in_width,
    )
    .map_err(|e| VerifyError::WeightValidation {
        op: "Conv2d",
        reason: format!("layer construction failed: {e}"),
    })?;

    let node_name = format!("t{}", node_id.index());
    let layer = Layer::Conv2d(conv_layer);
    add_unary_node(&node_name, layer, &input_name, graph);

    Ok(TensorNodeValue::Variable(node_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_dilated_kernel_2d_passthrough() {
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let expanded = expand_dilated_kernel_2d(&kernel, 1, 1);
        assert_eq!(expanded.shape(), &[1, 1, 3, 3]);
        assert_eq!(expanded, kernel);
    }

    #[test]
    fn test_expand_dilated_kernel_2d_dilation_2() {
        // kernel [1, 1, 2, 2] = [[1, 2], [3, 4]]
        // dilation (2, 2) -> expanded [1, 1, 3, 3]
        // [[1, 0, 2], [0, 0, 0], [3, 0, 4]]
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let expanded = expand_dilated_kernel_2d(&kernel, 2, 2);
        assert_eq!(expanded.shape(), &[1, 1, 3, 3]);
        assert_eq!(expanded[[0, 0, 0, 0]], 1.0);
        assert_eq!(expanded[[0, 0, 0, 1]], 0.0);
        assert_eq!(expanded[[0, 0, 0, 2]], 2.0);
        assert_eq!(expanded[[0, 0, 1, 0]], 0.0);
        assert_eq!(expanded[[0, 0, 1, 1]], 0.0);
        assert_eq!(expanded[[0, 0, 1, 2]], 0.0);
        assert_eq!(expanded[[0, 0, 2, 0]], 3.0);
        assert_eq!(expanded[[0, 0, 2, 1]], 0.0);
        assert_eq!(expanded[[0, 0, 2, 2]], 4.0);
    }

    #[test]
    fn test_expand_dilated_kernel_2d_asymmetric() {
        // dilation (1, 3) — expand only width
        // kernel [1, 1, 2, 2] -> expanded [1, 1, 2, 4]
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let expanded = expand_dilated_kernel_2d(&kernel, 1, 3);
        assert_eq!(expanded.shape(), &[1, 1, 2, 4]); // h: 1*(2-1)+1=2, w: 3*(2-1)+1=4
        assert_eq!(expanded[[0, 0, 0, 0]], 1.0);
        assert_eq!(expanded[[0, 0, 0, 3]], 2.0);
        assert_eq!(expanded[[0, 0, 1, 0]], 3.0);
        assert_eq!(expanded[[0, 0, 1, 3]], 4.0);
        // Zeros at non-dilation positions
        assert_eq!(expanded[[0, 0, 0, 1]], 0.0);
        assert_eq!(expanded[[0, 0, 0, 2]], 0.0);
    }
}
