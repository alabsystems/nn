// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d tensor-level IR → NY `Conv2dLayer` translation.
//!
//! Maps `TensorOpKind::Conv2d` to `Layer::Conv2d(Conv2dLayer)`, extracting
//! weight and bias tensors from `ConstantTensor` bindings.
//!
//! ## Dilation — native, not zero-inserted
//!
//! This module used to work around an NY limitation by expanding a dilated
//! kernel with zero-insertion (`[oc, ic, kh, kw]` → `[oc, ic, d_h*(kh-1)+1,
//! d_w*(kw-1)+1]`) and handing NY a `dilation=(1,1)` layer. **That limitation no
//! longer exists**: `ny_propagate::layers::Conv2dLayer` carries a `dilation`
//! field (`Conv2dLayer::new_dilated`), threads it through `output_size`, the
//! im2col/GEMM IBP forward (`conv2d_ibp_forward_grouped`), and both the dense
//! and patches CROWN backward transposes. We therefore emit the dilation
//! directly.
//!
//! Zero-insertion was never *unsound* — it computes the identical function —
//! but it inflates the kernel by up to `d_h*d_w`, and every inserted zero is a
//! real tap in NY's im2col matrix and in the CROWN coefficient block.
//!
//! MEASURED (`tests::dilated_conv2d_native_is_no_looser_than_zero_insertion`, a
//! `[2,8,8]` input with a 3x3 kernel at dilation `(2,2)`, stride 1, padding 2):
//! the two forms give **bit-identical IBP widths** — expected, since a zero
//! weight contributes exactly `0` on both sides of NY's W⁺/W⁻ interval split —
//! while the kernel shrinks from **150 elements to 54** (5x5 vs 3x3 per
//! `(oc, ic)` plane, a 2.78x reduction in taps). So the gain here is not
//! tightness on this fixture but the work every dilated conv does: 2.78x fewer
//! im2col columns and a 2.78x smaller CROWN coefficient block, and one less
//! place where nn's translation and NY's geometry can disagree. The test asserts
//! *no looser*, elementwise and in aggregate, so a future NY change that made
//! the native path wider would fail it.

use ndarray::{Array1, ArrayD, IxDyn};
use nn_dsl::tensor_ir::TensorNodeId;
use ny_propagate::layers::Conv2dLayer;
use ny_propagate::{GraphNetwork, Layer};

use super::{TensorNodeValue, TensorTranslationContext};
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

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
/// The dilation `(dh, dw)` passed here is the SAME dilation handed to
/// `Conv2dLayer::new_dilated` on the variable path, so both paths evaluate the
/// identical tap geometry.
#[allow(clippy::too_many_arguments)]
fn constant_fold_conv2d(
    input: &ArrayD<f32>,  // [C, H, W] or [B, C, H, W]
    kernel: &ArrayD<f32>, // [out_c, in_c, kh, kw]
    bias: Option<&Array1<f32>>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
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

    // Effective (dilated) kernel span — the same formula NY's
    // `Conv2dLayer::output_size` uses: `dilation * (kernel - 1) + 1`.
    let eff_kh = dilation_h * (kh - 1) + 1;
    let eff_kw = dilation_w * (kw - 1) + 1;
    let padded_h = in_h + 2 * padding_h;
    let padded_w = in_w + 2 * padding_w;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(VerifyError::UnsupportedOp(format!(
            "Conv2d constant-fold: effective kernel ({eff_kh},{eff_kw}) larger than padded input \
             ({padded_h},{padded_w})"
        )));
    }
    let out_h = (padded_h - eff_kh) / stride_h + 1;
    let out_w = (padded_w - eff_kw) / stride_w + 1;

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
                            let ih =
                                (oh * stride_h + ki * dilation_h) as isize - padding_h as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kj in 0..kw {
                                let iw =
                                    (ow * stride_w + kj * dilation_w) as isize - padding_w as isize;
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
/// optional bias, stride, padding, dilation, and input spatial dims set for
/// CROWN backward propagation.
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
    // Grouped Conv2d does not arrive here: `graph_tensor_dispatch.rs` routes
    // `groups > 1` to `pool::translate_conv2d_grouped`, which forwards `groups`
    // natively (NY's `Conv2dLayer` models it — `new_dilated` takes a `groups`
    // argument and `conv2d_ibp_forward_grouped`/the grouped CROWN transpose are
    // group-aware). This translator serves the single-group case, whose
    // constant-fold below reads the kernel as ungrouped `[out_c, in_c, kh, kw]`.
    // `groups == 0` is malformed geometry and DOES reach here (the dispatcher
    // tests `groups <= 1`), so the guard stays, fail-closed.
    if groups != 1 {
        return Err(VerifyError::UnsupportedOp(format!(
            "Conv2d groups={groups} is not a single-group convolution — groups>1 is routed to \
             the grouped translator, groups=0 is malformed"
        )));
    }

    // Weight must be a WeightTensor (constant kernel parameters).
    let kernel_array = match get_value(node_values, weight.index(), "Conv2d weight")? {
        TensorNodeValue::WeightTensor(arr) => arr.clone(),
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "Conv2d",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // Dilation is emitted NATIVELY (see the module docs): no zero-insertion,
    // so the kernel NY sees is the model's own `[oc, ic, kh, kw]` rather than a
    // `d_h*d_w`-times larger mostly-zero one.
    if dilation_h == 0 || dilation_w == 0 {
        return Err(VerifyError::UnsupportedOp(format!(
            "Conv2d dilation must be >= 1 in both dimensions, got ({dilation_h},{dilation_w})"
        )));
    }

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
                dilation_h,
                dilation_w,
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

    // Build NY Conv2dLayer with the native dilation and input_shape set.
    // `new_dilated` validates stride/dilation/groups and the kernel rank;
    // `set_input_shape` is what CROWN backward needs (there is no
    // `with_input_shape` constructor carrying dilation).
    let mut conv_layer = Conv2dLayer::new_dilated(
        kernel_array,
        bias_array,
        (stride_h, stride_w),
        (padding_h, padding_w),
        (dilation_h, dilation_w),
        1,
    )
    .map_err(|e| VerifyError::WeightValidation {
        op: "Conv2d",
        reason: format!("layer construction failed: {e}"),
    })?;
    conv_layer.set_input_shape(in_height, in_width);

    let node_name = format!("t{}", node_id.index());
    let layer = Layer::Conv2d(conv_layer);
    add_unary_node(&node_name, layer, &input_name, graph);

    Ok(TensorNodeValue::Variable(node_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundedTensor;
    use ny_propagate::BoundPropagation;

    /// The RETIRED zero-insertion workaround, kept as the "before" oracle for
    /// the tightness measurement below: a Conv2d with dilation `(dh, dw)` and
    /// kernel `(kh, kw)` equals a `dilation=(1,1)` Conv2d whose kernel is
    /// `(dh*(kh-1)+1, dw*(kw-1)+1)` with zeros at non-dilation positions.
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
                        expanded[[oc, ic, ih * dilation_h, iw * dilation_w]] =
                            kernel[[oc, ic, ih, iw]];
                    }
                }
            }
        }
        expanded
    }

    /// Deterministic pseudo-random spread over `[-1, 1]`, so the fixture is a
    /// real mixed-sign kernel rather than an accidentally-symmetric one.
    fn spread(i: usize, salt: usize) -> f32 {
        ((i * 37 + salt * 101) % 199) as f32 / 99.0 - 1.0
    }

    /// Shared fixture: DETR/DConv-shaped dilated conv — `[2, 8, 8]` input,
    /// 3 output channels, 3x3 kernel, dilation (2, 2), stride 1, padding 2.
    /// Zero-insertion turns the 3x3 kernel into 5x5 (9 real taps -> 25).
    struct DilatedFixture {
        kernel: ArrayD<f32>,
        bias: Array1<f32>,
        input: BoundedTensor,
        dilation: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        in_h: usize,
        in_w: usize,
    }

    fn dilated_fixture() -> DilatedFixture {
        let (out_c, in_c, kh, kw) = (3usize, 2usize, 3usize, 3usize);
        let kdata: Vec<f32> = (0..out_c * in_c * kh * kw).map(|i| spread(i, 3)).collect();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), kdata).expect("kernel");
        let bias = Array1::from_vec(vec![0.25, -0.5, 0.125]);

        let (in_h, in_w) = (8usize, 8usize);
        let center: Vec<f32> = (0..in_c * in_h * in_w).map(|i| spread(i, 7)).collect();
        let lower: Vec<f32> = center.iter().map(|&c| c - 0.05).collect();
        let upper: Vec<f32> = center.iter().map(|&c| c + 0.05).collect();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), lower).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), upper).expect("upper"),
        )
        .expect("valid input box");

        DilatedFixture {
            kernel,
            bias,
            input,
            dilation: (2, 2),
            stride: (1, 1),
            padding: (2, 2),
            in_h,
            in_w,
        }
    }

    /// Sum of interval widths over every output element — the tightness metric.
    fn total_width(t: &BoundedTensor) -> f64 {
        let (lo, hi) = t.lower_upper();
        lo.iter()
            .zip(hi.iter())
            .map(|(&l, &u)| f64::from(u) - f64::from(l))
            .sum()
    }

    // -- Task 2: native dilation replaces the zero-insertion workaround --

    /// BEFORE/AFTER measurement. `ny_propagate::Conv2dLayer` carries a real
    /// `dilation` field today, so nn no longer zero-inserts. Both forms compute
    /// the same function; this pins that the native form is (a) an enclosure of
    /// the same concrete evaluations and (b) NO LOOSER than the zero-inserted
    /// form it replaced, while carrying a 25/9-times smaller kernel.
    #[test]
    fn dilated_conv2d_native_is_no_looser_than_zero_insertion() {
        let f = dilated_fixture();

        // AFTER: native dilation, model-sized 3x3 kernel.
        let mut native = Conv2dLayer::new_dilated(
            f.kernel.clone(),
            Some(f.bias.clone()),
            f.stride,
            f.padding,
            f.dilation,
            1,
        )
        .expect("native dilated layer");
        native.set_input_shape(f.in_h, f.in_w);

        // BEFORE: zero-inserted 5x5 kernel, dilation (1, 1).
        let expanded = expand_dilated_kernel_2d(&f.kernel, f.dilation.0, f.dilation.1);
        let mut inserted = Conv2dLayer::new_dilated(
            expanded.clone(),
            Some(f.bias.clone()),
            f.stride,
            f.padding,
            (1, 1),
            1,
        )
        .expect("zero-inserted layer");
        inserted.set_input_shape(f.in_h, f.in_w);

        // The workaround really did inflate the kernel.
        assert_eq!(f.kernel.shape(), &[3, 2, 3, 3]);
        assert_eq!(expanded.shape(), &[3, 2, 5, 5]);
        assert_eq!(
            expanded.len(),
            f.kernel.len() * 25 / 9,
            "zero-insertion inflates 9 taps per (oc, ic) plane to 25"
        );

        let native_out = native.propagate_ibp(&f.input).expect("native IBP");
        let inserted_out = inserted.propagate_ibp(&f.input).expect("zero-inserted IBP");
        assert_eq!(native_out.shape(), inserted_out.shape());

        let (nlo, nhi) = native_out.lower_upper();
        let (ilo, ihi) = inserted_out.lower_upper();

        // (a) SOUNDNESS: the native bounds enclose the exact concrete evaluation
        // at the box centre, at the lower corner, and at the upper corner.
        let (blo, bhi) = f.input.lower_upper();
        let centre = (blo.to_owned() + bhi.to_owned()).mapv(|v| v / 2.0);
        for concrete in [&blo.to_owned(), &bhi.to_owned(), &centre] {
            let exact = constant_fold_conv2d(
                concrete,
                &f.kernel,
                Some(&f.bias),
                f.stride.0,
                f.stride.1,
                f.padding.0,
                f.padding.1,
                f.dilation.0,
                f.dilation.1,
            )
            .expect("exact dilated evaluation");
            assert_eq!(exact.shape(), nlo.shape());
            for (i, ((&v, &l), &u)) in exact.iter().zip(nlo.iter()).zip(nhi.iter()).enumerate() {
                assert!(
                    l <= v + 1e-5 && v <= u + 1e-5,
                    "native bound [{l}, {u}] must enclose the concrete value {v} at index {i}"
                );
            }
        }

        // (b) TIGHTNESS: no looser than the zero-inserted form, elementwise.
        const SLACK: f32 = 1e-6;
        for (i, (((&nl, &nu), &il), &iu)) in nlo
            .iter()
            .zip(nhi.iter())
            .zip(ilo.iter())
            .zip(ihi.iter())
            .enumerate()
        {
            assert!(
                nu - nl <= (iu - il) + SLACK,
                "index {i}: native width {} must not exceed zero-inserted width {}",
                nu - nl,
                iu - il
            );
        }

        let native_width = total_width(&native_out);
        let inserted_width = total_width(&inserted_out);
        println!(
            "Conv2d dilation (2,2) 3x3 kernel: native total IBP width={native_width:.6} \
             (kernel {} elems) vs zero-inserted {inserted_width:.6} (kernel {} elems)",
            f.kernel.len(),
            expanded.len(),
        );
        assert!(
            native_width <= inserted_width + 1e-4,
            "aggregate: native total width {native_width} must not exceed zero-inserted \
             {inserted_width}"
        );
    }

    /// The emitted layer must carry the model's own kernel and the dilation —
    /// i.e. the workaround is really gone from the translation, not just from
    /// the docs.
    #[test]
    fn translate_conv2d_emits_native_dilation_and_unexpanded_kernel() {
        let f = dilated_fixture();
        let node = nn_dsl::tensor_ir::TensorNode::new(
            TensorNodeId::new(0),
            nn_dsl::tensor_ir::TensorOpKind::Reshape {
                input: TensorNodeId::new(0),
                target_shape: vec![2, f.in_h, f.in_w],
            },
            vec![2, f.in_h, f.in_w],
        );
        let all_nodes = vec![node];
        let ctx = TensorTranslationContext {
            input_bindings: &[],
            input_node_names: &[],
            axis_offset: 0,
            all_nodes: &all_nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };
        let node_values = vec![
            TensorNodeValue::Variable(ny_propagate::NETWORK_INPUT.to_string()),
            TensorNodeValue::WeightTensor(f.kernel.clone()),
        ];
        let mut graph = GraphNetwork::new();
        let out = translate_conv2d(
            &ctx,
            TensorNodeId::new(2),
            &TensorNodeId::new(0),
            &TensorNodeId::new(1),
            &None,
            f.stride.0,
            f.stride.1,
            f.padding.0,
            f.padding.1,
            f.dilation.0,
            f.dilation.1,
            1,
            &node_values,
            &mut graph,
        )
        .expect("conv2d translation");
        assert!(matches!(out, TensorNodeValue::Variable(ref n) if n == "t2"));

        let emitted = graph.node("t2").expect("emitted node");
        let Layer::Conv2d(layer) = emitted.layer() else {
            panic!("expected Conv2d layer, got {:?}", emitted.layer());
        };
        assert_eq!(layer.dilation, f.dilation, "dilation emitted natively");
        assert_eq!(
            layer.kernel.shape(),
            &[3, 2, 3, 3],
            "kernel must NOT be zero-inserted"
        );
        assert_eq!(
            layer.input_shape,
            Some((f.in_h, f.in_w)),
            "CROWN backward needs input_shape"
        );
    }

    /// The constant-fold path must use the SAME tap geometry as the layer it
    /// replaces: fold a point input and compare against NY's own IBP on a
    /// degenerate (point) box.
    #[test]
    fn constant_fold_conv2d_matches_native_dilated_layer_on_a_point() {
        let f = dilated_fixture();
        let (blo, bhi) = f.input.lower_upper();
        let point = (blo.to_owned() + bhi.to_owned()).mapv(|v| v / 2.0);

        let folded = constant_fold_conv2d(
            &point,
            &f.kernel,
            Some(&f.bias),
            f.stride.0,
            f.stride.1,
            f.padding.0,
            f.padding.1,
            f.dilation.0,
            f.dilation.1,
        )
        .expect("constant fold");

        let mut native = Conv2dLayer::new_dilated(
            f.kernel.clone(),
            Some(f.bias.clone()),
            f.stride,
            f.padding,
            f.dilation,
            1,
        )
        .expect("native dilated layer");
        native.set_input_shape(f.in_h, f.in_w);
        let point_box = BoundedTensor::new(point.clone(), point).expect("degenerate point box");
        let out = native.propagate_ibp(&point_box).expect("point IBP");
        let (lo, hi) = out.lower_upper();

        assert_eq!(
            folded.shape(),
            lo.shape(),
            "fold must match NY output shape"
        );
        for (i, ((&v, &l), &u)) in folded.iter().zip(lo.iter()).zip(hi.iter()).enumerate() {
            assert!(
                (v - l).abs() < 1e-4 && (v - u).abs() < 1e-4,
                "index {i}: fold {v} must agree with NY's point bounds [{l}, {u}]"
            );
        }
    }

    /// Dilation 1 must still work and must not be perturbed by the change.
    #[test]
    fn constant_fold_conv2d_dilation_one_unchanged() {
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).expect("kernel");
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .expect("input");
        let out = constant_fold_conv2d(&input, &kernel, None, 1, 1, 0, 0, 1, 1).expect("fold");
        assert_eq!(out.shape(), &[1, 2, 2]);
        // out[0,0] = 1*1 + 2*2 + 4*3 + 5*4 = 1 + 4 + 12 + 20 = 37
        assert!(
            (out[[0, 0, 0]] - 37.0).abs() < 1e-5,
            "got {}",
            out[[0, 0, 0]]
        );
    }

    /// Dilation 2 on a hand-checkable fixture: taps land two apart.
    #[test]
    fn constant_fold_conv2d_dilation_two_is_hand_checkable() {
        // kernel [[1, 2], [3, 4]], dilation (2, 2) -> taps at (0,0), (0,2),
        // (2,0), (2,2) of a 3x3 input.
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).expect("kernel");
        let input = ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .expect("input");
        let out = constant_fold_conv2d(&input, &kernel, None, 1, 1, 0, 0, 2, 2).expect("fold");
        // Effective span 3x3 == input, so a single output element:
        // 1*1 + 3*2 + 7*3 + 9*4 = 1 + 6 + 21 + 36 = 64
        assert_eq!(out.shape(), &[1, 1, 1]);
        assert!(
            (out[[0, 0, 0]] - 64.0).abs() < 1e-5,
            "got {}",
            out[[0, 0, 0]]
        );
    }

    /// Zero dilation is malformed geometry — fail closed rather than divide the
    /// tap stride by zero downstream.
    #[test]
    fn zero_dilation_is_rejected() {
        let f = dilated_fixture();
        let all_nodes = vec![nn_dsl::tensor_ir::TensorNode::new(
            TensorNodeId::new(0),
            nn_dsl::tensor_ir::TensorOpKind::Reshape {
                input: TensorNodeId::new(0),
                target_shape: vec![2, f.in_h, f.in_w],
            },
            vec![2, f.in_h, f.in_w],
        )];
        let ctx = TensorTranslationContext {
            input_bindings: &[],
            input_node_names: &[],
            axis_offset: 0,
            all_nodes: &all_nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };
        let node_values = vec![
            TensorNodeValue::Variable(ny_propagate::NETWORK_INPUT.to_string()),
            TensorNodeValue::WeightTensor(f.kernel),
        ];
        let mut graph = GraphNetwork::new();
        let err = translate_conv2d(
            &ctx,
            TensorNodeId::new(2),
            &TensorNodeId::new(0),
            &TensorNodeId::new(1),
            &None,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            &node_values,
            &mut graph,
        );
        assert!(err.is_err(), "dilation 0 must be rejected");
    }
}
