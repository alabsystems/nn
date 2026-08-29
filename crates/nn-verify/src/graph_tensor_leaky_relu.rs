// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LeakyReLU tensor op translation via decomposition: α*x + (1-α)*ReLU(x).
//!
//! Used by Kokoro decoder (ISTFTNet vocoder): `LeakyReLU(0.1)` per upsample
//! stage, `LeakyReLU(0.01)` before `conv_post`.
//!
//! ## Why the decomposition survives — the old reason is stale, the new one is measured
//!
//! The historical justification was that NY's `LeakyReLULayer` "returns IBP-wide
//! CROWN bounds (#2977)". **That is no longer true.**
//! `ny_propagate::layers::activations::leaky_relu::leaky_relu_linear_relaxation`
//! is a complete chord/tangent relaxation today: exact identity/α-scaling on the
//! non-crossing sides, an f64-computed chord with directed rounding on the
//! crossing side, and dedicated arms for ±inf endpoints and α > 1. So the
//! premise this module was written against has been fixed upstream.
//!
//! The decomposition is nevertheless RETAINED, on measured evidence rather than
//! on the retired bug reference. On a `Linear -> act -> Linear -> act -> Linear`
//! fixture (`tests::decomposition_crown_is_tighter_than_native_leaky_relu`) the
//! ReLU decomposition's CROWN bounds are **1.3x–1.7x tighter** than the native
//! layer's across every crossing regime (`u > |l|`, `u == |l|`, `u < |l|`,
//! `u << |l|`) at both α = 0.01 and α = 0.1 — e.g. α = 0.1 on `[-1, 2]`: total
//! output width 86.1 decomposed vs 147.8 native. IBP is a tie to ~1e-7 (the
//! native form is a few ULPs narrower, having three fewer rounding steps), so
//! the whole difference is in the CROWN relaxation.
//!
//! α-CROWN returned the same numbers as plain CROWN for BOTH forms on that
//! fixture, so the measured gap is entirely in the fixed relaxations, and the
//! mechanism is visible in NY's source:
//!
//!   * The two relaxations agree exactly on the upper chord (algebraically
//!     identical: slope `(u - αl)/(u - l)`, intercept `-(1-α)lu/(u-l)` — the
//!     decomposition's `α*x + (1-α)*ReLU_chord` expands to precisely that), but
//!     differ on the lower tangent. `relu_crown_relaxation` switches to slope 1
//!     at `u > |l|`, which is the gap-area-minimizing threshold;
//!     `leaky_relu_linear_relaxation` switches at `u > α|l|`, so throughout
//!     `α|l| < u <= |l|` it takes the wider of the two sound tangents. At
//!     α = 0.01 that window is almost the whole crossing regime.
//!   * `ReLU` is also dispatched OUT of the generic unary CROWN backward
//!     (`dispatch_backward_layer` excludes it: "alpha/beta/gradient logic varies
//!     per site") into the per-site ReLU path, and NY's α-CROWN entry points
//!     gate on `Layer::ReLU` (`alpha_crown.rs` counts
//!     `matches!(l, Layer::ReLU(_))`; `graph_alpha::resnet_decompose` matches
//!     the same). A fused `Layer::LeakyReLU` node is outside that machinery, so
//!     it also forfeits any future α/β refinement.
//!
//! If NY later aligns the LeakyReLU lower-slope threshold and admits the layer
//! to the α-CROWN neuron set, re-run that test: the moment native stops being
//! looser, this decomposition is dead weight and should be replaced by
//! `Layer::LeakyReLU`.
//!
//! Part of #1741 — extending moonshot proofs to real Kokoro model graph.

use nn_dsl::tensor_ir::TensorNodeId;
use ny_propagate::layers::{AddLayer, MulConstantLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::LeakyRelu` node to a NY `LeakyReLULayer`.
///
/// Constant-folds when the input is a known constant:
/// `leaky_relu(c, alpha) = c if c >= 0, else alpha * c`.
/// For variable inputs, emits a `LeakyReLULayer` node in the graph.
pub(crate) fn translate_leaky_relu(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    negative_slope: f32,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    if !negative_slope.is_finite() {
        return Err(VerifyError::UnsupportedOp(format!(
            "LeakyRelu negative_slope is non-finite ({negative_slope})"
        )));
    }
    let input_val = get_value(node_values, input.index(), "LeakyRelu input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = if c.get() >= 0.0 {
                c.get()
            } else {
                negative_slope * c.get()
            };
            checked_tensor_constant(
                val,
                &format!("leaky_relu constant fold t{}", node_id.index()),
            )
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            // Decompose LeakyReLU(x, alpha) = alpha*x + (1-alpha)*ReLU(x).
            // NOT because NY's LeakyReLULayer is IBP-wide under CROWN — that
            // #2977 limitation is fixed upstream — but because the ReLU form's
            // CROWN is MEASURABLY 1.3-1.7x tighter (see the module docs and
            // `tests::decomposition_crown_is_tighter_than_native_leaky_relu`).
            let scale_name = format!("{node_name}_alpha_x");
            add_unary_node(
                &scale_name,
                Layer::MulConstant(MulConstantLayer::scalar(negative_slope)),
                input_name,
                graph,
            );
            let relu_name = format!("{node_name}_relu");
            add_unary_node(&relu_name, Layer::ReLU(ReLULayer::new()), input_name, graph);
            let relu_scaled_name = format!("{node_name}_relu_scaled");
            add_unary_node(
                &relu_scaled_name,
                Layer::MulConstant(MulConstantLayer::scalar(1.0 - negative_slope)),
                &relu_name,
                graph,
            );
            graph.add_node(GraphNode::new(
                node_name.clone(),
                Layer::Add(AddLayer),
                vec![scale_name, relu_scaled_name],
            ));
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "LeakyRelu on WeightTensor not supported".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FiniteF32;
    use crate::BoundedTensor;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use ny_propagate::layers::{LeakyReLULayer, LinearLayer};
    use ny_propagate::NETWORK_INPUT;

    /// Deterministic mixed-sign weight matrix — a stand-in for a real
    /// ISTFTNet-style conv/linear stage, chosen so BOTH the lower and the upper
    /// CROWN relaxation of every activation reaches the output.
    fn mixed_weights(rows: usize, cols: usize, salt: usize) -> Array2<f32> {
        let mut w = Array2::<f32>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let k = (i * 31 + j * 17 + salt * 7) % 23;
                w[[i, j]] = (k as f32) / 11.0 - 1.0;
            }
        }
        w
    }

    fn push_linear(
        graph: &mut GraphNetwork,
        name: &str,
        input: &str,
        rows: usize,
        cols: usize,
        salt: usize,
    ) {
        let lin = LinearLayer::new(mixed_weights(rows, cols, salt), Some(Array1::zeros(rows)))
            .expect("linear stage");
        graph.add_node(GraphNode::new(
            name.to_string(),
            Layer::Linear(lin),
            vec![input.to_string()],
        ));
    }

    /// `input -> Linear -> act -> Linear -> act -> Linear`, so the activation's
    /// input bounds come from a preceding affine stage (where a CROWN
    /// relaxation gap is actually visible) and two relaxation gaps compound.
    /// `native == false` emits this module's 4-node decomposition; `true` emits
    /// NY's single `Layer::LeakyReLU`.
    fn stack(alpha: f32, n: usize, native: bool) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        push_linear(&mut graph, "l0", NETWORK_INPUT, n, n, 1);
        let mut prev = "l0".to_string();
        for stage in 0..2 {
            let act = format!("a{stage}");
            if native {
                graph.add_node(GraphNode::new(
                    act.clone(),
                    Layer::LeakyReLU(LeakyReLULayer::new(alpha)),
                    vec![prev.clone()],
                ));
            } else {
                let node_values = vec![
                    TensorNodeValue::Variable(prev.clone()),
                    TensorNodeValue::Variable(prev.clone()),
                ];
                // `translate_leaky_relu` names its output `t{id}`; give each
                // stage a distinct id so the names do not collide.
                let out = translate_leaky_relu(
                    TensorNodeId::new(100 + stage),
                    &TensorNodeId::new(0),
                    alpha,
                    &node_values,
                    &mut graph,
                )
                .expect("decomposed translation");
                let TensorNodeValue::Variable(name) = out else {
                    panic!("expected a Variable output");
                };
                assert_eq!(name, format!("t{}", 100 + stage));
                graph.add_node(GraphNode::new(
                    act.clone(),
                    Layer::MulConstant(MulConstantLayer::scalar(1.0)),
                    vec![name],
                ));
            }
            let lin = format!("l{}", stage + 1);
            push_linear(&mut graph, &lin, &act, n, n, stage + 2);
            prev = lin;
        }
        graph.set_output(prev);
        graph
    }

    fn box_of(lo: f32, hi: f32, n: usize) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[n]), lo),
            ArrayD::from_elem(IxDyn(&[n]), hi),
        )
        .expect("valid box")
    }

    fn total_width(t: &BoundedTensor) -> f64 {
        let (lo, hi) = t.lower_upper();
        lo.iter()
            .zip(hi.iter())
            .map(|(&l, &u)| f64::from(u) - f64::from(l))
            .sum()
    }

    /// The decomposition and NY's native `Layer::LeakyReLU` are the SAME
    /// FUNCTION — the precondition for even comparing their bounds. Checked by
    /// running both graphs on degenerate (point) boxes across the sign change.
    #[test]
    fn decomposition_and_native_leaky_relu_are_the_same_function() {
        let n = 4;
        for alpha in [0.01f32, 0.1, 0.5] {
            let dec = stack(alpha, n, false);
            let nat = stack(alpha, n, true);
            for x in [-3.0f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0] {
                let point = box_of(x, x, n);
                let d = dec.propagate_ibp(&point).expect("decomposed point IBP");
                let m = nat.propagate_ibp(&point).expect("native point IBP");
                let (dlo, dhi) = d.lower_upper();
                let (mlo, mhi) = m.lower_upper();
                for (i, (((&a, &b), &c), &e)) in dlo
                    .iter()
                    .zip(dhi.iter())
                    .zip(mlo.iter())
                    .zip(mhi.iter())
                    .enumerate()
                {
                    let dmid = f64::midpoint(f64::from(a), f64::from(b));
                    let nmid = f64::midpoint(f64::from(c), f64::from(e));
                    assert!(
                        (dmid - nmid).abs() < 1e-3,
                        "alpha={alpha} x={x} index {i}: decomposed {dmid} != native {nmid}"
                    );
                }
            }
        }
    }

    /// BEFORE/AFTER measurement (task 3). The `#2977` premise this module was
    /// written against — "NY's `LeakyReLULayer` returns IBP-wide CROWN bounds" —
    /// is STALE: `leaky_relu_linear_relaxation` is a full chord/tangent
    /// relaxation today. So the decomposition has to justify itself by
    /// measurement instead, and it does: its CROWN bounds are strictly tighter
    /// than the native layer's in every crossing regime, while IBP ties.
    ///
    /// This is the test to re-run if NY aligns the LeakyReLU lower-slope
    /// threshold with `relu_crown_relaxation`'s or admits the layer to the
    /// α-CROWN neuron set — when it stops failing to beat native, the
    /// decomposition should be deleted in favour of `Layer::LeakyReLU`.
    #[test]
    fn decomposition_crown_is_tighter_than_native_leaky_relu() {
        let n = 6;
        let mut worst_ratio = f64::INFINITY;
        for alpha in [0.01f32, 0.1] {
            for (lo, hi, label) in [
                (-1.0f32, 2.0f32, "u > |l|"),
                (-1.0, 1.0, "u == |l|"),
                (-2.0, 1.0, "u < |l|"),
                (-4.0, 0.25, "u << |l|"),
            ] {
                let input = box_of(lo, hi, n);
                let dec = stack(alpha, n, false);
                let nat = stack(alpha, n, true);

                let dec_ibp = total_width(&dec.propagate_ibp(&input).expect("decomposed IBP"));
                let nat_ibp = total_width(&nat.propagate_ibp(&input).expect("native IBP"));
                let dec_crown =
                    total_width(&dec.propagate_crown(&input).expect("decomposed CROWN"));
                let nat_crown = total_width(&nat.propagate_crown(&input).expect("native CROWN"));
                let dec_alpha = total_width(
                    &dec.propagate_alpha_crown(&input)
                        .expect("decomposed a-CROWN"),
                );
                let nat_alpha =
                    total_width(&nat.propagate_alpha_crown(&input).expect("native a-CROWN"));

                println!(
                    "alpha={alpha} box=[{lo},{hi}] ({label}) IBP dec={dec_ibp:.6} nat={nat_ibp:.6} \
                     | CROWN dec={dec_crown:.6} nat={nat_crown:.6} \
                     | a-CROWN dec={dec_alpha:.6} nat={nat_alpha:.6}"
                );

                // IBP is a tie (the native form is a few ULPs narrower — three
                // fewer rounding steps), so a small relative tolerance.
                assert!(
                    (dec_ibp - nat_ibp).abs() <= 1e-4 * dec_ibp.max(1.0),
                    "alpha={alpha} {label}: IBP should tie, got dec={dec_ibp} nat={nat_ibp}"
                );

                // CROWN: the decomposition must be STRICTLY tighter — that is
                // the whole reason it is still here.
                assert!(
                    dec_crown < nat_crown,
                    "alpha={alpha} {label}: decomposed CROWN {dec_crown} must beat native \
                     {nat_crown}; if this now ties, replace the decomposition with \
                     Layer::LeakyReLU (see the module docs)"
                );
                assert!(
                    dec_alpha <= nat_alpha + 1e-4,
                    "alpha={alpha} {label}: decomposed a-CROWN {dec_alpha} must not be looser \
                     than native {nat_alpha}"
                );
                worst_ratio = worst_ratio.min(nat_crown / dec_crown);
            }
        }
        assert!(
            worst_ratio > 1.25,
            "the decomposition's CROWN advantage collapsed to {worst_ratio}x — re-evaluate \
             whether Layer::LeakyReLU should now be emitted directly"
        );
    }

    #[test]
    fn test_translate_leaky_relu_variable() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.01,
            &node_values,
            &mut graph,
        );
        let val = result.expect("variable path should succeed");
        let TensorNodeValue::Variable(name) = val else {
            panic!("expected Variable, got {val:?}");
        };
        assert_eq!(name, "t1");
    }

    #[test]
    fn test_translate_leaky_relu_constant_positive() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(5.0).unwrap())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.01,
            &node_values,
            &mut graph,
        );
        let val = result.expect("positive constant should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!((c.get() - 5.0).abs() < 1e-6, "leaky_relu(5) = 5");
    }

    #[test]
    fn test_translate_leaky_relu_constant_negative() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(-5.0).unwrap())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.1,
            &node_values,
            &mut graph,
        );
        let val = result.expect("negative constant should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!(
            (c.get() - (-0.5)).abs() < 1e-6,
            "leaky_relu(-5, 0.1) = -0.5"
        );
    }

    #[test]
    fn test_translate_leaky_relu_nan_slope_rejected() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            f32::NAN,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "NaN slope must be rejected");
    }

    #[test]
    fn test_translate_leaky_relu_inf_slope_rejected() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            f32::INFINITY,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "Inf slope must be rejected");
    }

    #[test]
    fn test_translate_leaky_relu_weight_tensor_unsupported() {
        let mut graph = GraphNetwork::new();
        let weight = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).unwrap();
        let node_values = vec![TensorNodeValue::WeightTensor(weight)];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.01,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "WeightTensor should be unsupported");
    }
}
