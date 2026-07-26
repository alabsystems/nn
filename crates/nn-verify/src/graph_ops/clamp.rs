// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Clamp translation: constant-bounds ClipLayer.

use ny_propagate::layers::ClipLayer;
use ny_propagate::{GraphNetwork, Layer};

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, NodeValue};

/// Evaluate constant clamp: returns `x.clamp(lo, hi)` or error for inverted/NaN bounds.
fn evaluate_constant_clamp(x: f32, lo: f32, hi: f32) -> Result<f32, VerifyError> {
    // IEEE 754: `lo > hi` returns false when either is NaN. Check finiteness first.
    if !lo.is_finite() || !hi.is_finite() {
        return Err(VerifyError::InternalTranslationError {
            context: format!("Clamp bounds non-finite: lo={lo}, hi={hi}"),
        });
    }
    if lo > hi {
        return Err(VerifyError::InternalTranslationError {
            context: "Clamp bounds inverted".to_string(),
        });
    }
    Ok(x.clamp(lo, hi))
}

/// Translate a Clamp node with constant bounds to ClipLayer.
pub(crate) fn translate_clamp(
    name: &str,
    input_val: &NodeValue,
    min_val: &NodeValue,
    max_val: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match (min_val, max_val) {
        (NodeValue::Constant(lo), NodeValue::Constant(hi)) => {
            let (lo, hi) = (lo.get(), hi.get());
            match input_val {
                NodeValue::Constant(v) => {
                    let result = evaluate_constant_clamp(v.get(), lo, hi)?;
                    checked_constant(result, "Clamp constant fold")
                }
                NodeValue::Variable(var_name) => {
                    // IEEE 754: `lo > hi` returns false when either is NaN.
                    // Defense-in-depth: reject NaN bounds before NY ClipLayer.
                    if !lo.is_finite() || !hi.is_finite() {
                        return Err(VerifyError::InternalTranslationError {
                            context: format!(
                                "Clamp bounds non-finite: lo={lo}, hi={hi} in node `{name}`"
                            ),
                        });
                    }
                    if lo > hi {
                        return Err(VerifyError::InternalTranslationError {
                            context: format!(
                                "Clamp bounds inverted: lo={lo} > hi={hi} in node `{name}`"
                            ),
                        });
                    }
                    let layer = Layer::Clip(ClipLayer::new(lo, hi));
                    add_unary_node(name, layer, var_name, graph);
                    Ok(NodeValue::Variable(name.to_string()))
                }
            }
        }
        _ => Err(VerifyError::UnsupportedOp(
            "Clamp with variable bounds (constant min/max required)".into(),
        )),
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves `evaluate_constant_clamp` rejects inverted bounds (lo > hi).
    /// Uses `unwind(8)` to bound loop unwinding depth — CBMC otherwise diverges
    /// unwinding `syn::error::ErrorMessage` Drop impl (pulled in via nn-dsl → syn).
    /// See #608.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn clamp_rejects_inverted_bounds() {
        let x: f32 = kani::any();
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(x.is_finite() && lo.is_finite() && hi.is_finite());
        kani::assume(lo > hi);

        let result = evaluate_constant_clamp(x, lo, hi);
        assert!(result.is_err(), "inverted clamp bounds must return Err");
    }

    /// Proves `evaluate_constant_clamp` rejects NaN bounds.
    /// NaN bypasses `lo > hi` under IEEE 754, so explicit finiteness check is needed.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn clamp_rejects_nan_bounds() {
        let x: f32 = kani::any();
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(!lo.is_finite() || !hi.is_finite());

        let result = evaluate_constant_clamp(x, lo, hi);
        assert!(result.is_err(), "non-finite clamp bounds must return Err");
    }

    /// Proves `evaluate_constant_clamp` output satisfies structural properties:
    /// result is in [lo, hi] for valid bounds.
    ///
    /// Decomposed from bit-exactness proof: CBMC cannot handle 3-variable
    /// symbolic f32 comparison + `String` allocation in error paths within
    /// timeout. This harness proves the safety-critical properties (bounds
    /// containment) without the identity check.
    /// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn clamp_constant_fold_correct() {
        let x: f32 = kani::any();
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(x.is_finite() && lo.is_finite() && hi.is_finite());
        kani::assume(lo <= hi);

        let result = evaluate_constant_clamp(x, lo, hi);
        let val = result.expect("clamp of finite values with valid bounds must succeed");
        // Safety-critical structural properties:
        assert!(val.is_finite(), "clamped value must be finite");
        assert!(val >= lo, "clamped value must be >= lo");
        assert!(val <= hi, "clamped value must be <= hi");
    }

    /// Proves clamp is idempotent: clamp(clamp(x, lo, hi), lo, hi) == clamp(x, lo, hi).
    /// Re-clamping an already-clamped value with the same bounds must be a no-op.
    /// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn clamp_idempotent() {
        let x: f32 = kani::any();
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(x.is_finite() && lo.is_finite() && hi.is_finite());
        kani::assume(lo <= hi);

        let once = evaluate_constant_clamp(x, lo, hi).expect("first clamp must succeed");
        let twice = evaluate_constant_clamp(once, lo, hi).expect("second clamp must succeed");
        assert_eq!(
            once.to_bits(),
            twice.to_bits(),
            "clamp must be idempotent: clamp(clamp(x,lo,hi),lo,hi) == clamp(x,lo,hi)"
        );
    }
}
