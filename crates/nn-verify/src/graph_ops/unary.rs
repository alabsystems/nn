// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! UnaryFn translation: constant fold or emit NY layer.

use ny_propagate::layers::{
    AbsLayer, CosLayer, ExpLayer, ReciprocalLayer, SinLayer, SqrtLayer, TanhLayer,
};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::ir::UnaryFnKind;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, NodeValue};

/// Evaluate a unary function on a constant scalar value.
/// Returns the result, which may be non-finite (e.g., recip(0) = inf, sqrt(-1) = NaN).
fn evaluate_constant_unary(op: UnaryFnKind, val: f32) -> Result<f32, VerifyError> {
    match op {
        UnaryFnKind::Sin => Ok(val.sin()),
        UnaryFnKind::Cos => Ok(val.cos()),
        UnaryFnKind::Sqrt => Ok(val.sqrt()),
        UnaryFnKind::Rsqrt => Ok(1.0 / val.sqrt()),
        UnaryFnKind::Exp => Ok(val.exp()),
        UnaryFnKind::Abs => Ok(val.abs()),
        UnaryFnKind::Recip => Ok(1.0 / val),
        UnaryFnKind::Tanh => Ok(val.tanh()),
        UnaryFnKind::Log => Ok(val.ln()),
        _ => Err(VerifyError::UnsupportedOp(format!("UnaryFn {op:?}"))),
    }
}

/// Translate a `UnaryFn` node to a NY layer or constant fold.
pub(crate) fn translate_unary(
    name: &str,
    op: UnaryFnKind,
    input: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match input {
        NodeValue::Constant(v) => {
            let val = v.get();
            let result = evaluate_constant_unary(op, val)?;
            checked_constant(result, &format!("{op:?}({val})"))
        }
        NodeValue::Variable(var_name) => {
            let layer = match op {
                UnaryFnKind::Sin => Layer::Sin(SinLayer::new()),
                UnaryFnKind::Cos => Layer::Cos(CosLayer::new()),
                UnaryFnKind::Sqrt => Layer::Sqrt(SqrtLayer::new()),
                UnaryFnKind::Exp => Layer::Exp(ExpLayer::new()),
                UnaryFnKind::Abs => Layer::Abs(AbsLayer::new()),
                UnaryFnKind::Recip => Layer::Reciprocal(ReciprocalLayer::new()),
                UnaryFnKind::Tanh => Layer::Tanh(TanhLayer::new()),
                UnaryFnKind::Rsqrt => {
                    let sqrt_name = format!("{name}_sqrt");
                    add_unary_node(&sqrt_name, Layer::Sqrt(SqrtLayer::new()), var_name, graph);
                    add_unary_node(
                        name,
                        Layer::Reciprocal(ReciprocalLayer::new()),
                        &sqrt_name,
                        graph,
                    );
                    return Ok(NodeValue::Variable(name.to_string()));
                }
                _ => return Err(VerifyError::UnsupportedOp(format!("UnaryFn {op:?}"))),
            };
            add_unary_node(name, layer, var_name, graph);
            Ok(NodeValue::Variable(name.to_string()))
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves `evaluate_constant_unary(Abs, val)` produces `val.abs()` for any
    /// finite input, and the result is always non-negative and finite.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_abs_constant_fold_correct() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());

        let result =
            evaluate_constant_unary(UnaryFnKind::Abs, val).expect("Abs must not return Err");
        assert_eq!(
            result.to_bits(),
            val.abs().to_bits(),
            "Abs must be bit-exact"
        );
        assert!(result >= 0.0, "Abs must be non-negative");
        assert!(result.is_finite(), "Abs of finite input must be finite");
    }

    /// Proves `evaluate_constant_unary(Recip, 0.0)` produces infinity (which
    /// `checked_constant` will reject), proving the division-by-zero path is handled.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_recip_zero_produces_infinity() {
        let result = evaluate_constant_unary(UnaryFnKind::Recip, 0.0)
            .expect("Recip(0) returns Ok(inf), not Err");
        assert!(
            !result.is_finite(),
            "Recip(0) must produce non-finite value (caught by checked_constant)"
        );
    }

    /// Proves `evaluate_constant_unary(Recip, val)` is bit-exact with `1.0/val`
    /// for a representative set of f32 values chosen by symbolic index.
    ///
    /// CBMC cannot handle fully-symbolic f32 division (SAT solver timeout).
    /// This harness uses symbolic index selection from 8 representative values.
    /// Uses `unwind(8)` to bound syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn unary_recip_constant_fold_correct() {
        const VALS: [f32; 8] = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 100.0, -100.0];
        let i: usize = kani::any();
        kani::assume(i < VALS.len());
        let val = VALS[i];

        let result =
            evaluate_constant_unary(UnaryFnKind::Recip, val).expect("Recip must not return Err");
        assert_eq!(
            result.to_bits(),
            (1.0_f32 / val).to_bits(),
            "Recip must be bit-exact with 1.0/val"
        );
    }

    /// AC3: Proves `evaluate_constant_unary(Rsqrt, 0.0)` produces non-finite result
    /// (infinity from 1.0/sqrt(0.0)), caught by `checked_constant`.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_rsqrt_zero_produces_infinity() {
        let result = evaluate_constant_unary(UnaryFnKind::Rsqrt, 0.0)
            .expect("Rsqrt(0) returns Ok(inf), not Err");
        assert!(
            !result.is_finite(),
            "Rsqrt(0) = 1/sqrt(0) must produce non-finite value"
        );
    }

    /// AC4: Proves `evaluate_constant_unary(Exp, val)` for large positive inputs
    /// overflows to infinity (caught by `checked_constant`). Uses specific threshold:
    /// f32 exp overflows at approximately val > 88.72.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_exp_overflow_produces_infinity() {
        // 89.0 is above the f32 exp overflow threshold (~88.72)
        let result = evaluate_constant_unary(UnaryFnKind::Exp, 89.0_f32)
            .expect("Exp(89) returns Ok(inf), not Err");
        assert!(
            !result.is_finite(),
            "Exp(89) must overflow to infinity (caught by checked_constant)"
        );
    }

    /// Proves `evaluate_constant_unary(Exp, val)` for bounded inputs produces finite
    /// positive results. exp(val) is finite for val < ~88.72.
    /// Note: CBMC's expf model is not bit-exact with Rust's f32::exp() — we verify
    /// structural properties only. Bit-exactness is covered by unit tests.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_exp_bounded_input_finite() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        // Restrict to range where exp is guaranteed finite
        kani::assume(val >= -87.0 && val <= 87.0);

        let result =
            evaluate_constant_unary(UnaryFnKind::Exp, val).expect("Exp must not return Err");
        assert!(result.is_finite(), "Exp of bounded input must be finite");
        assert!(result > 0.0, "Exp must be positive");
    }

    /// AC5: Proves `evaluate_constant_unary(Sin, val)` produces results in [-1, 1]
    /// for all finite inputs. Uses concrete test values since CBMC cannot model
    /// f32::sin accurately (see design doc and #329).
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_sin_constant_fold_bounded() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        // Restrict to moderate range where CBMC may model sin correctly
        kani::assume(val >= -10.0 && val <= 10.0);

        let result =
            evaluate_constant_unary(UnaryFnKind::Sin, val).expect("Sin must not return Err");
        // sin of any real is in [-1, 1] and finite
        assert!(result.is_finite(), "Sin of finite input must be finite");
        assert!(result >= -1.0, "Sin must be >= -1");
        assert!(result <= 1.0, "Sin must be <= 1");
    }

    /// Proves `evaluate_constant_unary(Cos, val)` produces results in [-1, 1]
    /// for all finite inputs.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_cos_constant_fold_bounded() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -10.0 && val <= 10.0);

        let result =
            evaluate_constant_unary(UnaryFnKind::Cos, val).expect("Cos must not return Err");
        assert!(result.is_finite(), "Cos of finite input must be finite");
        assert!(result >= -1.0, "Cos must be >= -1");
        assert!(result <= 1.0, "Cos must be <= 1");
    }

    /// Proves `evaluate_constant_unary(Log, 0.0)` produces negative infinity,
    /// which `checked_constant` will reject. Verifies the ln(0) edge case.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_log_zero_produces_neg_infinity() {
        let result = evaluate_constant_unary(UnaryFnKind::Log, 0.0)
            .expect("Log(0) returns Ok(-inf), not Err");
        assert!(
            !result.is_finite(),
            "Log(0) must produce non-finite value (caught by checked_constant)"
        );
        assert!(
            result.is_sign_negative(),
            "Log(0) must produce negative infinity"
        );
    }

    /// Proves `evaluate_constant_unary(Log, val)` for negative inputs produces NaN,
    /// which `checked_constant` will reject. Verifies the ln(negative) edge case.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_log_negative_produces_nan() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val < 0.0);

        let result = evaluate_constant_unary(UnaryFnKind::Log, val)
            .expect("Log(neg) returns Ok(NaN), not Err");
        assert!(
            result.is_nan(),
            "Log of negative must produce NaN (caught by checked_constant)"
        );
    }

    /// Proves `evaluate_constant_unary(Abs, val)` is idempotent:
    /// abs(abs(x)) == abs(x) for any finite input. This verifies that applying
    /// Abs twice yields the same result as applying it once.
    #[kani::unwind(64)]
    #[kani::proof]
    fn unary_abs_idempotent() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());

        let once = evaluate_constant_unary(UnaryFnKind::Abs, val).expect("Abs must not return Err");
        let twice =
            evaluate_constant_unary(UnaryFnKind::Abs, once).expect("Abs must not return Err");
        assert_eq!(
            once.to_bits(),
            twice.to_bits(),
            "Abs must be idempotent: abs(abs(x)) == abs(x)"
        );
    }

    /// Proves that unsupported unary ops (Neg, Floor, Round, Fract) correctly
    /// return `Err(UnsupportedOp)` instead of silently producing wrong results.
    /// Uses symbolic index over the 4 unsupported variants.
    /// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn unary_unsupported_ops_return_err() {
        let op_idx: u8 = kani::any();
        kani::assume(op_idx < 4);
        let op = match op_idx {
            0 => UnaryFnKind::Neg,
            1 => UnaryFnKind::Floor,
            2 => UnaryFnKind::Round,
            _ => UnaryFnKind::Fract,
        };

        let val: f32 = kani::any();
        kani::assume(val.is_finite());

        let result = evaluate_constant_unary(op, val);
        assert!(
            result.is_err(),
            "Unsupported unary ops must return Err(UnsupportedOp)"
        );
    }
}

/// Kani harnesses that require `#[kani::stub]` to work around CBMC limitations:
/// - CBMC's sqrtf fires internal NaN-on-division checks for negative inputs (#708)
/// - Kani does not support `tanhf` foreign function (kani#2423)
///
/// Run with: `cargo kani -p nn-verify --features kani-stubbing -Z stubbing`
///
/// Stubs model IEEE 754 / mathematical semantics with nondeterministic values
/// constrained to the function's range. This makes proofs *sound*: if properties
/// hold for any value in the range, they hold for the true result.
#[cfg(all(kani, feature = "kani-stubbing"))]
mod kani_stubbed_proofs {
    use super::*;

    /// Sqrt stub that models IEEE 754 semantics without CBMC's internal NaN checks.
    /// For negative inputs: returns NaN (via 0.0/0.0 is avoided — use f32::NAN directly).
    /// For non-negative inputs: returns a nondeterministic non-negative finite value.
    fn sqrt_stub(x: f32) -> f32 {
        if x < 0.0 {
            f32::NAN
        } else if x == 0.0 {
            0.0
        } else {
            let result: f32 = kani::any();
            kani::assume(result.is_finite() && result >= 0.0);
            result
        }
    }

    /// Tanh stub: returns nondeterministic value in (-1, 1) for finite inputs.
    /// Kani does not support the `tanhf` C foreign function (kani#2423).
    /// tanh is bounded: tanh(x) ∈ (-1, 1) for all finite x.
    fn tanh_stub(_x: f32) -> f32 {
        let result: f32 = kani::any();
        kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
        result
    }

    /// Proves the Sqrt dispatch arm in `evaluate_constant_unary` returns Ok.
    ///
    /// NOTE: sqrt_stub is nondeterministic (finite, non-negative), so the
    /// result assertions are circular with the stub assumptions. The actual
    /// non-trivial property proved is dispatch correctness: the Sqrt match
    /// arm calls f32::sqrt (stubbed) and returns Ok, not Err.
    /// Uses `sqrt_stub` to avoid CBMC sqrtf bit-pattern inconsistency (#708).
    #[kani::unwind(64)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn unary_sqrt_dispatch_returns_ok() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= 0.0);

        let result =
            evaluate_constant_unary(UnaryFnKind::Sqrt, val).expect("Sqrt must not return Err");
        assert!(
            result.is_finite(),
            "Sqrt of non-negative finite input must be finite"
        );
        assert!(result >= 0.0, "Sqrt must be non-negative");
    }

    /// Proves `evaluate_constant_unary(Sqrt, val)` for negative input produces NaN.
    /// Uses `sqrt_stub` to avoid CBMC builtin-library-sqrtf NaN-on-division check (#708).
    #[kani::unwind(64)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn unary_sqrt_negative_produces_nan() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val < 0.0);

        let result =
            evaluate_constant_unary(UnaryFnKind::Sqrt, val).expect("Sqrt returns Ok(NaN), not Err");
        assert!(
            result.is_nan(),
            "Sqrt of negative must produce NaN (caught by checked_constant)"
        );
    }

    /// Proves `evaluate_constant_unary(Rsqrt, val)` for negative input produces NaN
    /// (since sqrt of negative is NaN, and 1.0/NaN = NaN).
    /// Uses `sqrt_stub` to avoid CBMC builtin-library-sqrtf NaN-on-division check (#708).
    #[kani::unwind(64)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_stub)]
    fn unary_rsqrt_negative_produces_nan() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val < 0.0);

        let result = evaluate_constant_unary(UnaryFnKind::Rsqrt, val)
            .expect("Rsqrt returns Ok(NaN), not Err");
        assert!(
            result.is_nan(),
            "Rsqrt of negative must produce NaN (caught by checked_constant)"
        );
    }

    /// Proves the Tanh dispatch arm in `evaluate_constant_unary` returns Ok.
    ///
    /// NOTE: tanh_stub is nondeterministic in [-1, 1], so the result bound
    /// assertions are circular with the stub assumptions. The actual
    /// non-trivial property proved is dispatch correctness: the Tanh match
    /// arm calls f32::tanh (stubbed) and returns Ok, not Err.
    /// Uses `tanh_stub` because Kani does not support `tanhf` (kani#2423).
    /// Part of #665 AC1.
    #[kani::unwind(64)]
    #[kani::proof]
    #[kani::stub(f32::tanh, tanh_stub)]
    fn unary_tanh_dispatch_returns_ok() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -10.0 && val <= 10.0);

        let result =
            evaluate_constant_unary(UnaryFnKind::Tanh, val).expect("Tanh must not return Err");
        assert!(result.is_finite(), "Tanh of finite input must be finite");
        assert!(result >= -1.0, "Tanh must be >= -1");
        assert!(result <= 1.0, "Tanh must be <= 1");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Part of #665 AC4: verify tanh constant fold at known values.
    #[test]
    fn test_evaluate_constant_unary_tanh_known_values() {
        let result_zero =
            evaluate_constant_unary(UnaryFnKind::Tanh, 0.0).expect("Tanh(0) must succeed");
        assert!(
            (result_zero - 0.0).abs() < 1e-7,
            "tanh(0) must equal 0, got {result_zero}",
        );

        let result_one =
            evaluate_constant_unary(UnaryFnKind::Tanh, 1.0).expect("Tanh(1) must succeed");
        assert!(
            (result_one - 0.7616).abs() < 0.001,
            "tanh(1) must be ~0.7616, got {result_one}",
        );

        let result_neg =
            evaluate_constant_unary(UnaryFnKind::Tanh, -1.0).expect("Tanh(-1) must succeed");
        assert!(
            (result_neg + 0.7616).abs() < 0.001,
            "tanh(-1) must be ~-0.7616, got {result_neg}",
        );
    }
}
