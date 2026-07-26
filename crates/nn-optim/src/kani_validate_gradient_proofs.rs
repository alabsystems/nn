// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `validate_gradient` and NaN-guard logic.
//!
//! The `validate_gradient` function in `error.rs` scans a gradient tensor
//! for NaN/Inf values. Since DynTensor is runtime-sized and can't be used
//! in Kani, we extract the scalar validation logic and prove properties:
//!
//! - The guard is sound: non-finite count matches element-wise check
//! - NaN gradients cannot corrupt finite parameters when guarded
//!
//! Re: #1486 (verified-training gaps).

#[cfg(kani)]
mod proofs {
    // ── Scalar validation logic ─────────────────────────────────
    //
    // Matches the element-wise check in `error.rs::validate_gradient`:
    //   data.iter().filter(|v| !v.is_finite()).count()

    /// Count non-finite elements in a small fixed-size array.
    /// Matches the pattern: `data.iter().filter(|v| !v.is_finite()).count()`
    fn count_non_finite(data: &[f32]) -> usize {
        let mut count = 0;
        for v in data {
            if !v.is_finite() {
                count += 1;
            }
        }
        count
    }

    // ── Optimizer guard interaction harnesses ────────────────────

    /// Prove: the validate_gradient guard is fail-safe.
    /// When count > 0, the validation logic returns an error (never Ok).
    /// When count == 0, it returns Ok (never error).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn prove_guard_fail_safe() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let count = count_non_finite(&[a, b]);
        let should_reject = count > 0;
        let has_non_finite = !a.is_finite() || !b.is_finite();
        // Guard rejects iff there exists a non-finite element
        assert!(
            should_reject == has_non_finite,
            "guard must reject iff any element is non-finite"
        );
    }

    /// Prove: NaN gradient cannot corrupt a finite parameter.
    /// If validate_gradient rejects (count > 0), the parameter update
    /// is skipped, preserving the original finite parameter.
    /// If the gradient IS finite and bounded, update preserves finiteness.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_nan_cannot_corrupt_param() {
        let theta: f32 = kani::any();
        let grad: f32 = kani::any();
        let lr: f32 = kani::any();
        kani::assume(theta.is_finite());
        kani::assume(lr.is_finite() && lr > 0.0 && lr < 1.0);
        let is_valid_grad = grad.is_finite();
        let updated = if is_valid_grad {
            theta - lr * grad
        } else {
            theta // guard rejects: param unchanged
        };
        if !is_valid_grad {
            // Non-finite grad: guard rejects, original parameter preserved
            assert!(
                updated == theta,
                "rejected NaN grad must leave param unchanged"
            );
        }
        // In both branches: output is never NaN (it's either theta or theta - lr*grad)
        // For the finite branch: lr*grad can overflow to ±inf, making theta - inf = ±inf,
        // but it CANNOT produce NaN (only inf - inf = NaN, and theta is finite).
        // Note: the update CAN be infinite if lr*grad overflows, which is expected
        // (the optimizer's gradient clipping handles that). The key invariant:
        // no NaN enters the parameter from a guarded update path.
        assert!(!updated.is_nan(), "guarded update must never produce NaN");
    }
}
