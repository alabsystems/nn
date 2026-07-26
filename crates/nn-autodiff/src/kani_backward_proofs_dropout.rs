// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for dropout backward/forward scalar functions.
//! Extracted from `kani_backward_proofs.rs` for 500-line compliance.
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.

// ── Scalar dropout backward function ────────────────────────────────────

/// Dropout forward: output = x * mask * scale.
/// Dropout backward: grad_input = grad * mask * scale.
///
/// mask ∈ {0.0, 1.0}, scale = 1/(1-p) where p ∈ (0, 1).
///
/// SYNC: backward_rules.rs:62-65 (grad.mul(mask).mul_scalar(scale) pattern).
#[allow(dead_code)]
fn dropout_backward_scalar(grad: f32, mask: f32, scale: f32) -> f32 {
    grad * mask * scale
}

/// Dropout forward scalar: output = x * mask * scale.
#[allow(dead_code)]
fn dropout_forward_scalar(x: f32, mask: f32, scale: f32) -> f32 {
    x * mask * scale
}

#[cfg(kani)]
mod dropout_proofs {
    use super::*;

    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    /// Prove dropout backward preserves gradient direction when mask is 1 (kept).
    /// The gradient is scaled by `scale` but preserves sign.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_dropout_backward_preserves_sign() {
        let grad: f32 = kani::any();
        let scale: f32 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(scale, 1.0, 10.0);
        kani::assume(grad != 0.0);
        let result = dropout_backward_scalar(grad, 1.0, scale);
        assert!(!result.is_nan() && !result.is_infinite());
        // sign(result) == sign(grad) when scale > 0
        assert!(
            (result > 0.0) == (grad > 0.0),
            "dropout backward must preserve gradient sign"
        );
    }

    /// Prove dropout forward output is finite for finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_dropout_forward_finite() {
        let x: f32 = kani::any();
        let mask: f32 = kani::any();
        let scale: f32 = kani::any();
        assume_bounded(x, -1e4, 1e4);
        kani::assume(mask == 0.0 || mask == 1.0);
        assume_bounded(scale, 1.0, 10.0);
        let result = dropout_forward_scalar(x, mask, scale);
        assert!(!result.is_nan() && !result.is_infinite());
    }

    /// Prove dropout forward zeros dropped elements exactly.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_dropout_forward_zeros_dropped() {
        let x: f32 = kani::any();
        let scale: f32 = kani::any();
        assume_bounded(x, -1e4, 1e4);
        assume_bounded(scale, 1.0, 10.0);
        let result = dropout_forward_scalar(x, 0.0, scale);
        assert!(result == 0.0, "dropped elements must be exactly zero");
    }
}
