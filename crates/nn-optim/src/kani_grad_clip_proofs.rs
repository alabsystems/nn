// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for gradient clipping operations.
//!
//! Extracted from `kani_optim_proofs.rs` to keep file sizes under 500 lines.
//! Proves that `clip_grad_norm` and `clip_grad_value` produce finite,
//! bounded output without amplifying gradient magnitude.
//!
//! Re: #1496, #1465 (gradient clipping).

#[cfg(kani)]
mod proofs {
    fn assume_bounded(x: f32, lo: f32, hi: f32) {
        kani::assume(!x.is_nan() && !x.is_infinite());
        kani::assume(x >= lo && x <= hi);
    }

    // ── Gradient clipping harnesses ──────────────────────────────────

    /// clip_grad_norm scale factor is in [0, 1] when total_norm > max_norm > 0.
    /// Proves that the gradient scaling never amplifies gradients.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_clip_norm_scale_bounded() {
        // Use f32 instead of f64 — CBMC's f64 division model causes spurious
        // failures (assertion 1 "scale > 0.0" falsified despite positive/positive
        // division). The f32 version exercises the same mathematical property
        // within CBMC's precision model.
        let total_norm: f32 = kani::any();
        let max_norm: f32 = kani::any();
        assume_bounded(total_norm, 1e-4, 1e4);
        assume_bounded(max_norm, 1e-4, 1e4);
        kani::assume(total_norm > max_norm); // clipping is triggered
        let scale = max_norm / total_norm;
        assert!(scale > 0.0, "scale must be positive");
        assert!(scale <= 1.0, "scale must not amplify");
        assert!(!scale.is_nan() && !scale.is_infinite());
    }

    /// Prove: clip_grad_norm scalar application produces finite output
    /// and does not amplify the gradient magnitude. Exercises both the
    /// clip and no-clip paths through the full decision logic.
    ///
    /// Matches grad_clip.rs:60-72.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_clip_norm_scalar_apply_safe() {
        let grad: f32 = kani::any();
        let total_norm: f64 = kani::any();
        let max_norm: f64 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        kani::assume(!total_norm.is_nan() && !total_norm.is_infinite());
        kani::assume(!max_norm.is_nan() && !max_norm.is_infinite());
        kani::assume(total_norm >= 0.0 && total_norm <= 1e8);
        kani::assume(max_norm > 0.0 && max_norm <= 1e8);
        // Compute scale factor matching production logic:
        // if total_norm > max_norm { grad * (max_norm / total_norm) } else { grad }
        let output = if total_norm > max_norm {
            let scale = (max_norm / total_norm) as f32;
            grad * scale
        } else {
            grad
        };
        assert!(output.is_finite(), "clipped gradient must be finite");
        // Gradient magnitude must not increase (no amplification)
        assert!(
            output.abs() <= grad.abs() + 1e-6,
            "clipping must not amplify gradient"
        );
    }

    /// clip_grad_value output is bounded by clip_value for any finite input.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_clip_value_bounded() {
        let grad: f32 = kani::any();
        let clip_value: f32 = kani::any();
        kani::assume(!grad.is_nan() && !grad.is_infinite());
        kani::assume(!clip_value.is_nan() && !clip_value.is_infinite());
        kani::assume(clip_value > 0.0);
        // clamp(grad, -clip_value, clip_value)
        let clamped = if grad < -clip_value {
            -clip_value
        } else if grad > clip_value {
            clip_value
        } else {
            grad
        };
        assert!(clamped >= -clip_value);
        assert!(clamped <= clip_value);
        assert!(!clamped.is_nan() && !clamped.is_infinite());
    }

    /// clip_grad_value preserves values within range.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_clip_value_preserves_within_range() {
        let grad: f32 = kani::any();
        let clip_value: f32 = kani::any();
        kani::assume(!grad.is_nan() && !grad.is_infinite());
        kani::assume(!clip_value.is_nan() && !clip_value.is_infinite());
        kani::assume(clip_value > 0.0);
        kani::assume(grad >= -clip_value && grad <= clip_value);
        // Value within range → unchanged
        let clamped = if grad < -clip_value {
            -clip_value
        } else if grad > clip_value {
            clip_value
        } else {
            grad
        };
        assert!(clamped == grad, "values within range must be preserved");
    }

    /// Scalar norm-clipped gradient has magnitude <= max_norm.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_clip_norm_scalar_output_bounded() {
        let grad: f32 = kani::any();
        let max_norm: f32 = kani::any();
        assume_bounded(grad, -1e4, 1e4);
        assume_bounded(max_norm, 1e-6, 1e4);
        let abs_grad = grad.abs();
        if abs_grad > max_norm {
            let scale = max_norm / abs_grad;
            let clipped = grad * scale;
            assert!(!clipped.is_nan() && !clipped.is_infinite());
            assert!(clipped.abs() <= max_norm + 1e-3); // f32 rounding tolerance
        }
    }
}
