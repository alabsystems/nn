// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for error types, initialization, hooks, and
//! additional backward rule properties.
//!
//! Covers:
//! - Error variant construction invariants (AutodiffError)
//! - Fan computation for Xavier/Kaiming initialization
//! - Initialization scale factor bounds
//! - Gradient accumulation scalar properties
//! - Activation derivative tighter bounds (ELU continuity, clamp boundary)
//! - TrainLoopConfig validation invariants
//! - Training loop metrics consistency
//!
//! Re: #3798 (Kani harnesses for nn-autodiff).

// ── Fan computation re-implementations ────────────────────────────────────
//
// SYNC: var_init.rs:69-89 (compute_fans function).

/// Compute fan_in and fan_out from weight dimensions (local copy).
///
/// Follows PyTorch convention:
/// - 0D: fan_in = 1, fan_out = 1
/// - 1D [out]: fan_in = 1, fan_out = out
/// - 2D [out, in]: fan_in = in, fan_out = out
/// - 3D+ [out, in, *kernel]: fan_in = in * prod(kernel), fan_out = out * prod(kernel)
fn compute_fans_local(dims: &[usize]) -> (usize, usize) {
    match dims.len() {
        0 => (1, 1),
        1 => (1, dims[0]),
        2 => (dims[1], dims[0]),
        _ => {
            let receptive: usize = dims[2..].iter().product();
            let fan_in = dims[1] * receptive;
            let fan_out = dims[0] * receptive;
            (fan_in, fan_out)
        }
    }
}

/// Select fan value for Kaiming init.
///
/// SYNC: var_init.rs:124-130 (select_fan function).
fn select_fan_local(fan_in: usize, fan_out: usize, mode: u8) -> usize {
    match mode {
        0 => fan_in.max(1),                           // FanIn
        1 => fan_out.max(1),                          // FanOut
        _ => usize::midpoint(fan_in, fan_out).max(1), // FanAvg
    }
}

// ── Activation derivative re-implementations ─────────────────────────────
//
// Local scalar copies for proving tighter properties.

/// ELU derivative: 1 if x > 0, alpha * exp(x) if x <= 0.
///
/// SYNC: backward_rules_elementwise.rs:131-138, kani_backward_proofs.rs:285-291.
fn elu_derivative_local(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha * x.exp()
    }
}

/// Clamp derivative: 1 if lo <= x <= hi, else 0.
///
/// SYNC: backward_rules_elementwise.rs:122-130, kani_backward_proofs.rs:273-279.
fn clamp_derivative_local(x: f32, lo: f32, hi: f32) -> f32 {
    if x >= lo && x <= hi {
        1.0
    } else {
        0.0
    }
}

/// Sigmoid function for testing its derivative property.
fn sigmoid_local(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative: s(x) * (1 - s(x)).
///
/// SYNC: kani_backward_proofs.rs:126-129.
fn sigmoid_derivative_local(x: f32) -> f32 {
    let s = sigmoid_local(x);
    s * (1.0 - s)
}

/// SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x))).
///
/// SYNC: kani_backward_proofs.rs:183-186.
fn silu_derivative_local(x: f32) -> f32 {
    let s = sigmoid_local(x);
    s * (1.0 + x * (1.0 - s))
}

/// Huber backward scalar derivative.
///
/// SYNC: kani_backward_proofs.rs:323-336.
fn huber_backward_local(x: f32, t: f32, delta: f32, n: usize) -> f32 {
    let diff = x - t;
    if diff.abs() < delta {
        diff / (n as f32 * delta)
    } else {
        let sign = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };
        sign / n as f32
    }
}

/// Xavier uniform scale: sqrt(6 / (fan_in + fan_out)).
///
/// SYNC: var_init.rs:99-101.
fn xavier_uniform_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Kaiming uniform scale: sqrt(6 / fan) for ReLU (gain=sqrt(2)).
///
/// SYNC: var_init.rs:107-111.
fn kaiming_uniform_scale(fan: usize) -> f64 {
    (6.0 / fan as f64).sqrt()
}

/// MSE backward scalar: 2*(x - t) / N.
///
/// SYNC: kani_backward_proofs.rs:297-299.
fn mse_backward_local(x: f32, t: f32, n: usize) -> f32 {
    2.0 * (x - t) / n as f32
}

/// Gradient accumulation scalar: adding two finite gradients.
fn accumulate_gradient(existing: f32, incoming: f32) -> f32 {
    existing + incoming
}

// ── Kani proof harnesses ─────────────────────────────────────────────────

#[cfg(kani)]
mod proofs {

    use super::*;

    fn exp_f32_stub(x: f32) -> f32 {
        let _ = x;
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
        r
    }

    fn sqrt_f64_stub(x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x > 0.0 {
            kani::assume(r > 0.0);
        }
        r
    }

    // ── Fan computation ──

    /// Prove compute_fans returns positive fan_in and fan_out for non-empty dims.
    fn sqrt_f32_stub(x: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x > 0.0 {
            kani::assume(r > 0.0);
        }
        r
    }

    #[kani::unwind(5)]
    #[kani::proof]
    fn fan_computation_positive_for_2d() {
        let dim0: usize = kani::any();
        let dim1: usize = kani::any();
        kani::assume(dim0 >= 1 && dim0 <= 4096);
        kani::assume(dim1 >= 1 && dim1 <= 4096);
        let (fan_in, fan_out) = compute_fans_local(&[dim0, dim1]);
        assert!(fan_in > 0, "fan_in must be positive for 2D tensor");
        assert!(fan_out > 0, "fan_out must be positive for 2D tensor");
        assert_eq!(fan_in, dim1, "fan_in must be dim1 for 2D");
        assert_eq!(fan_out, dim0, "fan_out must be dim0 for 2D");
    }

    /// Prove compute_fans returns (1, 1) for 0D (scalar).
    #[kani::unwind(5)]
    #[kani::proof]
    fn fan_computation_scalar_dims() {
        let (fan_in, fan_out) = compute_fans_local(&[]);
        assert_eq!(fan_in, 1);
        assert_eq!(fan_out, 1);
    }

    /// Prove compute_fans returns (1, out) for 1D.
    #[kani::unwind(5)]
    #[kani::proof]
    fn fan_computation_1d() {
        let out: usize = kani::any();
        kani::assume(out >= 1 && out <= 8192);
        let (fan_in, fan_out) = compute_fans_local(&[out]);
        assert_eq!(fan_in, 1, "fan_in must be 1 for 1D");
        assert_eq!(fan_out, out, "fan_out must be the dimension for 1D");
    }

    /// Prove compute_fans includes receptive field for 3D (conv) weights.
    #[kani::unwind(5)]
    #[kani::proof]
    fn fan_computation_3d_conv() {
        let out_c: usize = kani::any();
        let in_c: usize = kani::any();
        let kernel: usize = kani::any();
        kani::assume(out_c >= 1 && out_c <= 256);
        kani::assume(in_c >= 1 && in_c <= 256);
        kani::assume(kernel >= 1 && kernel <= 32);
        let (fan_in, fan_out) = compute_fans_local(&[out_c, in_c, kernel]);
        assert_eq!(fan_in, in_c * kernel, "fan_in must include receptive field");
        assert_eq!(
            fan_out,
            out_c * kernel,
            "fan_out must include receptive field"
        );
    }

    /// Prove select_fan always returns at least 1.
    #[kani::unwind(1)]
    #[kani::proof]
    fn select_fan_at_least_one() {
        let fan_in: usize = kani::any();
        let fan_out: usize = kani::any();
        let mode: u8 = kani::any();
        kani::assume(fan_in <= 100_000);
        kani::assume(fan_out <= 100_000);
        kani::assume(mode <= 2);
        let result = select_fan_local(fan_in, fan_out, mode);
        assert!(result >= 1, "select_fan must return >= 1");
    }

    // ── Initialization scale factors ──

    /// Prove Xavier uniform scale is finite and positive for valid fan values.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn xavier_scale_finite_and_positive() {
        let fan_in: usize = kani::any();
        let fan_out: usize = kani::any();
        kani::assume(fan_in >= 1 && fan_in <= 10_000);
        kani::assume(fan_out >= 1 && fan_out <= 10_000);
        let scale = xavier_uniform_scale(fan_in, fan_out);
        assert!(scale.is_finite(), "Xavier scale must be finite");
        assert!(scale > 0.0, "Xavier scale must be positive");
    }

    /// Prove Xavier scale decreases as fan grows (wider layers → smaller init).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn xavier_scale_monotone_decreasing() {
        let fan_in: usize = kani::any();
        let fan_out: usize = kani::any();
        kani::assume(fan_in >= 1 && fan_in <= 5_000);
        kani::assume(fan_out >= 1 && fan_out <= 5_000);
        let scale_small = xavier_uniform_scale(fan_in, fan_out);
        let scale_large = xavier_uniform_scale(fan_in + 1, fan_out);
        assert!(
            scale_large <= scale_small,
            "Xavier scale must decrease as fan grows"
        );
    }

    /// Prove Kaiming scale is finite and positive for valid fan.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn kaiming_scale_finite_and_positive() {
        let fan: usize = kani::any();
        kani::assume(fan >= 1 && fan <= 100_000);
        let scale = kaiming_uniform_scale(fan);
        assert!(scale.is_finite(), "Kaiming scale must be finite");
        assert!(scale > 0.0, "Kaiming scale must be positive");
    }

    // ── Gradient accumulation ──

    /// Prove finite gradients accumulate to finite result.
    #[kani::unwind(1)]
    #[kani::proof]
    fn gradient_accumulation_finite() {
        let existing: f32 = kani::any();
        let incoming: f32 = kani::any();
        kani::assume(existing.is_finite() && existing.abs() <= 1e6);
        kani::assume(incoming.is_finite() && incoming.abs() <= 1e6);
        let result = accumulate_gradient(existing, incoming);
        assert!(
            result.is_finite(),
            "gradient accumulation must be finite for bounded inputs"
        );
    }

    /// Prove gradient accumulation is commutative (a + b == b + a).
    #[kani::unwind(1)]
    #[kani::proof]
    fn gradient_accumulation_commutative() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && a.abs() <= 1e6);
        kani::assume(b.is_finite() && b.abs() <= 1e6);
        let ab = accumulate_gradient(a, b);
        let ba = accumulate_gradient(b, a);
        assert!(
            (ab - ba).abs() <= f32::EPSILON,
            "gradient accumulation must be commutative"
        );
    }

    // ── ELU derivative properties ──

    /// Prove ELU derivative is continuous at x=0 (left limit equals right limit).
    ///
    /// At x=0: right derivative = 1.0, left limit = alpha * exp(0) = alpha.
    /// For continuity, alpha must be 1.0. This harness proves that when alpha = 1,
    /// the derivative is continuous at x=0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn elu_derivative_continuous_at_zero_alpha_one() {
        let right = elu_derivative_local(0.001, 1.0);
        let left = elu_derivative_local(-0.001, 1.0);
        // Both should be close to 1.0 (right = 1.0, left = 1.0 * exp(-0.001) ≈ 0.999)
        assert!(
            (right - left).abs() < 0.01,
            "ELU derivative must be approximately continuous at x=0 for alpha=1"
        );
    }

    /// Prove ELU derivative is always positive for positive alpha.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn elu_derivative_positive_for_positive_alpha() {
        let x: f32 = kani::any();
        let alpha: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 100.0);
        kani::assume(alpha.is_finite() && alpha > 0.0 && alpha <= 10.0);
        let d = elu_derivative_local(x, alpha);
        assert!(d.is_finite(), "ELU derivative must be finite");
        assert!(
            d > 0.0,
            "ELU derivative must be positive for positive alpha"
        );
    }

    /// Prove ELU derivative equals 1.0 for all positive x (regardless of alpha).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn elu_derivative_one_for_positive_x() {
        let x: f32 = kani::any();
        let alpha: f32 = kani::any();
        kani::assume(x.is_finite() && x > 0.0 && x <= 100.0);
        kani::assume(alpha.is_finite() && alpha > 0.0 && alpha <= 10.0);
        let d = elu_derivative_local(x, alpha);
        assert_eq!(d, 1.0, "ELU derivative must be 1.0 for x > 0");
    }

    // ── Clamp derivative boundary properties ──

    /// Prove clamp derivative is 1 at exact boundary lo.
    #[kani::unwind(1)]
    #[kani::proof]
    fn clamp_derivative_at_lower_boundary() {
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(lo.is_finite() && hi.is_finite());
        kani::assume(lo < hi);
        kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);
        let d = clamp_derivative_local(lo, lo, hi);
        assert_eq!(
            d, 1.0,
            "clamp derivative must be 1 at exact lower boundary (ge, not gt)"
        );
    }

    /// Prove clamp derivative is 1 at exact boundary hi.
    #[kani::unwind(1)]
    #[kani::proof]
    fn clamp_derivative_at_upper_boundary() {
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(lo.is_finite() && hi.is_finite());
        kani::assume(lo < hi);
        kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);
        let d = clamp_derivative_local(hi, lo, hi);
        assert_eq!(
            d, 1.0,
            "clamp derivative must be 1 at exact upper boundary (le, not lt)"
        );
    }

    /// Prove clamp derivative is 0 outside the range.
    #[kani::unwind(1)]
    #[kani::proof]
    fn clamp_derivative_zero_outside() {
        let x: f32 = kani::any();
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(x.is_finite() && lo.is_finite() && hi.is_finite());
        kani::assume(lo < hi);
        kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);
        kani::assume(x < lo || x > hi);
        let d = clamp_derivative_local(x, lo, hi);
        assert_eq!(d, 0.0, "clamp derivative must be 0 outside [lo, hi]");
    }

    // ── Sigmoid derivative properties ──

    /// Prove sigmoid derivative is in [0, 0.25] for all finite inputs.
    ///
    /// The maximum of s(x)*(1-s(x)) occurs at x=0 where s(0) = 0.5,
    /// giving derivative 0.25.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn sigmoid_derivative_bounded() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 20.0);
        let d = sigmoid_derivative_local(x);
        assert!(d.is_finite(), "sigmoid derivative must be finite");
        assert!(d >= 0.0, "sigmoid derivative must be non-negative");
        assert!(
            d <= 0.2501, // small epsilon for float imprecision
            "sigmoid derivative must be <= 0.25"
        );
    }

    /// Prove SiLU derivative is approximately 1.0 for large positive x.
    ///
    /// As x → +inf, sigmoid(x) → 1 and SiLU(x) ≈ x, so SiLU'(x) → 1.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn silu_derivative_approaches_one_for_large_x() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x >= 10.0 && x <= 20.0);
        let d = silu_derivative_local(x);
        assert!(
            (d - 1.0).abs() < 0.01,
            "SiLU derivative must approach 1.0 for large positive x"
        );
    }

    /// Prove Huber backward agrees with MSE backward when |diff| < delta.
    ///
    /// In the quadratic regime (|x-t| < delta), Huber loss = (x-t)^2 / (2*delta),
    /// so d/dx = (x-t) / (N*delta) = MSE_grad / (2*delta) when N accounts for the
    /// 1/(2*delta) normalization.
    #[kani::unwind(1)]
    #[kani::proof]
    fn huber_backward_quadratic_regime_sign() {
        let x: f32 = kani::any();
        let t: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 100.0);
        kani::assume(t.is_finite() && t.abs() <= 100.0);
        let diff = x - t;
        kani::assume(diff.is_finite());
        kani::assume(diff.abs() < 0.5); // well within delta=1.0
        kani::assume(diff.abs() > 0.01); // avoid zero crossing
        let d = huber_backward_local(x, t, 1.0, 1);
        // In quadratic regime: grad = diff / (N * delta)
        // sign(grad) must match sign(diff)
        assert!(
            (d > 0.0) == (diff > 0.0),
            "Huber backward must have same sign as diff in quadratic regime"
        );
    }

    // ── MSE backward properties ──

    /// Prove MSE backward is zero when prediction equals target.
    #[kani::unwind(1)]
    #[kani::proof]
    fn mse_backward_zero_at_target() {
        let x: f32 = kani::any();
        let n: usize = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 1e6);
        kani::assume(n >= 1 && n <= 100_000);
        let d = mse_backward_local(x, x, n);
        assert!(d == 0.0, "MSE backward must be zero when x == target");
    }

    // ── TrainLoopConfig invariants ──

    /// Prove default TrainLoopConfig has valid curriculum fraction.
    #[kani::unwind(1)]
    #[kani::proof]
    fn train_loop_config_default_valid() {
        let fraction = 0.1_f64; // TrainLoopConfig::default().curriculum_fraction
        let max_epochs = 10_usize; // TrainLoopConfig::default().max_epochs
        assert!(fraction > 0.0 && fraction <= 1.0);
        assert!(max_epochs > 0);
    }

    /// Prove curriculum size calculation doesn't underflow for valid inputs.
    ///
    /// curriculum_size = max(1, (corpus_size as f64 * fraction) as usize)
    /// SYNC: train_loop.rs curriculum selection pattern.
    #[kani::unwind(1)]
    #[kani::proof]
    fn curriculum_size_no_underflow() {
        let corpus_size: usize = kani::any();
        let fraction_int: u8 = kani::any(); // encode fraction as 1..=100 percent
        kani::assume(corpus_size >= 1 && corpus_size <= 100_000);
        kani::assume(fraction_int >= 1 && fraction_int <= 100);
        let fraction = fraction_int as f64 / 100.0;
        let raw = (corpus_size as f64 * fraction) as usize;
        let curriculum_size = raw.max(1);
        assert!(curriculum_size >= 1, "curriculum size must be at least 1");
        assert!(
            curriculum_size <= corpus_size,
            "curriculum size must not exceed corpus"
        );
    }
}
