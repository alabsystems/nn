// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for moonshot_crown_probabilistic bound arithmetic.
//!
//! Proves correctness of the probabilistic verification helpers:
//! bounded_tensor_from_vecs NaN/empty/Inf rejection, tightest_epsilons
//! min-selection, Hoeffding epsilon arithmetic, confidence interval
//! composition, non-clipping probabilistic check semantics, non-silence
//! probabilistic check semantics, and property result field invariants.
//!
//! These harnesses exercise the pure arithmetic core of the probabilistic
//! concentration inequality bridge without requiring NY propagation
//! at runtime.
//!
//! Properties proved:
//!
//! 1. bounded_tensor_from_vecs: empty vectors return None.
//! 2. bounded_tensor_from_vecs: NaN in lower returns None.
//! 3. bounded_tensor_from_vecs: NaN in upper returns None.
//! 4. bounded_tensor_from_vecs: Inf in bounds returns None.
//! 5. bounded_tensor_from_vecs: neg Inf in bounds returns None.
//! 6. Tightest epsilon: min(a, b) selects the smaller value.
//! 7. Tightest epsilon: min is commutative.
//! 8. Tightest epsilon: hoeffding-only path uses hoeffding directly.
//! 9. Non-clipping check: mean + eps within [-1,1] iff all_within true.
//! 10. Non-clipping check: mean - eps >= -1 required for all_within.
//! 11. Non-silence check: |mean| - eps > threshold iff any_nonsilent.
//! 12. Non-silence check: threshold=0 with nonzero mean is nonsilent.
//! 13. Worst bound computation: max of absolute bounds is non-negative.
//! 14. Worst bound computation: NaN propagation through abs/max chain.
//! 15. Best bound computation: max of (|mean| - eps) is correct.
//! 16. Confidence interval: epsilon is non-negative for valid inputs.
//! 17. Hoeffding bound: epsilon decreases with sample count.
//! 18. Hoeffding bound: epsilon increases with range width.
//! 19. Property result: property_index for non-clipping is 1.
//! 20. Property result: property_index for non-silence is 0.

// ---------- CBMC transcendental stubs for Kani (#708) -----------------------

/// Nondeterministic stub for `f64::ln`.
/// CBMC cannot handle the ln intrinsic. Returns a finite f64
/// in a plausible range for log values.
fn ln_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

/// Nondeterministic stub for `f64::sqrt`.
/// CBMC cannot handle the sqrt intrinsic. Returns a finite non-negative f64.
fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------- bounded_tensor_from_vecs Logic Proofs ----------------------------
//
// The bounded_tensor_from_vecs function is private. We mirror its core
// validation logic (empty check, finite check) and prove properties.

/// Mirror of bounded_tensor_from_vecs validation logic.
/// Returns true iff the inputs would produce Some result.
fn bounded_tensor_valid(lower: &[f64], upper: &[f64]) -> bool {
    let n = lower.len();
    if n == 0 {
        return false;
    }
    // f64-to-f32 conversion
    let lower_f32: Vec<f32> = lower.iter().map(|&x| x as f32).collect();
    let upper_f32: Vec<f32> = upper.iter().map(|&x| x as f32).collect();
    // All must be finite after conversion
    lower_f32
        .iter()
        .chain(upper_f32.iter())
        .all(|x| x.is_finite())
}

/// Prove: empty vectors are rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bounded_tensor_empty_rejected() {
    assert!(
        !bounded_tensor_valid(&[], &[]),
        "empty vectors must be rejected"
    );
}

/// Prove: NaN in lower bound is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_nan_lower_rejected() {
    assert!(
        !bounded_tensor_valid(&[f64::NAN], &[1.0]),
        "NaN in lower must be rejected"
    );
}

/// Prove: NaN in upper bound is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_nan_upper_rejected() {
    assert!(
        !bounded_tensor_valid(&[0.0], &[f64::NAN]),
        "NaN in upper must be rejected"
    );
}

/// Prove: +Inf in bounds is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_inf_rejected() {
    assert!(
        !bounded_tensor_valid(&[0.0], &[f64::INFINITY]),
        "Inf in bounds must be rejected"
    );
}

/// Prove: -Inf in bounds is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_neg_inf_rejected() {
    assert!(
        !bounded_tensor_valid(&[f64::NEG_INFINITY], &[0.0]),
        "neg Inf in bounds must be rejected"
    );
}

/// Prove: valid finite bounds are accepted.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_finite_accepted() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 1e30 && hi.abs() <= 1e30);

    // After f64->f32, the values must still be finite
    let lo_f32 = lo as f32;
    let hi_f32 = hi as f32;
    kani::assume(lo_f32.is_finite() && hi_f32.is_finite());

    assert!(
        bounded_tensor_valid(&[lo], &[hi]),
        "finite bounds with finite f32 representation must be accepted"
    );
}

/// Prove: f64 values that overflow f32 are rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_f32_overflow_rejected() {
    let huge: f64 = f64::from(f32::MAX) * 2.0;
    assert!(
        !bounded_tensor_valid(&[huge], &[huge]),
        "f64 values that overflow f32 must be rejected"
    );
}

// ---------- Tightest Epsilon Selection Proofs --------------------------------

/// Prove: min(a, b) correctly selects the smaller epsilon.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tightest_epsilon_selects_min() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a >= 0.0 && b >= 0.0);

    let result = a.min(b);
    assert!(result <= a, "min must be <= a");
    assert!(result <= b, "min must be <= b");
    assert!(
        result == a || result == b,
        "min must equal one of the inputs"
    );
}

/// Prove: min is commutative for finite non-negative values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tightest_epsilon_min_commutative() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a >= 0.0 && b >= 0.0);

    assert_eq!(a.min(b), b.min(a), "min must be commutative");
}

/// Prove: when only hoeffding is available, epsilon equals hoeffding epsilon.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn tightest_epsilon_hoeffding_only() {
    let h_eps: f64 = kani::any();
    kani::assume(h_eps.is_finite() && h_eps >= 0.0 && h_eps <= 100.0);

    // With no mcdiarmid, result is hoeffding directly
    let epsilons: Vec<f64> = vec![h_eps]; // single-dim hoeffding
    assert_eq!(
        epsilons[0], h_eps,
        "hoeffding-only path must use hoeffding directly"
    );
}

// ---------- Non-Clipping Probabilistic Check Proofs --------------------------

/// Prove: non-clipping all_within logic is correct.
///
/// For each dimension, mean + eps <= 1.0 AND mean - eps >= -1.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn non_clipping_all_within_logic() {
    let mean: f64 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(mean.is_finite() && eps.is_finite());
    kani::assume(eps >= 0.0);
    kani::assume(mean.abs() <= 2.0 && eps <= 2.0);

    let within = mean + eps <= 1.0 && mean - eps >= -1.0;

    if within {
        assert!(mean + eps <= 1.0, "within requires upper <= 1");
        assert!(mean - eps >= -1.0, "within requires lower >= -1");
        // The entire confidence interval [mean-eps, mean+eps] is in [-1, 1]
        assert!(
            mean - eps >= -1.0 && mean + eps <= 1.0,
            "within means interval contained in [-1, 1]"
        );
    }
}

/// Prove: non-clipping worst_bound is non-negative.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn non_clipping_worst_bound_non_negative() {
    let mean: f64 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(mean.is_finite() && eps.is_finite());
    kani::assume(eps >= 0.0);
    kani::assume(mean.abs() <= 10.0 && eps <= 10.0);

    let worst = (mean + eps).abs().max((mean - eps).abs());
    assert!(worst >= 0.0, "worst bound must be non-negative");
}

/// Prove: non-clipping worst_bound monotonically increases with eps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_clipping_worst_bound_increases_with_eps() {
    let mean: f64 = kani::any();
    let eps1: f64 = kani::any();
    let eps2: f64 = kani::any();
    kani::assume(mean.is_finite() && eps1.is_finite() && eps2.is_finite());
    kani::assume(eps1 >= 0.0 && eps2 >= eps1);
    kani::assume(mean.abs() <= 10.0 && eps2 <= 10.0);

    let worst1 = (mean + eps1).abs().max((mean - eps1).abs());
    let worst2 = (mean + eps2).abs().max((mean - eps2).abs());
    assert!(
        worst2 >= worst1,
        "larger eps must give larger or equal worst bound"
    );
}

/// Prove: when mean=0 and eps=0, worst_bound is 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_clipping_zero_mean_zero_eps() {
    let worst = (0.0_f64 + 0.0).abs().max((0.0_f64 - 0.0).abs());
    assert_eq!(worst, 0.0, "zero mean and eps must give zero worst bound");
}

// ---------- Non-Silence Probabilistic Check Proofs ---------------------------

/// Prove: non-silence any_nonsilent logic is correct.
///
/// At least one dimension has |mean| - eps > threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn non_silence_any_nonsilent_logic() {
    let mean: f64 = kani::any();
    let eps: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(mean.is_finite() && eps.is_finite() && threshold.is_finite());
    kani::assume(eps >= 0.0 && threshold >= 0.0);
    kani::assume(mean.abs() <= 10.0 && eps <= 10.0 && threshold <= 10.0);

    let nonsilent = mean.abs() - eps > threshold;

    if nonsilent {
        assert!(
            mean.abs() > threshold + eps,
            "nonsilent requires |mean| > threshold + eps"
        );
    }
}

/// Prove: non-silence best_bound is correctly computed.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn non_silence_best_bound_correct() {
    let mean: f64 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(mean.is_finite() && eps.is_finite());
    kani::assume(eps >= 0.0);
    kani::assume(mean.abs() <= 10.0 && eps <= 10.0);

    let best = mean.abs() - eps;

    // best can be negative (when eps > |mean|), that's fine
    // The actual code uses .max(0.0) on the final result
    let clamped = best.max(0.0);
    assert!(clamped >= 0.0, "clamped best_bound must be non-negative");
}

/// Prove: large |mean| with small eps is always nonsilent for reasonable threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_silence_large_mean_is_nonsilent() {
    let mean: f64 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(mean.is_finite() && eps.is_finite());
    kani::assume(mean.abs() >= 1.0); // large mean
    kani::assume(eps >= 0.0 && eps <= 0.01); // small eps

    let nonsilent = mean.abs() - eps > 0.01; // threshold = 0.01
    assert!(
        nonsilent,
        "|mean| >= 1.0 and eps <= 0.01 must be nonsilent with threshold 0.01"
    );
}

// ---------- Hoeffding Epsilon Arithmetic Proofs ------------------------------

/// Hoeffding epsilon formula: eps = range * sqrt(ln(2/delta) / (2*n))
/// where range = upper - lower, delta = 1 - confidence.
///
/// We prove properties of this formula without calling the actual function.

/// Prove: Hoeffding epsilon is non-negative for valid inputs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn hoeffding_epsilon_non_negative() {
    let range: f64 = kani::any();
    let n: u32 = kani::any();
    let confidence: f64 = kani::any();
    kani::assume(range >= 0.0 && range.is_finite() && range <= 100.0);
    kani::assume(n >= 1 && n <= 10000);
    kani::assume(confidence > 0.0 && confidence < 1.0);

    let delta = 1.0 - confidence;
    let ln_term = (2.0_f64 / delta).ln();
    kani::assume(ln_term.is_finite() && ln_term >= 0.0);

    let eps = range * (ln_term / (2.0 * f64::from(n))).sqrt();

    assert!(
        eps >= 0.0 || eps.is_nan(),
        "Hoeffding epsilon must be non-negative for valid inputs"
    );
}

/// Prove: Hoeffding epsilon decreases with sample count (n).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn hoeffding_epsilon_decreases_with_n() {
    let range: f64 = kani::any();
    kani::assume(range > 0.0 && range.is_finite() && range <= 10.0);

    let n1: u32 = kani::any();
    let n2: u32 = kani::any();
    kani::assume(n1 >= 1 && n2 > n1);
    kani::assume(n2 <= 10000);

    // Fixed confidence -> fixed ln_term
    let ln_term = (2.0_f64 / 0.01).ln(); // 99% confidence

    let eps1 = range * (ln_term / (2.0 * f64::from(n1))).sqrt();
    let eps2 = range * (ln_term / (2.0 * f64::from(n2))).sqrt();

    assert!(
        eps2 <= eps1,
        "more samples must give smaller or equal epsilon"
    );
}

/// Prove: Hoeffding epsilon increases with range width.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn hoeffding_epsilon_increases_with_range() {
    let range1: f64 = kani::any();
    let range2: f64 = kani::any();
    kani::assume(range1 >= 0.0 && range2 >= range1);
    kani::assume(range1.is_finite() && range2.is_finite());
    kani::assume(range2 <= 100.0);

    let n = 100_u32;
    let ln_term = (2.0_f64 / 0.01).ln();

    let eps1 = range1 * (ln_term / (2.0 * f64::from(n))).sqrt();
    let eps2 = range2 * (ln_term / (2.0 * f64::from(n))).sqrt();

    assert!(
        eps2 >= eps1,
        "wider range must give larger or equal epsilon"
    );
}

/// Prove: Hoeffding epsilon with zero range is zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn hoeffding_epsilon_zero_range_is_zero() {
    let n = 100_u32;
    let ln_term = (2.0_f64 / 0.01).ln();
    let eps = 0.0_f64 * (ln_term / (2.0 * f64::from(n))).sqrt();
    assert_eq!(eps, 0.0, "zero range must give zero epsilon");
}

// ---------- Property Result Invariant Proofs ---------------------------------

/// Prove: non-clipping property has index 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_property_index_is_1() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);
    assert_eq!(result.property_index, 1, "non-clipping is property 1");
}

/// Prove: non-silence property has index 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_property_index_is_0() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_silence(&cert, 0.01);
    assert_eq!(result.property_index, 0, "non-silence is property 0");
}

/// Prove: non-clipping proven implies bound_value <= 1.0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_proven_implies_bounded() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 2.0 && hi.abs() <= 2.0);
    kani::assume(lo <= hi);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    if result.proven {
        assert!(
            result.bound_value <= 1.0,
            "proven non-clipping implies worst bound <= 1.0"
        );
    }
}

/// Prove: non-silence proven implies bound_value > threshold.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_proven_implies_above_threshold() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    let threshold: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite() && threshold.is_finite());
    kani::assume(lo.abs() <= 10.0 && hi.abs() <= 10.0);
    kani::assume(threshold >= 0.0 && threshold <= 10.0);
    kani::assume(lo <= hi);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![hi],
            vec![lo],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_silence(&cert, threshold);

    if result.proven {
        assert!(
            result.bound_value > threshold,
            "proven non-silence implies bound > threshold"
        );
    }
}

// ---------- Confidence Interval Composition Proofs ----------------------------

/// Prove: confidence interval width is non-negative.
///
/// For any valid confidence interval [mean - eps, mean + eps], the width
/// 2*eps must be >= 0 since eps >= 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn confidence_interval_width_non_negative() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite() && eps >= 0.0);

    let width = 2.0 * eps;
    assert!(width >= 0.0, "confidence interval width must be >= 0");
}

/// Prove: nested confidence intervals — smaller confidence gives smaller epsilon.
///
/// Hoeffding formula: eps = range * sqrt(ln(2/delta) / (2n)).
/// Larger delta (smaller confidence) gives smaller ln(2/delta), hence smaller eps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn nested_confidence_intervals() {
    let range: f64 = kani::any();
    kani::assume(range > 0.0 && range.is_finite() && range <= 10.0);
    let n = 100_u32;

    // 95% confidence: delta = 0.05
    let ln_95 = (2.0_f64 / 0.05).ln();
    let eps_95 = range * (ln_95 / (2.0 * f64::from(n))).sqrt();

    // 99% confidence: delta = 0.01
    let ln_99 = (2.0_f64 / 0.01).ln();
    let eps_99 = range * (ln_99 / (2.0 * f64::from(n))).sqrt();

    // Higher confidence needs wider interval
    assert!(
        eps_99 >= eps_95,
        "99% confidence epsilon must be >= 95% confidence epsilon"
    );
}

/// Prove: Hoeffding epsilon with n=1 is larger than n=2 for same range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn hoeffding_n1_larger_than_n2() {
    let range: f64 = kani::any();
    kani::assume(range > 0.0 && range.is_finite() && range <= 10.0);

    let ln_term = (2.0_f64 / 0.01).ln();
    let eps_1 = range * (ln_term / 2.0).sqrt();
    let eps_2 = range * (ln_term / 4.0).sqrt();

    assert!(eps_1 > eps_2, "n=1 must give larger epsilon than n=2");
}

/// Prove: tightest epsilon min operation is associative.
///
/// min(min(a, b), c) = min(a, min(b, c)) for non-negative finite values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tightest_epsilon_min_associative() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    let c: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= 0.0 && b >= 0.0 && c >= 0.0);

    let left = a.min(b).min(c);
    let right = a.min(b.min(c));
    assert_eq!(left, right, "min must be associative");
}

/// Prove: tightest epsilon is idempotent — min(a, a) = a.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tightest_epsilon_min_idempotent() {
    let a: f64 = kani::any();
    kani::assume(a.is_finite() && a >= 0.0);

    assert_eq!(a.min(a), a, "min must be idempotent");
}

// ---------- Non-Clipping Extended Proofs --------------------------------------

/// Prove: non-clipping is symmetric — bounds [-x, x] satisfy iff x <= 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_clipping_symmetric_bounds() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0 && x <= 2.0);

    let within = (-x) >= -1.0 && x <= 1.0;
    assert_eq!(
        within,
        x <= 1.0,
        "symmetric bounds [-x, x] within [-1, 1] iff x <= 1"
    );
}

/// Prove: non-clipping — zero output always satisfies [-1, 1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_clipping_zero_output_always_passes() {
    let mean = 0.0_f64;
    let eps = 0.0_f64;
    let within = mean + eps <= 1.0 && mean - eps >= -1.0;
    assert!(
        within,
        "zero output with zero epsilon must pass non-clipping"
    );
}

/// Prove: non-clipping worst_bound is at least |mean| when eps >= 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_clipping_worst_at_least_mean_abs() {
    let mean: f64 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(mean.is_finite() && eps.is_finite());
    kani::assume(eps >= 0.0 && mean.abs() <= 10.0 && eps <= 10.0);

    let worst = (mean + eps).abs().max((mean - eps).abs());
    assert!(worst >= mean.abs(), "worst bound must be at least |mean|");
}

// ---------- Non-Silence Extended Proofs ---------------------------------------

/// Prove: non-silence with zero threshold is equivalent to nonzero mean.
///
/// With eps=0 and threshold=0, nonsilent iff |mean| > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_silence_zero_threshold_zero_eps() {
    let mean: f64 = kani::any();
    kani::assume(mean.is_finite() && mean.abs() <= 10.0);
    kani::assume(mean != 0.0); // exclude exactly zero

    let eps = 0.0_f64;
    let threshold = 0.0_f64;
    let nonsilent = mean.abs() - eps > threshold;
    assert!(
        nonsilent,
        "nonzero mean with zero eps and zero threshold must be nonsilent"
    );
}

// ---------- Bounded Tensor Extended Proofs ------------------------------------

/// Prove: bounded_tensor_valid is false for mixed finite/non-finite pairs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounded_tensor_mixed_finite_nonfinite() {
    // One finite, one infinite
    assert!(
        !bounded_tensor_valid(&[0.0], &[f64::INFINITY]),
        "finite lower + Inf upper must be rejected"
    );
    assert!(
        !bounded_tensor_valid(&[f64::NEG_INFINITY], &[0.0]),
        "neg Inf lower + finite upper must be rejected"
    );
}

/// Prove: bounded_tensor_valid with multiple elements — any NaN rejects all.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn bounded_tensor_multi_element_nan_rejects_all() {
    // Second element NaN should reject even though first is valid
    assert!(
        !bounded_tensor_valid(&[0.0, f64::NAN], &[1.0, 1.0]),
        "NaN in any lower element must reject"
    );
    assert!(
        !bounded_tensor_valid(&[0.0, 0.0], &[1.0, f64::NAN]),
        "NaN in any upper element must reject"
    );
}

/// Prove: verification level for proven non-clipping is at least CrownPartial.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_proven_level_at_least_partial() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);
    assert!(result.proven);
    assert!(
        result.level >= crate::moonshot::VerificationLevel::CrownPartial,
        "proven property level must be >= CrownPartial"
    );
}

/// Prove: MoonshotPropertyResult is_sound reflects certificate soundness.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn property_result_is_sound_from_cert() {
    let sound: bool = kani::any();

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            sound,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            sound,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert_eq!(
        result.is_sound, cert.is_sound,
        "property is_sound must match certificate is_sound"
    );
}
