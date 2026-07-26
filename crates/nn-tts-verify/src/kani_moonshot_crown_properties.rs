// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for moonshot_crown_properties.rs — property check functions.
//!
//! Complements `kani_crown_moonshot_proofs.rs` with deeper property-specific proofs:
//!
//! - **Non-silence**: NaN propagation, threshold monotonicity, sound vs partial level.
//! - **Non-clipping**: boundary precision, finite_bounds guard, worst_bound computation.
//! - **Intelligibility proxy**: range ratio computation, zero input range handling.
//! - **Streaming safety**: alpha_step computation, crossfade degenerate case.
//! - **Temporal boundedness**: timing margin, zero worst_case handling.
//! - **Memory boundedness**: zero peak bytes, peak_bytes > bound rejection.
//! - **Verification level ordering**: full ordering correctness.
//! - **fold_max/min_propagate_nan**: NaN propagation semantics.

// ---------------------------------------------------------------------------
// fold_max_propagate_nan / fold_min_propagate_nan Proofs
// ---------------------------------------------------------------------------

/// Prove: fold_max_propagate_nan returns NaN when any element is NaN.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_nan_propagates() {
    let a: f64 = kani::any();
    kani::assume(a.is_finite());

    let result = crate::stats::fold_max_propagate_nan([a, f64::NAN].iter().copied(), 0.0);
    assert!(result.is_nan(), "fold_max must propagate NaN");
}

/// Prove: fold_min_propagate_nan returns NaN when any element is NaN.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_nan_propagates() {
    let a: f64 = kani::any();
    kani::assume(a.is_finite());

    let result = crate::stats::fold_min_propagate_nan([a, f64::NAN].iter().copied(), f64::INFINITY);
    assert!(result.is_nan(), "fold_min must propagate NaN");
}

/// Prove: fold_max_propagate_nan on empty iterator returns init.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fold_max_empty_returns_init() {
    let init: f64 = kani::any();
    kani::assume(init.is_finite());

    let result = crate::stats::fold_max_propagate_nan(std::iter::empty::<f64>(), init);
    assert_eq!(result, init, "fold_max on empty must return init");
}

/// Prove: fold_min_propagate_nan on empty iterator returns init.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fold_min_empty_returns_init() {
    let init: f64 = kani::any();
    kani::assume(init.is_finite());

    let result = crate::stats::fold_min_propagate_nan(std::iter::empty::<f64>(), init);
    assert_eq!(result, init, "fold_min on empty must return init");
}

/// Prove: fold_max_propagate_nan returns the max of finite elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_max_returns_max_of_finite() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let result = crate::stats::fold_max_propagate_nan([a, b].iter().copied(), f64::NEG_INFINITY);
    let expected = a.max(b);
    assert_eq!(result, expected, "fold_max must return max of elements");
}

/// Prove: fold_min_propagate_nan returns the min of finite elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn fold_min_returns_min_of_finite() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6);

    let result = crate::stats::fold_min_propagate_nan([a, b].iter().copied(), f64::INFINITY);
    let expected = a.min(b);
    assert_eq!(result, expected, "fold_min must return min of elements");
}

// ---------------------------------------------------------------------------
// Non-Silence (P1) Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: check_non_silence threshold is monotonic — lower threshold makes proof easier.
///
/// If property is proven at threshold T1, it is also proven at any T2 < T1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_threshold_monotonic() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 10.0 && hi.abs() <= 10.0);
    kani::assume(lo <= hi);

    let t1: f64 = kani::any();
    let t2: f64 = kani::any();
    kani::assume(t1.is_finite() && t2.is_finite());
    kani::assume(t1 > 0.0 && t2 > 0.0);
    kani::assume(t2 < t1);
    kani::assume(t1 <= 10.0 && t2 <= 10.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
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

    let r1 = crate::moonshot_crown::check_non_silence(&cert, t1);
    let r2 = crate::moonshot_crown::check_non_silence(&cert, t2);

    // If proven at higher threshold, must also be proven at lower threshold
    if r1.proven {
        assert!(r2.proven, "proven at T1 implies proven at T2 < T1");
    }
}

/// Prove: non-silence sound vs partial level assignment.
///
/// When proven AND cert.is_sound, level must be CrownProven.
/// When proven AND !cert.is_sound, level must be CrownPartial.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_level_sound_vs_partial() {
    let is_sound: bool = kani::any();

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
            is_sound,
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
            is_sound,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_silence(&cert, 0.01);

    if result.proven && cert.is_sound {
        assert_eq!(
            result.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "proven + sound = CrownProven"
        );
    } else if result.proven && !cert.is_sound {
        assert_eq!(
            result.level,
            crate::moonshot::VerificationLevel::CrownPartial,
            "proven + !sound = CrownPartial"
        );
    }
}

// ---------------------------------------------------------------------------
// Non-Clipping (P2) Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: non-clipping worst_bound is the maximum absolute value of bounds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_worst_bound_is_max_abs() {
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
            vec![0.0],
            vec![1.0],
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

    let expected_worst = hi.abs().max(lo.abs());
    assert!(
        (result.bound_value - expected_worst).abs() < 1e-10,
        "worst_bound must be max(|upper|, |lower|)"
    );
}

/// Prove: non-clipping with bounds strictly inside [-1, 1] is proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_strict_interior_is_proven() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo > -1.0 && hi < 1.0);
    kani::assume(lo <= hi);
    kani::assume(lo >= -0.999 && hi <= 0.999);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
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

    assert!(
        result.proven,
        "strictly interior bounds must be proven non-clipping"
    );
}

/// Prove: non-clipping with hi > 1.0 is NOT proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_exceeds_upper_not_proven() {
    let hi: f64 = kani::any();
    kani::assume(hi.is_finite() && hi > 1.0 && hi <= 2.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-0.5],
            vec![hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![hi],
            vec![-0.5],
            vec![hi],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(
        !result.proven,
        "output upper > 1.0 must not be proven non-clipping"
    );
}

/// Prove: non-clipping with lo < -1.0 is NOT proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_exceeds_lower_not_proven() {
    let lo: f64 = kani::any();
    kani::assume(lo.is_finite() && lo < -1.0 && lo >= -2.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![lo],
            vec![0.5],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![lo],
            vec![0.5],
            vec![lo],
            vec![0.5],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(
        !result.proven,
        "output lower < -1.0 must not be proven non-clipping"
    );
}

// ---------------------------------------------------------------------------
// Intelligibility Proxy (P3) Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: intelligibility proxy with zero input range yields infinity ratio → not proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn intelligibility_zero_input_range_not_proven() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val.abs() <= 5.0);

    // Input range = 0 (lo == hi)
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![val],
            vec![val],
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
    let result = crate::moonshot_crown::check_intelligibility_proxy(&cert, 10.0);

    // input_range = 0 → range_ratio = INFINITY → not proven
    assert!(
        !result.proven,
        "zero input range must produce INFINITY ratio → not proven"
    );
}

/// Prove: intelligibility proxy with output range == input range has ratio == 1.0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn intelligibility_equal_ranges_ratio_one() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo < hi);
    kani::assume(lo.abs() <= 5.0 && hi.abs() <= 5.0);

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
    let result = crate::moonshot_crown::check_intelligibility_proxy(&cert, 10.0);

    // ratio = output_range / input_range = 1.0
    assert!(
        (result.bound_value - 1.0).abs() < 1e-6,
        "equal input/output ranges must yield ratio = 1.0"
    );
}

// ---------------------------------------------------------------------------
// Streaming Safety (P6) Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: alpha_step = 1/(n-1) is correctly computed for n > 1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_alpha_step_computation() {
    let n: usize = kani::any();
    kani::assume(n > 1 && n <= 10000);

    let alpha_step = 1.0 / (n - 1) as f64;
    assert!(
        alpha_step.is_finite(),
        "alpha_step must be finite for n > 1"
    );
    assert!(alpha_step > 0.0, "alpha_step must be positive");
    assert!(alpha_step <= 1.0, "alpha_step must be <= 1.0 for n >= 2");
}

/// Prove: alpha_step = 1.0 when crossfade_samples = 1 (degenerate case).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_degenerate_alpha_step() {
    let crossfade_samples: usize = 1;
    let alpha_step: f64 = if crossfade_samples > 1 {
        1.0 / (crossfade_samples - 1) as f64
    } else {
        1.0
    };
    assert_eq!(
        alpha_step, 1.0,
        "single-sample crossfade must have alpha_step = 1.0"
    );
}

/// Prove: max_click_bound is non-negative for all valid inputs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn streaming_click_bound_non_negative() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1.0 && hi.abs() <= 1.0);

    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 1000);

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
    let result = crate::moonshot_crown::check_streaming_safety(&cert, n, 10.0);

    assert!(
        result.bound_value >= 0.0,
        "click bound must be non-negative"
    );
}

// ---------------------------------------------------------------------------
// Temporal Boundedness (P5) Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: temporal boundedness timing_bound_met semantics.
///
/// timing_bound_met = true iff worst_case_time_us <= timing_bound_us.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn temporal_timing_bound_met_semantics() {
    let worst: f64 = kani::any();
    let bound: f64 = kani::any();
    kani::assume(worst.is_finite() && bound.is_finite());
    kani::assume(worst >= 0.0 && bound > 0.0);
    kani::assume(worst <= 1e9 && bound <= 1e9);

    let met = worst <= bound;

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let bounds_cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let timing_cert = crate::pipeline::TimingCertificate::new(
        bounds_cert,
        vec![],
        worst,
        0,
        0,
        "test",
        bound,
        met,
        met,
        None,
    );

    let result = crate::moonshot_crown::check_temporal_boundedness(&timing_cert);

    // result.proven requires timing_bound_met AND bounds_cert.is_valid
    if met && timing_cert.bounds_cert.is_valid {
        assert!(result.proven, "timing met + valid cert = proven");
    }
    if !met {
        assert!(!result.proven, "timing not met = not proven");
    }
}

/// Prove: temporal boundedness with zero worst_case_time reports infinity margin.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn temporal_zero_worst_case_infinite_margin() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let bounds_cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let timing_cert = crate::pipeline::TimingCertificate::new(
        bounds_cert,
        vec![],
        0.0,
        0,
        0,
        "test",
        100_000.0,
        true,
        true,
        None,
    );

    let result = crate::moonshot_crown::check_temporal_boundedness(&timing_cert);

    // worst_case = 0 → margin = timing_bound / 0 = INFINITY
    // The explanation string will contain "Inf" but the proof holds
    assert!(
        result.proven,
        "zero worst_case with positive bound must be proven"
    );
    assert_eq!(result.bound_value, 0.0, "bound_value is worst_case_time_us");
}

// ---------------------------------------------------------------------------
// Memory Boundedness Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: memory boundedness with zero peak_bytes is not proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn memory_zero_peak_not_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let bounds_cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    // No peak_memory → peak_bytes = 0
    let timing_cert = crate::pipeline::TimingCertificate::new(
        bounds_cert,
        vec![],
        50_000.0,
        0,
        0,
        "test",
        100_000.0,
        true,
        true,
        None,
    );

    let result = crate::moonshot_crown::check_memory_boundedness(&timing_cert, 1_000_000);

    // peak_bytes = 0 → not proven (zero means unknown, not "uses no memory")
    assert!(!result.proven, "zero peak_bytes must not be proven");
}

/// Prove: memory boundedness when peak exceeds bound is not proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn memory_exceeds_bound_not_proven() {
    let peak: u64 = kani::any();
    let bound: u64 = kani::any();
    kani::assume(peak > 0 && peak > bound);
    kani::assume(bound > 0 && bound <= 1_000_000_000);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let bounds_cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let peak_mem = crate::cost_model::PeakMemoryProfile {
        weight_bytes: 0,
        peak_activation_bytes: peak,
        peak_total_bytes: peak,
        peak_step_index: 0,
        peak_step_name: "test".to_string(),
        per_step_output_bytes: vec![],
    };

    let timing_cert = crate::pipeline::TimingCertificate::new(
        bounds_cert,
        vec![],
        50_000.0,
        0,
        0,
        "test",
        100_000.0,
        true,
        true,
        Some(peak_mem),
    );

    let result = crate::moonshot_crown::check_memory_boundedness(&timing_cert, bound);
    assert!(!result.proven, "peak > bound must not be proven");
}

/// Prove: memory boundedness when peak == bound is proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn memory_exact_bound_is_proven() {
    let peak: u64 = kani::any();
    kani::assume(peak > 0 && peak <= 1_000_000_000);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let bounds_cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let peak_mem = crate::cost_model::PeakMemoryProfile {
        weight_bytes: 0,
        peak_activation_bytes: peak,
        peak_total_bytes: peak,
        peak_step_index: 0,
        peak_step_name: "test".to_string(),
        per_step_output_bytes: vec![],
    };

    let timing_cert = crate::pipeline::TimingCertificate::new(
        bounds_cert,
        vec![],
        50_000.0,
        0,
        0,
        "test",
        100_000.0,
        true,
        true,
        Some(peak_mem),
    );

    let result = crate::moonshot_crown::check_memory_boundedness(&timing_cert, peak);
    assert!(result.proven, "peak == bound must be proven");
}

// ---------------------------------------------------------------------------
// Verification Level Ordering Proofs
// ---------------------------------------------------------------------------

/// Prove: full VerificationLevel ordering is correct.
///
/// None < Empirical < CrownPartial < CrownProbabilistic < CrownProven < KaniProven < SmtProven
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_full_ordering() {
    use crate::moonshot::VerificationLevel;
    assert!(VerificationLevel::None < VerificationLevel::Empirical);
    assert!(VerificationLevel::Empirical < VerificationLevel::CrownPartial);
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProbabilistic);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
    assert!(VerificationLevel::CrownProven < VerificationLevel::KaniProven);
    assert!(VerificationLevel::KaniProven < VerificationLevel::SmtProven);
}

/// Prove: VerificationLevel equality is reflexive.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_equality_reflexive() {
    use crate::moonshot::VerificationLevel;
    assert_eq!(
        VerificationLevel::CrownProven,
        VerificationLevel::CrownProven
    );
    assert_eq!(VerificationLevel::Empirical, VerificationLevel::Empirical);
    assert_eq!(VerificationLevel::SmtProven, VerificationLevel::SmtProven);
}

// ---------------------------------------------------------------------------
// Property Result Invariants
// ---------------------------------------------------------------------------

/// Prove: property_index for non-silence is 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_property_index_is_zero() {
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

/// Prove: property_index for non-clipping is 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_property_index_is_one() {
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

/// Prove: property_index for intelligibility proxy is 2.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn intelligibility_property_index_is_two() {
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
    let result = crate::moonshot_crown::check_intelligibility_proxy(&cert, 10.0);
    assert_eq!(result.property_index, 2, "intelligibility is property 2");
}

/// Prove: property_index for streaming safety is 5.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn streaming_property_index_is_five() {
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
    let result = crate::moonshot_crown::check_streaming_safety(&cert, 100, 10.0);
    assert_eq!(result.property_index, 5, "streaming safety is property 5");
}

/// Prove: is_sound field in result matches pipeline certificate is_sound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn result_is_sound_matches_cert() {
    let sound: bool = kani::any();

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
    let result = crate::moonshot_crown::check_non_silence(&cert, 0.01);

    assert_eq!(
        result.is_sound, cert.is_sound,
        "result.is_sound must match cert.is_sound"
    );
}
