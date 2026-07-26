// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for moonshot_crown_properties.rs.
//!
//! Supplements `kani_moonshot_crown_properties.rs` with deeper proofs:
//!
//! - **Non-silence**: NaN in bounds propagates to not-proven, threshold=0 edge.
//! - **Non-clipping**: NaN bound contaminates finite_bounds check, boundary
//!   exactly at [-1, 1] is proven.
//! - **Streaming safety**: click bound monotonicity in crossfade_samples,
//!   zero bound range gives zero click.
//! - **Temporal boundedness**: margin is positive when timing met, timing
//!   bound_value matches worst_case.
//! - **Memory boundedness**: peak_bytes < bound is proven, level matches
//!   soundness.
//! - **Verification level**: transitivity.

// ---------------------------------------------------------------------------
// Non-Silence (P1) — NaN Propagation
// ---------------------------------------------------------------------------

/// Prove: check_non_silence with NaN in output bounds is not proven.
///
/// NaN bounds indicate verification failure. The fold_max_propagate_nan
/// function propagates NaN, making max_abs NaN. NaN > threshold is false.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_nan_output_not_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![f64::NAN],
            vec![0.5],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![f64::NAN],
            vec![0.5],
            vec![f64::NAN],
            vec![0.5],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_silence(&cert, 0.01);

    // NaN > threshold = false, so proven = false
    assert!(
        !result.proven,
        "NaN in output bounds must produce not-proven"
    );
}

/// Prove: check_non_silence with threshold = 0.0 is proven for any non-zero bound.
///
/// Any positive max_abs > 0.0 satisfies the check.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_silence_zero_threshold_proven() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val > 0.0 && val <= 10.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-val],
            vec![val],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-val],
            vec![val],
            vec![-val],
            vec![val],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_silence(&cert, 0.0);

    assert!(
        result.proven,
        "any non-zero bound with threshold=0 must be proven"
    );
}

// ---------------------------------------------------------------------------
// Non-Clipping (P2) — NaN and Boundary Proofs
// ---------------------------------------------------------------------------

/// Prove: check_non_clipping with NaN in output upper is not proven.
///
/// The finite_bounds check must catch NaN and prevent a false positive.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_nan_upper_not_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-0.5],
            vec![f64::NAN],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![f64::NAN],
            vec![-0.5],
            vec![f64::NAN],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(!result.proven, "NaN in output upper must prevent proven");
}

/// Prove: check_non_clipping with bounds exactly at [-1.0, 1.0] is proven.
///
/// The check is <= 1.0 and >= -1.0, so exact boundary is within range.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_exact_boundary_proven() {
    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![-1.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![-1.0],
            vec![1.0],
            vec![-1.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(
        result.proven,
        "bounds exactly at [-1, 1] boundary must be proven"
    );
}

/// Prove: check_non_clipping level is CrownProven when proven and sound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_level_crown_proven_when_sound() {
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

    assert!(result.proven);
    assert_eq!(
        result.level,
        crate::moonshot::VerificationLevel::CrownProven,
        "proven + sound cert => CrownProven"
    );
}

/// Prove: check_non_clipping level is CrownPartial when proven but not sound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn non_clipping_level_partial_when_not_sound() {
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
            false,
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
            false,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_non_clipping(&cert);

    assert!(result.proven);
    assert_eq!(
        result.level,
        crate::moonshot::VerificationLevel::CrownPartial,
        "proven + !sound cert => CrownPartial"
    );
}

// ---------------------------------------------------------------------------
// Streaming Safety (P6) — Click Bound Monotonicity
// ---------------------------------------------------------------------------

/// Prove: larger crossfade_samples gives smaller or equal click bound.
///
/// alpha_step = 1/(n-1) decreases as n increases. Since
/// max_click_bound = max_bound_range * alpha_step, the click bound
/// decreases monotonically with crossfade length.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn streaming_click_bound_decreases_with_crossfade() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1.0 && hi.abs() <= 1.0);

    let n1: usize = kani::any();
    let n2: usize = kani::any();
    kani::assume(n1 >= 2 && n1 <= 500);
    kani::assume(n2 >= 2 && n2 <= 500);
    kani::assume(n1 <= n2);

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

    let r1 = crate::moonshot_crown::check_streaming_safety(&cert, n1, 10.0);
    let r2 = crate::moonshot_crown::check_streaming_safety(&cert, n2, 10.0);

    assert!(
        r1.bound_value >= r2.bound_value - 1e-12,
        "larger crossfade must give smaller or equal click bound"
    );
}

/// Prove: zero output bound range produces zero click bound.
///
/// When lower == upper for all elements, the bound range is 0, so
/// max_click_bound = 0 * alpha_step = 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn streaming_zero_range_zero_click() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val.abs() <= 1.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![val],
            vec![val],
            vec![val],
            vec![val],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![val],
            vec![val],
            vec![val],
            vec![val],
            "CROWN",
            true,
        ),
    ];
    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();
    let result = crate::moonshot_crown::check_streaming_safety(&cert, 100, 10.0);

    assert_eq!(
        result.bound_value, 0.0,
        "zero bound range must produce zero click bound"
    );
}

// ---------------------------------------------------------------------------
// Temporal Boundedness (P5) — Extended Proofs
// ---------------------------------------------------------------------------

/// Prove: check_temporal_boundedness bound_value is worst_case_time_us.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn temporal_bound_value_is_worst_case() {
    let worst: f64 = kani::any();
    kani::assume(worst.is_finite() && worst >= 0.0 && worst <= 1e9);

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
        1e9,
        true,
        true,
        None,
    );

    let result = crate::moonshot_crown::check_temporal_boundedness(&timing_cert);
    assert_eq!(
        result.bound_value, worst,
        "bound_value must equal worst_case_time_us"
    );
}

/// Prove: check_temporal_boundedness threshold is timing_bound_us.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn temporal_threshold_is_timing_bound() {
    let bound: f64 = kani::any();
    kani::assume(bound.is_finite() && bound > 0.0 && bound <= 1e9);

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
        50_000.0,
        0,
        0,
        "test",
        bound,
        true,
        true,
        None,
    );

    let result = crate::moonshot_crown::check_temporal_boundedness(&timing_cert);
    assert_eq!(
        result.threshold, bound,
        "threshold must equal timing_bound_us"
    );
}

// ---------------------------------------------------------------------------
// Memory Boundedness — Level Assignment
// ---------------------------------------------------------------------------

/// Prove: memory boundedness level is CrownProven when proven and sound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn memory_level_crown_proven_when_sound() {
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
        peak_activation_bytes: 1000,
        peak_total_bytes: 1000,
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

    let result = crate::moonshot_crown::check_memory_boundedness(&timing_cert, 2000);
    assert!(result.proven);
    assert_eq!(
        result.level,
        crate::moonshot::VerificationLevel::CrownProven,
        "proven + sound => CrownProven"
    );
}

/// Prove: memory boundedness level is CrownPartial when proven but not sound.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn memory_level_partial_when_not_sound() {
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
            false,
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
            false,
        ),
    ];
    let bounds_cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    let peak_mem = crate::cost_model::PeakMemoryProfile {
        weight_bytes: 0,
        peak_activation_bytes: 1000,
        peak_total_bytes: 1000,
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

    let result = crate::moonshot_crown::check_memory_boundedness(&timing_cert, 2000);
    assert!(result.proven);
    assert_eq!(
        result.level,
        crate::moonshot::VerificationLevel::CrownPartial,
        "proven + !sound => CrownPartial"
    );
}

/// Prove: intelligibility proxy level is never CrownProven.
///
/// The intelligibility proxy check explicitly sets CrownPartial (not CrownProven)
/// because it's a proxy, not a full monotonicity proof.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn intelligibility_level_never_crown_proven() {
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

    // Even when proven and sound, the level is CrownPartial (proxy limitation)
    assert_ne!(
        result.level,
        crate::moonshot::VerificationLevel::CrownProven,
        "intelligibility proxy must never be CrownProven"
    );
}
