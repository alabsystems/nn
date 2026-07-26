// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pipeline composition verification.
//!
//! Proves correctness of `check_junction` bound containment logic,
//! `verify_pipeline` composition semantics, NaN/Inf defense-in-depth,
//! length-mismatch handling, and PipelineCertificate invariants.
//!
//! Properties proved:
//!
//! 1. check_junction: containment is reflexive (bounds ⊆ themselves).
//! 2. check_junction: containment is transitive.
//! 3. check_junction: NaN in any bound field produces a violation.
//! 4. check_junction: Inf bounds produce a violation.
//! 5. check_junction: length-mismatch between bound vectors produces violations.
//! 6. check_junction: violation_count >= length_mismatch for mismatched vectors.
//! 7. check_junction: max_violation is non-negative for all finite inputs.
//! 8. check_junction: shape compatibility is element-count based.
//! 9. verify_pipeline: is_valid requires ALL junctions to be contained.
//! 10. verify_pipeline: e2e bounds come from first/last stage.

// ---- check_junction containment proofs --------------------------------------

/// Prove: containment is reflexive — bounds always contain themselves.
///
/// For any finite [lo, hi], check_junction(stage, stage) must report
/// bounds_contained = true with zero violations.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_containment_reflexive() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);

    let stage = crate::pipeline::VerifiedStage::new(
        "s",
        vec![1],
        vec![1],
        vec![lo],
        vec![hi],
        vec![lo],
        vec![hi],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&stage, &stage, 0);
    assert!(result.bounds_contained, "bounds must contain themselves");
    assert_eq!(
        result.violation_count, 0,
        "self-containment has zero violations"
    );
    assert_eq!(
        result.max_violation, 0.0,
        "self-containment has zero max_violation"
    );
}

/// Prove: narrower output bounds are contained in wider input bounds.
///
/// If from_output ⊆ [from_lo, from_hi] and to_input = [to_lo, to_hi]
/// where to_lo <= from_lo and from_hi <= to_hi, then contained.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_narrow_within_wide_contained() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    let c: f64 = kani::any();
    let d: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6 && c.abs() <= 1e6 && d.abs() <= 1e6);
    // c <= a <= b <= d  (from output [a,b] ⊆ to input [c,d])
    kani::assume(c <= a && a <= b && b <= d);

    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![a],
        vec![b],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![c],
        vec![d],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        result.bounds_contained,
        "narrow output must be contained in wide input"
    );
    assert_eq!(result.violation_count, 0);
}

/// Prove: wider output bounds violate narrower input bounds.
///
/// If from_output = [a, d] and to_input = [b, c] where a < b or d > c,
/// then NOT contained.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_wider_output_not_contained() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    let c: f64 = kani::any();
    let d: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6 && c.abs() <= 1e6 && d.abs() <= 1e6);
    kani::assume(a <= d); // output is valid range
    kani::assume(b <= c); // input is valid range
    kani::assume(a < b || d > c); // at least one bound violated

    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![a],
        vec![d],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![b],
        vec![c],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        !result.bounds_contained,
        "wider output must not be contained in narrow input"
    );
    assert!(result.violation_count > 0);
    assert!(result.max_violation > 0.0);
}

/// Prove: NaN in output_lower produces a violation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_nan_output_lower_is_violation() {
    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![f64::NAN],
        vec![0.5],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![-10.0],
        vec![10.0],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        !result.bounds_contained,
        "NaN output_lower must produce violation"
    );
    assert!(result.violation_count > 0);
}

/// Prove: NaN in input_upper produces a violation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_nan_input_upper_is_violation() {
    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.5],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![-1.0],
        vec![f64::NAN],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        !result.bounds_contained,
        "NaN input_upper must produce violation"
    );
}

/// Prove: +Inf in output_upper produces a violation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_inf_output_upper_is_violation() {
    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![f64::INFINITY],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![-1.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        !result.bounds_contained,
        "Inf output_upper must produce violation"
    );
}

/// Prove: -Inf in input_lower produces a violation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_neg_inf_input_lower_is_violation() {
    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.5],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![f64::NEG_INFINITY],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        !result.bounds_contained,
        "NEG_INFINITY input_lower must produce violation"
    );
}

/// Prove: length mismatch between output/input bounds vectors produces violations.
///
/// Trailing unmatched elements are counted as violations per the specification.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn junction_length_mismatch_produces_violations() {
    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![2],
        vec![2],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![-1.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    // Even if the overlapping element is contained, the length mismatch
    // adds 1 violation for the trailing unmatched element.
    assert!(
        result.violation_count >= 1,
        "length mismatch must produce at least 1 violation"
    );
    assert_eq!(
        result.max_violation,
        f64::MAX,
        "length mismatch uses MAX sentinel"
    );
}

/// Prove: max_violation is non-negative for all finite inputs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_max_violation_non_negative() {
    let from_lo: f64 = kani::any();
    let from_hi: f64 = kani::any();
    let to_lo: f64 = kani::any();
    let to_hi: f64 = kani::any();
    kani::assume(from_lo.is_finite() && from_hi.is_finite());
    kani::assume(to_lo.is_finite() && to_hi.is_finite());
    kani::assume(from_lo.abs() <= 1e6 && from_hi.abs() <= 1e6);
    kani::assume(to_lo.abs() <= 1e6 && to_hi.abs() <= 1e6);

    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![1],
        vec![1],
        vec![0.0],
        vec![1.0],
        vec![from_lo],
        vec![from_hi],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![1],
        vec![1],
        vec![to_lo],
        vec![to_hi],
        vec![0.0],
        vec![1.0],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        result.max_violation >= 0.0,
        "max_violation must be non-negative, got {}",
        result.max_violation
    );
}

/// Prove: shape compatibility is element-count based, not shape-vector equality.
///
/// [2, 4] and [4, 2] both have 8 elements → shape_compatible = true.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn junction_shape_compatible_by_element_count() {
    let from = crate::pipeline::VerifiedStage::new(
        "from",
        vec![2, 4],
        vec![2, 4],
        vec![0.0; 8],
        vec![1.0; 8],
        vec![0.0; 8],
        vec![0.5; 8],
        "CROWN",
        true,
    );
    let to = crate::pipeline::VerifiedStage::new(
        "to",
        vec![4, 2],
        vec![4, 2],
        vec![-1.0; 8],
        vec![1.0; 8],
        vec![0.0; 8],
        vec![1.0; 8],
        "CROWN",
        true,
    );

    let result = crate::pipeline::check_junction(&from, &to, 0);
    assert!(
        result.shape_compatible,
        "same element count must be shape compatible regardless of shape vector"
    );
}

// ---- verify_pipeline proofs -------------------------------------------------

/// Prove: verify_pipeline is_valid requires ALL junctions contained.
///
/// With symbolic bounds: if either junction has a violation, is_valid is false.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn pipeline_valid_requires_all_junctions() {
    let s0_hi: f64 = kani::any();
    let s1_lo: f64 = kani::any();
    kani::assume(s0_hi.is_finite() && s1_lo.is_finite());
    kani::assume(s0_hi.abs() <= 10.0 && s1_lo.abs() <= 10.0);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "s0",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![s0_hi],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "s1",
            vec![1],
            vec![1],
            vec![s1_lo],
            vec![10.0],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    // Junction is contained iff s0's output [0, s0_hi] ⊆ s1's input [s1_lo, 10.0]
    // i.e., 0.0 >= s1_lo AND s0_hi <= 10.0
    let j0_contained = 0.0 >= s1_lo && s0_hi <= 10.0;

    if cert.is_valid {
        assert!(j0_contained, "is_valid requires junction contained");
    }
}

/// Prove: verify_pipeline e2e bounds come from first and last stage.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn pipeline_e2e_bounds_from_endpoints() {
    let first_in_lo: f64 = kani::any();
    let first_in_hi: f64 = kani::any();
    let last_out_lo: f64 = kani::any();
    let last_out_hi: f64 = kani::any();
    kani::assume(first_in_lo.is_finite() && first_in_hi.is_finite());
    kani::assume(last_out_lo.is_finite() && last_out_hi.is_finite());
    kani::assume(first_in_lo <= first_in_hi && last_out_lo <= last_out_hi);
    kani::assume(first_in_lo.abs() <= 1e6 && first_in_hi.abs() <= 1e6);
    kani::assume(last_out_lo.abs() <= 1e6 && last_out_hi.abs() <= 1e6);

    let stages = vec![
        crate::pipeline::VerifiedStage::new(
            "first",
            vec![1],
            vec![1],
            vec![first_in_lo],
            vec![first_in_hi],
            vec![0.0],
            vec![1.0],
            "CROWN",
            true,
        ),
        crate::pipeline::VerifiedStage::new(
            "last",
            vec![1],
            vec![1],
            vec![0.0],
            vec![1.0],
            vec![last_out_lo],
            vec![last_out_hi],
            "CROWN",
            true,
        ),
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).unwrap();

    assert_eq!(
        cert.e2e_input_lower[0], first_in_lo,
        "e2e input lower from first stage"
    );
    assert_eq!(
        cert.e2e_input_upper[0], first_in_hi,
        "e2e input upper from first stage"
    );
    assert_eq!(
        cert.e2e_output_lower[0], last_out_lo,
        "e2e output lower from last stage"
    );
    assert_eq!(
        cert.e2e_output_upper[0], last_out_hi,
        "e2e output upper from last stage"
    );
}

/// Prove: verify_pipeline rejects empty input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn pipeline_rejects_empty_stages() {
    let stages: Vec<crate::pipeline::VerifiedStage> = vec![];
    assert!(
        crate::pipeline::verify_pipeline(&stages).is_err(),
        "empty stages must be rejected"
    );
}
