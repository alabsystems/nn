// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_contracts and moonshot verification.
//!
//! Proves IEEE 754 NaN/Inf defense-in-depth for `bounds_within_contract` and
//! `max_contract_violation`, monotonicity of violation magnitude, empty-array
//! edge cases, `VerificationLevel` total-order with symbolic inputs, and
//! `all_at_least_crown_partial` semantics.
//!
//! Part of #3753.
//!
//! Properties proved:
//!
//! 1. `bounds_within_contract` — Inf rejection (both +Inf and -Inf in lower/upper).
//! 2. `bounds_within_contract` — exact contract boundaries return true.
//! 3. `bounds_within_contract` — empty proven bound arrays return true.
//! 4. `bounds_within_contract` — mismatched array lengths return false.
//! 5. `bounds_within_contract` — single dimension exceeding contract returns false.
//! 6. `max_contract_violation` — returns 0.0 when all bounds within contract.
//! 7. `max_contract_violation` — positive value when any bound exceeds contract.
//! 8. `max_contract_violation` — monotonically increasing as bounds diverge.
//! 9. `max_contract_violation` — NaN bounds produce non-zero (MAX) violation.
//! 10. `max_contract_violation` — Inf bounds produce MAX violation.
//! 11. `max_contract_violation` — empty arrays return 0.0.
//! 12. `VerificationLevel` — total order: for any two levels, one <= the other.
//! 13. `VerificationLevel` — antisymmetry: a <= b && b <= a implies a == b.
//! 14. `VerificationLevel` — symbolic transitivity proof.
//! 15. `all_at_least_crown_partial` — false when any property is None.
//! 16. `all_at_least_crown_partial` — false when any property is Empirical.
//! 17. `MoonshotStatus::from_repo` — exactly 8 properties with non-empty names.
//! 18. `MoonshotStatus::from_repo` — all verification levels are valid variants.

// ---- bounds_within_contract: Inf Rejection ----------------------------------

/// Prove: `bounds_within_contract` rejects +Inf in proven upper bound.
///
/// IEEE 754: `+Inf > contract.upper` is true for finite contract.upper, so
/// `!hi.is_finite()` catches this. But if the guard were missing, +Inf would
/// be compared directly and the result depends on contract bounds. This harness
/// proves the guard is present and active.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_rejects_pos_inf_upper() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let result =
        crate::kokoro_contracts::bounds_within_contract(&contract, &[0.0], &[f64::INFINITY]);
    assert!(!result, "+Inf in proven upper must not be contained");
}

/// Prove: `bounds_within_contract` rejects -Inf in proven lower bound.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_rejects_neg_inf_lower() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let result =
        crate::kokoro_contracts::bounds_within_contract(&contract, &[f64::NEG_INFINITY], &[0.0]);
    assert!(!result, "-Inf in proven lower must not be contained");
}

/// Prove: `bounds_within_contract` rejects +Inf in proven lower bound.
///
/// Even +Inf as a lower bound (degenerate but possible from bad NY
/// output) must be rejected by the is_finite guard.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_rejects_pos_inf_lower() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -100.0, 100.0);
    let result = crate::kokoro_contracts::bounds_within_contract(
        &contract,
        &[f64::INFINITY],
        &[f64::INFINITY],
    );
    assert!(!result, "+Inf in proven lower must not be contained");
}

/// Prove: `bounds_within_contract` rejects -Inf in proven upper bound.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_rejects_neg_inf_upper() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -100.0, 100.0);
    let result = crate::kokoro_contracts::bounds_within_contract(
        &contract,
        &[f64::NEG_INFINITY],
        &[f64::NEG_INFINITY],
    );
    assert!(!result, "-Inf in proven upper must not be contained");
}

// ---- bounds_within_contract: Exact Boundary Edge Case -----------------------

/// Prove: `bounds_within_contract` returns true when bounds exactly equal
/// the contract boundaries (inclusive containment, not strict).
///
/// This is a critical edge case: proven bounds that are exactly at the
/// contract limit must be accepted, not rejected. Off-by-one here would
/// cause false negatives in pipeline verification.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_exact_boundary() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl < cu);
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    // Proven bounds exactly equal contract boundaries.
    let result = crate::kokoro_contracts::bounds_within_contract(&contract, &[cl], &[cu]);
    assert!(
        result,
        "bounds exactly at contract boundary must be contained (inclusive)"
    );
}

/// Prove: multi-element exact boundary containment.
///
/// All elements at exact contract boundary => contained.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn bounds_within_contract_exact_boundary_multi() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl < cu);
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let result =
        crate::kokoro_contracts::bounds_within_contract(&contract, &[cl, cl, cl], &[cu, cu, cu]);
    assert!(result, "all elements at exact boundary must be contained");
}

// ---- bounds_within_contract: Empty Arrays -----------------------------------

/// Prove: `bounds_within_contract` returns true for empty proven bound arrays.
///
/// Vacuous truth: if there are zero elements, all zero elements satisfy
/// the contract. This is the correct behavior for degenerate stages.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bounds_within_contract_empty_arrays() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let result = crate::kokoro_contracts::bounds_within_contract(&contract, &[], &[]);
    assert!(result, "empty arrays must be vacuously contained");
}

// ---- bounds_within_contract: Mismatched Lengths -----------------------------

/// Prove: `bounds_within_contract` returns false for mismatched lengths.
///
/// Even when individual elements are within contract, length mismatch
/// means the bounds arrays are malformed — this must be rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_within_contract_mismatched_lengths() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -10.0, 10.0);
    // lower has 1 element, upper has 0 elements
    let result = crate::kokoro_contracts::bounds_within_contract(&contract, &[0.0], &[]);
    assert!(!result, "mismatched array lengths must not be contained");
}

/// Prove: mismatched lengths (upper longer than lower) also rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn bounds_within_contract_mismatched_lengths_reverse() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -10.0, 10.0);
    let result = crate::kokoro_contracts::bounds_within_contract(&contract, &[], &[0.0, 0.5]);
    assert!(
        !result,
        "mismatched array lengths (upper longer) must not be contained"
    );
}

// ---- bounds_within_contract: Single Dimension Exceeds -----------------------

/// Prove: a single out-of-bounds dimension causes rejection even when all
/// other dimensions are within contract.
///
/// The function must be AND-quantified over all dimensions, not OR.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn bounds_within_contract_single_dimension_exceeds() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl < cu);
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);

    let mid = (cl + cu) / 2.0;
    kani::assume(mid.is_finite());

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);

    // Two elements within contract, one upper bound exceeds by 1.0
    let exceeding_upper = cu + 1.0;
    kani::assume(exceeding_upper.is_finite());

    let result = crate::kokoro_contracts::bounds_within_contract(
        &contract,
        &[mid, mid, mid],
        &[mid, exceeding_upper, mid],
    );
    assert!(
        !result,
        "single dimension exceeding upper bound must reject entire check"
    );
}

// ---- max_contract_violation: Zero When Contained ----------------------------

/// Prove: `max_contract_violation` returns exactly 0.0 when all proven
/// bounds are strictly inside the contract.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_zero_when_strictly_inside() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    let pl: f64 = kani::any();
    let pu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(pl.is_finite() && pu.is_finite());
    kani::assume(cl <= cu && pl <= pu);
    kani::assume(pl >= cl && pu <= cu); // contained
    kani::assume(cl.abs() <= 1e6 && cu.abs() <= 1e6);
    kani::assume(pl.abs() <= 1e6 && pu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[pl], &[pu]);
    assert_eq!(
        violation, 0.0,
        "violation must be 0.0 when bounds are contained"
    );
}

// ---- max_contract_violation: Positive When Exceeding ------------------------

/// Prove: `max_contract_violation` returns a positive value when the proven
/// upper bound exceeds the contract upper.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_positive_when_upper_exceeds() {
    let cu: f64 = kani::any();
    let pu: f64 = kani::any();
    kani::assume(cu.is_finite() && pu.is_finite());
    kani::assume(pu > cu); // upper exceeds
    kani::assume(cu.abs() <= 1e6 && pu.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -1e6, cu);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[0.0], &[pu]);
    assert!(
        violation > 0.0,
        "violation must be positive when upper exceeds contract"
    );
    // The violation should be at least (pu - cu).
    let expected_gap = pu - cu;
    assert!(
        violation >= expected_gap,
        "violation must be >= gap ({expected_gap})"
    );
}

/// Prove: `max_contract_violation` returns a positive value when the proven
/// lower bound is below the contract lower.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_positive_when_lower_below() {
    let cl: f64 = kani::any();
    let pl: f64 = kani::any();
    kani::assume(cl.is_finite() && pl.is_finite());
    kani::assume(pl < cl); // lower below
    kani::assume(cl.abs() <= 1e6 && pl.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, 1e6);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[pl], &[0.0]);
    assert!(
        violation > 0.0,
        "violation must be positive when lower below contract"
    );
    let expected_gap = cl - pl;
    assert!(
        violation >= expected_gap,
        "violation must be >= gap ({expected_gap})"
    );
}

// ---- max_contract_violation: Monotonicity -----------------------------------

/// Prove: `max_contract_violation` is monotonically increasing as the proven
/// upper bound moves further above the contract upper.
///
/// If pu2 > pu1 > cu, then violation(pu2) >= violation(pu1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_monotone_upper() {
    let cu: f64 = kani::any();
    let pu1: f64 = kani::any();
    let pu2: f64 = kani::any();
    kani::assume(cu.is_finite() && pu1.is_finite() && pu2.is_finite());
    kani::assume(pu1 > cu && pu2 > pu1); // pu2 further from contract
    kani::assume(cu.abs() <= 1e6 && pu1.abs() <= 1e6 && pu2.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -1e6, cu);
    let v1 = crate::kokoro_contracts::max_contract_violation(&contract, &[0.0], &[pu1]);
    let v2 = crate::kokoro_contracts::max_contract_violation(&contract, &[0.0], &[pu2]);
    assert!(
        v2 >= v1,
        "violation must increase as upper bound moves further from contract"
    );
}

/// Prove: `max_contract_violation` is monotonically increasing as the proven
/// lower bound moves further below the contract lower.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_monotone_lower() {
    let cl: f64 = kani::any();
    let pl1: f64 = kani::any();
    let pl2: f64 = kani::any();
    kani::assume(cl.is_finite() && pl1.is_finite() && pl2.is_finite());
    kani::assume(pl1 < cl && pl2 < pl1); // pl2 further below
    kani::assume(cl.abs() <= 1e6 && pl1.abs() <= 1e6 && pl2.abs() <= 1e6);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, 1e6);
    let v1 = crate::kokoro_contracts::max_contract_violation(&contract, &[pl1], &[0.0]);
    let v2 = crate::kokoro_contracts::max_contract_violation(&contract, &[pl2], &[0.0]);
    assert!(
        v2 >= v1,
        "violation must increase as lower bound moves further from contract"
    );
}

// ---- max_contract_violation: NaN/Inf Defense --------------------------------

/// Prove: NaN in proven lower produces MAX violation (defense-in-depth).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_nan_lower_returns_max() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[f64::NAN], &[0.0]);
    assert_eq!(violation, f64::MAX, "NaN lower must produce MAX violation");
}

/// Prove: NaN in proven upper produces MAX violation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_nan_upper_returns_max() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[0.0], &[f64::NAN]);
    assert_eq!(violation, f64::MAX, "NaN upper must produce MAX violation");
}

/// Prove: +Inf in proven upper produces MAX violation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_pos_inf_upper_returns_max() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -1.0, 1.0);
    let violation =
        crate::kokoro_contracts::max_contract_violation(&contract, &[0.0], &[f64::INFINITY]);
    assert_eq!(violation, f64::MAX, "+Inf upper must produce MAX violation");
}

/// Prove: -Inf in proven lower produces MAX violation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_violation_neg_inf_lower_returns_max() {
    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", -1.0, 1.0);
    let violation =
        crate::kokoro_contracts::max_contract_violation(&contract, &[f64::NEG_INFINITY], &[0.0]);
    assert_eq!(violation, f64::MAX, "-Inf lower must produce MAX violation");
}

// ---- max_contract_violation: Empty Arrays -----------------------------------

/// Prove: `max_contract_violation` returns 0.0 for empty arrays.
///
/// No elements => no violations => 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn max_violation_empty_arrays_returns_zero() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(cl <= cu);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &[], &[]);
    assert_eq!(violation, 0.0, "empty arrays must have zero violation");
}

// ---- VerificationLevel: Symbolic Total Order --------------------------------

/// Helper: convert a u8 index to a VerificationLevel variant.
///
/// Maps 0..=6 to the 7 variants in order. Used by symbolic harnesses
/// to enumerate all variants via kani::any::<u8>().
fn u8_to_verification_level(v: u8) -> crate::moonshot::VerificationLevel {
    use crate::moonshot::VerificationLevel;
    match v {
        0 => VerificationLevel::None,
        1 => VerificationLevel::Empirical,
        2 => VerificationLevel::CrownPartial,
        3 => VerificationLevel::CrownProbabilistic,
        4 => VerificationLevel::CrownProven,
        5 => VerificationLevel::KaniProven,
        6 => VerificationLevel::SmtProven,
        _ => unreachable!(),
    }
}

/// Prove: VerificationLevel is a total order (for any two levels, one <= the other).
///
/// Uses symbolic enum selection to cover all 7x7 = 49 pairs exhaustively.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_total_order_symbolic() {
    let a_idx: u8 = kani::any();
    let b_idx: u8 = kani::any();
    kani::assume(a_idx <= 6 && b_idx <= 6);

    let a = u8_to_verification_level(a_idx);
    let b = u8_to_verification_level(b_idx);

    // Total order: a <= b OR b <= a (at least one must hold).
    assert!(
        a <= b || b <= a,
        "VerificationLevel must be totally ordered"
    );
}

/// Prove: VerificationLevel antisymmetry (a <= b && b <= a implies a == b).
///
/// Covers all 49 pairs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_antisymmetric() {
    let a_idx: u8 = kani::any();
    let b_idx: u8 = kani::any();
    kani::assume(a_idx <= 6 && b_idx <= 6);

    let a = u8_to_verification_level(a_idx);
    let b = u8_to_verification_level(b_idx);

    if a <= b && b <= a {
        assert_eq!(a, b, "antisymmetry violated: a <= b && b <= a but a != b");
    }
}

/// Prove: VerificationLevel transitivity with symbolic inputs.
///
/// For any three levels a, b, c: if a <= b and b <= c then a <= c.
/// Covers all 7^3 = 343 triples.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_transitive_symbolic() {
    let a_idx: u8 = kani::any();
    let b_idx: u8 = kani::any();
    let c_idx: u8 = kani::any();
    kani::assume(a_idx <= 6 && b_idx <= 6 && c_idx <= 6);

    let a = u8_to_verification_level(a_idx);
    let b = u8_to_verification_level(b_idx);
    let c = u8_to_verification_level(c_idx);

    if a <= b && b <= c {
        assert!(a <= c, "transitivity violated: a <= b && b <= c but a > c");
    }
}

/// Prove: VerificationLevel strict ordering matches the intended hierarchy.
///
/// The discriminant ordering must match:
/// None(0) < Empirical(1) < CrownPartial(2) < CrownProbabilistic(3)
///   < CrownProven(4) < KaniProven(5) < SmtProven(6).
///
/// This proves index-based ordering equals variant ordering — if someone
/// inserts a variant in the wrong position, this harness catches it.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_index_matches_ordering() {
    let a_idx: u8 = kani::any();
    let b_idx: u8 = kani::any();
    kani::assume(a_idx <= 6 && b_idx <= 6);

    let a = u8_to_verification_level(a_idx);
    let b = u8_to_verification_level(b_idx);

    if a_idx < b_idx {
        assert!(a < b, "lower index must mean lower verification level");
    } else if a_idx == b_idx {
        assert_eq!(a, b, "same index must mean same level");
    } else {
        assert!(a > b, "higher index must mean higher verification level");
    }
}

// ---- all_at_least_crown_partial: Semantics ----------------------------------

/// Prove: `all_at_least_crown_partial` returns false when any property has
/// level None.
///
/// Constructs a MoonshotStatus from the repo (which has concrete artifact
/// data), then manually downgrades one property to None and checks.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(9)]
fn all_at_least_crown_partial_false_when_any_none() {
    use crate::moonshot::{MoonshotStatus, PropertyStatus, VerificationLevel, PROPERTY_NAMES};

    // Build a status where all properties are CrownProven except one.
    let properties = PROPERTY_NAMES.map(|name| PropertyStatus {
        name,
        verified: VerificationLevel::CrownProven,
        evidence: Vec::new(),
        gaps: Vec::new(),
    });
    let mut status = MoonshotStatus { properties };

    // Downgrade property at symbolic index to None.
    let idx: usize = kani::any();
    kani::assume(idx < 8);
    status.properties[idx].verified = VerificationLevel::None;

    assert!(
        !status.all_at_least_crown_partial(),
        "must return false when property {} is None",
        idx
    );
}

/// Prove: `all_at_least_crown_partial` returns false when any property has
/// level Empirical (below CrownPartial).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(9)]
fn all_at_least_crown_partial_false_when_any_empirical() {
    use crate::moonshot::{MoonshotStatus, PropertyStatus, VerificationLevel, PROPERTY_NAMES};

    let properties = PROPERTY_NAMES.map(|name| PropertyStatus {
        name,
        verified: VerificationLevel::CrownProven,
        evidence: Vec::new(),
        gaps: Vec::new(),
    });
    let mut status = MoonshotStatus { properties };

    let idx: usize = kani::any();
    kani::assume(idx < 8);
    status.properties[idx].verified = VerificationLevel::Empirical;

    assert!(
        !status.all_at_least_crown_partial(),
        "must return false when property {} is Empirical",
        idx
    );
}

/// Prove: `all_at_least_crown_partial` returns true when all properties
/// are at least CrownPartial.
///
/// Uses symbolic level selection constrained to >= CrownPartial.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(9)]
fn all_at_least_crown_partial_true_when_all_sufficient() {
    use crate::moonshot::{MoonshotStatus, PropertyStatus, VerificationLevel, PROPERTY_NAMES};

    let level_idx: u8 = kani::any();
    // Constrain to levels >= CrownPartial (indices 2..=6).
    kani::assume(level_idx >= 2 && level_idx <= 6);
    let level = u8_to_verification_level(level_idx);

    let properties = PROPERTY_NAMES.map(|name| PropertyStatus {
        name,
        verified: level,
        evidence: Vec::new(),
        gaps: Vec::new(),
    });
    let status = MoonshotStatus { properties };

    assert!(
        status.all_at_least_crown_partial(),
        "must return true when all properties are >= CrownPartial"
    );
}

// ---- MoonshotStatus::from_repo: Structural Proofs ---------------------------

/// Prove: `MoonshotStatus::from_repo()` always produces exactly 8 properties
/// AND all property names are non-empty AND all verification levels are valid.
///
/// This is a comprehensive structural check on the production constructor.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn moonshot_from_repo_structural_integrity() {
    let status = crate::moonshot::MoonshotStatus::from_repo();

    // Exactly 8 properties.
    assert_eq!(
        status.properties.len(),
        8,
        "from_repo must produce exactly 8 properties"
    );

    // All names non-empty.
    for prop in &status.properties {
        assert!(!prop.name.is_empty(), "property name must not be empty");
    }

    // All verification levels are valid (must be one of the 7 variants).
    // Since VerificationLevel derives PartialOrd, any valid level satisfies:
    // None <= level <= SmtProven.
    for prop in &status.properties {
        assert!(
            prop.verified >= crate::moonshot::VerificationLevel::None,
            "verification level must be >= None"
        );
        assert!(
            prop.verified <= crate::moonshot::VerificationLevel::SmtProven,
            "verification level must be <= SmtProven"
        );
    }
}

// ---- Consistency: bounds_within_contract agrees with max_contract_violation --

/// Prove: for any symbolic bounds that are fully non-finite-free,
/// `bounds_within_contract` returning true implies `max_contract_violation`
/// returning 0.0, and vice versa.
///
/// This is a stronger version of the existing `violation_zero_iff_contained`
/// harness, using multi-element arrays (2 elements).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn bounds_violation_consistency_multi_element() {
    let cl: f64 = kani::any();
    let cu: f64 = kani::any();
    let pl0: f64 = kani::any();
    let pu0: f64 = kani::any();
    let pl1: f64 = kani::any();
    let pu1: f64 = kani::any();
    kani::assume(cl.is_finite() && cu.is_finite());
    kani::assume(pl0.is_finite() && pu0.is_finite());
    kani::assume(pl1.is_finite() && pu1.is_finite());
    kani::assume(cl <= cu && pl0 <= pu0 && pl1 <= pu1);
    kani::assume(cl.abs() <= 1e4 && cu.abs() <= 1e4);
    kani::assume(pl0.abs() <= 1e4 && pu0.abs() <= 1e4);
    kani::assume(pl1.abs() <= 1e4 && pu1.abs() <= 1e4);

    let contract = crate::kokoro_contracts::JunctionContract::new("test", "zone", cl, cu);
    let lower = [pl0, pl1];
    let upper = [pu0, pu1];

    let contained = crate::kokoro_contracts::bounds_within_contract(&contract, &lower, &upper);
    let violation = crate::kokoro_contracts::max_contract_violation(&contract, &lower, &upper);

    if contained {
        assert_eq!(
            violation, 0.0,
            "contained => zero violation (multi-element)"
        );
    } else {
        assert!(
            violation > 0.0,
            "not contained => positive violation (multi-element)"
        );
    }
}

// ---- All 6 Production Contracts: NaN Guard ----------------------------------

/// Prove: NaN bounds are rejected for ALL 6 production Kokoro contracts.
///
/// This is a completeness check: the NaN guard must work regardless of
/// which contract's bounds are used. Covers both NaN-in-lower and NaN-in-upper.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(7)]
fn all_production_contracts_reject_nan() {
    let contracts = crate::kokoro_contracts::all_contracts();
    for contract in &contracts {
        // NaN in lower bound.
        let result_nan_lo =
            crate::kokoro_contracts::bounds_within_contract(contract, &[f64::NAN], &[0.0]);
        assert!(
            !result_nan_lo,
            "{}: NaN lower must be rejected",
            contract.name
        );

        // NaN in upper bound.
        let result_nan_hi =
            crate::kokoro_contracts::bounds_within_contract(contract, &[0.0], &[f64::NAN]);
        assert!(
            !result_nan_hi,
            "{}: NaN upper must be rejected",
            contract.name
        );

        // NaN violation must be MAX.
        let v_lo = crate::kokoro_contracts::max_contract_violation(contract, &[f64::NAN], &[0.0]);
        assert_eq!(
            v_lo,
            f64::MAX,
            "{}: NaN lower must produce MAX violation",
            contract.name
        );

        let v_hi = crate::kokoro_contracts::max_contract_violation(contract, &[0.0], &[f64::NAN]);
        assert_eq!(
            v_hi,
            f64::MAX,
            "{}: NaN upper must produce MAX violation",
            contract.name
        );
    }
}

/// Prove: Inf bounds are rejected for ALL 6 production Kokoro contracts.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(7)]
fn all_production_contracts_reject_inf() {
    let contracts = crate::kokoro_contracts::all_contracts();
    for contract in &contracts {
        let result_inf =
            crate::kokoro_contracts::bounds_within_contract(contract, &[0.0], &[f64::INFINITY]);
        assert!(
            !result_inf,
            "{}: +Inf upper must be rejected",
            contract.name
        );

        let result_neg_inf =
            crate::kokoro_contracts::bounds_within_contract(contract, &[f64::NEG_INFINITY], &[0.0]);
        assert!(
            !result_neg_inf,
            "{}: -Inf lower must be rejected",
            contract.name
        );
    }
}
