// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Property-based verification tests for the precision module.
//!
//! These tests verify the mathematical invariants that Kani harnesses would
//! prove exhaustively. Each test systematically checks a property across
//! boundary values and representative samples of the input domain.
//!
//! Properties verified:
//! 1. differential_tolerance is always non-negative for valid contracts
//! 2. within_differential_budget is reflexive (x is within budget of itself)
//! 3. bootstrap_budget produces positive budgets for all dtype/tier combinations
//! 4. tolerance grows monotonically with |reference|
//! 5. within_differential_budget is symmetric around the reference
//! 6. PrecisionContract fields are consistent with tier

use nn_dsl::ir::ScalarType;
use nn_dsl::{
    bootstrap_budget, differential_tolerance, within_differential_budget, PrecisionContract,
    PrecisionTier,
};

// ======================== Constants ========================

const ALL_TIERS: [PrecisionTier; 3] = [
    PrecisionTier::Strict,
    PrecisionTier::Normal,
    PrecisionTier::Relaxed,
];

const ALL_DTYPES: [ScalarType; 2] = [ScalarType::F32, ScalarType::F16];

/// Representative reference values spanning the practical ML domain.
const REFERENCE_VALUES: [f32; 15] = [
    -1e6,
    -1e4,
    -1e2,
    -10.0,
    -1.0,
    -1e-6,
    0.0,
    1e-6,
    1.0,
    10.0,
    1e2,
    1e4,
    1e6,
    f32::MIN_POSITIVE,
    f32::MAX / 2.0,
];

// ======================== Property 1: tolerance non-negative ========================

#[test]
fn property_tolerance_non_negative_all_contracts() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            for &reference in &REFERENCE_VALUES {
                let tol = differential_tolerance(reference, contract);
                assert!(
                    tol >= 0.0,
                    "tolerance must be >= 0 for tier={tier:?}, dtype={dtype:?}, ref={reference}; got {tol}"
                );
                assert!(
                    tol.is_finite(),
                    "tolerance must be finite for tier={tier:?}, dtype={dtype:?}, ref={reference}; got {tol}"
                );
            }
        }
    }
}

#[test]
fn property_tolerance_non_negative_negative_refs() {
    // Specifically check that negative references don't produce negative tolerance
    // (tolerance uses reference.abs() internally)
    for tier in ALL_TIERS {
        let contract = PrecisionContract::bootstrap(tier, ScalarType::F32);
        for ref_val in [-1e8_f32, -1e4, -100.0, -1.0, -1e-10] {
            let tol = differential_tolerance(ref_val, contract);
            assert!(
                tol >= 0.0,
                "tolerance({ref_val}) = {tol} must be >= 0 for tier={tier:?}"
            );
        }
    }
}

// ======================== Property 2: reflexivity ========================

#[test]
fn property_within_budget_reflexive_all_contracts() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            for &reference in &REFERENCE_VALUES {
                assert!(
                    within_differential_budget(reference, reference, contract),
                    "x must be within budget of itself: tier={tier:?}, dtype={dtype:?}, ref={reference}"
                );
            }
        }
    }
}

#[test]
fn property_within_budget_reflexive_zero() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            assert!(
                within_differential_budget(0.0, 0.0, contract),
                "0.0 must be within budget of itself for tier={tier:?}, dtype={dtype:?}"
            );
        }
    }
}

// ======================== Property 3: positive budgets ========================

#[test]
fn property_bootstrap_budgets_positive_all_combinations() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let (abs_budget, rel_budget) = bootstrap_budget(dtype, tier);
            assert!(
                abs_budget > 0.0,
                "abs_budget must be > 0 for tier={tier:?}, dtype={dtype:?}; got {abs_budget}"
            );
            assert!(
                rel_budget > 0.0,
                "rel_budget must be > 0 for tier={tier:?}, dtype={dtype:?}; got {rel_budget}"
            );
            assert!(
                abs_budget.is_finite(),
                "abs_budget must be finite for tier={tier:?}, dtype={dtype:?}"
            );
            assert!(
                rel_budget.is_finite(),
                "rel_budget must be finite for tier={tier:?}, dtype={dtype:?}"
            );
        }
    }
}

// ======================== Property 4: monotonicity in |reference| ========================

#[test]
fn property_tolerance_monotone_in_abs_reference() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            // Tolerance should grow as |reference| grows
            let ref_pairs = [
                (0.0_f32, 1.0),
                (1.0, 10.0),
                (10.0, 100.0),
                (100.0, 1000.0),
                (1000.0, 1e6),
            ];
            for (smaller, larger) in ref_pairs {
                let tol_small = differential_tolerance(smaller, contract);
                let tol_large = differential_tolerance(larger, contract);
                assert!(
                    tol_large >= tol_small,
                    "tolerance must grow with |reference|: tol({larger}) = {tol_large} < tol({smaller}) = {tol_small} for tier={tier:?}, dtype={dtype:?}"
                );
            }
        }
    }
}

#[test]
fn property_tolerance_symmetric_in_sign() {
    // tolerance(x) == tolerance(-x) because tolerance uses reference.abs()
    for tier in ALL_TIERS {
        let contract = PrecisionContract::bootstrap(tier, ScalarType::F32);
        for ref_val in [1.0_f32, 10.0, 100.0, 1e4, 1e-3] {
            let tol_pos = differential_tolerance(ref_val, contract);
            let tol_neg = differential_tolerance(-ref_val, contract);
            assert!(
                (tol_pos - tol_neg).abs() < f32::EPSILON,
                "tolerance({ref_val}) = {tol_pos} != tolerance({}) = {tol_neg} for tier={:?}",
                -ref_val,
                tier
            );
        }
    }
}

// ======================== Property 5: budget ordering by tier ========================

#[test]
fn property_strict_tighter_than_normal_tighter_than_relaxed() {
    for dtype in ALL_DTYPES {
        let (strict_abs, _) = bootstrap_budget(dtype, PrecisionTier::Strict);
        let (normal_abs, _) = bootstrap_budget(dtype, PrecisionTier::Normal);
        let (relaxed_abs, _) = bootstrap_budget(dtype, PrecisionTier::Relaxed);

        assert!(
            strict_abs < normal_abs,
            "strict should be tighter than normal for dtype={dtype:?}: {strict_abs} >= {normal_abs}"
        );
        assert!(
            normal_abs < relaxed_abs,
            "normal should be tighter than relaxed for dtype={dtype:?}: {normal_abs} >= {relaxed_abs}"
        );
    }
}

#[test]
fn property_f16_wider_than_f32_same_tier() {
    // f16 has less precision, so budgets should be wider
    for tier in ALL_TIERS {
        let (f32_abs, _) = bootstrap_budget(ScalarType::F32, tier);
        let (f16_abs, _) = bootstrap_budget(ScalarType::F16, tier);

        assert!(
            f16_abs > f32_abs,
            "f16 budget should be wider than f32 for tier={tier:?}: f16={f16_abs} <= f32={f32_abs}"
        );
    }
}

// ======================== Property 6: contract consistency ========================

#[test]
fn property_contract_fast_math_only_relaxed() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            match tier {
                PrecisionTier::Relaxed => {
                    assert!(
                        contract.fast_math,
                        "Relaxed tier should have fast_math=true"
                    );
                }
                _ => {
                    assert!(
                        !contract.fast_math,
                        "{tier:?} tier should have fast_math=false"
                    );
                }
            }
            assert_eq!(
                contract.tier, tier,
                "contract.tier should match bootstrap tier"
            );
        }
    }
}

// ======================== Property 7: budget boundary check ========================

#[test]
fn property_within_budget_rejects_beyond_tolerance() {
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            let reference = 1.0_f32;
            let tol = differential_tolerance(reference, contract);

            // Value well within tolerance should be within budget
            let within = reference + tol * 0.5;
            assert!(
                within_differential_budget(reference, within, contract),
                "value at half tolerance should be within budget for tier={tier:?}, dtype={dtype:?}"
            );

            // Value well beyond tolerance should NOT be within budget
            let well_beyond = reference + tol * 100.0;
            assert!(
                !within_differential_budget(reference, well_beyond, contract),
                "value 100x beyond tolerance should NOT be within budget for tier={tier:?}, dtype={dtype:?}"
            );
        }
    }
}

#[test]
fn property_within_budget_negative_direction() {
    // Budget should work symmetrically for candidate < reference
    for tier in ALL_TIERS {
        let contract = PrecisionContract::bootstrap(tier, ScalarType::F32);
        let reference = 10.0_f32;

        assert!(
            within_differential_budget(reference, reference - 1e-7, contract),
            "small negative diff should be within budget for tier={tier:?}"
        );
    }
}

// ======================== Property 8: tolerance formula correctness ========================

#[test]
fn property_tolerance_equals_abs_plus_rel_times_abs_ref() {
    // Verify: tolerance = abs_budget + rel_budget * |reference|
    for tier in ALL_TIERS {
        for dtype in ALL_DTYPES {
            let contract = PrecisionContract::bootstrap(tier, dtype);
            for &reference in &REFERENCE_VALUES {
                if !reference.is_finite() || reference.abs() > 1e30 {
                    continue; // skip values that could overflow
                }
                let tol = differential_tolerance(reference, contract);
                let expected = contract.differential_abs_budget
                    + contract.differential_rel_budget * reference.abs();
                assert!(
                    (tol - expected).abs() < f32::EPSILON * 10.0,
                    "tolerance({reference}) = {tol} != expected {expected} for tier={tier:?}, dtype={dtype:?}"
                );
            }
        }
    }
}
