// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani harnesses for precision-budget invariants.
//!
//! Keeping these in `src/` lets `cargo kani --harness <name>` run them
//! directly without relying on `--tests` discovery.

use crate::ir::ScalarType;
use crate::{
    bootstrap_budget, differential_tolerance, within_differential_budget, PrecisionContract,
    PrecisionTier,
};

/// Proves differential_tolerance is non-negative and finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn kani_precision_tolerance_non_negative_f32_normal() {
    let reference: f32 = kani::any();
    kani::assume(reference.is_finite());
    kani::assume(reference.abs() <= 1e6);

    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let tolerance = differential_tolerance(reference, contract);
    assert!(tolerance >= 0.0, "tolerance must be non-negative");
    assert!(tolerance.is_finite(), "tolerance must stay finite");
}

/// Proves within_differential_budget is reflexive for strict f32 contracts.
#[kani::unwind(1)]
#[kani::proof]
fn kani_precision_within_budget_reflexive_f32_strict() {
    let value: f32 = kani::any();
    kani::assume(value.is_finite());
    kani::assume(value.abs() <= 1e6);

    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        within_differential_budget(value, value, contract),
        "value must be within budget of itself"
    );
}

/// Proves bootstrap budgets are positive, finite, and tier-ordered for f32.
#[kani::unwind(1)]
#[kani::proof]
fn kani_precision_bootstrap_budgets_positive_f32_all_tiers() {
    let (strict_abs, strict_rel) = bootstrap_budget(ScalarType::F32, PrecisionTier::Strict);
    let (normal_abs, normal_rel) = bootstrap_budget(ScalarType::F32, PrecisionTier::Normal);
    let (relaxed_abs, relaxed_rel) = bootstrap_budget(ScalarType::F32, PrecisionTier::Relaxed);

    assert!(strict_abs > 0.0 && strict_abs.is_finite());
    assert!(strict_rel > 0.0 && strict_rel.is_finite());
    assert!(normal_abs > 0.0 && normal_abs.is_finite());
    assert!(normal_rel > 0.0 && normal_rel.is_finite());
    assert!(relaxed_abs > 0.0 && relaxed_abs.is_finite());
    assert!(relaxed_rel > 0.0 && relaxed_rel.is_finite());

    assert!(strict_abs < normal_abs, "strict budget should be tighter");
    assert!(normal_abs < relaxed_abs, "normal budget should be tighter");
}

/// Proves tolerance is sign-symmetric because the formula uses `abs(reference)`.
#[kani::unwind(1)]
#[kani::proof]
fn kani_precision_tolerance_symmetric_in_sign() {
    let selector: u8 = kani::any();
    kani::assume(selector < 6);
    let value = match selector {
        0 => 1.0_f32,
        1 => 10.0_f32,
        2 => 1e-3_f32,
        3 => 100.0_f32,
        4 => 4096.0_f32,
        _ => 1e6_f32,
    };

    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let positive = differential_tolerance(value, contract);
    let negative = differential_tolerance(-value, contract);
    assert!(positive == negative, "tolerance must be sign-symmetric");
}
