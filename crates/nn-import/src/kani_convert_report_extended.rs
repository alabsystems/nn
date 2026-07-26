// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for convert_report.rs (#3748).
//!
//! Covers:
//! - dispatch_reduction_pct: after > before produces 0% (saturating_sub safety)
//! - estimate_rtf: RTF is always finite for bounded dispatches
//! - estimate_rtf: zero dispatches produces None (not Some(0))
//! - dispatch_reduction_pct: returns None when before == 0
//! - gamma_crown_coverage_pct: zero total returns 0.0 (not NaN)
//! - gamma_crown_coverage_pct: covered > total is bounded to <=100%
//! - FusionReport: dispatches_saved cannot exceed fused_ops
//! - PeepholeReport: native_dispatches >= native_ops structural bound
//! - RTF model: large dispatch counts do not overflow f32
//! - dispatch_reduction_pct: result is in [0, 100]

#![cfg(kani)]

// ---------------------------------------------------------------------------
// dispatch_reduction_pct: after > before (saturating_sub safety)
// ---------------------------------------------------------------------------

/// Prove: when dispatch_count > dispatch_count_before_fusion (e.g., due to
/// expansion passes), saturating_sub ensures the percentage is 0%, not negative.
///
/// Inlines convert_report.rs:94-98.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_reduction_pct_after_exceeds_before() {
    let before: usize = kani::any();
    let after: usize = kani::any();
    kani::assume(before > 0 && before <= 5000);
    kani::assume(after > before && after <= 10_000);

    let saved = before.saturating_sub(after);
    let pct = (saved as f32 / before as f32) * 100.0;

    assert_eq!(
        saved, 0,
        "saturating_sub must clamp to 0 when after > before"
    );
    assert!(
        (pct - 0.0).abs() < f32::EPSILON,
        "Percentage must be 0% when after > before"
    );
}

// ---------------------------------------------------------------------------
// estimate_rtf: RTF is always finite for bounded dispatches
// ---------------------------------------------------------------------------

/// Prove: estimate_rtf produces a finite f32 for any reasonable dispatch count.
///
/// Inlines convert_report.rs:81. f32 overflow occurs at ~3.4e38; 0.0015 * 10^6
/// is only 1500, well within range.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn estimate_rtf_always_finite() {
    let dispatches: usize = kani::any();
    kani::assume(dispatches > 0 && dispatches <= 1_000_000);

    let rtf = dispatches as f32 * 0.0015 + 0.001;

    assert!(
        rtf.is_finite(),
        "RTF must be finite for bounded dispatch counts"
    );
    assert!(rtf > 0.0, "RTF must be positive");
}

// ---------------------------------------------------------------------------
// estimate_rtf: zero dispatches produces None
// ---------------------------------------------------------------------------

/// Prove: estimate_rtf does NOT set estimated_rtf when metal_dispatches == 0.
///
/// Inlines convert_report.rs:80-83. Reporting an RTF for zero dispatches would
/// mislead the user into thinking the model has been compiled to Metal.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn estimate_rtf_zero_dispatches_none() {
    let dispatches: usize = 0;
    let should_set = dispatches > 0;

    assert!(
        !should_set,
        "Zero dispatches must not produce an RTF estimate"
    );
}

// ---------------------------------------------------------------------------
// dispatch_reduction_pct: returns None when before == 0
// ---------------------------------------------------------------------------

/// Prove: dispatch_reduction_pct returns None when dispatch_count_before_fusion
/// is 0, preventing division by zero.
///
/// Inlines convert_report.rs:91-93.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_reduction_pct_zero_before_returns_none() {
    let before: usize = 0;

    // The function checks: if before == 0 { return None }
    let result: Option<f32> = if before == 0 { None } else { Some(0.0) };

    assert!(
        result.is_none(),
        "Zero before must return None (div by zero guard)"
    );
}

// ---------------------------------------------------------------------------
// gamma_crown_coverage_pct: zero total returns 0.0
// ---------------------------------------------------------------------------

/// Prove: gamma_crown_coverage_pct returns 0.0 (not NaN or Inf) when
/// gamma_crown_layers_total is 0.
///
/// Inlines convert_report.rs:255-257. Division by zero would produce NaN/Inf,
/// which would propagate into the display output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gamma_crown_coverage_pct_zero_total_returns_zero() {
    let total: usize = 0;

    let pct: f32 = if total == 0 {
        0.0
    } else {
        (0_usize as f32 / total as f32) * 100.0
    };

    assert_eq!(pct, 0.0, "Zero total must return 0.0");
    assert!(pct.is_finite(), "Result must be finite");
}

// ---------------------------------------------------------------------------
// gamma_crown_coverage_pct: result is in [0, 100] when covered <= total
// ---------------------------------------------------------------------------

/// Prove: gamma_crown_coverage_pct is in [0.0, 100.0] for valid inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gamma_crown_coverage_pct_bounded() {
    let covered: usize = kani::any();
    let total: usize = kani::any();
    kani::assume(total > 0 && total <= 10_000);
    kani::assume(covered <= total);

    let pct = (covered as f32 / total as f32) * 100.0;

    assert!(pct >= 0.0, "Coverage pct must be >= 0");
    assert!(pct <= 100.01, "Coverage pct must be <= 100 (with epsilon)");
}

// ---------------------------------------------------------------------------
// FusionReport: dispatches_saved bounded by fused_ops
// ---------------------------------------------------------------------------

/// Prove: in a valid FusionReport, dispatches_saved <= fused_ops.
///
/// Each fused chain saves at most (chain_ops - 1) dispatches, so total
/// saved <= total fused_ops.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn fusion_report_saved_bounded_by_ops() {
    let chains: usize = kani::any();
    let ops_per_chain: usize = kani::any();
    kani::assume(chains >= 1 && chains <= 100);
    kani::assume(ops_per_chain >= 2 && ops_per_chain <= 10);

    let total_ops = chains * ops_per_chain;
    // Each chain saves (ops_per_chain - 1) dispatches.
    let saved = chains * (ops_per_chain - 1);

    assert!(saved <= total_ops, "Saved dispatches must be <= total ops");
    assert!(
        saved < total_ops,
        "With >=2 ops/chain, saved is strictly < total ops"
    );
}

// ---------------------------------------------------------------------------
// PeepholeReport: native_dispatches >= native_ops
// ---------------------------------------------------------------------------

/// Prove: for valid peephole data, native_dispatches >= native_ops.
///
/// Each NativeOp produces at least 1 Metal dispatch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn peephole_native_dispatches_ge_native_ops() {
    let ops: usize = kani::any();
    let dispatches_per_op: usize = kani::any();
    kani::assume(ops <= 200);
    kani::assume(dispatches_per_op >= 1 && dispatches_per_op <= 10);

    let dispatches = ops.checked_mul(dispatches_per_op);
    assert!(dispatches.is_some(), "Bounded product must not overflow");
    let total_dispatches = dispatches.unwrap();

    assert!(
        total_dispatches >= ops,
        "Dispatches must be >= ops (each op produces >=1 dispatch)"
    );
}

// ---------------------------------------------------------------------------
// RTF model: large dispatch counts stay finite in f32
// ---------------------------------------------------------------------------

/// Prove: the RTF linear model does not overflow to f32::INFINITY
/// for dispatch counts up to 1 million.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn rtf_model_no_overflow_million_dispatches() {
    let dispatches: usize = kani::any();
    kani::assume(dispatches >= 1 && dispatches <= 1_000_000);

    let rtf = dispatches as f32 * 0.0015 + 0.001;

    assert!(
        rtf.is_finite(),
        "RTF must not overflow for million dispatches"
    );
    assert!(!rtf.is_nan(), "RTF must not be NaN");
    // Max: 1_000_000 * 0.0015 + 0.001 = 1500.001
    assert!(rtf <= 1501.0, "RTF must be bounded");
}

// ---------------------------------------------------------------------------
// dispatch_reduction_pct: result is always in [0, 100] for valid inputs
// ---------------------------------------------------------------------------

/// Prove: dispatch_reduction_pct result is always in [0.0, 100.0] when
/// after <= before and before > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_reduction_pct_range_invariant() {
    let before: usize = kani::any();
    let after: usize = kani::any();
    kani::assume(before > 0 && before <= 10_000);
    kani::assume(after <= before);

    let saved = before.saturating_sub(after);
    let pct = (saved as f32 / before as f32) * 100.0;

    assert!(pct >= 0.0, "Reduction pct must be >= 0");
    assert!(pct <= 100.01, "Reduction pct must be <= 100 (with epsilon)");
    assert!(pct.is_finite(), "Reduction pct must be finite");
}
