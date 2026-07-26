// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional tests for verify_trace.rs — NaN fold semantics and is_tight boundaries.
//!
//! Wire from verify_trace.rs with cfg(test) + path attribute mod declaration.

/// Test NaN-propagating fold semantics used by `compute_width` (#3196).
///
/// `BoundedTensor::new()` rejects NaN, so we can't test through the full
/// function. In production, NaN can arrive via `from_parts_unchecked`
/// during CROWN propagation. This test validates the fold logic directly.
///
/// When gamma-api exposes `new_unchecked` via test-utils feature,
/// upgrade this to call `compute_width` on a NaN-bearing BoundedTensor.
#[test]
fn test_compute_width_nan_fold_propagates() {
    // The NaN-propagating fold used in compute_width (lines 132-138).
    let nan_fold = |widths: &[f32]| -> f32 {
        widths.iter().copied().fold(0.0f32, |acc, w| {
            if w.is_nan() || acc.is_nan() {
                f32::NAN
            } else {
                acc.max(w)
            }
        })
    };

    // NaN in the middle: must propagate.
    let widths = [2.0f32, f32::NAN, 1.0];
    assert!(nan_fold(&widths).is_nan());

    // NaN at start: must propagate.
    assert!(nan_fold(&[f32::NAN, 5.0, 3.0]).is_nan());

    // NaN at end: must propagate.
    assert!(nan_fold(&[1.0, 2.0, f32::NAN]).is_nan());

    // All finite: normal max behavior.
    let result = nan_fold(&[1.0, 4.0, 2.0]);
    assert!((result - 4.0).abs() < 1e-6);

    // Empty: returns 0.0 (fold initial value).
    assert!((nan_fold(&[]) - 0.0).abs() < 1e-6);

    // Demonstrate the bug that #3196 fixed: f32::max drops NaN.
    let old_fold = widths.iter().copied().fold(0.0f32, f32::max);
    assert!(
        !old_fold.is_nan(),
        "f32::max silently drops NaN — the pre-fix bug"
    );
}

// ---------------------------------------------------------------------------
// is_tight boundary condition tests (algorithm_audit, handoff from P10)
// ---------------------------------------------------------------------------
//
// is_tight() returns `self.ibp_width < 100.0` (strict less-than).
// These tests call verify_trace() to get a real VerifyTraceResult, then
// override ibp_width to test boundary conditions.
//
// Blocked on NY (#3118) — same as all verify_trace tests.

use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;
use ndarray::{ArrayD, IxDyn};

/// Build a valid VerifyTraceResult from a simple Linear+ReLU model.
fn build_result() -> super::VerifyTraceResult {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    let bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32),
    )
    .unwrap();

    super::verify_trace(&graph, &bounds).unwrap()
}

/// ibp_width = 1000.0: vacuous bounds, NOT tight.
#[test]
fn test_is_tight_vacuous() {
    let mut result = build_result();
    result.ibp_width = 1000.0;
    assert!(!result.is_tight(), "width 1000.0 should not be tight");
}

/// ibp_width = 100.0: at the boundary, NOT tight (strict <).
#[test]
fn test_is_tight_at_boundary() {
    let mut result = build_result();
    result.ibp_width = 100.0;
    assert!(
        !result.is_tight(),
        "width 100.0 should not be tight (strict < 100.0)"
    );
}

/// ibp_width = 99.9: just below boundary, IS tight.
#[test]
fn test_is_tight_just_below() {
    let mut result = build_result();
    result.ibp_width = 99.9;
    assert!(result.is_tight(), "width 99.9 should be tight");
}

/// ibp_width = NaN: NOT tight (IEEE 754: NaN < 100.0 is false).
#[test]
fn test_is_tight_nan() {
    let mut result = build_result();
    result.ibp_width = f32::NAN;
    assert!(
        !result.is_tight(),
        "NaN width should not be tight (IEEE 754)"
    );
}
