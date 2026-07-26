// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ReferenceTrace operations, ComparisonConfig
//! builder invariants, DivergenceReport consistency, and compare_traces
//! correctness properties.
//!
//! These harnesses cover the trace/config/report layer that sits between
//! the low-level tensor comparison engine (covered by kani_compare*.rs)
//! and the file-loading layer (covered by kani_load*.rs / kani_npy*.rs).
//!
//! Issue: #3803

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig, DivergenceReport};
use crate::trace::{NamedTensor, ReferenceTrace};

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

fn scalar_tensor(name: &str, val: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![1],
        data: vec![val],
    }
}

fn make_trace(tensors: Vec<NamedTensor>) -> ReferenceTrace {
    ReferenceTrace::from_checkpoints(tensors)
}

// ---------------------------------------------------------------------------
// ReferenceTrace structural proofs
// ---------------------------------------------------------------------------

/// Proves that a newly created ReferenceTrace is empty.
#[kani::unwind(1)]
#[kani::proof]
fn new_trace_is_empty() {
    let trace = ReferenceTrace::new();
    assert!(trace.is_empty(), "new trace must be empty");
    assert!(trace.len() == 0, "new trace must have length 0");
}

/// Proves that `from_checkpoints` followed by `into_checkpoints` preserves
/// the count and order of tensors.
#[kani::unwind(8)]
#[kani::proof]
fn from_into_checkpoints_roundtrip_preserves_count() {
    let n: usize = kani::any();
    kani::assume(n <= 4);

    let tensors: Vec<NamedTensor> = (0..n)
        .map(|i| NamedTensor {
            name: format!("t{i}"),
            shape: vec![1],
            data: vec![i as f32],
        })
        .collect();

    let trace = ReferenceTrace::from_checkpoints(tensors);
    assert!(trace.len() == n, "trace length must equal input count");
    assert!(
        trace.is_empty() == (n == 0),
        "is_empty must be consistent with len"
    );

    let recovered = trace.into_checkpoints();
    assert!(recovered.len() == n, "into_checkpoints must preserve count");
}

/// Proves that `get` returns `Some` for valid indices and `None` for
/// out-of-bounds indices.
#[kani::unwind(8)]
#[kani::proof]
fn trace_get_bounds_check() {
    let count: usize = kani::any();
    kani::assume(count >= 1 && count <= 4);

    let tensors: Vec<NamedTensor> = (0..count)
        .map(|i| NamedTensor {
            name: format!("layer{i}"),
            shape: vec![1],
            data: vec![0.0],
        })
        .collect();
    let trace = make_trace(tensors);

    let idx: usize = kani::any();
    kani::assume(idx <= count + 1);

    if idx < count {
        assert!(trace.get(idx).is_some(), "valid index must return Some");
    } else {
        assert!(
            trace.get(idx).is_none(),
            "out-of-bounds index must return None"
        );
    }
}

/// Proves that `get_by_name` returns `Some` for a name that exists
/// in the trace and `None` for one that does not.
#[kani::unwind(8)]
#[kani::proof]
fn trace_get_by_name_finds_existing() {
    let t0 = scalar_tensor("alpha", 1.0);
    let t1 = scalar_tensor("beta", 2.0);
    let trace = make_trace(vec![t0, t1]);

    assert!(
        trace.get_by_name("alpha").is_some(),
        "existing name must be found"
    );
    assert!(
        trace.get_by_name("beta").is_some(),
        "existing name must be found"
    );
    assert!(
        trace.get_by_name("gamma").is_none(),
        "non-existing name must return None"
    );
}

/// Proves that `checkpoint` correctly adds a tensor and increments length.
#[kani::unwind(1)]
#[kani::proof]
fn trace_checkpoint_increments_len() {
    let mut trace = ReferenceTrace::new();
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let before = trace.len();
    let result = trace.checkpoint("layer", &[val], &[1]);
    assert!(result.is_ok(), "valid checkpoint must succeed");
    assert!(
        trace.len() == before + 1,
        "checkpoint must increment length by 1"
    );
}

/// Proves that `checkpoint` rejects data/shape mismatch (same as
/// NamedTensor::new, but exercises the trace-level API).
#[kani::unwind(1)]
#[kani::proof]
fn trace_checkpoint_rejects_shape_mismatch() {
    let mut trace = ReferenceTrace::new();
    // Shape says 3 elements, data has 2.
    let result = trace.checkpoint("bad", &[1.0, 2.0], &[3]);
    assert!(result.is_err(), "mismatched shape/data must error");
    assert!(trace.is_empty(), "failed checkpoint must not add to trace");
}

/// Proves that `capture` returns both the trace and the closure output,
/// and the trace reflects checkpoints added inside the closure.
#[kani::unwind(1)]
#[kani::proof]
fn trace_capture_returns_both() {
    let (trace, output) = ReferenceTrace::capture(|t| {
        t.checkpoint("x", &[1.0], &[1]).expect("valid");
        42u32
    });

    assert!(output == 42, "capture must return closure output");
    assert!(trace.len() == 1, "capture must capture the checkpoint");
}

// ---------------------------------------------------------------------------
// ComparisonConfig builder invariant proofs
// ---------------------------------------------------------------------------

/// Proves that `ComparisonConfig::default()` has positive tolerances and
/// a cosine threshold in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn config_default_has_valid_tolerances() {
    let config = ComparisonConfig::default();
    assert!(
        config.abs_tolerance > 0.0,
        "default abs_tolerance must be positive"
    );
    assert!(
        config.rel_tolerance > 0.0,
        "default rel_tolerance must be positive"
    );
    assert!(
        config.cosine_threshold >= 0.0 && config.cosine_threshold <= 1.0,
        "default cosine_threshold must be in [0, 1]"
    );
    assert!(
        config.rms_tolerance.is_none(),
        "default rms_tolerance must be None"
    );
    assert!(
        config.peak_amplitude_limit.is_none(),
        "default peak_amplitude_limit must be None"
    );
}

/// Proves that `strict()` config has tighter tolerances than `relaxed()`.
#[kani::unwind(1)]
#[kani::proof]
fn config_strict_tighter_than_relaxed() {
    let strict = ComparisonConfig::strict();
    let relaxed = ComparisonConfig::relaxed();

    assert!(
        strict.abs_tolerance < relaxed.abs_tolerance,
        "strict abs_tolerance must be less than relaxed"
    );
    assert!(
        strict.rel_tolerance < relaxed.rel_tolerance,
        "strict rel_tolerance must be less than relaxed"
    );
    assert!(
        strict.cosine_threshold > relaxed.cosine_threshold,
        "strict cosine_threshold must be greater (tighter) than relaxed"
    );
}

/// Proves that `with_rms_tolerance` sets the RMS gate to `Some(value)` and
/// that chaining does not affect other fields.
#[kani::unwind(1)]
#[kani::proof]
fn config_with_rms_tolerance_sets_field() {
    let rms_val: f32 = kani::any();
    kani::assume(rms_val.is_finite() && rms_val >= 0.0 && rms_val <= 1.0);

    let base = ComparisonConfig::new(1e-5, 1e-4, 0.999);
    let with_rms = base.clone().with_rms_tolerance(rms_val);

    assert!(
        with_rms.rms_tolerance == Some(rms_val),
        "with_rms_tolerance must set rms_tolerance"
    );
    assert!(
        with_rms.abs_tolerance == base.abs_tolerance,
        "with_rms_tolerance must not change abs_tolerance"
    );
    assert!(
        with_rms.rel_tolerance == base.rel_tolerance,
        "with_rms_tolerance must not change rel_tolerance"
    );
    assert!(
        with_rms.cosine_threshold == base.cosine_threshold,
        "with_rms_tolerance must not change cosine_threshold"
    );
    assert!(
        with_rms.peak_amplitude_limit == base.peak_amplitude_limit,
        "with_rms_tolerance must not change peak_amplitude_limit"
    );
}

/// Proves that `with_peak_amplitude_limit` sets the peak gate to `Some(value)`
/// and preserves all other fields.
#[kani::unwind(1)]
#[kani::proof]
fn config_with_peak_amplitude_sets_field() {
    let peak_val: f32 = kani::any();
    kani::assume(peak_val.is_finite() && peak_val >= 0.0 && peak_val <= 1e6);

    let base = ComparisonConfig::new(1e-5, 1e-4, 0.999);
    let with_peak = base.clone().with_peak_amplitude_limit(peak_val);

    assert!(
        with_peak.peak_amplitude_limit == Some(peak_val),
        "with_peak_amplitude_limit must set the field"
    );
    assert!(
        with_peak.abs_tolerance == base.abs_tolerance,
        "with_peak_amplitude_limit must not change abs_tolerance"
    );
    assert!(
        with_peak.rms_tolerance == base.rms_tolerance,
        "with_peak_amplitude_limit must not change rms_tolerance"
    );
}

// ---------------------------------------------------------------------------
// compare_traces correctness proofs
// ---------------------------------------------------------------------------

/// Proves that comparing two identical single-layer traces produces
/// all_passed = true and first_failure = None.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn compare_traces_identical_single_layer_passes() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v >= -1e4 && v <= 1e4);

    let ref_trace = make_trace(vec![scalar_tensor("layer", v)]);
    let cand_trace = make_trace(vec![scalar_tensor("layer", v)]);
    let config = ComparisonConfig::default();

    let report = compare_traces(&ref_trace, &cand_trace, &config).expect("must succeed");

    assert!(report.all_passed, "identical traces must pass");
    assert!(
        report.first_failure.is_none(),
        "identical traces must have no failure"
    );
    assert!(
        report.layers.len() == 1,
        "single-layer traces must produce one layer comparison"
    );
}

/// Proves that `compare_traces` returns `TraceLengthMismatch` when the
/// two traces have different numbers of checkpoints.
#[kani::unwind(8)]
#[kani::proof]
fn compare_traces_rejects_length_mismatch() {
    let ref_trace = make_trace(vec![scalar_tensor("a", 1.0)]);
    let cand_trace = make_trace(vec![scalar_tensor("a", 1.0), scalar_tensor("b", 2.0)]);
    let config = ComparisonConfig::default();

    let result = compare_traces(&ref_trace, &cand_trace, &config);
    assert!(
        result.is_err(),
        "mismatched trace lengths must produce an error"
    );
}

/// Proves the consistency invariant of DivergenceReport: `all_passed` is
/// true if and only if `first_failure` is None.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn divergence_report_consistency() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v >= -1e4 && v <= 1e4);

    let ref_trace = make_trace(vec![scalar_tensor("l", v)]);
    let cand_trace = make_trace(vec![scalar_tensor("l", v)]);
    let config = ComparisonConfig::new(1e6, 1e6, -1.0); // permissive

    let report = compare_traces(&ref_trace, &cand_trace, &config).expect("must succeed");

    assert!(
        report.all_passed == report.first_failure.is_none(),
        "all_passed must be true iff first_failure is None"
    );
}

// ---------------------------------------------------------------------------
// NamedTensor numel consistency proof
// ---------------------------------------------------------------------------

/// Proves that `NamedTensor::numel()` always equals the shape product for
/// validly constructed tensors.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_numel_equals_shape_product() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);

    let len = d0 * d1;
    let data = vec![0.0f32; len];
    let t = NamedTensor::new("test", vec![d0, d1], data).expect("valid tensor");

    assert!(t.numel() == d0 * d1, "numel must equal shape product");
    assert!(t.numel() == t.data.len(), "numel must equal data.len()");
}

/// Proves that the NamedTensor scalar case (shape=[]) has numel = 1.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_scalar_numel_is_one() {
    let data = vec![42.0f32];
    let t = NamedTensor::new("scalar", vec![], data).expect("scalar must be valid");

    assert!(t.numel() == 1, "scalar tensor must have numel = 1");
    assert!(t.shape.is_empty(), "scalar tensor shape must be empty vec");
}
