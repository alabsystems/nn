// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses — wave 11 for nn-reftest.
//!
//! Covers tolerance arithmetic, statistics computation bounds, error type
//! preservation, comparison dimension validation, NPY header parsing edge
//! cases, trace-level invariants, and config monotonicity properties.
//!
//! Issue: #3822

use crate::compare::{compare_tensors, compare_traces, ComparisonConfig};
use crate::error::ReftestError;
use crate::npy::{extract_bool_value, extract_shape, extract_string_value, parse_npy_header};
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

fn pair_tensor(name: &str, a: f32, b: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![2],
        data: vec![a, b],
    }
}

fn triple_tensor(name: &str, a: f32, b: f32, c: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![3],
        data: vec![a, b, c],
    }
}

fn make_trace(tensors: Vec<NamedTensor>) -> ReferenceTrace {
    ReferenceTrace::from_checkpoints(tensors)
}

fn assume_finite_bounded(v: f32) {
    kani::assume(v.is_finite());
    kani::assume(v >= -1.0e4 && v <= 1.0e4);
}

// ---------------------------------------------------------------------------
// 1. Tolerance monotonicity: widening abs_tolerance never rejects
// ---------------------------------------------------------------------------

/// Proves that if a comparison passes with abs_tolerance = t, it also passes
/// with abs_tolerance = 2*t (wider tolerance is weaker gate).
///
/// This is the fundamental monotonicity property of the comparison engine:
/// relaxing a tolerance never turns a pass into a fail.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn wider_abs_tolerance_never_rejects_what_tighter_accepts() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    let t: f32 = kani::any();
    assume_finite_bounded(r);
    assume_finite_bounded(c);
    kani::assume(t > 0.0 && t <= 1.0e3 && t.is_finite());

    let tight = ComparisonConfig::new(t, 1e6, -1.0);
    let wide = ComparisonConfig::new(2.0 * t, 1e6, -1.0);

    let r_ref = scalar_tensor("ref", r);
    let r_cand = scalar_tensor("cand", c);

    let tight_result = compare_tensors(&r_ref, &r_cand, &tight).expect("must succeed");
    let wide_result = compare_tensors(&r_ref, &r_cand, &wide).expect("must succeed");

    if tight_result.passed {
        assert!(
            wide_result.passed,
            "wider abs_tolerance must not reject what tighter accepts"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Tolerance monotonicity: widening rel_tolerance
// ---------------------------------------------------------------------------

/// Proves that widening rel_tolerance never turns a pass into a fail.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn wider_rel_tolerance_never_rejects_what_tighter_accepts() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    let t: f32 = kani::any();
    assume_finite_bounded(r);
    assume_finite_bounded(c);
    kani::assume(t > 0.0 && t <= 1.0e3 && t.is_finite());

    let tight = ComparisonConfig::new(1e6, t, -1.0);
    let wide = ComparisonConfig::new(1e6, 2.0 * t, -1.0);

    let tight_result =
        compare_tensors(&scalar_tensor("r", r), &scalar_tensor("c", c), &tight).expect("ok");
    let wide_result =
        compare_tensors(&scalar_tensor("r", r), &scalar_tensor("c", c), &wide).expect("ok");

    if tight_result.passed {
        assert!(
            wide_result.passed,
            "wider rel_tolerance must not reject what tighter accepts"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Tolerance monotonicity: lowering cosine_threshold
// ---------------------------------------------------------------------------

/// Proves that lowering cosine_threshold (making it less strict) never
/// turns a pass into a fail.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn lower_cosine_threshold_never_rejects_what_higher_accepts() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    assume_finite_bounded(r);
    assume_finite_bounded(c);

    let high_thresh = ComparisonConfig::new(1e6, 1e6, 0.5);
    let low_thresh = ComparisonConfig::new(1e6, 1e6, -1.0);

    let high_result =
        compare_tensors(&scalar_tensor("r", r), &scalar_tensor("c", c), &high_thresh).expect("ok");
    let low_result =
        compare_tensors(&scalar_tensor("r", r), &scalar_tensor("c", c), &low_thresh).expect("ok");

    if high_result.passed {
        assert!(
            low_result.passed,
            "lower cosine_threshold must not reject what higher accepts"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. max_abs_diff is the true maximum over elements
// ---------------------------------------------------------------------------

/// Proves that for a 3-element tensor, max_abs_diff >= each individual
/// element-wise absolute difference.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn max_abs_diff_dominates_all_elements() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let r2: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    let c2: f32 = kani::any();
    assume_finite_bounded(r0);
    assume_finite_bounded(r1);
    assume_finite_bounded(r2);
    assume_finite_bounded(c0);
    assume_finite_bounded(c1);
    assume_finite_bounded(c2);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let reference = triple_tensor("ref", r0, r1, r2);
    let candidate = triple_tensor("cand", c0, c1, c2);

    let result = compare_tensors(&reference, &candidate, &config).expect("ok");

    let d0 = (r0 - c0).abs();
    let d1 = (r1 - c1).abs();
    let d2 = (r2 - c2).abs();

    assert!(result.max_abs_diff >= d0, "max_abs_diff must >= d0");
    assert!(result.max_abs_diff >= d1, "max_abs_diff must >= d1");
    assert!(result.max_abs_diff >= d2, "max_abs_diff must >= d2");
}

// ---------------------------------------------------------------------------
// 5. mean_abs_diff is the arithmetic mean of per-element diffs
// ---------------------------------------------------------------------------

/// Proves that for a 2-element tensor, mean_abs_diff equals (d0 + d1) / 2
/// within floating-point tolerance.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn mean_abs_diff_is_arithmetic_mean_of_diffs() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    assume_finite_bounded(r0);
    assume_finite_bounded(r1);
    assume_finite_bounded(c0);
    assume_finite_bounded(c1);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &pair_tensor("r", r0, r1),
        &pair_tensor("c", c0, c1),
        &config,
    )
    .expect("ok");

    let d0 = (r0 - c0).abs() as f64;
    let d1 = (r1 - c1).abs() as f64;
    let expected_mean = ((d0 + d1) / 2.0) as f32;

    // The comparison engine computes mean via f64 accumulation then casts to f32.
    let tol = expected_mean.abs() * 1e-5 + 1e-10;
    assert!(
        (result.mean_abs_diff - expected_mean).abs() <= tol,
        "mean_abs_diff must equal arithmetic mean of per-element diffs"
    );
}

// ---------------------------------------------------------------------------
// 6. rms_diff <= max_abs_diff for any tensor
// ---------------------------------------------------------------------------

/// Proves that rms_diff never exceeds max_abs_diff for 2-element tensors.
///
/// RMS = sqrt(mean(d_i^2)). Since mean(d_i^2) <= max(d_i)^2,
/// we have RMS <= max(d_i) = max_abs_diff.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn rms_diff_bounded_by_max_abs_diff() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    assume_finite_bounded(r0);
    assume_finite_bounded(r1);
    assume_finite_bounded(c0);
    assume_finite_bounded(c1);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &pair_tensor("r", r0, r1),
        &pair_tensor("c", c0, c1),
        &config,
    )
    .expect("ok");

    let tol = result.max_abs_diff * 1e-5 + 1e-10;
    assert!(
        result.rms_diff <= result.max_abs_diff + tol,
        "rms_diff must be <= max_abs_diff"
    );
}

// ---------------------------------------------------------------------------
// 7. cosine_similarity is bounded in [-1, 1] for finite non-zero tensors
// ---------------------------------------------------------------------------

/// Proves that cosine similarity is in [-1, 1] for non-zero finite tensors.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn cosine_similarity_bounded_minus_one_to_one() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    assume_finite_bounded(r0);
    assume_finite_bounded(r1);
    assume_finite_bounded(c0);
    assume_finite_bounded(c1);
    // Ensure both vectors are non-zero.
    kani::assume(r0 != 0.0 || r1 != 0.0);
    kani::assume(c0 != 0.0 || c1 != 0.0);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &pair_tensor("r", r0, r1),
        &pair_tensor("c", c0, c1),
        &config,
    )
    .expect("ok");

    // Small tolerance for floating-point round-off in cosine computation.
    let tol = 1e-5;
    assert!(
        result.cosine_similarity >= -1.0 - tol,
        "cosine similarity must be >= -1"
    );
    assert!(
        result.cosine_similarity <= 1.0 + tol,
        "cosine similarity must be <= 1"
    );
}

// ---------------------------------------------------------------------------
// 8. Negated vector has cosine similarity = -1
// ---------------------------------------------------------------------------

/// Proves that comparing a non-zero vector with its negation produces
/// cosine similarity approximately equal to -1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn cosine_similarity_negated_is_minus_one() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v != 0.0 && v >= -1e4 && v <= 1e4);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result =
        compare_tensors(&scalar_tensor("r", v), &scalar_tensor("c", -v), &config).expect("ok");

    let tol = 1e-5;
    assert!(
        (result.cosine_similarity - (-1.0)).abs() <= tol,
        "negated scalar must have cosine similarity = -1"
    );
}

// ---------------------------------------------------------------------------
// 9. peak_amplitude non-negative for finite inputs
// ---------------------------------------------------------------------------

/// Proves that peak_amplitude is always non-negative for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn peak_amplitude_non_negative_for_finite() {
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    assume_finite_bounded(c0);
    assume_finite_bounded(c1);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &pair_tensor("r", 0.0, 0.0),
        &pair_tensor("c", c0, c1),
        &config,
    )
    .expect("ok");

    assert!(
        result.peak_amplitude >= 0.0,
        "peak_amplitude must be non-negative"
    );
}

// ---------------------------------------------------------------------------
// 10. peak_amplitude equals max(|c_i|) for 2-element tensor
// ---------------------------------------------------------------------------

/// Proves that peak_amplitude equals the maximum absolute candidate value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn peak_amplitude_equals_max_candidate_abs() {
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    assume_finite_bounded(c0);
    assume_finite_bounded(c1);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &pair_tensor("r", 0.0, 0.0),
        &pair_tensor("c", c0, c1),
        &config,
    )
    .expect("ok");

    let expected_peak = c0.abs().max(c1.abs());
    assert!(
        result.peak_amplitude == expected_peak,
        "peak_amplitude must equal max(|c_i|)"
    );
}

// ---------------------------------------------------------------------------
// 11. Shape mismatch error preserves the tensor name
// ---------------------------------------------------------------------------

/// Proves that ShapeMismatch error carries the correct tensor name.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn shape_mismatch_preserves_tensor_name() {
    let reference = NamedTensor {
        name: "encoder".to_string(),
        shape: vec![3],
        data: vec![1.0, 2.0, 3.0],
    };
    let candidate = NamedTensor {
        name: "encoder".to_string(),
        shape: vec![2],
        data: vec![1.0, 2.0],
    };

    let config = ComparisonConfig::default();
    let result = compare_tensors(&reference, &candidate, &config);

    match result {
        Err(ReftestError::ShapeMismatch {
            ref name,
            ref expected,
            ref actual,
        }) => {
            assert!(name == "encoder", "error must carry tensor name");
            assert!(*expected == vec![3], "error must carry expected shape");
            assert!(*actual == vec![2], "error must carry actual shape");
        }
        _ => panic!("expected ShapeMismatch error"),
    }
}

// ---------------------------------------------------------------------------
// 12. Empty tensor is rejected
// ---------------------------------------------------------------------------

/// Proves that comparing zero-element tensors returns EmptyTensor error.
#[kani::unwind(8)]
#[kani::proof]
fn empty_tensor_rejected() {
    let reference = NamedTensor {
        name: "empty".to_string(),
        shape: vec![0],
        data: vec![],
    };
    let candidate = NamedTensor {
        name: "empty".to_string(),
        shape: vec![0],
        data: vec![],
    };

    let config = ComparisonConfig::default();
    let result = compare_tensors(&reference, &candidate, &config);
    assert!(result.is_err(), "empty tensors must be rejected");
}

// ---------------------------------------------------------------------------
// 13. NamedTensor::new with 3-D shape validates product correctly
// ---------------------------------------------------------------------------

/// Proves that NamedTensor::new correctly validates 3-D shapes.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_3d_shape_validation() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);
    kani::assume(d2 >= 1 && d2 <= 4);

    let product = d0 * d1 * d2;
    let data = vec![0.0f32; product];
    let result = NamedTensor::new("test3d", vec![d0, d1, d2], data);
    assert!(result.is_ok(), "matching 3-D data must be accepted");

    let t = result.unwrap();
    assert!(t.numel() == product, "numel must equal 3-D product");
}

// ---------------------------------------------------------------------------
// 14. NamedTensor::new rejects off-by-one data length
// ---------------------------------------------------------------------------

/// Proves that NamedTensor::new rejects data that is exactly one element
/// short of the shape product (common off-by-one bug).
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_rejects_off_by_one_short() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 64);

    let data = vec![0.0f32; n - 1];
    let result = NamedTensor::new("obo", vec![n], data);
    assert!(
        result.is_err(),
        "data one element short of shape must be rejected"
    );
}

/// Proves that NamedTensor::new rejects data that is one element too long.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_rejects_off_by_one_long() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 63);

    let data = vec![0.0f32; n + 1];
    let result = NamedTensor::new("obo", vec![n], data);
    assert!(
        result.is_err(),
        "data one element longer than shape must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 15. compare_traces: multi-layer first_failure index is minimal
// ---------------------------------------------------------------------------

/// Proves that for a 2-layer trace where the first layer fails, first_failure
/// index is 0 (not 1).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn compare_traces_first_failure_is_minimal_index() {
    // First layer: huge difference (must fail with tight tolerance).
    // Second layer: identical (must pass).
    let ref_trace = make_trace(vec![scalar_tensor("a", 0.0), scalar_tensor("b", 1.0)]);
    let cand_trace = make_trace(vec![scalar_tensor("a", 100.0), scalar_tensor("b", 1.0)]);
    let config = ComparisonConfig::strict();

    let report = compare_traces(&ref_trace, &cand_trace, &config).expect("ok");

    assert!(!report.all_passed, "trace with divergent layer must fail");
    assert!(
        report.first_failure == Some(0),
        "first_failure must be the earliest failing layer index"
    );
}

// ---------------------------------------------------------------------------
// 16. compare_traces: layers count matches trace length
// ---------------------------------------------------------------------------

/// Proves that the DivergenceReport always has exactly as many layer
/// comparisons as there are checkpoints in the input traces.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn compare_traces_layer_count_matches_trace_length() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    assume_finite_bounded(v0);
    assume_finite_bounded(v1);

    let ref_trace = make_trace(vec![scalar_tensor("a", v0), scalar_tensor("b", v1)]);
    let cand_trace = make_trace(vec![scalar_tensor("a", v0), scalar_tensor("b", v1)]);
    let config = ComparisonConfig::default();

    let report = compare_traces(&ref_trace, &cand_trace, &config).expect("ok");
    assert!(
        report.layers.len() == 2,
        "report must have one LayerComparison per checkpoint"
    );
}

// ---------------------------------------------------------------------------
// 17. NPY header: extract_string_value handles both quote styles
// ---------------------------------------------------------------------------

/// Proves that extract_string_value handles single-quoted values.
#[kani::unwind(1)]
#[kani::proof]
fn npy_extract_string_single_quotes() {
    let header = "'descr': '<f4', 'fortran_order': False, 'shape': (1,)";
    let result = extract_string_value(header, "descr");
    assert!(result.is_some(), "single-quoted descr must be found");
    assert!(result.unwrap() == "<f4", "value must be <f4");
}

/// Proves that extract_string_value handles double-quoted values.
#[kani::unwind(1)]
#[kani::proof]
fn npy_extract_string_double_quotes() {
    let header = "'descr': \"<f4\", 'fortran_order': False, 'shape': (1,)";
    let result = extract_string_value(header, "descr");
    assert!(result.is_some(), "double-quoted descr must be found");
    assert!(result.unwrap() == "<f4", "value must be <f4");
}

// ---------------------------------------------------------------------------
// 18. NPY header: extract_bool_value correctness
// ---------------------------------------------------------------------------

/// Proves extract_bool_value correctly parses True and False.
#[kani::unwind(1)]
#[kani::proof]
fn npy_extract_bool_true_and_false() {
    let h_true = "'fortran_order': True";
    let h_false = "'fortran_order': False";
    let h_missing = "'descr': '<f4'";

    assert!(
        extract_bool_value(h_true, "fortran_order") == Some(true),
        "True must parse as true"
    );
    assert!(
        extract_bool_value(h_false, "fortran_order") == Some(false),
        "False must parse as false"
    );
    assert!(
        extract_bool_value(h_missing, "fortran_order").is_none(),
        "missing key must return None"
    );
}

// ---------------------------------------------------------------------------
// 19. NPY header: extract_shape for scalar, 1-D, 2-D
// ---------------------------------------------------------------------------

/// Proves that extract_shape handles scalar shape `()`.
#[kani::unwind(8)]
#[kani::proof]
fn npy_extract_shape_scalar() {
    let header = "'descr': '<f4', 'shape': ()";
    let shape = extract_shape(header);
    assert!(shape == Some(vec![]), "scalar shape must be empty vec");
}

/// Proves that extract_shape handles 1-D shape `(5,)`.
#[kani::unwind(8)]
#[kani::proof]
fn npy_extract_shape_1d() {
    let header = "'descr': '<f4', 'shape': (5,)";
    let shape = extract_shape(header);
    assert!(shape == Some(vec![5]), "1-D shape must be [5]");
}

/// Proves that extract_shape handles 2-D shape `(3, 4)`.
#[kani::unwind(8)]
#[kani::proof]
fn npy_extract_shape_2d() {
    let header = "'descr': '<f4', 'shape': (3, 4)";
    let shape = extract_shape(header);
    assert!(shape == Some(vec![3, 4]), "2-D shape must be [3, 4]");
}

// ---------------------------------------------------------------------------
// 20. NPY: fortran_order is rejected
// ---------------------------------------------------------------------------

/// Proves that parse_npy_header with fortran_order=True returns the flag
/// and that the caller can reject it.
#[kani::unwind(1)]
#[kani::proof]
fn npy_fortran_order_detected() {
    let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }";
    let result = parse_npy_header(header);
    assert!(result.is_ok(), "header must parse successfully");
    let (_, _, fortran) = result.unwrap();
    assert!(fortran, "fortran_order: True must be detected");
}

// ---------------------------------------------------------------------------
// 21. NPY: integer dtype descriptors parse correctly
// ---------------------------------------------------------------------------

/// Proves that integer dtype descriptors are preserved through header parsing.
#[kani::unwind(8)]
#[kani::proof]
fn npy_integer_dtype_descriptors_preserved() {
    let idx: u8 = kani::any();
    kani::assume(idx < 4);

    let dtype = match idx {
        0 => "<i4",
        1 => "<i8",
        2 => "<i2",
        _ => "<i1",
    };

    let header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': (1,), }}");
    let (parsed, shape, fortran) = parse_npy_header(&header).expect("valid header");

    assert!(parsed == dtype, "integer dtype must be preserved");
    assert!(shape == vec![1], "shape must be [1]");
    assert!(!fortran, "fortran_order must be false");
}

// ---------------------------------------------------------------------------
// 22. ReferenceTrace: names iterator matches checkpoint order
// ---------------------------------------------------------------------------

/// Proves that `names()` returns checkpoint names in insertion order.
#[kani::unwind(1)]
#[kani::proof]
fn trace_names_match_insertion_order() {
    let mut trace = ReferenceTrace::new();
    trace.checkpoint("first", &[1.0], &[1]).expect("ok");
    trace.checkpoint("second", &[2.0], &[1]).expect("ok");
    trace.checkpoint("third", &[3.0], &[1]).expect("ok");

    let names: Vec<&str> = trace.names().collect();
    assert!(names.len() == 3, "must have 3 names");
    assert!(names[0] == "first", "first name must be first");
    assert!(names[1] == "second", "second name must be second");
    assert!(names[2] == "third", "third name must be third");
}

// ---------------------------------------------------------------------------
// 23. DivergenceReport summary: all_passed produces "All N layers passed"
// ---------------------------------------------------------------------------

/// Proves that the summary of an all-passed report contains "passed".
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn divergence_report_summary_contains_passed_when_all_pass() {
    let ref_trace = make_trace(vec![scalar_tensor("x", 1.0)]);
    let cand_trace = make_trace(vec![scalar_tensor("x", 1.0)]);
    let config = ComparisonConfig::new(1e6, 1e6, -1.0);

    let report = compare_traces(&ref_trace, &cand_trace, &config).expect("ok");
    assert!(report.all_passed, "identical traces must pass");

    let summary = report.summary();
    assert!(
        summary.contains("passed"),
        "summary of passed report must contain 'passed'"
    );
}

// ---------------------------------------------------------------------------
// 24. ComparisonConfig: chaining builders preserves all gates
// ---------------------------------------------------------------------------

/// Proves that chaining both `with_rms_tolerance` and `with_peak_amplitude_limit`
/// preserves both gate values.
#[kani::unwind(1)]
#[kani::proof]
fn config_chaining_preserves_both_gates() {
    let rms_val: f32 = kani::any();
    let peak_val: f32 = kani::any();
    kani::assume(rms_val.is_finite() && rms_val >= 0.0 && rms_val <= 100.0);
    kani::assume(peak_val.is_finite() && peak_val >= 0.0 && peak_val <= 1e6);

    let config = ComparisonConfig::new(1e-5, 1e-4, 0.999)
        .with_rms_tolerance(rms_val)
        .with_peak_amplitude_limit(peak_val);

    assert!(
        config.rms_tolerance == Some(rms_val),
        "rms_tolerance must be preserved after chaining"
    );
    assert!(
        config.peak_amplitude_limit == Some(peak_val),
        "peak_amplitude_limit must be preserved after chaining"
    );
    // Core tolerances unchanged.
    assert!(config.abs_tolerance == 1e-5, "abs_tolerance preserved");
    assert!(config.rel_tolerance == 1e-4, "rel_tolerance preserved");
}

// ---------------------------------------------------------------------------
// 25. Both NaN reference and candidate produce infinite divergence
// ---------------------------------------------------------------------------

/// Proves that when both reference and candidate are NaN, infinite
/// divergence is still reported.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn both_nan_produces_infinite_divergence() {
    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &scalar_tensor("r", f32::NAN),
        &scalar_tensor("c", f32::NAN),
        &config,
    )
    .expect("ok");

    assert!(
        result.max_abs_diff == f32::INFINITY,
        "both-NaN must produce INFINITY max_abs_diff"
    );
    assert!(
        result.max_rel_diff == f32::INFINITY,
        "both-NaN must produce INFINITY max_rel_diff"
    );
}

// ---------------------------------------------------------------------------
// 26. max_rel_diff non-negative for finite inputs
// ---------------------------------------------------------------------------

/// Proves that max_rel_diff is always non-negative for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn max_rel_diff_non_negative() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    assume_finite_bounded(r);
    assume_finite_bounded(c);

    let config = ComparisonConfig::new(1e-5, 1e6, -1.0);
    let result =
        compare_tensors(&scalar_tensor("r", r), &scalar_tensor("c", c), &config).expect("ok");

    assert!(
        result.max_rel_diff >= 0.0,
        "max_rel_diff must be non-negative"
    );
}

// ---------------------------------------------------------------------------
// 27. num_elements reflects actual tensor size
// ---------------------------------------------------------------------------

/// Proves that the LayerComparison num_elements field matches the tensor
/// element count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn num_elements_matches_tensor_size() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let r2: f32 = kani::any();
    assume_finite_bounded(r0);
    assume_finite_bounded(r1);
    assume_finite_bounded(r2);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let result = compare_tensors(
        &triple_tensor("r", r0, r1, r2),
        &triple_tensor("c", r0, r1, r2),
        &config,
    )
    .expect("ok");

    assert!(
        result.num_elements == 3,
        "num_elements must equal actual tensor size"
    );
}
