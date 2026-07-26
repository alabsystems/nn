// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `status_smt.rs` — SMT verification status types.

use super::*;

// ---------------------------------------------------------------------------
// SmtEncodingKind — serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_smt_encoding_kind_exact_serde_roundtrip() {
    let kind = SmtEncodingKind::Exact;
    let json = serde_json::to_string(&kind).unwrap();
    assert_eq!(json, "\"exact\"");
    let back: SmtEncodingKind = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtEncodingKind::Exact);
}

#[test]
fn test_smt_encoding_kind_uf_approx_serde_roundtrip() {
    let kind = SmtEncodingKind::UfApprox;
    let json = serde_json::to_string(&kind).unwrap();
    assert_eq!(json, "\"uf_approx\"");
    let back: SmtEncodingKind = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtEncodingKind::UfApprox);
}

// ---------------------------------------------------------------------------
// SmtOutcome — serde round-trip for all variants
// ---------------------------------------------------------------------------

#[test]
fn test_smt_outcome_proven_serde() {
    let outcome = SmtOutcome::Proven;
    let json = serde_json::to_string(&outcome).unwrap();
    assert_eq!(json, "\"proven\"");
    let back: SmtOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtOutcome::Proven);
}

#[test]
fn test_smt_outcome_counterexample_serde() {
    let outcome = SmtOutcome::Counterexample;
    let json = serde_json::to_string(&outcome).unwrap();
    assert_eq!(json, "\"counterexample\"");
    let back: SmtOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtOutcome::Counterexample);
}

#[test]
fn test_smt_outcome_unknown_serde() {
    let outcome = SmtOutcome::Unknown;
    let json = serde_json::to_string(&outcome).unwrap();
    assert_eq!(json, "\"unknown\"");
    let back: SmtOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtOutcome::Unknown);
}

#[test]
fn test_smt_outcome_unexecuted_serde() {
    let outcome = SmtOutcome::Unexecuted;
    let json = serde_json::to_string(&outcome).unwrap();
    assert_eq!(json, "\"unexecuted\"");
    let back: SmtOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtOutcome::Unexecuted);
}

#[test]
fn test_smt_outcome_execution_failed_serde() {
    let outcome = SmtOutcome::ExecutionFailed;
    let json = serde_json::to_string(&outcome).unwrap();
    assert_eq!(json, "\"execution_failed\"");
    let back: SmtOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(back, SmtOutcome::ExecutionFailed);
}

// ---------------------------------------------------------------------------
// BoundsSource — serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_bounds_source_analytical_serde() {
    let source = BoundsSource::Analytical;
    let json = serde_json::to_string(&source).unwrap();
    assert_eq!(json, "\"analytical\"");
    let back: BoundsSource = serde_json::from_str(&json).unwrap();
    assert_eq!(back, BoundsSource::Analytical);
}

#[test]
fn test_bounds_source_heuristic_serde() {
    let source = BoundsSource::Heuristic;
    let json = serde_json::to_string(&source).unwrap();
    assert_eq!(json, "\"heuristic\"");
    let back: BoundsSource = serde_json::from_str(&json).unwrap();
    assert_eq!(back, BoundsSource::Heuristic);
}

#[test]
fn test_bounds_source_caller_provided_serde() {
    let source = BoundsSource::CallerProvided;
    let json = serde_json::to_string(&source).unwrap();
    assert_eq!(json, "\"caller_provided\"");
    let back: BoundsSource = serde_json::from_str(&json).unwrap();
    assert_eq!(back, BoundsSource::CallerProvided);
}

// ---------------------------------------------------------------------------
// SmtStatusRecord — constructor and serde
// ---------------------------------------------------------------------------

#[test]
fn test_execution_failed_constructor_fields() {
    let record = SmtStatusRecord::execution_failed("timeout after 30s");
    assert_eq!(record.solver, "ay");
    assert_eq!(record.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(record.property, "pipeline_failure");
    assert_eq!(record.outcome, SmtOutcome::ExecutionFailed);
    assert_eq!(record.detail, Some("timeout after 30s".to_string()));
    assert_eq!(record.bounds_source, BoundsSource::Heuristic);
    assert!(record.expected_bounds.is_none());
}

#[test]
fn test_smt_status_record_serde_roundtrip_with_detail() {
    let record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_finite".to_string(),
        outcome: SmtOutcome::Proven,
        detail: Some("all outputs finite".to_string()),
        bounds_source: BoundsSource::Analytical,
        expected_bounds: Some((-1.0, 1.0)),
        proof_alethe: None,
        proof_verdict: None,
    };
    let json = serde_json::to_string_pretty(&record).unwrap();
    let back: SmtStatusRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.solver, "ay");
    assert_eq!(back.encoding, SmtEncodingKind::Exact);
    assert_eq!(back.property, "output_finite");
    assert_eq!(back.outcome, SmtOutcome::Proven);
    assert_eq!(back.detail, Some("all outputs finite".to_string()));
    assert_eq!(back.bounds_source, BoundsSource::Analytical);
    assert_eq!(back.expected_bounds, Some((-1.0, 1.0)));
}

#[test]
fn test_smt_status_record_serde_without_optional_fields() {
    let record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::UfApprox,
        property: "bound_check".to_string(),
        outcome: SmtOutcome::Unexecuted,
        detail: None,
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    let json = serde_json::to_string(&record).unwrap();
    // detail and expected_bounds should be absent (skip_serializing_if)
    assert!(!json.contains("detail"));
    assert!(!json.contains("expected_bounds"));

    let back: SmtStatusRecord = serde_json::from_str(&json).unwrap();
    assert!(back.detail.is_none());
    assert!(back.expected_bounds.is_none());
}

#[test]
fn test_smt_status_record_bounds_source_defaults_to_heuristic() {
    // When deserializing JSON that lacks `bounds_source`, the default should be Heuristic.
    let json = r#"{
        "solver": "ay",
        "encoding": "exact",
        "property": "output_finite",
        "outcome": "proven"
    }"#;
    let record: SmtStatusRecord = serde_json::from_str(json).unwrap();
    assert_eq!(record.bounds_source, BoundsSource::Heuristic);
    assert!(record.detail.is_none());
    assert!(record.expected_bounds.is_none());
}

#[test]
fn test_smt_status_record_counterexample_with_detail() {
    let record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "bound_check".to_string(),
        outcome: SmtOutcome::Counterexample,
        detail: Some("x = -3.14 violates upper bound".to_string()),
        bounds_source: BoundsSource::CallerProvided,
        expected_bounds: Some((-5.0, 5.0)),
        proof_alethe: None,
        proof_verdict: None,
    };
    let json = serde_json::to_string(&record).unwrap();
    let back: SmtStatusRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.outcome, SmtOutcome::Counterexample);
    assert!(back.detail.as_ref().unwrap().contains("x = -3.14"));
}

// ---------------------------------------------------------------------------
// Debug/Clone derives
// ---------------------------------------------------------------------------

#[test]
fn test_smt_encoding_kind_debug_clone() {
    let kind = SmtEncodingKind::Exact;
    let cloned = kind;
    assert_eq!(kind, cloned);
    let debug = format!("{kind:?}");
    assert!(debug.contains("Exact"));
}

#[test]
fn test_smt_outcome_debug_clone() {
    let outcome = SmtOutcome::Proven;
    let cloned = outcome;
    assert_eq!(outcome, cloned);
    let debug = format!("{outcome:?}");
    assert!(debug.contains("Proven"));
}

#[test]
fn test_bounds_source_debug_clone() {
    let source = BoundsSource::Analytical;
    let cloned = source;
    assert_eq!(source, cloned);
    let debug = format!("{source:?}");
    assert!(debug.contains("Analytical"));
}

#[test]
fn test_smt_status_record_clone() {
    let record = SmtStatusRecord::execution_failed("test");
    let cloned = record.clone();
    assert_eq!(record, cloned);
}
