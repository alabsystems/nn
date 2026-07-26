// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for `tightest_enclosing_interval` and `derive_fusion_bounds`
//! in `certify.rs`. Previously untested private functions promoted to pub(crate).
//!
//! Part of #3020 proof_coverage phase.

use super::*;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;
use ny_api::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

fn make_layer_bound(
    index: usize,
    layer_type: &str,
    output_bounds: Vec<(f32, f32)>,
) -> LayerBoundRecord {
    LayerBoundRecord {
        layer_index: index,
        layer_type: layer_type.to_string(),
        input_bounds: vec![],
        output_bounds,
        method: PropMethod::Ibp,
        node_name: None,
        input_sources: None,
    }
}

// ---------------------------------------------------------------------------
// tightest_enclosing_interval
// ---------------------------------------------------------------------------

#[test]
fn test_tightest_enclosing_interval_finite_bounds() {
    let records = vec![
        make_layer_bound(0, "Linear", vec![(-2.0, 3.0), (0.0, 1.5)]),
        make_layer_bound(1, "ReLU", vec![(-0.5, 5.0)]),
    ];

    let (lo, hi) = tightest_enclosing_interval(&records).unwrap();
    // Enclosing: min(-2.0, 0.0, -0.5) = -2.0, max(3.0, 1.5, 5.0) = 5.0
    assert!((lo - (-2.0)).abs() < 1e-6);
    assert!((hi - 5.0).abs() < 1e-6);
}

#[test]
fn test_tightest_enclosing_interval_empty_records() {
    assert!(tightest_enclosing_interval(&[]).is_none());
}

#[test]
fn test_tightest_enclosing_interval_empty_output_bounds() {
    let records = vec![make_layer_bound(0, "Linear", vec![])];
    assert!(tightest_enclosing_interval(&records).is_none());
}

#[test]
fn test_tightest_enclosing_interval_all_non_finite() {
    let records = vec![make_layer_bound(
        0,
        "Linear",
        vec![
            (f32::NEG_INFINITY, 1.0),
            (0.0, f32::INFINITY),
            (f32::NAN, 2.0),
        ],
    )];
    // All pairs have at least one non-finite element
    assert!(tightest_enclosing_interval(&records).is_none());
}

#[test]
fn test_tightest_enclosing_interval_mixed_finite_and_non_finite() {
    let records = vec![make_layer_bound(
        0,
        "Linear",
        vec![
            (f32::NEG_INFINITY, 1.0), // skipped: lower non-finite
            (-3.0, 4.0),              // included
            (0.0, f32::INFINITY),     // skipped: upper non-finite
        ],
    )];

    let (lo, hi) = tightest_enclosing_interval(&records).unwrap();
    assert!((lo - (-3.0)).abs() < 1e-6);
    assert!((hi - 4.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// derive_fusion_bounds
// ---------------------------------------------------------------------------

#[test]
fn test_derive_fusion_bounds_with_layer_bounds() {
    let lb = vec![make_layer_bound(0, "Linear", vec![(-2.0, 5.0)])];

    let lower = ArrayD::from_elem(IxDyn(&[2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&Some(lb), &input_bounds);
    assert_eq!(result.len(), 1);
    // Should use layer bounds: (-2.0, 5.0)
    assert!((result[0].0 - (-2.0)).abs() < 1e-6);
    assert!((result[0].1 - 5.0).abs() < 1e-6);
}

#[test]
fn test_derive_fusion_bounds_fallback_to_input_bounds() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 0.5f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    // No layer bounds -> fallback
    let result = derive_fusion_bounds(&None, &input_bounds);
    assert_eq!(result.len(), 1);
    // Fallback: lo.min(-3.0) = -3.0, hi.max(3.0) = 3.0
    assert!((result[0].0 - (-3.0)).abs() < 1e-6);
    assert!((result[0].1 - 3.0).abs() < 1e-6);
}

#[test]
fn test_derive_fusion_bounds_fallback_when_all_non_finite() {
    let lb = vec![make_layer_bound(
        0,
        "Linear",
        vec![(f32::NEG_INFINITY, f32::INFINITY)],
    )];

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    // Layer bounds all non-finite -> tightest_enclosing_interval returns None -> fallback
    let result = derive_fusion_bounds(&Some(lb), &input_bounds);
    assert_eq!(result.len(), 1);
    // Fallback: lo.min(-3.0) = -5.0, hi.max(3.0) = 5.0
    assert!((result[0].0 - (-5.0)).abs() < 1e-6);
    assert!((result[0].1 - 5.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// CertifyConfig signing key integration (#3253)
// ---------------------------------------------------------------------------

#[test]
fn test_certify_config_default_has_no_signing_key() {
    let config = CertifyConfig::new("test");
    assert!(config.signing_key.is_none());
}

#[test]
fn test_certify_config_with_signing_key() {
    use crate::signing_config::SigningKey;

    let mut config = CertifyConfig::new("test_signed");
    let key_bytes: Vec<u8> = (0..32).collect();
    config.signing_key = SigningKey::Raw(key_bytes);
    assert!(!config.signing_key.is_none());
    assert_eq!(config.signing_key.as_bytes().unwrap().len(), 32);
}

// ---------------------------------------------------------------------------
// CertifyConfig builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_certify_config_default_values() {
    let config = CertifyConfig::new("nn_model");
    assert_eq!(config.model_name, "nn_model");
    assert_eq!(config.fusion_epsilon, 1e-5);
    assert_eq!(config.production_dim, 256);
    assert!(config.enrichment.is_none());
    assert!(config.signing_key.is_none());
}

#[test]
fn test_certify_config_with_enrichment() {
    use crate::certificate::CertificateEnrichment;

    let mut config = CertifyConfig::new("enriched_model");
    config.enrichment = Some(CertificateEnrichment {
        source_path: None,
        weight_path: None,
        kani_status_path: None,
        layer_bounds: None,
        verifier_version: Some("NY-0.9.0".to_string()),
        smt_record: None,
    });
    assert!(config.enrichment.is_some());
    assert_eq!(
        config
            .enrichment
            .as_ref()
            .unwrap()
            .verifier_version
            .as_deref(),
        Some("NY-0.9.0")
    );
}

// ---------------------------------------------------------------------------
// tightest_enclosing_interval edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_tightest_enclosing_interval_single_element() {
    let records = vec![make_layer_bound(0, "ReLU", vec![(0.0, 1.0)])];
    let (lo, hi) = tightest_enclosing_interval(&records).unwrap();
    assert!((lo - 0.0).abs() < 1e-6);
    assert!((hi - 1.0).abs() < 1e-6);
}

#[test]
fn test_tightest_enclosing_interval_negative_range() {
    let records = vec![make_layer_bound(0, "Linear", vec![(-100.0, -50.0)])];
    let (lo, hi) = tightest_enclosing_interval(&records).unwrap();
    assert!((lo - (-100.0)).abs() < 1e-6);
    assert!((hi - (-50.0)).abs() < 1e-6);
}

#[test]
fn test_tightest_enclosing_interval_multiple_layers() {
    let records = vec![
        make_layer_bound(0, "Linear", vec![(-1.0, 1.0)]),
        make_layer_bound(1, "ReLU", vec![(0.0, 1.0)]),
        make_layer_bound(2, "Linear", vec![(-10.0, 10.0)]),
    ];
    let (lo, hi) = tightest_enclosing_interval(&records).unwrap();
    // Enclosing: min(-1.0, 0.0, -10.0)=-10.0, max(1.0, 1.0, 10.0)=10.0
    assert!((lo - (-10.0)).abs() < 1e-6);
    assert!((hi - 10.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// derive_fusion_bounds edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_derive_fusion_bounds_wide_input_bounds() {
    // Input bounds wider than fallback floor (-3, 3) should use input range.
    let lower = ArrayD::from_elem(IxDyn(&[4]), -100.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 100.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&None, &input_bounds);
    assert_eq!(result.len(), 1);
    // lo.min(-3.0) = -100.0, hi.max(3.0) = 100.0
    assert!((result[0].0 - (-100.0)).abs() < 1e-6);
    assert!((result[0].1 - 100.0).abs() < 1e-6);
}

#[test]
fn test_derive_fusion_bounds_narrow_input_widens_to_floor() {
    // Input bounds narrower than (-3, 3) should be widened to floor.
    let lower = ArrayD::from_elem(IxDyn(&[2]), -0.1f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 0.1f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&None, &input_bounds);
    assert_eq!(result.len(), 1);
    // lo.min(-3.0) = -3.0, hi.max(3.0) = 3.0
    assert!((result[0].0 - (-3.0)).abs() < 1e-6);
    assert!((result[0].1 - 3.0).abs() < 1e-6);
}

#[test]
fn test_derive_fusion_bounds_layer_bounds_tighter_than_input() {
    // Layer bounds tighter than input bounds → should use layer bounds.
    let lb = vec![make_layer_bound(0, "ReLU", vec![(0.0, 0.5)])];

    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&Some(lb), &input_bounds);
    assert_eq!(result.len(), 1);
    assert!((result[0].0 - 0.0).abs() < 1e-6);
    assert!((result[0].1 - 0.5).abs() < 1e-6);
}

#[test]
fn test_derive_fusion_bounds_empty_layer_bounds_list() {
    // Some(vec![]) — present but empty → tightest_enclosing_interval returns None → fallback.
    let lower = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&Some(vec![]), &input_bounds);
    assert_eq!(result.len(), 1);
    // Fallback: lo.min(-3.0) = -3.0, hi.max(3.0) = 3.0
    assert!((result[0].0 - (-3.0)).abs() < 1e-6);
    assert!((result[0].1 - 3.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// CertifyError tests
// ---------------------------------------------------------------------------

#[test]
fn test_certify_error_unverifiable_ops_display() {
    let err = CertifyError::UnverifiableOps {
        ops: vec!["mystery_op".to_string(), "custom_gate".to_string()],
    };
    let msg = format!("{err}");
    assert!(msg.contains("mystery_op"), "error should list op names");
    assert!(msg.contains("custom_gate"), "error should list all ops");
}

#[test]
fn test_certify_error_verify_display() {
    let inner = VerifyError::UnsupportedOp("bad_op".to_string());
    let err = CertifyError::Verify(inner);
    let msg = format!("{err}");
    assert!(msg.contains("bad_op"), "should show inner error");
}
