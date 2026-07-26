// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SpecVerification CROWN fallback provenance tracking (#111 AC1).
//!
//! SpecVerification distinguishes three provenance cases:
//!   1. IBP-only (threshold not exceeded)
//!   2. CROWN-success (escalated and CROWN succeeded)
//!   3. CROWN-failure-fallback (escalated, CROWN failed, IBP result retained)
//!
//! Cases 1 and 2 are covered by tests in `verify_bounds.rs`. This file covers
//! case 3 and serde roundtrip of the `crown_fallback_reason` field.

use nn_dsl::lower::Lowerer;
use nn_verify::{
    Bound, ParamBinding, PropMethod, SpecVerification, VerificationResult, VerifyConfig,
    VerifyRequest,
};

/// Test that SpecVerification correctly tracks propagation provenance when
/// CROWN escalation is triggered on a multi-variable kernel.
///
/// The spec path uses `Verifier::verify_graph()` which has internal CROWN->IBP
/// fallback in NY: when a layer lacks CROWN backward support,
/// NY falls back to IBP internally and returns Ok. This means the
/// `Err(crown_err)` branch in `verify_graph_against_spec_with_config` is
/// effectively unreachable for currently expressible kernels (all supported
/// ops either have CROWN backward or NY absorbs the UnsupportedOp).
///
/// The test uses conditional assertions: if method==Crown, crown_fallback_reason
/// must be None; if method==Ibp (fallback), crown_fallback_reason must be Some.
#[test]
fn test_spec_verification_crown_fallback_provenance() {
    let src = "fn max_xy(x: f32, y: f32) -> f32 { x.max(y) }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    // Tight spec to force Unknown -> escalation to CROWN
    let tight_spec = vec![Bound::new(-4.0, 4.0)];
    // Threshold 0.0 forces CROWN escalation for any non-trivial interval
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let spec_v = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-5.0, 5.0), (-5.0, 5.0)])
        .required_output_bounds(&tight_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should return a result");

    // Conditional provenance check: method and crown_fallback_reason must be consistent
    match spec_v.method {
        PropMethod::Crown => {
            assert!(
                spec_v.crown_fallback_reason.is_none(),
                "CROWN success must have crown_fallback_reason=None, got {:?}",
                spec_v.crown_fallback_reason
            );
        }
        PropMethod::Ibp => {
            // IBP with CROWN escalation attempted means CROWN failed
            assert!(
                spec_v.crown_fallback_reason.is_some(),
                "IBP fallback after CROWN escalation must have crown_fallback_reason=Some"
            );
            let reason = spec_v.crown_fallback_reason.as_ref().unwrap();
            assert!(
                !reason.is_empty(),
                "crown_fallback_reason should contain a non-empty error description"
            );
        }
        _ => panic!("unexpected PropMethod variant: {:?}", spec_v.method),
    }

    // The result should be Unknown (tight spec [-4,4] doesn't contain max(x,y)
    // output range [-5,5])
    assert!(
        matches!(spec_v.result, VerificationResult::Unknown { .. }),
        "expected Unknown with tight spec, got {:?}",
        spec_v.result
    );
}

/// Test serde roundtrip of SpecVerification with crown_fallback_reason=Some.
///
/// Since the CROWN failure path is effectively unreachable via the API (NY
/// absorbs UnsupportedOp errors internally), this test exercises the serialization
/// path by getting a real SpecVerification and injecting crown_fallback_reason
/// via JSON manipulation.
#[test]
fn test_spec_verification_crown_fallback_serde_roundtrip() {
    let src = "fn max_xy(x: f32, y: f32) -> f32 { x.max(y) }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let spec = vec![Bound::new(-10.0, 10.0)];

    let spec_v = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-5.0, 5.0), (-5.0, 5.0)])
        .required_output_bounds(&spec)
        .verify_spec()
        .expect("spec verification should return a result");

    // Roundtrip the real result
    let json = serde_json::to_string(&spec_v).expect("serialize");
    let roundtripped: SpecVerification =
        serde_json::from_str(&json).expect("deserialize roundtrip");
    assert_eq!(
        roundtripped.crown_fallback_reason,
        spec_v.crown_fallback_reason
    );
    assert_eq!(roundtripped.method, spec_v.method);

    // Now inject crown_fallback_reason via JSON manipulation to test the Some path
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse as Value");
    value["method"] = serde_json::json!("IBP");
    value["crown_fallback_reason"] =
        serde_json::json!("CROWN backward not supported for MaxBinaryLayer");
    let injected_json = serde_json::to_string(&value).expect("re-serialize");

    let injected: SpecVerification =
        serde_json::from_str(&injected_json).expect("deserialize injected");
    assert_eq!(injected.method, PropMethod::Ibp);
    assert_eq!(
        injected.crown_fallback_reason,
        Some("CROWN backward not supported for MaxBinaryLayer".to_string())
    );
}
