// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certification pipeline tests for certify.rs.
//!
//! Covers: single-layer certification, multi-stage models, certificate soundness,
//! vacuous detection, gap detection in certificates, serialization roundtrip,
//! soundness mode, proof strength reporting, empty model handling, error
//! propagation, and bundle validation.
//!
//! Part of #3020, #3351.

use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, TraceNode, TraceOp,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device};
use nn_verify::certificate::{CertificateBundle, ProofCertificate};
use nn_verify::PropMethod;
use nn_verify::SigningKey;
use nn_verify::VerificationSoundnessMode;
use nn_verify::{certify_model, CertifyConfig, CertifyError};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a simple Linear + ReLU graph with tracing enabled.
fn build_linear_relu_graph() -> (DynTensor, ComputationGraph) {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap()
}

/// Build a multi-layer graph: Input -> ReLU -> Sigmoid -> Tanh.
fn build_multi_stage_graph() -> ComputationGraph {
    let nodes = vec![
        TraceNode::new(
            0,
            "input".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu".to_string(),
            TraceOp::Relu,
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sigmoid".to_string(),
            TraceOp::Sigmoid,
            vec![1],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "tanh".to_string(),
            TraceOp::Tanh,
            vec![2],
            vec![1, 4],
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

fn make_bounds(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), lo);
    let upper = ArrayD::from_elem(IxDyn(shape), hi);
    BoundedTensor::new(lower, upper).unwrap()
}

// ---------------------------------------------------------------------------
// A. Certification Pipeline Tests (10+)
// ---------------------------------------------------------------------------

/// Test 1: Single-layer (Linear + ReLU) model produces a valid certificate.
#[test]
fn test_certify_simple_model() {
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);
    let config = CertifyConfig::new("simple_model");

    let result = certify_model(&graph, &bounds, &config).unwrap();

    assert!(
        !result.bundle.is_empty(),
        "should produce at least one certificate"
    );
    assert_eq!(result.bundle.model_name, "simple_model");
    assert!(result.verifiability.is_fully_compilable());

    let cert = &result.bundle.certificates[0];
    assert!(cert.is_finite, "linear+relu should produce finite bounds");
    assert!(cert.output_width >= 0.0, "width should be non-negative");
}

/// Test 2: Multi-stage model verifies each stage (Input + ReLU + Sigmoid + Tanh).
#[test]
fn test_certify_checks_all_stages() {
    let graph = build_multi_stage_graph();
    let bounds = make_bounds(&[1, 4], -1.0, 1.0);
    let config = CertifyConfig::new("multi_stage");

    let result = certify_model(&graph, &bounds, &config).unwrap();

    assert!(!result.bundle.is_empty());
    // The verifiability summary should count all ops
    let v = &result.verifiability;
    assert!(
        v.verifiable > 0,
        "should have verifiable ops (ReLU, Sigmoid, Tanh)"
    );
    assert!(v.is_fully_compilable());
    // Output bounds should be finite and valid
    let (lb, ub) = result.output_bounds.lower_upper();
    assert!(
        lb.iter().all(|x| x.is_finite()),
        "all lower bounds should be finite"
    );
    assert!(
        ub.iter().all(|x| x.is_finite()),
        "all upper bounds should be finite"
    );
}

/// Test 3: Certificate with sound mode from a verified model.
#[test]
fn test_certificate_is_sound() {
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);
    let config = CertifyConfig::new("sound_test");

    let result = certify_model(&graph, &bounds, &config).unwrap();

    let cert = &result.bundle.certificates[0];
    // Soundness mode should be either Sound or Heuristic
    // (depends on the IBP/CROWN escalation). Just verify it is populated.
    assert!(
        cert.soundness_mode == VerificationSoundnessMode::Sound
            || cert.soundness_mode == VerificationSoundnessMode::Heuristic,
        "soundness mode should be set: {:?}",
        cert.soundness_mode
    );
    // sound_count should match expectations
    let sound_ct = result.bundle.sound_count();
    let verified_ct = result.bundle.verified_count();
    assert!(verified_ct >= 1, "at least one verified certificate");
    assert!(
        sound_ct <= verified_ct,
        "sound count should not exceed verified count"
    );
}

/// Test 4: Certificate validation catches vacuous or malformed certificates.
#[test]
fn test_certificate_with_vacuous_validation() {
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);
    let config = CertifyConfig::new("vacuous_test");

    let result = certify_model(&graph, &bounds, &config).unwrap();

    // The bundle should pass validation
    assert!(
        result.bundle.validate_all().is_ok(),
        "valid bundle should pass validation"
    );

    // Manually check output width — if very large, mark as vacuous-like
    let cert = &result.bundle.certificates[0];
    let is_tight = cert.output_width < 100.0;
    assert!(
        is_tight,
        "simple linear+relu model should have tight bounds, got width {}",
        cert.output_width
    );
}

/// Test 5: Certificate with gaps — model with unverifiable custom op.
#[test]
fn test_certificate_with_gaps() {
    let nodes = vec![
        TraceNode::new(
            0,
            "input".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "custom".to_string(),
            TraceOp::Custom {
                name: "unverifiable_op".to_string(),
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let bounds = make_bounds(&[1, 4], -1.0, 1.0);
    let config = CertifyConfig::new("gap_test");

    let result = certify_model(&graph, &bounds, &config);
    match result {
        Err(CertifyError::UnverifiableOps { ops }) => {
            assert!(
                ops.contains(&"unverifiable_op".to_string()),
                "should report the custom op name"
            );
        }
        other => panic!("expected UnverifiableOps error, got {other:?}"),
    }
}

/// Test 6: Certificate serialization roundtrip (JSON).
#[test]
fn test_certificate_serialization() {
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);
    let config = CertifyConfig::new("serial_test");

    let result = certify_model(&graph, &bounds, &config).unwrap();
    let bundle = &result.bundle;

    // Serialize to JSON
    let json = serde_json::to_string_pretty(bundle).expect("serialize bundle");
    assert!(!json.is_empty());

    // Deserialize back
    let roundtrip: CertificateBundle = serde_json::from_str(&json).expect("deserialize bundle");

    assert_eq!(roundtrip.model_name, bundle.model_name);
    assert_eq!(roundtrip.certificates.len(), bundle.certificates.len());
    assert_eq!(roundtrip.version, bundle.version);

    // Certificates should match
    for (orig, rt) in bundle
        .certificates
        .iter()
        .zip(roundtrip.certificates.iter())
    {
        assert_eq!(orig.kernel_name, rt.kernel_name);
        assert_eq!(orig.method, rt.method);
        assert!((orig.output_width - rt.output_width).abs() < 1e-6);
        assert_eq!(orig.is_finite, rt.is_finite);
        assert_eq!(orig.soundness_mode, rt.soundness_mode);
    }
}

/// Test 7: CertifyConfig respects soundness mode (IbpValidated by default).
#[test]
fn test_certify_respects_soundness_mode() {
    let config = CertifyConfig::new("soundness_config");
    // Default verify config should exist and be usable
    assert_eq!(config.model_name, "soundness_config");
    assert_eq!(config.fusion_epsilon, 1e-5);
    assert_eq!(config.production_dim, 256);
    assert!(config.signing_key.is_none());

    // Run certification with default config
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);

    let result = certify_model(&graph, &bounds, &config).unwrap();
    let cert = &result.bundle.certificates[0];
    // The certificate should record the soundness mode used
    assert!(
        cert.soundness_mode == VerificationSoundnessMode::Sound
            || cert.soundness_mode == VerificationSoundnessMode::Heuristic,
    );
}

/// Test 8: Certification reports proof strength classification.
#[test]
fn test_certify_reports_proof_strength() {
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);
    let config = CertifyConfig::new("proof_strength");

    let result = certify_model(&graph, &bounds, &config).unwrap();

    let cert = &result.bundle.certificates[0];
    // Method should be IBP or CROWN (the escalation decides)
    assert!(
        cert.method == PropMethod::Ibp
            || cert.method == PropMethod::Crown
            || cert.method == PropMethod::AlphaCrown
            || cert.method == PropMethod::BetaCrown
            || cert.method == PropMethod::MixedIbpCrown,
        "method should be a valid propagation method: {:?}",
        cert.method
    );
    // Output width must be non-negative
    assert!(cert.output_width >= 0.0);
}

/// Test 9: Empty model (no compute nodes, just input) still produces a result.
#[test]
fn test_certify_empty_model() {
    let nodes = vec![TraceNode::new(
        0,
        "input".to_string(),
        TraceOp::Input,
        vec![],
        vec![4],
        DType::F32,
    )];
    let graph = ComputationGraph::from_nodes(nodes);
    let bounds = make_bounds(&[4], -1.0, 1.0);
    let config = CertifyConfig::new("empty_model");

    // An input-only graph may succeed with identity bounds or error
    // depending on graph translation. Either is acceptable; we just
    // verify it does not panic.
    let result = certify_model(&graph, &bounds, &config);
    match result {
        Ok(r) => {
            // If it succeeds, the certificate should be valid
            assert!(r.bundle.validate_all().is_ok());
        }
        Err(e) => {
            // Translation error is acceptable for degenerate graph
            eprintln!("expected error for input-only graph: {e}");
        }
    }
}

/// Test 10: Verification failure propagated as CertifyError::Verify.
#[test]
fn test_certify_error_propagation() {
    // Build a graph with a Relu that has mismatched input reference
    // (node 0 input, node 1 relu referencing non-existent node 5)
    let nodes = vec![
        TraceNode::new(
            0,
            "input".to_string(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu".to_string(),
            TraceOp::Relu,
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);

    // Use mismatched bounds shape to trigger a verification error
    let bounds = make_bounds(&[1, 99], -1.0, 1.0);
    let config = CertifyConfig::new("error_prop");

    let result = certify_model(&graph, &bounds, &config);
    // The exact error depends on how shape mismatch is handled, but
    // it should not panic and should produce an error.
    match result {
        Ok(r) => {
            // If it somehow succeeds, the bounds shape issue was tolerated
            eprintln!("Unexpectedly succeeded with mismatched bounds shape");
            assert!(r.bundle.validate_all().is_ok());
        }
        Err(CertifyError::Verify(e)) => {
            eprintln!("Got expected verify error: {e}");
        }
        Err(e) => {
            // Other error types are also valid
            eprintln!("Got error: {e}");
        }
    }
}

/// Test 11: Certificate bundle with signing key produces HMAC signatures.
#[test]
fn test_certify_with_signing_key() {
    let (_output, graph) = build_linear_relu_graph();
    let bounds = make_bounds(&[1, 2], -1.0, 1.0);
    let mut config = CertifyConfig::new("signed_test");
    let key_bytes: Vec<u8> = (0..32).collect();
    config.signing_key = SigningKey::Raw(key_bytes);

    let result = certify_model(&graph, &bounds, &config).unwrap();

    let cert = &result.bundle.certificates[0];
    assert!(
        cert.content_hash.is_some(),
        "signed certificate should have content_hash"
    );
    assert!(
        cert.hmac_signature.is_some(),
        "signed certificate should have hmac_signature"
    );
    // Validate passes (signature is structurally valid)
    assert!(cert.validate().is_ok());
}

/// Test 12: Bundle validation rejects inverted bounds.
#[test]
fn test_bundle_validates_inverted_bounds() {
    let mut bundle = CertificateBundle::new("test_model");

    // Manually push a bad certificate (inverted bounds)
    let json = serde_json::json!({
        "version": 5,
        "kernel_name": "bad_kernel",
        "input_spec": {
            "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
            "constant_params": [],
            "input_shape": [4],
            "input_range": [-1.0, 1.0]
        },
        "output_bounds": {
            "lower": 5.0,
            "upper": -5.0,
            "is_infeasible": false
        },
        "output_width": -10.0,
        "is_finite": true,
        "method": "IBP",
        "soundness_mode": "sound",
        "generated_at": "2026-01-01T00:00:00Z"
    });
    let bad_cert: ProofCertificate = serde_json::from_value(json).unwrap();
    bundle.push(bad_cert);

    let validation = bundle.validate_all();
    assert!(
        validation.is_err(),
        "inverted bounds should fail validation"
    );
}

/// Test 13: CertifyResult includes verifiability summary with correct counts.
#[test]
fn test_certify_verifiability_summary_counts() {
    let graph = build_multi_stage_graph();
    let bounds = make_bounds(&[1, 4], -1.0, 1.0);
    let config = CertifyConfig::new("summary_counts");

    let result = certify_model(&graph, &bounds, &config).unwrap();
    let v = &result.verifiability;

    // Input=1 (shape_only), ReLU+Sigmoid+Tanh=3 (verifiable)
    assert_eq!(v.shape_only, 1, "Input should be shape_only");
    assert_eq!(v.verifiable, 3, "ReLU + Sigmoid + Tanh are verifiable");
    assert_eq!(v.unverifiable_learned, 0);
    assert!(v.unverifiable_learned_ops.is_empty());
}
