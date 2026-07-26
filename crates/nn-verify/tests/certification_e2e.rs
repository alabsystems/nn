// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end model certification tests for the nn verification pipeline.
//!
//! Tests exercise the certification flow from model construction through
//! NY verification to proof certificate generation. Covers:
//!
//! - P1 Bounded outputs (Linear -> Sigmoid outputs in [0, 1])
//! - P2 Monotone confidence (tighter inputs -> tighter outputs)
//! - P6 Softmax normalization (outputs sum to ~1.0)
//! - P7 Sigmoid boundedness (outputs strictly in (0, 1))
//! - Certificate generation and serialization
//! - Gap detection with synthetic status data
//! - Status file read/write roundtrip
//! - Compose + certify flow (small model -> verify -> certificate)
//!
//! Part of #3942.

mod common;

use ny_propagate::layers::{LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_verify::certificate::{CertificateBundle, ProofCertificate, CERTIFICATE_VERSION};
use nn_verify::gap_detector::{detect_gaps, format_gap_report, kokoro_pipeline_stages};
use nn_verify::{
    BoundedTensor, InputBoundsRecord, KernelVerification, OutputTensorBounds, ParamInputRecord,
    PropMethod, VerifyStatus,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random LCG state advance.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 33) as f32) / (u32::MAX as f32) - 0.5
}

/// Xavier-uniform weight initialization.
fn xavier_weights(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let xavier_scale = (2.0 / (rows + cols) as f32).sqrt();
    let mut w = Array2::zeros((rows, cols));
    let mut rng_state = seed;
    for elem in w.iter_mut() {
        *elem = lcg_next(&mut rng_state) * xavier_scale;
    }
    w
}

/// Build a Linear -> Sigmoid model: output is sigmoid(Wx + b).
fn build_linear_sigmoid(d_in: usize, d_out: usize) -> GraphNetwork {
    let w = xavier_weights(d_out, d_in, 42);
    let b = Array1::zeros(d_out);
    let linear = LinearLayer::new(w, Some(b)).expect("linear");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("sigmoid");
    graph
}

/// Build a Linear -> Softmax model.
fn build_linear_softmax(d_in: usize, d_out: usize) -> GraphNetwork {
    let w = xavier_weights(d_out, d_in, 99);
    let b = Array1::zeros(d_out);
    let linear = LinearLayer::new(w, Some(b)).expect("linear");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "softmax",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["linear".to_string()],
    ));
    graph.set_output("softmax");
    graph
}

/// Build a two-layer MLP: Linear -> ReLU -> Linear -> Sigmoid.
fn build_mlp_sigmoid(d: usize) -> GraphNetwork {
    let w1 = xavier_weights(d, d, 42);
    let b1 = Array1::zeros(d);
    let w2 = xavier_weights(d, d, 137);
    let b2 = Array1::zeros(d);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("linear2")),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["linear2".to_string()],
    ));
    graph.set_output("sigmoid");
    graph
}

fn make_uniform_bounds(shape: &[usize], range: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), -range),
        ArrayD::from_elem(IxDyn(shape), range),
    )
    .expect("valid uniform bounds")
}

fn make_verification(lower: f32, upper: f32) -> KernelVerification {
    let mut v = KernelVerification::new(
        "test_model".to_string(),
        PropMethod::Ibp,
        lower,
        upper,
        upper - lower,
        true,
    );
    v.output_tensor = Some(OutputTensorBounds::new(vec![lower], vec![upper], vec![1]));
    v
}

fn make_input_spec() -> InputBoundsRecord {
    InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[1.0])
}

// ===========================================================================
// P1: Bounded outputs — Linear -> Sigmoid produces outputs in [0, 1]
// ===========================================================================

#[test]
fn test_p1_bounded_outputs_linear_sigmoid_ibp() {
    let d = 8;
    let graph = build_linear_sigmoid(d, d);
    let input = make_uniform_bounds(&[d], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
        // Sigmoid is bounded to (0, 1); IBP should prove this.
        assert!(l >= -1e-6, "sigmoid lower bound should be >= 0 (got {l})");
        assert!(
            u <= 1.0 + 1e-6,
            "sigmoid upper bound should be <= 1 (got {u})"
        );
    }
}

#[test]
fn test_p1_bounded_outputs_linear_sigmoid_crown() {
    let d = 8;
    let graph = build_linear_sigmoid(d, d);
    let input = make_uniform_bounds(&[d], 1.0);

    let (method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    let (lo, hi) = crown_output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l >= -1e-6, "sigmoid lower {l} should be >= 0");
        assert!(u <= 1.0 + 1e-6, "sigmoid upper {u} should be <= 1");
    }

    // CROWN should produce tighter bounds than IBP.
    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    if matches!(method, PropMethod::Crown) {
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let eps = 1e-4;
        for (&cl, &il) in lo.iter().zip(ibp_lo.iter()) {
            assert!(
                cl >= il - eps,
                "CROWN lower {cl} should be >= IBP lower {il}"
            );
        }
        for (&cu, &iu) in hi.iter().zip(ibp_hi.iter()) {
            assert!(
                cu <= iu + eps,
                "CROWN upper {cu} should be <= IBP upper {iu}"
            );
        }
    }
}

// ===========================================================================
// P2: Monotone confidence — tighter inputs produce tighter outputs
// ===========================================================================

#[test]
fn test_p2_monotone_confidence_tighter_inputs_tighter_outputs() {
    let d = 8;
    let graph = build_linear_sigmoid(d, d);

    // Wide input bounds
    let wide_input = make_uniform_bounds(&[d], 2.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");

    // Tight input bounds (subset of wide)
    let tight_input = make_uniform_bounds(&[d], 0.5);
    let tight_output = graph.propagate_ibp(&tight_input).expect("tight IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (tight_lo, tight_hi) = tight_output.lower_upper();
    let eps = 1e-6;

    for i in 0..d {
        // Tight outputs should be contained within wide outputs (soundness).
        assert!(
            tight_lo[i] >= wide_lo[i] - eps,
            "dim {i}: tight lower {} should be >= wide lower {} (containment)",
            tight_lo[i],
            wide_lo[i]
        );
        assert!(
            tight_hi[i] <= wide_hi[i] + eps,
            "dim {i}: tight upper {} should be <= wide upper {} (containment)",
            tight_hi[i],
            wide_hi[i]
        );
    }

    // At least one dimension should have a strictly tighter bound.
    let any_tighter = (0..d).any(|i| {
        let wide_width = wide_hi[i] - wide_lo[i];
        let tight_width = tight_hi[i] - tight_lo[i];
        tight_width < wide_width - eps
    });
    assert!(
        any_tighter,
        "tighter input bounds should produce at least one tighter output dimension"
    );
}

#[test]
fn test_p2_monotone_confidence_three_levels() {
    let d = 4;
    let graph = build_mlp_sigmoid(d);

    let ranges = [3.0, 1.0, 0.3];
    let mut widths: Vec<f32> = Vec::new();

    for &range in &ranges {
        let input = make_uniform_bounds(&[d], range);
        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo, hi) = output.lower_upper();
        let max_width = lo
            .iter()
            .zip(hi.iter())
            .map(|(&l, &u)| u - l)
            .fold(0.0_f32, f32::max);
        widths.push(max_width);
    }

    // Wider inputs should never produce narrower outputs.
    for i in 0..widths.len() - 1 {
        assert!(
            widths[i] >= widths[i + 1] - 1e-6,
            "input range {} (width {}) should produce >= width than input range {} (width {})",
            ranges[i],
            widths[i],
            ranges[i + 1],
            widths[i + 1]
        );
    }
}

// ===========================================================================
// P6: Softmax normalization — outputs sum to ~1.0
// ===========================================================================

#[test]
fn test_p6_softmax_normalization_ibp() {
    let d = 4;
    let graph = build_linear_softmax(d, d);
    let input = make_uniform_bounds(&[d], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // Each softmax output should be in [0, 1].
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l >= -1e-5, "softmax lower bound should be >= 0, got {l}");
        assert!(
            u <= 1.0 + 1e-5,
            "softmax upper bound should be <= 1, got {u}"
        );
    }

    // Sum of upper bounds should be >= 1.0 (soundness: true sum is exactly 1).
    let sum_upper: f32 = hi.iter().copied().sum();
    assert!(
        sum_upper >= 1.0 - 1e-5,
        "sum of softmax upper bounds should be >= 1.0 (got {sum_upper})"
    );

    // Sum of lower bounds should be <= 1.0.
    let sum_lower: f32 = lo.iter().copied().sum();
    assert!(
        sum_lower <= 1.0 + 1e-5,
        "sum of softmax lower bounds should be <= 1.0 (got {sum_lower})"
    );
}

#[test]
fn test_p6_softmax_normalization_crown() {
    let d = 4;
    let graph = build_linear_softmax(d, d);
    let input = make_uniform_bounds(&[d], 1.0);

    let (_method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l >= -1e-5, "softmax lower {l} should be >= 0");
        assert!(u <= 1.0 + 1e-5, "softmax upper {u} should be <= 1");
    }

    // CROWN should give tighter bounds: sum of upper bounds closer to 1.0.
    let sum_upper: f32 = hi.iter().copied().sum();
    assert!(
        sum_upper >= 1.0 - 1e-5,
        "CROWN softmax sum of upper bounds >= 1.0 (got {sum_upper})"
    );
}

// ===========================================================================
// P7: Sigmoid boundedness — strictly in (0, 1)
// ===========================================================================

#[test]
fn test_p7_sigmoid_boundedness_standalone() {
    // Pure sigmoid: any finite input maps to (0, 1).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer),
    ));
    graph.set_output("sigmoid");

    // Wide input range [-10, 10].
    let input = make_uniform_bounds(&[4], 10.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= 0.0, "sigmoid lower should be >= 0, got {l}");
        assert!(u <= 1.0, "sigmoid upper should be <= 1, got {u}");
        // With [-10, 10] input, sigmoid should be close to the full range.
        assert!(l < 0.01, "sigmoid lower should be near 0, got {l}");
        assert!(u > 0.99, "sigmoid upper should be near 1, got {u}");
    }
}

#[test]
fn test_p7_sigmoid_boundedness_narrow_input() {
    // Narrow input range should give narrow sigmoid output.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer),
    ));
    graph.set_output("sigmoid");

    let input = make_uniform_bounds(&[4], 0.1);
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= 0.0, "sigmoid lower should be >= 0, got {l}");
        assert!(u <= 1.0, "sigmoid upper should be <= 1, got {u}");
        // Output width should be small (sigmoid is nearly linear near 0).
        let width = u - l;
        assert!(
            width < 0.1,
            "sigmoid output width should be < 0.1 for narrow input, got {width}"
        );
    }
}

// ===========================================================================
// Certificate generation and serialization
// ===========================================================================

#[test]
fn test_certificate_from_verification_creates_valid() {
    let verification = make_verification(0.0, 1.0);
    let cert = ProofCertificate::from_verification(&verification, make_input_spec());

    assert_eq!(cert.kernel_name, "test_model");
    assert_eq!(cert.version, CERTIFICATE_VERSION);
    assert!(
        cert.validate().is_ok(),
        "generated certificate should be valid"
    );
}

#[test]
fn test_certificate_json_roundtrip() {
    let verification = make_verification(0.0, 1.0);
    let cert = ProofCertificate::from_verification(&verification, make_input_spec())
        .with_verifier_version("NY-test".to_string());

    let json = cert.to_json().expect("serialize to JSON");
    let parsed: ProofCertificate = serde_json::from_str(&json).expect("deserialize from JSON");
    assert_eq!(cert, parsed);
}

#[test]
fn test_certificate_bundle_creation_and_serialization() {
    let v1 = make_verification(0.0, 1.0);
    let v2 = {
        let mut v = make_verification(-0.5, 0.8);
        v.kernel_name = "secondary_model".to_string();
        v
    };

    let bundle = CertificateBundle::new("certification_e2e_test")
        .with_certificate(ProofCertificate::from_verification(&v1, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v2, make_input_spec()));

    assert_eq!(bundle.len(), 2);
    assert_eq!(bundle.model_name, "certification_e2e_test");
    assert!(bundle.all_sound(), "all IBP verifications should be sound");

    // Save/load roundtrip.
    let dir = std::env::temp_dir().join(format!("nn_cert_e2e_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("e2e_test.proof.json");
    let _ = std::fs::remove_file(&path);

    bundle.save(&path).expect("save bundle");
    let loaded = CertificateBundle::load(&path).expect("load bundle");
    assert_eq!(bundle, loaded);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_certificate_signing_roundtrip() {
    let verification = make_verification(0.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&verification, make_input_spec());
    let key: Vec<u8> = (0..32).collect();

    nn_verify::sign_certificate(&mut cert, &key).expect("sign");
    assert!(cert.content_hash.is_some());
    assert!(cert.hmac_signature.is_some());

    nn_verify::verify_signature(&cert, &key).expect("verify passes with correct key");
}

// ===========================================================================
// Gap detection with synthetic status data
// ===========================================================================

#[test]
fn test_gap_detection_empty_status_all_gaps() {
    let status = serde_json::json!({});
    let report = detect_gaps(&status);

    let stages = kokoro_pipeline_stages();
    assert_eq!(
        report.total_gaps,
        stages.len(),
        "empty status should mark all stages as gaps"
    );
    assert_eq!(report.vacuous_count, 0);
}

#[test]
fn test_gap_detection_partial_coverage() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": {
                "status": "verified",
                "method": "IBP",
                "output_width": 100.0,
                "proof_strength": "sound"
            },
            "kokoro_production_text_encoder": {
                "status": "verified",
                "method": "IBP",
                "output_width": 50.0,
                "proof_strength": "sound"
            }
        }
    });

    let report = detect_gaps(&status);
    // 2 stages verified, rest are gaps.
    assert_eq!(report.total_gaps, 6, "should have 6 gaps (8 - 2 verified)");
}

#[test]
fn test_gap_detection_vacuous_by_width() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": {
                "status": "verified",
                "method": "IBP",
                "output_width": 5000.0,
                "proof_strength": "heuristic"
            }
        }
    });

    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("bert"))
        .expect("should find bert stage");

    assert!(bert.is_vacuous, "width > 1000 should be vacuous");
    assert_eq!(report.vacuous_count, 1);
}

#[test]
fn test_gap_detection_format_report_includes_summary() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": {
                "status": "verified",
                "method": "CROWN",
                "output_width": 2.0,
                "proof_strength": "sound"
            }
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    assert!(formatted.contains("Summary:"), "report should have summary");
    assert!(
        formatted.contains("gaps"),
        "report should mention gap count"
    );
    assert!(
        formatted.contains("total stages"),
        "report should mention total stages"
    );
}

// ===========================================================================
// Status file integration — read/write roundtrip
// ===========================================================================

#[test]
fn test_status_file_roundtrip() {
    let dir = std::env::temp_dir().join(format!("nn_status_e2e_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_status.json");
    let _ = std::fs::remove_file(&path);

    // Create and save a status file.
    let status = VerifyStatus::default();
    status.save(&path).expect("save status");

    // Load it back.
    let loaded = VerifyStatus::load(&path).expect("load status");
    assert!(
        loaded.kernels().is_empty(),
        "fresh status should have no kernels"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_status_file_kernel_query() {
    let dir = std::env::temp_dir().join(format!("nn_status_query_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_status_query.json");
    let _ = std::fs::remove_file(&path);

    // Write a status file with a known kernel entry via JSON.
    let status_json = serde_json::json!({
        "kernels": {
            "test_kernel_alpha": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [1.0]
                },
                "output_bounds": {
                    "lower": -0.5,
                    "upper": 0.5
                },
                "output_width": 1.0,
                "soundness_mode": "sound"
            }
        }
    });
    std::fs::write(&path, serde_json::to_string_pretty(&status_json).unwrap())
        .expect("write status JSON");

    let loaded = VerifyStatus::load(&path).expect("load status");
    assert!(
        loaded.kernel("test_kernel_alpha").is_some(),
        "should find test_kernel_alpha"
    );
    assert!(
        loaded.kernel("nonexistent").is_none(),
        "should not find nonexistent kernel"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ===========================================================================
// Compose + certify flow — build model, verify, produce certificate
// ===========================================================================

#[test]
fn test_compose_certify_linear_sigmoid_ibp_certificate() {
    let d = 4;
    let graph = build_linear_sigmoid(d, d);
    let input = make_uniform_bounds(&[d], 1.0);

    // Run IBP.
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // Verify bounds are valid.
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} <= upper {u}");
    }

    // Create a certificate from the verification results.
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let verification = make_verification(lo_min, hi_max);
    let cert = ProofCertificate::from_verification(&verification, make_input_spec());

    assert!(cert.validate().is_ok(), "certificate should be valid");

    // Bundle it.
    let bundle = CertificateBundle::new("linear_sigmoid_e2e").with_certificate(cert);
    assert_eq!(bundle.len(), 1);
    assert!(bundle.all_sound());
}

#[test]
fn test_compose_certify_mlp_sigmoid_crown_certificate() {
    let d = 4;
    let graph = build_mlp_sigmoid(d);
    let input = make_uniform_bounds(&[d], 1.0);

    // Run CROWN with fallback.
    let (method, output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    let (lo, hi) = output.lower_upper();

    // All bounds should be valid.
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} <= upper {u}");
        // Final layer is sigmoid: bounded to [0, 1].
        assert!(l >= -1e-6, "sigmoid lower {l} >= 0");
        assert!(u <= 1.0 + 1e-6, "sigmoid upper {u} <= 1");
    }

    // Create certificate reflecting the actual method used.
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let prop_method = method;
    let mut verification = KernelVerification::new(
        "mlp_sigmoid_e2e".to_string(),
        prop_method,
        lo_min,
        hi_max,
        hi_max - lo_min,
        true,
    );
    verification.output_tensor = Some(OutputTensorBounds::new(
        lo.iter().copied().collect(),
        hi.iter().copied().collect(),
        vec![d],
    ));

    let cert = ProofCertificate::from_verification(&verification, make_input_spec())
        .with_verifier_version("NY-e2e-test".to_string());
    assert!(cert.validate().is_ok(), "CROWN certificate should be valid");

    let bundle = CertificateBundle::new("mlp_sigmoid_crown_e2e").with_certificate(cert);
    assert_eq!(bundle.len(), 1);
}

// ===========================================================================
// Moonshot property verification — high-level integration
// ===========================================================================

#[test]
fn test_moonshot_status_from_repo_runs() {
    // Verify the MoonshotStatus can be constructed from the repo.
    // This exercises the artifact registry scanning code.
    let status = nn_tts_verify::moonshot::MoonshotStatus::from_repo();
    assert_eq!(
        status.properties.len(),
        8,
        "should have exactly 8 moonshot properties"
    );

    // All properties should have names.
    for prop in &status.properties {
        assert!(!prop.name.is_empty(), "property name should not be empty");
    }

    // Report generation should succeed.
    let report = status.report();
    assert!(!report.is_empty(), "report should not be empty");
    assert!(
        report.contains("Moonshot Status"),
        "report should contain header"
    );
}

#[test]
fn test_junction_contracts_all_valid() {
    let contracts = nn_tts_verify::kokoro_contracts::all_contracts();
    assert_eq!(contracts.len(), 6, "should have 6 junction contracts");

    for contract in &contracts {
        assert!(
            contract.lower < contract.upper,
            "contract {} lower {} should be < upper {}",
            contract.name,
            contract.lower,
            contract.upper
        );
        assert!(
            contract.lower.is_finite() && contract.upper.is_finite(),
            "contract {} bounds must be finite",
            contract.name
        );
    }
}

#[test]
fn test_junction_contract_bounds_within() {
    let contracts = nn_tts_verify::kokoro_contracts::all_contracts();
    let audio_contract = contracts
        .iter()
        .find(|c| c.name == "J5_AUDIO")
        .expect("J5_AUDIO contract should exist");

    // Bounds that are within the contract.
    assert!(nn_tts_verify::kokoro_contracts::bounds_within_contract(
        audio_contract,
        &[-0.9, -0.5, 0.0],
        &[0.5, 0.8, 0.95]
    ));

    // Bounds that exceed the contract.
    assert!(!nn_tts_verify::kokoro_contracts::bounds_within_contract(
        audio_contract,
        &[-1.5],
        &[0.5]
    ));
}
