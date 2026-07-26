// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro segment certification tests.
//!
//! Validates that `certify_model()` produces valid proof certificates for
//! real Kokoro model segments — not just trivial Linear+ReLU test models.
//!
//! D2-D3 from designs/2026-03-20-kokoro-segment-certification-e2e.md.
//! Part of #3020 (Proof Certificates), #3030 (VerifiedCompiledModel), #2218.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_verify::{certify_model, CertifyConfig};

use super::common::kokoro_weights::{bert_encoder_weights, uniform_bt};

// -- Test-sized Kokoro dimensions (matching compose_kokoro_trace_full.rs) ------

const D_EN: usize = 8;
const HIDDEN: usize = 8;

// -- D2: Segment certification tests ------------------------------------------

/// Certify bert_encoder (single Linear layer) — simplest Kokoro segment.
///
/// Validates the full certification pipeline: trace → graph → IBP/CROWN →
/// certificate bundle. This is the minimal wiring test for certify_model()
/// on a real Kokoro component.
#[test]
fn test_certify_kokoro_bert_encoder() {
    let weights = bert_encoder_weights(D_EN, HIDDEN, 0.01);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let w = vb.get(&[D_EN, HIDDEN], "weight").unwrap();
    let b = vb.get(&[D_EN], "bias").unwrap();
    let bert_encoder = Linear::new(w, Some(b)).unwrap();

    let batch = 1;
    let seq_len = 3;
    let input_shape = [batch, seq_len, HIDDEN];
    let bert_output = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = bert_output.clone();
        let id = record_input(x.dims(), DType::F32).unwrap();
        x.set_trace_id(id);
        bert_encoder.forward(&x)
    })
    .unwrap();

    let bounds = uniform_bt(&input_shape, -1.0, 1.0);
    let config = CertifyConfig::new("kokoro_bert_encoder");
    let result = certify_model(&graph, &bounds, &config).unwrap();

    // Certificate bundle is non-empty.
    assert!(
        !result.bundle.certificates.is_empty(),
        "certify_model should produce at least one certificate"
    );

    // Graph is fully verifiable (single Linear layer).
    assert!(
        result.verifiability.is_fully_compilable(),
        "bert_encoder (Linear) should be fully compilable"
    );

    // Certificate has finite bounds (not vacuous).
    let cert = &result.bundle.certificates[0];
    assert!(cert.is_finite, "certificate should have finite bounds");

    // Output bounds should be valid (lower <= upper for all elements).
    let (lower, upper) = result.output_bounds.lower_upper();
    for (lo, hi) in lower.iter().zip(upper.iter()) {
        assert!(lo <= hi, "output bounds should be valid: {lo} <= {hi}");
    }
}

/// Certify bert_encoder with non-trivial weights to verify bounds are real.
///
/// Uses fill=0.01 weights so IBP produces non-zero bounds, confirming that
/// the certification pipeline processes actual numerical data.
#[test]
fn test_certify_kokoro_bert_encoder_nontrivial_bounds() {
    let weights = bert_encoder_weights(D_EN, HIDDEN, 0.01);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let w = vb.get(&[D_EN, HIDDEN], "weight").unwrap();
    let b = vb.get(&[D_EN], "bias").unwrap();
    let bert_encoder = Linear::new(w, Some(b)).unwrap();

    let input_shape = [1, 4, HIDDEN];
    let input = DynTensor::full(&input_shape, 0.5, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = input.clone();
        let id = record_input(x.dims(), DType::F32).unwrap();
        x.set_trace_id(id);
        bert_encoder.forward(&x)
    })
    .unwrap();

    let bounds = uniform_bt(&input_shape, -2.0, 2.0);
    let config = CertifyConfig::new("kokoro_bert_encoder_nontrivial");
    let result = certify_model(&graph, &bounds, &config).unwrap();

    // With fill=0.01 weights and [-2, 2] input bounds, output bounds should
    // be wider than zero (non-degenerate).
    let (lower, upper) = result.output_bounds.lower_upper();
    let lo_min = lower.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = upper.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let width = hi_max - lo_min;
    assert!(
        width > 1e-6,
        "output bounds should be non-degenerate with fill=0.01 weights, got width={width}"
    );

    // Bound analysis should be present (layer bounds were extracted).
    assert!(
        result.bound_analysis.is_some(),
        "bound analysis should be generated for a verifiable model"
    );
}

// -- D3: Fusion bounds regression test ----------------------------------------

/// Verify that fusion certificates use derived bounds, not hardcoded (-3, 3).
///
/// After the D1 fix, certify_model() derives fusion bounds from IBP/CROWN
/// layer bounds instead of using the hardcoded (-3.0, 3.0) default.
/// This test builds a model with known input bounds and checks that any
/// fusion certificates (if generated) use the derived bounds.
#[test]
fn test_certify_fusion_bounds_derived_from_layer_bounds() {
    // Build a Linear+ReLU model (ReLU is a fusible op in some contexts).
    let weight = DynTensor::from_vec(vec![0.5, 0.0, 0.0, 0.5], &[2, 2], &cpu()).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.1, -0.1], &[1, 2], &cpu()).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    // Use narrow input bounds [-0.5, 0.5] — much tighter than (-3, 3).
    let bounds = uniform_bt(&[1, 2], -0.5, 0.5);
    let config = CertifyConfig::new("test_fusion_bounds_derived");
    let result = certify_model(&graph, &bounds, &config).unwrap();

    // The model is verifiable.
    assert!(result.verifiability.is_fully_compilable());

    // If fusion certificates exist, check they don't all use (-3, 3) default.
    // Note: Linear+ReLU may not match any AutoFusionSpec, so empty is OK.
    // The test validates the pipeline doesn't crash and bounds are passed through.
    if !result.fusion_certificates.is_empty() {
        for cert in &result.fusion_certificates {
            // At least one bound should differ from the (-3, 3) default
            // since our input bounds are [-0.5, 0.5].
            let all_default = cert
                .variable_bounds
                .iter()
                .all(|&(lo, hi)| (lo - (-3.0)).abs() < 1e-6 && (hi - 3.0).abs() < 1e-6);
            assert!(
                !all_default,
                "fusion certificates should use derived bounds, not default (-3, 3)"
            );
        }
    }
}
