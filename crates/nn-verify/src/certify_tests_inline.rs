// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Inline tests extracted from certify.rs (#3350).

use super::*;
use crate::certificate_types::ConstructiveProofMethod;
use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::{record_input, trace_graph, TraceOp};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_certify_model_linear_relu() {
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

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_linear_relu");
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    assert!(!result.bundle.certificates.is_empty());
    assert!(result.verifiability.is_fully_compilable());

    let cert = &result.bundle.certificates[0];
    assert!(cert.is_finite);
}

#[test]
fn test_certify_model_unverifiable_ops() {
    use nn_core::dyn_tensor::trace::TraceNode;
    use nn_core::DType;

    let input_node = TraceNode::new(
        0,
        "input".to_string(),
        TraceOp::Input,
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let custom_node = TraceNode::new(
        1,
        "custom".to_string(),
        TraceOp::Custom {
            name: "mystery".to_string(),
        },
        vec![0],
        vec![1, 4],
        DType::F32,
    );
    let graph = ComputationGraph::from_nodes(vec![input_node, custom_node]);

    let lower = ArrayD::from_elem(IxDyn(&[1, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_unverifiable");
    let result = certify_model(&graph, &bounds, &config);

    match result {
        Err(CertifyError::UnverifiableOps { ops }) => {
            assert!(ops.contains(&"mystery".to_string()));
        }
        other => panic!("expected UnverifiableOps, got {other:?}"),
    }
}

#[test]
fn test_classify_graph_summary() {
    use nn_core::dyn_tensor::trace::TraceNode;
    use nn_core::DType;

    let nodes = vec![
        TraceNode::new(
            0,
            "in".to_string(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu".to_string(),
            TraceOp::Relu,
            vec![0],
            vec![4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);

    let summary = classify_graph(&graph);
    assert_eq!(summary.shape_only, 1); // Input
    assert_eq!(summary.verifiable, 1); // Relu
    assert!(summary.is_fully_compilable());
}

#[test]
fn test_certify_f32_model_has_f32only_precision() {
    // Pure F32 model (Linear + ReLU, no dtype casts) → PrecisionModel::F32Only.
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

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_f32_precision");
    let result = certify_model(&graph, &bounds, &config).unwrap();

    let cert = &result.bundle.certificates[0];
    assert_eq!(
        cert.precision_model,
        Some(PrecisionModel::F32Only),
        "F32-only model should have PrecisionModel::F32Only"
    );
}

#[test]
fn test_certify_f16_cast_model_has_f16aware_precision() {
    use nn_core::dyn_tensor::trace::TraceNode;
    use nn_core::DType;

    // Input → ReLU → ToDtype(F16) → ReLU.
    // The ToDtype(F16) Clamp layer → PrecisionModel::F16Aware { cast_count: 1 }.
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
            "relu1".to_string(),
            TraceOp::Relu,
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "cast_f16".to_string(),
            TraceOp::ToDtype {
                target_dtype: DType::F16,
            },
            vec![1],
            vec![1, 4],
            DType::F16,
        ),
        TraceNode::new(
            3,
            "relu2".to_string(),
            TraceOp::Relu,
            vec![2],
            vec![1, 4],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);

    let lower = ArrayD::from_elem(IxDyn(&[1, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_f16_precision");
    let result = certify_model(&graph, &bounds, &config).unwrap();

    let cert = &result.bundle.certificates[0];
    match &cert.precision_model {
        Some(PrecisionModel::F16Aware {
            cast_count,
            total_epsilon,
        }) => {
            assert_eq!(*cast_count, 1, "should have exactly 1 F16 cast");
            assert_eq!(*total_epsilon, 0.0, "Phase 1: epsilon is 0.0");
        }
        other => panic!("expected F16Aware {{ cast_count: 1 }}, got {other:?}"),
    }
}

#[test]
fn test_certify_generates_constructive_proof() {
    // Linear + ReLU model with finite bounds -> constructive proof should be generated.
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

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_constructive_proof");
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    // Constructive proof should be present (generate_constructive_proof defaults to true).
    assert!(
        result.has_constructive_proof(),
        "constructive proof should be generated for finite-bounds model"
    );

    let proof = result.constructive_proof().unwrap();
    // IBP escalation + layer bounds composition → IbpComposition.
    // The method reflects the actual escalation result (IBP) composed via
    // composition_from_prop_method. Previously hardcoded to Ibp; now
    // correctly reflects that composition was attempted (#4315).
    assert!(
        matches!(
            proof.method,
            ConstructiveProofMethod::Ibp | ConstructiveProofMethod::IbpComposition
        ),
        "method should be IBP or IBP composition, got {:?}",
        proof.method,
    );
    assert!(proof.verified, "proof should be self-verified");
    assert!(
        proof.is_machine_checkable(),
        "verified proof with output bounds should be machine-checkable"
    );
    assert!(
        !proof.output_lower.is_empty(),
        "output lower bounds should be non-empty"
    );
    assert_eq!(
        proof.output_lower.len(),
        proof.output_upper.len(),
        "output bounds should have matching lengths"
    );
    assert_eq!(
        proof.input_lower.len(),
        proof.input_upper.len(),
        "input bounds should have matching lengths"
    );

    // Verify bounds are finite (constructive proofs require finite bounds).
    for v in &proof.output_lower {
        assert!(v.is_finite(), "output lower bound should be finite: {v}");
    }
    for v in &proof.output_upper {
        assert!(v.is_finite(), "output upper bound should be finite: {v}");
    }

    // Constructive proof should also be embedded in the certificate.
    let cert = &result.bundle.certificates[0];
    assert!(
        cert.has_constructive_proof(),
        "certificate should contain the constructive proof"
    );
}

#[test]
fn test_certify_constructive_proof_disabled() {
    // When generate_constructive_proof is false, no proof should be generated.
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

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut config = CertifyConfig::new("test_no_constructive_proof");
    config.generate_constructive_proof = false;
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    assert!(
        !result.has_constructive_proof(),
        "constructive proof should NOT be generated when disabled"
    );
    assert!(
        result.constructive_proof.is_none(),
        "constructive_proof field should be None"
    );
}

#[test]
fn test_constructive_proof_validates_bounds() {
    // Verify that the constructive proof output bounds are consistent with
    // the CertifyResult output_bounds field.
    let weight = DynTensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2], &Device::Cpu).unwrap();
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

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_bounds_consistency");
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    let proof = result.constructive_proof().unwrap();
    let (out_lo, out_hi) = result.output_bounds.lower_upper();

    // Constructive proof bounds should match the verification output bounds.
    let out_lo_vec: Vec<f32> = out_lo.iter().copied().collect();
    let out_hi_vec: Vec<f32> = out_hi.iter().copied().collect();
    assert_eq!(
        proof.output_lower, out_lo_vec,
        "constructive proof lower bounds should match output_bounds"
    );
    assert_eq!(
        proof.output_upper, out_hi_vec,
        "constructive proof upper bounds should match output_bounds"
    );

    // Input bounds in the proof should match the original input bounds.
    let (in_lo, in_hi) = input_bounds.lower_upper();
    let in_lo_vec: Vec<f32> = in_lo.iter().copied().collect();
    let in_hi_vec: Vec<f32> = in_hi.iter().copied().collect();
    assert_eq!(
        proof.input_lower, in_lo_vec,
        "constructive proof input lower should match input_bounds"
    );
    assert_eq!(
        proof.input_upper, in_hi_vec,
        "constructive proof input upper should match input_bounds"
    );
}
