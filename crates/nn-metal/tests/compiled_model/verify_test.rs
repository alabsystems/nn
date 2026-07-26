// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `CompiledModel` auto-verification API (#3042).
//!
//! These tests require the `verify` feature:
//! `cargo test -p nn-metal --features verify -- verify_test`

use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;
use nn_verify::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

use super::helpers::input_node;

/// Compile + verify a simple Linear → ReLU graph.
/// Verifies that `from_trace_verified` returns both a compiled model
/// and a `VerifyTraceResult` with valid IBP bounds.
#[test]
fn test_from_trace_verified_linear_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

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

    let (compiled, result) =
        nn_metal::compiled_model::CompiledModel::from_trace_verified(&graph, &cache, &bounds)
            .expect("from_trace_verified should succeed");

    // Verification result should be meaningful.
    assert!(result.node_count > 0);
    assert_eq!(result.input_count, 1);
    assert!(result.ibp_width.is_finite());
    assert!(result.is_tight());

    // ReLU output lower bounds should be >= 0.
    let (lo, _hi) = result.ibp_bounds.lower_upper();
    for &l in lo.iter() {
        assert!(l >= 0.0, "ReLU lower bound should be >= 0, got {l}");
    }

    // Compiled model should be usable for execution.
    assert_eq!(compiled.num_inputs(), 1);
}

/// Compile + certify a simple Linear → ReLU graph.
/// Verifies that `from_trace_certified` returns a `VerifiedModel`
/// with a non-empty certificate bundle.
#[test]
fn test_from_trace_certified_linear_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

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

    let config = nn_verify::CertifyConfig::new("test_auto_certify");
    let verified = nn_metal::compiled_model::CompiledModel::from_trace_certified(
        &graph, &cache, &bounds, &config,
    )
    .expect("from_trace_certified should succeed");

    assert!(verified.is_fully_verified());
    assert!(!verified.certificate.bundle.certificates.is_empty());
    assert_eq!(verified.model.num_inputs(), 1);
}

/// Compile a simple Linear → ReLU model via `compile_forward` and verify
/// that a proof certificate is automatically populated (verified by default).
///
/// Part of #3042.
#[test]
fn test_compile_forward_auto_verifies() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let compiled = nn_metal::compiled_model::CompiledModel::compile_forward(
        &[&input],
        |traced| {
            let h = linear.forward(&traced[0])?;
            h.relu()
        },
        &cache,
    )
    .expect("compile_forward should succeed");

    // Certificate should be automatically populated when verify feature is active.
    let cert_json = compiled.proof_certificate_json();
    assert!(
        cert_json.is_some(),
        "verify feature is enabled; certificate should be auto-generated for Linear+ReLU"
    );

    // Certificate should be valid JSON containing expected fields.
    let json = cert_json.unwrap();
    assert!(
        json.contains("\"method\""),
        "certificate should contain method field"
    );
    assert!(
        json.contains("output_bounds"),
        "certificate should contain output_bounds"
    );

    // save_proof_certificate should write to disk.
    let tmp =
        std::env::temp_dir().join(format!("nn_auto_verify_test_{}.json", std::process::id()));
    let saved = compiled.save_proof_certificate(&tmp).unwrap();
    assert!(saved, "save should return true when certificate exists");
    assert!(tmp.exists(), "certificate file should exist");
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert_eq!(content, json);
    std::fs::remove_file(&tmp).ok();
}

/// Verify that `from_trace_verified` rejects a graph with unsupported ops.
#[test]
fn test_from_trace_verified_unsupported_op() {
    use nn_core::dyn_tensor::trace::TraceNode;
    use nn_core::DType;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "custom".to_string(),
            nn_core::dyn_tensor::trace::TraceOp::Custom {
                name: "unknown_op".to_string(),
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    let lower = ArrayD::from_elem(IxDyn(&[4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let result =
        nn_metal::compiled_model::CompiledModel::from_trace_verified(&graph, &cache, &bounds);

    // Should fail because the custom op can't be translated.
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("verification failed") || err_msg.contains("unsupported"),
        "expected verification/unsupported error, got: {err_msg}"
    );
}
