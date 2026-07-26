// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for estimate_peak_memory and PeakMemoryProfile (#1739 Phase 19).

use super::*;
use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::DispatchStep;

fn node(id: usize) -> TensorNodeId {
    TensorNodeId::new(id)
}

// --- estimate_peak_memory tests ---

#[test]
fn test_peak_memory_empty_plan() {
    let profile = estimate_peak_memory(&[]);
    assert_eq!(profile.weight_bytes, 0);
    assert_eq!(profile.peak_activation_bytes, 0);
    assert_eq!(profile.peak_total_bytes, 0);
    assert_eq!(profile.peak_step_index, 0);
    assert!(profile.per_step_output_bytes.is_empty());
}

#[test]
fn test_peak_memory_single_step() {
    let plan = vec![DispatchStep::Linear {
        kernel_name: "linear_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        bias: None,
        output: node(2),
        in_features: 768,
        out_features: 3072,
        batch_size: 1,
        total_elements: 3072,
    }];
    let profile = estimate_peak_memory(&plan);

    // Output: 3072 × 4 = 12,288 bytes
    // Weight: 768 × 3072 × 4 = 9,437,184 bytes
    // Peak activation: 0 (no input) + 12,288 (output) = 12,288
    assert_eq!(profile.per_step_output_bytes, vec![3072 * 4]);
    assert_eq!(profile.weight_bytes, 768 * 3072 * 4);
    assert_eq!(profile.peak_activation_bytes, 3072 * 4);
    assert_eq!(profile.peak_total_bytes, 768 * 3072 * 4 + 3072 * 4);
    assert_eq!(profile.peak_step_index, 0);
}

#[test]
fn test_peak_memory_two_steps_peak_at_second() {
    // Step 0: small output (1024 elements)
    // Step 1: large output (8192 elements) — peak is here
    let plan = vec![
        DispatchStep::Relu {
            kernel_name: "relu_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 1024,
        },
        DispatchStep::Relu {
            kernel_name: "relu_1".to_string(),
            dtype: ScalarType::F32,
            input: node(1),
            output: node(2),
            total_elements: 8192,
        },
    ];
    let profile = estimate_peak_memory(&plan);

    // Step 0: input=0, output=1024×4=4096 → live=4096
    // Step 1: input=4096, output=8192×4=32768 → live=36864
    assert_eq!(profile.peak_activation_bytes, 4096 + 32768);
    assert_eq!(profile.peak_step_index, 1);
    assert_eq!(profile.peak_step_name, "relu_1");
}

#[test]
fn test_peak_memory_peak_at_first_step() {
    // Step 0: large output (8192 elements) — peak here
    // Step 1: small output (1024 elements)
    let plan = vec![
        DispatchStep::Relu {
            kernel_name: "relu_big".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 8192,
        },
        DispatchStep::Relu {
            kernel_name: "relu_small".to_string(),
            dtype: ScalarType::F32,
            input: node(1),
            output: node(2),
            total_elements: 1024,
        },
    ];
    let profile = estimate_peak_memory(&plan);

    // Step 0: input=0, output=8192×4=32768 → live=32768
    // Step 1: input=32768, output=1024×4=4096 → live=36864
    // Peak is at step 1 (input is large even though output is small)
    assert_eq!(profile.peak_activation_bytes, 32768 + 4096);
    assert_eq!(profile.peak_step_index, 1);
}

#[test]
fn test_peak_memory_reshape_does_not_inflate() {
    // Reshape produces 0 output bytes → shouldn't inflate peak
    let plan = vec![
        DispatchStep::Relu {
            kernel_name: "relu_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 4096,
        },
        DispatchStep::Reshape {
            input: node(1),
            output: node(2),
        },
        DispatchStep::Relu {
            kernel_name: "relu_1".to_string(),
            dtype: ScalarType::F32,
            input: node(2),
            output: node(3),
            total_elements: 4096,
        },
    ];
    let profile = estimate_peak_memory(&plan);

    // Step 0: input=0, output=4096×4=16384 → live=16384
    // Step 1 (reshape): input=16384, output=0 → live=16384
    // Step 2: input=0 (reshape output), output=16384 → live=16384
    // All steps have same live → peak at step 0 (first maximum)
    assert_eq!(profile.per_step_output_bytes, vec![16384, 0, 16384]);
    assert_eq!(profile.peak_activation_bytes, 16384);
    assert_eq!(profile.peak_step_index, 0);
}

#[test]
fn test_peak_memory_weight_accumulation() {
    // Two linear layers: weights sum, not max
    let plan = vec![
        DispatchStep::Linear {
            kernel_name: "linear_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            weight: node(1),
            bias: None,
            output: node(2),
            in_features: 768,
            out_features: 3072,
            batch_size: 1,
            total_elements: 3072,
        },
        DispatchStep::Linear {
            kernel_name: "linear_1".to_string(),
            dtype: ScalarType::F32,
            input: node(2),
            weight: node(3),
            bias: None,
            output: node(4),
            in_features: 3072,
            out_features: 768,
            batch_size: 1,
            total_elements: 768,
        },
    ];
    let profile = estimate_peak_memory(&plan);

    let w0 = 768 * 3072 * 4; // 9,437,184
    let w1 = 3072 * 768 * 4; // 9,437,184
    assert_eq!(profile.weight_bytes, w0 + w1);
}

// --- PeakMemoryProfile API tests ---

#[test]
fn test_peak_total_mb() {
    let profile = PeakMemoryProfile {
        weight_bytes: 1024 * 1024,         // 1 MB
        peak_activation_bytes: 512 * 1024, // 0.5 MB
        peak_total_bytes: 1024 * 1024 + 512 * 1024,
        peak_step_index: 0,
        peak_step_name: "test".to_string(),
        per_step_output_bytes: vec![],
    };
    let mb = profile.peak_total_mb();
    assert!((mb - 1.5).abs() < 1e-6, "expected 1.5 MB, got {mb}");
}

#[test]
fn test_within_bound() {
    let profile = PeakMemoryProfile {
        weight_bytes: 1_000_000,
        peak_activation_bytes: 500_000,
        peak_total_bytes: 1_500_000,
        peak_step_index: 0,
        peak_step_name: "test".to_string(),
        per_step_output_bytes: vec![],
    };
    assert!(profile.within_bound(2_000_000));
    assert!(profile.within_bound(1_500_000));
    assert!(!profile.within_bound(1_499_999));
}

#[test]
fn test_peak_memory_report_format() {
    let profile = PeakMemoryProfile {
        weight_bytes: 1024 * 1024,
        peak_activation_bytes: 512 * 1024,
        peak_total_bytes: 1024 * 1024 + 512 * 1024,
        peak_step_index: 3,
        peak_step_name: "conv_3".to_string(),
        per_step_output_bytes: vec![],
    };
    let report = profile.report();
    assert!(report.contains("Peak Memory Profile"), "missing header");
    assert!(report.contains("Weight memory:"), "missing weight");
    assert!(report.contains("Peak activation:"), "missing activation");
    assert!(report.contains("conv_3"), "missing step name");
    assert!(report.contains("Peak total:"), "missing total");
}

// --- Integration test with TimingCertificate ---

#[test]
fn test_timing_certificate_includes_peak_memory() {
    use crate::pipeline::VerifiedStage;

    let stages = vec![
        VerifiedStage {
            name: "encoder".to_string(),
            input_lower: vec![0.0],
            input_upper: vec![1.0],
            output_lower: vec![-1.0],
            output_upper: vec![1.0],
            input_shape: vec![1],
            output_shape: vec![1],
            method: "test".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "decoder".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![-2.0],
            output_upper: vec![2.0],
            input_shape: vec![1],
            output_shape: vec![1],
            method: "test".to_string(),
            is_sound: true,
        },
    ];

    let plan = vec![DispatchStep::Linear {
        kernel_name: "linear_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        bias: None,
        output: node(2),
        in_features: 768,
        out_features: 3072,
        batch_size: 1,
        total_elements: 3072,
    }];

    let hw = HardwareCostModel::m4_max();
    let cert = crate::pipeline::verify_pipeline_with_timing(&stages, &plan, &hw, 1_000_000.0)
        .expect("verify_pipeline_with_timing should succeed for valid inputs");

    let peak = cert
        .peak_memory
        .as_ref()
        .expect("peak_memory should be populated");
    assert!(peak.peak_total_bytes > 0, "peak memory should be non-zero");
    assert!(peak.weight_bytes > 0, "weight memory should be non-zero");

    // Verify report includes peak memory
    let report = cert.report();
    assert!(
        report.contains("Peak memory:"),
        "report should include peak memory"
    );
}
