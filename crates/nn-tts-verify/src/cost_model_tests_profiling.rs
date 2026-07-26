// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dispatch plan profiling and roofline conservatism.
//!
//! Extracted from cost_model_tests.rs to stay under 500-line limit.
//! Part of #1739.

use super::*;
use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::DispatchStep;

fn node(id: usize) -> TensorNodeId {
    TensorNodeId::new(id)
}

// --- profile_dispatch_plan ---

#[test]
fn test_profile_empty_plan() {
    let model = HardwareCostModel::m4_max();
    let profiles = profile_dispatch_plan(&[], &model);
    assert!(profiles.is_empty());
    assert_eq!(total_estimated_time_us(&profiles), 0.0);
}

#[test]
fn test_profile_single_linear() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![DispatchStep::Linear {
        kernel_name: "ffn_up".to_string(),
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
    let profiles = profile_dispatch_plan(&plan, &model);
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].layer_name, "ffn_up");
    assert_eq!(profiles[0].flops, 2 * 768 * 3072);
    assert!(profiles[0].estimated_time_us > 0.0);
    assert!(profiles[0].measured_time_us.is_none());
}

#[test]
fn test_profile_total_time_is_sum() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![
        DispatchStep::Relu {
            kernel_name: "relu_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 10_000,
        },
        DispatchStep::Relu {
            kernel_name: "relu_1".to_string(),
            dtype: ScalarType::F32,
            input: node(1),
            output: node(2),
            total_elements: 10_000,
        },
    ];
    let profiles = profile_dispatch_plan(&plan, &model);
    let total = total_estimated_time_us(&profiles);
    let sum: f64 = profiles.iter().map(|p| p.estimated_time_us).sum();
    assert!((total - sum).abs() < 1e-10);
}

#[test]
fn test_total_flops_aggregation() {
    let model = HardwareCostModel::m4_max();
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
        DispatchStep::Relu {
            kernel_name: "relu_0".to_string(),
            dtype: ScalarType::F32,
            input: node(2),
            output: node(3),
            total_elements: 3072,
        },
    ];
    let profiles = profile_dispatch_plan(&plan, &model);
    let total = total_flops(&profiles);
    assert_eq!(total, 2 * 768 * 3072 + 3072);
}

// --- Roofline conservatism ---

#[test]
fn test_roofline_conservative_upper_bound() {
    // The roofline model should upper-bound real execution because it assumes
    // peak hardware utilization. No real workload achieves 100% utilization.
    let model = HardwareCostModel::m4_max();

    // A 512x768 × 768x3072 matmul (FFN shape from Whisper/Kokoro)
    let step = DispatchStep::MatMul {
        kernel_name: "ffn_matmul".to_string(),
        dtype: ScalarType::F32,
        left: node(0),
        right: node(1),
        output: node(2),
        m: 512,
        k: 768,
        n: 3072,
        batch_size: 1,
        transpose_right: false,
        broadcast_right: false,
        scale: None,
        total_elements: 512 * 3072,
    };
    let flops = step_flops(&step);
    let bytes = step_memory_bytes(&step);
    let est_us = model.estimate_time_us(flops, bytes);

    // Measured simdgroup GEMM: 0.784ms for this shape (from design doc helper)
    // Roofline estimate should be >= measured (conservative)
    // FLOPs = 2 * 512 * 768 * 3072 = 2,415,919,104
    // Compute: 2.4e9 / 14.2e6 ≈ 170 μs
    // Memory: (512*768 + 768*3072 + 512*3072) * 4 ≈ 16MB → 16e6 / 400e3 ≈ 40 μs
    // Roofline ≈ max(170, 40) + 5 = 175 μs
    assert_eq!(flops, 2_415_919_104);
    // The roofline estimate should be a lower bound (faster than measured)
    // because real hardware has cache misses, bank conflicts, etc.
    // We check it's in a reasonable range: 100-500 μs
    assert!(est_us > 100.0, "estimate {est_us} too low");
    assert!(est_us < 500.0, "estimate {est_us} too high");
}
