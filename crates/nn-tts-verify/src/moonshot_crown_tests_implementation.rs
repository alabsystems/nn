// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for moonshot Property 8 (implementation correctness) dispatch plan analysis.

use super::*;
use crate::moonshot::VerificationLevel;
use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::{Conv1dParams, DispatchStep};

fn node(id: usize) -> TensorNodeId {
    TensorNodeId::new(id)
}

/// Minimal KernelDef for test Elementwise steps.
fn dummy_kernel(name: &str) -> nn_dsl::ir::KernelDef {
    nn_dsl::ir::KernelDef::new(
        name,
        vec![],
        ScalarType::F32,
        vec![],
        nn_dsl::NodeId::new(0),
    )
}

// --- ay_kernel_category tests ---

#[test]
fn test_category_sigmoid() {
    let step = DispatchStep::Sigmoid {
        kernel_name: "sig_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("sigmoid"));
}

#[test]
fn test_category_gelu() {
    let step = DispatchStep::Gelu {
        kernel_name: "gelu_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("gelu"));
}

#[test]
fn test_category_relu() {
    let step = DispatchStep::Relu {
        kernel_name: "relu_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("relu"));
}

#[test]
fn test_category_tanh() {
    let step = DispatchStep::Tanh {
        kernel_name: "tanh_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("tanh_act"));
}

#[test]
fn test_category_elementwise_snake() {
    let step = DispatchStep::Elementwise {
        kernel_name: "snake_activation".to_string(),
        scalar_kernel: dummy_kernel("snake"),
        inputs: vec![node(0)],
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("snake"));
}

#[test]
fn test_category_elementwise_silu_mul() {
    let step = DispatchStep::Elementwise {
        kernel_name: "silu_mul_fused".to_string(),
        scalar_kernel: dummy_kernel("silu_mul"),
        inputs: vec![node(0)],
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("silu_mul"));
}

#[test]
fn test_category_elementwise_adain_snake() {
    let step = DispatchStep::Elementwise {
        kernel_name: "adain_snake_fused".to_string(),
        scalar_kernel: dummy_kernel("adain_snake"),
        inputs: vec![node(0)],
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("adain_snake"));
}

#[test]
fn test_category_elementwise_adain_no_snake() {
    let step = DispatchStep::Elementwise {
        kernel_name: "adain_layer".to_string(),
        scalar_kernel: dummy_kernel("adain"),
        inputs: vec![node(0)],
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), Some("adain"));
}

#[test]
fn test_category_elementwise_unknown() {
    let step = DispatchStep::Elementwise {
        kernel_name: "custom_op".to_string(),
        scalar_kernel: dummy_kernel("custom"),
        inputs: vec![node(0)],
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(ay_kernel_category(&step), None);
}

#[test]
fn test_category_linear_no_proof() {
    let step = DispatchStep::Linear {
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
    };
    assert_eq!(ay_kernel_category(&step), None);
}

#[test]
fn test_category_conv1d_no_proof() {
    let step = DispatchStep::Conv1d(Conv1dParams::new(
        "conv_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        None,
        node(2),
        1,
        48,
        8,
        1000,
        12_000,
        4,
        0,
        1,
        1,
    ));
    assert_eq!(ay_kernel_category(&step), None);
}

// --- is_metadata_only tests ---

#[test]
fn test_metadata_reshape() {
    let step = DispatchStep::Reshape {
        input: node(0),
        output: node(1),
    };
    assert!(is_metadata_only(&step));
}

#[test]
fn test_metadata_narrow() {
    let step = DispatchStep::Narrow {
        kernel_name: "narrow_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        input_shape: vec![2, 16, 4],
        axis: 1,
        start: 4,
        length: 8,
    };
    assert!(is_metadata_only(&step));
}

#[test]
fn test_metadata_transpose() {
    let step = DispatchStep::Transpose {
        kernel_name: "trans_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        input_shape: vec![4, 8, 16],
        axes: vec![0, 2, 1],
        total_elements: 512,
    };
    assert!(is_metadata_only(&step));
}

#[test]
fn test_metadata_zero_pad() {
    let step = DispatchStep::ZeroPad1d {
        kernel_name: "zp_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        channels: 48,
        in_length: 250,
        pad_left: 5,
        out_length: 260,
    };
    assert!(is_metadata_only(&step));
}

#[test]
fn test_not_metadata_sigmoid() {
    let step = DispatchStep::Sigmoid {
        kernel_name: "sig_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 1024,
    };
    assert!(!is_metadata_only(&step));
}

#[test]
fn test_not_metadata_linear() {
    let step = DispatchStep::Linear {
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
    };
    assert!(!is_metadata_only(&step));
}

// --- analyze_dispatch_plan tests ---

#[test]
fn test_analyze_all_proven() {
    let steps = vec![
        DispatchStep::Sigmoid {
            kernel_name: "sig_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 1024,
        },
        DispatchStep::Relu {
            kernel_name: "relu_0".to_string(),
            dtype: ScalarType::F32,
            input: node(1),
            output: node(2),
            total_elements: 1024,
        },
        // Metadata-only, excluded from fraction
        DispatchStep::Reshape {
            input: node(2),
            output: node(3),
        },
    ];
    let evidence = analyze_dispatch_plan(&steps);
    assert_eq!(evidence.total_steps, 2); // Reshape excluded
    assert_eq!(evidence.proven_steps, 2);
    assert!(evidence.all_proven);
    assert!(evidence.proven_categories.contains(&"sigmoid".to_string()));
    assert!(evidence.proven_categories.contains(&"relu".to_string()));
    assert!(evidence.unproven_categories.is_empty());
}

#[test]
fn test_analyze_partial_coverage() {
    let steps = vec![
        DispatchStep::Sigmoid {
            kernel_name: "sig_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 1024,
        },
        DispatchStep::Linear {
            kernel_name: "linear_0".to_string(),
            dtype: ScalarType::F32,
            input: node(1),
            weight: node(2),
            bias: None,
            output: node(3),
            in_features: 768,
            out_features: 3072,
            batch_size: 1,
            total_elements: 3072,
        },
    ];
    let evidence = analyze_dispatch_plan(&steps);
    assert_eq!(evidence.total_steps, 2);
    assert_eq!(evidence.proven_steps, 1);
    assert!(!evidence.all_proven);
    assert!(evidence.proven_categories.contains(&"sigmoid".to_string()));
    assert!(evidence.unproven_categories.contains(&"linear".to_string()));
}

#[test]
fn test_analyze_empty_plan() {
    let steps: Vec<DispatchStep> = vec![];
    let evidence = analyze_dispatch_plan(&steps);
    assert_eq!(evidence.total_steps, 0);
    assert_eq!(evidence.proven_steps, 0);
    assert!(!evidence.all_proven); // 0 == 0 but > 0 guard
}

#[test]
fn test_analyze_metadata_only_plan() {
    let steps = vec![
        DispatchStep::Reshape {
            input: node(0),
            output: node(1),
        },
        DispatchStep::Reshape {
            input: node(1),
            output: node(2),
        },
    ];
    let evidence = analyze_dispatch_plan(&steps);
    assert_eq!(evidence.total_steps, 0); // All metadata
    assert_eq!(evidence.proven_steps, 0);
    assert!(!evidence.all_proven);
}

#[test]
fn test_analyze_dedup_categories() {
    let steps = vec![
        DispatchStep::Sigmoid {
            kernel_name: "sig_0".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 1024,
        },
        DispatchStep::Sigmoid {
            kernel_name: "sig_1".to_string(),
            dtype: ScalarType::F32,
            input: node(1),
            output: node(2),
            total_elements: 512,
        },
    ];
    let evidence = analyze_dispatch_plan(&steps);
    assert_eq!(evidence.proven_steps, 2);
    // Category "sigmoid" appears only once despite two sigmoid steps
    assert_eq!(evidence.proven_categories.len(), 1);
    assert_eq!(evidence.proven_categories[0], "sigmoid");
}

// --- check_implementation_correctness tests ---

#[test]
fn test_check_all_proven_smt() {
    let evidence = ImplementationCorrectnessEvidence {
        total_steps: 5,
        proven_steps: 5,
        proven_categories: vec!["sigmoid".to_string(), "relu".to_string()],
        unproven_categories: vec![],
        all_proven: true,
    };
    let result = check_implementation_correctness(&evidence);
    assert_eq!(result.property_index, 7);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::SmtProven);
    assert_eq!(result.bound_value, 5.0);
    assert_eq!(result.threshold, 5.0);
}

#[test]
fn test_check_partial_above_50() {
    let evidence = ImplementationCorrectnessEvidence {
        total_steps: 10,
        proven_steps: 6,
        proven_categories: vec![
            "sigmoid".to_string(),
            "relu".to_string(),
            "gelu".to_string(),
        ],
        unproven_categories: vec!["linear".to_string(), "matmul".to_string()],
        all_proven: false,
    };
    let result = check_implementation_correctness(&evidence);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert_eq!(result.bound_value, 6.0);
    assert_eq!(result.threshold, 10.0);
}

#[test]
fn test_check_low_coverage() {
    let evidence = ImplementationCorrectnessEvidence {
        total_steps: 10,
        proven_steps: 3,
        proven_categories: vec!["sigmoid".to_string()],
        unproven_categories: vec![
            "linear".to_string(),
            "matmul".to_string(),
            "conv1d".to_string(),
        ],
        all_proven: false,
    };
    let result = check_implementation_correctness(&evidence);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_check_zero_steps() {
    let evidence = ImplementationCorrectnessEvidence {
        total_steps: 0,
        proven_steps: 0,
        proven_categories: vec![],
        unproven_categories: vec![],
        all_proven: false,
    };
    let result = check_implementation_correctness(&evidence);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_check_exactly_50_percent() {
    let evidence = ImplementationCorrectnessEvidence {
        total_steps: 10,
        proven_steps: 5,
        proven_categories: vec!["sigmoid".to_string()],
        unproven_categories: vec!["linear".to_string()],
        all_proven: false,
    };
    let result = check_implementation_correctness(&evidence);
    // >= 0.5 threshold
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

// --- ay_proven_kernel_names ---

#[test]
fn test_ay_proven_kernel_names_count() {
    let names = ay_proven_kernel_names();
    assert_eq!(names.len(), 20);
    assert!(names.contains(&"snake"));
    assert!(names.contains(&"silu_mul"));
    assert!(names.contains(&"tanh_act"));
}
