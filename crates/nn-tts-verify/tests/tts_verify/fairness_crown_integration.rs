// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for CROWN-verified fairness bounds (Phase 2 of #1728).
//!
//! These tests build real NY GraphNetworks and run
//! `verify_fairness_bounds()` end-to-end with CROWN/IBP propagation.
//!
//! Requires the `NY` feature to be enabled.

#![cfg(feature = "ny")]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_tts_verify::fairness::Group;
use nn_tts_verify::{verify_fairness_bounds, GroupInputRegion};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a simple Conv1d + ReLU model for fairness testing.
/// Returns (TensorKernelDef, variable_input_shape).
fn build_test_model() -> (nn_dsl::TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("fairness_test_model");
    let input = b.add_input("audio_features", &[4, 8]); // [channels=4, time=8]
    let conv_weight = b.add_input("conv_weight", &[4, 4, 3]); // [out_ch, in_ch, kernel]
    let conv_bias = b.add_input("conv_bias", &[4]);

    // Conv1d: stride=1, padding=1, kernel=3 → output shape = input shape = [4, 8]
    let conv_out = b.add_conv1d(input, conv_weight, Some(conv_bias), 1, 1, &[4, 8]);
    let relu_out = b.add_relu(conv_out, &[4, 8]);
    let def = b.build(relu_out).expect("valid graph");

    (def, vec![4, 8])
}

#[test]
fn test_verify_fairness_bounds_identical_regions() {
    let (def, input_shape) = build_test_model();

    // Create constant weight/bias bindings
    let weight_data = ArrayD::from_elem(IxDyn(&[4, 4, 3]), 0.1f32);
    let bias_data = ArrayD::from_elem(IxDyn(&[4]), 0.01f32);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight_data),
        TensorParamBinding::ConstantTensor(bias_data),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let flat_len: usize = input_shape.iter().product();

    // Two groups with identical input regions → should have identical bound widths
    let regions = vec![
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![1.0; flat_len],
        },
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "ja".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![1.0; flat_len],
        },
    ];

    let cert = verify_fairness_bounds(&graph, &regions, &input_shape, 2.0)
        .expect("verification should succeed");

    // Identical input regions → ratio should be 1.0 (perfectly fair)
    assert!(
        (cert.max_width_ratio - 1.0).abs() < 1e-6,
        "Identical regions should give ratio 1.0, got {}",
        cert.max_width_ratio,
    );
    assert!(cert.is_fair, "Identical regions should be fair");
    assert_eq!(cert.group_results.len(), 2);

    // Both groups should have the same mean output width
    let w0 = cert.group_results[0].mean_output_width;
    let w1 = cert.group_results[1].mean_output_width;
    assert!(
        (w0 - w1).abs() < 1e-6,
        "Identical regions should produce equal widths: {w0} vs {w1}",
    );
}

#[test]
fn test_verify_fairness_bounds_asymmetric_regions() {
    let (def, input_shape) = build_test_model();

    let weight_data = ArrayD::from_elem(IxDyn(&[4, 4, 3]), 0.1f32);
    let bias_data = ArrayD::from_elem(IxDyn(&[4]), 0.01f32);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight_data),
        TensorParamBinding::ConstantTensor(bias_data),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let flat_len: usize = input_shape.iter().product();

    // English: narrow region [0, 1], Japanese: wider region [0, 5]
    // Wider input region → wider output bounds → larger quality variation
    let regions = vec![
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![1.0; flat_len],
        },
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "ja".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![5.0; flat_len], // 5x wider
        },
    ];

    let cert = verify_fairness_bounds(&graph, &regions, &input_shape, 2.0)
        .expect("verification should succeed");

    // Wider input region should produce wider output bounds
    let en_width = cert.group_results[0].mean_output_width;
    let ja_width = cert.group_results[1].mean_output_width;
    assert!(
        ja_width > en_width,
        "Wider input region should produce wider output bounds: ja={ja_width} > en={en_width}",
    );

    // With 5x wider input, the ratio should be > 1.0
    assert!(
        cert.max_width_ratio > 1.0,
        "Asymmetric regions should give ratio > 1.0, got {}",
        cert.max_width_ratio,
    );
}

#[test]
fn test_verify_fairness_bounds_three_groups_threshold() {
    let (def, input_shape) = build_test_model();

    let weight_data = ArrayD::from_elem(IxDyn(&[4, 4, 3]), 0.1f32);
    let bias_data = ArrayD::from_elem(IxDyn(&[4]), 0.01f32);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight_data),
        TensorParamBinding::ConstantTensor(bias_data),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let flat_len: usize = input_shape.iter().product();

    // Three language groups with progressively wider regions
    let regions = vec![
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "en".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![1.0; flat_len],
        },
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "zh".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![2.0; flat_len],
        },
        GroupInputRegion {
            group: Group {
                dimension: "language".to_string(),
                value: "ko".to_string(),
            },
            lower: vec![0.0; flat_len],
            upper: vec![3.0; flat_len],
        },
    ];

    // With loose threshold (10.0) → fair
    let cert_loose = verify_fairness_bounds(&graph, &regions, &input_shape, 10.0)
        .expect("verification should succeed");
    assert!(cert_loose.is_fair, "Loose threshold should pass");
    assert_eq!(cert_loose.group_results.len(), 3);

    // With tight threshold (1.1) → unfair (since ko has 3x the input range of en)
    let cert_tight = verify_fairness_bounds(&graph, &regions, &input_shape, 1.1)
        .expect("verification should succeed");
    assert!(!cert_tight.is_fair, "Tight threshold should fail");
}

#[test]
fn test_verify_fairness_bounds_validation_errors() {
    let (def, input_shape) = build_test_model();

    let weight_data = ArrayD::from_elem(IxDyn(&[4, 4, 3]), 0.1f32);
    let bias_data = ArrayD::from_elem(IxDyn(&[4]), 0.01f32);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight_data),
        TensorParamBinding::ConstantTensor(bias_data),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Empty regions
    let result = verify_fairness_bounds(&graph, &[], &input_shape, 2.0);
    assert!(result.is_err(), "Empty regions should return error");

    let flat_len: usize = input_shape.iter().product();

    // Mismatched lower/upper lengths
    let bad_region = GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "en".to_string(),
        },
        lower: vec![0.0; flat_len],
        upper: vec![1.0; flat_len - 1], // wrong length
    };
    let result = verify_fairness_bounds(&graph, &[bad_region], &input_shape, 2.0);
    assert!(result.is_err(), "Mismatched lengths should return error");

    // NaN in bounds
    let mut lower_with_nan = vec![0.0; flat_len];
    lower_with_nan[0] = f64::NAN;
    let nan_region = GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "ja".to_string(),
        },
        lower: lower_with_nan,
        upper: vec![1.0; flat_len],
    };
    let result = verify_fairness_bounds(&graph, &[nan_region], &input_shape, 2.0);
    assert!(result.is_err(), "NaN bounds should return error");

    // Lower > upper
    let inverted_region = GroupInputRegion {
        group: Group {
            dimension: "language".to_string(),
            value: "ko".to_string(),
        },
        lower: vec![2.0; flat_len],
        upper: vec![1.0; flat_len], // lower > upper
    };
    let result = verify_fairness_bounds(&graph, &[inverted_region], &input_shape, 2.0);
    assert!(result.is_err(), "Inverted bounds should return error");
}
