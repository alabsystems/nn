// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Monolithic TensorOpKind::LayerNorm → LayerNormLayer translation tests (#746).
//!
//! Extracted from `graph_translate_layer_norm.rs` for file size (#763).

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a monolithic LayerNorm kernel using TensorBlockBuilder.
fn monolithic_layer_norm_kernel(batch: usize, hidden: usize) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("layer_norm_mono_test");
    let x = b.add_input("x", &[batch, hidden]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[hidden]);
    let bias = b.add_input("bias", &[hidden]);
    let axis = 1; // last axis
    let out = b.add_layer_norm(x, eps, axis, weight, bias, &[batch, hidden]);
    b.build(out).expect("valid graph")
}

#[test]
fn test_layer_norm_monolithic_validates() {
    let _def = monolithic_layer_norm_kernel(2, 8);
}

#[test]
fn test_layer_norm_monolithic_output_shape() {
    let def = monolithic_layer_norm_kernel(2, 8);
    let output = &def.nodes[def.output.index()];
    assert_eq!(output.shape, vec![2, 8], "output shape should match input");
}

#[test]
fn test_layer_norm_monolithic_builds_gamma_crown_graph() {
    let def = monolithic_layer_norm_kernel(2, 8);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("monolithic LayerNorm should build NY graph");
    // Monolithic LayerNormLayer is a single NY layer (NETWORK_INPUT
    // is implicit), so num_nodes() >= 1.
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the LayerNorm node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_layer_norm_monolithic_ibp_bounds_finite() {
    let batch = 2;
    let hidden = 4;
    let def = monolithic_layer_norm_kernel(batch, hidden);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, hidden]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[batch, hidden]), 1.0f32),
    )
    .expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    for &val in lo.iter() {
        assert!(val.is_finite(), "lower bound must be finite, got {val}");
    }
    for &val in hi.iter() {
        assert!(val.is_finite(), "upper bound must be finite, got {val}");
    }
}

#[test]
fn test_layer_norm_monolithic_ibp_with_affine() {
    let batch = 1;
    let hidden = 4;
    let def = monolithic_layer_norm_kernel(batch, hidden);
    // gamma=2.0, beta=0.5
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(0.5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, hidden]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[batch, hidden]), 5.0f32),
    )
    .expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    for &val in lo.iter() {
        assert!(
            val.is_finite(),
            "affine lower bound must be finite, got {val}"
        );
    }
    for &val in hi.iter() {
        assert!(
            val.is_finite(),
            "affine upper bound must be finite, got {val}"
        );
    }
}

#[test]
fn test_layer_norm_monolithic_point_input_equals_beta() {
    // Constant input: all same value → normalized = 0, output = beta.
    let batch = 1;
    let hidden = 4;
    let def = monolithic_layer_norm_kernel(batch, hidden);
    let beta = 0.5f32;
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(beta),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, hidden]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[batch, hidden]), 1.0f32),
    )
    .expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    // gamma * 0 + beta = beta for constant input
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            (l - beta).abs() < 1e-2,
            "lower should be ~{beta} for constant input, got {l}"
        );
        assert!(
            (u - beta).abs() < 1e-2,
            "upper should be ~{beta} for constant input, got {u}"
        );
    }
}

#[test]
fn test_layer_norm_monolithic_constant_tensor_bindings() {
    // Per-element weight and bias via ConstantTensor.
    let batch = 2;
    let hidden = 4;
    let def = monolithic_layer_norm_kernel(batch, hidden);
    let gamma_data: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.5 * i as f32).collect();
    let beta_data: Vec<f32> = (0..hidden).map(|i| 0.1 * i as f32).collect();
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[hidden]), gamma_data).expect("gamma shape"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[hidden]), beta_data).expect("beta shape"),
        ),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConstantTensor LayerNorm graph");
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least one node"
    );
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, hidden]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[batch, hidden]), 1.0f32),
    )
    .expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    for &val in lo.iter() {
        assert!(val.is_finite(), "lower bound must be finite, got {val}");
    }
    for &val in hi.iter() {
        assert!(val.is_finite(), "upper bound must be finite, got {val}");
    }
}

#[test]
fn test_layer_norm_monolithic_constant_input_nonuniform_beta_rejected() {
    // Constant input + non-uniform ConstantTensor beta cannot be folded
    // to a single scalar — the translation rejects this degenerate case.
    let def = monolithic_layer_norm_kernel(2, 4);
    let beta_data = vec![0.1, 0.2, 0.3, 0.4];
    let err = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(1.0),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantScalar(1.0),
            TensorParamBinding::ConstantTensor(
                ArrayD::from_shape_vec(IxDyn(&[4]), beta_data).unwrap(),
            ),
        ],
    )
    .expect_err("non-uniform beta with constant input should be rejected");
    let msg = format!("{err:?}");
    assert!(msg.contains("non-uniform beta"), "unexpected error: {msg}");
}
