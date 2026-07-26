// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `TensorOpKind::BinaryAdd` → NY `AddLayer`.
//!
//! Tests:
//! - Two-variable IBP bounds propagation: bounds(a+b) = bounds(a)+bounds(b)
//! - Composition: Conv1d output + skip input through NY
//! - Constant-folding: BinaryAdd of two constants
//! - Mixed: variable + constant

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Composition: Conv1d output + skip input through NY (AC7)
// ---------------------------------------------------------------------------

/// Build a Conv1d + BinaryAdd skip-connection kernel, modeling the Demucs decoder
/// pattern: `output = conv1d(data, weight) + skip`.
///
/// Nodes:
/// - N0: Input "data" [in_ch, in_len]  — Variable
/// - N1: Input "weight" [out_ch, in_ch, k] — ConstantTensor
/// - N2: Input "skip" [out_ch, out_len]  — Variable
/// - N3: Conv1d(N0, N1) → [out_ch, out_len]
/// - N4: BinaryAdd(N3, N2) → [out_ch, out_len]
fn conv1d_skip_add_kernel(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
) -> TensorKernelDef {
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    let conv_out_shape = vec![out_channels, out_length];

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".to_string(),
                shape: vec![in_channels, in_length],
            },
            vec![in_channels, in_length],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".to_string(),
                shape: vec![out_channels, in_channels, kernel_size],
            },
            vec![out_channels, in_channels, kernel_size],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "skip".to_string(),
                shape: conv_out_shape.clone(),
            },
            conv_out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride,
                padding,
                dilation: 1,
                groups: 1,
            },
            conv_out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::BinaryAdd {
                left: TensorNodeId::new(3),
                right: TensorNodeId::new(2),
            },
            conv_out_shape,
        ),
    ];

    TensorKernelDef::new("conv1d_skip_add", nodes, TensorNodeId::new(4))
}

/// Conv1d + BinaryAdd skip connection translates to a valid NY graph.
///
/// Uses in_ch = out_ch so both Variable inputs have the same shape (required by
/// the multi-variable SliceLayer stacking mechanism which slices axis 0 per variable).
#[test]
fn test_conv1d_skip_add_graph_builds() {
    let ch = 2;
    let def = conv1d_skip_add_kernel(ch, ch, 3, 8, 1, 1);
    let weight = ArrayD::from_elem(IxDyn(&[ch, ch, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::Variable,
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("conv1d + skip add graph should build");
    // 2 variables → 2 SliceLayer + Conv1dLayer + AddLayer = 4 minimum
    assert!(
        graph.num_nodes() >= 4,
        "expected >= 4 nodes (2 SliceLayer + Conv1d + AddLayer), got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through Conv1d + BinaryAdd skip connection.
///
/// Verifies the Demucs decoder pattern: `output = conv1d(data) + encoder_skip`.
/// Uses in_ch = out_ch = 2 so both variables have shape [2, 8], enabling the
/// multi-variable stacking along axis 0: NETWORK_INPUT shape = [2, 2, 8].
#[test]
fn test_conv1d_skip_add_ibp_propagates() {
    let ch = 2; // in_ch = out_ch for matching variable shapes
    let k = 3;
    let in_len = 8;
    let stride = 1;
    let padding = 1;
    let out_len = (in_len + 2 * padding - k) / stride + 1; // = 8

    let def = conv1d_skip_add_kernel(ch, ch, k, in_len, stride, padding);
    let weight = ArrayD::from_elem(IxDyn(&[ch, ch, k]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Multi-variable: 2 variables stacked along axis 0.
    // Both have shape [ch, in_len] = [2, 8].
    // Stacked NETWORK_INPUT shape: [2, 2, 8].
    // SliceLayer(axis=0, 0, 1) → data [2, 8], SliceLayer(axis=0, 1, 2) → skip [2, 8].
    let mut lower = ArrayD::from_elem(IxDyn(&[2, ch, in_len]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[2, ch, in_len]), 0.0f32);

    // Variable 0 (data): bounds [-1, 1]
    for c in 0..ch {
        for t in 0..in_len {
            lower[[0, c, t]] = -1.0;
            upper[[0, c, t]] = 1.0;
        }
    }
    // Variable 1 (skip): bounds [-2, 2]
    for c in 0..ch {
        for t in 0..out_len {
            lower[[1, c, t]] = -2.0;
            upper[[1, c, t]] = 2.0;
        }
    }

    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv1d + skip add");

    let out_lower = output.lower();
    let out_upper = output.upper();

    // Multi-variable SliceLayer retains a leading singleton dimension.
    // Output shape: [1, ch, out_len] = [1, 2, 8].
    let out_elements = ch * out_len;
    assert_eq!(
        out_lower.len(),
        out_elements,
        "output should have ch*out_len = {} elements, shape={:?}",
        out_elements,
        out_lower.shape()
    );

    for (l, u) in out_lower.iter().zip(out_upper.iter()) {
        assert!(l.is_finite(), "lower must be finite, got {l}");
        assert!(u.is_finite(), "upper must be finite, got {u}");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // Sanity: skip bounds [-2, 2], Conv1d with weight=0.1 adds bounded contribution.
    // Output bounds should not be drastically wider than the input ranges.
    for &v in out_lower.iter() {
        assert!(v >= -10.0, "lower bound unexpectedly wide: {v}");
    }
    for &v in out_upper.iter() {
        assert!(v <= 10.0, "upper bound unexpectedly wide: {v}");
    }
}

/// Build a simple binary_add kernel: two inputs of the same shape, output = left + right.
fn binary_add_kernel(name: &str, shape: &[usize]) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "left".into(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "right".into(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                shape.to_vec(),
            ),
        ],
        TensorNodeId::new(2),
    )
}

#[test]
fn test_binary_add_two_variables_builds_graph() {
    let def = binary_add_kernel("add_test", &[4, 32]);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("binary add graph should build");
    // 2 variables → multi-variable setup: 2 SliceLayer nodes + 1 AddLayer = 3 minimum
    assert!(
        graph.num_nodes() >= 3,
        "expected >= 3 nodes (2 SliceLayer + 1 AddLayer), got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_binary_add_ibp_bounds_correct() {
    // bounds(a + b) should equal bounds(a) + bounds(b):
    //   lower = a_lower + b_lower
    //   upper = a_upper + b_upper
    let shape = &[2, 4];
    let def = binary_add_kernel("ibp_add", shape);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("build add graph");

    // Multi-variable: inputs are stacked along axis 0 → shape [2, 2, 4]
    let mut lower = ArrayD::from_elem(IxDyn(&[2, 2, 4]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[2, 2, 4]), 0.0f32);
    // left input (slice 0): bounds [-1, 3]
    for i in 0..2 {
        for j in 0..4 {
            lower[[0, i, j]] = -1.0;
            upper[[0, i, j]] = 3.0;
        }
    }
    // right input (slice 1): bounds [2, 5]
    for i in 0..2 {
        for j in 0..4 {
            lower[[1, i, j]] = 2.0;
            upper[[1, i, j]] = 5.0;
        }
    }
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    // Expected: lower = -1 + 2 = 1, upper = 3 + 5 = 8
    let out_lower = output.lower();
    let out_upper = output.upper();
    for &v in out_lower.iter() {
        assert!((v - 1.0).abs() < 1e-5, "expected lower ~1.0, got {v}");
    }
    for &v in out_upper.iter() {
        assert!((v - 8.0).abs() < 1e-5, "expected upper ~8.0, got {v}");
    }
}

#[test]
fn test_binary_add_constant_fold() {
    // BinaryAdd of two constant scalars should fold to a constant output.
    let shape = &[2, 4];
    let def = binary_add_kernel("const_add", shape);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(3.0),
            TensorParamBinding::ConstantScalar(7.0),
        ],
    )
    .expect("constant-fold binary add should succeed");

    // Constant output → graph wraps it in AddConstant(10.0) identity
    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    for &v in output.lower().iter() {
        assert!(
            (v - 10.0).abs() < 1e-5,
            "expected constant fold result 10.0, got {v}"
        );
    }
}

#[test]
fn test_binary_add_variable_plus_constant() {
    // One variable + one constant: output = var + 5.0
    let shape = &[2, 4];
    let def = binary_add_kernel("mixed_add", shape);
    let graph = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(5.0),
        ],
    )
    .expect("mixed binary add should succeed");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 7.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    // Expected: lower = -3 + 5 = 2, upper = 7 + 5 = 12
    for &v in output.lower().iter() {
        assert!((v - 2.0).abs() < 1e-5, "expected lower ~2.0, got {v}");
    }
    for &v in output.upper().iter() {
        assert!((v - 12.0).abs() < 1e-5, "expected upper ~12.0, got {v}");
    }
}

#[test]
fn test_binary_add_validation_rejects_shape_mismatch() {
    let def = TensorKernelDef::new(
        "bad_add",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let result = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    );
    assert!(
        result.is_err(),
        "BinaryAdd with shape mismatch should fail validation"
    );
}
