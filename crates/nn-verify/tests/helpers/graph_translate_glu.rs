// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for GLU (Gated Linear Unit) decomposition through NY.
//!
//! GLU = narrow(data) * sigmoid(narrow(gate)), composed from:
//! - `TensorOpKind::Narrow` × 2 → `SliceLayer` × 2
//! - `TensorOpKind::Sigmoid` → `SigmoidLayer`
//! - `TensorOpKind::BinaryMul` → `MulBinaryLayer`
//!
//! Part of #660.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a GLU kernel manually: input [2C, T] → GLU(axis=0) → [C, T].
fn glu_kernel(channels_2x: usize, time: usize) -> TensorKernelDef {
    let half = channels_2x / 2;
    let in_shape = vec![channels_2x, time];
    let half_shape = vec![half, time];

    TensorKernelDef::new(
        "glu_test",
        vec![
            // 0: input [2C, T]
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: in_shape.clone(),
                },
                in_shape,
            ),
            // 1: data = narrow(x, axis=0, start=0, length=C) → [C, T]
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Narrow {
                    input: TensorNodeId::new(0),
                    axis: 0,
                    start: 0,
                    length: half,
                },
                half_shape.clone(),
            ),
            // 2: gate = narrow(x, axis=0, start=C, length=C) → [C, T]
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Narrow {
                    input: TensorNodeId::new(0),
                    axis: 0,
                    start: half,
                    length: half,
                },
                half_shape.clone(),
            ),
            // 3: sigmoid(gate) → [C, T]
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(2),
                },
                half_shape.clone(),
            ),
            // 4: data * sigmoid(gate) → [C, T]
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::BinaryMul {
                    left: TensorNodeId::new(1),
                    right: TensorNodeId::new(3),
                },
                half_shape,
            ),
        ],
        TensorNodeId::new(4),
    )
}

/// GLU graph builds successfully from a single variable input.
#[test]
fn test_glu_variable_builds_graph() {
    let def = glu_kernel(8, 16);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("GLU graph should build");
    // Two SliceLayer + SigmoidLayer + MulBinaryLayer = 4 layers minimum
    assert!(
        graph.num_nodes() >= 4,
        "GLU graph should have at least 4 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate correctly through the full GLU composition.
///
/// GLU(x) = data * sigmoid(gate) where data,gate are halves of x.
/// With uniform input bounds [lo, hi]:
/// - data bounds: [lo, hi]
/// - gate bounds: [lo, hi]
/// - sigmoid(gate) bounds: [sigmoid(lo), sigmoid(hi)]
/// - output = data * sigmoid(gate): McCormick envelope of the product
///
/// Since sigmoid output is in (0,1), and data has the same bounds as input,
/// the output bounds are tighter than the raw product.
#[test]
fn test_glu_ibp_bounds_correct() {
    let ch2 = 4; // 2C = 4, so C = 2
    let time = 8;
    let def = glu_kernel(ch2, time);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    // Input bounds: [-2, 3] uniformly across [4, 8] = 32 elements
    let lower = ArrayD::from_elem(IxDyn(&[ch2, time]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[ch2, time]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP through GLU");

    let out_lower = output.lower();
    let out_upper = output.upper();

    // Output shape: [C=2, T=8]
    let half = ch2 / 2;
    assert_eq!(
        out_lower.len(),
        half * time,
        "output should have C*T = {} elements, got {}",
        half * time,
        out_lower.len()
    );

    // sigmoid([-2, 3]) ≈ [0.119, 0.952]
    // data = [-2, 3]
    // Products: (-2)*0.119=-0.238, (-2)*0.952=-1.905, 3*0.119=0.358, 3*0.952=2.857
    // Expected lower ≥ min(-0.238, -1.905, 0.358, 2.857) ≈ -1.905
    // Expected upper ≤ max(-0.238, -1.905, 0.358, 2.857) ≈ 2.857
    //
    // Observed IBP output: lower=-1.9051483, upper=2.8577223
    // Expected: data=[-2,3], sigmoid([-2,3])≈[0.119,0.953]
    //   min corner: -2*0.953 ≈ -1.905
    //   max corner: 3*0.953 ≈ 2.858
    // IBP is near-exact for McCormick product of these ranges.
    //
    // AC3: Tighter windows centered on observed values (±0.3 each side)
    for &v in out_lower.iter() {
        assert!(
            v > -2.2 && v < -1.6,
            "expected GLU lower ~-1.905 (±0.3), got {v}"
        );
    }
    for &v in out_upper.iter() {
        assert!(
            v > 2.55 && v < 3.15,
            "expected GLU upper ~2.858 (±0.3), got {v}"
        );
    }
}

/// GLU constant-fold: when input is constant, the entire GLU folds.
///
/// GLU(c) = data_half * sigmoid(gate_half). If input is constant c:
/// both halves are c, so output = c * sigmoid(c).
#[test]
fn test_glu_constant_fold() {
    let def = glu_kernel(4, 2);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(1.0)])
        .expect("constant GLU should succeed");

    let lower = ArrayD::from_elem(IxDyn(&[2, 2]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 2]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    // Expected: 1.0 * sigmoid(1.0) = 1.0 / (1 + exp(-1)) ≈ 0.7311
    let expected = 1.0 / (1.0 + (-1.0f32).exp());
    for &v in output.lower().iter() {
        assert!(
            (v - expected).abs() < 1e-4,
            "expected constant fold ~{expected:.4}, got {v}"
        );
    }
}

/// GLU graph wiring: fan-out from single input → two Narrow → Sigmoid + BinaryMul merge.
///
/// Validates the DAG structure: both Narrow nodes reference the same input,
/// Sigmoid references the gate Narrow, BinaryMul references data Narrow and Sigmoid.
#[test]
fn test_glu_graph_wiring_correct() {
    let def = glu_kernel(8, 4);
    // Validate the kernel is well-formed
    def.validate().expect("GLU kernel should validate");

    // Verify the structural wiring
    assert_eq!(def.nodes.len(), 5, "GLU kernel should have 5 nodes");

    // Both narrows reference input 0
    match &def.nodes[1].kind {
        TensorOpKind::Narrow {
            input,
            start,
            length,
            ..
        } => {
            assert_eq!(
                *input,
                TensorNodeId::new(0),
                "data narrow should reference input"
            );
            assert_eq!(*start, 0, "data narrow starts at 0");
            assert_eq!(*length, 4, "data narrow length = C");
        }
        other => panic!("expected Narrow for node 1, got {other:?}"),
    }
    match &def.nodes[2].kind {
        TensorOpKind::Narrow {
            input,
            start,
            length,
            ..
        } => {
            assert_eq!(
                *input,
                TensorNodeId::new(0),
                "gate narrow should reference input"
            );
            assert_eq!(*start, 4, "gate narrow starts at C");
            assert_eq!(*length, 4, "gate narrow length = C");
        }
        other => panic!("expected Narrow for node 2, got {other:?}"),
    }

    // Sigmoid references gate narrow (node 2)
    match &def.nodes[3].kind {
        TensorOpKind::Sigmoid { input } => {
            assert_eq!(
                *input,
                TensorNodeId::new(2),
                "sigmoid should reference gate narrow"
            );
        }
        other => panic!("expected Sigmoid for node 3, got {other:?}"),
    }

    // BinaryMul references data (node 1) and sigmoid(gate) (node 3)
    match &def.nodes[4].kind {
        TensorOpKind::BinaryMul { left, right } => {
            assert_eq!(
                *left,
                TensorNodeId::new(1),
                "mul left should be data narrow"
            );
            assert_eq!(
                *right,
                TensorNodeId::new(3),
                "mul right should be sigmoid(gate)"
            );
        }
        other => panic!("expected BinaryMul for node 4, got {other:?}"),
    }
}

/// GLU gate-data swap detection: asymmetric input detects if data and gate are swapped.
///
/// With uniform input, `data * sigmoid(gate) == gate * sigmoid(data)` when data==gate,
/// so a swap is invisible. This test uses distinct data=[2.0, 2.0] and gate=[-1.0, -1.0]
/// (concatenated as input [2.0, 2.0, -1.0, -1.0] along axis 0 with shape [4, 1]).
///
/// Correct: output = data * sigmoid(gate) = 2.0 * sigmoid(-1.0) = 2.0 * 0.2689 ≈ 0.5379
/// Swapped: output = gate * sigmoid(data) = -1.0 * sigmoid(2.0) = -1.0 * 0.8808 ≈ -0.8808
///
/// The difference (0.5379 vs -0.8808) is ~1.42 — far exceeding any reasonable tolerance.
/// Uses IBP point bounds (lower == upper) for exact computation.
///
/// Part of #789 AC3.
#[test]
fn test_glu_detects_gate_data_swap() {
    let ch2 = 4; // 2C = 4, so C = 2
    let time = 1;
    let def = glu_kernel(ch2, time);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    // Asymmetric input: data half = [2.0, 2.0], gate half = [-1.0, -1.0]
    // Shape: [4, 1] — first 2 rows are data, last 2 rows are gate
    let vals = vec![2.0f32, 2.0, -1.0, -1.0];
    let lower = ArrayD::from_shape_vec(IxDyn(&[ch2, time]), vals.clone()).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[ch2, time]), vals).unwrap();
    let input = BoundedTensor::new(lower, upper).expect("point bounds");

    let output = graph.propagate_ibp(&input).expect("IBP on point bounds");
    let out_lo = output.lower();
    let out_hi = output.upper();

    // Correct: data * sigmoid(gate) = 2.0 * sigmoid(-1.0) ≈ 2.0 * 0.2689 ≈ 0.5379
    let expected = 2.0 * (1.0 / (1.0 + 1.0f32.exp())); // sigmoid(-1) = 1/(1+e^1)
    assert!(
        (expected - 0.5379).abs() < 0.001,
        "analytical reference check: expected ~0.5379, got {expected}"
    );

    // Point bounds should produce tight intervals — verify output matches correct GLU
    for (i, (&lo, &hi)) in out_lo.iter().zip(out_hi.iter()).enumerate() {
        let width = (hi - lo).abs();
        assert!(
            width < 0.01,
            "point bounds should produce tight interval at [{i}], width={width}"
        );
        let mid = f32::midpoint(lo, hi);
        assert!(
            (mid - expected).abs() < 0.01,
            "GLU output[{i}]: expected {expected:.4} (data*sigmoid(gate)), got {mid:.4}. \
             If ~-0.8808, data and gate are swapped."
        );
    }
}

/// GLU via builder produces the same structure as manual construction.
#[test]
fn test_glu_builder_matches_manual() {
    let mut b = TensorBlockBuilder::new("glu_builder");
    let x = b.add_input("x", &[8, 4]);
    let glu = b.add_glu(x, 0, &[8, 4]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    // Should produce the same 5-node structure
    assert_eq!(def.nodes.len(), 5);

    // Verify output shape is halved
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 4]);

    // Build and propagate
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("builder GLU graph should build");
    assert!(graph.num_nodes() >= 4, "graph should have >= 4 nodes");
}
