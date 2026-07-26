// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for trace-to-graph translation gap-fill ops (#3557):
//!   - Powf (general exponents: -0.5, -1, 0, 3, 1/3)
//!   - SwiGlu decomposition (Silu + Mul)
//!   - ScatterAdd / IndexAdd (conservative identity passthrough)
//!   - GridSample (conservative identity passthrough)
//!
//! Uses programmatic `ComputationGraph` construction to test TraceOp variants
//! that DynTensor may decompose at trace time.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::trace_to_graph_model;

// ---------------------------------------------------------------------------
// Powf: extended exponent support
// ---------------------------------------------------------------------------

/// Powf(0) = constant 1.0.
#[test]
fn test_powf_zero_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow0".into(),
            TraceOp::Powf { exponent: 0.0 },
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(0) translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 4], 10.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // x^0 = 1.0, so output bounds should be [1.0, 1.0].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!((v - 1.0).abs() < 1e-5, "Powf(0) lo should be ~1.0, got {v}");
    }
    for &v in hi.iter() {
        assert!((v - 1.0).abs() < 1e-5, "Powf(0) hi should be ~1.0, got {v}");
    }
}

/// Powf(-1) = Reciprocal.
#[test]
fn test_powf_neg1_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow_neg1".into(),
            TraceOp::Powf { exponent: -1.0 },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(-1) translation")
        .graph;
    // Use positive-only bounds to avoid division by zero.
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), 0.5_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), 2.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // 1/x for x in [0.5, 2.0] → output in [0.5, 2.0].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 0.5 - 1e-4, "Powf(-1) lo >= 0.5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.0 + 1e-4, "Powf(-1) hi <= 2.0, got {v}");
    }
}

/// Powf(-0.5) = 1/sqrt(x).
#[test]
fn test_powf_neg_half_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow_neghalf".into(),
            TraceOp::Powf { exponent: -0.5 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(-0.5) translation")
        .graph;
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), 4.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // 1/sqrt(x) for x in [1, 4] → output in [0.5, 1.0].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 0.5 - 1e-3, "Powf(-0.5) lo >= 0.5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.0 + 1e-3, "Powf(-0.5) hi <= 1.0, got {v}");
    }
}

/// Powf(3) = x^3 (general case via Exp(3*Log(Abs(x)))).
#[test]
fn test_powf_cube_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 2],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow3".into(),
            TraceOp::Powf { exponent: 3.0 },
            vec![0],
            vec![2, 2],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(3) translation")
        .graph;
    // Use positive bounds to keep the decomposition in valid domain.
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 1.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 3.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // x^3 for x in [1, 3] → output in [1, 27].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 1.0 - 1e-2, "Powf(3) lo >= 1.0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "Powf(3) hi must be finite, got {v}");
    }
}

/// Powf(1/3) = cube root (general case).
#[test]
fn test_powf_third_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow_third".into(),
            TraceOp::Powf {
                exponent: 1.0 / 3.0,
            },
            vec![0],
            vec![3],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(1/3) translation")
        .graph;
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[3]), 8.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // x^(1/3) for x in [1, 8] → output in [1, 2].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 1.0 - 1e-2, "Powf(1/3) lo >= 1.0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "Powf(1/3) hi must be finite, got {v}");
    }
}

// ---------------------------------------------------------------------------
// SwiGlu: Silu(gate) * up
// ---------------------------------------------------------------------------

/// SwiGlu with two inputs: gate and up.
#[test]
fn test_swiglu_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "gate".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "up".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "swiglu".into(),
            TraceOp::SwiGlu,
            vec![0, 1],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = nn_verify::trace_to_graph_model_multi_input(&graph)
        .expect("SwiGlu translation")
        .graph;

    // Multi-input: stacked [2*4 + 2*4] = [16].
    let input_bounds = uniform_bounds(&[16], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "swiglu lo must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "swiglu hi must be finite, got {v}");
    }
}

/// SwiGlu with insufficient inputs should error.
#[test]
fn test_swiglu_one_input_returns_error() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "swiglu".into(),
            TraceOp::SwiGlu,
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    let err = trace_to_graph_model(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("SwiGlu") || msg.contains("2 inputs"),
        "SwiGlu with 1 input should error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// ScatterAdd
// ---------------------------------------------------------------------------
//
// INC-FINAL soundness fix: ScatterAdd/IndexAdd are REFUSED. The deleted
// legacy translator modeled them as identity `Add(x, 0)`, which is NOT a
// sound over-approximation: scatter/index ACCUMULATION can move output
// values outside the destination input's bounds (e.g. two +5 sources added
// into one destination slot exceeds a [-5, 5] passthrough).

/// ScatterAdd refused (accumulation breaks the identity passthrough).
#[test]
fn test_scatter_add_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "dest".into(),
            TraceOp::Input,
            vec![],
            vec![3, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "src".into(),
            TraceOp::Input,
            vec![],
            vec![3, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "idx".into(),
            TraceOp::Input,
            vec![],
            vec![3, 2],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "scatter".into(),
            TraceOp::ScatterAdd { dim: 1 },
            vec![0, 1, 2],
            vec![3, 4],
            DType::F32,
        ),
    ]);

    // Sound refusal (bridge coverage taxonomy).
    {
        let err = nn_verify::trace_to_graph_model_multi_input(&graph)
            .expect_err("ScatterAdd must be refused (unsound identity lowering)");
        let msg = err.to_string();
        assert!(
            msg.contains("ScatterAdd") && msg.contains("not supported"),
            "refusal should name ScatterAdd, got: {msg}"
        );
    }

}

// ---------------------------------------------------------------------------
// IndexAdd
// ---------------------------------------------------------------------------

/// IndexAdd refused (accumulation breaks the identity passthrough — see the
/// ScatterAdd section note above).
#[test]
fn test_index_add_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "dest".into(),
            TraceOp::Input,
            vec![],
            vec![4, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "src".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(2, "idx".into(), TraceOp::Input, vec![], vec![2], DType::F32),
        TraceNode::new(
            3,
            "index_add".into(),
            TraceOp::IndexAdd { dim: 0 },
            vec![0, 1, 2],
            vec![4, 3],
            DType::F32,
        ),
    ]);

    // Sound refusal (bridge coverage taxonomy).
    {
        let err = nn_verify::trace_to_graph_model_multi_input(&graph)
            .expect_err("IndexAdd must be refused (unsound identity lowering)");
        let msg = err.to_string();
        assert!(
            msg.contains("IndexAdd") && msg.contains("not supported"),
            "refusal should name IndexAdd, got: {msg}"
        );
    }

}

// ---------------------------------------------------------------------------
// GridSample: conservative identity passthrough
// ---------------------------------------------------------------------------

/// GridSample passes bounds through from the input tensor.
#[test]
fn test_grid_sample_ibp() {
    use nn_core::dyn_tensor::GridSamplePaddingMode;

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 3, 4, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "grid".into(),
            TraceOp::Input,
            vec![],
            vec![1, 2, 2, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sample".into(),
            TraceOp::GridSample {
                padding_mode: GridSamplePaddingMode::Zeros,
                align_corners: true,
            },
            vec![0, 1],
            vec![1, 3, 2, 2],
            DType::F32,
        ),
    ]);

    let gn = nn_verify::trace_to_graph_model_multi_input(&graph)
        .expect("GridSample translation")
        .graph;

    // Multi-input: stacked [1*3*4*4 + 1*2*2*2] = [56].
    let input_bounds = uniform_bounds(&[56], 4.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -4.0 - 1e-5, "grid_sample lo >= -4, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 4.0 + 1e-5, "grid_sample hi <= 4, got {v}");
    }
}

/// GridSample graph builds successfully.
#[test]
fn test_grid_sample_graph_builds() {
    use nn_core::dyn_tensor::GridSamplePaddingMode;

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "grid".into(),
            TraceOp::Input,
            vec![],
            vec![1, 3, 3, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sample".into(),
            TraceOp::GridSample {
                padding_mode: GridSamplePaddingMode::Border,
                align_corners: false,
            },
            vec![0, 1],
            vec![1, 1, 3, 3],
            DType::F32,
        ),
    ]);

    let result =
        nn_verify::trace_to_graph_model_multi_input(&graph).expect("GridSample should translate");
    assert!(
        result.graph.num_nodes() >= 2,
        "expected >=2 nodes, got {}",
        result.graph.num_nodes()
    );
}
