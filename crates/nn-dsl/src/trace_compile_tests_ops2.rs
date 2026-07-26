// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace compilation tests for extended ops: QLinear, GroupNorm,
//! ConvTranspose2d, AvgPool2d, MaxPool2d, Permute.
//!
//! Extracted from `trace_compile_tests.rs` to keep files under 500 lines.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::dyn_tensor::CompareOp;
use nn_core::DType;

use crate::trace_compile::{compile_trace, CompiledStep};

// -- Helpers (duplicated from trace_compile_tests.rs) -------------------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

// -- QLinear / GroupNorm (#2113) -----------------------------------------------

#[test]
fn test_compile_qlinear_with_weights() {
    let weight = WeightRef::new(vec![1.0; 12], vec![3, 4]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 3], vec![3]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "qlinear_0".into(),
            TraceOp::QLinear { weight, bias },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("qlinear should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(weight_data.contains_key("weight"));
            assert_eq!(kernel.name(), "linear");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_qlinear_no_bias() {
    let weight = WeightRef::new(vec![1.0; 12], vec![3, 4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "qlinear_0".into(),
            TraceOp::QLinear { weight, bias: None },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("qlinear without bias should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_group_norm() {
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups: 2,
                eps: 1e-5,
                weight,
                bias,
            },
            vec![0],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("group_norm should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(
                weight_data.contains_key("weight"),
                "group_norm should have weight data"
            );
            assert!(
                weight_data.contains_key("bias"),
                "group_norm should have bias data"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_group_norm_single_group() {
    // num_groups=1 with 2D input hits the optimized add_group_norm_g1 path.
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[4, 8]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups: 1,
                eps: 1e-5,
                weight,
                bias,
            },
            vec![0],
            vec![4, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("group_norm_g1 should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

// -- ConvTranspose2d (Part of #2113) -----------------------------------------

#[test]
fn test_compile_conv_transpose2d() {
    // weight: [in_ch=3, out_ch/groups=2, kH=3, kW=3] => 3*2*3*3 = 54
    let weight = WeightRef::new(vec![1.0; 54], vec![3, 2, 3, 3]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 2], vec![2]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 4, 4]),
        TraceNode::new(
            1,
            "conv_transpose2d_0".into(),
            TraceOp::ConvTranspose2d {
                weight,
                bias,
                padding: [1, 1],
                output_padding: [0, 0],
                stride: [2, 2],
                dilation: [1, 1],
                groups: 1,
            },
            vec![0],
            vec![2, 7, 7],
            DType::F32,
        ),
    ]);
    // ConvTranspose2d should fail at compile time because MSL codegen is
    // not yet implemented (#2274).
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ConvTranspose2d"),
        "error should mention ConvTranspose2d, got: {msg}"
    );
}

#[test]
fn test_compile_conv_transpose2d_no_bias() {
    let weight = WeightRef::new(vec![1.0; 54], vec![3, 2, 3, 3]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 4, 4]),
        TraceNode::new(
            1,
            "conv_transpose2d_0".into(),
            TraceOp::ConvTranspose2d {
                weight,
                bias: None,
                padding: [0, 0],
                output_padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            vec![0],
            vec![2, 6, 6],
            DType::F32,
        ),
    ]);
    // ConvTranspose2d should fail at compile time because MSL codegen is
    // not yet implemented (#2274).
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ConvTranspose2d"),
        "error should mention ConvTranspose2d, got: {msg}"
    );
}

// -- AvgPool2d / MaxPool2d (Part of #2113) -----------------------------------

#[test]
fn test_compile_avg_pool2d() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 8, 8]),
        TraceNode::new(
            1,
            "avg_pool2d_0".into(),
            TraceOp::AvgPool2d {
                kernel_size: [2, 2],
                stride: [2, 2],
                padding: [0, 0],
            },
            vec![0],
            vec![3, 4, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("avg_pool2d should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "avg_pool2d");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_max_pool2d() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 8, 8]),
        TraceNode::new(
            1,
            "max_pool2d_0".into(),
            TraceOp::MaxPool2d {
                kernel_size: [3, 3],
                stride: [2, 2],
                padding: [1, 1],
            },
            vec![0],
            vec![3, 4, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("max_pool2d should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "max_pool2d");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_avg_pool2d_with_padding() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16, 14, 14]),
        TraceNode::new(
            1,
            "avg_pool2d_0".into(),
            TraceOp::AvgPool2d {
                kernel_size: [3, 3],
                stride: [1, 1],
                padding: [1, 1],
            },
            vec![0],
            vec![16, 14, 14],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("avg_pool2d with padding should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

// -- Permute (must compile to Dispatch, not Passthrough) ----------------------

#[test]
fn test_compile_permute_produces_dispatch() {
    // Permute([0, 2, 1]) on [2, 3, 4] tensor → [2, 4, 3].
    // Must produce a Dispatch step (GPU data reordering), NOT a Passthrough
    // (which would silently return data in the wrong memory layout).
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3, 4]),
        TraceNode::new(
            1,
            "permute_0".into(),
            TraceOp::Permute {
                axes: vec![0, 2, 1],
            },
            vec![0],
            vec![2, 4, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("permute should compile");
    assert_eq!(steps.len(), 2);
    assert!(
        matches!(steps[1], CompiledStep::Dispatch { .. }),
        "Permute must compile to Dispatch, not Passthrough — got {:?}",
        std::mem::discriminant(&steps[1])
    );
}

// -- ToDtype (#2177) ----------------------------------------------------------

/// ToDtype is a passthrough in the compile pipeline because DynTensor uses F32
/// storage internally for all float types.
#[test]
fn test_compile_todtype_passthrough() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "to_dtype_0".into(),
            TraceOp::ToDtype {
                target_dtype: DType::BF16,
            },
            vec![0],
            vec![2, 3],
            DType::BF16,
        ),
    ]);
    let steps = compile_trace(&graph).expect("todtype should compile");
    assert_eq!(steps.len(), 2);
    assert!(
        matches!(steps[1], CompiledStep::Passthrough { .. }),
        "ToDtype should be Passthrough (F32-only execution per #2169)"
    );
}

// -- Compare (scalar comparison producing mask) (#3214) -----------------------

#[test]
fn test_compile_compare_gt() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4, 16]),
        TraceNode::new(
            1,
            "compare_gt".into(),
            TraceOp::Compare {
                op: CompareOp::Gt,
                value: 0.5,
            },
            vec![0],
            vec![4, 16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("compare gt should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(weight_data.contains_key("compare_threshold"));
            let w = &weight_data["compare_threshold"];
            assert_eq!(w.data(), &[0.5_f32]);
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_compare_all_ops() {
    let ops = [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ];
    for op in &ops {
        let graph = graph_from_nodes(vec![
            input_node(0, &[8]),
            TraceNode::new(
                1,
                format!("cmp_{op:?}"),
                TraceOp::Compare {
                    op: *op,
                    value: 1.0,
                },
                vec![0],
                vec![8],
                DType::F32,
            ),
        ]);
        let steps = compile_trace(&graph)
            .unwrap_or_else(|e| panic!("compare {op:?} should compile, got: {e}"));
        assert_eq!(steps.len(), 2, "compare {op:?} should produce 2 steps");
        assert!(
            matches!(steps[1], CompiledStep::Dispatch { .. }),
            "compare {op:?} should produce Dispatch step"
        );
    }
}

#[test]
fn test_compile_compare_nan_threshold_rejected() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "cmp_nan".into(),
            TraceOp::Compare {
                op: CompareOp::Gt,
                value: f64::NAN,
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "NaN threshold should produce NonFiniteConstant error, got: {msg}"
    );
}
