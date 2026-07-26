// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `build_edge_map` patches in `compiled_model_build.rs`.
//!
//! Tests the AdainSnake/AdainLeakyRelu edge_map routing patch (#3254)
//! that resolves gamma/beta inputs from successor IdentityPassthrough steps
//! when the NativeOp is placed at a 1-input InstanceNorm position.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_dsl::trace_compile::{CompiledStep, NativeOpKind};

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

/// Test that `build_edge_map` patches AdainSnake at a 1-input InstanceNorm
/// position to have [x, gamma, beta] edges by walking successor
/// IdentityPassthrough steps.
///
/// Graph layout (simulating graph-level detection):
///   0: Input(x)           inputs=[]
///   1: Input(gamma)       inputs=[]
///   2: Input(beta)        inputs=[]
///   3: InstanceNorm(x)    inputs=[0]       → AdainSnake NativeOp
///   4: Mul(gamma, normed) inputs=[1, 3]    → IdentityPassthrough
///   5: Add(beta, scaled)  inputs=[2, 4]    → IdentityPassthrough
///   6: Snake(adain_out)   inputs=[5]       → IdentityPassthrough
///
/// Without the patch: edge_map[3] = [0] (only x).
/// With the patch: edge_map[3] = [0, 1, 2] (x, gamma, beta).
#[test]
fn test_build_edge_map_patches_adain_snake_at_instance_norm_position() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        input_node(1, &[1, 4, 1]),
        input_node(2, &[1, 4, 1]),
        TraceNode::new(
            3,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![1, 4, 16],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "mul_gamma".into(),
            TraceOp::Mul,
            vec![1, 3],
            vec![1, 4, 16],
            DType::F32,
        ),
        TraceNode::new(
            5,
            "add_beta".into(),
            TraceOp::Add,
            vec![2, 4],
            vec![1, 4, 16],
            DType::F32,
        ),
        TraceNode::new(
            6,
            "snake_out".into(),
            TraceOp::Relu, // placeholder op
            vec![5],
            vec![1, 4, 16],
            DType::F32,
        ),
    ]);

    let alpha = WeightRef::new(vec![1.0; 4], vec![4]).expect("alpha");
    let mut weight_data = HashMap::new();
    weight_data.insert("alpha".to_string(), alpha);

    let steps = vec![
        CompiledStep::InputForward, // 0: x
        CompiledStep::InputForward, // 1: gamma
        CompiledStep::InputForward, // 2: beta
        CompiledStep::NativeOp {
            // 3: AdainSnake (at InstanceNorm pos)
            op: NativeOpKind::AdainSnake {
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
                channels: 4,
                residual_gamma: true,
                external_node_ids: Some(vec![0, 1, 2]), // x, gamma, beta (#3261)
            },
            weight_data,
        },
        CompiledStep::IdentityPassthrough, // 4: was Mul
        CompiledStep::IdentityPassthrough, // 5: was Add
        CompiledStep::IdentityPassthrough, // 6: was Snake
    ];

    let edge_map = super::build_edge_map(&graph, &steps).expect("build_edge_map should succeed");

    // Before patch: edge_map[3] would be [0] (only x from InstanceNorm's inputs).
    // After patch: edge_map[3] should be [0, 1, 2] (x, gamma, beta).
    assert_eq!(
        edge_map[3],
        vec![0, 1, 2],
        "AdainSnake edge_map should be [x, gamma, beta] after patch"
    );
}

/// Same test for AdainLeakyRelu (defense-in-depth).
#[test]
fn test_build_edge_map_patches_adain_leaky_relu_at_instance_norm_position() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 32]),
        input_node(1, &[1, 8, 1]),
        input_node(2, &[1, 8, 1]),
        TraceNode::new(
            3,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![1, 8, 32],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "mul_gamma".into(),
            TraceOp::Mul,
            vec![1, 3],
            vec![1, 8, 32],
            DType::F32,
        ),
        TraceNode::new(
            5,
            "add_beta".into(),
            TraceOp::Add,
            vec![2, 4],
            vec![1, 8, 32],
            DType::F32,
        ),
    ]);

    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::NativeOp {
            op: NativeOpKind::AdainLeakyRelu {
                eps: 1e-5,
                slope: 0.2,
                input_shape: vec![1, 8, 32],
                external_node_ids: Some(vec![0, 1, 2]), // x, gamma, beta (#3261)
            },
            weight_data: HashMap::new(),
        },
        CompiledStep::IdentityPassthrough,
        CompiledStep::IdentityPassthrough,
    ];

    let edge_map = super::build_edge_map(&graph, &steps).expect("build_edge_map should succeed");

    assert_eq!(
        edge_map[3],
        vec![0, 1, 2],
        "AdainLeakyRelu edge_map should be [x, gamma, beta] after patch"
    );
}

/// Verify that `upload_weights` applies step dtype to ALL weight keys,
/// including bias. For F16 steps, both "weight" and "bias" must be F16
/// because the MSL codegen generates `half*` for all input buffers when
/// the step's scalar type is F16. This test validates the invariant that
/// upload and MSL types are consistent. See #3342.
///
/// Analysis: all F16-autocast MSL kernels read bias as `half*` (not `float*`):
/// - Generated Dispatch MSL: `device const {t}*` where t="half" for F16
/// - NormActivConv1d fused MSL: `device const {scalar_type}* bias`
/// - Mixed GEMM MSL: `device const half* bias`
/// - NormLinear MSL: `device const {scalar_type}* bias`
/// - LSTM MSL is the ONLY `float* bias`, but LSTM stays F32 by classification
///
/// Therefore, upload_weights uploading bias as F16 for F16 steps is CORRECT.
/// Uploading bias as F32 would BREAK F16 steps (MSL reads half*, buffer is float).
#[test]
fn test_upload_weights_applies_step_dtype_to_all_keys_including_bias() {
    use nn_dsl::ir::ScalarType;

    // Build a NativeOp step with both "conv_weight" and "conv_bias" keys.
    let weight = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("weight");
    let bias = WeightRef::new(vec![0.5, -0.5], vec![2]).expect("bias");
    let mut weight_data = HashMap::new();
    weight_data.insert("conv_weight".to_string(), weight);
    weight_data.insert("conv_bias".to_string(), bias);

    let steps = vec![CompiledStep::NativeOp {
        op: NativeOpKind::NormActivConv1d {
            activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 0,
            input_shape: vec![1, 2, 4],
            output_channels: 2,
            kernel_size: 2,
            external_node_ids: None,
        },
        weight_data,
    }];

    // For F16 steps: both conv_weight (4 elems × 2 bytes) and conv_bias (2 elems × 2 bytes)
    // should be uploaded as F16 (half). Buffer byte sizes confirm this.
    let f16_types = vec![ScalarType::F16];
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buffers = super::upload_weights(&steps, &f16_types, &ctx, None).expect("upload");

    let weight_buf = buffers
        .get(&(0, "conv_weight".to_string()))
        .expect("conv_weight buffer");
    let bias_buf = buffers
        .get(&(0, "conv_bias".to_string()))
        .expect("conv_bias buffer");

    // F16: 2 bytes per element. F32 would be 4 bytes per element.
    assert_eq!(
        weight_buf.len(),
        4 * 2,
        "conv_weight should be F16 (4 elements × 2 bytes)"
    );
    assert_eq!(
        bias_buf.len(),
        2 * 2,
        "conv_bias should be F16 (2 elements × 2 bytes)"
    );

    // For F32 steps: both should be F32 (4 bytes per element).
    let f32_types = vec![ScalarType::F32];
    let buffers_f32 = super::upload_weights(&steps, &f32_types, &ctx, None).expect("upload");

    let weight_buf_f32 = buffers_f32
        .get(&(0, "conv_weight".to_string()))
        .expect("conv_weight buffer");
    let bias_buf_f32 = buffers_f32
        .get(&(0, "conv_bias".to_string()))
        .expect("conv_bias buffer");

    assert_eq!(
        weight_buf_f32.len(),
        4 * 4,
        "conv_weight should be F32 (4 elements × 4 bytes)"
    );
    assert_eq!(
        bias_buf_f32.len(),
        2 * 4,
        "conv_bias should be F32 (2 elements × 4 bytes)"
    );
}

/// Verify that the patch is a no-op when AdainSnake already has 3 inputs
/// (the normal KokoroFusedOp::AdainSnake path).
#[test]
fn test_build_edge_map_no_patch_when_adain_already_has_3_inputs() {
    let alpha = WeightRef::new(vec![1.0; 4], vec![4]).expect("alpha");
    let mut weight_data = HashMap::new();
    weight_data.insert("alpha".to_string(), alpha);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        input_node(1, &[1, 4, 1]),
        input_node(2, &[1, 4, 1]),
        TraceNode::new(
            3,
            "adain_snake_0".into(),
            TraceOp::Relu, // placeholder
            vec![0, 1, 2],
            vec![1, 4, 16],
            DType::F32,
        ),
    ]);

    let steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::NativeOp {
            op: NativeOpKind::AdainSnake {
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
                channels: 4,
                residual_gamma: true,
                external_node_ids: None, // graph already has 3 inputs
            },
            weight_data,
        },
    ];

    let edge_map = super::build_edge_map(&graph, &steps).expect("build_edge_map should succeed");

    // Graph node already has [0, 1, 2] inputs — external_node_ids: None
    // means the base graph-topology edges are used directly.
    assert_eq!(
        edge_map[3],
        vec![0, 1, 2],
        "3-input graph node gives correct edges without external_node_ids"
    );
}

#[test]
fn test_upload_weights_aliases_shared_invariant_model_weights() {
    use nn_dsl::ir::ScalarType;

    let weight = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("weight");
    let mut weight_data = HashMap::new();
    weight_data.insert("conv_weight".to_string(), weight);

    let steps = vec![CompiledStep::NativeOp {
        op: NativeOpKind::NormActivConv1d {
            activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 0,
            input_shape: vec![1, 2, 4],
            output_channels: 2,
            kernel_size: 2,
            external_node_ids: None,
        },
        weight_data,
    }];

    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let shared_weight = ctx
        .create_buffer(&[7.0_f32, 8.0, 9.0, 10.0])
        .expect("shared buffer");
    let shared = HashMap::from([((0, "conv_weight".to_string()), shared_weight)]);

    let buffers = super::upload_weights(&steps, &[ScalarType::F32], &ctx, Some(&shared))
        .expect("upload with shared invariant weights");
    let uploaded = buffers
        .get(&(0, "conv_weight".to_string()))
        .expect("aliased shared weight");

    assert!(
        uploaded.is_same_allocation(shared.get(&(0, "conv_weight".to_string())).unwrap()),
        "true model weights should still alias from the shared store"
    );
}

#[test]
fn test_upload_weights_does_not_alias_shared_constant_weight_buffers() {
    use nn_dsl::ir::ScalarType;

    let constant = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).expect("constant");
    let mut weight_data = HashMap::new();
    weight_data.insert("constant_weight".to_string(), constant);

    let steps = vec![CompiledStep::NativeOp {
        op: NativeOpKind::ConstantWeight {
            name: "constant_weight".into(),
            shape: vec![5],
        },
        weight_data,
    }];

    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let stale_shared = ctx
        .create_buffer(&[99.0_f32, 98.0, 97.0])
        .expect("stale shared buffer");
    let shared = HashMap::from([((0, "constant_weight".to_string()), stale_shared)]);

    let buffers = super::upload_weights(&steps, &[ScalarType::F32], &ctx, Some(&shared))
        .expect("upload with constant weight");
    let uploaded = buffers
        .get(&(0, "constant_weight".to_string()))
        .expect("fresh constant weight buffer");
    let shared_buf = shared.get(&(0, "constant_weight".to_string())).unwrap();

    assert_eq!(
        uploaded.len(),
        5 * size_of::<f32>(),
        "ConstantWeight should be uploaded for the current shape, not aliased from a stale shared buffer"
    );
    assert!(
        !uploaded.is_same_allocation(shared_buf),
        "ConstantWeight buffers must not alias the shared store"
    );
}

#[test]
fn test_upload_weights_skips_shared_buffer_when_length_mismatches() {
    use nn_dsl::ir::ScalarType;

    let weight = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("weight");
    let mut weight_data = HashMap::new();
    weight_data.insert("conv_weight".to_string(), weight);

    let steps = vec![CompiledStep::NativeOp {
        op: NativeOpKind::NormActivConv1d {
            activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 0,
            input_shape: vec![1, 2, 4],
            output_channels: 2,
            kernel_size: 2,
            external_node_ids: None,
        },
        weight_data,
    }];

    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let stale_shared = ctx
        .create_buffer(&[9.0_f32, 8.0, 7.0])
        .expect("stale shared buffer");
    let shared = HashMap::from([((0, "conv_weight".to_string()), stale_shared)]);

    let buffers = super::upload_weights(&steps, &[ScalarType::F32], &ctx, Some(&shared))
        .expect("upload with mismatched shared len");
    let uploaded = buffers
        .get(&(0, "conv_weight".to_string()))
        .expect("fresh weight buffer");
    let shared_buf = shared.get(&(0, "conv_weight".to_string())).unwrap();

    assert_eq!(
        uploaded.len(),
        4 * size_of::<f32>(),
        "mismatched shared buffer length must force a fresh upload"
    );
    assert!(
        !uploaded.is_same_allocation(shared_buf),
        "length-mismatched shared buffers must not alias"
    );
}
