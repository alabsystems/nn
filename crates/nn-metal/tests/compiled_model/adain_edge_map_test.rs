// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test for the AdaIN edge_map patch (#3254).
//!
//! Exercises the full pipeline: build a CompiledModel via `from_plan` where
//! the plan simulates graph-level detection output (AdainSnake at InstanceNorm
//! position + IdentityPassthrough successors), then execute on GPU and verify
//! the output matches the CPU reference.
//!
//! This tests that `build_edge_map` correctly patches `[x]` → `[x, gamma, beta]`
//! when AdainSnake is placed at a 1-input InstanceNorm step.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_dsl::trace_compile::{CompiledStep, NativeOpKind};
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{assert_close, create_input_buffer, read_output_n};

// -- CPU reference (same as adain_nativeop_test.rs) ----------------------------

fn cpu_adain_snake(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    alpha: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
) -> Vec<f32> {
    let mut normed = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let offset = (b * channels + c) * time;
            let slice = &x[offset..offset + time];
            let mean: f32 = slice.iter().sum::<f32>() / time as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / time as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for t in 0..time {
                normed[offset + t] = (slice[t] - mean) * inv_std;
            }
        }
    }
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let g = gamma[b * channels + c];
            let be = beta[b * channels + c];
            let a = alpha[c];
            let inv_a = 1.0 / a;
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                output[offset + t] = y + inv_a * (a * y).sin().powi(2);
            }
        }
    }
    output
}

// -- Integration test: edge_map patch through from_plan → GPU execute ----------

/// Build a CompiledModel via `from_plan` where the plan simulates graph-level
/// detection: AdainSnake at the InstanceNorm step position with 1 graph edge,
/// successor IdentityPassthrough steps carrying gamma/beta edges.
///
/// The edge_map patch (#3254) must expand `[x]` → `[x, gamma, beta]` so that
/// `execute_native_adain_snake` can resolve all 3 inputs.
///
/// Graph topology (7 nodes):
///   0: Input(x)          [1, 4, 16]  inputs=[]
///   1: Input(gamma)      [1, 4, 1]   inputs=[]
///   2: Input(beta)       [1, 4, 1]   inputs=[]
///   3: InstanceNorm(x)   [1, 4, 16]  inputs=[0]     → AdainSnake NativeOp
///   4: Mul(gamma, norm)  [1, 4, 16]  inputs=[1, 3]  → IdentityPassthrough
///   5: Add(beta, scaled) [1, 4, 16]  inputs=[2, 4]  → IdentityPassthrough
///   6: Relu(adain_out)   [1, 4, 16]  inputs=[5]     → IdentityPassthrough
///
/// Output is at step 3 (AdainSnake), not step 6.
#[test]
fn test_adain_edge_map_patch_from_plan_gpu_execute() {
    use nn_dsl::compile_trace_to_plan_with_fusion;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 16);
    let eps = 1e-5_f64;
    let shape_bct = [batch, channels, time];
    let shape_bc1 = [batch, channels, 1];

    // Random test data.
    let x_data = super::test_utils::rand_f32_vec(0x3254_0001, batch * channels * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0x3254_0002, batch * channels, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0x3254_0003, batch * channels, -0.2, 0.2);
    let alpha_data = super::test_utils::rand_f32_vec(0x3254_0004, channels, 0.5, 2.0);

    // Build the graph with individual ops (InstanceNorm, Mul, Add, Relu).
    // This mimics the graph topology that graph-level detection would see.
    let mut graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_x".into(),
            TraceOp::Input,
            vec![],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "input_gamma".into(),
            TraceOp::Input,
            vec![],
            shape_bc1.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "input_beta".into(),
            TraceOp::Input,
            vec![],
            shape_bc1.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "mul_gamma".into(),
            TraceOp::Mul,
            vec![1, 3],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            5,
            "add_beta".into(),
            TraceOp::Add,
            vec![2, 4],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            6,
            "relu_placeholder".into(),
            TraceOp::Relu,
            vec![5],
            shape_bct.to_vec(),
            DType::F32,
        ),
    ]);
    // Mark node 3 as the output — after fusion, the AdainSnake at step 3
    // produces the actual result. Steps 4-6 are dead IdentityPassthrough.
    assert!(graph.mark_output(3), "node 3 must exist in graph");

    // Compile the graph to get a structurally valid plan.
    // The actual steps will be: [Input×3, InstanceNorm NativeOp, IdPass×2, FusedChain].
    // We then replace them to simulate graph-level detection output.
    let mut plan = compile_trace_to_plan_with_fusion(&graph).expect("compile plan");
    assert_eq!(
        plan.steps.len(),
        7,
        "plan should have 7 steps (1:1 with graph nodes)"
    );

    // Replace steps to simulate graph-level AdaIN detection:
    // Step 3: AdainSnake NativeOp (was InstanceNorm)
    // Steps 4-6: IdentityPassthrough (absorbed Mul, Add, Relu)
    let alpha = WeightRef::new(alpha_data.clone(), vec![channels]).expect("alpha weight");
    let mut weight_data = HashMap::new();
    weight_data.insert("alpha".to_string(), alpha);
    plan.steps[3] = CompiledStep::NativeOp {
        op: NativeOpKind::AdainSnake {
            eps: eps as f32,
            input_shape: shape_bct.to_vec(),
            channels,
            residual_gamma: true,
            external_node_ids: Some(vec![0, 1, 2]), // x, gamma, beta (#3261)
        },
        weight_data,
    };
    plan.steps[4] = CompiledStep::IdentityPassthrough;
    plan.steps[5] = CompiledStep::IdentityPassthrough;
    plan.steps[6] = CompiledStep::IdentityPassthrough;
    // Note: plan.output_step is NOT used by from_plan — output routing
    // is derived from graph.output_nodes() (set by mark_output(3) above).

    // Build CompiledModel via from_plan. This calls build_edge_map() which
    // must patch edge_map[3] from [0] to [0, 1, 2] for the 3-input AdainSnake.
    let compiled = CompiledModel::from_plan(&plan, &graph, &cache)
        .expect("from_plan should succeed with edge_map patch");

    // Execute on GPU with 3 input buffers.
    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);
    let out_buf = compiled
        .execute(&cache, &[&x_buf, &gamma_buf, &beta_buf])
        .expect("execute should resolve all 3 inputs via patched edge_map");

    // Verify output matches CPU reference.
    let result = read_output_n(&out_buf, batch * channels * time);
    let expected = cpu_adain_snake(
        &x_data,
        &gamma_data,
        &beta_data,
        &alpha_data,
        batch,
        channels,
        time,
        eps as f32,
    );
    assert_close("adain_edge_map_patch_gpu", &result, &expected, 1e-3);
}

/// Same test with AdainLeakyRelu to verify defense-in-depth edge_map patch.
#[test]
fn test_adain_leaky_relu_edge_map_patch_from_plan_gpu_execute() {
    use nn_dsl::compile_trace_to_plan_with_fusion;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 32);
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let shape_bct = [batch, channels, time];
    let shape_bc1 = [batch, channels, 1];

    let x_data = super::test_utils::rand_f32_vec(0x3254_0010, batch * channels * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0x3254_0011, batch * channels, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0x3254_0012, batch * channels, -0.2, 0.2);

    let mut graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_x".into(),
            TraceOp::Input,
            vec![],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "input_gamma".into(),
            TraceOp::Input,
            vec![],
            shape_bc1.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "input_beta".into(),
            TraceOp::Input,
            vec![],
            shape_bc1.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "mul_gamma".into(),
            TraceOp::Mul,
            vec![1, 3],
            shape_bct.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            5,
            "add_beta".into(),
            TraceOp::Add,
            vec![2, 4],
            shape_bct.to_vec(),
            DType::F32,
        ),
    ]);
    assert!(graph.mark_output(3), "node 3 must exist in graph");

    let mut plan = compile_trace_to_plan_with_fusion(&graph).expect("compile plan");
    assert_eq!(plan.steps.len(), 6, "plan should have 6 steps");

    plan.steps[3] = CompiledStep::NativeOp {
        op: NativeOpKind::AdainLeakyRelu {
            eps: eps as f32,
            slope: slope as f32,
            input_shape: shape_bct.to_vec(),
            external_node_ids: Some(vec![0, 1, 2]), // x, gamma, beta (#3261)
        },
        weight_data: HashMap::new(),
    };
    plan.steps[4] = CompiledStep::IdentityPassthrough;
    plan.steps[5] = CompiledStep::IdentityPassthrough;
    // Note: plan.output_step is NOT used by from_plan — output routing
    // is derived from graph.output_nodes() (set by mark_output(3) above).

    let compiled = CompiledModel::from_plan(&plan, &graph, &cache)
        .expect("from_plan should succeed with edge_map patch");

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);
    let out_buf = compiled
        .execute(&cache, &[&x_buf, &gamma_buf, &beta_buf])
        .expect("execute should resolve all 3 inputs via patched edge_map");

    let result = read_output_n(&out_buf, batch * channels * time);

    // CPU reference: InstanceNorm → affine → leaky_relu
    let mut normed = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let offset = (b * channels + c) * time;
            let slice = &x_data[offset..offset + time];
            let mean: f32 = slice.iter().sum::<f32>() / time as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / time as f32;
            let inv_std = 1.0 / (var + eps as f32).sqrt();
            for t in 0..time {
                normed[offset + t] = (slice[t] - mean) * inv_std;
            }
        }
    }
    let mut expected = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let g = gamma_data[b * channels + c];
            let be = beta_data[b * channels + c];
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                expected[offset + t] = if y >= 0.0 { y } else { slope as f32 * y };
            }
        }
    }
    assert_close(
        "adain_leaky_relu_edge_map_patch_gpu",
        &result,
        &expected,
        1e-4,
    );
}
