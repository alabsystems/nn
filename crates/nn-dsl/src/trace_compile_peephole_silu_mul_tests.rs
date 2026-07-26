// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for peephole pass 14: Silu + Mul → SiluMul fusion.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::trace_compile::CompiledKernel;
use crate::CompiledStep;

use super::fuse_silu_mul;

/// Helper: build a TraceNode for graph construction.
fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Relu,
        inputs,
        shape,
        DType::F32,
    )
}

/// Helper: create a Silu dispatch step.
fn silu_step(shape: &[usize]) -> CompiledStep {
    let scalar = crate::kernel_util::build_scalar_kernel(
        "fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }",
    )
    .expect("silu kernel");

    let mut b = TensorBlockBuilder::new("silu");
    let input = b.add_input("x", shape);
    let output = b.add_elementwise(scalar, &[input], shape);
    let def = b.build(output).expect("valid silu IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

/// Helper: create a Mul dispatch step.
fn mul_step(shape: &[usize]) -> CompiledStep {
    let scalar = crate::kernel_util::build_scalar_kernel("fn mul(a: f32, b: f32) -> f32 { a * b }")
        .expect("mul kernel");

    let mut b = TensorBlockBuilder::new("mul");
    let a = b.add_input("a", shape);
    let b_in = b.add_input("b", shape);
    let output = b.add_elementwise(scalar, &[a, b_in], shape);
    let def = b.build(output).expect("valid mul IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

#[test]
fn test_fuse_silu_mul_basic() {
    let shape = vec![2, 3];
    // Graph: input(0) → silu(1) → mul(2), input(3) → mul(2)
    //   gate=0, silu=1, up=3, mul=2
    // Steps align: step 0=gate input, step 1=silu, step 2=mul, step 3=up input
    // But peephole only looks at consecutive steps, so we set it up as:
    //   step 0: silu (graph node 0, inputs=[10])
    //   step 1: mul  (graph node 1, inputs=[0, 11])
    let mut steps = vec![silu_step(&shape), mul_step(&shape)];

    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "silu", vec![10], shape.clone()),
        test_node(1, "mul", vec![0, 11], shape),
    ]);

    // use_counts: step 0 (silu) has 1 consumer (the mul).
    let use_counts = vec![1, 0];

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    // Step 0 should be IdentityPassthrough (was silu).
    assert!(
        matches!(steps[0], CompiledStep::IdentityPassthrough),
        "silu step should become IdentityPassthrough"
    );
    // Step 1 should be fused silu_mul Dispatch.
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            external_node_ids,
            ..
        } => {
            assert_eq!(kernel.name(), "silu_mul");
            // external_node_ids = [gate=10, up=11]
            assert_eq!(external_node_ids.as_deref(), Some(&[10_u64, 11][..]));
        }
        other => panic!("expected Dispatch silu_mul, got {other:?}"),
    }
}

#[test]
fn test_fuse_silu_mul_reversed_inputs() {
    let shape = vec![4, 8];
    // Mul inputs reversed: mul(up, silu_out) instead of mul(silu_out, up).
    let mut steps = vec![silu_step(&shape), mul_step(&shape)];

    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "silu", vec![10], shape.clone()),
        test_node(1, "mul", vec![11, 0], shape), // reversed: up first
    ]);

    let use_counts = vec![1, 0];

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    assert!(matches!(steps[0], CompiledStep::IdentityPassthrough));
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            external_node_ids,
            ..
        } => {
            assert_eq!(kernel.name(), "silu_mul");
            // Gate=10, up=11 — correct order regardless of original mul input order.
            assert_eq!(external_node_ids.as_deref(), Some(&[10_u64, 11][..]));
        }
        other => panic!("expected Dispatch silu_mul, got {other:?}"),
    }
}

#[test]
fn test_no_fuse_when_silu_multi_use() {
    let shape = vec![2, 3];
    let mut steps = vec![silu_step(&shape), mul_step(&shape)];

    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "silu", vec![10], shape.clone()),
        test_node(1, "mul", vec![0, 11], shape),
    ]);

    // Silu output has 2 consumers — should NOT fuse.
    let use_counts = vec![2, 0];

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    assert!(
        matches!(steps[0], CompiledStep::Dispatch { .. }),
        "silu should stay as Dispatch when multi-use"
    );
    assert!(
        matches!(steps[1], CompiledStep::Dispatch { .. }),
        "mul should stay as Dispatch when silu is multi-use"
    );
}

#[test]
fn test_no_fuse_non_adjacent() {
    let shape = vec![2, 3];
    let mut steps = vec![
        silu_step(&shape),
        CompiledStep::IdentityPassthrough, // gap
        mul_step(&shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "silu", vec![10], shape.clone()),
        test_node(1, "passthrough", vec![0], shape.clone()),
        test_node(2, "mul", vec![0, 11], shape),
    ]);

    let use_counts = vec![1, 0, 0];

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    // Should NOT fuse — silu and mul are not adjacent.
    assert!(matches!(steps[0], CompiledStep::Dispatch { .. }));
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_fuse_multiple_pairs() {
    let shape = vec![2, 3];
    let mut steps = vec![
        silu_step(&shape),
        mul_step(&shape),
        silu_step(&shape),
        mul_step(&shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "silu", vec![10], shape.clone()),
        test_node(1, "mul", vec![0, 11], shape.clone()),
        test_node(2, "silu", vec![12], shape.clone()),
        test_node(3, "mul", vec![2, 13], shape),
    ]);

    let use_counts = vec![1, 0, 1, 0];

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    // Both pairs should be fused.
    assert!(matches!(steps[0], CompiledStep::IdentityPassthrough));
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "silu_mul")
    );
    assert!(matches!(steps[2], CompiledStep::IdentityPassthrough));
    assert!(
        matches!(&steps[3], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "silu_mul")
    );
}

#[test]
fn test_no_fuse_mul_without_silu_input() {
    let shape = vec![2, 3];
    // Mul's inputs don't include the silu output (node 0).
    let mut steps = vec![silu_step(&shape), mul_step(&shape)];

    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "silu", vec![10], shape.clone()),
        test_node(1, "mul", vec![11, 12], shape), // neither input is node 0
    ]);

    let use_counts = vec![1, 0];

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    // Should NOT fuse — mul doesn't consume silu output.
    assert!(matches!(steps[0], CompiledStep::Dispatch { .. }));
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_empty_steps() {
    let mut steps: Vec<CompiledStep> = vec![];
    let use_counts: Vec<usize> = vec![];
    let graph = ComputationGraph::from_nodes(vec![]);
    fuse_silu_mul(&mut steps, &use_counts, &graph); // should not panic
}

/// Helper: create a Linear dispatch step for MLP tests.
///
/// Builds a minimal Dispatch{linear} with the correct kernel name and
/// weight_data entries so the peephole recognizes it as a linear layer.
fn linear_step(input_shape: &[usize], out_features: usize) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;
    use nn_core::dyn_tensor::trace::WeightRef;

    let in_features = *input_shape.last().unwrap();
    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().unwrap() = out_features;
    let weight_shape = [out_features, in_features];

    let mut b = TensorBlockBuilder::new("linear");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", &weight_shape);
    let bi = b.add_input("bias", &[out_features]);
    let output = b.add_linear(input, w, Some(bi), &output_shape);
    let def = b.build(output).expect("valid linear IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; out_features * in_features],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; out_features], vec![out_features]).expect("valid bias"),
    );

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

/// Test: SwiGLU MLP block dispatch count reduction.
///
/// SwiGLU MLP (used in Qwen3, GLM5, LLaMA, Whisper decoder):
///   gate = gate_proj(x)     -- Linear [B, S, D] → [B, S, I]
///   up   = up_proj(x)       -- Linear [B, S, D] → [B, S, I]
///   act  = silu(gate)       -- elementwise
///   h    = act * up          -- elementwise
///   out  = down_proj(h)     -- Linear [B, S, I] → [B, S, D]
///
/// Before fusion: 5 dispatches (gate_proj, up_proj, silu, mul, down_proj).
/// After fuse_silu_mul: silu → IdentityPassthrough, mul → silu_mul.
///   Effective dispatches: 4 (gate_proj, up_proj, silu_mul, down_proj).
///   Saved: 1 dispatch per transformer layer.
///
/// In a 32-layer Qwen3 model, this saves 32 dispatches total.
#[test]
fn test_swiglu_mlp_dispatch_reduction() {
    // Typical transformer dimensions: batch=1, seq_len=32, d_model=768, intermediate=2048.
    let input_shape = vec![1, 32, 768];
    let intermediate = 2048;
    let intermediate_shape = vec![1, 32, intermediate];

    // Build 5 compiled steps representing the SwiGLU MLP block.
    // Step 0: gate_proj Linear (input → intermediate)
    // Step 1: up_proj Linear (input → intermediate)
    // Step 2: silu(gate_proj output)
    // Step 3: mul(silu output, up_proj output)
    // Step 4: down_proj Linear (intermediate → d_model)
    let mut steps = vec![
        linear_step(&input_shape, intermediate), // 0: gate_proj
        linear_step(&input_shape, intermediate), // 1: up_proj
        silu_step(&intermediate_shape),          // 2: silu
        mul_step(&intermediate_shape),           // 3: mul
        linear_step(&intermediate_shape, 768),   // 4: down_proj
    ];

    // Graph topology:
    //   node 100 (external input x)
    //   node 0: gate_proj(100)  → intermediate
    //   node 1: up_proj(100)    → intermediate
    //   node 2: silu(0)         → intermediate  (consumes gate_proj)
    //   node 3: mul(2, 1)       → intermediate  (consumes silu + up_proj)
    //   node 4: down_proj(3)    → d_model       (consumes mul)
    let graph = ComputationGraph::from_nodes(vec![
        test_node(0, "gate_proj", vec![100], intermediate_shape.clone()),
        test_node(1, "up_proj", vec![100], intermediate_shape.clone()),
        test_node(2, "silu", vec![0], intermediate_shape.clone()),
        test_node(3, "mul", vec![2, 1], intermediate_shape),
        test_node(4, "down_proj", vec![3], input_shape),
    ]);

    // use_counts: how many times each step's output is consumed.
    // step 0 (gate_proj): consumed by silu → 1
    // step 1 (up_proj): consumed by mul → 1
    // step 2 (silu): consumed by mul → 1
    // step 3 (mul): consumed by down_proj → 1
    // step 4 (down_proj): terminal → 0
    let use_counts = vec![1, 1, 1, 1, 0];

    let dispatch_count_before = steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    assert_eq!(
        dispatch_count_before, 5,
        "SwiGLU MLP should start with 5 dispatches"
    );

    fuse_silu_mul(&mut steps, &use_counts, &graph);

    // After fusion:
    // step 0: gate_proj Linear   → unchanged (Dispatch)
    // step 1: up_proj Linear     → unchanged (Dispatch)
    // step 2: silu               → IdentityPassthrough (absorbed into silu_mul)
    // step 3: mul                → silu_mul Dispatch (fused kernel)
    // step 4: down_proj Linear   → unchanged (Dispatch)
    let dispatch_count_after = steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    let identity_count = steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::IdentityPassthrough))
        .count();

    // 5 dispatches → 4 dispatches + 1 IdentityPassthrough.
    // Dispatch reduction: 1 per SwiGLU MLP block.
    assert_eq!(
        dispatch_count_after, 4,
        "After fusion: 4 dispatches (gate_proj, up_proj, silu_mul, down_proj)"
    );
    assert_eq!(
        identity_count, 1,
        "After fusion: 1 IdentityPassthrough (absorbed silu)"
    );

    // Verify the specific step transformations.
    assert!(
        matches!(&steps[0], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
        "step 0 (gate_proj) should remain as linear Dispatch"
    );
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
        "step 1 (up_proj) should remain as linear Dispatch"
    );
    assert!(
        matches!(&steps[2], CompiledStep::IdentityPassthrough),
        "step 2 (silu) should become IdentityPassthrough"
    );
    match &steps[3] {
        CompiledStep::Dispatch {
            kernel,
            external_node_ids,
            ..
        } => {
            assert_eq!(kernel.name(), "silu_mul", "step 3 should be fused silu_mul");
            // external_node_ids = [gate=0 (gate_proj input to silu), up=1 (up_proj output)]
            // gate_id comes from silu's input (node 0's input = 100... but actually
            // gate_id = silu_node.inputs()[0] = node 0, and up_id = mul's other input = 1)
            let ids = external_node_ids
                .as_ref()
                .expect("silu_mul should have external_node_ids");
            assert_eq!(
                ids.len(),
                2,
                "silu_mul should have exactly 2 external inputs (gate, up)"
            );
            assert_eq!(ids[0], 0, "gate input should be node 0 (gate_proj output)");
            assert_eq!(ids[1], 1, "up input should be node 1 (up_proj output)");
        }
        other => panic!("step 3 should be silu_mul Dispatch, got: {other:?}"),
    }
    assert!(
        matches!(&steps[4], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
        "step 4 (down_proj) should remain as linear Dispatch"
    );
}
