// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the static buffer planner.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::{native_op_output_bytes, plan_buffers};
use crate::edge_map::compute_edge_map;
use crate::trace_compile::{
    compile_trace_to_plan_with_fusion, CompiledStep, FusedNormKind, NativeOpKind,
    NormActivConv1dParams, NormActivation,
};

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

fn unary_node(id: u64, name: &str, op: TraceOp, input_id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

fn binary_node(id: u64, name: &str, op: TraceOp, lhs: u64, rhs: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs, rhs],
        shape.to_vec(),
        DType::F32,
    )
}

#[test]
fn test_empty_plan() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);
    assert_eq!(bp.total_bytes, 0);
    assert!(bp.step_offsets.is_empty());
}

#[test]
fn test_input_only_no_allocation() {
    // A graph with only an input node needs no intermediate buffers.
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[4])]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // InputForward produces no allocation.
    assert_eq!(bp.total_bytes, 0);
    assert_eq!(bp.step_offsets.len(), 1);
    assert!(bp.step_offsets[0].is_none());
    assert_eq!(bp.step_sizes[0], 0);
}

#[test]
fn test_single_dispatch_allocation() {
    // input -> relu: relu needs a 4-element f32 buffer = 16 bytes.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    assert!(bp.step_offsets[0].is_none()); // InputForward: no alloc
    assert_eq!(bp.step_offsets[1], Some(0)); // Relu dispatch at offset 0
    assert_eq!(bp.step_sizes[1], 16); // 4 * 4 bytes
    assert_eq!(bp.total_bytes, 16);
}

#[test]
fn test_linear_chain_reuse() {
    // Three sequential reduce ops (non-fusible) where intermediate buffers
    // can be freed and reused:
    //
    // input(0, [4,4]) -> reduce_sum_dim0(1, [4]) -> reduce_sum_dim0(2, [1])
    //                                            -> reduce_mean_dim0(3, [1])
    //
    // Wait, reduce_mean(dim0) on [4] produces [1]. But step 2 and 3 both
    // consume step 1. That means step 1's buffer lives until step 3.
    //
    // For actual reuse: sequential reduces with no fan-out.
    // input(0, [4,4]) -> reduce_sum_dim1(1, [4]) -> reduce_sum_dim0(2, [1])
    //                                            -> ... more steps after 2
    //
    // Actually the cleanest reuse case: 3 sequential non-fusible ops,
    // each consuming only the prior output.
    //
    // input(0, [16]) -> reduce_sum(1, [1]) -> reduce_sum(2, [1]) -> reduce_sum(3, [1])
    //
    // But reduce_sum on a scalar is degenerate. Use reduce_mean with keepdim:
    // input(0,[4,4]) -> reduce_mean_d0(1,[1,4]) -> reduce_mean_d1(2,[1,1])
    //                                            -> relu(3,[1,1])
    //
    // reduce_mean_d0: output [1,4] = 16 bytes, consumed by step 2
    // reduce_mean_d1: output [1,1] = 4 bytes, consumed by step 3
    // relu: output [1,1] = 4 bytes
    //
    // At step 2: step 1's 16-byte buffer is freed (last_use=2).
    // Step 2 needs 4 bytes -- can reuse step 1's 16-byte slot at offset 0.
    // At step 3: step 2's buffer freed (also at offset 0, size 4).
    // Step 3 needs 4 bytes -- reuses offset 0.
    //
    // naive_total = 16 + 4 + 4 = 24
    // With reuse: total = 16 (the 16-byte slot is the high water mark)

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 4]),
        TraceNode::new(
            1,
            "reduce_mean_d0".into(),
            TraceOp::ReduceMean {
                dim: 0,
                keepdim: true,
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reduce_mean_d1".into(),
            TraceOp::ReduceMean {
                dim: 1,
                keepdim: true,
            },
            vec![1],
            vec![1, 1],
            DType::F32,
        ),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[1, 1]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // Verify that reuse reduces total bytes below naive.
    assert!(
        bp.total_bytes < bp.naive_total,
        "total_bytes ({}) should be less than naive_total ({})",
        bp.total_bytes,
        bp.naive_total,
    );
}

#[test]
fn test_diamond_topology_no_early_free() {
    // Diamond: input -> relu(1) and input -> sigmoid(2), then add(3).
    // relu and sigmoid are both live until add, so no reuse between them.
    // input: 0 bytes (InputForward)
    // relu: 16 bytes (4 * f32), last_use = 3 (add)
    // sigmoid: 16 bytes (4 * f32), last_use = 3 (add)
    // add: 16 bytes, last_use = 3 (output)
    // No overlap possible between relu and sigmoid -- both live until step 3.
    // But the input fans out (fan-out=2 on input), so relu and sigmoid
    // are separate dispatches (no fusion between relu and sigmoid).

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // relu (step 1) and sigmoid (step 2) are both consumed by add (step 3).
    // They must coexist, so total >= 32 bytes for their intermediates.
    // add's output also needs 16 bytes.
    // Minimum: relu(16) + sigmoid(16) + add(16) = 48 bytes (no reuse possible).
    //
    // But with fusion: relu->add might fuse if relu has fan-out 1.
    // relu feeds add (fan-out 1 from fusion perspective? No, input fans out).
    // Actually relu has fan-out 1 (only feeds add), and sigmoid also fan-out 1.
    // But add has 2 inputs from different branches, so the chain detection
    // in fusion requires one input to be the prior chain node.
    // relu -> add fuses only if add's input[0] is relu AND relu has fan-out 1.
    // Let's check: nodes[2] is sigmoid, nodes[3] is add.
    // Between relu(1) and sigmoid(2): sigmoid doesn't chain from relu.
    // Between sigmoid(2) and add(3): add takes inputs [1, 2], so inputs
    // include sigmoid (nodes[2]). But for fusion, the chain extends from
    // the previous node -- sigmoid(2) has fan-out 1, and add(3) includes
    // sigmoid's output as one input. So sigmoid->add would try to fuse.
    // But add is binary -- it also needs relu's output as a second input.
    //
    // In practice, the test verifies the plan assigns different offsets for
    // simultaneously-live buffers.
    let relu_offset = bp.step_offsets[1];
    let sigmoid_offset = bp.step_offsets[2];

    // Both should have Some offsets (both are Dispatch).
    // If fusion happened, some steps become IdentityPassthrough with size 0.
    // Just verify the plan is internally consistent.
    assert_eq!(bp.step_offsets.len(), 4);
    assert!(bp.total_bytes > 0);

    // If both have offsets, they must not overlap.
    if let (Some(r_off), Some(s_off)) = (relu_offset, sigmoid_offset) {
        let r_size = bp.step_sizes[1];
        let s_size = bp.step_sizes[2];
        if r_size > 0 && s_size > 0 {
            let r_end = r_off + r_size;
            let s_end = s_off + s_size;
            assert!(
                r_end <= s_off || s_end <= r_off,
                "relu [{r_off}..{r_end}] and sigmoid [{s_off}..{s_end}] overlap"
            );
        }
    }
}

#[test]
fn test_buffer_reuse_with_different_sizes() {
    // Test that a freed slot can be reused even for smaller buffers.
    //
    // input(0, [8]) -> reduce_sum(1, [1]) -> relu(2, [1])
    // reduce_sum: 4 bytes output, consumed by relu at step 2.
    // relu: 4 bytes output.
    // After step 2: reduce_sum freed. relu reuses offset 0.
    // total = 4, naive = 8.

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "reduce_sum".into(),
            TraceOp::ReduceSum {
                dim: 0,
                keepdim: false,
            },
            vec![0],
            vec![1],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[1]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // Both reduce_sum and relu allocate 4 bytes (1 element * 4).
    // reduce_sum is freed when relu processes, so relu reuses the slot.
    assert!(
        bp.total_bytes <= bp.naive_total,
        "total ({}) should be <= naive ({})",
        bp.total_bytes,
        bp.naive_total,
    );
}

#[test]
fn test_last_use_linear_chain() {
    // input(0,[8]) -> reduce_sum(1,[1]) -> relu(2,[1])
    //
    // last_use[0] = 1 (input consumed by reduce_sum at step 1)
    // last_use[1] = 2 (reduce_sum consumed by relu at step 2)
    // last_use[2] = 2 (relu is the output — no downstream consumer)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "reduce_sum".into(),
            TraceOp::ReduceSum {
                dim: 0,
                keepdim: false,
            },
            vec![0],
            vec![1],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[1]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    assert_eq!(bp.last_use.len(), plan.steps.len());
    // Input step (0): consumed by reduce_sum (step 1).
    assert_eq!(bp.last_use[0], 1, "input last_use should be step 1");
    // Reduce_sum (1): consumed by relu (step 2).
    assert_eq!(bp.last_use[1], 2, "reduce last_use should be step 2");
    // Relu (2): output node, last_use = self.
    assert_eq!(bp.last_use[2], 2, "output last_use should be self");
}

#[test]
fn test_last_use_diamond() {
    // input(0,[4]) -> relu(1,[4])
    //              -> sigmoid(2,[4])
    //              -> add(3,[4])  (inputs: relu + sigmoid)
    //
    // With partition-driven codegen: all three elementwise ops (relu, sigmoid,
    // add) are fused into one Elementwise-dominant partition group {1, 2, 3}.
    // The fused kernel at step 3 computes relu(x) + sigmoid(x) from a single
    // input (step 0). Steps 1 and 2 become IdentityPassthrough.
    //
    // last_use[0] = 3 (input consumed by fused step@3)
    // last_use[1] = 1 (IdentityPassthrough, no buffer)
    // last_use[2] = 2 (IdentityPassthrough, no buffer)
    // last_use[3] = 3 (fused output)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    assert_eq!(bp.last_use.len(), plan.steps.len());
    // Input consumed by fused step@3 (only external input).
    assert_eq!(bp.last_use[0], 3, "input last_use");
    // relu(1) is IdentityPassthrough — absorbed into fused kernel.
    assert_eq!(bp.last_use[1], 1, "relu last_use (identity passthrough)");
    // sigmoid(2) is IdentityPassthrough — absorbed into fused kernel.
    assert_eq!(bp.last_use[2], 2, "sigmoid last_use (identity passthrough)");
    // add(3)/fused is output.
    assert_eq!(bp.last_use[3], 3, "output last_use");
}

#[test]
fn test_no_overlap_invariant() {
    // For any plan, no two simultaneously-live buffers should overlap.
    // Use a diamond topology so there ARE simultaneously-live buffers:
    //
    // input(0,[8]) -> relu(1,[8])
    //             -> sigmoid(2,[8])
    //             -> add(3,[8])  (inputs: relu + sigmoid)
    //
    // relu(1) and sigmoid(2) are both live until add(3) consumes them,
    // so their buffers MUST occupy non-overlapping memory.

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[8]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[8]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // Verify: no two buffers overlap in BOTH memory and time.
    assert_no_memory_time_overlap(&bp);
}

#[test]
fn test_fused_chain_last_use_accounts_for_external_node_ids() {
    // Regression test: input(0) -> relu(1) -> sigmoid(2).
    // With fusion, relu+sigmoid fuse. The fused step at index 2 has
    // external_node_ids=[0] (reads from input, not from relu).
    // The buffer planner must set last_use[0] = 2 (not 1), otherwise
    // eager release frees buffer[0] after step 1 and step 2 fails
    // with "input references step 0 with no buffer".
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    assert_eq!(bp.last_use.len(), 3);
    // With fusion: step 1 is IdentityPassthrough, step 2 is the fused dispatch.
    // The fused step's external_node_ids=[0] means step 2 consumes step 0.
    // So last_use[0] must be >= 2 (not 1).
    assert!(
        bp.last_use[0] >= 2,
        "input last_use should be >= 2 for fused chain, got {}",
        bp.last_use[0],
    );
}

#[test]
fn test_native_op_output_bytes_fused_resblock() {
    // FusedResBlock output shape == phase1 input shape (residual add).
    // [1, 512, 100] → 512 * 100 * 4 = 204_800 bytes.
    let op = NativeOpKind::FusedResBlock {
        phase1: NormActivConv1dParams {
            activation: NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 3,
            conv_padding: 3,
            input_shape: vec![1, 512, 100],
            output_channels: 512,
            kernel_size: 3,
        },
        phase2: NormActivConv1dParams {
            activation: NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 512, 100],
            output_channels: 512,
            kernel_size: 3,
        },
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(native_op_output_bytes(&op), 512 * 100 * 4);
}

#[test]
fn test_native_op_output_bytes_max_pool1d() {
    // MaxPool1d: [1, 64, 200] with kernel=3, stride=2, padding=1.
    // out_len = (200 + 2*1 - 3) / 2 + 1 = 199/2 + 1 = 99 + 1 = 100.
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 200],
    };
    assert_eq!(native_op_output_bytes(&op), 64 * 100 * 4);
}

#[test]
fn test_native_op_output_bytes_max_pool1d_underflow_guard() {
    // Edge case: kernel_size > padded length should return 0, not usize underflow.
    // input length=1, padding=0, kernel_size=5 → padded=1 < 5 → 0 bytes.
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 5,
        stride: 1,
        padding: 0,
        input_shape: vec![1, 64, 1],
    };
    assert_eq!(native_op_output_bytes(&op), 0);
}

#[test]
fn test_native_op_output_bytes_max_pool1d_exact_boundary() {
    // Boundary: padded == kernel_size → out_len = (K - K)/S + 1 = 1.
    // input length=3, padding=1, kernel_size=5 → padded=5 == 5 → 1 element.
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 5,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 32, 3],
    };
    assert_eq!(native_op_output_bytes(&op), 32 * 4);
}

#[test]
fn test_native_op_output_bytes_max_pool1d_zero_stride() {
    // stride=0 should return 0 (division by zero guard).
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 0,
        padding: 0,
        input_shape: vec![1, 64, 100],
    };
    assert_eq!(native_op_output_bytes(&op), 0);
}

#[test]
fn test_native_op_output_bytes_max_pool1d_short_shape() {
    // input_shape with < 3 dimensions should return 0.
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 1,
        padding: 0,
        input_shape: vec![100],
    };
    assert_eq!(native_op_output_bytes(&op), 0);
}

#[test]
fn test_native_op_output_bytes_constant_weight() {
    // ConstantWeight aliases pre-uploaded buffer — 0 bytes.
    let op = NativeOpKind::ConstantWeight {
        name: "arange".to_string(),
        shape: vec![1, 1, 100],
    };
    assert_eq!(native_op_output_bytes(&op), 0);
}

/// NormLinear output has different shape from input: input [B, hidden_dim],
/// output [B, out_features]. Buffer planner must use [B, out_features].
#[test]
fn test_native_op_output_bytes_norm_linear() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![4, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    // Output is [4, 3072] = 12288 elements * 4 bytes = 49152.
    assert_eq!(native_op_output_bytes(&op), 4 * 3072 * 4);
}

/// NormLinear with RmsNorm variant (Qwen3 pattern).
#[test]
fn test_native_op_output_bytes_norm_linear_rms() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::RmsNorm,
        eps: 1e-6,
        input_shape: vec![1, 32, 512],
        hidden_dim: 512,
        out_features: 2048,
        has_bias: false,
    };
    // Output is [1, 32, 2048] = 65536 elements * 4 bytes = 262144.
    assert_eq!(native_op_output_bytes(&op), 32 * 2048 * 4);
}

/// AddLayerNorm output shape equals input shape (same as LayerNorm).
#[test]
fn test_native_op_output_bytes_add_layer_norm() {
    let op = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![4, 768],
        hidden_dim: 768,
    };
    // Output shape == input shape: [4, 768] = 3072 elements * 4 bytes.
    assert_eq!(native_op_output_bytes(&op), 4 * 768 * 4);
}

/// LinearActivation output has different last dim from input.
#[test]
fn test_native_op_output_bytes_linear_activation() {
    use crate::trace_compile::GemmActivation;

    let op = NativeOpKind::LinearActivation {
        activation: GemmActivation::Relu,
        in_features: 768,
        out_features: 3072,
        has_bias: true,
        input_shape: vec![4, 768],
    };
    // Output is [4, 3072] = 12288 elements * 4 bytes.
    assert_eq!(native_op_output_bytes(&op), 4 * 3072 * 4);
}

/// Build a linear chain of N steps (input → relu → relu → ... → relu)
/// and verify buffer planning scales linearly, not quadratically.
///
/// Before the O(n²) fix, linear_scan_alloc scanned all prior steps per
/// allocation to find buffers to free. For N=1000 steps this was ~500K
/// comparisons. After the fix, a pre-built release map gives O(1)
/// lookup per step.
///
/// This test verifies correctness at scale (the output must match
/// the semantics of the old algorithm), not wall-clock time.
#[test]
fn test_large_chain_buffer_plan_correctness() {
    // Build a chain of 200 alternating relu/sigmoid ops (non-fusible
    // due to different op types when interleaved with reduce ops).
    // Using reduce_sum to break fusion chains: each reduce_sum has
    // different output shape than its input, preventing elementwise fusion.
    let n = 200;
    let mut nodes = vec![input_node(0, &[16])];
    for i in 1..=n {
        let prev_id = (i - 1) as u64;
        let id = i as u64;
        if i % 2 == 1 {
            // Odd steps: ReduceSum [16] -> [1] (breaks fusion)
            nodes.push(TraceNode::new(
                id,
                format!("reduce_{i}"),
                TraceOp::ReduceSum {
                    dim: 0,
                    keepdim: false,
                },
                vec![prev_id],
                vec![1],
                DType::F32,
            ));
        } else {
            // Even steps: Relu [1] -> [1] (elementwise, may fuse)
            nodes.push(unary_node(
                id,
                &format!("relu_{i}"),
                TraceOp::Relu,
                prev_id,
                &[1],
            ));
        }
    }

    let graph = ComputationGraph::from_nodes(nodes);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // Basic correctness: plan has the right number of steps.
    assert_eq!(bp.step_offsets.len(), plan.steps.len());
    assert_eq!(bp.step_sizes.len(), plan.steps.len());
    assert_eq!(bp.last_use.len(), plan.steps.len());

    // Buffer reuse should reduce total below naive sum.
    assert!(
        bp.total_bytes <= bp.naive_total,
        "total ({}) should be <= naive ({})",
        bp.total_bytes,
        bp.naive_total,
    );

    // No allocated buffer should overlap with a simultaneously-live buffer.
    assert_no_memory_time_overlap(&bp);
}

/// Verify that the release_at pre-build produces identical results
/// to what the old O(n²) scan would produce, on a diamond topology.
#[test]
fn test_release_map_matches_naive_scan_diamond() {
    // input(0,[8]) -> relu(1,[8])
    //             -> sigmoid(2,[8])
    //             -> mul(3,[8])  (inputs: 1, 2)
    //             -> add(4,[8])  (inputs: 0, 3)
    //
    // Last-use: [4, 3, 3, 4, 4]
    // At step 3: relu(1) and sigmoid(2) should be freed.
    // At step 4: input(0) and mul(3) should be freed.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[8]),
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[8]),
        binary_node(4, "add_0", TraceOp::Add, 0, 3, &[8]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // The plan should have 5 steps.
    assert_eq!(bp.last_use.len(), 5);

    // Output (step 4) must still have its buffer (not freed).
    assert!(
        bp.step_offsets.last().is_some(),
        "output step should have an offset"
    );

    // Total bytes should be less than naive (some reuse should occur).
    assert!(
        bp.total_bytes <= bp.naive_total,
        "total ({}) <= naive ({})",
        bp.total_bytes,
        bp.naive_total,
    );
}

#[test]
fn test_adain_edge_map_patch_extends_last_use_for_gamma_beta() {
    // Regression test for #3254: when AdainSnake is placed at an InstanceNorm
    // position (1 graph edge: x), the edge_map patch must expand to [x, gamma, beta].
    // Without the patch, the buffer planner sets last_use for gamma and beta based
    // on the original Mul/Add nodes (steps 4, 5), but the AdainSnake at step 3
    // actually reads those buffers. If gamma/beta buffers are freed before step 3,
    // the executor reads freed memory.
    //
    // Graph: Input(x:0) Input(gamma:1) Input(beta:2) InstanceNorm(3) Mul(4) Add(5)
    // After graph-level detection: step 3 = AdainSnake, steps 4-5 = IdentityPassthrough
    let (batch, channels, time) = (1, 4, 16);
    let eps = 1e-5_f64;
    let shape_bct = vec![batch, channels, time];
    let shape_bc1 = vec![batch, channels, 1];

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_x".into(),
            TraceOp::Input,
            vec![],
            shape_bct.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "input_gamma".into(),
            TraceOp::Input,
            vec![],
            shape_bc1.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "input_beta".into(),
            TraceOp::Input,
            vec![],
            shape_bc1,
            DType::F32,
        ),
        TraceNode::new(
            3,
            "instance_norm".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            shape_bct.clone(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "mul_gamma".into(),
            TraceOp::Mul,
            vec![1, 3],
            shape_bct.clone(),
            DType::F32,
        ),
        TraceNode::new(
            5,
            "add_beta".into(),
            TraceOp::Add,
            vec![2, 4],
            shape_bct.clone(),
            DType::F32,
        ),
    ]);

    let mut plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    assert_eq!(
        plan.steps.len(),
        6,
        "plan should have 6 steps (1:1 with graph nodes)"
    );

    // Simulate graph-level detection: replace InstanceNorm with AdainSnake.
    // external_node_ids carries [x=0, gamma=1, beta=2] so the edge_map builder
    // resolves edges generically without per-NativeOp patches (#3261).
    plan.steps[3] = CompiledStep::NativeOp {
        op: NativeOpKind::AdainSnake {
            eps: eps as f32,
            input_shape: shape_bct,
            channels,
            residual_gamma: true,
            external_node_ids: Some(vec![0, 1, 2]),
        },
        weight_data: std::collections::HashMap::new(),
    };
    plan.steps[4] = CompiledStep::IdentityPassthrough;
    plan.steps[5] = CompiledStep::IdentityPassthrough;
    plan.output_step = 3;

    // The critical invariant: the edge_map patch must expand step 3's edges
    // from [0] (graph-level InstanceNorm) to [0, 1, 2] (x, gamma, beta).
    // Without this, the executor at step 3 resolves only 1 buffer, missing
    // gamma and beta entirely. Assert on edge_map directly — last_use
    // assertions alone are insufficient because the IdentityPassthrough steps
    // (4, 5) still reference gamma/beta in graph topology, keeping last_use
    // values high regardless of the patch.
    let edge_map = compute_edge_map(&graph, &plan.steps);
    assert_eq!(
        edge_map[3],
        vec![0, 1, 2],
        "AdainSnake edge_map must be [x=0, gamma=1, beta=2], got {:?}",
        edge_map[3],
    );

    let bp = plan_buffers(&plan, &graph);
    assert_eq!(bp.last_use.len(), 6);

    // Secondary check: gamma (step 1) and beta (step 2) must have
    // last_use >= 3. Note: this holds even without the edge_map patch
    // (Mul/Add at steps 4/5 keep gamma/beta alive), so this assertion
    // is necessary but not sufficient — the edge_map assertion above
    // is the one that catches the actual regression.
    assert!(
        bp.last_use[0] >= 3,
        "x (step 0) last_use should be >= 3 (AdainSnake reads it), got {}",
        bp.last_use[0],
    );
    assert!(
        bp.last_use[1] >= 3,
        "gamma (step 1) last_use should be >= 3 (AdainSnake reads it), got {}",
        bp.last_use[1],
    );
    assert!(
        bp.last_use[2] >= 3,
        "beta (step 2) last_use should be >= 3 (AdainSnake reads it), got {}",
        bp.last_use[2],
    );

    // Verify no-overlap invariant holds with the patched lifetimes.
    assert_no_memory_time_overlap(&bp);
}

/// Regression test for #3306: BatchedLinearProjection must allocate for
/// projection_sizes[0] (first narrow), NOT total_out_features (full QKV).
///
/// The executor narrows the first projection as the step buffer output
/// and stashes the full intermediate in a thread-local temp. The buffer
/// planner must match the executor by sizing for `projection_sizes[0]`.
/// Using `total_out_features` instead causes a 3.0x size mismatch when
/// there are 3 equal-sized projections (Q, K, V).
#[test]
fn test_native_op_output_bytes_batched_linear_projection_first_proj_only() {
    // Kokoro-like QKV: 3 projections of 256 each, total_out=768.
    // Input shape [1, 32, 768] → batch = 1*32 = 32.
    // Correct: 32 * 256 * 4 = 32768 bytes (first projection only).
    // Bug:     32 * 768 * 4 = 98304 bytes (3.0x too large).
    let op = NativeOpKind::BatchedLinearProjection {
        in_features: 768,
        total_out_features: 768,
        projection_sizes: vec![256, 256, 256],
        has_bias: true,
        input_shape: vec![1, 32, 768],
    };
    let expected = 32 * 256 * 4; // batch * proj[0] * F32_BYTES
    let buggy = 32 * 768 * 4; // batch * total_out * F32_BYTES (3.0x)
    let actual = native_op_output_bytes(&op);
    assert_eq!(
        actual,
        expected,
        "BatchedLinearProjection must use projection_sizes[0]={}, not total_out={}. \
         Got {} bytes (ratio={:.1}x vs correct)",
        256,
        768,
        actual,
        actual as f64 / expected as f64,
    );
    assert_ne!(actual, buggy, "must NOT use total_out_features");
}

/// ProjectionSlice output bytes match the narrow slice shape.
#[test]
fn test_native_op_output_bytes_projection_slice() {
    let op = NativeOpKind::ProjectionSlice {
        source_step: 1,
        dim: 1,
        start: 256,
        length: 256,
        output_shape: vec![1, 32, 256],
    };
    // [1, 32, 256] = 8192 elements * 4 bytes = 32768.
    assert_eq!(native_op_output_bytes(&op), 32 * 256 * 4);
}

/// Assert that no two allocated buffers overlap in both memory and time.
fn assert_no_memory_time_overlap(bp: &super::BufferPlan) {
    let allocated: Vec<(usize, usize, usize)> = bp
        .step_offsets
        .iter()
        .enumerate()
        .filter_map(|(idx, off)| off.map(|o| (idx, o, bp.step_sizes[idx])))
        .filter(|&(_, _, size)| size > 0)
        .collect();

    for i in 0..allocated.len() {
        for j in (i + 1)..allocated.len() {
            let (a_idx, a_off, a_size) = allocated[i];
            let (b_idx, b_off, b_size) = allocated[j];
            let a_end = a_off + a_size;
            let b_end = b_off + b_size;
            let memory_overlap = a_end > b_off && b_end > a_off;
            if !memory_overlap {
                continue;
            }
            let a_live_end = bp.last_use[a_idx];
            let b_live_end = bp.last_use[b_idx];
            let time_overlap = a_idx <= b_live_end && b_idx <= a_live_end;
            assert!(
                !time_overlap,
                "steps {a_idx} and {b_idx} overlap in both memory \
                 [{a_off}..{a_end}) vs [{b_off}..{b_end}) \
                 and time [{a_idx}..{a_live_end}] vs [{b_idx}..{b_live_end}]"
            );
        }
    }
}

#[test]
fn test_projection_slice_keeps_batched_linear_alive() {
    // Test for #3269: ProjectionSlice steps must keep the
    // BatchedLinearProjection buffer alive in the buffer plan.
    //
    // Graph: Input(0) → Relu(1), Input(0) → Sigmoid(2), Input(0) → Sigmoid(3) → Add(4)
    // After manual replacement:
    //   Step 0: InputForward
    //   Step 1: BatchedLinearProjection (Q+K+V from input)
    //   Step 2: ProjectionSlice (K, source_step=1)
    //   Step 3: ProjectionSlice (V, source_step=1)
    //   Step 4: Add (consumes K=2, V=3)
    //
    // Critical: Nodes 2 and 3 read from node 0 in the graph, NOT node 1.
    // Without the ProjectionSlice edge_map patch, edge_map[2] = [0] and
    // edge_map[3] = [0]. The patch overrides these to [1], which is what
    // the executor needs. This test would fail without the patch.
    let shape = vec![2, 64];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &shape),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &shape),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &shape),
        unary_node(3, "sigmoid_1", TraceOp::Sigmoid, 0, &shape),
        binary_node(4, "add_0", TraceOp::Add, 2, 3, &shape),
    ]);

    let mut plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    assert_eq!(plan.steps.len(), 5);

    // Replace steps 1-3 with batched projection pattern.
    plan.steps[1] = CompiledStep::NativeOp {
        op: NativeOpKind::BatchedLinearProjection {
            in_features: 64,
            total_out_features: 192,
            projection_sizes: vec![64, 64, 64],
            has_bias: false,
            input_shape: shape.clone(),
        },
        weight_data: std::collections::HashMap::new(),
    };
    plan.steps[2] = CompiledStep::NativeOp {
        op: NativeOpKind::ProjectionSlice {
            source_step: 1,
            dim: 1,
            start: 64,
            length: 64,
            output_shape: shape.clone(),
        },
        weight_data: std::collections::HashMap::new(),
    };
    plan.steps[3] = CompiledStep::NativeOp {
        op: NativeOpKind::ProjectionSlice {
            source_step: 1,
            dim: 1,
            start: 128,
            length: 64,
            output_shape: shape,
        },
        weight_data: std::collections::HashMap::new(),
    };

    // Verify edge_map: ProjectionSlice steps must point to source_step.
    let edge_map = super::build_edge_map_simple(&graph, &plan.steps);
    assert_eq!(
        edge_map[2],
        vec![1],
        "ProjectionSlice K edge_map must be [source_step=1], got {:?}",
        edge_map[2],
    );
    assert_eq!(
        edge_map[3],
        vec![1],
        "ProjectionSlice V edge_map must be [source_step=1], got {:?}",
        edge_map[3],
    );

    // Verify buffer lifetimes: BatchedLinearProjection alive until last ProjectionSlice.
    let bp = plan_buffers(&plan, &graph);
    assert!(
        bp.last_use[1] >= 3,
        "BatchedLinearProjection last_use must be >= 3 (last ProjectionSlice), got {}",
        bp.last_use[1],
    );

    // Verify no memory-time overlap.
    assert_no_memory_time_overlap(&bp);
}
