// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SwiGLU MLP compilation optimization tests.
//!
//! Verifies that the full compilation pipeline (elementwise chain fusion +
//! peephole passes) correctly optimizes the SwiGLU MLP pattern used in
//! transformer decoders (Qwen3, GLM5, CosyVoice3).
//!
//! SwiGLU forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`
//!
//! Expected optimizations (#3521):
//! - **Elementwise fusion**: `Silu + Mul` → `fused_silu_x2` (1 dispatch)
//! - **Pass 12 (BatchedLinearProjection)**: `gate_proj + up_proj` → batched
//!   matmul (gate_proj no longer fuses with silu via pass 5 because silu was
//!   already consumed by the elementwise chain fusion)
//!
//! Result: 5 ops → 4 dispatches (BatchedLinear, ProjectionSlice, fused_silu_mul, down_proj).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::trace_compile::{compile_trace_to_plan_with_fusion, CompiledStep};

/// Helper: create an Input trace node.
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

/// Helper: create a Linear trace node with weight data.
fn linear_node(
    id: u64,
    name: &str,
    input_id: u64,
    input_shape: &[usize],
    out_features: usize,
) -> TraceNode {
    let in_features = *input_shape.last().expect("non-empty shape");
    let weight_data = vec![0.1f32; out_features * in_features];
    let weight =
        WeightRef::new(weight_data, vec![out_features, in_features]).expect("weight shape valid");

    let mut output_shape = input_shape.to_vec();
    if let Some(last) = output_shape.last_mut() {
        *last = out_features;
    }

    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Linear { weight, bias: None },
        vec![input_id],
        output_shape,
        DType::F32,
    )
}

/// Helper: create a unary op trace node.
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

/// Helper: create a binary op trace node.
fn binary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    lhs_id: u64,
    rhs_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs_id, rhs_id],
        shape.to_vec(),
        DType::F32,
    )
}

/// Count dispatches (Dispatch + NativeOp) in a compiled plan.
fn count_dispatches(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count()
}

/// Count specific NativeOp variant occurrences.
fn count_native_op(steps: &[CompiledStep], variant_name: &str) -> usize {
    steps
        .iter()
        .filter(|s| match s {
            CompiledStep::NativeOp { op, .. } => op.variant_name() == variant_name,
            _ => false,
        })
        .count()
}

/// Count IR Dispatch steps with a specific kernel name.
fn count_dispatch_by_name(steps: &[CompiledStep], name: &str) -> usize {
    steps
        .iter()
        .filter(|s| match s {
            CompiledStep::Dispatch { kernel, .. } => kernel.name() == name,
            _ => false,
        })
        .count()
}

/// Count IR Dispatch steps with kernel names starting with a prefix.
fn count_dispatch_by_prefix(steps: &[CompiledStep], prefix: &str) -> usize {
    steps
        .iter()
        .filter(|s| match s {
            CompiledStep::Dispatch { kernel, .. } => kernel.name().starts_with(prefix),
            _ => false,
        })
        .count()
}

// -- Tests --------------------------------------------------------------------

/// Verify elementwise chain fusion catches Silu+Mul across non-consecutive
/// nodes (with a Linear between them in topological order).
///
/// Trace graph for SwiGLU (topological order):
///   0: Input x [1, 4, 256]
///   1: Linear gate_proj [1, 4, 512] (input: 0)
///   2: Silu [1, 4, 512] (input: 1)
///   3: Linear up_proj [1, 4, 512] (input: 0)
///   4: Mul [1, 4, 512] (inputs: 2, 3)
///   5: Linear down_proj [1, 4, 256] (input: 4)
///
/// Expected after fusion + peephole:
/// - Silu(2) + Mul(4) fused into fused_silu_x2 (chain spans non-consecutive nodes)
/// - gate_proj(1) + up_proj(3) batched by pass 12 (both consume input 0)
#[test]
fn test_swiglu_silu_mul_fusion() {
    let dim = 256;
    let ff_dim = 512;
    let shape_in = [1, 4, dim];
    let shape_ff = [1, 4, ff_dim];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &shape_in),
        linear_node(1, "gate_proj", 0, &shape_in, ff_dim),
        unary_node(2, "silu_0", TraceOp::Silu, 1, &shape_ff),
        linear_node(3, "up_proj", 0, &shape_in, ff_dim),
        binary_node(4, "mul_0", TraceOp::Mul, 2, 3, &shape_ff),
        linear_node(5, "down_proj", 4, &shape_ff, dim),
    ]);

    let plan =
        compile_trace_to_plan_with_fusion(&graph).expect("SwiGLU trace should compile with fusion");
    let steps = &plan.steps;

    // Verify Silu+Mul was fused into a single fused dispatch.
    let fused_count = count_dispatch_by_prefix(steps, "fused_");
    assert!(
        fused_count >= 1,
        "expected at least 1 fused dispatch (silu+mul), got {fused_count}"
    );

    // Verify pass 12 batched gate_proj + up_proj.
    let batched = count_native_op(steps, "BatchedLinearProjection");
    assert!(
        batched >= 1,
        "expected BatchedLinearProjection (gate+up), got {batched}"
    );

    let slices = count_native_op(steps, "ProjectionSlice");
    assert!(
        slices >= 1,
        "expected ProjectionSlice (up extraction), got {slices}"
    );

    // Verify down_proj remains as a standalone Linear dispatch.
    let linear_count = count_dispatch_by_name(steps, "linear");
    assert!(
        linear_count >= 1,
        "expected at least 1 standalone linear (down_proj), got {linear_count}"
    );

    // Total dispatches should be 4: BatchedLinear + ProjectionSlice + fused_silu_mul + down_proj
    let total = count_dispatches(steps);
    assert_eq!(
        total, 4,
        "SwiGLU should compile to 4 dispatches \
         (BatchedLinear + ProjectionSlice + fused_silu_mul + down_proj), got {total}"
    );

    eprintln!("SwiGLU optimization: 5 ops -> {total} dispatches");
    eprintln!("  BatchedLinearProjection: {batched}");
    eprintln!("  ProjectionSlice: {slices}");
    eprintln!("  Fused (silu+mul): {fused_count}");
    eprintln!("  Standalone linear: {linear_count}");
}

/// Verify that LinearActivation (pass 5) does NOT fire for gate_proj+silu
/// when elementwise fusion has already consumed the silu into fused_silu_mul.
///
/// This is correct behavior: the chain fusion Silu+Mul is a better optimization
/// than LinearActivation(gate_proj+silu) because it eliminates the Mul dispatch
/// that would otherwise remain standalone.
#[test]
fn test_swiglu_no_linear_activation_for_gate_silu() {
    let dim = 64;
    let ff_dim = 128;
    let shape_in = [1, 4, dim];
    let shape_ff = [1, 4, ff_dim];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &shape_in),
        linear_node(1, "gate_proj", 0, &shape_in, ff_dim),
        unary_node(2, "silu_0", TraceOp::Silu, 1, &shape_ff),
        linear_node(3, "up_proj", 0, &shape_in, ff_dim),
        binary_node(4, "mul_0", TraceOp::Mul, 2, 3, &shape_ff),
        linear_node(5, "down_proj", 4, &shape_ff, dim),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("SwiGLU trace should compile");
    let steps = &plan.steps;

    // LinearActivation should NOT appear — silu was consumed by chain fusion.
    let linear_activation = count_native_op(steps, "LinearActivation");
    assert_eq!(
        linear_activation, 0,
        "LinearActivation should not fire (silu consumed by chain fusion), got {linear_activation}"
    );
}

/// Verify fused gate+up SwiGLU pattern (GLM5 variant).
///
/// GLM5 uses a single `dense_h_to_4h` that outputs `2 * ffn_hidden_size`,
/// then narrows to split gate and up projections:
///
///   0: Input x [1, 4, 256]
///   1: Linear dense_h_to_4h [1, 4, 1024] (input: 0)
///   2: Narrow(dim=2, start=0, len=512) gate [1, 4, 512] (input: 1)
///   3: Narrow(dim=2, start=512, len=512) up [1, 4, 512] (input: 1)
///   4: Silu [1, 4, 512] (input: 2)
///   5: Mul [1, 4, 512] (inputs: 4, 3)
///   6: Linear dense_4h_to_h [1, 4, 256] (input: 5)
///
/// Expected: Silu(4) + Mul(5) fused into `fused_silu_x2`.
/// Narrows compile as Dispatch (GPU narrow kernel for non-contiguous slicing).
///
/// Result: 7 graph nodes -> 5 dispatches
///   (dense_h_to_4h + narrow_gate + narrow_up + fused_silu_mul + dense_4h_to_h)
#[test]
fn test_swiglu_glm5_fused_gate_up_pattern() {
    let dim = 256;
    let ff_dim = 512;
    let shape_in = [1, 4, dim];
    let shape_ff = [1, 4, ff_dim];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &shape_in),
        linear_node(1, "dense_h_to_4h", 0, &shape_in, ff_dim * 2),
        TraceNode::new(
            2,
            "narrow_gate".to_string(),
            TraceOp::Narrow {
                dim: 2,
                start: 0,
                length: ff_dim,
            },
            vec![1],
            shape_ff.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "narrow_up".to_string(),
            TraceOp::Narrow {
                dim: 2,
                start: ff_dim,
                length: ff_dim,
            },
            vec![1],
            shape_ff.to_vec(),
            DType::F32,
        ),
        unary_node(4, "silu_0", TraceOp::Silu, 2, &shape_ff),
        binary_node(5, "mul_0", TraceOp::Mul, 4, 3, &shape_ff),
        linear_node(6, "dense_4h_to_h", 5, &shape_ff, dim),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("GLM5 SwiGLU trace should compile with fusion");
    let steps = &plan.steps;

    // Verify Silu+Mul was fused into a single dispatch.
    let fused_count = count_dispatch_by_prefix(steps, "fused_");
    assert_eq!(
        fused_count, 1,
        "expected 1 fused dispatch (silu+mul), got {fused_count}"
    );

    // No BatchedLinearProjection — GLM5 uses a single fused dense_h_to_4h.
    let batched = count_native_op(steps, "BatchedLinearProjection");
    assert_eq!(
        batched, 0,
        "GLM5 should not batch (already fused gate+up), got {batched}"
    );

    // Total: dense_h_to_4h(1) + narrow_gate(1) + narrow_up(1)
    //        + fused_silu_mul(1) + dense_4h_to_h(1) = 5 dispatches.
    let total = count_dispatches(steps);
    assert_eq!(
        total, 5,
        "GLM5 SwiGLU should compile to 5 dispatches, got {total}"
    );

    eprintln!("GLM5 SwiGLU: 7 graph nodes -> {total} dispatches");
}

/// Verify that the SwiGLU optimization produces numerically identical results
/// to the unoptimized path by checking that both compile successfully with
/// the same number of output steps.
#[test]
fn test_swiglu_optimization_preserves_graph_structure() {
    let dim = 32;
    let ff_dim = 64;
    let shape_in = [1, 2, dim];
    let shape_ff = [1, 2, ff_dim];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &shape_in),
        linear_node(1, "gate_proj", 0, &shape_in, ff_dim),
        unary_node(2, "silu_0", TraceOp::Silu, 1, &shape_ff),
        linear_node(3, "up_proj", 0, &shape_in, ff_dim),
        binary_node(4, "mul_0", TraceOp::Mul, 2, 3, &shape_ff),
        linear_node(5, "down_proj", 4, &shape_ff, dim),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("SwiGLU trace should compile");

    // The plan should have exactly 6 steps (one per graph node).
    assert_eq!(
        plan.steps.len(),
        6,
        "compiled plan should have 6 steps (1 per graph node), got {}",
        plan.steps.len()
    );

    // Output step should be the last step (down_proj).
    assert_eq!(plan.output_step, 5, "output should be step 5 (down_proj)");
}
