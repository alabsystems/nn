// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ICB analysis functions.
//!
//! Tests the pure logic in `compiled_model_icb_analysis.rs`:
//! segment detection, barrier computation, eligibility analysis.
//!
//! Part of #3410.

use std::collections::HashMap;

use super::{
    build_segment_starts, compute_concurrent_barriers, detect_icb_segments, summarize_barriers,
};

// ── detect_icb_segments ──────────────────────────────────────────────

#[test]
fn detect_segments_empty_input() {
    let result = detect_icb_segments(&[], 1);
    assert!(result.is_empty());
}

#[test]
fn detect_segments_all_eligible_single_segment() {
    let eligible = vec![true; 8];
    let result = detect_icb_segments(&eligible, 4);
    assert_eq!(result, vec![(0, 7)]);
}

#[test]
fn detect_segments_none_eligible() {
    let eligible = vec![false; 8];
    let result = detect_icb_segments(&eligible, 1);
    assert!(result.is_empty());
}

#[test]
fn detect_segments_split_by_ineligible() {
    // [T T T T F F T T T T] → two segments: (0,3) and (6,9)
    let eligible = vec![true, true, true, true, false, false, true, true, true, true];
    let result = detect_icb_segments(&eligible, 4);
    assert_eq!(result, vec![(0, 3), (6, 9)]);
}

#[test]
fn detect_segments_below_min_length_filtered() {
    // [T T F T T T T T] → only (3,7) qualifies at min_len=4
    let eligible = vec![true, true, false, true, true, true, true, true];
    let result = detect_icb_segments(&eligible, 4);
    assert_eq!(result, vec![(3, 7)]);
}

#[test]
fn detect_segments_exactly_min_length() {
    let eligible = vec![true, true, true, true];
    let result = detect_icb_segments(&eligible, 4);
    assert_eq!(result, vec![(0, 3)]);
}

#[test]
fn detect_segments_one_below_min_length() {
    let eligible = vec![true, true, true];
    let result = detect_icb_segments(&eligible, 4);
    assert!(result.is_empty());
}

#[test]
fn detect_segments_min_length_one() {
    let eligible = vec![false, true, false, true, true, false];
    let result = detect_icb_segments(&eligible, 1);
    assert_eq!(result, vec![(1, 1), (3, 4)]);
}

#[test]
fn detect_segments_trailing_eligible() {
    // Eligible run at end of array
    let eligible = vec![false, false, true, true, true, true];
    let result = detect_icb_segments(&eligible, 4);
    assert_eq!(result, vec![(2, 5)]);
}

#[test]
fn detect_segments_single_element_eligible() {
    let eligible = vec![true];
    let result = detect_icb_segments(&eligible, 1);
    assert_eq!(result, vec![(0, 0)]);
}

// ── build_segment_starts ─────────────────────────────────────────────

#[test]
fn segment_starts_empty() {
    let result = build_segment_starts(&[]);
    assert!(result.is_empty());
}

#[test]
fn segment_starts_maps_start_to_index() {
    let segments = vec![(0, 3), (6, 9), (15, 20)];
    let result = build_segment_starts(&segments);
    assert_eq!(result.len(), 3);
    assert_eq!(result[&0], 0);
    assert_eq!(result[&6], 1);
    assert_eq!(result[&15], 2);
}

#[test]
fn segment_starts_single_segment() {
    let segments = vec![(5, 10)];
    let result = build_segment_starts(&segments);
    assert_eq!(result.len(), 1);
    assert_eq!(result[&5], 0);
}

// ── compute_concurrent_barriers ──────────────────────────────────────

#[test]
fn barriers_empty() {
    let result = compute_concurrent_barriers(&[], &[], &[]);
    assert!(result.is_empty());
}

#[test]
fn barriers_no_dependencies() {
    // Three independent GPU dispatches writing to different offsets
    let edge_map = vec![vec![], vec![], vec![]];
    let step_offsets = vec![Some(0), Some(100), Some(200)];
    let is_gpu = vec![true, true, true];
    let result = compute_concurrent_barriers(&edge_map, &step_offsets, &is_gpu);
    assert_eq!(result, vec![false, false, false]);
}

#[test]
fn barriers_chain_dependency() {
    // Step 1 reads from step 0's output offset
    let edge_map = vec![vec![], vec![0], vec![1]];
    let step_offsets = vec![Some(0), Some(100), Some(200)];
    let is_gpu = vec![true, true, true];
    let result = compute_concurrent_barriers(&edge_map, &step_offsets, &is_gpu);
    // Step 0: no dependency → false
    // Step 1: reads from step 0 (offset 0 is dirty) → true
    // Step 1's barrier clears dirty set, then step 1 writes offset 100
    // Step 2: reads from step 1 (offset 100 is dirty) → true
    assert_eq!(result, vec![false, true, true]);
}

#[test]
fn barriers_non_dispatch_clears_dirty() {
    // Non-dispatch step in the middle clears the dirty set
    let edge_map = vec![vec![], vec![], vec![0]];
    let step_offsets = vec![Some(0), None, Some(200)];
    let is_gpu = vec![true, false, true];
    let result = compute_concurrent_barriers(&edge_map, &step_offsets, &is_gpu);
    // Step 0: GPU, no deps → false, writes offset 0
    // Step 1: non-GPU → clears dirty set
    // Step 2: GPU, reads step 0, but dirty set was cleared → false
    assert_eq!(result, vec![false, false, false]);
}

#[test]
fn barriers_barrier_clears_dirty_set() {
    // After a barrier, the dirty set is cleared
    let edge_map = vec![vec![], vec![0], vec![]];
    let step_offsets = vec![Some(0), Some(100), Some(200)];
    let is_gpu = vec![true, true, true];
    let result = compute_concurrent_barriers(&edge_map, &step_offsets, &is_gpu);
    // Step 1 gets barrier (reads dirty offset 0), clears dirty set
    // Step 2 has no dependency → no barrier (dirty set only has step 1's offset 100,
    // but step 2 has no edges)
    assert_eq!(result, vec![false, true, false]);
}

#[test]
fn barriers_no_planned_offset_for_source() {
    // Source step has None offset → resolve walks edges
    let edge_map = vec![vec![], vec![0]];
    let step_offsets = vec![None, Some(100)];
    let is_gpu = vec![true, true];
    let result = compute_concurrent_barriers(&edge_map, &step_offsets, &is_gpu);
    // Step 1 reads step 0, but step 0 has no planned offset → no dirty match
    assert_eq!(result, vec![false, false]);
}

// ── summarize_barriers ───────────────────────────────────────────────

#[test]
fn summarize_barriers_basic() {
    let eligible = vec![true, true, true, false, true];
    let barriers = vec![false, true, false, false, true];
    let summary = summarize_barriers(&eligible, &barriers);
    assert_eq!(summary.eligible, 4); // indices 0,1,2,4
    assert_eq!(summary.barriers, 2); // indices 1,4 (eligible AND barrier)
    assert_eq!(summary.concurrent, 2); // 4 - 2
}

#[test]
fn summarize_barriers_none_eligible() {
    let eligible = vec![false; 5];
    let barriers = vec![true; 5];
    let summary = summarize_barriers(&eligible, &barriers);
    assert_eq!(summary.eligible, 0);
    assert_eq!(summary.barriers, 0);
    assert_eq!(summary.concurrent, 0);
}

#[test]
fn summarize_barriers_all_concurrent() {
    let eligible = vec![true; 4];
    let barriers = vec![false; 4];
    let summary = summarize_barriers(&eligible, &barriers);
    assert_eq!(summary.eligible, 4);
    assert_eq!(summary.barriers, 0);
    assert_eq!(summary.concurrent, 4);
}

// ── analyze_icb_eligibility ──────────────────────────────────────────

#[test]
fn eligibility_dispatch_no_autocast() {
    let steps = vec![make_dispatch_step(), make_dispatch_step(), make_dispatch_step()];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, false, false);
    assert_eq!(result, vec![true, true, true]);
}

#[test]
fn eligibility_dispatch_with_autocast_matching_dtypes() {
    // With autocast, Dispatch steps with matching input dtypes ARE eligible.
    let steps = vec![make_dispatch_step(), make_dispatch_step()];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0])];
    let gemm = no_gemm(2);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    // Both F32 → F32, no boundary cast needed → eligible.
    assert_eq!(result, vec![true, true]);
}

#[test]
fn eligibility_dispatch_with_mixed_precision() {
    let steps = vec![make_dispatch_step()];
    let metas = vec![f32_meta(vec![])];
    let gemm = no_gemm(1);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, false, true);
    assert_eq!(result, vec![false]);
}

#[test]
fn eligibility_mixed_steps() {
    let steps = vec![
        make_dispatch_step(),
        make_passthrough_step(),
        make_dispatch_step(),
        make_native_op_step(),
        make_dispatch_step(),
    ];
    let metas = vec![
        f32_meta(vec![]),
        f32_meta(vec![0]),
        f32_meta(vec![1]),
        f32_meta(vec![2]),
        f32_meta(vec![3]),
    ];
    let gemm = no_gemm(5);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, false, false);
    // Only Dispatch steps are eligible
    assert_eq!(result, vec![true, false, true, false, true]);
}

#[test]
fn eligibility_non_dispatch_always_false() {
    let steps = vec![
        make_passthrough_step(),
        nn_dsl::CompiledStep::InputForward,
        nn_dsl::CompiledStep::IdentityPassthrough,
    ];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, false, false);
    assert_eq!(result, vec![false, false, false]);
}

// ── autocast static dtype analysis (#3426) ──────────────────────────

#[test]
fn eligibility_autocast_dtype_boundary_ineligible() {
    // Step 0 is F16, step 1 is F32 reading from step 0 → dtype mismatch → ineligible.
    use nn_dsl::ir::ScalarType;
    let steps = vec![make_dispatch_step(), make_dispatch_step()];
    let metas = vec![
        meta(ScalarType::F16, vec![]),
        meta(ScalarType::F32, vec![0]),
    ];
    let gemm = no_gemm(2);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    // Step 0: F16, no edges → eligible (no inputs to cast)
    // Step 1: F32 reading F16 source → needs cast → ineligible
    assert_eq!(result, vec![true, false]);
}

#[test]
fn eligibility_autocast_consistent_f16_chain() {
    // All F16 → F16, consistent dtype throughout → all eligible.
    use nn_dsl::ir::ScalarType;
    let steps = vec![make_dispatch_step(), make_dispatch_step(), make_dispatch_step()];
    let metas = vec![
        meta(ScalarType::F16, vec![]),
        meta(ScalarType::F16, vec![0]),
        meta(ScalarType::F16, vec![1]),
    ];
    let gemm = no_gemm(3);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    assert_eq!(result, vec![true, true, true]);
}

#[test]
fn eligibility_autocast_mixed_gemm_excluded() {
    // Mixed GEMM step is always ineligible under autocast.
    use nn_dsl::ir::ScalarType;
    let steps = vec![make_dispatch_step(), make_dispatch_step()];
    let metas = vec![
        meta(ScalarType::F16, vec![]),
        meta(ScalarType::F16, vec![0]),
    ];
    let gemm = vec![
        None,
        Some(crate::compiled_model::MixedGemmInfo {
            m: 8,
            k: 256,
            n: 256,
            batch_count: 1,
            transpose_b: true,
            broadcast_b: false,
            has_bias: false,
            activation: None,
        }),
    ];
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    // Step 0: F16, no edges → eligible
    // Step 1: Mixed GEMM → ineligible (uses separate dispatch path)
    assert_eq!(result, vec![true, false]);
}

#[test]
fn eligibility_autocast_mixed_gemm_poisons_downstream() {
    // Mixed GEMM outputs F32 at runtime. Downstream F16 step sees F32 source → ineligible.
    use nn_dsl::ir::ScalarType;
    let steps = vec![make_dispatch_step(), make_dispatch_step(), make_dispatch_step()];
    let metas = vec![
        meta(ScalarType::F16, vec![]),
        meta(ScalarType::F16, vec![0]),
        meta(ScalarType::F16, vec![1]),
    ];
    let gemm = vec![
        None,
        Some(crate::compiled_model::MixedGemmInfo {
            m: 8,
            k: 256,
            n: 256,
            batch_count: 1,
            transpose_b: true,
            broadcast_b: false,
            has_bias: false,
            activation: None,
        }),
        None,
    ];
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    // Step 0: F16, no edges → eligible
    // Step 1: Mixed GEMM → ineligible
    // Step 2: F16 reading step 1 which sim_dtype is F32 → needs cast → ineligible
    assert_eq!(result, vec![true, false, false]);
}

#[test]
fn eligibility_autocast_runtime_op_poisons_downstream() {
    // RuntimeOp always outputs F32. Downstream F16 Dispatch sees F32 → ineligible.
    use nn_dsl::ir::ScalarType;
    let steps = vec![
        make_dispatch_step(),
        nn_dsl::CompiledStep::RuntimeOp {
            op: nn_dsl::trace_compile::RuntimeOpKind::RepeatInterleave {
                dim: 0,
                input_shape: vec![4],
                counts_shape: vec![4],
            },
        },
        make_dispatch_step(),
    ];
    let metas = vec![
        meta(ScalarType::F16, vec![]),
        meta(ScalarType::F16, vec![0]),
        meta(ScalarType::F16, vec![1]),
    ];
    let gemm = no_gemm(3);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    // Step 0: F16, no edges → eligible
    // Step 1: RuntimeOp → not a Dispatch → false
    // Step 2: F16 reading step 1 (sim F32) → needs cast → ineligible
    assert_eq!(result, vec![true, false, false]);
}

#[test]
fn eligibility_autocast_passthrough_propagation() {
    // Passthrough inherits source dtype: F16 → passthrough → F16 Dispatch = eligible.
    use nn_dsl::ir::ScalarType;
    let steps = vec![
        make_dispatch_step(),
        make_passthrough_step(),
        make_dispatch_step(),
    ];
    let metas = vec![
        meta(ScalarType::F16, vec![]),
        meta(ScalarType::F32, vec![0]), // compiled as F32 but propagates from F16 source
        meta(ScalarType::F16, vec![1]),
    ];
    let gemm = no_gemm(3);
    let result = super::analyze_icb_eligibility(&steps, &metas, &gemm, true, false);
    // Step 0: F16 → eligible
    // Step 1: Passthrough → false (never eligible)
    // Step 2: F16, reads passthrough whose sim_dtype propagated to F16 → eligible
    assert_eq!(result, vec![true, false, true]);
}

// ── analyze_gpu_dispatch_steps ───────────────────────────────────────

#[test]
fn gpu_dispatch_detects_dispatch_steps() {
    let steps = vec![
        make_dispatch_step(),
        make_passthrough_step(),
        make_dispatch_step(),
    ];
    let result = super::analyze_gpu_dispatch_steps(&steps);
    assert_eq!(result, vec![true, false, true]);
}

#[test]
fn gpu_dispatch_empty() {
    let result = super::analyze_gpu_dispatch_steps(&[]);
    assert!(result.is_empty());
}

// ── Helpers ──────────────────────────────────────────────────────────

fn meta(scalar_type: nn_dsl::ir::ScalarType, edges: Vec<usize>) -> crate::compiled_model::StepMeta {
    crate::compiled_model::StepMeta {
        edges,
        scalar_type,
        numel: 1,
    }
}

fn f32_meta(edges: Vec<usize>) -> crate::compiled_model::StepMeta {
    meta(nn_dsl::ir::ScalarType::F32, edges)
}

fn no_gemm(n: usize) -> Vec<Option<crate::compiled_model::MixedGemmInfo>> {
    vec![None; n]
}

fn make_dispatch_step() -> nn_dsl::CompiledStep {
    use nn_dsl::{CompiledKernel, CompiledStep, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

    let node = TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: "x".into(),
            shape: vec![1],
        },
        vec![1],
    );
    let def = TensorKernelDef::new("test_kernel", vec![node], TensorNodeId::new(0));
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn make_passthrough_step() -> nn_dsl::CompiledStep {
    nn_dsl::CompiledStep::Passthrough {
        op_name: "reshape".into(),
        output_shape: vec![1],
    }
}

fn make_native_op_step() -> nn_dsl::CompiledStep {
    use nn_dsl::{CompiledStep, NativeOpKind};

    CompiledStep::NativeOp {
        op: NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 1, 4],
        },
        weight_data: HashMap::new(),
    }
}
