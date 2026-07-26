// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `fingerprint.rs`.
//!
//! Proves safety and correctness properties of:
//! - SHA-256 hash assembly (no panics, 32-byte output)
//! - `fingerprint_trace` determinism and output structure
//! - `fingerprint_trace_with_weights` divergence from structural fingerprint
//! - `diff_fingerprints` region detection, boundary arithmetic, coverage
//! - `classify_change` classification correctness (OpChanged vs WeightChanged)
//! - `ChangeReason` Display completeness
//! - Hash domain separation: ops with different hyperparameters differ
//! - Hash content sensitivity: weight content changes detected in parametric mode
//! - `fingerprint_graph` / `fingerprint_graph_with_weights` convenience parity
//!
//! Part of #3682.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::fingerprint::{
    diff_fingerprints, fingerprint_graph, fingerprint_graph_with_weights, fingerprint_trace,
    fingerprint_trace_with_weights, ChangeReason, SubgraphFingerprint,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_input(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn make_relu(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("relu_{id}"),
        TraceOp::Relu,
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_sigmoid(id: u64, input_id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("sigmoid_{id}"),
        TraceOp::Sigmoid,
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_softmax(id: u64, input_id: u64, dim: usize, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("softmax_{id}"),
        TraceOp::Softmax { dim },
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_reshape(id: u64, input_id: u64, target: Vec<usize>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("reshape_{id}"),
        TraceOp::Reshape {
            target_shape: target,
        },
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_linear(id: u64, input_id: u64, weight: WeightRef, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("linear_{id}"),
        TraceOp::Linear { weight, bias: None },
        vec![input_id],
        shape,
        DType::F32,
    )
}

fn make_conv1d(
    id: u64,
    input_id: u64,
    weight: WeightRef,
    stride: usize,
    shape: Vec<usize>,
) -> TraceNode {
    TraceNode::new(
        id,
        format!("conv1d_{id}"),
        TraceOp::Conv1d {
            weight,
            bias: None,
            padding: 1,
            stride,
            dilation: 1,
            groups: 1,
        },
        vec![input_id],
        shape,
        DType::F32,
    )
}

// ===========================================================================
// DETERMINISM HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. fingerprint_trace is deterministic on multi-node graph
// ---------------------------------------------------------------------------

/// Prove: fingerprint_trace called twice on a 3-node chain yields identical
/// hashes at every position.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn fp_deterministic_three_node_chain() {
    let nodes = vec![
        make_input(1, vec![2, 4]),
        make_relu(2, 1, vec![2, 4]),
        make_sigmoid(3, 2, vec![2, 4]),
    ];
    let fp1 = fingerprint_trace(&nodes);
    let fp2 = fingerprint_trace(&nodes);
    assert_eq!(fp1.len(), fp2.len());
    for i in 0..fp1.len() {
        assert_eq!(fp1[i].hash, fp2[i].hash, "position {i} hash must match");
    }
}

// ---------------------------------------------------------------------------
// 2. fingerprint_trace_with_weights is deterministic
// ---------------------------------------------------------------------------

/// Prove: parametric fingerprints are deterministic for nodes with weights.
#[kani::unwind(128)]
#[kani::proof]
fn fp_with_weights_deterministic() {
    let w = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("valid");
    let nodes = vec![make_linear(1, 0, w, vec![1, 2])];
    let fp1 = fingerprint_trace_with_weights(&nodes);
    let fp2 = fingerprint_trace_with_weights(&nodes);
    assert_eq!(fp1[0].hash, fp2[0].hash);
}

// ===========================================================================
// OUTPUT STRUCTURE HARNESSES
// ===========================================================================

// ---------------------------------------------------------------------------
// 3. op_summary populated for each node
// ---------------------------------------------------------------------------

/// Prove: every fingerprint has a non-empty `op_summary`.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn fp_op_summary_nonempty() {
    let nodes = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_sigmoid(3, 2, vec![4]),
    ];
    let fps = fingerprint_trace(&nodes);
    for fp in &fps {
        assert!(!fp.op_summary.is_empty(), "op_summary must be non-empty");
    }
}

// ---------------------------------------------------------------------------
// 4. node_indices singleton per node
// ---------------------------------------------------------------------------

/// Prove: each fingerprint from `fingerprint_trace` has exactly one node_index.
#[kani::unwind(128)]
#[kani::proof]
fn fp_node_indices_singleton() {
    let nodes = vec![make_input(1, vec![8])];
    let fps = fingerprint_trace(&nodes);
    assert_eq!(fps[0].node_indices.len(), 1);
}

// ---------------------------------------------------------------------------
// 5. hash is always 32 bytes (SHA-256)
// ---------------------------------------------------------------------------

/// Prove: the hash field is always exactly 32 bytes.
#[kani::unwind(128)]
#[kani::proof]
fn fp_hash_always_32_bytes() {
    let nodes = vec![make_relu(1, 0, vec![16])];
    let fps = fingerprint_trace(&nodes);
    assert_eq!(fps[0].hash.len(), 32);
}

// ===========================================================================
// HASH DOMAIN SEPARATION
// ===========================================================================

// ---------------------------------------------------------------------------
// 6. Different ops produce different hashes (Relu vs Sigmoid)
// ---------------------------------------------------------------------------

/// Prove: Relu and Sigmoid nodes with the same shape produce different hashes.
#[kani::unwind(128)]
#[kani::proof]
fn fp_relu_vs_sigmoid_different_hash() {
    let relu = vec![make_relu(1, 0, vec![8])];
    let sig = vec![make_sigmoid(1, 0, vec![8])];
    let fp_r = fingerprint_trace(&relu);
    let fp_s = fingerprint_trace(&sig);
    assert_ne!(fp_r[0].hash, fp_s[0].hash);
}

// ---------------------------------------------------------------------------
// 7. Same op, different hyperparameter (Softmax dim=0 vs dim=1)
// ---------------------------------------------------------------------------

/// Prove: Softmax with dim=0 and dim=1 produce different hashes.
#[kani::unwind(128)]
#[kani::proof]
fn fp_softmax_different_dim_different_hash() {
    let sm0 = vec![make_softmax(1, 0, 0, vec![2, 4])];
    let sm1 = vec![make_softmax(1, 0, 1, vec![2, 4])];
    let fp0 = fingerprint_trace(&sm0);
    let fp1 = fingerprint_trace(&sm1);
    assert_ne!(
        fp0[0].hash, fp1[0].hash,
        "different dim must produce different hash"
    );
}

// ---------------------------------------------------------------------------
// 8. Same op, different Reshape target shapes
// ---------------------------------------------------------------------------

/// Prove: Reshape to [4,2] vs [8] produce different hashes.
#[kani::unwind(128)]
#[kani::proof]
fn fp_reshape_different_target_different_hash() {
    let r1 = vec![make_reshape(1, 0, vec![4, 2], vec![4, 2])];
    let r2 = vec![make_reshape(1, 0, vec![8], vec![8])];
    let fp1 = fingerprint_trace(&r1);
    let fp2 = fingerprint_trace(&r2);
    assert_ne!(fp1[0].hash, fp2[0].hash);
}

// ---------------------------------------------------------------------------
// 9. Conv1d with different stride => different hash
// ---------------------------------------------------------------------------

/// Prove: Conv1d with stride=1 vs stride=2 produces different hashes.
#[kani::unwind(128)]
#[kani::proof]
fn fp_conv1d_different_stride_different_hash() {
    let w = WeightRef::from_shape(&[4, 2, 3]);
    let c1 = vec![make_conv1d(1, 0, w.clone(), 1, vec![1, 4, 8])];
    let c2 = vec![make_conv1d(1, 0, w, 2, vec![1, 4, 4])];
    let fp1 = fingerprint_trace(&c1);
    let fp2 = fingerprint_trace(&c2);
    assert_ne!(fp1[0].hash, fp2[0].hash);
}

// ---------------------------------------------------------------------------
// 10. Same op and shape but different node IDs => same hash
// ---------------------------------------------------------------------------

/// Prove: node ID is NOT part of the hash (fingerprint is structural).
#[kani::unwind(128)]
#[kani::proof]
fn fp_node_id_not_in_hash() {
    let n1 = vec![make_input(1, vec![4])];
    let n2 = vec![make_input(999, vec![4])];
    let fp1 = fingerprint_trace(&n1);
    let fp2 = fingerprint_trace(&n2);
    assert_eq!(fp1[0].hash, fp2[0].hash, "node ID must not affect hash");
}

// ---------------------------------------------------------------------------
// 11. Same op and shape but different node name => same hash
// ---------------------------------------------------------------------------

/// Prove: node name is NOT part of the hash.
#[kani::unwind(128)]
#[kani::proof]
fn fp_node_name_not_in_hash() {
    let n1 = TraceNode::new(
        1,
        "foo".to_string(),
        TraceOp::Relu,
        vec![0],
        vec![4],
        DType::F32,
    );
    let n2 = TraceNode::new(
        1,
        "bar".to_string(),
        TraceOp::Relu,
        vec![0],
        vec![4],
        DType::F32,
    );
    let fp1 = fingerprint_trace(&[n1]);
    let fp2 = fingerprint_trace(&[n2]);
    assert_eq!(fp1[0].hash, fp2[0].hash, "node name must not affect hash");
}

// ===========================================================================
// STRUCTURAL vs PARAMETRIC MODE
// ===========================================================================

// ---------------------------------------------------------------------------
// 12. Structural: same weight shape, different content => same hash
// ---------------------------------------------------------------------------

/// Prove: structural fingerprint does NOT include weight content.
#[kani::unwind(128)]
#[kani::proof]
fn fp_structural_ignores_weight_content() {
    let w1 = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
    let w2 = WeightRef::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("ok");
    let n1 = vec![make_linear(1, 0, w1, vec![1, 2])];
    let n2 = vec![make_linear(1, 0, w2, vec![1, 2])];
    let fp1 = fingerprint_trace(&n1);
    let fp2 = fingerprint_trace(&n2);
    assert_eq!(
        fp1[0].hash, fp2[0].hash,
        "structural mode ignores weight content"
    );
}

// ---------------------------------------------------------------------------
// 13. Parametric: same weight shape, different content => different hash
// ---------------------------------------------------------------------------

/// Prove: parametric fingerprint DOES include weight content.
#[kani::unwind(128)]
#[kani::proof]
fn fp_parametric_detects_weight_content() {
    let w1 = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
    let w2 = WeightRef::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("ok");
    let n1 = vec![make_linear(1, 0, w1, vec![1, 2])];
    let n2 = vec![make_linear(1, 0, w2, vec![1, 2])];
    let fp1 = fingerprint_trace_with_weights(&n1);
    let fp2 = fingerprint_trace_with_weights(&n2);
    assert_ne!(
        fp1[0].hash, fp2[0].hash,
        "parametric mode detects weight content"
    );
}

// ---------------------------------------------------------------------------
// 14. Parametric: different weight shape => different hash
// ---------------------------------------------------------------------------

/// Prove: parametric fingerprints differ when weight shapes differ.
#[kani::unwind(128)]
#[kani::proof]
fn fp_parametric_different_weight_shape() {
    let w1 = WeightRef::from_shape(&[4, 4]);
    let w2 = WeightRef::from_shape(&[8, 2]);
    let n1 = vec![make_linear(1, 0, w1, vec![1, 4])];
    let n2 = vec![make_linear(1, 0, w2, vec![1, 8])];
    let fp1 = fingerprint_trace_with_weights(&n1);
    let fp2 = fingerprint_trace_with_weights(&n2);
    assert_ne!(fp1[0].hash, fp2[0].hash);
}

// ===========================================================================
// CONVENIENCE FUNCTIONS (graph wrappers)
// ===========================================================================

// ---------------------------------------------------------------------------
// 15. fingerprint_graph parity with fingerprint_trace
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_graph` and `fingerprint_trace` produce identical output.
#[kani::unwind(128)]
#[kani::proof]
fn fp_graph_parity_with_trace() {
    let nodes = vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes.clone());
    let fp_trace = fingerprint_trace(&nodes);
    let fp_graph = fingerprint_graph(&graph);
    assert_eq!(fp_trace.len(), fp_graph.len());
    assert_eq!(fp_trace[0].hash, fp_graph[0].hash);
    assert_eq!(fp_trace[1].hash, fp_graph[1].hash);
}

// ---------------------------------------------------------------------------
// 16. fingerprint_graph_with_weights parity
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_graph_with_weights` matches `fingerprint_trace_with_weights`.
#[kani::unwind(128)]
#[kani::proof]
fn fp_graph_with_weights_parity() {
    let nodes = vec![make_input(1, vec![4])];
    let graph = ComputationGraph::from_nodes(nodes.clone());
    let fp_trace = fingerprint_trace_with_weights(&nodes);
    let fp_graph = fingerprint_graph_with_weights(&graph);
    assert_eq!(fp_trace[0].hash, fp_graph[0].hash);
}

// ===========================================================================
// DIFF FINGERPRINTS — BOUNDARY ARITHMETIC
// ===========================================================================

// ---------------------------------------------------------------------------
// 17. diff: regions are non-overlapping and ordered
// ---------------------------------------------------------------------------

/// Prove: `diff_fingerprints` returns non-overlapping regions in ascending order.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn diff_regions_nonoverlapping_ordered() {
    let old = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_relu(3, 2, vec![4]),
        make_relu(4, 3, vec![4]),
    ];
    let new = vec![
        make_input(1, vec![4]),
        make_sigmoid(2, 1, vec![4]), // changed
        make_relu(3, 2, vec![4]),    // same
        make_sigmoid(4, 3, vec![4]), // changed
    ];
    let old_fps = fingerprint_trace(&old);
    let new_fps = fingerprint_trace(&new);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    for i in 1..changes.len() {
        assert!(
            changes[i].start >= changes[i - 1].end,
            "regions must not overlap"
        );
    }
}

// ---------------------------------------------------------------------------
// 18. diff: changed regions cover exactly the differing indices
// ---------------------------------------------------------------------------

/// Prove: union of changed regions covers all indices where hashes differ
/// (within the common-length portion).
#[kani::unwind(128)]
#[kani::proof]
fn diff_changed_region_covers_all_diffs() {
    let old = vec![make_input(1, vec![4]), make_relu(2, 1, vec![4])];
    let new = vec![
        make_sigmoid(1, 0, vec![4]), // changed
        make_relu(2, 1, vec![4]),    // same
    ];
    let old_fps = fingerprint_trace(&old);
    let new_fps = fingerprint_trace(&new);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    // Index 0 should be covered by a changed region.
    let covers_0 = changes.iter().any(|r| r.start <= 0 && r.end > 0);
    assert!(covers_0, "index 0 differs and must be covered");
    // Index 1 should NOT be covered.
    let covers_1 = changes.iter().any(|r| r.start <= 1 && r.end > 1);
    assert!(!covers_1, "index 1 is same and must not be covered");
}

// ---------------------------------------------------------------------------
// 19. diff: old longer => Removed region
// ---------------------------------------------------------------------------

/// Prove: when old is longer than new, trailing region reason is Removed
/// with correct start/end.
#[kani::unwind(128)]
#[kani::proof]
fn diff_old_longer_removed_bounds() {
    let old = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_sigmoid(3, 2, vec![4]),
    ];
    let new = vec![make_input(1, vec![4])];
    let old_fps = fingerprint_trace(&old);
    let new_fps = fingerprint_trace(&new);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    let removed = changes.iter().find(|r| r.reason == ChangeReason::Removed);
    assert!(removed.is_some(), "must have Removed region");
    let r = removed.unwrap();
    assert_eq!(r.start, 1, "Removed starts at new.len()");
    assert_eq!(r.end, 3, "Removed ends at old.len()");
}

// ---------------------------------------------------------------------------
// 20. diff: consecutive changed nodes merge into one region
// ---------------------------------------------------------------------------

/// Prove: three consecutive changed nodes produce exactly one ChangedRegion.
#[kani::unwind(128)]
#[kani::proof]
fn diff_consecutive_changes_merge() {
    let old = vec![
        make_relu(1, 0, vec![4]),
        make_relu(2, 1, vec![4]),
        make_relu(3, 2, vec![4]),
    ];
    let new = vec![
        make_sigmoid(1, 0, vec![4]),
        make_sigmoid(2, 1, vec![4]),
        make_sigmoid(3, 2, vec![4]),
    ];
    let old_fps = fingerprint_trace(&old);
    let new_fps = fingerprint_trace(&new);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    assert_eq!(changes.len(), 1, "consecutive changes must merge");
    assert_eq!(changes[0].start, 0);
    assert_eq!(changes[0].end, 3);
}

// ===========================================================================
// CLASSIFY_CHANGE
// ===========================================================================

// ---------------------------------------------------------------------------
// 21. classify: same op different content => WeightChanged
// ---------------------------------------------------------------------------

/// Prove: when op_summary matches but hashes differ (e.g., shape differs
/// for the same op), classify_change returns WeightChanged.
#[kani::unwind(128)]
#[kani::proof]
fn classify_weight_changed_same_op_different_shape() {
    let old = vec![make_relu(1, 0, vec![4])];
    let new = vec![make_relu(1, 0, vec![8])];
    let old_fps = fingerprint_trace(&old);
    let new_fps = fingerprint_trace(&new);
    let changes = diff_fingerprints(&old_fps, &new_fps);
    // Relu has canonical_name "relu" in both. Shapes differ => hash differs.
    // classify_change sees same op_summary => returns WeightChanged.
    assert_eq!(changes[0].reason, ChangeReason::WeightChanged);
}

// ===========================================================================
// CHANGE_REASON DISPLAY
// ===========================================================================

// ---------------------------------------------------------------------------
// 22. ChangeReason Display values are distinct
// ---------------------------------------------------------------------------

/// Prove: all five ChangeReason variants produce distinct Display strings.
#[kani::unwind(64)]
#[kani::proof]
fn change_reason_display_distinct() {
    let reasons = [
        ChangeReason::OpChanged,
        ChangeReason::ShapeChanged,
        ChangeReason::WeightChanged,
        ChangeReason::Inserted,
        ChangeReason::Removed,
    ];
    let strings: Vec<String> = reasons.iter().map(|r| format!("{r}")).collect();
    for i in 0..strings.len() {
        for j in (i + 1)..strings.len() {
            assert_ne!(
                strings[i], strings[j],
                "variants must have distinct Display"
            );
        }
    }
}

// ===========================================================================
// EMPTY INPUT EDGE CASES
// ===========================================================================

// ---------------------------------------------------------------------------
// 23. fingerprint_trace on empty slice returns empty
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_trace(&[])` returns an empty Vec.
#[kani::unwind(1)]
#[kani::proof]
fn fp_empty_input_returns_empty() {
    let fps = fingerprint_trace(&[]);
    assert!(fps.is_empty());
}

// ---------------------------------------------------------------------------
// 24. fingerprint_trace_with_weights on empty slice returns empty
// ---------------------------------------------------------------------------

/// Prove: `fingerprint_trace_with_weights(&[])` returns an empty Vec.
#[kani::unwind(1)]
#[kani::proof]
fn fp_with_weights_empty_input_returns_empty() {
    let fps = fingerprint_trace_with_weights(&[]);
    assert!(fps.is_empty());
}

// ===========================================================================
// NEW HARNESSES -- Hash domain separation for additional TraceOp variants
// ===========================================================================

/// Prove: Transpose(dim0=0,dim1=1) vs Transpose(dim0=0,dim1=2) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_transpose_different_dims_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "t1".into(),
        TraceOp::Transpose { dim0: 0, dim1: 1 },
        vec![0],
        vec![4, 2],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "t2".into(),
        TraceOp::Transpose { dim0: 0, dim1: 2 },
        vec![0],
        vec![4, 2],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Narrow(start=0) vs Narrow(start=1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_narrow_different_start_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "n1".into(),
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 2,
        },
        vec![0],
        vec![2],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "n2".into(),
        TraceOp::Narrow {
            dim: 0,
            start: 1,
            length: 2,
        },
        vec![0],
        vec![2],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Unsqueeze(dim=0) vs Unsqueeze(dim=1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_unsqueeze_different_dim_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "u1".into(),
        TraceOp::Unsqueeze { dim: 0 },
        vec![0],
        vec![1, 4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "u2".into(),
        TraceOp::Unsqueeze { dim: 1 },
        vec![0],
        vec![4, 1],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Cat(dim=0) vs Cat(dim=1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_cat_different_dim_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "c1".into(),
        TraceOp::Cat {
            dim: 0,
            num_inputs: 2,
        },
        vec![0],
        vec![8],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "c2".into(),
        TraceOp::Cat {
            dim: 1,
            num_inputs: 2,
        },
        vec![0],
        vec![8],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Cat(num_inputs=2) vs Cat(num_inputs=3) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_cat_different_num_inputs_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "c1".into(),
        TraceOp::Cat {
            dim: 0,
            num_inputs: 2,
        },
        vec![0],
        vec![8],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "c2".into(),
        TraceOp::Cat {
            dim: 0,
            num_inputs: 3,
        },
        vec![0],
        vec![8],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: InstanceNorm(eps=1e-5) vs InstanceNorm(eps=1e-3) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_instance_norm_different_eps_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "in1".into(),
        TraceOp::InstanceNorm { eps: 1e-5 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "in2".into(),
        TraceOp::InstanceNorm { eps: 1e-3 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: ReduceSum(keepdim=true) vs ReduceSum(keepdim=false) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_reduce_sum_different_keepdim_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "rs1".into(),
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: true,
        },
        vec![0],
        vec![1, 4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "rs2".into(),
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false,
        },
        vec![0],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Elu(alpha=1.0) vs Elu(alpha=0.5) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_elu_different_alpha_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "e1".into(),
        TraceOp::Elu { alpha: 1.0 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "e2".into(),
        TraceOp::Elu { alpha: 0.5 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: LeakyRelu(slope=0.01) vs LeakyRelu(slope=0.1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_leaky_relu_different_slope_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "lr1".into(),
        TraceOp::LeakyRelu { slope: 0.01 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "lr2".into(),
        TraceOp::LeakyRelu { slope: 0.1 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Constant(0.0) vs Constant(1.0) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_constant_different_value_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "const1".into(),
        TraceOp::Constant { value: 0.0 },
        vec![],
        vec![1],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "const2".into(),
        TraceOp::Constant { value: 1.0 },
        vec![],
        vec![1],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Permute([0,1,2]) vs Permute([2,1,0]) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_permute_different_axes_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "p1".into(),
        TraceOp::Permute {
            axes: vec![0, 1, 2],
        },
        vec![0],
        vec![2, 3, 4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "p2".into(),
        TraceOp::Permute {
            axes: vec![2, 1, 0],
        },
        vec![0],
        vec![4, 3, 2],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Triu(diagonal=0) vs Triu(diagonal=1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_triu_different_diagonal_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "t1".into(),
        TraceOp::Triu { diagonal: 0 },
        vec![0],
        vec![4, 4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "t2".into(),
        TraceOp::Triu { diagonal: 1 },
        vec![0],
        vec![4, 4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Arange(start=0) vs Arange(start=1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_arange_different_start_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "a1".into(),
        TraceOp::Arange {
            start: 0.0,
            end: 4.0,
            step: 1.0,
        },
        vec![],
        vec![4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "a2".into(),
        TraceOp::Arange {
            start: 1.0,
            end: 5.0,
            step: 1.0,
        },
        vec![],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

// ===========================================================================
// NEW HARNESSES -- Diff edge cases
// ===========================================================================

/// Prove: interspersed changes produce separate regions.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn diff_interspersed_changes_separate_regions() {
    let old = vec![
        make_relu(1, 0, vec![4]),
        make_relu(2, 1, vec![4]),
        make_relu(3, 2, vec![4]),
    ];
    let new = vec![
        make_sigmoid(1, 0, vec![4]),
        make_relu(2, 1, vec![4]),
        make_sigmoid(3, 2, vec![4]),
    ];
    let changes = diff_fingerprints(&fingerprint_trace(&old), &fingerprint_trace(&new));
    assert_eq!(changes.len(), 2);
    assert_eq!(changes[0].start, 0);
    assert_eq!(changes[0].end, 1);
    assert_eq!(changes[1].start, 2);
    assert_eq!(changes[1].end, 3);
}

/// Prove: all nodes changed produces single region.
#[kani::unwind(128)]
#[kani::proof]
fn diff_all_changed_single_region() {
    let old = vec![make_relu(1, 0, vec![4]), make_relu(2, 1, vec![4])];
    let new = vec![make_sigmoid(1, 0, vec![4]), make_sigmoid(2, 1, vec![4])];
    let changes = diff_fingerprints(&fingerprint_trace(&old), &fingerprint_trace(&new));
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].start, 0);
    assert_eq!(changes[0].end, 2);
}

/// Prove: new longer produces Inserted region.
#[kani::unwind(128)]
#[kani::proof]
fn diff_new_longer_inserted_region() {
    let old = vec![make_input(1, vec![4])];
    let new = vec![
        make_input(1, vec![4]),
        make_relu(2, 1, vec![4]),
        make_sigmoid(3, 2, vec![4]),
    ];
    let changes = diff_fingerprints(&fingerprint_trace(&old), &fingerprint_trace(&new));
    let ins = changes
        .iter()
        .find(|r| r.reason == ChangeReason::Inserted)
        .unwrap();
    assert_eq!(ins.start, 1);
    assert_eq!(ins.end, 3);
}

/// Prove: Clamp(min=0) vs Clamp(min=1) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_clamp_different_min_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "cl1".into(),
        TraceOp::Clamp {
            min: Some(0.0),
            max: None,
        },
        vec![0],
        vec![4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "cl2".into(),
        TraceOp::Clamp {
            min: Some(1.0),
            max: None,
        },
        vec![0],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}

/// Prove: Powf(2.0) vs Powf(3.0) differ.
#[kani::unwind(8)]
#[kani::proof]
fn fp_powf_different_exponent_different_hash() {
    let n1 = vec![TraceNode::new(
        1,
        "p1".into(),
        TraceOp::Powf { exponent: 2.0 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    let n2 = vec![TraceNode::new(
        1,
        "p2".into(),
        TraceOp::Powf { exponent: 3.0 },
        vec![0],
        vec![4],
        DType::F32,
    )];
    assert_ne!(
        fingerprint_trace(&n1)[0].hash,
        fingerprint_trace(&n2)[0].hash
    );
}
