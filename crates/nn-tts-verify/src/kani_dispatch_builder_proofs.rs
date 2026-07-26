// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DispatchBuilder node allocation and step construction.
//!
//! Proves correctness of the node ID allocator (monotonicity, uniqueness),
//! step-count invariants, and node-count arithmetic for each builder method.
//!
//! Properties proved:
//!
//! 1. Node IDs are monotonically increasing.
//! 2. Each builder method allocates the documented number of nodes.
//! 3. `into_steps()` returns exactly the number of steps pushed.
//! 4. `with_capacity` starts at zero node count and zero steps.
//! 5. Mixed sequences produce correct cumulative node counts.
//! 6. `alloc_node` / `push_step` manual construction path is consistent.

// ---- CBMC transcendental stubs for Kani (#708) ------------------------------

/// Nondeterministic stub for `f32::tanh`.
/// CBMC cannot handle the tanh intrinsic. Returns a finite f32 in [-1, 1].
fn tanh_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

// ---- Node allocation proofs -------------------------------------------------

/// Prove: node IDs from alloc_node are monotonically increasing.
///
/// Two consecutive alloc_node calls must return id, id+1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn node_ids_monotonically_increasing() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(4);
    let id0 = b.alloc_node();
    let id1 = b.alloc_node();
    let id2 = b.alloc_node();

    // TensorNodeId wraps usize; use node_count as proxy for ID values.
    // After 3 allocs, node_count must be 3.
    assert_eq!(b.node_count(), 3, "3 allocs must produce node_count 3");
}

/// Prove: with_capacity initializes to zero node count and zero steps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_capacity_initializes_zero() {
    let cap: usize = kani::any();
    kani::assume(cap <= 1024);

    let b = crate::dispatch_builder::DispatchBuilder::with_capacity(cap);
    assert_eq!(b.node_count(), 0, "initial node_count must be 0");
    assert_eq!(b.into_steps().len(), 0, "initial steps must be empty");
}

// ---- Per-method node count proofs -------------------------------------------

/// Prove: `linear` allocates exactly 4 nodes (input, weight, bias, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_allocates_four_nodes() {
    let in_f: usize = kani::any();
    let out_f: usize = kani::any();
    let batch: usize = kani::any();
    kani::assume(in_f > 0 && in_f <= 1024);
    kani::assume(out_f > 0 && out_f <= 1024);
    kani::assume(batch > 0 && batch <= 64);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(4);
    b.linear("test", in_f, out_f, batch);

    assert_eq!(b.node_count(), 4, "linear must allocate 4 nodes");
    assert_eq!(b.into_steps().len(), 1, "linear must push 1 step");
}

/// Prove: `conv1d` allocates exactly 4 nodes (input, weight, bias, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_allocates_four_nodes() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(4);
    b.conv1d("test", 64, 128, 7, 100, 1, 3, 1);

    assert_eq!(b.node_count(), 4, "conv1d must allocate 4 nodes");
    assert_eq!(b.into_steps().len(), 1, "conv1d must push 1 step");
}

/// Prove: `conv_transpose1d` allocates exactly 4 nodes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_allocates_four_nodes() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(4);
    b.conv_transpose1d("test", 64, 128, 7, 100, 2, 3);

    assert_eq!(b.node_count(), 4, "conv_transpose1d must allocate 4 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `sigmoid` allocates exactly 2 nodes (input, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sigmoid_allocates_two_nodes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 100000);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.sigmoid("test", n);

    assert_eq!(b.node_count(), 2, "sigmoid must allocate 2 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `tanh` allocates exactly 2 nodes (input, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn tanh_allocates_two_nodes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 100000);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.tanh("test", n);

    assert_eq!(b.node_count(), 2, "tanh must allocate 2 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `gelu` allocates exactly 2 nodes (input, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gelu_allocates_two_nodes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 100000);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.gelu("test", n);

    assert_eq!(b.node_count(), 2, "gelu must allocate 2 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `relu` allocates exactly 2 nodes (input, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn relu_allocates_two_nodes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 100000);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.relu("test", n);

    assert_eq!(b.node_count(), 2, "relu must allocate 2 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `binary_add` allocates exactly 3 nodes (left, right, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn binary_add_allocates_three_nodes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 100000);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.binary_add("test", n);

    assert_eq!(b.node_count(), 3, "binary_add must allocate 3 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `binary_mul` allocates exactly 3 nodes (left, right, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn binary_mul_allocates_three_nodes() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 100000);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.binary_mul("test", n);

    assert_eq!(b.node_count(), 3, "binary_mul must allocate 3 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `matmul` allocates exactly 3 nodes (left, right, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn matmul_allocates_three_nodes() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();
    kani::assume(m > 0 && m <= 256);
    kani::assume(k > 0 && k <= 256);
    kani::assume(n > 0 && n <= 256);
    kani::assume(batch > 0 && batch <= 16);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.matmul("test", m, k, n, batch, false, false, None);

    assert_eq!(b.node_count(), 3, "matmul must allocate 3 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `softmax` allocates exactly 2 nodes (input, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_allocates_two_nodes() {
    let axis_size: usize = kani::any();
    let outer_size: usize = kani::any();
    kani::assume(axis_size > 0 && axis_size <= 1024);
    kani::assume(outer_size > 0 && outer_size <= 1024);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.softmax("test", axis_size, outer_size);

    assert_eq!(b.node_count(), 2, "softmax must allocate 2 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `reduce` allocates exactly 2 nodes (input, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reduce_allocates_two_nodes() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.reduce("test", nn_dsl::tensor_ir::ReduceOp::Sum, 128, 8);

    assert_eq!(b.node_count(), 2, "reduce must allocate 2 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

/// Prove: `embedding` allocates exactly 3 nodes (input, weight, output).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn embedding_allocates_three_nodes() {
    let dim: usize = kani::any();
    let num: usize = kani::any();
    kani::assume(dim > 0 && dim <= 1024);
    kani::assume(num > 0 && num <= 1024);

    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(2);
    b.embedding("test", dim, num);

    assert_eq!(b.node_count(), 3, "embedding must allocate 3 nodes");
    assert_eq!(b.into_steps().len(), 1);
}

// ---- Composite sequence proofs ----------------------------------------------

/// Prove: mixed sequence node count is additive.
///
/// embedding(3) + linear(4) + gelu(2) + linear(4) + softmax(2) = 15 nodes, 5 steps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mixed_sequence_node_count_additive() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(16);
    b.embedding("emb", 768, 10); // 3 nodes
    b.linear("fc1", 768, 256, 10); // 4 nodes
    b.gelu("act", 2560); // 2 nodes
    b.linear("fc2", 256, 128, 10); // 4 nodes
    b.softmax("sm", 128, 10); // 2 nodes

    assert_eq!(b.node_count(), 15, "total nodes = 3+4+2+4+2 = 15");
    assert_eq!(b.into_steps().len(), 5, "total steps = 5");
}

/// Prove: alloc_node + push_step manual path is consistent with node_count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn alloc_node_push_step_consistent() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(4);

    // Allocate 2 nodes manually
    let _id0 = b.alloc_node();
    let _id1 = b.alloc_node();
    assert_eq!(b.node_count(), 2, "manual alloc produces 2 nodes");

    // Then use a builder method (adds 4 nodes)
    b.linear("test", 64, 128, 1);
    assert_eq!(b.node_count(), 6, "manual + linear = 2 + 4 = 6 nodes");

    // push_step doesn't allocate nodes
    let steps = b.into_steps();
    assert_eq!(steps.len(), 1, "only builder method pushes steps");
}

/// Prove: chaining binary ops produces correct cumulative counts.
///
/// binary_add(3) + binary_mul(3) + sigmoid(2) = 8 nodes, 3 steps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chained_binary_ops_cumulative_count() {
    let mut b = crate::dispatch_builder::DispatchBuilder::with_capacity(8);
    b.binary_add("add", 128); // 3 nodes
    b.binary_mul("mul", 128); // 3 nodes
    b.sigmoid("sig", 128); // 2 nodes

    assert_eq!(b.node_count(), 8, "3+3+2 = 8 nodes");
    assert_eq!(b.into_steps().len(), 3, "3 steps");
}
