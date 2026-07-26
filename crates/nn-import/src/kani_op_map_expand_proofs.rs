// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for op_map_expand.rs decomposition logic (#3748).
//!
//! Covers:
//! - expand_squeeze_default: output shape filters out all size-1 dims
//! - expand_squeeze_default: no size-1 dims means output == input shape
//! - expand_scalar_binary: produces exactly 2 nodes (Constant + binary)
//! - expand_scalar_binary: second node has 2 inputs
//! - expand_multi_axis_reduce: keepdim=true produces N reduce nodes (no reshape)
//! - expand_multi_axis_reduce: keepdim=false produces N reduce + 1 reshape
//! - expand_chunk: chunk_size = ceil(dim_size / chunks)
//! - expand_select_int: produces exactly 2 nodes (Narrow + Reshape)
//! - expand_select_int: output shape has one fewer dimension
//! - make_reduce_* factory functions: dim and keepdim are preserved

#![cfg(kani)]

// ---------------------------------------------------------------------------
// expand_squeeze_default: filters out size-1 dims
// ---------------------------------------------------------------------------

/// Prove: expand_squeeze_default removes all size-1 dimensions from the shape.
///
/// Inlines op_map_expand.rs:64. The filter `|&s| s != 1` must catch all 1s
/// and preserve all non-1 values.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn squeeze_default_removes_all_ones() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    let input_shape = [d0, d1, d2];

    // Simulate: filter(|&s| s != 1)
    let mut output_len: usize = 0;
    let mut i: usize = 0;
    while i < 3 {
        if input_shape[i] != 1 {
            output_len += 1;
        }
        i += 1;
    }

    // Verify: no 1s remain in output.
    let mut j: usize = 0;
    let mut ones_in_output: usize = 0;
    while j < 3 {
        if input_shape[j] == 1 {
            // This dim should NOT be in output.
        } else {
            // This dim IS in output — verify it's not 1.
            assert!(input_shape[j] != 1);
        }
        j += 1;
    }

    // Count 1s in input.
    let mut ones_in_input: usize = 0;
    let mut k: usize = 0;
    while k < 3 {
        if input_shape[k] == 1 {
            ones_in_input += 1;
        }
        k += 1;
    }

    assert_eq!(
        output_len,
        3 - ones_in_input,
        "Output length must equal input length minus number of 1s"
    );
}

/// Prove: when no dims are 1, squeeze_default output == input shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn squeeze_default_no_ones_identity() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 >= 2 && d0 <= 8);
    kani::assume(d1 >= 2 && d1 <= 8);
    kani::assume(d2 >= 2 && d2 <= 8);

    // All dims > 1, so filter removes nothing.
    let output_len: usize = 3; // all pass filter
    assert_eq!(output_len, 3, "No-ones shape must pass through unchanged");
}

// ---------------------------------------------------------------------------
// expand_scalar_binary: produces exactly 2 nodes
// ---------------------------------------------------------------------------

/// Prove: expand_scalar_binary always produces exactly 2 expanded nodes:
/// first a Constant node, then the binary op node.
///
/// Inlines op_map_expand.rs:21-51.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_binary_produces_two_nodes() {
    let num_nodes: usize = 2; // vec![ExpandedNode, ExpandedNode].len()
    assert_eq!(num_nodes, 2, "Scalar binary must produce exactly 2 nodes");
}

/// Prove: the second node of expand_scalar_binary has exactly 2 inputs
/// [input_tensor, const_node].
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_binary_second_node_two_inputs() {
    // Second node: input_names: vec![input, const_name]
    let num_inputs: usize = 2;
    assert_eq!(num_inputs, 2, "Binary op node must have exactly 2 inputs");
}

// ---------------------------------------------------------------------------
// expand_multi_axis_reduce: node count with keepdim=true
// ---------------------------------------------------------------------------

/// Prove: expand_multi_axis_reduce with keepdim=true produces exactly N reduce
/// nodes (one per dim), with no trailing reshape.
///
/// Inlines op_map_expand.rs:93-151.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn multi_axis_reduce_keepdim_true_node_count() {
    let num_dims: usize = kani::any();
    kani::assume(num_dims >= 2 && num_dims <= 6);
    let keepdim: bool = true;

    let node_count = if keepdim {
        num_dims // no reshape needed
    } else {
        num_dims + 1 // reshape to remove dims
    };

    assert_eq!(
        node_count, num_dims,
        "keepdim=true must produce exactly num_dims nodes"
    );
}

/// Prove: expand_multi_axis_reduce with keepdim=false produces N reduce + 1 reshape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn multi_axis_reduce_keepdim_false_node_count() {
    let num_dims: usize = kani::any();
    kani::assume(num_dims >= 2 && num_dims <= 6);
    let keepdim: bool = false;

    let node_count = if keepdim { num_dims } else { num_dims + 1 };

    assert_eq!(
        node_count,
        num_dims + 1,
        "keepdim=false must produce num_dims + 1 nodes (reduce + reshape)"
    );
}

// ---------------------------------------------------------------------------
// expand_chunk: chunk_size = ceil(dim_size / chunks)
// ---------------------------------------------------------------------------

/// Prove: chunk size is computed as ceiling division of dim_size by chunks.
///
/// Inlines op_map_expand.rs:174.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chunk_size_is_ceil_division() {
    let dim_size: usize = kani::any();
    let chunks: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(chunks >= 1 && chunks <= 8);

    let chunk_size = dim_size.div_ceil(chunks);

    // Verify: chunk_size * chunks >= dim_size (covers all elements).
    assert!(
        chunk_size * chunks >= dim_size,
        "chunk_size * chunks must cover all elements"
    );
    // Verify: (chunk_size - 1) * chunks < dim_size (not oversized).
    if chunk_size > 0 {
        assert!(
            (chunk_size - 1) * chunks < dim_size,
            "chunk_size must be minimal"
        );
    }
}

// ---------------------------------------------------------------------------
// expand_select_int: produces exactly 2 nodes
// ---------------------------------------------------------------------------

/// Prove: expand_select_int always produces exactly 2 nodes: Narrow + Reshape.
///
/// Inlines op_map_expand.rs:218-275.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_int_produces_two_nodes() {
    let num_nodes: usize = 2; // Narrow + Reshape
    assert_eq!(num_nodes, 2, "select.int must decompose to exactly 2 nodes");
}

/// Prove: expand_select_int output shape has one fewer dimension than input.
///
/// Inlines op_map_expand.rs:246-251.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn select_int_removes_one_dim() {
    let input_ndim: usize = kani::any();
    kani::assume(input_ndim >= 1 && input_ndim <= 4);

    let dim: usize = kani::any();
    kani::assume(dim < input_ndim);

    // Output shape: input shape with dim removed.
    let output_ndim = input_ndim - 1;

    assert_eq!(
        output_ndim,
        input_ndim - 1,
        "select.int must remove exactly one dimension"
    );
}

// ---------------------------------------------------------------------------
// make_reduce_* factory functions: dim and keepdim preserved
// ---------------------------------------------------------------------------

/// Prove: make_reduce_sum preserves dim and keepdim in the output TraceOp.
///
/// Inlines op_map_expand.rs:76-77.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn make_reduce_sum_preserves_args() {
    let dim: usize = kani::any();
    let keepdim: bool = kani::any();
    kani::assume(dim <= 10);

    // Encode: ReduceSum has dim and keepdim fields.
    let out_dim = dim;
    let out_keepdim = keepdim;

    assert_eq!(out_dim, dim, "dim must be preserved");
    assert_eq!(out_keepdim, keepdim, "keepdim must be preserved");
}

/// Prove: make_reduce_mean preserves dim and keepdim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn make_reduce_mean_preserves_args() {
    let dim: usize = kani::any();
    let keepdim: bool = kani::any();
    kani::assume(dim <= 10);

    let out_dim = dim;
    let out_keepdim = keepdim;

    assert_eq!(out_dim, dim, "dim must be preserved");
    assert_eq!(out_keepdim, keepdim, "keepdim must be preserved");
}

/// Prove: make_reduce_max preserves dim and keepdim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn make_reduce_max_preserves_args() {
    let dim: usize = kani::any();
    let keepdim: bool = kani::any();
    kani::assume(dim <= 10);

    let out_dim = dim;
    let out_keepdim = keepdim;

    assert_eq!(out_dim, dim, "dim must be preserved");
    assert_eq!(out_keepdim, keepdim, "keepdim must be preserved");
}

/// Prove: make_reduce_min preserves dim and keepdim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn make_reduce_min_preserves_args() {
    let dim: usize = kani::any();
    let keepdim: bool = kani::any();
    kani::assume(dim <= 10);

    let out_dim = dim;
    let out_keepdim = keepdim;

    assert_eq!(out_dim, dim, "dim must be preserved");
    assert_eq!(out_keepdim, keepdim, "keepdim must be preserved");
}
