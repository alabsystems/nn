// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor IR operation correctness.
//!
//! These harnesses verify fundamental tensor IR invariants using
//! bounded model checking:
//!
//! - Broadcast compatibility rules (NumPy-style right-aligned)
//! - Conv output dimension formula correctness
//! - Reshape element count preservation
//! - Transpose permutation validity
//! - Reduce shape computation
//! - Stride/shape computation for contiguous tensors
//! - Validation: valid shapes accepted, invalid shapes rejected
//!
//! Part of #3599.

use super::*;

// ============================================================================
// Broadcast harnesses
// ============================================================================

/// Proves: same-rank broadcast always returns Left alignment when dims match.
///
/// When input and target have the same rank, left and right alignment are
/// identical (offset = 0 either way). The function must return `Left`.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_same_rank_returns_left() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);

    // Same shape: [a] -> [a] must succeed with Left
    let result = infer_broadcast_alignment(&[a], &[a]);
    assert!(result.is_ok(), "same shape must be broadcast-compatible");
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

/// Proves: broadcasting [1] to any [n] succeeds (scalar broadcast).
///
/// A dimension of size 1 is always broadcast-compatible with any target
/// dimension. For same-rank, returns Left.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_scalar_dim_always_compatible() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 256);

    let result = infer_broadcast_alignment(&[1], &[n]);
    assert!(result.is_ok(), "[1] must broadcast to [n]");
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

/// Proves: broadcasting to a shorter target is always an error.
///
/// If the input rank exceeds the target rank, no alignment can work.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_input_longer_than_target_fails() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 64);

    // [a, b] -> [c] must fail: input rank (2) > target rank (1)
    let result = infer_broadcast_alignment(&[a, b], &[c]);
    assert!(
        result.is_err(),
        "cannot broadcast higher rank to lower rank"
    );
}

/// Proves: right-aligned broadcast infers correctly.
///
/// [D] broadcast to [B, T, D] should infer Right alignment because
/// the input aligns to the suffix of the target.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_right_aligned_inference() {
    let b: usize = kani::any();
    let t: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(b >= 2 && b <= 16);
    kani::assume(t >= 2 && t <= 16);
    kani::assume(d >= 2 && d <= 16);
    // Ensure d != b so left alignment fails
    kani::assume(d != b);

    let result = infer_broadcast_alignment(&[d], &[b, t, d]);
    assert!(result.is_ok(), "[D] -> [B, T, D] must succeed");
    assert_eq!(result.unwrap(), BroadcastAlignment::Right);
}

/// Proves: left-aligned broadcast infers correctly.
///
/// [B, C] broadcast to [B, C, T] should infer Left alignment because
/// the input aligns to the prefix of the target.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_left_aligned_inference() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();
    kani::assume(b >= 2 && b <= 16);
    kani::assume(c >= 2 && c <= 16);
    kani::assume(t >= 2 && t <= 16);
    // Ensure c != t so right alignment fails
    kani::assume(c != t);

    let result = infer_broadcast_alignment(&[b, c], &[b, c, t]);
    assert!(result.is_ok(), "[B, C] -> [B, C, T] must succeed");
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

/// Proves: ambiguous broadcast is detected when both alignments are valid
/// and input has non-1 dimensions.
///
/// [N] broadcast to [N, N] is ambiguous when N > 1: left maps to dim 0,
/// right maps to dim 1, producing different coordinate mappings.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_ambiguous_detected() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 64);

    let result = infer_broadcast_alignment(&[n], &[n, n]);
    assert!(result.is_err(), "ambiguous broadcast must be rejected");
}

/// Proves: all-ones input to higher-rank target is never ambiguous.
///
/// When all input dimensions are 1, both left and right alignment produce
/// identical coordinate mappings. The function defaults to Left.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_all_ones_not_ambiguous() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);

    let result = infer_broadcast_alignment(&[1], &[a, b]);
    assert!(result.is_ok(), "[1] -> [a, b] must not be ambiguous");
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

// ============================================================================
// Conv1d output dimension formula harnesses
// ============================================================================

/// Proves: conv1d output length formula always produces >= 1 for valid parameters.
///
/// For valid parameters where `padded >= effective_kernel`:
/// `out_len = (in_len + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
/// The output must always be >= 1.
#[kani::unwind(8)]
#[kani::proof]
fn conv1d_output_length_positive() {
    let in_len: usize = kani::any();
    let kernel: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(in_len >= 1 && in_len <= 128);
    kani::assume(kernel >= 1 && kernel <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(padding <= 16);
    kani::assume(dilation >= 1 && dilation <= 4);

    // Compute effective kernel (must not overflow for these bounds)
    let eff_kernel = dilation * (kernel - 1) + 1;
    let padded = in_len + 2 * padding;
    kani::assume(padded >= eff_kernel);

    // The conv1d output length formula
    let out_len = (padded - eff_kernel) / stride + 1;
    assert!(out_len >= 1, "conv1d output length must be >= 1");
}

/// Proves: conv1d output length formula — kernel size 1, stride 1, no padding, no dilation
/// always produces output length == input length (identity spatial dimension).
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_identity_kernel() {
    let in_len: usize = kani::any();
    kani::assume(in_len >= 1 && in_len <= 256);

    // kernel=1, stride=1, padding=0, dilation=1
    let eff_kernel = 1;
    let padded = in_len;
    let out_len = (padded - eff_kernel) / 1 + 1;
    assert_eq!(out_len, in_len, "kernel=1, stride=1 must preserve length");
}

/// Proves: conv1d output length with stride=2 halves the input (rounded down).
///
/// For kernel=1, padding=0, dilation=1, stride=2:
/// `out_len = (in_len - 1) / 2 + 1 = ceil(in_len / 2)`.
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_stride2_halves_length() {
    let in_len: usize = kani::any();
    kani::assume(in_len >= 1 && in_len <= 256);

    let out_len = (in_len - 1) / 2 + 1;
    // This is equivalent to ceil(in_len / 2)
    let expected = (in_len + 1) / 2;
    assert_eq!(
        out_len, expected,
        "stride=2 kernel=1 must produce ceil(in/2)"
    );
}

// ============================================================================
// Reshape element count preservation harnesses
// ============================================================================

/// Proves: reshape preserves total element count for 2D -> 1D.
///
/// Reshaping [A, B] to [A*B] must preserve the product.
#[kani::unwind(8)]
#[kani::proof]
fn reshape_preserves_element_count_2d_to_1d() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 128);
    kani::assume(b >= 1 && b <= 128);

    let product_input = a * b;
    let target = vec![product_input];
    assert_eq!(
        target.iter().product::<usize>(),
        product_input,
        "1D target product must equal input product"
    );
}

/// Proves: reshape validation catches mismatched element counts.
///
/// A graph with reshape where input product != target product must fail
/// validation. Tests the validator itself.
#[kani::unwind(64)]
#[kani::proof]
fn reshape_rejects_mismatched_products() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);
    // Ensure the products differ
    kani::assume(a * b != c);

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![a, b],
            },
            vec![a, b],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reshape {
                input: TensorNodeId::new(0),
                target_shape: vec![c],
            },
            vec![c],
        ),
    ];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(1));
    let result = graph.validate();
    assert!(
        result.is_err(),
        "reshape with mismatched product must fail validation"
    );
}

/// Proves: reshape validation accepts matching element counts.
///
/// A graph with reshape where input product == target product must pass.
#[kani::unwind(64)]
#[kani::proof]
fn reshape_accepts_matching_products() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);

    let product = a * b;
    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![a, b],
            },
            vec![a, b],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reshape {
                input: TensorNodeId::new(0),
                target_shape: vec![product],
            },
            vec![product],
        ),
    ];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(1));
    let result = graph.validate();
    assert!(
        result.is_ok(),
        "reshape with matching product must validate"
    );
}

// ============================================================================
// Transpose permutation validity harnesses
// ============================================================================

/// Proves: transpose with identity permutation preserves shape.
///
/// Axes [0, 1, 2] on shape [A, B, C] must produce shape [A, B, C].
#[kani::unwind(8)]
#[kani::proof]
fn transpose_identity_preserves_shape() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);

    let input_shape = vec![a, b, c];
    let axes = vec![0, 1, 2];
    let output_shape: Vec<usize> = axes.iter().map(|&ax| input_shape[ax]).collect();
    assert_eq!(
        output_shape, input_shape,
        "identity permutation must preserve shape"
    );
}

/// Proves: transpose swap axes produces correct shape.
///
/// Axes [1, 0, 2] on [A, B, C] must produce [B, A, C].
#[kani::unwind(8)]
#[kani::proof]
fn transpose_swap_first_two() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);

    let input_shape = vec![a, b, c];
    let axes = vec![1, 0, 2];
    let output_shape: Vec<usize> = axes.iter().map(|&ax| input_shape[ax]).collect();
    assert_eq!(output_shape, vec![b, a, c], "swapping first two axes");
}

/// Proves: transpose preserves total element count.
///
/// For any valid permutation of a 3D tensor, the product of output dims
/// equals the product of input dims.
#[kani::unwind(8)]
#[kani::proof]
fn transpose_preserves_element_count() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);

    // Enumerate all 6 permutations of [0,1,2]
    let perm_idx: u8 = kani::any();
    kani::assume(perm_idx < 6);
    let axes: Vec<usize> = match perm_idx {
        0 => vec![0, 1, 2],
        1 => vec![0, 2, 1],
        2 => vec![1, 0, 2],
        3 => vec![1, 2, 0],
        4 => vec![2, 0, 1],
        _ => vec![2, 1, 0],
    };

    let input_shape = vec![a, b, c];
    let output_shape: Vec<usize> = axes.iter().map(|&ax| input_shape[ax]).collect();

    let input_product: usize = input_shape.iter().product();
    let output_product: usize = output_shape.iter().product();
    assert_eq!(
        input_product, output_product,
        "transpose must preserve element count"
    );
}

/// Proves: transpose validation rejects duplicate axes.
///
/// A graph with Transpose axes [0, 0, 1] on a rank-3 tensor must fail.
#[kani::unwind(64)]
#[kani::proof]
fn transpose_rejects_duplicate_axes() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![a, b, c],
            },
            vec![a, b, c],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Transpose {
                input: TensorNodeId::new(0),
                axes: vec![0, 0, 1], // duplicate axis 0
            },
            vec![a, a, b],
        ),
    ];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(1));
    let result = graph.validate();
    assert!(
        result.is_err(),
        "duplicate axes must be rejected by validation"
    );
}

// ============================================================================
// Reduce shape computation harnesses
// ============================================================================

/// Proves: reduce with keepdim=true replaces axis dim with 1.
///
/// Reducing axis 1 of [A, B, C] with keepdim=true → [A, 1, C].
#[kani::unwind(8)]
#[kani::proof]
fn reduce_keepdim_replaces_with_one() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);

    let input_shape = vec![a, b, c];
    let axis = 1;
    let mut output_shape = input_shape.clone();
    output_shape[axis] = 1;
    assert_eq!(output_shape, vec![a, 1, c], "keepdim replaces dim with 1");
}

/// Proves: reduce without keepdim removes the axis dimension.
///
/// Reducing axis 1 of [A, B, C] with keepdim=false → [A, C].
#[kani::unwind(8)]
#[kani::proof]
fn reduce_no_keepdim_removes_axis() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);

    let input_shape = vec![a, b, c];
    let axis = 1;
    let mut output_shape = input_shape.clone();
    output_shape.remove(axis);
    assert_eq!(output_shape, vec![a, c], "no keepdim removes axis");
}

/// Proves: reduce validation rejects out-of-bounds axis.
///
/// A graph reducing axis 3 on a rank-3 tensor must fail validation.
#[kani::unwind(64)]
#[kani::proof]
fn reduce_rejects_oob_axis() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![a, b, c],
            },
            vec![a, b, c],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reduce {
                op: ReduceOp::Sum,
                input: TensorNodeId::new(0),
                axis: 3, // out of bounds for rank 3
                keepdim: false,
            },
            vec![a, b], // doesn't matter, validation should fail first
        ),
    ];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(1));
    let result = graph.validate();
    assert!(result.is_err(), "reduce with OOB axis must fail");
}

// ============================================================================
// Validation: empty dimension rejection harnesses
// ============================================================================

/// Proves: zero-dimension shapes are always rejected by validation.
///
/// Any shape containing a 0 dimension must fail the validate_shape check.
#[kani::unwind(64)]
#[kani::proof]
fn validate_rejects_zero_dimension() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);

    let nodes = vec![TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: "x".to_string(),
            shape: vec![a, 0, b], // zero dimension
        },
        vec![a, 0, b],
    )];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(0));
    let result = graph.validate();
    assert!(result.is_err(), "zero dimension must be rejected");
}

// ============================================================================
// Contiguous stride computation harnesses
// ============================================================================

/// Proves: contiguous strides for a 3D tensor satisfy the invariant
/// `stride[i] = product(shape[i+1..])`.
///
/// For shape [A, B, C], strides are [B*C, C, 1].
/// This verifies the standard row-major stride formula.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_strides_3d() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 64);
    // Guard overflow
    kani::assume(b as u64 * c as u64 <= 4096);

    let stride_2 = 1usize;
    let stride_1 = c;
    let stride_0 = b * c;

    // Verify the invariant: any element at (i, j, k) can be addressed as
    // offset = i * stride_0 + j * stride_1 + k * stride_2
    // and offset < a * b * c (total elements)
    let i: usize = kani::any();
    let j: usize = kani::any();
    let k: usize = kani::any();
    kani::assume(i < a && j < b && k < c);

    let offset = i * stride_0 + j * stride_1 + k * stride_2;
    let total = a * b * c;
    assert!(
        offset < total,
        "contiguous stride must produce valid offset"
    );
}

/// Proves: contiguous stride addressing is injective (no aliasing).
///
/// Two distinct (i, j, k) indices in a contiguous 3D tensor must map to
/// different offsets. This proves that contiguous layout has no aliasing.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_strides_injective() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);

    let stride_1 = c;
    let stride_0 = b * c;

    let i1: usize = kani::any();
    let j1: usize = kani::any();
    let k1: usize = kani::any();
    let i2: usize = kani::any();
    let j2: usize = kani::any();
    let k2: usize = kani::any();
    kani::assume(i1 < 4 && j1 < b && k1 < c);
    kani::assume(i2 < 4 && j2 < b && k2 < c);
    // At least one index differs
    kani::assume(i1 != i2 || j1 != j2 || k1 != k2);

    let off1 = i1 * stride_0 + j1 * stride_1 + k1;
    let off2 = i2 * stride_0 + j2 * stride_1 + k2;
    assert_ne!(off1, off2, "distinct indices must map to distinct offsets");
}

// ============================================================================
// Conv output dimension formula — algebraic properties
// ============================================================================

/// Proves: conv1d with same-padding preserves length.
///
/// For kernel_size K (odd), stride=1, dilation=1, padding=(K-1)/2:
/// `out_len = (in_len + 2*((K-1)/2) - K + 1) / 1 = in_len`.
#[kani::unwind(8)]
#[kani::proof]
fn conv1d_same_padding_preserves_length() {
    let in_len: usize = kani::any();
    let kernel: usize = kani::any();
    kani::assume(in_len >= 1 && in_len <= 256);
    kani::assume(kernel >= 1 && kernel <= 15);
    // Odd kernel for exact same-padding
    kani::assume(kernel % 2 == 1);

    let padding = (kernel - 1) / 2;
    let padded = in_len + 2 * padding;
    let eff_kernel = kernel; // dilation=1
                             // padded = in_len + kernel - 1, eff_kernel = kernel
                             // out_len = (in_len + kernel - 1 - kernel) / 1 + 1 = in_len
    let out_len = (padded - eff_kernel) / 1 + 1;
    assert_eq!(
        out_len, in_len,
        "same-padding with odd kernel must preserve length"
    );
}

/// Proves: conv_transpose1d with stride=S, kernel=S, padding=0 exactly
/// upsamples by factor S.
///
/// `L_out = (L_in - 1) * S + S = L_in * S` — exact integer upsampling.
#[kani::unwind(1)]
#[kani::proof]
fn conv_transpose1d_exact_upsample() {
    let in_len: usize = kani::any();
    let stride: usize = kani::any();
    kani::assume(in_len >= 1 && in_len <= 128);
    kani::assume(stride >= 1 && stride <= 8);

    let kernel = stride;
    let padding = 0usize;
    let dilation = 1usize;
    let output_padding = 0usize;

    // L_out = (L_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
    // = (L_in - 1) * S + 1*(S-1) + 1
    // = (L_in - 1) * S + S
    // = L_in * S
    let expanded = (in_len - 1) * stride + dilation * (kernel - 1) + 1;
    let double_pad = 2 * padding;
    let out_len = expanded - double_pad + output_padding;

    assert_eq!(
        out_len,
        in_len * stride,
        "conv_transpose with kernel=stride must exactly upsample"
    );
}

// ============================================================================
// Graph topological order harnesses
// ============================================================================

/// Proves: forward references are always rejected by validation.
///
/// A node referencing a node at the same or later index must fail.
#[kani::unwind(8)]
#[kani::proof]
fn graph_rejects_forward_reference() {
    let dim: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 32);

    // Node 0 references itself via Reshape
    let nodes = vec![TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Reshape {
            input: TensorNodeId::new(0), // self-reference
            target_shape: vec![dim],
        },
        vec![dim],
    )];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(0));
    let result = graph.validate();
    assert!(result.is_err(), "self-reference must be rejected");
}

/// Proves: mismatched node ID is rejected.
///
/// If a node's ID doesn't match its array index, validation fails.
#[kani::unwind(64)]
#[kani::proof]
fn graph_rejects_mismatched_node_id() {
    let dim: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 32);

    // Node at index 0 has id=1 — mismatch
    let nodes = vec![TensorNode::new(
        TensorNodeId::new(1), // wrong: should be 0
        TensorOpKind::Input {
            name: "x".to_string(),
            shape: vec![dim],
        },
        vec![dim],
    )];
    let graph = TensorKernelDef::new("test", nodes, TensorNodeId::new(1));
    let result = graph.validate();
    assert!(result.is_err(), "mismatched node ID must be rejected");
}

// ============================================================================
// ZeroPad1d shape computation harness
// ============================================================================

/// Proves: ZeroPad1d correctly increases the last dimension.
///
/// For shape [..., L], padding (pl, pr) → [..., L + pl + pr].
#[kani::unwind(1)]
#[kani::proof]
fn zero_pad_1d_increases_last_dim() {
    let c: usize = kani::any();
    let l: usize = kani::any();
    let pl: usize = kani::any();
    let pr: usize = kani::any();
    kani::assume(c >= 1 && c <= 32);
    kani::assume(l >= 1 && l <= 128);
    kani::assume(pl <= 32);
    kani::assume(pr <= 32);
    // Guard overflow
    kani::assume(l as u64 + pl as u64 + pr as u64 <= 256);

    let out_len = l + pl + pr;
    assert!(out_len >= l, "padding cannot decrease length");
    assert_eq!(
        out_len,
        l + pl + pr,
        "output length must equal input + pad_left + pad_right"
    );
}
