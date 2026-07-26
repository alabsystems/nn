// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Concat tensor op → NY ConcatLayer.
//!
//! Verifies that `TensorOpKind::Concat` translates correctly to NY
//! `ConcatLayer` with IBP and CROWN bound propagation.
//!
//! Multi-variable stacking requires all Variable inputs to share the same
//! shape (axis 0 is the stacking dimension). Tests use equal-shape inputs
//! and concat along axis 1 to double the channel dimension.
//!
//! Part of #810 — Concat along existing axis.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Two-input concat along axis 1 (channel dimension)
// ---------------------------------------------------------------------------

/// Build a kernel that concatenates two [B, C, T] inputs along axis 1 (channels).
///
/// Both inputs have the same shape [B, C, T]; output is [B, 2*C, T].
/// Multi-variable stacking requires equal shapes.
fn build_concat_same_shape(
    batch: usize,
    channels: usize,
    t: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("concat_channels");
    let a = b.add_input("a", &[batch, channels, t]);
    let b_input = b.add_input("b", &[batch, channels, t]);
    let _out = b.add_concat(&[a, b_input], 1, &[batch, 2 * channels, t]);
    b.build(_out).expect("valid graph")
}

/// Concat graph builds and has correct node count.
#[test]
fn test_concat_graph_builds() {
    let def = build_concat_same_shape(1, 4, 16);
    def.validate().expect("concat should validate");

    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("concat graph");
    // 2 variable inputs → SliceLayer each, plus ConcatLayer = at least 3 nodes
    assert!(
        graph.num_nodes() >= 3,
        "concat graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through concat.
#[test]
fn test_concat_ibp_propagates() {
    let def = build_concat_same_shape(1, 4, 16);
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Multi-variable input is a flat element buffer (the translator reshapes the
    // network input to [-1] and peels each variable's elements by count, then
    // restores its TRUE shape — see #358). Total = 2 vars * (1*4*16) = 128 elems.
    let num_vars = 2;
    let lower = ArrayD::from_elem(IxDyn(&[num_vars, 1, 4, 16]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[num_vars, 1, 4, 16]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through concat");
    let (lo, hi) = output.lower_upper();

    // Output is the natural concat shape [B, 2*C, T] = [1, 8, 16] — no leading
    // stacking axis (variables enter at their declared rank, axis_offset = 0).
    assert_eq!(lo.shape(), &[1, 8, 16]);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "lower must be finite, got {l}");
        assert!(u.is_finite(), "upper must be finite, got {u}");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// CROWN propagation through concat.
#[test]
fn test_concat_crown_propagates() {
    let def = build_concat_same_shape(1, 4, 16);
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let num_vars = 2;
    let lower = ArrayD::from_elem(IxDyn(&[num_vars, 1, 4, 16]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[num_vars, 1, 4, 16]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (_, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through concat");
    let (lo, hi) = output.lower_upper();

    // Natural concat shape [B, 2*C, T] = [1, 8, 16] — no leading stacking axis.
    assert_eq!(lo.shape(), &[1, 8, 16]);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "lower must be finite, got {l}");
        assert!(u.is_finite(), "upper must be finite, got {u}");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// ---------------------------------------------------------------------------
// Three-input concat (multi-head merge pattern)
// ---------------------------------------------------------------------------

/// Build a 3-input concat along axis 1, simulating head merging.
///
/// 3 heads of [B, H, T] → concatenate to [B, 3*H, T].
fn build_concat_three_heads(
    batch: usize,
    head_dim: usize,
    t: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("concat_3heads");
    let h0 = b.add_input("head0", &[batch, head_dim, t]);
    let h1 = b.add_input("head1", &[batch, head_dim, t]);
    let h2 = b.add_input("head2", &[batch, head_dim, t]);
    let _out = b.add_concat(&[h0, h1, h2], 1, &[batch, 3 * head_dim, t]);
    b.build(_out).expect("valid graph")
}

/// Three-input concat IBP propagation.
#[test]
fn test_concat_three_inputs_ibp() {
    let def = build_concat_three_heads(1, 4, 8);
    def.validate().expect("3-head concat should validate");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("3-head concat graph");

    let num_vars = 3;
    let lower = ArrayD::from_elem(IxDyn(&[num_vars, 1, 4, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[num_vars, 1, 4, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-head concat");
    let (lo, hi) = output.lower_upper();

    // Output is the natural concat shape [B, 3*H, T] = [1, 12, 8] — no leading
    // stacking axis (variables enter at their declared rank, axis_offset = 0).
    assert_eq!(lo.shape(), &[1, 12, 8]);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "lower must be finite, got {l}");
        assert!(u.is_finite(), "upper must be finite, got {u}");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------

/// Concat rejects mismatched non-concat dimensions.
#[test]
fn test_concat_rejects_shape_mismatch() {
    let mut b = TensorBlockBuilder::new("bad_concat");
    let a = b.add_input("a", &[1, 4, 16]);
    let b_input = b.add_input("b", &[1, 4, 32]); // T mismatch (16 vs 32)
    let _out = b.add_concat(&[a, b_input], 1, &[1, 8, 16]);
    assert!(
        b.build(_out).is_err(),
        "concat with mismatched non-concat dims should fail validation"
    );
}

/// Concat ALONG axis 0 is a legitimate user-space data axis (e.g. sequence/token
/// concat for [T, D] kernels). The framework's variable/batch stacking axis is
/// injected below user axes at translation time (axis_offset, #358), so user
/// axis 0 never aliases the packing axis — it is accepted, not reserved. See
/// `validate_concat` in nn-dsl/tensor_ir_validate_structural.rs.
#[test]
fn test_concat_accepts_axis_zero() {
    let mut b = TensorBlockBuilder::new("axis0_concat");
    let a = b.add_input("a", &[4, 16]);
    let b_input = b.add_input("b", &[4, 16]);
    let _out = b.add_concat(&[a, b_input], 0, &[8, 16]);
    b.build(_out)
        .expect("concat along user axis 0 should validate");
}

/// Concat rejects single input.
#[test]
fn test_concat_rejects_single_input() {
    let mut b = TensorBlockBuilder::new("single_concat");
    let a = b.add_input("a", &[4, 16]);
    let _out = b.add_concat(&[a], 1, &[4, 16]);
    assert!(
        b.build(_out).is_err(),
        "concat with single input should fail validation"
    );
}
