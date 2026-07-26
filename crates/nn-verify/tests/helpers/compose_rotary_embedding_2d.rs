// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: 2D Rotary Position Embedding NY composition.
//!
//! Models the RotaryEmbedding2d mechanism from Qwen2-VL and modern vision
//! transformers that operate on 2D spatial grids:
//!
//!   1. Split input [S, D] into height-half [S, D/2] and width-half [S, D/2]
//!   2. Each half is further split into even/odd pairs for rotation
//!   3. Height rotation: y_h_even = x_h_even * h_cos - x_h_odd * h_sin
//!      y_h_odd  = x_h_even * h_sin + x_h_odd * h_cos
//!   4. Width rotation: same formula with w_cos, w_sin
//!   5. Reassemble even/odd pairs within each half
//!   6. Concatenate height-half and width-half back to [S, D]
//!
//! The cos/sin tables are precomputed constants (not learned weights).
//! head_dim must be divisible by 4: half for H, half for W, each further
//! split into rotation pairs.
//!
//! Simplification for verification: single spatial grid flattened to sequence.
//! The rotation is a block-diagonal linear transform per pair, amenable to
//! linear bound propagation (IBP and CROWN).
//!
//! Dimensions: HEAD_DIM=16, SEQ_LEN=8 (e.g., 2x4 spatial grid).
//!
//! Part of #3563: SlidingWindowAttention + RotaryEmbedding2d compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length (flattened spatial grid, e.g. H=2 x W=4 = 8).
const SEQ_LEN: usize = 8;
/// Per-head dimension (must be divisible by 4 for 2D RoPE).
/// Production: 64-128 for vision transformers.
const HEAD_DIM: usize = 16;
/// Half of head_dim — dimension for each spatial axis (height or width).
const HALF_DIM: usize = HEAD_DIM / 2; // 8
/// Quarter of head_dim — number of rotation pairs per spatial axis.
const QUARTER_DIM: usize = HEAD_DIM / 4; // 4

/// Frequency base for RoPE.
const BASE: f64 = 10000.0;

// ---------------------------------------------------------------------------
// RoPE cos/sin table construction (per spatial axis)
// ---------------------------------------------------------------------------

/// Precomputed cos table for one spatial axis: cos(pos * inv_freq[i]).
///
/// Shape: `[SEQ_LEN, QUARTER_DIM]` — one value per (position, pair_index).
fn axis_cos_table(positions: &[usize]) -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * QUARTER_DIM];
    let inv_freq: Vec<f64> = (0..QUARTER_DIM)
        .map(|i| {
            let exponent = (2 * i) as f64 / HALF_DIM as f64;
            1.0 / BASE.powf(exponent)
        })
        .collect();
    for (s, &pos) in positions.iter().enumerate() {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = (pos as f64 * freq) as f32;
            data[s * QUARTER_DIM + i] = angle.cos();
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, QUARTER_DIM]), data).expect("valid cos table")
}

/// Precomputed sin table for one spatial axis: sin(pos * inv_freq[i]).
///
/// Shape: `[SEQ_LEN, QUARTER_DIM]`.
fn axis_sin_table(positions: &[usize]) -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * QUARTER_DIM];
    let inv_freq: Vec<f64> = (0..QUARTER_DIM)
        .map(|i| {
            let exponent = (2 * i) as f64 / HALF_DIM as f64;
            1.0 / BASE.powf(exponent)
        })
        .collect();
    for (s, &pos) in positions.iter().enumerate() {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = (pos as f64 * freq) as f32;
            data[s * QUARTER_DIM + i] = angle.sin();
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, QUARTER_DIM]), data).expect("valid sin table")
}

/// Height positions for a 2x4 spatial grid (raster scan order).
fn height_positions() -> Vec<usize> {
    // Grid: row 0 has 4 tokens, row 1 has 4 tokens
    // h_pos = [0, 0, 0, 0, 1, 1, 1, 1]
    vec![0, 0, 0, 0, 1, 1, 1, 1]
}

/// Width positions for a 2x4 spatial grid (raster scan order).
fn width_positions() -> Vec<usize> {
    // w_pos = [0, 1, 2, 3, 0, 1, 2, 3]
    vec![0, 1, 2, 3, 0, 1, 2, 3]
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a single-axis 1D RoPE rotation as part of the graph.
///
/// Input: `x_half` of shape [SEQ_LEN, HALF_DIM] — one spatial half.
/// cos/sin: [SEQ_LEN, QUARTER_DIM] — precomputed tables for this axis.
/// neg_one: scalar [-1] for subtraction.
///
/// Decomposes x_half into even/odd pairs, applies rotation, reassembles:
///   x_even = narrow(x_half, axis=1, start=0,           len=QUARTER_DIM)
///   x_odd  = narrow(x_half, axis=1, start=QUARTER_DIM, len=QUARTER_DIM)
///   y_even = x_even * cos - x_odd * sin
///   y_odd  = x_even * sin + x_odd * cos
///   y_half = concat([y_even, y_odd], axis=1)
fn build_rope_1d_rotation(
    b: &mut TensorBlockBuilder,
    x_half: nn_dsl::tensor_ir::TensorNodeId,
    cos: nn_dsl::tensor_ir::TensorNodeId,
    sin: nn_dsl::tensor_ir::TensorNodeId,
    neg_one_bc: nn_dsl::tensor_ir::TensorNodeId,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let pair_shape = [SEQ_LEN, QUARTER_DIM];

    // Split into even and odd components
    let x_even = b.add_narrow(x_half, 1, 0, QUARTER_DIM, &pair_shape);
    let x_odd = b.add_narrow(x_half, 1, QUARTER_DIM, QUARTER_DIM, &pair_shape);

    // y_even = x_even * cos - x_odd * sin
    let ec = b.add_binary_mul(x_even, cos, &pair_shape);
    let os = b.add_binary_mul(x_odd, sin, &pair_shape);
    let neg_os = b.add_binary_mul(os, neg_one_bc, &pair_shape);
    let y_even = b.add_binary_add(ec, neg_os, &pair_shape);

    // y_odd = x_even * sin + x_odd * cos
    let es = b.add_binary_mul(x_even, sin, &pair_shape);
    let oc = b.add_binary_mul(x_odd, cos, &pair_shape);
    let y_odd = b.add_binary_add(es, oc, &pair_shape);

    // Concatenate back to half_dim
    b.add_concat(&[y_even, y_odd], 1, &[SEQ_LEN, HALF_DIM])
}

/// Build a RotaryEmbedding2d kernel as a TensorKernelDef.
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (Variable — post-projection activations).
/// H/W cos/sin tables: each `[SEQ_LEN, QUARTER_DIM]` (Constant).
/// neg_one: scalar (Constant).
/// Output: `[SEQ_LEN, HEAD_DIM]`.
///
/// Decomposition:
///   x_h = narrow(input, axis=1, 0, HALF_DIM)              [S, D/2]
///   x_w = narrow(input, axis=1, HALF_DIM, HALF_DIM)       [S, D/2]
///   y_h = rope_1d(x_h, h_cos, h_sin)                      [S, D/2]
///   y_w = rope_1d(x_w, w_cos, w_sin)                      [S, D/2]
///   output = concat([y_h, y_w], axis=1)                    [S, D]
fn build_rotary_embedding_2d_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("rotary_embedding_2d");

    // Inputs
    let input = b.add_input("x", &[SEQ_LEN, HEAD_DIM]);
    let h_cos = b.add_input("h_cos", &[SEQ_LEN, QUARTER_DIM]);
    let h_sin = b.add_input("h_sin", &[SEQ_LEN, QUARTER_DIM]);
    let w_cos = b.add_input("w_cos", &[SEQ_LEN, QUARTER_DIM]);
    let w_sin = b.add_input("w_sin", &[SEQ_LEN, QUARTER_DIM]);
    let neg_one = b.add_input("neg_one", &[1]);

    // Broadcast neg_one to pair shape for subtraction
    let neg_one_bc = b.add_broadcast(neg_one, &[SEQ_LEN, QUARTER_DIM]);

    // Split input into height-half and width-half along dim 1
    let half_shape = [SEQ_LEN, HALF_DIM];
    let x_h = b.add_narrow(input, 1, 0, HALF_DIM, &half_shape);
    let x_w = b.add_narrow(input, 1, HALF_DIM, HALF_DIM, &half_shape);

    // Apply 1D RoPE to each half independently
    let y_h = build_rope_1d_rotation(&mut b, x_h, h_cos, h_sin, neg_one_bc);
    let y_w = build_rope_1d_rotation(&mut b, x_w, w_cos, w_sin, neg_one_bc);

    // Concatenate halves back to full dimension
    let output = b.add_concat(&[y_h, y_w], 1, &[SEQ_LEN, HEAD_DIM]);

    b.build(output).expect("valid RotaryEmbedding2d kernel")
}

/// Build parameter bindings for the RotaryEmbedding2d kernel.
///
/// x = Variable, h_cos/h_sin/w_cos/w_sin = ConstantTensor, neg_one = ConstantScalar(-1).
fn rotary_2d_bindings() -> Vec<TensorParamBinding> {
    let h_pos = height_positions();
    let w_pos = width_positions();
    vec![
        TensorParamBinding::Variable, // x [SEQ_LEN, HEAD_DIM]
        TensorParamBinding::ConstantTensor(axis_cos_table(&h_pos)), // h_cos
        TensorParamBinding::ConstantTensor(axis_sin_table(&h_pos)), // h_sin
        TensorParamBinding::ConstantTensor(axis_cos_table(&w_pos)), // w_cos
        TensorParamBinding::ConstantTensor(axis_sin_table(&w_pos)), // w_sin
        TensorParamBinding::ConstantScalar(-1.0), // neg_one
    ]
}

/// Input bounds for 2D RoPE: post-projection activations in [-2, 2].
fn rotary_2d_input_bounds() -> BoundedTensor {
    uniform_bounds(&[SEQ_LEN, HEAD_DIM], 2.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// RotaryEmbedding2d TensorKernelDef validates.
#[test]
fn test_rotary_embedding_2d_def_validates() {
    let def = build_rotary_embedding_2d_kernel();
    def.validate()
        .expect("RotaryEmbedding2d kernel should validate");
}

/// RotaryEmbedding2d translates to NY GraphNetwork.
#[test]
fn test_rotary_embedding_2d_graph_builds() {
    let def = build_rotary_embedding_2d_kernel();
    let bindings = rotary_2d_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("RotaryEmbedding2d graph should translate");

    // Narrow(2 outer + 2*2 inner) + Broadcast(1) + BinaryMul(2*4) + BinaryAdd(2*2)
    // + Concat(2 inner + 1 outer) = ~20+ nodes
    assert!(
        graph.num_nodes() >= 15,
        "RotaryEmbedding2d graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through RotaryEmbedding2d.
///
/// 2D RoPE is a rotation applied independently to height and width halves.
/// For inputs in [-2, 2], each output element is of the form:
///   y = x_a * cos(θ) ± x_b * sin(θ)  where |cos|, |sin| <= 1
/// IBP worst case: |y| <= |x_a| + |x_b| <= 4.
#[test]
fn test_rotary_embedding_2d_ibp_propagates() {
    let def = build_rotary_embedding_2d_kernel();
    let bindings = rotary_2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = rotary_2d_input_bounds();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RotaryEmbedding2d");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RotaryEmbedding2d IBP: bounds=[{lo_min}, {hi_max}]");

    // With [-2, 2] input and |cos|, |sin| <= 1:
    // Worst case per element: ±(2*1 + 2*1) = ±4.
    assert!(
        lo_min >= -5.0,
        "IBP lower should be >= -5 for [-2,2] input with 2D rotation, got {lo_min}"
    );
    assert!(
        hi_max <= 5.0,
        "IBP upper should be <= 5 for [-2,2] input with 2D rotation, got {hi_max}"
    );
}

/// CROWN bounds propagate through RotaryEmbedding2d.
///
/// 2D RoPE is linear (rotation matrix applied to input per spatial axis),
/// so CROWN should produce tight bounds (no nonlinear relaxation needed).
#[test]
fn test_rotary_embedding_2d_crown_propagation() {
    let def = build_rotary_embedding_2d_kernel();
    let bindings = rotary_2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = rotary_2d_input_bounds();

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RotaryEmbedding2d: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// RotaryEmbedding2d verify and record under "rotary_embedding_2d" key.
#[test]
fn test_rotary_embedding_2d_verify_and_record() {
    let def = build_rotary_embedding_2d_kernel();
    let bindings = rotary_2d_bindings();
    let input = rotary_2d_input_bounds();

    let result = verify_and_assert(&def, &bindings, &input, "rotary_embedding_2d");
    assert_eq!(result.num_variables, 1, "single Variable input (x)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HEAD_DIM]);
}
