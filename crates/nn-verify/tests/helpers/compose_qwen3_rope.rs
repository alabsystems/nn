// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3 RoPE (Rotary Position Embedding) NY composition.
//!
//! Models the RoPE rotation applied to Q/K projections in Qwen3 attention:
//!
//!   x_rot[i]   = x[i]   * cos(θ_i) - x[i+d/2] * sin(θ_i)
//!   x_rot[i+d/2] = x[i] * sin(θ_i) + x[i+d/2] * cos(θ_i)
//!
//! where θ_i = pos / 10000^(2i/d) for position `pos` and dimension index `i`.
//!
//! The cos/sin tables are precomputed constants (not learned weights).
//! The rotation is a 2x2 block-diagonal linear transform per pair, making
//! it amenable to linear bound propagation (IBP and CROWN).
//!
//! Simplification: single position (no sequence dimension batching) to keep
//! the graph small for verification. Multi-position is a broadcast extension.
//!
//! Part of #3560: Qwen3 RoPE + GQA NY compose verification tests.

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

/// Sequence length (number of token positions).
const SEQ_LEN: usize = 8;
/// Per-head dimension (production: 128 for Qwen3-8B with 4096/32).
const HEAD_DIM: usize = 16;
/// Half of head_dim — number of rotation pairs.
const HALF_DIM: usize = HEAD_DIM / 2;

// ---------------------------------------------------------------------------
// RoPE cos/sin table construction
// ---------------------------------------------------------------------------

/// Build precomputed RoPE cos table: cos(pos / 10000^(2i/d)).
///
/// Shape: `[SEQ_LEN, HALF_DIM]` — one value per (position, pair_index).
fn rope_cos_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    for pos in 0..SEQ_LEN {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HALF_DIM]), data).expect("valid cos table")
}

/// Build precomputed RoPE sin table: sin(pos / 10000^(2i/d)).
///
/// Shape: `[SEQ_LEN, HALF_DIM]`.
fn rope_sin_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    for pos in 0..SEQ_LEN {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.sin() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HALF_DIM]), data).expect("valid sin table")
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a RoPE kernel as a TensorKernelDef.
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (Variable — the Q or K projection for one head).
/// Cos table: `[SEQ_LEN, HALF_DIM]` (Constant).
/// Sin table: `[SEQ_LEN, HALF_DIM]` (Constant).
/// Output: `[SEQ_LEN, HEAD_DIM]`.
///
/// Decomposition:
///   x_first  = narrow(input, axis=1, start=0, len=HALF_DIM)        [S, d/2]
///   x_second = narrow(input, axis=1, start=HALF_DIM, len=HALF_DIM) [S, d/2]
///   rot_first  = x_first * cos - x_second * sin                    [S, d/2]
///   rot_second = x_first * sin + x_second * cos                    [S, d/2]
///   output = concat([rot_first, rot_second], axis=1)                [S, d]
fn build_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_rope");

    // Inputs
    let input = b.add_input("x", &[SEQ_LEN, HEAD_DIM]);
    let cos_table = b.add_input("cos", &[SEQ_LEN, HALF_DIM]);
    let sin_table = b.add_input("sin", &[SEQ_LEN, HALF_DIM]);

    let half_shape = [SEQ_LEN, HALF_DIM];

    // Split input into first half and second half along dim 1
    let x_first = b.add_narrow(input, 1, 0, HALF_DIM, &half_shape);
    let x_second = b.add_narrow(input, 1, HALF_DIM, HALF_DIM, &half_shape);

    // rot_first = x_first * cos - x_second * sin
    let fc = b.add_binary_mul(x_first, cos_table, &half_shape);
    let ss = b.add_binary_mul(x_second, sin_table, &half_shape);
    // Negate ss: multiply by -1 constant, then add
    let neg_one = b.add_input("neg_one", &[1]);
    let neg_one_bc = b.add_broadcast(neg_one, &half_shape);
    let neg_ss = b.add_binary_mul(ss, neg_one_bc, &half_shape);
    let rot_first = b.add_binary_add(fc, neg_ss, &half_shape);

    // rot_second = x_first * sin + x_second * cos
    let fs = b.add_binary_mul(x_first, sin_table, &half_shape);
    let sc = b.add_binary_mul(x_second, cos_table, &half_shape);
    let rot_second = b.add_binary_add(fs, sc, &half_shape);

    // Concatenate back to full dimension
    let output = b.add_concat(&[rot_first, rot_second], 1, &[SEQ_LEN, HEAD_DIM]);

    b.build(output).expect("valid RoPE kernel")
}

/// Build parameter bindings for the RoPE kernel.
///
/// x = Variable, cos/sin = ConstantTensor, neg_one = ConstantScalar(-1).
fn rope_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x [SEQ_LEN, HEAD_DIM]
        TensorParamBinding::ConstantTensor(rope_cos_table()), // cos [SEQ_LEN, HALF_DIM]
        TensorParamBinding::ConstantTensor(rope_sin_table()), // sin [SEQ_LEN, HALF_DIM]
        TensorParamBinding::ConstantScalar(-1.0), // neg_one
    ]
}

/// Input bounds for RoPE: post-projection activations in [-2, 2].
fn rope_input_bounds() -> BoundedTensor {
    uniform_bounds(&[SEQ_LEN, HEAD_DIM], 2.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// RoPE TensorKernelDef validates.
#[test]
fn test_qwen3_rope_def_validates() {
    let def = build_rope_kernel();
    def.validate().expect("RoPE kernel should validate");
}

/// RoPE translates to NY GraphNetwork.
#[test]
fn test_qwen3_rope_graph_builds() {
    let def = build_rope_kernel();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("RoPE graph should translate");

    // Narrow(2) + BinaryMul(4) + Broadcast(1) + BinaryAdd(2) + Concat(1) = ~10+ nodes
    assert!(
        graph.num_nodes() >= 8,
        "RoPE graph should have >= 8 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through RoPE.
///
/// RoPE is a rotation: for inputs in [-2, 2], the rotation preserves magnitude.
/// Each output element is `x_i * cos(θ) ± x_j * sin(θ)` where |cos|, |sin| <= 1.
/// IBP upper bound: |x_i| * |cos| + |x_j| * |sin| <= 2 * 1 + 2 * 1 = 4.
#[test]
fn test_qwen3_rope_ibp_propagates() {
    let def = build_rope_kernel();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = rope_input_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP through RoPE");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RoPE IBP: bounds=[{lo_min}, {hi_max}]");

    // With [-2, 2] input and |cos|, |sin| <= 1:
    // Worst case per element: ±(2*1 + 2*1) = ±4.
    assert!(
        lo_min >= -5.0,
        "IBP lower should be >= -5 for [-2,2] input with rotation, got {lo_min}"
    );
    assert!(
        hi_max <= 5.0,
        "IBP upper should be <= 5 for [-2,2] input with rotation, got {hi_max}"
    );
}

/// CROWN bounds propagate through RoPE.
///
/// RoPE is linear (rotation matrix applied to input), so CROWN should
/// produce tight bounds (no nonlinear relaxation needed).
#[test]
fn test_qwen3_rope_crown_propagation() {
    let def = build_rope_kernel();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = rope_input_bounds();

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RoPE: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// RoPE verify and record under "qwen3_rope" key.
#[test]
fn test_qwen3_rope_verify_and_record() {
    let def = build_rope_kernel();
    let bindings = rope_bindings();
    let input = rope_input_bounds();

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_rope");
    assert_eq!(result.num_variables, 1, "single Variable input (x)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HEAD_DIM]);
}
