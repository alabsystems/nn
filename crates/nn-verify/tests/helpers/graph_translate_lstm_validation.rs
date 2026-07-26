// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numeric correctness tests for decomposed LSTM NY translation.
//!
//! Tests verify exact output values against analytically-computed LSTM gate
//! equations with known weights. Detects: wrong gate order, wrong activations,
//! incorrect c_new/h_new formulas.
//!
//! Extracted from `graph_translate_lstm.rs` (#923).
//! Part of #789 AC3.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helpers (shared with graph_translate_lstm.rs builder convention)
// ---------------------------------------------------------------------------

/// Build an LSTM cell kernel using the builder API.
fn build_lstm_kernel(
    name: &str,
    input_size: usize,
    hidden_size: usize,
    with_bias: bool,
) -> TensorKernelDef {
    let four_h = 4 * hidden_size;
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input", &[input_size]);
    let hidden = b.add_input("hidden", &[hidden_size]);
    let cell = b.add_input("cell", &[hidden_size]);
    let w_ih = b.add_input("weight_ih", &[four_h, input_size]);
    let w_hh = b.add_input("weight_hh", &[four_h, hidden_size]);
    let bias = if with_bias {
        let b_id = b.add_input("bias", &[four_h]);
        Some(b_id)
    } else {
        None
    };
    let out = b.add_lstm(input, hidden, cell, w_ih, w_hh, bias, &[hidden_size]);
    b.build(out).expect("LSTM kernel build should succeed")
}

/// Build a constant-weight array from a flat slice and shape.
fn constant_from_slice(shape: &[usize], values: &[f32]) -> TensorParamBinding {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec())
        .expect("shape must match values length");
    TensorParamBinding::ConstantTensor(arr)
}

/// Build LSTM bindings with specific W_ih weights, zero W_hh, zero state, uniform bias.
///
/// Returns (kernel, bindings) for a 2x2 LSTM cell (input_size=2, hidden_size=2).
/// W_ih layout: rows [0-1] i-gate, [2-3] f-gate, [4-5] g-gate, [6-7] o-gate.
fn build_known_weight_lstm() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let (input_size, hidden_size) = (2, 2);
    let four_h = 4 * hidden_size;
    let kernel = build_lstm_kernel("lstm_numeric", input_size, hidden_size, true);

    #[rustfmt::skip]
    let w_ih: Vec<f32> = vec![
        0.1, 0.2,  0.3, 0.1,     // i-gate rows
        0.2, 0.4,  0.1, 0.3,     // f-gate rows
        0.5, -0.1, -0.2, 0.3,    // g-gate rows
        0.3, 0.2,  0.1, 0.4,     // o-gate rows
    ];
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::zeros(IxDyn(&[hidden_size]))),
        TensorParamBinding::ConstantTensor(ArrayD::zeros(IxDyn(&[hidden_size]))),
        constant_from_slice(&[four_h, input_size], &w_ih),
        constant_from_slice(&[four_h, hidden_size], &vec![0.0f32; four_h * hidden_size]),
        constant_from_slice(&[four_h], &vec![0.01f32; four_h]),
    ];
    (kernel, bindings)
}

/// Assert IBP output elements match expected values within tolerance.
fn assert_ibp_matches(lo: &ArrayD<f32>, hi: &ArrayD<f32>, expected: &[f32], tol: f32) {
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(lo_flat.len(), expected.len(), "output size mismatch");

    for (idx, (&exp, (&l, &u))) in expected
        .iter()
        .zip(lo_flat.iter().zip(hi_flat.iter()))
        .enumerate()
    {
        let width = u - l;
        assert!(
            width.abs() < 0.1,
            "interval too wide at [{idx}]: width={width:.6}"
        );
        let mid = f32::midpoint(l, u);
        assert!(
            (mid - exp).abs() < tol,
            "LSTM output[{idx}]: expected {exp:.6}, got {mid:.6} (lo={l:.6}, hi={u:.6})"
        );
    }
}

// ---------------------------------------------------------------------------
// Numeric correctness tests
// ---------------------------------------------------------------------------

/// Numeric correctness: decomposed LSTM produces correct output for known weights.
///
/// Point bounds (lower == upper) make IBP compute exact values.
/// Catches: wrong gate order, wrong activations, incorrect c_new/h_new formulas.
///
/// Analytical reference (x=[0.5, -0.3], h=0, c=0, W_hh=0, bias=0.01):
///   gates = x @ W_ih^T + bias → split [i,f,g,o] → activations → c_new, h_new
///   h_new ≈ [0.0736, -0.0459]
#[test]
fn test_lstm_numeric_correctness_known_weights() {
    let (kernel, bindings) = build_known_weight_lstm();
    let graph = tensor_kernel_to_graph(&kernel, &bindings)
        .expect("LSTM with known weights should build graph");

    // Point input: x = [0.5, -0.3].
    let x = vec![0.5f32, -0.3];
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2]), x.clone()).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2]), x).unwrap();
    let input = BoundedTensor::new(lower, upper).expect("point bounds");

    let output = graph.propagate_ibp(&input).expect("IBP on point bounds");
    let (lo, hi) = output.lower_upper();

    // Expected h_new (analytically computed, see build_known_weight_lstm doc).
    assert_ibp_matches(lo, hi, &[0.0736, -0.0459], 0.01);
}

/// Numeric correctness with large distinct input — detects gate-order swaps.
///
/// Uses x=[3.0, -2.0] which produces distinct pre-activations for each gate,
/// making gate-order swaps (e.g., swapping i-gate and f-gate) produce
/// detectably wrong outputs that exceed the 0.01 tolerance.
///
/// Analytical reference (x=[3.0, -2.0], h=0, c=0, W_hh=0, bias=0.01):
///   i_pre = [-0.09, 0.71], f_pre = [-0.19, -0.29]
///   g_pre = [1.71, -1.19],  o_pre = [0.51, -0.49]
///   c_new = sigmoid(i_pre) * tanh(g_pre) = [0.4474, -0.5571]
///   h_new = sigmoid(o_pre) * tanh(c_new) ≈ [0.2630, -0.1922]
///
/// An i/f swap would produce h_new ≈ [0.2502, -0.1297], differing by
/// >0.01 at both elements.
#[test]
fn test_lstm_numeric_large_input_detects_gate_swap() {
    let (kernel, bindings) = build_known_weight_lstm();
    let graph = tensor_kernel_to_graph(&kernel, &bindings)
        .expect("LSTM with known weights should build graph");

    // Large distinct input: x = [3.0, -2.0].
    let x = vec![3.0f32, -2.0];
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2]), x.clone()).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2]), x).unwrap();
    let input = BoundedTensor::new(lower, upper).expect("point bounds");

    let output = graph.propagate_ibp(&input).expect("IBP on point bounds");
    let (lo, hi) = output.lower_upper();

    // Expected h_new (analytically computed).
    // If i/f gates are swapped, h_new ≈ [0.2502, -0.1297] — both elements
    // differ by >0.01 from correct values, so the swap is detected.
    assert_ibp_matches(lo, hi, &[0.2630, -0.1922], 0.01);
}
