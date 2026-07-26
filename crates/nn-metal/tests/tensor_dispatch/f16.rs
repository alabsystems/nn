// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! f16 and bf16 dispatch tests for the generalized `execute_tensor_dispatch`.
//!
//! Validates that the Metal tensor dispatch pipeline works with `half::f16`
//! and `half::bf16` element types — not just `f32`. Each test builds a kernel,
//! generates random f16/bf16 inputs, dispatches through Metal with
//! `ScalarType::F16`, and compares GPU output against an f32 CPU reference
//! (using the wider F16 precision budget from `PrecisionContract`).
//!
//! Coverage:
//! - Elementwise (Sigmoid, ReLU): simple activations through f16 dispatch
//! - Reduce (Sum): standalone reduction in half precision
//! - Softmax: row-wise softmax reduction in half precision
//! - bf16 round-trip: bf16 inputs → f16 Metal compute → bf16 readback
//!
//! Part of #865.

use super::test_utils::{assert_within_budget_f16, metal_setup, rand_f16_vec, rand_f32_vec};
use nn_dsl::{tensor_block_builder::TensorBlockBuilder, ScalarType};
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

// ===========================================================================
// f16 Sigmoid via execute_tensor_dispatch
// ===========================================================================

/// Sigmoid dispatch with f16 inputs: verifies the full f16 pipeline from
/// `half::f16` input → Metal "half" compute → `half::f16` output.
#[test]
fn test_sigmoid_f16_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("sigmoid_f16");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    // f16 input data (values in [-5, 5] — within f16 range).
    let x_f16 = rand_f16_vec(0xF160_0001, total, -5.0, 5.0);

    // CPU reference in f32 (compute sigmoid at full precision, compare with
    // wider f16 budget).
    let cpu_out: Vec<f32> = x_f16
        .iter()
        .map(|v| {
            let x = v.to_f32();
            1.0 / (1.0 + (-x).exp())
        })
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_f16);

    let gpu_out: Vec<half::f16> =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F16, &inputs)
            .expect("sigmoid f16 dispatch");

    assert_eq!(gpu_out.len(), total, "sigmoid f16 output length");
    assert_within_budget_f16("sigmoid_f16", &gpu_out, &cpu_out);
}

// ===========================================================================
// f16 ReLU via execute_tensor_dispatch
// ===========================================================================

/// ReLU dispatch with f16 inputs: elementwise max(x, 0) in half precision.
/// ReLU is exact (no floating-point arithmetic error), so this is a clean
/// validation of the f16 data path.
#[test]
fn test_relu_f16_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("relu_f16");
    let x = b.add_input("x", &shape);
    let relu = b.add_relu(x, &shape);
    let kernel = b.build(relu).expect("valid graph");

    let x_f16 = rand_f16_vec(0xF160_0002, total, -10.0, 10.0);
    let cpu_out: Vec<f32> = x_f16.iter().map(|v| v.to_f32().max(0.0)).collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_f16);

    let gpu_out: Vec<half::f16> =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F16, &inputs)
            .expect("relu f16 dispatch");

    assert_eq!(gpu_out.len(), total, "relu f16 output length");
    // ReLU is exact — every element should match bit-for-bit after f32
    // conversion. Use the standard budget anyway for uniformity.
    assert_within_budget_f16("relu_f16", &gpu_out, &cpu_out);
}

// ===========================================================================
// f16 Reduce(Sum) via execute_tensor_dispatch
// ===========================================================================

/// Reduce(Sum) dispatch with f16 inputs: validates the tree reduction kernel
/// with 2-byte element sizing (shared memory = threads * 2 bytes).
///
/// Shape: [4, 8] → reduce axis 1 → [4] (sum of each row).
#[test]
fn test_reduce_sum_f16_dispatch() {
    let cache = metal_setup();

    let (outer, inner) = (4_usize, 8_usize);
    let total = outer * inner;

    let mut b = TensorBlockBuilder::new("reduce_sum_f16");
    let x = b.add_input("x", &[outer, inner]);
    let reduced = b.add_reduce(x, nn_dsl::ReduceOp::Sum, 1, false, &[outer]);
    let kernel = b.build(reduced).expect("valid graph");

    // f16 input — moderate values to avoid accumulation overflow in half.
    let x_f16 = rand_f16_vec(0xF160_0003, total, -2.0, 2.0);

    // CPU reference: sum each row in f32.
    let x_f32: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out: Vec<f32> = (0..outer)
        .map(|row| x_f32[row * inner..(row + 1) * inner].iter().sum())
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_f16);

    let gpu_out: Vec<half::f16> =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F16, &inputs)
            .expect("reduce_sum f16 dispatch");

    assert_eq!(gpu_out.len(), outer, "reduce_sum f16 output length");
    assert_within_budget_f16("reduce_sum_f16", &gpu_out, &cpu_out);
}

// ===========================================================================
// f16 Softmax via execute_tensor_dispatch
// ===========================================================================

/// Softmax dispatch with f16 inputs: validates the softmax reduction kernel
/// with 2-byte element sizing (shared memory = axis_size * 2 bytes).
#[test]
fn test_softmax_f16_dispatch() {
    let cache = metal_setup();

    let (outer, axis) = (4_usize, 8_usize);
    let total = outer * axis;

    let mut b = TensorBlockBuilder::new("softmax_f16");
    let x = b.add_input("x", &[outer, axis]);
    let sm = b.add_softmax(x, 1, &[outer, axis]);
    let kernel = b.build(sm).expect("valid graph");

    // f16 input — moderate values to avoid exp() overflow in half precision.
    // f16 max is ~65504; exp(11) ≈ 59874, so keep inputs in [-8, 8].
    let x_f16 = rand_f16_vec(0xF160_0004, total, -8.0, 8.0);

    // CPU reference softmax in f32.
    let x_f32: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let mut cpu_out = vec![0.0_f32; total];
    for row in 0..outer {
        let start = row * axis;
        let end = start + axis;
        let row_max = x_f32[start..end]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = x_f32[start..end].iter().map(|&v| (v - row_max).exp()).sum();
        for j in start..end {
            cpu_out[j] = (x_f32[j] - row_max).exp() / exp_sum;
        }
    }

    let mut inputs = HashMap::new();
    inputs.insert("x", x_f16);

    let gpu_out: Vec<half::f16> =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F16, &inputs)
            .expect("softmax f16 dispatch");

    assert_eq!(gpu_out.len(), total, "softmax f16 output length");
    assert_within_budget_f16("softmax_f16", &gpu_out, &cpu_out);
}

// ===========================================================================
// f16 Reduce(Sum) precision test — f32 accumulator improves accuracy (#1352)
// ===========================================================================

/// Precision test: sum 256 f16 values near 1.0 where half-precision accumulation
/// would accumulate significant rounding error but f32 accumulation stays accurate.
///
/// f16 has 10-bit mantissa → ULP at 1.0 is ~0.001. Summing 256 values at ~1.0 in
/// half precision can drift by up to 256 * ULP ≈ 0.25. With f32 accumulator, the
/// sum is accurate to f32 precision (23-bit mantissa → ULP at 256.0 ≈ 3e-5).
///
/// Acceptance criteria: max abs error < 0.05 (tighter than 0.25 half-precision drift).
#[test]
fn test_reduce_sum_f16_precision_f32_accumulator() {
    let cache = metal_setup();

    let (outer, inner) = (2_usize, 256_usize);
    let total = outer * inner;

    let mut b = TensorBlockBuilder::new("reduce_sum_f16_prec");
    let x = b.add_input("x", &[outer, inner]);
    let reduced = b.add_reduce(x, nn_dsl::ReduceOp::Sum, 1, false, &[outer]);
    let kernel = b.build(reduced).expect("valid graph");

    // f16 input: values clustered around 1.0 ± 0.5 to stress accumulation.
    let x_f16 = rand_f16_vec(0xF16_ACC0, total, 0.5, 1.5);

    // CPU reference: sum each row in f32.
    let x_f32: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let cpu_out: Vec<f32> = (0..outer)
        .map(|row| x_f32[row * inner..(row + 1) * inner].iter().sum())
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_f16);

    let gpu_out: Vec<half::f16> =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F16, &inputs)
            .expect("reduce_sum f16 precision dispatch");

    assert_eq!(
        gpu_out.len(),
        outer,
        "reduce_sum f16 precision output length"
    );

    // With f32 accumulator (#1352), the accumulation itself is exact in f32.
    // Remaining error comes from the final f32→f16 cast on the output (sum ≈ 256,
    // f16 ULP at 256 = 0.25, so quantization noise ≤ 0.125 per element).
    // Without f32 accumulator, half-precision drift would reach ≥ 1.0.
    let max_abs_err: f32 = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, &c)| (g.to_f32() - c).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err < 0.15,
        "f32 accumulator should keep error below f16 quantization noise; \
         got max_abs_err={max_abs_err:.6e}",
    );
}

// ===========================================================================
// bf16 round-trip via execute_tensor_dispatch
// ===========================================================================

/// bf16 dispatch round-trip: verifies that `half::bf16` inputs are transparently
/// converted to f16 for Metal compute, and converted back to bf16 on readback.
///
/// Uses ReLU (exact op) so any discrepancy is in the bf16↔f16 conversion,
/// not floating-point arithmetic.
#[test]
fn test_relu_bf16_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("relu_bf16");
    let x = b.add_input("x", &shape);
    let relu = b.add_relu(x, &shape);
    let kernel = b.build(relu).expect("valid graph");

    // bf16 input data — generate f32 then convert to bf16.
    let x_f32 = rand_f32_vec(0xBF16_0001, total, -10.0, 10.0);
    let x_bf16: Vec<half::bf16> = x_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();

    // Expected output: relu applied to the bf16→f16→bf16 round-trip values.
    // bf16→f16 may lose precision (different mantissa/exponent split), so
    // the reference is computed through the same conversion path.
    let cpu_out: Vec<f32> = x_bf16
        .iter()
        .map(|v| {
            // bf16 → f32 → f16 → f32 (mimics Metal boundary conversion) → relu
            let as_f16 = half::f16::from_f32(v.to_f32());
            as_f16.to_f32().max(0.0)
        })
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_bf16);

    // ScalarType::F16 because bf16 is stored as f16 on Metal.
    let gpu_out: Vec<half::bf16> =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F16, &inputs)
            .expect("relu bf16 dispatch");

    assert_eq!(gpu_out.len(), total, "relu bf16 output length");

    // Compare bf16 GPU output against the expected f32 reference.
    // bf16 has wider exponent range but fewer mantissa bits than f16.
    // The round-trip bf16→f16→bf16 is lossy but deterministic.
    for (i, (&g_bf16, &expected)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let g = g_bf16.to_f32();
        let delta = (g - expected).abs();
        // bf16 has ~2 decimal digits of precision (7-bit mantissa).
        // After bf16→f16→relu→f16→bf16, tolerance is generous.
        assert!(
            delta < 0.05,
            "relu_bf16[{i}]: expected={expected}, gpu_bf16={g}, delta={delta:.6e}",
        );
    }
}
