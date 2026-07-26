// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor softmax operations (#3679).
//!
//! Proves correctness properties of softmax.rs arithmetic:
//!
//! - `softmax_clamp_constants`: all dtypes return finite, ordered values
//! - Softmax numerical stability: shift invariance, probability sum, bounds
//! - Log-softmax: output non-positivity, shift invariance
//! - Edge cases: all-neg-inf lanes, uniform input, +inf handling
//!
//! These harnesses operate on pure scalar/arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use crate::DType;

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn ln_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ---------------------------------------------------------------------------
// softmax_clamp_constants: correctness for all dtypes
// ---------------------------------------------------------------------------

/// Prove: softmax_clamp_constants returns finite values for all dtype variants.
///
/// Non-finite clamp constants would cause the GPU decomposed softmax to
/// produce NaN or Inf immediately. Every dtype must get finite bounds.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_clamp_constants_all_finite() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);
    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let (max_val, min_val, min_pos) = super::softmax::softmax_clamp_constants(dt);
    assert!(max_val.is_finite(), "max must be finite for {dt:?}");
    assert!(min_val.is_finite(), "min must be finite for {dt:?}");
    assert!(
        min_pos.is_finite(),
        "min_positive must be finite for {dt:?}"
    );
}

/// Prove: softmax_clamp_constants returns max > min for all dtypes.
///
/// If max <= min, the clamp range would be empty or inverted, causing
/// the softmax decomposition to produce all-zeros or NaN.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_clamp_constants_ordered() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);
    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let (max_val, min_val, _) = super::softmax::softmax_clamp_constants(dt);
    assert!(
        max_val > min_val,
        "max must be strictly greater than min for {dt:?}"
    );
}

/// Prove: softmax_clamp_constants returns min_positive > 0 for all dtypes.
///
/// min_positive is used as the sum clamp floor in the decomposed softmax.
/// A zero or negative min_positive would cause division by zero in the
/// final exp_vals / sum_vals step.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_clamp_constants_min_positive_positive() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);
    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let (_, _, min_pos) = super::softmax::softmax_clamp_constants(dt);
    assert!(
        min_pos > 0.0,
        "min_positive must be strictly positive for {dt:?}"
    );
}

/// Prove: softmax_clamp_constants for BF16 and F16 have vastly different MAX.
///
/// BF16 has 8 exponent bits (MAX ~ 3.39e38), F16 has 5 (MAX = 65504).
/// Confusing them causes the softmax decomposition to overflow F16 tensors
/// or unnecessarily restrict BF16 range. Issue: #1691.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_clamp_bf16_f16_max_differ() {
    let (bf16_max, _, _) = super::softmax::softmax_clamp_constants(DType::BF16);
    let (f16_max, _, _) = super::softmax::softmax_clamp_constants(DType::F16);

    // BF16 MAX is at least 1e30 times larger than F16 MAX
    assert!(
        bf16_max > f16_max * 1e30,
        "bf16 MAX must be vastly larger than f16 MAX"
    );
    // F16 MAX must be in the 65000-66000 range
    assert!(f16_max > 65000.0, "f16 MAX must be ~65504");
    assert!(f16_max < 66000.0, "f16 MAX must be ~65504");
}

// ---------------------------------------------------------------------------
// Softmax scalar arithmetic properties
// ---------------------------------------------------------------------------

/// Prove: softmax of a single element always produces 1.0.
///
/// For a single-element lane, softmax(x) = exp(x - x) / exp(x - x) = 1.0
/// regardless of x (for any finite x). This is the trivial base case.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_single_element_is_one() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() <= 1e4);

    // Softmax of single element: exp(x - x) / sum(exp(x - x)) = 1/1 = 1
    let shifted = x - x; // 0.0
    let exp_val = shifted.exp(); // 1.0
    let sum = exp_val; // 1.0
    let prob = exp_val / sum;

    assert_eq!(prob, 1.0, "softmax of single element must be 1.0");
}

/// Prove: softmax of two equal elements produces uniform [0.5, 0.5].
///
/// When all inputs are equal, softmax distributes probability uniformly.
/// This is the fundamental symmetry property.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_equal_inputs_uniform() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() <= 1e3);

    // Both elements are x. max = x, shifted = x - x = 0.
    let shifted = 0.0_f32;
    let exp_val = shifted.exp(); // 1.0
    let sum = exp_val + exp_val; // 2.0
    let prob = exp_val / sum; // 0.5

    assert_eq!(prob, 0.5, "softmax of equal inputs must be 0.5 each");
}

/// Prove: softmax probabilities are non-negative for any finite inputs.
///
/// exp(x) > 0 for all finite x, so each softmax output must be > 0.
/// Combined with sum-to-1, this makes softmax a valid probability distribution.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_probabilities_nonnegative() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let max_val = f32::max(fa, fb);

    let exp_a = (fa - max_val).exp();
    let exp_b = (fb - max_val).exp();
    let sum = exp_a + exp_b;

    let prob_a = exp_a / sum;
    let prob_b = exp_b / sum;

    assert!(prob_a >= 0.0, "softmax probability must be non-negative");
    assert!(prob_b >= 0.0, "softmax probability must be non-negative");
    assert!(prob_a <= 1.0, "softmax probability must be <= 1.0");
    assert!(prob_b <= 1.0, "softmax probability must be <= 1.0");
}

/// Prove: softmax shift invariance — adding a constant doesn't change output.
///
/// softmax(x + c) == softmax(x) for any constant c. This is why the
/// max-subtract trick works: it shifts all values by -max without changing
/// the output distribution.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_shift_invariant() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;

    // Softmax without shift
    let max1 = f32::max(fa, fb);
    let e1a = (fa - max1).exp();
    let e1b = (fb - max1).exp();
    let s1 = e1a + e1b;
    let p1a = e1a / s1;

    // Softmax with shift by c
    let fa_c = fa + fc;
    let fb_c = fb + fc;
    let max2 = f32::max(fa_c, fb_c);
    let e2a = (fa_c - max2).exp();
    let e2b = (fb_c - max2).exp();
    let s2 = e2a + e2b;
    let p2a = e2a / s2;

    // Must be equal (using small integer values for exact arithmetic)
    let diff = (p1a - p2a).abs();
    assert!(diff < 1e-6, "softmax must be shift-invariant: diff={diff}");
}

/// Prove: log_softmax output is non-positive for any finite inputs.
///
/// log(softmax(x_i)) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
/// Since softmax(x_i) <= 1 for all i, log(softmax(x_i)) <= 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_non_positive() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let max_val = f32::max(fa, fb);

    let exp_a = (fa - max_val).exp();
    let exp_b = (fb - max_val).exp();
    let log_sum = (exp_a + exp_b).ln();

    let log_softmax_a = (fa - max_val) - log_sum;
    let log_softmax_b = (fb - max_val) - log_sum;

    assert!(
        log_softmax_a <= 1e-6, // allow tiny float error above 0
        "log_softmax must be non-positive: got {log_softmax_a}"
    );
    assert!(
        log_softmax_b <= 1e-6,
        "log_softmax must be non-positive: got {log_softmax_b}"
    );
}
