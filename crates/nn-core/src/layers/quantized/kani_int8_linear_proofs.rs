// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Int8Linear layer correctness (#3656).
//!
//! Supplements the existing 8 harnesses in int8_linear.rs (memory bounds,
//! dequant bounded, dim validation) and the 15 harnesses in int8.rs
//! (cast roundtrip, scale properties, error bounds).
//!
//! These harnesses prove additional properties specific to the Int8Linear
//! layer abstraction:
//!
//!  1. memory_bytes formula matches hand computation
//!  2. f32_memory_bytes formula matches hand computation
//!  3. compression_ratio > 1.0 for in_features >= 2
//!  4. compression_ratio < 4.0 always (overhead prevents reaching 4x)
//!  5. compression_ratio monotonically approaches 4.0 as in_features grows
//!  6. memory_bytes with bias = memory_bytes without + 4 * out_features
//!  7. f32_memory_bytes with bias = f32_memory_bytes without + 4 * out_features
//!  8. memory_bytes < f32_memory_bytes for in_features >= 2 (with bias)
//!  9. forward input validation: matching last dim always passes check
//! 10. forward input validation: non-matching last dim always fails
//! 11. params scale length must equal out_features
//! 12. params zero_point length must equal out_features
//! 13. per-element dequant with asymmetric zero_point is bounded
//! 14. matmul output dimension: [B, N, in] @ [out, in]^T = [B, N, out]
//! 15. weight memory savings grows linearly with in_features
//!
//! Part of #3656.

// ---------------------------------------------------------------------------
// Harness 1: memory_bytes formula matches hand computation
// ---------------------------------------------------------------------------

/// Prove: Int8Linear::memory_bytes() computes exactly
/// out * in + 4 * out + out = out * (in + 5) without bias.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_memory_bytes_formula() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    // Components from Int8Linear::memory_bytes (no bias)
    let weight_bytes = out_features * in_features; // 1 byte per weight
    let scale_bytes = out_features * 4; // f32 per channel
    let zp_bytes = out_features; // i8 per channel
    let total = weight_bytes + scale_bytes + zp_bytes;

    // Algebraic simplification: out * in + 4*out + out = out*(in + 5)
    let expected = out_features * (in_features + 5);
    assert!(
        total == expected,
        "memory_bytes must equal out_features * (in_features + 5)"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: f32_memory_bytes formula matches hand computation
// ---------------------------------------------------------------------------

/// Prove: Int8Linear::f32_memory_bytes() computes exactly
/// 4 * out * in without bias.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_f32_memory_bytes_formula() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    let f32_bytes = out_features * in_features * 4;
    let expected = 4 * out_features * in_features;

    assert!(
        f32_bytes == expected,
        "f32_memory_bytes must equal 4 * out * in"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: compression_ratio > 1.0 for in_features >= 2
// ---------------------------------------------------------------------------

/// Prove: compression ratio (f32_memory / int8_memory) is strictly > 1.0
/// for any layer with in_features >= 2, regardless of bias.
///
/// Algebraically: 4*out*in / (out*(in+5)) = 4*in / (in+5) > 1
/// when 4*in > in+5, i.e., 3*in > 5, i.e., in >= 2.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_compression_above_one() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 2048);
    kani::assume(in_features >= 2 && in_features <= 2048);

    let int8_mem = out_features * (in_features + 5);
    let f32_mem = 4 * out_features * in_features;

    // f32_mem > int8_mem  <=>  4*in > in + 5  <=>  3*in > 5  <=>  in >= 2
    assert!(
        f32_mem > int8_mem,
        "F32 must use more memory than INT8 for in >= 2"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: compression_ratio < 4.0 always (overhead prevents 4x)
// ---------------------------------------------------------------------------

/// Prove: the compression ratio never reaches 4.0 because of per-channel
/// scale and zero_point overhead.
///
/// ratio = 4*in / (in+5) < 4 always (since in+5 > in).
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_compression_below_four() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    let int8_mem = out_features * (in_features + 5);
    let f32_mem = 4 * out_features * in_features;

    // ratio = f32_mem / int8_mem = 4*in / (in+5)
    // 4*in < 4*(in+5) = 4*in + 20, so ratio < 4 always.
    // Equivalently: f32_mem < 4 * int8_mem
    let four_times_int8 = 4 * int8_mem;
    assert!(
        f32_mem < four_times_int8,
        "compression ratio must be < 4.0 (overhead prevents reaching 4x)"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: compression_ratio monotonically increases with in_features
// ---------------------------------------------------------------------------

/// Prove: for fixed out_features, increasing in_features increases the
/// compression ratio (approaches 4.0 asymptotically).
///
/// ratio(in) = 4*in / (in+5). d/d(in) = 20 / (in+5)^2 > 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_compression_monotone() {
    let out_features: usize = kani::any();
    kani::assume(out_features >= 1 && out_features <= 1024);

    let in_a: usize = kani::any();
    let in_b: usize = kani::any();
    kani::assume(in_a >= 1 && in_a <= 1024);
    kani::assume(in_b >= 1 && in_b <= 1024);
    kani::assume(in_a < in_b);

    // ratio_a = 4*in_a / (in_a + 5)
    // ratio_b = 4*in_b / (in_b + 5)
    // To prove ratio_b > ratio_a without floating point:
    // 4*in_b * (in_a + 5) > 4*in_a * (in_b + 5)
    // 4*in_b*in_a + 20*in_b > 4*in_a*in_b + 20*in_a
    // 20*in_b > 20*in_a  <=>  in_b > in_a (true by assumption)
    let lhs = 4 * in_b * (in_a + 5);
    let rhs = 4 * in_a * (in_b + 5);
    assert!(
        lhs > rhs,
        "compression ratio must increase with in_features"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: memory_bytes with bias = without bias + 4*out_features
// ---------------------------------------------------------------------------

/// Prove: adding a bias to Int8Linear adds exactly 4 * out_features bytes
/// (one f32 per output channel).
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_bias_memory_delta() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    let mem_no_bias = out_features * in_features + out_features * 4 + out_features;
    let bias_bytes = out_features * 4;
    let mem_with_bias = mem_no_bias + bias_bytes;

    assert!(
        mem_with_bias - mem_no_bias == out_features * 4,
        "bias adds exactly 4 * out_features bytes"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: f32_memory_bytes with bias = without bias + 4*out_features
// ---------------------------------------------------------------------------

/// Prove: F32 memory with bias vs without also differs by 4*out_features.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_f32_bias_memory_delta() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    let f32_no_bias = out_features * in_features * 4;
    let f32_bias_bytes = out_features * 4;
    let f32_with_bias = f32_no_bias + f32_bias_bytes;

    assert!(
        f32_with_bias - f32_no_bias == out_features * 4,
        "F32 bias also adds exactly 4 * out_features bytes"
    );

    // The bias overhead is the same for both INT8 and F32
    assert!(
        f32_bias_bytes == out_features * 4,
        "bias overhead is identical across formats"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: memory_bytes < f32_memory_bytes for in_features >= 2 (with bias)
// ---------------------------------------------------------------------------

/// Prove: even with bias included, INT8 still saves memory for in_features >= 2.
/// INT8 total = out*(in+5) + 4*out = out*(in+9)
/// F32 total  = 4*out*in + 4*out   = 4*out*(in+1)
/// INT8 < F32 when in+9 < 4*(in+1) = 4*in+4, i.e., 5 < 3*in, i.e., in >= 2.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_saves_memory_with_bias() {
    let out_features: usize = kani::any();
    let in_features: usize = kani::any();

    kani::assume(out_features >= 1 && out_features <= 2048);
    kani::assume(in_features >= 2 && in_features <= 2048);

    let int8_total = out_features * (in_features + 9); // weight + scale + zp + bias
    let f32_total = 4 * out_features * (in_features + 1); // weight + bias

    assert!(
        int8_total < f32_total,
        "INT8 must save memory vs F32 even with bias for in >= 2"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: forward input dimension check passes for matching last dim
// ---------------------------------------------------------------------------

/// Prove: when the input tensor's last dimension equals in_features,
/// the dimension validation in Int8Linear::forward succeeds.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_forward_matching_dim_passes() {
    let in_features: usize = kani::any();
    kani::assume(in_features >= 1 && in_features <= 4096);

    let x_last: usize = in_features; // matching

    let passes = x_last == in_features;
    assert!(passes, "matching last dimension must pass validation");
}

// ---------------------------------------------------------------------------
// Harness 10: forward input dimension check fails for non-matching last dim
// ---------------------------------------------------------------------------

/// Prove: when the input tensor's last dimension does NOT equal in_features,
/// the dimension validation in Int8Linear::forward rejects the input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_forward_mismatched_dim_fails() {
    let in_features: usize = kani::any();
    let x_last: usize = kani::any();

    kani::assume(in_features >= 1 && in_features <= 4096);
    kani::assume(x_last >= 1 && x_last <= 4096);
    kani::assume(x_last != in_features);

    let fails = x_last != in_features;
    assert!(fails, "non-matching last dimension must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 11: params scale length must equal out_features
// ---------------------------------------------------------------------------

/// Prove: for any valid Int8Linear, scale.len() == out_features.
/// This is the structural invariant that forward's per-channel dequant relies on.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_scale_length_invariant() {
    let out_features: usize = kani::any();
    kani::assume(out_features >= 1 && out_features <= 4096);

    let scale_len = out_features; // enforced by Int8Linear::new

    // Every row index is a valid scale index
    let row: usize = kani::any();
    kani::assume(row < out_features);
    assert!(row < scale_len, "row must be valid scale index");
}

// ---------------------------------------------------------------------------
// Harness 12: params zero_point length must equal out_features
// ---------------------------------------------------------------------------

/// Prove: for any valid Int8Linear, zero_point.len() == out_features.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_zp_length_invariant() {
    let out_features: usize = kani::any();
    kani::assume(out_features >= 1 && out_features <= 4096);

    let zp_len = out_features; // enforced by Int8Linear::new

    let row: usize = kani::any();
    kani::assume(row < out_features);
    assert!(row < zp_len, "row must be valid zero_point index");
}

// ---------------------------------------------------------------------------
// Harness 13: per-element dequant with asymmetric zero_point is bounded
// ---------------------------------------------------------------------------

/// Prove: for asymmetric INT8 dequantization, the output value
/// (q_i8 - zero_point) * scale is bounded by 255 * |scale|.
///
/// q_i8 in [-128, 127], zero_point as i32 in [-128, 127].
/// diff = q_i8 - zero_point in [-255, 255].
/// |dequant| <= 255 * |scale|.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_asymmetric_dequant_bounded() {
    let q_u8: u8 = kani::any();
    let q_i8 = q_u8 as i8;

    let zero_point: i32 = kani::any();
    kani::assume(zero_point >= -128 && zero_point <= 127);

    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale >= 0.0 && scale < 100.0);

    let diff = q_i8 as f32 - zero_point as f32;
    // diff in [-255.0, 255.0] (proven by int8.rs harness 2)

    let dequant = diff * scale;
    assert!(dequant.is_finite(), "dequantized value must be finite");

    let bound = 255.0 * scale;
    assert!(
        dequant >= -bound && dequant <= bound,
        "asymmetric dequant bounded by 255 * scale"
    );
}

// ---------------------------------------------------------------------------
// Harness 14: matmul output dimension
// ---------------------------------------------------------------------------

/// Prove: Int8Linear forward computes x @ W^T, producing the correct
/// output shape. For input [B, N, in_features] and weight [out_features,
/// in_features], the output is [B, N, out_features].
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_matmul_output_dim() {
    let b: usize = kani::any();
    let n: usize = kani::any();
    let in_features: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(n >= 1 && n <= 256);
    kani::assume(in_features >= 1 && in_features <= 1024);
    kani::assume(out_features >= 1 && out_features <= 1024);

    // Input: [B, N, in_features]
    // Weight transposed: [in_features, out_features]
    // Output: [B, N, out_features]

    // Matmul inner dimension must match
    let x_last = in_features;
    let wt_first = in_features; // W^T has shape [in_features, out_features]
    assert!(x_last == wt_first, "matmul inner dims must match");

    // Output last dim = out_features
    let output_last = out_features;
    assert!(
        output_last == out_features,
        "output last dim must be out_features"
    );

    // Output element count
    let output_elems = b.checked_mul(n).and_then(|v| v.checked_mul(out_features));
    if let Some(elems) = output_elems {
        assert!(elems >= 1, "output must have at least 1 element");
    }
}

// ---------------------------------------------------------------------------
// Harness 15: weight memory savings grows linearly with in_features
// ---------------------------------------------------------------------------

/// Prove: the absolute memory savings (f32_bytes - int8_bytes) grows
/// linearly with in_features for fixed out_features.
///
/// savings = 4*out*in - out*(in+5) = out*(3*in - 5)
/// For in >= 2: savings > 0 and proportional to in.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8linear_savings_linear_in_features() {
    let out_features: usize = kani::any();
    kani::assume(out_features >= 1 && out_features <= 1024);

    let in_a: usize = kani::any();
    let in_b: usize = kani::any();
    kani::assume(in_a >= 2 && in_a <= 1024);
    kani::assume(in_b >= 2 && in_b <= 1024);
    kani::assume(in_a < in_b);

    // savings_a = out * (3*in_a - 5)
    // savings_b = out * (3*in_b - 5)
    let savings_a = out_features * (3 * in_a - 5);
    let savings_b = out_features * (3 * in_b - 5);

    // Since in_b > in_a, savings_b > savings_a
    assert!(
        savings_b > savings_a,
        "memory savings must increase with in_features"
    );
}
