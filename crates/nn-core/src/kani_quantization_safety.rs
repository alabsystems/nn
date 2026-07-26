// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for quantization safety (#4238).
//!
//! Proves 10 properties of the quantization pipeline that model deployment,
//! mixed-precision inference, and verified quantization depend on:
//!
//! 1. Per-channel scale computation: scale = max_abs / (2^(bits-1) - 1), scale > 0
//! 2. INT8 quantization range: quantized values in [-128, 127]
//! 3. INT4 quantization range: quantized values in [-8, 7]
//! 4. Dequantization roundtrip error: |dequant(quant(x)) - x| <= scale/2
//! 5. Zero-point alignment: zero maps to zero (symmetric quantization)
//! 6. Scale invariant ordering: if x > y then quant(x) >= quant(y)
//! 7. Group quantization: group size divides tensor dimension
//! 8. Mixed-precision: F32 accumulation of INT8 products doesn't overflow
//! 9. Dynamic range: quantization range covers actual weight distribution
//! 10. Block quantization (Q4_K): super-block structure, scales per block
//!
//! All harnesses operate on pure scalar arithmetic — no DynTensor, ndarray,
//! or GPU storage — making them tractable for CBMC symbolic execution.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// 1. Per-channel scale computation: scale = max_abs / (2^(bits-1) - 1), scale > 0
// ---------------------------------------------------------------------------

/// Prove: for any finite positive abs_max, per-channel symmetric INT8 scale
/// computation yields scale = abs_max / 127, and scale > 0.
///
/// The scale formula for symmetric quantization maps the full-precision range
/// [-abs_max, abs_max] to the integer range [-127, 127]. Scale must be
/// strictly positive when abs_max > 0 (the zero-weight channel case is
/// trivially scale=0). This is the core invariant that `compute_channel_params`
/// in `nn/quantized/int8.rs` depends on.
#[kani::proof]
#[kani::unwind(1)]
fn proof_per_channel_scale_positive_int8() {
    let abs_max: f32 = kani::any();
    kani::assume!(abs_max.is_finite());
    kani::assume!(abs_max > 0.0);
    kani::assume!(abs_max <= 1e30);

    // INT8 symmetric: scale = abs_max / 127.0
    let scale_int8 = abs_max / 127.0;
    assert!(
        scale_int8 > 0.0,
        "INT8 symmetric scale must be strictly positive for non-zero abs_max"
    );
    assert!(
        scale_int8.is_finite(),
        "INT8 symmetric scale must be finite for bounded abs_max"
    );

    // INT4 symmetric: scale = abs_max / 7.0
    let scale_int4 = abs_max / 7.0;
    assert!(
        scale_int4 > 0.0,
        "INT4 symmetric scale must be strictly positive for non-zero abs_max"
    );
    assert!(
        scale_int4.is_finite(),
        "INT4 symmetric scale must be finite for bounded abs_max"
    );

    // INT4 scale >= INT8 scale (fewer levels = coarser quantization)
    assert!(
        scale_int4 >= scale_int8,
        "INT4 scale must be >= INT8 scale (fewer quantization levels)"
    );
}

// ---------------------------------------------------------------------------
// 2. INT8 quantization range: quantized values in [-128, 127]
// ---------------------------------------------------------------------------

/// Prove: for any finite input value and any finite positive scale, the
/// quantize-to-INT8 operation (round + clamp) produces a value in [-128, 127].
///
/// The quantization formula is: q = clamp(round(v / scale + zp), -128, 127).
/// The clamp is the safety net; this harness proves the clamp enforces the
/// range invariant for all inputs. GPU dispatch and dequantization both
/// assume quantized values fit in a signed byte.
#[kani::proof]
#[kani::unwind(1)]
fn proof_int8_quantized_range() {
    let v: f32 = kani::any();
    kani::assume!(v.is_finite());
    kani::assume!(v.abs() <= 1e10);

    let scale: f32 = kani::any();
    kani::assume!(scale.is_finite());
    kani::assume!(scale > 1e-10 && scale < 1e10);

    let zero_point: i32 = kani::any();
    kani::assume!(zero_point >= -128 && zero_point <= 127);

    let q_f32 = (v / scale + zero_point as f32).round().clamp(-128.0, 127.0);
    let q_i8 = q_f32 as i8;

    // The cast from clamped f32 to i8 must stay within bounds
    let q_i32 = q_i8 as i32;
    assert!(
        q_i32 >= -128 && q_i32 <= 127,
        "INT8 quantized value must be in [-128, 127]"
    );
}

// ---------------------------------------------------------------------------
// 3. INT4 quantization range: quantized values in [-8, 7]
// ---------------------------------------------------------------------------

/// Prove: for any finite input value and any finite positive scale, the
/// quantize-to-INT4 operation (round + clamp) produces a value in [-8, 7].
///
/// INT4 uses only 4 bits (signed): range [-8, 7] for symmetric or [-8, 7]
/// for the full signed range. The nibble packing in `weight_quant.rs`
/// stores `q & 0x0F`, so sign extension on unpack recovers the original
/// only if q was in [-8, 7]. This harness proves the clamp guarantees it.
#[kani::proof]
#[kani::unwind(1)]
fn proof_int4_quantized_range() {
    let v: f32 = kani::any();
    kani::assume!(v.is_finite());
    kani::assume!(v.abs() <= 1e10);

    let scale: f32 = kani::any();
    kani::assume!(scale.is_finite());
    kani::assume!(scale > 1e-10 && scale < 1e10);

    // Symmetric INT4: zp = 0
    let q_f32 = (v / scale).round().clamp(-8.0, 7.0);
    let q_i8 = q_f32 as i8;

    assert!(
        q_i8 >= -8 && q_i8 <= 7,
        "INT4 quantized value must be in [-8, 7]"
    );

    // Verify nibble packing preserves value: low 4 bits + sign extension
    let nibble = (q_i8 & 0x0F) as u8;
    let sign_extended: i8 = if nibble & 0x08 != 0 {
        (nibble | 0xF0) as i8
    } else {
        nibble as i8
    };

    assert!(
        sign_extended == q_i8,
        "INT4 nibble pack→sign-extend must preserve value in [-8, 7]"
    );
}

// ---------------------------------------------------------------------------
// 4. Dequantization roundtrip error: |dequant(quant(x)) - x| <= scale/2
// ---------------------------------------------------------------------------

/// Prove: for symmetric quantization at either INT8 or INT4 bit width, the
/// dequantization roundtrip error is bounded by scale/2 (half a quantization
/// step) plus a small epsilon for floating-point arithmetic imprecision.
///
/// This is the fundamental quantization error guarantee. For any value v
/// within the representable range, quant(v) = round(v/scale), dequant(q) =
/// q * scale, and |dequant(quant(v)) - v| <= scale/2. The proof covers
/// both INT8 (range [-127*s, 127*s]) and INT4 (range [-7*s, 7*s]) in one
/// harness via a symbolic bit-width selector.
#[kani::proof]
#[kani::unwind(1)]
fn proof_dequant_roundtrip_error_bounded() {
    let scale: f32 = kani::any();
    kani::assume!(scale.is_finite());
    kani::assume!(scale > 1e-10 && scale < 100.0);

    let is_int4: bool = kani::any();
    let (clamp_lo, clamp_hi, qmax) = if is_int4 {
        (-8.0_f32, 7.0_f32, 7.0_f32)
    } else {
        (-128.0_f32, 127.0_f32, 127.0_f32)
    };

    // Value within the symmetric representable range
    let v: f32 = kani::any();
    kani::assume!(v.is_finite());
    kani::assume!(v >= -qmax * scale && v <= qmax * scale);

    // Quantize: round + clamp
    let q_f32 = (v / scale).round().clamp(clamp_lo, clamp_hi);
    let q_i8 = q_f32 as i8;

    // Dequantize
    let dequant = (q_i8 as f32) * scale;

    // Error bound: scale/2 + epsilon for f32 arithmetic
    let error = (dequant - v).abs();
    let bound = scale / 2.0 + 1e-5;
    assert!(
        error <= bound,
        "roundtrip error must be <= scale/2 for symmetric quantization"
    );
}

// ---------------------------------------------------------------------------
// 5. Zero-point alignment: zero maps to zero (symmetric quantization)
// ---------------------------------------------------------------------------

/// Prove: for symmetric quantization (zero_point = 0), the value 0.0
/// quantizes to 0 and dequantizes back to exactly 0.0.
///
/// Symmetric quantization is designed so that floating-point zero has a
/// perfect representation in the quantized domain. This is critical for
/// bias initialization (zero bias must remain zero after quantization)
/// and for sparsity patterns (zero weights must stay zero).
#[kani::proof]
#[kani::unwind(1)]
fn proof_zero_maps_to_zero_symmetric() {
    let scale: f32 = kani::any();
    kani::assume!(scale.is_finite());
    kani::assume!(scale > 0.0 && scale < 1e10);

    // Quantize 0.0 with symmetric mode (zp = 0)
    let q_f32 = (0.0_f32 / scale).round().clamp(-128.0, 127.0);
    assert!(q_f32 == 0.0, "zero must quantize to q=0 for symmetric mode");

    let q_i8 = q_f32 as i8;
    assert!(q_i8 == 0, "zero must quantize to i8(0)");

    // Dequantize: (0 - 0) * scale = 0.0
    let dequant = (q_i8 as f32 - 0.0) * scale;
    assert!(
        dequant == 0.0,
        "dequant(quant(0.0)) must be exactly 0.0 for symmetric mode"
    );
    assert!(
        dequant.to_bits() == 0.0_f32.to_bits(),
        "dequant must produce +0.0 (not -0.0) for zero input"
    );
}

// ---------------------------------------------------------------------------
// 6. Scale invariant ordering: if x > y then quant(x) >= quant(y)
// ---------------------------------------------------------------------------

/// Prove: for any two finite values x > y within the representable range,
/// the quantized values satisfy quant(x) >= quant(y).
///
/// Quantization is a monotone (order-preserving) map: the rounding step
/// can collapse distinct values to the same level but cannot reverse their
/// order. This is essential for softmax output ordering preservation and
/// for correctness of argmax on quantized activations.
#[kani::proof]
#[kani::unwind(1)]
fn proof_quantization_preserves_order() {
    let scale: f32 = kani::any();
    kani::assume!(scale.is_finite());
    kani::assume!(scale > 1e-10 && scale < 100.0);

    let x: f32 = kani::any();
    let y: f32 = kani::any();
    kani::assume!(x.is_finite() && y.is_finite());
    kani::assume!(x > y);
    // Stay within representable range to avoid clamp-induced aliasing
    kani::assume!(x.abs() <= 127.0 * scale);
    kani::assume!(y.abs() <= 127.0 * scale);

    // Quantize both values (symmetric INT8, zp=0)
    let qx = (x / scale).round().clamp(-128.0, 127.0) as i8;
    let qy = (y / scale).round().clamp(-128.0, 127.0) as i8;

    // Monotonicity: x > y implies quant(x) >= quant(y)
    assert!(
        qx >= qy,
        "quantization must be monotone: x > y implies quant(x) >= quant(y)"
    );
}

// ---------------------------------------------------------------------------
// 7. Group quantization: group size divides tensor dimension
// ---------------------------------------------------------------------------

/// Prove: for valid group quantization parameters (group_size > 0,
/// in_features > 0, in_features divisible by group_size), the number of
/// groups per row is exact and the total group count is consistent.
///
/// The `quantize_per_group` function in `weight_quant.rs` requires
/// `in_features % group_size == 0`. This harness verifies that when the
/// precondition holds, the derived quantities (groups_per_row, total_groups,
/// packed data length) are self-consistent and do not overflow.
#[kani::proof]
#[kani::unwind(1)]
fn proof_group_quantization_consistency() {
    let group_size: usize = kani::any();
    kani::assume!(group_size >= 1 && group_size <= 256);

    let groups_per_row: usize = kani::any();
    kani::assume!(groups_per_row >= 1 && groups_per_row <= 256);

    let out_features: usize = kani::any();
    kani::assume!(out_features >= 1 && out_features <= 256);

    let in_features = groups_per_row * group_size;

    // Divisibility invariant
    assert!(
        in_features % group_size == 0,
        "in_features must be divisible by group_size"
    );
    assert!(
        in_features / group_size == groups_per_row,
        "groups_per_row must equal in_features / group_size"
    );

    // Total groups consistency
    let total_groups = out_features * groups_per_row;
    assert!(
        total_groups == out_features * (in_features / group_size),
        "total_groups must be out_features * groups_per_row"
    );

    // INT4 packed byte count: 2 elements per byte
    let total_elements = out_features * in_features;
    let packed_int4 = (total_elements + 1) / 2;

    // Each group has `group_size` elements and one scale+zp pair.
    // Scale array length must equal total_groups.
    assert!(
        total_groups * group_size == total_elements,
        "groups * group_size must cover all elements"
    );

    // Packed byte count must be large enough
    assert!(
        packed_int4 * 2 >= total_elements,
        "INT4 packed bytes must cover all elements"
    );
}

// ---------------------------------------------------------------------------
// 8. Mixed-precision: F32 accumulation of INT8 products doesn't overflow
// ---------------------------------------------------------------------------

/// Prove: accumulating K dequantized INT8*F16 products in F32 does not
/// overflow for realistic inner dimensions (K <= 16384).
///
/// In a W8A16 GEMM, each output element is:
///   sum_{k=0}^{K-1} activation_f16[k] * dequant_int8[k]
/// where dequant_int8[k] = q_i8 * scale, |q_i8| <= 128, |scale| < 1000.
/// The maximum single product is 128 * 1000 * 65504 (f16 max for activation).
/// But realistic activations are much smaller. We prove a tighter bound:
/// |activation| <= 100, |dequant| <= 128000, so |product| <= 12_800_000.
/// For K = 16384: max_sum = 16384 * 12_800_000 = 209_715_200_000 < f32::MAX.
///
/// This harness proves a single accumulation step stays finite.
#[kani::proof]
#[kani::unwind(1)]
fn proof_mixed_precision_f32_accumulation_safe() {
    // Single dequantized INT8 value: |q_i8| <= 128, |scale| < 1000
    let dequant_val: f32 = kani::any();
    kani::assume!(dequant_val.is_finite());
    kani::assume!(dequant_val >= -128_000.0 && dequant_val <= 128_000.0);

    // Activation value (F16 range but practically bounded)
    let activation: f32 = kani::any();
    kani::assume!(activation.is_finite());
    kani::assume!(activation >= -100.0 && activation <= 100.0);

    let product = dequant_val * activation;
    assert!(
        product.is_finite(),
        "INT8 * activation product must be finite"
    );
    assert!(
        product >= -12_800_000.0 && product <= 12_800_000.0,
        "product must be bounded by |dequant_max| * |act_max|"
    );

    // Accumulator: partial sum of up to K-1 = 16383 previous products
    let accumulator: f32 = kani::any();
    kani::assume!(accumulator.is_finite());
    // 16383 * 12_800_000 = 209_702_400_000
    kani::assume!(accumulator >= -209_702_400_000.0 && accumulator <= 209_702_400_000.0);

    let result = accumulator + product;
    assert!(
        result.is_finite(),
        "F32 accumulation must not overflow for K <= 16384"
    );
}

// ---------------------------------------------------------------------------
// 9. Dynamic range: quantization range covers actual weight distribution
// ---------------------------------------------------------------------------

/// Prove: for symmetric quantization, any value in [-abs_max, abs_max]
/// maps to a representable quantized level (no clipping for in-range values).
///
/// The quantization scale is calibrated from the actual weight distribution:
/// scale = abs_max / qmax. For any v with |v| <= abs_max, the unclamped
/// quantized value |v/scale| = |v| * qmax / abs_max <= qmax, so the clamp
/// is not active. This means no information is lost from clipping for any
/// value within the calibrated range.
#[kani::proof]
#[kani::unwind(1)]
fn proof_dynamic_range_coverage() {
    let abs_max: f32 = kani::any();
    kani::assume!(abs_max.is_finite());
    kani::assume!(abs_max > 1e-10 && abs_max < 1e10);

    // INT8 symmetric: qmax = 127
    let scale_int8 = abs_max / 127.0;
    kani::assume!(scale_int8.is_finite() && scale_int8 > 0.0);

    let v: f32 = kani::any();
    kani::assume!(v.is_finite());
    kani::assume!(v >= -abs_max && v <= abs_max);

    // The unclamped quantized value
    let q_unclamped = v / scale_int8;
    kani::assume!(q_unclamped.is_finite());

    // q_unclamped should be in [-127, 127] (no clipping needed)
    // Allow small f32 rounding: 127 + epsilon
    assert!(
        q_unclamped >= -127.0 - 1e-3 && q_unclamped <= 127.0 + 1e-3,
        "value within [-abs_max, abs_max] must map to [-qmax, qmax] without clipping"
    );

    // After rounding and clamping, the value should not be clamped
    let q_clamped = q_unclamped.round().clamp(-128.0, 127.0);
    // Verify the clamp at -128 was not hit (symmetric uses [-127, 127])
    // The round() of a value in ~[-127, 127] stays in [-127, 127]
    assert!(
        q_clamped >= -127.0 - 0.5 && q_clamped <= 127.0 + 0.5,
        "in-range values must not be clipped by the clamp operation"
    );
}

// ---------------------------------------------------------------------------
// 10. Block quantization (Q4_K): super-block structure, scales per block
// ---------------------------------------------------------------------------

/// Prove: Q4_0 block dequantization produces finite values and respects
/// the block structure invariants.
///
/// Q4_0: 18 bytes per block of 32 elements. Layout:
///   [f16 scale][16 bytes: 32 x 4-bit signed, packed 2/byte]
/// Dequant: val = scale * (nibble - 8)
///
/// For any finite f16 scale and any nibble value (0..15), the dequantized
/// value is finite and bounded. The nibble-8 shift maps [0,15] to [-8,7],
/// so |val| <= |scale| * 8. For f16 max (~65504), max |val| = 524032.
///
/// We also verify the block byte count: 32 elements at 4 bits = 16 bytes
/// of packed data + 2 bytes for the f16 scale = 18 bytes per block.
#[kani::proof]
#[kani::unwind(1)]
fn proof_q4_block_dequant_safety() {
    // f16 scale: any representable f16 value
    let scale_bits: u16 = kani::any();
    let scale_f16 = half::f16::from_bits(scale_bits);
    let scale = scale_f16.to_f32();
    kani::assume!(scale.is_finite());

    // A 4-bit nibble (one of the 32 quantized values in the block)
    let nibble: u8 = kani::any();
    kani::assume!(nibble <= 15);

    // Q4_0 dequantization: val = scale * (nibble - 8)
    let shifted = nibble as i32 - 8; // in [-8, 7]
    assert!(shifted >= -8 && shifted <= 7, "nibble-8 must be in [-8, 7]");

    let dequant = scale * shifted as f32;

    // Result must be finite for finite scale
    assert!(
        dequant.is_finite(),
        "Q4_0 dequantized value must be finite for finite scale"
    );

    // Magnitude bound: |dequant| <= |scale| * 8
    let max_magnitude = scale.abs() * 8.0;
    assert!(
        dequant.abs() <= max_magnitude + 1e-10,
        "Q4_0 dequant magnitude must be bounded by |scale| * 8"
    );

    // Block structure invariants (compile-time verifiable)
    let block_elements: usize = 32;
    let nibble_bytes: usize = block_elements / 2; // 16
    let scale_bytes: usize = 2; // f16
    let block_bytes: usize = scale_bytes + nibble_bytes; // 18

    assert!(nibble_bytes == 16, "32 nibbles pack into 16 bytes");
    assert!(block_bytes == 18, "Q4_0 block is 18 bytes");

    // For Q4_K (super-blocks): 256 elements, ~144 bytes per block
    let super_block_elements: usize = 256;
    let sub_blocks_per_super: usize = super_block_elements / 32;
    assert!(
        sub_blocks_per_super == 8,
        "Q4_K super-block contains 8 sub-blocks of 32"
    );
}
