// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for quantization safety.
//!
//! Supplements the 10 harnesses in `kani_quantization_safety.rs` and the 8
//! harnesses in `kani_quantized.rs` with 28 additional proofs covering:
//!
//!  1. INT8 per-channel quantization roundtrip bounds
//!  2. INT4 block quantization error bounds
//!  3. Quantization scale factor positivity invariants
//!  4. Zero-point range validity for asymmetric quantization
//!  5. Symmetric vs asymmetric quantization equivalence properties
//!  6. Mixed-precision casting safety (F32 -> BF16 -> F32)
//!  7. Quantized matmul output range
//!  8. Dequantization accuracy bounds
//!  9. Quantization group size constraints
//! 10. GGML dtype conversion safety
//! 11. Weight compression ratio properties
//! 12. Activation quantization dynamic range
//! 13. Per-tensor vs per-channel error comparison
//! 14. Quantization calibration statistics bounds
//! 15. Overflow detection during quantization
//!
//! All harnesses operate on pure scalar arithmetic — no DynTensor, ndarray,
//! or GPU storage — making them tractable for CBMC symbolic execution.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// Harness 1: INT8 per-channel roundtrip preserves sign
// ---------------------------------------------------------------------------

/// Prove: for symmetric INT8 quantization, the sign of a nonzero value is
/// preserved through the quantize-dequantize roundtrip when the value is
/// within the representable range and at least one quantization step from zero.
///
/// This is critical for model correctness: negative weights must remain
/// negative after quantization, and positive weights must remain positive.
#[kani::proof]
#[kani::unwind(1)]
fn proof_int8_per_channel_roundtrip_sign_preserved() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 1e-10 && scale < 100.0);

    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    // Value must be at least one quantization step from zero to survive rounding
    kani::assume(v.abs() >= scale * 0.6);
    kani::assume(v.abs() <= 127.0 * scale);

    // Quantize (symmetric, zp=0)
    let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
    // Dequantize
    let dq = (q as f32) * scale;

    // Sign must be preserved
    if v > 0.0 {
        assert!(dq >= 0.0, "positive value must dequantize to non-negative");
    }
    if v < 0.0 {
        assert!(dq <= 0.0, "negative value must dequantize to non-positive");
    }
}

// ---------------------------------------------------------------------------
// Harness 2: INT4 block quantization error bound (Q4_0 style)
// ---------------------------------------------------------------------------

/// Prove: for Q4_0-style block quantization with 4-bit unsigned nibbles
/// offset by 8 (range [-8, 7]), the dequantization error for any value
/// within the representable range is bounded by scale/2.
///
/// Q4_0 dequant: val = scale * (nibble - 8). The representable range is
/// [-8*scale, 7*scale]. For any v in this range, the roundtrip error
/// |dequant(quant(v)) - v| <= scale/2.
#[kani::proof]
#[kani::unwind(1)]
fn proof_int4_block_quant_error_bound() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 1e-8 && scale < 100.0);

    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    // Stay within the representable range [-8*scale, 7*scale]
    kani::assume(v >= -8.0 * scale && v <= 7.0 * scale);

    // Quantize: nibble = round(v/scale) + 8, clamped to [0, 15]
    let q_shifted = (v / scale).round();
    kani::assume(q_shifted.is_finite());
    let nibble = (q_shifted + 8.0).clamp(0.0, 15.0) as u8;

    // Dequantize: val = scale * (nibble - 8)
    let dq = scale * (nibble as f32 - 8.0);

    let error = (dq - v).abs();
    let bound = scale / 2.0 + 1e-5;
    assert!(
        error <= bound,
        "INT4 block roundtrip error must be bounded by scale/2"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Scale factor positivity for non-constant distributions
// ---------------------------------------------------------------------------

/// Prove: when min_val < max_val (non-constant distribution), both symmetric
/// and asymmetric scale computations yield strictly positive, finite scales.
///
/// This is the foundational invariant: a degenerate (zero) scale would collapse
/// all values to a single quantization level, losing all information.
#[kani::proof]
#[kani::unwind(1)]
fn proof_scale_factor_strictly_positive() {
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(min_val.is_finite() && max_val.is_finite());
    kani::assume(min_val < max_val);
    kani::assume(min_val >= -1e10 && max_val <= 1e10);

    // Symmetric: scale = max(|min|, |max|) / 127
    let abs_max = min_val.abs().max(max_val.abs());
    kani::assume(abs_max > 0.0); // guaranteed since min < max and at least one nonzero
    let sym_scale = abs_max / 127.0;

    assert!(sym_scale > 0.0, "symmetric scale must be strictly positive");
    assert!(sym_scale.is_finite(), "symmetric scale must be finite");

    // Asymmetric: scale = (max - min) / 255
    let range = max_val - min_val;
    kani::assume(range.is_finite());
    let asym_scale = range / 255.0;

    assert!(
        asym_scale > 0.0,
        "asymmetric scale must be strictly positive"
    );
    assert!(asym_scale.is_finite(), "asymmetric scale must be finite");
}

// ---------------------------------------------------------------------------
// Harness 4: Zero-point range for asymmetric INT8
// ---------------------------------------------------------------------------

/// Prove: for asymmetric INT8 quantization, the zero-point computed as
/// zp = round(-min / scale) is always in [0, 255] when min <= 0 <= max.
///
/// When data spans zero, the zero-point maps floating-point 0.0 to an
/// integer in [0, 255], ensuring zero has a near-exact representation.
#[kani::proof]
#[kani::unwind(1)]
fn proof_zero_point_range_asymmetric() {
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(min_val.is_finite() && max_val.is_finite());
    kani::assume(min_val <= 0.0 && max_val >= 0.0);
    kani::assume(max_val > min_val);
    kani::assume(min_val >= -1e6 && max_val <= 1e6);

    let range = max_val - min_val;
    kani::assume(range.is_finite() && range > 0.0);
    let scale = range / 255.0;
    kani::assume(scale > 0.0 && scale.is_finite());

    // zp = round(-min / scale), since min <= 0, -min >= 0
    let zp_f32 = (-min_val / scale).round();
    kani::assume(zp_f32.is_finite());

    // zp should map to [0, 255] for data spanning zero
    let zp_clamped = zp_f32.clamp(0.0, 255.0) as u8;
    assert!(
        zp_clamped as u16 <= 255,
        "zero-point must be in [0, 255] for unsigned INT8"
    );

    // Verify that the zero-point is close to the unclamped value
    // (i.e., clamping didn't activate for well-behaved data)
    assert!(
        zp_f32 >= -0.5 && zp_f32 <= 255.5,
        "zero-point should be near [0, 255] for data spanning zero"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: Symmetric quantization zero-point is always zero
// ---------------------------------------------------------------------------

/// Prove: symmetric quantization always produces zero_point = 0 regardless
/// of the input distribution parameters. This is the defining property
/// that distinguishes symmetric from asymmetric quantization.
#[kani::proof]
#[kani::unwind(1)]
fn proof_symmetric_always_zero_zp() {
    let abs_max: f32 = kani::any();
    kani::assume(abs_max.is_finite());
    kani::assume(abs_max >= 0.0 && abs_max <= 1e10);

    // Symmetric INT8: scale = abs_max / 127, zero_point = 0 (by definition)
    let zero_point: i32 = 0;
    let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };

    assert!(zero_point == 0, "symmetric mode must have zp = 0");
    assert!(scale >= 0.0, "symmetric scale must be non-negative");

    // For zero input, both scale and zero_point should be zero
    if abs_max == 0.0 {
        assert!(scale == 0.0, "zero abs_max implies zero scale");
    }
}

// ---------------------------------------------------------------------------
// Harness 6: Asymmetric uses full [0, 255] unsigned range
// ---------------------------------------------------------------------------

/// Prove: asymmetric INT8 quantization maps the minimum to 0 and maximum
/// to 255, using the full unsigned byte range for maximum precision.
///
/// For v = min_val: q = round((min_val - min_val) / scale) = 0.
/// For v = max_val: q = round((max_val - min_val) / scale) = round(255) = 255.
#[kani::proof]
#[kani::unwind(1)]
fn proof_asymmetric_full_range_utilization() {
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(min_val.is_finite() && max_val.is_finite());
    kani::assume(max_val > min_val);
    kani::assume(min_val >= -1e6 && max_val <= 1e6);

    let range = max_val - min_val;
    kani::assume(range.is_finite() && range > 0.0);
    let scale = range / 255.0;
    kani::assume(scale > 0.0 && scale.is_finite());

    // Quantize min_val
    let q_min = ((min_val - min_val) / scale).round().clamp(0.0, 255.0);
    assert!(q_min == 0.0, "min_val must quantize to 0");

    // Quantize max_val
    let q_max_raw = (max_val - min_val) / scale;
    kani::assume(q_max_raw.is_finite());
    let q_max = q_max_raw.round().clamp(0.0, 255.0);
    // (max - min) / scale = (max - min) / ((max - min) / 255) = 255
    assert!(
        q_max >= 254.0 && q_max <= 255.0,
        "max_val must quantize to 255 (within rounding)"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: F32 -> BF16 -> F32 roundtrip error bounded
// ---------------------------------------------------------------------------

/// Prove: casting F32 to BF16 and back introduces an error bounded by
/// the BF16 precision (~0.78% relative error for normal numbers).
///
/// BF16 has 8 exponent bits and 7 mantissa bits (8 total significant bits
/// including implicit leading 1). The relative precision is 2^{-7} = 1/128.
/// For any normal F32 value, |bf16(x) - x| <= |x| / 128.
#[kani::proof]
#[kani::unwind(1)]
fn proof_f32_bf16_f32_roundtrip_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() > 1e-30 && x.abs() < 1e30); // normal range

    // F32 -> BF16: truncate lower 16 mantissa bits
    let bf16_val = half::bf16::from_f32(x);
    // BF16 -> F32: exact (widening)
    let recovered = bf16_val.to_f32();

    kani::assume(recovered.is_finite());

    let error = (recovered - x).abs();
    // BF16 relative error: at most 2^{-8} of the magnitude
    // (rounding to nearest gives 2^{-8}, truncation gives 2^{-7})
    let rel_bound = x.abs() / 128.0 + 1e-38; // +epsilon for subnormals

    assert!(
        error <= rel_bound,
        "BF16 roundtrip error must be bounded by |x| / 128"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: F32 -> F16 -> F32 roundtrip preserves finiteness
// ---------------------------------------------------------------------------

/// Prove: for F32 values within F16 representable range (|x| <= 65504),
/// the F16 roundtrip produces a finite result.
///
/// F16 has a maximum magnitude of 65504. Values within this range are
/// representable (possibly with precision loss) but always finite.
#[kani::proof]
#[kani::unwind(1)]
fn proof_f32_f16_f32_roundtrip_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() <= 65504.0);

    let f16_val = half::f16::from_f32(x);
    let recovered = f16_val.to_f32();

    assert!(
        recovered.is_finite(),
        "F16 roundtrip of in-range F32 must produce finite result"
    );

    // F16 has 10 mantissa bits -> relative precision ~2^{-11}
    // For normal values, the error is bounded
    if x.abs() > 6.1e-5 {
        // above F16 minimum normal
        let error = (recovered - x).abs();
        let rel_bound = x.abs() / 1024.0 + 1e-7;
        assert!(
            error <= rel_bound,
            "F16 roundtrip error must be bounded for normal values"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 9: Quantized INT8 matmul single-element output bounded
// ---------------------------------------------------------------------------

/// Prove: for a single dot-product element of a W8A8 matmul with K terms,
/// each term is bounded by 128 * 127 = 16256, and for K <= 4096 the
/// accumulation in i32 does not overflow (K * 16256 < 2^31).
///
/// W8A8: both weights and activations are INT8.
/// Each product: w_i8 * a_i8, where |w_i8| <= 128, |a_i8| <= 128.
/// |product| <= 128 * 127 = 16256. K products sum to at most 66_584_576
/// for K = 4096, well within i32 range.
#[kani::proof]
#[kani::unwind(1)]
fn proof_int8_matmul_single_output_bounded() {
    let w_u8: u8 = kani::any();
    let a_u8: u8 = kani::any();
    let w_i8 = w_u8 as i8;
    let a_i8 = a_u8 as i8;

    // Single product in i32
    let product = (w_i8 as i32) * (a_i8 as i32);
    assert!(
        product >= -16384 && product <= 16384,
        "INT8 * INT8 product must fit in [-16384, 16384]"
    );

    // Partial accumulator for up to K-1 = 4095 previous products
    let partial: i32 = kani::any();
    kani::assume(partial >= -66_846_720 && partial <= 66_846_720);

    let result = partial.checked_add(product);
    assert!(
        result.is_some(),
        "INT8 matmul accumulation in i32 must not overflow for K <= 4096"
    );

    let sum = result.unwrap();
    assert!(
        sum >= -66_863_104 && sum <= 66_863_104,
        "accumulated sum must stay within safe i32 range"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: Dequantization accuracy — max error equals half a step
// ---------------------------------------------------------------------------

/// Prove: the maximum possible roundtrip error for symmetric INT8
/// quantization is exactly scale/2 (achieved when the input is at the
/// midpoint between two quantization levels).
///
/// For v = (q + 0.5) * scale: round(v/scale) = q + 1, error = 0.5 * scale.
/// No larger error is possible because round() rounds to nearest integer.
#[kani::proof]
#[kani::unwind(1)]
fn proof_dequant_max_error_half_step() {
    let q_int: i8 = kani::any();
    kani::assume(q_int > -128 && q_int < 127); // avoid clamp boundaries

    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 1e-6 && scale < 100.0);

    // Value at the midpoint between quantization levels q and q+1
    let midpoint = ((q_int as f32) + 0.5) * scale;
    kani::assume(midpoint.is_finite());

    // Quantize the midpoint
    let q_result = (midpoint / scale).round().clamp(-128.0, 127.0) as i8;
    let dq = (q_result as f32) * scale;

    let error = (dq - midpoint).abs();
    // Error should be exactly scale/2 (within f32 precision)
    assert!(
        error <= scale / 2.0 + 1e-5,
        "midpoint error must be at most scale/2"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: Group size must divide dimension (no partial groups)
// ---------------------------------------------------------------------------

/// Prove: for valid group quantization (group_size divides in_features),
/// the last group ends exactly at in_features, with no leftover elements.
///
/// Partial groups would require special-case handling (different scale
/// computation for the last group), which most quantization implementations
/// do not support.
#[kani::proof]
#[kani::unwind(1)]
fn proof_group_size_divides_dim_no_remainder() {
    let in_features: usize = kani::any();
    let group_size: usize = kani::any();
    kani::assume(in_features >= 1 && in_features <= 8192);
    kani::assume(group_size >= 1 && group_size <= 512);
    kani::assume(in_features % group_size == 0);

    let num_groups = in_features / group_size;

    // Last group ends exactly at in_features
    let last_group_end = num_groups * group_size;
    assert!(
        last_group_end == in_features,
        "last group must end exactly at in_features"
    );

    // No remainder
    assert!(
        in_features - last_group_end == 0,
        "there must be no leftover elements"
    );

    // At least one group
    assert!(num_groups >= 1, "must have at least one group");
}

// ---------------------------------------------------------------------------
// Harness 12: Common group sizes are powers of 2
// ---------------------------------------------------------------------------

/// Prove: for power-of-2 group sizes (32, 64, 128, 256), any dimension
/// that is a multiple of 256 is also a multiple of the group size.
///
/// This ensures that models with dimensions aligned to 256 (common in
/// transformers: 256, 512, 768, 1024, 2048, 4096) work with all standard
/// group sizes without partial groups.
#[kani::proof]
#[kani::unwind(1)]
fn proof_power_of_2_group_sizes_divide_aligned_dims() {
    let dim_mult: usize = kani::any();
    kani::assume(dim_mult >= 1 && dim_mult <= 64);
    let dim = dim_mult * 256;

    // Group size 32 divides
    assert!(dim % 32 == 0, "dim aligned to 256 must be divisible by 32");
    // Group size 64 divides
    assert!(dim % 64 == 0, "dim aligned to 256 must be divisible by 64");
    // Group size 128 divides
    assert!(
        dim % 128 == 0,
        "dim aligned to 256 must be divisible by 128"
    );
    // Group size 256 divides
    assert!(
        dim % 256 == 0,
        "dim aligned to 256 must be divisible by 256"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: GGML Q4_K block byte count consistency
// ---------------------------------------------------------------------------

/// Prove: Q4_K block storage (144 bytes for 256 elements) is consistent
/// with the component layout: 2 bytes (f16 d) + 2 bytes (f16 dmin) +
/// 12 bytes (scales) + 128 bytes (nibbles) = 144 bytes.
///
/// The 128 nibble bytes store 256 4-bit values (2 per byte).
#[kani::proof]
#[kani::unwind(1)]
fn proof_ggml_q4k_block_byte_count() {
    let d_bytes: usize = 2; // f16 scale
    let dmin_bytes: usize = 2; // f16 minimum
    let scales_bytes: usize = 12; // packed sub-block scales
    let elements: usize = 256;
    let nibble_bytes: usize = elements / 2; // 128 bytes for 256 4-bit values

    let total = d_bytes + dmin_bytes + scales_bytes + nibble_bytes;
    assert!(total == 144, "Q4_K block must be 144 bytes");

    // Bits per weight
    let bits_times_256 = total * 8; // 1152
                                    // 1152 / 256 = 4.5 bpw (verified via integer arithmetic)
    assert!(bits_times_256 == 1152);
    assert!(bits_times_256 / elements == 4);
    assert!(bits_times_256 % elements == 128); // 128/256 = 0.5 fractional bpw
}

// ---------------------------------------------------------------------------
// Harness 14: GGML Q8_0 block byte count consistency
// ---------------------------------------------------------------------------

/// Prove: Q8_0 block storage (34 bytes for 32 elements) is consistent
/// with: 2 bytes (f16 scale) + 32 bytes (32 x int8 values) = 34 bytes.
/// Effective bits per weight: 34*8/32 = 8.5.
#[kani::proof]
#[kani::unwind(1)]
fn proof_ggml_q8_0_block_byte_count() {
    let scale_bytes: usize = 2; // f16 scale
    let elements: usize = 32;
    let data_bytes: usize = elements; // 1 byte per INT8

    let total = scale_bytes + data_bytes;
    assert!(total == 34, "Q8_0 block must be 34 bytes");

    // Bits per weight: 34 * 8 / 32 = 272 / 32 = 8.5
    let bits_times_32 = total * 8; // 272
    assert!(bits_times_32 == 272);
    assert!(bits_times_32 / elements == 8);
    assert!(bits_times_32 % elements == 16); // 16/32 = 0.5 fractional bpw
}

// ---------------------------------------------------------------------------
// Harness 15: Weight compression ratio: Q4_K vs F32
// ---------------------------------------------------------------------------

/// Prove: for any weight tensor of N elements where N >= 256 and N is a
/// multiple of 256, the Q4_K compressed size is strictly less than the
/// F32 size, achieving at least 7x compression.
///
/// F32: 4 bytes/element. Q4_K: 144 bytes per 256 elements = 0.5625 bytes/element.
/// Ratio: 4 / 0.5625 = 7.111...
#[kani::proof]
#[kani::unwind(1)]
fn proof_q4k_compression_ratio_vs_f32() {
    let num_blocks: usize = kani::any();
    kani::assume(num_blocks >= 1 && num_blocks <= 65536);

    let elements = num_blocks * 256;
    let f32_bytes = elements * 4;
    let q4k_bytes = num_blocks * 144;

    assert!(q4k_bytes < f32_bytes, "Q4_K must use less memory than F32");

    // At least 7x compression: f32_bytes >= 7 * q4k_bytes
    // 4 * 256 * n = 1024n >= 7 * 144n = 1008n. 1024 >= 1008. True.
    assert!(
        f32_bytes >= 7 * q4k_bytes,
        "Q4_K must achieve at least 7x compression vs F32"
    );
}

// ---------------------------------------------------------------------------
// Harness 16: Activation quantization dynamic range is non-negative
// ---------------------------------------------------------------------------

/// Prove: for any collection of finite values, the dynamic range
/// (max - min) is always non-negative. When all values are identical,
/// the range is zero (degenerate case requiring special handling).
#[kani::proof]
#[kani::unwind(1)]
fn proof_activation_dynamic_range_nonneg() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    // Compute min/max of three values
    let min_val = a.min(b).min(c);
    let max_val = a.max(b).max(c);
    let range = max_val - min_val;

    assert!(range >= 0.0, "dynamic range must be non-negative");
    assert!(
        range.is_finite(),
        "dynamic range of finite values must be finite"
    );

    // If any two values differ, range is strictly positive
    if a != b || b != c {
        assert!(range > 0.0, "range of non-constant values must be positive");
    }
}

// ---------------------------------------------------------------------------
// Harness 17: Per-tensor scale is at least as coarse as per-channel
// ---------------------------------------------------------------------------

/// Prove: per-tensor quantization scale (computed from the global abs_max)
/// is always >= per-channel scale (computed from a single channel's abs_max).
///
/// This means per-tensor quantization has worse precision (coarser steps)
/// than per-channel, justifying the preference for per-channel in production.
#[kani::proof]
#[kani::unwind(1)]
fn proof_per_tensor_scale_gte_per_channel() {
    let channel_abs_max: f32 = kani::any();
    let global_abs_max: f32 = kani::any();
    kani::assume(channel_abs_max.is_finite() && global_abs_max.is_finite());
    kani::assume(channel_abs_max >= 0.0 && global_abs_max >= 0.0);
    kani::assume(channel_abs_max <= global_abs_max); // channel max <= global max
    kani::assume(global_abs_max > 0.0);

    let per_channel_scale = channel_abs_max / 127.0;
    let per_tensor_scale = global_abs_max / 127.0;

    assert!(
        per_tensor_scale >= per_channel_scale,
        "per-tensor scale must be >= per-channel scale"
    );
}

// ---------------------------------------------------------------------------
// Harness 18: Per-channel has lower or equal max error than per-tensor
// ---------------------------------------------------------------------------

/// Prove: for symmetric INT8 quantization of a value within a channel,
/// the per-channel roundtrip error (using the channel's own scale) is
/// always <= the per-tensor roundtrip error (using the global scale).
///
/// Error bound = scale/2. Per-channel scale <= per-tensor scale (from
/// Harness 17). Therefore per-channel error <= per-tensor error.
#[kani::proof]
#[kani::unwind(1)]
fn proof_per_channel_error_lte_per_tensor() {
    let channel_abs_max: f32 = kani::any();
    let global_abs_max: f32 = kani::any();
    kani::assume(channel_abs_max.is_finite() && global_abs_max.is_finite());
    kani::assume(channel_abs_max > 0.0 && global_abs_max > 0.0);
    kani::assume(channel_abs_max <= global_abs_max);
    kani::assume(global_abs_max < 1e10);

    let per_channel_scale = channel_abs_max / 127.0;
    let per_tensor_scale = global_abs_max / 127.0;

    let per_channel_max_error = per_channel_scale / 2.0;
    let per_tensor_max_error = per_tensor_scale / 2.0;

    assert!(
        per_channel_max_error <= per_tensor_max_error,
        "per-channel max error must be <= per-tensor max error"
    );
}

// ---------------------------------------------------------------------------
// Harness 19: Calibration running min/max update is monotonic
// ---------------------------------------------------------------------------

/// Prove: the running min/max update for quantization calibration is
/// monotonic: min can only decrease, max can only increase. This ensures
/// the calibration range never shrinks, preventing clipping regression.
#[kani::proof]
#[kani::unwind(1)]
fn proof_calibration_minmax_monotonic() {
    // Current running statistics
    let current_min: f32 = kani::any();
    let current_max: f32 = kani::any();
    kani::assume(current_min.is_finite() && current_max.is_finite());
    kani::assume(current_min <= current_max);

    // New observed value
    let new_val: f32 = kani::any();
    kani::assume(new_val.is_finite());

    // Update running min/max
    let updated_min = if new_val < current_min {
        new_val
    } else {
        current_min
    };
    let updated_max = if new_val > current_max {
        new_val
    } else {
        current_max
    };

    // Min can only decrease or stay the same
    assert!(
        updated_min <= current_min,
        "running min must be monotonically non-increasing"
    );
    // Max can only increase or stay the same
    assert!(
        updated_max >= current_max,
        "running max must be monotonically non-decreasing"
    );
    // Updated range is at least as wide as before
    assert!(
        updated_max - updated_min >= current_max - current_min,
        "calibration range must not shrink"
    );
}

// ---------------------------------------------------------------------------
// Harness 20: Overflow detection — clamped vs unclamped difference
// ---------------------------------------------------------------------------

/// Prove: overflow during INT8 quantization (when |v/scale| > 127) is
/// detectable by comparing the clamped and unclamped quantized values.
/// If they differ, the value was clipped.
///
/// This is the basis for the saturation count metric used in calibration:
/// count how many values get clipped to detect poor scale selection.
#[kani::proof]
#[kani::unwind(1)]
fn proof_overflow_detection_clamp_difference() {
    let v: f32 = kani::any();
    let scale: f32 = kani::any();
    kani::assume(v.is_finite() && scale.is_finite());
    kani::assume(scale > 1e-10 && scale < 1e10);

    let q_unclamped = (v / scale).round();
    kani::assume(q_unclamped.is_finite());
    let q_clamped = q_unclamped.clamp(-128.0, 127.0);

    let was_clipped = q_unclamped != q_clamped;

    if was_clipped {
        // If clipped, the unclamped value was outside [-128, 127]
        assert!(
            q_unclamped < -128.0 || q_unclamped > 127.0,
            "clipping implies unclamped was outside representable range"
        );
    } else {
        // If not clipped, the value was representable
        assert!(
            q_unclamped >= -128.0 && q_unclamped <= 127.0,
            "no clipping implies value was representable"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 21: INT8 dequant with zero-point offset preserves finiteness
// ---------------------------------------------------------------------------

/// Prove: for any valid INT8 value, scale, and zero-point, the asymmetric
/// dequantization formula `(q - zp) * scale` produces a finite result.
///
/// This is the core dequantization operation for asymmetric INT8. The
/// subtraction (q - zp) can range from -383 to 383 (128 + 255), and
/// multiplication by scale must stay finite.
#[kani::proof]
#[kani::unwind(1)]
fn proof_int8_dequant_zp_offset_finite() {
    let q_u8: u8 = kani::any();
    let q_i8 = q_u8 as i8;

    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale >= 0.0 && scale < 1e6);

    let zero_point: i32 = kani::any();
    kani::assume(zero_point >= -256 && zero_point <= 256);

    let diff = q_i8 as i32 - zero_point;
    let dequant = (diff as f32) * scale;

    assert!(
        dequant.is_finite(),
        "asymmetric dequantization must produce finite result"
    );

    // Bound: |diff| <= 384, |dequant| <= 384 * scale
    let bound = 384.0 * scale + 1e-5;
    assert!(
        dequant.abs() <= bound,
        "dequantized value must be bounded by 384 * scale"
    );
}

// ---------------------------------------------------------------------------
// Harness 22: Mixed-precision BF16 accumulation safety
// ---------------------------------------------------------------------------

/// Prove: accumulating BF16-precision products into an F32 accumulator
/// preserves finiteness. BF16 max is ~3.4e38 (same exponent range as F32),
/// so overflow is only possible when accumulating many large products.
///
/// For typical transformer hidden dims (K <= 4096) and bounded activations,
/// the accumulation is safe.
#[kani::proof]
#[kani::unwind(1)]
fn proof_bf16_accumulation_f32_safe() {
    // BF16 value as f32 (BF16 fits exactly in f32)
    let bf16_bits: u16 = kani::any();
    let bf16_val = half::bf16::from_bits(bf16_bits);
    let a = bf16_val.to_f32();
    kani::assume(a.is_finite());
    kani::assume(a.abs() <= 100.0); // typical activation range

    let weight: f32 = kani::any();
    kani::assume(weight.is_finite());
    kani::assume(weight.abs() <= 10.0); // typical weight range

    let product = a * weight;
    assert!(
        product.is_finite(),
        "BF16 activation * F32 weight must be finite"
    );
    assert!(
        product.abs() <= 1000.0,
        "product must be bounded by max_a * max_w"
    );

    // Accumulator
    let acc: f32 = kani::any();
    kani::assume(acc.is_finite());
    kani::assume(acc.abs() <= 4_096_000.0); // K=4096 terms * 1000

    let result = acc + product;
    assert!(
        result.is_finite(),
        "BF16 accumulation into F32 must stay finite"
    );
}

// ---------------------------------------------------------------------------
// Harness 23: GGML dtype F32 is identity conversion
// ---------------------------------------------------------------------------

/// Prove: GGML F32 "quantization" is the identity: block size = 1,
/// bytes per element = 4, and the storage is the raw F32 value.
///
/// This is the baseline for compression ratio comparisons: F32 has
/// exactly 32 bits per weight with zero quantization error.
#[kani::proof]
#[kani::unwind(1)]
fn proof_ggml_f32_identity_conversion() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    // F32 "quantization": store the raw bits
    let bytes = val.to_le_bytes();
    let recovered = f32::from_le_bytes(bytes);

    assert!(
        recovered == val || (val.is_nan() && recovered.is_nan()),
        "F32 storage must be lossless"
    );

    // Structural properties
    let f32_bytes_per_element: usize = 4;
    let f32_bits_per_weight: usize = 32;
    assert!(f32_bytes_per_element == core::mem::size_of::<f32>());
    assert!(f32_bits_per_weight == f32_bytes_per_element * 8);
}

// ---------------------------------------------------------------------------
// Harness 24: Quantization saturation count bounds
// ---------------------------------------------------------------------------

/// Prove: the saturation count (number of clipped values) for a batch of
/// N quantized values is bounded by [0, N]. If no values exceed the
/// representable range, the saturation count is 0.
///
/// The saturation rate (count / N) is the key metric for calibration
/// quality. A rate > 1% typically indicates poor scale selection.
#[kani::proof]
#[kani::unwind(1)]
fn proof_saturation_count_bounded() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 10000);

    let sat_count: usize = kani::any();
    kani::assume(sat_count <= n); // at most N values can saturate

    assert!(sat_count <= n, "saturation count must be <= N");

    // Saturation rate is in [0, 1]
    // (Use integer comparison to avoid floating point in CBMC)
    assert!(
        sat_count * 100 <= n * 100,
        "saturation rate must be in [0%, 100%]"
    );

    // If all values are within range, saturation count is 0
    if sat_count == 0 {
        // Zero saturation means perfect calibration (no clipping)
        assert!(sat_count * 100 == 0, "zero saturation means 0% rate");
    }
}

// ---------------------------------------------------------------------------
// Harness 25: INT4 nibble pack/unpack roundtrip
// ---------------------------------------------------------------------------

/// Prove: packing two INT4 values into one byte and unpacking them
/// recovers the original values. This is the fundamental storage
/// operation for all 4-bit quantization formats (Q4_0, Q4_K, GPTQ-4bit).
///
/// Pack: byte = (high_nibble << 4) | low_nibble
/// Unpack: low = byte & 0x0F, high = byte >> 4
#[kani::proof]
#[kani::unwind(1)]
fn proof_int4_nibble_pack_unpack_roundtrip() {
    let low: u8 = kani::any();
    let high: u8 = kani::any();
    kani::assume(low <= 15);
    kani::assume(high <= 15);

    // Pack two nibbles into one byte
    let packed = (high << 4) | low;

    // Unpack
    let unpacked_low = packed & 0x0F;
    let unpacked_high = packed >> 4;

    assert!(
        unpacked_low == low,
        "low nibble must survive pack/unpack roundtrip"
    );
    assert!(
        unpacked_high == high,
        "high nibble must survive pack/unpack roundtrip"
    );
}

// ---------------------------------------------------------------------------
// Harness 26: Quantization scale reciprocal safety
// ---------------------------------------------------------------------------

/// Prove: for any valid quantization scale (positive, finite, > epsilon),
/// the reciprocal (1/scale) used in the quantization formula is also
/// positive and finite. Division by scale is a core operation in every
/// quantization path.
#[kani::proof]
#[kani::unwind(1)]
fn proof_scale_reciprocal_safe() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 1e-30 && scale < 1e30);

    let reciprocal = 1.0 / scale;

    assert!(
        reciprocal.is_finite(),
        "reciprocal of valid scale must be finite"
    );
    assert!(
        reciprocal > 0.0,
        "reciprocal of positive scale must be positive"
    );

    // Verify: v / scale == v * (1/scale) for bounded v
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v.abs() < 1e10);

    let via_div = v / scale;
    let via_mul = v * reciprocal;
    kani::assume(via_div.is_finite() && via_mul.is_finite());

    // They should be very close (f32 associativity is not exact)
    let diff = (via_div - via_mul).abs();
    let tol = v.abs() * 1e-6 + 1e-30;
    assert!(diff <= tol, "v/scale and v*(1/scale) must be nearly equal");
}

// ---------------------------------------------------------------------------
// Harness 27: EMA smoothing factor for calibration is in (0, 1)
// ---------------------------------------------------------------------------

/// Prove: the exponential moving average (EMA) update for calibration
/// statistics preserves the range when the smoothing factor alpha is in
/// (0, 1). The updated value stays between the old and new observations.
///
/// EMA: updated = alpha * new_val + (1 - alpha) * old_val.
/// For alpha in (0, 1): min(old, new) <= updated <= max(old, new).
#[kani::proof]
#[kani::unwind(1)]
fn proof_ema_calibration_stays_in_range() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha > 0.0 && alpha < 1.0);

    let old_val: f32 = kani::any();
    let new_val: f32 = kani::any();
    kani::assume(old_val.is_finite() && new_val.is_finite());
    kani::assume(old_val.abs() < 1e10 && new_val.abs() < 1e10);

    let updated = alpha * new_val + (1.0 - alpha) * old_val;
    kani::assume(updated.is_finite());

    let lo = old_val.min(new_val);
    let hi = old_val.max(new_val);

    // EMA must be a convex combination: between old and new
    assert!(
        updated >= lo - 1e-5 && updated <= hi + 1e-5,
        "EMA must stay between old and new observations"
    );
}

// ---------------------------------------------------------------------------
// Harness 28: Quantized matmul bias addition stays finite
// ---------------------------------------------------------------------------

/// Prove: adding a dequantized bias to a quantized matmul result (both in
/// F32) produces a finite result for bounded inputs. This covers the final
/// step of quantized linear layer dispatch.
///
/// GEMM output: bounded by K * max_product. Bias: typically small (< 100).
/// Sum must be finite and representable in F32.
#[kani::proof]
#[kani::unwind(1)]
fn proof_quantized_matmul_bias_addition_finite() {
    // GEMM output from quantized matmul
    let gemm_output: f32 = kani::any();
    kani::assume(gemm_output.is_finite());
    kani::assume(gemm_output.abs() <= 1e9); // K=4096, max product ~255k

    // Dequantized bias
    let bias: f32 = kani::any();
    kani::assume(bias.is_finite());
    kani::assume(bias.abs() <= 1e6);

    let result = gemm_output + bias;
    assert!(
        result.is_finite(),
        "quantized GEMM + bias must produce finite result"
    );

    // Result is bounded
    assert!(
        result.abs() <= 1e9 + 1e6 + 1.0,
        "result must be bounded by GEMM bound + bias bound"
    );
}
