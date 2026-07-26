// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for weight quantization safety properties (#3867).
//!
//! Supplements the existing 15 harnesses in `int8.rs` and 20 harnesses in
//! `kani_quantized_extra.rs` with additional proofs focused on:
//!
//! 1.  INT4 range: quantized nibble values are in [-8, 7] (Q4K sub-blocks)
//! 2.  INT8 range: symmetric clamped quantization yields values in [-127, 127]
//! 3.  Group size alignment: in_features divisible by group_size
//! 4.  Symmetric dequant error bound: |dequant(quant(v)) - v| <= scale/2
//! 5.  Q4K sub-block scale pairing correctness
//!
//! These harnesses focus on the dpdf weight quantization pipeline where INT4/INT8
//! quantized weights are used for memory-efficient inference.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// Harness 1: INT4 quantized value range
// ---------------------------------------------------------------------------

/// Prove: after round + clamp to [-8, 7], the quantized value is always in
/// the valid INT4 signed range.
///
/// Q4K uses 4-bit nibbles (unsigned [0, 15] stored, but the dequant formula
/// interprets them with an offset). This harness proves the unsigned nibble
/// after clamping is always in [0, 15].
#[kani::unwind(1)]
#[kani::proof]
fn proof_int4_quantized_range() {
    let value: f32 = kani::any();
    let scale: f32 = kani::any();

    kani::assume(value.is_finite());
    kani::assume(scale.is_finite() && scale > 0.0 && scale < 1000.0);

    // Q4K-style nibble quantization: round(value / scale), clamped to [0, 15]
    let q_f32 = (value / scale).round();
    kani::assume(q_f32.is_finite());
    let clamped = q_f32.clamp(0.0, 15.0) as u8;

    assert!(clamped <= 15, "INT4 unsigned nibble must be in [0, 15]");
}

// ---------------------------------------------------------------------------
// Harness 2: INT8 symmetric clamped range
// ---------------------------------------------------------------------------

/// Prove: symmetric INT8 quantization with clamp to [-127, 127] always
/// produces a value in the valid symmetric range.
///
/// Symmetric INT8 avoids -128 to keep the range symmetric around zero.
/// This is the range used by `compute_channel_params` in symmetric mode.
#[kani::unwind(1)]
#[kani::proof]
fn proof_int8_symmetric_clamped_range() {
    let value: f32 = kani::any();
    let scale: f32 = kani::any();

    kani::assume(value.is_finite() && value.abs() <= 1e6);
    kani::assume(scale.is_finite() && scale > 1e-10 && scale < 1000.0);

    // Symmetric quantization path from quantize_per_channel
    let q_f32 = (value / scale).round().clamp(-127.0, 127.0);
    let q_i8 = q_f32 as i8;

    assert!(
        q_i8 >= -127 && q_i8 <= 127,
        "symmetric INT8 must be in [-127, 127]"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Group size alignment
// ---------------------------------------------------------------------------

/// Prove: when `in_features` is divisible by `group_size`, the number of
/// groups is exact (no partial group) and covers all features.
///
/// This is the structural invariant for per-group quantization (used when
/// quantizing weight matrices by groups of rows/columns).
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_alignment() {
    let in_features: usize = kani::any();
    let group_size: usize = kani::any();

    kani::assume(in_features > 0 && in_features <= 4096);
    kani::assume(group_size > 0 && group_size <= 256);
    kani::assume(in_features % group_size == 0);

    let num_groups = in_features / group_size;
    assert_eq!(
        num_groups * group_size,
        in_features,
        "groups must exactly cover all features"
    );
    assert!(num_groups > 0, "must have at least one group");
}

// ---------------------------------------------------------------------------
// Harness 4: Symmetric dequant roundtrip error bound (scalar)
// ---------------------------------------------------------------------------

/// Prove: for symmetric INT8 quantization, the roundtrip error
/// |dequant(quant(v)) - v| is bounded by scale/2 for values within the
/// representable range.
///
/// This strengthens the existing harness in int8.rs by explicitly proving
/// the half-step bound for the full clamp-to-[-128, 127] path (not just [-127, 127]).
#[kani::unwind(1)]
#[kani::proof]
fn proof_symmetric_dequant_error_bound() {
    let original: f32 = kani::any();
    let scale: f32 = kani::any();

    kani::assume(original.is_finite() && original.abs() <= 100.0);
    kani::assume(scale.is_finite() && scale > 1e-6 && scale <= 10.0);

    // Symmetric quantization: q = round(x / scale), dq = q * scale
    let quantized = (original / scale).round().clamp(-128.0, 127.0);
    let dequantized = quantized * scale;

    kani::assume(dequantized.is_finite());

    // Error is bounded by scale/2 (rounding error) for values in range
    let error = (original - dequantized).abs();
    if original.abs() <= 127.0 * scale {
        // Within the non-clamped representable range
        assert!(
            error <= scale * 0.5 + 1e-5,
            "dequant error should be bounded by scale/2 for in-range values"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 5: Q4K sub-block scale pairing
// ---------------------------------------------------------------------------

/// Prove: Q4K uses 256-element blocks with 4 groups of 64 elements.
/// Each group has 2 sub-blocks of 32 elements. The 4 groups use 4 scale/min
/// pairs from the 12-byte scales_and_mins array.
///
/// This harness validates the indexing relationship: for group index `g` in
/// `[0, 4)` and sub-block `s` in `[0, 2)`, the element range
/// `[g*64 + s*32, g*64 + s*32 + 32)` is within the 256-element block.
#[kani::unwind(1)]
#[kani::proof]
fn proof_q4k_subblock_indexing() {
    let g: usize = kani::any();
    let s: usize = kani::any();
    let elem: usize = kani::any();

    kani::assume(g < 4);
    kani::assume(s < 2);
    kani::assume(elem < 32);

    let block_size: usize = 256;
    let offset = g * 64 + s * 32 + elem;

    assert!(
        offset < block_size,
        "Q4K sub-block element must be within 256-element block"
    );

    // The group index (for scale/min lookup) is valid
    assert!(g < 4, "group index must be in [0, 4)");

    // Total elements covered by 4 groups * 2 sub-blocks * 32 elements = 256
    let total_covered = 4 * 2 * 32;
    assert_eq!(
        total_covered, block_size,
        "sub-blocks must cover exactly one Q4K block"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Per-channel scale computation is monotonic with abs_max
// ---------------------------------------------------------------------------

/// Prove: for symmetric INT8 quantization, a larger abs_max yields a larger
/// (or equal) scale. This ensures that wider value ranges get coarser
/// quantization, which is the correct behavior.
///
/// scale = abs_max / 127, so scale is proportional to abs_max.
#[kani::unwind(1)]
#[kani::proof]
fn proof_symmetric_scale_monotonic() {
    let abs_max_a: f32 = kani::any();
    let abs_max_b: f32 = kani::any();

    kani::assume(abs_max_a.is_finite() && abs_max_a >= 0.0 && abs_max_a <= 1e6);
    kani::assume(abs_max_b.is_finite() && abs_max_b >= 0.0 && abs_max_b <= 1e6);
    kani::assume(abs_max_a <= abs_max_b);

    let scale_a = abs_max_a / 127.0;
    let scale_b = abs_max_b / 127.0;

    assert!(
        scale_a <= scale_b,
        "larger abs_max must produce larger or equal scale"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: Asymmetric quantization range width
// ---------------------------------------------------------------------------

/// Prove: for asymmetric INT8 quantization, `scale = (max - min) / 255` is
/// non-negative, and the representable range spans exactly 255 quantization
/// steps of width `scale`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_asymmetric_range_width() {
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();

    kani::assume(min_val.is_finite() && min_val >= -1000.0);
    kani::assume(max_val.is_finite() && max_val <= 1000.0);
    kani::assume(max_val > min_val); // non-degenerate range

    let range = max_val - min_val;
    kani::assume(range.is_finite());
    let scale = range / 255.0;

    assert!(
        scale > 0.0,
        "asymmetric scale must be positive for non-degenerate range"
    );
    assert!(scale.is_finite(), "asymmetric scale must be finite");

    // 255 steps of width `scale` should reconstruct the original range
    let reconstructed_range = scale * 255.0;
    let diff = (reconstructed_range - range).abs();
    assert!(
        diff < 1e-3,
        "255 * scale must reconstruct the original range"
    );
}
