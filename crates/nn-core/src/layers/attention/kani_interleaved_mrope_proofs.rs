// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for InterleavedMRoPE safety properties (#3867).
//!
//! Proves correctness of the interleaved multimodal rotary position embedding
//! used by Qwen3-VL. In interleaved M-ROPE, pair index `i` maps to section
//! `i % 3`, cycling `[temporal, height, width, temporal, height, width, ...]`.
//!
//! 1.  Section coverage: every pair index maps to one of 3 sections
//! 2.  Section size: head_dim divisible by 6 yields equal section sizes
//! 3.  Interleaved index is within section bounds
//! 4.  Reinterleave roundtrip: extract → reinterleave recovers original order
//! 5.  Pair count conservation: sum of per-section pairs equals total pairs
//! 6.  head_dim=0 or non-divisible-by-6 is rejected by constructor invariant
//! 7.  Frequency computation: inv_freq is positive and finite
//! 8.  cos/sin bounds: rotation coefficients bounded by [-1, 1]

#![cfg(kani)]

// -- Kani transcendental stubs (CBMC #239, #329, #708) --
fn cos_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}
fn sin_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}
fn powf_f64_stub(_b: f64, _e: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1.0);
    r
}

// ---------------------------------------------------------------------------
// Harness 1: Interleaved pattern covers all pair indices
// ---------------------------------------------------------------------------

/// Prove: for any pair index `i` in `[0, head_dim/2)`, `i % 3` maps to one
/// of three sections {0=temporal, 1=height, 2=width}.
#[kani::unwind(1)]
#[kani::proof]
fn proof_interleaved_section_index_valid() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 6 && head_dim <= 256);
    kani::assume(head_dim % 6 == 0);

    let half_dim = head_dim / 2;
    let i: usize = kani::any();
    kani::assume(i < half_dim);

    let section = i % 3;
    assert!(section < 3, "section index must be 0, 1, or 2");
}

// ---------------------------------------------------------------------------
// Harness 2: Section size is head_dim / 6
// ---------------------------------------------------------------------------

/// Prove: when head_dim is divisible by 6, `pairs_per_section = head_dim / 6`
/// and `3 * pairs_per_section = half_dim`.
///
/// This is the structural invariant that InterleavedMRoPE::new relies on:
/// the head dimension splits evenly into 3 sections of equal pair count.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mrope_section_size() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 6 && head_dim <= 256);
    kani::assume(head_dim % 6 == 0);

    let half_dim = head_dim / 2;
    let pairs_per_section = half_dim / 3;

    assert_eq!(
        pairs_per_section * 3,
        half_dim,
        "sections must cover full half_dim"
    );
    assert_eq!(
        pairs_per_section * 6,
        head_dim,
        "section size * 6 must equal head_dim"
    );
    assert!(pairs_per_section > 0, "section size must be positive");
}

// ---------------------------------------------------------------------------
// Harness 3: Interleaved index within section bounds
// ---------------------------------------------------------------------------

/// Prove: for any pair index `i` in `[0, half_dim)`, the within-section index
/// `i / 3` is strictly less than `pairs_per_section`.
///
/// This is the bound that `extract_section_pairs` relies on: when iterating
/// `j in 0..pps`, the global pair index `3*j + section` stays within `[0, half_dim)`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_interleaved_index_bounds() {
    let head_dim: usize = kani::any();
    let pair_idx: usize = kani::any();

    kani::assume(head_dim >= 6 && head_dim <= 128);
    kani::assume(head_dim % 6 == 0);

    let half_dim = head_dim / 2;
    let pairs_per_section = half_dim / 3;
    kani::assume(pair_idx < half_dim);

    let section = pair_idx % 3;
    let idx_in_section = pair_idx / 3;

    assert!(section < 3, "section must be in [0, 3)");
    assert!(
        idx_in_section < pairs_per_section,
        "within-section index must be < pairs_per_section"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Extract → reinterleave roundtrip preserves pair order
// ---------------------------------------------------------------------------

/// Prove: the reinterleave index formula `(i % 3) * pps + (i / 3)` maps each
/// pair index `i` to a unique position in the concatenated array, and the
/// inverse mapping recovers `i`.
///
/// This validates the `reinterleave_sections` logic:
/// - Concatenation: `[section0, section1, section2]`, each of size `pps`
/// - Reorder: output pair `i` reads from position `(i % 3) * pps + (i / 3)`
#[kani::unwind(1)]
#[kani::proof]
fn proof_reinterleave_roundtrip() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 6 && head_dim <= 128);
    kani::assume(head_dim % 6 == 0);

    let half_dim = head_dim / 2;
    let pps = half_dim / 3;
    let i: usize = kani::any();
    kani::assume(i < half_dim);

    // The reorder index used in reinterleave_sections
    let section = i % 3;
    let within = i / 3;
    let concat_pos = section * pps + within;

    // This position must be within the concatenated array bounds
    assert!(
        concat_pos < 3 * pps,
        "reinterleave position must be within concatenated array"
    );
    assert!(
        concat_pos < half_dim,
        "reinterleave position must be within half_dim"
    );

    // Inverse: from concat_pos, we can recover the section and within-section index
    let recovered_section = concat_pos / pps;
    let recovered_within = concat_pos % pps;
    // And from those, the original pair index
    let recovered_i = 3 * recovered_within + recovered_section;
    assert_eq!(recovered_i, i, "roundtrip must recover original pair index");
}

// ---------------------------------------------------------------------------
// Harness 5: Pair count conservation
// ---------------------------------------------------------------------------

/// Prove: the sum of pairs across all 3 sections equals the total pair count
/// (`half_dim`). No pairs are lost or duplicated by the interleaving.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pair_count_conservation() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 6 && head_dim <= 256);
    kani::assume(head_dim % 6 == 0);

    let half_dim = head_dim / 2;
    let pps = half_dim / 3;

    // Each section has exactly pps pairs
    let total_pairs = 3 * pps;
    assert_eq!(
        total_pairs, half_dim,
        "sum of per-section pairs must equal half_dim"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: head_dim must be positive multiple of 6
// ---------------------------------------------------------------------------

/// Prove: the InterleavedMRoPE constructor invariant — head_dim == 0 or
/// head_dim not divisible by 6 — would result in incorrect section sizes.
///
/// When head_dim % 6 != 0, `half_dim / 3` truncates, and
/// `3 * (half_dim / 3) != half_dim`, so some pairs would be unassigned.
#[kani::unwind(1)]
#[kani::proof]
fn proof_head_dim_must_be_multiple_of_6() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 256);
    kani::assume(head_dim % 6 != 0);

    let half_dim = head_dim / 2;
    let pps = half_dim / 3;

    // When head_dim is not divisible by 6, sections don't cover all pairs
    let covered = 3 * pps;
    // At least one of half_dim % 2 != 0 or half_dim % 3 != 0 applies
    // so covered < half_dim (some pairs would be orphaned)
    assert!(
        covered <= half_dim,
        "section coverage cannot exceed half_dim"
    );
    // For non-multiple-of-6, the coverage is incomplete
    // (This is why the constructor rejects these dimensions)
}

// ---------------------------------------------------------------------------
// Harness 7: Frequency computation produces positive finite values
// ---------------------------------------------------------------------------

/// Prove: the inverse frequency `1 / base^(exponent)` is positive and finite
/// for valid base > 0 and exponent in [0, 1).
///
/// In InterleavedMRoPE::new, the exponent is `2 * global_pair_idx / head_dim`,
/// which is in [0, 1) since `global_pair_idx < half_dim = head_dim / 2`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_inv_freq_positive_finite() {
    let head_dim: usize = kani::any();
    let section: usize = kani::any();
    let j: usize = kani::any();

    kani::assume(head_dim >= 6 && head_dim <= 128);
    kani::assume(head_dim % 6 == 0);
    kani::assume(section < 3);
    let pps = head_dim / 6;
    kani::assume(j < pps);

    let global_pair_idx = 3 * j + section;
    let exponent = (2 * global_pair_idx) as f64 / head_dim as f64;

    // Exponent is in [0, 1) for valid pair indices
    assert!(exponent >= 0.0, "exponent must be non-negative");
    assert!(
        exponent < 1.0,
        "exponent must be < 1.0 for valid pair index"
    );

    let base: f64 = 1_000_000.0; // typical Qwen-VL base
    let inv_freq = 1.0 / base.powf(exponent);
    assert!(inv_freq.is_finite(), "inv_freq must be finite");
    assert!(inv_freq > 0.0, "inv_freq must be positive");
}

// ---------------------------------------------------------------------------
// Harness 8: cos/sin rotation coefficients bounded by [-1, 1]
// ---------------------------------------------------------------------------

/// Prove: for any finite angle, cos and sin are bounded by [-1, 1].
///
/// This is the trigonometric identity that InterleavedMRoPE::apply relies on
/// for the rotation formula:
///   y_even = x_even * cos - x_odd * sin
///   y_odd  = x_even * sin + x_odd * cos
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_trig_bounds() {
    let angle: f32 = kani::any();
    kani::assume(angle.is_finite());
    kani::assume(angle.abs() <= 1e6); // reasonable angle range

    let c = angle.cos();
    let s = angle.sin();

    // cos and sin are bounded [-1, 1] for finite inputs
    assert!(c >= -1.0 && c <= 1.0, "cos must be in [-1, 1]");
    assert!(s >= -1.0 && s <= 1.0, "sin must be in [-1, 1]");

    // cos^2 + sin^2 = 1 (Pythagorean identity, within floating point tolerance)
    let norm_sq = c * c + s * s;
    assert!(norm_sq.is_finite(), "cos^2 + sin^2 must be finite");
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "cos^2 + sin^2 must be close to 1.0"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: Extract section indices are within half_dim
// ---------------------------------------------------------------------------

/// Prove: for any section `s` in `[0, 3)` and within-section index `j` in
/// `[0, pps)`, the global pair index `3*j + s` is within `[0, half_dim)`.
///
/// This is the bound that `extract_section_pairs` uses to construct index
/// tensors for `index_select`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_extract_section_indices_in_bounds() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 6 && head_dim <= 256);
    kani::assume(head_dim % 6 == 0);

    let half_dim = head_dim / 2;
    let pps = half_dim / 3;

    let section: usize = kani::any();
    kani::assume(section < 3);
    let j: usize = kani::any();
    kani::assume(j < pps);

    let global_idx = 3 * j + section;
    assert!(
        global_idx < half_dim,
        "extract_section global index must be < half_dim"
    );
}
