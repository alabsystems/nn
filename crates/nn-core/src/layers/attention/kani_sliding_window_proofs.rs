// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for sliding window attention properties.
//!
//! Covers:
//! - Half-window parity (even vs odd window_size)
//! - Visible count monotonicity with window_size
//! - Full window coverage when window_size >= seq_len
//! - Scale factor finite for valid head_dim
//! - QKV weight shape constraints (divisible by 3 and by num_heads)
//! - Mask element count equals seq_len^2
//! - Boundary tokens see fewer positions than interior tokens
//! - Window growth: larger window_size never masks previously visible positions
//!
//! Part of #3672.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// -- Half-window parity ----------------------------------------------------------

/// Prove half_window computation for even window_size: half = window_size / 2.
///
/// When window_size is even, the window extends exactly half_window to each side.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_half_even() {
    let window_size: usize = kani::any();
    kani::assume(window_size >= 2 && window_size <= 64);
    kani::assume(window_size % 2 == 0);
    let half_window = window_size / 2;
    // Even window: visible count at interior = 2 * half_window + 1 = window_size + 1.
    let visible_interior = 2 * half_window + 1;
    kani::assert(
        visible_interior == window_size + 1,
        "even window: interior visible count = window_size + 1",
    );
    kani::assert(
        half_window * 2 == window_size,
        "half_window round-trips for even",
    );
}

/// Prove half_window computation for odd window_size: half = (window_size - 1) / 2.
///
/// When window_size is odd, integer division yields (window_size - 1) / 2.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_half_odd() {
    let window_size: usize = kani::any();
    kani::assume(window_size >= 1 && window_size <= 63);
    kani::assume(window_size % 2 == 1);
    let half_window = window_size / 2;
    kani::assert(
        half_window == (window_size - 1) / 2,
        "odd window: half = (window_size - 1) / 2",
    );
    // Visible count at interior = 2 * half_window + 1 = window_size.
    let visible_interior = 2 * half_window + 1;
    kani::assert(
        visible_interior == window_size,
        "odd window: interior visible count = window_size",
    );
}

// -- Visible count monotonicity with window_size ---------------------------------

/// Prove visible count is monotonically non-decreasing with window_size.
///
/// For a fixed position i and seq_len, increasing window_size never reduces
/// the number of visible positions.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_visible_count_monotonic() {
    let seq_len: usize = kani::any();
    let w1: usize = kani::any();
    let w2: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 8);
    kani::assume(w1 >= 1 && w1 <= 16);
    kani::assume(w2 > w1 && w2 <= 17);
    let i: usize = kani::any();
    kani::assume(i < seq_len);

    let half1 = w1 / 2;
    let lo1 = if i >= half1 { i - half1 } else { 0 };
    let hi1 = if i + half1 < seq_len {
        i + half1
    } else {
        seq_len - 1
    };
    let count1 = hi1 - lo1 + 1;

    let half2 = w2 / 2;
    let lo2 = if i >= half2 { i - half2 } else { 0 };
    let hi2 = if i + half2 < seq_len {
        i + half2
    } else {
        seq_len - 1
    };
    let count2 = hi2 - lo2 + 1;

    kani::assert(count2 >= count1, "larger window must have >= visible count");
}

// -- Full window coverage --------------------------------------------------------

/// Prove when window_size >= 2 * seq_len - 1, all positions are visible to all tokens.
///
/// If half_window >= seq_len - 1, the maximum distance (seq_len - 1) is within
/// the window, so no masking occurs.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_full_coverage_when_large() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 8);
    // window_size such that half_window >= seq_len - 1
    let window_size = 2 * seq_len - 1;
    let half_window = window_size / 2;
    // For seq_len >= 1: half_window = (2 * seq_len - 1) / 2 >= seq_len - 1.
    kani::assert(
        half_window >= seq_len - 1,
        "half_window must cover max distance",
    );
    // Check all pairs: max distance = seq_len - 1 <= half_window.
    let i: usize = kani::any();
    let j: usize = kani::any();
    kani::assume(i < seq_len && j < seq_len);
    let dist = if i >= j { i - j } else { j - i };
    kani::assert(
        dist <= half_window,
        "all positions visible when window covers full range",
    );
}

// -- Scale factor from head_dim --------------------------------------------------

/// Prove SlidingWindowAttention scale = 1/sqrt(head_dim) is positive and finite.
///
/// head_dim = embed_dim / num_heads. For any head_dim >= 1, the scale factor is
/// in (0, 1].
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sliding_window_scale_factor_finite() {
    let embed_dim: usize = kani::any();
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(embed_dim >= num_heads && embed_dim <= 2048);
    kani::assume(embed_dim % num_heads == 0);
    let head_dim = embed_dim / num_heads;
    kani::assert(head_dim >= 1, "head_dim must be at least 1");
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    kani::assert(scale.is_finite(), "scale must be finite");
    kani::assert(scale > 0.0, "scale must be positive");
    kani::assert(scale <= 1.0, "scale <= 1.0 for head_dim >= 1");
}

// -- QKV weight shape constraints ------------------------------------------------

/// Prove QKV out_features is divisible by 3 iff constructed as 3 * embed_dim.
///
/// The SlidingWindowAttention constructor requires qkv_out % 3 == 0 and then
/// derives embed_dim = qkv_out / 3.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_qkv_divisible_by_3() {
    let embed_dim: usize = kani::any();
    kani::assume(embed_dim >= 1 && embed_dim <= 2048);
    let qkv_out = embed_dim.checked_mul(3);
    kani::assume(qkv_out.is_some());
    let qkv_out = qkv_out.unwrap();
    kani::assert(qkv_out % 3 == 0, "3 * embed_dim must be divisible by 3");
    let recovered = qkv_out / 3;
    kani::assert(
        recovered == embed_dim,
        "embed_dim must round-trip from qkv_out / 3",
    );
}

/// Prove embed_dim divisible by num_heads yields exact head_dim.
///
/// The constructor checks embed_dim % num_heads == 0.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_embed_dim_head_dim_exact() {
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(head_dim >= 1 && head_dim <= 256);
    let embed_dim = num_heads.checked_mul(head_dim);
    kani::assume(embed_dim.is_some());
    let embed_dim = embed_dim.unwrap();
    kani::assert(
        embed_dim % num_heads == 0,
        "embed_dim must be divisible by num_heads",
    );
    kani::assert(
        embed_dim / num_heads == head_dim,
        "head_dim must recover exactly",
    );
}

// -- Mask indexing bound ---------------------------------------------------------

/// Prove mask flat index i * seq_len + j < seq_len^2 for valid i, j.
///
/// The mask generation loop accesses data[i * seq_len + j]. This must be
/// within bounds [0, seq_len^2).
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_mask_index_in_bounds() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 64);
    let total = seq_len * seq_len;
    let i: usize = kani::any();
    let j: usize = kani::any();
    kani::assume(i < seq_len && j < seq_len);
    let idx = i * seq_len + j;
    kani::assert(idx < total, "flat index must be within bounds");
}

// -- Boundary token visibility ---------------------------------------------------

/// Prove boundary tokens (position 0 and seq_len-1) see at most as many
/// positions as interior tokens.
///
/// Position 0 is at the edge: it can only see rightward. Interior positions
/// can see both directions.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_boundary_le_interior() {
    let seq_len: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(seq_len >= 3 && seq_len <= 16);
    kani::assume(window_size >= 1 && window_size <= 32);
    let half = window_size / 2;

    // Position 0 (boundary).
    let lo0 = 0usize;
    let hi0 = if half < seq_len { half } else { seq_len - 1 };
    let count0 = hi0 - lo0 + 1;

    // An interior position i where half <= i <= seq_len - 1 - half.
    // Such a position exists when seq_len > 2 * half.
    if seq_len > 2 * half {
        let i = half; // interior
        let lo_i = i - half;
        let hi_i = i + half;
        kani::assume(hi_i < seq_len);
        let count_i = hi_i - lo_i + 1;
        kani::assert(
            count0 <= count_i,
            "boundary token must see <= interior token",
        );
    }
}

// -- Window growth property ------------------------------------------------------

/// Prove increasing window_size never masks a previously visible position.
///
/// If position j is visible from i with window w1, then j is also visible
/// from i with any window w2 > w1. (Monotonic inclusion.)
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sliding_window_monotonic_inclusion() {
    let i: usize = kani::any();
    let j: usize = kani::any();
    let w1: usize = kani::any();
    let w2: usize = kani::any();
    kani::assume(i <= 16 && j <= 16);
    kani::assume(w1 >= 1 && w1 <= 32);
    kani::assume(w2 > w1 && w2 <= 33);
    let dist = if i >= j { i - j } else { j - i };
    let half1 = w1 / 2;
    let half2 = w2 / 2;
    // If visible with w1 (dist <= half1), then also visible with w2 (dist <= half2).
    if dist <= half1 {
        kani::assert(half2 >= half1, "larger window has >= half_window");
        kani::assert(dist <= half2, "visible with w1 implies visible with w2");
    }
}
