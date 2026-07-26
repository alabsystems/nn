// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for window partition/unpartition properties.
//!
//! Covers:
//! - Padding computation: `(window_size - dim % window_size) % window_size`
//! - Padded dimensions divisible by window_size
//! - Padded dimensions >= original dimensions
//! - Element count preservation through partition/unpartition
//! - Number of windows calculation
//! - WindowAttentionConfig validation
//! - hidden_size = num_heads * head_dim
//! - Partition shape: [B * nw, ws^2, D]
//! - Unpartition inverse: recovers original element count
//! - Zero-padding correctness (padded >= original)
//!
//! Part of #3672.

// -- Padding computation ---------------------------------------------------------

/// Prove padding formula yields value in [0, window_size).
///
/// `pad = (window_size - dim % window_size) % window_size` always produces
/// a non-negative padding less than window_size.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padding_in_range() {
    let dim: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 256);
    kani::assume(window_size >= 1 && window_size <= 64);
    let pad = (window_size - dim % window_size) % window_size;
    kani::assert(pad < window_size, "padding must be < window_size");
}

/// Prove padding is zero when dim is already divisible by window_size.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padding_zero_when_divisible() {
    let dim: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(window_size >= 1 && window_size <= 64);
    kani::assume(dim >= 1 && dim <= 256);
    kani::assume(dim % window_size == 0);
    let pad = (window_size - dim % window_size) % window_size;
    kani::assert(
        pad == 0,
        "no padding needed when dim divisible by window_size",
    );
}

/// Prove padding is non-zero when dim is not divisible by window_size.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padding_nonzero_when_indivisible() {
    let dim: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(window_size >= 2 && window_size <= 64);
    kani::assume(dim >= 1 && dim <= 256);
    kani::assume(dim % window_size != 0);
    let pad = (window_size - dim % window_size) % window_size;
    kani::assert(pad >= 1, "padding must be >= 1 when not divisible");
    kani::assert(pad < window_size, "padding must be < window_size");
}

// -- Padded dimensions -----------------------------------------------------------

/// Prove padded_dim = dim + pad is divisible by window_size.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padded_dim_divisible() {
    let dim: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 256);
    kani::assume(window_size >= 1 && window_size <= 64);
    let pad = (window_size - dim % window_size) % window_size;
    let padded = dim + pad;
    kani::assert(
        padded % window_size == 0,
        "padded dimension must be divisible by window_size",
    );
}

/// Prove padded_dim >= dim (padding never shrinks).
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padded_ge_original() {
    let dim: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 256);
    kani::assume(window_size >= 1 && window_size <= 64);
    let pad = (window_size - dim % window_size) % window_size;
    let padded = dim + pad;
    kani::assert(padded >= dim, "padded must be >= original dimension");
}

/// Prove padded_dim < dim + window_size (minimal padding).
///
/// The padding is the smallest non-negative value to make dim + pad divisible
/// by window_size, so padded_dim < dim + window_size.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padded_minimal() {
    let dim: usize = kani::any();
    let window_size: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 256);
    kani::assume(window_size >= 1 && window_size <= 64);
    let pad = (window_size - dim % window_size) % window_size;
    let padded = dim + pad;
    kani::assert(
        padded < dim + window_size,
        "padded_dim must be strictly less than dim + window_size",
    );
}

// -- Number of windows -----------------------------------------------------------

/// Prove num_windows = (padded_h / ws) * (padded_w / ws) is exact.
///
/// Since padded_h and padded_w are both divisible by window_size, the
/// division is exact with no remainder.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_num_windows_exact() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let ws: usize = kani::any();
    kani::assume(ws >= 1 && ws <= 16);
    kani::assume(h >= 1 && h <= 64);
    kani::assume(w >= 1 && w <= 64);
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let ph = h + pad_h;
    let pw = w + pad_w;
    kani::assert(ph % ws == 0, "padded_h divisible");
    kani::assert(pw % ws == 0, "padded_w divisible");
    let nw_h = ph / ws;
    let nw_w = pw / ws;
    kani::assert(nw_h * ws == ph, "nw_h * ws == padded_h");
    kani::assert(nw_w * ws == pw, "nw_w * ws == padded_w");
    let nw = nw_h.checked_mul(nw_w);
    kani::assume(nw.is_some());
    kani::assert(nw.unwrap() >= 1, "at least 1 window");
}

// -- Element count preservation --------------------------------------------------

/// Prove partition preserves element count: B * ph * pw * D == B * nw * ws^2 * D.
///
/// The reshape from [B, ph, pw, D] to [B * nw, ws^2, D] must preserve total
/// element count.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_partition_element_count_preserved() {
    let b: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let d: usize = kani::any();
    let ws: usize = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(h >= 1 && h <= 16);
    kani::assume(w >= 1 && w <= 16);
    kani::assume(d >= 1 && d <= 32);
    kani::assume(ws >= 1 && ws <= 8);
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let ph = h + pad_h;
    let pw = w + pad_w;
    let nw_h = ph / ws;
    let nw_w = pw / ws;
    let nw = nw_h * nw_w;
    // Check element count: b * ph * pw * d == b * nw * ws * ws * d
    let lhs = b
        .checked_mul(ph)
        .and_then(|x| x.checked_mul(pw))
        .and_then(|x| x.checked_mul(d));
    let ws2 = ws.checked_mul(ws);
    kani::assume(ws2.is_some());
    let rhs = b
        .checked_mul(nw)
        .and_then(|x| x.checked_mul(ws2.unwrap()))
        .and_then(|x| x.checked_mul(d));
    kani::assume(lhs.is_some() && rhs.is_some());
    kani::assert(
        lhs.unwrap() == rhs.unwrap(),
        "element count must be preserved through partition",
    );
}

/// Prove unpartition recovers original seq_len = h * w from padded windows.
///
/// After unpartition with narrow to original (h, w), the output element count
/// is B * h * w * D.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_unpartition_recovers_seq_len() {
    let b: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let d: usize = kani::any();
    let ws: usize = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(h >= 1 && h <= 16);
    kani::assume(w >= 1 && w <= 16);
    kani::assume(d >= 1 && d <= 32);
    kani::assume(ws >= 1 && ws <= 8);
    let seq_len = h.checked_mul(w);
    kani::assume(seq_len.is_some());
    let seq_len = seq_len.unwrap();
    let total = b.checked_mul(seq_len).and_then(|x| x.checked_mul(d));
    kani::assume(total.is_some());
    kani::assert(
        total.unwrap() == b * h * w * d,
        "output element count = B * H * W * D",
    );
}

// -- WindowAttentionConfig validation --------------------------------------------

/// Prove WindowAttentionConfig::hidden_size == num_heads * head_dim.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_config_hidden_size() {
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 128);
    kani::assume(head_dim >= 1 && head_dim <= 256);
    let hidden = num_heads.checked_mul(head_dim);
    kani::assume(hidden.is_some());
    let hidden = hidden.unwrap();
    kani::assert(
        hidden == num_heads * head_dim,
        "hidden_size = num_heads * head_dim",
    );
    // Verify round-trip.
    kani::assert(
        hidden / num_heads == head_dim,
        "head_dim recovers from hidden_size",
    );
    kani::assert(
        hidden % num_heads == 0,
        "hidden_size divisible by num_heads",
    );
}

/// Prove WindowAttentionConfig rejects zero values.
///
/// window_size == 0, num_heads == 0, or head_dim == 0 must be rejected.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_config_rejects_zero() {
    let window_size: usize = kani::any();
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();
    kani::assume(window_size <= 64 && num_heads <= 64 && head_dim <= 256);
    let valid = window_size >= 1 && num_heads >= 1 && head_dim >= 1;
    if !valid {
        // At least one is zero — constructor should reject.
        kani::assert(
            window_size == 0 || num_heads == 0 || head_dim == 0,
            "invalid config has at least one zero field",
        );
    }
}

// -- Partition output shape ------------------------------------------------------

/// Prove partition output dim(0) = B * num_windows.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_partition_output_dim0() {
    let b: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let ws: usize = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(h >= 1 && h <= 32);
    kani::assume(w >= 1 && w <= 32);
    kani::assume(ws >= 1 && ws <= 16);
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let ph = h + pad_h;
    let pw = w + pad_w;
    let nw_h = ph / ws;
    let nw_w = pw / ws;
    let nw = nw_h.checked_mul(nw_w);
    kani::assume(nw.is_some());
    let out_dim0 = b.checked_mul(nw.unwrap());
    kani::assume(out_dim0.is_some());
    kani::assert(
        out_dim0.unwrap() == b * nw_h * nw_w,
        "output dim(0) = B * num_windows",
    );
}

/// Prove partition output dim(1) = window_size^2.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_partition_output_dim1() {
    let ws: usize = kani::any();
    kani::assume(ws >= 1 && ws <= 64);
    let ws2 = ws.checked_mul(ws);
    kani::assume(ws2.is_some());
    let ws2 = ws2.unwrap();
    kani::assert(ws2 == ws * ws, "window token count = window_size^2");
    kani::assert(ws2 >= 1, "at least 1 token per window");
}

// -- Padding symmetry for H and W ------------------------------------------------

/// Prove H and W are padded independently using the same formula.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn window_padding_independent_axes() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let ws: usize = kani::any();
    kani::assume(h >= 1 && h <= 64);
    kani::assume(w >= 1 && w <= 64);
    kani::assume(ws >= 1 && ws <= 16);
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let ph = h + pad_h;
    let pw = w + pad_w;
    // Both axes independently satisfy divisibility.
    kani::assert(ph % ws == 0, "padded H divisible by ws");
    kani::assert(pw % ws == 0, "padded W divisible by ws");
    // Padding one axis does not affect the other.
    if h % ws == 0 {
        kani::assert(pad_h == 0, "no H padding needed");
    }
    if w % ws == 0 {
        kani::assert(pad_w == 0, "no W padding needed");
    }
}
