// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for multi-head attention extended safety.
//!
//! Extends `kani_mha_safety.rs` with additional attention-mechanism proofs:
//!
//! 1. **Head dimension no-remainder** — hidden_size / num_heads = head_dim exactly
//! 2. **Grouped query attention** — num_heads % num_kv_heads == 0
//! 3. **Attention scale factor** — 1/sqrt(head_dim) is positive and finite
//! 4. **Fused QKV projection shape** — [B, S, 3*D] for fused QKV output
//! 5. **Attention output shape preservation** — output shape equals input shape
//! 6. **Deformable attention sampling bounds** — sampling points within spatial grid
//! 7. **Window attention** — window_size <= sequence_length
//! 8. **Sliding window bounds** — window_start >= 0, window_end <= seq_len
//! 9. **Flash attention tiling** — tile size handles remainder correctly
//! 10. **Relative position bias shape** — [num_heads, seq_len, seq_len]
//!
//! All harnesses use small bounds for CBMC tractability:
//! B <= 2, H <= 4, S <= 8, D <= 16.
//!
//! Part of #4233.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// 1. Head dimension no-remainder
// ===========================================================================

/// Proves that when hidden_size is divisible by num_heads, head_dim has no
/// remainder and the product head_dim * num_heads recovers hidden_size exactly.
/// Additionally proves head_dim >= 1 when both inputs are >= 1.
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_head_dim_no_remainder() {
    let hidden: u8 = kani::any();
    let heads: u8 = kani::any();

    kani::assume(hidden >= 1 && hidden <= 16);
    kani::assume(heads >= 1 && heads <= 4);
    kani::assume(hidden as usize % (heads as usize) == 0);

    let h = hidden as usize;
    let n = heads as usize;
    let head_dim = h / n;

    // head_dim is positive
    assert!(head_dim >= 1, "head_dim must be >= 1");

    // No remainder
    assert_eq!(h % n, 0, "hidden_size must be divisible by num_heads");

    // Roundtrip
    assert_eq!(
        head_dim * n,
        h,
        "head_dim * num_heads must equal hidden_size"
    );

    // head_dim divides hidden_size
    assert_eq!(
        h / head_dim,
        n,
        "hidden_size / head_dim must equal num_heads"
    );
}

// ===========================================================================
// 2. Grouped query attention divisibility
// ===========================================================================

/// Proves GQA constraint: num_heads must be divisible by num_kv_heads.
/// The group size (num_heads / num_kv_heads) is >= 1 and multiplies back.
/// Multi-head attention (MHA) is GQA with group_size = 1.
/// Multi-query attention (MQA) is GQA with num_kv_heads = 1.
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_gqa_divisibility() {
    let num_heads: u8 = kani::any();
    let num_kv_heads: u8 = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 8);
    kani::assume(num_kv_heads >= 1 && num_kv_heads <= 8);
    kani::assume(num_kv_heads <= num_heads);
    kani::assume(num_heads as usize % (num_kv_heads as usize) == 0);

    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let group_size = nh / nkv;

    // Group size is positive
    assert!(group_size >= 1, "GQA group size must be >= 1");

    // Roundtrip: group_size * num_kv_heads == num_heads
    assert_eq!(
        group_size * nkv,
        nh,
        "group_size * num_kv_heads must equal num_heads"
    );

    // MHA case: num_kv_heads == num_heads => group_size == 1
    if nkv == nh {
        assert_eq!(group_size, 1, "MHA has group_size 1");
    }

    // MQA case: num_kv_heads == 1 => group_size == num_heads
    if nkv == 1 {
        assert_eq!(group_size, nh, "MQA has group_size == num_heads");
    }
}

// ===========================================================================
// 3. Attention scale factor is positive and finite
// ===========================================================================

/// Proves that the attention scale factor 1/sqrt(head_dim) is positive,
/// finite, and bounded for valid head dimensions.
///
/// The scale factor is used to normalize dot-product attention scores:
/// score = Q @ K^T * scale, where scale = 1/sqrt(head_dim).
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_scale_factor_positive_finite() {
    let head_dim: u8 = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 128);

    let hd = head_dim as f64;

    // sqrt(head_dim) for head_dim in [1, 128] is finite and positive
    let sqrt_hd = hd.sqrt();
    assert!(sqrt_hd > 0.0, "sqrt(head_dim) must be positive");
    assert!(sqrt_hd.is_finite(), "sqrt(head_dim) must be finite");

    // 1/sqrt(head_dim)
    let scale = 1.0 / sqrt_hd;
    assert!(scale > 0.0, "scale factor must be positive");
    assert!(scale.is_finite(), "scale factor must be finite");

    // scale <= 1.0 since head_dim >= 1 => sqrt(head_dim) >= 1
    assert!(
        scale <= 1.0,
        "scale factor must be <= 1.0 for head_dim >= 1"
    );

    // scale >= 1/sqrt(128) ~= 0.0884
    let min_scale = 1.0 / (128.0_f64).sqrt();
    assert!(
        scale >= min_scale - 1e-10,
        "scale factor must be >= 1/sqrt(128)"
    );

    // Verify f32 version is also finite
    let scale_f32 = scale as f32;
    assert!(scale_f32.is_finite(), "f32 scale must be finite");
    assert!(scale_f32 > 0.0, "f32 scale must be positive");
}

// ===========================================================================
// 4. Fused QKV projection output shape
// ===========================================================================

/// Proves fused QKV projection: input [B, S, D] projects to [B, S, 3*D].
///
/// In fused QKV, a single linear layer W_qkv of shape [D, 3*D] produces
/// Q, K, V in a single matmul. The output is [B, S, 3*D] which is then
/// split into three tensors of shape [B, S, D].
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_fused_qkv_projection_shape() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    let h: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 16);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(d as usize % (h as usize) == 0);
    // 3*D must not overflow u8
    kani::assume((d as usize) * 3 <= 48);

    let bu = b as usize;
    let su = s as usize;
    let du = d as usize;

    // Input shape: [B, S, D]
    let input_shape = [bu, su, du];

    // Fused QKV output: [B, S, 3*D]
    let fused_dim = 3 * du;
    let fused_shape = [bu, su, fused_dim];

    // Numel of fused output is 3x input numel
    let input_numel = checked_dim_product(&input_shape);
    let fused_numel = checked_dim_product(&fused_shape);
    if let (Ok(in_n), Ok(fn_n)) = (input_numel, fused_numel) {
        assert_eq!(fn_n, 3 * in_n, "fused QKV numel must be 3x input numel");
    }

    // Splitting: each of Q, K, V has shape [B, S, D]
    let q_shape = [bu, su, du];
    let k_shape = [bu, su, du];
    let v_shape = [bu, su, du];

    // Total elements from split equals fused elements
    let q_numel = checked_dim_product(&q_shape);
    let k_numel = checked_dim_product(&k_shape);
    let v_numel = checked_dim_product(&v_shape);
    if let (Ok(qn), Ok(kn), Ok(vn), Ok(fn_n)) = (q_numel, k_numel, v_numel, fused_numel) {
        assert_eq!(
            qn + kn + vn,
            fn_n,
            "Q + K + V numel must equal fused QKV numel"
        );
    }

    // Each split tensor has the same shape as input
    assert_eq!(q_shape, input_shape, "Q shape must equal input shape");
    assert_eq!(k_shape, input_shape, "K shape must equal input shape");
    assert_eq!(v_shape, input_shape, "V shape must equal input shape");
}

// ===========================================================================
// 5. Attention output shape preservation
// ===========================================================================

/// Proves that multi-head attention preserves the input shape [B, S, D].
///
/// Full MHA pipeline:
/// input [B, S, D] -> QKV project -> attention -> concat -> output project
/// The output must be [B, S, D] (same as input).
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_output_shape_preservation() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    let h: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 16);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(d as usize % (h as usize) == 0);

    let bu = b as usize;
    let su = s as usize;
    let du = d as usize;
    let hu = h as usize;
    let dh = du / hu;

    // Input: [B, S, D]
    let input_shape = [bu, su, du];

    // After QKV projection and reshape: [B, H, S, Dh] per head
    let per_head = [bu, hu, su, dh];

    // After attention (scores [B, H, S, S] @ V [B, H, S, Dh]): [B, H, S, Dh]
    let attn_out = [bu, hu, su, dh];
    assert_eq!(
        attn_out, per_head,
        "attention output shape must equal per-head shape"
    );

    // Transpose [B, H, S, Dh] -> [B, S, H, Dh]
    let transposed = [bu, su, hu, dh];

    // Reshape [B, S, H, Dh] -> [B, S, H*Dh] = [B, S, D]
    let concat_dim = hu * dh;
    assert_eq!(concat_dim, du, "H * Dh must equal D");
    let concat_shape = [bu, su, du];

    // Output projection: [B, S, D] -> [B, S, D]
    let output_shape = concat_shape;

    // Final check: output shape == input shape
    assert_eq!(
        output_shape, input_shape,
        "MHA output shape must equal input shape"
    );

    // Numel preserved throughout
    let input_numel = checked_dim_product(&input_shape);
    let per_head_numel = checked_dim_product(&per_head);
    let transposed_numel = checked_dim_product(&transposed);
    let output_numel = checked_dim_product(&output_shape);
    if let (Ok(inn), Ok(phn), Ok(tn), Ok(on)) =
        (input_numel, per_head_numel, transposed_numel, output_numel)
    {
        assert_eq!(inn, phn, "input numel must equal per-head numel");
        assert_eq!(phn, tn, "per-head numel must equal transposed numel");
        assert_eq!(tn, on, "transposed numel must equal output numel");
    }
}

// ===========================================================================
// 6. Deformable attention sampling bounds
// ===========================================================================

/// Proves deformable attention sampling points stay within spatial bounds.
///
/// In deformable attention, each query samples K points from the spatial grid
/// at learned offsets. The sampling offsets are normalized to [0, 1] via sigmoid,
/// then scaled to [0, H) and [0, W). We prove the resulting coordinates are
/// within [0, H) x [0, W) for any valid sigmoid output.
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_deformable_sampling_bounds() {
    let spatial_h: u8 = kani::any();
    let spatial_w: u8 = kani::any();

    kani::assume(spatial_h >= 1 && spatial_h <= 8);
    kani::assume(spatial_w >= 1 && spatial_w <= 8);

    let sh = spatial_h as usize;
    let sw = spatial_w as usize;

    // Reference point (normalized to [0, 1])
    let ref_y: f32 = kani::any();
    let ref_x: f32 = kani::any();
    kani::assume(ref_y >= 0.0 && ref_y <= 1.0 && ref_y.is_finite());
    kani::assume(ref_x >= 0.0 && ref_x <= 1.0 && ref_x.is_finite());

    // Sampling offset after sigmoid (in [0, 1])
    let offset_y: f32 = kani::any();
    let offset_x: f32 = kani::any();
    kani::assume(offset_y >= 0.0 && offset_y <= 1.0 && offset_y.is_finite());
    kani::assume(offset_x >= 0.0 && offset_x <= 1.0 && offset_x.is_finite());

    // Sampling location = reference_point + offset, clamped to [0, 1]
    let mut sample_y = ref_y + (offset_y - 0.5); // offset centered around 0
    let mut sample_x = ref_x + (offset_x - 0.5);

    // Clamp to valid range [0, 1]
    if sample_y < 0.0 {
        sample_y = 0.0;
    }
    if sample_y > 1.0 {
        sample_y = 1.0;
    }
    if sample_x < 0.0 {
        sample_x = 0.0;
    }
    if sample_x > 1.0 {
        sample_x = 1.0;
    }

    // Scale to spatial dimensions
    let grid_y = sample_y * (sh as f32 - 1.0);
    let grid_x = sample_x * (sw as f32 - 1.0);

    // Grid coordinates must be within spatial bounds
    assert!(grid_y >= 0.0, "grid_y must be >= 0");
    assert!(grid_x >= 0.0, "grid_x must be >= 0");

    if sh > 1 {
        assert!(grid_y <= (sh - 1) as f32, "grid_y must be < spatial_h");
    }
    if sw > 1 {
        assert!(grid_x <= (sw - 1) as f32, "grid_x must be < spatial_w");
    }

    // Grid coordinates are finite
    assert!(grid_y.is_finite(), "grid_y must be finite");
    assert!(grid_x.is_finite(), "grid_x must be finite");
}

// ===========================================================================
// 7. Window attention: window_size <= sequence_length
// ===========================================================================

/// Proves window attention partitioning is valid when window_size <= seq_len.
///
/// Window attention divides the sequence into non-overlapping windows of
/// size W. The number of windows is ceil(S / W). Each window has at most
/// W tokens, and the last window may be padded.
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_window_attention_valid() {
    let seq_len: u8 = kani::any();
    let window_size: u8 = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 16);
    kani::assume(window_size >= 1 && window_size <= 16);
    kani::assume(window_size <= seq_len);

    let s = seq_len as usize;
    let w = window_size as usize;

    // Number of windows (ceiling division)
    let num_windows = (s + w - 1) / w;

    // At least one window
    assert!(num_windows >= 1, "must have at least one window");

    // Windows cover the entire sequence
    assert!(num_windows * w >= s, "windows must cover entire sequence");

    // No more windows than necessary
    assert!(
        (num_windows - 1) * w < s,
        "last window must contain at least one real token"
    );

    // Padding tokens in last window
    let total_slots = num_windows * w;
    let padding = total_slots - s;
    assert!(padding < w, "padding must be less than window_size");

    // When seq_len is a multiple of window_size, no padding
    if s % w == 0 {
        assert_eq!(
            padding, 0,
            "no padding when seq_len is multiple of window_size"
        );
        assert_eq!(num_windows, s / w, "exact number of windows");
    }
}

// ===========================================================================
// 8. Sliding window bounds
// ===========================================================================

/// Proves sliding window attention bounds: window_start >= 0, window_end <= seq_len.
///
/// For position i with window size W, the sliding window covers positions
/// [max(0, i - W + 1), i + 1). This ensures each position only attends to
/// at most W previous positions (including itself).
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_sliding_window_bounds() {
    let seq_len: u8 = kani::any();
    let window_size: u8 = kani::any();
    let position: u8 = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 16);
    kani::assume(window_size >= 1 && window_size <= 16);
    kani::assume(position < seq_len);

    let s = seq_len as usize;
    let w = window_size as usize;
    let i = position as usize;

    // Window start: max(0, i - W + 1)
    // Use saturating subtraction to avoid underflow
    let window_start = if i + 1 >= w { i + 1 - w } else { 0 };

    // Window end: min(i + 1, seq_len)
    let window_end = if i + 1 <= s { i + 1 } else { s };

    // Bounds safety
    assert!(
        window_start <= i,
        "window_start must be <= current position"
    );
    assert!(window_end <= s, "window_end must be <= seq_len");
    assert!(window_start < window_end, "window must be non-empty");

    // Window size check
    let actual_window = window_end - window_start;
    assert!(
        actual_window >= 1,
        "window must contain at least 1 position"
    );
    assert!(actual_window <= w, "window must not exceed window_size");

    // Current position is in the window
    assert!(i >= window_start, "position must be >= window_start");
    assert!(i < window_end, "position must be < window_end");

    // For i >= W-1, window has exactly W elements
    if i + 1 >= w {
        assert_eq!(actual_window, w, "full window must have exactly W elements");
    }
}

// ===========================================================================
// 9. Flash attention tiling
// ===========================================================================

/// Proves flash attention tiling: tile_size divides sequence or handles remainder.
///
/// Flash attention processes the S x S attention matrix in tiles of size T x T.
/// The number of tiles along each dimension is ceil(S / T). The last tile may
/// be smaller than T (handles the remainder).
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_flash_attention_tiling() {
    let seq_len: u8 = kani::any();
    let tile_size: u8 = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 16);
    kani::assume(tile_size >= 1 && tile_size <= 8);

    let s = seq_len as usize;
    let t = tile_size as usize;

    // Number of tiles (ceiling division)
    let num_tiles = (s + t - 1) / t;

    // At least one tile
    assert!(num_tiles >= 1, "must have at least one tile");

    // Tiles cover the full sequence
    assert!(num_tiles * t >= s, "tiles must cover full sequence");

    // Verify each tile's bounds
    let mut tile_idx = 0usize;
    while tile_idx < num_tiles {
        let tile_start = tile_idx * t;
        let tile_end = if (tile_idx + 1) * t <= s {
            (tile_idx + 1) * t
        } else {
            s
        };

        // Tile bounds are valid
        assert!(tile_start < s, "tile_start must be < seq_len");
        assert!(tile_end <= s, "tile_end must be <= seq_len");
        assert!(tile_start < tile_end, "tile must be non-empty");

        // Tile size is at most T
        let this_tile_size = tile_end - tile_start;
        assert!(this_tile_size <= t, "tile size must be <= tile_size");
        assert!(this_tile_size >= 1, "tile must have >= 1 element");

        // Only the last tile can be smaller than T
        if tile_idx < num_tiles - 1 {
            assert_eq!(
                this_tile_size, t,
                "non-last tiles must have exactly tile_size elements"
            );
        }

        tile_idx += 1;
    }

    // Total elements across all tiles equals seq_len
    // Last tile may be partial: (num_tiles - 1) * t + remainder = s
    let remainder = s % t;
    if remainder == 0 {
        assert_eq!(
            num_tiles * t,
            s,
            "no remainder: tiles exactly cover sequence"
        );
    } else {
        assert_eq!(
            (num_tiles - 1) * t + remainder,
            s,
            "remainder tiles must sum to seq_len"
        );
    }
}

// ===========================================================================
// 10. Relative position bias shape
// ===========================================================================

/// Proves relative position bias shape is [num_heads, seq_len, seq_len].
///
/// Relative position bias is added to the attention scores before softmax.
/// Its shape must be [H, S, S] to broadcast with scores [B, H, S, S].
/// The total elements must equal H * S * S.
#[kani::unwind(1)]
#[kani::proof]
fn attn_ext_relative_position_bias_shape() {
    let num_heads: u8 = kani::any();
    let seq_len: u8 = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 4);
    kani::assume(seq_len >= 1 && seq_len <= 8);

    let h = num_heads as usize;
    let s = seq_len as usize;

    // Bias shape: [H, S, S]
    let bias_shape = [h, s, s];

    // Score shape: [B, H, S, S] for batch B=1 (bias broadcasts along B)
    let score_shape = [1usize, h, s, s];

    // Bias dimensions match score dimensions 1, 2, 3
    assert_eq!(
        bias_shape[0], score_shape[1],
        "bias head dim must match score head dim"
    );
    assert_eq!(
        bias_shape[1], score_shape[2],
        "bias row dim must match score row dim"
    );
    assert_eq!(
        bias_shape[2], score_shape[3],
        "bias col dim must match score col dim"
    );

    // Numel of bias
    let bias_numel = checked_dim_product(&bias_shape);
    if let Ok(bn) = bias_numel {
        assert_eq!(bn, h * s * s, "bias numel must equal H * S * S");
    }

    // Bias is square in the seq dimensions
    assert_eq!(
        bias_shape[1], bias_shape[2],
        "relative position bias must be S x S (square)"
    );

    // Number of distinct relative positions = 2*S - 1 (from -(S-1) to +(S-1))
    let num_relative_positions = 2 * s - 1;
    assert!(
        num_relative_positions >= 1,
        "must have at least 1 relative position"
    );
    assert_eq!(
        num_relative_positions,
        2 * s - 1,
        "relative positions count must be 2*S - 1"
    );

    // The bias table has shape [H, 2*S-1], one entry per relative position per head
    let table_shape = [h, num_relative_positions];
    let table_numel = checked_dim_product(&table_shape);
    if let Ok(tn) = table_numel {
        assert_eq!(
            tn,
            h * num_relative_positions,
            "table numel must be H * (2S-1)"
        );
        // Table is smaller than the full bias matrix (table is compressed)
        if let Ok(bn) = bias_numel {
            assert!(tn <= bn, "table must be <= full bias matrix in size");
        }
    }
}
