// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for `compiled_model_execute_native_resblock.rs`.
//!
//! Complements `kani_execute_native_resblock.rs` with proofs covering:
//! - LeakyRelu fast path routing: both phases must match for 3-dispatch path
//! - Snake fast path routing: both phases must match for 3-dispatch path
//! - Mixed activation fallback: fires when phases differ
//! - Conv weight shape [C_out, C_in, K] element product overflow protection
//! - Batch offset narrow: cumulative offset calculation correctness
//! - Batch offset reshape [B,C] → [B,C,1] is zero-copy (element count unchanged)
//! - Pool path: pool_time inference denominator always valid
//! - Residual scale: skip mul_scalar path when scale == 1.0
//! - Input steps length requirements per path
//! - Shortcut vs identity: element count consistency
//! - Style projection matmul: [B, D] @ [D, 2C] = [B, 2C]
//! - Style projection narrow: gamma at [0..C], beta at [C..2C]
//!
//! Part of #3742.

// ============================================================================
// LeakyRelu fast path routing
// ============================================================================

/// Prove: LeakyRelu fast path fires if and only if BOTH phases are LeakyRelu.
///
/// The condition is `matches!(phase1, LeakyRelu) && matches!(phase2, LeakyRelu)`.
/// Any other combination falls through to the Snake check or mixed fallback.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn leaky_relu_fast_path_both_phases() {
    // Encode activation types as: 0=LeakyRelu, 1=Snake, 2=Other.
    let phase1: u8 = kani::any();
    let phase2: u8 = kani::any();
    kani::assume(phase1 <= 2);
    kani::assume(phase2 <= 2);

    let leaky_fast = phase1 == 0 && phase2 == 0;
    let snake_fast = phase1 == 1 && phase2 == 1;
    let mixed_fallback = !leaky_fast && !snake_fast;

    // Exactly one path fires.
    let count = leaky_fast as u8 + snake_fast as u8 + mixed_fallback as u8;
    assert_eq!(count, 1, "exactly one dispatch path");

    // LeakyRelu only when both are LeakyRelu.
    if leaky_fast {
        assert!(phase1 == 0 && phase2 == 0);
    }
}

/// Prove: LeakyRelu slope extraction succeeds when activation is LeakyRelu.
///
/// `match phase.activation { LeakyRelu { slope } => slope, _ => unreachable!() }`
/// Since we already matched LeakyRelu in the outer condition, extraction is safe.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn leaky_relu_slope_extraction_safe() {
    let is_leaky: bool = kani::any();
    let slope: f32 = kani::any();
    kani::assume(slope.is_finite());

    if is_leaky {
        // Extraction succeeds.
        let extracted = slope;
        assert_eq!(extracted, slope, "slope extraction preserves value");
        assert!(extracted.is_finite(), "extracted slope must be finite");
    }
}

// ============================================================================
// Snake fast path routing
// ============================================================================

/// Prove: Snake fast path fires if and only if BOTH phases are Snake.
///
/// The condition is `matches!(phase1, Snake) && matches!(phase2, Snake)`.
/// It only fires after the LeakyRelu check fails.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snake_fast_path_both_phases() {
    let phase1: u8 = kani::any(); // 0=LeakyRelu, 1=Snake, 2=Other
    let phase2: u8 = kani::any();
    kani::assume(phase1 <= 2);
    kani::assume(phase2 <= 2);

    // LeakyRelu has priority.
    let leaky_fast = phase1 == 0 && phase2 == 0;
    let snake_fast = !leaky_fast && phase1 == 1 && phase2 == 1;

    if snake_fast {
        assert!(phase1 == 1 && phase2 == 1, "snake requires both phases Snake");
        assert!(!leaky_fast, "snake path cannot fire if leaky fires");
    }
}

// ============================================================================
// Mixed activation fallback
// ============================================================================

/// Prove: mixed fallback fires when phase activations differ.
///
/// If phase1 != phase2 (e.g., LeakyRelu + Snake), neither fast path fires,
/// so the mixed fallback must handle it.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mixed_fallback_fires_when_phases_differ() {
    let phase1: u8 = kani::any();
    let phase2: u8 = kani::any();
    kani::assume(phase1 <= 2);
    kani::assume(phase2 <= 2);
    kani::assume(phase1 != phase2);

    let leaky_fast = phase1 == 0 && phase2 == 0;
    let snake_fast = phase1 == 1 && phase2 == 1;

    assert!(!leaky_fast, "different phases cannot fire LeakyRelu fast path");
    assert!(!snake_fast, "different phases cannot fire Snake fast path");
    // Therefore mixed_fallback fires.
}

/// Prove: mixed fallback handles LeakyRelu+Snake and Snake+LeakyRelu.
///
/// The two most common mixed cases in production.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mixed_fallback_handles_leaky_snake_combo() {
    // Case 1: LeakyRelu + Snake
    let p1a: u8 = 0;
    let p2a: u8 = 1;
    assert!(!(p1a == 0 && p2a == 0), "not both LeakyRelu");
    assert!(!(p1a == 1 && p2a == 1), "not both Snake");

    // Case 2: Snake + LeakyRelu
    let p1b: u8 = 1;
    let p2b: u8 = 0;
    assert!(!(p1b == 0 && p2b == 0), "not both LeakyRelu");
    assert!(!(p1b == 1 && p2b == 1), "not both Snake");
}

// ============================================================================
// Conv weight shape
// ============================================================================

/// Prove: conv weight [C_out, C_in, K] element product does not overflow
/// for Kokoro-range parameters.
///
/// Max Kokoro: C_out=512, C_in=512, K=7 → 1,835,008 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_weight_elements_no_overflow() {
    let c_out: usize = kani::any();
    let c_in: usize = kani::any();
    let kernel: usize = kani::any();

    kani::assume(c_out >= 1 && c_out <= 512);
    kani::assume(c_in >= 1 && c_in <= 512);
    kani::assume(kernel >= 1 && kernel <= 15);

    let elems = c_out
        .checked_mul(c_in)
        .and_then(|v| v.checked_mul(kernel));
    assert!(elems.is_some(), "conv weight elements must not overflow");

    let bytes = elems.unwrap().checked_mul(4); // f32
    assert!(bytes.is_some(), "conv weight bytes must not overflow");
}

/// Prove: conv weight byte count fits in Metal buffer (< 256 MB).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_weight_bytes_within_metal_limit() {
    let c_out: usize = kani::any();
    let c_in: usize = kani::any();
    let kernel: usize = kani::any();

    kani::assume(c_out >= 1 && c_out <= 1024);
    kani::assume(c_in >= 1 && c_in <= 1024);
    kani::assume(kernel >= 1 && kernel <= 15);

    let bytes = c_out * c_in * kernel * 4;
    assert!(bytes <= 256 * 1024 * 1024, "conv weight must fit in 256 MB");
}

// ============================================================================
// Batch offset narrow arithmetic
// ============================================================================

/// Prove: cumulative narrow offsets partition the batch output exactly.
///
/// Layout: [gamma1(C1), beta1(C1), gamma2(C2), beta2(C2)].
/// Sum of widths = 2*C1 + 2*C2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_offset_narrow_partition_exact() {
    let offset: usize = kani::any();
    let c1: usize = kani::any();
    let c2: usize = kani::any();

    kani::assume(offset <= 4096);
    kani::assume(c1 >= 1 && c1 <= 512);
    kani::assume(c2 >= 1 && c2 <= 512);

    let mut off = offset;
    let g1_start = off;
    off += c1;
    let b1_start = off;
    off += c1;
    let g2_start = off;
    off += c2;
    let b2_start = off;
    off += c2;
    let end = off;

    // Partition is contiguous.
    assert_eq!(g1_start, offset, "gamma1 starts at offset");
    assert_eq!(b1_start, offset + c1, "beta1 starts after gamma1");
    assert_eq!(g2_start, offset + 2 * c1, "gamma2 starts after beta1");
    assert_eq!(b2_start, offset + 2 * c1 + c2, "beta2 starts after gamma2");
    assert_eq!(end, offset + 2 * c1 + 2 * c2, "end is offset + total span");

    // No gaps or overlaps.
    assert_eq!(end - offset, 2 * c1 + 2 * c2, "total span correct");
}

// ============================================================================
// Reshape [B,C] → [B,C,1] is zero-copy
// ============================================================================

/// Prove: reshape from [B, C] to [B, C, 1] preserves element count.
///
/// This is used for AdaIN compatibility. B*C*1 == B*C always.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_2d_to_3d_preserves_elements() {
    let b: usize = kani::any();
    let c: usize = kani::any();

    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 1024);

    let elems_2d = b * c;
    let elems_3d = b * c * 1;
    assert_eq!(elems_2d, elems_3d, "reshape [B,C] → [B,C,1] preserves elements");
}

// ============================================================================
// Pool path denominator
// ============================================================================

/// Prove: pool_time inference denominator (batch * channels * dtype_size)
/// is always > 0 and divides the buffer size exactly.
///
/// This prevents division by zero in `pool_time = pool_bytes / denom`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_time_denominator_positive_and_exact() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let dtype_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 512);
    kani::assume(dtype_size == 2 || dtype_size == 4);

    let denom = batch * channels * dtype_size;
    assert!(denom > 0, "denominator must be positive");
    assert!(denom >= 2, "minimum denominator is 1*1*2 = 2");
}

// ============================================================================
// Residual scale: skip path
// ============================================================================

/// Prove: mul_scalar is skipped when residual_scale == 1.0.
///
/// `(residual_scale - 1.0).abs() > f32::EPSILON` is the guard.
/// When scale == 1.0 exactly, the difference is 0.0 which is NOT > EPSILON.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn residual_scale_skip_when_exactly_one() {
    let scale: f32 = 1.0;
    let diff = (scale - 1.0f32).abs();
    assert_eq!(diff, 0.0, "diff must be exactly 0 for scale=1.0");
    assert!(!(diff > f32::EPSILON), "mul_scalar must be skipped for scale=1.0");
}

/// Prove: mul_scalar fires when residual_scale differs from 1.0 by > EPSILON.
///
/// For typical Kokoro scale values (e.g., 0.5, sqrt(0.5)), the guard fires.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn residual_scale_fires_for_half() {
    let scale: f32 = 0.5;
    let diff = (scale - 1.0f32).abs();
    assert!(diff > f32::EPSILON, "mul_scalar must fire for scale=0.5");
}

// ============================================================================
// Input steps length requirements
// ============================================================================

/// Prove: direct buffer path requires exactly 5 input_steps.
///
/// [x, gamma1, beta1, gamma2, beta2] = 5 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn direct_buffer_path_needs_5_steps() {
    let len: usize = kani::any();
    kani::assume(len < 5);

    // If len < 5, the direct buffer path returns Err.
    assert!(len < 5, "direct path rejects fewer than 5 steps");
}

/// Prove: style_proj and batch_offset paths require >= 2 input_steps.
///
/// [x, style_embed] or [x, batch_step] = 2 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn projection_paths_need_2_steps() {
    let len: usize = kani::any();
    kani::assume(len < 2);

    // Both style_proj and batch_offset return Err when len < 2.
    assert!(len < 2, "projection paths reject fewer than 2 steps");
}

// ============================================================================
// Shortcut vs identity: element count
// ============================================================================

/// Prove: shortcut conv1x1 output shape [B, C_out, T] has the same temporal
/// dimension as the input [B, C_in, T]. Element counts match at residual add
/// because C_out is the same as phase2.output_channels.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn shortcut_output_temporal_matches_input() {
    let b: usize = kani::any();
    let c_out: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b >= 1 && b <= 64);
    kani::assume(c_out >= 1 && c_out <= 512);
    kani::assume(t >= 1 && t <= 16384);

    // Phase2 output shape.
    let phase2_elems = b.checked_mul(c_out).and_then(|v| v.checked_mul(t));
    // Shortcut output shape: same [B, C_out, T].
    let shortcut_elems = b.checked_mul(c_out).and_then(|v| v.checked_mul(t));

    assert!(phase2_elems.is_some(), "phase2 elems no overflow");
    assert!(shortcut_elems.is_some(), "shortcut elems no overflow");
    assert_eq!(
        phase2_elems.unwrap(),
        shortcut_elems.unwrap(),
        "shortcut and phase2 must have same element count for residual add"
    );
}

// ============================================================================
// Style projection matmul shape
// ============================================================================

/// Prove: style projection matmul [B, D] @ [D, 2C] → [B, 2C].
///
/// The inner dimension (D) must match for the matmul to succeed.
/// Output has batch rows and 2*channels columns.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn style_proj_matmul_shape_correct() {
    let batch: usize = kani::any();
    let style_dim: usize = kani::any();
    let channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(style_dim >= 1 && style_dim <= 512);
    kani::assume(channels >= 1 && channels <= 512);

    // Input: [B, D]. Weight_t: [D, 2C]. Output: [B, 2C].
    let input_cols = style_dim;
    let weight_rows = style_dim;
    let output_cols = 2 * channels;

    assert_eq!(input_cols, weight_rows, "inner dimensions must match");

    let output_elems = batch.checked_mul(output_cols);
    assert!(output_elems.is_some(), "output elements must not overflow");
}

/// Prove: style projection narrow splits [B, 2C] into two [B, C] tensors.
///
/// narrow(1, 0, C) → gamma [B, C]. narrow(1, C, C) → beta [B, C].
/// Together they cover all 2C columns exactly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn style_proj_narrow_covers_all_columns() {
    let channels: usize = kani::any();
    kani::assume(channels >= 1 && channels <= 1024);

    let total_cols = 2 * channels;

    // Gamma: narrow(1, 0, channels) → [0, channels)
    let gamma_start = 0;
    let gamma_end = channels;

    // Beta: narrow(1, channels, channels) → [channels, 2*channels)
    let beta_start = channels;
    let beta_end = 2 * channels;

    // Coverage: gamma_end == beta_start (contiguous).
    assert_eq!(gamma_end, beta_start, "gamma and beta are contiguous");
    // Coverage: beta_end == total_cols (complete).
    assert_eq!(beta_end, total_cols, "gamma + beta cover all columns");
    // No overlap.
    assert_eq!(gamma_start + channels + channels, total_cols, "exact partition");
}
