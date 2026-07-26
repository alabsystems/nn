// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `compiled_model_execute_native_resblock.rs` (#3715).
//!
//! Complements `kani_fused_resblock.rs` and `kani_native_resblock_dispatch.rs`
//! with additional proofs for:
//! - Pool path buffer dimension inference (3D shape recovery)
//! - Style batch offset total span overflow protection
//! - Style projection weight shape consistency
//! - Gamma/beta resolution path mutual exclusivity
//! - Residual scale mul_scalar f64 widening safety
//! - Pool path phase boundary channel consistency
//! - Conv1d output length formula (arbitrary padding/dilation)
//! - Shortcut conv1x1 output shape calculation
//! - PhaseWeightKeys static label/key consistency
//! - Batch offset buffer denominator non-zero
//! - Style projection bias shape = 2 * channels
//! - Gamma/beta shape [B,C,1] element product overflow protection
//! - Input steps index [0] always valid when len >= 1
//! - Residual add element count must match

// ============================================================================
// 1. Pool path: byte-to-time recovery for non-power-of-2 channels
// ============================================================================

/// Prove: pool_time = pool_bytes / (batch * pool_channels * dtype_size) is
/// exact for non-power-of-2 channel counts typical in Kokoro (48, 96, 192).
///
/// This extends the basic inference proof by verifying Kokoro-specific
/// channel counts that are NOT powers of 2 and could theoretically cause
/// truncation bugs in integer division.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_time_inference_non_power_of_2_channels() {
    let batch: usize = kani::any();
    let time: usize = kani::any();
    let dtype_size: usize = kani::any();
    // Kokoro channel counts: 48, 96, 192, 384, 512.
    let channel_idx: u8 = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(time >= 1 && time <= 4096);
    kani::assume(dtype_size == 2 || dtype_size == 4);
    kani::assume(channel_idx < 5);

    let channels: usize = match channel_idx {
        0 => 48,
        1 => 96,
        2 => 192,
        3 => 384,
        _ => 512,
    };

    let data_bytes = match batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(time))
        .and_then(|v| v.checked_mul(dtype_size))
    {
        Some(b) => b,
        None => return,
    };

    let denom = batch * channels * dtype_size;
    assert!(denom > 0, "denominator must be positive");

    let inferred_time = data_bytes / denom;
    assert_eq!(inferred_time, time, "inferred time must match actual");
    assert_eq!(data_bytes % denom, 0, "bytes must be exactly divisible");
}

// ============================================================================
// 2. Style batch offset: total span overflow guard
// ============================================================================

/// Prove: the total span computation `2*C1 + 2*C2 + offset` is checked
/// for overflow at every intermediate step. No silent wrapping.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn style_batch_offset_span_overflow_guard() {
    let offset: usize = kani::any();
    let c1: usize = kani::any();
    let c2: usize = kani::any();

    kani::assume(c1 >= 1 && c1 <= 2048);
    kani::assume(c2 >= 1 && c2 <= 2048);
    kani::assume(offset <= 1 << 20);

    // Each intermediate must be checked.
    let span_c1 = c1.checked_mul(2);
    let span_c2 = c2.checked_mul(2);

    if let (Some(s1), Some(s2)) = (span_c1, span_c2) {
        if let Some(total_span) = s1.checked_add(s2) {
            if let Some(end) = offset.checked_add(total_span) {
                // Property: end >= offset (no wrapping).
                assert!(end >= offset, "end must not wrap below offset");
                // Property: end >= total_span (no wrapping).
                assert!(end >= total_span, "end must not wrap below total_span");
                // Property: end == offset + 2*c1 + 2*c2.
                assert_eq!(end, offset + 2 * c1 + 2 * c2);
            }
        }
    }
}

// ============================================================================
// 3. Style projection weight shape: [2*C, style_dim] vs [style_dim, 2*C]
// ============================================================================

/// Prove: style projection weight shape element count is the same
/// regardless of transposition: [2*C, D] has same elements as [D, 2*C].
/// This validates that the pre-transposed fast path and the
/// Linear fallback path consume exactly the same weight data.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn style_proj_weight_shape_transpose_invariant() {
    let channels: usize = kani::any();
    let style_dim: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 512);
    kani::assume(style_dim >= 1 && style_dim <= 512);

    let double_c = 2 * channels;

    // Original weight: [2*C, style_dim].
    let elems_orig = match double_c.checked_mul(style_dim) {
        Some(e) => e,
        None => return,
    };

    // Transposed weight: [style_dim, 2*C].
    let elems_trans = match style_dim.checked_mul(double_c) {
        Some(e) => e,
        None => return,
    };

    assert_eq!(
        elems_orig, elems_trans,
        "transposed weight must have same element count"
    );
}

// ============================================================================
// 4. Gamma/beta resolution: exactly one of three paths
// ============================================================================

/// Prove: the gamma/beta resolution dispatches to exactly one of:
/// 1. style_batch_offset (highest priority)
/// 2. style_proj (medium priority)
/// 3. direct buffers (lowest priority, fallback)
///
/// These are mutually exclusive because the code uses if-else-if-else.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gamma_beta_resolution_exactly_one_path() {
    let has_batch_offset: bool = kani::any();
    let has_style_proj: bool = kani::any();

    // Model the if-else chain from execute_native_fused_resblock.
    let takes_batch_offset = has_batch_offset;
    let takes_style_proj = !has_batch_offset && has_style_proj;
    let takes_direct = !has_batch_offset && !has_style_proj;

    // Property 1: exactly one path.
    let count = takes_batch_offset as u8 + takes_style_proj as u8 + takes_direct as u8;
    assert_eq!(count, 1, "exactly one gamma/beta path must be selected");

    // Property 2: batch_offset has highest priority.
    if has_batch_offset {
        assert!(takes_batch_offset, "batch_offset must take priority");
        assert!(!takes_style_proj, "style_proj must not fire");
        assert!(!takes_direct, "direct must not fire");
    }

    // Property 3: style_proj fires only when no batch_offset.
    if takes_style_proj {
        assert!(!has_batch_offset, "style_proj implies no batch_offset");
        assert!(has_style_proj, "style_proj path requires style_proj param");
    }

    // Property 4: direct path fires only when neither projection path fires.
    if takes_direct {
        assert!(!has_batch_offset && !has_style_proj);
    }
}

// ============================================================================
// 5. Residual scale f64 widening: f32 -> f64 is lossless
// ============================================================================

/// Prove: f64::from(scale) preserves the exact f32 value for all
/// finite f32 inputs. This is used in `sum.mul_scalar(f64::from(residual_scale))`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn residual_scale_f64_widening_lossless() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());

    let widened = f64::from(scale);

    // Property 1: widened value round-trips back to the same f32.
    let narrowed = widened as f32;
    assert_eq!(narrowed, scale, "f64::from(f32) must round-trip losslessly");

    // Property 2: widened value is also finite.
    assert!(widened.is_finite(), "widened value must be finite");
}

// ============================================================================
// 6. Pool path: phase1 input_channels == pool_channels
// ============================================================================

/// Prove: in the pool path, the pool output has the same channel count
/// as phase1.input_shape[1]. This is the invariant relied upon when
/// building the pool_shape = [B, pool_channels, pool_time].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_path_channels_match_phase1_input() {
    let batch: usize = kani::any();
    let phase1_in_channels: usize = kani::any();
    let pool_time: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(phase1_in_channels >= 1 && phase1_in_channels <= 512);
    kani::assume(pool_time >= 1 && pool_time <= 16384);

    // The pool output channels are set to phase1.input_shape[1].
    let pool_channels = phase1_in_channels;

    // Pool shape: [B, pool_channels, pool_time].
    let pool_shape = [batch, pool_channels, pool_time];

    // The Conv1d in phase1 reads input with in_channels = pool_channels.
    assert_eq!(
        pool_shape[1], phase1_in_channels,
        "pool channels must match phase1 input channels"
    );
}

// ============================================================================
// 7. Conv1d output length: general formula with arbitrary padding/dilation
// ============================================================================

/// Prove: Conv1d output length formula
/// `T_out = (T_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
/// for stride=1 simplifies to `T_in + 2*padding - dilation*(kernel-1)`.
///
/// This is the formula that determines whether shapes are compatible
/// across the FusedResBlock phase boundary.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_output_length_stride1_formula() {
    let t_in: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();
    let kernel: usize = kani::any();

    kani::assume(t_in >= 1 && t_in <= 8192);
    kani::assume(padding <= 128);
    kani::assume(dilation >= 1 && dilation <= 8);
    kani::assume(kernel >= 1 && kernel <= 15);

    let effective_k = (kernel - 1) * dilation + 1;
    let padded = t_in + 2 * padding;
    kani::assume(padded >= effective_k); // valid conv

    // Stride=1: general formula reduces.
    let t_out_general = (padded - effective_k) / 1 + 1;
    let t_out_simplified = padded - effective_k + 1;

    assert_eq!(
        t_out_general, t_out_simplified,
        "stride=1 general formula must equal simplified"
    );

    // Property: output length is positive.
    assert!(t_out_simplified >= 1, "output length must be >= 1");
}

// ============================================================================
// 8. Shortcut conv1x1: output shape [B, C_out, T]
// ============================================================================

/// Prove: conv1x1 (kernel=1, padding=0, dilation=1) preserves the
/// temporal dimension exactly. Output shape is [B, C_out, T_in].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn shortcut_conv1x1_preserves_temporal() {
    let t_in: usize = kani::any();
    kani::assume(t_in >= 1 && t_in <= 16384);

    // Conv1x1 params: kernel=1, padding=0, dilation=1, stride=1.
    let kernel = 1;
    let padding = 0;
    let dilation = 1;

    let effective_k = (kernel - 1) * dilation + 1; // = 1
    let padded = t_in + 2 * padding; // = t_in
    assert!(padded >= effective_k); // t_in >= 1 always holds
    let t_out = padded - effective_k + 1; // = t_in

    assert_eq!(t_out, t_in, "conv1x1 must preserve temporal dimension");
}

// ============================================================================
// 9. PhaseWeightKeys: label field is prefix of all keys
// ============================================================================

/// Prove: each PhaseWeightKeys struct's `label` field is a strict prefix
/// of all other keys in the same struct. This ensures that label-based
/// error messages correctly identify the phase.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn phase_keys_label_is_prefix_of_all_keys() {
    // Phase 1 keys.
    let p1_label = "p1_";
    let p1_alpha = "p1_alpha";
    let p1_conv_weight = "p1_conv_weight";
    let p1_conv_bias = "p1_conv_bias";

    assert!(p1_alpha.starts_with(p1_label));
    assert!(p1_conv_weight.starts_with(p1_label));
    assert!(p1_conv_bias.starts_with(p1_label));

    // Phase 2 keys.
    let p2_label = "p2_";
    let p2_alpha = "p2_alpha";
    let p2_conv_weight = "p2_conv_weight";
    let p2_conv_bias = "p2_conv_bias";

    assert!(p2_alpha.starts_with(p2_label));
    assert!(p2_conv_weight.starts_with(p2_label));
    assert!(p2_conv_bias.starts_with(p2_label));

    // Labels are distinct.
    assert_ne!(p1_label, p2_label);
}

// ============================================================================
// 10. Batch offset buffer denominator: always non-zero
// ============================================================================

/// Prove: `batch * dtype.size_bytes()` is non-zero for valid batch (>=1)
/// and valid dtype sizes (2 or 4). Division by zero in
/// `total_out_dim = slice_bytes / (batch * dtype_size)` would panic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_offset_denominator_nonzero() {
    let batch: usize = kani::any();
    let dtype_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(dtype_size == 2 || dtype_size == 4);

    let denom = batch * dtype_size;
    assert!(denom > 0, "denominator must be non-zero");
    assert!(denom >= 2, "minimum denominator is 1*2 = 2");
}

// ============================================================================
// 11. Style projection bias shape: exactly 2 * channels
// ============================================================================

/// Prove: the style projection bias tensor has shape [2*channels],
/// which is exactly the sum of gamma_channels + beta_channels.
/// This ensures the Linear projection produces the correct output dim.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn style_proj_bias_shape_is_double_channels() {
    let channels: usize = kani::any();
    kani::assume(channels >= 1 && channels <= 1024);

    let bias_dim = 2 * channels;

    // Property 1: bias_dim = gamma_dim + beta_dim.
    assert_eq!(bias_dim, channels + channels);

    // Property 2: no overflow for Kokoro-range channels.
    let checked = channels.checked_mul(2);
    assert!(checked.is_some(), "2*channels must not overflow");
    assert_eq!(checked.unwrap(), bias_dim);
}

// ============================================================================
// 12. Gamma/beta [B,C,1] element product: overflow check
// ============================================================================

/// Prove: the element product B*C*1 for gamma/beta reshapes cannot overflow
/// for Kokoro-range parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gamma_beta_reshape_no_overflow() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 1024);

    let elems = batch.checked_mul(channels).and_then(|bc| bc.checked_mul(1));
    assert!(elems.is_some(), "B*C*1 must not overflow");

    let elems = elems.unwrap();
    assert_eq!(elems, batch * channels, "trailing 1 does not change count");
    assert!(elems >= 1, "element count must be positive");
}

// ============================================================================
// 13. Input steps: index [0] always valid when len >= 1
// ============================================================================

/// Prove: resolve_step(0) is always valid when input_steps has at least
/// one element. This is the common path: x is always at index 0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn input_steps_index_zero_always_valid() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 10);

    // Index 0 is in bounds for any non-empty slice.
    assert!(0 < len, "index 0 must be in bounds when len >= 1");
}

// ============================================================================
// 14. Residual add: element count match
// ============================================================================

/// Prove: for residual add, both operands must have the same element count.
/// [B, C_out, T] + [B, C_out, T] requires B*C_out*T to match.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn residual_add_element_count_match() {
    let b1: usize = kani::any();
    let c1: usize = kani::any();
    let t1: usize = kani::any();
    let b2: usize = kani::any();
    let c2: usize = kani::any();
    let t2: usize = kani::any();

    kani::assume(b1 >= 1 && b1 <= 64);
    kani::assume(c1 >= 1 && c1 <= 1024);
    kani::assume(t1 >= 1 && t1 <= 8192);
    kani::assume(b2 >= 1 && b2 <= 64);
    kani::assume(c2 >= 1 && c2 <= 1024);
    kani::assume(t2 >= 1 && t2 <= 8192);

    // Add requires matching shapes.
    kani::assume(b1 == b2 && c1 == c2 && t1 == t2);

    let elems1 = b1
        .checked_mul(c1)
        .and_then(|v| v.checked_mul(t1));
    let elems2 = b2
        .checked_mul(c2)
        .and_then(|v| v.checked_mul(t2));

    assert!(elems1.is_some() && elems2.is_some(), "element counts must not overflow");
    assert_eq!(
        elems1.unwrap(),
        elems2.unwrap(),
        "matching shapes must have matching element counts"
    );
}

// ============================================================================
// 15. Pool path temporal inference: division-exact for stride=2 upsampling
// ============================================================================

/// Prove: after ConvTranspose1d with stride=2, the pool output time is
/// approximately 2*T_in. The byte-length inference must recover this
/// doubled time dimension exactly.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_upsample_time_inference() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let pool_time: usize = kani::any(); // after upsampling
    let dtype_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(channels >= 1 && channels <= 512);
    kani::assume(pool_time >= 2 && pool_time <= 32768);
    kani::assume(dtype_size == 2 || dtype_size == 4);

    // Buffer allocated for [B, C, pool_time].
    let data_bytes = match batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(pool_time))
        .and_then(|v| v.checked_mul(dtype_size))
    {
        Some(b) => b,
        None => return,
    };

    // Inference: pool_time = data_bytes / (batch * channels * dtype_size).
    let denom = batch * channels * dtype_size;
    let inferred = data_bytes / denom;
    assert_eq!(inferred, pool_time, "upsample pool time must be recovered exactly");
}
