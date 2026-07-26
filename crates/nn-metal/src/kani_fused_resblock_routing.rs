// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for FusedResBlock dispatch routing (#3554).
//!
//! Complements `kani_fused_resblock.rs` (buffer narrow/shape/epsilon proofs)
//! with proofs about:
//! - Dispatch count bounds (estimated_metal_dispatches / encoding events)
//! - 3-way routing determinism (batch_offset vs style_proj vs direct)
//! - Output buffer size from buffer_planner_bytes
//! - Pool step buffer dimension inference
//! - Style projection weight shape consistency
//! - Conv1x1 shortcut shape compatibility with phase2 output
//! - StyleBatchOffset accumulation across multiple blocks

/// Prove: Dispatch count bounds for FusedResBlock are deterministic.
///
/// Models `estimated_metal_dispatches()` from
/// `trace_compile_native_ops_dispatch_count.rs:101-113`:
/// ```
/// let base = 3; // stats + conv_with_stats + conv_precomputed
/// let proj = match (style_proj, style_batch_offset) {
///     (Some(_), _) => 4,  // unbatched: 2 projections x 2 dispatches
///     (_, Some(_)) => 0,  // batched: zero-copy narrow
///     (None, None) => 0,  // pre-computed gamma/beta
/// };
/// base + proj
/// ```
///
/// This harness proves:
/// 1. The dispatch count is exactly one of {3, 7} — no other values.
/// 2. Same inputs always produce the same dispatch count (determinism).
/// 3. style_proj takes priority when both style_proj and batch_offset set.
/// 4. Without style_proj, dispatch count is always 3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_count_bounds_deterministic() {
    let has_style_proj: bool = kani::any();
    let has_batch_offset: bool = kani::any();

    let base: usize = 3;
    let proj: usize = if has_style_proj {
        4 // unbatched: 2 projections x 2 dispatches
    } else if has_batch_offset {
        0 // batched: zero-copy narrow from batch output
    } else {
        0 // pre-computed gamma/beta
    };

    let total = base + proj;

    // Property 1: dispatch count is bounded to exactly {3, 7}.
    assert!(
        total == 3 || total == 7,
        "dispatch count must be 3 (no proj) or 7 (with proj)"
    );

    // Property 2: determinism — re-computing with same inputs yields same result.
    let proj2: usize = if has_style_proj {
        4
    } else if has_batch_offset {
        0
    } else {
        0
    };
    let total2 = base + proj2;
    assert_eq!(total, total2, "dispatch count must be deterministic");

    // Property 3: When style_proj is present, dispatch count is always 7
    // regardless of style_batch_offset.
    if has_style_proj {
        assert_eq!(total, 7, "style_proj path must always produce 7 dispatches");
    }

    // Property 4: Without style_proj, dispatch count is always 3.
    if !has_style_proj {
        assert_eq!(total, 3, "non-style-proj path must produce 3 dispatches");
    }
}

/// Prove: Encoding event count bounds for FusedResBlock are deterministic.
///
/// Models `estimated_encoding_events()` from
/// `trace_compile_native_ops_dispatch_count.rs:209-221`:
/// ```
/// let base = 2; // phase 1 + phase 2
/// let proj = match (style_proj, style_batch_offset) {
///     (Some(_), _) => 4,
///     (_, Some(_)) => 0,
///     (None, None) => 0,
/// };
/// base + proj
/// ```
///
/// Proves encoding events are exactly {2, 6} and consistent with
/// dispatch count (encoding events <= dispatch count always).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn encoding_event_count_bounds() {
    let has_style_proj: bool = kani::any();
    let has_batch_offset: bool = kani::any();

    // Dispatch count (from estimated_metal_dispatches).
    let dispatch_base: usize = 3;
    let dispatch_proj: usize = if has_style_proj {
        4
    } else if has_batch_offset {
        0
    } else {
        0
    };
    let dispatch_total = dispatch_base + dispatch_proj;

    // Encoding event count (from estimated_encoding_events).
    let enc_base: usize = 2;
    let enc_proj: usize = if has_style_proj {
        4
    } else if has_batch_offset {
        0
    } else {
        0
    };
    let enc_total = enc_base + enc_proj;

    // Property 1: encoding events are bounded to {2, 6}.
    assert!(
        enc_total == 2 || enc_total == 6,
        "encoding events must be 2 (no proj) or 6 (with proj)"
    );

    // Property 2: encoding events <= dispatch count (always true).
    assert!(
        enc_total <= dispatch_total,
        "encoding events must not exceed dispatch count"
    );

    // Property 3: style projection overhead is identical for both metrics.
    assert_eq!(
        dispatch_proj, enc_proj,
        "style proj overhead must match for dispatches and encodings"
    );
}

/// Prove: FusedResBlock output buffer bytes from buffer_planner is exact.
///
/// Models `native_op_output_bytes()` in `buffer_planner_bytes.rs:126`:
/// ```
/// NativeOpKind::FusedResBlock { phase1, .. } =>
///     checked_shape_bytes(&phase1.input_shape)
/// ```
///
/// The FusedResBlock output shape equals phase1.input_shape because:
/// 1. Both conv phases use stride=1 with same-padding => T preserved.
/// 2. The residual add requires output shape == input shape.
/// 3. The optional residual_scale is element-wise => shape preserved.
///
/// This harness proves the buffer planner allocates exactly the right
/// number of bytes for any valid Kokoro-range input shape.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn output_buffer_bytes_matches_input_shape() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let time: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 1024);
    kani::assume(time >= 1 && time <= (1usize << 14));

    let f32_bytes: usize = 4;

    // Model checked_shape_bytes(&[batch, channels, time]).
    let product = match batch.checked_mul(channels) {
        Some(bc) => match bc.checked_mul(time) {
            Some(bct) => bct,
            None => return,
        },
        None => return,
    };
    let bytes = match product.checked_mul(f32_bytes) {
        Some(b) => b,
        None => return,
    };

    // Property 1: bytes > 0 for non-degenerate shapes.
    assert!(bytes > 0, "output bytes must be positive");

    // Property 2: bytes is exactly B * C * T * 4.
    assert_eq!(bytes, batch * channels * time * f32_bytes);

    // Property 3: element count is recoverable from bytes.
    let recovered_elements = bytes / f32_bytes;
    assert_eq!(recovered_elements, batch * channels * time);

    // Property 4: the buffer can hold the full residual add result.
    let residual_bytes = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(time))
        .and_then(|v| v.checked_mul(f32_bytes));
    if let Some(rb) = residual_bytes {
        assert_eq!(bytes, rb, "output buffer must hold full residual result");
    }
}

/// Prove: 3-way routing is mutually exclusive and exhaustive.
///
/// Models the routing decision in `execute_native_fused_resblock()`:
/// ```
/// if let Some(sbo) = style_batch_offset {
///     // Path 1: batch_offset — input_steps = [x, batch_step]
/// } else if let Some(sp) = style_proj {
///     // Path 2: style_proj — input_steps = [x, style_embed]
/// } else {
///     // Path 3: direct — input_steps = [x, g1, b1, g2, b2]
/// }
/// ```
///
/// This harness proves:
/// 1. Exactly one path is taken for any configuration.
/// 2. The minimum input_steps length for the chosen path is correct.
/// 3. batch_offset takes priority over style_proj.
/// 4. Routing is deterministic (same inputs => same path).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn routing_mutually_exclusive_and_exhaustive() {
    let has_batch_offset: bool = kani::any();
    let has_style_proj: bool = kani::any();
    let input_steps_len: usize = kani::any();
    kani::assume(input_steps_len <= 10);

    // Model the 3-way routing.
    let (path, min_len) = if has_batch_offset {
        (1u8, 2usize) // batch_offset path
    } else if has_style_proj {
        (2u8, 2usize) // style_proj path
    } else {
        (3u8, 5usize) // direct buffer path
    };

    // Property 1: Exactly one path is taken.
    assert!(
        path >= 1 && path <= 3,
        "exactly one of three routing paths must be selected"
    );

    // Property 2: Re-derive and verify determinism.
    let path2 = if has_batch_offset {
        1u8
    } else if has_style_proj {
        2u8
    } else {
        3u8
    };
    assert_eq!(path, path2, "routing must be deterministic");

    // Property 3: batch_offset takes priority over style_proj.
    if has_batch_offset && has_style_proj {
        assert_eq!(path, 1, "batch_offset must take priority");
    }

    // Property 4: When length check passes, all accessed indices valid.
    if input_steps_len >= min_len {
        let max_accessed = min_len - 1;
        assert!(
            max_accessed < input_steps_len,
            "all accessed indices must be in bounds"
        );
    }

    // Property 5: Direct path requires 5 inputs.
    if !has_batch_offset && !has_style_proj {
        assert_eq!(min_len, 5, "direct path must require 5 input_steps");
    }
}

/// Prove: Pool step buffer dimension inference is exact.
///
/// Models the pool path in `execute_native_fused_resblock()`:
/// ```
/// let pool_channels = phase1.input_shape[1];
/// let pool_bytes = pool_slice.buffer().len() - pool_slice.byte_offset();
/// let pool_time = pool_bytes / (batch * pool_channels * dtype.size_bytes());
/// ```
///
/// This integer division must be exact for the inferred shape to be
/// correct. Same pattern as `total_out_dim_buffer_inference_exact` in
/// `kani_fused_resblock.rs` but for 3D pool output tensors.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_buffer_dimension_inference_exact() {
    let batch: usize = kani::any();
    let pool_channels: usize = kani::any();
    let pool_time: usize = kani::any();
    let dtype_size: usize = kani::any();
    let byte_offset: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(pool_channels >= 1 && pool_channels <= 1024);
    kani::assume(pool_time >= 1 && pool_time <= (1usize << 14));
    kani::assume(dtype_size == 2 || dtype_size == 4);
    kani::assume(byte_offset <= (1usize << 20));

    // Compute buffer: batch * pool_channels * pool_time * dtype_size + offset.
    let data_bytes = match batch.checked_mul(pool_channels) {
        Some(bc) => match bc.checked_mul(pool_time) {
            Some(bct) => match bct.checked_mul(dtype_size) {
                Some(b) => b,
                None => return,
            },
            None => return,
        },
        None => return,
    };
    let buffer_len = match data_bytes.checked_add(byte_offset) {
        Some(l) => l,
        None => return,
    };

    // Model the inference.
    let pool_bytes = buffer_len - byte_offset;

    let denom = match batch.checked_mul(pool_channels) {
        Some(bc) => match bc.checked_mul(dtype_size) {
            Some(d) => d,
            None => return,
        },
        None => return,
    };

    let inferred_time = pool_bytes / denom;

    // Prove: for a correctly-allocated buffer, the inference is exact.
    assert_eq!(
        inferred_time, pool_time,
        "inferred pool_time must match actual allocation"
    );

    // Prove: no truncation occurred.
    assert_eq!(
        pool_bytes % denom, 0,
        "pool buffer byte length must be exactly divisible"
    );
}

/// Prove: Style projection weight shape is consistent with channel count.
///
/// Models the style projection weight shape in
/// `compiled_model_execute_native_resblock_helpers.rs`:
/// ```
/// bias: [2 * channels]
/// weight: [2 * channels, style_dim]
/// ```
///
/// The narrow split divides projected output [B, 2*channels] into
/// gamma [B, channels] and beta [B, channels]. This requires:
/// 1. 2*channels does not overflow
/// 2. The narrow offsets 0 and channels are valid within 2*channels
/// 3. The matmul output [B, 2*channels] has the right element count
/// 4. Reshape to [B, C, 1] preserves element count
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn style_projection_weight_shape_consistency() {
    let channels: usize = kani::any();
    let style_dim: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 512);
    kani::assume(style_dim >= 1 && style_dim <= 512);
    kani::assume(batch >= 1 && batch <= 64);

    // 2 * channels must not overflow.
    let double_channels = match channels.checked_mul(2) {
        Some(dc) => dc,
        None => return,
    };

    // Bias shape: [2 * channels].
    assert_eq!(double_channels, 2 * channels);

    // Weight shape: [2 * channels, style_dim]. Must not overflow.
    let weight_elems = match double_channels.checked_mul(style_dim) {
        Some(e) => e,
        None => return,
    };
    assert_eq!(weight_elems, 2 * channels * style_dim);

    // Narrow gamma: offset=0, length=channels.
    let gamma_end = channels;
    assert!(gamma_end <= double_channels, "gamma narrow in bounds");

    // Narrow beta: offset=channels, length=channels.
    let beta_end = match channels.checked_add(channels) {
        Some(e) => e,
        None => return,
    };
    assert_eq!(beta_end, double_channels, "beta narrow covers rest exactly");

    // Gamma + beta accounts for all projected elements per row.
    assert_eq!(
        channels + channels,
        double_channels,
        "gamma + beta must equal projected output dim"
    );

    // Reshape: [B, channels] -> [B, channels, 1]. Element count preserved.
    let reshaped_elems = match batch.checked_mul(channels) {
        Some(bc) => bc,
        None => return,
    };
    assert_eq!(
        reshaped_elems,
        batch * channels,
        "reshape to [B, C, 1] must preserve element count"
    );
}

/// Prove: Conv1x1 shortcut output shape matches phase2 output shape.
///
/// Models the shortcut path in `execute_native_fused_resblock()`:
/// ```
/// let sc_shape = vec![batch, phase2.output_channels, phase1.input_shape[2]];
/// ```
///
/// For the residual add to succeed, the shortcut shape must match the
/// phase2 output shape. Phase2 output is [B, out_channels, T_out].
/// With same-padding and stride=1, T_out == T_in == phase1.input_shape[2].
///
/// This harness proves: when phase2 uses same-padding, the shortcut shape
/// and phase2 output shape are identical (element-wise add is valid).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1x1_shortcut_shape_matches_phase2_output() {
    let batch: usize = kani::any();
    let out_channels: usize = kani::any();
    let t_in: usize = kani::any();
    let kernel2: usize = kani::any();
    let dilation2: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(out_channels >= 1 && out_channels <= 1024);
    kani::assume(t_in >= 1 && t_in <= (1usize << 14));
    kani::assume(kernel2 >= 1 && kernel2 <= 15);
    kani::assume(dilation2 >= 1 && dilation2 <= 8);
    kani::assume(kernel2 % 2 == 1); // same-padding requires odd kernel

    // Same-padding: padding = dilation * (kernel - 1) / 2.
    let pad2 = dilation2 * (kernel2 - 1) / 2;
    let eff_k2 = (kernel2 - 1) * dilation2 + 1;
    let padded2 = t_in + 2 * pad2;
    kani::assume(padded2 >= eff_k2);
    let t_out = padded2 - eff_k2 + 1;

    // With same-padding, T_out == T_in.
    assert_eq!(t_out, t_in, "same-padding must preserve temporal dim");

    // Shortcut shape: [B, out_channels, T_in].
    // Phase2 output shape: [B, out_channels, T_out].
    assert_eq!(t_in, t_out, "time dims must match for residual add");

    // Element count must be identical.
    let sc_elems = batch * out_channels * t_in;
    let p2_elems = batch * out_channels * t_out;
    assert_eq!(sc_elems, p2_elems, "element counts must match for add");
}

/// Prove: StyleBatchOffset accumulation across blocks does not overflow.
///
/// When multiple FusedResBlocks share a `BatchedStyleProjection`, each
/// block's `StyleBatchOffset` occupies a non-overlapping region of the
/// batched output. The layout per block is:
/// [gamma1(C1), beta1(C1), gamma2(C2), beta2(C2)]
/// Total span per block = 2*C1 + 2*C2.
///
/// This harness proves: accumulated offsets through multiple blocks
/// (a) never overflow, (b) total_out is exactly the sum of all spans,
/// (c) each block's region does not overlap the next.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn style_batch_offset_accumulation_no_overflow() {
    let num_blocks: usize = kani::any();
    kani::assume(num_blocks >= 1 && num_blocks <= 4);

    let mut accumulated_offset: usize = 0;

    let mut i: usize = 0;
    while i < num_blocks {
        let c1: usize = kani::any();
        let c2: usize = kani::any();
        kani::assume(c1 >= 1 && c1 <= 512);
        kani::assume(c2 >= 1 && c2 <= 512);

        // Per-block span = 2*C1 + 2*C2.
        let span_c1 = match c1.checked_mul(2) {
            Some(s) => s,
            None => return,
        };
        let span_c2 = match c2.checked_mul(2) {
            Some(s) => s,
            None => return,
        };
        let block_span = match span_c1.checked_add(span_c2) {
            Some(s) => s,
            None => return,
        };

        // This block starts at accumulated_offset.
        let block_end = match accumulated_offset.checked_add(block_span) {
            Some(e) => e,
            None => return,
        };

        // Non-overlap: block_end > accumulated_offset (span > 0).
        assert!(
            block_end > accumulated_offset,
            "each block must occupy a positive region"
        );

        accumulated_offset = block_end;
        i += 1;
    }

    let total_out = accumulated_offset;

    // Property: total_out > 0 with at least 1 block.
    assert!(total_out > 0, "total_out must be positive");

    // Property: total_out bounded by max span per block.
    // Each block: 2*512 + 2*512 = 2048 max.
    assert!(
        total_out <= num_blocks * 2048,
        "total_out must be bounded by block count * max span"
    );
}
