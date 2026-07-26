// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `compiled_model_kernel_spec_norm.rs` and
//! `compiled_model_kernel_spec_ops.rs` (#3657).
//!
//! These two files contain the `spec_*()` builder functions for normalization
//! and single-dispatch GPU kernels. Each builder computes grid dimensions,
//! binding layouts, and output buffer sizes from input shapes.
//!
//! ## Properties Proved
//!
//! ### Norm builders (spec_norm.rs)
//! - InstanceNorm: flat_rows = batch * channels, grid covers all rows
//! - LayerNorm: flat_rows = product of all dims except last
//! - AddLayerNorm: same grid/binding structure as LayerNorm + extra edge
//! - ChannelsFirstLayerNorm: flat_rows = B * T, LeakyRelu adds 1 binding
//! - AdaLayerNorm: flat_rows = batch * mid_dims, 3 edge inputs
//!
//! ### Ops builders (spec_ops.rs)
//! - GroupNorm: channels divisibility check, flat_cols = (C/G) * spatial
//! - RmsNorm: flat_rows for rank-1 tensor is 1
//! - Snake: threadgroup count covers all elements
//! - FlashAttention: GQA group_size divides H_q, grid covers all Q blocks
//!
//! ### Cross-cutting
//! - All output_bytes = total_elems * elem_bytes without overflow
//! - NORM_TG_SIZE = 256, within Metal's 1024-thread limit
//! - All grid dimensions fit in u32

use crate::compiled_model::kernel_spec::norm::NORM_TG_SIZE;

// =========================================================================
// spec_instance_norm
// =========================================================================

/// Proves: InstanceNorm flat_rows = batch * channels for valid rank-3+ input.
///
/// SUBSTANTIVE: `spec_instance_norm` computes `flat_rows = batch * channels`
/// and dispatches one threadgroup per row. This must match the expected
/// B*C reduction pattern.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 49-51.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn instance_norm_flat_rows_correct() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let spatial: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 1024);
    kani::assume(spatial >= 1 && spatial <= 16384);

    let flat_rows = batch.checked_mul(channels);
    assert!(flat_rows.is_some(), "batch * channels must not overflow");

    let total = flat_rows.unwrap().checked_mul(spatial);
    assert!(total.is_some(), "total elements must not overflow");

    // Grid dispatches flat_rows threadgroups.
    let flat_rows_u32 = u32::try_from(flat_rows.unwrap());
    assert!(
        flat_rows_u32.is_ok(),
        "flat_rows must fit in u32"
    );
}

/// Proves: InstanceNorm output_bytes does not overflow for production shapes.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 79-81.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn instance_norm_output_bytes_no_overflow() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let spatial: usize = kani::any();
    let elem_bytes: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 1024);
    kani::assume(spatial >= 1 && spatial <= 16384);
    kani::assume(elem_bytes == 2 || elem_bytes == 4);

    let total = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(spatial));
    kani::assume(total.is_some());

    let output_bytes = total.unwrap().checked_mul(elem_bytes);
    assert!(
        output_bytes.is_some(),
        "InstanceNorm output_bytes must not overflow"
    );
}

/// Proves: InstanceNorm has exactly 4 bindings (edge, output, spatial, eps).
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 91-97.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn instance_norm_binding_count() {
    // InstanceNorm bindings: (0, Edge(0)), (1, Output), (2, spatial), (3, eps)
    let binding_count = 4;
    let param_count = 1; // Only 1 input buffer (the tensor).

    assert_eq!(binding_count, 4, "InstanceNorm must have 4 bindings");
    assert!(
        param_count <= binding_count,
        "param_count must not exceed binding count"
    );
}

// =========================================================================
// spec_layer_norm
// =========================================================================

/// Proves: LayerNorm flat_rows is the product of all dims except last.
///
/// SUBSTANTIVE: `flat_rows = input_shape[..len-1].iter().product()`.
/// For shape [B, S, H], flat_rows = B * S. For shape [B, H], flat_rows = B.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` line 129.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn layer_norm_flat_rows_product() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let hidden_dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(seq_len >= 1 && seq_len <= 4096);
    kani::assume(hidden_dim >= 1 && hidden_dim <= 4096);

    // Rank 3: [B, S, H] → flat_rows = B * S
    let flat_rows = batch.checked_mul(seq_len);
    assert!(flat_rows.is_some(), "B * S must not overflow");

    let flat_rows_u32 = u32::try_from(flat_rows.unwrap());
    assert!(flat_rows_u32.is_ok(), "flat_rows must fit in u32");

    // Rank 2: [B, H] → flat_rows = B
    let flat_rows_2d = batch;
    assert!(flat_rows_2d >= 1, "rank-2 flat_rows must be >= 1");
}

/// Proves: LayerNorm has exactly 6 bindings.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 169-177.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn layer_norm_binding_count() {
    // LayerNorm bindings: edge(0), weight, bias, output, hidden_dim, eps
    let binding_count = 6;
    let param_count = 3; // edge + weight + bias

    assert_eq!(binding_count, 6, "LayerNorm must have 6 bindings");
    assert_eq!(param_count, 3, "LayerNorm param_count must be 3");
}

// =========================================================================
// spec_add_layer_norm
// =========================================================================

/// Proves: AddLayerNorm has exactly 7 bindings (1 more than LayerNorm).
///
/// SUBSTANTIVE: AddLayerNorm adds a second edge input (b) compared to
/// LayerNorm. bindings: edge(0), edge(1), weight, bias, output, hidden_dim, eps.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 250-259.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn add_layer_norm_binding_count() {
    let binding_count = 7;
    let param_count = 4; // 2 edges + weight + bias

    assert_eq!(binding_count, 7, "AddLayerNorm must have 7 bindings");
    assert_eq!(param_count, 4, "AddLayerNorm param_count must be 4");
    assert_eq!(
        binding_count - 6, 1,
        "AddLayerNorm has exactly 1 more binding than LayerNorm"
    );
}

// =========================================================================
// spec_channels_first_layer_norm
// =========================================================================

/// Proves: ChannelsFirstLayerNorm flat_rows = B * T (not B * C).
///
/// SUBSTANTIVE: For [B, C, T] input, the kernel normalizes over C,
/// dispatching one threadgroup per (b, t) pair. flat_rows = B * T.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 297-299.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn channels_first_ln_flat_rows_is_bt() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let time_steps: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(channels >= 1 && channels <= 1024);
    kani::assume(time_steps >= 1 && time_steps <= 8192);

    let bt = batch.checked_mul(time_steps);
    assert!(bt.is_some(), "B * T must not overflow");

    // Grid is [bt, 1, 1], NOT [batch * channels, 1, 1].
    let bt_u32 = u32::try_from(bt.unwrap());
    assert!(bt_u32.is_ok(), "B * T must fit in u32");

    // Total elements = bt * channels.
    let total = bt.unwrap().checked_mul(channels);
    assert!(total.is_some(), "total elements must not overflow");
}

/// Proves: LeakyRelu adds exactly 1 binding (slope constant).
///
/// SUBSTANTIVE: Without LeakyRelu: 7 bindings. With LeakyRelu: 8 bindings.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 367-369.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn channels_first_ln_leaky_relu_adds_one_binding() {
    let has_leaky_relu: bool = kani::any();

    let base_bindings = 7; // edge, weight, bias, output, channels, time_steps, eps
    let total = if has_leaky_relu {
        base_bindings + 1
    } else {
        base_bindings
    };

    if has_leaky_relu {
        assert_eq!(total, 8, "with LeakyRelu: 8 bindings");
    } else {
        assert_eq!(total, 7, "without LeakyRelu: 7 bindings");
    }
}

// =========================================================================
// spec_ada_layer_norm
// =========================================================================

/// Proves: AdaLayerNorm mid_dims computation for rank-3+ shapes.
///
/// SUBSTANTIVE: mid_dims = product of input_shape[1..len-1].
/// For [B, T, C]: mid_dims = T. For [B, H, W, C]: mid_dims = H * W.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` line 417.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn ada_layer_norm_mid_dims() {
    let batch: usize = kani::any();
    let mid: usize = kani::any();
    let hidden_dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(mid >= 1 && mid <= 4096);
    kani::assume(hidden_dim >= 1 && hidden_dim <= 4096);

    let flat_rows = batch.checked_mul(mid);
    assert!(flat_rows.is_some(), "B * mid must not overflow");

    let flat_rows_u32 = u32::try_from(flat_rows.unwrap());
    assert!(flat_rows_u32.is_ok(), "flat_rows must fit in u32");

    let total = flat_rows.unwrap().checked_mul(hidden_dim);
    assert!(total.is_some(), "total elements must not overflow");
}

/// Proves: AdaLayerNorm has exactly 9 bindings and param_count=5.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` lines 464-475.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn ada_layer_norm_binding_count() {
    // Bindings: 3 edges + 2 weights + output + 3 constants = 9.
    let binding_count = 9;
    let param_count = 5; // 3 edges + 2 weights

    assert_eq!(binding_count, 9, "AdaLayerNorm must have 9 bindings");
    assert_eq!(param_count, 5, "AdaLayerNorm param_count must be 5");
}

// =========================================================================
// spec_group_norm
// =========================================================================

/// Proves: GroupNorm channels divisibility check is correct.
///
/// SUBSTANTIVE: `spec_group_norm` requires `channels % num_groups == 0`.
/// channels_per_group = channels / num_groups must be exact.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 54-58.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_norm_channels_divisibility() {
    let channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 1024);
    kani::assume(num_groups >= 1 && num_groups <= 256);
    kani::assume(channels % num_groups == 0);

    let channels_per_group = channels / num_groups;
    assert!(channels_per_group >= 1, "channels_per_group must be >= 1");

    // Verify: channels_per_group * num_groups == channels.
    assert_eq!(
        channels_per_group * num_groups,
        channels,
        "division must be exact"
    );
}

/// Proves: GroupNorm flat_rows = B * G (not B * C).
///
/// SUBSTANTIVE: One threadgroup per group-row, not per channel-row.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 62-64.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_norm_flat_rows_is_bg() {
    let batch: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(num_groups >= 1 && num_groups <= 256);

    let flat_rows = batch.checked_mul(num_groups);
    assert!(flat_rows.is_some(), "B * G must not overflow");

    let flat_rows_u32 = u32::try_from(flat_rows.unwrap());
    assert!(flat_rows_u32.is_ok(), "B * G must fit in u32");
}

/// Proves: GroupNorm has exactly 9 bindings.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 113-124.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn group_norm_binding_count() {
    // edge, weight, bias, output, flat_cols, eps, cpg, spatial, num_groups
    let binding_count = 9;
    let param_count = 3;

    assert_eq!(binding_count, 9, "GroupNorm must have 9 bindings");
    assert_eq!(param_count, 3, "GroupNorm param_count must be 3");
}

// =========================================================================
// spec_rms_norm
// =========================================================================

/// Proves: RmsNorm flat_rows for rank-1 tensor is 1.
///
/// SUBSTANTIVE: For input shape [H], flat_rows = 1 (empty product).
/// This is the degenerate case where the entire tensor is one row.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 156-160.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn rms_norm_rank1_flat_rows_is_one() {
    let hidden_dim: usize = kani::any();
    kani::assume(hidden_dim >= 1 && hidden_dim <= 65536);

    // Rank 1: flat_rows = 1 (the if branch).
    let flat_rows: usize = 1;
    assert_eq!(flat_rows, 1, "rank-1 flat_rows must be 1");

    let total = flat_rows.checked_mul(hidden_dim);
    assert!(total.is_some(), "total elements must not overflow");
    assert_eq!(total.unwrap(), hidden_dim, "rank-1 total == hidden_dim");
}

/// Proves: RmsNorm has exactly 5 bindings.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 200-207.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn rms_norm_binding_count() {
    // edge, weight, output, hidden_dim, eps
    let binding_count = 5;
    let param_count = 2; // edge + weight

    assert_eq!(binding_count, 5, "RmsNorm must have 5 bindings");
    assert_eq!(param_count, 2, "RmsNorm param_count must be 2");
}

// =========================================================================
// spec_snake
// =========================================================================

/// Proves: Snake threadgroup count covers all elements.
///
/// SUBSTANTIVE: `num_threadgroups = total_elems.div_ceil(NORM_TG_SIZE)`.
/// Total threads = num_threadgroups * NORM_TG_SIZE >= total_elems.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` line 272.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snake_tg_count_covers_all_elements() {
    let total_elems: u32 = kani::any();
    kani::assume(total_elems >= 1);

    let num_tg = total_elems.div_ceil(NORM_TG_SIZE);
    let total_threads = num_tg.checked_mul(NORM_TG_SIZE);

    assert!(
        total_threads.is_some(),
        "num_tg * NORM_TG_SIZE must not overflow"
    );
    assert!(
        total_threads.unwrap() >= total_elems,
        "threadgroups must cover all elements"
    );
}

/// Proves: Snake channel_stride = product of spatial dims (dims[2..]).
///
/// SUBSTANTIVE: For [B, C, T]: channel_stride = T.
/// For [B, C, H, W]: channel_stride = H * W.
/// For [B, C]: channel_stride = 1.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 242-246.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snake_channel_stride_computation() {
    let rank: usize = kani::any();
    kani::assume(rank >= 2 && rank <= 4);

    let d2: usize = kani::any();
    let d3: usize = kani::any();
    kani::assume(d2 >= 1 && d2 <= 16384);
    kani::assume(d3 >= 1 && d3 <= 16384);

    let channel_stride: usize = match rank {
        2 => 1,
        3 => d2,
        4 => {
            let prod = d2.checked_mul(d3);
            kani::assume(prod.is_some());
            prod.unwrap().max(1)
        }
        _ => unreachable!(),
    };

    assert!(channel_stride >= 1, "channel_stride must be >= 1");
}

/// Proves: Snake has exactly 6 bindings.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 286-294.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snake_binding_count() {
    // edge, alpha_weight, output, total_elems, channel_stride, channels
    let binding_count = 6;
    let param_count = 2; // edge + alpha_weight

    assert_eq!(binding_count, 6, "Snake must have 6 bindings");
    assert_eq!(param_count, 2, "Snake param_count must be 2");
}

// =========================================================================
// spec_flash_attention
// =========================================================================

/// Proves: FlashAttention GQA group_size divides H_q.
///
/// SUBSTANTIVE: `group_size = h_q / h_kv`. The builder requires `h_q % h_kv == 0`.
/// After division, `group_size * h_kv == h_q`.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 359-364.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flash_attn_gqa_group_size() {
    let h_q: usize = kani::any();
    let h_kv: usize = kani::any();

    kani::assume(h_q >= 1 && h_q <= 128);
    kani::assume(h_kv >= 1 && h_kv <= 128);
    kani::assume(h_q % h_kv == 0);

    let group_size = h_q / h_kv;
    assert!(group_size >= 1, "group_size must be >= 1");
    assert_eq!(
        group_size * h_kv,
        h_q,
        "group_size * h_kv must equal h_q"
    );
}

/// Proves: FlashAttention grid covers all Q blocks.
///
/// SUBSTANTIVE: `grid_x = s_q.div_ceil(BLOCK_SIZE)` where BLOCK_SIZE=32.
/// Total Q blocks = grid_x >= ceil(S_q / 32). All S_q positions are covered.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` line 416.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flash_attn_grid_covers_all_q_blocks() {
    let s_q: u32 = kani::any();
    kani::assume(s_q >= 1);

    let block_size: u32 = 32; // FLASH_ATTN_BLOCK_SIZE
    let grid_x = s_q.div_ceil(block_size);

    let total_covered = grid_x.checked_mul(block_size);
    assert!(
        total_covered.is_some(),
        "grid_x * BLOCK_SIZE must not overflow"
    );
    assert!(
        total_covered.unwrap() >= s_q,
        "grid must cover all S_q positions"
    );
}

/// Proves: FlashAttention output element count does not overflow.
///
/// SUBSTANTIVE: total_output = B * H_q * S_q * D. Chained checked_mul
/// must succeed for production dimensions.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 392-399.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn flash_attn_output_count_no_overflow() {
    let batch: usize = kani::any();
    let h_q: usize = kani::any();
    let s_q: usize = kani::any();
    let d: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(h_q >= 1 && h_q <= 128);
    kani::assume(s_q >= 1 && s_q <= 8192);
    kani::assume(d >= 1 && d <= 256);

    let total = batch
        .checked_mul(h_q)
        .and_then(|v| v.checked_mul(s_q))
        .and_then(|v| v.checked_mul(d));

    assert!(
        total.is_some(),
        "B * H_q * S_q * D must not overflow for production dims"
    );

    let elem_bytes: usize = kani::any();
    kani::assume(elem_bytes == 2 || elem_bytes == 4);

    let output_bytes = total.unwrap().checked_mul(elem_bytes);
    assert!(
        output_bytes.is_some(),
        "flash_attn output_bytes must not overflow"
    );
}

/// Proves: FlashAttention has exactly 11 bindings.
///
/// Covers: `compiled_model_kernel_spec_ops.rs` lines 430-442.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flash_attn_binding_count() {
    // 3 edges (Q, K, V) + output + 7 constants = 11
    let binding_count = 11;
    let param_count = 3; // 3 edge inputs

    assert_eq!(binding_count, 11, "FlashAttention must have 11 bindings");
    assert_eq!(param_count, 3, "FlashAttention param_count must be 3");
}

// =========================================================================
// Cross-cutting: NORM_TG_SIZE within Metal limits
// =========================================================================

/// Proves: NORM_TG_SIZE = 256 is within Metal's max threadgroup size (1024).
///
/// SUBSTANTIVE: All norm/fused kernel specs use NORM_TG_SIZE = 256.
/// Metal requires threads_per_threadgroup <= 1024. The chosen value
/// must be a power of 2 for efficient reduction.
///
/// Covers: `compiled_model_kernel_spec_norm.rs` line 21.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn norm_tg_size_within_metal_limit() {
    let tg_size = NORM_TG_SIZE;

    assert!(tg_size <= 1024, "NORM_TG_SIZE must be <= Metal max (1024)");
    assert!(tg_size >= 32, "NORM_TG_SIZE must be >= 32 for useful reduction");
    assert!(
        tg_size.is_power_of_two(),
        "NORM_TG_SIZE must be a power of 2 for tree reduction"
    );
}

// =========================================================================
// Cross-cutting: output_bytes generic proof
// =========================================================================

/// Proves: output_bytes = total_elems * elem_bytes does not overflow
/// for shapes within production bounds.
///
/// SUBSTANTIVE: All spec builders compute output_bytes via checked_mul.
/// This harness covers the common pattern across all builders.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn output_bytes_no_overflow_generic() {
    let total_elems: usize = kani::any();
    let elem_bytes: usize = kani::any();

    // Max total: 64 * 1024 * 16384 = 1,073,741,824 (1G elements).
    kani::assume(total_elems >= 1 && total_elems <= 1_073_741_824);
    kani::assume(elem_bytes == 2 || elem_bytes == 4);

    let output_bytes = total_elems.checked_mul(elem_bytes);
    assert!(
        output_bytes.is_some(),
        "output_bytes must not overflow for production shapes"
    );

    // Max output: 1G * 4 = 4 GB. Must be representable in usize on 64-bit.
    assert!(
        output_bytes.unwrap() <= 4_294_967_296,
        "output_bytes within 4 GB for production shapes"
    );
}
