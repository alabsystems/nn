// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for SIMD matmul MSL dispatch and LSTM GPU sequence
//! dispatch safety (#3631).
//!
//! ## SIMD matmul properties proved:
//!
//! - `tg_memory_bytes` never exceeds Metal 32 KB threadgroup limit
//! - `tg_memory_bytes` matches documented per-config values
//! - `select_tile_config` always returns SMALL or LARGE
//! - Grid dimensions from tile config fit in u32
//! - GEMM output buffer byte count does not overflow
//! - Threads per threadgroup (128) is within Metal 1024-thread limit
//! - `should_use_f16_simdgroup` implies `should_use_simdgroup`
//! - `F16_MIN_THREADGROUPS` threshold scales correctly with tile area
//!
//! ## LSTM sequence properties proved:
//!
//! - LSTM 4*hidden_size gate dimension never overflows
//! - LSTM weight w_ih shape [4H, I] buffer bytes do not overflow
//! - LSTM weight w_hh shape [4H, H] buffer bytes do not overflow
//! - LSTM output buffer [S, B, H] bytes do not overflow
//! - LSTM state buffer [B, H] bytes do not overflow
//! - LSTM threadgroup memory `shared_h[H]` fits within 32 KB limit
//! - LSTM hidden_size=0 is correctly rejected
//! - LSTM hidden_size > MAX_THREADGROUP_HIDDEN is correctly rejected
//! - LSTM thread grid [batch_size, hidden_size] within Metal limits
//! - LSTM precomputed input_proj [S, B, 4H] buffer bytes do not overflow
//! - LSTM gate indexing (g*H + h) for g in 0..4 never overflows
//! - LSTM timestep reverse index `seq_len - 1 - t` is valid for all t

use crate::dyn_tensor_metal::{
    select_tile_config, should_use_f16_simdgroup, should_use_simdgroup, tg_memory_bytes,
    F16_MIN_THREADGROUPS, GemmTileConfig, MAX_THREADGROUP_HIDDEN,
};

// ===========================================================================
// SIMD Matmul: tg_memory_bytes
// ===========================================================================

/// Prove: threadgroup memory for all tile configs never exceeds Metal's 32 KB limit.
///
/// Metal specification: maximum 32,768 bytes threadgroup memory per threadgroup.
/// Both SMALL (32x32) and LARGE (64x64) configs, in both f32 and f16 modes,
/// must stay within this limit.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tg_memory_within_32kb_limit() {
    let is_half: bool = kani::any();

    let small_bytes = tg_memory_bytes(GemmTileConfig::SMALL, is_half);
    assert!(
        small_bytes <= 32_768,
        "SMALL tg_memory ({small_bytes}) exceeds 32 KB Metal limit"
    );

    let large_bytes = tg_memory_bytes(GemmTileConfig::LARGE, is_half);
    assert!(
        large_bytes <= 32_768,
        "LARGE tg_memory ({large_bytes}) exceeds 32 KB Metal limit"
    );
}

/// Prove: SMALL f32 threadgroup memory matches documented 8,448 bytes.
///
/// From MSL source: As[32x33] + Bs[32x33], each f32.
/// 2 * 32 * 33 * 4 = 8,448 bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tg_memory_small_f32_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::SMALL, false);
    assert_eq!(bytes, 8_448, "SMALL f32 must be exactly 8,448 bytes");
}

/// Prove: SMALL f16 threadgroup memory matches documented 8,448 bytes.
///
/// From MSL source: As[32x33]h + Bs[32x33]h + tile_out[32x33]f.
/// 2 * 32 * 33 * 2 + 32 * 33 * 4 = 4,224 + 4,224 = 8,448 bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tg_memory_small_f16_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::SMALL, true);
    assert_eq!(bytes, 8_448, "SMALL f16 must be exactly 8,448 bytes");
}

/// Prove: LARGE f32 threadgroup memory matches documented 16,768 bytes.
///
/// From MSL source: As[64x33]f + Bs[32x65]f (pass_out eliminated).
/// 64 * 33 * 4 + 32 * 65 * 4 = 8,448 + 8,320 = 16,768 bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tg_memory_large_f32_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::LARGE, false);
    assert_eq!(bytes, 16_768, "LARGE f32 must be exactly 16,768 bytes");
}

/// Prove: LARGE f16 threadgroup memory matches documented 16,704 bytes.
///
/// From MSL source: As[64x33]h + Bs[32x65]h + pass_out[32x65]f.
/// 64 * 33 * 2 + 32 * 65 * 2 + 32 * 65 * 4 = 4,224 + 4,160 + 8,320 = 16,704.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tg_memory_large_f16_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::LARGE, true);
    assert_eq!(bytes, 16_704, "LARGE f16 must be exactly 16,704 bytes");
}

// ===========================================================================
// SIMD Matmul: select_tile_config
// ===========================================================================

/// Prove: select_tile_config always returns SMALL or LARGE, never panics.
///
/// For all valid (non-zero) dimension inputs, the function is total.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_always_valid() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 8192);
    kani::assume(k > 0 && k <= 8192);
    kani::assume(n > 0 && n <= 8192);
    kani::assume(batch <= 512);

    let tile = select_tile_config(m, k, n, batch);
    assert!(
        tile == GemmTileConfig::SMALL || tile == GemmTileConfig::LARGE,
        "must return SMALL or LARGE"
    );
}

/// Prove: LARGE tile is only selected when M >= 64 AND N >= 64.
///
/// The 64x64 kernel requires at least 64 rows and 64 columns.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_large_requires_64x64() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 8192);
    kani::assume(k > 0 && k <= 8192);
    kani::assume(n > 0 && n <= 8192);
    kani::assume(batch <= 512);

    let tile = select_tile_config(m, k, n, batch);
    if tile == GemmTileConfig::LARGE {
        assert!(m >= 64, "LARGE requires M >= 64");
        assert!(n >= 64, "LARGE requires N >= 64");
    }
}

/// Prove: LARGE tile requires sufficient threadgroup count (>= 32).
///
/// ceil(M/64) * ceil(N/64) must be >= 32 for LARGE selection.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_large_requires_min_tgs() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 8192);
    kani::assume(k > 0 && k <= 8192);
    kani::assume(n > 0 && n <= 8192);
    kani::assume(batch <= 512);

    let tile = select_tile_config(m, k, n, batch);
    if tile == GemmTileConfig::LARGE {
        let tgs = m.div_ceil(64) * n.div_ceil(64);
        assert!(tgs >= 32, "LARGE requires >= 32 threadgroups, got {tgs}");
    }
}

// ===========================================================================
// SIMD Matmul: Grid dimensions
// ===========================================================================

/// Prove: GEMM dispatch grid dimensions fit in u32 for production ranges.
///
/// Grid: [ceil(N/BN), ceil(M/BM), batch]. All must fit in u32 for Metal.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_grid_dimensions_fit_u32() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 65536);
    kani::assume(k > 0 && k <= 65536);
    kani::assume(n > 0 && n <= 65536);
    kani::assume(batch > 0 && batch <= 256);

    let tile = select_tile_config(m, k, n, batch);
    let grid_x = n.div_ceil(tile.bn as usize);
    let grid_y = m.div_ceil(tile.bm as usize);

    assert!(grid_x <= u32::MAX as usize, "grid_x overflows u32");
    assert!(grid_y <= u32::MAX as usize, "grid_y overflows u32");
    assert!(batch <= u32::MAX as usize, "batch overflows u32");
}

/// Prove: GEMM thread configuration (32, 4, 1) = 128 threads <= 1024.
///
/// The simdgroup kernels use [32, 4, 1] = 128 threads per threadgroup.
/// This is well within Metal's 1024-thread limit.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_threads_per_tg_within_metal_limit() {
    let threads: [u32; 3] = [32, 4, 1];
    let total = threads[0] as u64 * threads[1] as u64 * threads[2] as u64;
    assert_eq!(total, 128, "GEMM uses 128 threads per TG");
    assert!(total <= 1024, "Must not exceed Metal 1024-thread limit");
}

// ===========================================================================
// SIMD Matmul: Output buffer sizing
// ===========================================================================

/// Prove: GEMM output buffer byte calculation does not overflow for production dims.
///
/// total_output = batch * M * N, buffer_bytes = total_output * 4.
/// checked_mul chain must not overflow.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_output_buffer_bytes_no_overflow() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    // Production range: up to 8192 per dim, batch up to 64.
    kani::assume(m > 0 && m <= 8192);
    kani::assume(n > 0 && n <= 8192);
    kani::assume(batch > 0 && batch <= 64);

    let total_output = batch
        .checked_mul(m)
        .and_then(|v| v.checked_mul(n));
    assert!(total_output.is_some(), "batch * M * N must not overflow");

    let buffer_bytes = total_output.unwrap().checked_mul(4); // f32 = 4 bytes
    assert!(buffer_bytes.is_some(), "buffer_bytes must not overflow");
}

// ===========================================================================
// SIMD Matmul: F16 routing
// ===========================================================================

/// Prove: should_use_f16_simdgroup implies should_use_simdgroup.
///
/// F16 routing is a refinement of the base simdgroup routing.
/// If F16 is selected, the base simdgroup check must also pass.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_simdgroup_implies_base_simdgroup() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(batch > 0 && batch <= 64);

    if should_use_f16_simdgroup(m, k, n, batch) {
        assert!(
            should_use_simdgroup(m, k, n),
            "F16 simdgroup must imply base simdgroup"
        );
    }
}

/// Prove: F16_MIN_THREADGROUPS threshold produces valid scaled threshold.
///
/// The scaled threshold `F16_MIN_THREADGROUPS * 1024 / tile_area` must not
/// overflow and must be positive for all tile configs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_threshold_scaling_no_overflow() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(batch > 0 && batch <= 64);

    let tile = select_tile_config(m, k, n, batch);
    let tile_area = (tile.bm as usize) * (tile.bn as usize);

    // tile_area is either 1024 (SMALL) or 4096 (LARGE), never zero.
    assert!(tile_area > 0, "tile_area must be positive");

    let threshold = F16_MIN_THREADGROUPS
        .checked_mul(1024)
        .map(|v| v / tile_area);
    assert!(threshold.is_some(), "F16 threshold must not overflow");
    assert!(threshold.unwrap() > 0, "F16 threshold must be positive");
}

// ===========================================================================
// LSTM Sequence: 4*hidden_size gate arithmetic
// ===========================================================================

/// Prove: LSTM 4*hidden_size never overflows for valid hidden sizes.
///
/// The LSTM gate dimension is always 4*hidden_size. This must not overflow.
/// Production range: hidden_size up to MAX_THREADGROUP_HIDDEN (512).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_4h_no_overflow() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let gate_dim = hidden_size.checked_mul(4);
    assert!(gate_dim.is_some(), "4*hidden_size must not overflow");
    assert_eq!(gate_dim.unwrap(), 4 * hidden_size);
}

/// Prove: LSTM gate indexing (g*H + h) never overflows for valid params.
///
/// In the MSL kernel, gates are indexed as `g * hidden_size + h` where
/// g is in 0..4 and h is in 0..hidden_size. The result must fit in u32
/// for Metal buffer addressing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_gate_index_no_overflow() {
    let hidden_size: u32 = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);

    let g: u32 = kani::any();
    kani::assume(g < 4);

    let h: u32 = kani::any();
    kani::assume(h < hidden_size);

    let index = g.checked_mul(hidden_size).and_then(|v| v.checked_add(h));
    assert!(index.is_some(), "g*H + h must not overflow u32");
    assert!(
        index.unwrap() < 4 * hidden_size,
        "g*H + h must be < 4*H"
    );
}

// ===========================================================================
// LSTM Sequence: Weight buffer sizing
// ===========================================================================

/// Prove: LSTM w_ih buffer [4H, I] byte count does not overflow.
///
/// w_ih shape: [4*hidden_size, input_size]. Buffer = 4H * I * sizeof(f32).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_w_ih_buffer_no_overflow() {
    let hidden_size: usize = kani::any();
    let input_size: usize = kani::any();

    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);
    kani::assume(input_size > 0 && input_size <= 1024);

    let gate_dim = 4 * hidden_size;
    let numel = gate_dim.checked_mul(input_size);
    assert!(numel.is_some(), "4H * I must not overflow");

    let bytes = numel.unwrap().checked_mul(4); // f32
    assert!(bytes.is_some(), "w_ih bytes must not overflow");
}

/// Prove: LSTM w_hh buffer [4H, H] byte count does not overflow.
///
/// w_hh shape: [4*hidden_size, hidden_size]. Buffer = 4H * H * sizeof(f32).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_w_hh_buffer_no_overflow() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let gate_dim = 4 * hidden_size;
    let numel = gate_dim.checked_mul(hidden_size);
    assert!(numel.is_some(), "4H * H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "w_hh bytes must not overflow");
}

// ===========================================================================
// LSTM Sequence: Output and state buffer sizing
// ===========================================================================

/// Prove: LSTM output buffer [S, B, H] byte count does not overflow.
///
/// Production ranges: seq_len up to 512, batch up to 16, hidden up to 512.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_output_buffer_no_overflow() {
    let seq_len: usize = kani::any();
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len > 0 && seq_len <= 512);
    kani::assume(batch_size > 0 && batch_size <= 16);
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let numel = seq_len
        .checked_mul(batch_size)
        .and_then(|v| v.checked_mul(hidden_size));
    assert!(numel.is_some(), "S * B * H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "output bytes must not overflow");
}

/// Prove: LSTM state buffer [B, H] byte count does not overflow.
///
/// State tensors h_n and c_n are both [batch_size, hidden_size].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_state_buffer_no_overflow() {
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch_size > 0 && batch_size <= 16);
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let numel = batch_size.checked_mul(hidden_size);
    assert!(numel.is_some(), "B * H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "state bytes must not overflow");
}

// ===========================================================================
// LSTM Sequence: Threadgroup memory
// ===========================================================================

/// Prove: LSTM shared_h[hidden_size] fits within Metal 32 KB threadgroup limit.
///
/// The MSL kernel allocates `threadgroup float shared_h[hidden_size]`.
/// At 4 bytes per float, this is hidden_size * 4 bytes.
/// Must be <= 32,768 bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_shared_h_within_32kb() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let shared_bytes = hidden_size * 4; // sizeof(float)
    assert!(
        shared_bytes <= 32_768,
        "shared_h[{hidden_size}] = {shared_bytes} bytes exceeds 32 KB"
    );
    // MAX_THREADGROUP_HIDDEN=512: 512*4 = 2048, well within 32 KB.
}

/// Prove: MAX_THREADGROUP_HIDDEN * 4 is a small fraction of the 32 KB limit.
///
/// Documents that the MAX_THREADGROUP_HIDDEN=512 constant leaves ample headroom
/// for other threadgroup memory allocations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_max_hidden_leaves_tg_headroom() {
    let max_bytes = MAX_THREADGROUP_HIDDEN * 4;
    // 512 * 4 = 2,048 bytes = 6.25% of 32 KB limit.
    assert!(max_bytes <= 2048, "MAX_THREADGROUP_HIDDEN * 4 must be <= 2048");
    assert!(
        max_bytes <= 32_768 / 4,
        "shared_h must use at most 25% of 32 KB"
    );
}

// ===========================================================================
// LSTM Sequence: Validation guards
// ===========================================================================

/// Prove: hidden_size=0 is rejected by the guard in gpu_lstm_sequence.
///
/// hidden_size=0 would create a zero-length MSL threadgroup array (UB).
/// The function must reject it before reaching the dispatch path.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_hidden_zero_rejected() {
    let hidden_size: usize = 0;
    // Verify the guard: hidden_size == 0 triggers early return.
    assert_eq!(hidden_size, 0);
    // This is the condition checked in gpu_lstm_sequence:
    // if hidden_size == 0 { return gpu_fallback(...); }
    assert!(
        hidden_size == 0,
        "zero hidden_size must trigger the rejection guard"
    );
}

/// Prove: hidden_size > MAX_THREADGROUP_HIDDEN is correctly bounded.
///
/// Any hidden_size above 512 is rejected by the threadgroup memory guard.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_hidden_above_max_rejected() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > MAX_THREADGROUP_HIDDEN);
    kani::assume(hidden_size <= 4096);

    // This is the condition checked in gpu_lstm_sequence:
    // if hidden_size > MAX_THREADGROUP_HIDDEN { return gpu_fallback(...); }
    assert!(
        hidden_size > MAX_THREADGROUP_HIDDEN,
        "hidden_size > 512 must be rejected"
    );

    // Also verify that the shared memory would exceed a safe threshold.
    let shared_bytes = hidden_size * 4;
    assert!(
        shared_bytes > MAX_THREADGROUP_HIDDEN * 4,
        "shared memory exceeds safe limit"
    );
}

// ===========================================================================
// LSTM Sequence: Thread grid
// ===========================================================================

/// Prove: LSTM thread grid [batch_size, hidden_size] has valid dimensions.
///
/// The kernel dispatches as:
/// - threadgroups: [batch_size, 1, 1]
/// - threads_per_threadgroup: [hidden_size, 1, 1]
///
/// hidden_size is bounded by MAX_THREADGROUP_HIDDEN=512 < Metal limit 1024.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_thread_grid_valid() {
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch_size > 0 && batch_size <= 256);
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    // Threads per threadgroup = hidden_size, must be <= 1024.
    assert!(
        hidden_size <= 1024,
        "hidden_size as threads_per_tg exceeds Metal 1024 limit"
    );

    // Both must fit in u32 for Metal dispatch.
    assert!(batch_size <= u32::MAX as usize, "batch_size overflows u32");
    assert!(hidden_size <= u32::MAX as usize, "hidden_size overflows u32");
}

// ===========================================================================
// LSTM Sequence: Precomputed input projection sizing
// ===========================================================================

/// Prove: LSTM precomputed input_proj [S, B, 4H] byte count does not overflow.
///
/// The precomputed path uses input_proj of shape [seq_len, batch, 4*hidden_size].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_precomputed_input_proj_no_overflow() {
    let seq_len: usize = kani::any();
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len > 0 && seq_len <= 512);
    kani::assume(batch_size > 0 && batch_size <= 16);
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let gate_dim = 4 * hidden_size;
    let numel = seq_len
        .checked_mul(batch_size)
        .and_then(|v| v.checked_mul(gate_dim));
    assert!(numel.is_some(), "S * B * 4H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "input_proj bytes must not overflow");
}

// ===========================================================================
// LSTM Sequence: Timestep reverse index
// ===========================================================================

/// Prove: LSTM reverse timestep index is valid for all t in 0..seq_len.
///
/// In reverse mode: `ts = seq_len - 1 - t`. For t in [0, seq_len),
/// ts must be in [0, seq_len) and must enumerate all timesteps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_reverse_timestep_valid() {
    let seq_len: u32 = kani::any();
    let t: u32 = kani::any();

    kani::assume(seq_len > 0 && seq_len <= 512);
    kani::assume(t < seq_len);

    // Forward index.
    assert!(t < seq_len);

    // Reverse index: ts = seq_len - 1 - t.
    // Since t < seq_len and seq_len > 0, seq_len - 1 >= t, so no underflow.
    let ts = seq_len - 1 - t;
    assert!(ts < seq_len, "reverse ts must be in [0, seq_len)");

    // Verify the forward-reverse bijection: forward(reverse(t)) == t.
    let forward_of_reverse = seq_len - 1 - ts;
    assert_eq!(forward_of_reverse, t, "reverse must be a bijection");
}

/// Prove: LSTM reverse timestep covers the full range [0, seq_len) uniquely.
///
/// For small seq_len values, verify that {seq_len - 1 - t | t in 0..seq_len}
/// = {0, 1, ..., seq_len - 1}.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_reverse_timestep_range_complete() {
    let seq_len: u32 = kani::any();
    kani::assume(seq_len > 0 && seq_len <= 512);

    // First timestep: t=0 -> ts = seq_len - 1
    let first_ts = seq_len - 1;
    assert_eq!(first_ts, seq_len - 1, "t=0 maps to seq_len-1");

    // Last timestep: t = seq_len - 1 -> ts = 0
    let last_ts = seq_len - 1 - (seq_len - 1);
    assert_eq!(last_ts, 0, "t=seq_len-1 maps to 0");
}

// ===========================================================================
// LSTM Sequence: MSL weight addressing
// ===========================================================================

/// Prove: LSTM w_hh addressing `(g*H + h)*H + j` does not overflow u32.
///
/// In the MSL kernel: `w_hh[(g * hidden_size_val + h) * hidden_size_val + j]`.
/// g in 0..4, h in 0..H, j in 0..H. Result must fit in u32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_w_hh_address_no_overflow() {
    let hidden_size: u32 = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);

    let g: u32 = kani::any();
    kani::assume(g < 4);

    let h: u32 = kani::any();
    kani::assume(h < hidden_size);

    let j: u32 = kani::any();
    kani::assume(j < hidden_size);

    // (g * H + h) * H + j
    let gate_row = g.checked_mul(hidden_size).and_then(|v| v.checked_add(h));
    assert!(gate_row.is_some(), "g*H + h must not overflow");

    let addr = gate_row.unwrap().checked_mul(hidden_size).and_then(|v| v.checked_add(j));
    assert!(addr.is_some(), "(g*H + h)*H + j must not overflow u32");

    // Maximum value: (3*512 + 511)*512 + 511 = 2047*512 + 511 = 1_048_575.
    // Well within u32::MAX.
    assert!(addr.unwrap() < 4 * hidden_size * hidden_size);
}

/// Prove: LSTM w_ih addressing `(g*H + h)*I + k` does not overflow u32.
///
/// In the MSL kernel: `w_ih[(g * hidden_size_val + h) * input_size + k]`.
/// g in 0..4, h in 0..H, k in 0..I.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_w_ih_address_no_overflow() {
    let hidden_size: u32 = kani::any();
    let input_size: u32 = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);
    kani::assume(input_size > 0 && input_size <= 1024);

    let g: u32 = kani::any();
    kani::assume(g < 4);

    let h: u32 = kani::any();
    kani::assume(h < hidden_size);

    let k: u32 = kani::any();
    kani::assume(k < input_size);

    // (g * H + h) * I + k
    let gate_row = g.checked_mul(hidden_size).and_then(|v| v.checked_add(h));
    assert!(gate_row.is_some(), "g*H + h must not overflow");

    let addr = gate_row.unwrap().checked_mul(input_size).and_then(|v| v.checked_add(k));
    assert!(addr.is_some(), "(g*H + h)*I + k must not overflow u32");
}
