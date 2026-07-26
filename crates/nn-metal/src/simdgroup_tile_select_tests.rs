// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for adaptive GEMM tile selection (#3479).
//!
//! Validates tile routing for common Kokoro shapes, boundary conditions,
//! and shape class detection (tall-skinny, wide, small-K, tiny).

use super::{
    is_scalar_fallback, select_gemm_tiles, TileConfig, SIMDGROUP_ALIGN, SMALL_K_THRESHOLD,
    TALL_SKINNY_RATIO, TINY_THRESHOLD, WIDE_RATIO,
};

// ---------------------------------------------------------------------------
// Kokoro model shapes
// ---------------------------------------------------------------------------

/// LSTM recurrent step: M=1, K=640, N=256.
/// M*N=256 < 1024 → tiny, scalar fallback.
#[test]
fn test_kokoro_lstm_recurrent_step() {
    assert!(select_gemm_tiles(1, 640, 256).is_none());
}

/// Conv1d GEMM: M=8, K=256, N=512.
/// M*N=4096 >= 1024. M=8, N=512, N/M=64 >= 4, both aligned → WIDE.
#[test]
fn test_kokoro_conv1d_gemm() {
    let cfg = select_gemm_tiles(8, 256, 512).unwrap();
    assert_eq!(cfg, TileConfig::WIDE);
    assert_eq!(cfg.tile_m, 16);
    assert_eq!(cfg.tile_n, 64);
}

/// Style projection: M=1, K=768, N=512.
/// M*N=512 < 1024 → tiny, scalar fallback.
#[test]
fn test_kokoro_style_projection() {
    assert!(select_gemm_tiles(1, 768, 512).is_none());
}

/// Large matmul: M=256, K=768, N=768.
/// M*N=196,608 >= 1024. K > 32. M/N < 4, N/M < 4 → SQUARE.
#[test]
fn test_kokoro_large_matmul() {
    let cfg = select_gemm_tiles(256, 768, 768).unwrap();
    assert_eq!(cfg, TileConfig::SQUARE);
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
    assert_eq!(cfg.tile_k, 32);
}

// ---------------------------------------------------------------------------
// Tiny threshold boundary
// ---------------------------------------------------------------------------

/// M*N = 1023 < 1024 → None (scalar fallback).
#[test]
fn test_tiny_below_threshold() {
    assert!(select_gemm_tiles(31, 256, 33).is_none());
    assert!(select_gemm_tiles(1, 128, 1023).is_none());
}

/// M*N = 1024 exactly → Some (not scalar fallback).
#[test]
fn test_tiny_at_threshold() {
    let cfg = select_gemm_tiles(32, 256, 32);
    assert!(cfg.is_some());
}

/// M*N = 1025 → Some.
#[test]
fn test_tiny_above_threshold() {
    assert!(select_gemm_tiles(32, 128, 33).is_some());
}

// ---------------------------------------------------------------------------
// Small K routing
// ---------------------------------------------------------------------------

/// K=16 (<= 32): tile_k should be rounded up to 16 (already aligned to 8).
#[test]
fn test_small_k_aligned() {
    let cfg = select_gemm_tiles(64, 16, 64).unwrap();
    assert_eq!(cfg.tile_k, 16);
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
}

/// K=24 (<= 32): tile_k should be rounded up to 24 (next_multiple_of(8)=24).
#[test]
fn test_small_k_unaligned() {
    let cfg = select_gemm_tiles(64, 24, 64).unwrap();
    assert_eq!(cfg.tile_k, 24);
}

/// K=32 exactly (boundary): still small-K path.
#[test]
fn test_small_k_at_threshold() {
    let cfg = select_gemm_tiles(64, 32, 64).unwrap();
    assert_eq!(cfg.tile_k, 32);
    assert_eq!(cfg.tile_m, 32);
}

/// K=33 (> 32): NOT small-K path → standard routing.
#[test]
fn test_k_above_threshold() {
    let cfg = select_gemm_tiles(64, 33, 64).unwrap();
    // Standard square, tile_k = 32 (the default).
    assert_eq!(cfg.tile_k, 32);
}

// ---------------------------------------------------------------------------
// Tall-skinny routing
// ---------------------------------------------------------------------------

/// M=256, N=32 → M/N=8 >= 4, both aligned → TALL_SKINNY.
#[test]
fn test_tall_skinny_basic() {
    let cfg = select_gemm_tiles(256, 512, 32).unwrap();
    assert_eq!(cfg, TileConfig::TALL_SKINNY);
    assert_eq!(cfg.tile_m, 64);
    assert_eq!(cfg.tile_n, 16);
}

/// M=128, N=32 → M/N=4, exactly at ratio → TALL_SKINNY.
#[test]
fn test_tall_skinny_exact_ratio() {
    let cfg = select_gemm_tiles(128, 256, 32).unwrap();
    assert_eq!(cfg, TileConfig::TALL_SKINNY);
}

/// M=120, N=32 → M/N=3 < 4, NOT tall-skinny → SQUARE.
/// (But K=256 > 32, so not small-K either.)
#[test]
fn test_not_tall_skinny_below_ratio() {
    let cfg = select_gemm_tiles(120, 256, 48).unwrap();
    assert_eq!(cfg, TileConfig::SQUARE);
}

// ---------------------------------------------------------------------------
// Wide routing
// ---------------------------------------------------------------------------

/// M=32, N=256 → N/M=8 >= 4, both aligned → WIDE.
#[test]
fn test_wide_basic() {
    let cfg = select_gemm_tiles(32, 512, 256).unwrap();
    assert_eq!(cfg, TileConfig::WIDE);
    assert_eq!(cfg.tile_m, 16);
    assert_eq!(cfg.tile_n, 64);
}

/// M=32, N=128 → N/M=4, exactly at ratio → WIDE.
#[test]
fn test_wide_exact_ratio() {
    let cfg = select_gemm_tiles(32, 256, 128).unwrap();
    assert_eq!(cfg, TileConfig::WIDE);
}

/// M=48, N=128 → N/M=2 < 4, NOT wide → SQUARE.
#[test]
fn test_not_wide_below_ratio() {
    let cfg = select_gemm_tiles(48, 256, 128).unwrap();
    assert_eq!(cfg, TileConfig::SQUARE);
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

#[test]
fn test_output_per_threadgroup() {
    assert_eq!(TileConfig::SQUARE.output_per_threadgroup(), 1024);
    assert_eq!(TileConfig::TALL_SKINNY.output_per_threadgroup(), 1024);
    assert_eq!(TileConfig::WIDE.output_per_threadgroup(), 1024);
}

#[test]
fn test_threadgroup_count() {
    // 256 x 256 with 32x32 tiles → 8 x 8 = 64 TGs.
    assert_eq!(TileConfig::SQUARE.threadgroup_count(256, 256), 64);
    // 256 x 256 with 64x16 tiles → 4 x 16 = 64 TGs.
    assert_eq!(TileConfig::TALL_SKINNY.threadgroup_count(256, 256), 64);
    // Edge: not a multiple → rounds up.
    assert_eq!(TileConfig::SQUARE.threadgroup_count(33, 33), 4);
}

#[test]
fn test_threads_per_threadgroup() {
    assert_eq!(TileConfig::SQUARE.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::TALL_SKINNY.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::WIDE.threads_per_threadgroup(), 128);
}

#[test]
fn test_is_scalar_fallback() {
    assert!(is_scalar_fallback(1, 256));   // 256 < 1024
    assert!(is_scalar_fallback(31, 33));   // 1023 < 1024
    assert!(!is_scalar_fallback(32, 32));  // 1024 == threshold
    assert!(!is_scalar_fallback(64, 64));  // 4096 > 1024
}

// ---------------------------------------------------------------------------
// Constants sanity
// ---------------------------------------------------------------------------

#[test]
fn test_constants() {
    assert_eq!(SIMDGROUP_ALIGN, 8);
    assert_eq!(TINY_THRESHOLD, 1024);
    assert_eq!(SMALL_K_THRESHOLD, 32);
    assert_eq!(TALL_SKINNY_RATIO, 4);
    assert_eq!(WIDE_RATIO, 4);
}

// ---------------------------------------------------------------------------
// Priority ordering: small-K checked before tall/wide
// ---------------------------------------------------------------------------

/// K=16 with tall-skinny shape: small-K takes priority over tall-skinny.
#[test]
fn test_small_k_takes_priority_over_tall_skinny() {
    // M=256, K=16, N=32 → K <= 32 so small-K wins.
    let cfg = select_gemm_tiles(256, 16, 32).unwrap();
    assert_eq!(cfg.tile_k, 16);
    assert_eq!(cfg.tile_m, 32); // square, not 64x16
}

/// K=24 with wide shape: small-K takes priority over wide.
#[test]
fn test_small_k_takes_priority_over_wide() {
    // M=32, K=24, N=256 → K <= 32 so small-K wins.
    let cfg = select_gemm_tiles(32, 24, 256).unwrap();
    assert_eq!(cfg.tile_k, 24);
    assert_eq!(cfg.tile_m, 32);
}

// ---------------------------------------------------------------------------
// Tile validity invariants (runtime counterparts of Kani proofs, #3549)
// ---------------------------------------------------------------------------

/// All returned tile configs have dimensions that are multiples of SIMDGROUP_ALIGN.
/// Runtime sweep over representative shapes.
#[test]
fn test_tile_alignment_sweep() {
    let shapes: &[(usize, usize, usize)] = &[
        (64, 64, 64),
        (8, 256, 512),
        (256, 512, 32),
        (32, 512, 256),
        (256, 768, 768),
        (1024, 1024, 1024),
        (32, 16, 32),
        (64, 24, 64),
        (128, 8, 128),
    ];
    for &(m, k, n) in shapes {
        if let Some(cfg) = select_gemm_tiles(m, k, n) {
            assert_eq!(
                cfg.tile_m % SIMDGROUP_ALIGN, 0,
                "tile_m not aligned for ({m},{k},{n})"
            );
            assert_eq!(
                cfg.tile_n % SIMDGROUP_ALIGN, 0,
                "tile_n not aligned for ({m},{k},{n})"
            );
            assert_eq!(
                cfg.tile_k % SIMDGROUP_ALIGN, 0,
                "tile_k not aligned for ({m},{k},{n})"
            );
        }
    }
}

/// Threads per threadgroup never exceeds Metal's 1024-thread limit.
#[test]
fn test_threads_within_metal_limit() {
    let shapes: &[(usize, usize, usize)] = &[
        (64, 256, 64),
        (256, 768, 768),
        (1024, 1024, 1024),
        (8, 256, 512),
        (256, 512, 32),
    ];
    for &(m, k, n) in shapes {
        if let Some(cfg) = select_gemm_tiles(m, k, n) {
            let threads = cfg.threads_per_threadgroup();
            assert!(
                threads <= 1024,
                "threads {threads} > 1024 for ({m},{k},{n})"
            );
        }
    }
}

/// threadgroup_count does not overflow for large-but-valid shapes.
#[test]
fn test_threadgroup_count_no_overflow() {
    let shapes: &[(usize, usize, usize)] = &[
        (4096, 4096, 4096),
        (2048, 768, 3072),
        (1, 640, 256),
    ];
    for &(m, k, n) in shapes {
        if let Some(cfg) = select_gemm_tiles(m, k, n) {
            let tg_m = m.div_ceil(cfg.tile_m);
            let tg_n = n.div_ceil(cfg.tile_n);
            let tg_count = tg_m.checked_mul(tg_n);
            assert!(
                tg_count.is_some(),
                "TG count overflow for ({m},{k},{n})"
            );
        }
    }
}

/// is_scalar_fallback and select_gemm_tiles are consistent across shapes.
#[test]
fn test_scalar_fallback_consistency() {
    let shapes: &[(usize, usize, usize)] = &[
        (1, 640, 256),    // tiny
        (31, 256, 33),    // tiny
        (32, 256, 32),    // at threshold
        (64, 256, 64),    // above
        (256, 768, 768),  // large
    ];
    for &(m, k, n) in shapes {
        let fallback = is_scalar_fallback(m, n);
        let tiles = select_gemm_tiles(m, k, n);
        if fallback {
            assert!(tiles.is_none(), "fallback=true but Some for ({m},{k},{n})");
        }
        if tiles.is_some() {
            assert!(!fallback, "Some but fallback=true for ({m},{k},{n})");
        }
    }
}

/// Small-K tile_k is always >= K and properly rounded.
#[test]
fn test_small_k_tile_k_bounds() {
    for k in 1..=SMALL_K_THRESHOLD {
        // M*N must be >= TINY_THRESHOLD for non-None result.
        let cfg = select_gemm_tiles(64, k, 64);
        if let Some(cfg) = cfg {
            assert!(cfg.tile_k >= k, "tile_k < K for k={k}");
            assert_eq!(cfg.tile_k % SIMDGROUP_ALIGN, 0, "tile_k not aligned for k={k}");
            // Minimal rounding.
            assert!(
                cfg.tile_k <= k + SIMDGROUP_ALIGN - 1,
                "excessive rounding for k={k}: tile_k={}", cfg.tile_k,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Kokoro Conv1d GEMM shape detection (#4264)
// ---------------------------------------------------------------------------

use super::{
    is_kokoro_conv1d_shape, select_conv1d_gemm_tiles, Conv1dGemmShape, KOKORO_SMALL_M_THRESHOLD,
};

/// Kokoro 512x512xK3 is recognized.
#[test]
fn test_kokoro_conv1d_shape_512_512_k3() {
    assert!(is_kokoro_conv1d_shape(512, 512, 3));
}

/// Kokoro 256x512xK3 (upsampling path output) is recognized.
#[test]
fn test_kokoro_conv1d_shape_512_256_k3() {
    assert!(is_kokoro_conv1d_shape(256, 512, 3));
}

/// Kokoro 128x256xK3 is recognized.
#[test]
fn test_kokoro_conv1d_shape_256_128_k3() {
    assert!(is_kokoro_conv1d_shape(128, 256, 3));
}

/// Non-Kokoro shapes are not recognized.
#[test]
fn test_non_kokoro_conv1d_shapes() {
    assert!(!is_kokoro_conv1d_shape(64, 64, 3));
    assert!(!is_kokoro_conv1d_shape(512, 512, 5));
    assert!(!is_kokoro_conv1d_shape(256, 256, 3));
}

/// Conv1dGemmShape GEMM dimensions are correct.
#[test]
fn test_conv1d_gemm_shape_dims() {
    let s = Conv1dGemmShape::KOKORO_512_512_K3;
    assert_eq!(s.gemm_m(), 512);
    assert_eq!(s.gemm_k(), 1536); // 512 * 3
    assert!(s.supports_direct_conv());

    let s = Conv1dGemmShape::KOKORO_512_256_K3;
    assert_eq!(s.gemm_m(), 256);
    assert_eq!(s.gemm_k(), 1536); // 512 * 3

    let s = Conv1dGemmShape::KOKORO_256_128_K3;
    assert_eq!(s.gemm_m(), 128);
    assert_eq!(s.gemm_k(), 768); // 256 * 3
}

/// select_conv1d_gemm_tiles returns square tiles for short sequences.
#[test]
fn test_conv1d_gemm_tiles_short_sequence() {
    let shape = Conv1dGemmShape::KOKORO_512_512_K3;
    // L_out=32 < KOKORO_SMALL_M_THRESHOLD=64 → SQUARE
    let cfg = select_conv1d_gemm_tiles(&shape, 32).unwrap();
    assert_eq!(cfg, TileConfig::SQUARE);
}

/// select_conv1d_gemm_tiles returns tall-skinny tiles when M >> L_out.
#[test]
fn test_conv1d_gemm_tiles_tall_skinny_sequence() {
    let shape = Conv1dGemmShape::KOKORO_512_512_K3;
    // L_out=128, M=512, ratio=512/128=4 → TALL_SKINNY (M >> L_out)
    let cfg = select_conv1d_gemm_tiles(&shape, 128).unwrap();
    assert_eq!(cfg, TileConfig::TALL_SKINNY);
    assert_eq!(cfg.tile_m, 64);
    assert_eq!(cfg.tile_n, 16);
}

/// select_conv1d_gemm_tiles returns square tiles for balanced sequences.
#[test]
fn test_conv1d_gemm_tiles_balanced_sequence() {
    let shape = Conv1dGemmShape::KOKORO_512_512_K3;
    // L_out=256, M=512, ratio=2 < 4 → neither tall-skinny nor wide → square
    let cfg = select_conv1d_gemm_tiles(&shape, 256).unwrap();
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
}

/// select_conv1d_gemm_tiles returns wide tiles for long sequences.
#[test]
fn test_conv1d_gemm_tiles_long_sequence() {
    let shape = Conv1dGemmShape::KOKORO_256_128_K3;
    // M=128, L_out=1024, ratio=8 → WIDE (if aligned)
    let cfg = select_conv1d_gemm_tiles(&shape, 1024).unwrap();
    assert_eq!(cfg, TileConfig::WIDE);
}

/// select_conv1d_gemm_tiles rejects tiny outputs.
#[test]
fn test_conv1d_gemm_tiles_tiny() {
    let shape = Conv1dGemmShape::KOKORO_256_128_K3;
    // M=128, L_out=4 → M*N=512 < TINY_THRESHOLD → None
    assert!(select_conv1d_gemm_tiles(&shape, 4).is_none());
}

/// KOKORO_SMALL_M_THRESHOLD constant sanity check.
#[test]
fn test_kokoro_small_m_threshold() {
    assert_eq!(KOKORO_SMALL_M_THRESHOLD, 64);
}
