// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Adaptive GEMM tile selection for Metal simdgroup matmul (#3479).
//!
//! Selects optimal tile dimensions based on M, K, N shape analysis:
//!
//! | Shape class     | Tile (BM x BN) | BK | Threadgroup       | Output/TG |
//! |-----------------|----------------|----|-------------------|-----------|
//! | Large square    | 32 x 32        | 32 | [32, 4, 1] (128)  | 1,024     |
//! | Tall-skinny     | 64 x 16        | 32 | [32, 4, 1] (128)  | 1,024     |
//! | Wide            | 16 x 64        | 32 | [32, 4, 1] (128)  | 1,024     |
//! | Small K         | 32 x 32        | K  | [32, 4, 1] (128)  | 1,024     |
//! | Tiny (M*N<1024) | —              | —  | scalar fallback    | —         |
//!
//! The tall-skinny and wide configs cover more output in the dominant
//! dimension per threadgroup, improving L1/L2 utilization when one
//! dimension greatly exceeds the other. Small-K avoids unnecessary
//! K-loop iterations when K fits in a single tile. Tiny matrices fall
//! back to scalar dispatch (returns `None`).
//!
//! Note: The existing `GemmTileConfig::LARGE` (64x64, BK=32) from the
//! matmul_simd module handles the very-large-square case and produces
//! 4,096 output elements per TG. This module provides finer-grained
//! routing for non-square and smaller shapes where 64x64 is not optimal.

/// Minimum alignment for simdgroup_matrix dimensions.
///
/// All tile dimensions (BM, BN, BK) and matrix dimensions (M, K, N)
/// must be multiples of this value for `simdgroup_matrix<T, 8, 8>`.
pub(crate) const SIMDGROUP_ALIGN: usize = 8;

/// Minimum M * N product below which simdgroup dispatch is not worthwhile.
/// Dispatch overhead dominates compute at this scale; use scalar fallback.
pub(crate) const TINY_THRESHOLD: usize = 1024;

/// Minimum M * N for the standard simdgroup path (existing threshold from
/// `should_use_simdgroup`). Below this but above `TINY_THRESHOLD`, shapes
/// are eligible for narrow tile configs but not the standard 32x32 path.
pub(crate) const STANDARD_MN_THRESHOLD: usize = 16_384;

/// Minimum K for full simdgroup tiling benefit.
/// Below this, BK is clamped to K to eliminate the K-loop.
pub(crate) const SMALL_K_THRESHOLD: usize = 32;

/// Ratio threshold for tall-skinny detection: M / N >= this value.
pub(crate) const TALL_SKINNY_RATIO: usize = 4;

/// Ratio threshold for wide detection: N / M >= this value.
pub(crate) const WIDE_RATIO: usize = 4;

/// M threshold below which 8x8 simdgroup tiles are preferred for Kokoro
/// Conv1d GEMM shapes. Short sequences (M < 64) benefit from smaller tiles
/// that avoid under-filled threadgroups.
pub(crate) const KOKORO_SMALL_M_THRESHOLD: usize = 64;

/// GEMM tile configuration for adaptive simdgroup kernel dispatch.
///
/// Encodes the output tile dimensions (BM x BN), the K-dimension tile
/// width (BK), and the Metal threadgroup size. The kernel loops over
/// K in increments of `tile_k`, accumulating partial sums in registers
/// before writing the BM x BN output tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileConfig {
    /// Output tile rows (BM). Must be a multiple of [`SIMDGROUP_ALIGN`].
    pub(crate) tile_m: usize,
    /// Output tile columns (BN). Must be a multiple of [`SIMDGROUP_ALIGN`].
    pub(crate) tile_n: usize,
    /// K-dimension tile width (BK). Must be a multiple of [`SIMDGROUP_ALIGN`].
    /// When K <= `SMALL_K_THRESHOLD`, this equals K (no K-loop).
    pub(crate) tile_k: usize,
    /// Metal threadgroup size `[threads_x, threads_y, threads_z]`.
    pub(crate) threadgroup_size: [usize; 3],
}

impl TileConfig {
    /// Standard square tile: 32x32, BK=32, 128 threads.
    pub(crate) const SQUARE: Self = Self {
        tile_m: 32,
        tile_n: 32,
        tile_k: 32,
        threadgroup_size: [32, 4, 1],
    };

    /// Tall-skinny tile: 64x16, BK=32, 128 threads.
    /// Covers 64 M-rows per TG when M >> N.
    pub(crate) const TALL_SKINNY: Self = Self {
        tile_m: 64,
        tile_n: 16,
        tile_k: 32,
        threadgroup_size: [32, 4, 1],
    };

    /// Wide tile: 16x64, BK=32, 128 threads.
    /// Covers 64 N-columns per TG when N >> M.
    pub(crate) const WIDE: Self = Self {
        tile_m: 16,
        tile_n: 64,
        tile_k: 32,
        threadgroup_size: [32, 4, 1],
    };

    /// Total output elements per threadgroup: `tile_m * tile_n`.
    pub(crate) fn output_per_threadgroup(&self) -> usize {
        self.tile_m * self.tile_n
    }

    /// Number of threadgroups needed for a given M x N output.
    pub(crate) fn threadgroup_count(&self, m: usize, n: usize) -> usize {
        m.div_ceil(self.tile_m) * n.div_ceil(self.tile_n)
    }

    /// Number of threads per threadgroup (product of threadgroup_size).
    pub(crate) fn threads_per_threadgroup(&self) -> usize {
        self.threadgroup_size[0] * self.threadgroup_size[1] * self.threadgroup_size[2]
    }
}

// ---------------------------------------------------------------------------
// Kokoro generator Conv1d shape descriptors (#4264)
// ---------------------------------------------------------------------------

/// Describes a Conv1d GEMM shape for Kokoro-specific tile optimization.
///
/// Conv1d via im2col produces GEMM: [C_out, C_in*K] x [C_in*K, L_out].
/// The common Kokoro generator shapes (K=3, stride=1) are:
///   512x512xK3 → M=512, K=1536, N=L_out
///   512x256xK3 → M=256, K=1536, N=L_out (upsampling path output)
///   256x128xK3 → M=128, K=768,  N=L_out
///
/// Issue: #4264 (RTF 0.082 → 0.03)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Conv1dGemmShape {
    /// Output channels (M in GEMM).
    pub(crate) c_out: usize,
    /// Input channels (part of K in GEMM).
    pub(crate) c_in: usize,
    /// Kernel spatial size.
    pub(crate) kernel_size: usize,
}

impl Conv1dGemmShape {
    /// Kokoro generator most common: 512→512, K=3.
    pub(crate) const KOKORO_512_512_K3: Self = Self {
        c_out: 512,
        c_in: 512,
        kernel_size: 3,
    };

    /// Kokoro generator upsampling path: 512→256, K=3.
    pub(crate) const KOKORO_512_256_K3: Self = Self {
        c_out: 256,
        c_in: 512,
        kernel_size: 3,
    };

    /// Kokoro generator later stage: 256→128, K=3.
    pub(crate) const KOKORO_256_128_K3: Self = Self {
        c_out: 128,
        c_in: 256,
        kernel_size: 3,
    };

    /// GEMM M dimension (= c_out).
    pub(crate) fn gemm_m(&self) -> usize {
        self.c_out
    }

    /// GEMM K dimension (= c_in * kernel_size).
    pub(crate) fn gemm_k(&self) -> usize {
        self.c_in * self.kernel_size
    }

    /// Whether this shape benefits from the direct sliding-window kernel
    /// (avoids im2col allocation + blit for K=3, stride=1).
    pub(crate) fn supports_direct_conv(&self) -> bool {
        self.kernel_size == 3
    }
}

/// Select optimal GEMM tile config for a Conv1d GEMM shape.
///
/// Conv1d GEMM is weight[C_out, C_in*K] x col[C_in*K, L_out] → [C_out, L_out].
/// The K dimension is always large (768+), so we never hit the small-K path.
/// M varies (128-512), N=L_out varies with sequence length.
///
/// For Kokoro generator shapes, we optimize:
/// - Short sequences (L_out < 64): 32x32 tiles to avoid under-filled TGs
/// - Medium sequences (64 <= L_out < 256): 32x32 with full K-loop utilization
/// - Long sequences (L_out >= 256): 32x32 still optimal because M is moderate
///   (128-512) and K is large enough to saturate ALU
///
/// Issue: #4264
pub(crate) fn select_conv1d_gemm_tiles(
    shape: &Conv1dGemmShape,
    l_out: usize,
) -> Option<TileConfig> {
    let m = shape.gemm_m();
    let k = shape.gemm_k();
    let mn = m.saturating_mul(l_out);

    // Tiny: scalar fallback.
    if mn < TINY_THRESHOLD {
        return None;
    }

    // For Kokoro Conv1d shapes, K is always large (768-1536), so we get
    // excellent K-loop utilization with BK=32 (24-48 iterations).
    // The tile choice depends on M vs L_out aspect ratio.

    if l_out < KOKORO_SMALL_M_THRESHOLD {
        // Short sequence: M is larger than L_out.
        // Use standard 32x32 tiles — they fill well in the M dimension.
        return Some(TileConfig::SQUARE);
    }

    if m > 0 && l_out / m >= WIDE_RATIO && is_aligned(m) && is_aligned(l_out) {
        // L_out >> M: wide tile covers more output columns per TG.
        return Some(TileConfig::WIDE);
    }

    if l_out > 0 && m / l_out >= TALL_SKINNY_RATIO && is_aligned(m) && is_aligned(l_out) {
        // M >> L_out: tall-skinny tile covers more output rows per TG.
        return Some(TileConfig::TALL_SKINNY);
    }

    // Default: 32x32 square tiles with BK=32.
    // For K=1536 (512*3), this gives 48 K-loop iterations — enough to
    // keep simdgroup ALU saturated while amortizing shared memory loads.
    Some(TileConfig {
        tile_m: 32,
        tile_n: 32,
        tile_k: k.min(32).next_multiple_of(SIMDGROUP_ALIGN),
        threadgroup_size: [32, 4, 1],
    })
}

/// Returns `true` if the given Conv1d shape is a known Kokoro generator shape
/// that benefits from specialized tile selection.
///
/// Issue: #4264
pub(crate) fn is_kokoro_conv1d_shape(c_out: usize, c_in: usize, kernel_size: usize) -> bool {
    matches!(
        (c_out, c_in, kernel_size),
        (512, 512, 3) | (256, 512, 3) | (128, 256, 3)
    )
}

/// Select the optimal GEMM tile configuration for given dimensions.
///
/// Returns `None` when M * N < [`TINY_THRESHOLD`] (1024), indicating
/// the caller should use scalar dispatch instead of simdgroup tiling.
///
/// # Shape-aware routing
///
/// 1. **Tiny** (M * N < 1024): `None` — scalar fallback.
/// 2. **Small K** (K <= 32): 32x32 tile with `tile_k = K` (no K-loop).
/// 3. **Tall-skinny** (M >= 4*N, both aligned): 64x16 tile.
/// 4. **Wide** (N >= 4*M, both aligned): 16x64 tile.
/// 5. **Default**: 32x32 tile (standard square config).
///
/// All returned configs require M, K, N to be multiples of
/// [`SIMDGROUP_ALIGN`] (8). The caller is responsible for verifying
/// alignment before dispatch.
///
/// # Examples
///
/// ```ignore
/// // LSTM recurrent step: M=1, K=640, N=256 → tiny, scalar fallback
/// assert!(select_gemm_tiles(1, 640, 256).is_none());
///
/// // Large square: M=256, K=768, N=768 → standard 32x32
/// let cfg = select_gemm_tiles(256, 768, 768).unwrap();
/// assert_eq!(cfg.tile_m, 32);
/// ```
///
/// Issue: #3479
pub(crate) fn select_gemm_tiles(m: usize, k: usize, n: usize) -> Option<TileConfig> {
    let mn = m.saturating_mul(n);

    // 1. Tiny: scalar fallback.
    if mn < TINY_THRESHOLD {
        return None;
    }

    // 2. Small K: clamp tile_k to K, use square tile.
    if k <= SMALL_K_THRESHOLD && k > 0 {
        // Round tile_k up to alignment boundary.
        let tile_k = k.next_multiple_of(SIMDGROUP_ALIGN);
        return Some(TileConfig {
            tile_m: 32,
            tile_n: 32,
            tile_k,
            threadgroup_size: [32, 4, 1],
        });
    }

    // 3. Tall-skinny: M >> N.
    if n > 0 && m / n >= TALL_SKINNY_RATIO && is_aligned(m) && is_aligned(n) {
        return Some(TileConfig::TALL_SKINNY);
    }

    // 4. Wide: N >> M.
    if m > 0 && n / m >= WIDE_RATIO && is_aligned(m) && is_aligned(n) {
        return Some(TileConfig::WIDE);
    }

    // 5. Default: standard square.
    Some(TileConfig::SQUARE)
}

/// Returns `true` if `v` is a multiple of [`SIMDGROUP_ALIGN`] (8).
fn is_aligned(v: usize) -> bool {
    v.is_multiple_of(SIMDGROUP_ALIGN)
}

/// Returns `true` if the given dimensions should use scalar dispatch
/// rather than simdgroup tiling.
///
/// This is the tiny-matrix gate used by `should_use_simdgroup()` to
/// reject shapes where dispatch overhead dominates compute. Shapes
/// with M * N < [`TINY_THRESHOLD`] (1024) are rejected.
///
/// Issue: #3479
pub(crate) fn is_scalar_fallback(m: usize, n: usize) -> bool {
    m.saturating_mul(n) < TINY_THRESHOLD
}

#[cfg(test)]
#[path = "simdgroup_tile_select_tests.rs"]
mod tests;
