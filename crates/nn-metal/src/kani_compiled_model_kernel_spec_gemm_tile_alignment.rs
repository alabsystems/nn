// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Focused Kani proof harnesses for GEMM KernelSpec routing.
//!
//! These proofs target the issue #3728 invariants around simdgroup tile-size
//! selection and 8-lane alignment requirements.

#[cfg(kani)]
mod proofs {
    use kani::assume;

    use crate::dyn_tensor_metal::{
        select_tile_config, should_use_simdgroup, tg_memory_bytes, GemmTileConfig,
    };

    const TG_MEM_LIMIT_BYTES: u64 = 32 * 1024;

    /// Prove: simdgroup GEMM eligibility implies the production alignment and
    /// size preconditions used by `spec_linear_activation`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn simdgroup_gate_implies_aligned_dims() {
        let m: usize = kani::any();
        let k: usize = kani::any();
        let n: usize = kani::any();

        assume(m >= 1 && m <= 4096);
        assume(k >= 1 && k <= 4096);
        assume(n >= 1 && n <= 4096);
        assume(should_use_simdgroup(m, k, n));

        assert_eq!(m % 8, 0);
        assert_eq!(k % 8, 0);
        assert_eq!(n % 8, 0);
        assert!(m * n >= 16_384);
        assert!(k >= 128);
    }

    /// Prove: a misaligned M, K, or N dimension is enough to disable the
    /// simdgroup path entirely.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn misalignment_disables_simdgroup_gemm() {
        let m: usize = kani::any();
        let k: usize = kani::any();
        let n: usize = kani::any();
        let misaligned_axis: u8 = kani::any();

        assume(m >= 8 && m <= 4096);
        assume(k >= 128 && k <= 4096);
        assume(n >= 8 && n <= 4096);
        assume(m * n >= 16_384);
        assume(misaligned_axis < 3);

        let (m, k, n) = match misaligned_axis {
            0 => (m + 1, k, n),
            1 => (m, k + 1, n),
            _ => (m, k, n + 1),
        };

        assume(m % 8 != 0 || k % 8 != 0 || n % 8 != 0);

        assert!(!should_use_simdgroup(m, k, n));
    }

    /// Prove: the large 64x64 tile is selected exactly when both output
    /// dimensions are large enough and the 64x64 occupancy threshold is met.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn large_tile_selection_matches_shape_threshold() {
        let m: usize = kani::any();
        let k: usize = kani::any();
        let n: usize = kani::any();

        assume(m >= 1 && m <= 8192);
        assume(k >= 1 && k <= 4096);
        assume(n >= 1 && n <= 8192);

        let tile = select_tile_config(m, k, n, 1);
        let large_shape = m >= 64 && n >= 64;
        let tg_count_64 = m.div_ceil(64) * n.div_ceil(64);
        let should_be_large = large_shape && tg_count_64 >= 32;

        assert_eq!(tile == GemmTileConfig::LARGE, should_be_large);
        if tile == GemmTileConfig::LARGE {
            assert_eq!(tile.bm, 64);
            assert_eq!(tile.bn, 64);
        } else {
            assert_eq!(tile, GemmTileConfig::SMALL);
        }
    }

    /// Prove: both supported GEMM tile configurations fit within the Metal
    /// 32 KB threadgroup-memory budget for f32 and half-precision kernels.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn gemm_tile_threadgroup_memory_stays_within_limit() {
        let use_large_tile: bool = kani::any();
        let is_half: bool = kani::any();

        let tile = if use_large_tile {
            GemmTileConfig::LARGE
        } else {
            GemmTileConfig::SMALL
        };
        let tg_bytes = tg_memory_bytes(tile, is_half);

        assert!(tg_bytes > 0);
        assert!(tg_bytes <= TG_MEM_LIMIT_BYTES);
    }
}
