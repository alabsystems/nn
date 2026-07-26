// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deterministic PRNG for reproducible test inputs.
//!
//! Provides [`splitmix64`] and [`rand_f32_vec`] used by integration tests
//! across nn-metal, nn-verify, and nn-dsl. Consolidated here to eliminate
//! duplication (previously 3 copies of splitmix64, 2 of rand_f32_vec).
//!
//! Part of #1411.

/// SplitMix64 hash function (Vigna 2015).
///
/// Deterministic bijective mixing: given a `u64` input, produces a
/// well-distributed `u64` output. Used as the core PRNG primitive for
/// test data generation.
pub fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Generate `count` deterministic f32 values in `[lo, hi]`.
///
/// Each element is derived from `seed` and its index via [`splitmix64`],
/// producing reproducible test vectors that are uniform across the range.
pub fn rand_f32_vec(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let z = splitmix64(seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let t = (z >> 11) as f64 * (1.0 / f64::from(1u32 << 23) / f64::from(1u32 << 30));
            let val = f64::from(lo) + t * f64::from(hi - lo);
            val as f32
        })
        .collect()
}
