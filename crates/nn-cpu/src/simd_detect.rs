// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Runtime SIMD capability detection.
//!
//! Detects NEON on aarch64 (always available) and AVX2 on x86_64 (runtime check).
//! Other architectures fall back to scalar.

/// SIMD instruction set available on the current CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimdLevel {
    /// No SIMD available; use scalar fallback.
    Scalar,
    /// ARM NEON (128-bit, 4x f32). Always available on aarch64.
    Neon,
    /// x86 AVX2 (256-bit, 8x f32). Requires runtime detection.
    Avx2,
}

/// Detect the best SIMD level available at runtime.
///
/// On aarch64 this always returns `Neon` (NEON is mandatory in ARMv8).
/// On x86_64 this checks for AVX2 support via `is_x86_feature_detected!`.
/// On other architectures this returns `Scalar`.
#[must_use]
pub fn detect() -> SimdLevel {
    #[cfg(target_arch = "aarch64")]
    {
        SimdLevel::Neon
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else {
            SimdLevel::Scalar
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        SimdLevel::Scalar
    }
}

/// Width of the SIMD register in f32 lanes.
#[must_use]
pub const fn lane_count(level: SimdLevel) -> usize {
    match level {
        SimdLevel::Scalar => 1,
        SimdLevel::Neon => 4, // 128-bit / 32-bit
        SimdLevel::Avx2 => 8, // 256-bit / 32-bit
    }
}
