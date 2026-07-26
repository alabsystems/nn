// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extension trait providing kernel-only math operations on scalar types.
//!
//! The `#[nn::kernel]` proc-macro preserves the original Rust function as a
//! reference implementation. Some operations (like `rsqrt`) are recognized by
//! the lowerer and emit efficient GPU instructions, but have no equivalent
//! method on `f32` in standard Rust.
//!
//! Import this trait to make those methods available:
//!
//! ```rust
//! use nn_dsl::KernelOps;
//!
//! fn inv_sqrt(x: f32) -> f32 {
//!     x.rsqrt()
//! }
//! ```

/// Extension methods for scalar types used in nn kernels.
///
/// These methods have no standard Rust equivalent but map to efficient
/// GPU intrinsics (e.g., `metal::precise::rsqrt` in MSL).
pub trait KernelOps {
    /// Reciprocal square root: `1.0 / self.sqrt()`.
    ///
    /// Maps to `metal::precise::rsqrt` in MSL codegen, which is a single
    /// hardware instruction on Apple Silicon GPUs.
    #[must_use]
    fn rsqrt(self) -> Self;
}

impl KernelOps for f32 {
    #[inline]
    fn rsqrt(self) -> Self {
        1.0 / self.sqrt()
    }
}

impl KernelOps for f64 {
    #[inline]
    fn rsqrt(self) -> Self {
        1.0 / self.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::KernelOps;

    #[test]
    fn test_rsqrt_f32_basic() {
        let result = 4.0_f32.rsqrt();
        assert!(
            (result - 0.5).abs() < 1e-7,
            "rsqrt(4) should be 0.5, got {result}"
        );
    }

    #[test]
    fn test_rsqrt_f32_one() {
        let result = 1.0_f32.rsqrt();
        assert!(
            (result - 1.0).abs() < 1e-7,
            "rsqrt(1) should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_rsqrt_f64_basic() {
        let result = 4.0_f64.rsqrt();
        assert!(
            (result - 0.5).abs() < 1e-15,
            "rsqrt(4) should be 0.5, got {result}"
        );
    }
}
