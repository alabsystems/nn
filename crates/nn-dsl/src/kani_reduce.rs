// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rust reference implementations and Kani verification harnesses for
//! tensor reduction operations (Sum, Mean).
//!
//! The reference implementations serve two purposes:
//! 1. Kani model-checks them with bounded symbolic inputs
//! 2. Differential tests compare GPU output against them
//!
//! See `designs/2026-02-26-kernelir-tensor-ops.md` Phase C.

use crate::kernel_error::KernelError;

/// Compute a sum reduction along the last axis.
///
/// Input is a flat slice of `outer_size * reduce_dim` elements in row-major
/// order. Returns a `Vec<f32>` of length `outer_size`.
///
/// # Errors
///
/// Returns `KernelError::InvalidDimension` if `reduce_dim == 0`, or
/// `KernelError::ShapeMismatch` if `input.len()` is not divisible by `reduce_dim`.
pub(crate) fn reduce_sum_ref(input: &[f32], reduce_dim: usize) -> Result<Vec<f32>, KernelError> {
    if reduce_dim == 0 {
        return Err(KernelError::InvalidDimension {
            name: "reduce_dim",
            value: 0,
        });
    }
    if !input.len().is_multiple_of(reduce_dim) {
        return Err(KernelError::ShapeMismatch {
            expected: input.len() - (input.len() % reduce_dim) + reduce_dim,
            got: input.len(),
        });
    }
    let outer_size = input.len() / reduce_dim;
    Ok((0..outer_size)
        .map(|i| {
            let slice = &input[i * reduce_dim..(i + 1) * reduce_dim];
            slice.iter().sum::<f32>()
        })
        .collect())
}

/// Compute a mean reduction along the last axis.
///
/// Input is a flat slice of `outer_size * reduce_dim` elements in row-major
/// order. Returns a `Vec<f32>` of length `outer_size`.
///
/// # Errors
///
/// Returns `KernelError::InvalidDimension` if `reduce_dim == 0`,
/// `KernelError::DimensionExceedsF32Precision` if `reduce_dim > 2^24`, or
/// `KernelError::ShapeMismatch` if `input.len()` is not divisible by `reduce_dim`.
pub(crate) fn reduce_mean_ref(input: &[f32], reduce_dim: usize) -> Result<Vec<f32>, KernelError> {
    if reduce_dim == 0 {
        return Err(KernelError::InvalidDimension {
            name: "reduce_dim",
            value: 0,
        });
    }
    if reduce_dim > (1 << 24) {
        return Err(KernelError::DimensionExceedsF32Precision {
            name: "reduce_dim",
            value: reduce_dim,
        });
    }
    if !input.len().is_multiple_of(reduce_dim) {
        return Err(KernelError::ShapeMismatch {
            expected: input.len() - (input.len() % reduce_dim) + reduce_dim,
            got: input.len(),
        });
    }
    let outer_size = input.len() / reduce_dim;
    Ok((0..outer_size)
        .map(|i| {
            let slice = &input[i * reduce_dim..(i + 1) * reduce_dim];
            slice.iter().sum::<f32>() / reduce_dim as f32
        })
        .collect())
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Prove: reduce_sum output is finite for bounded finite inputs.
    ///
    /// Dimension bound: 1..=4 (reduced from 8 for CBMC tractability, #767 AC3).
    /// Input bound: [-100, 100] (realistic ML range).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn reduce_sum_finite() {
        let dim: u8 = kani::any();
        kani::assume((1..=4).contains(&dim));
        let dim = dim as usize;

        let mut input = vec![0.0f32; dim];
        for v in &mut input {
            *v = kani::any();
            kani::assume(v.is_finite() && *v >= -100.0 && *v <= 100.0);
        }

        let result = reduce_sum_ref(&input, dim)
            .expect("invariant: kani assumes valid dim and aligned input");
        assert_eq!(result.len(), 1);
        assert!(
            result[0].is_finite(),
            "reduce_sum must produce finite output for finite inputs"
        );
    }

    /// Prove: reduce_mean output is finite for bounded finite inputs.
    /// Dimension reduced from 8 to 4 for CBMC tractability (#767 AC3).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn reduce_mean_finite() {
        let dim: u8 = kani::any();
        kani::assume((1..=4).contains(&dim));
        let dim = dim as usize;

        let mut input = vec![0.0f32; dim];
        for v in &mut input {
            *v = kani::any();
            kani::assume(v.is_finite() && *v >= -100.0 && *v <= 100.0);
        }

        let result = reduce_mean_ref(&input, dim)
            .expect("invariant: kani assumes valid dim and aligned input");
        assert_eq!(result.len(), 1);
        assert!(
            result[0].is_finite(),
            "reduce_mean must produce finite output for finite inputs"
        );
    }

    /// Prove: reduce_mean output is within input bounds (with epsilon).
    ///
    /// For mean: min(input) <= mean(input) <= max(input), modulo
    /// floating-point accumulation error proportional to dimension size.
    /// Dimension reduced from 8 to 4 for CBMC tractability (#767 AC3).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn reduce_mean_bounded() {
        let dim: u8 = kani::any();
        kani::assume((1..=4).contains(&dim));
        let dim = dim as usize;

        let mut input = vec![0.0f32; dim];
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for v in &mut input {
            *v = kani::any();
            kani::assume(v.is_finite() && *v >= -100.0 && *v <= 100.0);
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }

        let result = reduce_mean_ref(&input, dim)
            .expect("invariant: kani assumes valid dim and aligned input");
        // Accumulation error bound: N * eps * max_value
        let eps = dim as f32 * f32::EPSILON * 100.0;
        assert!(result[0] >= lo - eps, "mean must be >= min(input) - eps");
        assert!(result[0] <= hi + eps, "mean must be <= max(input) + eps");
    }

    /// Prove: reduce_sum with dimension 1 is identity.
    /// Unwind(16) added for CBMC Vec loop tractability (#767 AC3).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn reduce_sum_dim1_identity() {
        let outer: u8 = kani::any();
        kani::assume((1..=4).contains(&outer));
        let outer = outer as usize;

        let mut input = vec![0.0f32; outer];
        for v in &mut input {
            *v = kani::any();
            kani::assume(v.is_finite() && *v >= -100.0 && *v <= 100.0);
        }

        let result = reduce_sum_ref(&input, 1).expect("invariant: dim=1 always valid");
        assert_eq!(result.len(), outer);
        for (i, &r) in result.iter().enumerate() {
            assert_eq!(r, input[i], "sum over dim=1 is identity");
        }
    }
}

#[cfg(test)]
#[path = "kani_reduce_tests.rs"]
mod tests;
