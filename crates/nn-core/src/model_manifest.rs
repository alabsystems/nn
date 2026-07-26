// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model manifest trait and static configuration types for compile-time
//! verification of model graph structure.
//!
//! [`ModelManifest`] allows models to declare their graph structure statically
//! (input/output rank, weight names, layer count, parameter count bound) so
//! that verification tools can validate model properties at compile time via
//! const assertions.
//!
//! [`StaticModelConfig`] carries runtime configuration that can be validated
//! against a [`ModelManifest`] to ensure dimension compatibility.
//!
//! # Const Assertion Helpers
//!
//! The module provides three const-fn helpers for use in const contexts:
//! - [`assert_divisible`] — ensures one dimension divides another evenly
//! - [`assert_shape_compatible`] — ensures input dimension matches expected
//! - [`assert_positive`] — ensures a value is nonzero

/// Trait for models to declare their graph structure statically.
///
/// Implementing this trait lets verification tools (NY, ay, Kani)
/// inspect model metadata at compile time without constructing an instance.
///
/// # Example
///
/// ```rust
/// use nn_core::model_manifest::ModelManifest;
///
/// struct NnEncoder;
///
/// impl ModelManifest for NnEncoder {
///     const INPUT_RANK: usize = 3;
///     const OUTPUT_RANK: usize = 3;
///     const WEIGHT_NAMES: &'static [&'static str] = &[
///         "encoder.conv1.weight",
///         "encoder.conv1.bias",
///     ];
///     const LAYER_COUNT: usize = 4;
///     const PARAM_COUNT_BOUND: usize = 1_000_000;
/// }
/// ```
pub trait ModelManifest {
    /// Rank (number of dimensions) of the model's primary input tensor.
    ///
    /// For example, a 1-D audio model might use rank 3: `[B, C, T]`.
    const INPUT_RANK: usize;

    /// Rank (number of dimensions) of the model's primary output tensor.
    const OUTPUT_RANK: usize;

    /// Static list of weight tensor names expected by the model.
    ///
    /// These correspond to keys in a safetensors file or [`VarBuilder`]
    /// namespace. The verifier checks that all listed weights are present
    /// during weight loading.
    const WEIGHT_NAMES: &'static [&'static str];

    /// Total number of layers (operations) in the model graph.
    ///
    /// Used by verification tools to allocate intermediate-bound storage
    /// and to sanity-check that the traced graph matches the declared
    /// structure.
    const LAYER_COUNT: usize;

    /// Upper bound on the total number of scalar parameters.
    ///
    /// This is a compile-time constant, so it cannot reflect runtime
    /// dynamic shapes. It should be an upper bound that the verifier
    /// can use for resource estimation and certificate metadata.
    const PARAM_COUNT_BOUND: usize;
}

/// Runtime model configuration validated against a [`ModelManifest`].
///
/// Holds dimension sizes and other runtime values that cannot be encoded
/// as const generics today (blocked on Rust `adt_const_params`).
/// [`StaticModelConfig::validate`] checks these values for consistency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticModelConfig {
    /// Input dimension sizes (length must equal `ModelManifest::INPUT_RANK`).
    pub input_dims: Vec<usize>,
    /// Output dimension sizes (length must equal `ModelManifest::OUTPUT_RANK`).
    pub output_dims: Vec<usize>,
    /// Number of weight tensors loaded (must equal `WEIGHT_NAMES.len()`).
    pub weight_count: usize,
}

/// Error type for [`StaticModelConfig`] validation failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ManifestValidationError {
    /// Input dimensions length does not match `INPUT_RANK`.
    #[error("input rank mismatch: expected {expected} dimensions, got {actual}")]
    InputRankMismatch { expected: usize, actual: usize },

    /// Output dimensions length does not match `OUTPUT_RANK`.
    #[error("output rank mismatch: expected {expected} dimensions, got {actual}")]
    OutputRankMismatch { expected: usize, actual: usize },

    /// Weight count does not match `WEIGHT_NAMES.len()`.
    #[error("weight count mismatch: manifest declares {expected} weights, got {actual}")]
    WeightCountMismatch { expected: usize, actual: usize },

    /// A dimension size is zero.
    #[error("dimension {index} is zero")]
    ZeroDimension { index: usize },
}

impl StaticModelConfig {
    /// Validate this runtime configuration against a [`ModelManifest`].
    ///
    /// Returns `Ok(())` if all dimensions, ranks, and weight counts are
    /// consistent. Returns a typed error describing the first mismatch.
    pub fn validate<M: ModelManifest>(&self) -> Result<(), ManifestValidationError> {
        // Check input rank.
        if self.input_dims.len() != M::INPUT_RANK {
            return Err(ManifestValidationError::InputRankMismatch {
                expected: M::INPUT_RANK,
                actual: self.input_dims.len(),
            });
        }

        // Check output rank.
        if self.output_dims.len() != M::OUTPUT_RANK {
            return Err(ManifestValidationError::OutputRankMismatch {
                expected: M::OUTPUT_RANK,
                actual: self.output_dims.len(),
            });
        }

        // Check weight count.
        if self.weight_count != M::WEIGHT_NAMES.len() {
            return Err(ManifestValidationError::WeightCountMismatch {
                expected: M::WEIGHT_NAMES.len(),
                actual: self.weight_count,
            });
        }

        // Check for zero dimensions in input.
        for (i, &dim) in self.input_dims.iter().enumerate() {
            if dim == 0 {
                return Err(ManifestValidationError::ZeroDimension { index: i });
            }
        }

        // Check for zero dimensions in output.
        for (i, &dim) in self.output_dims.iter().enumerate() {
            if dim == 0 {
                return Err(ManifestValidationError::ZeroDimension {
                    index: self.input_dims.len() + i,
                });
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Const assertion helpers
// ---------------------------------------------------------------------------

/// Asserts at compile time that `a` is divisible by `b`.
///
/// Panics (compile-time error in const context) if `b == 0` or `a % b != 0`.
///
/// # Example
///
/// ```rust
/// use nn_core::model_manifest::assert_divisible;
/// const _: () = assert_divisible(768, 12); // 768 / 12 = 64 heads
/// ```
#[allow(clippy::must_use_unit, clippy::manual_is_multiple_of)]
#[must_use = "call in a `const _: () = ...` context for compile-time checking"]
pub const fn assert_divisible(a: usize, b: usize) {
    assert!(b != 0, "assert_divisible: divisor must be nonzero");
    assert!(a % b == 0, "assert_divisible: a is not divisible by b");
}

/// Asserts at compile time that `input_dim == expected`.
///
/// Panics (compile-time error in const context) if the values differ.
///
/// # Example
///
/// ```rust
/// use nn_core::model_manifest::assert_shape_compatible;
/// const _: () = assert_shape_compatible(512, 512);
/// ```
#[allow(clippy::must_use_unit)]
#[must_use = "call in a `const _: () = ...` context for compile-time checking"]
pub const fn assert_shape_compatible(input_dim: usize, expected: usize) {
    assert!(
        input_dim == expected,
        "assert_shape_compatible: input_dim does not match expected"
    );
}

/// Asserts at compile time that `val` is positive (nonzero).
///
/// Panics (compile-time error in const context) if `val == 0`.
///
/// # Example
///
/// ```rust
/// use nn_core::model_manifest::assert_positive;
/// const _: () = assert_positive(64);
/// ```
#[allow(clippy::must_use_unit)]
#[must_use = "call in a `const _: () = ...` context for compile-time checking"]
pub const fn assert_positive(val: usize) {
    assert!(val != 0, "assert_positive: value must be nonzero");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal test model implementing ModelManifest.
    struct TestModel;

    impl ModelManifest for TestModel {
        const INPUT_RANK: usize = 3;
        const OUTPUT_RANK: usize = 2;
        const WEIGHT_NAMES: &'static [&'static str] = &["linear.weight", "linear.bias"];
        const LAYER_COUNT: usize = 5;
        const PARAM_COUNT_BOUND: usize = 10_000;
    }

    // -- assert_divisible tests -----------------------------------------------

    #[test]
    fn test_assert_divisible_exact() {
        let () = assert_divisible(768, 12); // 768 / 12 = 64
        let () = assert_divisible(100, 10);
        let () = assert_divisible(0, 1); // 0 is divisible by anything nonzero
        let () = assert_divisible(7, 1);
    }

    #[test]
    #[should_panic(expected = "divisor must be nonzero")]
    fn test_assert_divisible_zero_divisor() {
        let () = assert_divisible(10, 0);
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn test_assert_divisible_not_divisible() {
        let () = assert_divisible(10, 3);
    }

    // -- assert_shape_compatible tests ----------------------------------------

    #[test]
    fn test_assert_shape_compatible_match() {
        let () = assert_shape_compatible(512, 512);
        let () = assert_shape_compatible(0, 0);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn test_assert_shape_compatible_mismatch() {
        let () = assert_shape_compatible(512, 256);
    }

    // -- assert_positive tests ------------------------------------------------

    #[test]
    fn test_assert_positive_valid() {
        let () = assert_positive(1);
        let () = assert_positive(usize::MAX);
    }

    #[test]
    #[should_panic(expected = "must be nonzero")]
    fn test_assert_positive_zero() {
        let () = assert_positive(0);
    }

    // -- const evaluation tests -----------------------------------------------

    /// Verify helpers actually work in const context.
    const _DIVISIBLE_CHECK: () = assert_divisible(768, 12);
    const _SHAPE_CHECK: () = assert_shape_compatible(512, 512);
    const _POSITIVE_CHECK: () = assert_positive(64);

    // -- StaticModelConfig validation tests -----------------------------------

    #[test]
    fn test_validate_success() {
        let cfg = StaticModelConfig {
            input_dims: vec![1, 1, 16000],
            output_dims: vec![1, 512],
            weight_count: 2,
        };
        cfg.validate::<TestModel>().unwrap();
    }

    #[test]
    fn test_validate_input_rank_mismatch() {
        let cfg = StaticModelConfig {
            input_dims: vec![1, 16000], // rank 2, expected 3
            output_dims: vec![1, 512],
            weight_count: 2,
        };
        let err = cfg.validate::<TestModel>().unwrap_err();
        assert_eq!(
            err,
            ManifestValidationError::InputRankMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn test_validate_output_rank_mismatch() {
        let cfg = StaticModelConfig {
            input_dims: vec![1, 1, 16000],
            output_dims: vec![1, 1, 512], // rank 3, expected 2
            weight_count: 2,
        };
        let err = cfg.validate::<TestModel>().unwrap_err();
        assert_eq!(
            err,
            ManifestValidationError::OutputRankMismatch {
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn test_validate_weight_count_mismatch() {
        let cfg = StaticModelConfig {
            input_dims: vec![1, 1, 16000],
            output_dims: vec![1, 512],
            weight_count: 5, // expected 2
        };
        let err = cfg.validate::<TestModel>().unwrap_err();
        assert_eq!(
            err,
            ManifestValidationError::WeightCountMismatch {
                expected: 2,
                actual: 5,
            }
        );
    }

    #[test]
    fn test_validate_zero_input_dim() {
        let cfg = StaticModelConfig {
            input_dims: vec![1, 0, 16000], // zero at index 1
            output_dims: vec![1, 512],
            weight_count: 2,
        };
        let err = cfg.validate::<TestModel>().unwrap_err();
        assert_eq!(err, ManifestValidationError::ZeroDimension { index: 1 });
    }

    #[test]
    fn test_validate_zero_output_dim() {
        let cfg = StaticModelConfig {
            input_dims: vec![1, 1, 16000],
            output_dims: vec![0, 512], // zero at output index 0 → overall index 3
            weight_count: 2,
        };
        let err = cfg.validate::<TestModel>().unwrap_err();
        assert_eq!(err, ManifestValidationError::ZeroDimension { index: 3 });
    }

    // -- ModelManifest associated const tests ---------------------------------

    #[test]
    fn test_manifest_consts() {
        assert_eq!(TestModel::INPUT_RANK, 3);
        assert_eq!(TestModel::OUTPUT_RANK, 2);
        assert_eq!(TestModel::WEIGHT_NAMES.len(), 2);
        assert_eq!(TestModel::WEIGHT_NAMES[0], "linear.weight");
        assert_eq!(TestModel::WEIGHT_NAMES[1], "linear.bias");
        assert_eq!(TestModel::LAYER_COUNT, 5);
        assert_eq!(TestModel::PARAM_COUNT_BOUND, 10_000);
    }

    // -- Error Display tests --------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = ManifestValidationError::InputRankMismatch {
            expected: 3,
            actual: 2,
        };
        assert!(err.to_string().contains("input rank mismatch"));
        assert!(err.to_string().contains("expected 3"));
        assert!(err.to_string().contains("got 2"));
    }
}
