// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape policy for compiled model execution.
//!
//! Controls whether tensor shapes are baked at compile time (maximum
//! performance, zero overhead) or resolved at runtime (eliminates
//! recompilation when sequence dimensions change).
//!
//! Part of #3873 (shape-polymorphic Metal compilation).

/// Controls whether shapes are baked at compile time or resolved at runtime.
///
/// For TTS streaming, `Polymorphic` eliminates the ~1.2s recompilation
/// latency when input text length changes. The tradeoff is slightly higher
/// GPU memory usage (buffer plan computed at max dimensions) and minor
/// per-dispatch overhead for runtime grid computation (~100ns/step).
///
/// # Example
///
/// ```rust,ignore
/// use nn_metal::compiled_model::ShapePolicy;
///
/// // Fixed shapes (default, current behavior):
/// let model = CompiledModel::builder(&graph, &cache).build()?;
///
/// // Polymorphic shapes for streaming TTS:
/// let model = CompiledModel::builder(&graph, &cache)
///     .shape_policy(ShapePolicy::Polymorphic {
///         max_seq_len: 512,
///         max_t_mel: 1024,
///     })
///     .build()?;
/// ```
///
/// # Design
///
/// Metal compute pipelines (`.metallib`) are compiled from shader source at
/// build time. The shader code itself is shape-agnostic -- it processes
/// elements indexed by `thread_position_in_grid`. What changes with input
/// shape are:
///
/// - **Buffer sizes**: how many bytes to allocate for intermediates.
/// - **Threadgroup grid counts**: how many threadgroups to launch.
/// - **Input validation**: which shapes to accept vs reject.
///
/// `Polymorphic` mode pre-allocates buffers at max dimensions and computes
/// threadgroup grids from actual input sizes at dispatch time. The Metal
/// pipeline objects are reused without recompilation.
///
/// # Shape Classification
///
/// Tensor dimensions are classified into two categories:
///
/// - **Structural** (channels, hidden_size, kernel_size): fixed by model
///   architecture, must match exactly between compile and runtime.
/// - **Sequence** (batch, seq_len, t_mel, total_samples): vary with input
///   text length, allowed to differ at runtime as long as they stay within
///   the declared maximums.
///
/// For TTS models like Kokoro, the last dimension(s) of 2D+ tensors are
/// typically sequence dimensions (token count, mel frames, audio samples).
/// The classification heuristic: for rank >= 2 inputs, the last dimension
/// is treated as a sequence dimension. For rank-1 inputs, the single
/// dimension is structural.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ShapePolicy {
    /// Current behavior: shapes baked into `CompiledPlan` at trace time.
    ///
    /// Maximum performance. Zero per-dispatch overhead. Requires
    /// recompilation when input shapes change.
    #[default]
    Fixed,

    /// Sequence dimensions resolved at runtime from input `GpuSlice` metadata.
    ///
    /// Eliminates recompilation when only sequence dimensions change.
    /// Structural dimensions (channels, hidden_size, kernel_size) remain
    /// fixed from the trace-time graph.
    ///
    /// Buffer plan is computed at `(max_seq_len, max_t_mel)` to avoid
    /// per-forward replanning. Smaller inputs use a subset of the
    /// pre-allocated buffer.
    Polymorphic {
        /// Maximum token sequence length for buffer pre-allocation.
        ///
        /// Buffer plan is computed as if all sequence dimensions are at
        /// this maximum. Must be >= the largest `seq_len` that will be
        /// passed at runtime. Typical values: 128 (chat), 512 (narration).
        max_seq_len: usize,

        /// Maximum mel frame count for buffer pre-allocation.
        ///
        /// Used by generator and sinegen segments. Must be >= the largest
        /// `t_mel` at runtime. Typical values: 320 (chat), 1024 (narration).
        max_t_mel: usize,
    },
}


impl ShapePolicy {
    /// Returns `true` if this policy requires runtime shape resolution.
    #[must_use]
    pub fn is_polymorphic(&self) -> bool {
        matches!(self, Self::Polymorphic { .. })
    }

    /// Returns `true` if this is the default fixed-shape policy.
    #[must_use]
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }

    /// Returns the maximum sequence length, or `None` for `Fixed` policy.
    #[must_use]
    pub fn max_seq_len(&self) -> Option<usize> {
        match self {
            Self::Polymorphic { max_seq_len, .. } => Some(*max_seq_len),
            Self::Fixed => None,
        }
    }

    /// Returns the maximum mel frame count, or `None` for `Fixed` policy.
    #[must_use]
    pub fn max_t_mel(&self) -> Option<usize> {
        match self {
            Self::Polymorphic { max_t_mel, .. } => Some(*max_t_mel),
            Self::Fixed => None,
        }
    }

    /// Validate that an actual input shape is compatible with a compiled
    /// (expected) input shape under this policy.
    ///
    /// For `Fixed`, shapes must match exactly.
    ///
    /// For `Polymorphic`, structural dimensions (all except the last for
    /// rank >= 2 tensors) must match exactly. The last dimension (sequence
    /// dim) may differ as long as it is > 0 and <= the compiled dimension.
    /// Rank must always match.
    ///
    /// Returns `Ok(())` on compatible shapes, `Err(reason)` otherwise.
    pub fn validate_shape(
        &self,
        expected: &[usize],
        actual: &[usize],
        input_index: usize,
    ) -> Result<(), ShapePolicyError> {
        // Rank must always match.
        if expected.len() != actual.len() {
            return Err(ShapePolicyError::RankMismatch {
                index: input_index,
                expected_rank: expected.len(),
                actual_rank: actual.len(),
            });
        }

        match self {
            Self::Fixed => {
                if expected != actual {
                    return Err(ShapePolicyError::ExactMismatch {
                        index: input_index,
                        expected: expected.to_vec(),
                        actual: actual.to_vec(),
                    });
                }
            }
            Self::Polymorphic { .. } => {
                // For rank-0 and rank-1 tensors, all dims are structural.
                // For rank >= 2, the last dim is the sequence dimension.
                let seq_dim_start = if expected.len() >= 2 {
                    expected.len() - 1
                } else {
                    expected.len() // no sequence dims for rank 0/1
                };

                // Structural dims must match exactly.
                for (dim_idx, (&exp, &act)) in
                    expected.iter().zip(actual.iter()).enumerate()
                {
                    if dim_idx < seq_dim_start {
                        if exp != act {
                            return Err(ShapePolicyError::StructuralMismatch {
                                index: input_index,
                                dim: dim_idx,
                                expected: exp,
                                actual: act,
                            });
                        }
                    } else {
                        // Sequence dimension: must be > 0 and <= compiled max.
                        if act == 0 {
                            return Err(ShapePolicyError::ZeroSequenceDim {
                                index: input_index,
                                dim: dim_idx,
                            });
                        }
                        if act > exp {
                            return Err(ShapePolicyError::SequenceDimExceedsMax {
                                index: input_index,
                                dim: dim_idx,
                                max: exp,
                                actual: act,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Errors from shape policy validation.
///
/// These are structured to give clear diagnostics about which dimension
/// failed and why, unlike the generic `ShapeMismatch` error in
/// `CompiledModelError`.
///
/// Part of #3873.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ShapePolicyError {
    /// Tensor ranks don't match (always fatal regardless of policy).
    #[error("input {index} rank mismatch: expected {expected_rank}, got {actual_rank}")]
    RankMismatch {
        index: usize,
        expected_rank: usize,
        actual_rank: usize,
    },

    /// Exact shape mismatch under `Fixed` policy.
    #[error("input {index} shape mismatch: expected {expected:?}, got {actual:?}")]
    ExactMismatch {
        index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// A structural (non-sequence) dimension doesn't match.
    #[error(
        "input {index} structural dim {dim} mismatch: expected {expected}, got {actual}"
    )]
    StructuralMismatch {
        index: usize,
        dim: usize,
        expected: usize,
        actual: usize,
    },

    /// A sequence dimension is zero (invalid).
    #[error("input {index} sequence dim {dim} is zero")]
    ZeroSequenceDim { index: usize, dim: usize },

    /// A sequence dimension exceeds the compiled maximum.
    #[error(
        "input {index} sequence dim {dim} exceeds compiled max: {actual} > {max}"
    )]
    SequenceDimExceedsMax {
        index: usize,
        dim: usize,
        max: usize,
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::ShapePolicy;

    #[test]
    fn default_is_fixed() {
        assert_eq!(ShapePolicy::default(), ShapePolicy::Fixed);
    }

    #[test]
    fn fixed_is_not_polymorphic() {
        let policy = ShapePolicy::Fixed;
        assert!(policy.is_fixed());
        assert!(!policy.is_polymorphic());
        assert_eq!(policy.max_seq_len(), None);
        assert_eq!(policy.max_t_mel(), None);
    }

    #[test]
    fn polymorphic_has_max_dims() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        assert!(!policy.is_fixed());
        assert!(policy.is_polymorphic());
        assert_eq!(policy.max_seq_len(), Some(512));
        assert_eq!(policy.max_t_mel(), Some(1024));
    }

    #[test]
    fn polymorphic_eq() {
        let a = ShapePolicy::Polymorphic {
            max_seq_len: 128,
            max_t_mel: 320,
        };
        let b = ShapePolicy::Polymorphic {
            max_seq_len: 128,
            max_t_mel: 320,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn different_max_dims_not_eq() {
        let a = ShapePolicy::Polymorphic {
            max_seq_len: 128,
            max_t_mel: 320,
        };
        let b = ShapePolicy::Polymorphic {
            max_seq_len: 256,
            max_t_mel: 320,
        };
        assert_ne!(a, b);
    }

    // -- Shape validation tests --

    #[test]
    fn fixed_exact_match_ok() {
        let policy = ShapePolicy::Fixed;
        assert!(policy.validate_shape(&[1, 128], &[1, 128], 0).is_ok());
    }

    #[test]
    fn fixed_rejects_different_shapes() {
        let policy = ShapePolicy::Fixed;
        assert!(policy.validate_shape(&[1, 128], &[1, 64], 0).is_err());
    }

    #[test]
    fn fixed_rejects_different_ranks() {
        let policy = ShapePolicy::Fixed;
        assert!(policy.validate_shape(&[1, 128], &[128], 0).is_err());
    }

    #[test]
    fn polymorphic_accepts_smaller_seq_dim() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        // [1, 128] compiled at max, [1, 64] actual -- last dim is seq, allowed to shrink.
        assert!(policy.validate_shape(&[1, 128], &[1, 64], 0).is_ok());
    }

    #[test]
    fn polymorphic_rejects_larger_seq_dim() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        // Actual seq dim 256 > compiled max 128.
        assert!(policy.validate_shape(&[1, 128], &[1, 256], 0).is_err());
    }

    #[test]
    fn polymorphic_rejects_structural_mismatch() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        // Structural dim (dim 0) mismatch: 1 vs 2.
        assert!(policy.validate_shape(&[1, 128], &[2, 128], 0).is_err());
    }

    #[test]
    fn polymorphic_rejects_zero_seq_dim() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        assert!(policy.validate_shape(&[1, 128], &[1, 0], 0).is_err());
    }

    #[test]
    fn polymorphic_rank_1_all_structural() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        // Rank-1: single dim is structural, must match exactly.
        assert!(policy.validate_shape(&[256], &[256], 0).is_ok());
        assert!(policy.validate_shape(&[256], &[128], 0).is_err());
    }

    #[test]
    fn polymorphic_rank_3_last_dim_is_seq() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        // [B, C, T] -- B and C are structural, T is sequence.
        assert!(policy.validate_shape(&[1, 512, 128], &[1, 512, 64], 0).is_ok());
        assert!(policy.validate_shape(&[1, 512, 128], &[1, 256, 64], 0).is_err());
    }

    #[test]
    fn polymorphic_exact_match_ok() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        // Exact match is always fine under polymorphic.
        assert!(policy.validate_shape(&[1, 128], &[1, 128], 0).is_ok());
    }

    #[test]
    fn polymorphic_rank_mismatch_rejected() {
        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        assert!(policy.validate_shape(&[1, 128], &[128], 0).is_err());
    }

    #[test]
    fn error_variants_are_descriptive() {
        use super::ShapePolicyError;

        let policy = ShapePolicy::Polymorphic {
            max_seq_len: 512,
            max_t_mel: 1024,
        };
        let err = policy.validate_shape(&[1, 128], &[2, 128], 0).unwrap_err();
        match err {
            ShapePolicyError::StructuralMismatch { index, dim, expected, actual } => {
                assert_eq!(index, 0);
                assert_eq!(dim, 0);
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            other => panic!("expected StructuralMismatch, got {other:?}"),
        }

        let err = policy.validate_shape(&[1, 128], &[1, 256], 0).unwrap_err();
        match err {
            ShapePolicyError::SequenceDimExceedsMax { index, dim, max, actual } => {
                assert_eq!(index, 0);
                assert_eq!(dim, 1);
                assert_eq!(max, 128);
                assert_eq!(actual, 256);
            }
            other => panic!("expected SequenceDimExceedsMax, got {other:?}"),
        }
    }
}
