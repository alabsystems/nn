// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder and reference implementation for `TensorOpKind::Softmax`.
//!
//! Softmax normalization along an axis:
//! `y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`.
//!
//! The `max(x)` subtraction is for numerical stability (log-sum-exp trick).
//! Output shape matches input shape. Used in transformer attention.
//!
//! Maps to NY's `SoftmaxLayer::new(axis)` for IBP bound propagation.
//!
//! # Naming convention (#336)
//!
//! - `softmax_ref` — 1D vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_softmax` — `TensorKernelDef` IR builder
//! - `resolve_softmax_axis` — negative-to-positive axis resolution
//!
//! Issue: #737.

#[cfg(any(test, kani))]
use crate::kernel_error::KernelError;
#[cfg(any(test, kani))]
use crate::kernel_util::{checked_slice_output, validate_finite_slice};
use crate::tensor_ir::{
    TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a `TensorKernelDef` for a softmax operation.
///
/// Constructs a 2-node graph:
/// - `%0 = Input("data", shape)`
/// - `%1 = Softmax { input: %0, axis }`
///
/// # Parameters
///
/// - `name`: Kernel name for the generated code.
/// - `shape`: Input (and output) tensor shape.
/// - `axis`: Axis along which to compute softmax. Supports Python-style negative
///   indexing (e.g., -1 = last axis).
///
/// # Errors
///
/// Returns `TensorIRLayerError::SoftmaxInputScalar` if `shape` is empty.
/// Returns `TensorIRLayerError::SoftmaxAxisOutOfBounds` if `axis` is out of range.
/// Returns `TensorIRError::EmptyDimension` if any dimension is 0.
pub fn build_softmax(
    name: &str,
    shape: &[usize],
    axis: i32,
) -> Result<TensorKernelDef, TensorIRError> {
    if shape.is_empty() {
        return Err(TensorIRLayerError::SoftmaxInputScalar.into());
    }
    if shape.contains(&0) {
        return Err(TensorIRError::EmptyDimension(shape.to_vec()));
    }
    let rank = shape.len();
    let rank_i32 = i32::try_from(rank).map_err(|_| TensorIRLayerError::SoftmaxAxisOutOfBounds {
        axis,
        rank,
        neg_rank: i32::MIN,
    })?;
    if axis < -rank_i32 || axis >= rank_i32 {
        return Err(TensorIRLayerError::SoftmaxAxisOutOfBounds {
            axis,
            rank,
            neg_rank: -rank_i32,
        }
        .into());
    }

    let mut nodes = Vec::new();

    // %0: input data.
    nodes.push(TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: crate::input_names::DATA.into(),
            shape: shape.to_vec(),
        },
        shape.to_vec(),
    ));

    // %1: Softmax operation — output shape matches input.
    let softmax_id = TensorNodeId::new(1);
    nodes.push(TensorNode::new(
        softmax_id,
        TensorOpKind::Softmax {
            input: TensorNodeId::new(0),
            axis,
        },
        shape.to_vec(),
    ));

    Ok(TensorKernelDef::new(name, nodes, softmax_id))
}

#[cfg(test)]
/// Resolve a potentially negative axis to a non-negative index.
///
/// Returns the non-negative axis index for a given rank, using Python-style
/// negative indexing rules. Panics if axis is out of bounds — callers should
/// validate with `build_softmax()` or `validate_softmax()` first.
#[must_use]
pub(crate) fn resolve_softmax_axis(axis: i32, rank: usize) -> usize {
    let rank_i32 = rank as i32;
    assert!(
        axis >= -rank_i32 && axis < rank_i32,
        "axis {axis} out of bounds for rank {rank}"
    );
    if axis < 0 {
        (rank_i32 + axis) as usize
    } else {
        axis as usize
    }
}

#[cfg(any(test, kani))]
/// Compute softmax over a 1D slice (numerically stable log-sum-exp).
///
/// `softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)) for all j)`
///
/// # Properties
///
/// - All output elements are in `[0, 1]`.
/// - Output elements sum to `1.0` (within floating-point tolerance).
/// - Numerically stable via max subtraction (log-sum-exp trick).
///
/// # Errors
///
/// Returns [`KernelError::InvalidDimension`] if the input is empty.
/// Returns [`KernelError::NonFiniteSliceElement`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteSliceOutput`] if any output is non-finite.
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn softmax_ref(x: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "length",
            value: 0,
        });
    }
    validate_finite_slice("x", x)?;

    // Phase 1: Find max for numerical stability.
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Phase 2: Compute exp(x[i] - max) and sum.
    let exps: Vec<f32> = x.iter().map(|&xi| (xi - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    // Phase 3: Normalize.
    let result: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    checked_slice_output(&result)?;
    Ok(result)
}

#[cfg(kani)]
mod kani_builder_proofs {
    //! Kani proof harnesses for the softmax IR builder.
    //!
    //! Same pattern as conv1d/conv_transpose_1d/causal_conv1d/linear builder
    //! harnesses: prove no-panic and output shape correct.

    use super::*;

    /// Prove `build_softmax` never panics for bounded params.
    ///
    /// Domain: rank in [0, 4], each dim in [0, 4], axis in [-4, 4].
    /// Reduced from [0, 8] and added unwind(16) for CBMC Vec heap
    /// unwinding tractability (#767 AC3).
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn softmax_build_no_panic() {
        let rank: usize = kani::any();
        kani::assume(rank <= 4);

        let d0: usize = kani::any();
        let d1: usize = kani::any();
        let d2: usize = kani::any();
        let d3: usize = kani::any();
        kani::assume(d0 <= 4);
        kani::assume(d1 <= 4);
        kani::assume(d2 <= 4);
        kani::assume(d3 <= 4);

        let shape: &[usize] = match rank {
            0 => &[],
            1 => &[d0],
            2 => &[d0, d1],
            3 => &[d0, d1, d2],
            4 => &[d0, d1, d2, d3],
            _ => unreachable!(),
        };

        let axis: i32 = kani::any();
        kani::assume(axis >= -4 && axis <= 4);

        // Must not panic — returns Err for invalid params.
        let _ = build_softmax("kani_test", shape, axis);
    }

    /// Prove `build_softmax` output shape matches input shape.
    #[kani::unwind(8)]
    #[kani::proof]
    fn softmax_output_shape_preserved() {
        let d0: usize = kani::any();
        let d1: usize = kani::any();
        kani::assume(d0 >= 1 && d0 <= 64);
        kani::assume(d1 >= 1 && d1 <= 64);

        let shape = [d0, d1];

        // axis=-1 is always valid for rank >= 1.
        let def = build_softmax("kani_test", &shape, -1).expect("valid params must succeed");
        let out_node = &def.nodes[def.output.index()];
        assert_eq!(out_node.shape.len(), 2, "output rank must match input");
        assert_eq!(out_node.shape[0], d0, "output dim 0 must match input");
        assert_eq!(out_node.shape[1], d1, "output dim 1 must match input");
    }
}

#[cfg(all(kani, feature = "kani-stubbing"))]
mod kani_proofs {
    //! Kani proof harnesses for softmax reference implementation.
    //!
    //! Softmax is a vector operation, so we use fixed small sizes (2, 3)
    //! to keep CBMC tractable. Uses `exp_stub` to work around CBMC's
    //! inaccurate `f32::exp()` model (#239).
    //!
    //! ## Proof strategy
    //!
    //! Softmax with `exp_stub` (nondeterministic positive finite) proves:
    //! 1. **Output in [0, 1]**: Each `exp(x_i) / sum(exp(x_j))` is a ratio of
    //!    positive values where the numerator ≤ denominator, so output ∈ (0, 1].
    //! 2. **Finiteness**: Sum of positive finite values is finite (for small N);
    //!    division of finite by finite is finite.
    //!
    //! Sum-to-one requires deterministic exp (actual exp or det_stub), since
    //! nondeterministic exp values don't satisfy the algebraic identity.

    use super::*;
    use crate::kani_stubs::exp_stub;

    /// Prove softmax outputs are in [0, 1] for 2-element input.
    ///
    /// Domain: x in [-100, 100].
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f32::exp, exp_stub)]
    fn softmax_output_in_unit_interval_n2() {
        let x0: f32 = kani::any();
        let x1: f32 = kani::any();
        kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
        kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);

        let result = softmax_ref(&[x0, x1]).expect("softmax_ref must succeed for bounded inputs");

        assert_eq!(result.len(), 2, "output length must match input");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "softmax output [{i}] must be finite");
            assert!(v >= 0.0, "softmax output [{i}] must be >= 0");
            assert!(v <= 1.0, "softmax output [{i}] must be <= 1");
        }
    }

    /// Prove softmax outputs are in [0, 1] for 3-element input.
    ///
    /// Domain: x in [-100, 100].
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(6)]
    #[kani::stub(f32::exp, exp_stub)]
    fn softmax_output_in_unit_interval_n3() {
        let x0: f32 = kani::any();
        let x1: f32 = kani::any();
        let x2: f32 = kani::any();
        kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
        kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);
        kani::assume(x2.is_finite() && x2 >= -100.0 && x2 <= 100.0);

        let result =
            softmax_ref(&[x0, x1, x2]).expect("softmax_ref must succeed for bounded inputs");

        assert_eq!(result.len(), 3, "output length must match input");
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "softmax output [{i}] must be finite");
            assert!(v >= 0.0, "softmax output [{i}] must be >= 0");
            assert!(v <= 1.0, "softmax output [{i}] must be <= 1");
        }
    }

    /// Prove softmax rejects empty input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(2)]
    #[kani::stub(f32::exp, exp_stub)]
    fn softmax_rejects_empty_input() {
        let result = softmax_ref(&[]);
        assert!(result.is_err(), "softmax must reject empty input");
    }
}

#[cfg(test)]
#[path = "softmax_tests.rs"]
mod tests;
