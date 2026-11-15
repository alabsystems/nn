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

#[cfg(test)]
use crate::kernel_error::KernelError;
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
#[path = "softmax_tests.rs"]
mod tests;
