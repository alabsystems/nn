// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Input bounds construction and validation helpers for kernel verification.

use ny_api::{Bound, BoundedTensor, VerificationSpec};

use crate::error::{StructuralError, VerifyError};
use crate::graph::ParamBinding;

/// Validated scalar input bounds for verification.
///
/// Enforces that both bounds are finite and `lower <= upper` at construction.
/// This eliminates the repeated `(input_lower: f32, input_upper: f32)` parameter
/// pattern throughout the verification API and prevents swapped-argument bugs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarInputBounds {
    lower: f32,
    upper: f32,
}

impl ScalarInputBounds {
    /// Create validated scalar input bounds.
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::InvalidInputBounds`] if either bound is non-finite
    /// (NaN/Inf) or if `lower > upper`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn new(lower: f32, upper: f32) -> Result<Self, VerifyError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(VerifyError::InvalidInputBounds { lower, upper });
        }
        Ok(Self { lower, upper })
    }

    /// Lower bound (guaranteed finite).
    #[must_use]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Upper bound (guaranteed finite, `>= lower`).
    #[must_use]
    pub fn upper(&self) -> f32 {
        self.upper
    }

    /// Convert to a `BoundedTensor` for NY verification.
    ///
    /// Creates a 1-element tensor with these bounds.
    #[must_use = "returns a Result that may contain an error"]
    pub fn to_bounded_tensor(&self) -> Result<BoundedTensor, VerifyError> {
        use ndarray::{ArrayD, IxDyn};
        let lo = ArrayD::from_elem(IxDyn(&[1]), self.lower);
        let hi = ArrayD::from_elem(IxDyn(&[1]), self.upper);
        Ok(BoundedTensor::new(lo, hi)?)
    }
}

/// Construct a `BoundedTensor` for scalar kernel verification.
///
/// Creates a 1-element tensor with the given lower and upper bounds.
///
/// # Errors
///
/// Returns an error if bounds are non-finite (NaN/Inf), `lower > upper`, or
/// if `BoundedTensor` construction fails.
#[must_use = "returns a Result that may contain an error"]
pub fn scalar_input_bounds(lower: f32, upper: f32) -> Result<BoundedTensor, VerifyError> {
    ScalarInputBounds::new(lower, upper)?.to_bounded_tensor()
}

/// Construct a `BoundedTensor` for multi-variable scalar kernel verification.
///
/// Creates an N-element tensor where each element has its own (lower, upper) bound.
///
/// # Errors
///
/// Returns an error if any bounds are non-finite (NaN/Inf), `lower > upper`,
/// or if bounds are empty.
#[must_use = "returns a Result that may contain an error"]
pub fn multi_scalar_input_bounds(bounds: &[(f32, f32)]) -> Result<BoundedTensor, VerifyError> {
    if bounds.is_empty() {
        return Err(VerifyError::InvalidInputBounds {
            lower: 0.0,
            upper: 0.0,
        });
    }
    for &(lower, upper) in bounds {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(VerifyError::InvalidInputBounds { lower, upper });
        }
    }
    use ndarray::{ArrayD, IxDyn};
    let lo = ArrayD::from_shape_vec(
        IxDyn(&[bounds.len()]),
        bounds.iter().map(|&(l, _)| l).collect(),
    )
    .map_err(|e| {
        VerifyError::from(StructuralError::Shape {
            reason: e.to_string(),
        })
    })?;
    let hi = ArrayD::from_shape_vec(
        IxDyn(&[bounds.len()]),
        bounds.iter().map(|&(_, u)| u).collect(),
    )
    .map_err(|e| {
        VerifyError::from(StructuralError::Shape {
            reason: e.to_string(),
        })
    })?;
    Ok(BoundedTensor::new(lo, hi)?)
}

/// Create uniform `BoundedTensor`: all lower = `-range`, all upper = `+range`.
///
/// Replaces the common 4-line `ArrayD::from_elem` + `BoundedTensor::new` pattern
/// for constructing symmetric input bounds. Used by `verify_model!` and test files.
///
/// # Examples
///
/// ```rust,ignore
/// let bounds = nn_verify::uniform_bounds(&[1, 128], 1.0);
/// // => BoundedTensor with lower = -1.0 everywhere, upper = 1.0 everywhere
/// ```
///
/// # Errors
///
/// Returns [`VerifyError`] if `range` is not finite or is negative.
pub fn uniform_bounds(shape: &[usize], range: f32) -> Result<BoundedTensor, VerifyError> {
    if !range.is_finite() || range < 0.0 {
        return Err(VerifyError::InvalidInputBounds {
            lower: -range,
            upper: range,
        });
    }
    use ndarray::{ArrayD, IxDyn};
    let lower = ArrayD::from_elem(IxDyn(shape), -range);
    let upper = ArrayD::from_elem(IxDyn(shape), range);
    Ok(BoundedTensor::new(lower, upper)?)
}

pub(crate) fn count_variable_bindings(bindings: &[ParamBinding]) -> usize {
    bindings
        .iter()
        .filter(|binding| matches!(binding, ParamBinding::Variable))
        .count()
}

pub(crate) fn validate_variable_bounds(
    bindings: &[ParamBinding],
    variable_bounds: &[(f32, f32)],
) -> Result<(), VerifyError> {
    let variable_count = count_variable_bindings(bindings);
    if variable_count == 0 {
        return Err(VerifyError::NoVariableBindings);
    }

    if variable_bounds.len() != variable_count {
        return Err(VerifyError::VariableBoundsMismatch {
            variable_count,
            bounds_count: variable_bounds.len(),
        });
    }

    Ok(())
}

pub(crate) fn verification_spec_from_tensors(
    input_bounds: &BoundedTensor,
    required_output_bounds: &[Bound],
) -> Result<VerificationSpec, VerifyError> {
    let input_spec_bounds: Vec<Bound> = input_bounds
        .lower()
        .iter()
        .zip(input_bounds.upper().iter())
        .map(|(&lower, &upper)| Bound::try_new(lower, upper))
        .collect::<ny_api::Result<Vec<_>>>()?;

    let input_shape = Some(input_bounds.lower().shape().to_vec());
    Ok(VerificationSpec::from_parts(
        input_spec_bounds,
        required_output_bounds.to_vec(),
        None,
        input_shape,
    )?)
}

#[cfg(test)]
#[path = "verify_input_tests.rs"]
mod tests;
