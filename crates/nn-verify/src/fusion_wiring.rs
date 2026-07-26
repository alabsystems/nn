// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generic fusion wiring verification utility.
//!
//! Provides a one-call function for Workers to verify that a fused scalar
//! kernel is equivalent to its decomposed sequence. Wraps [`FusionSpec`]
//! construction and [`verify_fusion_equivalence`] into a single call.

use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::fusion::verify_fusion_equivalence;
use crate::fusion_spec::{FusionSpec, FusionVerification};

/// Verify that a fused scalar kernel is equivalent to its decomposed sequence
/// for all inputs within the given bounds.
///
/// This is the standard utility for Workers wiring new fused NativeOps.
/// It constructs a [`FusionSpec`], builds the diamond DAG, propagates bounds
/// via CROWN (with IBP fallback), and checks the diff is within `epsilon`.
///
/// # Arguments
///
/// * `fused` — The fused kernel IR (single kernel computing the composed result)
/// * `first` — The first kernel in the decomposed sequence
/// * `second` — The second kernel (receives output of `first`)
/// * `num_shared_inputs` — Total number of shared input variables
/// * `first_param_indices` — Maps each `first` param to a shared input index
/// * `second_param_indices` — Maps each `second` param to a shared input index
///   (the entry at `second_input_from_first` is a placeholder)
/// * `second_input_from_first` — Which `second` param receives `first`'s output
/// * `variable_bounds` — Per-variable `(lower, upper)` bounds
/// * `epsilon` — Maximum tolerable absolute difference
///
/// # Errors
///
/// Returns [`VerifyError`] if kernel IR is invalid, bounds length doesn't
/// match `num_shared_inputs`, epsilon is NaN, or propagation fails.
pub fn verify_fusion_wiring(
    fused: &KernelDef,
    first: &KernelDef,
    second: &KernelDef,
    num_shared_inputs: usize,
    first_param_indices: &[usize],
    second_param_indices: &[usize],
    second_input_from_first: usize,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    let spec = FusionSpec {
        fused,
        first,
        second,
        num_shared_inputs,
        first_param_indices,
        second_param_indices,
        second_input_from_first,
    };
    verify_fusion_equivalence(&spec, variable_bounds, epsilon)
}
