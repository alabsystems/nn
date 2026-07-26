// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for certificate checker tests.
//!
//! Consolidates `sample_verification()`, `sample_input_spec()`, and
//! `consistent_layer_bounds()` which were duplicated across 6 test files.
//! Test modules access these via `super::checker_test_shared::*`.

use crate::certificate_types::LayerBoundRecord;
use crate::status::{InputBoundsRecord, ParamInputRecord};
use crate::verify_types::{KernelVerification, OutputTensorBounds, PropMethod};
use crate::VerificationSoundnessMode;

/// Standard KernelVerification for certificate checker tests.
/// Snake kernel, IBP, output [-5, 5], width 10.
pub(super) fn sample_verification() -> KernelVerification {
    sample_verification_with_bounds(-5.0, 5.0)
}

/// Parameterized KernelVerification with custom output bounds.
///
/// Consolidates 6 near-identical helpers that differed only in bound values:
/// - `sample_verification()` → (-5.0, 5.0)
/// - `sample_verification_wide()` → (-1000.0, 1000.0) [soundness_3200]
/// - `sample_verification_with_width(w)` → (-w/2, w/2) [vacuity]
/// - local `sample_verification()` with Crown method [smt_proof]
///
/// For method override, call this then set `.method` on the result.
pub(super) fn sample_verification_with_bounds(lower: f32, upper: f32) -> KernelVerification {
    KernelVerification {
        kernel_name: "snake".to_string(),
        method: PropMethod::Ibp,
        output_lower: lower,
        output_upper: upper,
        output_width: upper - lower,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: Some(OutputTensorBounds {
            lower: vec![lower],
            upper: vec![upper],
            shape: vec![1],
            finite_mask: vec![true],
        }),
    }
}

/// Standard InputBoundsRecord for certificate checker tests.
/// Single variable input [-10, 10], one constant param.
pub(super) fn sample_input_spec() -> InputBoundsRecord {
    sample_input_spec_with_bounds(-10.0, 10.0, vec![1.0])
}

/// Parameterized InputBoundsRecord with custom input bounds and constants.
///
/// Consolidates 5 near-identical helpers that differed only in bound/constant values:
/// - `sample_input_spec()` → (-10.0, 10.0), constants [1.0]
/// - local `sample_input_spec()` → (-1.0, 1.0), constants [] [soundness_3200]
/// - local `sample_input_spec()` → (-10.0, 10.0), constants [1.0] [smt_proof, vacuity]
pub(super) fn sample_input_spec_with_bounds(
    lower: f32,
    upper: f32,
    constant_params: Vec<f32>,
) -> InputBoundsRecord {
    InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower,
            upper,
        }],
        constant_params,
        input_shape: Some(vec![1]),
        input_range: Some((lower, upper)),
    }
}

/// Consistent layer bounds using a specific PropMethod.
///
/// Consolidates the local `consistent_layer_bounds()` in smt_proof tests
/// which was identical except using Crown instead of Ibp.
pub(super) fn consistent_layer_bounds_with_method(method: PropMethod) -> Vec<LayerBoundRecord> {
    vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(0.0, 5.0)],
            method,
            node_name: None,
            input_sources: Some(vec![0]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(0.0, 5.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method,
            node_name: None,
            input_sources: Some(vec![1]),
        },
    ]
}

/// A consistent 3-layer trace: Linear → ReLU → Linear.
/// Each layer's output matches the next layer's input.
pub(super) fn consistent_layer_bounds() -> Vec<LayerBoundRecord> {
    consistent_layer_bounds_with_method(PropMethod::Ibp)
}
