// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for proof certificate tests.
//!
//! Used by `certificate_tests.rs` and `certificate_v2_tests.rs`.

use super::*;

pub(super) fn sample_verification() -> KernelVerification {
    sample_verification_with_bounds(-9.704, 10.296)
}

/// Parameterized KernelVerification with custom output bounds.
///
/// Replaces copy-pasted helpers that differed only in bound values.
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

pub(super) fn sample_input_spec() -> InputBoundsRecord {
    sample_input_spec_with_bounds(-10.0, 10.0, vec![1.0])
}

/// Parameterized InputBoundsRecord with custom input bounds and constants.
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

pub(super) fn sample_layer_bounds() -> Vec<LayerBoundRecord> {
    vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(0.0, 5.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ]
}

#[allow(dead_code)] // Test helper available for future certificate tests.
pub(super) fn sample_layer_bound(index: usize) -> LayerBoundRecord {
    LayerBoundRecord {
        layer_index: index,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-10.0, 10.0)],
        output_bounds: vec![(-9.704, 10.296)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![]),
    }
}

pub(super) fn sample_kani_record() -> KaniProofRecord {
    KaniProofRecord {
        harness_count: 3,
        status: KaniOutcome::Passed,
        properties: vec![
            "no_overflow".to_string(),
            "no_nan".to_string(),
            "bounds_preservation".to_string(),
        ],
        cbmc_version: Some("6.0.0".to_string()),
    }
}
