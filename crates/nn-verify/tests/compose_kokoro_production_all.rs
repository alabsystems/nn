// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Kokoro production-weight verification tests.
//!
//! Merges 5 `compose_kokoro_production*.rs` binaries into one to reduce
//! link-time overhead from redundant NY linkage.

#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_production_weights.rs"]
mod kokoro_production_weights;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_production_segments.rs"]
mod kokoro_production_segments;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production.rs"]
mod compose_kokoro_production;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_crown.rs"]
mod compose_kokoro_production_crown;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_moonshot.rs"]
mod compose_kokoro_production_moonshot;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_segments.rs"]
mod compose_kokoro_production_segments;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_production_crown_extended.rs"]
mod compose_kokoro_production_crown_extended;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_kokoro_conv_transpose_layernorm.rs"]
mod compose_kokoro_conv_transpose_layernorm;

#[cfg(test)]
mod tighter_crown_adoption_tests {
    use super::kokoro_production_weights::{
        prefer_tighter_recorded_output, propagate_with_tight_crown_fallback,
    };
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;
    use nn_verify::{tensor_kernel_to_graph, BoundedTensor, PropMethod, TensorParamBinding};
    use ndarray::{ArrayD, IxDyn};

    fn uniform_bt(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
        let lower = ArrayD::from_elem(IxDyn(shape), lo);
        let upper = ArrayD::from_elem(IxDyn(shape), hi);
        BoundedTensor::new(lower, upper).expect("valid bounds")
    }

    fn bounds_min_max(bounds: &BoundedTensor) -> (f32, f32) {
        let (lower, upper) = bounds.lower_upper();
        let lo = lower.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = upper.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lo, hi)
    }

    fn build_linear_graph(in_features: usize, out_features: usize) -> nn_verify::GraphNetwork {
        let mut b = TensorBlockBuilder::new("tight_crown_linear");
        let data = b.add_input("data", &[in_features]);
        let weight = b.add_input("weight", &[out_features, in_features]);
        let bias = b.add_input("bias", &[out_features]);
        let linear = b.add_linear(data, weight, Some(bias), &[out_features]);
        let def = b.build(linear).expect("valid linear graph");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[out_features, in_features]),
                0.01_f32,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_features]), 0.0_f32)),
        ];

        tensor_kernel_to_graph(&def, &bindings).expect("graph translation")
    }

    #[test]
    fn test_production_helper_prefers_alpha_crown_when_graph_supports_it() {
        let graph = build_linear_graph(8, 8);
        let input = uniform_bt(&[8], -1.0, 1.0);

        let (method, output, fallback_reason) =
            propagate_with_tight_crown_fallback(&graph, &input).expect("alpha-first propagation");

        assert_eq!(method, PropMethod::AlphaCrown);
        assert!(fallback_reason.is_none());
        assert_eq!(output.lower_upper().0.shape(), &[8]);
    }

    #[test]
    fn test_prefer_tighter_recorded_output_preserves_alpha_crown() {
        let ibp_output = uniform_bt(&[2], -2.0, 2.0);
        let alpha_output = uniform_bt(&[2], -0.5, 0.5);

        let (method, output, ibp_width, _) =
            prefer_tighter_recorded_output(PropMethod::AlphaCrown, &ibp_output, &alpha_output);

        assert_eq!(method, PropMethod::AlphaCrown);
        assert_eq!(ibp_width, Some(4.0));
        assert_eq!(bounds_min_max(&output), (-0.5, 0.5));
    }

    #[test]
    fn test_prefer_tighter_recorded_output_preserves_beta_crown() {
        let ibp_output = uniform_bt(&[2], -3.0, 3.0);
        let beta_output = uniform_bt(&[2], -1.0, 1.0);

        let (method, output, ibp_width, _) =
            prefer_tighter_recorded_output(PropMethod::BetaCrown, &ibp_output, &beta_output);

        assert_eq!(method, PropMethod::BetaCrown);
        assert_eq!(ibp_width, Some(6.0));
        assert_eq!(bounds_min_max(&output), (-1.0, 1.0));
    }
}
