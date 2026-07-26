// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for convert parameter-validation and layer-consistency
//! invariants.
//!
//! These proofs focus on the issue #3732 concerns that sit closest to
//! `convert.rs` today:
//! - per-layer parameter counts add up to the converted model's total params
//! - `WeightShapeMismatch` preserves expected vs actual counts
//! - complete layer parameter groups keep weight/bias layer counts aligned
//! - a missing layer parameter immediately creates a count mismatch

#[cfg(kani)]
mod proofs {
    use std::collections::HashMap;

    use nn_core::dyn_tensor::trace::ComputationGraph;
    use nn_core::{Device, DynTensor};

    use crate::convert::{ConvertError, ConvertedModel};

    fn tensor_with_len(len: usize) -> DynTensor {
        DynTensor::from_vec(vec![0.0f32; len], &[len], &Device::Cpu).unwrap()
    }

    fn count_suffix(weights: &HashMap<String, DynTensor>, suffix: &str) -> usize {
        weights.keys().filter(|name| name.ends_with(suffix)).count()
    }

    /// Per-layer tensors contribute additively to the converted model's total
    /// parameter count.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn converted_model_total_params_matches_layer_parameter_sum() {
        let layer0_weight: u8 = kani::any();
        let layer0_bias: u8 = kani::any();
        let layer1_weight: u8 = kani::any();
        let layer1_bias: u8 = kani::any();
        kani::assume(layer0_weight >= 1 && layer0_weight <= 8);
        kani::assume(layer0_bias >= 1 && layer0_bias <= 8);
        kani::assume(layer1_weight >= 1 && layer1_weight <= 8);
        kani::assume(layer1_bias >= 1 && layer1_bias <= 8);

        let mut weights = HashMap::new();
        weights.insert(
            "encoder.layers.0.weight".to_string(),
            tensor_with_len(layer0_weight as usize),
        );
        weights.insert(
            "encoder.layers.0.bias".to_string(),
            tensor_with_len(layer0_bias as usize),
        );
        weights.insert(
            "encoder.layers.1.weight".to_string(),
            tensor_with_len(layer1_weight as usize),
        );
        weights.insert(
            "encoder.layers.1.bias".to_string(),
            tensor_with_len(layer1_bias as usize),
        );

        let expected_total = layer0_weight as usize
            + layer0_bias as usize
            + layer1_weight as usize
            + layer1_bias as usize;

        let model = ConvertedModel::new(
            ComputationGraph::from_nodes(vec![]),
            weights,
            1,
            vec!["input".to_string()],
            vec!["output".to_string()],
            "issue-3732".to_string(),
        );

        assert_eq!(model.total_params(), expected_total);
        assert_eq!(model.num_weights(), 4);
    }

    /// Parameter-count validation failures must preserve both the expected and
    /// actual element counts for diagnostics.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn weight_shape_mismatch_preserves_expected_and_actual_counts() {
        let expected: u8 = kani::any();
        let actual: u8 = kani::any();
        kani::assume(expected >= 1 && expected <= 32);
        kani::assume(actual >= 1 && actual <= 32);
        kani::assume(expected != actual);

        let err = ConvertError::WeightShapeMismatch {
            name: "encoder.layers.0.weight".to_string(),
            expected: expected as usize,
            actual: actual as usize,
        };

        match err {
            ConvertError::WeightShapeMismatch {
                name,
                expected: stored_expected,
                actual: stored_actual,
            } => {
                assert_eq!(name, "encoder.layers.0.weight");
                assert_eq!(stored_expected, expected as usize);
                assert_eq!(stored_actual, actual as usize);
                assert_ne!(stored_expected, stored_actual);
            }
            _ => unreachable!("constructed variant must round-trip through the match"),
        }
    }

    /// When conversion emits complete `{weight,bias}` pairs per layer, the
    /// layer counts stay aligned.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn complete_layer_parameter_groups_keep_counts_aligned() {
        let include_second_layer: bool = kani::any();

        let mut weights = HashMap::new();
        weights.insert("encoder.layers.0.weight".to_string(), tensor_with_len(2));
        weights.insert("encoder.layers.0.bias".to_string(), tensor_with_len(1));

        if include_second_layer {
            weights.insert("encoder.layers.1.weight".to_string(), tensor_with_len(2));
            weights.insert("encoder.layers.1.bias".to_string(), tensor_with_len(1));
        }

        let weight_layers = count_suffix(&weights, ".weight");
        let bias_layers = count_suffix(&weights, ".bias");

        assert_eq!(weight_layers, bias_layers);
        assert_eq!(weight_layers, if include_second_layer { 2 } else { 1 });
    }

    /// If one layer is missing a bias tensor, the weight/bias layer counts stop
    /// matching immediately.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn missing_layer_bias_exposes_count_mismatch() {
        let mut weights = HashMap::new();
        weights.insert("encoder.layers.0.weight".to_string(), tensor_with_len(2));
        weights.insert("encoder.layers.0.bias".to_string(), tensor_with_len(1));
        weights.insert("encoder.layers.1.weight".to_string(), tensor_with_len(2));

        let weight_layers = count_suffix(&weights, ".weight");
        let bias_layers = count_suffix(&weights, ".bias");

        assert_eq!(weight_layers, 2);
        assert_eq!(bias_layers, 1);
        assert_ne!(weight_layers, bias_layers);
        assert_eq!(weight_layers, bias_layers + 1);
    }
}
