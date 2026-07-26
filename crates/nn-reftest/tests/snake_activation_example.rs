// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Example: Kokoro-style Snake activation layer comparison (reference vs Rust).

use nn_dsl::{snake_ref_f32, PrecisionContract, PrecisionTier, ScalarType};
use nn_reftest::{assert_traces_match, load_safetensors_from_bytes, ReferenceTrace};

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn test_kokoro_snake_activation_matches_reference_trace() {
    let channels = 2usize;
    let length = 4usize;
    let input: Vec<f32> = vec![-1.2, -0.5, 0.0, 0.75, 1.1, -0.3, 0.2, 2.0];
    let alpha: Vec<f32> = vec![0.5, 2.0];

    // Simulated "Python-exported" reference tensor from the Snake layer.
    let reference_values: Vec<f32> = vec![
        -0.562_357_7,
        -0.377_582_55,
        0.0,
        1.018_311_1,
        1.426_833_3,
        -0.140_589_45,
        0.275_823_32,
        2.286_375,
    ];
    let reference_bytes = f32_to_le_bytes(&reference_values);
    let reference_tensor = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![1, channels, length],
        &reference_bytes,
    )
    .expect("reference tensor should be valid");
    let reference_serialized = safetensors::tensor::serialize(
        vec![("kokoro.encoder.snake0".to_string(), reference_tensor)],
        None,
    )
    .expect("reference serialization should succeed");
    let reference_trace =
        load_safetensors_from_bytes(&reference_serialized).expect("reference trace should load");

    let (candidate_trace, output_len) = ReferenceTrace::capture(|trace| {
        let candidate =
            snake_ref_f32(&input, &alpha, channels, length).expect("snake layout should be valid");
        trace
            .checkpoint("kokoro.encoder.snake0", &candidate, &[1, channels, length])
            .expect("valid checkpoint");
        candidate.len()
    });
    assert_eq!(output_len, input.len());

    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    assert_traces_match!(
        candidate_trace,
        reference_trace,
        epsilon = contract.differential_abs_budget
    );
}
