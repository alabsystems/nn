// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for the decoder composition fragment:
//! GPU output of a BinaryAdd → Conv1d → GLU → GELU chain within NY
//! composed IBP bounds.
//!
//! This is the AC3 test for #652: proving that the Metal GPU execution of a
//! composed Demucs temporal decoder fragment matches the composed verification
//! bounds — not just each individual kernel.
//!
//! The decoder fragment has 2 variable inputs (data + skip) which are stacked
//! along axis 0 for NY multi-variable verification.
//!
//! Part of #652.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Decoder fragment builder (mirrors compose_decoder_chain.rs in nn-verify)
// ---------------------------------------------------------------------------

/// Build the Demucs decoder fragment: BinaryAdd(data, skip) → Conv1d → GLU → GELU.
///
/// Inputs: data [C, T], skip [C, T], weight [2C, C, K].
/// Output: [C, out_length] where out_length = T + 2*padding - K + 1.
fn build_decoder_fragment(
    channels: usize,
    length: usize,
    kernel_size: usize,
    padding: usize,
) -> TensorKernelDef {
    let doubled = channels * 2;
    let out_length = length + 2 * padding - kernel_size + 1;

    let mut b = TensorBlockBuilder::new("demucs_decoder_fragment");

    let data = b.add_input("data", &[channels, length]);
    let skip = b.add_input("skip", &[channels, length]);
    let weight = b.add_input("weight", &[doubled, channels, kernel_size]);

    let added = b.add_binary_add(data, skip, &[channels, length]);
    let conv_out = b.add_conv1d(added, weight, None, 1, padding, &[doubled, out_length]);
    let glu_out = b
        .add_glu(conv_out, 0, &[doubled, out_length])
        .expect("even dim");
    let gelu_out = b.add_gelu(glu_out, &[channels, out_length]);

    b.build(gelu_out).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------

/// Prove IBP bounds for the decoder fragment with 2 variable inputs.
///
/// Multi-variable inputs are stacked along axis 0: BoundedTensor shape = [2, C, T].
/// Returns (proved_lower, proved_upper) arrays over the output shape.
fn prove_decoder_bounds(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    channels: usize,
    length: usize,
    input_lo: f32,
    input_hi: f32,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("decoder graph must build");

    // 2 variable inputs stacked along axis 0
    let lower_in = ArrayD::from_elem(IxDyn(&[2, channels, length]), input_lo);
    let upper_in = ArrayD::from_elem(IxDyn(&[2, channels, length]), input_hi);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");

    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through decoder fragment");
    let (lo, hi) = output_bounds.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // Multi-variable stacking adds a leading dimension [1, C, T].
    // Squeeze it to match the Metal dispatch output shape [C, T].
    let squeeze = |arr: &ArrayD<f32>| -> ArrayD<f32> {
        if arr.shape().first() == Some(&1) {
            let new_shape: Vec<usize> = arr.shape()[1..].to_vec();
            let flat: Vec<f32> = arr.iter().copied().collect();
            ArrayD::from_shape_vec(IxDyn(&new_shape), flat).expect("squeeze reshape")
        } else {
            arr.clone()
        }
    };
    (squeeze(lo), squeeze(hi))
}

// ===========================================================================
// Decoder composition GPU contract tests
// ===========================================================================

/// Small decoder fragment: BinaryAdd(2,8) → Conv1d(2→4,k=3,pad=1) → GLU → GELU.
/// Exercises all 4 dispatch steps with minimal dimensions for fast execution.
/// Part of #652.
#[test]
fn test_decoder_fragment_gpu_within_bounds_small() {
    let (channels, length, kernel_size, padding) = (2, 8, 3, 1);
    let doubled = channels * 2;
    let out_length = length + 2 * padding - kernel_size + 1;

    let def = build_decoder_fragment(channels, length, kernel_size, padding);

    let weight_data = rand_f32_vec(0xDEC0_0001, doubled * channels * kernel_size, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[doubled, channels, kernel_size]),
                weight_data.clone(),
            )
            .expect("weight shape"),
        ),
    ];

    let (proved_lo, proved_hi) = prove_decoder_bounds(&def, &bindings, channels, length, -1.0, 1.0);
    assert_eq!(
        proved_lo.shape(),
        &[channels, out_length],
        "output bounds shape"
    );

    // Run on Metal GPU with random inputs within [-1, 1].
    let cache = metal_setup();
    let data = rand_f32_vec(0xDEC0_0002, channels * length, -1.0, 1.0);
    let skip = rand_f32_vec(0xDEC0_0003, channels * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("skip", skip);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("decoder GPU dispatch");
    assert_eq!(gpu_out.len(), channels * out_length, "output length");

    assert_gpu_within_bounds("decoder_small", &gpu_out, &proved_lo, &proved_hi);
}

/// Dvoice-realistic parameters: Conv1d(48→96, k=3, stride=1, pad=1).
/// Tests at production channel counts where precision accumulation matters.
/// Part of #652.
#[test]
fn test_decoder_fragment_gpu_within_bounds_dvoice() {
    let (channels, length, kernel_size, padding) = (48, 16, 3, 1);
    let doubled = channels * 2;
    let out_length = length + 2 * padding - kernel_size + 1;

    let def = build_decoder_fragment(channels, length, kernel_size, padding);

    // Small weights to prevent IBP blow-up at dvoice scale.
    let weight_data = rand_f32_vec(0xDA5D_0001, doubled * channels * kernel_size, -0.05, 0.05);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[doubled, channels, kernel_size]),
                weight_data.clone(),
            )
            .expect("weight shape"),
        ),
    ];

    let (proved_lo, proved_hi) = prove_decoder_bounds(&def, &bindings, channels, length, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[channels, out_length]);

    // Vacuous widening guard: IBP width should stay reasonable with small weights.
    let max_width = proved_lo
        .iter()
        .zip(proved_hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 100.0,
        "IBP width {max_width} exceeds threshold — possible vacuous widening"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA5D_0002, channels * length, -1.0, 1.0);
    let skip = rand_f32_vec(0xDA5D_0003, channels * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("skip", skip);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dvoice decoder GPU dispatch");
    assert_eq!(gpu_out.len(), channels * out_length);

    assert_gpu_within_bounds("decoder_dvoice", &gpu_out, &proved_lo, &proved_hi);
}

/// Larger kernel Conv1d(4→8, k=5, pad=2) with wider input range.
/// Exercises the decoder fragment with different kernel geometry.
/// Part of #652.
#[test]
fn test_decoder_fragment_gpu_within_bounds_wider_kernel() {
    let (channels, length, kernel_size, padding) = (4, 12, 5, 2);
    let doubled = channels * 2;
    let out_length = length + 2 * padding - kernel_size + 1;

    let def = build_decoder_fragment(channels, length, kernel_size, padding);

    let weight_data = rand_f32_vec(0xDEC5_0001, doubled * channels * kernel_size, -0.3, 0.3);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[doubled, channels, kernel_size]),
                weight_data.clone(),
            )
            .expect("weight shape"),
        ),
    ];

    let (proved_lo, proved_hi) = prove_decoder_bounds(&def, &bindings, channels, length, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[channels, out_length]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDEC5_0002, channels * length, -1.0, 1.0);
    let skip = rand_f32_vec(0xDEC5_0003, channels * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("skip", skip);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("wider kernel decoder GPU dispatch");
    assert_eq!(gpu_out.len(), channels * out_length);

    assert_gpu_within_bounds("decoder_wider_kernel", &gpu_out, &proved_lo, &proved_hi);
}
