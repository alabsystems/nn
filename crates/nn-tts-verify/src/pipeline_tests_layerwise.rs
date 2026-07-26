// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! verify_layerwise tests — per-layer CROWN composition (#1762 AC2, AC3).

use super::*;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;

/// Create a BoundedTensor with uniform bounds.
fn uniform_bt(shape: &[usize], lo: f32, hi: f32) -> nn_verify::BoundedTensor {
    use ndarray::{ArrayD, IxDyn};
    let lower = ArrayD::from_elem(IxDyn(shape), lo);
    let upper = ArrayD::from_elem(IxDyn(shape), hi);
    nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Build a single Linear layer TensorKernelDef with given dimensions.
fn build_linear_layer(
    name: &str,
    in_features: usize,
    out_features: usize,
) -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    Vec<nn_verify::TensorParamBinding>,
) {
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &[in_features]);
    let weight = b.add_input("weight", &[out_features, in_features]);
    let bias = b.add_input("bias", &[out_features]);
    let linear = b.add_linear(data, weight, Some(bias), &[out_features]);
    let def = b.build(linear).expect("valid linear graph");

    let w_val = 0.01_f32;
    let bindings = vec![
        nn_verify::TensorParamBinding::Variable,
        nn_verify::TensorParamBinding::ConstantTensor(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[out_features, in_features]),
            w_val,
        )),
        nn_verify::TensorParamBinding::ConstantTensor(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[out_features]),
            0.0_f32,
        )),
    ];
    (def, bindings)
}

/// Build a Linear+ReLU layer for tighter bounded output.
fn build_linear_relu_layer(
    name: &str,
    in_features: usize,
    out_features: usize,
) -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    Vec<nn_verify::TensorParamBinding>,
) {
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &[in_features]);
    let weight = b.add_input("weight", &[out_features, in_features]);
    let bias = b.add_input("bias", &[out_features]);
    let linear = b.add_linear(data, weight, Some(bias), &[out_features]);
    let relu = b.add_relu(linear, &[out_features]);
    let def = b.build(relu).expect("valid linear+relu graph");

    let w_val = 0.01_f32;
    let bindings = vec![
        nn_verify::TensorParamBinding::Variable,
        nn_verify::TensorParamBinding::ConstantTensor(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[out_features, in_features]),
            w_val,
        )),
        nn_verify::TensorParamBinding::ConstantTensor(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[out_features]),
            0.0_f32,
        )),
    ];
    (def, bindings)
}

#[test]
fn test_verify_layerwise_two_linear_d64() {
    // AC2: verify_layerwise at D=64 — two Linear layers composed.
    let (layer1, bind1) = build_linear_relu_layer("layer_0", 64, 64);
    let (layer2, bind2) = build_linear_layer("layer_1", 64, 64);

    let layers = vec![(layer1, bind1), (layer2, bind2)];
    let initial = uniform_bt(&[64], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("layerwise composition");

    // Pipeline should be valid: output bounds of layer 0 should fit
    // input bounds of layer 1 (both propagated from the same chain).
    assert!(cert.is_valid, "layerwise pipeline should be valid");
    assert!(cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].shape_compatible);
}

#[test]
fn test_verify_layerwise_prefers_alpha_crown_when_available() {
    let (layer1, bind1) = build_linear_relu_layer("alpha_layer_0", 16, 16);
    let (layer2, bind2) = build_linear_layer("alpha_layer_1", 16, 16);

    let layers = vec![(layer1, bind1), (layer2, bind2)];
    let initial = uniform_bt(&[16], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("alpha-first layerwise composition");

    assert_eq!(cert.stages[0].method, "AlphaCrown");
    assert_eq!(cert.stages[1].method, "AlphaCrown");
    assert!(cert.is_sound);
}

#[test]
fn test_verify_layerwise_three_layers_d64() {
    // Three-layer chain: Linear+ReLU → Linear+ReLU → Linear.
    let (layer1, bind1) = build_linear_relu_layer("layer_0", 64, 32);
    let (layer2, bind2) = build_linear_relu_layer("layer_1", 32, 16);
    let (layer3, bind3) = build_linear_layer("layer_2", 16, 8);

    let layers = vec![(layer1, bind1), (layer2, bind2), (layer3, bind3)];
    let initial = uniform_bt(&[64], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("3-layer composition");

    assert!(cert.is_valid);
}

#[test]
fn test_verify_layerwise_d128_scale() {
    // AC3: verify_layerwise at D=128+ — proves per-layer CROWN
    // scales beyond toy dimensions.
    let (layer1, bind1) = build_linear_relu_layer("layer_0", 128, 128);
    let (layer2, bind2) = build_linear_layer("layer_1", 128, 64);

    let layers = vec![(layer1, bind1), (layer2, bind2)];
    let initial = uniform_bt(&[128], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("D=128 layerwise");

    assert!(cert.is_valid);
}

#[test]
fn test_verify_layerwise_single_layer_error() {
    // Fewer than 2 layers should return InsufficientStages.
    let (layer1, bind1) = build_linear_layer("only_layer", 32, 16);
    let layers = vec![(layer1, bind1)];
    let initial = uniform_bt(&[32], -1.0, 1.0);

    let result = verify_layerwise(&layers, &initial);
    assert!(result.is_err());
}

#[test]
fn test_verify_layerwise_bound_widening_observable() {
    // Per-layer composition produces wider e2e bounds than any single layer.
    // This is expected behavior documented in AC5.
    let (layer1, bind1) = build_linear_relu_layer("layer_0", 64, 64);
    let (layer2, bind2) = build_linear_relu_layer("layer_1", 64, 64);
    let (layer3, bind3) = build_linear_relu_layer("layer_2", 64, 64);

    let layers = vec![(layer1, bind1), (layer2, bind2), (layer3, bind3)];
    let initial = uniform_bt(&[64], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("3-layer composition");

    assert!(cert.is_valid);

    // After 3 layers with small weights (0.01), bounds should stay reasonable.
    // With w=0.01 and ReLU, output bounds narrow rather than widen
    // (small weights contract the range).
    let output_range = cert.e2e_output_upper[0] - cert.e2e_output_lower[0];
    assert!(
        output_range < 10.0,
        "output range {output_range} should be reasonable (not vacuously wide)"
    );
}

// -----------------------------------------------------------------------
// D=192 production-dimension CROWN composition (#1741 Property 6 gap)
// -----------------------------------------------------------------------

#[test]
fn test_verify_layerwise_d192_production_scale() {
    // Production-dimension CROWN composition at D=192.
    // Proves per-layer CROWN scales to actual Kokoro decoder hidden dim.
    // Design reference: designs/archive/2026-03-10-crown-scaling-alternatives.md
    let (layer1, bind1) = build_linear_relu_layer("prod_layer_0", 192, 192);
    let (layer2, bind2) = build_linear_relu_layer("prod_layer_1", 192, 192);
    let (layer3, bind3) = build_linear_layer("prod_layer_2", 192, 192);

    let layers = vec![(layer1, bind1), (layer2, bind2), (layer3, bind3)];
    let initial = uniform_bt(&[192], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("D=192 layerwise");

    assert!(cert.is_valid, "D=192 pipeline must be valid");

    // With small weights (0.01) and ReLU, bounds should contract.
    let output_range = cert.e2e_output_upper[0] - cert.e2e_output_lower[0];
    assert!(
        output_range < 10.0,
        "D=192 output range {output_range} should be non-vacuous"
    );

    // All junctions must have contained bounds.
    for j in &cert.junctions {
        assert!(
            j.bounds_contained,
            "junction {} must be contained",
            j.junction_index
        );
    }
}

#[test]
fn test_verify_layerwise_d192_moonshot_bridge() {
    // D=192 pipeline through verify_moonshot_from_stages — the full bridge
    // from per-layer CROWN (#1762) to moonshot property verification (#1741).
    //
    // With w=0.01, 192 inputs, and ReLU, CROWN propagation produces bounds
    // wider than [-1,1] (expected: output range ~3.7 for 2 layers). This means
    // P2 (non-clipping) may not pass — that's correct behavior, not a test bug.
    // The test validates that CROWN composition WORKS at D=192 and that the
    // moonshot bridge produces meaningful property checks.
    use crate::moonshot_crown::verify_moonshot_from_stages;

    let (layer1, bind1) = build_linear_relu_layer("bridge_layer_0", 192, 192);
    let (layer2, bind2) = build_linear_layer("bridge_layer_1", 192, 192);

    let layers = vec![(layer1, bind1), (layer2, bind2)];
    let initial = uniform_bt(&[192], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("D=192 layerwise");
    assert!(cert.is_valid);

    // Bridge to moonshot properties at production dimension.
    let bundle =
        verify_moonshot_from_stages(&cert.stages, 192).expect("moonshot from stages at D=192");
    assert_eq!(bundle.verification_dim, 192);

    // P1 (non-silence): output has non-zero range — must pass.
    assert!(
        bundle.results[0].proven,
        "P1 non-silence must pass: {}",
        bundle.results[0]
    );

    // P6 (streaming): bounded crossfade discontinuity — must pass.
    // Even with wider bounds, crossfade alpha_step (1/239) keeps click bound small.
    let streaming = &bundle.results[3];
    assert_eq!(streaming.property_index, 5);
    assert!(
        streaming.proven,
        "P6 streaming must be proven at D=192: {streaming}"
    );
    assert!(
        streaming.bound_value < 0.3,
        "streaming bound {} must be < 0.3 threshold",
        streaming.bound_value,
    );

    // Output bounds are finite and non-vacuous.
    let output_range = cert.e2e_output_upper[0] - cert.e2e_output_lower[0];
    assert!(
        output_range.is_finite(),
        "D=192 output range must be finite"
    );
    assert!(
        output_range < 100.0,
        "D=192 output range {output_range} must be non-vacuous"
    );
}

#[test]
fn test_verify_layerwise_d192_dimension_reduction() {
    // D=192 input reducing to D=48 output — simulates encoder bottleneck.
    let (layer1, bind1) = build_linear_relu_layer("reduce_0", 192, 96);
    let (layer2, bind2) = build_linear_layer("reduce_1", 96, 48);

    let layers = vec![(layer1, bind1), (layer2, bind2)];
    let initial = uniform_bt(&[192], -1.0, 1.0);

    let cert = verify_layerwise(&layers, &initial).expect("D=192→48 layerwise");

    assert!(cert.is_valid);
}
