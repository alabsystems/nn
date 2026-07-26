// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Gated DeltaNet cell decomposition.

use super::*;

#[test]
fn test_decomposed_validates() {
    let def = build_gated_delta_net_decomposed(4, 8, 8, 0.125).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.name, "gated_delta_net_decomposed");
}

#[test]
fn test_decomposed_output_shape() {
    let def = build_gated_delta_net_decomposed(4, 8, 16, 0.125).expect("valid dims");
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 16]);
}

#[test]
fn test_decomposed_qwen35_config() {
    // Qwen3.5: 32 heads, head_k_dim=128, head_v_dim=128, scale=1/sqrt(128)
    let scale = 1.0 / (128.0_f32).sqrt();
    let def = build_gated_delta_net_decomposed(32, 128, 128, scale).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.nodes[def.output.index()].shape, vec![32, 128]);
}

#[test]
fn test_decomposed_dual_validates() {
    let def = build_gated_delta_net_decomposed_dual(4, 8, 8, 0.125).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.name, "gated_delta_net_decomposed_dual");
    // Output: [H, 1+K, V] = [4, 9, 8]
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 9, 8]);
}

#[test]
fn test_decomposed_dual_qwen35_shape() {
    let scale = 1.0 / (128.0_f32).sqrt();
    let def = build_gated_delta_net_decomposed_dual(32, 128, 128, scale).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    // Output: [32, 1+128, 128] = [32, 129, 128]
    assert_eq!(def.nodes[def.output.index()].shape, vec![32, 129, 128]);
}

#[test]
fn test_zero_heads_returns_error() {
    assert!(matches!(
        build_gated_delta_net_decomposed(0, 8, 8, 0.125).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetZeroDimension { param: "num_heads" })
    ));
}

#[test]
fn test_zero_key_dim_returns_error() {
    assert!(matches!(
        build_gated_delta_net_decomposed(4, 0, 8, 0.125).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetZeroDimension { param: "key_dim" })
    ));
}

#[test]
fn test_zero_value_dim_returns_error() {
    assert!(matches!(
        build_gated_delta_net_decomposed(4, 8, 0, 0.125).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetZeroDimension { param: "value_dim" })
    ));
}

#[test]
fn test_invalid_scale_zero() {
    assert!(matches!(
        build_gated_delta_net_decomposed(4, 8, 8, 0.0).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetScaleInvalid { .. })
    ));
}

#[test]
fn test_invalid_scale_negative() {
    assert!(matches!(
        build_gated_delta_net_decomposed(4, 8, 8, -1.0).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetScaleInvalid { .. })
    ));
}

#[test]
fn test_invalid_scale_nan() {
    assert!(matches!(
        build_gated_delta_net_decomposed(4, 8, 8, f32::NAN).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetScaleInvalid { .. })
    ));
}

#[test]
fn test_invalid_scale_inf() {
    assert!(matches!(
        build_gated_delta_net_decomposed(4, 8, 8, f32::INFINITY).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::GatedDeltaNetScaleInvalid { .. })
    ));
}

#[test]
fn test_dual_zero_dims_returns_error() {
    assert!(build_gated_delta_net_decomposed_dual(0, 8, 8, 0.125).is_err());
    assert!(build_gated_delta_net_decomposed_dual(4, 0, 8, 0.125).is_err());
    assert!(build_gated_delta_net_decomposed_dual(4, 8, 0, 0.125).is_err());
}

#[test]
fn test_decompose_in_builder_returns_both() {
    let mut builder = TensorBlockBuilder::new("test_dual");
    let q = builder.add_input("q", &[2, 4]);
    let k = builder.add_input("k", &[2, 4]);
    let v = builder.add_input("v", &[2, 8]);
    let state = builder.add_input("state", &[2, 4, 8]);
    let gate = builder.add_input("gate", &[2, 1, 1]);
    let beta = builder.add_input("beta", &[2, 1]);

    let outputs = decompose_gated_delta_net(&mut builder, q, k, v, state, gate, beta, 0.5, 2, 4, 8);
    assert_ne!(outputs.output, outputs.new_state);

    let def = builder.build(outputs.output).expect("valid graph");
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[def.output.index()].shape, vec![2, 8]);
}

#[test]
fn test_new_state_shape() {
    let mut builder = TensorBlockBuilder::new("test_state");
    let q = builder.add_input("q", &[2, 4]);
    let k = builder.add_input("k", &[2, 4]);
    let v = builder.add_input("v", &[2, 8]);
    let state = builder.add_input("state", &[2, 4, 8]);
    let gate = builder.add_input("gate", &[2, 1, 1]);
    let beta = builder.add_input("beta", &[2, 1]);

    let outputs = decompose_gated_delta_net(&mut builder, q, k, v, state, gate, beta, 0.5, 2, 4, 8);

    let def = builder.build(outputs.new_state).expect("valid graph");
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[def.output.index()].shape, vec![2, 4, 8]);
}
