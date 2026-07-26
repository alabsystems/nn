// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fused GPU normalization kernels (LayerNorm, RmsNorm, GroupNorm, InstanceNorm).
//!
//! Verifies that fused GPU dispatch produces results within tolerance of the
//! CPU decomposed path. Edge-case tests (constant input, dim=1, large values)
//! and NaN/Inf propagation tests are in `dyn_tensor_metal_norm_tests_edge.rs`.
//!
//! Part of #1290.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{GroupNorm, InstanceNorm, LayerNorm, Module, RmsNorm};
use nn_core::{DType, Device};

use crate::test_common::{assert_gpu_matches_cpu, init};

// -- Fused LayerNorm GPU tests ------------------------------------------------

#[test]
fn test_fused_layer_norm_basic() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "fused_layer_norm_basic",
    );
}

#[test]
fn test_fused_layer_norm_batched_3d() {
    // [2, 3, 4] — batch of 2, sequence length 3, hidden dim 4
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();
            let x = DynTensor::new(&data, &[2, 3, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_layer_norm_batched_3d",
    );
}

#[test]
fn test_fused_layer_norm_nonunit_weight_bias() {
    // Weight != 1, bias != 0
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[2.0, 0.5, 3.0], &[3], dev).unwrap();
            let b = DynTensor::new(&[1.0, -1.0, 0.0], &[3], dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-6).unwrap();
            let x = DynTensor::new(&[10.0, 20.0, 30.0, 5.0, 15.0, 25.0], &[2, 3], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_layer_norm_nonunit_wb",
    );
}

// -- Fused RmsNorm GPU tests --------------------------------------------------

#[test]
fn test_fused_rms_norm_basic() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let layer = RmsNorm::new(w, 1e-5).unwrap();
            let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "fused_rms_norm_basic",
    );
}

#[test]
fn test_fused_rms_norm_with_weight() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[2.0, 0.5], &[2], dev).unwrap();
            let layer = RmsNorm::new(w, 1e-5).unwrap();
            let x = DynTensor::new(&[3.0, 4.0], &[1, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "fused_rms_norm_weight",
    );
}

#[test]
fn test_fused_rms_norm_batched() {
    // Batch of 3, hidden dim 8 — tests batched GPU dispatch
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[8], DType::F32, dev).unwrap();
            let layer = RmsNorm::new(w, 1e-5).unwrap();
            let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.3 + 0.1).collect();
            let x = DynTensor::new(&data, &[3, 8], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_rms_norm_batched",
    );
}

#[test]
fn test_fused_rms_norm_3d() {
    // [2, 3, 4] — tests 3D input (common in transformer hidden states)
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let layer = RmsNorm::new(w, 1e-6).unwrap();
            let data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.5).collect();
            let x = DynTensor::new(&data, &[2, 3, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_rms_norm_3d",
    );
}

// -- Fused GroupNorm GPU tests ------------------------------------------------

#[test]
fn test_fused_group_norm_1group() {
    // 1 group = equivalent to LayerNorm over channels*spatial
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(1, 4, w, b, 1e-5).unwrap();
            let x =
                DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 4, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_group_norm_1group",
    );
}

#[test]
fn test_fused_group_norm_2groups() {
    // 2 groups of 2 channels each
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[2.0, 1.0, 0.5, 3.0], &[4], dev).unwrap();
            let b = DynTensor::new(&[0.1, -0.1, 0.2, -0.2], &[4], dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0,
                ],
                &[1, 4, 3],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "fused_group_norm_2groups",
    );
}

#[test]
fn test_fused_group_norm_batched() {
    // Batch of 2, 4 channels, 2 groups, spatial dim 3
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 + 1.0).collect();
            let x = DynTensor::new(&data, &[2, 4, 3], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "fused_group_norm_batched",
    );
}

#[test]
fn test_fused_group_norm_8groups() {
    // Production-scale: 8 groups across 16 channels, spatial dim 8.
    // Exercises multi-group reduce path at higher group count than existing tests.
    assert_gpu_matches_cpu(
        |dev| {
            let w_data: Vec<f32> = (0..16).map(|i| 1.0 + (i as f32) * 0.1).collect();
            let b_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.01).collect();
            let w = DynTensor::new(&w_data, &[16], dev).unwrap();
            let b = DynTensor::new(&b_data, &[16], dev).unwrap();
            let layer = GroupNorm::new(8, 16, w, b, 1e-5).unwrap();
            let data: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
            let x = DynTensor::new(&data, &[1, 16, 8], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "fused_group_norm_8groups",
    );
}

#[test]
fn test_fused_group_norm_16groups_batched() {
    // 16 groups (one channel per group), batch=2, spatial=4.
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[16], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[16], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(16, 16, w, b, 1e-5).unwrap();
            let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.2).collect();
            let x = DynTensor::new(&data, &[2, 16, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "fused_group_norm_16groups_batched",
    );
}

// -- Fused InstanceNorm GPU tests (#2040) -------------------------------------

#[test]
fn test_fused_instance_norm_basic() {
    // [1, 2, 4] — 1 batch, 2 channels, spatial dim 4
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let x = DynTensor::new(
                &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                &[1, 2, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "fused_instance_norm_basic",
    );
}

#[test]
fn test_fused_instance_norm_batched() {
    // [2, 3, 4] — batch of 2, 3 channels, spatial dim 4
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.3 + 0.1).collect();
            let x = DynTensor::new(&data, &[2, 3, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_instance_norm_batched",
    );
}

#[test]
fn test_fused_instance_norm_large_spatial() {
    // [1, 4, 16] — larger spatial dimension exercises threadgroup reduction
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-6).unwrap();
            let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.5).collect();
            let x = DynTensor::new(&data, &[1, 4, 16], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_instance_norm_large_spatial",
    );
}

#[test]
fn test_fused_instance_norm_many_channels() {
    // [2, 8, 4] — 8 channels, exercises multi-row GPU dispatch
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let data: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
            let x = DynTensor::new(&data, &[2, 8, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "fused_instance_norm_many_channels",
    );
}

// -- Direct GPU-stays-on-GPU validation tests ---------------------------------

#[test]
fn test_fused_layer_norm_output_on_gpu() {
    init();
    let w = DynTensor::ones(&[3], DType::F32, &Device::metal()).unwrap();
    let b = DynTensor::zeros(&[3], DType::F32, &Device::metal()).unwrap();
    let layer = LayerNorm::new(w, b, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &Device::metal()).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(
        out.device(),
        Device::metal(),
        "fused LayerNorm output must stay on GPU"
    );
}

#[test]
fn test_fused_rms_norm_output_on_gpu() {
    init();
    let w = DynTensor::ones(&[3], DType::F32, &Device::metal()).unwrap();
    let layer = RmsNorm::new(w, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &Device::metal()).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(
        out.device(),
        Device::metal(),
        "fused RmsNorm output must stay on GPU"
    );
}

#[test]
fn test_fused_group_norm_output_on_gpu() {
    init();
    let w = DynTensor::ones(&[4], DType::F32, &Device::metal()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &Device::metal()).unwrap();
    let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 4, 2],
        &Device::metal(),
    )
    .unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(
        out.device(),
        Device::metal(),
        "fused GroupNorm output must stay on GPU"
    );
}

#[test]
fn test_fused_instance_norm_output_on_gpu() {
    init();
    let layer = InstanceNorm::new(1e-5).unwrap();
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
        &Device::metal(),
    )
    .unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(
        out.device(),
        Device::metal(),
        "fused InstanceNorm output must stay on GPU"
    );
}

// -- Fused vs Decomposed LayerNorm parity tests (#2939) -----------------------

/// Helper: compare fused single-dispatch LayerNorm against decomposed 14-dispatch path.
fn assert_fused_layer_norm_matches_decomposed(shape: &[usize], hidden_dim: usize, label: &str) {
    init();
    let dev = Device::metal();
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total)
        .map(|i| ((i as f32) - (total as f32) / 2.0) * 0.01)
        .collect();
    let x = DynTensor::new(&data, shape, &dev).unwrap();
    let w_data: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let b_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.0005).collect();
    let weight = DynTensor::new(&w_data, &[hidden_dim], &dev).unwrap();
    let bias = DynTensor::new(&b_data, &[hidden_dim], &dev).unwrap();

    let decomposed = super::MetalDynBackend::gpu_layer_norm(&x, &weight, &bias, 1e-5).unwrap();
    let fused = super::MetalDynBackend::gpu_layer_norm_fused(&x, &weight, &bias, 1e-5).unwrap();

    let dec_data = decomposed.to_flat_vec::<f32>().unwrap();
    let fused_data = fused.to_flat_vec::<f32>().unwrap();
    assert_eq!(dec_data.len(), fused_data.len(), "{label}: length mismatch");
    for (i, (d, f)) in dec_data.iter().zip(fused_data.iter()).enumerate() {
        let diff = (d - f).abs();
        assert!(
            diff < 1e-4,
            "{label}: mismatch at {i}: decomposed={d}, fused={f}, diff={diff}"
        );
    }
}

#[test]
fn test_fused_vs_decomposed_layer_norm_768() {
    // PlBert hidden dim
    assert_fused_layer_norm_matches_decomposed(&[1, 128, 768], 768, "PlBert_768");
}

#[test]
fn test_fused_vs_decomposed_layer_norm_256() {
    // TextEncoder hidden dim
    assert_fused_layer_norm_matches_decomposed(&[1, 50, 256], 256, "TextEncoder_256");
}

#[test]
fn test_fused_vs_decomposed_layer_norm_1024() {
    // Edge case: larger hidden dim
    assert_fused_layer_norm_matches_decomposed(&[2, 32, 1024], 1024, "large_1024");
}

#[test]
fn test_fused_vs_decomposed_layer_norm_small() {
    // Small hidden dim: exercises non-full threadgroup
    assert_fused_layer_norm_matches_decomposed(&[4, 8, 16], 16, "small_16");
}
