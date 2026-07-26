#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ALBERT-scale and decomposed GPU normalization tests.
//!
//! Extracted from `dyn_tensor_metal_norm_tests_edge.rs` — tests fused and
//! decomposed GPU norm paths at ALBERT-scale dimensions (hidden=768, eps=1e-12).
//!
//! Part of #1656, #1678.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{LayerNorm, RmsNorm};
use nn_core::{DType, Device};

use crate::test_common::{assert_gpu_matches_cpu, init};

// -- ALBERT-scale GPU norm tests (#1656) --------------------------------------
//
// ALBERT uses LayerNorm with hidden_size=768 and eps=1e-12.
// dvoice reports 100% NaN output on Metal GPU for these dimensions.

/// LayerNorm with ALBERT-scale hidden dim (768) and small eps (1e-12).
/// Reproduces #1656: Metal GPU produces NaN for [1, T, 768] inputs.
#[test]
fn test_fused_layer_norm_albert_768_eps12() {
    assert_gpu_matches_cpu(
        |dev| {
            let hidden = 768;
            let w = DynTensor::ones(&[hidden], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[hidden], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-12).unwrap();
            // ALBERT-like input: [1, 4, 768] with typical activation range
            let data: Vec<f32> = (0..4 * hidden)
                .map(|i| ((i as f32) * 0.013 - 20.0).sin())
                .collect();
            let x = DynTensor::new(&data, &[1, 4, hidden], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "LayerNorm ALBERT-scale hidden=768, eps=1e-12 (#1656)",
    );
}

/// LayerNorm with dim=768, eps=1e-5 (standard eps) — isolates dim vs eps.
#[test]
fn test_fused_layer_norm_dim768_standard_eps() {
    assert_gpu_matches_cpu(
        |dev| {
            let hidden = 768;
            let w = DynTensor::ones(&[hidden], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[hidden], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let data: Vec<f32> = (0..2 * hidden)
                .map(|i| ((i as f32) * 0.01 - 5.0).sin())
                .collect();
            let x = DynTensor::new(&data, &[1, 2, hidden], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "LayerNorm dim=768, eps=1e-5",
    );
}

/// LayerNorm with dim=256 — tests medium-scale reduction.
#[test]
fn test_fused_layer_norm_dim256() {
    assert_gpu_matches_cpu(
        |dev| {
            let hidden = 256;
            let w = DynTensor::ones(&[hidden], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[hidden], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-12).unwrap();
            let data: Vec<f32> = (0..3 * hidden).map(|i| (i as f32) * 0.005 - 1.0).collect();
            let x = DynTensor::new(&data, &[1, 3, hidden], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "LayerNorm dim=256, eps=1e-12",
    );
}

/// LayerNorm with dim=768, T=64 — exercises larger sequence length.
#[test]
fn test_fused_layer_norm_albert_768_long_seq() {
    assert_gpu_matches_cpu(
        |dev| {
            let hidden = 768;
            let seq = 64;
            let w = DynTensor::ones(&[hidden], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[hidden], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-12).unwrap();
            let data: Vec<f32> = (0..seq * hidden)
                .map(|i| ((i as f32) * 0.007 - 100.0).sin() * 2.0)
                .collect();
            let x = DynTensor::new(&data, &[1, seq, hidden], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "LayerNorm ALBERT hidden=768, T=64, eps=1e-12 (#1656)",
    );
}

/// RmsNorm with dim=768 — tests if RmsNorm also has the large-dim issue.
#[test]
fn test_fused_rms_norm_dim768() {
    assert_gpu_matches_cpu(
        |dev| {
            let hidden = 768;
            let w = DynTensor::ones(&[hidden], DType::F32, dev).unwrap();
            let layer = RmsNorm::new(w, 1e-12).unwrap();
            let data: Vec<f32> = (0..2 * hidden)
                .map(|i| ((i as f32) * 0.02 - 10.0).sin())
                .collect();
            let x = DynTensor::new(&data, &[1, 2, hidden], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "RmsNorm dim=768, eps=1e-12",
    );
}

// -- Decomposed GPU path tests (#1656) ----------------------------------------
//
// dvoice reports NaN from `mean_keepdim` on Metal for dim=768.
// The fused kernel path bypasses `mean_keepdim` — these tests exercise
// the decomposed path that dvoice may be using directly.

/// GPU mean_keepdim with dim=768 — tests standalone reduction at ALBERT scale.
/// Regression test for dvoice#1122 hypothesis: "Metal mean_keepdim
/// corruption for long T instance-norm inputs."
#[test]
fn test_gpu_mean_keepdim_dim768() {
    init();
    let hidden = 768;
    let seq = 4;
    let data: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) * 0.013 - 20.0).sin())
        .collect();
    let gpu = DynTensor::new(&data, &[1, seq, hidden], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[1, seq, hidden], &Device::Cpu).unwrap();

    // mean_keepdim over last dim (768) — this is what LayerNorm decomposed uses
    let gpu_mean = gpu.mean_keepdim(2).unwrap();
    let cpu_mean = cpu.mean_keepdim(2).unwrap();

    let gpu_vals = gpu_mean
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_mean.to_flat_vec::<f32>().unwrap();

    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "mean_keepdim mismatch at {i}: gpu={g}, cpu={c}"
        );
        assert!(g.is_finite(), "GPU mean at {i} is not finite: {g}");
    }
}

/// Decomposed LayerNorm on GPU (bypassing fused kernel) with dim=768.
/// Exercises the exact code path dvoice#1122 hit: mean_keepdim →
/// broadcast_sub → sqr → mean_keepdim → broadcast_add(eps) → sqrt →
/// recip → multiply → affine.
#[test]
fn test_decomposed_layer_norm_gpu_dim768() {
    init();
    let hidden = 768;
    let seq = 4;
    let eps = 1e-12_f64;

    let data: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) * 0.013 - 20.0).sin())
        .collect();

    let gpu_x = DynTensor::new(&data, &[1, seq, hidden], &Device::metal()).unwrap();
    let cpu_x = DynTensor::new(&data, &[1, seq, hidden], &Device::Cpu).unwrap();

    // Decomposed LayerNorm (same ops as layers.rs:172-180, without fused path)
    let decomposed_ln = |x: &DynTensor| -> DynTensor {
        let rank = x.rank();
        let last_dim = rank - 1;
        let mean = x.mean_keepdim(last_dim).unwrap();
        let centered = x.broadcast_sub(&mean).unwrap();
        let var = centered.sqr().unwrap().mean_keepdim(last_dim).unwrap();
        let eps_t = DynTensor::full(vec![1; rank], eps, DType::F32, &x.device()).unwrap();
        let std_inv = var
            .broadcast_add(&eps_t)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        centered.broadcast_mul(&std_inv).unwrap()
    };

    let gpu_result = decomposed_ln(&gpu_x);
    let cpu_result = decomposed_ln(&cpu_x);

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    // Check no NaN in GPU output
    let nan_count = gpu_vals.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        nan_count,
        0,
        "decomposed GPU LayerNorm produced {nan_count}/{} NaN values",
        gpu_vals.len()
    );

    // Check CPU/GPU parity
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "decomposed LN mismatch at {i}: gpu={g}, cpu={c}"
        );
    }
}
