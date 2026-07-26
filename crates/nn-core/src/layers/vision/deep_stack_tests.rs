// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`DeepStackFusion`].
//!
//! VitEncoder::forward_deepstack tests are in `vit_tests_deepstack.rs`
//! (within the `vit` module where `pub(super)` fields are accessible).

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module};
use crate::Device;

// -- Helpers ------------------------------------------------------------------

fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect()
}

fn make_linear(out: usize, inp: usize, seed: f32) -> Linear {
    let w = DynTensor::from_vec(det_data(out * inp, seed), &[out, inp], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(det_data(out, seed + 100.0), &[out], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

fn make_fusion(input_hidden: usize, num_layers: usize, output_hidden: usize) -> DeepStackFusion {
    let concat_dim = num_layers * input_hidden;
    let proj = make_linear(output_hidden, concat_dim, 42.0);
    DeepStackFusion::new(proj, input_hidden, num_layers, output_hidden).unwrap()
}

// -- DeepStackFusion construction ---------------------------------------------

#[test]
fn test_fusion_new_valid() {
    let fusion = make_fusion(64, 3, 128);
    assert_eq!(fusion.num_layers(), 3);
    assert_eq!(fusion.input_hidden_size(), 64);
    assert_eq!(fusion.output_hidden_size(), 128);
}

#[test]
fn test_fusion_new_zero_layers() {
    let proj = make_linear(128, 64, 1.0);
    let err = DeepStackFusion::new(proj, 64, 0, 128)
        .unwrap_err()
        .to_string();
    assert!(err.contains("num_layers"), "error: {err}");
}

#[test]
fn test_fusion_new_zero_input_hidden() {
    let proj = make_linear(128, 64, 1.0);
    let err = DeepStackFusion::new(proj, 0, 3, 128)
        .unwrap_err()
        .to_string();
    assert!(err.contains("input_hidden_size"), "error: {err}");
}

#[test]
fn test_fusion_new_zero_output_hidden() {
    let proj = make_linear(1, 192, 1.0);
    let err = DeepStackFusion::new(proj, 64, 3, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("output_hidden_size"), "error: {err}");
}

// -- forward_multi ------------------------------------------------------------

#[test]
fn test_fusion_forward_multi_shape() {
    let hidden = 32;
    let num_layers = 3;
    let output = 64;
    let fusion = make_fusion(hidden, num_layers, output);

    let b = 2;
    let s = 4;
    let intermediates: Vec<DynTensor> = (0..num_layers)
        .map(|i| {
            DynTensor::from_vec(
                det_data(b * s * hidden, i as f32),
                &[b, s, hidden],
                &Device::Cpu,
            )
            .unwrap()
        })
        .collect();

    let fused = fusion.forward_multi(&intermediates).unwrap();
    assert_eq!(fused.dims(), &[b, s, output]);
}

#[test]
fn test_fusion_forward_multi_single_layer() {
    let hidden = 16;
    let output = 32;
    let fusion = make_fusion(hidden, 1, output);

    let x =
        DynTensor::from_vec(det_data(2 * 5 * hidden, 0.0), &[2, 5, hidden], &Device::Cpu).unwrap();
    let fused = fusion.forward_multi(&[x]).unwrap();
    assert_eq!(fused.dims(), &[2, 5, output]);
}

#[test]
fn test_fusion_forward_multi_wrong_count() {
    let fusion = make_fusion(32, 3, 64);
    let x = DynTensor::from_vec(det_data(2 * 4 * 32, 0.0), &[2, 4, 32], &Device::Cpu).unwrap();
    let err = fusion
        .forward_multi(&[x.clone(), x])
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected 3"), "error: {err}");
}

#[test]
fn test_fusion_forward_multi_wrong_hidden_size() {
    let fusion = make_fusion(32, 2, 64);
    let wrong = DynTensor::from_vec(det_data(2 * 4 * 16, 0.0), &[2, 4, 16], &Device::Cpu).unwrap();
    let correct =
        DynTensor::from_vec(det_data(2 * 4 * 32, 0.0), &[2, 4, 32], &Device::Cpu).unwrap();
    let err = fusion
        .forward_multi(&[wrong, correct])
        .unwrap_err()
        .to_string();
    assert!(err.contains("hidden_size"), "error: {err}");
}

#[test]
fn test_fusion_forward_multi_mismatched_shapes() {
    let fusion = make_fusion(32, 2, 64);
    let a = DynTensor::from_vec(det_data(2 * 4 * 32, 0.0), &[2, 4, 32], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(det_data(2 * 5 * 32, 1.0), &[2, 5, 32], &Device::Cpu).unwrap();
    let err = fusion.forward_multi(&[a, b]).unwrap_err().to_string();
    assert!(
        err.contains("mismatch") || err.contains("shape"),
        "error: {err}"
    );
}

#[test]
fn test_fusion_forward_multi_output_finite() {
    let fusion = make_fusion(32, 3, 64);
    let intermediates: Vec<DynTensor> = (0..3)
        .map(|i| {
            DynTensor::from_vec(det_data(4 * 32, i as f32), &[1, 4, 32], &Device::Cpu).unwrap()
        })
        .collect();
    let fused = fusion.forward_multi(&intermediates).unwrap();
    let data = fused.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

// -- Module trait (pre-concatenated input) ------------------------------------

#[test]
fn test_fusion_module_trait() {
    let hidden = 32;
    let num_layers = 3;
    let output = 64;
    let fusion = make_fusion(hidden, num_layers, output);

    let concat_dim = num_layers * hidden;
    let x = DynTensor::from_vec(
        det_data(2 * 4 * concat_dim, 0.0),
        &[2, 4, concat_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = Module::forward(&fusion, &x).unwrap();
    assert_eq!(out.dims(), &[2, 4, output]);
}
