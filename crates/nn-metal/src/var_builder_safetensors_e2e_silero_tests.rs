#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E Silero VAD architecture test via safetensors + VarBuilder.
//!
//! Extracted from `var_builder_safetensors_e2e_tests.rs` for 500-line compliance (#1306).

use std::path::Path;

use nn_core::{DType, Device, VarBuilder};

use crate::context::MetalContext;
use crate::var_builder_safetensors::from_mmaped_safetensors_with_ctx;

/// Silero VAD encoder: [(in_ch, out_ch, kernel, stride, padding)].
const SILERO_ENCODER: [(usize, usize, usize, usize, usize); 4] = [
    (129, 128, 3, 1, 1),
    (128, 64, 3, 2, 1),
    (64, 64, 3, 2, 1),
    (64, 128, 3, 1, 1),
];
const SILERO_HIDDEN: usize = 128;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_vb_st_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Generate deterministic weight values for a given size and seed.
fn det_weights(size: usize, seed: usize) -> Vec<f32> {
    (0..size)
        .map(|j| ((j as f32 * 0.01 + seed as f32 * 0.001) % 0.1) - 0.05)
        .collect()
}

/// Create a safetensors file with Silero VAD weights (PyTorch naming convention).
fn create_silero_vad_safetensors(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let four_h = 4 * SILERO_HIDDEN;
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

    // Encoder Conv1d weights
    for (i, &(in_ch, out_ch, k, _, _)) in SILERO_ENCODER.iter().enumerate() {
        tensors.push((
            format!("encoder.{i}.weight"),
            vec![out_ch, in_ch, k],
            det_weights(out_ch * in_ch * k, i),
        ));
        tensors.push((format!("encoder.{i}.bias"), vec![out_ch], vec![0.0; out_ch]));
    }
    // LSTM weights
    tensors.push((
        "lstm.weight_ih_l0".into(),
        vec![four_h, SILERO_HIDDEN],
        det_weights(four_h * SILERO_HIDDEN, 10),
    ));
    tensors.push((
        "lstm.weight_hh_l0".into(),
        vec![four_h, SILERO_HIDDEN],
        det_weights(four_h * SILERO_HIDDEN, 11),
    ));
    tensors.push(("lstm.bias_ih_l0".into(), vec![four_h], vec![0.0; four_h]));
    tensors.push(("lstm.bias_hh_l0".into(), vec![four_h], vec![0.0; four_h]));
    // Output linear: 128 → 1
    let out_w: Vec<f32> = (0..SILERO_HIDDEN)
        .map(|j| (j as f32 * 0.01) - 0.64)
        .collect();
    tensors.push(("output.weight".into(), vec![1, SILERO_HIDDEN], out_w));
    tensors.push(("output.bias".into(), vec![1], vec![0.0]));

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            let bytes = bytemuck::cast_slice::<f32, u8>(values);
            (
                name.clone(),
                TensorView::new(StDtype::F32, shape.clone(), bytes).expect("valid view"),
            )
        })
        .collect();
    std::fs::write(path, serialize(views, None).expect("serialize")).expect("write");
}

/// Load Silero VAD layers from a VarBuilder.
fn load_silero_vad_layers(
    vb: &VarBuilder,
) -> (
    Vec<nn_core::layers::Conv1d>,
    nn_core::layers::Lstm,
    nn_core::layers::Linear,
) {
    use nn_core::layers::{conv1d, linear, lstm, Conv1dConfig};

    let mut convs = Vec::with_capacity(4);
    for (i, &(in_ch, out_ch, k, stride, padding)) in SILERO_ENCODER.iter().enumerate() {
        let cfg = Conv1dConfig::default()
            .with_padding(padding)
            .with_stride(stride);
        convs.push(
            conv1d(in_ch, out_ch, k, cfg, vb.pp(format!("encoder.{i}")))
                .unwrap_or_else(|e| panic!("load encoder.{i}: {e}")),
        );
    }
    let lstm = lstm(SILERO_HIDDEN, SILERO_HIDDEN, vb.pp("lstm")).expect("load lstm");
    let linear = linear(SILERO_HIDDEN, 1, vb.pp("output")).expect("load output");
    (convs, lstm, linear)
}

/// Demonstrates the full PyTorch-to-nn import path on Silero VAD architecture:
///   safetensors → VarBuilder → Conv1d + Lstm + Linear → sigmoid probability.
#[test]
fn test_e2e_silero_vad_architecture_via_var_builder() {
    use nn_core::layers::{Activation, Module};

    let dir = temp_dir("e2e_silero_vad");
    let path = dir.join("silero_vad.safetensors");
    create_silero_vad_safetensors(&path);

    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test file is not modified during the test.
    let vb = unsafe {
        from_mmaped_safetensors_with_ctx(&[path.as_path()], DType::F32, &Device::Cpu, &ctx)
            .expect("load silero_vad.safetensors")
    };
    let (conv_layers, lstm, output_linear) = load_silero_vad_layers(&vb);

    // Forward pass: STFT magnitude [1, 129, 4] → encoder → LSTM → output
    let relu = Activation::Relu;
    let input_data: Vec<f32> = (0..129 * 4).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut x =
        nn_core::DynTensor::from_vec(input_data, &[1, 129, 4], &Device::Cpu).expect("input");

    // Encoder: 4× (Conv1d + ReLU), temporal: 4 → 4 → 2 → 1 → 1
    for (i, conv) in conv_layers.iter().enumerate() {
        x = relu
            .forward(&conv.forward(&x).unwrap_or_else(|e| panic!("enc{i}: {e}")))
            .unwrap_or_else(|e| panic!("relu{i}: {e}"));
    }
    assert_eq!(x.dims(), &[1, 128, 1], "encoder output shape");

    // Squeeze → LSTM → ReLU → Linear → Sigmoid
    x = x.squeeze(2).expect("squeeze");
    let (h, _) = lstm.forward(&x, None).expect("lstm");
    let logit = output_linear
        .forward(&relu.forward(&h).expect("relu"))
        .expect("linear");
    let prob = logit
        .sigmoid()
        .expect("sigmoid")
        .to_flat_vec::<f32>()
        .expect("readback");

    assert_eq!(prob.len(), 1);
    assert!(
        prob[0].is_finite(),
        "probability must be finite: {}",
        prob[0]
    );
    assert!(
        (0.0..=1.0).contains(&prob[0]),
        "probability in [0,1]: {}",
        prob[0]
    );

    std::fs::remove_dir_all(&dir).ok();
}
