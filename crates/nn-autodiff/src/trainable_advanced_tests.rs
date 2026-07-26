#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for advanced trainable layers: ConvTranspose1d, LSTM, MHA, SwiGlu.
//!
//! Split from `trainable_extra_tests.rs` to stay under 500-line limit.
//! Tests follow the same pattern: forward shape, backward gradients, vars() count.

use super::*;
use crate::grad::backward;
use nn_core::{Device, DynTensor};

// ── TrainableConvTranspose1d tests ──────────────────────────────────

#[test]
fn test_trainable_conv_transpose1d_forward_shape() {
    // Weight: [in_ch=2, out_ch=3, kernel=3], no bias
    let weight = DynTensor::from_vec(
        (0..18).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[2, 3, 3],
        &Device::Cpu,
    )
    .unwrap();
    let layer = TrainableConvTranspose1d::from_tensors(
        weight, None, 0, // padding
        1, // stride
        1, // dilation
        1, // groups
        0, // output_padding
    );

    // Input: [batch=1, in_ch=2, length=4]
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&tx).unwrap();

    // ConvTranspose1d output length: (4-1)*1 - 2*0 + 1*(3-1) + 0 + 1 = 6
    assert_eq!(y.dims(), &[1, 3, 6]);
}

#[test]
fn test_trainable_conv_transpose1d_with_bias() {
    let weight = DynTensor::from_vec(
        (0..18).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[2, 3, 3],
        &Device::Cpu,
    )
    .unwrap();
    let bias = DynTensor::from_vec(vec![0.1, 0.2, 0.3], &[3], &Device::Cpu).unwrap();
    let layer = TrainableConvTranspose1d::from_tensors(weight, Some(bias), 0, 1, 1, 1, 0);

    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&tx).unwrap();

    assert_eq!(y.dims(), &[1, 3, 6]);
    // Bias adds constant per channel — verify output is non-trivial
    let vals = y.tensor().to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().any(|&v| v > 0.1), "output should be non-zero");
}

#[test]
fn test_trainable_conv_transpose1d_backward_produces_gradients() {
    let weight = DynTensor::from_vec(
        (0..18).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[2, 3, 3],
        &Device::Cpu,
    )
    .unwrap();
    let layer = TrainableConvTranspose1d::from_tensors(weight, None, 0, 1, 1, 1, 0);

    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&tx).unwrap();

    let loss = y
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let w_grad = grads.get(layer.weight()).expect("weight gradient");
    assert_eq!(w_grad.dims(), &[2, 3, 3]);

    let vals = w_grad.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().any(|&v| v.abs() > 1e-7),
        "weight gradient should be non-zero"
    );
}

#[test]
fn test_trainable_conv_transpose1d_vars() {
    let weight = DynTensor::from_vec(vec![0.0; 18], &[2, 3, 3], &Device::Cpu).unwrap();
    let bias = DynTensor::from_vec(vec![0.0; 3], &[3], &Device::Cpu).unwrap();

    let layer_no_bias = TrainableConvTranspose1d::from_tensors(weight.clone(), None, 0, 1, 1, 1, 0);
    assert_eq!(layer_no_bias.vars().len(), 1);

    let layer_with_bias = TrainableConvTranspose1d::from_tensors(weight, Some(bias), 0, 1, 1, 1, 0);
    assert_eq!(layer_with_bias.vars().len(), 2);
}

// ── TrainableLstm tests ─────────────────────────────────────────────

#[test]
fn test_trainable_lstm_forward_shape() {
    let lstm = TrainableLstm::new(4, 8, true).unwrap();
    // Input: [batch=2, input_size=4]
    let x = DynTensor::from_vec(
        (0..8).map(|i| i as f32 * 0.1).collect(),
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = lstm.forward(&tx).unwrap();

    // Output: [batch=2, hidden_size=8]
    assert_eq!(y.dims(), &[2, 8]);
}

#[test]
fn test_trainable_lstm_forward_seq_shape() {
    let lstm = TrainableLstm::new(4, 8, true).unwrap();
    // Input: [batch=1, seq_len=3, input_size=4]
    let x = DynTensor::from_vec(
        (0..12).map(|i| i as f32 * 0.1).collect(),
        &[1, 3, 4],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let (outputs, state) = lstm.forward_seq(&tx, None).unwrap();

    assert_eq!(outputs.len(), 3);
    for h in &outputs {
        assert_eq!(h.dims(), &[1, 8]);
    }
    assert_eq!(state.h.dims(), &[1, 8]);
    assert_eq!(state.c.dims(), &[1, 8]);
}

#[test]
fn test_trainable_lstm_backward_produces_gradients() {
    // Use small non-zero weights to ensure non-trivial gradients.
    let input_size = 3;
    let hidden_size = 4;
    let w_ih_data: Vec<f32> = (0..(4 * hidden_size * input_size))
        .map(|i| (i as f32 + 1.0) * 0.001)
        .collect();
    let w_hh_data: Vec<f32> = (0..(4 * hidden_size * hidden_size))
        .map(|i| (i as f32 + 1.0) * 0.001)
        .collect();
    let w_ih =
        DynTensor::from_vec(w_ih_data, &[4 * hidden_size, input_size], &Device::Cpu).unwrap();
    let w_hh =
        DynTensor::from_vec(w_hh_data, &[4 * hidden_size, hidden_size], &Device::Cpu).unwrap();
    let lstm = TrainableLstm::from_tensors(w_ih, w_hh, None, None, hidden_size);

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = lstm.forward(&tx).unwrap();

    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let w_ih_grad = grads.get(lstm.w_ih()).expect("w_ih gradient");
    assert_eq!(w_ih_grad.dims(), &[4 * hidden_size, input_size]);

    let w_hh_grad = grads.get(lstm.w_hh()).expect("w_hh gradient");
    assert_eq!(w_hh_grad.dims(), &[4 * hidden_size, hidden_size]);

    let vals = w_ih_grad.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().any(|&v| v.abs() > 1e-10),
        "w_ih gradient should be non-zero"
    );
}

#[test]
fn test_trainable_lstm_vars() {
    let lstm_with_bias = TrainableLstm::new(4, 8, true).unwrap();
    assert_eq!(lstm_with_bias.vars().len(), 4); // w_ih, w_hh, b_ih, b_hh

    let lstm_no_bias = TrainableLstm::new(4, 8, false).unwrap();
    assert_eq!(lstm_no_bias.vars().len(), 2); // w_ih, w_hh only
}

// ── TrainableMultiHeadAttention tests ───────────────────────────────

#[test]
fn test_trainable_mha_forward_shape() {
    let mha = TrainableMultiHeadAttention::zeros(8, 2, true).unwrap();
    // Input: [batch=1, seq_len=3, model_dim=8]
    let x = DynTensor::from_vec(
        (0..24).map(|i| i as f32 * 0.01).collect(),
        &[1, 3, 8],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = mha.forward(&tx).unwrap();

    assert_eq!(y.dims(), &[1, 3, 8]);
}

#[test]
fn test_trainable_mha_invalid_heads() {
    let result = TrainableMultiHeadAttention::zeros(8, 0, false);
    assert!(result.is_err(), "num_heads=0 should be rejected");

    let result = TrainableMultiHeadAttention::zeros(8, 3, false);
    assert!(result.is_err(), "model_dim=8 not divisible by num_heads=3");
}

#[test]
fn test_trainable_mha_backward_produces_gradients() {
    use crate::trainable::TrainableLinear;

    let model_dim = 4;
    let num_heads = 2;

    // Use small non-zero weights so Q/K produce distinguishable vectors
    // (zero weights → uniform softmax → zero Q/K gradients).
    let make_weight = |seed: usize| {
        DynTensor::from_vec(
            (0..model_dim * model_dim)
                .map(|i| ((i + seed) as f32 + 1.0) * 0.01)
                .collect(),
            &[model_dim, model_dim],
            &Device::Cpu,
        )
        .unwrap()
    };
    let q = TrainableLinear::from_tensors(make_weight(0), None);
    let k = TrainableLinear::from_tensors(make_weight(16), None);
    let v = TrainableLinear::from_tensors(make_weight(32), None);
    let out = TrainableLinear::from_tensors(make_weight(48), None);
    let mha = TrainableMultiHeadAttention::new(q, k, v, out, num_heads, model_dim).unwrap();

    // Use non-zero input so gradients are non-trivial.
    let x = DynTensor::from_vec(
        (0..12).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[1, 3, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = mha.forward(&tx).unwrap();

    let loss = y
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // All 4 projections (Q, K, V, out) should receive gradients.
    let all_vars = mha.vars();
    assert_eq!(all_vars.len(), 4); // Q, K, V, out (no bias)

    for (i, var) in all_vars.iter().enumerate() {
        let g = grads
            .get(var)
            .unwrap_or_else(|| panic!("projection {i} should have gradient"));
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().any(|&v| v.abs() > 1e-10),
            "projection {i} gradient should be non-zero"
        );
    }
}

#[test]
fn test_trainable_mha_vars() {
    let mha_no_bias = TrainableMultiHeadAttention::zeros(8, 2, false).unwrap();
    assert_eq!(mha_no_bias.vars().len(), 4); // Q, K, V, out weights

    let mha_with_bias = TrainableMultiHeadAttention::zeros(8, 2, true).unwrap();
    assert_eq!(mha_with_bias.vars().len(), 8); // Q, K, V, out × (weight + bias)
}

// ── TrainableSwiGlu tests ───────────────────────────────────────────

#[test]
fn test_trainable_swiglu_forward_shape() {
    let swiglu = TrainableSwiGlu::zeros(8, 16, true).unwrap();
    // Input: [batch=2, dim=8]
    let x = DynTensor::from_vec(
        (0..16).map(|i| i as f32 * 0.1).collect(),
        &[2, 8],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = swiglu.forward(&tx).unwrap();

    // Output: [batch=2, dim=8] (w_down projects hidden_dim=16 back to dim=8)
    assert_eq!(y.dims(), &[2, 8]);
}

#[test]
fn test_trainable_swiglu_backward_produces_gradients() {
    use crate::trainable::TrainableLinear;

    let dim = 4;
    let hidden_dim = 8;

    // Use small non-zero weights so SiLU(W_gate @ x) is non-zero
    // (zero weights → gate output = 0 → SiLU(0) = 0 → zero gradients).
    // Weight shapes: [out_features, in_features] (PyTorch convention).
    // w_gate: dim → hidden_dim → shape [hidden_dim, dim]
    // w_up:   dim → hidden_dim → shape [hidden_dim, dim]
    // w_down: hidden_dim → dim → shape [dim, hidden_dim]
    let w_gate = TrainableLinear::from_tensors(
        DynTensor::from_vec(
            (0..hidden_dim * dim)
                .map(|i| (i as f32 + 1.0) * 0.01)
                .collect(),
            &[hidden_dim, dim],
            &Device::Cpu,
        )
        .unwrap(),
        None,
    );
    let w_up = TrainableLinear::from_tensors(
        DynTensor::from_vec(
            (0..hidden_dim * dim)
                .map(|i| (i as f32 + 33.0) * 0.01)
                .collect(),
            &[hidden_dim, dim],
            &Device::Cpu,
        )
        .unwrap(),
        None,
    );
    let w_down = TrainableLinear::from_tensors(
        DynTensor::from_vec(
            (0..dim * hidden_dim)
                .map(|i| (i as f32 + 65.0) * 0.01)
                .collect(),
            &[dim, hidden_dim],
            &Device::Cpu,
        )
        .unwrap(),
        None,
    );
    let swiglu = TrainableSwiGlu::new(w_gate, w_up, w_down);

    let x = DynTensor::from_vec(
        (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[2, dim],
        &Device::Cpu,
    )
    .unwrap();
    let tx = Arc::new(TrackedTensor::from_tensor(x));
    let y = swiglu.forward(&tx).unwrap();

    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    // All 3 projections (gate, up, down) should receive gradients.
    let all_vars = swiglu.vars();
    assert_eq!(all_vars.len(), 3); // gate, up, down weights (no bias)

    for (i, var) in all_vars.iter().enumerate() {
        let g = grads
            .get(var)
            .unwrap_or_else(|| panic!("projection {i} should have gradient"));
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().any(|&v| v.abs() > 1e-10),
            "projection {i} gradient should be non-zero"
        );
    }
}

#[test]
fn test_trainable_swiglu_vars() {
    let swiglu_no_bias = TrainableSwiGlu::zeros(4, 8, false).unwrap();
    assert_eq!(swiglu_no_bias.vars().len(), 3); // gate, up, down weights

    let swiglu_with_bias = TrainableSwiGlu::zeros(4, 8, true).unwrap();
    assert_eq!(swiglu_with_bias.vars().len(), 6); // gate, up, down × (weight + bias)
}
