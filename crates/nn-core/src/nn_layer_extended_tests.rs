// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended nn layer configuration and shape tests.
//!
//! Covers forward-pass shape contracts, numerical properties, and edge cases
//! for Linear, Conv1d/Conv2d, LayerNorm, BatchNorm, Dropout, Embedding,
//! LSTM/BiLSTM, Activation functions, Softmax/LogSoftmax, and
//! MultiHeadAttention.
//!
//! Part of #4560.

use crate::dyn_tensor::DynTensor;
use crate::layers::{
    Activation, BatchNorm, BiLstm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Dropout, Embedding,
    LayerNorm, Linear, Lstm, LstmState, Module, MultiHeadAttention,
};
use crate::{conv1d_out_len, conv2d_out_len, DType, Device};

// =============================================================================
// 1. Linear layer
// =============================================================================

#[test]
fn test_linear_weight_shape_matches_constructor() {
    let w = DynTensor::from_vec(vec![1.0; 12], &[4, 3], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![0.0; 4], &[4], &Device::Cpu).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    assert_eq!(lin.out_features(), 4);
    assert_eq!(lin.in_features(), 3);
    assert_eq!(lin.weight().dims(), &[4, 3]);
    assert_eq!(lin.bias().unwrap().dims(), &[4]);
}

#[test]
fn test_linear_no_bias_variant() {
    let w = DynTensor::from_vec(vec![1.0; 6], &[2, 3], &Device::Cpu).unwrap();
    let lin = Linear::new(w, None).unwrap();
    assert!(lin.bias().is_none());
    assert_eq!(lin.out_features(), 2);
}

#[test]
fn test_linear_output_shape_2d() {
    // weight [4, 3], input [2, 3] -> output [2, 4]
    let w = DynTensor::from_vec(vec![1.0; 12], &[4, 3], &Device::Cpu).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 6], &[2, 3], &Device::Cpu).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
}

#[test]
fn test_linear_output_shape_3d() {
    // weight [8, 4], input [2, 5, 4] -> output [2, 5, 8]
    let w = DynTensor::from_vec(vec![0.1; 32], &[8, 4], &Device::Cpu).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 40], &[2, 5, 4], &Device::Cpu).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 5, 8]);
}

#[test]
fn test_linear_bias_adds_correctly() {
    // weight = identity 2x2, bias = [10, 20]
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0], &[2], &Device::Cpu).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    let x = DynTensor::from_vec(vec![3.0, 5.0], &[1, 2], &Device::Cpu).unwrap();
    let y = lin.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 13.0).abs() < 1e-5);
    assert!((vals[1] - 25.0).abs() < 1e-5);
}

#[test]
fn test_linear_rejects_1d_weight() {
    let w = DynTensor::from_vec(vec![1.0; 4], &[4], &Device::Cpu).unwrap();
    assert!(Linear::new(w, None).is_err());
}

#[test]
fn test_linear_rejects_mismatched_bias_size() {
    let w = DynTensor::from_vec(vec![1.0; 6], &[2, 3], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![0.0; 3], &[3], &Device::Cpu).unwrap();
    assert!(Linear::new(w, Some(b)).is_err());
}

// =============================================================================
// 2. Conv1d / Conv2d
// =============================================================================

#[test]
fn test_conv1d_output_size_formula_stride2_padding1() {
    // out = (L_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1
    // = (20 + 2 - 1*(3-1) - 1) / 2 + 1 = 19/2 + 1 = 10
    let out = conv1d_out_len(20, 3, 1, 2, 1).unwrap();
    assert_eq!(out, 10);
}

#[test]
fn test_conv1d_output_size_dilation3() {
    // effective_k = (3-1)*3 + 1 = 7
    // out = (20 - 7)/1 + 1 = 14
    let out = conv1d_out_len(20, 3, 0, 1, 3).unwrap();
    assert_eq!(out, 14);
}

#[test]
fn test_conv1d_forward_output_shape() {
    // in_channels=3, out_channels=8, kernel=3, padding=1, stride=1
    // weight: [8, 3, 3], input: [1, 3, 16] -> output: [1, 8, 16]
    let w = DynTensor::from_vec(vec![0.01; 72], &[8, 3, 3], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![0.0; 8], &[8], &Device::Cpu).unwrap();
    let conv = Conv1d::new(w, Some(b), Conv1dConfig::new(1, 1, 1)).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 48], &[1, 3, 16], &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 16]);
}

#[test]
fn test_conv1d_stride_reduces_length() {
    // weight: [4, 2, 5], stride=2, padding=0, dilation=1
    // input: [1, 2, 20] -> out_len = (20 - 5)/2 + 1 = 8
    let w = DynTensor::from_vec(vec![0.01; 40], &[4, 2, 5], &Device::Cpu).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::new(0, 2, 1)).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 40], &[1, 2, 20], &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 8]);
}

#[test]
fn test_conv1d_groups_depthwise() {
    // Depthwise: groups = in_channels = out_channels = 4
    // weight: [4, 1, 3], groups=4
    let w = DynTensor::from_vec(vec![1.0; 12], &[4, 1, 3], &Device::Cpu).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::new(1, 1, 1).with_groups(4)).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 40], &[1, 4, 10], &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 10]);
}

#[test]
fn test_conv2d_output_size_formula() {
    // H=32, W=32, kernel=3, padding=1, stride=1 -> same padding -> 32x32
    let out_h = conv2d_out_len(32, 3, 1, 1, 1).unwrap();
    let out_w = conv2d_out_len(32, 3, 1, 1, 1).unwrap();
    assert_eq!(out_h, 32);
    assert_eq!(out_w, 32);
}

#[test]
fn test_conv2d_stride2_halves_spatial() {
    // H=32, kernel=3, padding=1, stride=2 -> out = (32+2-3)/2 + 1 = 16
    let out = conv2d_out_len(32, 3, 1, 2, 1).unwrap();
    assert_eq!(out, 16);
}

#[test]
fn test_conv2d_forward_output_shape() {
    // weight: [16, 3, 3, 3], input: [1, 3, 8, 8], padding=1, stride=1
    // -> [1, 16, 8, 8]
    let w = DynTensor::from_vec(vec![0.01; 16 * 3 * 3 * 3], &[16, 3, 3, 3], &Device::Cpu).unwrap();
    let conv = Conv2d::new(w, None, Conv2dConfig::new(1, 1, 1)).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 3 * 8 * 8], &[1, 3, 8, 8], &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 16, 8, 8]);
}

#[test]
fn test_conv2d_groups_depthwise() {
    // Depthwise conv2d: groups = channels = 8
    // weight: [8, 1, 3, 3], groups=8
    let w = DynTensor::from_vec(vec![0.1; 8 * 3 * 3], &[8, 1, 3, 3], &Device::Cpu).unwrap();
    let conv = Conv2d::new(w, None, Conv2dConfig::new(1, 1, 1).with_groups(8)).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 8 * 6 * 6], &[1, 8, 6, 6], &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 6, 6]);
}

#[test]
fn test_conv1d_no_bias_forward() {
    let w = DynTensor::from_vec(vec![1.0; 6], &[2, 1, 3], &Device::Cpu).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    assert!(conv.bias().is_none());
    let x = DynTensor::from_vec(vec![1.0; 5], &[1, 1, 5], &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);
}

// =============================================================================
// 3. LayerNorm
// =============================================================================

#[test]
fn test_layer_norm_normalized_mean_near_zero() {
    let d = 16;
    let weight = DynTensor::ones(&[d], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[d], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let data: Vec<f32> = (0..d).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let x = DynTensor::from_vec(data, &[1, d], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // Mean should be approximately 0
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(mean.abs() < 1e-4, "LayerNorm mean should be ~0, got {mean}");
}

#[test]
fn test_layer_norm_normalized_variance_near_one() {
    let d = 32;
    let weight = DynTensor::ones(&[d], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[d], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let data: Vec<f32> = (0..d).map(|i| (i as f32) * 0.3 - 2.0).collect();
    let x = DynTensor::from_vec(data, &[1, d], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    assert!(
        (var - 1.0).abs() < 0.05,
        "LayerNorm variance should be ~1, got {var}"
    );
}

#[test]
fn test_layer_norm_affine_transform() {
    let d = 4;
    // weight = 2.0, bias = 1.0 -> output = 2 * normalized + 1
    let weight = DynTensor::full(&[d], 2.0, DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::full(&[d], 1.0, DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, d], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // The mean of the output should be bias value (1.0) since mean of normalized is 0
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(
        (mean - 1.0).abs() < 0.05,
        "Affine LayerNorm mean should be ~1.0, got {mean}"
    );
}

#[test]
fn test_layer_norm_preserves_shape_batched() {
    let d = 8;
    let weight = DynTensor::ones(&[d], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[d], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::full(&[4, 6, d], 1.0, DType::F32, &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, 6, d]);
}

// =============================================================================
// 4. BatchNorm
// =============================================================================

#[test]
fn test_batch_norm_inference_preserves_shape() {
    let c = 4;
    let mean = DynTensor::zeros(&[c], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[c], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::ones(&[c], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[c], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(mean, var, Some(weight), Some(bias), 1e-5).unwrap();

    let x = DynTensor::full(&[2, c, 8], 1.0, DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, c, 8]);
}

#[test]
fn test_batch_norm_with_identity_stats() {
    // running_mean = 0, running_var = 1, weight = 1, bias = 0 -> identity-like
    let c = 2;
    let mean = DynTensor::zeros(&[c], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[c], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::ones(&[c], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[c], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(mean, var, Some(weight), Some(bias), 1e-5).unwrap();

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::from_vec(data.clone(), &[1, c, 4], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // With identity stats, output should be close to input
    for (a, b) in vals.iter().zip(data.iter()) {
        assert!(
            (a - b).abs() < 0.01,
            "BatchNorm with identity stats should be ~identity, got {a} vs {b}"
        );
    }
}

#[test]
fn test_batch_norm_no_affine() {
    let c = 3;
    let mean = DynTensor::zeros(&[c], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[c], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::full(&[2, c, 4], 2.0, DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, c, 4]);
}

// =============================================================================
// 5. Dropout
// =============================================================================

#[test]
fn test_dropout_p0_is_identity() {
    let d = Dropout::new(0.0);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dropout_p1_still_identity_at_inference() {
    // nn targets inference only -- dropout is always identity
    let d = Dropout::new(1.0);
    let x = DynTensor::from_vec(vec![5.0, 6.0], &[2], &Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 6.0]);
}

#[test]
fn test_dropout_preserves_multidim_shape() {
    let d = Dropout::new(0.5);
    let x = DynTensor::full(&[2, 3, 4, 5], 1.0, DType::F32, &Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 5]);
}

// =============================================================================
// 6. Embedding
// =============================================================================

#[test]
fn test_embedding_output_shape_1d_input() {
    // vocab=10, dim=4, input=[3] -> output=[3, 4]
    let w = DynTensor::from_vec(vec![0.1; 40], &[10, 4], &Device::Cpu).unwrap();
    let emb = Embedding::new(w).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 5, 9], &[3], &Device::Cpu).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[3, 4]);
}

#[test]
fn test_embedding_output_shape_2d_input() {
    // vocab=10, dim=4, input=[2, 3] -> output=[2, 3, 4]
    let w = DynTensor::from_vec(vec![0.1; 40], &[10, 4], &Device::Cpu).unwrap();
    let emb = Embedding::new(w).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &Device::Cpu).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_embedding_index_out_of_bounds() {
    let w = DynTensor::from_vec(vec![0.1; 20], &[5, 4], &Device::Cpu).unwrap();
    let emb = Embedding::new(w).unwrap();
    // Index 5 is out of range for vocab_size=5
    let result = emb.forward_ids(&[5]);
    assert!(result.is_err(), "index >= vocab_size should fail");
}

#[test]
fn test_embedding_forward_ids_shape() {
    let w = DynTensor::from_vec(vec![0.1; 30], &[6, 5], &Device::Cpu).unwrap();
    let emb = Embedding::new(w).unwrap();
    let y = emb.forward_ids(&[0, 3, 5]).unwrap();
    assert_eq!(y.dims(), &[3, 5]);
}

#[test]
fn test_embedding_weight_accessor() {
    let w = DynTensor::from_vec(vec![1.0; 12], &[3, 4], &Device::Cpu).unwrap();
    let emb = Embedding::new(w).unwrap();
    assert_eq!(emb.weight().dims(), &[3, 4]);
    // embeddings() is an alias for weight()
    assert_eq!(emb.embeddings().dims(), &[3, 4]);
}

#[test]
fn test_embedding_rejects_non_2d_weight() {
    let w = DynTensor::from_vec(vec![1.0; 8], &[2, 2, 2], &Device::Cpu).unwrap();
    assert!(Embedding::new(w).is_err());
}

// =============================================================================
// 7. LSTM / BiLSTM
// =============================================================================

#[test]
fn test_lstm_single_step_output_shape() {
    let input_size = 4;
    let hidden_size = 3;
    let batch = 2;
    let w_ih = DynTensor::from_vec(
        vec![0.01; 4 * hidden_size * input_size],
        &[4 * hidden_size, input_size],
        &Device::Cpu,
    )
    .unwrap();
    let w_hh = DynTensor::from_vec(
        vec![0.01; 4 * hidden_size * hidden_size],
        &[4 * hidden_size, hidden_size],
        &Device::Cpu,
    )
    .unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap();

    let x = DynTensor::from_vec(
        vec![1.0; batch * input_size],
        &[batch, input_size],
        &Device::Cpu,
    )
    .unwrap();
    let (output, state) = lstm.forward(&x, None).unwrap();
    assert_eq!(output.dims(), &[batch, hidden_size]);
    assert_eq!(state.h.dims(), &[batch, hidden_size]);
    assert_eq!(state.c.dims(), &[batch, hidden_size]);
}

#[test]
fn test_lstm_seq_output_shape() {
    let input_size = 4;
    let hidden_size = 3;
    let batch = 2;
    let seq_len = 5;

    let w_ih = DynTensor::from_vec(
        vec![0.01; 4 * hidden_size * input_size],
        &[4 * hidden_size, input_size],
        &Device::Cpu,
    )
    .unwrap();
    let w_hh = DynTensor::from_vec(
        vec![0.01; 4 * hidden_size * hidden_size],
        &[4 * hidden_size, hidden_size],
        &Device::Cpu,
    )
    .unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap();

    // time-first: [seq_len, batch, input_size]
    let x = DynTensor::from_vec(
        vec![1.0; seq_len * batch * input_size],
        &[seq_len, batch, input_size],
        &Device::Cpu,
    )
    .unwrap();
    let (outputs, final_state) = lstm.forward_seq(&x, None).unwrap();
    assert_eq!(outputs.dims(), &[seq_len, batch, hidden_size]);
    assert_eq!(final_state.h.dims(), &[batch, hidden_size]);
    assert_eq!(final_state.c.dims(), &[batch, hidden_size]);
}

#[test]
fn test_lstm_state_new_validates_shapes() {
    let h = DynTensor::zeros(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    let c = DynTensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(
        LstmState::new(h, c).is_err(),
        "h and c must have matching shapes"
    );
}

#[test]
fn test_lstm_rejects_zero_hidden_size() {
    let w_ih = DynTensor::from_vec(vec![1.0; 4], &[4, 1], &Device::Cpu).unwrap();
    let w_hh = DynTensor::from_vec(vec![1.0; 4], &[4, 1], &Device::Cpu).unwrap();
    assert!(Lstm::new(w_ih, w_hh, None, None, 0).is_err());
}

#[test]
fn test_bilstm_output_shape() {
    let input_size = 4;
    let hidden_size = 3;
    let batch = 2;
    let seq_len = 5;

    let make_lstm = || {
        let w_ih = DynTensor::from_vec(
            vec![0.01; 4 * hidden_size * input_size],
            &[4 * hidden_size, input_size],
            &Device::Cpu,
        )
        .unwrap();
        let w_hh = DynTensor::from_vec(
            vec![0.01; 4 * hidden_size * hidden_size],
            &[4 * hidden_size, hidden_size],
            &Device::Cpu,
        )
        .unwrap();
        Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap()
    };

    let bilstm = BiLstm::new(make_lstm(), make_lstm()).unwrap();
    assert_eq!(bilstm.hidden_size(), hidden_size);

    let x = DynTensor::from_vec(
        vec![1.0; seq_len * batch * input_size],
        &[seq_len, batch, input_size],
        &Device::Cpu,
    )
    .unwrap();
    let (outputs, fwd_state, bwd_state) = bilstm.forward_seq(&x, None, None).unwrap();
    // BiLSTM concatenates forward + backward: 2 * hidden_size
    assert_eq!(outputs.dims(), &[seq_len, batch, 2 * hidden_size]);
    assert_eq!(fwd_state.h.dims(), &[batch, hidden_size]);
    assert_eq!(bwd_state.h.dims(), &[batch, hidden_size]);
}

// =============================================================================
// 8. Activation functions
// =============================================================================

#[test]
fn test_relu_output_range() {
    let x = DynTensor::from_vec(vec![-3.0, -1.0, 0.0, 0.5, 2.0], &[5], &Device::Cpu).unwrap();
    let y = Activation::Relu.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v >= 0.0), "ReLU output must be >= 0");
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[4], 2.0);
}

#[test]
fn test_gelu_output_range() {
    let x = DynTensor::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], &[5], &Device::Cpu).unwrap();
    let y = Activation::Gelu.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // GELU(0) = 0
    assert!(vals[2].abs() < 1e-5, "GELU(0) should be 0");
    // GELU(x) ~ x for large positive x
    assert!((vals[4] - 5.0).abs() < 0.01, "GELU(5) should be ~5");
    // GELU(x) ~ 0 for large negative x
    assert!(vals[0].abs() < 0.01, "GELU(-5) should be ~0");
}

#[test]
fn test_silu_output_range() {
    let x = DynTensor::from_vec(vec![-5.0, 0.0, 5.0], &[3], &Device::Cpu).unwrap();
    let y = Activation::Silu.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // SiLU(0) = 0
    assert!(vals[1].abs() < 1e-5, "SiLU(0) should be 0");
    // SiLU(x) ~ x for large positive x
    assert!((vals[2] - 5.0).abs() < 0.05, "SiLU(5) should be ~5");
}

#[test]
fn test_sigmoid_output_range() {
    let x = DynTensor::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], &Device::Cpu).unwrap();
    let y = Activation::Sigmoid.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "Sigmoid output must be in [0, 1]"
    );
    assert!((vals[2] - 0.5).abs() < 1e-5, "Sigmoid(0) should be 0.5");
    assert!(vals[0] < 0.001, "Sigmoid(-10) should be ~0");
    assert!(vals[4] > 0.999, "Sigmoid(10) should be ~1");
}

#[test]
fn test_tanh_output_range() {
    let x = DynTensor::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], &Device::Cpu).unwrap();
    let y = Activation::Tanh.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| (-1.0..=1.0).contains(&v)),
        "Tanh output must be in [-1, 1]"
    );
    assert!(vals[2].abs() < 1e-5, "Tanh(0) should be 0");
    assert!(vals[0] < -0.999, "Tanh(-10) should be ~-1");
    assert!(vals[4] > 0.999, "Tanh(10) should be ~1");
}

#[test]
fn test_elu_output_range() {
    let alpha = 1.0;
    let x = DynTensor::from_vec(vec![-2.0, 0.0, 2.0], &[3], &Device::Cpu).unwrap();
    let y = Activation::Elu(alpha).forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // ELU(x) = x for x > 0, alpha*(exp(x)-1) for x <= 0
    assert!(
        vals[0] < 0.0 && vals[0] > -(alpha as f32),
        "ELU(-2) in (-alpha, 0)"
    );
    assert!(vals[1].abs() < 1e-5, "ELU(0) should be 0");
    assert_eq!(vals[2], 2.0);
}

#[test]
fn test_leaky_relu_output_range() {
    let slope = 0.01;
    let x = DynTensor::from_vec(vec![-10.0, 0.0, 10.0], &[3], &Device::Cpu).unwrap();
    let y = Activation::LeakyRelu(slope).forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-0.1)).abs() < 1e-4, "LeakyReLU(-10) = -0.1");
    assert!(vals[1].abs() < 1e-5, "LeakyReLU(0) = 0");
    assert_eq!(vals[2], 10.0);
}

// =============================================================================
// 9. Softmax / LogSoftmax
// =============================================================================

#[test]
fn test_softmax_sums_to_one() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap();
    let y = x.softmax(0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let sum: f32 = vals.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax should sum to 1, got {sum}"
    );
}

#[test]
fn test_softmax_monotonic() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = x.softmax(0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < vals[1], "Softmax should preserve order");
    assert!(vals[1] < vals[2], "Softmax should preserve order");
}

#[test]
fn test_softmax_2d_rows_sum_to_one() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5], &[2, 3], &Device::Cpu).unwrap();
    let y = x.softmax(1).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let row0_sum: f32 = vals[..3].iter().sum();
    let row1_sum: f32 = vals[3..].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5);
    assert!((row1_sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_numerical_stability_large_values() {
    // Large input values should not cause overflow
    let x = DynTensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[3], &Device::Cpu).unwrap();
    let y = x.softmax(0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let sum: f32 = vals.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Softmax with large values should still sum to 1"
    );
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "No NaN/Inf in softmax output"
    );
}

#[test]
fn test_log_softmax_exp_sums_to_one() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = x.log_softmax(0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // All values should be negative
    assert!(
        vals.iter().all(|&v| v < 0.0),
        "LogSoftmax values should be negative"
    );
    // exp(log_softmax) should sum to 1
    let exp_sum: f32 = vals.iter().map(|v| v.exp()).sum();
    assert!(
        (exp_sum - 1.0).abs() < 1e-5,
        "exp(LogSoftmax) should sum to 1, got {exp_sum}"
    );
}

#[test]
fn test_log_softmax_numerical_stability_large_values() {
    let x = DynTensor::from_vec(vec![500.0, 501.0, 502.0], &[3], &Device::Cpu).unwrap();
    let y = x.log_softmax(0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "No NaN/Inf in log_softmax output"
    );
}

// =============================================================================
// 10. MultiHeadAttention
// =============================================================================

#[test]
fn test_mha_output_shape_self_attention() {
    let dim = 16;
    let num_heads = 4;
    let batch = 2;
    let seq_len = 5;

    // Create projection weights
    let make_proj = |out_dim: usize| {
        let w =
            DynTensor::from_vec(vec![0.01; out_dim * dim], &[out_dim, dim], &Device::Cpu).unwrap();
        Linear::new(w, None).unwrap()
    };

    let q_proj = make_proj(dim);
    let k_proj = make_proj(dim);
    let v_proj = make_proj(dim);
    let out_proj = make_proj(dim);

    let mha =
        MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, num_heads).unwrap();

    let x = DynTensor::from_vec(
        vec![0.1; batch * seq_len * dim],
        &[batch, seq_len, dim],
        &Device::Cpu,
    )
    .unwrap();

    let y = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(y.dims(), &[batch, seq_len, dim]);
}

#[test]
fn test_mha_gqa_output_shape() {
    // Grouped-query attention: num_kv_heads < num_heads
    let dim = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let kv_dim = num_kv_heads * (dim / num_heads);
    let batch = 1;
    let seq_len = 3;

    let w_q = DynTensor::from_vec(vec![0.01; dim * dim], &[dim, dim], &Device::Cpu).unwrap();
    let w_k = DynTensor::from_vec(vec![0.01; kv_dim * dim], &[kv_dim, dim], &Device::Cpu).unwrap();
    let w_v = DynTensor::from_vec(vec![0.01; kv_dim * dim], &[kv_dim, dim], &Device::Cpu).unwrap();
    let w_o = DynTensor::from_vec(vec![0.01; dim * dim], &[dim, dim], &Device::Cpu).unwrap();

    let q_proj = Linear::new(w_q, None).unwrap();
    let k_proj = Linear::new(w_k, None).unwrap();
    let v_proj = Linear::new(w_v, None).unwrap();
    let out_proj = Linear::new(w_o, None).unwrap();

    let mha =
        MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, num_kv_heads).unwrap();

    let x = DynTensor::from_vec(
        vec![0.1; batch * seq_len * dim],
        &[batch, seq_len, dim],
        &Device::Cpu,
    )
    .unwrap();

    let y = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(y.dims(), &[batch, seq_len, dim]);
}

#[test]
fn test_mha_rejects_zero_heads() {
    let dim = 8;
    let w = DynTensor::from_vec(vec![0.01; dim * dim], &[dim, dim], &Device::Cpu).unwrap();
    let proj = Linear::new(w, None).unwrap();
    let result = MultiHeadAttention::new(
        proj.clone(),
        proj.clone(),
        proj.clone(),
        proj,
        0, // zero heads
        1,
    );
    assert!(result.is_err(), "num_heads=0 should be rejected");
}

#[test]
fn test_mha_rejects_heads_not_divisible() {
    let dim = 8;
    let w = DynTensor::from_vec(vec![0.01; dim * dim], &[dim, dim], &Device::Cpu).unwrap();
    let proj = Linear::new(w, None).unwrap();
    let result = MultiHeadAttention::new(
        proj.clone(),
        proj.clone(),
        proj.clone(),
        proj,
        4, // num_heads
        3, // num_kv_heads -- 4 is not divisible by 3
    );
    assert!(
        result.is_err(),
        "num_heads not divisible by num_kv_heads should be rejected"
    );
}

#[test]
fn test_mha_with_mask() {
    let dim = 8;
    let num_heads = 2;
    let batch = 1;
    let seq_len = 4;

    let make_proj = |out_dim: usize| {
        let w =
            DynTensor::from_vec(vec![0.01; out_dim * dim], &[out_dim, dim], &Device::Cpu).unwrap();
        Linear::new(w, None).unwrap()
    };

    let mha = MultiHeadAttention::new(
        make_proj(dim),
        make_proj(dim),
        make_proj(dim),
        make_proj(dim),
        num_heads,
        num_heads,
    )
    .unwrap();

    let x = DynTensor::from_vec(
        vec![0.1; batch * seq_len * dim],
        &[batch, seq_len, dim],
        &Device::Cpu,
    )
    .unwrap();

    // Create a causal mask: [1, 1, seq_len, seq_len]
    let mask = crate::layers::causal_mask(seq_len, &Device::Cpu).unwrap();
    let y = mha.forward(&x, None, Some(&mask), None, 0).unwrap();
    assert_eq!(y.dims(), &[batch, seq_len, dim]);
}
