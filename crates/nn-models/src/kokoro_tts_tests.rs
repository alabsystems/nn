#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro TTS model components.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;
use std::collections::HashMap;

fn z(tensors: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    tensors.insert(
        name.to_string(),
        DynTensor::zeros(shape, DType::F32, &cpu()).unwrap(),
    );
}

#[test]
fn test_length_regulate_basic() {
    // features [1, 2, 3], durations [1, 3] = [2, 1, 3] → T_mel = 6
    let features = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();
    let durations = DynTensor::new(&[2.0, 1.0, 3.0], &[1, 3], &cpu()).unwrap();
    let result = super::length_regulate(&features, &durations).unwrap();
    assert_eq!(result.dims(), &[1, 2, 6]);
    // Channel 0: [1, 1, 2, 3, 3, 3], Channel 1: [4, 4, 5, 6, 6, 6]
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0]
    );
}

#[test]
fn test_length_regulate_rank_error() {
    let features = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let durations = DynTensor::new(&[1.0, 1.0], &[1, 2], &cpu()).unwrap();
    assert!(super::length_regulate(&features, &durations).is_err());
}

#[test]
fn test_length_regulate_batch_not_1_rejected() {
    // length_regulate currently supports batch=1 only.
    let features = DynTensor::zeros(&[2, 4, 3], DType::F32, &cpu()).unwrap();
    let durations = DynTensor::new(&[1.0, 1.0, 1.0, 2.0, 1.0, 1.0], &[2, 3], &cpu()).unwrap();
    let err = super::length_regulate(&features, &durations).unwrap_err();
    match err {
        crate::kokoro_error::KokoroError::Tensor(nn_core::TensorError::Unsupported(msg)) => {
            assert!(msg.contains("batch=1"), "should mention batch=1: {msg}");
        }
        other => panic!("expected Tensor(Unsupported), got: {other:?}"),
    }
}

#[test]
fn test_length_regulate_durations_rank_error() {
    // durations must be rank 2 [B, T], not rank 1.
    let features = DynTensor::zeros(&[1, 2, 3], DType::F32, &cpu()).unwrap();
    let durations = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &cpu()).unwrap();
    let err = super::length_regulate(&features, &durations).unwrap_err();
    assert!(
        matches!(
            err,
            crate::kokoro_error::KokoroError::Tensor(nn_core::TensorError::RankMismatch {
                expected: 2,
                actual: 1
            })
        ),
        "expected Tensor(RankMismatch) for rank-1 durations, got: {err:?}"
    );
}

#[test]
fn test_length_regulate_rounding_and_clamp_min() {
    // Fractional durations are rounded (banker's rounding) then clamped to min=1.
    // Duration 0.3 → round → 0 → clamp_min(1) → 1
    // Duration 0.5 → round → 0 (banker's: half rounds to even) → clamp_min(1) → 1
    // Duration 1.5 → round → 2 (banker's: half rounds to even)
    // Duration 2.7 → round → 3
    let features = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4], &cpu()).unwrap();
    let durations = DynTensor::new(&[0.3, 0.5, 1.5, 2.7], &[1, 4], &cpu()).unwrap();
    let result = super::length_regulate(&features, &durations).unwrap();
    // Durations after round+clamp: [1, 1, 2, 3] → T_mel = 7
    assert_eq!(result.dims(), &[1, 1, 7]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Feature 0 (10.0) × 1 frame, Feature 1 (20.0) × 1 frame,
    // Feature 2 (30.0) × 2 frames, Feature 3 (40.0) × 3 frames
    assert_eq!(vals, vec![10.0, 20.0, 30.0, 30.0, 40.0, 40.0, 40.0]);
}

#[test]
fn test_harmonic_source_basic() {
    // Constant F0 at 100Hz, sampling rate 1000Hz → phase increments of 2π*100/1000 = 0.2π
    let f0 = DynTensor::new(&[100.0, 100.0, 100.0, 100.0], &[1, 1, 4], &cpu()).unwrap();
    let source = super::harmonic_source(&f0, 1000.0).unwrap();
    assert_eq!(source.dims(), &[1, 1, 4]);
    let vals = source.to_flat_vec::<f32>().unwrap();
    // sin(0.2π), sin(0.4π), sin(0.6π), sin(0.8π) ≈ [0.588, 0.951, 0.951, 0.588]
    assert!((vals[0] - 0.588).abs() < 0.01);
    assert!((vals[1] - 0.951).abs() < 0.01);
}

const T_D_EN: usize = 8;
const T_TE_VOCAB: usize = 10;

#[test]
fn test_text_encoder_load() {
    let mut tensors = HashMap::new();
    insert_text_encoder_weights(&mut tensors);
    let vb = nn_core::VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let enc = super::TextEncoder::load(vb.pp("text_encoder"), T_TE_VOCAB, T_D_EN);
    assert!(enc.is_ok(), "TextEncoder::load failed: {:?}", enc.err());
}

#[test]
fn test_text_encoder_forward() {
    let mut tensors = HashMap::new();
    insert_text_encoder_weights(&mut tensors);
    let vb = nn_core::VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let enc = super::TextEncoder::load(vb.pp("text_encoder"), T_TE_VOCAB, T_D_EN).unwrap();

    // Input: token IDs [B=1, T=3]
    let tokens = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let result = enc.forward(&tokens);
    assert!(
        result.is_ok(),
        "TextEncoder::forward failed: {:?}",
        result.err()
    );
    let out = result.unwrap();
    assert_eq!(out.dims(), &[1, T_D_EN, 3]);
}

#[test]
fn test_text_encoder_odd_d_en_rejected() {
    let d_en = 7; // Odd: BiLSTM requires even d_en.
    let vb = nn_core::VarBuilder::from_tensors(HashMap::new(), DType::F32, &cpu());
    let result = super::TextEncoder::load(&vb, 10, d_en);
    assert!(result.is_err(), "odd d_en should be rejected");
}

#[test]
fn test_prepare_istft_input_basic() {
    // Decoder output: [1, 4, 2] where n_fft=4, T=2
    // half=2, n_bins=3. Real = channels [0,1], Imag = channels [2,3], both padded to 3.
    let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let decoder_out = DynTensor::new(&data, &[1, 4, 2], &cpu()).unwrap();
    let (real, imag, n_frames) = super::prepare_istft_input(&decoder_out).unwrap();
    assert_eq!(n_frames, 2);
    // real: ch0=[0,1], ch1=[2,3], pad=[0,0] → 6 elements
    assert_eq!(real.len(), 3 * 2);
    assert_eq!(real, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    // imag: ch2=[4,5], ch3=[6,7], pad=[0,0] → 6 elements
    assert_eq!(imag.len(), 3 * 2);
    assert_eq!(imag, vec![4.0, 5.0, 6.0, 7.0, 0.0, 0.0]);
}

#[test]
fn test_prepare_istft_input_rank_error() {
    let x = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(super::prepare_istft_input(&x).is_err());
}

#[test]
fn test_kokoro_constants() {
    assert_eq!(super::KOKORO_N_FFT, 20);
    assert_eq!(super::KOKORO_HOP_LENGTH, 5);
    assert_eq!(super::KOKORO_SAMPLE_RATE, 24000);
    assert_eq!(super::KOKORO_N_BINS, 11);
}

fn insert_text_encoder_weights(m: &mut HashMap<String, DynTensor>) {
    let hidden = T_D_EN / 2;
    let p = "text_encoder";
    // Embedding(vocab_size, d_en)
    z(m, &format!("{p}.embedding.weight"), &[T_TE_VOCAB, T_D_EN]);
    // 3× Conv1d(d_en, d_en, k=5) + LayerNorm(d_en)
    for i in 0..3 {
        z(m, &format!("{p}.convs.{i}.weight"), &[T_D_EN, T_D_EN, 5]);
        z(m, &format!("{p}.convs.{i}.bias"), &[T_D_EN]);
        m.insert(
            format!("{p}.norms.{i}.weight"),
            DynTensor::full(&[T_D_EN], 1.0, DType::F32, &cpu()).unwrap(),
        );
        z(m, &format!("{p}.norms.{i}.bias"), &[T_D_EN]);
    }
    // BiLSTM
    z(m, &format!("{p}.lstm.weight_ih_l0"), &[4 * hidden, T_D_EN]);
    z(m, &format!("{p}.lstm.weight_hh_l0"), &[4 * hidden, hidden]);
    z(m, &format!("{p}.lstm.bias_ih_l0"), &[4 * hidden]);
    z(m, &format!("{p}.lstm.bias_hh_l0"), &[4 * hidden]);
    z(
        m,
        &format!("{p}.lstm.weight_ih_l0_reverse"),
        &[4 * hidden, T_D_EN],
    );
    z(
        m,
        &format!("{p}.lstm.weight_hh_l0_reverse"),
        &[4 * hidden, hidden],
    );
    z(m, &format!("{p}.lstm.bias_ih_l0_reverse"), &[4 * hidden]);
    z(m, &format!("{p}.lstm.bias_hh_l0_reverse"), &[4 * hidden]);
    z(m, &format!("{p}.lstm.linear.weight"), &[T_D_EN, T_D_EN]);
    z(m, &format!("{p}.lstm.linear.bias"), &[T_D_EN]);
}

#[path = "kokoro_tts_tests_model.rs"]
mod model;
use model::{make_kokoro_model_weights, test_kokoro_config, T_HIDDEN, T_STYLE};

#[path = "kokoro_tts_validation_tests.rs"]
mod validation;
