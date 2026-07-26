// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KokoroModel integration tests: config, weight loading, and forward passes.
//! Extracted from `kokoro_tts_tests.rs` to keep files under 500 lines.
//! Weight insertion helpers live in `kokoro_tts_tests_model_weights.rs`.

#[path = "kokoro_tts_tests_model_weights.rs"]
mod weights;
pub(crate) use weights::make_kokoro_model_weights;

use crate::plbert::PlbertConfig;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

const T_D_EN: usize = 8;
pub(super) const T_STYLE: usize = 4;
pub(super) const T_HIDDEN: usize = 8;
const T_EMB: usize = 4;
const T_VOCAB: usize = 10;
const T_N_FFT: usize = 4;
const T_GEN_CH: usize = 8;

fn test_plbert_config() -> PlbertConfig {
    PlbertConfig {
        vocab_size: T_VOCAB,
        embedding_dim: T_EMB,
        hidden_size: T_HIDDEN,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 16,
        num_hidden_layers: 1,
        layer_norm_eps: 1e-12,
    }
}

const T_F0_HIDDEN: usize = 4;

pub(super) fn test_kokoro_config() -> super::super::KokoroConfig {
    super::super::KokoroConfig {
        d_en: T_D_EN,
        n_prosody_layers: 1,
        style_dim: T_STYLE,
        upsample_rates: vec![2],
        upsample_kernel_sizes: vec![4],
        resblock_kernel_sizes: vec![3],
        resblock_dilations: vec![vec![1, 2]],
        gen_initial_channels: T_GEN_CH,
        n_fft: T_N_FFT,
        f0_bilstm_hidden: T_F0_HIDDEN,
        max_dur: 50,
        plbert: test_plbert_config(),
    }
}

#[test]
fn test_kokoro_model_load() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config);
    assert!(model.is_ok(), "KokoroModel::load failed: {:?}", model.err());
}

#[test]
fn test_kokoro_model_forward_text() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward_text(&input_ids, &bert_out, &style, 1.0);
    assert!(result.is_ok(), "forward_text failed: {:?}", result.err());
    let r = result.unwrap();
    assert_eq!(r.dur_logits.dims(), &[1, 3, 50]);
    assert_eq!(r.aligned_dur.dims()[0], 1);
    // regulated: TextEncoder features → FullDecoder (asr input, d_en channels)
    assert_eq!(r.regulated.dims()[0], 1);
    assert_eq!(r.regulated.dims()[1], T_D_EN);
}

#[test]
fn test_kokoro_model_forward_full() {
    let weights = make_kokoro_model_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = super::super::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward(&input_ids, &style, 1.0);
    assert!(result.is_ok(), "full forward failed: {:?}", result.err());
    let (mag, phase) = result.unwrap();
    let n_bins = T_N_FFT / 2 + 1;
    assert_eq!(mag.dims()[0], 1);
    assert_eq!(mag.dims()[1], n_bins);
    assert_eq!(phase.dims()[0], 1);
    assert_eq!(phase.dims()[1], n_bins);
    assert_eq!(mag.dims()[2], phase.dims()[2]);
}

#[test]
fn test_kokoro_config_default() {
    let config = super::super::KokoroConfig::default();
    assert_eq!(config.d_en, 512);
    assert_eq!(config.n_prosody_layers, 3);
    assert_eq!(config.style_dim, 128);
    assert_eq!(config.upsample_rates, vec![10, 6]);
    assert_eq!(config.n_fft, 20);
    assert_eq!(config.f0_bilstm_hidden, 256);
    assert_eq!(config.plbert.vocab_size, 178);
}
