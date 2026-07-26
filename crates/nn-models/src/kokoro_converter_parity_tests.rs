// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro auto-converter parity test scaffolding (Part of #4276).
//!
//! Focuses on dtype handling (F32, BF16, F16 conversion paths), config
//! validation edge cases, model architecture consistency checks, and
//! VarBuilder weight loading under different dtype configurations.

#[path = "kokoro_convert_parity_weights.rs"]
mod weights;

use crate::convert::ConvertedModel;
use crate::kokoro_tts::KokoroConfig;
use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::test_utils::cpu;
use nn_core::DType;

// ---------------------------------------------------------------------------
// Test-scale config constants (must match kokoro_convert_parity_tests.rs)
// ---------------------------------------------------------------------------

const T_D_EN: usize = 8;
const T_STYLE: usize = 4;
const T_HIDDEN: usize = 8;
const T_EMB: usize = 4;
const T_VOCAB: usize = 10;
const T_N_FFT: usize = 4;
const T_GEN_CH: usize = 8;
const T_F0_HIDDEN: usize = 4;

fn test_plbert_config() -> crate::plbert::PlbertConfig {
    crate::plbert::PlbertConfig {
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

fn test_kokoro_config() -> KokoroConfig {
    KokoroConfig {
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

fn load_test_model() -> crate::kokoro_tts::KokoroModel {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    crate::kokoro_tts::KokoroModel::load(&vb, &config).unwrap()
}

// ===========================================================================
// Dtype handling: VarBuilder with F32
// ===========================================================================

/// VarBuilder created with F32 dtype loads model successfully.
#[test]
fn test_varbuilder_f32_loads_model() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    assert_eq!(vb.dtype(), DType::F32);
    let result = crate::kokoro_tts::KokoroModel::load(&vb, &config);
    assert!(
        result.is_ok(),
        "F32 VarBuilder load failed: {:?}",
        result.err()
    );
}

/// VarBuilder created with BF16 dtype reports BF16.
#[test]
fn test_varbuilder_bf16_dtype_propagation() {
    let weights = weights::make_kokoro_weights();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let vb_bf16 = vb.to_dtype(DType::BF16);
    assert_eq!(vb_bf16.dtype(), DType::BF16);
}

/// VarBuilder created with F16 dtype reports F16.
#[test]
fn test_varbuilder_f16_dtype_propagation() {
    let weights = weights::make_kokoro_weights();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let vb_f16 = vb.to_dtype(DType::F16);
    assert_eq!(vb_f16.dtype(), DType::F16);
}

// ===========================================================================
// Dtype handling: DynTensor conversion paths
// ===========================================================================

/// F32 tensor converts to BF16 and back without shape change.
#[test]
fn test_dyntensor_f32_to_bf16_roundtrip_shape() {
    let t = DynTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F32);

    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(bf16.dims(), &[2, 2]);

    let back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    assert_eq!(back.dims(), &[2, 2]);
}

/// F32 tensor converts to F16 and back without shape change.
#[test]
fn test_dyntensor_f32_to_f16_roundtrip_shape() {
    let t = DynTensor::from_vec(vec![0.5_f32, -0.5, 1.0, -1.0], &[4], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[4]);

    let back = f16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    assert_eq!(back.dims(), &[4]);
}

/// BF16 conversion preserves values within expected tolerance.
#[test]
fn test_bf16_conversion_numerical_tolerance() {
    let vals = vec![0.0_f32, 1.0, -1.0, 0.125, 100.0];
    let t = DynTensor::from_vec(vals.clone(), &[5], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    let back = bf16.to_dtype(DType::F32).unwrap();
    let result = back.to_vec1::<f32>().unwrap();

    for (i, (&orig, &converted)) in vals.iter().zip(result.iter()).enumerate() {
        // BF16 has ~7 bits mantissa, so relative tolerance is ~1%
        let tol = if orig.abs() > 1e-6 {
            orig.abs() * 0.01
        } else {
            1e-4
        };
        assert!(
            (orig - converted).abs() <= tol,
            "BF16 roundtrip mismatch at {i}: orig={orig}, got={converted}, tol={tol}"
        );
    }
}

/// F16 conversion preserves values within expected tolerance.
#[test]
fn test_f16_conversion_numerical_tolerance() {
    let vals = vec![0.0_f32, 1.0, -1.0, 0.5, 10.0];
    let t = DynTensor::from_vec(vals.clone(), &[5], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let result = back.to_vec1::<f32>().unwrap();

    for (i, (&orig, &converted)) in vals.iter().zip(result.iter()).enumerate() {
        // F16 has ~10 bits mantissa, so relative tolerance is ~0.1%
        let tol = if orig.abs() > 1e-6 {
            orig.abs() * 0.002
        } else {
            1e-4
        };
        assert!(
            (orig - converted).abs() <= tol,
            "F16 roundtrip mismatch at {i}: orig={orig}, got={converted}, tol={tol}"
        );
    }
}

/// F32 to F32 conversion is identity (no-op path).
#[test]
fn test_f32_to_f32_is_identity() {
    let vals = vec![3.14_f32, 2.718, -1.414];
    let t = DynTensor::from_vec(vals.clone(), &[3], &cpu()).unwrap();
    let same = t.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    let result = same.to_vec1::<f32>().unwrap();
    assert_eq!(result, vals);
}

/// Weight tensors can be converted to BF16 and maintain expected shapes.
#[test]
fn test_weight_tensors_bf16_shape_preserved() {
    let weights = weights::make_kokoro_weights();
    for (name, tensor) in &weights {
        let original_dims = tensor.dims().to_vec();
        let bf16 = tensor.to_dtype(DType::BF16).unwrap();
        assert_eq!(
            bf16.dims(),
            original_dims.as_slice(),
            "BF16 shape mismatch for weight '{name}'"
        );
        assert_eq!(bf16.dtype(), DType::BF16);
    }
}

/// Weight tensors can be converted to F16 and maintain expected shapes.
#[test]
fn test_weight_tensors_f16_shape_preserved() {
    let weights = weights::make_kokoro_weights();
    for (name, tensor) in &weights {
        let original_dims = tensor.dims().to_vec();
        let f16 = tensor.to_dtype(DType::F16).unwrap();
        assert_eq!(
            f16.dims(),
            original_dims.as_slice(),
            "F16 shape mismatch for weight '{name}'"
        );
        assert_eq!(f16.dtype(), DType::F16);
    }
}

// ===========================================================================
// Config parsing validation: edge cases and field constraints
// ===========================================================================

/// KokoroConfig::validate rejects d_en = 0.
#[test]
fn test_config_rejects_zero_d_en() {
    let mut config = test_kokoro_config();
    config.d_en = 0;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("d_en"), "error should mention d_en: {msg}");
}

/// KokoroConfig::validate rejects style_dim = 0.
#[test]
fn test_config_rejects_zero_style_dim() {
    let mut config = test_kokoro_config();
    config.style_dim = 0;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("style_dim"),
        "error should mention style_dim: {msg}"
    );
}

/// KokoroConfig::validate rejects max_dur = 0.
#[test]
fn test_config_rejects_zero_max_dur() {
    let mut config = test_kokoro_config();
    config.max_dur = 0;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("max_dur"),
        "error should mention max_dur: {msg}"
    );
}

/// KokoroConfig::validate rejects n_fft = 0.
#[test]
fn test_config_rejects_zero_n_fft() {
    let mut config = test_kokoro_config();
    config.n_fft = 0;
    let result = config.validate();
    assert!(result.is_err());
}

/// KokoroConfig::validate rejects n_fft not divisible by 4.
#[test]
fn test_config_rejects_n_fft_not_div4() {
    let mut config = test_kokoro_config();
    config.n_fft = 5;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("divisible by 4"),
        "error should mention divisibility: {msg}"
    );
}

/// KokoroConfig::validate rejects empty upsample_rates.
#[test]
fn test_config_rejects_empty_upsample_rates() {
    let mut config = test_kokoro_config();
    config.upsample_rates = vec![];
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("upsample_rates"),
        "error should mention upsample_rates: {msg}"
    );
}

/// KokoroConfig::new() produces same defaults as Default.
#[test]
fn test_config_new_matches_default() {
    let from_new = KokoroConfig::new();
    let from_default = KokoroConfig::default();
    assert_eq!(from_new.d_en, from_default.d_en);
    assert_eq!(from_new.style_dim, from_default.style_dim);
    assert_eq!(from_new.max_dur, from_default.max_dur);
    assert_eq!(from_new.n_fft, from_default.n_fft);
    assert_eq!(
        from_new.gen_initial_channels,
        from_default.gen_initial_channels
    );
    assert_eq!(from_new.upsample_rates, from_default.upsample_rates);
    assert_eq!(
        from_new.resblock_kernel_sizes,
        from_default.resblock_kernel_sizes
    );
}

/// PlbertConfig defaults match expected production values.
#[test]
fn test_plbert_config_production_defaults() {
    let config = crate::plbert::PlbertConfig::default();
    assert_eq!(config.vocab_size, 178);
    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.intermediate_size, 2048);
    assert_eq!(config.max_position_embeddings, 512);
    assert_eq!(config.num_hidden_layers, 12);
    assert!((config.layer_norm_eps - 1e-12).abs() < 1e-20);
}

// ===========================================================================
// Model architecture consistency: dimensions derived from config
// ===========================================================================

/// Bert encoder weight shape matches [d_en, hidden_size] from config.
#[test]
fn test_bert_encoder_weight_shape_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights.get("bert_encoder.weight").unwrap();
    assert_eq!(w.dims(), &[config.d_en, config.plbert.hidden_size]);
    let b = weights.get("bert_encoder.bias").unwrap();
    assert_eq!(b.dims(), &[config.d_en]);
}

/// PlBert embedding shape matches [vocab_size, embedding_dim].
#[test]
fn test_plbert_embedding_shape_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights
        .get("plbert.embeddings.word_embeddings.weight")
        .unwrap();
    assert_eq!(
        w.dims(),
        &[config.plbert.vocab_size, config.plbert.embedding_dim]
    );
}

/// PlBert hidden mapping shape matches [hidden_size, embedding_dim].
#[test]
fn test_plbert_hidden_mapping_shape_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights
        .get("plbert.encoder.embedding_hidden_mapping_in.weight")
        .unwrap();
    assert_eq!(
        w.dims(),
        &[config.plbert.hidden_size, config.plbert.embedding_dim]
    );
}

/// Text encoder embedding shape matches [vocab_size, d_en].
#[test]
fn test_text_encoder_embedding_shape_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights.get("text_encoder.embedding.weight").unwrap();
    assert_eq!(w.dims(), &[config.plbert.vocab_size, config.d_en]);
}

/// Text encoder conv layers have [d_en, d_en, kernel] shape.
#[test]
fn test_text_encoder_conv_shapes_match_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    for i in 0..3 {
        let key = format!("text_encoder.convs.{i}.weight");
        let w = weights.get(&key).unwrap();
        assert_eq!(
            w.dims()[0],
            config.d_en,
            "conv.{i} out_channels should be d_en"
        );
        assert_eq!(
            w.dims()[1],
            config.d_en,
            "conv.{i} in_channels should be d_en"
        );
    }
}

/// Duration projection weight shape matches [max_dur, d_en].
#[test]
fn test_duration_proj_shape_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights
        .get("prosody_predictor.duration.duration_proj.weight")
        .unwrap();
    assert_eq!(w.dims(), &[config.max_dur, config.d_en]);
}

/// Generator conv_pre input channels match gen_initial_channels.
#[test]
fn test_generator_conv_pre_channels_match_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights.get("decoder.generator.conv_pre.weight").unwrap();
    assert_eq!(
        w.dims()[0],
        config.gen_initial_channels,
        "conv_pre out_channels should be gen_initial_channels"
    );
    assert_eq!(
        w.dims()[1],
        config.gen_initial_channels,
        "conv_pre in_channels should be gen_initial_channels"
    );
}

/// Generator conv_post output channels match 2 * n_bins.
#[test]
fn test_generator_conv_post_output_matches_n_fft() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let n_bins = config.n_fft / 2 + 1;
    let w = weights.get("decoder.generator.conv_post.weight").unwrap();
    assert_eq!(
        w.dims()[0],
        2 * n_bins,
        "conv_post out_channels should be 2*n_bins for mag+phase"
    );
}

/// Upsample weight shape uses upsample_kernel_sizes from config.
#[test]
fn test_upsample_kernel_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let w = weights.get("decoder.generator.ups.0.weight").unwrap();
    assert_eq!(
        w.dims()[2],
        config.upsample_kernel_sizes[0],
        "upsample kernel size should match config"
    );
}

/// F0 predictor shared BiLSTM input dimension matches d_en + style_dim.
#[test]
fn test_f0_predictor_bilstm_input_matches_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let expected_input = config.d_en + config.style_dim;
    let w = weights.get("predictor.shared.weight_ih_l0").unwrap();
    assert_eq!(
        w.dims()[1],
        expected_input,
        "F0 predictor BiLSTM input should be d_en + style_dim"
    );
    assert_eq!(
        w.dims()[0],
        4 * config.f0_bilstm_hidden,
        "F0 predictor BiLSTM gate dim should be 4*f0_bilstm_hidden"
    );
}

// ===========================================================================
// Forward pass shape consistency with config
// ===========================================================================

/// PlBert output rank is 3 and last dim matches hidden_size.
#[test]
fn test_plbert_output_shape_matches_config() {
    let model = load_test_model();
    let config = test_kokoro_config();
    let input = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let out = model.plbert().forward(&input).unwrap();
    assert_eq!(out.rank(), 3);
    assert_eq!(out.dims()[0], 1, "batch");
    assert_eq!(out.dims()[1], 3, "seq_len preserved");
    assert_eq!(
        out.dims()[2],
        config.plbert.hidden_size,
        "hidden_size from config"
    );
}

/// Bert encoder projects from hidden_size to d_en.
#[test]
fn test_bert_encoder_projects_to_d_en() {
    let model = load_test_model();
    let config = test_kokoro_config();
    let input = DynTensor::zeros(&[1, 4, config.plbert.hidden_size], DType::F32, &cpu()).unwrap();
    let out = model.bert_encoder().forward(&input).unwrap();
    assert_eq!(out.dims(), &[1, 4, config.d_en]);
}

/// TextEncoder output channels match d_en from config.
#[test]
fn test_text_encoder_output_channels_match_d_en() {
    let model = load_test_model();
    let config = test_kokoro_config();
    let input = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4], &[1, 4], &cpu()).unwrap();
    let out = model.text_encoder().forward(&input).unwrap();
    assert_eq!(out.dims()[1], config.d_en, "channel dim should be d_en");
}

/// Full model forward produces n_bins = n_fft/2 + 1 frequency bins.
#[test]
fn test_forward_frequency_bins_match_config() {
    let model = load_test_model();
    let config = test_kokoro_config();
    let n_bins = config.n_fft / 2 + 1;

    let input = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * config.style_dim], DType::F32, &cpu()).unwrap();
    let (mag, phase) = model.forward(&input, &style, 1.0).unwrap();

    assert_eq!(mag.dims()[1], n_bins, "magnitude bins should be n_fft/2+1");
    assert_eq!(phase.dims()[1], n_bins, "phase bins should be n_fft/2+1");
}

/// ConvertedModel preserves all weight dtypes as F32 (DynTensor invariant).
#[test]
fn test_converted_model_weights_all_f32() {
    let weights = weights::make_kokoro_weights();
    let converted = ConvertedModel::new(
        ComputationGraph::from_nodes(vec![]),
        weights,
        1,
        vec!["input".to_string()],
        vec!["output".to_string()],
        "kokoro-dtype-test".to_string(),
    );
    for (name, tensor) in &converted.weights {
        assert_eq!(
            tensor.dtype(),
            DType::F32,
            "weight '{name}' should be F32 per DynTensor invariant"
        );
    }
}

/// Style embedding split produces two halves each of style_dim.
#[test]
fn test_style_embedding_split_dim_matches_config() {
    let config = test_kokoro_config();
    let style = DynTensor::zeros(&[1, 2 * config.style_dim], DType::F32, &cpu()).unwrap();
    let (dec, pros) = crate::kokoro_tts::split_style_embedding(&style, config.style_dim).unwrap();
    assert_eq!(dec.dims(), &[1, config.style_dim]);
    assert_eq!(pros.dims(), &[1, config.style_dim]);
}

/// Production config validates successfully.
#[test]
fn test_production_config_validates() {
    let config = KokoroConfig::default();
    assert!(config.validate().is_ok(), "production config must validate");
}

/// Production config n_fft is divisible by 4.
#[test]
fn test_production_config_n_fft_div4() {
    let config = KokoroConfig::default();
    assert_eq!(
        config.n_fft % 4,
        0,
        "production n_fft must be divisible by 4"
    );
}

/// Production config upsample_rates and upsample_kernel_sizes have equal length.
#[test]
fn test_production_config_upsample_arrays_aligned() {
    let config = KokoroConfig::default();
    assert_eq!(
        config.upsample_rates.len(),
        config.upsample_kernel_sizes.len(),
        "upsample_rates and upsample_kernel_sizes must have equal length"
    );
}
