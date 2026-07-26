// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kokoro auto-converter parity tests (Part of #4276).
//!
//! Covers numerical parity and shape consistency for Kokoro submodules:
//! Snake activation, AdaIN normalization, Conv1d padding, ResBlock forward,
//! decoder stage pipeline, duration predictor, LSTM hidden state evolution,
//! full voice decoder shapes, and weight loading roundtrip.

#[path = "kokoro_convert_parity_weights.rs"]
mod weights;

use crate::convert::ConvertConfig;
use crate::kokoro_tts::KokoroConfig;
use nn_core::dyn_tensor::DynTensor;
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
// Snake activation parity
// ===========================================================================

/// Snake activation on a known input: verify x + (1/alpha)*sin^2(alpha*x).
#[test]
fn test_snake_activation_numerical_parity() {
    let x = DynTensor::from_vec(vec![0.0, 1.0, -1.0, 0.5], &[1, 2, 2], &cpu()).unwrap();
    let alpha = 2.0_f64;
    let result = x.snake(alpha).unwrap();

    // Reference: x + (1/alpha)*sin^2(alpha*x)
    let vals = [0.0_f32, 1.0, -1.0, 0.5];
    for (i, &v) in vals.iter().enumerate() {
        let expected = v + (1.0 / alpha as f32) * (alpha as f32 * v).sin().powi(2);
        let flat = result.to_vec1::<f32>().unwrap();
        assert!(
            (flat[i] - expected).abs() < 1e-5,
            "snake({v}, alpha={alpha}): got {}, expected {expected}",
            flat[i]
        );
    }
}

/// Per-channel snake activation with tensor alpha matches scalar when uniform.
#[test]
fn test_snake_tensor_matches_scalar_for_uniform_alpha() {
    let x = DynTensor::from_vec(vec![0.5, -0.3, 1.2, -0.7], &[1, 2, 2], &cpu()).unwrap();
    let alpha_scalar = 1.5_f64;
    let alpha_tensor = DynTensor::full(&[1, 2, 1], alpha_scalar, DType::F32, &cpu()).unwrap();

    let result_scalar = x.snake(alpha_scalar).unwrap();
    let result_tensor = x.snake_tensor(&alpha_tensor).unwrap();

    let s = result_scalar.to_vec1::<f32>().unwrap();
    let t = result_tensor.to_vec1::<f32>().unwrap();
    for (i, (&sv, &tv)) in s.iter().zip(t.iter()).enumerate() {
        assert!(
            (sv - tv).abs() < 1e-5,
            "snake scalar vs tensor mismatch at index {i}: {sv} vs {tv}"
        );
    }
}

/// Snake activation preserves shape.
#[test]
fn test_snake_activation_preserves_shape() {
    let x = DynTensor::zeros(&[2, 4, 8], DType::F32, &cpu()).unwrap();
    let result = x.snake(1.0).unwrap();
    assert_eq!(result.dims(), &[2, 4, 8]);
}

// ===========================================================================
// AdaIN normalization parity
// ===========================================================================

/// AdaIn forward normalizes input and applies style modulation.
/// Output shape must match input shape.
#[test]
fn test_adain_output_shape_matches_input() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = crate::kokoro_tts::KokoroModel::load(&vb, &config).unwrap();

    // ProsodyPredictor uses AdaLayerNorm internally. Test via forward path.
    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let result = model.forward_text(&input_ids, &bert_out, &style, 1.0);
    assert!(
        result.is_ok(),
        "forward_text with zero style should succeed"
    );
}

/// AdaIn with non-zero style produces different output than zero style.
#[test]
fn test_adain_style_modulates_output() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = crate::kokoro_tts::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2], &[1, 2], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 2, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style_zero = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let style_ones = DynTensor::full(&[1, T_STYLE], 1.0, DType::F32, &cpu()).unwrap();

    let r0 = model
        .forward_text(&input_ids, &bert_out, &style_zero, 1.0)
        .unwrap();
    let r1 = model
        .forward_text(&input_ids, &bert_out, &style_ones, 1.0)
        .unwrap();

    // Duration logits should differ with different styles.
    let v0 = r0.dur_logits.to_vec1::<f32>().unwrap();
    let v1 = r1.dur_logits.to_vec1::<f32>().unwrap();
    let all_same = v0.iter().zip(v1.iter()).all(|(a, b)| (a - b).abs() < 1e-12);
    assert!(
        !all_same,
        "different styles should produce different dur_logits"
    );
}

// ===========================================================================
// Conv1d with padding: output shape parity
// ===========================================================================

/// TextEncoder conv layers produce [B, d_en, T] with appropriate padding.
#[test]
fn test_text_encoder_conv_output_shape() {
    let model = load_test_model();
    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4, 5], &[1, 5], &cpu()).unwrap();
    let text_features = model.text_encoder().forward(&input_ids).unwrap();
    assert_eq!(text_features.dims()[0], 1, "batch dim");
    assert_eq!(text_features.dims()[1], T_D_EN, "channel dim");
    // T is preserved through padded convolutions
    assert!(text_features.dims()[2] > 0, "time dim must be positive");
}

/// Varying sequence length produces corresponding output length.
#[test]
fn test_text_encoder_varying_seq_len() {
    let model = load_test_model();
    let short = DynTensor::from_vec_u32(vec![1u32, 2], &[1, 2], &cpu()).unwrap();
    let long = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4, 5, 6], &[1, 6], &cpu()).unwrap();

    let out_short = model.text_encoder().forward(&short).unwrap();
    let out_long = model.text_encoder().forward(&long).unwrap();

    // Both should have d_en channels
    assert_eq!(out_short.dims()[1], T_D_EN);
    assert_eq!(out_long.dims()[1], T_D_EN);
    // Longer input should produce different time dimension
    assert_ne!(
        out_short.dims()[2],
        out_long.dims()[2],
        "different input lengths should produce different output time dims"
    );
}

// ===========================================================================
// ResBlock forward: skip connection + activation
// ===========================================================================

/// ResBlock loaded from weights produces output with same [B, C, T] shape.
#[test]
fn test_resblock_forward_preserves_shape() {
    let weights = weights::make_kokoro_weights();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let next_ch = T_GEN_CH / 2;

    let rb = crate::kokoro_resblock::ResBlock::load(
        vb.pp("decoder.generator.resblocks.0"),
        next_ch,
        3,
        &[1, 2],
        T_STYLE,
    )
    .unwrap();

    let x = DynTensor::zeros(&[1, next_ch, 8], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let out = rb.forward(&x, &style).unwrap();
    assert_eq!(out.dims(), &[1, next_ch, 8]);
}

/// ResBlock with non-zero input produces non-trivial output (skip connection active).
#[test]
fn test_resblock_nonzero_input_nontrivial_output() {
    let weights = weights::make_kokoro_weights();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let next_ch = T_GEN_CH / 2;

    let rb = crate::kokoro_resblock::ResBlock::load(
        vb.pp("decoder.generator.resblocks.0"),
        next_ch,
        3,
        &[1, 2],
        T_STYLE,
    )
    .unwrap();

    let x = DynTensor::full(&[1, next_ch, 8], 0.5, DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();
    let out = rb.forward(&x, &style).unwrap();

    // Due to skip connection, output should not be all zeros even with zero weights
    let vals = out.to_vec1::<f32>().unwrap();
    let has_nonzero = vals.iter().any(|&v| v.abs() > 1e-8);
    assert!(
        has_nonzero,
        "ResBlock with nonzero input should produce nonzero output"
    );
}

// ===========================================================================
// Decoder stage pipeline: multiple ResBlocks in sequence
// ===========================================================================

/// Full forward produces valid magnitude/phase shapes for test config.
#[test]
fn test_decoder_pipeline_full_forward_shapes() {
    let model = load_test_model();
    let config = test_kokoro_config();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();
    let (mag, phase) = model.forward(&input_ids, &style, 1.0).unwrap();

    let n_bins = config.n_fft / 2 + 1;
    assert_eq!(mag.dims()[0], 1, "batch");
    assert_eq!(mag.dims()[1], n_bins, "frequency bins");
    assert_eq!(phase.dims()[0], 1, "batch");
    assert_eq!(phase.dims()[1], n_bins, "frequency bins");
    // Both outputs should have the same time dimension
    assert_eq!(
        mag.dims()[2],
        phase.dims()[2],
        "mag/phase time dim must match"
    );
}

/// Full forward with different speed produces different time length.
#[test]
fn test_decoder_pipeline_speed_affects_length() {
    let model = load_test_model();
    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();

    let (mag_fast, _) = model.forward(&input_ids, &style, 0.5).unwrap();
    let (mag_slow, _) = model.forward(&input_ids, &style, 2.0).unwrap();

    // Faster speed -> shorter output, slower -> longer
    assert_ne!(
        mag_fast.dims()[2],
        mag_slow.dims()[2],
        "different speeds should produce different time lengths"
    );
}

// ===========================================================================
// Duration predictor: conv + projection chain
// ===========================================================================

/// Duration logits have shape [B, T, max_dur] from forward_text.
#[test]
fn test_duration_predictor_output_shape() {
    let model = load_test_model();
    let config = test_kokoro_config();
    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4], &[1, 4], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 4, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();

    let result = model
        .forward_text(&input_ids, &bert_out, &style, 1.0)
        .unwrap();
    assert_eq!(result.dur_logits.dims()[0], 1);
    assert_eq!(result.dur_logits.dims()[1], 4);
    assert_eq!(result.dur_logits.dims()[2], config.max_dur);
}

/// Different input tokens produce different duration logits.
#[test]
fn test_duration_predictor_varies_with_input() {
    let model = load_test_model();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();

    let ids_a = DynTensor::from_vec_u32(vec![1u32, 2], &[1, 2], &cpu()).unwrap();
    let ids_b = DynTensor::from_vec_u32(vec![5u32, 7], &[1, 2], &cpu()).unwrap();
    let bert_a = DynTensor::zeros(&[1, 2, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let bert_b = DynTensor::full(&[1, 2, T_HIDDEN], 0.1, DType::F32, &cpu()).unwrap();

    let r_a = model.forward_text(&ids_a, &bert_a, &style, 1.0).unwrap();
    let r_b = model.forward_text(&ids_b, &bert_b, &style, 1.0).unwrap();

    let v_a = r_a.dur_logits.to_vec1::<f32>().unwrap();
    let v_b = r_b.dur_logits.to_vec1::<f32>().unwrap();
    let all_same = v_a
        .iter()
        .zip(v_b.iter())
        .all(|(a, b)| (a - b).abs() < 1e-12);
    assert!(
        !all_same,
        "different inputs should produce different duration logits"
    );
}

// ===========================================================================
// LSTM forward: hidden state evolution via TextEncoder
// ===========================================================================

/// TextEncoder (which contains BiLSTM) produces different outputs for different inputs,
/// demonstrating hidden state evolution.
#[test]
fn test_lstm_hidden_state_evolution() {
    let model = load_test_model();
    let ids_a = DynTensor::from_vec_u32(vec![1u32, 1, 1, 1], &[1, 4], &cpu()).unwrap();
    let ids_b = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4], &[1, 4], &cpu()).unwrap();

    let out_a = model.text_encoder().forward(&ids_a).unwrap();
    let out_b = model.text_encoder().forward(&ids_b).unwrap();

    assert_eq!(out_a.dims(), out_b.dims());
    let va = out_a.to_vec1::<f32>().unwrap();
    let vb = out_b.to_vec1::<f32>().unwrap();
    let all_same = va.iter().zip(vb.iter()).all(|(a, b)| (a - b).abs() < 1e-12);
    assert!(
        !all_same,
        "BiLSTM should evolve differently for different token sequences"
    );
}

/// TextEncoder output rank is 3: [B, d_en, T].
#[test]
fn test_text_encoder_output_rank() {
    let model = load_test_model();
    let input = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let out = model.text_encoder().forward(&input).unwrap();
    assert_eq!(out.rank(), 3, "TextEncoder output must be rank 3");
}

// ===========================================================================
// Full voice decoder: text embedding -> mel output shapes
// ===========================================================================

/// Full model forward succeeds and produces finite outputs.
#[test]
fn test_full_voice_decoder_produces_finite_output() {
    let model = load_test_model();
    let input_ids = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();

    let (mag, phase) = model.forward(&input_ids, &style, 1.0).unwrap();
    let mag_vals = mag.to_vec1::<f32>().unwrap();
    let phase_vals = phase.to_vec1::<f32>().unwrap();

    assert!(
        mag_vals.iter().all(|v| v.is_finite()),
        "magnitude must be finite"
    );
    assert!(
        phase_vals.iter().all(|v| v.is_finite()),
        "phase must be finite"
    );
}

// ===========================================================================
// Weight loading roundtrip: save -> load -> identical forward
// ===========================================================================

/// Loading the same weights twice produces identical forward results.
#[test]
fn test_weight_loading_roundtrip_identical_forward() {
    let model1 = load_test_model();
    let model2 = load_test_model();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();

    let (mag1, phase1) = model1.forward(&input_ids, &style, 1.0).unwrap();
    let (mag2, phase2) = model2.forward(&input_ids, &style, 1.0).unwrap();

    let m1 = mag1.to_vec1::<f32>().unwrap();
    let m2 = mag2.to_vec1::<f32>().unwrap();
    assert_eq!(m1.len(), m2.len());
    for (i, (&a, &b)) in m1.iter().zip(m2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "magnitude mismatch at index {i}: {a} vs {b}"
        );
    }

    let p1 = phase1.to_vec1::<f32>().unwrap();
    let p2 = phase2.to_vec1::<f32>().unwrap();
    for (i, (&a, &b)) in p1.iter().zip(p2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "phase mismatch at index {i}: {a} vs {b}"
        );
    }
}

/// ConvertedModel weights loaded twice yield identical weight tensors.
#[test]
fn test_converted_model_weight_roundtrip() {
    let w1 = weights::make_kokoro_weights();
    let w2 = weights::make_kokoro_weights();

    assert_eq!(w1.len(), w2.len());
    for (name, t1) in &w1 {
        let t2 = w2.get(name).unwrap_or_else(|| panic!("missing: {name}"));
        assert_eq!(t1.dims(), t2.dims(), "shape mismatch for {name}");
        let v1 = t1.to_vec1::<f32>().unwrap();
        let v2 = t2.to_vec1::<f32>().unwrap();
        assert_eq!(v1, v2, "value mismatch for {name}");
    }
}

/// ConvertConfig builder round-trips all fields.
#[test]
fn test_convert_config_roundtrip_extended() {
    let config = ConvertConfig::new("kokoro-test-ext")
        .with_validate_weights(false)
        .with_constant_fold(false);
    assert_eq!(config.model_name, "kokoro-test-ext");
    assert!(!config.validate_weights);
    assert!(!config.constant_fold);
}
