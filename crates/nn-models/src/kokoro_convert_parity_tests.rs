// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro auto-converter parity tests (Part of #4276).
//!
//! Verifies that a `ConvertedModel` built from the same weights as the manual
//! `KokoroModel` builder produces consistent metadata: parameter counts, weight
//! names, output shapes, and config-derived dimensions all agree.

#[path = "kokoro_convert_parity_weights.rs"]
mod weights;

use crate::convert::{ConvertConfig, ConvertedModel};
use crate::kokoro_tts::KokoroConfig;
use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::test_utils::cpu;
use nn_core::DType;

// ---------------------------------------------------------------------------
// Test-scale config constants (must match kokoro_tts_tests_model.rs)
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

// ---------------------------------------------------------------------------
// Helper: build ConvertedModel from manual weights
// ---------------------------------------------------------------------------

fn converted_from_manual_weights() -> ConvertedModel {
    let weights = weights::make_kokoro_weights();
    ConvertedModel::new(
        ComputationGraph::from_nodes(vec![]),
        weights,
        1,
        vec!["input_ids".to_string()],
        vec!["magnitude".to_string(), "phase".to_string()],
        "kokoro-82m-test".to_string(),
    )
}

// ===========================================================================
// Tests
// ===========================================================================

/// Verify that KokoroModel::load succeeds with the same weights that
/// populate the ConvertedModel (weight name coverage parity).
#[test]
fn test_manual_model_loads_from_same_weights() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = crate::kokoro_tts::KokoroModel::load(&vb, &config);
    assert!(model.is_ok(), "KokoroModel::load failed: {:?}", model.err());
}

/// Verify ConvertedModel total_params matches the sum of all manual weights.
#[test]
fn test_converted_model_param_count_matches_manual() {
    let manual_weights = weights::make_kokoro_weights();
    let manual_total: usize = manual_weights.values().map(DynTensor::elem_count).sum();

    let converted = converted_from_manual_weights();
    assert_eq!(
        converted.total_params(),
        manual_total,
        "ConvertedModel total_params must equal sum of manual weight elements"
    );
}

/// Verify ConvertedModel num_weights matches the number of distinct tensors.
#[test]
fn test_converted_model_weight_count_matches_manual() {
    let manual_weights = weights::make_kokoro_weights();
    let converted = converted_from_manual_weights();
    assert_eq!(
        converted.num_weights(),
        manual_weights.len(),
        "ConvertedModel num_weights must equal number of manual weight tensors"
    );
}

/// Verify every weight from the manual builder is accessible by name.
#[test]
fn test_converted_model_all_weights_accessible() {
    let manual_weights = weights::make_kokoro_weights();
    let converted = converted_from_manual_weights();
    for name in manual_weights.keys() {
        assert!(
            converted.weight(name).is_some(),
            "ConvertedModel missing weight: {name}"
        );
    }
}

/// Verify weight shapes match between manual and converted.
#[test]
fn test_converted_model_weight_shapes_match() {
    let manual_weights = weights::make_kokoro_weights();
    let converted = converted_from_manual_weights();
    for (name, manual_tensor) in &manual_weights {
        let converted_tensor = converted
            .weight(name)
            .unwrap_or_else(|| panic!("missing weight: {name}"));
        assert_eq!(
            manual_tensor.dims(),
            converted_tensor.dims(),
            "shape mismatch for weight '{name}': manual={:?}, converted={:?}",
            manual_tensor.dims(),
            converted_tensor.dims(),
        );
    }
}

/// Verify ConvertedModel metadata fields.
#[test]
fn test_converted_model_metadata() {
    let converted = converted_from_manual_weights();
    assert_eq!(converted.num_inputs(), 1);
    assert_eq!(converted.input_names(), &["input_ids"]);
    assert_eq!(converted.output_names(), &["magnitude", "phase"]);
    assert_eq!(converted.model_name, "kokoro-82m-test");
}

/// Verify model_name round-trips through from_imported.
#[test]
fn test_from_imported_metadata_roundtrip() {
    let weights = weights::make_kokoro_weights();
    let weight_count = weights.len();
    let model = ConvertedModel::from_imported(
        ComputationGraph::from_nodes(vec![]),
        2,
        vec!["tokens".to_string(), "style".to_string()],
        vec!["audio".to_string()],
        weights,
        "kokoro-imported",
    );
    assert_eq!(model.num_inputs(), 2);
    assert_eq!(model.num_weights(), weight_count);
    assert_eq!(model.model_name, "kokoro-imported");
    assert_eq!(model.input_names(), &["tokens", "style"]);
    assert_eq!(model.output_names(), &["audio"]);
}

/// Verify per-submodule weight prefixes are all present.
#[test]
fn test_weight_prefix_coverage() {
    let converted = converted_from_manual_weights();
    let weight_names: Vec<&String> = converted.weights.keys().collect();

    let required_prefixes = [
        "plbert.",
        "bert_encoder.",
        "text_encoder.",
        "prosody_predictor.",
        "predictor.",
        "decoder.F0_conv.",
        "decoder.N_conv.",
        "decoder.asr_res.",
        "decoder.encode.",
        "decoder.decode.",
        "decoder.generator.",
    ];

    for prefix in &required_prefixes {
        let has_prefix = weight_names.iter().any(|n| n.starts_with(prefix));
        assert!(has_prefix, "no weights found with prefix '{prefix}'");
    }
}

/// Verify KokoroModel sub-module output shapes match config expectations.
#[test]
fn test_manual_model_submodule_output_shapes() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = crate::kokoro_tts::KokoroModel::load(&vb, &config).unwrap();

    // PlBert: [B, T, hidden_size]
    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4], &[1, 4], &cpu()).unwrap();
    let bert_out = model.plbert().forward(&input_ids).unwrap();
    assert_eq!(bert_out.dims(), &[1, 4, T_HIDDEN]);

    // bert_encoder Linear: [B, T, d_en]
    let encoded = model.bert_encoder().forward(&bert_out).unwrap();
    assert_eq!(encoded.dims(), &[1, 4, T_D_EN]);

    // TextEncoder: [B, d_en, T]
    let text_features = model.text_encoder().forward(&input_ids).unwrap();
    assert_eq!(text_features.dims()[0], 1);
    assert_eq!(text_features.dims()[1], T_D_EN);
}

/// Verify forward_text output shapes agree with config dimensions.
#[test]
fn test_forward_text_output_shapes_from_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = crate::kokoro_tts::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &cpu()).unwrap();
    let bert_out = DynTensor::zeros(&[1, 3, T_HIDDEN], DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, T_STYLE], DType::F32, &cpu()).unwrap();

    let result = model
        .forward_text(&input_ids, &bert_out, &style, 1.0)
        .unwrap();

    // dur_logits: [B, T, max_dur]
    assert_eq!(result.dur_logits.dims()[0], 1);
    assert_eq!(result.dur_logits.dims()[1], 3);
    assert_eq!(result.dur_logits.dims()[2], config.max_dur);

    // regulated: [B, d_en, T_mel]
    assert_eq!(result.regulated.dims()[0], 1);
    assert_eq!(result.regulated.dims()[1], config.d_en);

    // aligned_dur: [B, d_en+style_dim, T_mel]
    assert_eq!(result.aligned_dur.dims()[0], 1);
}

/// Verify full forward produces output shapes consistent with config.
#[test]
fn test_full_forward_output_shapes_from_config() {
    let weights = weights::make_kokoro_weights();
    let config = test_kokoro_config();
    let vb = nn_core::VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let model = crate::kokoro_tts::KokoroModel::load(&vb, &config).unwrap();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::zeros(&[1, 2 * T_STYLE], DType::F32, &cpu()).unwrap();
    let (mag, phase) = model.forward(&input_ids, &style, 1.0).unwrap();

    let n_bins = config.n_fft / 2 + 1;
    assert_eq!(mag.dims()[0], 1);
    assert_eq!(mag.dims()[1], n_bins);
    assert_eq!(phase.dims()[0], 1);
    assert_eq!(phase.dims()[1], n_bins);
    assert_eq!(mag.dims()[2], phase.dims()[2]);
}

/// Verify ConvertConfig round-trips for Kokoro.
#[test]
fn test_convert_config_for_kokoro() {
    let config = ConvertConfig::new("kokoro-82m")
        .with_validate_weights(true)
        .with_constant_fold(true);
    assert_eq!(config.model_name, "kokoro-82m");
    assert!(config.validate_weights);
    assert!(config.constant_fold);
    assert!(config.model_type.is_none());
}

/// Verify per-submodule parameter counts are non-zero and sum to total.
#[test]
fn test_submodule_parameter_counts() {
    let converted = converted_from_manual_weights();
    let count_with_prefix = |prefix: &str| -> usize {
        converted
            .weights
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(_, v)| v.elem_count())
            .sum()
    };

    let plbert_params = count_with_prefix("plbert.");
    let text_enc_params = count_with_prefix("text_encoder.");
    let prosody_params = count_with_prefix("prosody_predictor.");
    let predictor_params = count_with_prefix("predictor.");
    let decoder_params = count_with_prefix("decoder.");
    let bert_enc_params = count_with_prefix("bert_encoder.");

    assert!(plbert_params > 0, "plbert should have parameters");
    assert!(text_enc_params > 0, "text_encoder should have parameters");
    assert!(
        prosody_params > 0,
        "prosody_predictor should have parameters"
    );
    assert!(predictor_params > 0, "predictor should have parameters");
    assert!(decoder_params > 0, "decoder should have parameters");
    assert!(bert_enc_params > 0, "bert_encoder should have parameters");

    let submodule_total = plbert_params
        + text_enc_params
        + prosody_params
        + predictor_params
        + decoder_params
        + bert_enc_params;
    assert_eq!(
        submodule_total,
        converted.total_params(),
        "sum of submodule params must equal total params"
    );
}

/// Verify weight name listing sorted alphabetically is deterministic.
#[test]
fn test_weight_names_deterministic() {
    let m1 = converted_from_manual_weights();
    let m2 = converted_from_manual_weights();

    let mut names1: Vec<&String> = m1.weights.keys().collect();
    let mut names2: Vec<&String> = m2.weights.keys().collect();
    names1.sort();
    names2.sort();
    assert_eq!(names1, names2, "weight name lists should be identical");
}

/// Verify Debug output includes key metadata.
#[test]
fn test_converted_model_debug_format() {
    let converted = converted_from_manual_weights();
    let debug = format!("{converted:?}");
    assert!(debug.contains("kokoro-82m-test"));
    assert!(debug.contains("num_ops"));
    assert!(debug.contains("total_params"));
}

/// Verify ConvertedModel with empty graph has 0 ops.
#[test]
fn test_converted_model_empty_graph_ops() {
    let converted = converted_from_manual_weights();
    assert_eq!(converted.num_ops(), 0);
}

/// Verify KokoroConfig::validate passes for test config.
#[test]
fn test_kokoro_config_validates() {
    let config = test_kokoro_config();
    assert!(config.validate().is_ok());
}

/// Verify KokoroConfig::default validates.
#[test]
fn test_kokoro_default_config_validates() {
    let config = KokoroConfig::default();
    assert!(config.validate().is_ok());
}

/// Verify production-scale Kokoro parameter structure.
#[test]
fn test_production_config_parameter_structure() {
    let config = KokoroConfig::default();
    assert_eq!(config.d_en, 512);
    assert_eq!(config.gen_initial_channels, 512);
    assert_eq!(config.style_dim, 128);
    assert_eq!(config.upsample_rates, vec![10, 6]);
    assert_eq!(config.upsample_kernel_sizes, vec![20, 12]);
    assert_eq!(config.resblock_kernel_sizes, vec![3, 7, 11]);
    assert_eq!(config.n_fft, 20);
    assert_eq!(config.plbert.vocab_size, 178);
    assert_eq!(config.plbert.hidden_size, 768);
}
