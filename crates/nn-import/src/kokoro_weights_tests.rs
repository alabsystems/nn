// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro weight name mapping, quantization detection, and op map context.
//! Part of #2465, #4276.

use super::*;

// ===========================================================================
// map_pytorch_key: identity-mapped prefixes
// ===========================================================================

#[test]
fn test_map_plbert_key() {
    let key = "plbert.embeddings.word_embeddings.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_bert_encoder_key() {
    let key = "bert_encoder.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_text_encoder_key() {
    let key = "text_encoder.lstm.weight_ih_l0";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_prosody_predictor_key() {
    let key = "prosody_predictor.shared.0.conv.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_predictor_f0_key() {
    let key = "predictor.F0.0.n1.fc.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_key() {
    let key = "decoder.conv_pre.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_unknown_key_returns_none() {
    assert_eq!(map_pytorch_key("unknown.weight"), None);
    assert_eq!(map_pytorch_key("model.encoder.weight"), None);
}

// -- plbert sub-keys --

#[test]
fn test_map_plbert_position_embeddings() {
    let key = "plbert.embeddings.position_embeddings.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_plbert_token_type_embeddings() {
    let key = "plbert.embeddings.token_type_embeddings.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_plbert_layer_norm_weight_and_bias() {
    for suffix in &["weight", "bias"] {
        let key = format!("plbert.embeddings.LayerNorm.{suffix}");
        assert_eq!(map_pytorch_key(&key), Some(key.clone()), "failed for {key}");
    }
}

#[test]
fn test_map_plbert_encoder_hidden_mapping() {
    let key = "plbert.encoder.embedding_hidden_mapping_in.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_plbert_encoder_attention_qkv() {
    for proj in &["query", "key", "value", "dense"] {
        let key =
            format!("plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.{proj}.weight");
        assert_eq!(
            map_pytorch_key(&key),
            Some(key.clone()),
            "failed for {proj}"
        );
    }
}

#[test]
fn test_map_plbert_encoder_ffn() {
    for layer in &["ffn", "ffn_output"] {
        for suffix in &["weight", "bias"] {
            let key =
                format!("plbert.encoder.albert_layer_groups.0.albert_layers.0.{layer}.{suffix}");
            assert_eq!(map_pytorch_key(&key), Some(key.clone()));
        }
    }
}

// -- text_encoder sub-keys --

#[test]
fn test_map_text_encoder_lstm_reverse() {
    let key = "text_encoder.lstm.weight_ih_l0_reverse";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_text_encoder_lstm_linear() {
    let key = "text_encoder.lstm.linear.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

// -- prosody_predictor sub-keys --

#[test]
fn test_map_prosody_predictor_lstm() {
    let key = "prosody_predictor.shared.0.lstm.weight_ih_l0";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_prosody_predictor_norms() {
    let key = "prosody_predictor.norms.0.norm.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_prosody_predictor_duration_proj() {
    let key = "prosody_predictor.duration_proj.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

// -- predictor sub-keys --

#[test]
fn test_map_predictor_shared_bilstm() {
    let key = "predictor.shared.weight_ih_l0";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_predictor_shared_bilstm_reverse() {
    let key = "predictor.shared.weight_hh_l0_reverse";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_predictor_f0_proj() {
    let key = "predictor.F0_proj.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_predictor_n_resblock() {
    let key = "predictor.N.0.c1.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_predictor_n_proj() {
    let key = "predictor.N_proj.bias";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

// -- decoder sub-keys --

#[test]
fn test_map_decoder_ups() {
    let key = "decoder.ups.0.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_noise_convs() {
    let key = "decoder.noise_convs.0.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_noise_res_convs() {
    let key = "decoder.noise_res.0.convs1.0.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_noise_res_adain() {
    let key = "decoder.noise_res.0.adain1.0.fc.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_noise_res_alpha() {
    let key = "decoder.noise_res.0.alpha1.0";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_resblocks_convs() {
    let key = "decoder.resblocks.0.convs1.0.weight";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_resblocks_adain() {
    let key = "decoder.resblocks.0.adain2.0.fc.bias";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_resblocks_alpha() {
    let key = "decoder.resblocks.0.alpha2.0";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

#[test]
fn test_map_decoder_conv_post() {
    let key = "decoder.conv_post.bias";
    assert_eq!(map_pytorch_key(key), Some(key.to_string()));
}

// -- edge cases --

#[test]
fn test_map_empty_string_returns_none() {
    assert_eq!(map_pytorch_key(""), None);
}

#[test]
fn test_map_partial_prefix_not_matched() {
    // "plber" is not "plbert." -- should not match.
    assert_eq!(map_pytorch_key("plber.weight"), None);
}

#[test]
fn test_map_prefix_only_no_suffix() {
    // Just the prefix with nothing after the dot is still a valid prefix match.
    assert_eq!(map_pytorch_key("plbert."), Some("plbert.".to_string()));
}

// ===========================================================================
// validate_kokoro_keys
// ===========================================================================

#[test]
fn test_validate_all_prefixes_present() {
    let keys = vec![
        "plbert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.F0.0.n1.fc.weight",
        "decoder.conv_pre.weight",
    ];
    let missing = validate_kokoro_keys(&keys);
    assert!(missing.is_empty(), "unexpected missing: {missing:?}");
}

#[test]
fn test_validate_missing_prefix() {
    let keys = vec![
        "plbert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        // text_encoder, prosody_predictor, predictor, decoder missing
    ];
    let missing = validate_kokoro_keys(&keys);
    assert_eq!(missing.len(), 4);
    assert!(missing.contains(&"text_encoder."));
    assert!(missing.contains(&"prosody_predictor."));
    assert!(missing.contains(&"predictor."));
    assert!(missing.contains(&"decoder."));
}

#[test]
fn test_validate_missing_single_decoder() {
    let keys = vec![
        "plbert.x",
        "bert_encoder.x",
        "text_encoder.x",
        "prosody_predictor.x",
        "predictor.x",
    ];
    let missing = validate_kokoro_keys(&keys);
    assert_eq!(missing, vec!["decoder."]);
}

#[test]
fn test_validate_empty_keys() {
    let keys: Vec<&str> = vec![];
    let missing = validate_kokoro_keys(&keys);
    assert_eq!(missing.len(), 6, "all 6 prefixes should be missing");
}

#[test]
fn test_validate_duplicate_prefix_still_passes() {
    // Multiple keys with same prefix should still count as present.
    let keys = vec![
        "plbert.a",
        "plbert.b",
        "bert_encoder.a",
        "text_encoder.a",
        "prosody_predictor.a",
        "predictor.a",
        "decoder.a",
    ];
    let missing = validate_kokoro_keys(&keys);
    assert!(missing.is_empty());
}

// ===========================================================================
// validate_kokoro_safetensors
// ===========================================================================

#[test]
fn test_validate_safetensors_ok() {
    let keys: Vec<String> = vec![
        "plbert.x",
        "bert_encoder.w",
        "text_encoder.y",
        "prosody_predictor.z",
        "predictor.a",
        "decoder.b",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let result = validate_kokoro_safetensors(&keys);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 6);
}

#[test]
fn test_validate_safetensors_missing() {
    let keys: Vec<String> = vec!["plbert.x".to_string()];
    let result = validate_kokoro_safetensors(&keys);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("missing Kokoro weight groups"));
}

#[test]
fn test_validate_safetensors_empty_returns_error() {
    let keys: Vec<String> = vec![];
    let result = validate_kokoro_safetensors(&keys);
    assert!(result.is_err());
}

#[test]
fn test_validate_safetensors_counts_only_mapped_keys() {
    let keys: Vec<String> = vec![
        "plbert.x",
        "bert_encoder.w",
        "text_encoder.y",
        "prosody_predictor.z",
        "predictor.a",
        "decoder.b",
        "unknown.extra.tensor", // not a known prefix
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let result = validate_kokoro_safetensors(&keys);
    assert!(result.is_ok());
    // Only the 6 known-prefix keys map; the unknown one is filtered.
    assert_eq!(result.unwrap(), 6);
}

// ===========================================================================
// kokoro_name_mapping closure
// ===========================================================================

#[test]
fn test_kokoro_name_mapping_closure() {
    let mapper = kokoro_name_mapping();
    assert_eq!(
        mapper("plbert.embeddings.weight"),
        "plbert.embeddings.weight"
    );
    assert_eq!(mapper("decoder.conv_pre.weight"), "decoder.conv_pre.weight");
    // Unknown keys pass through unchanged.
    assert_eq!(mapper("unknown.key"), "unknown.key");
}

#[test]
fn test_kokoro_name_mapping_is_send_sync() {
    let mapper = kokoro_name_mapping();
    // Verify the closure can be sent across threads (required for VarBuilder).
    fn assert_send_sync<T: Send + Sync>(_t: &T) {}
    assert_send_sync(&mapper);
}

// ===========================================================================
// Quantization report tests
// ===========================================================================

/// Helper: build a minimal safetensors blob in memory with given tensors.
fn build_safetensors_bytes(tensors: &[(&str, safetensors::Dtype, &[usize])]) -> Vec<u8> {
    use safetensors::serialize;
    use safetensors::tensor::TensorView;

    let owned_data: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_name, dtype, shape)| {
            let num_elements: usize = shape.iter().product();
            let bytes_per_elem = match dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::I8 | safetensors::Dtype::U8 => 1,
                safetensors::Dtype::F64 => 8,
                _ => 4,
            };
            vec![0u8; num_elements * bytes_per_elem]
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .zip(owned_data.iter())
        .map(|((name, dtype, shape), data)| {
            let view = TensorView::new(*dtype, shape.to_vec(), data).unwrap();
            (*name, view)
        })
        .collect();

    serialize(views.iter().map(|(n, v)| (*n, v)), None).unwrap()
}

#[test]
fn test_detect_quantization_f32_only() {
    let bytes = build_safetensors_bytes(&[
        ("layer.weight", safetensors::Dtype::F32, &[256, 256]),
        ("layer.bias", safetensors::Dtype::F32, &[256]),
    ]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_tensors, 2);
    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(
        report.dtype_breakdown[0].dtype,
        crate::quantization::DetectedDtype::F32,
    );
}

#[test]
fn test_detect_quantization_mixed_f16_f32() {
    let bytes = build_safetensors_bytes(&[
        ("big.weight", safetensors::Dtype::F32, &[1024, 1024]),
        ("small.weight", safetensors::Dtype::F16, &[64, 64]),
    ]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);
}

#[test]
fn test_detect_quantization_f16_only_no_f16_recommendation() {
    let bytes = build_safetensors_bytes(&[("layer.weight", safetensors::Dtype::F16, &[256, 256])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    assert!(!report.is_mixed_precision());
    let has_f16_rec = report
        .recommendations
        .iter()
        .any(|r| r.target_dtype == crate::quantization::DetectedDtype::F16);
    assert!(
        !has_f16_rec,
        "F16-only model should not recommend further F16 quantization"
    );
}

#[test]
fn test_quant_recommendation_f16_savings_arithmetic() {
    let bytes = build_safetensors_bytes(&[("w", safetensors::Dtype::F32, &[1024, 1024])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == crate::quantization::DetectedDtype::F16)
        .expect("F32 model should have F16 recommendation");
    assert_eq!(
        f16_rec.savings_bytes,
        f16_rec.current_bytes - f16_rec.projected_bytes
    );
    assert_eq!(f16_rec.projected_bytes, f16_rec.current_bytes / 2);
}

#[test]
fn test_quant_recommendation_i8_savings_arithmetic() {
    let bytes = build_safetensors_bytes(&[("w", safetensors::Dtype::F32, &[1024, 1024])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let i8_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == crate::quantization::DetectedDtype::I8)
        .expect("F32 model should have I8 recommendation");
    assert_eq!(i8_rec.projected_bytes, i8_rec.current_bytes / 4);
}

#[test]
fn test_dtype_breakdown_accuracy() {
    let bytes = build_safetensors_bytes(&[
        ("a", safetensors::Dtype::F32, &[100, 100]),
        ("b", safetensors::Dtype::F32, &[50, 50]),
        ("c", safetensors::Dtype::F16, &[200]),
    ]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let f32_bd = report
        .dtype_breakdown
        .iter()
        .find(|b| b.dtype == crate::quantization::DetectedDtype::F32)
        .unwrap();
    assert_eq!(f32_bd.tensor_count, 2);
    assert_eq!(f32_bd.total_parameters, 100 * 100 + 50 * 50);
    assert_eq!(f32_bd.total_bytes, (100 * 100 + 50 * 50) * 4);
}

#[test]
fn test_dtype_fraction() {
    let bytes = build_safetensors_bytes(&[
        ("a", safetensors::Dtype::F32, &[1000]),
        ("b", safetensors::Dtype::F16, &[1000]),
    ]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let f32_frac = report.dtype_fraction(crate::quantization::DetectedDtype::F32);
    let f16_frac = report.dtype_fraction(crate::quantization::DetectedDtype::F16);
    let expected_f32 = 4000.0 / 6000.0;
    let expected_f16 = 2000.0 / 6000.0;
    assert!((f32_frac - expected_f32).abs() < 1e-10);
    assert!((f16_frac - expected_f16).abs() < 1e-10);
}

#[test]
fn test_tensor_quant_info_fields() {
    let bytes = build_safetensors_bytes(&[("layer.weight", safetensors::Dtype::F32, &[64, 32])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.tensors.len(), 1);
    let t = &report.tensors[0];
    assert_eq!(t.name, "layer.weight");
    assert_eq!(t.dtype, crate::quantization::DetectedDtype::F32);
    assert_eq!(t.shape, vec![64, 32]);
    assert_eq!(t.num_elements, 64 * 32);
    assert_eq!(t.size_bytes, 64 * 32 * 4);
}

#[test]
fn test_total_savings_bytes() {
    let bytes = build_safetensors_bytes(&[("w", safetensors::Dtype::F32, &[2048, 2048])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let total_savings = report.total_savings_bytes();
    let manual: usize = report.recommendations.iter().map(|r| r.savings_bytes).sum();
    assert_eq!(total_savings, manual);
    assert!(total_savings > 0);
}

#[test]
fn test_small_tensors_no_recommendation() {
    let bytes = build_safetensors_bytes(&[("bias", safetensors::Dtype::F32, &[512])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    assert!(
        report.recommendations.is_empty(),
        "small F32 tensors (<1024 elements) should not be recommended for quantization"
    );
}

#[test]
fn test_f64_to_f32_recommendation() {
    let bytes = build_safetensors_bytes(&[("big", safetensors::Dtype::F64, &[1000])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let f32_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == crate::quantization::DetectedDtype::F32);
    assert!(
        f32_rec.is_some(),
        "F64 tensors should get F32 recommendation"
    );
    let rec = f32_rec.unwrap();
    assert_eq!(rec.projected_bytes, rec.current_bytes / 2);
}

#[test]
fn test_quantization_report_summary_contains_sections() {
    let bytes = build_safetensors_bytes(&[("w", safetensors::Dtype::F32, &[1024, 1024])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();
    assert!(summary.contains("Quantization Report"));
    assert!(summary.contains("Dtype Breakdown"));
    assert!(summary.contains("F32"));
    assert!(summary.contains("Recommendations"));
}

#[test]
fn test_quantization_report_display_matches_summary() {
    let bytes = build_safetensors_bytes(&[("w", safetensors::Dtype::F32, &[1024])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    let display = format!("{report}");
    assert_eq!(display, report.summary());
}

#[test]
fn test_quantization_report_no_recommendations_for_compact() {
    let bytes = build_safetensors_bytes(&[("tiny", safetensors::Dtype::I8, &[100])]);
    let report = crate::quantization::detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.recommendations.is_empty());
    assert!(report.summary().contains("already compact"));
}

// ===========================================================================
// DetectedDtype unit tests
// ===========================================================================

#[test]
fn test_detected_dtype_bytes_per_element() {
    use crate::quantization::DetectedDtype;
    assert_eq!(DetectedDtype::F32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::F16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::BF16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::F64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::I8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::U8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::I16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::I32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::I64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::Bool.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::SubByte.bytes_per_element(), None);
    assert_eq!(DetectedDtype::Other.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_label_all_variants() {
    use crate::quantization::DetectedDtype;
    assert_eq!(DetectedDtype::F32.label(), "F32");
    assert_eq!(DetectedDtype::F16.label(), "F16");
    assert_eq!(DetectedDtype::BF16.label(), "BF16");
    assert_eq!(DetectedDtype::F64.label(), "F64");
    assert_eq!(DetectedDtype::I8.label(), "I8");
    assert_eq!(DetectedDtype::U8.label(), "U8");
    assert_eq!(DetectedDtype::F8.label(), "F8");
    assert_eq!(DetectedDtype::SubByte.label(), "SubByte");
    assert_eq!(DetectedDtype::Other.label(), "Other");
}

#[test]
fn test_detected_dtype_display() {
    use crate::quantization::DetectedDtype;
    assert_eq!(format!("{}", DetectedDtype::F32), "F32");
    assert_eq!(format!("{}", DetectedDtype::BF16), "BF16");
}

#[test]
fn test_detected_dtype_from_safetensors_roundtrip() {
    use crate::quantization::DetectedDtype;
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F32),
        DetectedDtype::F32,
    );
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F16),
        DetectedDtype::F16,
    );
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::BF16),
        DetectedDtype::BF16,
    );
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::I8),
        DetectedDtype::I8,
    );
}

// ===========================================================================
// OpMapContext / ResolvedWeight / supported_ops tests
// ===========================================================================

#[test]
fn test_resolved_weight_new() {
    let rw = crate::op_map::ResolvedWeight::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(rw.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(rw.shape, vec![3]);
}

#[test]
fn test_resolved_weight_empty() {
    let rw = crate::op_map::ResolvedWeight::new(vec![], vec![0]);
    assert!(rw.data.is_empty());
    assert_eq!(rw.shape, vec![0]);
}

#[test]
fn test_resolved_weight_multidim() {
    let rw = crate::op_map::ResolvedWeight::new(vec![1.0; 12], vec![3, 4]);
    assert_eq!(rw.data.len(), 12);
    assert_eq!(rw.shape, vec![3, 4]);
}

#[test]
fn test_op_map_context_construction() {
    use std::collections::HashMap;
    let tensor_meta: HashMap<String, crate::parse::TensorMeta> = HashMap::new();
    let weights: HashMap<String, crate::op_map::ResolvedWeight> = HashMap::new();
    let ctx = crate::op_map::OpMapContext {
        tensor_meta: &tensor_meta,
        weights: &weights,
    };
    assert!(ctx.tensor_meta.is_empty());
    assert!(ctx.weights.is_empty());
}

#[test]
fn test_op_map_context_with_weights() {
    use std::collections::HashMap;
    let tensor_meta: HashMap<String, crate::parse::TensorMeta> = HashMap::new();
    let mut weights: HashMap<String, crate::op_map::ResolvedWeight> = HashMap::new();
    weights.insert(
        "layer.weight".to_string(),
        crate::op_map::ResolvedWeight::new(vec![0.5; 6], vec![2, 3]),
    );
    let ctx = crate::op_map::OpMapContext {
        tensor_meta: &tensor_meta,
        weights: &weights,
    };
    assert_eq!(ctx.weights.len(), 1);
    let w = ctx.weights.get("layer.weight").unwrap();
    assert_eq!(w.shape, vec![2, 3]);
}

#[test]
fn test_supported_ops_non_empty() {
    let ops = crate::op_map::supported_ops();
    assert!(
        !ops.is_empty(),
        "supported_ops should return at least one op"
    );
}

#[test]
fn test_supported_ops_sorted() {
    let ops = crate::op_map::supported_ops();
    for window in ops.windows(2) {
        assert!(
            window[0] <= window[1],
            "supported_ops should be sorted, but {:?} > {:?}",
            window[0],
            window[1],
        );
    }
}

#[test]
fn test_supported_ops_deduplicated() {
    let ops = crate::op_map::supported_ops();
    for window in ops.windows(2) {
        assert_ne!(
            window[0], window[1],
            "supported_ops should be deduplicated, found duplicate: {:?}",
            window[0],
        );
    }
}

#[test]
fn test_supported_ops_contains_kokoro_required_ops() {
    let ops = crate::op_map::supported_ops();
    let kokoro_required = [
        "aten::conv1d",
        "aten::conv_transpose1d",
        "aten::linear",
        "aten::lstm",
        "aten::embedding",
        "aten::softmax",
        "aten::relu",
        "aten::sigmoid",
        "aten::tanh",
        "aten::layer_norm",
        "aten::group_norm",
        "aten::cat",
        "aten::reshape",
        "aten::transpose",
        "aten::add",
        "aten::mul",
    ];
    for required in &kokoro_required {
        assert!(
            ops.contains(required),
            "supported_ops is missing Kokoro-required op: {required}",
        );
    }
}

#[test]
fn test_supported_ops_contains_attention_ops() {
    let ops = crate::op_map::supported_ops();
    assert!(ops.contains(&"aten::scaled_dot_product_attention"));
    assert!(ops.contains(&"aten::softmax"));
}

#[test]
fn test_map_node_unsupported_target_returns_error() {
    use std::collections::HashMap;
    let node = crate::parse::Node {
        target: "torch.ops.aten.fake_op_that_does_not_exist.default".to_string(),
        inputs: vec![],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let tensor_meta: HashMap<String, crate::parse::TensorMeta> = HashMap::new();
    let weights: HashMap<String, crate::op_map::ResolvedWeight> = HashMap::new();
    let ctx = crate::op_map::OpMapContext {
        tensor_meta: &tensor_meta,
        weights: &weights,
    };
    let result = crate::op_map::map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_err(), "unsupported op should return error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("unsupported") || msg.contains("Unsupported"),
        "error: {msg}"
    );
}
