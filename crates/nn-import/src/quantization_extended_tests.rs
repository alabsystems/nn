// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended quantization detection and import tests for nn-import.
//! Part of #4186.

use crate::quantization::{
    detect_quantization_from_bytes, DetectedDtype, DtypeBreakdown, QuantRecommendation,
    TensorQuantInfo,
};

// ---------------------------------------------------------------------------
// Helper: build safetensors bytes for testing
// ---------------------------------------------------------------------------

/// Build a minimal safetensors blob from typed tensor specs.
fn build_safetensors(tensors: &[(&str, safetensors::Dtype, &[usize])]) -> Vec<u8> {
    use safetensors::tensor::TensorView;

    let owned_data: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_name, dtype, shape)| {
            let num_elements: usize = shape.iter().product();
            let bytes_per_elem = match dtype {
                safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
                safetensors::Dtype::F16
                | safetensors::Dtype::BF16
                | safetensors::Dtype::I16
                | safetensors::Dtype::U16 => 2,
                safetensors::Dtype::I8 | safetensors::Dtype::U8 | safetensors::Dtype::BOOL => 1,
                safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
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

    safetensors::serialize(views.iter().map(|(n, v)| (*n, v)), None).unwrap()
}

// ===========================================================================
// QuantizationReport structure tests
// ===========================================================================

#[test]
fn test_quantization_report_default_from_empty() {
    // An empty safetensors file produces a report with zeroed/empty fields.
    let bytes = build_safetensors(&[]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse empty model");

    assert_eq!(report.total_tensors, 0);
    assert_eq!(report.total_parameters, 0);
    assert_eq!(report.total_bytes, 0);
    assert!(report.tensors.is_empty());
    assert!(report.dtype_breakdown.is_empty());
    assert!(report.recommendations.is_empty());
    assert!(!report.is_mixed_precision());
    assert_eq!(report.total_savings_bytes(), 0);
    // dtype_fraction for any dtype on empty report should be 0.0.
    assert_eq!(report.dtype_fraction(DetectedDtype::F32), 0.0);
    assert_eq!(report.dtype_fraction(DetectedDtype::BF16), 0.0);
}

#[test]
fn test_dtype_breakdown_empty_has_no_dtype() {
    let breakdown = DtypeBreakdown {
        dtype: DetectedDtype::F32,
        tensor_count: 0,
        total_parameters: 0,
        total_bytes: 0,
    };
    assert_eq!(breakdown.tensor_count, 0);
    assert_eq!(breakdown.total_parameters, 0);
    assert_eq!(breakdown.total_bytes, 0);
    assert_eq!(breakdown.dtype, DetectedDtype::F32);
}

#[test]
fn test_quant_recommendation_fields() {
    let rec = QuantRecommendation {
        target_dtype: DetectedDtype::F16,
        tensor_names: vec!["w1".to_string(), "w2".to_string()],
        current_bytes: 8192,
        projected_bytes: 4096,
        savings_bytes: 4096,
    };
    assert_eq!(rec.target_dtype, DetectedDtype::F16);
    assert_eq!(rec.tensor_names.len(), 2);
    assert_eq!(rec.current_bytes, 8192);
    assert_eq!(rec.projected_bytes, 4096);
    assert_eq!(rec.savings_bytes, 4096);
}

#[test]
fn test_tensor_quant_info_creation() {
    let info = TensorQuantInfo {
        name: "encoder.weight".to_string(),
        dtype: DetectedDtype::BF16,
        shape: vec![512, 768],
        num_elements: 512 * 768,
        size_bytes: 512 * 768 * 2,
    };
    assert_eq!(info.name, "encoder.weight");
    assert_eq!(info.dtype, DetectedDtype::BF16);
    assert_eq!(info.shape, vec![512, 768]);
    assert_eq!(info.num_elements, 393216);
    assert_eq!(info.size_bytes, 786432);
}

// ===========================================================================
// detect_quantization tests
// ===========================================================================

#[test]
fn test_detect_quantization_empty_safetensors() {
    let bytes = build_safetensors(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_tensors, 0);
    assert!(report.dtype_breakdown.is_empty());
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_detect_quantization_all_f32() {
    let bytes = build_safetensors(&[
        ("weight_a", safetensors::Dtype::F32, &[128, 64]),
        ("weight_b", safetensors::Dtype::F32, &[64, 32]),
        ("bias_a", safetensors::Dtype::F32, &[64]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 3);
    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 3);

    let expected_params = 128 * 64 + 64 * 32 + 64;
    assert_eq!(report.total_parameters, expected_params);
    assert_eq!(report.total_bytes, expected_params * 4);
}

#[test]
fn test_detected_dtype_display_all_variants() {
    // Every DetectedDtype variant should have a meaningful (non-empty) Display output.
    let variants = [
        DetectedDtype::F32,
        DetectedDtype::F16,
        DetectedDtype::BF16,
        DetectedDtype::F64,
        DetectedDtype::I8,
        DetectedDtype::U8,
        DetectedDtype::F8,
        DetectedDtype::SubByte,
        DetectedDtype::I16,
        DetectedDtype::I32,
        DetectedDtype::I64,
        DetectedDtype::Bool,
        DetectedDtype::C64,
        DetectedDtype::Other,
    ];
    for variant in &variants {
        let display = format!("{variant}");
        assert!(
            !display.is_empty(),
            "DetectedDtype::{variant:?} should have non-empty Display"
        );
        // Display should match label().
        assert_eq!(
            display,
            variant.label(),
            "Display and label() disagree for {variant:?}"
        );
    }
}

// ===========================================================================
// detect_quantization_from_bytes edge cases
// ===========================================================================

#[test]
fn test_detect_from_empty_bytes_returns_error() {
    // Completely empty bytes (not even a valid safetensors header) should error.
    let result = detect_quantization_from_bytes(&[]);
    assert!(result.is_err(), "empty bytes should fail to parse");
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("safetensors") || msg.contains("I/O"),
        "error should mention safetensors parse failure: {msg}"
    );
}

#[test]
fn test_detect_from_minimal_safetensors() {
    // A valid safetensors file with a single scalar F32 tensor.
    let bytes = build_safetensors(&[("scalar", safetensors::Dtype::F32, &[1])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.total_parameters, 1);
    assert_eq!(report.total_bytes, 4); // 1 element * 4 bytes
    assert_eq!(report.tensors.len(), 1);
    assert_eq!(report.tensors[0].name, "scalar");
    assert_eq!(report.tensors[0].dtype, DetectedDtype::F32);
    assert_eq!(report.tensors[0].shape, vec![1]);
}

#[test]
fn test_detect_from_garbage_bytes_returns_error() {
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
    let result = detect_quantization_from_bytes(&garbage);
    assert!(result.is_err(), "garbage bytes should fail to parse");
}

#[test]
fn test_detect_bf16_tensors() {
    let bytes = build_safetensors(&[("layer.weight", safetensors::Dtype::BF16, &[256, 256])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::BF16);
    assert_eq!(report.total_parameters, 256 * 256);
    assert_eq!(report.total_bytes, 256 * 256 * 2);
}

#[test]
fn test_detect_i8_tensors() {
    let bytes = build_safetensors(&[("quant.weight", safetensors::Dtype::I8, &[1024, 512])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::I8);
    assert_eq!(report.total_bytes, 1024 * 512); // 1 byte per element
                                                // I8 tensors should have no quantization recommendations.
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_detect_multidtype_model() {
    let bytes = build_safetensors(&[
        ("f32.weight", safetensors::Dtype::F32, &[1024, 1024]),
        ("f16.weight", safetensors::Dtype::F16, &[512, 512]),
        ("i8.weight", safetensors::Dtype::I8, &[128]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 3);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 3);

    // F32 fraction should be the largest.
    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    let f16_frac = report.dtype_fraction(DetectedDtype::F16);
    assert!(f32_frac > f16_frac, "F32 should dominate by bytes");
}

#[test]
fn test_detect_tensors_sorted_by_name() {
    let bytes = build_safetensors(&[
        ("z_layer", safetensors::Dtype::F32, &[10]),
        ("a_layer", safetensors::Dtype::F32, &[10]),
        ("m_layer", safetensors::Dtype::F32, &[10]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["a_layer", "m_layer", "z_layer"]);
}

#[test]
fn test_report_summary_no_recs_contains_compact() {
    let bytes = build_safetensors(&[("small", safetensors::Dtype::F16, &[32])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();
    assert!(
        summary.contains("already compact"),
        "summary should mention 'already compact' when no recs: {summary}"
    );
}

#[test]
fn test_report_display_equals_summary() {
    let bytes = build_safetensors(&[("w", safetensors::Dtype::F32, &[2048, 2048])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let display = format!("{report}");
    assert_eq!(display, report.summary());
}

#[test]
fn test_report_mixed_precision_flag() {
    // Single dtype = not mixed.
    let single = build_safetensors(&[
        ("a", safetensors::Dtype::F32, &[100]),
        ("b", safetensors::Dtype::F32, &[200]),
    ]);
    let r1 = detect_quantization_from_bytes(&single).unwrap();
    assert!(!r1.is_mixed_precision());

    // Two dtypes = mixed.
    let mixed = build_safetensors(&[
        ("a", safetensors::Dtype::F32, &[100]),
        ("b", safetensors::Dtype::F16, &[200]),
    ]);
    let r2 = detect_quantization_from_bytes(&mixed).unwrap();
    assert!(r2.is_mixed_precision());
}

// ===========================================================================
// Kokoro weight validation
// ===========================================================================

#[test]
fn test_kokoro_name_mapping_not_empty() {
    let mapper = crate::kokoro_weights::kokoro_name_mapping();
    // The mapping closure should produce output for known keys.
    let result = mapper("plbert.embeddings.word_embeddings.weight");
    assert!(
        !result.is_empty(),
        "kokoro_name_mapping should return non-empty string for known keys"
    );
}

#[test]
fn test_map_pytorch_key_known_keys() {
    use crate::kokoro_weights::map_pytorch_key;

    // All expected prefixes should map.
    let known_keys = [
        "plbert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.F0.0.n1.fc.weight",
        "decoder.conv_pre.weight",
    ];
    for key in &known_keys {
        let mapped = map_pytorch_key(key);
        assert!(mapped.is_some(), "key '{key}' should map successfully");
        // Identity mapping expected.
        assert_eq!(
            mapped.as_deref(),
            Some(*key),
            "key '{key}' should be identity-mapped"
        );
    }

    // Unknown keys should return None.
    assert_eq!(map_pytorch_key("totally_unknown.weight"), None);
    assert_eq!(map_pytorch_key(""), None);
}

#[test]
fn test_validate_kokoro_keys_empty() {
    use crate::kokoro_weights::validate_kokoro_keys;

    let empty_keys: Vec<&str> = vec![];
    let missing = validate_kokoro_keys(&empty_keys);
    // All 6 expected prefixes should be missing.
    assert_eq!(
        missing.len(),
        6,
        "empty key set should be missing all 6 required prefixes, got {missing:?}"
    );
}

#[test]
fn test_validate_kokoro_keys_partial() {
    use crate::kokoro_weights::validate_kokoro_keys;

    // Provide only 2 out of 6 required prefixes.
    let keys = vec!["plbert.something", "decoder.something"];
    let missing = validate_kokoro_keys(&keys);
    assert_eq!(
        missing.len(),
        4,
        "should be missing 4 prefixes, got {missing:?}"
    );
    assert!(missing.contains(&"bert_encoder."));
    assert!(missing.contains(&"text_encoder."));
    assert!(missing.contains(&"prosody_predictor."));
    assert!(missing.contains(&"predictor."));
}

#[test]
fn test_validate_kokoro_safetensors_error_on_empty() {
    use crate::kokoro_weights::validate_kokoro_safetensors;

    let keys: Vec<String> = vec![];
    let result = validate_kokoro_safetensors(&keys);
    assert!(result.is_err(), "empty key set should fail validation");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("missing Kokoro weight groups"),
        "error should mention missing weight groups: {err_msg}"
    );
}

// ===========================================================================
// DetectedDtype additional coverage
// ===========================================================================

#[test]
fn test_detected_dtype_ord_consistency() {
    // PartialOrd/Ord should be consistent with Eq.
    let a = DetectedDtype::F32;
    let b = DetectedDtype::F16;
    // Different variants should not be equal.
    assert_ne!(a, b);
    // Same variant should be equal.
    assert_eq!(a, DetectedDtype::F32);
}

#[test]
fn test_detected_dtype_hash_works() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(DetectedDtype::F32);
    set.insert(DetectedDtype::F16);
    set.insert(DetectedDtype::F32); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_detected_dtype_clone_copy() {
    let original = DetectedDtype::BF16;
    let cloned = original;
    let copied = original; // Copy
    assert_eq!(original, cloned);
    assert_eq!(original, copied);
}

#[test]
fn test_detected_dtype_debug() {
    let dbg = format!("{:?}", DetectedDtype::C64);
    assert!(
        dbg.contains("C64"),
        "Debug should contain variant name: {dbg}"
    );
}

#[test]
fn test_detected_dtype_bytes_per_element_c64() {
    // C64 = 8 bytes (32-bit real + 32-bit imag)
    assert_eq!(DetectedDtype::C64.bytes_per_element(), Some(8));
}

#[test]
fn test_detected_dtype_bytes_per_element_f8() {
    assert_eq!(DetectedDtype::F8.bytes_per_element(), Some(1));
}

// ===========================================================================
// QuantizationReport method coverage
// ===========================================================================

#[test]
fn test_dtype_fraction_nonexistent_dtype() {
    // Report with only F32 tensors: fraction of BF16 should be 0.0.
    let bytes = build_safetensors(&[("w", safetensors::Dtype::F32, &[1024])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.dtype_fraction(DetectedDtype::BF16), 0.0);
    assert_eq!(report.dtype_fraction(DetectedDtype::I8), 0.0);
}

#[test]
fn test_dtype_fraction_sums_to_one() {
    let bytes = build_safetensors(&[
        ("f32", safetensors::Dtype::F32, &[1000]),
        ("f16", safetensors::Dtype::F16, &[2000]),
        ("i8", safetensors::Dtype::I8, &[500]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let sum = report.dtype_fraction(DetectedDtype::F32)
        + report.dtype_fraction(DetectedDtype::F16)
        + report.dtype_fraction(DetectedDtype::I8);
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "dtype fractions should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_total_savings_consistent_with_recommendations() {
    let bytes = build_safetensors(&[("w", safetensors::Dtype::F32, &[2048, 2048])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let manual_sum: usize = report.recommendations.iter().map(|r| r.savings_bytes).sum();
    assert_eq!(report.total_savings_bytes(), manual_sum);
    // Large F32 tensor should have at least F16 and I8 recommendations.
    assert!(
        report.recommendations.len() >= 2,
        "large F32 model should have F16 + I8 recs, got {}",
        report.recommendations.len()
    );
}

#[test]
fn test_f32_below_threshold_no_recommendations() {
    // F32 tensor with < 1024 elements should not trigger recommendations.
    let bytes = build_safetensors(&[
        ("small", safetensors::Dtype::F32, &[31, 32]), // 992 elements < 1024
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(
        report.recommendations.is_empty(),
        "F32 tensor below 1024 elements should get no recommendations"
    );
}

#[test]
fn test_f32_at_threshold_gets_recommendations() {
    // F32 tensor with exactly 1024 elements should trigger recommendations.
    let bytes = build_safetensors(&[
        ("exact", safetensors::Dtype::F32, &[32, 32]), // 1024 elements
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(
        !report.recommendations.is_empty(),
        "F32 tensor at exactly 1024 elements should get recommendations"
    );
}

#[test]
fn test_summary_contains_tensor_count_and_size() {
    let bytes = build_safetensors(&[
        ("layer1.weight", safetensors::Dtype::F32, &[1024, 1024]),
        ("layer2.weight", safetensors::Dtype::BF16, &[512, 256]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();

    assert!(
        summary.contains("2 tensors"),
        "summary should mention '2 tensors': {summary}"
    );
    assert!(
        summary.contains("Dtype Breakdown"),
        "summary should have 'Dtype Breakdown'"
    );
    assert!(summary.contains("F32"), "summary should mention F32");
    assert!(summary.contains("BF16"), "summary should mention BF16");
}
