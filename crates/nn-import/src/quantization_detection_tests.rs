// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended quantization detection tests for nn-import.
//!
//! Covers DtypeBreakdown computation, DetectedDtype classification,
//! QuantRecommendation generation, TensorQuantInfo validation,
//! QuantizationReport formatting, and detect_quantization_from_bytes
//! edge cases.

use crate::quantization::{detect_quantization_from_bytes, DetectedDtype, TensorQuantInfo};

// ---------------------------------------------------------------------------
// Helper: build safetensors bytes for testing
// ---------------------------------------------------------------------------

/// Build a minimal safetensors blob from typed tensor specs.
/// Each entry is (name, dtype, shape). Data is zero-filled to the correct size.
fn build_st(tensors: &[(&str, safetensors::Dtype, &[usize])]) -> Vec<u8> {
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
                safetensors::Dtype::F64
                | safetensors::Dtype::I64
                | safetensors::Dtype::U64
                | safetensors::Dtype::C64 => 8,
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
// 1. DtypeBreakdown computation from tensor lists
// ===========================================================================

#[test]
fn test_dtype_breakdown_single_dtype_aggregation() {
    // Three F32 tensors should produce a single DtypeBreakdown entry.
    let bytes = build_st(&[
        ("a", safetensors::Dtype::F32, &[100]),
        ("b", safetensors::Dtype::F32, &[200]),
        ("c", safetensors::Dtype::F32, &[300]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_breakdown.len(), 1);
    let bd = &report.dtype_breakdown[0];
    assert_eq!(bd.dtype, DetectedDtype::F32);
    assert_eq!(bd.tensor_count, 3);
    assert_eq!(bd.total_parameters, 600);
    assert_eq!(bd.total_bytes, 600 * 4);
}

#[test]
fn test_dtype_breakdown_multi_dtype_separate_buckets() {
    let bytes = build_st(&[
        ("f32_w", safetensors::Dtype::F32, &[50]),
        ("f16_w", safetensors::Dtype::F16, &[80]),
        ("bf16_w", safetensors::Dtype::BF16, &[120]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_breakdown.len(), 3);
    // Breakdown is sorted by BTreeMap key (DetectedDtype Ord).
    for bd in &report.dtype_breakdown {
        match bd.dtype {
            DetectedDtype::F32 => {
                assert_eq!(bd.tensor_count, 1);
                assert_eq!(bd.total_parameters, 50);
                assert_eq!(bd.total_bytes, 200);
            }
            DetectedDtype::F16 => {
                assert_eq!(bd.tensor_count, 1);
                assert_eq!(bd.total_parameters, 80);
                assert_eq!(bd.total_bytes, 160);
            }
            DetectedDtype::BF16 => {
                assert_eq!(bd.tensor_count, 1);
                assert_eq!(bd.total_parameters, 120);
                assert_eq!(bd.total_bytes, 240);
            }
            _ => panic!("unexpected dtype in breakdown: {:?}", bd.dtype),
        }
    }
}

#[test]
fn test_dtype_breakdown_parameters_sum_matches_total() {
    let bytes = build_st(&[
        ("a", safetensors::Dtype::F32, &[64, 64]),
        ("b", safetensors::Dtype::F16, &[32, 32]),
        ("c", safetensors::Dtype::I8, &[16]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let sum_params: usize = report
        .dtype_breakdown
        .iter()
        .map(|b| b.total_parameters)
        .sum();
    assert_eq!(sum_params, report.total_parameters);

    let sum_bytes: usize = report.dtype_breakdown.iter().map(|b| b.total_bytes).sum();
    assert_eq!(sum_bytes, report.total_bytes);

    let sum_count: usize = report.dtype_breakdown.iter().map(|b| b.tensor_count).sum();
    assert_eq!(sum_count, report.total_tensors);
}

#[test]
fn test_dtype_breakdown_i8_and_u8_separate() {
    // I8 and U8 should produce separate breakdown entries.
    let bytes = build_st(&[
        ("signed", safetensors::Dtype::I8, &[100]),
        ("unsigned", safetensors::Dtype::U8, &[200]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_breakdown.len(), 2);
    let dtypes: Vec<DetectedDtype> = report.dtype_breakdown.iter().map(|b| b.dtype).collect();
    assert!(dtypes.contains(&DetectedDtype::I8));
    assert!(dtypes.contains(&DetectedDtype::U8));
}

#[test]
fn test_dtype_breakdown_i16_u16_merge_to_i16() {
    // Both I16 and U16 map to DetectedDtype::I16, so they merge into one bucket.
    let bytes = build_st(&[
        ("signed16", safetensors::Dtype::I16, &[50]),
        ("unsigned16", safetensors::Dtype::U16, &[50]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::I16);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 2);
    assert_eq!(report.dtype_breakdown[0].total_parameters, 100);
}

// ===========================================================================
// 2. DetectedDtype classification
// ===========================================================================

#[test]
fn test_detected_dtype_from_safetensors_f32() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F32),
        DetectedDtype::F32
    );
}

#[test]
fn test_detected_dtype_from_safetensors_f16() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F16),
        DetectedDtype::F16
    );
}

#[test]
fn test_detected_dtype_from_safetensors_bf16() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::BF16),
        DetectedDtype::BF16
    );
}

#[test]
fn test_detected_dtype_from_safetensors_f64() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F64),
        DetectedDtype::F64
    );
}

#[test]
fn test_detected_dtype_from_safetensors_i8() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::I8),
        DetectedDtype::I8
    );
}

#[test]
fn test_detected_dtype_from_safetensors_u8() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::U8),
        DetectedDtype::U8
    );
}

#[test]
fn test_detected_dtype_from_safetensors_bool() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::BOOL),
        DetectedDtype::Bool
    );
}

#[test]
fn test_detected_dtype_from_safetensors_i64() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::I64),
        DetectedDtype::I64
    );
}

#[test]
fn test_detected_dtype_from_safetensors_u64() {
    // U64 maps to I64 bucket.
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::U64),
        DetectedDtype::I64
    );
}

#[test]
fn test_detected_dtype_from_safetensors_i32() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::I32),
        DetectedDtype::I32
    );
}

#[test]
fn test_detected_dtype_from_safetensors_u32() {
    // U32 maps to I32 bucket.
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::U32),
        DetectedDtype::I32
    );
}

#[test]
fn test_detected_dtype_from_safetensors_c64() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::C64),
        DetectedDtype::C64
    );
}

#[test]
fn test_detected_dtype_bytes_per_element_all_known() {
    // Verify bytes_per_element for all dtypes that have known sizes.
    assert_eq!(DetectedDtype::F32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::F16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::BF16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::F64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::I8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::U8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::F8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::I16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::I32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::I64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::Bool.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::C64.bytes_per_element(), Some(8));
}

#[test]
fn test_detected_dtype_bytes_per_element_unknown_returns_none() {
    assert_eq!(DetectedDtype::SubByte.bytes_per_element(), None);
    assert_eq!(DetectedDtype::Other.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_label_roundtrip() {
    // Every variant's label() should be non-empty and match Display.
    let all = [
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
    for dt in &all {
        let label = dt.label();
        let display = format!("{dt}");
        assert!(!label.is_empty());
        assert_eq!(label, display.as_str());
    }
}

// ===========================================================================
// 3. QuantRecommendation generation
// ===========================================================================

#[test]
fn test_recommendations_f32_large_yields_f16_and_i8() {
    let bytes = build_st(&[("w", safetensors::Dtype::F32, &[2048])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.recommendations.len(), 2);

    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16)
        .expect("should have F16 recommendation");
    assert_eq!(f16_rec.tensor_names, vec!["w"]);
    assert_eq!(f16_rec.current_bytes, 2048 * 4);
    assert_eq!(f16_rec.projected_bytes, 2048 * 4 / 2);
    assert_eq!(f16_rec.savings_bytes, 2048 * 4 / 2);

    let i8_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::I8)
        .expect("should have I8 recommendation");
    assert_eq!(i8_rec.projected_bytes, 2048 * 4 / 4);
    assert_eq!(i8_rec.savings_bytes, 2048 * 4 * 3 / 4);
}

#[test]
fn test_recommendations_f32_small_no_recs() {
    // Tensor with 512 elements (< 1024 threshold) should not produce recs.
    let bytes = build_st(&[("small", safetensors::Dtype::F32, &[16, 32])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_recommendations_f32_exactly_1024_elements() {
    let bytes = build_st(&[("exact", safetensors::Dtype::F32, &[1024])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.recommendations.len(), 2);
}

#[test]
fn test_recommendations_f64_yields_f32() {
    let bytes = build_st(&[("d", safetensors::Dtype::F64, &[256])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.recommendations.len(), 1);
    let rec = &report.recommendations[0];
    assert_eq!(rec.target_dtype, DetectedDtype::F32);
    assert_eq!(rec.current_bytes, 256 * 8);
    assert_eq!(rec.projected_bytes, 256 * 8 / 2);
    assert_eq!(rec.savings_bytes, 256 * 8 / 2);
}

#[test]
fn test_recommendations_f16_no_recs() {
    // F16 tensors should not produce any recommendation.
    let bytes = build_st(&[("w", safetensors::Dtype::F16, &[4096])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_recommendations_bf16_no_recs() {
    let bytes = build_st(&[("w", safetensors::Dtype::BF16, &[4096])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_recommendations_i8_no_recs() {
    let bytes = build_st(&[("q", safetensors::Dtype::I8, &[4096])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_recommendations_mixed_f32_and_f16_only_f32_recs() {
    let bytes = build_st(&[
        ("f32_big", safetensors::Dtype::F32, &[2048]),
        ("f16_big", safetensors::Dtype::F16, &[4096]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    // Only F32 tensor should produce recommendations.
    for rec in &report.recommendations {
        assert!(
            rec.tensor_names.contains(&"f32_big".to_string()),
            "recommendations should only reference F32 tensors"
        );
    }
}

#[test]
fn test_recommendations_multiple_f32_tensors_grouped() {
    let bytes = build_st(&[
        ("w1", safetensors::Dtype::F32, &[2048]),
        ("w2", safetensors::Dtype::F32, &[4096]),
        ("bias", safetensors::Dtype::F32, &[8]), // below threshold
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16)
        .unwrap();
    // Only w1 and w2 should be recommended (bias has < 1024 elements).
    assert_eq!(f16_rec.tensor_names.len(), 2);
    assert!(f16_rec.tensor_names.contains(&"w1".to_string()));
    assert!(f16_rec.tensor_names.contains(&"w2".to_string()));
    assert!(!f16_rec.tensor_names.contains(&"bias".to_string()));
}

#[test]
fn test_recommendations_f32_and_f64_both_generate_recs() {
    let bytes = build_st(&[
        ("f32_w", safetensors::Dtype::F32, &[2048]),
        ("f64_w", safetensors::Dtype::F64, &[512]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    // F32 -> F16 rec, F32 -> I8 rec, F64 -> F32 rec = 3 recs.
    assert_eq!(report.recommendations.len(), 3);
    let targets: Vec<DetectedDtype> = report
        .recommendations
        .iter()
        .map(|r| r.target_dtype)
        .collect();
    assert!(targets.contains(&DetectedDtype::F16));
    assert!(targets.contains(&DetectedDtype::I8));
    assert!(targets.contains(&DetectedDtype::F32));
}

// ===========================================================================
// 4. TensorQuantInfo construction and validation
// ===========================================================================

#[test]
fn test_tensor_quant_info_scalar_shape() {
    let info = TensorQuantInfo {
        name: "scalar_param".to_string(),
        dtype: DetectedDtype::F32,
        shape: vec![1],
        num_elements: 1,
        size_bytes: 4,
    };
    assert_eq!(info.shape.len(), 1);
    assert_eq!(info.num_elements, 1);
    assert_eq!(info.size_bytes, 4);
}

#[test]
fn test_tensor_quant_info_multidim_shape() {
    let info = TensorQuantInfo {
        name: "conv.weight".to_string(),
        dtype: DetectedDtype::F16,
        shape: vec![64, 3, 7, 7],
        num_elements: 64 * 3 * 7 * 7,
        size_bytes: 64 * 3 * 7 * 7 * 2,
    };
    assert_eq!(info.shape, vec![64, 3, 7, 7]);
    assert_eq!(info.num_elements, 9408);
    assert_eq!(info.size_bytes, 18816);
}

#[test]
fn test_tensor_quant_info_clone() {
    let info = TensorQuantInfo {
        name: "test".to_string(),
        dtype: DetectedDtype::BF16,
        shape: vec![10, 20],
        num_elements: 200,
        size_bytes: 400,
    };
    let cloned = info.clone();
    assert_eq!(cloned.name, info.name);
    assert_eq!(cloned.dtype, info.dtype);
    assert_eq!(cloned.shape, info.shape);
    assert_eq!(cloned.num_elements, info.num_elements);
    assert_eq!(cloned.size_bytes, info.size_bytes);
}

#[test]
fn test_tensor_quant_info_debug() {
    let info = TensorQuantInfo {
        name: "layer.weight".to_string(),
        dtype: DetectedDtype::F32,
        shape: vec![128],
        num_elements: 128,
        size_bytes: 512,
    };
    let dbg = format!("{info:?}");
    assert!(dbg.contains("layer.weight"));
    assert!(dbg.contains("F32"));
}

// ===========================================================================
// 5. QuantizationReport formatting
// ===========================================================================

#[test]
fn test_report_summary_no_recommendations_compact_message() {
    let bytes = build_st(&[("small", safetensors::Dtype::F16, &[32])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();
    assert!(summary.contains("already compact"));
    assert!(summary.contains("1 tensors"));
}

#[test]
fn test_report_summary_with_recommendations_shows_savings() {
    let bytes = build_st(&[("big", safetensors::Dtype::F32, &[1024, 1024])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();

    assert!(summary.contains("Recommendations:"));
    assert!(summary.contains("Quantize"));
    assert!(summary.contains("F16"));
    assert!(summary.contains("Total potential savings"));
}

#[test]
fn test_report_summary_shows_dtype_breakdown_percentages() {
    let bytes = build_st(&[
        ("f32", safetensors::Dtype::F32, &[1000]),
        ("f16", safetensors::Dtype::F16, &[1000]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();

    // Should contain percentage values.
    assert!(summary.contains('%'), "summary should contain percentages");
    assert!(summary.contains("F32"));
    assert!(summary.contains("F16"));
}

#[test]
fn test_report_display_matches_summary() {
    let bytes = build_st(&[
        ("w1", safetensors::Dtype::F32, &[4096]),
        ("w2", safetensors::Dtype::BF16, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(format!("{report}"), report.summary());
}

#[test]
fn test_report_summary_empty_model() {
    let bytes = build_st(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();
    assert!(summary.contains("0 tensors"));
    assert!(summary.contains("0 parameters"));
    assert!(summary.contains("already compact"));
}

// ===========================================================================
// 6. detect_quantization_from_bytes with various dtype distributions
// ===========================================================================

#[test]
fn test_detect_pure_f32_model() {
    let bytes = build_st(&[
        ("w1", safetensors::Dtype::F32, &[1024]),
        ("w2", safetensors::Dtype::F32, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    assert_eq!(report.dtype_fraction(DetectedDtype::F32), 1.0);
}

#[test]
fn test_detect_pure_f16_model() {
    let bytes = build_st(&[
        ("w1", safetensors::Dtype::F16, &[4096]),
        ("w2", safetensors::Dtype::F16, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F16);
    assert_eq!(report.dtype_fraction(DetectedDtype::F16), 1.0);
    // No recommendations for F16 model.
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_detect_pure_bf16_model() {
    let bytes = build_st(&[("w", safetensors::Dtype::BF16, &[512, 512])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(!report.is_mixed_precision());
    assert_eq!(report.dtype_fraction(DetectedDtype::BF16), 1.0);
}

#[test]
fn test_detect_mixed_f32_bf16_model() {
    let bytes = build_st(&[
        ("encoder.weight", safetensors::Dtype::F32, &[1024, 768]),
        ("encoder.bias", safetensors::Dtype::F32, &[768]),
        ("decoder.weight", safetensors::Dtype::BF16, &[768, 512]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);

    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    let bf16_frac = report.dtype_fraction(DetectedDtype::BF16);
    assert!((f32_frac + bf16_frac - 1.0).abs() < 1e-10);
}

#[test]
fn test_detect_f32_f16_i8_three_way_mixed() {
    let bytes = build_st(&[
        ("float32", safetensors::Dtype::F32, &[2048]),
        ("float16", safetensors::Dtype::F16, &[2048]),
        ("quantized", safetensors::Dtype::I8, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 3);
    assert_eq!(report.total_tensors, 3);

    // F32 should be the largest by bytes.
    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    let f16_frac = report.dtype_fraction(DetectedDtype::F16);
    let i8_frac = report.dtype_fraction(DetectedDtype::I8);
    assert!(f32_frac > f16_frac);
    assert!(f16_frac > i8_frac);
}

#[test]
fn test_detect_all_integer_model() {
    let bytes = build_st(&[
        ("ids", safetensors::Dtype::I32, &[1024]),
        ("labels", safetensors::Dtype::I64, &[512]),
        ("mask", safetensors::Dtype::U8, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(report.is_mixed_precision());
    assert_eq!(report.total_tensors, 3);
    // No quantization recommendations for integer-only models.
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_detect_bool_tensors() {
    let bytes = build_st(&[("mask", safetensors::Dtype::BOOL, &[256, 256])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::Bool);
    assert_eq!(report.total_bytes, 256 * 256); // 1 byte per bool
}

// ===========================================================================
// 7. detect_quantization_from_bytes with safetensors byte arrays
// ===========================================================================

#[test]
fn test_detect_from_bytes_invalid_empty() {
    let result = detect_quantization_from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_detect_from_bytes_invalid_garbage() {
    let garbage = vec![0xFF; 128];
    let result = detect_quantization_from_bytes(&garbage);
    assert!(result.is_err());
}

#[test]
fn test_detect_from_bytes_truncated_header() {
    // A truncated safetensors file (just the header length prefix).
    let truncated = vec![0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = detect_quantization_from_bytes(&truncated);
    assert!(result.is_err());
}

#[test]
fn test_detect_from_bytes_valid_roundtrip() {
    let bytes = build_st(&[
        ("layer.weight", safetensors::Dtype::F32, &[64, 64]),
        ("layer.bias", safetensors::Dtype::F32, &[64]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 2);
    assert_eq!(report.total_parameters, 64 * 64 + 64);
    assert_eq!(report.total_bytes, (64 * 64 + 64) * 4);
}

// ===========================================================================
// 8. Edge cases: empty tensors, single tensor, all same dtype
// ===========================================================================

#[test]
fn test_edge_case_empty_safetensors() {
    let bytes = build_st(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 0);
    assert_eq!(report.total_parameters, 0);
    assert_eq!(report.total_bytes, 0);
    assert!(report.dtype_breakdown.is_empty());
    assert!(report.recommendations.is_empty());
    assert!(!report.is_mixed_precision());
    assert_eq!(report.total_savings_bytes(), 0);
    assert_eq!(report.dtype_fraction(DetectedDtype::F32), 0.0);
}

#[test]
fn test_edge_case_single_scalar_tensor() {
    let bytes = build_st(&[("scale", safetensors::Dtype::F32, &[1])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.total_parameters, 1);
    assert_eq!(report.total_bytes, 4);
    assert_eq!(report.tensors[0].name, "scale");
    // Single scalar should not trigger quantization.
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_edge_case_single_large_tensor() {
    let bytes = build_st(&[("embedding", safetensors::Dtype::F32, &[50000, 768])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.total_parameters, 50000 * 768);
    // Should produce F16 and I8 recommendations.
    assert_eq!(report.recommendations.len(), 2);
}

#[test]
fn test_edge_case_many_small_tensors_no_recs() {
    // 20 small F32 tensors, each below the 1024-element threshold.
    let specs: Vec<(&str, safetensors::Dtype, &[usize])> = (0..20)
        .map(|i| {
            // We leak strings for convenience in this test.
            let name: &'static str = Box::leak(format!("param_{i}").into_boxed_str());
            (name, safetensors::Dtype::F32, [16usize, 16].as_slice())
        })
        .collect();
    let bytes = build_st(&specs);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 20);
    // 256 elements per tensor < 1024, so no recommendations.
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_edge_case_tensors_sorted_by_name() {
    let bytes = build_st(&[
        ("z.weight", safetensors::Dtype::F32, &[10]),
        ("a.weight", safetensors::Dtype::F32, &[10]),
        ("m.weight", safetensors::Dtype::F32, &[10]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["a.weight", "m.weight", "z.weight"]);
}

#[test]
fn test_edge_case_dtype_fraction_all_one_type() {
    let bytes = build_st(&[
        ("a", safetensors::Dtype::BF16, &[100]),
        ("b", safetensors::Dtype::BF16, &[200]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.dtype_fraction(DetectedDtype::BF16), 1.0);
    assert_eq!(report.dtype_fraction(DetectedDtype::F32), 0.0);
}

#[test]
fn test_edge_case_total_savings_with_no_recs() {
    let bytes = build_st(&[("tiny", safetensors::Dtype::I8, &[100])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_savings_bytes(), 0);
}

// ===========================================================================
// 9. Additional QuantizationReport method coverage
// ===========================================================================

#[test]
fn test_is_mixed_precision_one_dtype() {
    let bytes = build_st(&[
        ("a", safetensors::Dtype::F32, &[100]),
        ("b", safetensors::Dtype::F32, &[200]),
        ("c", safetensors::Dtype::F32, &[300]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(!report.is_mixed_precision());
}

#[test]
fn test_is_mixed_precision_two_dtypes() {
    let bytes = build_st(&[
        ("a", safetensors::Dtype::F32, &[100]),
        ("b", safetensors::Dtype::F16, &[100]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.is_mixed_precision());
}

#[test]
fn test_dtype_fraction_precise_values() {
    // 1000 F32 elements = 4000 bytes, 1000 F16 elements = 2000 bytes
    // Total = 6000 bytes
    let bytes = build_st(&[
        ("f32", safetensors::Dtype::F32, &[1000]),
        ("f16", safetensors::Dtype::F16, &[1000]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    let f16_frac = report.dtype_fraction(DetectedDtype::F16);

    // F32: 4000/6000 = 2/3
    assert!((f32_frac - 2.0 / 3.0).abs() < 1e-10);
    // F16: 2000/6000 = 1/3
    assert!((f16_frac - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_recommendation_savings_arithmetic() {
    // F32 tensor with 4096 elements = 16384 bytes
    // F16 projected = 8192, savings = 8192
    // I8 projected = 4096, savings = 12288
    let bytes = build_st(&[("w", safetensors::Dtype::F32, &[4096])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    for rec in &report.recommendations {
        assert_eq!(
            rec.savings_bytes,
            rec.current_bytes - rec.projected_bytes,
            "savings must equal current - projected for {:?}",
            rec.target_dtype
        );
    }

    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16)
        .unwrap();
    assert_eq!(f16_rec.current_bytes, 16384);
    assert_eq!(f16_rec.projected_bytes, 8192);
    assert_eq!(f16_rec.savings_bytes, 8192);

    let i8_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::I8)
        .unwrap();
    assert_eq!(i8_rec.current_bytes, 16384);
    assert_eq!(i8_rec.projected_bytes, 4096);
    assert_eq!(i8_rec.savings_bytes, 12288);
}
