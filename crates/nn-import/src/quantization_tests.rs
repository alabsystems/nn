// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the quantization detection module.

use super::*;

/// Helper: build safetensors bytes from typed tensor specs.
fn build_safetensors_typed(tensors: &[(&str, &[usize], &[u8], safetensors::Dtype)]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, shape, data, dtype) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

#[test]
fn test_detect_empty_model() {
    // An empty safetensors file (no tensors).
    let bytes = build_safetensors_typed(&[]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse empty model");

    assert_eq!(report.total_tensors, 0);
    assert_eq!(report.total_parameters, 0);
    assert_eq!(report.total_bytes, 0);
    assert!(report.dtype_breakdown.is_empty());
    assert!(report.recommendations.is_empty());
    assert!(!report.is_mixed_precision());
}

#[test]
fn test_detect_pure_f32_model() {
    // 2 F32 tensors: a weight [4, 4] = 16 elements = 64 bytes,
    // and a bias [4] = 4 elements = 16 bytes.
    let w_data: Vec<u8> = vec![0u8; 64]; // 16 * 4 bytes
    let b_data: Vec<u8> = vec![0u8; 16]; // 4 * 4 bytes

    let bytes = build_safetensors_typed(&[
        ("layer.weight", &[4, 4], &w_data, safetensors::Dtype::F32),
        ("layer.bias", &[4], &b_data, safetensors::Dtype::F32),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");

    assert_eq!(report.total_tensors, 2);
    assert_eq!(report.total_parameters, 20);
    assert_eq!(report.total_bytes, 80);
    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::F32);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 2);
    assert!(!report.is_mixed_precision());
}

#[test]
fn test_detect_mixed_precision_f32_f16() {
    // F32 weight [8, 8] = 64 elements = 256 bytes.
    let f32_data: Vec<u8> = vec![0u8; 256];
    // F16 weight [4, 4] = 16 elements = 32 bytes.
    let f16_data: Vec<u8> = vec![0u8; 32];

    let bytes = build_safetensors_typed(&[
        (
            "encoder.weight",
            &[8, 8],
            &f32_data,
            safetensors::Dtype::F32,
        ),
        (
            "decoder.weight",
            &[4, 4],
            &f16_data,
            safetensors::Dtype::F16,
        ),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");

    assert_eq!(report.total_tensors, 2);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 2);

    // Check dtype fractions.
    let f32_frac = report.dtype_fraction(DetectedDtype::F32);
    let f16_frac = report.dtype_fraction(DetectedDtype::F16);
    assert!(f32_frac > 0.8, "F32 should be majority: {f32_frac}");
    assert!(f16_frac > 0.0 && f16_frac < 0.2, "F16 fraction: {f16_frac}");
    assert!((f32_frac + f16_frac - 1.0).abs() < 1e-10);
}

#[test]
fn test_detect_mixed_f32_bf16_i8() {
    // 3 dtypes: F32 [32, 32] = 1024 elems, BF16 [16, 16] = 256 elems, I8 [8] = 8 elems.
    let f32_data: Vec<u8> = vec![0u8; 1024 * 4];
    let bf16_data: Vec<u8> = vec![0u8; 256 * 2];
    let i8_data: Vec<u8> = vec![0u8; 8];

    let bytes = build_safetensors_typed(&[
        ("big.weight", &[32, 32], &f32_data, safetensors::Dtype::F32),
        (
            "med.weight",
            &[16, 16],
            &bf16_data,
            safetensors::Dtype::BF16,
        ),
        ("small.scale", &[8], &i8_data, safetensors::Dtype::I8),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");

    assert_eq!(report.total_tensors, 3);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 3);
    assert_eq!(report.total_parameters, 1024 + 256 + 8);
    assert_eq!(report.total_bytes, 1024 * 4 + 256 * 2 + 8);
}

#[test]
fn test_recommendations_f32_large_tensors() {
    // A large F32 tensor (>= 1024 elements) should get F16 and I8 recommendations.
    let f32_data: Vec<u8> = vec![0u8; 2048 * 4]; // 2048 elements
    let bytes =
        build_safetensors_typed(&[("big.weight", &[64, 32], &f32_data, safetensors::Dtype::F32)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");

    // Should have 2 recommendations: F16 and I8.
    assert_eq!(report.recommendations.len(), 2);

    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16);
    let i8_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::I8);

    let f16_rec = f16_rec.expect("should have F16 recommendation");
    assert_eq!(f16_rec.current_bytes, 2048 * 4);
    assert_eq!(f16_rec.projected_bytes, 2048 * 2);
    assert_eq!(f16_rec.savings_bytes, 2048 * 2);
    assert_eq!(f16_rec.tensor_names, vec!["big.weight"]);

    let i8_rec = i8_rec.expect("should have I8 recommendation");
    assert_eq!(i8_rec.projected_bytes, 2048);
    assert_eq!(i8_rec.savings_bytes, 2048 * 3);
}

#[test]
fn test_no_recommendations_for_small_f32() {
    // A small F32 tensor (< 1024 elements) should NOT get recommendations.
    let f32_data: Vec<u8> = vec![0u8; 512 * 4]; // 512 elements
    let bytes =
        build_safetensors_typed(&[("small.bias", &[512], &f32_data, safetensors::Dtype::F32)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert!(
        report.recommendations.is_empty(),
        "small tensors should not get recommendations"
    );
}

#[test]
fn test_no_recommendations_for_already_quantized() {
    // I8 tensors should not get further quantization recommendations.
    let i8_data: Vec<u8> = vec![0u8; 4096];
    let bytes =
        build_safetensors_typed(&[("quant.weight", &[64, 64], &i8_data, safetensors::Dtype::I8)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_f64_recommendation() {
    // F64 tensors should get F32 recommendation.
    let f64_data: Vec<u8> = vec![0u8; 16 * 8]; // 16 elements * 8 bytes
    let bytes = build_safetensors_typed(&[(
        "precision.weight",
        &[4, 4],
        &f64_data,
        safetensors::Dtype::F64,
    )]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.recommendations.len(), 1);

    let rec = &report.recommendations[0];
    assert_eq!(rec.target_dtype, DetectedDtype::F32);
    assert_eq!(rec.current_bytes, 128);
    assert_eq!(rec.projected_bytes, 64);
    assert_eq!(rec.savings_bytes, 64);
}

#[test]
fn test_total_savings() {
    // Large F32 model: total_savings should sum both recs.
    let f32_data: Vec<u8> = vec![0u8; 4096 * 4];
    let bytes = build_safetensors_typed(&[(
        "layer.weight",
        &[64, 64],
        &f32_data,
        safetensors::Dtype::F32,
    )]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    // F16 saves 50%, I8 saves 75%. total_savings sums both.
    let total = report.total_savings_bytes();
    assert_eq!(total, 4096 * 4 / 2 + 4096 * 4 * 3 / 4);
}

#[test]
fn test_report_formatting() {
    let f32_data: Vec<u8> = vec![0u8; 2048 * 4];
    let f16_data: Vec<u8> = vec![0u8; 1024 * 2];

    let bytes = build_safetensors_typed(&[
        ("enc.weight", &[64, 32], &f32_data, safetensors::Dtype::F32),
        ("dec.weight", &[32, 32], &f16_data, safetensors::Dtype::F16),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    let summary = report.summary();

    assert!(summary.contains("Quantization Report:"), "missing header");
    assert!(summary.contains("2 tensors"), "missing tensor count");
    assert!(
        summary.contains("Dtype Breakdown:"),
        "missing breakdown header"
    );
    assert!(summary.contains("F32"), "missing F32 in breakdown");
    assert!(summary.contains("F16"), "missing F16 in breakdown");
    assert!(
        summary.contains("Recommendations:"),
        "missing recommendations header"
    );

    // Display impl should produce the same as summary.
    let display = format!("{report}");
    assert_eq!(display, summary);
}

#[test]
fn test_dtype_fraction_empty() {
    let bytes = build_safetensors_typed(&[]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.dtype_fraction(DetectedDtype::F32), 0.0);
}

#[test]
fn test_detected_dtype_labels() {
    assert_eq!(DetectedDtype::F32.label(), "F32");
    assert_eq!(DetectedDtype::F16.label(), "F16");
    assert_eq!(DetectedDtype::BF16.label(), "BF16");
    assert_eq!(DetectedDtype::I8.label(), "I8");
    assert_eq!(DetectedDtype::SubByte.label(), "SubByte");
    assert_eq!(format!("{}", DetectedDtype::F32), "F32");
}

#[test]
fn test_detected_dtype_bytes_per_element() {
    assert_eq!(DetectedDtype::F32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::F16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::BF16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::F64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::I8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::U8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::F8.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::SubByte.bytes_per_element(), None);
    assert_eq!(DetectedDtype::I16.bytes_per_element(), Some(2));
    assert_eq!(DetectedDtype::I32.bytes_per_element(), Some(4));
    assert_eq!(DetectedDtype::I64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::Bool.bytes_per_element(), Some(1));
    assert_eq!(DetectedDtype::C64.bytes_per_element(), Some(8));
    assert_eq!(DetectedDtype::Other.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_from_safetensors() {
    use safetensors::Dtype as SD;

    assert_eq!(DetectedDtype::from_safetensors(SD::F32), DetectedDtype::F32);
    assert_eq!(DetectedDtype::from_safetensors(SD::F16), DetectedDtype::F16);
    assert_eq!(
        DetectedDtype::from_safetensors(SD::BF16),
        DetectedDtype::BF16
    );
    assert_eq!(DetectedDtype::from_safetensors(SD::F64), DetectedDtype::F64);
    assert_eq!(DetectedDtype::from_safetensors(SD::I8), DetectedDtype::I8);
    assert_eq!(DetectedDtype::from_safetensors(SD::U8), DetectedDtype::U8);
    assert_eq!(
        DetectedDtype::from_safetensors(SD::F8_E5M2),
        DetectedDtype::F8
    );
    assert_eq!(
        DetectedDtype::from_safetensors(SD::F8_E4M3),
        DetectedDtype::F8
    );
    assert_eq!(
        DetectedDtype::from_safetensors(SD::F8_E8M0),
        DetectedDtype::F8
    );
    assert_eq!(
        DetectedDtype::from_safetensors(SD::F4),
        DetectedDtype::SubByte
    );
    assert_eq!(
        DetectedDtype::from_safetensors(SD::F6_E2M3),
        DetectedDtype::SubByte
    );
    assert_eq!(
        DetectedDtype::from_safetensors(SD::F6_E3M2),
        DetectedDtype::SubByte
    );
    assert_eq!(DetectedDtype::from_safetensors(SD::I16), DetectedDtype::I16);
    assert_eq!(DetectedDtype::from_safetensors(SD::U16), DetectedDtype::I16);
    assert_eq!(DetectedDtype::from_safetensors(SD::I32), DetectedDtype::I32);
    assert_eq!(DetectedDtype::from_safetensors(SD::U32), DetectedDtype::I32);
    assert_eq!(DetectedDtype::from_safetensors(SD::I64), DetectedDtype::I64);
    assert_eq!(DetectedDtype::from_safetensors(SD::U64), DetectedDtype::I64);
    assert_eq!(
        DetectedDtype::from_safetensors(SD::BOOL),
        DetectedDtype::Bool
    );
    assert_eq!(DetectedDtype::from_safetensors(SD::C64), DetectedDtype::C64);
}

#[test]
fn test_tensors_sorted_by_name() {
    let f32_a: Vec<u8> = vec![0u8; 16];
    let f32_b: Vec<u8> = vec![0u8; 16];
    let f32_c: Vec<u8> = vec![0u8; 16];

    let bytes = build_safetensors_typed(&[
        ("z_last", &[4], &f32_a, safetensors::Dtype::F32),
        ("a_first", &[4], &f32_b, safetensors::Dtype::F32),
        ("m_middle", &[4], &f32_c, safetensors::Dtype::F32),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    let names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["a_first", "m_middle", "z_last"]);
}

// =========================================================================
// GGUF dequantization tests (via nn-gguf)
// =========================================================================

// -- Q4_0 dequantization: known block -> expected float values --

#[test]
fn test_q4_0_dequant_known_block_values() {
    // Q4_0: scale=2.0, byte 0x37 -> lo=7, hi=3.
    //   lo: 2.0 * (7 - 8) = -2.0
    //   hi: 2.0 * (3 - 8) = -10.0
    let scale_bytes = half::f16::from_f32(2.0).to_le_bytes();
    let mut block = vec![0u8; 18];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..16 {
        block[2 + i] = 0x37;
    }
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for i in (0..32).step_by(2) {
        assert!(
            (result[i] - (-2.0)).abs() < 1e-3,
            "element {i}: expected -2.0, got {}",
            result[i]
        );
        assert!(
            (result[i + 1] - (-10.0)).abs() < 1e-3,
            "element {}: expected -10.0, got {}",
            i + 1,
            result[i + 1]
        );
    }
}

#[test]
fn test_q4_0_dequant_centered_nibbles() {
    // scale=1.0, nibble=8 -> val = 1.0 * (8 - 8) = 0.0
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 18];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..16 {
        block[2 + i] = 0x88;
    }
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for (i, &v) in result.iter().enumerate() {
        assert!((v).abs() < 1e-4, "element {i}: expected 0.0, got {v}");
    }
}

// -- Q4_0 block size validation --

#[test]
fn test_q4_0_block_size_is_32() {
    assert_eq!(nn_gguf::GgufDType::Q4_0.block_size(), 32);
    assert_eq!(nn_gguf::GgufDType::Q4_0.type_size(), 18);
}

// -- Q4_0 edge cases --

#[test]
fn test_q4_0_dequant_zero_scale() {
    // Zero scale with nonzero nibbles: all outputs should be 0.0.
    let mut block = vec![0u8; 18];
    for i in 0..16 {
        block[2 + i] = 0xFF;
    }
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0, "zero scale should produce zero output");
    }
}

#[test]
fn test_q4_0_dequant_max_scale() {
    // Large scale factor: verify extreme values.
    let scale_bytes = half::f16::from_f32(100.0).to_le_bytes();
    let mut block = vec![0u8; 18];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // nibble=15 -> 100*(15-8) = 700, nibble=0 -> 100*(0-8) = -800
    for i in 0..16 {
        block[2 + i] = 0xF0; // lo=0, hi=15
    }
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for i in (0..32).step_by(2) {
        assert!(
            (result[i] - (-800.0)).abs() < 1.0,
            "lo=0: expected -800.0, got {}",
            result[i]
        );
        assert!(
            (result[i + 1] - 700.0).abs() < 1.0,
            "hi=15: expected 700.0, got {}",
            result[i + 1]
        );
    }
}

#[test]
fn test_q4_0_dequant_all_zeros_block() {
    let block = vec![0u8; 18];
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_q4_0_dequant_extreme_nibbles() {
    // All nibbles min (0) and max (15) with scale=1.0.
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 18];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // 0x0F -> lo=15, hi=0
    for i in 0..16 {
        block[2 + i] = 0x0F;
    }
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    for i in (0..32).step_by(2) {
        assert!(
            (result[i] - 7.0).abs() < 1e-3,
            "lo=15: (15-8)=7, got {}",
            result[i]
        );
        assert!(
            (result[i + 1] - (-8.0)).abs() < 1e-3,
            "hi=0: (0-8)=-8, got {}",
            result[i + 1]
        );
    }
}

#[test]
fn test_q4_0_dequant_multi_block() {
    // 2 blocks: block 0 centered (0.0), block 1 with scale=3.0 and 0xA5
    let s1 = half::f16::from_f32(1.0).to_le_bytes();
    let s3 = half::f16::from_f32(3.0).to_le_bytes();
    let mut data = vec![0u8; 18 * 2];
    data[0] = s1[0];
    data[1] = s1[1];
    for i in 0..16 {
        data[2 + i] = 0x88; // centered
    }
    data[18] = s3[0];
    data[19] = s3[1];
    // 0xA5 -> lo=5, hi=10. 3*(5-8)=-9, 3*(10-8)=6
    for i in 0..16 {
        data[20 + i] = 0xA5;
    }

    let result = nn_gguf::dequantize_q4_0(&data, 64);
    assert_eq!(result.len(), 64);
    for i in 0..32 {
        assert!(
            (result[i]).abs() < 1e-4,
            "block 0 centered, got {}",
            result[i]
        );
    }
    for i in (32..64).step_by(2) {
        assert!(
            (result[i] - (-9.0)).abs() < 1e-2,
            "block 1 lo: expected -9.0, got {}",
            result[i]
        );
        assert!(
            (result[i + 1] - 6.0).abs() < 1e-2,
            "block 1 hi: expected 6.0, got {}",
            result[i + 1]
        );
    }
}

// -- Q4_1 dequantization: min/scale + nibbles -> expected values --

#[test]
fn test_q4_1_dequant_known_values() {
    // Q4_1: val = d * q + m. d=0.5, m=1.0, nibble=10 -> 0.5*10 + 1.0 = 6.0
    let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
    let m_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 20];
    block[0] = d_bytes[0];
    block[1] = d_bytes[1];
    block[2] = m_bytes[0];
    block[3] = m_bytes[1];
    // 0xAA -> lo=10, hi=10
    for i in 0..16 {
        block[4 + i] = 0xAA;
    }
    let result = nn_gguf::dequantize_q4_1(&block, 32);
    assert_eq!(result.len(), 32);
    for (i, &v) in result.iter().enumerate() {
        assert!((v - 6.0).abs() < 1e-3, "element {i}: expected 6.0, got {v}");
    }
}

#[test]
fn test_q4_1_dequant_min_offset_only() {
    // d=0.0, m=7.5: all outputs = 7.5
    let m_bytes = half::f16::from_f32(7.5).to_le_bytes();
    let mut block = vec![0u8; 20];
    block[2] = m_bytes[0];
    block[3] = m_bytes[1];
    for i in 0..16 {
        block[4 + i] = 0xCC;
    }
    let result = nn_gguf::dequantize_q4_1(&block, 32);
    for &v in &result {
        assert!((v - 7.5).abs() < 1e-2, "expected 7.5, got {v}");
    }
}

#[test]
fn test_q4_1_block_size_is_32() {
    assert_eq!(nn_gguf::GgufDType::Q4_1.block_size(), 32);
    assert_eq!(nn_gguf::GgufDType::Q4_1.type_size(), 20);
}

#[test]
fn test_q4_1_dequant_all_zeros_block() {
    let block = vec![0u8; 20];
    let result = nn_gguf::dequantize_q4_1(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_q4_1_dequant_values_non_negative() {
    // Q4_1 uses unsigned nibbles + non-negative d and m -> all outputs >= 0.
    let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
    let mut block = vec![0u8; 20];
    block[0] = d_bytes[0];
    block[1] = d_bytes[1];
    // m = 0.0 (already zero)
    for i in 0..16 {
        block[4 + i] = (i as u8) * 5;
    }
    let result = nn_gguf::dequantize_q4_1(&block, 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(v >= -1e-4, "element {i}: expected non-negative, got {v}");
    }
}

// -- Q8_0 dequantization: scale + int8 -> expected values --

#[test]
fn test_q8_0_dequant_known_values() {
    // Q8_0: val = scale * q. scale=0.25, q=100 -> 25.0
    let scale_bytes = half::f16::from_f32(0.25).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..32 {
        block[2 + i] = 100u8;
    }
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    assert_eq!(result.len(), 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 25.0).abs() < 0.1,
            "element {i}: expected 25.0, got {v}"
        );
    }
}

#[test]
fn test_q8_0_dequant_negative_values() {
    // scale=1.0, q=-50 (i8) -> val = -50.0
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..32 {
        block[2 + i] = (-50i8) as u8;
    }
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - (-50.0)).abs() < 0.1,
            "element {i}: expected -50.0, got {v}"
        );
    }
}

// -- Q8_0 block size validation --

#[test]
fn test_q8_0_block_size_is_32() {
    assert_eq!(nn_gguf::GgufDType::Q8_0.block_size(), 32);
    assert_eq!(nn_gguf::GgufDType::Q8_0.type_size(), 34);
}

// -- Q8_0 edge cases --

#[test]
fn test_q8_0_dequant_all_zeros_block() {
    let block = vec![0u8; 34];
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_q8_0_dequant_zero_scale() {
    let mut block = vec![0u8; 34];
    for i in 0..32 {
        block[2 + i] = 127;
    }
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    for &v in &result {
        assert_eq!(v, 0.0, "zero scale -> zero output");
    }
}

#[test]
fn test_q8_0_dequant_max_scale() {
    let scale_bytes = half::f16::from_f32(50.0).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..32 {
        block[2 + i] = 127;
    }
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    let expected = half::f16::from_f32(50.0).to_f32() * 127.0;
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1.0,
            "element {i}: expected {expected}, got {v}"
        );
    }
}

#[test]
fn test_q8_0_dequant_min_max_i8() {
    // i8::MIN (-128) and i8::MAX (127) with scale=1.0.
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    block[2] = 128; // -128 as u8
    block[3] = 127; // 127 as u8
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    assert!(
        (result[0] - (-128.0)).abs() < 0.5,
        "expected -128, got {}",
        result[0]
    );
    assert!(
        (result[1] - 127.0).abs() < 0.5,
        "expected 127, got {}",
        result[1]
    );
}

#[test]
fn test_q8_0_dequant_mixed_positive_negative() {
    // scale=2.0, alternating q=10 and q=-10.
    let scale_bytes = half::f16::from_f32(2.0).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..32 {
        block[2 + i] = if i % 2 == 0 { 10u8 } else { (-10i8) as u8 };
    }
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    for i in 0..32 {
        let expected = if i % 2 == 0 { 20.0 } else { -20.0 };
        assert!(
            (result[i] - expected).abs() < 0.1,
            "element {i}: expected {expected}, got {}",
            result[i]
        );
    }
}

// -- Q4_K dequantization --

#[test]
fn test_q4_k_block_size_is_256() {
    assert_eq!(nn_gguf::GgufDType::Q4K.block_size(), 256);
    assert_eq!(nn_gguf::GgufDType::Q4K.type_size(), 144);
}

#[test]
fn test_q4_k_dequant_all_zeros_block() {
    let block = vec![0u8; 144];
    let result = nn_gguf::dequantize_q4_k(&block, 256);
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_q4_k_dequant_zero_super_scale() {
    // d=0, dmin=0 with nonzero sub-block data.
    let mut block = vec![0u8; 144];
    for i in 4..16 {
        block[i] = 0xFF;
    }
    for i in 16..144 {
        block[i] = 0xAA;
    }
    let result = nn_gguf::dequantize_q4_k(&block, 256);
    for &v in &result {
        assert_eq!(v, 0.0, "d=0, dmin=0 -> all zeros");
    }
}

#[test]
fn test_q4_k_dequant_known_sub_block() {
    // d=1.0, dmin=0.0, sub-block 0: scale=2, all q=3 -> 1.0*2*3 - 0 = 6.0
    let mut block = vec![0u8; 144];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    block[0] = d_bytes[0];
    block[1] = d_bytes[1];
    // scales[0] = 2 (low nibble at byte 4)
    block[4] = 0x02;
    // q values: 0x33 -> lo=3, hi=3
    for i in 0..16 {
        block[16 + i] = 0x33;
    }
    let result = nn_gguf::dequantize_q4_k(&block, 256);
    assert_eq!(result.len(), 256);
    for i in 0..32 {
        assert!(
            (result[i] - 6.0).abs() < 1e-3,
            "sub-block 0 element {i}: expected 6.0, got {}",
            result[i]
        );
    }
    // Remaining sub-blocks: scale=0 -> 0.0
    for i in 32..256 {
        assert!(
            result[i].abs() < 1e-6,
            "element {i}: expected 0, got {}",
            result[i]
        );
    }
}

#[test]
fn test_q4_k_dequant_with_dmin() {
    // d=1.0, dmin=0.5, sub-block 0: scale=1, min=4, q=0
    // val = 1.0*1*0 - 0.5*4 = -2.0
    let mut block = vec![0u8; 144];
    let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
    block[0] = d_bytes[0];
    block[1] = d_bytes[1];
    block[2] = dmin_bytes[0];
    block[3] = dmin_bytes[1];
    block[4] = 0x01; // scales[0]=1
    block[8] = 0x04; // mins[0]=4
    let result = nn_gguf::dequantize_q4_k(&block, 256);
    for i in 0..32 {
        assert!(
            (result[i] - (-2.0)).abs() < 1e-3,
            "element {i}: expected -2.0, got {}",
            result[i]
        );
    }
}

// -- Round-trip sanity: dequant values in reasonable range --

#[test]
fn test_q4_0_dequant_values_in_reasonable_range() {
    // scale=1.0, max |q-8| = 8.
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 18];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..16 {
        block[2 + i] = (i as u8).wrapping_mul(17).wrapping_add(3);
    }
    let result = nn_gguf::dequantize_q4_0(&block, 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.abs() <= 8.5, "element {i}: value {v} out of range");
        assert!(v.is_finite(), "element {i}: must be finite");
    }
}

#[test]
fn test_q8_0_dequant_values_in_reasonable_range() {
    // scale=0.1, i8 in [-128,127] -> output in [-12.8, 12.7].
    let scale_bytes = half::f16::from_f32(0.1).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..32 {
        block[2 + i] = i as u8;
    }
    let result = nn_gguf::dequantize_q8_0(&block, 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "element {i}: must be finite");
        assert!(v.abs() < 20.0, "element {i}: value {v} unexpectedly large");
    }
}

// -- Format detection from GGUF dtype --

#[test]
fn test_gguf_dtype_from_u32_all_standard_types() {
    use nn_gguf::GgufDType;
    assert_eq!(GgufDType::from_u32(0), Some(GgufDType::F32));
    assert_eq!(GgufDType::from_u32(1), Some(GgufDType::F16));
    assert_eq!(GgufDType::from_u32(2), Some(GgufDType::Q4_0));
    assert_eq!(GgufDType::from_u32(3), Some(GgufDType::Q4_1));
    assert_eq!(GgufDType::from_u32(6), Some(GgufDType::Q5_0));
    assert_eq!(GgufDType::from_u32(7), Some(GgufDType::Q5_1));
    assert_eq!(GgufDType::from_u32(8), Some(GgufDType::Q8_0));
    assert_eq!(GgufDType::from_u32(10), Some(GgufDType::Q2K));
    assert_eq!(GgufDType::from_u32(11), Some(GgufDType::Q3K));
    assert_eq!(GgufDType::from_u32(12), Some(GgufDType::Q4K));
    assert_eq!(GgufDType::from_u32(13), Some(GgufDType::Q5K));
    assert_eq!(GgufDType::from_u32(14), Some(GgufDType::Q6K));
    assert_eq!(GgufDType::from_u32(30), Some(GgufDType::BF16));
}

#[test]
fn test_gguf_dtype_from_u32_invalid_ids() {
    assert_eq!(nn_gguf::GgufDType::from_u32(4), None);
    assert_eq!(nn_gguf::GgufDType::from_u32(5), None);
    assert_eq!(nn_gguf::GgufDType::from_u32(15), None);
    assert_eq!(nn_gguf::GgufDType::from_u32(31), None);
    assert_eq!(nn_gguf::GgufDType::from_u32(100), None);
    assert_eq!(nn_gguf::GgufDType::from_u32(u32::MAX), None);
}

// -- Block size validation --

#[test]
fn test_gguf_dtype_block_sizes() {
    use nn_gguf::GgufDType;
    // 32-element block types.
    assert_eq!(GgufDType::Q4_0.block_size(), 32);
    assert_eq!(GgufDType::Q4_1.block_size(), 32);
    assert_eq!(GgufDType::Q5_0.block_size(), 32);
    assert_eq!(GgufDType::Q5_1.block_size(), 32);
    assert_eq!(GgufDType::Q8_0.block_size(), 32);
    // 256-element block types.
    assert_eq!(GgufDType::Q2K.block_size(), 256);
    assert_eq!(GgufDType::Q3K.block_size(), 256);
    assert_eq!(GgufDType::Q4K.block_size(), 256);
    assert_eq!(GgufDType::Q5K.block_size(), 256);
    assert_eq!(GgufDType::Q6K.block_size(), 256);
    // Non-quantized types have block_size 1.
    assert_eq!(GgufDType::F32.block_size(), 1);
    assert_eq!(GgufDType::F16.block_size(), 1);
    assert_eq!(GgufDType::BF16.block_size(), 1);
}

#[test]
fn test_gguf_dtype_type_sizes() {
    use nn_gguf::GgufDType;
    assert_eq!(GgufDType::F32.type_size(), 4);
    assert_eq!(GgufDType::F16.type_size(), 2);
    assert_eq!(GgufDType::Q4_0.type_size(), 18);
    assert_eq!(GgufDType::Q4_1.type_size(), 20);
    assert_eq!(GgufDType::Q5_0.type_size(), 22);
    assert_eq!(GgufDType::Q5_1.type_size(), 24);
    assert_eq!(GgufDType::Q8_0.type_size(), 34);
    assert_eq!(GgufDType::Q2K.type_size(), 84);
    assert_eq!(GgufDType::Q3K.type_size(), 110);
    assert_eq!(GgufDType::Q4K.type_size(), 144);
    assert_eq!(GgufDType::Q5K.type_size(), 176);
    assert_eq!(GgufDType::Q6K.type_size(), 210);
}

// -- Output length matches element count for all formats --

#[test]
fn test_dequant_output_lengths_match_element_count() {
    let q4_0_data = vec![0u8; 18 * 4]; // 4 blocks = 128 elements
    let q4_1_data = vec![0u8; 20 * 4];
    let q8_0_data = vec![0u8; 34 * 4];
    let q4_k_data = vec![0u8; 144 * 2]; // 2 blocks = 512 elements
    let q6_k_data = vec![0u8; 210 * 2];
    let q2_k_data = vec![0u8; 84 * 2];
    let q3_k_data = vec![0u8; 110 * 2];
    let q5_k_data = vec![0u8; 176 * 2];
    let q5_0_data = vec![0u8; 22 * 4];
    let q5_1_data = vec![0u8; 24 * 4];

    assert_eq!(nn_gguf::dequantize_q4_0(&q4_0_data, 128).len(), 128);
    assert_eq!(nn_gguf::dequantize_q4_1(&q4_1_data, 128).len(), 128);
    assert_eq!(nn_gguf::dequantize_q8_0(&q8_0_data, 128).len(), 128);
    assert_eq!(nn_gguf::dequantize_q4_k(&q4_k_data, 512).len(), 512);
    assert_eq!(nn_gguf::dequantize_q6_k(&q6_k_data, 512).len(), 512);
    assert_eq!(nn_gguf::dequantize_q2_k(&q2_k_data, 512).len(), 512);
    assert_eq!(nn_gguf::dequantize_q3_k(&q3_k_data, 512).len(), 512);
    assert_eq!(nn_gguf::dequantize_q5_k(&q5_k_data, 512).len(), 512);
    assert_eq!(nn_gguf::dequantize_q5_0(&q5_0_data, 128).len(), 128);
    assert_eq!(nn_gguf::dequantize_q5_1(&q5_1_data, 128).len(), 128);
}

// -- All-zeros blocks produce all-zeros output for every format --

#[test]
fn test_dequant_all_zeros_produce_zeros() {
    for &v in &nn_gguf::dequantize_q4_0(&[0u8; 18], 32) {
        assert_eq!(v, 0.0, "q4_0");
    }
    for &v in &nn_gguf::dequantize_q4_1(&[0u8; 20], 32) {
        assert_eq!(v, 0.0, "q4_1");
    }
    for &v in &nn_gguf::dequantize_q8_0(&[0u8; 34], 32) {
        assert_eq!(v, 0.0, "q8_0");
    }
    for &v in &nn_gguf::dequantize_q4_k(&[0u8; 144], 256) {
        assert_eq!(v, 0.0, "q4_k");
    }
    for &v in &nn_gguf::dequantize_q6_k(&vec![0u8; 210], 256) {
        assert_eq!(v, 0.0, "q6_k");
    }
    for &v in &nn_gguf::dequantize_q2_k(&[0u8; 84], 256) {
        assert_eq!(v, 0.0, "q2_k");
    }
    for &v in &nn_gguf::dequantize_q3_k(&[0u8; 110], 256) {
        assert_eq!(v, 0.0, "q3_k");
    }
    for &v in &nn_gguf::dequantize_q5_0(&[0u8; 22], 32) {
        assert_eq!(v, 0.0, "q5_0");
    }
    for &v in &nn_gguf::dequantize_q5_1(&[0u8; 24], 32) {
        assert_eq!(v, 0.0, "q5_1");
    }
    for &v in &nn_gguf::dequantize_q5_k(&[0u8; 176], 256) {
        assert_eq!(v, 0.0, "q5_k");
    }
}

// -- All dequantized outputs are finite --

#[test]
fn test_dequant_all_outputs_finite() {
    let s = half::f16::from_f32(0.5).to_le_bytes();

    // Q4_0
    let mut q4_0 = vec![0u8; 18];
    q4_0[0] = s[0];
    q4_0[1] = s[1];
    for i in 0..16 {
        q4_0[2 + i] = (i as u8).wrapping_mul(13).wrapping_add(7);
    }
    for &v in &nn_gguf::dequantize_q4_0(&q4_0, 32) {
        assert!(v.is_finite(), "q4_0 must be finite, got {v}");
    }

    // Q8_0
    let s8 = half::f16::from_f32(0.1).to_le_bytes();
    let mut q8_0 = vec![0u8; 34];
    q8_0[0] = s8[0];
    q8_0[1] = s8[1];
    for i in 0..32 {
        q8_0[2 + i] = (i as u8).wrapping_mul(11);
    }
    for &v in &nn_gguf::dequantize_q8_0(&q8_0, 32) {
        assert!(v.is_finite(), "q8_0 must be finite, got {v}");
    }

    // Q4_1
    let mut q4_1 = vec![0u8; 20];
    q4_1[0] = s[0];
    q4_1[1] = s[1];
    q4_1[2] = s[0];
    q4_1[3] = s[1];
    for i in 0..16 {
        q4_1[4 + i] = (i as u8).wrapping_mul(7).wrapping_add(2);
    }
    for &v in &nn_gguf::dequantize_q4_1(&q4_1, 32) {
        assert!(v.is_finite(), "q4_1 must be finite, got {v}");
    }
}

// -- Model architecture detection --

#[test]
fn test_model_architecture_llama() {
    use nn_gguf::{GgufMetadata, GgufMetadataValue, ModelArchitecture};
    use std::collections::HashMap;

    let mut entries = HashMap::new();
    entries.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("llama".to_string()),
    );
    entries.insert(
        "llama.context_length".to_string(),
        GgufMetadataValue::U32(4096),
    );
    entries.insert(
        "llama.embedding_length".to_string(),
        GgufMetadataValue::U32(4096),
    );
    entries.insert("llama.block_count".to_string(), GgufMetadataValue::U32(32));
    entries.insert(
        "llama.attention.head_count".to_string(),
        GgufMetadataValue::U32(32),
    );
    entries.insert(
        "llama.attention.head_count_kv".to_string(),
        GgufMetadataValue::U32(8),
    );
    entries.insert(
        "llama.vocab_size".to_string(),
        GgufMetadataValue::U32(32000),
    );

    let meta = GgufMetadata { entries };
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.architecture, "llama");
    assert_eq!(arch.context_length, Some(4096));
    assert_eq!(arch.embedding_length, Some(4096));
    assert_eq!(arch.block_count, Some(32));
    assert_eq!(arch.head_count, Some(32));
    assert_eq!(arch.head_count_kv, Some(8));
    assert_eq!(arch.vocab_size, Some(32000));
}

#[test]
fn test_model_architecture_mistral_via_llama() {
    use nn_gguf::{GgufMetadata, GgufMetadataValue, ModelArchitecture};
    use std::collections::HashMap;

    let mut entries = HashMap::new();
    entries.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("llama".to_string()),
    );
    entries.insert(
        "llama.context_length".to_string(),
        GgufMetadataValue::U32(32768),
    );
    entries.insert(
        "llama.rope.freq_base".to_string(),
        GgufMetadataValue::F32(1000000.0),
    );

    let meta = GgufMetadata { entries };
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.architecture, "llama");
    assert_eq!(arch.context_length, Some(32768));
    assert!((arch.rope_freq_base.unwrap() - 1_000_000.0).abs() < 1.0);
}

#[test]
fn test_model_architecture_qwen2() {
    use nn_gguf::{GgufMetadata, GgufMetadataValue, ModelArchitecture};
    use std::collections::HashMap;

    let mut entries = HashMap::new();
    entries.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("qwen2".to_string()),
    );
    entries.insert(
        "qwen2.context_length".to_string(),
        GgufMetadataValue::U32(32768),
    );
    entries.insert(
        "qwen2.embedding_length".to_string(),
        GgufMetadataValue::U32(3584),
    );
    entries.insert(
        "qwen2.vocab_size".to_string(),
        GgufMetadataValue::U32(152064),
    );

    let meta = GgufMetadata { entries };
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.architecture, "qwen2");
    assert_eq!(arch.context_length, Some(32768));
    assert_eq!(arch.embedding_length, Some(3584));
    assert_eq!(arch.vocab_size, Some(152064));
}

#[test]
fn test_model_architecture_unknown_when_empty() {
    use nn_gguf::{GgufMetadata, ModelArchitecture};
    use std::collections::HashMap;

    let meta = GgufMetadata {
        entries: HashMap::new(),
    };
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.architecture, "unknown");
    assert_eq!(arch.context_length, None);
    assert_eq!(arch.block_count, None);
    assert_eq!(arch.head_count, None);
    assert_eq!(arch.quantization_version, None);
}

#[test]
fn test_model_architecture_phi() {
    use nn_gguf::{GgufMetadata, GgufMetadataValue, ModelArchitecture};
    use std::collections::HashMap;

    let mut entries = HashMap::new();
    entries.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("phi".to_string()),
    );
    entries.insert(
        "phi.context_length".to_string(),
        GgufMetadataValue::U32(4096),
    );
    entries.insert(
        "phi.embedding_length".to_string(),
        GgufMetadataValue::U32(2560),
    );
    entries.insert("phi.block_count".to_string(), GgufMetadataValue::U32(32));
    entries.insert(
        "phi.rope.freq_base".to_string(),
        GgufMetadataValue::F64(250000.0),
    );

    let meta = GgufMetadata { entries };
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.architecture, "phi");
    assert_eq!(arch.context_length, Some(4096));
    assert_eq!(arch.embedding_length, Some(2560));
    assert_eq!(arch.block_count, Some(32));
    assert!((arch.rope_freq_base.unwrap() - 250000.0).abs() < 1e-6);
}

#[test]
fn test_model_architecture_quantization_version() {
    use nn_gguf::{GgufMetadata, GgufMetadataValue, ModelArchitecture};
    use std::collections::HashMap;

    let mut entries = HashMap::new();
    entries.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("llama".to_string()),
    );
    entries.insert(
        "general.quantization_version".to_string(),
        GgufMetadataValue::U32(2),
    );

    let meta = GgufMetadata { entries };
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.quantization_version, Some(2));
}

// =========================================================================
// TensorQuantInfo construction tests
// =========================================================================

#[test]
fn test_tensor_quant_info_construction() {
    let info = TensorQuantInfo {
        name: "layer.weight".to_string(),
        dtype: DetectedDtype::F32,
        shape: vec![64, 32],
        num_elements: 2048,
        size_bytes: 8192,
    };
    assert_eq!(info.name, "layer.weight");
    assert_eq!(info.dtype, DetectedDtype::F32);
    assert_eq!(info.shape, vec![64, 32]);
    assert_eq!(info.num_elements, 2048);
    assert_eq!(info.size_bytes, 8192);
}

#[test]
fn test_tensor_quant_info_scalar_tensor() {
    let info = TensorQuantInfo {
        name: "scale".to_string(),
        dtype: DetectedDtype::F32,
        shape: vec![],
        num_elements: 1,
        size_bytes: 4,
    };
    assert_eq!(info.shape.len(), 0);
    assert_eq!(info.num_elements, 1);
}

#[test]
fn test_tensor_quant_info_debug_impl() {
    let info = TensorQuantInfo {
        name: "x".to_string(),
        dtype: DetectedDtype::BF16,
        shape: vec![4],
        num_elements: 4,
        size_bytes: 8,
    };
    let debug = format!("{info:?}");
    assert!(debug.contains("BF16"));
    assert!(debug.contains("x"));
}

// =========================================================================
// DtypeBreakdown aggregation tests
// =========================================================================

#[test]
fn test_dtype_breakdown_construction() {
    let bd = DtypeBreakdown {
        dtype: DetectedDtype::F16,
        tensor_count: 5,
        total_parameters: 10_000,
        total_bytes: 20_000,
    };
    assert_eq!(bd.dtype, DetectedDtype::F16);
    assert_eq!(bd.tensor_count, 5);
    assert_eq!(bd.total_parameters, 10_000);
    assert_eq!(bd.total_bytes, 20_000);
}

#[test]
fn test_dtype_breakdown_aggregation_multiple_f32_tensors() {
    // 3 F32 tensors of different shapes, verify aggregation sums correctly.
    let w1: Vec<u8> = vec![0u8; 256 * 4]; // [16, 16] = 256 elems
    let w2: Vec<u8> = vec![0u8; 128 * 4]; // [8, 16] = 128 elems
    let w3: Vec<u8> = vec![0u8; 64 * 4]; // [8, 8] = 64 elems

    let bytes = build_safetensors_typed(&[
        ("a.weight", &[16, 16], &w1, safetensors::Dtype::F32),
        ("b.weight", &[8, 16], &w2, safetensors::Dtype::F32),
        ("c.weight", &[8, 8], &w3, safetensors::Dtype::F32),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.dtype_breakdown.len(), 1);
    let bd = &report.dtype_breakdown[0];
    assert_eq!(bd.dtype, DetectedDtype::F32);
    assert_eq!(bd.tensor_count, 3);
    assert_eq!(bd.total_parameters, 256 + 128 + 64);
    assert_eq!(bd.total_bytes, (256 + 128 + 64) * 4);
}

#[test]
fn test_dtype_breakdown_four_dtypes() {
    // F32, F16, BF16, I8 each with 1 tensor.
    let f32_data: Vec<u8> = vec![0u8; 4 * 4]; // [4] = 4 elems
    let f16_data: Vec<u8> = vec![0u8; 4 * 2];
    let bf16_data: Vec<u8> = vec![0u8; 4 * 2];
    let i8_data: Vec<u8> = vec![0u8; 4];

    let bytes = build_safetensors_typed(&[
        ("a", &[4], &f32_data, safetensors::Dtype::F32),
        ("b", &[4], &f16_data, safetensors::Dtype::F16),
        ("c", &[4], &bf16_data, safetensors::Dtype::BF16),
        ("d", &[4], &i8_data, safetensors::Dtype::I8),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.dtype_breakdown.len(), 4);
    assert!(report.is_mixed_precision());
    assert_eq!(report.total_tensors, 4);
    assert_eq!(report.total_parameters, 16);
    assert_eq!(report.total_bytes, 4 * 4 + 4 * 2 + 4 * 2 + 4);
}

// =========================================================================
// QuantRecommendation tests
// =========================================================================

#[test]
fn test_quant_recommendation_savings_arithmetic() {
    let rec = QuantRecommendation {
        target_dtype: DetectedDtype::F16,
        tensor_names: vec!["w".to_string()],
        current_bytes: 1000,
        projected_bytes: 500,
        savings_bytes: 500,
    };
    assert_eq!(rec.savings_bytes, rec.current_bytes - rec.projected_bytes);
}

#[test]
fn test_quant_recommendation_multiple_tensors_in_one_rec() {
    // Two large F32 tensors should both appear in the same F16 recommendation.
    let w1: Vec<u8> = vec![0u8; 2048 * 4];
    let w2: Vec<u8> = vec![0u8; 4096 * 4];

    let bytes = build_safetensors_typed(&[
        ("layer1.weight", &[32, 64], &w1, safetensors::Dtype::F32),
        ("layer2.weight", &[64, 64], &w2, safetensors::Dtype::F32),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16)
        .expect("should have F16 rec");

    assert_eq!(f16_rec.tensor_names.len(), 2);
    assert!(f16_rec.tensor_names.contains(&"layer1.weight".to_string()));
    assert!(f16_rec.tensor_names.contains(&"layer2.weight".to_string()));
    assert_eq!(f16_rec.current_bytes, (2048 + 4096) * 4);
    assert_eq!(f16_rec.projected_bytes, (2048 + 4096) * 2);
    assert_eq!(f16_rec.savings_bytes, (2048 + 4096) * 2);
}

#[test]
fn test_no_recommendations_for_bf16_only() {
    // BF16 model gets no recommendations (already compact).
    let bf16_data: Vec<u8> = vec![0u8; 4096 * 2];
    let bytes = build_safetensors_typed(&[(
        "enc.weight",
        &[64, 64],
        &bf16_data,
        safetensors::Dtype::BF16,
    )]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert!(
        report.recommendations.is_empty(),
        "BF16 model should have no recommendations"
    );
}

#[test]
fn test_no_recommendations_for_f16_only() {
    // F16 model gets no recommendations (already compact).
    let f16_data: Vec<u8> = vec![0u8; 4096 * 2];
    let bytes =
        build_safetensors_typed(&[("dec.weight", &[64, 64], &f16_data, safetensors::Dtype::F16)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert!(
        report.recommendations.is_empty(),
        "F16 model should have no recommendations"
    );
}

// =========================================================================
// Report summary/formatting edge cases
// =========================================================================

#[test]
fn test_report_summary_no_recommendations_message() {
    // A model that is already compact should show "No quantization recommendations".
    let bf16_data: Vec<u8> = vec![0u8; 512 * 2];
    let bytes = build_safetensors_typed(&[("w", &[512], &bf16_data, safetensors::Dtype::BF16)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    let summary = report.summary();
    assert!(
        summary.contains("No quantization recommendations"),
        "compact model should show no-recommendation message"
    );
}

#[test]
fn test_report_total_savings_with_f64_and_f32() {
    // F64 + large F32: should have 3 recommendations (F64->F32, F32->F16, F32->I8).
    let f64_data: Vec<u8> = vec![0u8; 16 * 8]; // 16 elems * 8 bytes
    let f32_data: Vec<u8> = vec![0u8; 2048 * 4];

    let bytes = build_safetensors_typed(&[
        ("dbl.weight", &[4, 4], &f64_data, safetensors::Dtype::F64),
        ("flt.weight", &[32, 64], &f32_data, safetensors::Dtype::F32),
    ]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.recommendations.len(), 3);

    // Verify total_savings sums all three.
    let manual_sum: usize = report.recommendations.iter().map(|r| r.savings_bytes).sum();
    assert_eq!(report.total_savings_bytes(), manual_sum);
    assert!(report.total_savings_bytes() > 0);
}

#[test]
fn test_dtype_fraction_single_dtype_is_one() {
    let f32_data: Vec<u8> = vec![0u8; 64 * 4];
    let bytes = build_safetensors_typed(&[("w", &[8, 8], &f32_data, safetensors::Dtype::F32)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert!((report.dtype_fraction(DetectedDtype::F32) - 1.0).abs() < 1e-10);
    assert_eq!(report.dtype_fraction(DetectedDtype::F16), 0.0);
}

#[test]
fn test_detected_dtype_display_all_variants() {
    // Verify Display matches label() for all variants.
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
    for v in variants {
        assert_eq!(format!("{v}"), v.label());
    }
}

#[test]
fn test_detect_bool_tensor() {
    let bool_data: Vec<u8> = vec![0u8; 8]; // 8 elements, 1 byte each
    let bytes = build_safetensors_typed(&[("mask", &[2, 4], &bool_data, safetensors::Dtype::BOOL)]);

    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::Bool);
    assert_eq!(report.dtype_breakdown[0].total_parameters, 8);
}

#[test]
fn test_format_bytes_helper() {
    // Exercise the format_bytes helper via summary output.
    // Small: bytes
    let small_data: Vec<u8> = vec![0u8; 4]; // 1 F32 element = 4 bytes
    let bytes = build_safetensors_typed(&[("tiny", &[1], &small_data, safetensors::Dtype::F32)]);
    let report = detect_quantization_from_bytes(&bytes).expect("should parse");
    let summary = report.summary();
    assert!(
        summary.contains("4 B") || summary.contains("4.00"),
        "small bytes format: {summary}"
    );

    // Medium: KB
    let kb_data: Vec<u8> = vec![0u8; 4096]; // 1024 F32 elements = 4 KB
    let bytes2 = build_safetensors_typed(&[("med", &[1024], &kb_data, safetensors::Dtype::F32)]);
    let report2 = detect_quantization_from_bytes(&bytes2).expect("should parse");
    let summary2 = report2.summary();
    assert!(summary2.contains("KB"), "should format as KB: {summary2}");
}
