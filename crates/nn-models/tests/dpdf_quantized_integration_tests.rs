// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Qwen3-VL GPTQ/AWQ quantized inference.
//!
//! Exercises the public API of `nn_models::qwen3_vl_quantized` from an
//! external crate perspective: config presets, validation, `QuantizedLinearLayer`
//! dequantization roundtrips, forward passes, memory estimation, and error paths.
//!
//! All tests use synthetic data -- no external weight files needed.
//!
//! Part of #3923.

use nn_models::qwen3_vl::Qwen3VLConfig;
use nn_models::{
    estimate_memory_bytes, QuantMethod, QuantizedLayerError, QuantizedLinearLayer,
    Qwen3VLQuantConfig,
};

// ============================================================================
// Helpers
// ============================================================================

/// Pack INT4 nibbles (values in `[0, 15]`) into u32 words.
///
/// 8 nibbles per u32, little-endian nibble order: bits [3:0] = first,
/// bits [7:4] = second, ..., bits [31:28] = eighth.
fn pack_int4(values: &[u32]) -> Vec<u32> {
    values
        .chunks(8)
        .map(|chunk| {
            let mut word = 0u32;
            for (i, &val) in chunk.iter().enumerate() {
                word |= (val & 0xF) << (i as u32 * 4);
            }
            word
        })
        .collect()
}

/// Build a `QuantizedLinearLayer` with uniform INT4 values, a single scale
/// and zero-point. Dimensions chosen so everything fits in one group.
fn uniform_layer(
    nibble: u32,
    in_f: usize,
    out_f: usize,
    scale: f32,
    zero: f32,
) -> QuantizedLinearLayer {
    let total = in_f * out_f;
    // Pad to multiple of 8 for packing
    let padded_len = total.div_ceil(8) * 8;
    let nibbles = vec![nibble; padded_len];
    let packed = pack_int4(&nibbles);
    let group_size = total; // single group
    assert!(
        group_size.is_power_of_two(),
        "helper requires total to be power of two"
    );
    let num_groups = 1;
    QuantizedLinearLayer::new(
        packed,
        vec![scale; num_groups],
        vec![zero; num_groups],
        in_f,
        out_f,
        group_size,
    )
    .expect("uniform_layer: valid construction")
}

// ============================================================================
// 1. GPTQ INT4 dequantize roundtrip
// ============================================================================

#[test]
fn test_gptq_dequantize_roundtrip_known_values() {
    // 8x1 = 8 elements, group_size=8, 1 group.
    // Nibbles: [0, 3, 7, 15, 1, 9, 12, 5]
    // scale=0.25, zero=8.0 => dequant = (nibble - 8.0) * 0.25
    let nibbles: Vec<u32> = vec![0, 3, 7, 15, 1, 9, 12, 5];
    let packed = pack_int4(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![0.25], vec![8.0], 8, 1, 8)
        .expect("valid construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    assert_eq!(weights.len(), 8);

    let expected: Vec<f32> = nibbles.iter().map(|&n| (n as f32 - 8.0) * 0.25).collect();

    for (i, (&got, &exp)) in weights.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "dequant roundtrip mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_gptq_dequantize_multi_group_roundtrip() {
    // 16 elements = 2 groups of 8.
    // Group 0: scale=1.0, zero=0.0 => dequant = nibble as f32
    // Group 1: scale=0.5, zero=4.0 => dequant = (nibble - 4.0) * 0.5
    let nibbles: Vec<u32> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, // group 0
        8, 9, 10, 11, 12, 13, 14, 15, // group 1
    ];
    let packed = pack_int4(&nibbles);
    let layer = QuantizedLinearLayer::new(packed, vec![1.0, 0.5], vec![0.0, 4.0], 4, 4, 8)
        .expect("valid construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    assert_eq!(weights.len(), 16);

    // Verify group 0
    for i in 0..8 {
        let exp = nibbles[i] as f32;
        assert!(
            (weights[i] - exp).abs() < 1e-6,
            "group0[{i}]: got {}, exp {exp}",
            weights[i]
        );
    }
    // Verify group 1
    for i in 8..16 {
        let exp = (nibbles[i] as f32 - 4.0) * 0.5;
        assert!(
            (weights[i] - exp).abs() < 1e-6,
            "group1[{i}]: got {}, exp {exp}",
            weights[i]
        );
    }
}

// ============================================================================
// 2. Quantized linear forward pass
// ============================================================================

#[test]
fn test_quantized_linear_forward_identity_like() {
    // 2x2 weight matrix approximating identity.
    // W = [[1, 0], [0, 1]] via nibbles [4,0,0,4], scale=0.25, zero=0.
    let nibbles: Vec<u32> = vec![4, 0, 0, 4];
    let packed = pack_int4(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![0.25], vec![0.0], 2, 2, 4)
        .expect("valid construction");

    let input = vec![5.0_f32, 11.0];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    assert_eq!(output.len(), 2);
    assert!(
        (output[0] - 5.0).abs() < 1e-5,
        "identity forward: output[0] should be ~5.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 11.0).abs() < 1e-5,
        "identity forward: output[1] should be ~11.0, got {}",
        output[1]
    );
}

#[test]
fn test_quantized_linear_forward_random_input() {
    // 8 in_features, 4 out_features => 32 elements, group_size=32.
    // All nibbles = 8, zero=8.0, scale=1.0 => all weights = 0.
    // Output should be all zeros regardless of input.
    let layer = uniform_layer(8, 8, 4, 1.0, 8.0);

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    assert_eq!(output.len(), 4);
    for (j, &val) in output.iter().enumerate() {
        assert!(
            val.abs() < 1e-6,
            "zero-weight forward: output[{j}] should be 0, got {val}"
        );
    }
}

#[test]
fn test_quantized_linear_forward_nonzero_accumulation() {
    // 4 in_features, 2 out_features, group_size=8.
    // Nibbles: all 3, scale=1.0, zero=0.0 => all weights = 3.0.
    // y[j] = sum_i(input[i] * 3.0)
    let nibbles: Vec<u32> = vec![3, 3, 3, 3, 3, 3, 3, 3];
    let packed = pack_int4(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 2, 8)
        .expect("valid construction");

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    // y[0] = y[1] = (1+2+3+4) * 3 = 30
    assert_eq!(output.len(), 2);
    assert!(
        (output[0] - 30.0).abs() < 1e-4,
        "accumulation: output[0] should be 30.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 30.0).abs() < 1e-4,
        "accumulation: output[1] should be 30.0, got {}",
        output[1]
    );
}

// ============================================================================
// 3. Preset configs
// ============================================================================

#[test]
fn test_preset_gptq_produces_valid_config() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.validate().expect("GPTQ preset should validate");

    assert_eq!(cfg.quant_method, QuantMethod::Gptq);
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(cfg.desc_act);
    assert!(cfg.is_moe());
    assert_eq!(cfg.num_experts(), 60);
    assert_eq!(cfg.active_experts(), 2);
    assert_eq!(cfg.base.num_layers, 48);
    assert_eq!(cfg.base.hidden_size, 3584);
}

#[test]
fn test_preset_awq_produces_valid_config() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    cfg.validate().expect("AWQ preset should validate");

    assert_eq!(cfg.quant_method, QuantMethod::Awq);
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(!cfg.desc_act, "AWQ must not use desc_act");
    assert!(cfg.is_moe());
    assert_eq!(cfg.num_experts(), 60);
    assert_eq!(cfg.active_experts(), 2);
}

#[test]
fn test_presets_share_base_architecture() {
    let gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();

    assert_eq!(gptq.base.hidden_size, awq.base.hidden_size);
    assert_eq!(gptq.base.num_heads, awq.base.num_heads);
    assert_eq!(gptq.base.num_kv_heads, awq.base.num_kv_heads);
    assert_eq!(gptq.base.intermediate_size, awq.base.intermediate_size);
    assert_eq!(gptq.base.num_layers, awq.base.num_layers);
    assert_eq!(gptq.base.vocab_size, awq.base.vocab_size);
    assert_eq!(gptq.base.num_experts, awq.base.num_experts);
}

#[test]
fn test_preset_gptq_format_conversion() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let fmt = cfg
        .to_gptq_format()
        .expect("GPTQ config should convert to GptqFormat");
    assert_eq!(fmt.group_size, 128);
    assert_eq!(fmt.bits, 4);
    assert!(fmt.act_order);

    // Cross-format conversion must fail
    assert!(cfg.to_awq_format().is_err());
}

#[test]
fn test_preset_awq_format_conversion() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    let fmt = cfg
        .to_awq_format()
        .expect("AWQ config should convert to AwqFormat");
    assert_eq!(fmt.group_size, 128);
    assert_eq!(fmt.bits, 4);

    // Cross-format conversion must fail
    assert!(cfg.to_gptq_format().is_err());
}

// ============================================================================
// 4. Memory estimation: GPTQ vs AWQ
// ============================================================================

#[test]
fn test_memory_estimation_nonzero() {
    let gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();

    assert!(
        gptq.estimated_memory_bytes() > 0,
        "GPTQ memory estimate must be > 0"
    );
    assert!(
        awq.estimated_memory_bytes() > 0,
        "AWQ memory estimate must be > 0"
    );
}

#[test]
fn test_memory_estimation_gptq_equals_awq() {
    // Both use identical INT4 storage format; memory should match.
    let gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert_eq!(
        gptq.estimated_memory_bytes(),
        awq.estimated_memory_bytes(),
        "GPTQ and AWQ share INT4 format -- memory must match"
    );
}

#[test]
fn test_memory_estimation_reasonable_range() {
    // INT4 quantized 30B MoE: expect roughly 10--60 GB.
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let mem_gb = cfg.estimated_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);

    assert!(
        mem_gb > 10.0,
        "INT4 30B MoE should be > 10 GB, got {mem_gb:.2} GB"
    );
    assert!(
        mem_gb < 60.0,
        "INT4 30B MoE should be < 60 GB, got {mem_gb:.2} GB"
    );
}

#[test]
fn test_standalone_estimate_memory_matches_method() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert_eq!(
        estimate_memory_bytes(&cfg),
        cfg.estimated_memory_bytes(),
        "standalone function must agree with method"
    );
}

// ============================================================================
// 5. Group quantization edge cases (group_size=128 boundaries)
// ============================================================================

#[test]
fn test_group_size_128_exact_boundary() {
    // in=128, out=1, group_size=128 => exactly 1 group of 128 elements.
    let total = 128;
    let padded_len = ((total + 7) / 8) * 8; // 128 is divisible by 8
    assert_eq!(padded_len, 128);

    let nibbles = vec![7u32; padded_len];
    let packed = pack_int4(&nibbles);

    let scale = 0.1_f32;
    let zero = 7.0_f32;

    let layer = QuantizedLinearLayer::new(packed, vec![scale], vec![zero], 128, 1, 128)
        .expect("valid construction at group_size=128 boundary");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    assert_eq!(weights.len(), 128);

    // All nibbles = 7, zero = 7.0 => (7 - 7) * 0.1 = 0.0
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.abs() < 1e-7,
            "group_size=128 boundary: elem {i} should be 0, got {w}"
        );
    }
}

#[test]
fn test_group_size_128_two_groups() {
    // in=128, out=2, group_size=128 => 256 elements = 2 groups.
    let nibbles_g0 = vec![0u32; 128]; // group 0: all zeros
    let nibbles_g1 = vec![15u32; 128]; // group 1: all max
    let nibbles: Vec<u32> = [nibbles_g0, nibbles_g1].concat();
    let packed = pack_int4(&nibbles);

    let scales = vec![1.0_f32, 2.0];
    let zeros = vec![0.0_f32, 8.0];

    let layer =
        QuantizedLinearLayer::new(packed, scales, zeros, 128, 2, 128).expect("valid construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    assert_eq!(weights.len(), 256);

    // Group 0: (0 - 0) * 1.0 = 0.0
    for i in 0..128 {
        assert!(
            weights[i].abs() < 1e-7,
            "two-group g0 elem {i}: expected 0, got {}",
            weights[i]
        );
    }
    // Group 1: (15 - 8) * 2.0 = 14.0
    for i in 128..256 {
        assert!(
            (weights[i] - 14.0).abs() < 1e-5,
            "two-group g1 elem {i}: expected 14.0, got {}",
            weights[i]
        );
    }
}

#[test]
fn test_group_size_not_divisible_rejected() {
    // 10 elements with group_size=8 => 10 % 8 != 0, should be rejected.
    let packed = pack_int4(&[0; 16]); // 16 nibbles = 2 u32 words
    let err = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 5, 2, 8);
    assert!(
        err.is_err(),
        "total elements not divisible by group_size should be rejected"
    );
    match err.unwrap_err() {
        QuantizedLayerError::GroupSizeNotDivisible {
            in_features: 5,
            out_features: 2,
            group_size: 8,
        } => {} // expected
        other => panic!("expected GroupSizeNotDivisible, got: {other}"),
    }
}

// ============================================================================
// 6. QuantMethod enum variants
// ============================================================================

#[test]
fn test_quant_method_display() {
    assert_eq!(format!("{}", QuantMethod::Gptq), "GPTQ");
    assert_eq!(format!("{}", QuantMethod::Awq), "AWQ");
}

#[test]
fn test_quant_method_equality_and_inequality() {
    assert_eq!(QuantMethod::Gptq, QuantMethod::Gptq);
    assert_eq!(QuantMethod::Awq, QuantMethod::Awq);
    assert_ne!(QuantMethod::Gptq, QuantMethod::Awq);
}

#[test]
fn test_quant_method_debug() {
    let dbg = format!("{:?}", QuantMethod::Gptq);
    assert!(dbg.contains("Gptq"), "Debug should contain 'Gptq': {dbg}");

    let dbg = format!("{:?}", QuantMethod::Awq);
    assert!(dbg.contains("Awq"), "Debug should contain 'Awq': {dbg}");
}

#[test]
fn test_quant_method_clone_copy() {
    let a = QuantMethod::Gptq;
    let b = a; // Copy
    let c = a.clone(); // Clone
    assert_eq!(a, b);
    assert_eq!(a, c);
}

// ============================================================================
// 7. Config validation (invalid params should error)
// ============================================================================

#[test]
fn test_validate_rejects_8bit_quantization() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.bits = 8;
    assert!(
        cfg.validate().is_err(),
        "only 4-bit quantization is supported"
    );
}

#[test]
fn test_validate_rejects_zero_group_size() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.group_size = 0;
    assert!(
        cfg.validate().is_err(),
        "zero group_size should be rejected"
    );
}

#[test]
fn test_validate_rejects_non_power_of_two_group_size() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.group_size = 100;
    assert!(
        cfg.validate().is_err(),
        "non-power-of-two group_size should be rejected"
    );
}

#[test]
fn test_validate_rejects_awq_with_desc_act() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    cfg.desc_act = true;
    assert!(
        cfg.validate().is_err(),
        "AWQ with desc_act should be rejected"
    );
}

#[test]
fn test_validate_rejects_hidden_not_divisible_by_group_size() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.base.hidden_size = 3500; // not divisible by 128
    cfg.base.num_heads = 25; // keep head_dim integer
    cfg.base.num_kv_heads = 5;
    assert!(
        cfg.validate().is_err(),
        "hidden_size not divisible by group_size should be rejected"
    );
}

#[test]
fn test_validate_rejects_intermediate_not_divisible_by_group_size() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.base.intermediate_size = 2500; // not divisible by 128
    assert!(
        cfg.validate().is_err(),
        "intermediate_size not divisible by group_size should be rejected"
    );
}

#[test]
fn test_validate_accepts_group_size_32() {
    // group_size=32 is valid. hidden=3584 and intermediate=2560 are both
    // divisible by 32.
    let base = Qwen3VLConfig::preset_30b_a3b();
    let cfg = Qwen3VLQuantConfig::new(base, QuantMethod::Gptq, 4, 32, false);
    cfg.validate()
        .expect("group_size=32 with compatible dimensions should validate");
}

#[test]
fn test_validate_accepts_group_size_64() {
    let base = Qwen3VLConfig::preset_30b_a3b();
    let cfg = Qwen3VLQuantConfig::new(base, QuantMethod::Awq, 4, 64, false);
    cfg.validate()
        .expect("group_size=64 with compatible dimensions should validate");
}

#[test]
fn test_validate_rejects_1bit_quantization() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.bits = 1;
    assert!(cfg.validate().is_err(), "1-bit should be rejected");
}

// ============================================================================
// 8. Quantized output bounds (dequantized values should be finite, reasonable)
// ============================================================================

#[test]
fn test_dequantized_values_are_finite() {
    // All 16 INT4 values [0..15], scale=0.1, zero=7.5.
    // Dequant range: (0-7.5)*0.1 = -0.75 to (15-7.5)*0.1 = 0.75.
    let nibbles: Vec<u32> = (0..16).collect();
    let packed = pack_int4(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![0.1; 2], vec![7.5; 2], 4, 4, 8)
        .expect("valid construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.is_finite(),
            "dequantized weight at index {i} must be finite, got {w}"
        );
    }
}

#[test]
fn test_dequantized_values_in_reasonable_range() {
    // With scale=0.5, zero=8, INT4 range [0,15]:
    // min dequant: (0-8)*0.5 = -4.0
    // max dequant: (15-8)*0.5 = 3.5
    let nibbles: Vec<u32> = (0..8).collect();
    let packed = pack_int4(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![0.5], vec![8.0], 8, 1, 8)
        .expect("valid construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            (-4.0..=3.5).contains(&w),
            "weight at {i} should be in [-4.0, 3.5], got {w}"
        );
    }
}

#[test]
fn test_forward_output_values_are_finite() {
    // Construct a small layer and verify the forward output is all finite.
    let nibbles: Vec<u32> = vec![5, 5, 5, 5, 5, 5, 5, 5];
    let packed = pack_int4(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![0.3], vec![5.0], 4, 2, 8)
        .expect("valid construction");

    let input = vec![1.0, -1.0, 0.5, -0.5];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    for (j, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "forward output[{j}] must be finite, got {val}"
        );
    }
}

#[test]
fn test_dequantize_detects_non_finite_scale() {
    // Inf scale produces non-finite dequantized values.
    let packed = pack_int4(&[1, 0, 0, 0, 0, 0, 0, 0]);
    let layer = QuantizedLinearLayer::new(packed, vec![f32::INFINITY], vec![0.0], 4, 2, 8)
        .expect("construction succeeds -- validation is at dequant time");

    let result = layer.dequantize_weights();
    assert!(
        result.is_err(),
        "dequantize should detect non-finite values from Inf scale"
    );
    match result.unwrap_err() {
        QuantizedLayerError::NonFiniteValue { .. } => {}
        other => panic!("expected NonFiniteValue, got: {other}"),
    }
}

#[test]
fn test_dequantize_detects_nan_scale() {
    // NaN scale produces non-finite dequantized values.
    let packed = pack_int4(&[1, 0, 0, 0, 0, 0, 0, 0]);
    let layer = QuantizedLinearLayer::new(packed, vec![f32::NAN], vec![0.0], 4, 2, 8)
        .expect("construction succeeds");

    let result = layer.dequantize_weights();
    assert!(
        result.is_err(),
        "dequantize should detect non-finite values from NaN scale"
    );
}

// ============================================================================
// Additional edge cases: layer construction errors
// ============================================================================

#[test]
fn test_layer_rejects_wrong_packed_weight_length() {
    // 4x2 = 8 elements => 1 packed u32 expected. Provide 3.
    let err = QuantizedLinearLayer::new(vec![0, 0, 0], vec![1.0], vec![0.0], 4, 2, 8);
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::PackedWeightLengthMismatch {
            expected: 1,
            actual: 3,
            ..
        } => {}
        other => panic!("expected PackedWeightLengthMismatch, got: {other}"),
    }
}

#[test]
fn test_layer_rejects_wrong_scales_length() {
    // 8 elements, group_size=8 => 1 group. Provide 2 scales.
    let packed = pack_int4(&[0; 8]);
    let err = QuantizedLinearLayer::new(packed, vec![1.0, 2.0], vec![0.0], 4, 2, 8);
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::ScaleZeroLengthMismatch {
            expected: 1,
            actual: 2,
        } => {}
        other => panic!("expected ScaleZeroLengthMismatch, got: {other}"),
    }
}

#[test]
fn test_layer_rejects_wrong_zeros_length() {
    // 8 elements, group_size=8 => 1 group. Provide 3 zeros.
    let packed = pack_int4(&[0; 8]);
    let err = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0, 1.0, 2.0], 4, 2, 8);
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::ScaleZeroLengthMismatch {
            expected: 1,
            actual: 3,
        } => {}
        other => panic!("expected ScaleZeroLengthMismatch, got: {other}"),
    }
}

#[test]
fn test_forward_rejects_wrong_input_length() {
    let packed = pack_int4(&[0; 8]);
    let layer = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 2, 8)
        .expect("valid construction");

    let err = layer.forward_quantized_linear(&[1.0, 2.0, 3.0]); // expected 4
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::InputLengthMismatch {
            expected: 4,
            actual: 3,
        } => {}
        other => panic!("expected InputLengthMismatch, got: {other}"),
    }
}

// ============================================================================
// Layer accessor methods
// ============================================================================

#[test]
fn test_layer_accessors() {
    let packed = pack_int4(&[0; 8]);
    let layer =
        QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 2, 8).expect("valid layer");

    assert_eq!(layer.in_features(), 4);
    assert_eq!(layer.out_features(), 2);
    assert_eq!(layer.group_size(), 8);
}

// ============================================================================
// QuantizedLayerError display
// ============================================================================

#[test]
fn test_quantized_layer_error_display_messages() {
    let err = QuantizedLayerError::InvalidGroupSize { group_size: 7 };
    let msg = format!("{err}");
    assert!(msg.contains("7"), "should contain the invalid group_size");
    assert!(
        msg.contains("power of two"),
        "should mention power of two: {msg}"
    );

    let err = QuantizedLayerError::InputLengthMismatch {
        expected: 128,
        actual: 64,
    };
    let msg = format!("{err}");
    assert!(msg.contains("128"), "should contain expected: {msg}");
    assert!(msg.contains("64"), "should contain actual: {msg}");

    let err = QuantizedLayerError::NonFiniteValue { index: 42 };
    let msg = format!("{err}");
    assert!(msg.contains("42"), "should contain the index: {msg}");
}
