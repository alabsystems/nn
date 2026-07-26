// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quantized Qwen3-VL-30B-A3B config and weight loading.

use super::*;

// ---------------------------------------------------------------------------
// GPTQ preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_preset_30b_a3b_gptq_validates() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.validate().expect("GPTQ preset should validate");
}

#[test]
fn test_preset_30b_a3b_gptq_quant_params() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert_eq!(cfg.quant_method, QuantMethod::Gptq);
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(cfg.desc_act, "GPTQ preset should use desc_act");
}

#[test]
fn test_preset_30b_a3b_gptq_moe_config() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert!(cfg.is_moe());
    assert_eq!(cfg.num_experts(), 60);
    assert_eq!(cfg.active_experts(), 2);
    assert_eq!(cfg.base.num_layers, 48);
}

#[test]
fn test_preset_30b_a3b_gptq_architecture() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert_eq!(cfg.base.hidden_size, 3584);
    assert_eq!(cfg.base.num_heads, 28);
    assert_eq!(cfg.base.num_kv_heads, 4);
    assert_eq!(cfg.base.intermediate_size, 2560);
    assert_eq!(cfg.base.vocab_size, 152064);
}

// ---------------------------------------------------------------------------
// AWQ preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_preset_30b_a3b_awq_validates() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    cfg.validate().expect("AWQ preset should validate");
}

#[test]
fn test_preset_30b_a3b_awq_quant_params() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert_eq!(cfg.quant_method, QuantMethod::Awq);
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(!cfg.desc_act, "AWQ preset must not use desc_act");
}

#[test]
fn test_preset_30b_a3b_awq_moe_config() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert!(cfg.is_moe());
    assert_eq!(cfg.num_experts(), 60);
    assert_eq!(cfg.active_experts(), 2);
}

// ---------------------------------------------------------------------------
// Format conversion tests
// ---------------------------------------------------------------------------

#[test]
fn test_gptq_to_gptq_format() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let fmt = cfg.to_gptq_format().expect("should convert to GptqFormat");
    assert_eq!(fmt.group_size, 128);
    assert_eq!(fmt.bits, 4);
    assert!(fmt.act_order, "GPTQ desc_act maps to act_order");
}

#[test]
fn test_awq_to_awq_format() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    let fmt = cfg.to_awq_format().expect("should convert to AwqFormat");
    assert_eq!(fmt.group_size, 128);
    assert_eq!(fmt.bits, 4);
}

#[test]
fn test_gptq_to_awq_format_fails() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert!(
        cfg.to_awq_format().is_err(),
        "GPTQ config should not convert to AwqFormat"
    );
}

#[test]
fn test_awq_to_gptq_format_fails() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert!(
        cfg.to_gptq_format().is_err(),
        "AWQ config should not convert to GptqFormat"
    );
}

// ---------------------------------------------------------------------------
// Validation error tests
// ---------------------------------------------------------------------------

#[test]
fn test_validate_rejects_8bit() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.bits = 8;
    assert!(
        cfg.validate().is_err(),
        "8-bit quantization should be rejected"
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
fn test_validate_rejects_hidden_not_divisible_by_group() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.base.hidden_size = 3500; // not divisible by 128
                                 // Also fix num_heads to avoid head_dim validation failure
    cfg.base.num_heads = 25;
    cfg.base.num_kv_heads = 5;
    assert!(
        cfg.validate().is_err(),
        "hidden_size not divisible by group_size should be rejected"
    );
}

#[test]
fn test_validate_rejects_intermediate_not_divisible_by_group() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.base.intermediate_size = 2500; // not divisible by 128
    assert!(
        cfg.validate().is_err(),
        "intermediate_size not divisible by group_size should be rejected"
    );
}

// ---------------------------------------------------------------------------
// MoE parameter tests
// ---------------------------------------------------------------------------

#[test]
fn test_moe_expert_count_matches_spec() {
    // Spec: 30B total, 3B active, 60 experts, top-2 routing
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert_eq!(cfg.base.num_experts, 60);
    assert_eq!(cfg.base.active_experts, 2);
    assert!(cfg.is_moe());
}

#[test]
fn test_gptq_and_awq_share_same_base_architecture() {
    let gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();

    assert_eq!(gptq.base.hidden_size, awq.base.hidden_size);
    assert_eq!(gptq.base.num_heads, awq.base.num_heads);
    assert_eq!(gptq.base.num_kv_heads, awq.base.num_kv_heads);
    assert_eq!(gptq.base.intermediate_size, awq.base.intermediate_size);
    assert_eq!(gptq.base.num_layers, awq.base.num_layers);
    assert_eq!(gptq.base.vocab_size, awq.base.vocab_size);
    assert_eq!(gptq.base.num_experts, awq.base.num_experts);
    assert_eq!(gptq.base.active_experts, awq.base.active_experts);
}

// ---------------------------------------------------------------------------
// Memory estimation tests
// ---------------------------------------------------------------------------

#[test]
fn test_estimated_memory_nonzero() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let mem = cfg.estimated_memory_bytes();
    assert!(mem > 0, "estimated memory must be > 0");
}

#[test]
fn test_estimated_memory_reasonable() {
    // INT4 quantized 30B MoE model: ~60 experts * expert_params in INT4,
    // plus embeddings and LM head in F32. Should be well under full F32 size
    // (which would be ~120GB for 30B params). With INT4 quant on linear
    // layers and F32 embeddings, expect roughly 20-50GB.
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let mem_gb = cfg.estimated_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!(
        mem_gb < 60.0,
        "INT4-quantized 30B MoE should be under 60GB, got {mem_gb:.1}GB"
    );
    // Must be less than half of F32 (which would be ~120GB)
    assert!(
        mem_gb > 5.0,
        "memory estimate should be reasonable (>5GB), got {mem_gb:.1}GB"
    );
}

#[test]
fn test_gptq_and_awq_same_memory_estimate() {
    // Both use same INT4 format, so memory estimates should match
    let gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert_eq!(
        gptq.estimated_memory_bytes(),
        awq.estimated_memory_bytes(),
        "GPTQ and AWQ share the same INT4 format, memory should match"
    );
}

// ---------------------------------------------------------------------------
// QuantMethod display/equality tests
// ---------------------------------------------------------------------------

#[test]
fn test_quant_method_display() {
    assert_eq!(format!("{}", QuantMethod::Gptq), "GPTQ");
    assert_eq!(format!("{}", QuantMethod::Awq), "AWQ");
}

#[test]
fn test_quant_method_equality() {
    assert_eq!(QuantMethod::Gptq, QuantMethod::Gptq);
    assert_eq!(QuantMethod::Awq, QuantMethod::Awq);
    assert_ne!(QuantMethod::Gptq, QuantMethod::Awq);
}

// ---------------------------------------------------------------------------
// Constructor test
// ---------------------------------------------------------------------------

#[test]
fn test_new_constructor() {
    let base = base_30b_a3b_moe();
    let cfg = Qwen3VLQuantConfig::new(base, QuantMethod::Gptq, 4, 128, true);
    assert_eq!(cfg.quant_method, QuantMethod::Gptq);
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(cfg.desc_act);
    assert_eq!(cfg.base.num_experts, 60);
    cfg.validate().expect("constructed config should validate");
}

// ===========================================================================
// QuantizedLinearLayer tests (GPTQ INT4 dequantization + forward pass)
// ===========================================================================

/// Helper: pack INT4 nibbles into u32.
///
/// Takes a slice of values in [0, 15], packs 8 per u32 in little-endian
/// nibble order.
fn pack_int4_values(values: &[u32]) -> Vec<u32> {
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

// ---------------------------------------------------------------------------
// Dequantize tests
// ---------------------------------------------------------------------------

#[test]
fn test_dequantize_known_packed_values() {
    // 4 in_features x 2 out_features = 8 elements = 1 u32 packed word.
    // group_size = 8 => 1 group covering all 8 elements.
    //
    // Pack nibbles: [3, 7, 0, 15, 1, 2, 10, 5]
    let nibbles: Vec<u32> = vec![3, 7, 0, 15, 1, 2, 10, 5];
    let packed = pack_int4_values(&nibbles);
    assert_eq!(packed.len(), 1, "8 nibbles fit in 1 u32");

    let scale = 0.5_f32;
    let zero = 8.0_f32; // center zero at 8 (unsigned GPTQ convention)

    let layer = QuantizedLinearLayer::new(
        packed,
        vec![scale], // 1 group
        vec![zero],  // 1 group
        4,           // in_features
        2,           // out_features
        8,           // group_size
    )
    .expect("valid layer construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    assert_eq!(weights.len(), 8);

    // Expected: (nibble - 8.0) * 0.5
    let expected: Vec<f32> = nibbles.iter().map(|&n| (n as f32 - 8.0) * 0.5).collect();

    for (i, (&got, &exp)) in weights.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "element {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_dequantize_two_groups() {
    // 4 in x 4 out = 16 elements, group_size=8 => 2 groups.
    // Each group has its own scale/zero.
    let nibbles: Vec<u32> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, // group 0
        8, 9, 10, 11, 12, 13, 14, 15, // group 1
    ];
    let packed = pack_int4_values(&nibbles);
    assert_eq!(packed.len(), 2);

    let scales = vec![1.0, 2.0];
    let zeros = vec![0.0, 8.0];

    let layer = QuantizedLinearLayer::new(packed, scales, zeros, 4, 4, 8)
        .expect("valid construction");
    let weights = layer.dequantize_weights().expect("dequantize succeeds");

    // Group 0: (nibble - 0) * 1.0 = nibble as f32
    for i in 0..8 {
        let expected = nibbles[i] as f32 * 1.0;
        assert!(
            (weights[i] - expected).abs() < 1e-6,
            "group0 elem {i}: got {}, exp {}",
            weights[i],
            expected
        );
    }

    // Group 1: (nibble - 8.0) * 2.0
    for i in 8..16 {
        let expected = (nibbles[i] as f32 - 8.0) * 2.0;
        assert!(
            (weights[i] - expected).abs() < 1e-6,
            "group1 elem {i}: got {}, exp {}",
            weights[i],
            expected
        );
    }
}

#[test]
fn test_dequantize_all_zeros() {
    // All nibbles zero, scale=1.0, zero=0.0 => all weights = 0.0
    let packed = vec![0u32; 1]; // 8 zero nibbles
    let layer = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 2, 8)
        .expect("valid construction");

    let weights = layer.dequantize_weights().expect("dequantize succeeds");
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.abs() < 1e-10,
            "all-zero dequant: elem {i} should be 0, got {w}"
        );
    }
}

// ---------------------------------------------------------------------------
// Forward pass shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_quantized_linear_output_shape() {
    // in=4, out=2, group_size=8 => 8 elements = 1 group
    let nibbles: Vec<u32> = vec![8, 8, 8, 8, 8, 8, 8, 8]; // all neutral
    let packed = pack_int4_values(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![1.0], vec![8.0], 4, 2, 8)
        .expect("valid construction");

    let input = vec![1.0_f32; 4];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    assert_eq!(output.len(), 2, "output length should equal out_features");
}

#[test]
fn test_forward_quantized_linear_identity_like() {
    // 2x2 weight matrix that acts like identity (approximately).
    // in=2, out=2, group_size=4 (all 4 elements in one group).
    //
    // We want W = [[1, 0], [0, 1]].
    // With zero=0, scale chosen so nibble * scale = desired value.
    // Use scale = 0.25, then nibble 4 => 4*0.25 = 1.0, nibble 0 => 0.0.
    let nibbles: Vec<u32> = vec![4, 0, 0, 4]; // [1,0,0,1] after dequant
    let packed = pack_int4_values(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![0.25], vec![0.0], 2, 2, 4)
        .expect("valid construction");

    let input = vec![3.0_f32, 7.0_f32];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    // y = input @ W, W=[in,out] = [[1,0],[0,1]]
    // y[0] = 3*1 + 7*0 = 3, y[1] = 3*0 + 7*1 = 7
    assert!(
        (output[0] - 3.0).abs() < 1e-5,
        "output[0] should be ~3.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 7.0).abs() < 1e-5,
        "output[1] should be ~7.0, got {}",
        output[1]
    );
}

#[test]
fn test_forward_quantized_linear_accumulation() {
    // 4 in_features, 1 out_feature, group_size=4.
    // W = [2, 2, 2, 2] => y = sum(input) * 2
    // Use scale=1.0, zero=0.0, nibbles all = 2.
    // 4 in * 1 out = 4 elements, ceil(4/8) = 1 packed u32.
    // Pad to 8 nibbles for packing (trailing zeros don't matter).
    let nibbles: Vec<u32> = vec![2, 2, 2, 2, 0, 0, 0, 0];
    let packed = pack_int4_values(&nibbles);

    let layer = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 1, 4)
        .expect("valid construction");

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = layer
        .forward_quantized_linear(&input)
        .expect("forward succeeds");

    // y[0] = 1*2 + 2*2 + 3*2 + 4*2 = 20
    assert_eq!(output.len(), 1);
    assert!(
        (output[0] - 20.0).abs() < 1e-5,
        "expected 20.0, got {}",
        output[0]
    );
}

// ---------------------------------------------------------------------------
// Input validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_rejects_wrong_input_length() {
    let packed = pack_int4_values(&[0; 8]);
    let layer = QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 2, 8)
        .expect("valid construction");

    let wrong_input = vec![1.0_f32; 3]; // should be 4
    let err = layer.forward_quantized_linear(&wrong_input);
    assert!(err.is_err(), "should reject input of wrong length");

    match err.unwrap_err() {
        QuantizedLayerError::InputLengthMismatch {
            expected: 4,
            actual: 3,
        } => {} // expected
        other => panic!("unexpected error variant: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Memory estimation via standalone function
// ---------------------------------------------------------------------------

#[test]
fn test_estimate_memory_bytes_matches_method() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert_eq!(
        estimate_memory_bytes(&cfg),
        cfg.estimated_memory_bytes(),
        "standalone function must match method"
    );
}

#[test]
fn test_estimate_memory_30b_at_4bits_approximately_15gb() {
    // 30B params at 4 bits = ~30e9 * 0.5 bytes = 15 GB for pure weights.
    // Plus scales/zeros overhead and F32 embeddings, total should be in
    // the 15-50 GB range.
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let mem_gb = estimate_memory_bytes(&cfg) as f64 / (1024.0 * 1024.0 * 1024.0);

    // Lower bound: at least the pure INT4 weight footprint
    // (30B * 0.5 bytes = ~14 GB).
    assert!(
        mem_gb > 10.0,
        "30B at INT4 should be well above 10 GB, got {mem_gb:.2} GB"
    );

    // Upper bound: should be well under full F32 (120 GB).
    assert!(
        mem_gb < 60.0,
        "30B at INT4 with overhead should be under 60 GB, got {mem_gb:.2} GB"
    );
}

// ---------------------------------------------------------------------------
// Group size validation in QuantizedLinearLayer
// ---------------------------------------------------------------------------

#[test]
fn test_layer_rejects_zero_group_size() {
    let err = QuantizedLinearLayer::new(vec![], vec![], vec![], 8, 8, 0);
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::InvalidGroupSize { group_size: 0 } => {}
        other => panic!("expected InvalidGroupSize(0), got: {other}"),
    }
}

#[test]
fn test_layer_rejects_non_power_of_two_group_size() {
    let err = QuantizedLinearLayer::new(vec![], vec![], vec![], 8, 8, 3);
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::InvalidGroupSize { group_size: 3 } => {}
        other => panic!("expected InvalidGroupSize(3), got: {other}"),
    }
}

#[test]
fn test_layer_rejects_wrong_packed_length() {
    // 4 in x 2 out = 8 elements => 1 packed u32. Provide 2.
    let err = QuantizedLinearLayer::new(vec![0, 0], vec![1.0], vec![0.0], 4, 2, 8);
    assert!(err.is_err());
    match err.unwrap_err() {
        QuantizedLayerError::PackedWeightLengthMismatch {
            expected: 1,
            actual: 2,
            ..
        } => {}
        other => panic!("expected PackedWeightLengthMismatch, got: {other}"),
    }
}

#[test]
fn test_layer_rejects_wrong_scale_length() {
    // 8 elements, group_size=8 => 1 group. Provide 2 scales.
    let packed = pack_int4_values(&[0; 8]);
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

// ---------------------------------------------------------------------------
// QuantizedLayerError display
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_layer_error_display() {
    let err = QuantizedLayerError::InvalidGroupSize { group_size: 7 };
    let msg = format!("{err}");
    assert!(
        msg.contains("7"),
        "error message should contain the invalid group_size"
    );
    assert!(
        msg.contains("power of two"),
        "error message should mention power of two"
    );
}

// ---------------------------------------------------------------------------
// Accessor tests
// ---------------------------------------------------------------------------

#[test]
fn test_layer_accessors() {
    let packed = pack_int4_values(&[0; 8]);
    let layer =
        QuantizedLinearLayer::new(packed, vec![1.0], vec![0.0], 4, 2, 8).expect("valid layer");
    assert_eq!(layer.in_features(), 4);
    assert_eq!(layer.out_features(), 2);
    assert_eq!(layer.group_size(), 8);
}
