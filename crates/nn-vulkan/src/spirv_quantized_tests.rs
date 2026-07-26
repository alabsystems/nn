// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External tests for `spirv_quantized` module.
//!
//! Tests the public API from outside the module, exercising SPIR-V generation
//! for INT8 quantize/dequantize kernels, plus CPU reference correctness.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_quantized::{
    dequantize_reference, generate_dequantize_int8_spirv, generate_quantize_f32_to_int8_spirv,
    quantize_reference, QUANTIZED_WORKGROUP_SIZE,
};

// ---- test_quantize_f32_to_int8_spirv_valid ----

#[test]
fn test_quantize_f32_to_int8_spirv_valid() {
    let words = generate_quantize_f32_to_int8_spirv(1024);
    assert!(words.len() >= 5, "quantize module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "quantize: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("quantize must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("quantize must have workgroup size");
    assert_eq!(wg, [QUANTIZED_WORKGROUP_SIZE, 1, 1]);
}

// ---- test_dequantize_int8_spirv_valid ----

#[test]
fn test_dequantize_int8_spirv_valid() {
    let words = generate_dequantize_int8_spirv(1024);
    assert!(words.len() >= 5, "dequantize module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "dequantize: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("dequantize must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("dequantize must have workgroup size");
    assert_eq!(wg, [QUANTIZED_WORKGROUP_SIZE, 1, 1]);
}

// ---- test_quantize_reference_basic ----

#[test]
fn test_quantize_reference_basic() {
    // scale=0.5, zero_point=0: quantize(5.0) = round(5.0/0.5) + 0 = 10
    let data = vec![5.0f32, 10.0, -5.0];
    let result = quantize_reference(&data, 0.5, 0);
    assert_eq!(result, vec![10i8, 20, -10]);
}

// ---- test_dequantize_reference_basic ----

#[test]
fn test_dequantize_reference_basic() {
    // scale=0.5, zero_point=0: dequantize(10) = 0.5 * (10 - 0) = 5.0
    let data = vec![10i8, 20, -10];
    let result = dequantize_reference(&data, 0.5, 0);
    assert!((result[0] - 5.0).abs() < 1e-6);
    assert!((result[1] - 10.0).abs() < 1e-6);
    assert!((result[2] - (-5.0)).abs() < 1e-6);
}

// ---- test_quantize_dequantize_roundtrip ----

#[test]
fn test_quantize_dequantize_roundtrip() {
    let scale = 0.1;
    let zero_point = 0i8;
    // Values within representable range: [-12.8, 12.7] for scale=0.1
    let original = vec![0.0f32, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0];
    let quantized = quantize_reference(&original, scale, zero_point);
    let dequantized = dequantize_reference(&quantized, scale, zero_point);

    for (i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
        let error = (orig - deq).abs();
        assert!(
            error <= scale / 2.0 + 1e-6,
            "Roundtrip error at index {i}: orig={orig}, deq={deq}, error={error}, max_allowed={}",
            scale / 2.0
        );
    }
}

// ---- test_quantize_reference_clamp ----

#[test]
fn test_quantize_reference_clamp() {
    // Values outside the representable INT8 range should be clamped.
    let scale = 1.0;
    let zero_point = 0i8;

    // 1000.0 / 1.0 = 1000 -> clamped to 127
    let overflow = quantize_reference(&[1000.0f32], scale, zero_point);
    assert_eq!(overflow[0], 127, "positive overflow must clamp to 127");

    // -1000.0 / 1.0 = -1000 -> clamped to -128
    let underflow = quantize_reference(&[-1000.0f32], scale, zero_point);
    assert_eq!(underflow[0], -128, "negative overflow must clamp to -128");
}
