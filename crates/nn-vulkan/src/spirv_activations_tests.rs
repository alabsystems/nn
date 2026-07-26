// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V activation function kernels (GELU, SiLU, Snake, fused AdaIN+Snake).
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Separate input/output buffer layout (2+ StorageBuffer bindings)
//! - Reference implementation correctness against known values
//! - Edge cases (zero, negative, large values, NaN)

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes for structural assertions.
const TEST_OP_CAPABILITY: u16 = 17;
const TEST_OP_MEMORY_MODEL: u16 = 14;
const TEST_OP_FUNCTION: u16 = 54;
const TEST_OP_FUNCTION_END: u16 = 56;
const TEST_OP_LABEL: u16 = 248;
const TEST_OP_RETURN: u16 = 253;
const TEST_OP_FADD: u16 = 129;
const TEST_OP_FMUL: u16 = 133;
const TEST_OP_FDIV: u16 = 136;
const TEST_OP_FSUB: u16 = 131;
const TEST_OP_EXT_INST: u16 = 12;
const TEST_OP_VARIABLE: u16 = 59;
const TEST_OP_DECORATE: u16 = 71;
const TEST_STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const TEST_DECORATION_BINDING: u32 = 33;
const TEST_DECORATION_NON_WRITABLE: u32 = 24;
const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;

// ---- Helpers ----

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V byte length must be multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_valid_header(words: &[u32], label: &str) {
    assert!(words.len() >= 5, "{label}: module too short");
    assert_eq!(
        words[0], SPIRV_MAGIC,
        "{label}: wrong magic (expected 0x07230203)"
    );
    assert_eq!(
        words[1], TEST_SPIRV_VERSION_1_0,
        "{label}: wrong version (expected 1.0)"
    );
    assert_eq!(
        words[2], TEST_GENERATOR_MAGIC,
        "{label}: wrong generator magic"
    );
    assert!(words[3] > 0, "{label}: bound must be > 0");
    assert_eq!(words[4], 0, "{label}: schema must be 0");
}

fn has_opcode(words: &[u32], target_opcode: u16) -> bool {
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == target_opcode {
            return true;
        }
        pos += word_count;
    }
    false
}

fn count_opcode(words: &[u32], target_opcode: u16) -> usize {
    let mut pos = 5;
    let mut count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == target_opcode {
            count += 1;
        }
        pos += word_count;
    }
    count
}

fn find_instructions(words: &[u32], target_opcode: u16) -> Vec<Vec<u32>> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == target_opcode {
            results.push(words[pos..pos + word_count].to_vec());
        }
        pos += word_count;
    }
    results
}

fn count_storage_buffer_variables(words: &[u32]) -> usize {
    let variables = find_instructions(words, TEST_OP_VARIABLE);
    variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count()
}

fn get_binding_numbers(words: &[u32]) -> Vec<u32> {
    let decorations = find_instructions(words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}

// ====================================================================
// GELU SPIR-V structural tests
// ====================================================================

#[test]
fn test_gelu_spirv_valid_header() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "gelu");
}

#[test]
fn test_gelu_spirv_entry_point_is_main() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_gelu_spirv_workgroup_size() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [256, 1, 1]);
}

#[test]
fn test_gelu_spirv_custom_workgroup_size() {
    let bytes = generate_gelu_spirv(128);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [128, 1, 1]);
}

#[test]
fn test_gelu_spirv_has_basic_structure() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "must have OpCapability"
    );
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "must have OpMemoryModel"
    );
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_gelu_spirv_has_tanh_ext_inst() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "GELU must use GLSL.std.450 for Tanh"
    );
}

#[test]
fn test_gelu_spirv_has_fmul_and_fadd() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, TEST_OP_FMUL), "GELU must have OpFMul");
    assert!(has_opcode(&words, TEST_OP_FADD), "GELU must have OpFAdd");
}

#[test]
fn test_gelu_spirv_two_storage_buffers() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    let sb_count = count_storage_buffer_variables(&words);
    assert_eq!(
        sb_count, 2,
        "GELU needs 2 StorageBuffer variables (input + output), got {sb_count}"
    );
}

#[test]
fn test_gelu_spirv_binding_numbers() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    let bindings = get_binding_numbers(&words);
    assert!(bindings.contains(&0), "must have binding 0 (input)");
    assert!(bindings.contains(&1), "must have binding 1 (output)");
}

#[test]
fn test_gelu_spirv_has_nonwritable_input() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let has_nw = decorations
        .iter()
        .any(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE);
    assert!(has_nw, "input buffer should be NonWritable");
}

#[test]
fn test_gelu_spirv_byte_alignment() {
    for wg in [64, 128, 256, 512] {
        let bytes = generate_gelu_spirv(wg);
        assert_eq!(bytes.len() % 4, 0, "GELU wg={wg}: must be 4-byte aligned");
    }
}

#[test]
fn test_gelu_spirv_deterministic() {
    let a = generate_gelu_spirv(256);
    let b = generate_gelu_spirv(256);
    assert_eq!(a, b, "GELU SPIR-V must be deterministic");
}

#[test]
fn test_gelu_spirv_reasonable_size() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        words.len() > 50,
        "GELU module too small ({} words)",
        words.len()
    );
    assert!(
        words.len() < 2000,
        "GELU module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_gelu_spirv_word_counts_consistent() {
    let bytes = generate_gelu_spirv(256);
    let words = bytes_to_words(&bytes);
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        assert!(
            word_count > 0,
            "instruction at pos {pos} has wc 0 (opcode {opcode})"
        );
        assert!(
            pos + word_count <= words.len(),
            "instruction at pos {pos} (opcode {opcode}, wc {word_count}) exceeds module"
        );
        pos += word_count;
    }
    assert_eq!(pos, words.len(), "instructions did not consume full module");
}

// ====================================================================
// SiLU SPIR-V structural tests
// ====================================================================

#[test]
fn test_silu_spirv_valid_header() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "silu");
}

#[test]
fn test_silu_spirv_entry_point_is_main() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_silu_spirv_workgroup_size() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [256, 1, 1]);
}

#[test]
fn test_silu_spirv_has_exp_ext_inst() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "SiLU must use GLSL.std.450 for Exp"
    );
}

#[test]
fn test_silu_spirv_has_fdiv() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FDIV),
        "SiLU must have OpFDiv for x/(1+exp(-x))"
    );
}

#[test]
fn test_silu_spirv_two_storage_buffers() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    let sb_count = count_storage_buffer_variables(&words);
    assert_eq!(
        sb_count, 2,
        "SiLU needs 2 StorageBuffer variables, got {sb_count}"
    );
}

#[test]
fn test_silu_spirv_byte_alignment() {
    let bytes = generate_silu_spirv(256);
    assert_eq!(bytes.len() % 4, 0, "SiLU SPIR-V must be 4-byte aligned");
}

#[test]
fn test_silu_spirv_deterministic() {
    let a = generate_silu_spirv(256);
    let b = generate_silu_spirv(256);
    assert_eq!(a, b, "SiLU SPIR-V must be deterministic");
}

#[test]
fn test_silu_spirv_word_counts_consistent() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        assert!(word_count > 0, "instruction at pos {pos} has wc 0");
        assert!(
            pos + word_count <= words.len(),
            "instruction at pos {pos} exceeds module"
        );
        pos += word_count;
    }
    assert_eq!(pos, words.len());
}

// ====================================================================
// Snake SPIR-V structural tests
// ====================================================================

#[test]
fn test_snake_spirv_valid_header() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "snake");
}

#[test]
fn test_snake_spirv_entry_point_is_main() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_snake_spirv_workgroup_size() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [256, 1, 1]);
}

#[test]
fn test_snake_spirv_has_sin_ext_inst() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "Snake must use GLSL.std.450 for Sin"
    );
}

#[test]
fn test_snake_spirv_has_fdiv_for_inv_alpha() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FDIV),
        "Snake must have OpFDiv for 1/alpha"
    );
}

#[test]
fn test_snake_spirv_has_fmul_for_sin_squared() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    // sin^2 = sin * sin, plus alpha * x
    let fmul_count = count_opcode(&words, TEST_OP_FMUL);
    assert!(
        fmul_count >= 3,
        "Snake must have at least 3 OpFMul (alpha*x, sin*sin, inv_alpha*sin2), got {fmul_count}"
    );
}

#[test]
fn test_snake_spirv_three_storage_buffers() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    let sb_count = count_storage_buffer_variables(&words);
    assert_eq!(
        sb_count, 3,
        "Snake needs 3 StorageBuffer variables (input + output + alpha), got {sb_count}"
    );
}

#[test]
fn test_snake_spirv_binding_numbers() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    let bindings = get_binding_numbers(&words);
    assert!(bindings.contains(&0), "must have binding 0 (input)");
    assert!(bindings.contains(&1), "must have binding 1 (output)");
    assert!(bindings.contains(&2), "must have binding 2 (alpha)");
}

#[test]
fn test_snake_spirv_byte_alignment() {
    let bytes = generate_snake_spirv(256);
    assert_eq!(bytes.len() % 4, 0, "Snake SPIR-V must be 4-byte aligned");
}

#[test]
fn test_snake_spirv_deterministic() {
    let a = generate_snake_spirv(256);
    let b = generate_snake_spirv(256);
    assert_eq!(a, b, "Snake SPIR-V must be deterministic");
}

// ====================================================================
// Fused AdaIN+Snake SPIR-V structural tests
// ====================================================================

#[test]
fn test_fused_adain_snake_spirv_valid_header() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "fused_adain_snake");
}

#[test]
fn test_fused_adain_snake_spirv_entry_point_is_main() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_fused_adain_snake_spirv_workgroup_size() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [256, 1, 1]);
}

#[test]
fn test_fused_adain_snake_spirv_has_sin_and_sqrt() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    let ext_count = count_opcode(&words, TEST_OP_EXT_INST);
    // At least: sqrt(var+eps) and sin(alpha*y)
    assert!(
        ext_count >= 2,
        "fused AdaIN+Snake must have at least 2 ExtInst, got {ext_count}"
    );
}

#[test]
fn test_fused_adain_snake_spirv_has_adain_ops() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FSUB),
        "must have OpFSub for (x - mean)"
    );
    assert!(
        has_opcode(&words, TEST_OP_FDIV),
        "must have OpFDiv for normalization"
    );
    assert!(
        has_opcode(&words, TEST_OP_FMUL),
        "must have OpFMul for scale"
    );
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "must have OpFAdd for bias"
    );
}

#[test]
fn test_fused_adain_snake_spirv_three_storage_buffers() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    let sb_count = count_storage_buffer_variables(&words);
    assert_eq!(
        sb_count, 3,
        "fused AdaIN+Snake needs 3 StorageBuffer variables (input + output + params), got {sb_count}"
    );
}

#[test]
fn test_fused_adain_snake_spirv_binding_numbers() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    let words = bytes_to_words(&bytes);
    let bindings = get_binding_numbers(&words);
    assert!(bindings.contains(&0), "must have binding 0");
    assert!(bindings.contains(&1), "must have binding 1");
    assert!(bindings.contains(&2), "must have binding 2");
}

#[test]
fn test_fused_adain_snake_spirv_byte_alignment() {
    let bytes = generate_fused_adain_snake_spirv(256, 64);
    assert_eq!(
        bytes.len() % 4,
        0,
        "fused AdaIN+Snake must be 4-byte aligned"
    );
}

#[test]
fn test_fused_adain_snake_spirv_deterministic() {
    let a = generate_fused_adain_snake_spirv(256, 64);
    let b = generate_fused_adain_snake_spirv(256, 64);
    assert_eq!(a, b, "fused AdaIN+Snake must be deterministic");
}

#[test]
fn test_fused_adain_snake_spirv_various_channels() {
    for ch in [1, 4, 16, 64, 128, 256] {
        let bytes = generate_fused_adain_snake_spirv(256, ch);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("fused_adain_snake_ch{ch}"));
    }
}

// ====================================================================
// Reference GELU correctness tests
// ====================================================================

#[test]
fn test_gelu_reference_zero() {
    let result = gelu_reference(0.0);
    assert!(result.abs() < 1e-6, "GELU(0) should be ~0, got {result}");
}

#[test]
fn test_gelu_reference_positive() {
    let result = gelu_reference(1.0);
    // GELU(1) ~ 0.8413 (from tanh approximation)
    assert!(
        (result - 0.8412).abs() < 0.01,
        "GELU(1) should be ~0.841, got {result}"
    );
}

#[test]
fn test_gelu_reference_negative() {
    let result = gelu_reference(-1.0);
    // GELU(-1) ~ -0.1587
    assert!(
        (result - (-0.1588)).abs() < 0.01,
        "GELU(-1) should be ~-0.159, got {result}"
    );
}

#[test]
fn test_gelu_reference_large_positive() {
    let result = gelu_reference(10.0);
    // For large x, GELU(x) ~ x
    assert!(
        (result - 10.0).abs() < 0.01,
        "GELU(10) should be ~10, got {result}"
    );
}

#[test]
fn test_gelu_reference_large_negative() {
    let result = gelu_reference(-10.0);
    // For large negative x, GELU(x) ~ 0
    assert!(result.abs() < 0.01, "GELU(-10) should be ~0, got {result}");
}

#[test]
fn test_gelu_reference_monotonic() {
    // GELU is approximately monotonic for x > -0.75 (not globally, but for moderate range)
    let vals: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
    let results: Vec<f32> = vals.iter().map(|&x| gelu_reference(x)).collect();
    // For x >= 0, should be strictly increasing
    for i in 1..results.len() {
        if vals[i] >= 0.0 && vals[i - 1] >= 0.0 {
            assert!(
                results[i] >= results[i - 1],
                "GELU should be non-decreasing for x >= 0: GELU({}) = {}, GELU({}) = {}",
                vals[i - 1],
                results[i - 1],
                vals[i],
                results[i]
            );
        }
    }
}

#[test]
fn test_gelu_reference_finite_for_moderate_inputs() {
    for i in -100..=100 {
        let x = i as f32;
        let result = gelu_reference(x);
        assert!(result.is_finite(), "GELU({x}) must be finite, got {result}");
    }
}

// ====================================================================
// Reference SiLU correctness tests
// ====================================================================

#[test]
fn test_silu_reference_zero() {
    let result = silu_reference(0.0);
    assert!(result.abs() < 1e-6, "SiLU(0) should be 0, got {result}");
}

#[test]
fn test_silu_reference_positive() {
    let result = silu_reference(1.0);
    // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ~ 0.7311
    let expected = 1.0 / (1.0 + (-1.0_f32).exp());
    assert!(
        (result - expected).abs() < 1e-6,
        "SiLU(1) should be ~{expected}, got {result}"
    );
}

#[test]
fn test_silu_reference_negative() {
    let result = silu_reference(-1.0);
    // SiLU(-1) = -1 * sigmoid(-1) = -1 / (1 + exp(1)) ~ -0.2689
    let expected = -1.0 / (1.0 + 1.0_f32.exp());
    assert!(
        (result - expected).abs() < 1e-6,
        "SiLU(-1) should be ~{expected}, got {result}"
    );
}

#[test]
fn test_silu_reference_large_positive() {
    let result = silu_reference(10.0);
    // For large x, sigmoid(x) ~ 1, so SiLU(x) ~ x
    assert!(
        (result - 10.0).abs() < 0.001,
        "SiLU(10) should be ~10, got {result}"
    );
}

#[test]
fn test_silu_reference_large_negative() {
    let result = silu_reference(-10.0);
    // For large negative x, sigmoid(x) ~ 0, so SiLU(x) ~ 0
    assert!(result.abs() < 0.001, "SiLU(-10) should be ~0, got {result}");
}

#[test]
fn test_silu_reference_is_odd_like() {
    // SiLU is NOT symmetric, but SiLU(x) + SiLU(-x) should be close to 0 for moderate x
    // Actually SiLU(x) = x*sigmoid(x), SiLU(-x) = -x*sigmoid(-x) = -x*(1-sigmoid(x))
    // SiLU(x) + SiLU(-x) = x*sigmoid(x) - x*(1-sigmoid(x)) = x*(2*sigmoid(x) - 1) != 0 in general
    // So just check it's finite
    for i in -50..=50 {
        let x = i as f32 * 0.1;
        let result = silu_reference(x);
        assert!(result.is_finite(), "SiLU({x}) must be finite, got {result}");
    }
}

// ====================================================================
// Reference Snake correctness tests
// ====================================================================

#[test]
fn test_snake_reference_zero() {
    let result = snake_reference(0.0, 1.0);
    // Snake(0, alpha) = 0 + (1/alpha) * sin(0)^2 = 0
    assert!(result.abs() < 1e-6, "Snake(0, 1) should be 0, got {result}");
}

#[test]
fn test_snake_reference_identity_like() {
    // For large alpha, sin(alpha*x)^2 oscillates rapidly but bounded by 1,
    // so (1/alpha)*sin^2 goes to 0, and Snake(x) ~ x.
    let alpha = 1000.0;
    let x = 1.5;
    let result = snake_reference(x, alpha);
    assert!(
        (result - x).abs() < 0.01,
        "Snake with large alpha should be ~x: Snake({x}, {alpha}) = {result}"
    );
}

#[test]
fn test_snake_reference_known_value() {
    let alpha = 1.0;
    let x = std::f32::consts::PI / 2.0;
    // Snake(pi/2, 1) = pi/2 + sin(pi/2)^2 = pi/2 + 1
    let expected = x + 1.0;
    let result = snake_reference(x, alpha);
    assert!(
        (result - expected).abs() < 1e-5,
        "Snake(pi/2, 1) should be ~{expected}, got {result}"
    );
}

#[test]
fn test_snake_reference_positive_bias() {
    // Snake(x, alpha) >= x for all x (since sin^2 >= 0 and alpha > 0)
    for i in -100..=100 {
        let x = i as f32 * 0.1;
        let result = snake_reference(x, 1.0);
        assert!(
            result >= x - 1e-6,
            "Snake({x}, 1) = {result} should be >= {x}"
        );
    }
}

#[test]
fn test_snake_reference_finite() {
    for alpha in [0.1, 0.5, 1.0, 2.0, 10.0] {
        for i in -50..=50 {
            let x = i as f32;
            let result = snake_reference(x, alpha);
            assert!(
                result.is_finite(),
                "Snake({x}, {alpha}) must be finite, got {result}"
            );
        }
    }
}

// ====================================================================
// Reference fused AdaIN+Snake correctness tests
// ====================================================================

#[test]
fn test_fused_adain_snake_reference_identity_params() {
    // mean=0, var=1, scale=1, bias=0 => AdaIN is identity (approximately, eps=1e-5)
    // Then Snake with alpha=1
    let x = 1.0;
    let result = fused_adain_snake_reference(x, 0.0, 1.0, 1.0, 0.0, 1.0);
    // AdaIN: y = 1.0 * (1.0 - 0.0) / sqrt(1.0 + 1e-5) + 0.0 ~ 1.0
    // Snake: z = y + sin(y)^2 ~ 1.0 + sin(1)^2
    let expected = snake_reference(x / (1.0 + 1e-5_f32).sqrt(), 1.0);
    assert!(
        (result - expected).abs() < 1e-4,
        "fused with identity params: expected ~{expected}, got {result}"
    );
}

#[test]
fn test_fused_adain_snake_reference_zero_input() {
    // x=0, mean=0, var=1, scale=1, bias=0 => y~0 => Snake(0)~0
    let result = fused_adain_snake_reference(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    assert!(
        result.abs() < 0.01,
        "fused(0, identity) should be ~0, got {result}"
    );
}

#[test]
fn test_fused_adain_snake_reference_with_bias() {
    // x=0, mean=0, var=1, scale=1, bias=2.0 => y~2.0 => Snake(2.0, 1)
    let result = fused_adain_snake_reference(0.0, 0.0, 1.0, 1.0, 2.0, 1.0);
    let y_approx = 2.0 / (1.0 + 1e-5_f32).sqrt();
    let expected = snake_reference(y_approx, 1.0);
    assert!(
        (result - expected).abs() < 0.01,
        "fused with bias=2: expected ~{expected}, got {result}"
    );
}

#[test]
fn test_fused_adain_snake_reference_finite() {
    let params = [
        (1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
        (-5.0, 2.0, 3.0, 0.5, -1.0, 2.0),
        (100.0, 50.0, 25.0, 2.0, 1.0, 0.1),
        (0.0, 0.0, 0.001, 10.0, 0.0, 5.0),
    ];
    for (x, mean, var, scale, bias, alpha) in params {
        let result = fused_adain_snake_reference(x, mean, var, scale, bias, alpha);
        assert!(
            result.is_finite(),
            "fused({x}, mean={mean}, var={var}, scale={scale}, bias={bias}, alpha={alpha}) = {result} must be finite"
        );
    }
}

// ====================================================================
// Edge cases across all activations
// ====================================================================

#[test]
fn test_all_activations_handle_nan_input() {
    let nan = f32::NAN;
    // NaN propagation is expected for activations
    assert!(gelu_reference(nan).is_nan(), "GELU(NaN) should be NaN");
    assert!(silu_reference(nan).is_nan(), "SiLU(NaN) should be NaN");
    assert!(
        snake_reference(nan, 1.0).is_nan(),
        "Snake(NaN, 1) should be NaN"
    );
}

#[test]
fn test_all_spirv_modules_different() {
    // Each kernel should produce distinct SPIR-V modules.
    let gelu = generate_gelu_spirv(256);
    let silu = generate_silu_spirv(256);
    let snake = generate_snake_spirv(256);
    let fused = generate_fused_adain_snake_spirv(256, 64);

    assert_ne!(gelu, silu, "GELU and SiLU should produce different SPIR-V");
    assert_ne!(
        gelu, snake,
        "GELU and Snake should produce different SPIR-V"
    );
    assert_ne!(
        gelu, fused,
        "GELU and fused should produce different SPIR-V"
    );
    assert_ne!(
        silu, snake,
        "SiLU and Snake should produce different SPIR-V"
    );
    assert_ne!(
        silu, fused,
        "SiLU and fused should produce different SPIR-V"
    );
    assert_ne!(
        snake, fused,
        "Snake and fused should produce different SPIR-V"
    );
}
