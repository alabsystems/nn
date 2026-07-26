// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V fused residual kernels.
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Multiple StorageBuffer bindings for two-input and three-input kernels
//! - Reference implementation correctness against known values
//! - Edge cases (zero, negative, large values)

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const TEST_OP_FADD: u16 = 129;
const TEST_OP_FMUL: u16 = 133;
const TEST_OP_FUNCTION: u16 = 54;
const TEST_OP_FUNCTION_END: u16 = 56;
const TEST_OP_RETURN: u16 = 253;
const TEST_OP_EXT_INST: u16 = 12;
const TEST_OP_DECORATE: u16 = 71;
const TEST_DECORATION_BINDING: u32 = 33;

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
    assert_eq!(words[0], SPIRV_MAGIC, "{label}: wrong magic");
    assert_eq!(words[1], TEST_SPIRV_VERSION_1_0, "{label}: wrong version");
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

fn count_storage_buffer_bindings(words: &[u32]) -> usize {
    let mut pos = 5;
    let mut binding_count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > words.len() {
            break;
        }
        if opcode == TEST_OP_DECORATE && word_count >= 4 {
            let decoration = words[pos + 2];
            if decoration == TEST_DECORATION_BINDING {
                binding_count += 1;
            }
        }
        pos += word_count;
    }
    binding_count
}

// ---- Residual Add tests ----

#[test]
fn test_residual_add_spirv_valid_header() {
    let spirv = generate_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert_valid_header(&words, "residual_add");
}

#[test]
fn test_residual_add_spirv_entry_point() {
    let spirv = generate_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    let name = find_entry_point_name(&words);
    assert_eq!(name.as_deref(), Some("main"));
}

#[test]
fn test_residual_add_spirv_workgroup_size() {
    let spirv = generate_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_RESIDUAL_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_residual_add_spirv_has_fadd() {
    let spirv = generate_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert!(has_opcode(&words, TEST_OP_FADD), "should contain FAdd");
}

#[test]
fn test_residual_add_spirv_three_bindings() {
    let spirv = generate_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    // x (binding 0), residual (binding 1), output (binding 2), plus gid
    assert!(count_storage_buffer_bindings(&words) >= 3);
}

#[test]
fn test_residual_add_spirv_has_function_structure() {
    let spirv = generate_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert!(has_opcode(&words, TEST_OP_FUNCTION));
    assert!(has_opcode(&words, TEST_OP_FUNCTION_END));
    assert!(has_opcode(&words, TEST_OP_RETURN));
}

#[test]
fn test_residual_add_reference_basic() {
    assert_eq!(residual_add_reference(1.0, 2.0), 3.0);
    assert_eq!(residual_add_reference(-1.0, 1.0), 0.0);
    assert_eq!(residual_add_reference(0.0, 0.0), 0.0);
}

// ---- Residual Add + ReLU tests ----

#[test]
fn test_residual_add_relu_spirv_valid_header() {
    let spirv = generate_residual_add_relu_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert_valid_header(&words, "residual_add_relu");
}

#[test]
fn test_residual_add_relu_spirv_has_ext_inst() {
    let spirv = generate_residual_add_relu_spirv(1024);
    let words = bytes_to_words(&spirv);
    // FMax from GLSL.std.450 for ReLU
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "should use GLSL.std.450 FMax"
    );
}

#[test]
fn test_residual_add_relu_spirv_workgroup_size() {
    let spirv = generate_residual_add_relu_spirv(1024);
    let words = bytes_to_words(&spirv);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_RESIDUAL_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_residual_add_relu_reference_positive() {
    assert_eq!(residual_add_relu_reference(1.0, 2.0), 3.0);
}

#[test]
fn test_residual_add_relu_reference_negative_clamps() {
    assert_eq!(residual_add_relu_reference(-5.0, 2.0), 0.0);
}

#[test]
fn test_residual_add_relu_reference_zero() {
    assert_eq!(residual_add_relu_reference(-1.0, 1.0), 0.0);
}

// ---- Residual Add + GELU tests ----

#[test]
fn test_residual_add_gelu_spirv_valid_header() {
    let spirv = generate_residual_add_gelu_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert_valid_header(&words, "residual_add_gelu");
}

#[test]
fn test_residual_add_gelu_spirv_has_tanh() {
    let spirv = generate_residual_add_gelu_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "should use GLSL.std.450 Tanh"
    );
    assert!(
        has_opcode(&words, TEST_OP_FMUL),
        "should have FMul for GELU"
    );
}

#[test]
fn test_residual_add_gelu_spirv_workgroup_size() {
    let spirv = generate_residual_add_gelu_spirv(1024);
    let words = bytes_to_words(&spirv);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_RESIDUAL_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_residual_add_gelu_reference_zero_input() {
    // GELU(0) = 0
    let result = residual_add_gelu_reference(0.0, 0.0);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_residual_add_gelu_reference_positive() {
    // GELU(1.0) ~ 0.8413
    let result = residual_add_gelu_reference(0.5, 0.5);
    assert!(
        (result - 0.8413).abs() < 0.01,
        "GELU(1.0) should be ~0.8413, got {result}"
    );
}

#[test]
fn test_residual_add_gelu_reference_negative() {
    // GELU(-1.0) ~ -0.1587
    let result = residual_add_gelu_reference(-0.5, -0.5);
    assert!(
        (result - (-0.1587)).abs() < 0.01,
        "GELU(-1.0) should be ~-0.1587, got {result}"
    );
}

// ---- Bias + Residual Add tests ----

#[test]
fn test_bias_residual_add_spirv_valid_header() {
    let spirv = generate_bias_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    assert_valid_header(&words, "bias_residual_add");
}

#[test]
fn test_bias_residual_add_spirv_four_bindings() {
    let spirv = generate_bias_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    // x (0), bias (1), residual (2), output (3), plus gid
    assert!(count_storage_buffer_bindings(&words) >= 4);
}

#[test]
fn test_bias_residual_add_spirv_workgroup_size() {
    let spirv = generate_bias_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_RESIDUAL_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_bias_residual_add_spirv_entry_point() {
    let spirv = generate_bias_residual_add_spirv(1024);
    let words = bytes_to_words(&spirv);
    let name = find_entry_point_name(&words);
    assert_eq!(name.as_deref(), Some("main"));
}

#[test]
fn test_bias_residual_add_reference_basic() {
    assert_eq!(bias_residual_add_reference(1.0, 0.5, 2.0), 3.5);
    assert_eq!(bias_residual_add_reference(0.0, 0.0, 0.0), 0.0);
    assert_eq!(bias_residual_add_reference(-1.0, 0.5, 0.5), 0.0);
}

// ---- Cross-cutting tests ----

#[test]
fn test_all_residual_spirv_byte_alignment() {
    let spirv_add = generate_residual_add_spirv(1024);
    let spirv_relu = generate_residual_add_relu_spirv(1024);
    let spirv_gelu = generate_residual_add_gelu_spirv(1024);
    let spirv_bias = generate_bias_residual_add_spirv(1024);

    assert_eq!(spirv_add.len() % 4, 0, "residual_add not 4-byte aligned");
    assert_eq!(
        spirv_relu.len() % 4,
        0,
        "residual_add_relu not 4-byte aligned"
    );
    assert_eq!(
        spirv_gelu.len() % 4,
        0,
        "residual_add_gelu not 4-byte aligned"
    );
    assert_eq!(
        spirv_bias.len() % 4,
        0,
        "bias_residual_add not 4-byte aligned"
    );
}

#[test]
fn test_residual_add_gelu_larger_than_plain_add() {
    // GELU version should have more instructions than plain add.
    let spirv_add = generate_residual_add_spirv(1024);
    let spirv_gelu = generate_residual_add_gelu_spirv(1024);
    assert!(
        spirv_gelu.len() > spirv_add.len(),
        "GELU variant should be larger than plain add"
    );
}
