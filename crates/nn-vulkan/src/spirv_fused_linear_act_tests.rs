// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V fused linear + activation kernels.
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Loop structure for reduction (OpLoopMerge, OpPhi)
//! - Buffer binding count (4 bindings: input, weight, bias, output)
//! - Reference implementation correctness

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const TEST_OP_FUNCTION: u16 = 54;
const TEST_OP_FUNCTION_END: u16 = 56;
const TEST_OP_RETURN: u16 = 253;
const TEST_OP_LOOP_MERGE: u16 = 246;
const TEST_OP_PHI: u16 = 245;
const TEST_OP_EXT_INST: u16 = 12;
const TEST_OP_FMUL: u16 = 133;
const _TEST_OP_FADD: u16 = 129;
const TEST_OP_FDIV: u16 = 136;
const TEST_OP_DECORATE: u16 = 71;
const TEST_DECORATION_BINDING: u32 = 33;

#[allow(dead_code)]
fn bytes_to_words(words: &[u32]) -> Vec<u32> {
    // For these tests, the generator returns Vec<u32> directly.
    words.to_vec()
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

fn count_bindings(words: &[u32]) -> usize {
    let mut pos = 5;
    let mut count = 0;
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
                count += 1;
            }
        }
        pos += word_count;
    }
    count
}

// ---- Linear + ReLU tests ----

#[test]
fn test_fused_linear_relu_spirv_valid_header() {
    let words = generate_fused_linear_relu_spirv(4, 3);
    assert_valid_header(&words, "linear_relu");
}

#[test]
fn test_fused_linear_relu_spirv_entry_point() {
    let words = generate_fused_linear_relu_spirv(4, 3);
    let name = find_entry_point_name(&words);
    assert_eq!(name.as_deref(), Some("main"));
}

#[test]
fn test_fused_linear_relu_spirv_workgroup_size() {
    let words = generate_fused_linear_relu_spirv(4, 3);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_LINEAR_ACT_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_fused_linear_relu_spirv_has_loop() {
    let words = generate_fused_linear_relu_spirv(4, 3);
    assert!(
        has_opcode(&words, TEST_OP_LOOP_MERGE),
        "should have reduction loop"
    );
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "should have phi nodes for k and acc"
    );
}

#[test]
fn test_fused_linear_relu_spirv_has_ext_inst() {
    let words = generate_fused_linear_relu_spirv(4, 3);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "should use FMax for ReLU"
    );
}

#[test]
fn test_fused_linear_relu_spirv_four_bindings() {
    let words = generate_fused_linear_relu_spirv(4, 3);
    // input (0), weight (1), bias (2), output (3), plus gid
    assert!(count_bindings(&words) >= 4);
}

#[test]
fn test_fused_linear_relu_reference_basic() {
    // 1x2 input, 2x2 weight, 2 bias -> 1x2 output
    let input = [1.0, 2.0];
    let weight = [1.0, 0.0, 0.0, 1.0]; // identity-like: out[0]=in[0], out[1]=in[1]
    let bias = [0.5, -3.0];
    let output = fused_linear_relu_reference(&input, &weight, &bias, 1, 2, 2);
    // out[0] = max(0, 1*1 + 2*0 + 0.5) = max(0, 1.5) = 1.5
    // out[1] = max(0, 1*0 + 2*1 + (-3)) = max(0, -1) = 0.0
    assert!((output[0] - 1.5).abs() < 1e-6, "got {}", output[0]);
    assert!((output[1] - 0.0).abs() < 1e-6, "got {}", output[1]);
}

// ---- Linear + GELU tests ----

#[test]
fn test_fused_linear_gelu_spirv_valid_header() {
    let words = generate_fused_linear_gelu_spirv(4, 3);
    assert_valid_header(&words, "linear_gelu");
}

#[test]
fn test_fused_linear_gelu_spirv_has_tanh() {
    let words = generate_fused_linear_gelu_spirv(4, 3);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "should use Tanh for GELU"
    );
    assert!(has_opcode(&words, TEST_OP_FMUL), "should have FMul");
}

#[test]
fn test_fused_linear_gelu_spirv_workgroup_size() {
    let words = generate_fused_linear_gelu_spirv(4, 3);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_LINEAR_ACT_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_fused_linear_gelu_reference_zero() {
    let input = [0.0];
    let weight = [1.0];
    let bias = [0.0];
    let output = fused_linear_gelu_reference(&input, &weight, &bias, 1, 1, 1);
    // GELU(0) = 0
    assert!(
        (output[0]).abs() < 1e-6,
        "GELU(0) should be 0, got {}",
        output[0]
    );
}

#[test]
fn test_fused_linear_gelu_spirv_larger_than_relu() {
    let relu_words = generate_fused_linear_relu_spirv(4, 3);
    let gelu_words = generate_fused_linear_gelu_spirv(4, 3);
    assert!(
        gelu_words.len() > relu_words.len(),
        "GELU variant should have more instructions than ReLU"
    );
}

// ---- Linear + SiLU tests ----

#[test]
fn test_fused_linear_silu_spirv_valid_header() {
    let words = generate_fused_linear_silu_spirv(4, 3);
    assert_valid_header(&words, "linear_silu");
}

#[test]
fn test_fused_linear_silu_spirv_has_exp_and_div() {
    let words = generate_fused_linear_silu_spirv(4, 3);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "should use Exp for sigmoid"
    );
    assert!(
        has_opcode(&words, TEST_OP_FDIV),
        "should have FDiv for SiLU"
    );
}

#[test]
fn test_fused_linear_silu_spirv_workgroup_size() {
    let words = generate_fused_linear_silu_spirv(4, 3);
    let wg = find_workgroup_size(&words);
    assert_eq!(wg, Some([FUSED_LINEAR_ACT_WORKGROUP_SIZE, 1, 1]));
}

#[test]
fn test_fused_linear_silu_reference_basic() {
    let input = [1.0, 2.0];
    let weight = [1.0, 1.0]; // single output: dot product
    let bias = [0.0];
    let output = fused_linear_silu_reference(&input, &weight, &bias, 1, 2, 1);
    // linear_out = 1*1 + 2*1 + 0 = 3.0
    // SiLU(3.0) = 3.0 / (1 + exp(-3.0)) ~ 3.0 * 0.9526 ~ 2.8577
    assert!(
        (output[0] - 2.8577).abs() < 0.01,
        "SiLU(3.0) ~ 2.8577, got {}",
        output[0]
    );
}

// ---- Cross-cutting tests ----

#[test]
fn test_all_linear_act_have_function_structure() {
    for (label, words) in [
        ("relu", generate_fused_linear_relu_spirv(4, 3)),
        ("gelu", generate_fused_linear_gelu_spirv(4, 3)),
        ("silu", generate_fused_linear_silu_spirv(4, 3)),
    ] {
        assert!(
            has_opcode(&words, TEST_OP_FUNCTION),
            "{label}: missing OpFunction"
        );
        assert!(
            has_opcode(&words, TEST_OP_FUNCTION_END),
            "{label}: missing OpFunctionEnd"
        );
        assert!(
            has_opcode(&words, TEST_OP_RETURN),
            "{label}: missing OpReturn"
        );
    }
}

#[test]
fn test_all_linear_act_word_aligned() {
    // All generators return Vec<u32> which is inherently word-aligned.
    let relu = generate_fused_linear_relu_spirv(4, 3);
    let gelu = generate_fused_linear_gelu_spirv(4, 3);
    let silu = generate_fused_linear_silu_spirv(4, 3);
    assert!(relu.len() > 5, "relu module too short");
    assert!(gelu.len() > 5, "gelu module too short");
    assert!(silu.len() > 5, "silu module too short");
}
