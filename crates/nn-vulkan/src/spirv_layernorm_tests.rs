// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V binary generation of LayerNorm and RMSNorm shaders.

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes/constants used in assertions.
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const OP_LOOP_MERGE: u16 = 246;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_EXT_INST: u16 = 12;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_PHI: u16 = 245;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;

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
        words[1], SPIRV_VERSION_1_0,
        "{label}: wrong version (expected 1.0 = 0x00010000)"
    );
    assert_eq!(words[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
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

// ====================================================================
// LayerNorm tests
// ====================================================================

#[test]
fn test_layernorm_spirv_valid_header_768() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "layernorm_768");
}

#[test]
fn test_layernorm_spirv_valid_header_256() {
    let bytes = generate_layernorm_spirv(256, 1e-5);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "layernorm_256");
}

#[test]
fn test_layernorm_spirv_valid_header_1024() {
    let bytes = generate_layernorm_spirv(1024, 1e-6);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "layernorm_1024");
}

#[test]
fn test_layernorm_spirv_entry_point_is_main() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_layernorm_spirv_workgroup_size() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [LAYERNORM_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_layernorm_spirv_has_loop_merge() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "layernorm must have OpLoopMerge for accumulation loops"
    );
    // LayerNorm has at least 3 serial loops + 2 tree reduction loops = 5+ loops.
    let loop_count = count_opcode(&words, OP_LOOP_MERGE);
    assert!(
        loop_count >= 3,
        "layernorm should have at least 3 loops (sum, variance, normalize), got {loop_count}"
    );
}

#[test]
fn test_layernorm_spirv_has_fdiv() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FDIV),
        "layernorm must have OpFDiv for mean/variance computation"
    );
}

#[test]
fn test_layernorm_spirv_has_fsub() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FSUB),
        "layernorm must have OpFSub for (x - mean)"
    );
}

#[test]
fn test_layernorm_spirv_has_fmul() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FMUL),
        "layernorm must have OpFMul for weight scaling and variance"
    );
}

#[test]
fn test_layernorm_spirv_has_ext_inst_for_sqrt() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_EXT_INST),
        "layernorm must have OpExtInst for sqrt via GLSL.std.450"
    );
}

#[test]
fn test_layernorm_spirv_has_barrier() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CONTROL_BARRIER),
        "layernorm must have OpControlBarrier for shared memory sync"
    );
}

#[test]
fn test_layernorm_spirv_has_phi() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_PHI),
        "layernorm must have OpPhi for loop induction variables"
    );
}

#[test]
fn test_layernorm_spirv_has_function_structure() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, OP_RETURN), "must have OpReturn");
}

#[test]
fn test_layernorm_spirv_byte_alignment() {
    for norm_shape in [64, 128, 256, 512, 768, 1024, 2048, 4096] {
        let bytes = generate_layernorm_spirv(norm_shape, 1e-5);
        assert_eq!(
            bytes.len() % 4,
            0,
            "layernorm norm_shape={norm_shape}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_layernorm_spirv_different_sizes_produce_valid() {
    for norm_shape in [32, 64, 128, 256, 512, 768, 1024] {
        let bytes = generate_layernorm_spirv(norm_shape, 1e-5);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("layernorm_{norm_shape}"));
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main", "layernorm_{norm_shape} entry point");
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(
            wg,
            [LAYERNORM_WORKGROUP_SIZE, 1, 1],
            "layernorm_{norm_shape} workgroup"
        );
    }
}

#[test]
fn test_layernorm_spirv_different_eps_produce_valid() {
    for eps in [1e-5, 1e-6, 1e-8, 1e-12] {
        let bytes = generate_layernorm_spirv(768, eps);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("layernorm_eps_{eps}"));
    }
}

// ====================================================================
// RMSNorm tests
// ====================================================================

#[test]
fn test_rmsnorm_spirv_valid_header_768() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "rmsnorm_768");
}

#[test]
fn test_rmsnorm_spirv_valid_header_256() {
    let bytes = generate_rmsnorm_spirv(256, 1e-5);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "rmsnorm_256");
}

#[test]
fn test_rmsnorm_spirv_valid_header_4096() {
    let bytes = generate_rmsnorm_spirv(4096, 1e-6);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "rmsnorm_4096");
}

#[test]
fn test_rmsnorm_spirv_entry_point_is_main() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_rmsnorm_spirv_workgroup_size() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [LAYERNORM_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_rmsnorm_spirv_has_loop_merge() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "rmsnorm must have OpLoopMerge for accumulation loops"
    );
    let loop_count = count_opcode(&words, OP_LOOP_MERGE);
    assert!(
        loop_count >= 2,
        "rmsnorm should have at least 2 loops (sq_sum, normalize), got {loop_count}"
    );
}

#[test]
fn test_rmsnorm_spirv_has_fdiv() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FDIV),
        "rmsnorm must have OpFDiv for mean_sq computation"
    );
}

#[test]
fn test_rmsnorm_spirv_has_fmul() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FMUL),
        "rmsnorm must have OpFMul for x*x and weight*normalized"
    );
}

#[test]
fn test_rmsnorm_spirv_has_ext_inst_for_sqrt() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_EXT_INST),
        "rmsnorm must have OpExtInst for sqrt via GLSL.std.450"
    );
}

#[test]
fn test_rmsnorm_spirv_has_barrier() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CONTROL_BARRIER),
        "rmsnorm must have OpControlBarrier for shared memory sync"
    );
}

#[test]
fn test_rmsnorm_spirv_has_fadd() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FADD),
        "rmsnorm must have OpFAdd for sum accumulation"
    );
}

#[test]
fn test_rmsnorm_spirv_no_fsub() {
    // RMSNorm does not subtract the mean, so OpFSub should not appear
    // (unless the tree reduction or other internal code uses it).
    // Actually, the ISub for row index derivation uses OpISub not OpFSub,
    // so there should be no OpFSub in RMSNorm.
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(
        !has_opcode(&words, OP_FSUB),
        "rmsnorm should not have OpFSub (no mean subtraction)"
    );
}

#[test]
fn test_rmsnorm_spirv_has_function_structure() {
    let bytes = generate_rmsnorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, OP_RETURN), "must have OpReturn");
}

#[test]
fn test_rmsnorm_spirv_byte_alignment() {
    for norm_shape in [64, 128, 256, 512, 768, 1024, 2048, 4096] {
        let bytes = generate_rmsnorm_spirv(norm_shape, 1e-5);
        assert_eq!(
            bytes.len() % 4,
            0,
            "rmsnorm norm_shape={norm_shape}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_rmsnorm_spirv_different_sizes_produce_valid() {
    for norm_shape in [32, 64, 128, 256, 512, 768, 1024] {
        let bytes = generate_rmsnorm_spirv(norm_shape, 1e-5);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("rmsnorm_{norm_shape}"));
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main", "rmsnorm_{norm_shape} entry point");
        let wg = find_workgroup_size(&words).expect("must have workgroup size");
        assert_eq!(
            wg,
            [LAYERNORM_WORKGROUP_SIZE, 1, 1],
            "rmsnorm_{norm_shape} workgroup"
        );
    }
}

#[test]
fn test_rmsnorm_spirv_different_eps_produce_valid() {
    for eps in [1e-5, 1e-6, 1e-8, 1e-12] {
        let bytes = generate_rmsnorm_spirv(768, eps);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("rmsnorm_eps_{eps}"));
    }
}

// ====================================================================
// Cross-variant comparison tests
// ====================================================================

#[test]
fn test_layernorm_longer_than_rmsnorm() {
    // LayerNorm has more phases (mean + variance + affine with bias) than RMSNorm
    // (sq_sum + normalize), so the SPIR-V binary should be larger.
    let ln_bytes = generate_layernorm_spirv(768, 1e-5);
    let rms_bytes = generate_rmsnorm_spirv(768, 1e-5);
    assert!(
        ln_bytes.len() > rms_bytes.len(),
        "LayerNorm ({} bytes) should be larger than RMSNorm ({} bytes)",
        ln_bytes.len(),
        rms_bytes.len()
    );
}

#[test]
fn test_workgroup_size_constant_value() {
    assert_eq!(LAYERNORM_WORKGROUP_SIZE, 256);
}
