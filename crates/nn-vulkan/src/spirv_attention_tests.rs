// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V binary generation of scaled dot-product attention shaders.

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes/constants used in assertions.
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const OP_CAPABILITY: u16 = 17;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_EXT_INST: u16 = 12;
const OP_SELECT: u16 = 169;
const OP_U_GREATER_THAN: u16 = 172;

// ---- Helpers ----

/// Parse SPIR-V bytes back to words for header inspection.
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

// ---- Valid SPIR-V header tests ----

#[test]
fn test_attention_spirv_header_non_causal() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "attention_hd64_noncausal");
}

#[test]
fn test_attention_spirv_header_causal() {
    let bytes = generate_attention_spirv(64, true);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "attention_hd64_causal");
}

#[test]
fn test_attention_spirv_non_empty() {
    let bytes = generate_attention_spirv(64, false);
    assert!(!bytes.is_empty(), "attention SPIR-V must not be empty");
    assert!(
        bytes.len() > 100,
        "attention SPIR-V must have substantial content"
    );
}

// ---- Entry point name tests ----

#[test]
fn test_attention_spirv_entry_point_non_causal() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_attention_spirv_entry_point_causal() {
    let bytes = generate_attention_spirv(128, true);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

// ---- Workgroup size tests ----

#[test]
fn test_attention_spirv_workgroup_size_non_causal() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [ATTENTION_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_attention_spirv_workgroup_size_causal() {
    let bytes = generate_attention_spirv(64, true);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [ATTENTION_WORKGROUP_SIZE, 1, 1]);
}

// ---- Causal vs non-causal produce different binaries ----

#[test]
fn test_causal_vs_noncausal_different_binaries() {
    let causal_bytes = generate_attention_spirv(64, true);
    let noncausal_bytes = generate_attention_spirv(64, false);
    assert_ne!(
        causal_bytes, noncausal_bytes,
        "causal and non-causal attention must produce different SPIR-V binaries"
    );
}

#[test]
fn test_causal_has_select_and_ugt() {
    let bytes = generate_attention_spirv(64, true);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_SELECT),
        "causal attention must have OpSelect for masking"
    );
    assert!(
        has_opcode(&words, OP_U_GREATER_THAN),
        "causal attention must have UGreaterThan for col > row check"
    );
}

#[test]
fn test_noncausal_no_select() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    assert!(
        !has_opcode(&words, OP_SELECT),
        "non-causal attention must not have OpSelect"
    );
    assert!(
        !has_opcode(&words, OP_U_GREATER_THAN),
        "non-causal attention must not have UGreaterThan"
    );
}

// ---- Structural content tests ----

#[test]
fn test_attention_spirv_has_capability() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CAPABILITY),
        "attention must have OpCapability"
    );
}

#[test]
fn test_attention_spirv_has_loops() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    // Attention has multiple loops: 3 outer (phases) * 1 inner (dot product) each.
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "attention must have loop merge"
    );
    assert!(has_opcode(&words, OP_PHI), "attention must have phi nodes");
}

#[test]
fn test_attention_spirv_loop_count() {
    // Non-causal: 3 outer phase loops + 3 inner dot product loops = 6 total.
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    let loop_count = count_opcode(&words, OP_LOOP_MERGE);
    assert_eq!(
        loop_count, 6,
        "attention must have 6 loops (3 phases x 2 nested)"
    );
}

#[test]
fn test_attention_spirv_has_fmul_fadd() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FMUL),
        "attention must have FMul for dot products and scaling"
    );
    assert!(
        has_opcode(&words, OP_FADD),
        "attention must have FAdd for accumulation"
    );
}

#[test]
fn test_attention_spirv_has_fsub() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FSUB),
        "attention must have FSub for score - max_score"
    );
}

#[test]
fn test_attention_spirv_has_fdiv() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FDIV),
        "attention must have FDiv for softmax normalization"
    );
}

#[test]
fn test_attention_spirv_has_ext_inst() {
    let bytes = generate_attention_spirv(64, false);
    let words = bytes_to_words(&bytes);
    // Attention uses GLSL.std.450 for exp and fmax.
    assert!(
        has_opcode(&words, OP_EXT_INST),
        "attention must use GLSL ext inst (exp, fmax)"
    );
}

// ---- Multiple configurations ----

#[test]
fn test_attention_spirv_different_head_dims() {
    let head_dims = [32, 64, 80, 96, 128, 256];
    for hd in head_dims {
        let bytes = generate_attention_spirv(hd, false);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("attention_hd{hd}_noncausal"));
    }
}

#[test]
fn test_attention_spirv_different_head_dims_causal() {
    let head_dims = [32, 64, 80, 128];
    for hd in head_dims {
        let bytes = generate_attention_spirv(hd, true);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("attention_hd{hd}_causal"));
    }
}

#[test]
fn test_attention_spirv_small_head_dim() {
    // Head dim of 1 is degenerate but must produce valid SPIR-V.
    let bytes = generate_attention_spirv(1, false);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "attention_hd1_noncausal");
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

// ---- Byte alignment ----

#[test]
fn test_attention_spirv_aligned_bytes() {
    let noncausal = generate_attention_spirv(64, false);
    assert_eq!(noncausal.len() % 4, 0, "non-causal bytes must be 4-aligned");

    let causal = generate_attention_spirv(64, true);
    assert_eq!(causal.len() % 4, 0, "causal bytes must be 4-aligned");
}

// ---- Causal binary is larger (has additional masking instructions) ----

#[test]
fn test_causal_binary_larger_than_noncausal() {
    let causal = generate_attention_spirv(64, true);
    let noncausal = generate_attention_spirv(64, false);
    assert!(
        causal.len() > noncausal.len(),
        "causal binary ({} bytes) must be larger than non-causal ({} bytes)",
        causal.len(),
        noncausal.len(),
    );
}

// ---- Both modes share same workgroup size ----

#[test]
fn test_causal_noncausal_same_workgroup_size() {
    let causal = generate_attention_spirv(64, true);
    let noncausal = generate_attention_spirv(64, false);
    let causal_words = bytes_to_words(&causal);
    let noncausal_words = bytes_to_words(&noncausal);
    let causal_wg = find_workgroup_size(&causal_words).unwrap();
    let noncausal_wg = find_workgroup_size(&noncausal_words).unwrap();
    assert_eq!(
        causal_wg, noncausal_wg,
        "causal and non-causal must use same workgroup size"
    );
}

// ---- Constant: ATTENTION_WORKGROUP_SIZE ----

#[test]
fn test_workgroup_size_constant() {
    assert_eq!(ATTENTION_WORKGROUP_SIZE, 256);
}
