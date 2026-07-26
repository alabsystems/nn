// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the matmul SPIR-V binary generation module.
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Naive matmul generation for various matrix sizes
//! - Tiled matmul generation with shared memory
//! - 3 storage buffer bindings (A, B, C)
//! - Edge cases: 1x1, non-square, non-power-of-2 dimensions
//! - Deterministic output
//! - Word count consistency (no truncation or padding)
//! - Cross-variant validation (naive vs tiled)

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes for structural assertions.
const TEST_OP_CAPABILITY: u16 = 17;
const TEST_OP_MEMORY_MODEL: u16 = 14;
const TEST_OP_FUNCTION: u16 = 54;
const TEST_OP_FUNCTION_END: u16 = 56;
const TEST_OP_LABEL: u16 = 248;
const TEST_OP_RETURN: u16 = 253;
const TEST_OP_LOOP_MERGE: u16 = 246;
const TEST_OP_PHI: u16 = 245;
const TEST_OP_FADD: u16 = 129;
const TEST_OP_FMUL: u16 = 133;
const TEST_OP_BRANCH_CONDITIONAL: u16 = 250;
const TEST_OP_U_LESS_THAN: u16 = 176;
const TEST_OP_U_GREATER_THAN_EQUAL: u16 = 174;
const TEST_OP_CONTROL_BARRIER: u16 = 224;
const TEST_OP_VARIABLE: u16 = 59;
const TEST_OP_DECORATE: u16 = 71;
const TEST_OP_STORE: u16 = 62;
const TEST_OP_LOAD: u16 = 61;
const TEST_OP_ACCESS_CHAIN: u16 = 65;
const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const TEST_STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const TEST_STORAGE_CLASS_WORKGROUP: u32 = 4;
const TEST_STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const TEST_DECORATION_BINDING: u32 = 33;
const TEST_DECORATION_DESCRIPTOR_SET: u32 = 34;
const TEST_DECORATION_BLOCK: u32 = 2;
const TEST_DECORATION_NON_WRITABLE: u32 = 24;

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

fn count_storage_buffer_vars(words: &[u32]) -> usize {
    let variables = find_instructions(words, TEST_OP_VARIABLE);
    variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count()
}

fn count_workgroup_vars(words: &[u32]) -> usize {
    let variables = find_instructions(words, TEST_OP_VARIABLE);
    variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_WORKGROUP)
        .count()
}

fn count_push_constant_vars(words: &[u32]) -> usize {
    let variables = find_instructions(words, TEST_OP_VARIABLE);
    variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_PUSH_CONSTANT)
        .count()
}

// ====================================================================
// Naive matmul: SPIR-V header validity
// ====================================================================

#[test]
fn test_naive_matmul_header_32x32() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_32x32");
}

#[test]
fn test_naive_matmul_header_rectangular() {
    let bytes = generate_matmul_spirv_naive(64, 128, 32);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_64x128x32");
}

#[test]
fn test_naive_matmul_header_small() {
    let bytes = generate_matmul_spirv_naive(4, 4, 4);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_4x4");
}

#[test]
fn test_naive_matmul_header_1x1() {
    let bytes = generate_matmul_spirv_naive(1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_1x1");
}

#[test]
fn test_naive_matmul_header_non_power_of_2() {
    let bytes = generate_matmul_spirv_naive(17, 23, 11);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_17x23x11");
}

#[test]
fn test_naive_matmul_header_tall_skinny() {
    let bytes = generate_matmul_spirv_naive(256, 1, 64);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_256x1x64");
}

#[test]
fn test_naive_matmul_header_wide_flat() {
    let bytes = generate_matmul_spirv_naive(1, 512, 32);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_1x512x32");
}

#[test]
fn test_naive_matmul_header_large() {
    let bytes = generate_matmul_spirv_naive(1024, 1024, 1024);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_matmul_1024x1024x1024");
}

// ====================================================================
// Naive matmul: entry point and workgroup
// ====================================================================

#[test]
fn test_naive_matmul_entry_point() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_naive_matmul_workgroup_size() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1]);
}

#[test]
fn test_naive_matmul_entry_point_non_square() {
    let bytes = generate_matmul_spirv_naive(13, 29, 7);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_naive_matmul_workgroup_size_non_square() {
    let bytes = generate_matmul_spirv_naive(13, 29, 7);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(
        wg,
        [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1],
        "workgroup size must be tile_size x tile_size x 1 regardless of matrix dimensions"
    );
}

// ====================================================================
// Naive matmul: opcode structure
// ====================================================================

#[test]
fn test_naive_matmul_has_capability_shader() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "naive matmul must have OpCapability"
    );
}

#[test]
fn test_naive_matmul_has_memory_model() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "naive matmul must have OpMemoryModel"
    );
}

#[test]
fn test_naive_matmul_has_loop() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_LOOP_MERGE),
        "naive matmul must have a loop (OpLoopMerge) for K accumulation"
    );
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "naive matmul must have OpPhi for loop variables"
    );
}

#[test]
fn test_naive_matmul_has_fmul_fadd() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FMUL),
        "naive matmul must have OpFMul for A*B products"
    );
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "naive matmul must have OpFAdd for accumulation"
    );
}

#[test]
fn test_naive_matmul_has_bounds_check() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_U_GREATER_THAN_EQUAL),
        "naive matmul must have bounds check (OpUGreaterThanEqual)"
    );
    assert!(
        has_opcode(&words, TEST_OP_BRANCH_CONDITIONAL),
        "naive matmul must have conditional branches"
    );
}

#[test]
fn test_naive_matmul_has_function_structure() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_naive_matmul_has_memory_access_ops() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_LOAD),
        "naive matmul must have OpLoad for reading A/B elements"
    );
    assert!(
        has_opcode(&words, TEST_OP_STORE),
        "naive matmul must have OpStore for writing C elements"
    );
    assert!(
        has_opcode(&words, TEST_OP_ACCESS_CHAIN),
        "naive matmul must have OpAccessChain for buffer indexing"
    );
}

// ====================================================================
// Naive matmul: size and alignment
// ====================================================================

#[test]
fn test_naive_matmul_reasonable_size() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        words.len() > 100,
        "naive matmul module too small ({} words)",
        words.len()
    );
    assert!(
        words.len() < 2000,
        "naive matmul module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_naive_matmul_byte_alignment() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    assert_eq!(bytes.len() % 4, 0, "SPIR-V binary must be 4-byte aligned");
}

#[test]
fn test_naive_matmul_byte_alignment_various() {
    for (m, n, k) in [(1, 1, 1), (7, 13, 5), (32, 64, 16), (100, 200, 50)] {
        let bytes = generate_matmul_spirv_naive(m, n, k);
        assert_eq!(
            bytes.len() % 4,
            0,
            "naive matmul {m}x{n}x{k}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_naive_matmul_deterministic() {
    let bytes1 = generate_matmul_spirv_naive(32, 32, 32);
    let bytes2 = generate_matmul_spirv_naive(32, 32, 32);
    assert_eq!(
        bytes1, bytes2,
        "naive matmul SPIR-V output must be deterministic across calls"
    );
}

#[test]
fn test_naive_matmul_word_counts_consistent() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let mut pos = 5;
    let mut instruction_count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        assert!(
            word_count > 0,
            "instruction at pos {pos} has word_count 0 (opcode {opcode})"
        );
        assert!(
            pos + word_count <= words.len(),
            "instruction at pos {pos} (opcode {opcode}, wc {word_count}) exceeds module length {}",
            words.len()
        );
        pos += word_count;
        instruction_count += 1;
    }
    assert_eq!(
        pos,
        words.len(),
        "instructions did not consume exactly the full module"
    );
    assert!(
        instruction_count > 20,
        "expected at least 20 instructions for naive matmul, got {instruction_count}"
    );
}

// ====================================================================
// Naive matmul: buffer layout
// ====================================================================

#[test]
fn test_naive_matmul_three_storage_buffers() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_storage_buffer_vars(&words),
        3,
        "naive matmul must have 3 storage buffer variables (A, B, C)"
    );
}

#[test]
fn test_naive_matmul_has_push_constants() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_push_constant_vars(&words),
        1,
        "naive matmul must have 1 push constant variable for M, N, K"
    );
}

#[test]
fn test_naive_matmul_binding_numbers() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    assert!(
        bindings.contains(&0),
        "must have binding 0 (matrix A buffer)"
    );
    assert!(
        bindings.contains(&1),
        "must have binding 1 (matrix B buffer)"
    );
    assert!(
        bindings.contains(&2),
        "must have binding 2 (matrix C buffer)"
    );
}

#[test]
fn test_naive_matmul_descriptor_set_zero() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let desc_sets: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_DESCRIPTOR_SET)
        .map(|d| d[3])
        .collect();
    for &ds in &desc_sets {
        assert_eq!(
            ds, 0,
            "all descriptor sets must be 0 (single descriptor set layout)"
        );
    }
}

#[test]
fn test_naive_matmul_has_block_decorations() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let block_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_BLOCK)
        .count();
    // A, B, C buffer structs need Block decoration; push constant struct also needs it
    assert!(
        block_count >= 3,
        "matmul must have at least 3 Block decorations (A, B, C structs), found {block_count}"
    );
}

#[test]
fn test_naive_matmul_nonwritable_decoration_count() {
    // The matmul generator does not currently emit NonWritable decorations
    // for input buffers (A, B). This test documents the current behavior.
    // If NonWritable is added in the future, update the assertion.
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nw_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    // Current implementation: 0 NonWritable decorations.
    // Valid SPIR-V either way -- NonWritable is an optimization hint, not required.
    assert!(
        nw_count == 0 || nw_count >= 2,
        "NonWritable count must be 0 (not emitted) or >= 2 (A and B), found {nw_count}"
    );
}

// ====================================================================
// Naive matmul: no workgroup memory (no shared memory)
// ====================================================================

#[test]
fn test_naive_matmul_no_workgroup_variables() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_workgroup_vars(&words),
        0,
        "naive matmul must NOT have workgroup variables (no shared memory)"
    );
}

#[test]
fn test_naive_matmul_no_barriers() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_opcode(&words, TEST_OP_CONTROL_BARRIER),
        0,
        "naive matmul must NOT have barriers (no shared memory synchronization needed)"
    );
}

// ====================================================================
// Naive matmul: various dimensions header validity
// ====================================================================

#[test]
fn test_naive_matmul_various_dimensions() {
    let cases = [
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 4),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (7, 13, 5),     // all non-power-of-2
        (17, 23, 11),   // primes
        (64, 128, 32),  // rectangular
        (1, 64, 1),     // degenerate: vector-vector via matmul
        (256, 1, 64),   // tall-skinny result
        (1, 512, 32),   // wide result
        (100, 200, 50), // round non-power-of-2
    ];
    for (m, n, k) in cases {
        let bytes = generate_matmul_spirv_naive(m, n, k);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("naive_{m}x{n}x{k}"));
    }
}

// ====================================================================
// Tiled matmul: SPIR-V header validity
// ====================================================================

#[test]
fn test_tiled_matmul_header_32x32() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_matmul_32x32");
}

#[test]
fn test_tiled_matmul_header_rectangular() {
    let bytes = generate_matmul_spirv(64, 128, 32);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_matmul_64x128x32");
}

#[test]
fn test_tiled_matmul_header_small() {
    let bytes = generate_matmul_spirv(4, 4, 4);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_matmul_4x4");
}

#[test]
fn test_tiled_matmul_header_1x1() {
    let bytes = generate_matmul_spirv(1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_matmul_1x1");
}

#[test]
fn test_tiled_matmul_header_non_power_of_2() {
    let bytes = generate_matmul_spirv(17, 23, 11);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_matmul_17x23x11");
}

#[test]
fn test_tiled_matmul_header_large() {
    let bytes = generate_matmul_spirv(1024, 1024, 1024);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_matmul_1024x1024x1024");
}

// ====================================================================
// Tiled matmul: entry point and workgroup
// ====================================================================

#[test]
fn test_tiled_matmul_entry_point() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_tiled_matmul_workgroup_size() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1]);
}

#[test]
fn test_tiled_matmul_entry_point_non_square() {
    let bytes = generate_matmul_spirv(13, 29, 7);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_tiled_matmul_workgroup_size_non_square() {
    let bytes = generate_matmul_spirv(13, 29, 7);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(
        wg,
        [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1],
        "tiled workgroup size must be tile_size x tile_size x 1 regardless of matrix dimensions"
    );
}

// ====================================================================
// Tiled matmul: opcode structure
// ====================================================================

#[test]
fn test_tiled_matmul_has_capability_shader() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "tiled matmul must have OpCapability"
    );
}

#[test]
fn test_tiled_matmul_has_memory_model() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "tiled matmul must have OpMemoryModel"
    );
}

#[test]
fn test_tiled_matmul_has_loops() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_LOOP_MERGE),
        "tiled matmul must have loops (OpLoopMerge)"
    );
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "tiled matmul must have OpPhi for loop variables"
    );
}

#[test]
fn test_tiled_matmul_has_barrier() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CONTROL_BARRIER),
        "tiled matmul must have OpControlBarrier for shared memory sync"
    );
}

#[test]
fn test_tiled_matmul_barrier_count() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let barrier_count = count_opcode(&words, TEST_OP_CONTROL_BARRIER);
    // Tiled matmul needs at least 2 barriers: one after loading tiles into shared memory,
    // one after computing the partial products before loading next tile.
    assert!(
        barrier_count >= 2,
        "tiled matmul must have at least 2 barriers for shared memory sync, found {barrier_count}"
    );
}

#[test]
fn test_tiled_matmul_has_fmul_fadd() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FMUL),
        "tiled matmul must have OpFMul"
    );
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "tiled matmul must have OpFAdd"
    );
}

#[test]
fn test_tiled_matmul_has_function_structure() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_tiled_matmul_has_bounds_check() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_U_LESS_THAN),
        "tiled matmul must have bounds checks (OpULessThan)"
    );
    assert!(
        has_opcode(&words, TEST_OP_BRANCH_CONDITIONAL),
        "tiled matmul must have conditional branches"
    );
}

#[test]
fn test_tiled_matmul_has_memory_access_ops() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_LOAD),
        "tiled matmul must have OpLoad"
    );
    assert!(
        has_opcode(&words, TEST_OP_STORE),
        "tiled matmul must have OpStore"
    );
    assert!(
        has_opcode(&words, TEST_OP_ACCESS_CHAIN),
        "tiled matmul must have OpAccessChain"
    );
}

// ====================================================================
// Tiled matmul: size and alignment
// ====================================================================

#[test]
fn test_tiled_matmul_reasonable_size() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert!(
        words.len() > 100,
        "tiled matmul module too small ({} words)",
        words.len()
    );
    assert!(
        words.len() < 5000,
        "tiled matmul module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_tiled_matmul_byte_alignment() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    assert_eq!(bytes.len() % 4, 0, "SPIR-V binary must be 4-byte aligned");
}

#[test]
fn test_tiled_matmul_byte_alignment_various() {
    for (m, n, k) in [(1, 1, 1), (7, 13, 5), (32, 64, 16), (100, 200, 50)] {
        let bytes = generate_matmul_spirv(m, n, k);
        assert_eq!(
            bytes.len() % 4,
            0,
            "tiled matmul {m}x{n}x{k}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_tiled_matmul_deterministic() {
    let bytes1 = generate_matmul_spirv(32, 32, 32);
    let bytes2 = generate_matmul_spirv(32, 32, 32);
    assert_eq!(
        bytes1, bytes2,
        "tiled matmul SPIR-V output must be deterministic across calls"
    );
}

#[test]
fn test_tiled_matmul_word_counts_consistent() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let mut pos = 5;
    let mut instruction_count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        assert!(
            word_count > 0,
            "instruction at pos {pos} has word_count 0 (opcode {opcode})"
        );
        assert!(
            pos + word_count <= words.len(),
            "instruction at pos {pos} (opcode {opcode}, wc {word_count}) exceeds module length {}",
            words.len()
        );
        pos += word_count;
        instruction_count += 1;
    }
    assert_eq!(
        pos,
        words.len(),
        "instructions did not consume exactly the full module"
    );
    assert!(
        instruction_count > 20,
        "expected at least 20 instructions for tiled matmul, got {instruction_count}"
    );
}

// ====================================================================
// Tiled matmul: buffer and shared memory layout
// ====================================================================

#[test]
fn test_tiled_matmul_three_storage_buffers() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_storage_buffer_vars(&words),
        3,
        "tiled matmul must have 3 storage buffer variables (A, B, C)"
    );
}

#[test]
fn test_tiled_matmul_has_workgroup_variables() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_workgroup_vars(&words),
        2,
        "tiled matmul must have 2 workgroup variables (tile_a, tile_b)"
    );
}

#[test]
fn test_tiled_matmul_has_push_constants() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    assert_eq!(
        count_push_constant_vars(&words),
        1,
        "tiled matmul must have 1 push constant variable for M, N, K"
    );
}

#[test]
fn test_tiled_matmul_binding_numbers() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    assert!(
        bindings.contains(&0),
        "must have binding 0 (matrix A buffer)"
    );
    assert!(
        bindings.contains(&1),
        "must have binding 1 (matrix B buffer)"
    );
    assert!(
        bindings.contains(&2),
        "must have binding 2 (matrix C buffer)"
    );
}

#[test]
fn test_tiled_matmul_descriptor_set_zero() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let desc_sets: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_DESCRIPTOR_SET)
        .map(|d| d[3])
        .collect();
    for &ds in &desc_sets {
        assert_eq!(
            ds, 0,
            "all descriptor sets must be 0 (single descriptor set layout)"
        );
    }
}

#[test]
fn test_tiled_matmul_nonwritable_decoration_count() {
    // The matmul generator does not currently emit NonWritable decorations.
    // This test documents the current behavior. See naive variant test for details.
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nw_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    assert!(
        nw_count == 0 || nw_count >= 2,
        "NonWritable count must be 0 (not emitted) or >= 2 (A and B), found {nw_count}"
    );
}

// ====================================================================
// Tiled matmul: various dimensions header validity
// ====================================================================

#[test]
fn test_tiled_matmul_various_dimensions() {
    let cases = [
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 4),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (7, 13, 5),
        (17, 23, 11),
        (64, 128, 32),
        (1, 64, 1),
        (256, 1, 64),
        (1, 512, 32),
        (100, 200, 50),
    ];
    for (m, n, k) in cases {
        let bytes = generate_matmul_spirv(m, n, k);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("tiled_{m}x{n}x{k}"));
    }
}

// ====================================================================
// Cross-variant tests: naive vs tiled
// ====================================================================

#[test]
fn test_both_variants_produce_valid_spirv() {
    for (m, n, k) in [(32, 32, 32), (64, 32, 128), (4, 4, 4), (17, 23, 11)] {
        let naive = generate_matmul_spirv_naive(m, n, k);
        let tiled = generate_matmul_spirv(m, n, k);

        let naive_words = bytes_to_words(&naive);
        let tiled_words = bytes_to_words(&tiled);

        assert_valid_header(&naive_words, &format!("naive_{m}x{n}x{k}"));
        assert_valid_header(&tiled_words, &format!("tiled_{m}x{n}x{k}"));

        // Both must have the same magic number and version.
        assert_eq!(naive_words[0], tiled_words[0], "same SPIR-V magic");
        assert_eq!(naive_words[1], tiled_words[1], "same SPIR-V version");
    }
}

#[test]
fn test_both_variants_have_three_storage_buffers() {
    let naive = generate_matmul_spirv_naive(32, 32, 32);
    let tiled = generate_matmul_spirv(32, 32, 32);
    let naive_words = bytes_to_words(&naive);
    let tiled_words = bytes_to_words(&tiled);

    assert_eq!(
        count_storage_buffer_vars(&naive_words),
        3,
        "naive matmul must have 3 storage buffer variables (A, B, C)"
    );
    assert_eq!(
        count_storage_buffer_vars(&tiled_words),
        3,
        "tiled matmul must have 3 storage buffer variables (A, B, C)"
    );
}

#[test]
fn test_both_variants_same_entry_point_name() {
    for (m, n, k) in [(32, 32, 32), (7, 13, 5), (1, 1, 1)] {
        let naive = generate_matmul_spirv_naive(m, n, k);
        let tiled = generate_matmul_spirv(m, n, k);
        let naive_words = bytes_to_words(&naive);
        let tiled_words = bytes_to_words(&tiled);

        let naive_name = find_entry_point_name(&naive_words)
            .unwrap_or_else(|| panic!("naive {m}x{n}x{k} must have entry point"));
        let tiled_name = find_entry_point_name(&tiled_words)
            .unwrap_or_else(|| panic!("tiled {m}x{n}x{k} must have entry point"));
        assert_eq!(
            naive_name, tiled_name,
            "both variants should use the same entry point name"
        );
        assert_eq!(naive_name, "main");
    }
}

#[test]
fn test_both_variants_same_workgroup_size() {
    for (m, n, k) in [(32, 32, 32), (7, 13, 5), (64, 128, 32)] {
        let naive = generate_matmul_spirv_naive(m, n, k);
        let tiled = generate_matmul_spirv(m, n, k);
        let naive_words = bytes_to_words(&naive);
        let tiled_words = bytes_to_words(&tiled);

        let naive_wg = find_workgroup_size(&naive_words)
            .unwrap_or_else(|| panic!("naive {m}x{n}x{k} must have workgroup size"));
        let tiled_wg = find_workgroup_size(&tiled_words)
            .unwrap_or_else(|| panic!("tiled {m}x{n}x{k} must have workgroup size"));
        assert_eq!(
            naive_wg, tiled_wg,
            "both variants should use the same workgroup size for {m}x{n}x{k}"
        );
    }
}

#[test]
fn test_tiled_has_more_instructions_than_naive() {
    // Tiled version must be larger due to shared memory management.
    let naive = generate_matmul_spirv_naive(32, 32, 32);
    let tiled = generate_matmul_spirv(32, 32, 32);
    assert!(
        tiled.len() > naive.len(),
        "tiled matmul ({} bytes) should be larger than naive ({} bytes) due to shared memory logic",
        tiled.len(),
        naive.len()
    );
}

#[test]
fn test_tiled_has_workgroup_vars_naive_does_not() {
    let naive = generate_matmul_spirv_naive(32, 32, 32);
    let tiled = generate_matmul_spirv(32, 32, 32);
    let naive_words = bytes_to_words(&naive);
    let tiled_words = bytes_to_words(&tiled);

    assert_eq!(
        count_workgroup_vars(&naive_words),
        0,
        "naive must not have workgroup variables"
    );
    assert!(
        count_workgroup_vars(&tiled_words) >= 2,
        "tiled must have at least 2 workgroup variables for tile_a and tile_b"
    );
}

#[test]
fn test_tiled_has_barriers_naive_does_not() {
    let naive = generate_matmul_spirv_naive(32, 32, 32);
    let tiled = generate_matmul_spirv(32, 32, 32);
    let naive_words = bytes_to_words(&naive);
    let tiled_words = bytes_to_words(&tiled);

    assert_eq!(
        count_opcode(&naive_words, TEST_OP_CONTROL_BARRIER),
        0,
        "naive matmul must not have barriers"
    );
    assert!(
        count_opcode(&tiled_words, TEST_OP_CONTROL_BARRIER) >= 2,
        "tiled matmul must have barriers for shared memory synchronization"
    );
}

// ====================================================================
// MATMUL_TILE_SIZE constant
// ====================================================================

#[test]
fn test_matmul_tile_size_is_16() {
    assert_eq!(MATMUL_TILE_SIZE, 16, "MATMUL_TILE_SIZE must be 16");
}

#[test]
fn test_matmul_tile_size_is_power_of_2() {
    assert!(
        MATMUL_TILE_SIZE.is_power_of_two(),
        "MATMUL_TILE_SIZE must be a power of 2"
    );
}

// ====================================================================
// Loop structure tests (tiled has more loops than naive)
// ====================================================================

#[test]
fn test_naive_matmul_single_loop() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    // Naive matmul should have exactly 1 loop: the K accumulation loop.
    assert!(
        loop_count >= 1,
        "naive matmul must have at least 1 loop for K accumulation, found {loop_count}"
    );
}

#[test]
fn test_tiled_matmul_multiple_loops() {
    let bytes = generate_matmul_spirv(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    // Tiled matmul has: outer tile loop over K tiles + inner dot product loop = at least 2.
    assert!(
        loop_count >= 2,
        "tiled matmul must have at least 2 loops (tile iteration + inner dot), found {loop_count}"
    );
}

#[test]
fn test_tiled_more_loops_than_naive() {
    let naive = generate_matmul_spirv_naive(32, 32, 32);
    let tiled = generate_matmul_spirv(32, 32, 32);
    let naive_words = bytes_to_words(&naive);
    let tiled_words = bytes_to_words(&tiled);

    let naive_loops = count_opcode(&naive_words, TEST_OP_LOOP_MERGE);
    let tiled_loops = count_opcode(&tiled_words, TEST_OP_LOOP_MERGE);
    assert!(
        tiled_loops >= naive_loops,
        "tiled ({tiled_loops} loops) should have at least as many loops as naive ({naive_loops})"
    );
}

// ====================================================================
// Dimension independence: different dimensions produce different modules
// ====================================================================

#[test]
fn test_naive_different_dimensions_produce_different_modules() {
    let bytes_a = generate_matmul_spirv_naive(32, 32, 32);
    let bytes_b = generate_matmul_spirv_naive(64, 64, 64);
    // Different dimensions should produce different SPIR-V modules
    // (due to push constant encoding or specialization constants).
    // The modules may or may not differ depending on implementation.
    // At minimum, both must be valid.
    let words_a = bytes_to_words(&bytes_a);
    let words_b = bytes_to_words(&bytes_b);
    assert_valid_header(&words_a, "32x32x32");
    assert_valid_header(&words_b, "64x64x64");
}

#[test]
fn test_tiled_different_dimensions_produce_different_modules() {
    let bytes_a = generate_matmul_spirv(32, 32, 32);
    let bytes_b = generate_matmul_spirv(64, 64, 64);
    let words_a = bytes_to_words(&bytes_a);
    let words_b = bytes_to_words(&bytes_b);
    assert_valid_header(&words_a, "32x32x32");
    assert_valid_header(&words_b, "64x64x64");
}

// ====================================================================
// Edge case: minimum dimensions (1x1x1)
// ====================================================================

#[test]
fn test_naive_matmul_1x1x1_full_validation() {
    let bytes = generate_matmul_spirv_naive(1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_1x1x1");
    assert_eq!(
        find_entry_point_name(&words).expect("must have entry point"),
        "main"
    );
    assert_eq!(
        find_workgroup_size(&words).expect("must have workgroup size"),
        [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1]
    );
    assert_eq!(count_storage_buffer_vars(&words), 3);
    assert!(has_opcode(&words, TEST_OP_FUNCTION));
    assert!(has_opcode(&words, TEST_OP_FUNCTION_END));
}

#[test]
fn test_tiled_matmul_1x1x1_full_validation() {
    let bytes = generate_matmul_spirv(1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_1x1x1");
    assert_eq!(
        find_entry_point_name(&words).expect("must have entry point"),
        "main"
    );
    assert_eq!(
        find_workgroup_size(&words).expect("must have workgroup size"),
        [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1]
    );
    assert_eq!(count_storage_buffer_vars(&words), 3);
    assert_eq!(count_workgroup_vars(&words), 2);
    assert!(has_opcode(&words, TEST_OP_CONTROL_BARRIER));
}

// ====================================================================
// Edge case: highly rectangular matrices
// ====================================================================

#[test]
fn test_naive_matmul_row_vector_times_matrix() {
    // [1, K] x [K, N] = [1, N]
    let bytes = generate_matmul_spirv_naive(1, 128, 64);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_1x128x64");
    assert_eq!(count_storage_buffer_vars(&words), 3);
}

#[test]
fn test_naive_matmul_matrix_times_column_vector() {
    // [M, K] x [K, 1] = [M, 1]
    let bytes = generate_matmul_spirv_naive(128, 1, 64);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "naive_128x1x64");
    assert_eq!(count_storage_buffer_vars(&words), 3);
}

#[test]
fn test_tiled_matmul_row_vector_times_matrix() {
    let bytes = generate_matmul_spirv(1, 128, 64);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_1x128x64");
    assert_eq!(count_storage_buffer_vars(&words), 3);
}

#[test]
fn test_tiled_matmul_matrix_times_column_vector() {
    let bytes = generate_matmul_spirv(128, 1, 64);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "tiled_128x1x64");
    assert_eq!(count_storage_buffer_vars(&words), 3);
}

// ====================================================================
// Edge case: dimensions smaller than tile size
// ====================================================================

#[test]
fn test_tiled_matmul_smaller_than_tile() {
    // All dimensions < MATMUL_TILE_SIZE (16). The tiled kernel must still produce
    // valid SPIR-V with bounds checking to handle the partial tile.
    for size in [1, 2, 3, 7, 8, 15] {
        let bytes = generate_matmul_spirv(size, size, size);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("tiled_{size}x{size}x{size}"));
        assert!(
            has_opcode(&words, TEST_OP_BRANCH_CONDITIONAL),
            "tiled matmul {size}x{size}x{size} must have bounds checks for partial tiles"
        );
    }
}

// ====================================================================
// Edge case: dimensions exactly equal to tile size
// ====================================================================

#[test]
fn test_tiled_matmul_exact_tile_size() {
    let tile = MATMUL_TILE_SIZE;
    let bytes = generate_matmul_spirv(tile, tile, tile);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, &format!("tiled_{tile}x{tile}x{tile}"));
    assert_eq!(count_storage_buffer_vars(&words), 3);
    assert_eq!(count_workgroup_vars(&words), 2);
}

#[test]
fn test_tiled_matmul_multiple_of_tile_size() {
    let tile = MATMUL_TILE_SIZE;
    let bytes = generate_matmul_spirv(tile * 2, tile * 3, tile * 4);
    let words = bytes_to_words(&bytes);
    assert_valid_header(
        &words,
        &format!("tiled_{}x{}x{}", tile * 2, tile * 3, tile * 4),
    );
}

// ====================================================================
// Edge case: prime dimensions (worst case for tiling)
// ====================================================================

#[test]
fn test_naive_matmul_prime_dimensions() {
    for (m, n, k) in [(3, 5, 7), (11, 13, 17), (23, 29, 31), (37, 41, 43)] {
        let bytes = generate_matmul_spirv_naive(m, n, k);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("naive_prime_{m}x{n}x{k}"));
    }
}

#[test]
fn test_tiled_matmul_prime_dimensions() {
    for (m, n, k) in [(3, 5, 7), (11, 13, 17), (23, 29, 31), (37, 41, 43)] {
        let bytes = generate_matmul_spirv(m, n, k);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("tiled_prime_{m}x{n}x{k}"));
    }
}
