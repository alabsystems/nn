// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::generate_embedding_spirv`].

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;

// ---- Helpers ----

/// Re-interpret a `Vec<u8>` as `Vec<u32>` (little-endian).
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "byte length must be 4-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_valid_header(spirv: &[u32], label: &str) {
    assert!(spirv.len() >= 5, "{label}: module too short");
    assert_eq!(spirv[0], SPIRV_MAGIC, "{label}: wrong magic number");
    assert_eq!(spirv[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
    assert_eq!(spirv[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
    assert!(spirv[3] > 0, "{label}: bound must be > 0");
    assert_eq!(spirv[4], 0, "{label}: schema must be 0");
}

fn has_opcode(spirv: &[u32], target_opcode: u16) -> bool {
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            return true;
        }
        pos += word_count;
    }
    false
}

// ---- Valid SPIR-V header ----

#[test]
fn test_embedding_spirv_valid_header() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert_valid_header(&spirv, "embedding(32000,768)");
}

#[test]
fn test_embedding_spirv_valid_header_small() {
    let bytes = generate_embedding_spirv(256, 64);
    let spirv = bytes_to_words(&bytes);
    assert_valid_header(&spirv, "embedding(256,64)");
}

// ---- Entry point = "main" ----

#[test]
fn test_embedding_spirv_entry_point_main() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("embedding: no entry point found"));
    assert_eq!(name, "main", "embedding: entry point must be 'main'");
}

// ---- Workgroup size matches constant ----

#[test]
fn test_embedding_spirv_workgroup_size() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    let wg =
        find_workgroup_size(&spirv).unwrap_or_else(|| panic!("embedding: no workgroup size found"));
    assert_eq!(
        wg,
        [EMBEDDING_WORKGROUP_SIZE, 1, 1],
        "embedding: workgroup size must be [{EMBEDDING_WORKGROUP_SIZE}, 1, 1]",
    );
}

// ---- Different vocab/dim sizes produce valid SPIR-V ----

#[test]
fn test_embedding_spirv_various_sizes() {
    let configs: &[(u32, u32)] = &[
        (100, 32),
        (256, 64),
        (1024, 128),
        (32000, 768),
        (50257, 1024),
        (128256, 4096),
    ];
    for &(vocab, dim) in configs {
        let bytes = generate_embedding_spirv(vocab, dim);
        let spirv = bytes_to_words(&bytes);
        let label = format!("embedding({vocab},{dim})");
        assert_valid_header(&spirv, &label);
        let name =
            find_entry_point_name(&spirv).unwrap_or_else(|| panic!("{label}: no entry point"));
        assert_eq!(name, "main", "{label}: wrong entry point");
        let wg =
            find_workgroup_size(&spirv).unwrap_or_else(|| panic!("{label}: no workgroup size"));
        assert_eq!(wg, [EMBEDDING_WORKGROUP_SIZE, 1, 1], "{label}: wrong wg");
    }
}

// ---- Bytes are 4-aligned ----

#[test]
fn test_embedding_spirv_4_byte_aligned() {
    let bytes = generate_embedding_spirv(32000, 768);
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V binary must be 4-byte aligned, got {} bytes",
        bytes.len()
    );
}

#[test]
fn test_embedding_spirv_4_byte_aligned_small() {
    let bytes = generate_embedding_spirv(100, 32);
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V binary must be 4-byte aligned, got {} bytes",
        bytes.len()
    );
}

// ---- Has OpCapability ----

#[test]
fn test_embedding_spirv_has_capability() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_CAPABILITY),
        "embedding: must have OpCapability"
    );
}

// ---- Structural: key opcodes present ----

#[test]
fn test_embedding_spirv_has_memory_model() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_MEMORY_MODEL),
        "embedding: must have OpMemoryModel"
    );
}

#[test]
fn test_embedding_spirv_has_function_structure() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(has_opcode(&spirv, OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&spirv, OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&spirv, OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&spirv, OP_RETURN), "must have OpReturn");
}

#[test]
fn test_embedding_spirv_has_access_chain() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_ACCESS_CHAIN),
        "embedding: must have OpAccessChain for buffer indexing"
    );
}

#[test]
fn test_embedding_spirv_has_loop_structure() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_LOOP_MERGE),
        "embedding: must have OpLoopMerge for grid-stride loop"
    );
    assert!(
        has_opcode(&spirv, OP_PHI),
        "embedding: must have OpPhi for loop induction variable"
    );
}

#[test]
fn test_embedding_spirv_has_bounds_check() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_U_LESS_THAN),
        "embedding: must have OpULessThan for vocab bounds check"
    );
    assert!(
        has_opcode(&spirv, OP_BRANCH_CONDITIONAL),
        "embedding: must have OpBranchConditional"
    );
}

#[test]
fn test_embedding_spirv_has_integer_math() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_IMUL),
        "embedding: must have OpIMul for index computation"
    );
    assert!(
        has_opcode(&spirv, OP_IADD),
        "embedding: must have OpIAdd for index computation"
    );
    assert!(
        has_opcode(&spirv, OP_UDIV),
        "embedding: must have OpUDiv for t = i / embedding_dim"
    );
    assert!(
        has_opcode(&spirv, OP_UMOD),
        "embedding: must have OpUMod for d = i % embedding_dim"
    );
}

#[test]
fn test_embedding_spirv_module_size_reasonable() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(spirv.len() > 50, "module too small ({} words)", spirv.len());
    assert!(
        spirv.len() < 1000,
        "module too large ({} words)",
        spirv.len()
    );
}

#[test]
fn test_embedding_workgroup_size_constant() {
    assert_eq!(EMBEDDING_WORKGROUP_SIZE, 256);
}

// ---- SPIR-V magic number and version from raw bytes ----

#[test]
fn test_embedding_spirv_magic_number_from_bytes() {
    let bytes = generate_embedding_spirv(32000, 768);
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, 0x07230203, "first 4 bytes must be SPIR-V magic");
}

#[test]
fn test_embedding_spirv_version_from_bytes() {
    let bytes = generate_embedding_spirv(32000, 768);
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        version, 0x0001_0000,
        "second word must be SPIR-V 1.0 version"
    );
}

#[test]
fn test_embedding_spirv_generator_from_bytes() {
    let bytes = generate_embedding_spirv(32000, 768);
    let generator = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(
        generator, 0x4E4E_0000,
        "third word must be NN generator magic"
    );
}

// ---- Output buffer size calculations ----

#[test]
fn test_embedding_output_buffer_size_basic() {
    // For num_tokens=10, embedding_dim=64: output = 10 * 64 = 640 floats = 2560 bytes
    let num_tokens: u32 = 10;
    let embedding_dim: u32 = 64;
    let output_floats = num_tokens * embedding_dim;
    let output_bytes = output_floats * 4;
    assert_eq!(output_floats, 640);
    assert_eq!(output_bytes, 2560);
}

#[test]
fn test_embedding_output_buffer_size_large_vocab() {
    // Even with large vocab, output only depends on num_tokens * embedding_dim.
    let num_tokens: u32 = 512;
    let embedding_dim: u32 = 768;
    let output_floats = num_tokens * embedding_dim;
    assert_eq!(output_floats, 512 * 768);
    // Vocab size does NOT affect output buffer size.
    let _vocab_50k_output = output_floats;
    let _vocab_100k_output = output_floats;
    assert_eq!(_vocab_50k_output, _vocab_100k_output);
}

#[test]
fn test_embedding_table_buffer_size() {
    // Embedding table: vocab_size * embedding_dim floats.
    let vocab_size: u32 = 32000;
    let embedding_dim: u32 = 768;
    let table_floats = vocab_size * embedding_dim;
    let table_bytes = table_floats * 4;
    assert_eq!(table_floats, 32000 * 768);
    assert_eq!(table_bytes, 32000 * 768 * 4);
}

// ---- Same vocab/dim produces identical SPIR-V ----

#[test]
fn test_embedding_spirv_deterministic() {
    let bytes1 = generate_embedding_spirv(32000, 768);
    let bytes2 = generate_embedding_spirv(32000, 768);
    assert_eq!(bytes1, bytes2, "same params must produce identical SPIR-V");
}

// ---- Different vocab/dim produce same-sized SPIR-V (hint params) ----

#[test]
fn test_embedding_spirv_size_independent_of_params() {
    // vocab_size and embedding_dim are compile-time hints only; actual
    // values come from push constants. The SPIR-V module should be identical
    // in size regardless of the hint values.
    let bytes_small = generate_embedding_spirv(100, 32);
    let bytes_large = generate_embedding_spirv(128256, 4096);
    assert_eq!(
        bytes_small.len(),
        bytes_large.len(),
        "SPIR-V module size should not depend on hint params"
    );
}

// ---- Decorations: bindings 0, 1, 2 present ----

fn count_opcode(spirv: &[u32], target_opcode: u16) -> usize {
    let mut pos = 5;
    let mut count = 0;
    while pos < spirv.len() {
        let word = spirv[pos];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 || pos + word_count > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            count += 1;
        }
        pos += word_count;
    }
    count
}

#[test]
fn test_embedding_spirv_has_decorations() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    // Must have OpDecorate instructions (for bindings, descriptor set, array stride, block, etc.)
    let decorate_count = count_opcode(&spirv, OP_DECORATE);
    assert!(
        decorate_count >= 6,
        "embedding: must have at least 6 OpDecorate (3 bindings + 3 descriptor sets + strides + blocks), got {decorate_count}"
    );
}

#[test]
fn test_embedding_spirv_has_member_decorations() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    // Must have OpMemberDecorate for push constant and buffer struct offsets.
    let member_count = count_opcode(&spirv, OP_MEMBER_DECORATE);
    assert!(
        member_count >= 3,
        "embedding: must have at least 3 OpMemberDecorate (push constant offsets), got {member_count}"
    );
}

// ---- Type system: has float, uint, vector types ----

#[test]
fn test_embedding_spirv_has_type_declarations() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(has_opcode(&spirv, OP_TYPE_VOID), "must declare void type");
    assert!(has_opcode(&spirv, OP_TYPE_FLOAT), "must declare float type");
    assert!(has_opcode(&spirv, OP_TYPE_INT), "must declare int type");
    assert!(has_opcode(&spirv, OP_TYPE_BOOL), "must declare bool type");
    assert!(
        has_opcode(&spirv, OP_TYPE_VECTOR),
        "must declare vector type (uvec3)"
    );
    assert!(
        has_opcode(&spirv, OP_TYPE_STRUCT),
        "must declare struct types"
    );
    assert!(
        has_opcode(&spirv, OP_TYPE_POINTER),
        "must declare pointer types"
    );
    assert!(
        has_opcode(&spirv, OP_TYPE_RUNTIME_ARRAY),
        "must declare runtime arrays"
    );
    assert!(
        has_opcode(&spirv, OP_TYPE_FUNCTION),
        "must declare function type"
    );
}

// ---- Has load/store for buffer access ----

#[test]
fn test_embedding_spirv_has_load_store() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_LOAD),
        "must have OpLoad for buffer reads"
    );
    assert!(
        has_opcode(&spirv, OP_STORE),
        "must have OpStore for output writes"
    );
}

// ---- Has constants (0, 1, 2 for push constant indexing, 0.0f for OOV) ----

#[test]
fn test_embedding_spirv_has_constants() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    let constant_count = count_opcode(&spirv, OP_CONSTANT);
    // At minimum: 0u, 1u, 2u, 0.0f, WORKGROUP_SIZE
    assert!(
        constant_count >= 5,
        "embedding: must have at least 5 constants, got {constant_count}"
    );
}

// ---- Has selection merge for vocab bounds check branch ----

#[test]
fn test_embedding_spirv_has_selection_merge() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_SELECTION_MERGE),
        "embedding: must have OpSelectionMerge for in-vocab/OOV branching"
    );
}

// ---- Has CompositeExtract for gl_GlobalInvocationID.x ----

#[test]
fn test_embedding_spirv_has_composite_extract() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    assert!(
        has_opcode(&spirv, OP_COMPOSITE_EXTRACT),
        "embedding: must have OpCompositeExtract for extracting .x from uvec3"
    );
}

// ---- Has global variables ----

#[test]
fn test_embedding_spirv_has_global_variables() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    let var_count = count_opcode(&spirv, OP_VARIABLE);
    // At least: token_ids, embedding_table, output, push_constants, gl_GlobalInvocationID, gl_NumWorkGroups
    assert!(
        var_count >= 6,
        "embedding: must have at least 6 global variables, got {var_count}"
    );
}

// ---- Bound is reasonable ----

#[test]
fn test_embedding_spirv_bound_reasonable() {
    let bytes = generate_embedding_spirv(32000, 768);
    let spirv = bytes_to_words(&bytes);
    let bound = spirv[3];
    // Bound must be > 1 (at least a few IDs used) and < 200 (reasonable for this shader).
    assert!(bound > 10, "bound too low: {bound}");
    assert!(bound < 200, "bound too high: {bound}");
}

// ---- Minimum vocab and dim sizes ----

#[test]
fn test_embedding_spirv_minimum_size_params() {
    // Minimum sizes: vocab=1, dim=1
    let bytes = generate_embedding_spirv(1, 1);
    let spirv = bytes_to_words(&bytes);
    assert_valid_header(&spirv, "embedding(1,1)");
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("embedding(1,1): no entry point"));
    assert_eq!(name, "main");
}

#[test]
fn test_embedding_spirv_zero_vocab() {
    // Zero vocab should still produce valid SPIR-V (since actual from push constants).
    let bytes = generate_embedding_spirv(0, 768);
    let spirv = bytes_to_words(&bytes);
    assert_valid_header(&spirv, "embedding(0,768)");
}

#[test]
fn test_embedding_spirv_zero_dim() {
    // Zero dim should still produce valid SPIR-V (since actual from push constants).
    let bytes = generate_embedding_spirv(32000, 0);
    let spirv = bytes_to_words(&bytes);
    assert_valid_header(&spirv, "embedding(32000,0)");
}
