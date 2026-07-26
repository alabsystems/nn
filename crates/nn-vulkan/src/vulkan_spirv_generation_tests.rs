// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for Vulkan SPIR-V generation and dispatch.
//!
//! Covers:
//! 1.  SPIR-V header validity (magic, version, generator)
//! 2.  Workgroup size configuration (local dimensions match kernel requirements)
//! 3.  Storage buffer bindings (descriptor set bindings for input/output)
//! 4.  Push constant layout (push constant block matches kernel parameters)
//! 5.  Type declarations (float/int/vector types correctly declared)
//! 6.  Elementwise kernel structure (add/mul/relu correct SPIR-V structure)
//! 7.  Reduction kernel (sum/max with workgroup-level reduction pattern)
//! 8.  MatMul kernel (matrix multiply with correct loop nesting)
//! 9.  Dispatch dimension calculation (global/local workgroup size from tensor shape)
//! 10. DType to SPIR-V type mapping (F32->OpTypeFloat 32, F16->OpTypeFloat 16, etc.)

use crate::spirv_binary::{
    emit_add_spirv, emit_mul_spirv, emit_relu_spirv, emit_scalar_mul_spirv, emit_transpose_spirv,
    find_entry_point_name, find_workgroup_size, BINARY_WORKGROUP_SIZE,
};
use crate::spirv_cast::{
    generate_bf16_to_f32_spirv, generate_f16_to_f32_spirv, generate_f32_to_bf16_spirv,
    generate_f32_to_f16_spirv, CAST_WORKGROUP_SIZE,
};
use crate::spirv_emit::{
    emit_elementwise_glsl, emit_matmul_glsl, glsl_type,
    spirv_type_bytes, DEFAULT_WORKGROUP_SIZE, SPIRV_MAGIC,
};
use crate::spirv_matmul::{generate_matmul_spirv, generate_matmul_spirv_naive, MATMUL_TILE_SIZE};
use crate::spirv_reduction::{
    generate_max_spirv, generate_mean_spirv, generate_softmax_spirv, generate_sum_spirv,
    REDUCTION_WORKGROUP_SIZE,
};
use crate::workgroup::{
    optimal_elementwise_workgroup, push_constants_matmul,
    push_constants_reduction, validate_dispatch, workgroup_count_1d, workgroup_count_2d,
    workgroup_count_row_reduce,
};

// ---------------------------------------------------------------------------
// SPIR-V structural scanning helpers
// ---------------------------------------------------------------------------

/// SPIR-V opcodes referenced in tests.
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_FADD: u16 = 129;
const OP_FMUL: u16 = 133;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_EXT_INST: u16 = 12;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_SELECTION_MERGE: u16 = 247;

/// Decoration constants.
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

/// Storage classes.
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

/// SPIR-V version 1.0 word.
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;

/// Generator magic (nn-vulkan = "NN\0").
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

/// Check whether a SPIR-V word module contains a given opcode.
fn has_opcode(spirv: &[u32], target_opcode: u16) -> bool {
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            return true;
        }
        pos += wc;
    }
    false
}

/// Count occurrences of a given opcode in a SPIR-V word module.
fn count_opcode(spirv: &[u32], target_opcode: u16) -> usize {
    let mut pos = 5;
    let mut count = 0;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == target_opcode {
            count += 1;
        }
        pos += wc;
    }
    count
}

/// Collect all OpDecorate instructions and return (target_id, decoration, operands).
fn collect_decorations(spirv: &[u32]) -> Vec<(u32, u32, Vec<u32>)> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == OP_DECORATE && wc >= 3 {
            let target = spirv[pos + 1];
            let decoration = spirv[pos + 2];
            let ops: Vec<u32> = (3..wc).map(|i| spirv[pos + i]).collect();
            results.push((target, decoration, ops));
        }
        pos += wc;
    }
    results
}

/// Collect all OpMemberDecorate instructions: (struct_id, member, decoration, operands).
fn collect_member_decorations(spirv: &[u32]) -> Vec<(u32, u32, u32, Vec<u32>)> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == OP_MEMBER_DECORATE && wc >= 4 {
            let struct_id = spirv[pos + 1];
            let member = spirv[pos + 2];
            let decoration = spirv[pos + 3];
            let ops: Vec<u32> = (4..wc).map(|i| spirv[pos + i]).collect();
            results.push((struct_id, member, decoration, ops));
        }
        pos += wc;
    }
    results
}

/// Collect OpTypeFloat instructions: (result_id, width).
fn collect_type_floats(spirv: &[u32]) -> Vec<(u32, u32)> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == OP_TYPE_FLOAT && wc == 3 {
            results.push((spirv[pos + 1], spirv[pos + 2]));
        }
        pos += wc;
    }
    results
}

/// Collect OpTypeInt instructions: (result_id, width, signedness).
fn collect_type_ints(spirv: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == OP_TYPE_INT && wc == 4 {
            results.push((spirv[pos + 1], spirv[pos + 2], spirv[pos + 3]));
        }
        pos += wc;
    }
    results
}

/// Collect OpTypeVector instructions: (result_id, component_type, count).
fn collect_type_vectors(spirv: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == OP_TYPE_VECTOR && wc == 4 {
            results.push((spirv[pos + 1], spirv[pos + 2], spirv[pos + 3]));
        }
        pos += wc;
    }
    results
}

/// Collect OpVariable instructions: (result_id, storage_class).
fn collect_variables(spirv: &[u32]) -> Vec<(u32, u32)> {
    let mut results = Vec::new();
    let mut pos = 5;
    while pos < spirv.len() {
        let word = spirv[pos];
        let wc = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if wc == 0 || pos + wc > spirv.len() {
            break;
        }
        if opcode == OP_VARIABLE && wc >= 4 {
            let result_id = spirv[pos + 2];
            let sc = spirv[pos + 3];
            results.push((result_id, sc));
        }
        pos += wc;
    }
    results
}

/// Convert `Vec<u8>` (little-endian bytes) to `Vec<u32>` words for analysis.
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ===========================================================================
// 1. SPIR-V header validity: magic number, version, generator
// ===========================================================================

#[test]
fn test_header_magic_number_all_elementwise_ops() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        assert_eq!(spirv[0], SPIRV_MAGIC, "{name}: SPIR-V magic mismatch");
    }
}

#[test]
fn test_header_version_is_1_0_for_binary_ops() {
    // Binary-emitted ops use SPIR-V 1.0 for max compatibility.
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        assert_eq!(spirv[1], SPIRV_VERSION_1_0, "{name}: version != 1.0");
    }
}

#[test]
fn test_header_generator_magic_is_nn() {
    // Generator word should be 0x4E4E0000 ("NN\0").
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        assert_eq!(spirv[2], GENERATOR_MAGIC, "{name}: generator != NN");
    }
}

#[test]
fn test_header_bound_positive_and_reasonable() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        // Bound (word 3) must be positive and shouldn't be excessively large.
        let bound = spirv[3];
        assert!(bound > 0, "{name}: bound must be > 0");
        assert!(bound < 500, "{name}: bound={bound} unexpectedly large");
    }
}

#[test]
fn test_header_schema_is_zero() {
    // Word 4 (schema) must always be 0 in SPIR-V.
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        assert_eq!(spirv[4], 0, "{name}: schema word must be 0");
    }
}

#[test]
fn test_header_matmul_spirv_bytes() {
    let bytes = generate_matmul_spirv_naive(64, 64, 64);
    assert!(bytes.len() >= 20, "matmul SPIR-V module too small");
    assert_eq!(bytes.len() % 4, 0, "matmul SPIR-V not 4-byte aligned");
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC, "matmul: wrong magic");
    assert_eq!(words[1], SPIRV_VERSION_1_0, "matmul: wrong version");
    assert_eq!(words[2], GENERATOR_MAGIC, "matmul: wrong generator");
}

#[test]
fn test_header_reduction_spirv_bytes() {
    let bytes = generate_sum_spirv(1024);
    assert!(bytes.len() >= 20, "reduction SPIR-V module too small");
    assert_eq!(bytes.len() % 4, 0, "reduction SPIR-V not 4-byte aligned");
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC, "sum: wrong magic");
    assert_eq!(words[1], SPIRV_VERSION_1_0, "sum: wrong version");
    assert_eq!(words[2], GENERATOR_MAGIC, "sum: wrong generator");
}

// ===========================================================================
// 2. Workgroup size configuration
// ===========================================================================

#[test]
fn test_workgroup_size_elementwise_256x1x1() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        let wg = find_workgroup_size(&spirv).unwrap_or_else(|| panic!("{name}: no workgroup size"));
        assert_eq!(
            wg,
            [BINARY_WORKGROUP_SIZE, 1, 1],
            "{name}: expected [{BINARY_WORKGROUP_SIZE}, 1, 1], got {wg:?}"
        );
    }
}

#[test]
fn test_workgroup_size_matmul_naive_is_tile_size_squared() {
    let bytes = generate_matmul_spirv_naive(64, 64, 64);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("matmul naive: no workgroup size");
    assert_eq!(
        wg,
        [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1],
        "matmul naive workgroup should be [{MATMUL_TILE_SIZE}, {MATMUL_TILE_SIZE}, 1]"
    );
}

#[test]
fn test_workgroup_size_matmul_tiled_is_tile_size_squared() {
    let bytes = generate_matmul_spirv(128, 128, 64);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("matmul tiled: no workgroup size");
    assert_eq!(
        wg,
        [MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1],
        "matmul tiled workgroup should be [{MATMUL_TILE_SIZE}, {MATMUL_TILE_SIZE}, 1]"
    );
}

#[test]
fn test_workgroup_size_reduction_256x1x1() {
    let bytes = generate_sum_spirv(1024);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("sum reduction: no workgroup size");
    assert_eq!(
        wg,
        [REDUCTION_WORKGROUP_SIZE, 1, 1],
        "reduction workgroup should be [{REDUCTION_WORKGROUP_SIZE}, 1, 1]"
    );
}

#[test]
fn test_workgroup_size_max_reduction() {
    let bytes = generate_max_spirv(512);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("max reduction: no workgroup size");
    assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_workgroup_size_mean_reduction() {
    let bytes = generate_mean_spirv(2048);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("mean reduction: no workgroup size");
    assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_workgroup_size_softmax_reduction() {
    let bytes = generate_softmax_spirv(32, 256);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("softmax: no workgroup size");
    assert_eq!(wg, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_workgroup_size_cast_ops() {
    // Cast ops should use CAST_WORKGROUP_SIZE (256).
    let f32_to_f16 = generate_f32_to_f16_spirv(1024);
    let wg = find_workgroup_size(&f32_to_f16).expect("f32_to_f16: no workgroup size");
    assert_eq!(wg, [CAST_WORKGROUP_SIZE, 1, 1]);

    let f16_to_f32 = generate_f16_to_f32_spirv(1024);
    let wg = find_workgroup_size(&f16_to_f32).expect("f16_to_f32: no workgroup size");
    assert_eq!(wg, [CAST_WORKGROUP_SIZE, 1, 1]);
}

// ===========================================================================
// 3. Storage buffer bindings (descriptor set bindings for input/output)
// ===========================================================================

#[test]
fn test_add_spirv_has_three_storage_buffers() {
    // Add: binding 0 (A), binding 1 (B), binding 2 (C).
    let spirv = emit_add_spirv().unwrap();
    let decorations = collect_decorations(&spirv);
    let bindings: Vec<u32> = decorations
        .iter()
        .filter(|(_, dec, _)| *dec == DECORATION_BINDING)
        .map(|(_, _, ops)| ops[0])
        .collect();
    assert!(bindings.contains(&0), "add: missing binding 0");
    assert!(bindings.contains(&1), "add: missing binding 1");
    assert!(bindings.contains(&2), "add: missing binding 2");
}

#[test]
fn test_relu_spirv_has_two_storage_buffers() {
    // ReLU: binding 0 (input), binding 1 (output).
    let spirv = emit_relu_spirv().unwrap();
    let decorations = collect_decorations(&spirv);
    let bindings: Vec<u32> = decorations
        .iter()
        .filter(|(_, dec, _)| *dec == DECORATION_BINDING)
        .map(|(_, _, ops)| ops[0])
        .collect();
    assert!(bindings.contains(&0), "relu: missing binding 0");
    assert!(bindings.contains(&1), "relu: missing binding 1");
    // Should NOT have binding 2 (unary op).
    assert!(
        !bindings.contains(&2),
        "relu: unexpected binding 2 for unary op"
    );
}

#[test]
fn test_all_elementwise_ops_use_descriptor_set_0() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        let decorations = collect_decorations(&spirv);
        let sets: Vec<u32> = decorations
            .iter()
            .filter(|(_, dec, _)| *dec == DECORATION_DESCRIPTOR_SET)
            .map(|(_, _, ops)| ops[0])
            .collect();
        assert!(
            sets.iter().all(|s| *s == 0),
            "{name}: all bindings should use descriptor set 0, got {sets:?}"
        );
    }
}

#[test]
fn test_storage_buffer_variables_exist() {
    let spirv = emit_add_spirv().unwrap();
    let vars = collect_variables(&spirv);
    let sb_vars: Vec<_> = vars
        .iter()
        .filter(|(_, sc)| *sc == STORAGE_CLASS_STORAGE_BUFFER)
        .collect();
    // Add op has 3 storage buffers: A, B, C.
    assert_eq!(
        sb_vars.len(),
        3,
        "add op should have 3 StorageBuffer variables, got {}",
        sb_vars.len()
    );
}

#[test]
fn test_push_constant_variable_exists() {
    let spirv = emit_add_spirv().unwrap();
    let vars = collect_variables(&spirv);
    let pc_vars: Vec<_> = vars
        .iter()
        .filter(|(_, sc)| *sc == STORAGE_CLASS_PUSH_CONSTANT)
        .collect();
    assert_eq!(
        pc_vars.len(),
        1,
        "add op should have exactly 1 PushConstant variable"
    );
}

#[test]
fn test_matmul_has_three_storage_buffers() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let vars = collect_variables(&words);
    let sb_vars: Vec<_> = vars
        .iter()
        .filter(|(_, sc)| *sc == STORAGE_CLASS_STORAGE_BUFFER)
        .collect();
    assert_eq!(
        sb_vars.len(),
        3,
        "matmul should have 3 StorageBuffer variables (A, B, C), got {}",
        sb_vars.len()
    );
}

#[test]
fn test_reduction_has_two_storage_buffers() {
    let bytes = generate_sum_spirv(256);
    let words = bytes_to_words(&bytes);
    let vars = collect_variables(&words);
    let sb_vars: Vec<_> = vars
        .iter()
        .filter(|(_, sc)| *sc == STORAGE_CLASS_STORAGE_BUFFER)
        .collect();
    assert_eq!(
        sb_vars.len(),
        2,
        "reduction should have 2 StorageBuffer variables (input, output), got {}",
        sb_vars.len()
    );
}

// ===========================================================================
// 4. Push constant layout (matches kernel parameters)
// ===========================================================================

#[test]
fn test_add_push_constant_has_block_decoration() {
    let spirv = emit_add_spirv().unwrap();
    let decorations = collect_decorations(&spirv);
    let blocks: Vec<_> = decorations
        .iter()
        .filter(|(_, dec, _)| *dec == DECORATION_BLOCK)
        .collect();
    // At least 2 Block decorations: buffer structs + push constant struct.
    assert!(
        blocks.len() >= 2,
        "expected at least 2 Block decorations, got {}",
        blocks.len()
    );
}

#[test]
fn test_push_constant_member_offsets_elementwise() {
    // Elementwise push constants: { uint total_elements } -> offset 0.
    let spirv = emit_add_spirv().unwrap();
    let member_decs = collect_member_decorations(&spirv);
    let offsets: Vec<_> = member_decs
        .iter()
        .filter(|(_, _, dec, _)| *dec == DECORATION_OFFSET)
        .map(|(struct_id, member, _, ops)| (*struct_id, *member, ops[0]))
        .collect();
    // At minimum we should have offset 0 for the total_elements member.
    let has_zero_offset = offsets
        .iter()
        .any(|(_, member, off)| *member == 0 && *off == 0);
    assert!(
        has_zero_offset,
        "push constant must have member 0 at offset 0"
    );
}

#[test]
fn test_scalar_mul_push_constant_has_two_members() {
    // Scalar mul has { uint total_elements; float alpha; }
    let spirv = emit_scalar_mul_spirv().unwrap();
    let member_decs = collect_member_decorations(&spirv);
    let offset_decs: Vec<_> = member_decs
        .iter()
        .filter(|(_, _, dec, _)| *dec == DECORATION_OFFSET)
        .collect();
    // Should have offsets for member 0 (offset 0) and member 1 (offset 4).
    let has_offset_0 = offset_decs
        .iter()
        .any(|(_, member, _, ops)| *member == 0 && ops[0] == 0);
    let has_offset_4 = offset_decs
        .iter()
        .any(|(_, member, _, ops)| *member == 1 && ops[0] == 4);
    assert!(has_offset_0, "scalar_mul: missing member 0 at offset 0");
    assert!(has_offset_4, "scalar_mul: missing member 1 at offset 4");
}

#[test]
fn test_transpose_push_constant_has_three_members() {
    // Transpose has { uint total_elements; uint rows; uint cols; }
    let spirv = emit_transpose_spirv().unwrap();
    let member_decs = collect_member_decorations(&spirv);
    let offset_decs: Vec<_> = member_decs
        .iter()
        .filter(|(_, _, dec, _)| *dec == DECORATION_OFFSET)
        .collect();
    let has_offset_0 = offset_decs
        .iter()
        .any(|(_, member, _, ops)| *member == 0 && ops[0] == 0);
    let has_offset_4 = offset_decs
        .iter()
        .any(|(_, member, _, ops)| *member == 1 && ops[0] == 4);
    let has_offset_8 = offset_decs
        .iter()
        .any(|(_, member, _, ops)| *member == 2 && ops[0] == 8);
    assert!(has_offset_0, "transpose: missing offset 0");
    assert!(has_offset_4, "transpose: missing offset 4");
    assert!(has_offset_8, "transpose: missing offset 8");
}

#[test]
fn test_push_constants_matmul_bytes_encode_m_n_k() {
    let m = 128u32;
    let n = 256u32;
    let k = 64u32;
    let pc = push_constants_matmul(m, n, k);
    assert_eq!(pc.len(), 12);
    let m_dec = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]);
    let n_dec = u32::from_le_bytes([pc[4], pc[5], pc[6], pc[7]]);
    let k_dec = u32::from_le_bytes([pc[8], pc[9], pc[10], pc[11]]);
    assert_eq!(m_dec, m);
    assert_eq!(n_dec, n);
    assert_eq!(k_dec, k);
}

#[test]
fn test_push_constants_reduction_bytes_encode_row_size_num_rows() {
    let row_size = 512u32;
    let num_rows = 64u32;
    let pc = push_constants_reduction(row_size, num_rows);
    assert_eq!(pc.len(), 8);
    let rs = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]);
    let nr = u32::from_le_bytes([pc[4], pc[5], pc[6], pc[7]]);
    assert_eq!(rs, row_size);
    assert_eq!(nr, num_rows);
}

// ===========================================================================
// 5. Type declarations (float/int/vector types correctly declared)
// ===========================================================================

#[test]
fn test_elementwise_ops_declare_float32() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        let floats = collect_type_floats(&spirv);
        let has_f32 = floats.iter().any(|(_, width)| *width == 32);
        assert!(has_f32, "{name}: must declare OpTypeFloat 32");
    }
}

#[test]
fn test_elementwise_ops_declare_uint32() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        let ints = collect_type_ints(&spirv);
        let has_u32 = ints
            .iter()
            .any(|(_, width, signedness)| *width == 32 && *signedness == 0);
        assert!(has_u32, "{name}: must declare OpTypeInt 32 0 (unsigned)");
    }
}

#[test]
fn test_elementwise_ops_declare_uvec3_for_global_invocation_id() {
    // gl_GlobalInvocationID is uvec3.
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        let vecs = collect_type_vectors(&spirv);
        let has_uvec3 = vecs.iter().any(|(_, _, count)| *count == 3);
        assert!(
            has_uvec3,
            "{name}: must declare vector with 3 components (uvec3)"
        );
    }
}

#[test]
fn test_ops_declare_bool_type() {
    // All ops with bounds checks need OpTypeBool.
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        assert!(
            has_opcode(&spirv, 20), // OP_TYPE_BOOL = 20
            "{name}: must declare OpTypeBool for bounds check"
        );
    }
}

#[test]
fn test_ops_declare_void_type() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
    ] {
        assert!(
            has_opcode(&spirv, OP_TYPE_VOID),
            "{name}: must declare OpTypeVoid for function return"
        );
    }
}

#[test]
fn test_runtime_array_has_stride_decoration() {
    // Runtime arrays used for buffer data[] must have ArrayStride decoration.
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        let decorations = collect_decorations(&spirv);
        let stride_decs: Vec<_> = decorations
            .iter()
            .filter(|(_, dec, _)| *dec == DECORATION_ARRAY_STRIDE)
            .collect();
        assert!(
            !stride_decs.is_empty(),
            "{name}: runtime array must have ArrayStride decoration"
        );
        // Stride should be 4 for float.
        let all_stride_4 = stride_decs.iter().all(|(_, _, ops)| ops[0] == 4);
        assert!(all_stride_4, "{name}: float array stride must be 4 bytes");
    }
}

#[test]
fn test_matmul_declares_float32_and_uint32() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let floats = collect_type_floats(&words);
    let ints = collect_type_ints(&words);
    assert!(
        floats.iter().any(|(_, w)| *w == 32),
        "matmul must declare OpTypeFloat 32"
    );
    assert!(
        ints.iter().any(|(_, w, s)| *w == 32 && *s == 0),
        "matmul must declare OpTypeInt 32 0"
    );
}

#[test]
fn test_cast_f32_to_f16_declares_both_float_widths() {
    let words = generate_f32_to_f16_spirv(256);
    let floats = collect_type_floats(&words);
    let widths: Vec<u32> = floats.iter().map(|(_, w)| *w).collect();
    assert!(
        widths.contains(&32),
        "f32_to_f16: must declare OpTypeFloat 32"
    );
    assert!(
        widths.contains(&16),
        "f32_to_f16: must declare OpTypeFloat 16"
    );
}

// ===========================================================================
// 6. Elementwise kernel structure (add/mul/relu)
// ===========================================================================

#[test]
fn test_add_spirv_contains_fadd_opcode() {
    let spirv = emit_add_spirv().unwrap();
    assert!(has_opcode(&spirv, OP_FADD), "add must contain OpFAdd");
}

#[test]
fn test_add_spirv_does_not_contain_fmul() {
    // Pure add should not multiply.
    let spirv = emit_add_spirv().unwrap();
    assert!(
        !has_opcode(&spirv, OP_FMUL),
        "add should not contain OpFMul"
    );
}

#[test]
fn test_mul_spirv_contains_fmul_opcode() {
    let spirv = emit_mul_spirv().unwrap();
    assert!(has_opcode(&spirv, OP_FMUL), "mul must contain OpFMul");
}

#[test]
fn test_mul_spirv_does_not_contain_fadd() {
    let spirv = emit_mul_spirv().unwrap();
    assert!(
        !has_opcode(&spirv, OP_FADD),
        "mul should not contain OpFAdd"
    );
}

#[test]
fn test_relu_spirv_uses_glsl_ext_inst_fmax() {
    let spirv = emit_relu_spirv().unwrap();
    assert!(
        has_opcode(&spirv, OP_EXT_INST),
        "relu must use GLSL.std.450 FMax via OpExtInst"
    );
}

#[test]
fn test_all_elementwise_ops_have_bounds_check() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        assert!(
            has_opcode(&spirv, OP_U_GREATER_THAN_EQUAL),
            "{name}: must have bounds check (UGreaterThanEqual)"
        );
        assert!(
            has_opcode(&spirv, OP_BRANCH_CONDITIONAL),
            "{name}: must have BranchConditional for bounds check"
        );
        assert!(
            has_opcode(&spirv, OP_SELECTION_MERGE),
            "{name}: must have SelectionMerge"
        );
    }
}

#[test]
fn test_all_elementwise_ops_have_load_store() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        assert!(has_opcode(&spirv, OP_LOAD), "{name}: must have OpLoad");
        assert!(has_opcode(&spirv, OP_STORE), "{name}: must have OpStore");
        assert!(
            has_opcode(&spirv, OP_ACCESS_CHAIN),
            "{name}: must have OpAccessChain"
        );
    }
}

#[test]
fn test_all_elementwise_ops_have_function_structure() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        assert!(
            has_opcode(&spirv, OP_FUNCTION),
            "{name}: must have OpFunction"
        );
        assert!(
            has_opcode(&spirv, OP_FUNCTION_END),
            "{name}: must have OpFunctionEnd"
        );
        assert!(has_opcode(&spirv, OP_LABEL), "{name}: must have OpLabel");
        assert!(has_opcode(&spirv, OP_RETURN), "{name}: must have OpReturn");
    }
}

#[test]
fn test_all_elementwise_ops_extract_global_invocation_id() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
    ] {
        assert!(
            has_opcode(&spirv, OP_COMPOSITE_EXTRACT),
            "{name}: must extract gl_GlobalInvocationID.x"
        );
    }
}

#[test]
fn test_transpose_uses_udiv_umod_imul_iadd() {
    // Transpose computes row/col from linear index.
    let spirv = emit_transpose_spirv().unwrap();
    assert!(
        has_opcode(&spirv, OP_UDIV),
        "transpose: needs OpUDiv for row"
    );
    assert!(
        has_opcode(&spirv, OP_UMOD),
        "transpose: needs OpUMod for col"
    );
    assert!(
        has_opcode(&spirv, OP_IMUL),
        "transpose: needs OpIMul for dst index"
    );
    assert!(
        has_opcode(&spirv, OP_IADD),
        "transpose: needs OpIAdd for dst index"
    );
}

#[test]
fn test_entry_point_name_is_main_for_all_ops() {
    for (name, spirv) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", emit_transpose_spirv().unwrap()),
    ] {
        let ep = find_entry_point_name(&spirv).unwrap_or_else(|| panic!("{name}: no entry point"));
        assert_eq!(ep, "main", "{name}: entry point must be 'main'");
    }
}

// ===========================================================================
// 7. Reduction kernel structure (sum/max with workgroup-level reduction)
// ===========================================================================

#[test]
fn test_sum_reduction_contains_barrier() {
    let bytes = generate_sum_spirv(1024);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CONTROL_BARRIER),
        "sum reduction must use OpControlBarrier for workgroup sync"
    );
}

#[test]
fn test_max_reduction_contains_barrier() {
    let bytes = generate_max_spirv(1024);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CONTROL_BARRIER),
        "max reduction must use OpControlBarrier"
    );
}

#[test]
fn test_reduction_contains_loop_structure() {
    let bytes = generate_sum_spirv(1024);
    let words = bytes_to_words(&bytes);
    // Reductions use loops (OpLoopMerge + OpPhi for loop variable).
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "sum reduction must have OpLoopMerge for accumulation loop"
    );
    assert!(
        has_opcode(&words, OP_PHI),
        "sum reduction must have OpPhi for loop variable"
    );
}

#[test]
fn test_sum_reduction_contains_fadd() {
    let bytes = generate_sum_spirv(1024);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, OP_FADD), "sum must contain OpFAdd");
}

#[test]
fn test_mean_reduction_has_fadd_and_structure() {
    let bytes = generate_mean_spirv(1024);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, OP_FADD), "mean must have OpFAdd for sum");
    assert!(
        has_opcode(&words, OP_CONTROL_BARRIER),
        "mean must have barrier"
    );
}

#[test]
fn test_softmax_reduction_has_multiple_barriers() {
    // Softmax requires multiple barrier sync points: max pass, exp pass, sum pass.
    let bytes = generate_softmax_spirv(32, 256);
    let words = bytes_to_words(&bytes);
    let barrier_count = count_opcode(&words, OP_CONTROL_BARRIER);
    assert!(
        barrier_count >= 2,
        "softmax must have at least 2 barriers, got {barrier_count}"
    );
}

#[test]
fn test_reduction_entry_point_is_main() {
    for (name, bytes) in [
        ("sum", generate_sum_spirv(256)),
        ("max", generate_max_spirv(256)),
        ("mean", generate_mean_spirv(256)),
    ] {
        let words = bytes_to_words(&bytes);
        let ep = find_entry_point_name(&words).unwrap_or_else(|| panic!("{name}: no entry point"));
        assert_eq!(ep, "main", "{name}: entry point must be 'main'");
    }
}

#[test]
fn test_softmax_entry_point_is_main() {
    let bytes = generate_softmax_spirv(16, 128);
    let words = bytes_to_words(&bytes);
    let ep = find_entry_point_name(&words).expect("softmax: no entry point");
    assert_eq!(ep, "main");
}

#[test]
fn test_reduction_module_sizes_reasonable() {
    for (name, bytes) in [
        ("sum", generate_sum_spirv(1024)),
        ("max", generate_max_spirv(1024)),
        ("mean", generate_mean_spirv(1024)),
        ("softmax", generate_softmax_spirv(32, 256)),
    ] {
        let words = bytes_to_words(&bytes);
        assert!(
            words.len() > 100,
            "{name}: module too small ({} words)",
            words.len()
        );
        assert!(
            words.len() < 5000,
            "{name}: module too large ({} words)",
            words.len()
        );
    }
}

// ===========================================================================
// 8. MatMul kernel (matrix multiply with correct loop nesting)
// ===========================================================================

#[test]
fn test_matmul_naive_contains_loop_structure() {
    let bytes = generate_matmul_spirv_naive(64, 64, 64);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "matmul naive must have OpLoopMerge for k-loop"
    );
    assert!(
        has_opcode(&words, OP_PHI),
        "matmul naive must have OpPhi for accumulator/loop variable"
    );
}

#[test]
fn test_matmul_naive_contains_fmul_and_fadd() {
    // MatMul: acc += A[..] * B[..]
    let bytes = generate_matmul_spirv_naive(64, 64, 64);
    let words = bytes_to_words(&bytes);
    assert!(has_opcode(&words, OP_FMUL), "matmul must have OpFMul");
    assert!(has_opcode(&words, OP_FADD), "matmul must have OpFAdd");
}

#[test]
fn test_matmul_tiled_contains_barrier() {
    // Tiled matmul uses shared memory and barriers.
    let bytes = generate_matmul_spirv(128, 128, 64);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CONTROL_BARRIER),
        "tiled matmul must use OpControlBarrier for shared memory sync"
    );
}

#[test]
fn test_matmul_tiled_contains_loop_and_accumulate() {
    let bytes = generate_matmul_spirv(128, 128, 64);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "tiled matmul: needs loop"
    );
    assert!(has_opcode(&words, OP_FMUL), "tiled matmul: needs fmul");
    assert!(has_opcode(&words, OP_FADD), "tiled matmul: needs fadd");
}

#[test]
fn test_matmul_naive_entry_point_is_main() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let ep = find_entry_point_name(&words).expect("matmul naive: no entry point");
    assert_eq!(ep, "main");
}

#[test]
fn test_matmul_tiled_entry_point_is_main() {
    let bytes = generate_matmul_spirv(64, 64, 64);
    let words = bytes_to_words(&bytes);
    let ep = find_entry_point_name(&words).expect("matmul tiled: no entry point");
    assert_eq!(ep, "main");
}

#[test]
fn test_matmul_push_constant_has_m_n_k() {
    let bytes = generate_matmul_spirv_naive(32, 32, 32);
    let words = bytes_to_words(&bytes);
    let member_decs = collect_member_decorations(&words);
    let offset_decs: Vec<_> = member_decs
        .iter()
        .filter(|(_, _, dec, _)| *dec == DECORATION_OFFSET)
        .collect();
    // Should have offsets at 0, 4, 8 for M, N, K.
    let has_0 = offset_decs
        .iter()
        .any(|(_, m, _, ops)| *m == 0 && ops[0] == 0);
    let has_4 = offset_decs
        .iter()
        .any(|(_, m, _, ops)| *m == 1 && ops[0] == 4);
    let has_8 = offset_decs
        .iter()
        .any(|(_, m, _, ops)| *m == 2 && ops[0] == 8);
    assert!(
        has_0,
        "matmul: missing push constant member at offset 0 (M)"
    );
    assert!(
        has_4,
        "matmul: missing push constant member at offset 4 (N)"
    );
    assert!(
        has_8,
        "matmul: missing push constant member at offset 8 (K)"
    );
}

#[test]
fn test_matmul_tiled_larger_than_naive() {
    // Tiled matmul should be larger due to shared memory and multiple loops.
    let naive = generate_matmul_spirv_naive(64, 64, 64);
    let tiled = generate_matmul_spirv(64, 64, 64);
    assert!(
        tiled.len() > naive.len(),
        "tiled matmul ({} bytes) should be larger than naive ({} bytes)",
        tiled.len(),
        naive.len()
    );
}

#[test]
fn test_matmul_module_sizes_reasonable() {
    for (name, bytes) in [
        ("naive", generate_matmul_spirv_naive(64, 64, 64)),
        ("tiled", generate_matmul_spirv(64, 64, 64)),
    ] {
        let words = bytes_to_words(&bytes);
        assert!(
            words.len() > 100,
            "{name} matmul too small: {} words",
            words.len()
        );
        assert!(
            words.len() < 5000,
            "{name} matmul too large: {} words",
            words.len()
        );
    }
}

// ===========================================================================
// 9. Dispatch dimension calculation (global/local workgroup size from tensor shape)
// ===========================================================================

#[test]
fn test_workgroup_count_1d_exact_multiples() {
    assert_eq!(workgroup_count_1d(256, 256), 1);
    assert_eq!(workgroup_count_1d(512, 256), 2);
    assert_eq!(workgroup_count_1d(1024, 256), 4);
    assert_eq!(workgroup_count_1d(65536, 256), 256);
}

#[test]
fn test_workgroup_count_1d_remainder() {
    assert_eq!(workgroup_count_1d(1, 256), 1);
    assert_eq!(workgroup_count_1d(257, 256), 2);
    assert_eq!(workgroup_count_1d(511, 256), 2);
    assert_eq!(workgroup_count_1d(513, 256), 3);
}

#[test]
fn test_workgroup_count_1d_covers_all_elements() {
    for total in [
        1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1000, 10000,
    ] {
        let groups = workgroup_count_1d(total, 256);
        assert!(groups * 256 >= total, "groups*256 must cover {total}");
        assert!(
            (groups - 1) * 256 < total,
            "too many groups for {total}: {groups}"
        );
    }
}

#[test]
fn test_workgroup_count_2d_matmul_typical_sizes() {
    // Common matmul sizes in ML.
    let tile = MATMUL_TILE_SIZE;
    let cases: &[(u32, u32, [u32; 3])] = &[
        (64, 64, [4, 4, 1]),
        (128, 128, [8, 8, 1]),
        (256, 256, [16, 16, 1]),
        (768, 512, [48, 32, 1]),
        (65, 65, [5, 5, 1]),
    ];
    for &(dx, dy, expected) in cases {
        let actual = workgroup_count_2d(dx, dy, tile);
        assert_eq!(actual, expected, "workgroup_count_2d({dx}, {dy}, {tile})");
    }
}

#[test]
fn test_workgroup_count_row_reduce_matches_rows() {
    for rows in [1, 16, 32, 64, 128, 256, 1024] {
        let grid = workgroup_count_row_reduce(rows);
        assert_eq!(grid, [rows, 1, 1]);
    }
}

#[test]
fn test_optimal_workgroup_matches_binary_workgroup_for_large_tensors() {
    // For any tensor >= 256 elements with max_invocations >= 256, we get 256.
    for total in [256, 512, 1024, 65536, 1_000_000] {
        let wg = optimal_elementwise_workgroup(total, 1024);
        assert_eq!(
            wg, DEFAULT_WORKGROUP_SIZE,
            "optimal({total}) should be {DEFAULT_WORKGROUP_SIZE}"
        );
    }
}

#[test]
fn test_optimal_workgroup_reduces_for_small_tensors() {
    assert_eq!(optimal_elementwise_workgroup(1, 1024), 1);
    assert_eq!(optimal_elementwise_workgroup(2, 1024), 2);
    assert_eq!(optimal_elementwise_workgroup(3, 1024), 2);
    assert_eq!(optimal_elementwise_workgroup(4, 1024), 4);
    assert_eq!(optimal_elementwise_workgroup(5, 1024), 4);
    assert_eq!(optimal_elementwise_workgroup(8, 1024), 8);
    assert_eq!(optimal_elementwise_workgroup(128, 1024), 128);
    assert_eq!(optimal_elementwise_workgroup(255, 1024), 128);
}

#[test]
fn test_validate_dispatch_common_ml_configs() {
    // 1D elementwise: 10k elements, wg=256.
    assert!(validate_dispatch([40, 1, 1], [256, 1, 1], 65535, 1024).is_ok());
    // 2D matmul: 128x128 / tile=16 = 8x8 workgroups.
    assert!(validate_dispatch([8, 8, 1], [16, 16, 1], 65535, 1024).is_ok());
    // Reduction: 512 rows.
    assert!(validate_dispatch([512, 1, 1], [256, 1, 1], 65535, 1024).is_ok());
}

#[test]
fn test_validate_dispatch_rejects_zero_dimensions() {
    assert!(validate_dispatch([0, 1, 1], [256, 1, 1], 65535, 1024).is_err());
    assert!(validate_dispatch([1, 0, 1], [256, 1, 1], 65535, 1024).is_err());
    assert!(validate_dispatch([1, 1, 0], [256, 1, 1], 65535, 1024).is_err());
}

#[test]
fn test_validate_dispatch_rejects_exceeding_group_count() {
    assert!(validate_dispatch([65536, 1, 1], [256, 1, 1], 65535, 1024).is_err());
    assert!(validate_dispatch([1, 65536, 1], [256, 1, 1], 65535, 1024).is_err());
    assert!(validate_dispatch([1, 1, 65536], [256, 1, 1], 65535, 1024).is_err());
}

#[test]
fn test_validate_dispatch_rejects_exceeding_invocations() {
    // 32*32*2 = 2048 > 1024.
    assert!(validate_dispatch([1, 1, 1], [32, 32, 2], 65535, 1024).is_err());
}

// ===========================================================================
// 10. DType to SPIR-V type mapping
// ===========================================================================

#[test]
fn test_glsl_type_f32_maps_to_float() {
    use nn_dsl::ScalarType;
    assert_eq!(glsl_type(ScalarType::F32).unwrap(), "float");
}

#[test]
fn test_glsl_type_f16_maps_to_float16_t() {
    use nn_dsl::ScalarType;
    assert_eq!(glsl_type(ScalarType::F16).unwrap(), "float16_t");
}

#[test]
fn test_glsl_type_bf16_is_unsupported() {
    use nn_dsl::ScalarType;
    let result = glsl_type(ScalarType::BF16);
    assert!(result.is_err(), "bf16 should not have native GLSL type");
}

#[test]
fn test_spirv_type_bytes_f32_is_4() {
    use nn_dsl::ScalarType;
    assert_eq!(spirv_type_bytes(ScalarType::F32).unwrap(), 4);
}

#[test]
fn test_spirv_type_bytes_f16_is_2() {
    use nn_dsl::ScalarType;
    assert_eq!(spirv_type_bytes(ScalarType::F16).unwrap(), 2);
}

#[test]
fn test_spirv_type_bytes_bf16_is_2() {
    use nn_dsl::ScalarType;
    assert_eq!(spirv_type_bytes(ScalarType::BF16).unwrap(), 2);
}

#[test]
fn test_cast_f32_to_f16_spirv_structure() {
    let words = generate_f32_to_f16_spirv(1024);
    assert_eq!(words[0], SPIRV_MAGIC, "f32_to_f16: wrong magic");
    let ep = find_entry_point_name(&words).expect("f32_to_f16: no entry point");
    assert_eq!(ep, "main");
    // Should have OpTypeFloat 32 and OpTypeFloat 16.
    let floats = collect_type_floats(&words);
    let widths: Vec<u32> = floats.iter().map(|(_, w)| *w).collect();
    assert!(widths.contains(&32), "f32_to_f16: must have float32");
    assert!(widths.contains(&16), "f32_to_f16: must have float16");
}

#[test]
fn test_cast_f16_to_f32_spirv_structure() {
    let words = generate_f16_to_f32_spirv(1024);
    assert_eq!(words[0], SPIRV_MAGIC, "f16_to_f32: wrong magic");
    let ep = find_entry_point_name(&words).expect("f16_to_f32: no entry point");
    assert_eq!(ep, "main");
    let floats = collect_type_floats(&words);
    let widths: Vec<u32> = floats.iter().map(|(_, w)| *w).collect();
    assert!(widths.contains(&32), "f16_to_f32: must have float32");
    assert!(widths.contains(&16), "f16_to_f32: must have float16");
}

#[test]
fn test_cast_f32_to_bf16_uses_uint_emulation() {
    // BF16 has no native SPIR-V type. Uses uint16 bitwise truncation.
    let words = generate_f32_to_bf16_spirv(512);
    assert_eq!(words[0], SPIRV_MAGIC, "f32_to_bf16: wrong magic");
    let ints = collect_type_ints(&words);
    // Should have 32-bit uint and 16-bit uint for emulation.
    let has_u32 = ints.iter().any(|(_, w, _)| *w == 32);
    let has_u16 = ints.iter().any(|(_, w, _)| *w == 16);
    assert!(has_u32, "f32_to_bf16: must have uint32 for bitcast");
    assert!(has_u16, "f32_to_bf16: must have uint16 for bf16 storage");
}

#[test]
fn test_cast_bf16_to_f32_uses_uint_emulation() {
    let words = generate_bf16_to_f32_spirv(512);
    assert_eq!(words[0], SPIRV_MAGIC, "bf16_to_f32: wrong magic");
    let ints = collect_type_ints(&words);
    let has_u32 = ints.iter().any(|(_, w, _)| *w == 32);
    let has_u16 = ints.iter().any(|(_, w, _)| *w == 16);
    assert!(has_u32, "bf16_to_f32: must have uint32 for bitcast");
    assert!(has_u16, "bf16_to_f32: must have uint16 for bf16 storage");
}

#[test]
fn test_glsl_emission_elementwise_workgroup_matches_spirv_binary() {
    // GLSL emission and SPIR-V binary should use the same default workgroup size.
    let glsl = emit_elementwise_glsl("test", "x", DEFAULT_WORKGROUP_SIZE).unwrap();
    assert!(
        glsl.contains(&format!("local_size_x = {DEFAULT_WORKGROUP_SIZE}")),
        "GLSL should use DEFAULT_WORKGROUP_SIZE={DEFAULT_WORKGROUP_SIZE}"
    );
    assert_eq!(
        DEFAULT_WORKGROUP_SIZE, BINARY_WORKGROUP_SIZE,
        "GLSL and binary workgroup sizes must match"
    );
}

#[test]
fn test_glsl_matmul_tile_matches_spirv_matmul_tile() {
    // GLSL matmul and SPIR-V matmul should use consistent tile sizes.
    let glsl = emit_matmul_glsl(MATMUL_TILE_SIZE).unwrap();
    assert!(
        glsl.contains(&format!("local_size_x = {MATMUL_TILE_SIZE}")),
        "GLSL matmul should use MATMUL_TILE_SIZE={MATMUL_TILE_SIZE}"
    );
    let bytes = generate_matmul_spirv_naive(64, 64, 64);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).unwrap();
    assert_eq!(wg[0], MATMUL_TILE_SIZE);
    assert_eq!(wg[1], MATMUL_TILE_SIZE);
}

#[test]
fn test_all_binary_ops_return_nonempty_result() {
    assert!(!emit_add_spirv().unwrap().is_empty());
    assert!(!emit_mul_spirv().unwrap().is_empty());
    assert!(!emit_relu_spirv().unwrap().is_empty());
    assert!(!emit_scalar_mul_spirv().unwrap().is_empty());
    assert!(!emit_transpose_spirv().unwrap().is_empty());
}

#[test]
fn test_all_reduction_ops_return_4byte_aligned_bytes() {
    for (name, bytes) in [
        ("sum", generate_sum_spirv(256)),
        ("max", generate_max_spirv(256)),
        ("mean", generate_mean_spirv(256)),
        ("softmax", generate_softmax_spirv(16, 128)),
    ] {
        assert_eq!(
            bytes.len() % 4,
            0,
            "{name}: SPIR-V bytes must be 4-byte aligned"
        );
    }
}

#[test]
fn test_all_cast_ops_have_valid_spirv_headers() {
    for (name, words) in [
        ("f32_to_f16", generate_f32_to_f16_spirv(256)),
        ("f16_to_f32", generate_f16_to_f32_spirv(256)),
        ("f32_to_bf16", generate_f32_to_bf16_spirv(256)),
        ("bf16_to_f32", generate_bf16_to_f32_spirv(256)),
    ] {
        assert!(words.len() >= 5, "{name}: module too short");
        assert_eq!(words[0], SPIRV_MAGIC, "{name}: wrong magic");
        assert!(words[3] > 0, "{name}: bound must be > 0");
        assert_eq!(words[4], 0, "{name}: schema must be 0");
    }
}
