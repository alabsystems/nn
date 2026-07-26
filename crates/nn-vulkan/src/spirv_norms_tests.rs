// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V BatchNorm, GroupNorm, and InstanceNorm kernels.
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Buffer layout (bindings, NonWritable decorations)
//! - Reference implementation correctness against known values
//! - Various channel counts and group configurations

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
const TEST_OP_FSUB: u16 = 131;
const TEST_OP_FMUL: u16 = 133;
const TEST_OP_FDIV: u16 = 136;
const TEST_OP_EXT_INST: u16 = 12;
const TEST_OP_CONTROL_BARRIER: u16 = 224;
const TEST_OP_VARIABLE: u16 = 59;
const TEST_OP_DECORATE: u16 = 71;
const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const TEST_STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const TEST_STORAGE_CLASS_WORKGROUP: u32 = 4;
const TEST_DECORATION_BINDING: u32 = 33;
const TEST_DECORATION_NON_WRITABLE: u32 = 24;

// ---- Helpers ----

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

fn assert_word_counts_consistent(words: &[u32], label: &str) {
    let mut pos = 5;
    let mut instruction_count = 0;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xFFFF;
        assert!(
            word_count > 0,
            "{label}: instruction at pos {pos} has word_count 0 (opcode {opcode})"
        );
        assert!(
            pos + word_count <= words.len(),
            "{label}: instruction at pos {pos} (opcode {opcode}, wc {word_count}) exceeds module length {}",
            words.len()
        );
        pos += word_count;
        instruction_count += 1;
    }
    assert_eq!(
        pos,
        words.len(),
        "{label}: instructions did not consume exactly the full module"
    );
    assert!(
        instruction_count > 10,
        "{label}: expected at least 10 instructions, got {instruction_count}"
    );
}

// ====================================================================
// BatchNorm SPIR-V structural tests
// ====================================================================

#[test]
fn test_batchnorm_spirv_valid_header() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert_valid_header(&words, "batchnorm_64");
}

#[test]
fn test_batchnorm_spirv_valid_header_various_channels() {
    for ch in [16, 32, 64, 128, 256, 512] {
        let words = generate_batchnorm_spirv(ch, NORM_WORKGROUP_SIZE);
        assert_valid_header(&words, &format!("batchnorm_{ch}"));
    }
}

#[test]
fn test_batchnorm_spirv_entry_point_is_main() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_batchnorm_spirv_workgroup_size() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [NORM_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_batchnorm_spirv_has_capability() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "must have OpCapability"
    );
}

#[test]
fn test_batchnorm_spirv_has_memory_model() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "must have OpMemoryModel"
    );
}

#[test]
fn test_batchnorm_spirv_has_function_structure() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_batchnorm_spirv_has_fsub_for_mean_subtraction() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_FSUB),
        "batchnorm must have OpFSub for (x - running_mean)"
    );
}

#[test]
fn test_batchnorm_spirv_has_ext_inst_for_sqrt() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "batchnorm must have OpExtInst for sqrt(var + eps)"
    );
}

#[test]
fn test_batchnorm_spirv_has_fmul_and_fadd() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_FMUL),
        "batchnorm must have OpFMul for weight scaling"
    );
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "batchnorm must have OpFAdd for bias addition"
    );
}

#[test]
fn test_batchnorm_spirv_has_fdiv() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_FDIV),
        "batchnorm must have OpFDiv for normalization"
    );
}

#[test]
fn test_batchnorm_spirv_has_six_storage_buffer_variables() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 6,
        "batchnorm must have 6 StorageBuffer variables (input+mean+var+weight+bias+output), got {sb_count}"
    );
}

#[test]
fn test_batchnorm_spirv_binding_numbers() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    for i in 0..=5 {
        assert!(bindings.contains(&i), "batchnorm must have binding {i}");
    }
}

#[test]
fn test_batchnorm_spirv_nonwritable_count() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nw_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    // input, mean, var, weight, bias = 5 NonWritable
    assert_eq!(
        nw_count, 5,
        "batchnorm should have 5 NonWritable decorations, got {nw_count}"
    );
}

#[test]
fn test_batchnorm_spirv_word_counts_consistent() {
    let words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert_word_counts_consistent(&words, "batchnorm");
}

#[test]
fn test_batchnorm_spirv_deterministic() {
    let w1 = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let w2 = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert_eq!(w1, w2, "SPIR-V output must be deterministic");
}

// ====================================================================
// GroupNorm SPIR-V structural tests
// ====================================================================

#[test]
fn test_groupnorm_spirv_valid_header() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert_valid_header(&words, "groupnorm_8g_64ch");
}

#[test]
fn test_groupnorm_spirv_valid_header_various() {
    for (g, c) in [(1, 32), (2, 64), (4, 128), (8, 256), (32, 256)] {
        let words = generate_groupnorm_spirv(g, c, NORM_WORKGROUP_SIZE);
        assert_valid_header(&words, &format!("groupnorm_{g}g_{c}ch"));
    }
}

#[test]
fn test_groupnorm_spirv_entry_point_is_main() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_groupnorm_spirv_workgroup_size() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [NORM_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_groupnorm_spirv_has_function_structure() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_groupnorm_spirv_has_loops_and_phi() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert!(has_opcode(&words, TEST_OP_LOOP_MERGE), "must have loops");
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "must have OpPhi for loop variables"
    );
    // GroupNorm: sum loop + variance loop + normalize loop + 2 tree reductions = 5+
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    assert!(
        loop_count >= 3,
        "groupnorm should have at least 3 loops, found {loop_count}"
    );
}

#[test]
fn test_groupnorm_spirv_has_fsub_for_mean() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_FSUB),
        "groupnorm must have OpFSub for (x - mean)"
    );
}

#[test]
fn test_groupnorm_spirv_has_barriers() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let barrier_count = count_opcode(&words, TEST_OP_CONTROL_BARRIER);
    assert!(
        barrier_count >= 3,
        "groupnorm must have at least 3 barriers, found {barrier_count}"
    );
}

#[test]
fn test_groupnorm_spirv_has_ext_inst_for_sqrt() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "groupnorm must use sqrt via GLSL.std.450"
    );
}

#[test]
fn test_groupnorm_spirv_has_workgroup_shared_memory() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let wg_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_WORKGROUP)
        .count();
    assert!(
        wg_count >= 1,
        "groupnorm must have workgroup shared memory, found {wg_count}"
    );
}

#[test]
fn test_groupnorm_spirv_four_storage_buffers() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 4,
        "groupnorm must have 4 StorageBuffer variables (input+weight+bias+output), got {sb_count}"
    );
}

#[test]
fn test_groupnorm_spirv_binding_numbers() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    for i in 0..=3 {
        assert!(bindings.contains(&i), "groupnorm must have binding {i}");
    }
}

#[test]
fn test_groupnorm_spirv_nonwritable_count() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nw_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    assert_eq!(
        nw_count, 3,
        "groupnorm should have 3 NonWritable (input+weight+bias), got {nw_count}"
    );
}

#[test]
fn test_groupnorm_spirv_word_counts_consistent() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert_word_counts_consistent(&words, "groupnorm");
}

#[test]
fn test_groupnorm_spirv_deterministic() {
    let w1 = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let w2 = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    assert_eq!(w1, w2, "SPIR-V output must be deterministic");
}

// ====================================================================
// InstanceNorm SPIR-V structural tests
// ====================================================================

#[test]
fn test_instancenorm_spirv_valid_header() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert_valid_header(&words, "instancenorm_64");
}

#[test]
fn test_instancenorm_spirv_valid_header_various_channels() {
    for ch in [16, 32, 64, 128, 256, 512] {
        let words = generate_instancenorm_spirv(ch, NORM_WORKGROUP_SIZE);
        assert_valid_header(&words, &format!("instancenorm_{ch}"));
    }
}

#[test]
fn test_instancenorm_spirv_entry_point_is_main() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_instancenorm_spirv_workgroup_size() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [NORM_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_instancenorm_spirv_has_function_structure() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(has_opcode(&words, TEST_OP_FUNCTION), "must have OpFunction");
    assert!(
        has_opcode(&words, TEST_OP_FUNCTION_END),
        "must have OpFunctionEnd"
    );
    assert!(has_opcode(&words, TEST_OP_LABEL), "must have OpLabel");
    assert!(has_opcode(&words, TEST_OP_RETURN), "must have OpReturn");
}

#[test]
fn test_instancenorm_spirv_has_loops_and_phi() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(has_opcode(&words, TEST_OP_LOOP_MERGE), "must have loops");
    assert!(has_opcode(&words, TEST_OP_PHI), "must have OpPhi");
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    assert!(
        loop_count >= 3,
        "instancenorm should have at least 3 loops, found {loop_count}"
    );
}

#[test]
fn test_instancenorm_spirv_has_fsub() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_FSUB),
        "instancenorm must have OpFSub for (x - mean)"
    );
}

#[test]
fn test_instancenorm_spirv_has_barriers() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let barrier_count = count_opcode(&words, TEST_OP_CONTROL_BARRIER);
    assert!(
        barrier_count >= 3,
        "instancenorm must have at least 3 barriers, found {barrier_count}"
    );
}

#[test]
fn test_instancenorm_spirv_has_ext_inst() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "instancenorm must use sqrt via GLSL.std.450"
    );
}

#[test]
fn test_instancenorm_spirv_four_storage_buffers() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 4,
        "instancenorm must have 4 StorageBuffer variables, got {sb_count}"
    );
}

#[test]
fn test_instancenorm_spirv_has_workgroup_shared_memory() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let wg_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_WORKGROUP)
        .count();
    assert!(
        wg_count >= 1,
        "instancenorm must have shared memory, found {wg_count}"
    );
}

#[test]
fn test_instancenorm_spirv_nonwritable_count() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nw_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    assert_eq!(
        nw_count, 3,
        "instancenorm should have 3 NonWritable, got {nw_count}"
    );
}

#[test]
fn test_instancenorm_spirv_word_counts_consistent() {
    let words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert_word_counts_consistent(&words, "instancenorm");
}

#[test]
fn test_instancenorm_spirv_deterministic() {
    let w1 = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    let w2 = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert_eq!(w1, w2, "SPIR-V output must be deterministic");
}

// ====================================================================
// BatchNorm CPU reference tests
// ====================================================================

#[test]
fn test_batchnorm_reference_identity_params() {
    // With mean=0, var=1, weight=1, bias=0: output == input.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mean = vec![0.0];
    let var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 1, 4, 1e-5);
    for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
        assert!(
            (out - inp / (1.0 + 1e-5_f32).sqrt()).abs() < 1e-4,
            "output[{i}] = {out}, expected ~{inp}"
        );
    }
}

#[test]
fn test_batchnorm_reference_known_values() {
    // Simple test: 1 sample, 2 channels, 2 spatial.
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [N=1, C=2, S=2]
    let mean = vec![1.5, 3.5]; // channel means
    let var = vec![0.25, 0.25]; // channel variances
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 2, 2, eps);

    let inv_std_0 = 1.0 / (0.25 + eps).sqrt();
    let inv_std_1 = 1.0 / (0.25 + eps).sqrt();
    let expected = [(1.0 - 1.5) * inv_std_0,
        (2.0 - 1.5) * inv_std_0,
        (3.0 - 3.5) * inv_std_1,
        (4.0 - 3.5) * inv_std_1];
    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-4,
            "output[{i}] = {out}, expected {exp}"
        );
    }
}

#[test]
fn test_batchnorm_reference_weight_and_bias() {
    let input = vec![0.0; 4];
    let mean = vec![0.0, 0.0];
    let var = vec![1.0, 1.0];
    let weight = vec![2.0, 3.0];
    let bias = vec![1.0, -1.0];
    let eps = 1e-5;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 2, 2, eps);
    // normalized = 0, so output = bias
    for &v in &output[0..2] {
        assert!((v - 1.0).abs() < 1e-4, "channel 0 bias should be 1.0");
    }
    for &v in &output[2..4] {
        assert!((v - (-1.0)).abs() < 1e-4, "channel 1 bias should be -1.0");
    }
}

#[test]
fn test_batchnorm_reference_multi_batch() {
    let input = vec![
        1.0, 2.0, // batch 0, ch 0
        3.0, 4.0, // batch 0, ch 1
        5.0, 6.0, // batch 1, ch 0
        7.0, 8.0, // batch 1, ch 1
    ];
    let mean = vec![0.0, 0.0];
    let var = vec![1.0, 1.0];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 0.0;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 2, 2, 2, eps);
    // With zero mean, unit var, unit weight, zero bias: output = input
    for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
        assert!(
            (out - inp).abs() < 1e-6,
            "output[{i}] = {out}, expected {inp}"
        );
    }
}

// ====================================================================
// GroupNorm CPU reference tests
// ====================================================================

#[test]
fn test_groupnorm_reference_single_group_equals_layernorm_like() {
    // With groups=1, GroupNorm normalizes over all channels + spatial.
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [N=1, C=2, S=2]
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 1, 1, 2, 2, eps);

    let mean = 2.5;
    let var_val: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
    let inv_std = 1.0 / (var_val + eps).sqrt();
    for (i, &out) in output.iter().enumerate() {
        let expected = (input[i] - mean) * inv_std;
        assert!(
            (out - expected).abs() < 1e-4,
            "output[{i}] = {out}, expected {expected}"
        );
    }
}

#[test]
fn test_groupnorm_reference_groups_equal_channels_is_instancenorm() {
    // groups == channels => each group has 1 channel = InstanceNorm
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [N=1, C=3, S=2]
    let weight = vec![1.0; 3];
    let bias = vec![0.0; 3];
    let eps = 1e-5;
    let gn_output = groupnorm_reference(&input, &weight, &bias, 1, 3, 3, 2, eps);
    let in_output = instancenorm_reference(&input, &weight, &bias, 1, 3, 2, eps);
    for (i, (&gn, &ins)) in gn_output.iter().zip(in_output.iter()).enumerate() {
        assert!(
            (gn - ins).abs() < 1e-4,
            "groupnorm(g=C) should equal instancenorm: output[{i}]: gn={gn}, in={ins}"
        );
    }
}

#[test]
fn test_groupnorm_reference_known_values() {
    // [N=1, C=4, S=1], groups=2 => cpg=2
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 1, 2, 4, 1, eps);

    // Group 0: channels [0,1], values [1.0, 2.0]
    let mean0 = 1.5;
    let var0 = f32::midpoint((1.0 - 1.5_f32).powi(2), (2.0 - 1.5_f32).powi(2));
    let inv_std0 = 1.0 / (var0 + eps).sqrt();

    // Group 1: channels [2,3], values [3.0, 4.0]
    let mean1 = 3.5;
    let var1 = f32::midpoint((3.0 - 3.5_f32).powi(2), (4.0 - 3.5_f32).powi(2));
    let inv_std1 = 1.0 / (var1 + eps).sqrt();

    let expected = [(1.0 - mean0) * inv_std0,
        (2.0 - mean0) * inv_std0,
        (3.0 - mean1) * inv_std1,
        (4.0 - mean1) * inv_std1];
    for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (out - exp).abs() < 1e-4,
            "output[{i}] = {out}, expected {exp}"
        );
    }
}

#[test]
fn test_groupnorm_reference_weight_and_bias() {
    // Weight=2, Bias=1 for all channels.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0; 4];
    let bias = vec![1.0; 4];
    let eps = 1e-5;
    let output_wb = groupnorm_reference(&input, &weight, &bias, 1, 2, 4, 1, eps);
    let output_plain = groupnorm_reference(&input, &[1.0; 4], &[0.0; 4], 1, 2, 4, 1, eps);

    for (i, (&wb, &plain)) in output_wb.iter().zip(output_plain.iter()).enumerate() {
        let expected = plain * 2.0 + 1.0;
        assert!(
            (wb - expected).abs() < 1e-4,
            "output_wb[{i}] = {wb}, expected {expected}"
        );
    }
}

#[test]
fn test_groupnorm_reference_multi_batch() {
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 2, 2, 4, 1, eps);
    // Verify each batch is normalized independently.
    assert!(
        output.iter().all(|v| v.is_finite()),
        "all outputs must be finite"
    );
    assert_eq!(output.len(), 8);
}

// ====================================================================
// InstanceNorm CPU reference tests
// ====================================================================

#[test]
fn test_instancenorm_reference_unit_params() {
    // With weight=1, bias=0: pure normalization.
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [N=1, C=1, S=4]
    let weight = vec![1.0];
    let bias = vec![0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 1, 4, eps);

    let mean = 2.5;
    let var_val: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
    let inv_std = 1.0 / (var_val + eps).sqrt();
    for (i, &out) in output.iter().enumerate() {
        let expected = (input[i] - mean) * inv_std;
        assert!(
            (out - expected).abs() < 1e-4,
            "output[{i}] = {out}, expected {expected}"
        );
    }
}

#[test]
fn test_instancenorm_reference_known_values() {
    // [N=1, C=2, S=3]
    let input = vec![
        1.0, 2.0, 3.0, // channel 0
        4.0, 5.0, 6.0, // channel 1
    ];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 2, 3, eps);

    // Channel 0: mean=2.0, var=2/3
    let mean0 = 2.0;
    let var0 = ((1.0 - 2.0_f32).powi(2) + 0.0 + (3.0 - 2.0_f32).powi(2)) / 3.0;
    let inv_std0 = 1.0 / (var0 + eps).sqrt();
    for j in 0..3 {
        let expected = (input[j] - mean0) * inv_std0;
        assert!(
            (output[j] - expected).abs() < 1e-4,
            "ch0 output[{j}] = {}, expected {expected}",
            output[j]
        );
    }
}

#[test]
fn test_instancenorm_reference_all_same_value() {
    // If all values in a channel are the same, output should be zero (+ bias).
    let input = vec![5.0; 8]; // [N=1, C=2, S=4]
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 2, 4, eps);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-3,
            "output[{i}] = {v}, expected ~0 for constant input"
        );
    }
}

#[test]
fn test_instancenorm_reference_weight_and_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [N=1, C=2, S=2]
    let weight = vec![2.0, 3.0];
    let bias = vec![1.0, -1.0];
    let eps = 1e-5;
    let output_wb = instancenorm_reference(&input, &weight, &bias, 1, 2, 2, eps);
    let output_plain = instancenorm_reference(&input, &[1.0; 2], &[0.0; 2], 1, 2, 2, eps);

    // Channel 0: output = plain * 2 + 1
    for j in 0..2 {
        let expected = output_plain[j] * 2.0 + 1.0;
        assert!(
            (output_wb[j] - expected).abs() < 1e-4,
            "ch0 output_wb[{j}] = {}, expected {expected}",
            output_wb[j]
        );
    }
    // Channel 1: output = plain * 3 - 1
    for j in 0..2 {
        let expected = output_plain[2 + j] * 3.0 - 1.0;
        assert!(
            (output_wb[2 + j] - expected).abs() < 1e-4,
            "ch1 output_wb[{j}] = {}, expected {expected}",
            output_wb[2 + j]
        );
    }
}

#[test]
fn test_instancenorm_reference_multi_batch() {
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0: [C=2, S=2]
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 2, 2, 2, eps);
    assert!(
        output.iter().all(|v| v.is_finite()),
        "all outputs must be finite"
    );
    assert_eq!(output.len(), 8);
}

#[test]
fn test_instancenorm_reference_channel_independence() {
    // Single-channel result should match the same channel in multi-channel.
    let single_input = vec![1.0, 2.0, 3.0];
    let w = vec![1.5];
    let b_param = vec![0.5];
    let eps = 1e-6;
    let single_output = instancenorm_reference(&single_input, &w, &b_param, 1, 1, 3, eps);

    let multi_input = vec![
        1.0, 2.0, 3.0, // channel 0 (same as single)
        100.0, 200.0, 300.0, // channel 1
    ];
    let w2 = vec![1.5, 2.0];
    let b2 = vec![0.5, 1.0];
    let multi_output = instancenorm_reference(&multi_input, &w2, &b2, 1, 2, 3, eps);

    for i in 0..3 {
        assert!(
            (multi_output[i] - single_output[i]).abs() < 1e-5,
            "channel 0 of multi should match single: multi[{i}]={}, single[{i}]={}",
            multi_output[i],
            single_output[i]
        );
    }
}

#[test]
fn test_instancenorm_reference_numerical_stability() {
    // Very small and very large values.
    let input = vec![1e-10, 2e-10, 3e-10, 4e-10]; // tiny values
    let weight = vec![1.0];
    let bias = vec![0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 1, 4, eps);
    for &v in &output {
        assert!(v.is_finite(), "output must be finite for tiny inputs");
    }

    let input_large = vec![1e6, 2e6, 3e6, 4e6];
    let output_large = instancenorm_reference(&input_large, &weight, &bias, 1, 1, 4, eps);
    for &v in &output_large {
        assert!(v.is_finite(), "output must be finite for large inputs");
    }
}

// ====================================================================
// Cross-norm comparison tests
// ====================================================================

#[test]
fn test_instancenorm_spirv_larger_than_batchnorm() {
    // InstanceNorm has shared memory reductions; BatchNorm is element-wise.
    let bn_words = generate_batchnorm_spirv(64, NORM_WORKGROUP_SIZE);
    let in_words = generate_instancenorm_spirv(64, NORM_WORKGROUP_SIZE);
    assert!(
        in_words.len() > bn_words.len(),
        "InstanceNorm ({} words) should be larger than BatchNorm ({} words) due to reductions",
        in_words.len(),
        bn_words.len()
    );
}

#[test]
fn test_workgroup_size_constant() {
    assert_eq!(NORM_WORKGROUP_SIZE, 256);
}
