// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::generate_linear_spirv`] and [`super::generate_linear_no_bias_spirv`].

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;

// ---- Helpers ----

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
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

// ---- SPIR-V header validation ----

#[test]
fn test_linear_spirv_valid_header() {
    let spirv = generate_linear_spirv(768, 3072);
    assert_valid_header(&spirv, "linear(768,3072)");
}

#[test]
fn test_linear_no_bias_spirv_valid_header() {
    let spirv = generate_linear_no_bias_spirv(768, 3072);
    assert_valid_header(&spirv, "linear_no_bias(768,3072)");
}

// ---- SPIR-V magic from raw bytes ----

#[test]
fn test_linear_spirv_magic_from_bytes() {
    let spirv = generate_linear_spirv(64, 32);
    let bytes = words_to_bytes(&spirv);
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(magic, 0x07230203, "first 4 bytes must be SPIR-V magic");
}

// ---- Entry point ----

#[test]
fn test_linear_spirv_entry_point_main() {
    let spirv = generate_linear_spirv(768, 3072);
    let name =
        find_entry_point_name(&spirv).unwrap_or_else(|| panic!("linear: no entry point found"));
    assert_eq!(name, "main", "linear: entry point must be 'main'");
}

#[test]
fn test_linear_no_bias_spirv_entry_point_main() {
    let spirv = generate_linear_no_bias_spirv(768, 3072);
    let name = find_entry_point_name(&spirv)
        .unwrap_or_else(|| panic!("linear_no_bias: no entry point found"));
    assert_eq!(name, "main", "linear_no_bias: entry point must be 'main'");
}

// ---- Workgroup size ----

#[test]
fn test_linear_spirv_workgroup_size() {
    let spirv = generate_linear_spirv(768, 3072);
    let wg =
        find_workgroup_size(&spirv).unwrap_or_else(|| panic!("linear: no workgroup size found"));
    assert_eq!(
        wg,
        [LINEAR_WORKGROUP_SIZE, 1, 1],
        "linear: workgroup size must be [{LINEAR_WORKGROUP_SIZE}, 1, 1]",
    );
}

// ---- Reference: with bias ----

#[test]
fn test_linear_reference_with_bias() {
    // 2x3 input, 4x3 weight (4 out_features, 3 in_features), 4 bias
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = vec![
        1.0, 0.0, 0.0, // out_feature 0: picks input[0]
        0.0, 1.0, 0.0, // out_feature 1: picks input[1]
        0.0, 0.0, 1.0, // out_feature 2: picks input[2]
        1.0, 1.0, 1.0, // out_feature 3: sum of all
    ];
    let bias = vec![10.0, 20.0, 30.0, 40.0];
    let out = linear_reference(&input, &weight, Some(&bias), 3, 4);

    // Row 0: [1,2,3] -> [1+10, 2+20, 3+30, 6+40] = [11, 22, 33, 46]
    assert!((out[0] - 11.0).abs() < 1e-6);
    assert!((out[1] - 22.0).abs() < 1e-6);
    assert!((out[2] - 33.0).abs() < 1e-6);
    assert!((out[3] - 46.0).abs() < 1e-6);

    // Row 1: [4,5,6] -> [4+10, 5+20, 6+30, 15+40] = [14, 25, 36, 55]
    assert!((out[4] - 14.0).abs() < 1e-6);
    assert!((out[5] - 25.0).abs() < 1e-6);
    assert!((out[6] - 36.0).abs() < 1e-6);
    assert!((out[7] - 55.0).abs() < 1e-6);
}

// ---- Reference: without bias ----

#[test]
fn test_linear_reference_no_bias() {
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![
        1.0, 0.0, 0.0, // picks input[0]
        0.0, 0.0, 1.0, // picks input[2]
    ];
    let out = linear_reference(&input, &weight, None, 3, 2);

    // [1*1+2*0+3*0, 1*0+2*0+3*1] = [1, 3]
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 3.0).abs() < 1e-6);
}

// ---- Reference: single sample ----

#[test]
fn test_linear_reference_single_sample() {
    let input = vec![2.0, 3.0];
    let weight = vec![1.0, 1.0]; // 1x2 weight: sum of inputs
    let bias = vec![5.0];
    let out = linear_reference(&input, &weight, Some(&bias), 2, 1);

    // dot([2,3], [1,1]) + 5 = 5 + 5 = 10
    assert!((out[0] - 10.0).abs() < 1e-6);
}

// ---- Reference: batch ----

#[test]
fn test_linear_reference_batch() {
    // 3 samples, in_features=2, out_features=2
    let input = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let weight = vec![
        2.0, 0.0, // out 0: 2*x0
        0.0, 3.0, // out 1: 3*x1
    ];
    let out = linear_reference(&input, &weight, None, 2, 2);

    // Sample 0: [2*1, 3*0] = [2, 0]
    assert!((out[0] - 2.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);
    // Sample 1: [2*0, 3*1] = [0, 3]
    assert!((out[2] - 0.0).abs() < 1e-6);
    assert!((out[3] - 3.0).abs() < 1e-6);
    // Sample 2: [2*1, 3*1] = [2, 3]
    assert!((out[4] - 2.0).abs() < 1e-6);
    assert!((out[5] - 3.0).abs() < 1e-6);
}

// ---- Biased vs no-bias produce different SPIR-V ----

#[test]
fn test_linear_biased_vs_no_bias_differ() {
    let biased = generate_linear_spirv(768, 3072);
    let no_bias = generate_linear_no_bias_spirv(768, 3072);
    assert_valid_header(&biased, "biased");
    assert_valid_header(&no_bias, "no_bias");
    assert_ne!(
        biased, no_bias,
        "biased and no-bias linear should produce different SPIR-V"
    );
}

// ---- Has dot-product loop structure ----

#[test]
fn test_linear_spirv_has_loop_structure() {
    let spirv = generate_linear_spirv(768, 3072);
    assert!(
        has_opcode(&spirv, OP_LOOP_MERGE),
        "linear: must have OpLoopMerge for dot product loop"
    );
    assert!(
        has_opcode(&spirv, OP_PHI),
        "linear: must have OpPhi for accumulator"
    );
}

// ---- Has FMul and FAdd for dot product ----

#[test]
fn test_linear_spirv_has_float_arithmetic() {
    let spirv = generate_linear_spirv(768, 3072);
    assert!(
        has_opcode(&spirv, OP_FMUL),
        "linear: must have FMul for dot product"
    );
    assert!(
        has_opcode(&spirv, OP_FADD),
        "linear: must have FAdd for accumulation"
    );
}

// ---- Deterministic output ----

#[test]
fn test_linear_spirv_deterministic() {
    let a = generate_linear_spirv(768, 3072);
    let b = generate_linear_spirv(768, 3072);
    assert_eq!(a, b, "same params must produce identical SPIR-V");
}

// ---- Workgroup size constant value ----

#[test]
fn test_linear_workgroup_size_constant() {
    assert_eq!(LINEAR_WORKGROUP_SIZE, 64);
}

// ---- Various parameter sizes produce valid SPIR-V ----

#[test]
fn test_linear_spirv_various_sizes() {
    let configs: &[(u32, u32)] = &[(32, 16), (64, 64), (256, 512), (768, 3072), (4096, 4096)];
    for &(inf, outf) in configs {
        let spirv = generate_linear_spirv(inf, outf);
        let label = format!("linear({inf},{outf})");
        assert_valid_header(&spirv, &label);
        let name =
            find_entry_point_name(&spirv).unwrap_or_else(|| panic!("{label}: no entry point"));
        assert_eq!(name, "main", "{label}: wrong entry point");
    }
}

// ---- Biased variant has more storage buffer bindings ----

#[test]
fn test_linear_biased_has_more_bindings() {
    let biased = generate_linear_spirv(64, 32);
    let no_bias = generate_linear_no_bias_spirv(64, 32);

    let biased_vars = count_opcode(&biased, OP_VARIABLE);
    let no_bias_vars = count_opcode(&no_bias, OP_VARIABLE);

    // Biased has one extra buffer variable (bias buffer)
    assert!(
        biased_vars > no_bias_vars,
        "biased should have more variables ({biased_vars}) than no_bias ({no_bias_vars})"
    );
}
