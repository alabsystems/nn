// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the dedicated RMSNorm SPIR-V kernel with separate I/O buffers.
//!
//! Covers:
//! - Config validation
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Separate input/weight/output buffer layout (3 StorageBuffer bindings)
//! - Reference RMSNorm correctness with known values
//! - Various hidden dimensions (64, 128, 512, 1024, 4096)
//! - Epsilon edge cases
//! - Weight scaling verification
//! - Numerical stability with large/small values

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

fn default_config() -> RmsNormConfig {
    RmsNormConfig::new(768, 1e-5)
}

// ====================================================================
// Config validation tests
// ====================================================================

#[test]
fn test_rmsnorm_config_valid() {
    let config = RmsNormConfig::new(768, 1e-5);
    config.validate().expect("default config should be valid");
}

#[test]
fn test_rmsnorm_config_zero_hidden_dim_invalid() {
    let config = RmsNormConfig::new(0, 1e-5);
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_negative_eps_invalid() {
    let config = RmsNormConfig::new(768, -1e-5);
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_zero_eps_invalid() {
    let config = RmsNormConfig::new(768, 0.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_nan_eps_invalid() {
    let config = RmsNormConfig::new(768, f32::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_inf_eps_invalid() {
    let config = RmsNormConfig::new(768, f32::INFINITY);
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_empty_kernel_name_invalid() {
    let config = RmsNormConfig {
        hidden_dim: 768,
        eps: 1e-5,
        kernel_name: String::new(),
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_custom_kernel_name() {
    let config = RmsNormConfig {
        hidden_dim: 768,
        eps: 1e-5,
        kernel_name: "rmsnorm_layer0".to_string(),
    };
    config
        .validate()
        .expect("custom kernel name should be valid");
}

// ====================================================================
// SPIR-V structural validity tests
// ====================================================================

#[test]
fn test_rmsnorm_spirv_valid_header_768() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "rmsnorm_768");
}

#[test]
fn test_rmsnorm_spirv_valid_header_64() {
    let config = RmsNormConfig::new(64, 1e-5);
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "rmsnorm_64");
}

#[test]
fn test_rmsnorm_spirv_valid_header_4096() {
    let config = RmsNormConfig::new(4096, 1e-5);
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "rmsnorm_4096");
}

#[test]
fn test_rmsnorm_spirv_entry_point_is_main() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_rmsnorm_spirv_workgroup_size() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [RMSNORM_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_rmsnorm_spirv_has_capability() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "must have OpCapability"
    );
}

#[test]
fn test_rmsnorm_spirv_has_memory_model() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "must have OpMemoryModel"
    );
}

#[test]
fn test_rmsnorm_spirv_has_function_structure() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
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
fn test_rmsnorm_spirv_has_ext_inst_for_sqrt() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_EXT_INST),
        "rmsnorm must use GLSL.std.450 for Sqrt"
    );
}

#[test]
fn test_rmsnorm_spirv_has_fmul_for_x_squared_and_weight() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let fmul_count = count_opcode(&words, TEST_OP_FMUL);
    // At minimum: x*x in phase 1, x*inv_rms in phase 2, weight*normalized in phase 2 = 3
    assert!(
        fmul_count >= 3,
        "rmsnorm must have at least 3 OpFMul (x^2, x*inv_rms, weight*norm), found {fmul_count}"
    );
}

#[test]
fn test_rmsnorm_spirv_has_fdiv_for_mean_and_inv_rms() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let fdiv_count = count_opcode(&words, TEST_OP_FDIV);
    // sum/N for mean, 1.0/sqrt for inv_rms = 2
    assert!(
        fdiv_count >= 2,
        "rmsnorm must have at least 2 OpFDiv (mean + inv_rms), found {fdiv_count}"
    );
}

#[test]
fn test_rmsnorm_spirv_has_fadd_for_accumulation_and_eps() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "rmsnorm must have OpFAdd for sum accumulation and eps addition"
    );
}

#[test]
fn test_rmsnorm_spirv_has_loops_and_phi() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_LOOP_MERGE),
        "rmsnorm must have loops for strided element access"
    );
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "rmsnorm must have OpPhi for loop variable evolution"
    );
}

#[test]
fn test_rmsnorm_spirv_has_barriers() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let barrier_count = count_opcode(&words, TEST_OP_CONTROL_BARRIER);
    // At minimum: after sq_sum store, inside tree reduction, before phase 2
    assert!(
        barrier_count >= 3,
        "rmsnorm must have at least 3 barriers for shared memory synchronization, found {barrier_count}"
    );
}

// ====================================================================
// Separate I/O buffer layout tests
// ====================================================================

#[test]
fn test_rmsnorm_spirv_has_three_storage_buffer_variables() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 3,
        "rmsnorm must have 3 StorageBuffer variables (input + weight + output), got {sb_count}"
    );
}

#[test]
fn test_rmsnorm_spirv_has_workgroup_variable() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let wg_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_WORKGROUP)
        .count();
    assert!(
        wg_count >= 1,
        "rmsnorm must have at least 1 workgroup variable for shared memory, found {wg_count}"
    );
}

#[test]
fn test_rmsnorm_spirv_binding_numbers() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    assert!(bindings.contains(&0), "must have binding 0 (input buffer)");
    assert!(bindings.contains(&1), "must have binding 1 (weight buffer)");
    assert!(bindings.contains(&2), "must have binding 2 (output buffer)");
}

#[test]
fn test_rmsnorm_spirv_input_and_weight_are_nonwritable() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nonwritable_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    assert_eq!(
        nonwritable_count, 2,
        "input and weight buffers should both be NonWritable, found {nonwritable_count}"
    );
}

// ====================================================================
// Byte alignment and size tests
// ====================================================================

#[test]
fn test_rmsnorm_spirv_byte_alignment() {
    for dim in [64, 128, 256, 512, 768, 1024, 4096] {
        let config = RmsNormConfig::new(dim, 1e-5);
        let bytes = generate_rmsnorm_separate_io_spirv(&config);
        assert_eq!(
            bytes.len() % 4,
            0,
            "rmsnorm dim={dim}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_rmsnorm_spirv_reasonable_size() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    assert!(
        words.len() > 100,
        "rmsnorm module too small ({} words)",
        words.len()
    );
    assert!(
        words.len() < 5000,
        "rmsnorm module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_rmsnorm_spirv_deterministic() {
    let config = default_config();
    let bytes1 = generate_rmsnorm_separate_io_spirv(&config);
    let bytes2 = generate_rmsnorm_separate_io_spirv(&config);
    assert_eq!(
        bytes1, bytes2,
        "SPIR-V output must be deterministic across calls"
    );
}

#[test]
fn test_rmsnorm_spirv_various_hidden_dims() {
    for dim in [64, 128, 512, 1024, 4096] {
        let config = RmsNormConfig::new(dim, 1e-5);
        let bytes = generate_rmsnorm_separate_io_spirv(&config);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("rmsnorm_dim{dim}"));
    }
}

#[test]
fn test_rmsnorm_spirv_word_counts_consistent() {
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
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
        "expected at least 20 instructions for rmsnorm, got {instruction_count}"
    );
}

// ====================================================================
// Loop structure tests
// ====================================================================

#[test]
fn test_rmsnorm_spirv_loop_count_matches_two_phase_design() {
    // RMSNorm has 2 phases + 1 tree reduction = at least 3 loop constructs.
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    assert!(
        loop_count >= 3,
        "rmsnorm should have at least 3 loops (2 phases + 1 tree reduction), found {loop_count}"
    );
}

#[test]
fn test_rmsnorm_spirv_ext_inst_count() {
    // Must have GLSL.std.450 call for Sqrt.
    let config = default_config();
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let ext_count = count_opcode(&words, TEST_OP_EXT_INST);
    assert!(
        ext_count >= 1,
        "rmsnorm must have at least 1 ExtInst call (Sqrt), found {ext_count}"
    );
}

// ====================================================================
// Reference RMSNorm correctness tests (CPU implementation)
// ====================================================================

#[test]
fn test_rmsnorm_reference_unit_weight() {
    // With weight = [1, 1, 1, 1], RMSNorm is just x * rsqrt(mean(x^2) + eps).
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);

    // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // inv_rms = 1 / sqrt(7.5 + 1e-5) = 1 / 2.7386...
    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for (i, &v) in output.iter().enumerate() {
        let expected = input[i] * inv_rms;
        assert!(
            (v - expected).abs() < 1e-5,
            "output[{i}] = {v}, expected {expected}"
        );
    }
}

#[test]
fn test_rmsnorm_reference_known_values() {
    // Manually computed RMSNorm for [3.0, 4.0] with weight [2.0, 0.5], eps=0.
    let input = vec![3.0, 4.0];
    let weight = vec![2.0, 0.5];
    let eps = 0.0_f32;
    // In practice eps=0 is invalid for config, but reference function works fine.
    let output = rmsnorm_reference(&input, &weight, 1, 2, eps);

    // mean(x^2) = (9 + 16) / 2 = 12.5
    // inv_rms = 1 / sqrt(12.5) = 1 / 3.5355...
    let mean_sq: f32 = f32::midpoint(9.0, 16.0);
    let inv_rms = 1.0 / mean_sq.sqrt();
    let expected_0 = 3.0 * inv_rms * 2.0;
    let expected_1 = 4.0 * inv_rms * 0.5;

    assert!(
        (output[0] - expected_0).abs() < 1e-5,
        "output[0] = {}, expected {expected_0}",
        output[0]
    );
    assert!(
        (output[1] - expected_1).abs() < 1e-5,
        "output[1] = {}, expected {expected_1}",
        output[1]
    );
}

#[test]
fn test_rmsnorm_reference_all_ones() {
    // For x = [1, 1, 1], weight = [1, 1, 1], eps = 0:
    // mean(x^2) = 1, rms = 1, inv_rms = 1, output = [1, 1, 1].
    let input = vec![1.0; 3];
    let weight = vec![1.0; 3];
    let output = rmsnorm_reference(&input, &weight, 1, 3, 0.0);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "output[{i}] = {v}, expected 1.0 for all-ones input"
        );
    }
}

#[test]
fn test_rmsnorm_reference_all_zeros() {
    // For x = [0, 0, 0], output should be [0, 0, 0] regardless of weight.
    // inv_rms = 1 / sqrt(0 + eps) which is finite, but 0 * anything = 0.
    let input = vec![0.0; 4];
    let weight = vec![2.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "output[{i}] = {v}, expected 0.0 for all-zeros input"
        );
    }
}

#[test]
fn test_rmsnorm_reference_multi_row() {
    let input = vec![
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
    ];
    let weight = vec![1.0; 3];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 2, 3, eps);

    // Each row should be normalized independently.
    // Row 0: mean(x^2) = (1+4+9)/3 = 14/3
    let mean_sq_0 = 14.0 / 3.0;
    let inv_rms_0 = 1.0 / (mean_sq_0 + eps).sqrt();
    for j in 0..3 {
        let expected = input[j] * inv_rms_0;
        assert!(
            (output[j] - expected).abs() < 1e-5,
            "row 0, col {j}: output = {}, expected {expected}",
            output[j]
        );
    }

    // Row 1: mean(x^2) = (16+25+36)/3 = 77/3
    let mean_sq_1 = 77.0 / 3.0;
    let inv_rms_1 = 1.0 / (mean_sq_1 + eps).sqrt();
    for j in 0..3 {
        let expected = input[3 + j] * inv_rms_1;
        assert!(
            (output[3 + j] - expected).abs() < 1e-5,
            "row 1, col {j}: output = {}, expected {expected}",
            output[3 + j]
        );
    }
}

#[test]
fn test_rmsnorm_reference_row_independence() {
    // Single-row result should match corresponding row of multi-row result.
    let single_input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.5, 0.5, 2.0];
    let eps = 1e-6;
    let single_output = rmsnorm_reference(&single_input, &weight, 1, 3, eps);

    let multi_input = vec![
        1.0, 2.0, 3.0, // row 0 (same as single)
        100.0, 200.0, 300.0, // row 1 (different)
    ];
    let multi_output = rmsnorm_reference(&multi_input, &weight, 2, 3, eps);

    for i in 0..3 {
        assert!(
            (multi_output[i] - single_output[i]).abs() < 1e-6,
            "row 0 of multi-row must match single-row: multi[{i}]={}, single[{i}]={}",
            multi_output[i],
            single_output[i]
        );
    }
}

// ====================================================================
// Epsilon edge cases
// ====================================================================

#[test]
fn test_rmsnorm_reference_tiny_eps() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps = 1e-12;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    for &v in &output {
        assert!(v.is_finite(), "output must be finite with tiny eps");
    }
}

#[test]
fn test_rmsnorm_reference_large_eps() {
    // Large eps should heavily dampen the normalization.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps_small = 1e-5;
    let eps_large = 100.0;
    let output_small = rmsnorm_reference(&input, &weight, 1, 4, eps_small);
    let output_large = rmsnorm_reference(&input, &weight, 1, 4, eps_large);

    // With large eps, the rms value is dominated by eps, so inv_rms is small,
    // and output magnitudes should be smaller.
    let max_small = output_small.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let max_large = output_large.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(
        max_large < max_small,
        "large eps should produce smaller output magnitudes: max_large={max_large}, max_small={max_small}"
    );
}

#[test]
fn test_rmsnorm_reference_eps_prevents_division_by_zero() {
    // Very small inputs where mean(x^2) is nearly zero.
    let input = vec![1e-20, 1e-20, 1e-20, 1e-20];
    let weight = vec![1.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "output[{i}] must be finite even for near-zero input"
        );
    }
}

// ====================================================================
// Weight scaling verification
// ====================================================================

#[test]
fn test_rmsnorm_reference_weight_scaling() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight_1 = vec![1.0; 4];
    let weight_2 = vec![2.0; 4];
    let eps = 1e-5;
    let output_1 = rmsnorm_reference(&input, &weight_1, 1, 4, eps);
    let output_2 = rmsnorm_reference(&input, &weight_2, 1, 4, eps);

    // output_2 should be exactly 2x output_1 since weight is 2x.
    for i in 0..4 {
        assert!(
            (output_2[i] - 2.0 * output_1[i]).abs() < 1e-5,
            "2x weight should produce 2x output: output_2[{i}]={}, 2*output_1[{i}]={}",
            output_2[i],
            2.0 * output_1[i]
        );
    }
}

#[test]
fn test_rmsnorm_reference_per_element_weight() {
    // Different weights per element.
    let input = vec![2.0, 2.0, 2.0];
    let weight = vec![1.0, 2.0, 3.0];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 3, eps);

    // All inputs are the same, so normalized values should be equal.
    // mean(x^2) = 4, inv_rms = 1/sqrt(4 + eps) = 1/2
    let inv_rms = 1.0 / (4.0_f32 + eps).sqrt();
    let normalized = 2.0 * inv_rms;

    assert!(
        (output[0] - normalized * 1.0).abs() < 1e-5,
        "output[0] with weight 1.0"
    );
    assert!(
        (output[1] - normalized * 2.0).abs() < 1e-5,
        "output[1] with weight 2.0"
    );
    assert!(
        (output[2] - normalized * 3.0).abs() < 1e-5,
        "output[2] with weight 3.0"
    );
}

#[test]
fn test_rmsnorm_reference_zero_weight() {
    // Zero weight should produce zero output regardless of input.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "output[{i}] = {v}, expected 0.0 with zero weight"
        );
    }
}

#[test]
fn test_rmsnorm_reference_negative_weight() {
    // Negative weight should flip sign of output.
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight_pos = vec![1.0; 4];
    let weight_neg = vec![-1.0; 4];
    let eps = 1e-5;
    let output_pos = rmsnorm_reference(&input, &weight_pos, 1, 4, eps);
    let output_neg = rmsnorm_reference(&input, &weight_neg, 1, 4, eps);

    for i in 0..4 {
        assert!(
            (output_neg[i] + output_pos[i]).abs() < 1e-5,
            "negative weight should negate output: neg[{i}]={}, pos[{i}]={}",
            output_neg[i],
            output_pos[i]
        );
    }
}

// ====================================================================
// Numerical stability with large/small values
// ====================================================================

#[test]
fn test_rmsnorm_reference_large_values() {
    let input = vec![1000.0, 2000.0, 3000.0, 4000.0];
    let weight = vec![1.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "output[{i}] = {v} must be finite for large inputs"
        );
        assert!(
            v.abs() > 0.0,
            "output[{i}] = {v} must be non-zero for non-zero inputs"
        );
    }
}

#[test]
fn test_rmsnorm_reference_small_values() {
    let input = vec![1e-6, 2e-6, 3e-6, 4e-6];
    let weight = vec![1.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "output[{i}] = {v} must be finite for small inputs"
        );
    }
}

#[test]
fn test_rmsnorm_reference_mixed_sign_values() {
    let input = vec![-3.0, -1.0, 1.0, 3.0];
    let weight = vec![1.0; 4];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);

    // RMSNorm preserves sign: negative inputs produce negative outputs.
    assert!(
        output[0] < 0.0,
        "negative input should produce negative output"
    );
    assert!(
        output[1] < 0.0,
        "negative input should produce negative output"
    );
    assert!(
        output[2] > 0.0,
        "positive input should produce positive output"
    );
    assert!(
        output[3] > 0.0,
        "positive input should produce positive output"
    );

    // Symmetric magnitudes: |output[0]| == |output[3]|, |output[1]| == |output[2]|
    assert!(
        (output[0].abs() - output[3].abs()).abs() < 1e-5,
        "symmetric inputs should have symmetric output magnitudes"
    );
    assert!(
        (output[1].abs() - output[2].abs()).abs() < 1e-5,
        "symmetric inputs should have symmetric output magnitudes"
    );
}

#[test]
fn test_rmsnorm_reference_wide_hidden_dim() {
    // hidden_dim > RMSNORM_WORKGROUP_SIZE (256), ensuring strided access works.
    let hidden_dim = 1024;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01 + 0.1).collect();
    let weight = vec![1.0; hidden_dim];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, hidden_dim, eps);

    for (j, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "wide hidden_dim: output[{j}] = {v} must be finite"
        );
    }

    // All outputs should be positive (all inputs are positive, all weights are 1.0).
    for (j, &v) in output.iter().enumerate() {
        assert!(
            v > 0.0,
            "wide hidden_dim: output[{j}] = {v} must be > 0 for positive inputs"
        );
    }
}

#[test]
fn test_rmsnorm_reference_scale_invariance_of_direction() {
    // Scaling all inputs by a constant should not change the output direction
    // (only magnitude changes due to weight). With unit weight:
    // RMSNorm(c*x) = c*x / rms(c*x) * w = c*x / (|c| * rms(x)) * w = sign(c) * x / rms(x) * w
    // So for c > 0: RMSNorm(c*x) == RMSNorm(x).
    let input_1 = vec![1.0, 2.0, 3.0, 4.0];
    let input_10: Vec<f32> = input_1.iter().map(|&x| x * 10.0).collect();
    let weight = vec![1.0; 4];
    let eps = 1e-8; // small eps to minimize eps contribution
    let output_1 = rmsnorm_reference(&input_1, &weight, 1, 4, eps);
    let output_10 = rmsnorm_reference(&input_10, &weight, 1, 4, eps);

    for i in 0..4 {
        assert!(
            (output_1[i] - output_10[i]).abs() < 1e-4,
            "scale invariance: output_1[{i}]={}, output_10[{i}]={}",
            output_1[i],
            output_10[i]
        );
    }
}
