// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the grouped Conv1d SPIR-V kernel.
//!
//! Covers:
//! - Config validation (valid/invalid parameter combinations)
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Reference computation correctness with known values
//! - Various configs: kernel_size=1,3,5, stride>1, padding>0, dilation>1
//! - Depthwise convolution (groups == in_channels == out_channels)
//! - Grouped convolution (groups > 1, groups < in_channels)

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes/constants for structural assertions.
const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;
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
const TEST_OP_VARIABLE: u16 = 59;
const TEST_OP_DECORATE: u16 = 71;
const TEST_STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
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

fn default_config() -> Conv1dConfig {
    Conv1dConfig::new(4, 8, 3)
}

// ====================================================================
// Config validation tests
// ====================================================================

#[test]
fn test_config_valid_basic() {
    let cfg = Conv1dConfig::new(4, 8, 3);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_valid_with_groups() {
    let cfg = Conv1dConfig::new(4, 8, 3).groups(2);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_valid_depthwise() {
    let cfg = Conv1dConfig::new(8, 8, 3).groups(8);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_depthwise());
}

#[test]
fn test_config_invalid_zero_in_channels() {
    let cfg = Conv1dConfig {
        in_channels: 0,
        ..Conv1dConfig::new(1, 8, 3)
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_zero_out_channels() {
    let cfg = Conv1dConfig {
        out_channels: 0,
        ..Conv1dConfig::new(4, 1, 3)
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_zero_kernel_size() {
    let cfg = Conv1dConfig {
        kernel_size: 0,
        ..Conv1dConfig::new(4, 8, 1)
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_zero_stride() {
    let cfg = Conv1dConfig::new(4, 8, 3).stride(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_zero_dilation() {
    let cfg = Conv1dConfig::new(4, 8, 3).dilation(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_zero_groups() {
    let cfg = Conv1dConfig::new(4, 8, 3).groups(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_in_channels_not_divisible_by_groups() {
    let cfg = Conv1dConfig::new(5, 8, 3).groups(2);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_invalid_out_channels_not_divisible_by_groups() {
    let cfg = Conv1dConfig::new(4, 7, 3).groups(2);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_output_length_no_padding() {
    let cfg = Conv1dConfig::new(1, 1, 3);
    // length=10, ks=3, stride=1, pad=0, dil=1 => (10 - 3) / 1 + 1 = 8
    assert_eq!(cfg.output_length(10), 8);
}

#[test]
fn test_config_output_length_with_padding() {
    let cfg = Conv1dConfig::new(1, 1, 3).padding(1);
    // length=10, ks=3, stride=1, pad=1, dil=1 => (10+2-3)/1+1 = 10
    assert_eq!(cfg.output_length(10), 10);
}

#[test]
fn test_config_output_length_with_stride() {
    let cfg = Conv1dConfig::new(1, 1, 3).stride(2);
    // length=10, ks=3, stride=2, pad=0, dil=1 => (10-3)/2+1 = 4
    assert_eq!(cfg.output_length(10), 4);
}

#[test]
fn test_config_output_length_with_dilation() {
    let cfg = Conv1dConfig::new(1, 1, 3).dilation(2);
    // length=10, ks=3, stride=1, pad=0, dil=2 => effective_ks=5 => (10-5)/1+1 = 6
    assert_eq!(cfg.output_length(10), 6);
}

#[test]
fn test_config_output_length_kernel_size_1() {
    let cfg = Conv1dConfig::new(1, 1, 1);
    // length=10, ks=1 => out=10
    assert_eq!(cfg.output_length(10), 10);
}

#[test]
fn test_config_output_length_kernel_size_5_padding_2() {
    let cfg = Conv1dConfig::new(1, 1, 5).padding(2);
    // length=10, ks=5, pad=2, stride=1, dil=1 => (10+4-5)/1+1 = 10
    assert_eq!(cfg.output_length(10), 10);
}

#[test]
fn test_config_is_depthwise() {
    let depthwise = Conv1dConfig::new(8, 8, 3).groups(8);
    assert!(depthwise.is_depthwise());

    let grouped = Conv1dConfig::new(8, 8, 3).groups(4);
    assert!(!grouped.is_depthwise());

    let standard = Conv1dConfig::new(8, 8, 3);
    assert!(!standard.is_depthwise());
}

// ====================================================================
// SPIR-V structural validity tests
// ====================================================================

#[test]
fn test_conv1d_spirv_valid_header() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "conv1d_basic");
}

#[test]
fn test_conv1d_spirv_entry_point_is_main() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_conv1d_spirv_workgroup_size() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [CONV1D_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_conv1d_spirv_has_capability() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_CAPABILITY),
        "must have OpCapability"
    );
}

#[test]
fn test_conv1d_spirv_has_memory_model() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_MEMORY_MODEL),
        "must have OpMemoryModel"
    );
}

#[test]
fn test_conv1d_spirv_has_function_structure() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
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
fn test_conv1d_spirv_has_loops() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    // Need at least 2 loops: ic loop and k loop
    let loop_count = count_opcode(&words, TEST_OP_LOOP_MERGE);
    assert!(
        loop_count >= 2,
        "conv1d must have at least 2 loops (ic + k), found {loop_count}"
    );
}

#[test]
fn test_conv1d_spirv_has_phi_nodes() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_PHI),
        "conv1d must have OpPhi for loop variables"
    );
}

#[test]
fn test_conv1d_spirv_has_fmul_and_fadd() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, TEST_OP_FMUL),
        "conv1d must have OpFMul for weight * input"
    );
    assert!(
        has_opcode(&words, TEST_OP_FADD),
        "conv1d must have OpFAdd for accumulation"
    );
}

#[test]
fn test_conv1d_spirv_four_storage_buffer_variables() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    let variables = find_instructions(&words, TEST_OP_VARIABLE);
    let sb_count = variables
        .iter()
        .filter(|v| v.len() >= 4 && v[3] == TEST_STORAGE_CLASS_STORAGE_BUFFER)
        .count();
    assert_eq!(
        sb_count, 4,
        "grouped conv1d must have 4 StorageBuffer variables (input, weight, bias, output), got {sb_count}"
    );
}

#[test]
fn test_conv1d_spirv_binding_numbers() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let mut bindings: Vec<u32> = decorations
        .iter()
        .filter(|d| d.len() >= 4 && d[2] == TEST_DECORATION_BINDING)
        .map(|d| d[3])
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    assert!(bindings.contains(&0), "must have binding 0 (input)");
    assert!(bindings.contains(&1), "must have binding 1 (weight)");
    assert!(bindings.contains(&2), "must have binding 2 (bias)");
    assert!(bindings.contains(&3), "must have binding 3 (output)");
}

#[test]
fn test_conv1d_spirv_readonly_decorations() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    let decorations = find_instructions(&words, TEST_OP_DECORATE);
    let nonwritable_count = decorations
        .iter()
        .filter(|d| d.len() >= 3 && d[2] == TEST_DECORATION_NON_WRITABLE)
        .count();
    // input, weight, and bias should all be NonWritable (3 decorations)
    assert_eq!(
        nonwritable_count, 3,
        "input, weight, and bias buffers should be NonWritable, found {nonwritable_count}"
    );
}

#[test]
fn test_conv1d_spirv_byte_alignment() {
    for groups in [1, 2, 4] {
        let cfg = Conv1dConfig::new(4, 8, 3).groups(groups);
        let bytes = generate_conv1d_grouped_spirv(&cfg);
        assert_eq!(
            bytes.len() % 4,
            0,
            "groups={groups}: SPIR-V binary must be 4-byte aligned"
        );
    }
}

#[test]
fn test_conv1d_spirv_reasonable_size() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert!(
        words.len() > 50,
        "conv1d module too small ({} words)",
        words.len()
    );
    assert!(
        words.len() < 5000,
        "conv1d module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_conv1d_spirv_deterministic() {
    let cfg = default_config();
    let bytes1 = generate_conv1d_grouped_spirv(&cfg);
    let bytes2 = generate_conv1d_grouped_spirv(&cfg);
    assert_eq!(
        bytes1, bytes2,
        "SPIR-V output must be deterministic across calls"
    );
}

#[test]
fn test_conv1d_spirv_word_counts_consistent() {
    let cfg = default_config();
    let bytes = generate_conv1d_grouped_spirv(&cfg);
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
        "expected at least 20 instructions, got {instruction_count}"
    );
}

#[test]
fn test_conv1d_spirv_various_configs() {
    let configs = [
        Conv1dConfig::new(1, 1, 1),
        Conv1dConfig::new(4, 8, 3),
        Conv1dConfig::new(8, 16, 5),
        Conv1dConfig::new(4, 8, 3).stride(2),
        Conv1dConfig::new(4, 8, 3).padding(1),
        Conv1dConfig::new(4, 8, 3).dilation(2),
        Conv1dConfig::new(4, 8, 3).groups(2),
        Conv1dConfig::new(8, 8, 3).groups(8), // depthwise
    ];
    for (i, cfg) in configs.iter().enumerate() {
        let bytes = generate_conv1d_grouped_spirv(cfg);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("config_{i}"));
    }
}

// ====================================================================
// Reference computation tests
// ====================================================================

#[test]
fn test_reference_conv1d_identity_kernel() {
    // kernel_size=1, in_ch=1, out_ch=1, bias=0 => output == input
    let cfg = Conv1dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0]; // single weight
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 5);
    assert_eq!(output, input);
}

#[test]
fn test_reference_conv1d_bias_only() {
    // kernel_size=1, weight=0, bias=5 => output == 5
    let cfg = Conv1dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![0.0];
    let bias = vec![5.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 3);
    for &v in &output {
        assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
    }
}

#[test]
fn test_reference_conv1d_kernel_3_no_padding() {
    // in_ch=1, out_ch=1, ks=3, no padding
    // input = [1, 2, 3, 4, 5], weight = [1, 0, -1], bias = [0]
    // out[0] = 1*1 + 2*0 + 3*(-1) = -2
    // out[1] = 2*1 + 3*0 + 4*(-1) = -2
    // out[2] = 3*1 + 4*0 + 5*(-1) = -2
    let cfg = Conv1dConfig::new(1, 1, 3);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 0.0, -1.0];
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 5);
    assert_eq!(output.len(), 3);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - (-2.0)).abs() < 1e-6,
            "output[{i}] = {v}, expected -2.0"
        );
    }
}

#[test]
fn test_reference_conv1d_kernel_3_with_padding_1() {
    // in_ch=1, out_ch=1, ks=3, padding=1
    // input = [1, 2, 3], weight = [1, 1, 1], bias = [0]
    // Padded: [0, 1, 2, 3, 0]
    // out[0] = 0*1 + 1*1 + 2*1 = 3
    // out[1] = 1*1 + 2*1 + 3*1 = 6
    // out[2] = 2*1 + 3*1 + 0*1 = 5
    let cfg = Conv1dConfig::new(1, 1, 3).padding(1);
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0];
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 3);
    assert_eq!(output.len(), 3);
    assert!((output[0] - 3.0).abs() < 1e-6, "out[0]={}", output[0]);
    assert!((output[1] - 6.0).abs() < 1e-6, "out[1]={}", output[1]);
    assert!((output[2] - 5.0).abs() < 1e-6, "out[2]={}", output[2]);
}

#[test]
fn test_reference_conv1d_stride_2() {
    // in_ch=1, out_ch=1, ks=3, stride=2
    // input = [1, 2, 3, 4, 5, 6], weight = [1, 1, 1], bias = [0]
    // out[0] = 1+2+3 = 6
    // out[1] = 3+4+5 = 12
    let cfg = Conv1dConfig::new(1, 1, 3).stride(2);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = vec![1.0, 1.0, 1.0];
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 6);
    assert_eq!(output.len(), 2);
    assert!((output[0] - 6.0).abs() < 1e-6, "out[0]={}", output[0]);
    assert!((output[1] - 12.0).abs() < 1e-6, "out[1]={}", output[1]);
}

#[test]
fn test_reference_conv1d_dilation_2() {
    // in_ch=1, out_ch=1, ks=3, dilation=2
    // input = [1, 2, 3, 4, 5], weight = [1, 1, 1], bias = [0]
    // effective_ks = 5, out_length = 1
    // out[0] = input[0]*1 + input[2]*1 + input[4]*1 = 1+3+5 = 9
    let cfg = Conv1dConfig::new(1, 1, 3).dilation(2);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 5);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 9.0).abs() < 1e-6, "out[0]={}", output[0]);
}

#[test]
fn test_reference_conv1d_kernel_5_padding_2() {
    // ks=5, padding=2 -> same output length
    let cfg = Conv1dConfig::new(1, 1, 5).padding(2);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![0.0, 0.0, 1.0, 0.0, 0.0]; // center weight only
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 5);
    assert_eq!(output.len(), 5);
    // Center weight means output ≈ input
    for i in 0..5 {
        assert!(
            (output[i] - input[i]).abs() < 1e-6,
            "output[{i}]={}, expected {}",
            output[i],
            input[i]
        );
    }
}

#[test]
fn test_reference_conv1d_multi_channel() {
    // in_ch=2, out_ch=1, ks=1, groups=1
    // input = [[1, 2, 3], [4, 5, 6]], weight = [1, -1] (per channel), bias = [0]
    // out[j] = input[0,j]*1 + input[1,j]*(-1) = (1-4, 2-5, 3-6) = (-3, -3, -3)
    let cfg = Conv1dConfig::new(2, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let weight = vec![1.0, -1.0]; // [1, 2, 1]
    let bias = vec![0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 3);
    assert_eq!(output.len(), 3);
    for (i, &v) in output.iter().enumerate() {
        assert!((v - (-3.0)).abs() < 1e-6, "output[{i}] = {v}");
    }
}

#[test]
fn test_reference_conv1d_multi_output_channel() {
    // in_ch=1, out_ch=2, ks=1
    // weight[0] = [2.0], weight[1] = [3.0]
    // bias = [0, 0]
    // input = [1, 2, 3]
    // out_ch0 = [2, 4, 6], out_ch1 = [3, 6, 9]
    let cfg = Conv1dConfig::new(1, 2, 1);
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![2.0, 3.0]; // [2, 1, 1]
    let bias = vec![0.0, 0.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, 3);
    // output layout: [out_ch0, out_ch1] = [2,4,6, 3,6,9]
    assert_eq!(output.len(), 6);
    assert!((output[0] - 2.0).abs() < 1e-6);
    assert!((output[1] - 4.0).abs() < 1e-6);
    assert!((output[2] - 6.0).abs() < 1e-6);
    assert!((output[3] - 3.0).abs() < 1e-6);
    assert!((output[4] - 6.0).abs() < 1e-6);
    assert!((output[5] - 9.0).abs() < 1e-6);
}

#[test]
fn test_reference_conv1d_batch_2() {
    // batch=2, in_ch=1, out_ch=1, ks=1, weight=[2], bias=[1]
    // input_batch0 = [1, 2], input_batch1 = [3, 4]
    // out_batch0 = [2*1+1, 2*2+1] = [3, 5]
    // out_batch1 = [2*3+1, 2*4+1] = [7, 9]
    let cfg = Conv1dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [2, 1, 2]
    let weight = vec![2.0];
    let bias = vec![1.0];
    let output = conv1d_reference(&input, &weight, &bias, &cfg, 2, 2);
    assert_eq!(output.len(), 4);
    assert!((output[0] - 3.0).abs() < 1e-6);
    assert!((output[1] - 5.0).abs() < 1e-6);
    assert!((output[2] - 7.0).abs() < 1e-6);
    assert!((output[3] - 9.0).abs() < 1e-6);
}

// ====================================================================
// Depthwise convolution tests
// ====================================================================

#[test]
fn test_reference_conv1d_depthwise() {
    // in_ch=3, out_ch=3, groups=3, ks=3
    // Each output channel only sees one input channel.
    // weight shape: [3, 1, 3]
    let cfg = Conv1dConfig::new(3, 3, 3).groups(3);
    assert!(cfg.is_depthwise());

    let length = 5;
    // input: 3 channels, each with [1,2,3,4,5]
    let mut input = Vec::new();
    for ch in 0..3 {
        for i in 0..length {
            input.push((i + 1) as f32 * (ch as f32 + 1.0));
        }
    }
    // weight: 3 filters, each [1, 0, -1]
    let weight = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
    let bias = vec![0.0, 0.0, 0.0];

    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, length);
    let out_len = cfg.output_length(length);
    assert_eq!(out_len, 3);
    assert_eq!(output.len(), 3 * out_len);

    // For channel c with input [v, 2v, 3v, 4v, 5v] and filter [1, 0, -1]:
    // out[0] = v - 3v = -2v
    // out[1] = 2v - 4v = -2v
    // out[2] = 3v - 5v = -2v
    for ch in 0..3usize {
        let v = (ch + 1) as f32;
        for ox in 0..out_len {
            let idx = ch * out_len + ox;
            let expected = -2.0 * v;
            assert!(
                (output[idx] - expected).abs() < 1e-5,
                "ch={ch} ox={ox}: output={}, expected={expected}",
                output[idx]
            );
        }
    }
}

#[test]
fn test_reference_conv1d_grouped_2_groups() {
    // in_ch=4, out_ch=4, groups=2, ks=1
    // Group 0: in_ch [0,1] -> out_ch [0,1]
    // Group 1: in_ch [2,3] -> out_ch [2,3]
    // weight shape: [4, 2, 1]
    let cfg = Conv1dConfig::new(4, 4, 1).groups(2);

    let length = 2;
    // input: [ch0=[1,2], ch1=[3,4], ch2=[5,6], ch3=[7,8]]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // weight: oc0=[1,0], oc1=[0,1], oc2=[1,0], oc3=[0,1]
    let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let bias = vec![0.0; 4];

    let output = conv1d_reference(&input, &weight, &bias, &cfg, 1, length);
    assert_eq!(output.len(), 4 * length);

    // oc0 = 1*ch0 + 0*ch1 = [1,2]
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 2.0).abs() < 1e-6);
    // oc1 = 0*ch0 + 1*ch1 = [3,4]
    assert!((output[2] - 3.0).abs() < 1e-6);
    assert!((output[3] - 4.0).abs() < 1e-6);
    // oc2 = 1*ch2 + 0*ch3 = [5,6]
    assert!((output[4] - 5.0).abs() < 1e-6);
    assert!((output[5] - 6.0).abs() < 1e-6);
    // oc3 = 0*ch2 + 1*ch3 = [7,8]
    assert!((output[6] - 7.0).abs() < 1e-6);
    assert!((output[7] - 8.0).abs() < 1e-6);
}

// ====================================================================
// SPIR-V generation for various configs
// ====================================================================

#[test]
fn test_conv1d_spirv_depthwise_config() {
    let cfg = Conv1dConfig::new(8, 8, 3).groups(8);
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "depthwise");
    assert!(has_opcode(&words, TEST_OP_LOOP_MERGE));
}

#[test]
fn test_conv1d_spirv_grouped_config() {
    let cfg = Conv1dConfig::new(16, 32, 3).groups(4);
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "grouped");
}

#[test]
fn test_conv1d_spirv_kernel_1() {
    let cfg = Conv1dConfig::new(4, 8, 1);
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "ks1");
}

#[test]
fn test_conv1d_spirv_kernel_5_stride_2_padding_2() {
    let cfg = Conv1dConfig::new(4, 8, 5).stride(2).padding(2);
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "ks5_s2_p2");
}

#[test]
fn test_conv1d_spirv_dilation_3() {
    let cfg = Conv1dConfig::new(4, 8, 3).dilation(3);
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "dil3");
}

#[test]
fn test_conv1d_spirv_large_channels() {
    let cfg = Conv1dConfig::new(256, 512, 3).groups(1);
    let bytes = generate_conv1d_grouped_spirv(&cfg);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "large_channels");
}

#[test]
#[should_panic(expected = "Conv1dConfig validation failed")]
fn test_conv1d_spirv_panics_on_invalid_config() {
    let cfg = Conv1dConfig::new(5, 8, 3).groups(2); // 5 % 2 != 0
    let _bytes = generate_conv1d_grouped_spirv(&cfg);
}
