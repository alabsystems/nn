// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SPIR-V binary generation of convolution and pooling shaders.

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};

// SPIR-V opcodes/constants used in assertions (duplicated from spirv_conv.rs
// which defines them as private module constants).
const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;
const OP_CAPABILITY: u16 = 17;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_FADD: u16 = 129;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_IMUL: u16 = 132;
const OP_EXT_INST: u16 = 12;
const OP_CONVERT_U_TO_F: u16 = 112;

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

// ---- Output length tests ----

#[test]
fn test_conv1d_output_length_no_padding() {
    // length=10, kernel=3, stride=1, padding=0, dilation=1
    // effective_ks = 1*(3-1)+1 = 3
    // -> (10 - 3) / 1 + 1 = 8
    assert_eq!(conv1d_output_length(10, 3, 1, 0, 1), 8);
}

#[test]
fn test_conv1d_output_length_with_padding() {
    // length=10, kernel=3, stride=1, padding=1, dilation=1
    // effective_ks = 3, -> (10 + 2 - 3) / 1 + 1 = 10
    assert_eq!(conv1d_output_length(10, 3, 1, 1, 1), 10);
}

#[test]
fn test_conv1d_output_length_with_stride() {
    // length=10, kernel=3, stride=2, padding=0, dilation=1
    // -> (10 - 3) / 2 + 1 = 4
    assert_eq!(conv1d_output_length(10, 3, 2, 0, 1), 4);
}

#[test]
fn test_conv1d_output_length_stride_and_padding() {
    // length=16, kernel=4, stride=4, padding=0, dilation=1
    // -> (16 - 4) / 4 + 1 = 4
    assert_eq!(conv1d_output_length(16, 4, 4, 0, 1), 4);
}

#[test]
fn test_conv1d_output_length_with_dilation() {
    // length=10, kernel=3, stride=1, padding=0, dilation=2
    // effective_ks = 2*(3-1)+1 = 5
    // -> (10 - 5) / 1 + 1 = 6
    assert_eq!(conv1d_output_length(10, 3, 1, 0, 2), 6);
}

#[test]
fn test_conv1d_output_length_dilation_stride_padding() {
    // length=16, kernel=3, stride=2, padding=1, dilation=2
    // effective_ks = 2*(3-1)+1 = 5
    // -> (16 + 2 - 5) / 2 + 1 = 7
    assert_eq!(conv1d_output_length(16, 3, 2, 1, 2), 7);
}

#[test]
fn test_conv1d_output_length_dilation_3() {
    // length=20, kernel=3, stride=1, padding=0, dilation=3
    // effective_ks = 3*(3-1)+1 = 7
    // -> (20 - 7) / 1 + 1 = 14
    assert_eq!(conv1d_output_length(20, 3, 1, 0, 3), 14);
}

#[test]
fn test_pool1d_output_length_matches_conv_dilation1() {
    // pool1d has no dilation, so it should match conv1d with dilation=1.
    assert_eq!(
        pool1d_output_length(10, 3, 1, 0),
        conv1d_output_length(10, 3, 1, 0, 1),
    );
    assert_eq!(
        pool1d_output_length(10, 3, 2, 1),
        conv1d_output_length(10, 3, 2, 1, 1),
    );
}

#[test]
fn test_pool1d_output_length_typical_audio() {
    // length=16000, kernel=160, stride=80, padding=0
    // -> (16000 - 160) / 80 + 1 = 199
    assert_eq!(pool1d_output_length(16000, 160, 80, 0), 199);
}

// ---- Conv1d SPIR-V tests ----

#[test]
fn test_conv1d_spirv_header() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "conv1d_3x16_k3_s1_p1_d1");
}

#[test]
fn test_conv1d_spirv_non_empty() {
    let bytes = generate_conv1d_spirv(1, 1, 1, 1, 0, 1);
    assert!(!bytes.is_empty(), "conv1d SPIR-V must not be empty");
    assert!(
        bytes.len() > 20,
        "conv1d SPIR-V must have substantial content"
    );
}

#[test]
fn test_conv1d_spirv_magic() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC, "first word must be SPIR-V magic");
}

#[test]
fn test_conv1d_spirv_entry_point() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_conv1d_spirv_workgroup_size() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [CONV_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_conv1d_spirv_has_capability() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_CAPABILITY),
        "conv1d must have OpCapability"
    );
}

#[test]
fn test_conv1d_spirv_has_loops() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    // Conv1d has nested loops: ic loop and k loop.
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "conv1d must have loop merge for ic/k loops"
    );
    assert!(
        has_opcode(&words, OP_PHI),
        "conv1d must have phi nodes for loop accumulators"
    );
}

#[test]
fn test_conv1d_spirv_has_fmul_fadd() {
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_FMUL),
        "conv1d must have FMul for weight * input"
    );
    assert!(
        has_opcode(&words, OP_FADD),
        "conv1d must have FAdd for accumulation"
    );
}

#[test]
fn test_conv1d_spirv_has_imul_for_dilation() {
    // With dilation, the kernel must compute k * dilation, producing an IMul.
    let bytes = generate_conv1d_spirv(3, 16, 3, 1, 1, 2);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_IMUL),
        "conv1d with dilation must have IMul for k * dilation"
    );
}

#[test]
fn test_conv1d_spirv_different_params() {
    // Verify different parameters produce valid SPIR-V (dilation=1 for all).
    let configs = [
        (1, 1, 1, 1, 0, 1),
        (3, 16, 3, 1, 1, 1),
        (48, 96, 8, 4, 2, 1),
        (256, 512, 5, 2, 2, 1),
    ];
    for (ic, oc, ks, stride, pad, dil) in configs {
        let bytes = generate_conv1d_spirv(ic, oc, ks, stride, pad, dil);
        let words = bytes_to_words(&bytes);
        assert_valid_header(
            &words,
            &format!("conv1d_ic{ic}_oc{oc}_k{ks}_s{stride}_p{pad}_d{dil}"),
        );
    }
}

#[test]
fn test_conv1d_spirv_dilation_variants() {
    // Verify dilated convolution parameters produce valid SPIR-V.
    let configs = [
        (3, 16, 3, 1, 1, 2),  // dilation=2
        (3, 16, 3, 1, 2, 3),  // dilation=3
        (48, 96, 5, 2, 4, 4), // dilation=4
    ];
    for (ic, oc, ks, stride, pad, dil) in configs {
        let bytes = generate_conv1d_spirv(ic, oc, ks, stride, pad, dil);
        let words = bytes_to_words(&bytes);
        assert_valid_header(
            &words,
            &format!("conv1d_ic{ic}_oc{oc}_k{ks}_s{stride}_p{pad}_d{dil}"),
        );
    }
}

// ---- MaxPool1d SPIR-V tests ----

#[test]
fn test_max_pool1d_spirv_header() {
    let bytes = generate_max_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "max_pool1d_k3_s1_p0");
}

#[test]
fn test_max_pool1d_spirv_non_empty() {
    let bytes = generate_max_pool1d_spirv(2, 2, 0);
    assert!(!bytes.is_empty(), "max_pool1d SPIR-V must not be empty");
}

#[test]
fn test_max_pool1d_spirv_magic() {
    let bytes = generate_max_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC);
}

#[test]
fn test_max_pool1d_spirv_entry_point() {
    let bytes = generate_max_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_max_pool1d_spirv_workgroup_size() {
    let bytes = generate_max_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [CONV_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_max_pool1d_spirv_has_ext_inst() {
    let bytes = generate_max_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    // MaxPool uses GLSL.std.450 FMax.
    assert!(
        has_opcode(&words, OP_EXT_INST),
        "max_pool1d must use GLSL ext inst (FMax)"
    );
}

#[test]
fn test_max_pool1d_spirv_has_loop() {
    let bytes = generate_max_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "max_pool1d must have loop"
    );
    assert!(has_opcode(&words, OP_PHI), "max_pool1d must have phi nodes");
}

#[test]
fn test_max_pool1d_spirv_different_params() {
    let configs = [(2, 2, 0), (3, 1, 1), (4, 2, 1), (8, 4, 0)];
    for (ks, stride, pad) in configs {
        let bytes = generate_max_pool1d_spirv(ks, stride, pad);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("max_pool1d_k{ks}_s{stride}_p{pad}"));
    }
}

// ---- AvgPool1d SPIR-V tests ----

#[test]
fn test_avg_pool1d_spirv_header() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    assert_valid_header(&words, "avg_pool1d_k3_s1_p0");
}

#[test]
fn test_avg_pool1d_spirv_non_empty() {
    let bytes = generate_avg_pool1d_spirv(2, 2, 0);
    assert!(!bytes.is_empty(), "avg_pool1d SPIR-V must not be empty");
}

#[test]
fn test_avg_pool1d_spirv_magic() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC);
}

#[test]
fn test_avg_pool1d_spirv_entry_point() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_avg_pool1d_spirv_workgroup_size() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [CONV_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_avg_pool1d_spirv_has_fdiv() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    // AvgPool divides sum by kernel_size.
    assert!(
        has_opcode(&words, OP_FDIV),
        "avg_pool1d must have FDiv for averaging"
    );
}

#[test]
fn test_avg_pool1d_spirv_has_convert_u_to_f() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    // AvgPool converts uint kernel_size to float for division.
    assert!(
        has_opcode(&words, OP_CONVERT_U_TO_F),
        "avg_pool1d must convert kernel_size uint to float"
    );
}

#[test]
fn test_avg_pool1d_spirv_has_loop() {
    let bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let words = bytes_to_words(&bytes);
    assert!(
        has_opcode(&words, OP_LOOP_MERGE),
        "avg_pool1d must have loop"
    );
    assert!(has_opcode(&words, OP_PHI), "avg_pool1d must have phi nodes");
}

#[test]
fn test_avg_pool1d_spirv_different_params() {
    let configs = [(2, 2, 0), (3, 1, 1), (4, 2, 1), (8, 4, 0)];
    for (ks, stride, pad) in configs {
        let bytes = generate_avg_pool1d_spirv(ks, stride, pad);
        let words = bytes_to_words(&bytes);
        assert_valid_header(&words, &format!("avg_pool1d_k{ks}_s{stride}_p{pad}"));
    }
}

// ---- Cross-op consistency tests ----

#[test]
fn test_all_ops_produce_aligned_bytes() {
    // All SPIR-V binaries must have byte length that is a multiple of 4.
    let conv = generate_conv1d_spirv(3, 16, 3, 1, 1, 1);
    assert_eq!(conv.len() % 4, 0, "conv1d bytes must be 4-aligned");

    let conv_dil = generate_conv1d_spirv(3, 16, 3, 1, 1, 2);
    assert_eq!(
        conv_dil.len() % 4,
        0,
        "conv1d dilated bytes must be 4-aligned"
    );

    let max_pool = generate_max_pool1d_spirv(3, 1, 0);
    assert_eq!(max_pool.len() % 4, 0, "max_pool1d bytes must be 4-aligned");

    let avg_pool = generate_avg_pool1d_spirv(3, 1, 0);
    assert_eq!(avg_pool.len() % 4, 0, "avg_pool1d bytes must be 4-aligned");
}

#[test]
fn test_pool_ops_share_workgroup_size() {
    let max_bytes = generate_max_pool1d_spirv(3, 1, 0);
    let avg_bytes = generate_avg_pool1d_spirv(3, 1, 0);
    let max_words = bytes_to_words(&max_bytes);
    let avg_words = bytes_to_words(&avg_bytes);
    let max_wg = find_workgroup_size(&max_words).unwrap();
    let avg_wg = find_workgroup_size(&avg_words).unwrap();
    assert_eq!(
        max_wg, avg_wg,
        "max and avg pool must use same workgroup size"
    );
}

// ---- Edge case tests ----

#[test]
fn test_conv1d_output_length_kernel_equals_input() {
    // length=5, kernel=5, stride=1, padding=0, dilation=1
    // effective_ks = 5, -> (5 - 5) / 1 + 1 = 1
    assert_eq!(conv1d_output_length(5, 5, 1, 0, 1), 1);
}

#[test]
fn test_conv1d_output_length_kernel1() {
    // kernel_size=1 is identity-like: effective_ks = 1*(1-1)+1 = 1
    // -> (10 - 1) / 1 + 1 = 10
    assert_eq!(conv1d_output_length(10, 1, 1, 0, 1), 10);
}

#[test]
fn test_conv1d_output_length_large_dilation() {
    // length=100, kernel=3, stride=1, padding=0, dilation=10
    // effective_ks = 10*(3-1)+1 = 21
    // -> (100 - 21) / 1 + 1 = 80
    assert_eq!(conv1d_output_length(100, 3, 1, 0, 10), 80);
}

#[test]
fn test_pool1d_output_length_stride_equals_kernel() {
    // Non-overlapping pooling: stride == kernel_size
    // length=100, kernel=10, stride=10, padding=0
    // -> (100 - 10) / 10 + 1 = 10
    assert_eq!(pool1d_output_length(100, 10, 10, 0), 10);
}
