// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the depthwise Conv1d SPIR-V kernel.
//!
//! Covers:
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Reference computation correctness with known values
//! - Various configs: kernel_size, stride, padding
//! - Multi-channel depthwise, batched inputs

use super::*;
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;

const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;

fn assert_valid_header(words: &[u32], label: &str) {
    assert!(words.len() >= 5, "{label}: module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "{label}: wrong magic");
    assert_eq!(words[1], TEST_SPIRV_VERSION_1_0, "{label}: wrong version");
    assert_eq!(words[2], TEST_GENERATOR_MAGIC, "{label}: wrong generator");
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
// SPIR-V structural validity tests
// ====================================================================

#[test]
fn test_depthwise_conv_spirv_valid_header() {
    let words = generate_depthwise_conv1d_spirv(4, 3, 1, 0);
    assert_valid_header(&words, "depthwise_basic");
}

#[test]
fn test_depthwise_conv_spirv_entry_point_is_main() {
    let words = generate_depthwise_conv1d_spirv(4, 3, 1, 0);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_depthwise_conv_spirv_workgroup_size() {
    let words = generate_depthwise_conv1d_spirv(4, 3, 1, 0);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [DEPTHWISE_CONV_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_depthwise_conv_spirv_has_loop() {
    let words = generate_depthwise_conv1d_spirv(4, 3, 1, 0);
    let loop_count = count_opcode(&words, 246); // OP_LOOP_MERGE
    assert!(
        loop_count >= 1,
        "depthwise conv must have at least 1 loop (k loop), found {loop_count}"
    );
}

#[test]
fn test_depthwise_conv_spirv_has_fmul_fadd() {
    let words = generate_depthwise_conv1d_spirv(4, 3, 1, 0);
    assert!(has_opcode(&words, 133), "must have OpFMul");
    assert!(has_opcode(&words, 129), "must have OpFAdd");
}

#[test]
fn test_depthwise_conv_spirv_deterministic() {
    let w1 = generate_depthwise_conv1d_spirv(8, 5, 2, 1);
    let w2 = generate_depthwise_conv1d_spirv(8, 5, 2, 1);
    assert_eq!(w1, w2, "SPIR-V output must be deterministic");
}

#[test]
fn test_depthwise_conv_spirv_various_configs() {
    let configs: Vec<(u32, u32, u32, u32)> = vec![
        (1, 1, 1, 0),
        (4, 3, 1, 1),
        (8, 5, 2, 2),
        (16, 7, 1, 3),
        (32, 3, 4, 1),
    ];
    for (i, &(c, k, s, p)) in configs.iter().enumerate() {
        let words = generate_depthwise_conv1d_spirv(c, k, s, p);
        assert_valid_header(&words, &format!("config_{i}"));
    }
}

// ====================================================================
// Reference computation tests
// ====================================================================

#[test]
fn test_depthwise_conv_reference_identity_kernel() {
    // 1 channel, kernel_size=1, weight=1.0 => output == input
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0];
    let output = depthwise_conv1d_reference(&input, &weight, 1, 1, 5, 1, 1, 0);
    assert_eq!(output, input);
}

#[test]
fn test_depthwise_conv_reference_kernel3_no_padding() {
    // 1 channel, ks=3, weight=[1, 0, -1]
    // input = [1, 2, 3, 4, 5]
    // out[0] = 1*1 + 2*0 + 3*(-1) = -2
    // out[1] = 2*1 + 3*0 + 4*(-1) = -2
    // out[2] = 3*1 + 4*0 + 5*(-1) = -2
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 0.0, -1.0];
    let output = depthwise_conv1d_reference(&input, &weight, 1, 1, 5, 3, 1, 0);
    assert_eq!(output.len(), 3);
    for &v in &output {
        assert!((v - (-2.0)).abs() < 1e-6, "expected -2.0, got {v}");
    }
}

#[test]
fn test_depthwise_conv_reference_with_padding() {
    // 1 channel, ks=3, padding=1, weight=[1,1,1]
    // input = [1, 2, 3], padded: [0, 1, 2, 3, 0]
    // out[0] = 0+1+2 = 3
    // out[1] = 1+2+3 = 6
    // out[2] = 2+3+0 = 5
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0];
    let output = depthwise_conv1d_reference(&input, &weight, 1, 1, 3, 3, 1, 1);
    assert_eq!(output.len(), 3);
    assert!((output[0] - 3.0).abs() < 1e-6);
    assert!((output[1] - 6.0).abs() < 1e-6);
    assert!((output[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_depthwise_conv_reference_stride2() {
    // 1 channel, ks=3, stride=2, weight=[1,1,1]
    // input = [1,2,3,4,5,6]
    // out[0] = 1+2+3 = 6
    // out[1] = 3+4+5 = 12
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = vec![1.0, 1.0, 1.0];
    let output = depthwise_conv1d_reference(&input, &weight, 1, 1, 6, 3, 2, 0);
    assert_eq!(output.len(), 2);
    assert!((output[0] - 6.0).abs() < 1e-6);
    assert!((output[1] - 12.0).abs() < 1e-6);
}

#[test]
fn test_depthwise_conv_reference_multi_channel() {
    // 3 channels, ks=3, stride=1, padding=0
    // Each channel convolved independently with its own filter [1, 0, -1]
    let c = 3;
    let l = 5;
    // ch0 = [1,2,3,4,5], ch1 = [2,4,6,8,10], ch2 = [3,6,9,12,15]
    let mut input = Vec::new();
    for ch in 0..c {
        let scale = (ch + 1) as f32;
        for i in 0..l {
            input.push((i + 1) as f32 * scale);
        }
    }
    // weight for each channel: [1, 0, -1]
    let weight = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
    let output = depthwise_conv1d_reference(&input, &weight, 1, c, l, 3, 1, 0);
    let out_len = (l - 3) + 1; // = 3
    assert_eq!(output.len(), c * out_len);

    // For channel ch with scale s, input = [s, 2s, 3s, 4s, 5s]
    // filter [1, 0, -1]: out[j] = input[j] - input[j+2] = s*j - s*(j+2) = -2s
    for ch in 0..c {
        let s = (ch + 1) as f32;
        for ox in 0..out_len {
            let idx = ch * out_len + ox;
            let expected = -2.0 * s;
            assert!(
                (output[idx] - expected).abs() < 1e-5,
                "ch={ch} ox={ox}: got {}, expected {expected}",
                output[idx]
            );
        }
    }
}

#[test]
fn test_depthwise_conv_reference_batch2() {
    // batch=2, 1 channel, ks=1, weight=[2.0]
    // batch0 = [1, 2], batch1 = [3, 4]
    // out batch0 = [2, 4], out batch1 = [6, 8]
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0];
    let output = depthwise_conv1d_reference(&input, &weight, 2, 1, 2, 1, 1, 0);
    assert_eq!(output.len(), 4);
    assert!((output[0] - 2.0).abs() < 1e-6);
    assert!((output[1] - 4.0).abs() < 1e-6);
    assert!((output[2] - 6.0).abs() < 1e-6);
    assert!((output[3] - 8.0).abs() < 1e-6);
}

#[test]
fn test_depthwise_conv_reference_all_zeros_weight() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![0.0, 0.0, 0.0];
    let output = depthwise_conv1d_reference(&input, &weight, 1, 1, 5, 3, 1, 0);
    for &v in &output {
        assert!((v).abs() < 1e-6, "expected 0.0, got {v}");
    }
}

#[test]
#[should_panic(expected = "channels must be > 0")]
fn test_depthwise_conv_spirv_panics_zero_channels() {
    let _ = generate_depthwise_conv1d_spirv(0, 3, 1, 0);
}

#[test]
#[should_panic(expected = "kernel_size must be > 0")]
fn test_depthwise_conv_spirv_panics_zero_kernel_size() {
    let _ = generate_depthwise_conv1d_spirv(4, 0, 1, 0);
}

#[test]
#[should_panic(expected = "stride must be > 0")]
fn test_depthwise_conv_spirv_panics_zero_stride() {
    let _ = generate_depthwise_conv1d_spirv(4, 3, 0, 0);
}
