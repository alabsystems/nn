// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the 2D convolution SPIR-V kernel.
//!
//! Covers:
//! - Config validation (valid/invalid parameter combinations)
//! - SPIR-V structural validity (header, opcodes, entry point, workgroup size)
//! - Reference computation correctness with known values
//! - Various configs: 1x1, 3x3, strided, padded, grouped, depthwise
//! - Output size calculation

use super::*;
use crate::spirv_emit::SPIRV_MAGIC;

// ---- Helpers ----

fn assert_valid_spirv_header(words: &[u32], label: &str) {
    assert!(words.len() >= 5, "{label}: module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "{label}: wrong magic");
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

fn default_config() -> Conv2dConfig {
    Conv2dConfig::new(4, 8, 3, 3)
}

// ====================================================================
// Config validation tests
// ====================================================================

#[test]
fn test_conv2d_config_valid_basic() {
    let cfg = Conv2dConfig::new(4, 8, 3, 3);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_conv2d_config_valid_with_groups() {
    let cfg = Conv2dConfig::new(4, 8, 3, 3).groups(2);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_conv2d_config_valid_depthwise() {
    let cfg = Conv2dConfig::new(8, 8, 3, 3).groups(8);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_depthwise());
}

#[test]
fn test_conv2d_config_invalid_zero_in_channels() {
    let cfg = Conv2dConfig {
        in_channels: 0,
        ..Conv2dConfig::new(1, 8, 3, 3)
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_conv2d_config_invalid_zero_kernel() {
    let cfg = Conv2dConfig {
        kernel_h: 0,
        ..Conv2dConfig::new(4, 8, 3, 3)
    };
    assert!(cfg.validate().is_err());
    let cfg2 = Conv2dConfig {
        kernel_w: 0,
        ..Conv2dConfig::new(4, 8, 3, 3)
    };
    assert!(cfg2.validate().is_err());
}

#[test]
fn test_conv2d_config_invalid_zero_stride() {
    let cfg = Conv2dConfig::new(4, 8, 3, 3).stride(0, 1);
    assert!(cfg.validate().is_err());
    let cfg2 = Conv2dConfig::new(4, 8, 3, 3).stride(1, 0);
    assert!(cfg2.validate().is_err());
}

#[test]
fn test_conv2d_config_invalid_groups_not_divisible() {
    let cfg = Conv2dConfig::new(5, 8, 3, 3).groups(2);
    assert!(cfg.validate().is_err());
}

// ====================================================================
// Output size tests
// ====================================================================

#[test]
fn test_conv2d_output_size_no_padding() {
    let cfg = Conv2dConfig::new(1, 1, 3, 3);
    // in_h=5, in_w=5, kh=3, kw=3 => (5-3)/1+1 = 3
    assert_eq!(conv2d_output_size(5, 5, &cfg), (3, 3));
}

#[test]
fn test_conv2d_output_size_with_padding() {
    let cfg = Conv2dConfig::new(1, 1, 3, 3).padding(1, 1);
    // (5+2-3)/1+1 = 5
    assert_eq!(conv2d_output_size(5, 5, &cfg), (5, 5));
}

#[test]
fn test_conv2d_output_size_with_stride() {
    let cfg = Conv2dConfig::new(1, 1, 3, 3).stride(2, 2);
    // (6-3)/2+1 = 2
    assert_eq!(conv2d_output_size(6, 6, &cfg), (2, 2));
}

#[test]
fn test_conv2d_output_size_asymmetric() {
    let cfg = Conv2dConfig::new(1, 1, 3, 5).stride(1, 2).padding(1, 2);
    // out_h = (4+2-3)/1+1 = 4, out_w = (6+4-5)/2+1 = 3
    assert_eq!(conv2d_output_size(4, 6, &cfg), (4, 3));
}

#[test]
fn test_conv2d_output_size_1x1() {
    let cfg = Conv2dConfig::new(1, 1, 1, 1);
    // 1x1 conv: same as input
    assert_eq!(conv2d_output_size(7, 9, &cfg), (7, 9));
}

// ====================================================================
// SPIR-V structural validity tests
// ====================================================================

#[test]
fn test_conv2d_spirv_valid_header() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    assert_valid_spirv_header(&words, "conv2d_basic");
}

#[test]
fn test_conv2d_spirv_has_function_structure() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    assert!(has_opcode(&words, 54), "must have OpFunction");
    assert!(has_opcode(&words, 56), "must have OpFunctionEnd");
    assert!(has_opcode(&words, 248), "must have OpLabel");
    assert!(has_opcode(&words, 253), "must have OpReturn");
}

#[test]
fn test_conv2d_spirv_has_triple_nested_loops() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    // Need at least 3 loops: ic, kh, kw
    let loop_count = count_opcode(&words, 246); // OP_LOOP_MERGE
    assert!(
        loop_count >= 3,
        "conv2d must have at least 3 loops (ic + kh + kw), found {loop_count}"
    );
}

#[test]
fn test_conv2d_spirv_has_fmul_and_fadd() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    assert!(has_opcode(&words, 133), "must have OpFMul");
    assert!(has_opcode(&words, 129), "must have OpFAdd");
}

#[test]
fn test_conv2d_spirv_magic_number() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    assert_eq!(words[0], 0x07230203, "SPIR-V magic number check");
}

#[test]
fn test_conv2d_spirv_deterministic() {
    let cfg = default_config();
    let words1 = generate_conv2d_spirv(&cfg);
    let words2 = generate_conv2d_spirv(&cfg);
    assert_eq!(words1, words2, "SPIR-V output must be deterministic");
}

#[test]
fn test_conv2d_spirv_reasonable_size() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    assert!(words.len() > 50, "module too small ({} words)", words.len());
    assert!(
        words.len() < 5000,
        "module too large ({} words)",
        words.len()
    );
}

#[test]
fn test_conv2d_spirv_word_counts_consistent() {
    let cfg = default_config();
    let words = generate_conv2d_spirv(&cfg);
    let mut pos = 5;
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
    }
    assert_eq!(
        pos,
        words.len(),
        "instructions did not consume exactly the full module"
    );
}

#[test]
fn test_conv2d_spirv_various_configs() {
    let configs = [
        Conv2dConfig::new(1, 1, 1, 1),
        Conv2dConfig::new(3, 16, 3, 3),
        Conv2dConfig::new(3, 16, 3, 3).padding(1, 1),
        Conv2dConfig::new(3, 16, 3, 3).stride(2, 2),
        Conv2dConfig::new(64, 128, 3, 3).groups(2),
        Conv2dConfig::new(8, 8, 3, 3).groups(8), // depthwise
    ];
    for (i, cfg) in configs.iter().enumerate() {
        let words = generate_conv2d_spirv(cfg);
        assert_valid_spirv_header(&words, &format!("config_{i}"));
    }
}

// ====================================================================
// Reference computation tests
// ====================================================================

#[test]
fn test_reference_conv2d_1x1_identity() {
    // 1x1 conv, weight=1, bias=0 => output == input
    let cfg = Conv2dConfig::new(1, 1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0];
    let output = conv2d_reference(&input, &weight, None, &cfg, 2, 2);
    assert_eq!(output, input);
}

#[test]
fn test_reference_conv2d_1x1_with_bias() {
    let cfg = Conv2dConfig::new(1, 1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0];
    let bias = vec![10.0];
    let output = conv2d_reference(&input, &weight, Some(&bias), &cfg, 2, 2);
    assert_eq!(output, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_reference_conv2d_3x3_no_padding() {
    // 1x1x4x4 input, 1x1x3x3 weight (all 1s), no padding
    // Output: 1x1x2x2
    let cfg = Conv2dConfig::new(1, 1, 3, 3);
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let weight = vec![1.0; 9];
    let output = conv2d_reference(&input, &weight, None, &cfg, 4, 4);
    let (oh, ow) = conv2d_output_size(4, 4, &cfg);
    assert_eq!((oh, ow), (2, 2));
    assert_eq!(output.len(), 4);
    // out[0,0] = sum of 3x3 at (0,0) = 1+2+3+5+6+7+9+10+11 = 54
    assert!((output[0] - 54.0).abs() < 1e-5);
    // out[0,1] = sum of 3x3 at (0,1) = 2+3+4+6+7+8+10+11+12 = 63
    assert!((output[1] - 63.0).abs() < 1e-5);
    // out[1,0] = sum of 3x3 at (1,0) = 5+6+7+9+10+11+13+14+15 = 90
    assert!((output[2] - 90.0).abs() < 1e-5);
    // out[1,1] = sum of 3x3 at (1,1) = 6+7+8+10+11+12+14+15+16 = 99
    assert!((output[3] - 99.0).abs() < 1e-5);
}

#[test]
fn test_reference_conv2d_3x3_with_padding() {
    // 1x1x3x3 input, 3x3 kernel (center=1, rest=0), padding=1 => same size
    let cfg = Conv2dConfig::new(1, 1, 3, 3).padding(1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
    let mut weight = vec![0.0; 9];
    weight[4] = 1.0; // center
    let output = conv2d_reference(&input, &weight, None, &cfg, 3, 3);
    assert_eq!(output.len(), 9);
    for i in 0..9 {
        assert!(
            (output[i] - input[i]).abs() < 1e-6,
            "output[{i}]={}, expected {}",
            output[i],
            input[i]
        );
    }
}

#[test]
fn test_reference_conv2d_strided() {
    // 1x1x4x4 input, 1x1 kernel=1, stride=2 => 2x2 output
    let cfg = Conv2dConfig::new(1, 1, 1, 1).stride(2, 2);
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let weight = vec![1.0];
    let output = conv2d_reference(&input, &weight, None, &cfg, 4, 4);
    assert_eq!(output.len(), 4);
    // Picks input[0,0], input[0,2], input[2,0], input[2,2]
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 3.0).abs() < 1e-6);
    assert!((output[2] - 9.0).abs() < 1e-6);
    assert!((output[3] - 11.0).abs() < 1e-6);
}

#[test]
fn test_reference_conv2d_grouped() {
    // in_ch=4, out_ch=4, groups=2, 1x1 kernel
    // Group 0: in_ch [0,1] -> out_ch [0,1]
    // Group 1: in_ch [2,3] -> out_ch [2,3]
    let cfg = Conv2dConfig::new(4, 4, 1, 1).groups(2);
    // 1x4x2x2 input
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,  // ch0: 2x2
        5.0, 6.0, 7.0, 8.0,  // ch1: 2x2
        9.0, 10.0, 11.0, 12.0,  // ch2: 2x2
        13.0, 14.0, 15.0, 16.0, // ch3: 2x2
    ];
    // weight: [4, 2, 1, 1] - identity per group
    let weight = vec![
        1.0, 0.0, // oc0: ch0*1 + ch1*0
        0.0, 1.0, // oc1: ch0*0 + ch1*1
        1.0, 0.0, // oc2: ch2*1 + ch3*0
        0.0, 1.0, // oc3: ch2*0 + ch3*1
    ];
    let output = conv2d_reference(&input, &weight, None, &cfg, 2, 2);
    assert_eq!(output.len(), 16);
    // oc0 = ch0 = [1,2,3,4]
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[3] - 4.0).abs() < 1e-6);
    // oc1 = ch1 = [5,6,7,8]
    assert!((output[4] - 5.0).abs() < 1e-6);
    assert!((output[7] - 8.0).abs() < 1e-6);
    // oc2 = ch2 = [9,10,11,12]
    assert!((output[8] - 9.0).abs() < 1e-6);
    assert!((output[11] - 12.0).abs() < 1e-6);
    // oc3 = ch3 = [13,14,15,16]
    assert!((output[12] - 13.0).abs() < 1e-6);
    assert!((output[15] - 16.0).abs() < 1e-6);
}

#[test]
fn test_reference_conv2d_depthwise() {
    // in_ch=2, out_ch=2, groups=2, 3x3 kernel
    let cfg = Conv2dConfig::new(2, 2, 3, 3).groups(2);
    assert!(cfg.is_depthwise());

    // 1x2x3x3 input
    let input: Vec<f32> = (1..=18).map(|x| x as f32).collect();
    // weight: [2, 1, 3, 3] - all ones
    let weight = vec![1.0; 18];
    let output = conv2d_reference(&input, &weight, None, &cfg, 3, 3);
    let (oh, ow) = conv2d_output_size(3, 3, &cfg);
    assert_eq!((oh, ow), (1, 1));
    assert_eq!(output.len(), 2);
    // ch0: sum of 1..9 = 45
    assert!((output[0] - 45.0).abs() < 1e-5);
    // ch1: sum of 10..18 = 126
    assert!((output[1] - 126.0).abs() < 1e-5);
}

#[test]
fn test_reference_conv2d_multi_output_channel() {
    // in_ch=1, out_ch=2, 1x1 kernel
    let cfg = Conv2dConfig::new(1, 2, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![2.0, 3.0]; // oc0: *2, oc1: *3
    let output = conv2d_reference(&input, &weight, None, &cfg, 2, 2);
    assert_eq!(output.len(), 8);
    // oc0: [2, 4, 6, 8]
    assert!((output[0] - 2.0).abs() < 1e-6);
    assert!((output[1] - 4.0).abs() < 1e-6);
    // oc1: [3, 6, 9, 12]
    assert!((output[4] - 3.0).abs() < 1e-6);
    assert!((output[5] - 6.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "Conv2dConfig validation failed")]
fn test_conv2d_spirv_panics_on_invalid_config() {
    let cfg = Conv2dConfig::new(5, 8, 3, 3).groups(2);
    let _ = generate_conv2d_spirv(&cfg);
}
