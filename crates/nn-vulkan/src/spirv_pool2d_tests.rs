// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the 2D pooling SPIR-V kernels (max and average).
//!
//! Covers:
//! - Config validation
//! - SPIR-V structural validity (header, opcodes)
//! - Reference computation correctness
//! - Max vs avg behavior, stride/padding, output size
//! - SPIR-V magic number check

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

fn default_config() -> Pool2dConfig {
    Pool2dConfig::new(2, 2)
}

// ====================================================================
// Config validation tests
// ====================================================================

#[test]
fn test_pool2d_config_valid() {
    let cfg = Pool2dConfig::new(2, 2);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_pool2d_config_invalid_zero_kernel() {
    let cfg = Pool2dConfig {
        kernel_h: 0,
        ..Pool2dConfig::new(2, 2)
    };
    assert!(cfg.validate().is_err());
    let cfg2 = Pool2dConfig {
        kernel_w: 0,
        ..Pool2dConfig::new(2, 2)
    };
    assert!(cfg2.validate().is_err());
}

#[test]
fn test_pool2d_config_invalid_zero_stride() {
    let cfg = Pool2dConfig::new(2, 2).stride(0, 1);
    assert!(cfg.validate().is_err());
}

// ====================================================================
// Output size tests
// ====================================================================

#[test]
fn test_pool2d_output_size_basic() {
    let cfg = Pool2dConfig::new(2, 2);
    // 4x4, 2x2 kernel, stride=2 => 2x2
    assert_eq!(pool2d_output_size(4, 4, &cfg), (2, 2));
}

#[test]
fn test_pool2d_output_size_with_padding() {
    let cfg = Pool2dConfig::new(3, 3).stride(1, 1).padding(1, 1);
    // 4x4, 3x3 kernel, stride=1, pad=1 => (4+2-3)/1+1 = 4
    assert_eq!(pool2d_output_size(4, 4, &cfg), (4, 4));
}

#[test]
fn test_pool2d_output_size_stride_1() {
    let cfg = Pool2dConfig::new(2, 2).stride(1, 1);
    // 4x4, 2x2 kernel, stride=1 => 3x3
    assert_eq!(pool2d_output_size(4, 4, &cfg), (3, 3));
}

// ====================================================================
// SPIR-V structural tests: max pool
// ====================================================================

#[test]
fn test_max_pool2d_spirv_magic() {
    let cfg = default_config();
    let words = generate_max_pool2d_spirv(&cfg);
    assert_eq!(words[0], 0x07230203, "SPIR-V magic number");
}

#[test]
fn test_max_pool2d_spirv_valid_header() {
    let cfg = default_config();
    let words = generate_max_pool2d_spirv(&cfg);
    assert_valid_spirv_header(&words, "max_pool2d");
}

#[test]
fn test_max_pool2d_spirv_has_loops() {
    let cfg = default_config();
    let words = generate_max_pool2d_spirv(&cfg);
    let loop_count = count_opcode(&words, 246); // OP_LOOP_MERGE
    assert!(
        loop_count >= 2,
        "max_pool2d needs at least 2 loops (kh + kw), found {loop_count}"
    );
}

#[test]
fn test_max_pool2d_spirv_has_ext_inst() {
    // Max pool uses GLSL.std.450 FMax.
    let cfg = default_config();
    let words = generate_max_pool2d_spirv(&cfg);
    assert!(
        has_opcode(&words, 12),
        "max_pool2d must have OpExtInst for FMax"
    );
}

#[test]
fn test_max_pool2d_spirv_deterministic() {
    let cfg = default_config();
    let w1 = generate_max_pool2d_spirv(&cfg);
    let w2 = generate_max_pool2d_spirv(&cfg);
    assert_eq!(w1, w2);
}

#[test]
fn test_max_pool2d_spirv_word_counts_consistent() {
    let cfg = default_config();
    let words = generate_max_pool2d_spirv(&cfg);
    let mut pos = 5;
    while pos < words.len() {
        let word = words[pos];
        let word_count = (word >> 16) as usize;
        assert!(word_count > 0, "zero word_count at pos {pos}");
        assert!(pos + word_count <= words.len(), "overflow at pos {pos}");
        pos += word_count;
    }
    assert_eq!(pos, words.len());
}

// ====================================================================
// SPIR-V structural tests: avg pool
// ====================================================================

#[test]
fn test_avg_pool2d_spirv_magic() {
    let cfg = default_config();
    let words = generate_avg_pool2d_spirv(&cfg);
    assert_eq!(words[0], 0x07230203);
}

#[test]
fn test_avg_pool2d_spirv_valid_header() {
    let cfg = default_config();
    let words = generate_avg_pool2d_spirv(&cfg);
    assert_valid_spirv_header(&words, "avg_pool2d");
}

#[test]
fn test_avg_pool2d_spirv_has_fdiv() {
    // Avg pool divides sum by count.
    let cfg = default_config();
    let words = generate_avg_pool2d_spirv(&cfg);
    assert!(has_opcode(&words, 136), "avg_pool2d must have OpFDiv");
}

#[test]
fn test_avg_pool2d_spirv_has_convert_u_to_f() {
    // Avg pool converts count from uint to float.
    let cfg = default_config();
    let words = generate_avg_pool2d_spirv(&cfg);
    assert!(
        has_opcode(&words, 112),
        "avg_pool2d must have OpConvertUToF"
    );
}

// ====================================================================
// Reference computation tests: max pool
// ====================================================================

#[test]
fn test_max_pool2d_reference_basic() {
    // 1x1x4x4 input, 2x2 kernel, stride=2 => 1x1x2x2
    let cfg = Pool2dConfig::new(2, 2);
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let output = max_pool2d_reference(&input, &cfg, 4, 4);
    assert_eq!(output.len(), 4);
    // top-left 2x2: max(1,2,5,6) = 6
    assert!((output[0] - 6.0).abs() < 1e-6);
    // top-right 2x2: max(3,4,7,8) = 8
    assert!((output[1] - 8.0).abs() < 1e-6);
    // bottom-left 2x2: max(9,10,13,14) = 14
    assert!((output[2] - 14.0).abs() < 1e-6);
    // bottom-right 2x2: max(11,12,15,16) = 16
    assert!((output[3] - 16.0).abs() < 1e-6);
}

#[test]
fn test_max_pool2d_reference_stride_1() {
    // 1x1x3x3, 2x2, stride=1 => 2x2
    let cfg = Pool2dConfig::new(2, 2).stride(1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
    let output = max_pool2d_reference(&input, &cfg, 3, 3);
    assert_eq!(output.len(), 4);
    // max(1,2,4,5)=5, max(2,3,5,6)=6, max(4,5,7,8)=8, max(5,6,8,9)=9
    assert!((output[0] - 5.0).abs() < 1e-6);
    assert!((output[1] - 6.0).abs() < 1e-6);
    assert!((output[2] - 8.0).abs() < 1e-6);
    assert!((output[3] - 9.0).abs() < 1e-6);
}

#[test]
fn test_max_pool2d_reference_with_padding() {
    // 1x1x2x2, 2x2, stride=2, padding=1 => output (2+2-2)/2+1 = 2x2
    let cfg = Pool2dConfig::new(2, 2).stride(2, 2).padding(1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let output = max_pool2d_reference(&input, &cfg, 2, 2);
    let (oh, ow) = pool2d_output_size(2, 2, &cfg);
    assert_eq!((oh, ow), (2, 2));
    assert_eq!(output.len(), 4);
    // (oy=0,ox=0): only in-bounds cell is (0,0)=1.0
    assert!((output[0] - 1.0).abs() < 1e-6);
    // (oy=0,ox=1): only (0,1)=2.0
    assert!((output[1] - 2.0).abs() < 1e-6);
    // (oy=1,ox=0): only (1,0)=3.0
    assert!((output[2] - 3.0).abs() < 1e-6);
    // (oy=1,ox=1): only (1,1)=4.0
    assert!((output[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_max_pool2d_reference_negative_values() {
    let cfg = Pool2dConfig::new(2, 2);
    let input = vec![-4.0, -3.0, -2.0, -1.0]; // 2x2
    let output = max_pool2d_reference(&input, &cfg, 2, 2);
    assert_eq!(output.len(), 1);
    assert!((output[0] - (-1.0)).abs() < 1e-6);
}

// ====================================================================
// Reference computation tests: avg pool
// ====================================================================

#[test]
fn test_avg_pool2d_reference_basic() {
    let cfg = Pool2dConfig::new(2, 2);
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let output = avg_pool2d_reference(&input, &cfg, 4, 4);
    assert_eq!(output.len(), 4);
    // top-left: (1+2+5+6)/4 = 3.5
    assert!((output[0] - 3.5).abs() < 1e-6);
    // top-right: (3+4+7+8)/4 = 5.5
    assert!((output[1] - 5.5).abs() < 1e-6);
    // bottom-left: (9+10+13+14)/4 = 11.5
    assert!((output[2] - 11.5).abs() < 1e-6);
    // bottom-right: (11+12+15+16)/4 = 13.5
    assert!((output[3] - 13.5).abs() < 1e-6);
}

#[test]
fn test_avg_pool2d_reference_stride_1() {
    let cfg = Pool2dConfig::new(2, 2).stride(1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
    let output = avg_pool2d_reference(&input, &cfg, 3, 3);
    assert_eq!(output.len(), 4);
    // (1+2+4+5)/4=3.0, (2+3+5+6)/4=4.0, (4+5+7+8)/4=6.0, (5+6+8+9)/4=7.0
    assert!((output[0] - 3.0).abs() < 1e-6);
    assert!((output[1] - 4.0).abs() < 1e-6);
    assert!((output[2] - 6.0).abs() < 1e-6);
    assert!((output[3] - 7.0).abs() < 1e-6);
}

#[test]
fn test_avg_pool2d_vs_max_pool2d_uniform() {
    // For uniform input, avg == max.
    let cfg = Pool2dConfig::new(2, 2);
    let input = vec![5.0; 16]; // 4x4 all 5s
    let max_out = max_pool2d_reference(&input, &cfg, 4, 4);
    let avg_out = avg_pool2d_reference(&input, &cfg, 4, 4);
    assert_eq!(max_out.len(), avg_out.len());
    for i in 0..max_out.len() {
        assert!((max_out[i] - avg_out[i]).abs() < 1e-6);
    }
}

#[test]
fn test_pool2d_various_configs_spirv() {
    let configs = [
        Pool2dConfig::new(2, 2),
        Pool2dConfig::new(3, 3).stride(1, 1),
        Pool2dConfig::new(3, 3).stride(2, 2).padding(1, 1),
        Pool2dConfig::new(1, 1).stride(1, 1),
    ];
    for (i, cfg) in configs.iter().enumerate() {
        let max_words = generate_max_pool2d_spirv(cfg);
        assert_valid_spirv_header(&max_words, &format!("max_config_{i}"));
        let avg_words = generate_avg_pool2d_spirv(cfg);
        assert_valid_spirv_header(&avg_words, &format!("avg_config_{i}"));
    }
}

#[test]
fn test_max_pool2d_multi_channel() {
    // 1x2x2x2 input, 2x2 kernel
    let cfg = Pool2dConfig::new(2, 2);
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // ch0: 2x2
        5.0, 6.0, 7.0, 8.0, // ch1: 2x2
    ];
    let output = max_pool2d_reference(&input, &cfg, 2, 2);
    assert_eq!(output.len(), 2);
    assert!((output[0] - 4.0).abs() < 1e-6); // max of ch0
    assert!((output[1] - 8.0).abs() < 1e-6); // max of ch1
}
