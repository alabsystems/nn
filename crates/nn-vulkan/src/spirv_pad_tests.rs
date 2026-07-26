// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the 1D and 2D padding SPIR-V kernels.
//!
//! Covers:
//! - SPIR-V structural validity (header, entry point, workgroup size)
//! - 1D reference correctness (zero padding, symmetric, asymmetric, custom value)
//! - 2D reference correctness (zero padding, asymmetric, custom value)
//! - Various configurations for both 1D and 2D

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

// ====================================================================
// 1D Padding SPIR-V structural tests
// ====================================================================

#[test]
fn test_pad1d_spirv_valid_header() {
    let words = generate_pad_spirv(10, 2, 3, 0.0);
    assert_valid_header(&words, "pad1d_basic");
}

#[test]
fn test_pad1d_spirv_entry_point_is_main() {
    let words = generate_pad_spirv(10, 2, 3, 0.0);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_pad1d_spirv_workgroup_size() {
    let words = generate_pad_spirv(10, 2, 3, 0.0);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [PAD_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_pad1d_spirv_deterministic() {
    let w1 = generate_pad_spirv(8, 1, 1, -1.0);
    let w2 = generate_pad_spirv(8, 1, 1, -1.0);
    assert_eq!(w1, w2, "SPIR-V output must be deterministic");
}

#[test]
fn test_pad1d_spirv_various_configs() {
    let configs: Vec<(u32, u32, u32, f32)> = vec![
        (1, 0, 0, 0.0),
        (10, 2, 3, 0.0),
        (5, 5, 5, -1.0),
        (100, 0, 10, 42.0),
    ];
    for (i, &(n, pl, pr, pv)) in configs.iter().enumerate() {
        let words = generate_pad_spirv(n, pl, pr, pv);
        assert_valid_header(&words, &format!("pad1d_config_{i}"));
    }
}

// ====================================================================
// 1D Padding reference tests
// ====================================================================

#[test]
fn test_pad1d_reference_no_padding() {
    let input = vec![1.0, 2.0, 3.0];
    let output = pad_reference(&input, 0, 0, 0.0);
    assert_eq!(output, input);
}

#[test]
fn test_pad1d_reference_left_only() {
    let input = vec![1.0, 2.0, 3.0];
    let output = pad_reference(&input, 2, 0, 0.0);
    assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_pad1d_reference_right_only() {
    let input = vec![1.0, 2.0, 3.0];
    let output = pad_reference(&input, 0, 2, 0.0);
    assert_eq!(output, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn test_pad1d_reference_symmetric() {
    let input = vec![1.0, 2.0, 3.0];
    let output = pad_reference(&input, 1, 1, 0.0);
    assert_eq!(output, vec![0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_pad1d_reference_custom_value() {
    let input = vec![1.0, 2.0];
    let output = pad_reference(&input, 1, 2, -1.0);
    assert_eq!(output, vec![-1.0, 1.0, 2.0, -1.0, -1.0]);
}

#[test]
fn test_pad1d_reference_empty_input() {
    let input: Vec<f32> = vec![];
    let output = pad_reference(&input, 2, 3, 5.0);
    assert_eq!(output, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
}

// ====================================================================
// 2D Padding SPIR-V structural tests
// ====================================================================

#[test]
fn test_pad2d_spirv_valid_header() {
    let words = generate_pad2d_spirv(4, 4, 1, 1, 1, 1, 0.0);
    assert_valid_header(&words, "pad2d_basic");
}

#[test]
fn test_pad2d_spirv_entry_point_is_main() {
    let words = generate_pad2d_spirv(4, 4, 1, 1, 1, 1, 0.0);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_pad2d_spirv_workgroup_size() {
    let words = generate_pad2d_spirv(4, 4, 1, 1, 1, 1, 0.0);
    let wg = find_workgroup_size(&words).expect("must have workgroup size");
    assert_eq!(wg, [PAD_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_pad2d_spirv_deterministic() {
    let w1 = generate_pad2d_spirv(3, 5, 2, 1, 0, 3, -1.0);
    let w2 = generate_pad2d_spirv(3, 5, 2, 1, 0, 3, -1.0);
    assert_eq!(w1, w2, "SPIR-V output must be deterministic");
}

#[test]
fn test_pad2d_spirv_various_configs() {
    let configs: Vec<(u32, u32, u32, u32, u32, u32, f32)> = vec![
        (1, 1, 0, 0, 0, 0, 0.0),
        (4, 4, 1, 1, 1, 1, 0.0),
        (3, 5, 2, 1, 0, 3, -1.0),
        (8, 8, 0, 0, 2, 2, 42.0),
    ];
    for (i, &(h, w, pt, pb, pl, pr, pv)) in configs.iter().enumerate() {
        let words = generate_pad2d_spirv(h, w, pt, pb, pl, pr, pv);
        assert_valid_header(&words, &format!("pad2d_config_{i}"));
    }
}

// ====================================================================
// 2D Padding reference tests
// ====================================================================

#[test]
fn test_pad2d_reference_no_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = pad2d_reference(&input, 2, 2, 0, 0, 0, 0, 0.0);
    assert_eq!(output, input);
}

#[test]
fn test_pad2d_reference_symmetric_1() {
    // 2x2 input with pad=1 all sides -> 4x4 output
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = pad2d_reference(&input, 2, 2, 1, 1, 1, 1, 0.0);
    assert_eq!(output.len(), 4 * 4);
    // Row 0: [0, 0, 0, 0]
    // Row 1: [0, 1, 2, 0]
    // Row 2: [0, 3, 4, 0]
    // Row 3: [0, 0, 0, 0]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(output, expected);
}

#[test]
fn test_pad2d_reference_asymmetric() {
    // 2x2 input, pad_top=1, pad_bottom=0, pad_left=0, pad_right=2
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = pad2d_reference(&input, 2, 2, 1, 0, 0, 2, 0.0);
    // out_h = 3, out_w = 4
    assert_eq!(output.len(), 3 * 4);
    // Row 0: [0, 0, 0, 0]
    // Row 1: [1, 2, 0, 0]
    // Row 2: [3, 4, 0, 0]
    let expected = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
    assert_eq!(output, expected);
}

#[test]
fn test_pad2d_reference_custom_value() {
    let input = vec![5.0];
    let output = pad2d_reference(&input, 1, 1, 1, 1, 1, 1, -1.0);
    // 3x3 output
    let expected = vec![-1.0, -1.0, -1.0, -1.0, 5.0, -1.0, -1.0, -1.0, -1.0];
    assert_eq!(output, expected);
}

#[test]
fn test_pad2d_reference_top_bottom_only() {
    // 1x3 input, pad top=2, bottom=1
    let input = vec![1.0, 2.0, 3.0];
    let output = pad2d_reference(&input, 1, 3, 2, 1, 0, 0, 0.0);
    // out_h = 4, out_w = 3
    assert_eq!(output.len(), 4 * 3);
    let expected = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
    assert_eq!(output, expected);
}

#[test]
#[should_panic(expected = "n must be > 0")]
fn test_pad1d_spirv_panics_zero_n() {
    let _ = generate_pad_spirv(0, 1, 1, 0.0);
}

#[test]
#[should_panic(expected = "h must be > 0")]
fn test_pad2d_spirv_panics_zero_h() {
    let _ = generate_pad2d_spirv(0, 4, 1, 1, 1, 1, 0.0);
}

#[test]
#[should_panic(expected = "w must be > 0")]
fn test_pad2d_spirv_panics_zero_w() {
    let _ = generate_pad2d_spirv(4, 0, 1, 1, 1, 1, 0.0);
}
