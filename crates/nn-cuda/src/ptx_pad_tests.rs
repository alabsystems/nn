// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX 1D padding kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// Constant padding: PTX validity
// ---------------------------------------------------------------------------

#[test]
fn test_pad1d_ptx_contains_entry() {
    let ptx = generate_pad1d_ptx(10, 2, 3, 0.0);
    assert!(
        ptx.contains(".entry ptx_pad1d_const_f32"),
        "PTX must contain kernel entry point"
    );
}

#[test]
fn test_pad1d_ptx_contains_version() {
    let ptx = generate_pad1d_ptx(8, 1, 1, 0.0);
    assert!(
        ptx.contains(".version"),
        "PTX must contain version directive"
    );
    assert!(ptx.contains(".target"), "PTX must contain target directive");
    assert!(
        ptx.contains(".address_size 64"),
        "PTX must contain 64-bit address size"
    );
}

#[test]
fn test_pad1d_ptx_contains_params() {
    let ptx = generate_pad1d_ptx(10, 2, 2, 1.0);
    assert!(
        ptx.contains("param_input"),
        "PTX must reference input param"
    );
    assert!(
        ptx.contains("param_output"),
        "PTX must reference output param"
    );
    assert!(ptx.contains("param_n"), "PTX must reference n param");
}

// ---------------------------------------------------------------------------
// Constant padding: reference correctness
// ---------------------------------------------------------------------------

#[test]
fn test_pad1d_reference_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let result = pad1d_reference(&input, 2, 3, 0.0);
    assert_eq!(result.len(), 8);
    assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_pad1d_reference_nonzero_pad_value() {
    let input = vec![10.0, 20.0];
    let result = pad1d_reference(&input, 1, 1, -1.0);
    assert_eq!(result, vec![-1.0, 10.0, 20.0, -1.0]);
}

#[test]
fn test_pad1d_reference_zero_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = pad1d_reference(&input, 0, 0, 0.0);
    assert_eq!(result, input);
}

#[test]
fn test_pad1d_reference_left_only() {
    let input = vec![5.0];
    let result = pad1d_reference(&input, 3, 0, 0.0);
    assert_eq!(result, vec![0.0, 0.0, 0.0, 5.0]);
}

#[test]
fn test_pad1d_reference_right_only() {
    let input = vec![5.0];
    let result = pad1d_reference(&input, 0, 3, 0.0);
    assert_eq!(result, vec![5.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_pad1d_reference_empty_input() {
    let input: Vec<f32> = vec![];
    let result = pad1d_reference(&input, 2, 2, 7.0);
    assert_eq!(result, vec![7.0, 7.0, 7.0, 7.0]);
}

// ---------------------------------------------------------------------------
// Reflect padding: PTX validity
// ---------------------------------------------------------------------------

#[test]
fn test_reflect_pad1d_ptx_contains_entry() {
    let ptx = generate_reflect_pad1d_ptx(10, 2, 3);
    assert!(
        ptx.contains(".entry ptx_reflect_pad1d_f32"),
        "PTX must contain reflect kernel entry point"
    );
}

#[test]
fn test_reflect_pad1d_ptx_contains_version() {
    let ptx = generate_reflect_pad1d_ptx(8, 1, 1);
    assert!(
        ptx.contains(".version"),
        "PTX must contain version directive"
    );
    assert!(ptx.contains(".target"), "PTX must contain target directive");
}

// ---------------------------------------------------------------------------
// Reflect padding: reference correctness
// ---------------------------------------------------------------------------

#[test]
fn test_reflect_pad1d_reference_basic() {
    // Input: [1, 2, 3, 4, 5]
    // Pad left 2, right 2
    // Left reflect: [3, 2, ...] (input[2], input[1])
    // Right reflect: [..., 4, 3] (input[3], input[2])
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = reflect_pad1d_reference(&input, 2, 2);
    assert_eq!(result.len(), 9);
    assert_eq!(result, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
}

#[test]
fn test_reflect_pad1d_reference_small() {
    // Input: [10, 20, 30]
    // Pad left 1, right 1
    // Left: [20] (input[1])
    // Right: [20] (input[1])
    let input = vec![10.0, 20.0, 30.0];
    let result = reflect_pad1d_reference(&input, 1, 1);
    assert_eq!(result.len(), 5);
    assert_eq!(result, vec![20.0, 10.0, 20.0, 30.0, 20.0]);
}

#[test]
fn test_reflect_pad1d_reference_zero_padding() {
    let input = vec![1.0, 2.0, 3.0];
    let result = reflect_pad1d_reference(&input, 0, 0);
    assert_eq!(result, input);
}

#[test]
fn test_reflect_pad1d_reference_left_only() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = reflect_pad1d_reference(&input, 3, 0);
    // Left reflect: input[3], input[2], input[1] = 4, 3, 2
    assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_reflect_pad1d_reference_right_only() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = reflect_pad1d_reference(&input, 0, 3);
    // Right reflect: input[2], input[1], input[0] = 3, 2, 1
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_pad1d_ptx_zero_pad_left() {
    // Should still produce valid PTX
    let ptx = generate_pad1d_ptx(10, 0, 5, 0.0);
    assert!(ptx.contains(".entry ptx_pad1d_const_f32"));
}

#[test]
fn test_pad1d_ptx_zero_pad_right() {
    let ptx = generate_pad1d_ptx(10, 5, 0, 0.0);
    assert!(ptx.contains(".entry ptx_pad1d_const_f32"));
}

#[test]
fn test_pad1d_ptx_large_pad_value() {
    let ptx = generate_pad1d_ptx(4, 1, 1, 999.0);
    assert!(ptx.contains(".entry ptx_pad1d_const_f32"));
}

#[test]
fn test_reflect_pad1d_ptx_zero_left() {
    let ptx = generate_reflect_pad1d_ptx(10, 0, 3);
    assert!(ptx.contains(".entry ptx_reflect_pad1d_f32"));
}

#[test]
fn test_reflect_pad1d_ptx_zero_right() {
    let ptx = generate_reflect_pad1d_ptx(10, 3, 0);
    assert!(ptx.contains(".entry ptx_reflect_pad1d_f32"));
}
