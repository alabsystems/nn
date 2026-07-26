// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX nearest-neighbor upsampling kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// 1D Upsample: PTX validity
// ---------------------------------------------------------------------------

#[test]
fn test_upsample_nearest1d_ptx_contains_entry() {
    let ptx = generate_upsample_nearest1d_ptx(10, 2);
    assert!(
        ptx.contains(".entry ptx_upsample_nearest1d_f32"),
        "PTX must contain 1D upsample kernel entry point"
    );
}

#[test]
fn test_upsample_nearest1d_ptx_contains_version() {
    let ptx = generate_upsample_nearest1d_ptx(4, 3);
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
fn test_upsample_nearest1d_ptx_contains_params() {
    let ptx = generate_upsample_nearest1d_ptx(8, 2);
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
// 1D Upsample: reference correctness
// ---------------------------------------------------------------------------

#[test]
fn test_upsample_nearest1d_reference_scale2() {
    let input = vec![1.0, 2.0, 3.0];
    let result = upsample_nearest1d_reference(&input, 2);
    assert_eq!(result, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_upsample_nearest1d_reference_scale3() {
    let input = vec![10.0, 20.0];
    let result = upsample_nearest1d_reference(&input, 3);
    assert_eq!(result, vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
}

#[test]
fn test_upsample_nearest1d_reference_scale1_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = upsample_nearest1d_reference(&input, 1);
    assert_eq!(result, input, "Scale 1 must be identity");
}

#[test]
fn test_upsample_nearest1d_reference_single_element() {
    let input = vec![42.0];
    let result = upsample_nearest1d_reference(&input, 5);
    assert_eq!(result, vec![42.0, 42.0, 42.0, 42.0, 42.0]);
}

#[test]
fn test_upsample_nearest1d_reference_empty() {
    let input: Vec<f32> = vec![];
    let result = upsample_nearest1d_reference(&input, 3);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// 2D Upsample: PTX validity
// ---------------------------------------------------------------------------

#[test]
fn test_upsample_nearest2d_ptx_contains_entry() {
    let ptx = generate_upsample_nearest2d_ptx(4, 4, 2, 2);
    assert!(
        ptx.contains(".entry ptx_upsample_nearest2d_f32"),
        "PTX must contain 2D upsample kernel entry point"
    );
}

#[test]
fn test_upsample_nearest2d_ptx_contains_version() {
    let ptx = generate_upsample_nearest2d_ptx(2, 3, 2, 3);
    assert!(
        ptx.contains(".version"),
        "PTX must contain version directive"
    );
    assert!(ptx.contains(".target"), "PTX must contain target directive");
}

#[test]
fn test_upsample_nearest2d_ptx_contains_params() {
    let ptx = generate_upsample_nearest2d_ptx(4, 4, 2, 2);
    assert!(
        ptx.contains("param_input"),
        "PTX must reference input param"
    );
    assert!(
        ptx.contains("param_output"),
        "PTX must reference output param"
    );
    assert!(ptx.contains("param_h"), "PTX must reference h param");
    assert!(ptx.contains("param_w"), "PTX must reference w param");
}

// ---------------------------------------------------------------------------
// 2D Upsample: reference correctness
// ---------------------------------------------------------------------------

#[test]
fn test_upsample_nearest2d_reference_scale2() {
    // 2x2 input:
    // [1, 2]
    // [3, 4]
    // upsampled by 2x2 -> 4x4:
    // [1, 1, 2, 2]
    // [1, 1, 2, 2]
    // [3, 3, 4, 4]
    // [3, 3, 4, 4]
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = upsample_nearest2d_reference(&input, 2, 2, 2, 2);
    assert_eq!(result.len(), 16);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_upsample_nearest2d_reference_asymmetric_scale() {
    // 2x2 input, scale_h=1, scale_w=3
    // [1, 2]    ->  [1, 1, 1, 2, 2, 2]
    // [3, 4]        [3, 3, 3, 4, 4, 4]
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = upsample_nearest2d_reference(&input, 2, 2, 1, 3);
    assert_eq!(result.len(), 12);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_upsample_nearest2d_reference_scale1_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = upsample_nearest2d_reference(&input, 2, 3, 1, 1);
    assert_eq!(result, input, "Scale (1,1) must be identity");
}

#[test]
fn test_upsample_nearest2d_reference_single_element() {
    let input = vec![7.0];
    let result = upsample_nearest2d_reference(&input, 1, 1, 3, 3);
    assert_eq!(result.len(), 9);
    assert!(result.iter().all(|&v| v == 7.0));
}

#[test]
fn test_upsample_nearest2d_reference_1x3_scale2() {
    // 1x3 input: [10, 20, 30]
    // scale (2, 2) -> 2x6:
    // [10, 10, 20, 20, 30, 30]
    // [10, 10, 20, 20, 30, 30]
    let input = vec![10.0, 20.0, 30.0];
    let result = upsample_nearest2d_reference(&input, 1, 3, 2, 2);
    assert_eq!(result.len(), 12);
    #[rustfmt::skip]
    let expected = vec![
        10.0, 10.0, 20.0, 20.0, 30.0, 30.0,
        10.0, 10.0, 20.0, 20.0, 30.0, 30.0,
    ];
    assert_eq!(result, expected);
}
