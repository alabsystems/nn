// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `spirv_upsample` module: nearest-neighbor upsampling SPIR-V
//! generation and CPU reference implementations.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_upsample::{
    generate_upsample_nearest1d_spirv, generate_upsample_nearest2d_spirv,
    upsample_nearest1d_reference, upsample_nearest2d_reference, UPSAMPLE_WORKGROUP_SIZE,
};

// ---- 1D upsample SPIR-V validity ----

#[test]
fn test_upsample1d_spirv_starts_with_magic() {
    let words = generate_upsample_nearest1d_spirv(64, 2);
    assert!(words.len() >= 5, "upsample1d module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "upsample1d: wrong SPIR-V magic");
}

#[test]
fn test_upsample1d_spirv_has_entry_point_and_workgroup() {
    let words = generate_upsample_nearest1d_spirv(64, 3);
    let name = find_entry_point_name(&words).expect("upsample1d must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("upsample1d must have workgroup size");
    assert_eq!(wg, [UPSAMPLE_WORKGROUP_SIZE, 1, 1]);
}

// ---- 1D upsample reference correctness ----

#[test]
fn test_upsample1d_reference_scale2() {
    let input = vec![1.0, 2.0, 3.0];
    let result = upsample_nearest1d_reference(&input, 2);
    assert_eq!(result, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_upsample1d_reference_scale3() {
    let input = vec![10.0, 20.0];
    let result = upsample_nearest1d_reference(&input, 3);
    assert_eq!(result, vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
}

#[test]
fn test_upsample1d_reference_scale1() {
    let input = vec![5.0, 6.0, 7.0];
    let result = upsample_nearest1d_reference(&input, 1);
    assert_eq!(result, vec![5.0, 6.0, 7.0]);
}

#[test]
fn test_upsample1d_reference_empty() {
    let input: Vec<f32> = vec![];
    let result = upsample_nearest1d_reference(&input, 4);
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "scale must be > 0")]
fn test_upsample1d_reference_zero_scale_panics() {
    let _ = upsample_nearest1d_reference(&[1.0], 0);
}

// ---- 2D upsample SPIR-V validity ----

#[test]
fn test_upsample2d_spirv_starts_with_magic() {
    let words = generate_upsample_nearest2d_spirv(4, 4, 2, 2);
    assert!(words.len() >= 5, "upsample2d module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "upsample2d: wrong SPIR-V magic");
}

#[test]
fn test_upsample2d_spirv_has_entry_point_and_workgroup() {
    let words = generate_upsample_nearest2d_spirv(4, 4, 2, 3);
    let name = find_entry_point_name(&words).expect("upsample2d must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("upsample2d must have workgroup size");
    assert_eq!(wg, [UPSAMPLE_WORKGROUP_SIZE, 1, 1]);
}

// ---- 2D upsample reference correctness ----

#[test]
fn test_upsample2d_reference_scale2x2() {
    // 2x2 input → 4x4 output
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = upsample_nearest2d_reference(&input, 2, 2, 2, 2);
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
fn test_upsample2d_reference_asymmetric_scale() {
    // 2x3 input with scale_h=2, scale_w=1 → 4x3 output
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = upsample_nearest2d_reference(&input, 2, 3, 2, 1);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        4.0, 5.0, 6.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_upsample2d_reference_scale1x1() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = upsample_nearest2d_reference(&input, 2, 2, 1, 1);
    assert_eq!(result, input);
}

#[test]
#[should_panic(expected = "scale_h must be > 0")]
fn test_upsample2d_reference_zero_scale_h_panics() {
    let _ = upsample_nearest2d_reference(&[1.0], 1, 1, 0, 2);
}

#[test]
#[should_panic(expected = "scale_w must be > 0")]
fn test_upsample2d_reference_zero_scale_w_panics() {
    let _ = upsample_nearest2d_reference(&[1.0], 1, 1, 2, 0);
}
