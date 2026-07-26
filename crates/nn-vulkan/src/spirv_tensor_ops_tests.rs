// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `spirv_tensor_ops` module: concat, slice, repeat, fill SPIR-V
//! generation and CPU reference implementations.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_tensor_ops::{
    concat_reference, fill_reference, generate_concat_spirv, generate_fill_spirv,
    generate_repeat_spirv, generate_slice_spirv, repeat_reference, slice_reference,
    TENSOR_OPS_WORKGROUP_SIZE,
};

// ============================================================================
// Concat SPIR-V validity
// ============================================================================

#[test]
fn test_concat_spirv_starts_with_magic() {
    let words = generate_concat_spirv(128, 64);
    assert!(words.len() >= 5, "concat module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "concat: wrong SPIR-V magic");
}

#[test]
fn test_concat_spirv_has_entry_point_and_workgroup() {
    let words = generate_concat_spirv(128, 64);
    let name = find_entry_point_name(&words).expect("concat must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("concat must have workgroup size");
    assert_eq!(wg, [TENSOR_OPS_WORKGROUP_SIZE, 1, 1]);
}

// ============================================================================
// Concat reference correctness
// ============================================================================

#[test]
fn test_concat_reference_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0];
    let result = concat_reference(&a, &b);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_concat_reference_empty_a() {
    let a: Vec<f32> = vec![];
    let b = vec![10.0, 20.0];
    let result = concat_reference(&a, &b);
    assert_eq!(result, vec![10.0, 20.0]);
}

#[test]
fn test_concat_reference_empty_b() {
    let a = vec![10.0, 20.0];
    let b: Vec<f32> = vec![];
    let result = concat_reference(&a, &b);
    assert_eq!(result, vec![10.0, 20.0]);
}

#[test]
fn test_concat_reference_both_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let result = concat_reference(&a, &b);
    assert!(result.is_empty());
}

// ============================================================================
// Slice SPIR-V validity
// ============================================================================

#[test]
fn test_slice_spirv_starts_with_magic() {
    let words = generate_slice_spirv(256, 10, 50);
    assert!(words.len() >= 5, "slice module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "slice: wrong SPIR-V magic");
}

#[test]
fn test_slice_spirv_has_entry_point_and_workgroup() {
    let words = generate_slice_spirv(256, 10, 50);
    let name = find_entry_point_name(&words).expect("slice must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("slice must have workgroup size");
    assert_eq!(wg, [TENSOR_OPS_WORKGROUP_SIZE, 1, 1]);
}

// ============================================================================
// Slice reference correctness
// ============================================================================

#[test]
fn test_slice_reference_basic() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let result = slice_reference(&input, 1, 3);
    assert_eq!(result, vec![20.0, 30.0, 40.0]);
}

#[test]
fn test_slice_reference_full() {
    let input = vec![1.0, 2.0, 3.0];
    let result = slice_reference(&input, 0, 3);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_slice_reference_single_element() {
    let input = vec![10.0, 20.0, 30.0];
    let result = slice_reference(&input, 2, 1);
    assert_eq!(result, vec![30.0]);
}

#[test]
fn test_slice_reference_zero_length() {
    let input = vec![1.0, 2.0, 3.0];
    let result = slice_reference(&input, 1, 0);
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "slice out of bounds")]
fn test_slice_reference_out_of_bounds_panics() {
    let input = vec![1.0, 2.0, 3.0];
    let _ = slice_reference(&input, 2, 5);
}

// ============================================================================
// Repeat SPIR-V validity
// ============================================================================

#[test]
fn test_repeat_spirv_starts_with_magic() {
    let words = generate_repeat_spirv(64, 3);
    assert!(words.len() >= 5, "repeat module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "repeat: wrong SPIR-V magic");
}

#[test]
fn test_repeat_spirv_has_entry_point_and_workgroup() {
    let words = generate_repeat_spirv(64, 3);
    let name = find_entry_point_name(&words).expect("repeat must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("repeat must have workgroup size");
    assert_eq!(wg, [TENSOR_OPS_WORKGROUP_SIZE, 1, 1]);
}

// ============================================================================
// Repeat reference correctness
// ============================================================================

#[test]
fn test_repeat_reference_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let result = repeat_reference(&input, 2);
    assert_eq!(result, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_repeat_reference_single_repeat() {
    let input = vec![10.0, 20.0];
    let result = repeat_reference(&input, 1);
    assert_eq!(result, vec![10.0, 20.0]);
}

#[test]
fn test_repeat_reference_zero_repeats() {
    let input = vec![1.0, 2.0, 3.0];
    let result = repeat_reference(&input, 0);
    assert!(result.is_empty());
}

#[test]
fn test_repeat_reference_empty_input() {
    let input: Vec<f32> = vec![];
    let result = repeat_reference(&input, 5);
    assert!(result.is_empty());
}

#[test]
fn test_repeat_reference_many_repeats() {
    let input = vec![42.0];
    let result = repeat_reference(&input, 4);
    assert_eq!(result, vec![42.0, 42.0, 42.0, 42.0]);
}

// ============================================================================
// Fill SPIR-V validity
// ============================================================================

#[test]
fn test_fill_spirv_starts_with_magic() {
    let words = generate_fill_spirv(512, 3.14);
    assert!(words.len() >= 5, "fill module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "fill: wrong SPIR-V magic");
}

#[test]
fn test_fill_spirv_has_entry_point_and_workgroup() {
    let words = generate_fill_spirv(512, 3.14);
    let name = find_entry_point_name(&words).expect("fill must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("fill must have workgroup size");
    assert_eq!(wg, [TENSOR_OPS_WORKGROUP_SIZE, 1, 1]);
}

// ============================================================================
// Fill reference correctness
// ============================================================================

#[test]
fn test_fill_reference_basic() {
    let result = fill_reference(5, 7.0);
    assert_eq!(result, vec![7.0, 7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn test_fill_reference_zero_length() {
    let result = fill_reference(0, 99.0);
    assert!(result.is_empty());
}

#[test]
fn test_fill_reference_negative_value() {
    let result = fill_reference(3, -1.5);
    assert_eq!(result, vec![-1.5, -1.5, -1.5]);
}

#[test]
fn test_fill_reference_zero_value() {
    let result = fill_reference(4, 0.0);
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}
