// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `spirv_where` module: where/select and abs SPIR-V generation
//! and CPU reference implementations.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_where::{
    abs_reference, generate_abs_spirv, generate_where_spirv, where_reference, WHERE_WORKGROUP_SIZE,
};

// ---- Where SPIR-V validity ----

#[test]
fn test_where_spirv_starts_with_magic() {
    let words = generate_where_spirv(1024);
    assert!(words.len() >= 5, "where module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "where: wrong SPIR-V magic");
}

#[test]
fn test_where_spirv_has_entry_point_and_workgroup() {
    let words = generate_where_spirv(512);
    let name = find_entry_point_name(&words).expect("where must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("where must have workgroup size");
    assert_eq!(wg, [WHERE_WORKGROUP_SIZE, 1, 1]);
}

// ---- Where reference correctness ----

#[test]
fn test_where_reference_basic() {
    let condition = vec![1, 0, 1, 0, 1];
    let a = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = where_reference(&condition, &a, &b);
    assert_eq!(result, vec![10.0, 2.0, 30.0, 4.0, 50.0]);
}

#[test]
fn test_where_reference_all_true() {
    let condition = vec![1, 1, 1];
    let a = vec![10.0, 20.0, 30.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = where_reference(&condition, &a, &b);
    assert_eq!(result, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_where_reference_all_false() {
    let condition = vec![0, 0, 0];
    let a = vec![10.0, 20.0, 30.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = where_reference(&condition, &a, &b);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_where_reference_nonzero_condition_values() {
    // Any non-zero value is treated as true.
    let condition = vec![0, 5, 0, 255, 1];
    let a = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = where_reference(&condition, &a, &b);
    assert_eq!(result, vec![1.0, 20.0, 3.0, 40.0, 50.0]);
}

#[test]
fn test_where_reference_empty() {
    let condition: Vec<u32> = vec![];
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let result = where_reference(&condition, &a, &b);
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "same length")]
fn test_where_reference_mismatched_lengths_panics() {
    let condition = vec![1, 0];
    let a = vec![1.0, 2.0, 3.0]; // length mismatch
    let b = vec![4.0, 5.0];
    let _ = where_reference(&condition, &a, &b);
}

// ---- Abs SPIR-V validity ----

#[test]
fn test_abs_spirv_starts_with_magic() {
    let words = generate_abs_spirv(1024);
    assert!(words.len() >= 5, "abs module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "abs: wrong SPIR-V magic");
}

#[test]
fn test_abs_spirv_has_entry_point_and_workgroup() {
    let words = generate_abs_spirv(256);
    let name = find_entry_point_name(&words).expect("abs must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("abs must have workgroup size");
    assert_eq!(wg, [WHERE_WORKGROUP_SIZE, 1, 1]);
}

// ---- Abs reference correctness ----

#[test]
fn test_abs_reference_mixed() {
    let input = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
    let result = abs_reference(&input);
    assert_eq!(result, vec![3.0, 1.5, 0.0, 1.5, 3.0]);
}

#[test]
fn test_abs_reference_all_negative() {
    let input = vec![-10.0, -5.0, -0.1];
    let result = abs_reference(&input);
    assert_eq!(result, vec![10.0, 5.0, 0.1]);
}

#[test]
fn test_abs_reference_all_positive() {
    let input = vec![1.0, 2.0, 3.0];
    let result = abs_reference(&input);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_abs_reference_empty() {
    let input: Vec<f32> = vec![];
    let result = abs_reference(&input);
    assert!(result.is_empty());
}

#[test]
fn test_abs_reference_single_element() {
    assert_eq!(abs_reference(&[-42.0]), vec![42.0]);
    assert_eq!(abs_reference(&[42.0]), vec![42.0]);
}
