// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `spirv_clamp` module: clamp/where SPIR-V generation and CPU references.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_clamp::{
    clamp_reference, generate_clamp_spirv, generate_where_spirv, where_reference,
    CLAMP_WORKGROUP_SIZE,
};
use crate::spirv_emit::SPIRV_MAGIC;

// ---- Clamp SPIR-V validity ----

#[test]
fn test_clamp_spirv_starts_with_magic() {
    let words = generate_clamp_spirv(1024, 0.0, 1.0);
    assert!(words.len() >= 5, "clamp module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "clamp: wrong SPIR-V magic");
}

#[test]
fn test_clamp_spirv_has_entry_point_and_workgroup() {
    let words = generate_clamp_spirv(1024, -1.0, 1.0);
    let name = find_entry_point_name(&words).expect("clamp must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("clamp must have workgroup size");
    assert_eq!(wg, [CLAMP_WORKGROUP_SIZE, 1, 1]);
}

// ---- Where SPIR-V validity ----

#[test]
fn test_where_spirv_starts_with_magic() {
    let words = generate_where_spirv(1024);
    assert!(words.len() >= 5, "where module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "where: wrong SPIR-V magic");
}

#[test]
fn test_where_spirv_has_entry_point_and_workgroup() {
    let words = generate_where_spirv(1024);
    let name = find_entry_point_name(&words).expect("where must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("where must have workgroup size");
    assert_eq!(wg, [CLAMP_WORKGROUP_SIZE, 1, 1]);
}

// ---- Clamp reference correctness ----

#[test]
fn test_clamp_reference_basic() {
    let input = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let result = clamp_reference(&input, -1.0, 1.0);
    assert_eq!(result, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
}

#[test]
fn test_clamp_reference_all_below() {
    let input = vec![-10.0, -5.0, -3.0];
    let result = clamp_reference(&input, 0.0, 1.0);
    assert_eq!(result, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_clamp_reference_all_above() {
    let input = vec![10.0, 5.0, 3.0];
    let result = clamp_reference(&input, 0.0, 1.0);
    assert_eq!(result, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_clamp_reference_empty() {
    let input: Vec<f32> = vec![];
    let result = clamp_reference(&input, 0.0, 1.0);
    assert!(result.is_empty());
}

#[test]
fn test_clamp_reference_single_element() {
    let result = clamp_reference(&[5.0], 0.0, 3.0);
    assert_eq!(result, vec![3.0]);
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
fn test_where_reference_nonzero_condition() {
    // Any non-zero value counts as true.
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

// ---- Edge cases ----

#[test]
#[should_panic(expected = "same length")]
fn test_where_reference_mismatched_lengths_panics() {
    let condition = vec![1, 0];
    let a = vec![1.0, 2.0, 3.0]; // length mismatch
    let b = vec![4.0, 5.0];
    let _ = where_reference(&condition, &a, &b);
}
