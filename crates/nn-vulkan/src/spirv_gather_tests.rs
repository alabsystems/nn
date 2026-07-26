// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `spirv_gather` module: gather/scatter SPIR-V generation and CPU references.

use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_gather::{
    gather_reference, generate_gather_spirv, generate_scatter_spirv, scatter_reference,
    GATHER_WORKGROUP_SIZE,
};

// ---- Gather SPIR-V validity ----

#[test]
fn test_gather_spirv_starts_with_magic() {
    let words = generate_gather_spirv(1024, 256);
    assert!(words.len() >= 5, "gather module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "gather: wrong SPIR-V magic");
}

#[test]
fn test_gather_spirv_has_entry_point_and_workgroup() {
    let words = generate_gather_spirv(1024, 256);
    let name = find_entry_point_name(&words).expect("gather must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("gather must have workgroup size");
    assert_eq!(wg, [GATHER_WORKGROUP_SIZE, 1, 1]);
}

// ---- Scatter SPIR-V validity ----

#[test]
fn test_scatter_spirv_starts_with_magic() {
    let words = generate_scatter_spirv(1024, 256);
    assert!(words.len() >= 5, "scatter module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "scatter: wrong SPIR-V magic");
}

#[test]
fn test_scatter_spirv_has_entry_point_and_workgroup() {
    let words = generate_scatter_spirv(1024, 256);
    let name = find_entry_point_name(&words).expect("scatter must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("scatter must have workgroup size");
    assert_eq!(wg, [GATHER_WORKGROUP_SIZE, 1, 1]);
}

// ---- Gather reference correctness ----

#[test]
fn test_gather_reference_basic() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = vec![4, 2, 0, 3, 1];
    let result = gather_reference(&input, &indices);
    assert_eq!(result, vec![50.0, 30.0, 10.0, 40.0, 20.0]);
}

#[test]
fn test_gather_reference_duplicate_indices() {
    let input = vec![1.0, 2.0, 3.0];
    let indices = vec![0, 0, 0, 2, 2];
    let result = gather_reference(&input, &indices);
    assert_eq!(result, vec![1.0, 1.0, 1.0, 3.0, 3.0]);
}

#[test]
fn test_gather_reference_single_element() {
    let input = vec![42.0];
    let indices = vec![0];
    let result = gather_reference(&input, &indices);
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_gather_reference_empty() {
    let input = vec![1.0, 2.0, 3.0];
    let indices: Vec<u32> = vec![];
    let result = gather_reference(&input, &indices);
    assert!(result.is_empty());
}

// ---- Scatter reference correctness ----

#[test]
fn test_scatter_reference_basic() {
    let values = vec![10.0, 20.0, 30.0];
    let indices = vec![2, 0, 4];
    let result = scatter_reference(&values, &indices, 5);
    assert_eq!(result, vec![20.0, 0.0, 10.0, 0.0, 30.0]);
}

#[test]
fn test_scatter_reference_duplicate_indices_last_wins() {
    let values = vec![1.0, 2.0, 3.0];
    let indices = vec![0, 0, 0];
    let result = scatter_reference(&values, &indices, 3);
    // Last write wins: output[0] = 3.0 (from values[2]).
    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test]
fn test_scatter_reference_single_element() {
    let values = vec![99.0];
    let indices = vec![0];
    let result = scatter_reference(&values, &indices, 1);
    assert_eq!(result, vec![99.0]);
}

#[test]
fn test_scatter_reference_empty() {
    let values: Vec<f32> = vec![];
    let indices: Vec<u32> = vec![];
    let result = scatter_reference(&values, &indices, 4);
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

// ---- Edge cases ----

#[test]
#[should_panic(expected = "out of bounds")]
fn test_gather_reference_out_of_bounds_panics() {
    let input = vec![1.0, 2.0, 3.0];
    let indices = vec![5]; // out of bounds
    let _ = gather_reference(&input, &indices);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_scatter_reference_out_of_bounds_panics() {
    let values = vec![1.0];
    let indices = vec![10]; // out of bounds
    let _ = scatter_reference(&values, &indices, 3);
}
