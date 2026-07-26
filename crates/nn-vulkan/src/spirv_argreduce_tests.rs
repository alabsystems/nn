// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External tests for `spirv_argreduce` module.
//!
//! Tests the public API from outside the module, exercising SPIR-V generation
//! for argmax, argmin, and top-k kernels, plus CPU reference correctness.

use crate::spirv_argreduce::{
    argmax_reference, argmin_reference, generate_argmax_spirv, generate_argmin_spirv,
    generate_topk_spirv, ARGREDUCE_WORKGROUP_SIZE,
};
use crate::spirv_binary::find_entry_point_name;
use crate::spirv_binary::find_workgroup_size;
use crate::spirv_emit::SPIRV_MAGIC;

// ---- test_argmax_spirv_valid ----

#[test]
fn test_argmax_spirv_valid() {
    let words = generate_argmax_spirv(1024);
    assert!(words.len() >= 5, "argmax module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "argmax: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("argmax must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("argmax must have workgroup size");
    assert_eq!(wg, [ARGREDUCE_WORKGROUP_SIZE, 1, 1]);
}

// ---- test_argmin_spirv_valid ----

#[test]
fn test_argmin_spirv_valid() {
    let words = generate_argmin_spirv(1024);
    assert!(words.len() >= 5, "argmin module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "argmin: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("argmin must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("argmin must have workgroup size");
    assert_eq!(wg, [ARGREDUCE_WORKGROUP_SIZE, 1, 1]);
}

// ---- test_topk_spirv_valid ----

#[test]
fn test_topk_spirv_valid() {
    let words = generate_topk_spirv(100, 5);
    assert!(words.len() >= 5, "topk module too short");
    assert_eq!(words[0], SPIRV_MAGIC, "topk: wrong SPIR-V magic");
    let name = find_entry_point_name(&words).expect("topk must have entry point");
    assert_eq!(name, "main");
    let wg = find_workgroup_size(&words).expect("topk must have workgroup size");
    assert_eq!(wg, [ARGREDUCE_WORKGROUP_SIZE, 1, 1]);
}

// ---- test_argmax_reference_basic ----

#[test]
fn test_argmax_reference_basic() {
    let data = vec![1.0f32, 3.0, 2.0];
    assert_eq!(argmax_reference(&data), 1, "argmax of [1,3,2] should be 1");
}

// ---- test_argmin_reference_basic ----

#[test]
fn test_argmin_reference_basic() {
    let data = vec![3.0f32, 1.0, 2.0];
    assert_eq!(argmin_reference(&data), 1, "argmin of [3,1,2] should be 1");
}

// ---- test_argmax_reference_duplicate ----

#[test]
fn test_argmax_reference_duplicate() {
    // When duplicate max values exist, argmax returns the first (lowest) index.
    let data = vec![1.0f32, 5.0, 3.0, 5.0, 2.0];
    assert_eq!(
        argmax_reference(&data),
        1,
        "argmax with duplicate max should return first occurrence"
    );
}

// ---- test_argreduce_reference_single_element ----

#[test]
fn test_argreduce_reference_single_element() {
    let data = vec![42.0f32];
    assert_eq!(
        argmax_reference(&data),
        0,
        "argmax of single element should be 0"
    );
    assert_eq!(
        argmin_reference(&data),
        0,
        "argmin of single element should be 0"
    );
}
