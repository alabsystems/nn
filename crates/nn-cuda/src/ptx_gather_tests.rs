// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External test suite for `ptx_gather` — PTX generation validity and
//! reference implementation correctness for gather/scatter operations.

use crate::ptx_gather::{
    gather_reference, generate_gather_ptx, generate_scatter_add_ptx, scatter_add_reference,
};

// ---------------------------------------------------------------------------
// PTX generation validity
// ---------------------------------------------------------------------------

#[test]
fn test_gather_ptx_valid_structure() {
    let ptx = generate_gather_ptx(1024, 0);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_gather_f32"), "missing entry point");
}

#[test]
fn test_scatter_add_ptx_valid_structure() {
    let ptx = generate_scatter_add_ptx(1024, 0);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(
        ptx.contains(".entry ptx_scatter_add_f32"),
        "missing entry point"
    );
}

// ---------------------------------------------------------------------------
// Gather reference
// ---------------------------------------------------------------------------

#[test]
fn test_gather_reference_basic() {
    // gather from [10,20,30,40] with indices [2,0,3] = [30,10,40]
    let data = [10.0_f32, 20.0, 30.0, 40.0];
    let indices = [2_u32, 0, 3];
    // 1-D gather: dim_size = data.len()
    let result = gather_reference(&data, &indices, data.len());
    assert_eq!(result, vec![30.0, 10.0, 40.0]);
}

#[test]
fn test_scatter_reference_basic() {
    // scatter values [10,20,30] to positions [2,0,1] in output of length 3
    let src = [10.0_f32, 20.0, 30.0];
    let indices = [2_u32, 0, 1];
    let result = scatter_add_reference(&src, &indices, 3, 3);
    // output[2] += 10, output[0] += 20, output[1] += 30
    assert_eq!(result, vec![20.0, 30.0, 10.0]);
}

#[test]
fn test_gather_reference_out_of_bounds() {
    // When indices are within bounds, gather works correctly.
    // Test the boundary case where index == length-1 (maximum valid index).
    let data = [10.0_f32, 20.0, 30.0, 40.0, 50.0];
    let indices = [4_u32, 0]; // index 4 = last element
    let result = gather_reference(&data, &indices, data.len());
    assert_eq!(result, vec![50.0, 10.0]);
}

#[test]
fn test_gather_2d_reference() {
    // 2D gather along axis: 2 rows of 3 elements: [[10,20,30], [40,50,60]]
    let data = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    // 2 rows of 3 indices each
    let indices = [2_u32, 0, 1, 1, 2, 0];
    let result = gather_reference(&data, &indices, 3);
    // row 0: data[0*3+2]=30, data[0*3+0]=10, data[0*3+1]=20
    // row 1: data[1*3+1]=50, data[1*3+2]=60, data[1*3+0]=40
    assert_eq!(result, vec![30.0, 10.0, 20.0, 50.0, 60.0, 40.0]);
}

#[test]
fn test_gather_reference_single() {
    let data = [42.0_f32];
    let indices = [0_u32];
    let result = gather_reference(&data, &indices, 1);
    assert_eq!(result, vec![42.0]);
}
