// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX tensor manipulation operation generation.

use super::*;

// ---------------------------------------------------------------------------
// PTX generation: concat
// ---------------------------------------------------------------------------

#[test]
fn test_generate_concat_ptx_contains_entry_and_structure() {
    let ptx = generate_concat_ptx(512, 256);
    assert!(
        ptx.contains(".entry ptx_concat_f32"),
        "missing kernel entry"
    );
    assert!(ptx.contains(".version"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit addressing"
    );
    assert!(ptx.contains("param_a"), "missing param_a");
    assert!(ptx.contains("param_b"), "missing param_b");
    assert!(ptx.contains("param_output"), "missing param_output");
    assert!(ptx.contains("param_n_a"), "missing param_n_a");
    assert!(ptx.contains("param_n_b"), "missing param_n_b");
    assert!(ptx.contains("CONCAT_LOOP:"), "missing loop label");
    assert!(ptx.contains("CONCAT_EXIT:"), "missing exit label");
    assert!(ptx.contains("CONCAT_FROM_A:"), "missing branch label");
    assert!(ptx.contains("ld.global.f32"), "missing global load");
    assert!(ptx.contains("st.global.f32"), "missing global store");
}

#[test]
fn test_concat_reference_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0];
    let result = concat_reference(&a, &b);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_concat_reference_empty_inputs() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0, 2.0];
    assert_eq!(concat_reference(&a, &b), vec![1.0, 2.0]);
    assert_eq!(concat_reference(&b, &a), vec![1.0, 2.0]);
    let empty: Vec<f32> = vec![];
    assert_eq!(concat_reference(&empty, &empty), Vec::<f32>::new());
}

// ---------------------------------------------------------------------------
// PTX generation: slice
// ---------------------------------------------------------------------------

#[test]
fn test_generate_slice_ptx_contains_entry_and_structure() {
    let ptx = generate_slice_ptx(1024, 100, 200);
    assert!(ptx.contains(".entry ptx_slice_f32"), "missing kernel entry");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit addressing"
    );
    assert!(ptx.contains("param_input"), "missing param_input");
    assert!(ptx.contains("param_output"), "missing param_output");
    assert!(ptx.contains("param_start"), "missing param_start");
    assert!(ptx.contains("param_len"), "missing param_len");
    assert!(ptx.contains("SLICE_LOOP:"), "missing loop label");
    assert!(ptx.contains("SLICE_EXIT:"), "missing exit label");
    assert!(ptx.contains("ld.global.f32"), "missing global load");
    assert!(ptx.contains("st.global.f32"), "missing global store");
}

#[test]
fn test_slice_reference_basic() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let result = slice_reference(&input, 1, 3);
    assert_eq!(result, vec![20.0, 30.0, 40.0]);
}

#[test]
fn test_slice_reference_full_range() {
    let input = vec![1.0, 2.0, 3.0];
    let result = slice_reference(&input, 0, 3);
    assert_eq!(result, input);
}

#[test]
fn test_slice_reference_single_element() {
    let input = vec![10.0, 20.0, 30.0];
    let result = slice_reference(&input, 2, 1);
    assert_eq!(result, vec![30.0]);
}

// ---------------------------------------------------------------------------
// PTX generation: repeat
// ---------------------------------------------------------------------------

#[test]
fn test_generate_repeat_ptx_contains_entry_and_structure() {
    let ptx = generate_repeat_ptx(256, 4);
    assert!(
        ptx.contains(".entry ptx_repeat_f32"),
        "missing kernel entry"
    );
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit addressing"
    );
    assert!(ptx.contains("param_input"), "missing param_input");
    assert!(ptx.contains("param_output"), "missing param_output");
    assert!(ptx.contains("param_n"), "missing param_n");
    assert!(ptx.contains("param_repeats"), "missing param_repeats");
    assert!(ptx.contains("REPEAT_LOOP:"), "missing loop label");
    assert!(ptx.contains("REPEAT_EXIT:"), "missing exit label");
    assert!(
        ptx.contains("div.u32"),
        "missing integer division for index mapping"
    );
}

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
fn test_repeat_reference_triple() {
    let input = vec![5.0];
    let result = repeat_reference(&input, 3);
    assert_eq!(result, vec![5.0, 5.0, 5.0]);
}

// ---------------------------------------------------------------------------
// PTX generation: fill
// ---------------------------------------------------------------------------

#[test]
fn test_generate_fill_ptx_contains_entry_and_structure() {
    let ptx = generate_fill_ptx(1024, 3.14);
    assert!(ptx.contains(".entry ptx_fill_f32"), "missing kernel entry");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit addressing"
    );
    assert!(ptx.contains("param_output"), "missing param_output");
    assert!(ptx.contains("param_n"), "missing param_n");
    assert!(ptx.contains("param_value"), "missing param_value");
    assert!(ptx.contains("FILL_LOOP:"), "missing loop label");
    assert!(ptx.contains("FILL_EXIT:"), "missing exit label");
    assert!(ptx.contains("st.global.f32"), "missing global store");
}

#[test]
fn test_generate_fill_ptx_zero_value() {
    let ptx = generate_fill_ptx(512, 0.0);
    assert!(ptx.contains(".entry ptx_fill_f32"), "missing kernel entry");
}

#[test]
fn test_fill_reference_basic() {
    let result = fill_reference(4, 42.0);
    assert_eq!(result, vec![42.0, 42.0, 42.0, 42.0]);
}

#[test]
fn test_fill_reference_zero() {
    let result = fill_reference(3, 0.0);
    assert_eq!(result, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_fill_reference_empty() {
    let result = fill_reference(0, 1.0);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_ops_launch_config_small() {
    let (grid, block) = ptx_tensor_ops_launch_config(100);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [1, 1, 1]); // ceil(100/256) = 1
}

#[test]
fn test_tensor_ops_launch_config_exact() {
    let (grid, block) = ptx_tensor_ops_launch_config(256);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [1, 1, 1]);
}

#[test]
fn test_tensor_ops_launch_config_large() {
    let (grid, block) = ptx_tensor_ops_launch_config(1000);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [4, 1, 1]); // ceil(1000/256) = 4
}
