// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX elementwise operation generation.

use super::*;

// ---------------------------------------------------------------------------
// PTX generation: binary ops
// ---------------------------------------------------------------------------

#[test]
fn test_generate_add_ptx_contains_entry_and_instruction() {
    let ptx = generate_add_ptx(1024);
    assert!(ptx.contains(".entry ptx_add_f32"), "missing kernel entry");
    assert!(ptx.contains("add.f32"), "missing add instruction");
    assert!(ptx.contains(".version"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains("param_a"), "missing param_a");
    assert!(ptx.contains("param_b"), "missing param_b");
    assert!(ptx.contains("param_output"), "missing param_output");
    assert!(ptx.contains("param_n"), "missing param_n");
}

#[test]
fn test_generate_sub_ptx_contains_entry_and_instruction() {
    let ptx = generate_sub_ptx(512);
    assert!(ptx.contains(".entry ptx_sub_f32"), "missing kernel entry");
    assert!(ptx.contains("sub.f32"), "missing sub instruction");
}

#[test]
fn test_generate_mul_ptx_contains_entry_and_instruction() {
    let ptx = generate_mul_ptx(2048);
    assert!(ptx.contains(".entry ptx_mul_f32"), "missing kernel entry");
    assert!(ptx.contains("mul.f32"), "missing mul instruction");
}

#[test]
fn test_generate_div_ptx_contains_entry_and_instruction() {
    let ptx = generate_div_ptx(256);
    assert!(ptx.contains(".entry ptx_div_f32"), "missing kernel entry");
    assert!(ptx.contains("div.approx.f32"), "missing div instruction");
}

// ---------------------------------------------------------------------------
// PTX generation: unary ops
// ---------------------------------------------------------------------------

#[test]
fn test_generate_exp_ptx_contains_entry_and_instructions() {
    let ptx = generate_exp_ptx(1024);
    assert!(ptx.contains(".entry ptx_exp_f32"), "missing kernel entry");
    assert!(ptx.contains("ex2.approx.f32"), "missing ex2 instruction");
    assert!(ptx.contains("mul.f32"), "missing log2(e) prescale multiply");
}

#[test]
fn test_generate_log_ptx_contains_entry_and_instructions() {
    let ptx = generate_log_ptx(1024);
    assert!(ptx.contains(".entry ptx_log_f32"), "missing kernel entry");
    assert!(ptx.contains("lg2.approx.f32"), "missing lg2 instruction");
    assert!(ptx.contains("mul.f32"), "missing ln(2) postscale multiply");
}

#[test]
fn test_generate_sqrt_ptx_contains_entry_and_instruction() {
    let ptx = generate_sqrt_ptx(1024);
    assert!(ptx.contains(".entry ptx_sqrt_f32"), "missing kernel entry");
    assert!(ptx.contains("sqrt.approx.f32"), "missing sqrt instruction");
}

#[test]
fn test_generate_neg_ptx_contains_entry_and_instruction() {
    let ptx = generate_neg_ptx(1024);
    assert!(ptx.contains(".entry ptx_neg_f32"), "missing kernel entry");
    assert!(ptx.contains("neg.f32"), "missing neg instruction");
}

// ---------------------------------------------------------------------------
// PTX generation: scalar ops
// ---------------------------------------------------------------------------

#[test]
fn test_generate_scalar_mul_ptx_contains_entry_and_instruction() {
    let ptx = generate_scalar_mul_ptx(1024);
    assert!(
        ptx.contains(".entry ptx_scalar_mul_f32"),
        "missing kernel entry"
    );
    assert!(ptx.contains("param_scalar"), "missing scalar parameter");
    assert!(ptx.contains("mul.f32"), "missing mul instruction");
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_elementwise_launch_config_small() {
    let (grid, block) = ptx_elementwise_launch_config(100);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [1, 1, 1]); // ceil(100/256) = 1
}

#[test]
fn test_ptx_elementwise_launch_config_exact() {
    let (grid, block) = ptx_elementwise_launch_config(256);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [1, 1, 1]);
}

#[test]
fn test_ptx_elementwise_launch_config_large() {
    let (grid, block) = ptx_elementwise_launch_config(1000);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [4, 1, 1]); // ceil(1000/256) = 4
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

#[test]
fn test_add_reference() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = add_reference(&a, &b);
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_sub_reference() {
    let a = vec![5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let result = sub_reference(&a, &b);
    assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_mul_reference() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let result = mul_reference(&a, &b);
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_div_reference() {
    let a = vec![10.0, 20.0, 30.0, 40.0];
    let b = vec![2.0, 4.0, 5.0, 8.0];
    let result = div_reference(&a, &b);
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_exp_reference() {
    let input = vec![0.0, 1.0, -1.0];
    let result = exp_reference(&input);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
    assert!((result[2] - (-1.0_f32).exp()).abs() < 1e-6);
}

#[test]
fn test_log_reference() {
    let input = vec![1.0, std::f32::consts::E, 10.0];
    let result = log_reference(&input);
    assert!((result[0]).abs() < 1e-6); // ln(1) = 0
    assert!((result[1] - 1.0).abs() < 1e-5); // ln(e) = 1
    assert!((result[2] - 10.0_f32.ln()).abs() < 1e-6);
}

#[test]
fn test_sqrt_reference() {
    let input = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let result = sqrt_reference(&input);
    assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_neg_reference() {
    let input = vec![1.0, -2.0, 0.0, 3.5];
    let result = neg_reference(&input);
    assert_eq!(result, vec![-1.0, 2.0, -0.0, -3.5]);
}

#[test]
fn test_scalar_mul_reference() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = scalar_mul_reference(&input, 2.5);
    assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);
}

// ---------------------------------------------------------------------------
// PTX structural tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_binary_ops_have_grid_stride_loop() {
    for gen_fn in [
        generate_add_ptx,
        generate_sub_ptx,
        generate_mul_ptx,
        generate_div_ptx,
    ] {
        let ptx = gen_fn(1024);
        assert!(ptx.contains("EW_LOOP:"), "missing grid-stride loop label");
        assert!(ptx.contains("EW_EXIT:"), "missing exit label");
        assert!(ptx.contains("ld.global.f32"), "missing global load");
        assert!(ptx.contains("st.global.f32"), "missing global store");
    }
}

#[test]
fn test_all_unary_ops_have_address_size_64() {
    for ptx in [
        generate_exp_ptx(256),
        generate_log_ptx(256),
        generate_sqrt_ptx(256),
        generate_neg_ptx(256),
    ] {
        assert!(
            ptx.contains(".address_size 64"),
            "missing 64-bit addressing"
        );
    }
}
