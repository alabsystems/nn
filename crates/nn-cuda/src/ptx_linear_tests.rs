// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX Linear (matmul + bias) kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// CPU reference: linear_reference
// ---------------------------------------------------------------------------

#[test]
fn test_linear_reference_with_bias() {
    // input: [1, 3], weight: [3, 2], bias: [2]
    // input  = [1, 2, 3]
    // weight = [[1, 0],   -> weight[0*2+0]=1, weight[0*2+1]=0
    //           [0, 1],   -> weight[1*2+0]=0, weight[1*2+1]=1
    //           [1, 1]]   -> weight[2*2+0]=1, weight[2*2+1]=1
    // bias   = [10, 20]
    // out[0] = 1*1 + 2*0 + 3*1 + 10 = 14
    // out[1] = 1*0 + 2*1 + 3*1 + 20 = 25
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let bias = vec![10.0, 20.0];
    let output = linear_reference(&input, &weight, Some(&bias), 3, 2);
    assert_eq!(output, vec![14.0, 25.0]);
}

#[test]
fn test_linear_reference_without_bias() {
    // Same as above but without bias
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let output = linear_reference(&input, &weight, None, 3, 2);
    assert_eq!(output, vec![4.0, 5.0]);
}

#[test]
fn test_linear_reference_batch() {
    // input: [2, 2], weight: [2, 3], bias: [3]
    let input = vec![1.0, 0.0, 0.0, 1.0]; // two samples
    let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // identity-ish
    let bias = vec![0.5, 0.5, 0.5];
    let output = linear_reference(&input, &weight, Some(&bias), 2, 3);
    // sample 0: [1*1+0*4, 1*2+0*5, 1*3+0*6] + [0.5,0.5,0.5] = [1.5, 2.5, 3.5]
    // sample 1: [0*1+1*4, 0*2+1*5, 0*3+1*6] + [0.5,0.5,0.5] = [4.5, 5.5, 6.5]
    assert_eq!(output, vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
}

#[test]
fn test_linear_reference_single_neuron() {
    // in_features=4, out_features=1 -> dot product + bias
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0]; // sum weights
    let bias = vec![10.0];
    let output = linear_reference(&input, &weight, Some(&bias), 4, 1);
    // 1+2+3+4 + 10 = 20
    assert_eq!(output, vec![20.0]);
}

#[test]
fn test_linear_reference_identity_weight() {
    // Weight = identity -> output = input + bias
    let input = vec![5.0, 7.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let bias = vec![1.0, 2.0];
    let output = linear_reference(&input, &weight, Some(&bias), 2, 2);
    assert_eq!(output, vec![6.0, 9.0]);
}

#[test]
fn test_linear_reference_zero_bias() {
    let input = vec![1.0, 2.0];
    let weight = vec![3.0, 4.0];
    let bias = vec![0.0];
    let out_with_zero_bias = linear_reference(&input, &weight, Some(&bias), 2, 1);
    let out_no_bias = linear_reference(&input, &weight, None, 2, 1);
    assert_eq!(out_with_zero_bias, out_no_bias);
}

// ---------------------------------------------------------------------------
// Linear PTX generation: generate_linear_ptx
// ---------------------------------------------------------------------------

#[test]
fn test_generate_linear_ptx_entry_point() {
    let ptx = generate_linear_ptx(768, 3072);
    assert!(ptx.contains(".entry linear_bias_f32"));
}

#[test]
fn test_generate_linear_ptx_sm70_target() {
    let ptx = generate_linear_ptx(128, 256);
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_generate_linear_ptx_has_bias_param() {
    let ptx = generate_linear_ptx(64, 64);
    assert!(ptx.contains("param_bias"));
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_weight"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_batch"));
    assert!(ptx.contains("param_in_features"));
    assert!(ptx.contains("param_out_features"));
}

#[test]
fn test_generate_linear_ptx_has_fma() {
    let ptx = generate_linear_ptx(32, 32);
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_generate_linear_ptx_dimension_comment() {
    let ptx = generate_linear_ptx(768, 3072);
    assert!(ptx.contains("768"));
    assert!(ptx.contains("3072"));
}

// ---------------------------------------------------------------------------
// Linear no-bias PTX: generate_linear_no_bias_ptx
// ---------------------------------------------------------------------------

#[test]
fn test_generate_linear_no_bias_ptx_entry_point() {
    let ptx = generate_linear_no_bias_ptx(64, 128);
    assert!(ptx.contains(".entry linear_no_bias_f32"));
}

#[test]
fn test_generate_linear_no_bias_ptx_no_bias_param() {
    let ptx = generate_linear_no_bias_ptx(64, 128);
    // Should not have bias parameter
    assert!(!ptx.contains("param_bias"));
    // Should still have the other params
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_weight"));
    assert!(ptx.contains("param_output"));
}

#[test]
fn test_generate_linear_no_bias_ptx_has_fma() {
    let ptx = generate_linear_no_bias_ptx(32, 32);
    assert!(ptx.contains("fma.rn.f32"));
}

// ---------------------------------------------------------------------------
// Linear + ReLU PTX: generate_linear_relu_ptx
// ---------------------------------------------------------------------------

#[test]
fn test_generate_linear_relu_ptx_entry_point() {
    let ptx = generate_linear_relu_ptx(256, 512);
    assert!(ptx.contains(".entry linear_relu_f32"));
}

#[test]
fn test_generate_linear_relu_ptx_has_max_instruction() {
    let ptx = generate_linear_relu_ptx(32, 32);
    // ReLU = max(0, x) -> should contain max.f32
    assert!(
        ptx.contains("max.f32"),
        "fused ReLU must use max.f32 instruction"
    );
}

#[test]
fn test_generate_linear_relu_ptx_has_bias() {
    let ptx = generate_linear_relu_ptx(64, 64);
    assert!(ptx.contains("param_bias"));
}

#[test]
fn test_generate_linear_relu_ptx_relu_comment() {
    let ptx = generate_linear_relu_ptx(64, 64);
    assert!(ptx.contains("ReLU"));
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_linear_launch_config_basic() {
    let (grid, block) = ptx_linear_launch_config(4, 256);
    // total = 4 * 256 = 1024, block = 256, grid = 4
    assert_eq!(grid, [4, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_ptx_linear_launch_config_non_divisible() {
    let (grid, block) = ptx_linear_launch_config(1, 300);
    // total = 300, block = 256, grid = ceil(300/256) = 2
    assert_eq!(grid, [2, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_ptx_linear_launch_config_large_batch() {
    let (grid, _block) = ptx_linear_launch_config(32, 3072);
    // total = 32 * 3072 = 98304, block = 256, grid = 384
    assert_eq!(grid, [384, 1, 1]);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn test_linear_block_size_value() {
    assert_eq!(LINEAR_BLOCK_SIZE, 256);
}

// ---------------------------------------------------------------------------
// Cross-variant consistency
// ---------------------------------------------------------------------------

#[test]
fn test_linear_variants_all_have_ptx_header() {
    for ptx in [
        generate_linear_ptx(64, 64),
        generate_linear_no_bias_ptx(64, 64),
        generate_linear_relu_ptx(64, 64),
    ] {
        assert!(ptx.contains(".version"), "must contain PTX version");
        assert!(ptx.contains(".target sm_70"), "must target sm_70");
        assert!(
            ptx.contains(".address_size 64"),
            "must use 64-bit addressing"
        );
    }
}

#[test]
fn test_linear_variants_all_have_fma() {
    for ptx in [
        generate_linear_ptx(64, 64),
        generate_linear_no_bias_ptx(64, 64),
        generate_linear_relu_ptx(64, 64),
    ] {
        assert!(ptx.contains("fma.rn.f32"), "all variants must use fma");
    }
}

#[test]
fn test_linear_variants_different_entry_points() {
    let ptx_bias = generate_linear_ptx(64, 64);
    let ptx_no_bias = generate_linear_no_bias_ptx(64, 64);
    let ptx_relu = generate_linear_relu_ptx(64, 64);

    // Each variant must have a different entry point name
    assert!(ptx_bias.contains("linear_bias_f32"));
    assert!(ptx_no_bias.contains("linear_no_bias_f32"));
    assert!(ptx_relu.contains("linear_relu_f32"));

    // They should be distinct strings
    assert_ne!(ptx_bias, ptx_no_bias);
    assert_ne!(ptx_bias, ptx_relu);
    assert_ne!(ptx_no_bias, ptx_relu);
}
