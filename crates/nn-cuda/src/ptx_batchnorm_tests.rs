// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX BatchNorm kernel generation.
//!
//! Covers config validation, PTX structural checks, reference computation
//! verification, edge cases, and CNN-typical dimensions.

use super::*;

// =========================================================================
// Config construction and validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxBatchNormConfig::new("batchnorm_64", 64, 1e-5);
    assert_eq!(c.num_channels, 64);
    assert_eq!(c.kernel_name, "batchnorm_64");
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_channels_zero_rejected() {
    let c = PtxBatchNormConfig::new("bn", 0, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxBatchNormConfig::new("", 64, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_nan_eps_rejected() {
    let c = PtxBatchNormConfig::new("bn", 64, f32::NAN);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_negative_eps_rejected() {
    let c = PtxBatchNormConfig::new("bn", 64, -0.001);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_inf_eps_rejected() {
    let c = PtxBatchNormConfig::new("bn", 64, f32::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_eps_valid() {
    let c = PtxBatchNormConfig::new("bn", 64, 0.0);
    assert!(c.validate().is_ok());
}

// =========================================================================
// Block size
// =========================================================================

#[test]
fn test_block_size_always_256() {
    for channels in [1, 3, 16, 32, 64, 128, 256, 512] {
        let c = PtxBatchNormConfig::new("bn", channels, 1e-5);
        assert_eq!(c.block_size(), 256, "channels={channels}");
    }
}

// =========================================================================
// SM target configuration
// =========================================================================

#[test]
fn test_sm_target_custom() {
    let c = PtxBatchNormConfig::new("bn", 64, 1e-5).with_sm_target("sm_70");
    assert_eq!(c.sm_target, "sm_70");
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_sm_90_hopper() {
    let c = PtxBatchNormConfig::new("bn_hopper", 256, 1e-5).with_sm_target("sm_90");
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

// =========================================================================
// PTX structural validation
// =========================================================================

#[test]
fn test_ptx_contains_version_and_target() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_ptx_contains_entry_point() {
    let c = PtxBatchNormConfig::new("nn_batchnorm", 32, 1e-5);
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains(".visible .entry nn_batchnorm"));
}

#[test]
fn test_ptx_contains_kernel_params() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_running_mean"));
    assert!(ptx.contains("param_running_var"));
    assert!(ptx.contains("param_weight"));
    assert!(ptx.contains("param_bias"));
    assert!(ptx.contains("param_num_channels"));
    assert!(ptx.contains("param_spatial_size"));
    assert!(ptx.contains("param_total"));
}

#[test]
fn test_ptx_has_rsqrt() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(
        ptx.contains("rsqrt.approx.f32"),
        "BatchNorm must use rsqrt for 1/sqrt(var + eps)"
    );
}

#[test]
fn test_ptx_has_fma_for_affine() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(
        ptx.contains("fma.rn.f32"),
        "BatchNorm must use fma for weight*norm+bias"
    );
}

#[test]
fn test_ptx_has_mean_subtraction() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(
        ptx.contains("sub.f32"),
        "BatchNorm must subtract running_mean (sub.f32)"
    );
}

#[test]
fn test_ptx_no_warp_shuffle() {
    // BatchNorm inference is elementwise -- no reduction needed
    let ptx = generate_batchnorm_ptx(64);
    assert!(
        !ptx.contains("shfl.down.sync"),
        "BatchNorm inference should not use warp shuffle (no reduction)"
    );
}

#[test]
fn test_ptx_no_shared_memory() {
    // Elementwise kernel -- no shared memory needed
    let ptx = generate_batchnorm_ptx(64);
    assert!(
        !ptx.contains(".shared"),
        "BatchNorm inference should not use shared memory"
    );
}

#[test]
fn test_ptx_comment_header() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(ptx.contains("BatchNorm f32"));
}

#[test]
fn test_ptx_reqntid() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(ptx.contains(".reqntid 256"));
}

#[test]
fn test_ptx_loads_and_stores_global() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_ptx_channel_index_computation() {
    let ptx = generate_batchnorm_ptx(64);
    // BatchNorm computes channel via div and rem for NCHW
    assert!(ptx.contains("div.u32"));
    assert!(ptx.contains("rem.u32"));
}

#[test]
fn test_ptx_ends_with_closing_brace() {
    let ptx = generate_batchnorm_ptx(64);
    let trimmed = ptx.trim_end();
    assert!(trimmed.ends_with('}'));
}

#[test]
fn test_ptx_not_cuda_cpp() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("__shared__"));
}

#[test]
fn test_ptx_reasonable_size() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(
        ptx.len() > 500,
        "PTX should be substantial, got {} bytes",
        ptx.len()
    );
    assert!(ptx.len() < 50_000, "PTX too large: {} bytes", ptx.len());
}

// =========================================================================
// Different parameters produce different PTX
// =========================================================================

#[test]
fn test_different_channels_produce_different_ptx() {
    let ptx_16 = generate_batchnorm_ptx(16);
    let ptx_64 = generate_batchnorm_ptx(64);
    let ptx_256 = generate_batchnorm_ptx(256);
    assert_ne!(ptx_16, ptx_64);
    assert_ne!(ptx_64, ptx_256);
}

#[test]
fn test_different_eps_produce_different_ptx() {
    let ptx_1e5 = emit_ptx_batchnorm_default("bn", 64, 1e-5).unwrap();
    let ptx_1e6 = emit_ptx_batchnorm_default("bn", 64, 1e-6).unwrap();
    assert_ne!(ptx_1e5, ptx_1e6);
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_small() {
    let (grid, block) = ptx_batchnorm_launch_config(256);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_large() {
    // batch=4, channels=64, spatial=100*100 = 10000, total = 2_560_000
    let total = 4 * 64 * 10000;
    let (grid, block) = ptx_batchnorm_launch_config(total);
    assert_eq!(grid, [total.div_ceil(256), 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_not_multiple_of_block() {
    let (grid, _block) = ptx_batchnorm_launch_config(1000);
    assert_eq!(grid, [4, 1, 1]); // ceil(1000/256) = 4
}

// =========================================================================
// Reference computation: known values
// =========================================================================

#[test]
fn test_reference_identity_transform() {
    // mean=0, var=1, weight=1, bias=0 -> output = input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    // batch=1, channels=1, spatial=4
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        0.0,
    );
    for (o, &x) in output.iter().zip(input.iter()) {
        assert!((o - x).abs() < 1e-5, "identity: got {o}, expected {x}");
    }
}

#[test]
fn test_reference_mean_subtraction() {
    // mean=2.5, var=1, weight=1, bias=0 -> output = input - 2.5
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![2.5];
    let running_var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        0.0,
    );
    let expected = [-1.5, -0.5, 0.5, 1.5];
    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "got {o}, expected {e}");
    }
}

#[test]
fn test_reference_variance_scaling() {
    // mean=0, var=4, weight=1, bias=0 -> output = input / 2
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![4.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        0.0,
    );
    let expected = [1.0, 2.0, 3.0, 4.0];
    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "got {o}, expected {e}");
    }
}

#[test]
fn test_reference_weight_scaling() {
    // mean=0, var=1, weight=2, bias=0 -> output = 2 * input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let weight = vec![2.0];
    let bias = vec![0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        0.0,
    );
    let expected = [2.0, 4.0, 6.0, 8.0];
    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "got {o}, expected {e}");
    }
}

#[test]
fn test_reference_bias_shift() {
    // mean=0, var=1, weight=1, bias=3 -> output = input + 3
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![3.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        0.0,
    );
    let expected = [4.0, 5.0, 6.0, 7.0];
    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "got {o}, expected {e}");
    }
}

#[test]
fn test_reference_multi_channel() {
    // batch=1, channels=2, spatial=2
    // Channel 0: mean=1, var=1, weight=1, bias=0
    // Channel 1: mean=0, var=4, weight=2, bias=1
    // Input: [c0s0, c0s1, c1s0, c1s1] = [2, 3, 4, 6]
    let input = vec![2.0, 3.0, 4.0, 6.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![1.0, 0.0];
    let running_var = vec![1.0, 4.0];
    let weight = vec![1.0, 2.0];
    let bias = vec![0.0, 1.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        2,
        2,
        0.0,
    );
    // Channel 0: (2-1)/1*1+0=1, (3-1)/1*1+0=2
    // Channel 1: (4-0)/2*2+1=5, (6-0)/2*2+1=7
    let expected = [1.0, 2.0, 5.0, 7.0];
    for (i, (o, e)) in output.iter().zip(expected.iter()).enumerate() {
        assert!((o - e).abs() < 1e-5, "i={i}: got {o}, expected {e}");
    }
}

#[test]
fn test_reference_multi_batch() {
    // batch=2, channels=1, spatial=2
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        2,
        0.0,
    );
    // Identity: output = input
    for (o, &x) in output.iter().zip(input.iter()) {
        assert!((o - x).abs() < 1e-5);
    }
}

// =========================================================================
// Reference computation: edge cases
// =========================================================================

#[test]
fn test_reference_zero_input() {
    let input = vec![0.0f32; 4];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        1e-5,
    );
    for &v in &output {
        assert!(v.abs() < 1e-3, "zero input should yield ~0, got {v}");
    }
}

#[test]
fn test_reference_negative_input() {
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let mut output = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![1.0];
    let weight = vec![1.0];
    let bias = vec![0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        0.0,
    );
    for (o, &x) in output.iter().zip(input.iter()) {
        assert!(
            (o - x).abs() < 1e-5,
            "identity for negative: got {o}, expected {x}"
        );
    }
}

#[test]
fn test_reference_eps_effect() {
    // Small variance + large eps should attenuate output
    let input = vec![1.0; 4];
    let mut output_small = vec![0.0; 4];
    let mut output_large = vec![0.0; 4];
    let running_mean = vec![0.0];
    let running_var = vec![0.001];
    let weight = vec![1.0];
    let bias = vec![0.0];

    batchnorm_reference(
        &input,
        &mut output_small,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        1e-8,
    );
    batchnorm_reference(
        &input,
        &mut output_large,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        1,
        4,
        1.0,
    );

    let mag_small: f32 = output_small.iter().map(|x| x.abs()).sum();
    let mag_large: f32 = output_large.iter().map(|x| x.abs()).sum();
    assert!(
        mag_small > mag_large,
        "larger eps should reduce output: small={mag_small}, large={mag_large}"
    );
}

// =========================================================================
// Reference: various shapes
// =========================================================================

#[test]
fn test_reference_batch4_channels16_spatial100() {
    let channels = 16;
    let spatial = 100;
    let batch = 4;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let mut output = vec![0.0; total];

    let running_mean: Vec<f32> = (0..channels).map(|c| c as f32 * 0.1).collect();
    let running_var: Vec<f32> = (0..channels).map(|c| 1.0 + c as f32 * 0.05).collect();
    let weight: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..channels).map(|c| c as f32 * 0.01).collect();

    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        channels,
        spatial,
        1e-5,
    );

    // Verify a few elements manually
    // Element 0: batch=0, channel=0, spatial=0
    // x=0.0, mean=0.0, var=1.0, weight=0.5, bias=0.0
    // y = 0.5 * (0 - 0) / sqrt(1 + 1e-5) + 0 ~= 0
    assert!(output[0].abs() < 1e-3);

    // All outputs should be finite
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_reference_batch1_channels32_spatial1000() {
    let channels = 32;
    let spatial = 1000;
    let batch = 1;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.001).sin()).collect();
    let mut output = vec![0.0; total];

    let running_mean = vec![0.0; channels];
    let running_var = vec![1.0; channels];
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];

    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        channels,
        spatial,
        1e-5,
    );

    // With mean=0, var=1, weight=1, bias=0: output ~= input
    for i in 0..total {
        assert!(
            (output[i] - input[i]).abs() < 1e-3,
            "element {i}: got {}, expected {}",
            output[i],
            input[i]
        );
    }
}

#[test]
fn test_reference_batch4_channels64_spatial100() {
    let channels = 64;
    let spatial = 100;
    let batch = 4;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
    let mut output = vec![0.0; total];

    let running_mean = vec![0.5; channels];
    let running_var = vec![2.0; channels];
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];

    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        channels,
        spatial,
        1e-5,
    );

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}

// =========================================================================
// Convenience wrappers
// =========================================================================

#[test]
fn test_emit_ptx_batchnorm_default() {
    let ptx = emit_ptx_batchnorm_default("bn_default", 64, 1e-5).unwrap();
    assert!(ptx.contains(".entry bn_default"));
}

#[test]
fn test_generate_batchnorm_ptx() {
    let ptx = generate_batchnorm_ptx(64);
    assert!(ptx.contains(".entry ptx_batchnorm_f32"));
    assert!(ptx.contains("num_channels=64"));
}

// =========================================================================
// Config Clone and Debug
// =========================================================================

#[test]
fn test_config_clone() {
    let c = PtxBatchNormConfig::new("bn", 64, 1e-5);
    let c2 = c.clone();
    assert_eq!(c.num_channels, c2.num_channels);
    assert_eq!(c.kernel_name, c2.kernel_name);
    assert_eq!(c.eps, c2.eps);
}

#[test]
fn test_config_debug() {
    let c = PtxBatchNormConfig::new("bn", 64, 1e-5);
    let debug = format!("{c:?}");
    assert!(debug.contains("PtxBatchNormConfig"));
    assert!(debug.contains("64"));
}

// =========================================================================
// CNN-typical dimensions
// =========================================================================

#[test]
fn test_resnet_channels_64() {
    let c = PtxBatchNormConfig::new("resnet_bn", 64, 1e-5);
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains("num_channels=64"));
}

#[test]
fn test_resnet_channels_256() {
    let c = PtxBatchNormConfig::new("resnet_bn256", 256, 1e-5);
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains("num_channels=256"));
}

#[test]
fn test_resnet_channels_512() {
    let c = PtxBatchNormConfig::new("resnet_bn512", 512, 1e-5);
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains("num_channels=512"));
}

#[test]
fn test_efficientnet_channels_1280() {
    let c = PtxBatchNormConfig::new("eff_bn", 1280, 1e-3);
    let ptx = emit_ptx_batchnorm(&c).unwrap();
    assert!(ptx.contains("num_channels=1280"));
}
