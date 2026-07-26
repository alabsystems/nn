// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX GroupNorm kernel generation.
//!
//! Covers config validation, PTX structural checks, reference computation
//! verification, edge cases, and model-typical dimensions.

use super::*;

// =========================================================================
// Config construction and validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxGroupNormConfig::new("groupnorm_32_256", 32, 256, 1e-5);
    assert_eq!(c.num_groups, 32);
    assert_eq!(c.num_channels, 256);
    assert_eq!(c.kernel_name, "groupnorm_32_256");
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_groups_zero_rejected() {
    let c = PtxGroupNormConfig::new("gn", 0, 64, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_channels_zero_rejected() {
    let c = PtxGroupNormConfig::new("gn", 4, 0, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_channels_not_divisible_rejected() {
    let c = PtxGroupNormConfig::new("gn", 3, 64, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_channels_equals_groups_valid() {
    // Each group has 1 channel -> instance norm equivalent
    let c = PtxGroupNormConfig::new("gn", 64, 64, 1e-5);
    assert!(c.validate().is_ok());
    assert_eq!(c.channels_per_group(), 1);
}

#[test]
fn test_config_one_group_valid() {
    // One group = all channels -> layer norm equivalent
    let c = PtxGroupNormConfig::new("gn", 1, 64, 1e-5);
    assert!(c.validate().is_ok());
    assert_eq!(c.channels_per_group(), 64);
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxGroupNormConfig::new("", 4, 64, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_nan_eps_rejected() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, f32::NAN);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_negative_eps_rejected() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, -0.001);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_inf_eps_rejected() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, f32::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_eps_valid() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, 0.0);
    assert!(c.validate().is_ok());
}

// =========================================================================
// Block size and warp configuration
// =========================================================================

#[test]
fn test_channels_per_group() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, 1e-5);
    assert_eq!(c.channels_per_group(), 16);
}

#[test]
fn test_block_size_small_group() {
    // 64 channels / 64 groups = 1 channel per group -> block = 32 (1 warp)
    let c = PtxGroupNormConfig::new("gn", 64, 64, 1e-5);
    assert_eq!(c.channels_per_group(), 1);
    assert_eq!(c.block_size(), 32);
    assert!(c.is_warp_only());
    assert_eq!(c.shared_memory_bytes(), 0);
}

#[test]
fn test_block_size_medium_group() {
    // 256 channels / 4 groups = 64 channels per group -> block = 64 (2 warps)
    let c = PtxGroupNormConfig::new("gn", 4, 256, 1e-5);
    assert_eq!(c.channels_per_group(), 64);
    assert_eq!(c.block_size(), 64);
    assert_eq!(c.num_warps(), 2);
    assert!(!c.is_warp_only());
}

#[test]
fn test_block_size_large_group() {
    // 256 channels / 1 group = 256 -> block = 256 (8 warps)
    let c = PtxGroupNormConfig::new("gn", 1, 256, 1e-5);
    assert_eq!(c.channels_per_group(), 256);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
}

#[test]
fn test_block_size_capped_at_256() {
    // 1024 channels / 1 group = 1024 -> capped at 256
    let c = PtxGroupNormConfig::new("gn", 1, 1024, 1e-5);
    assert_eq!(c.block_size(), 256);
}

// =========================================================================
// SM target configuration
// =========================================================================

#[test]
fn test_sm_target_custom() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, 1e-5).with_sm_target("sm_70");
    assert_eq!(c.sm_target, "sm_70");
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_sm_90_hopper() {
    let c = PtxGroupNormConfig::new("gn_hopper", 32, 256, 1e-5).with_sm_target("sm_90");
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

// =========================================================================
// PTX structural validation
// =========================================================================

#[test]
fn test_ptx_contains_version_and_target() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_ptx_contains_entry_point() {
    let c = PtxGroupNormConfig::new("nn_groupnorm", 4, 64, 1e-5);
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains(".visible .entry nn_groupnorm"));
}

#[test]
fn test_ptx_contains_kernel_params() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_weight"));
    assert!(ptx.contains("param_bias"));
    assert!(ptx.contains("param_group_size"));
    assert!(ptx.contains("param_spatial_size"));
    assert!(ptx.contains("param_group_elems"));
}

#[test]
fn test_ptx_has_rsqrt() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(
        ptx.contains("rsqrt.approx.f32"),
        "GroupNorm must use rsqrt for 1/sqrt(var + eps)"
    );
}

#[test]
fn test_ptx_has_fma_for_affine() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(
        ptx.contains("fma.rn.f32"),
        "GroupNorm must use fma for weight*norm+bias"
    );
}

#[test]
fn test_ptx_has_mean_subtraction() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(
        ptx.contains("sub.f32"),
        "GroupNorm must subtract group mean"
    );
}

#[test]
fn test_ptx_has_warp_shuffle() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_ptx_comment_header() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains("GroupNorm f32"));
}

#[test]
fn test_ptx_loads_and_stores_global() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_ptx_has_phase_labels() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains("Phase 1: compute group mean"));
    assert!(ptx.contains("Phase 2: compute group variance"));
    assert!(ptx.contains("Phase 3: normalize + affine"));
}

#[test]
fn test_ptx_has_loop_labels() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains("GN_MEAN_LOOP"));
    assert!(ptx.contains("GN_MEAN_REDUCE"));
    assert!(ptx.contains("GN_VAR_LOOP"));
    assert!(ptx.contains("GN_VAR_REDUCE"));
    assert!(ptx.contains("GN_NORM_LOOP"));
}

#[test]
fn test_ptx_ends_with_closing_brace() {
    let ptx = generate_groupnorm_ptx(4, 64);
    let trimmed = ptx.trim_end();
    assert!(trimmed.ends_with('}'));
}

#[test]
fn test_ptx_not_cuda_cpp() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("__shared__"));
}

#[test]
fn test_ptx_reasonable_size() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(
        ptx.len() > 500,
        "PTX should be substantial, got {} bytes",
        ptx.len()
    );
    assert!(ptx.len() < 50_000, "PTX too large: {} bytes", ptx.len());
}

// =========================================================================
// Warp-only vs multi-warp reduction structure
// =========================================================================

#[test]
fn test_warp_only_small_group() {
    // 16 channels / 16 groups = 1 cpg -> warp-only
    let c = PtxGroupNormConfig::new("gn_16", 16, 16, 1e-5);
    assert!(c.is_warp_only());
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(!ptx.contains("warp_scratch"));
    assert!(!ptx.contains(".shared"));
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_multi_warp_uses_shared_memory() {
    // 256 channels / 4 groups = 64 cpg -> 2 warps -> shared memory
    let c = PtxGroupNormConfig::new("gn_multi", 4, 256, 1e-5);
    assert!(!c.is_warp_only());
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains("warp_scratch"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_multi_warp_cross_warp_labels() {
    let c = PtxGroupNormConfig::new("gn_multi", 4, 256, 1e-5);
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains("CROSS_GN_MEAN"));
    assert!(ptx.contains("CROSS_GN_VAR"));
}

// =========================================================================
// Different parameters produce different PTX
// =========================================================================

#[test]
fn test_different_groups_produce_different_ptx() {
    let ptx_4 = generate_groupnorm_ptx(4, 64);
    let ptx_16 = generate_groupnorm_ptx(16, 64);
    assert_ne!(ptx_4, ptx_16);
}

#[test]
fn test_different_channels_produce_different_ptx() {
    let ptx_64 = generate_groupnorm_ptx(4, 64);
    let ptx_256 = generate_groupnorm_ptx(4, 256);
    assert_ne!(ptx_64, ptx_256);
}

#[test]
fn test_different_eps_produce_different_ptx() {
    let ptx_1e5 = emit_ptx_groupnorm_default("gn", 4, 64, 1e-5).unwrap();
    let ptx_1e6 = emit_ptx_groupnorm_default("gn", 4, 64, 1e-6).unwrap();
    assert_ne!(ptx_1e5, ptx_1e6);
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_basic() {
    let (grid, block) = ptx_groupnorm_launch_config(4, 4, 64);
    // 4 samples * 4 groups = 16 blocks
    assert_eq!(grid, [16, 1, 1]);
    // 64/4 = 16 cpg -> block = 32
    assert_eq!(block, [32, 1, 1]);
}

#[test]
fn test_launch_config_large_batch() {
    let (grid, block) = ptx_groupnorm_launch_config(32, 32, 256);
    // 32 samples * 32 groups = 1024 blocks
    assert_eq!(grid, [1024, 1, 1]);
    // 256/32 = 8 cpg -> block = 32
    assert_eq!(block, [32, 1, 1]);
}

#[test]
fn test_launch_config_single_group() {
    let (grid, block) = ptx_groupnorm_launch_config(1, 1, 256);
    // 1 sample * 1 group = 1 block
    assert_eq!(grid, [1, 1, 1]);
    // 256/1 = 256 cpg -> block = 256
    assert_eq!(block, [256, 1, 1]);
}

// =========================================================================
// Reference computation: known values
// =========================================================================

#[test]
fn test_reference_identity_transform() {
    // 1 group, 2 channels, spatial=2: groups the whole tensor
    // All weight=1, bias=0, constant input -> normalized to 0
    let input = vec![5.0, 5.0, 5.0, 5.0]; // batch=1, channels=2, spatial=2
    let mut output = vec![0.0; 4];
    let weight = vec![1.0, 1.0];
    let bias = vec![2.0, 2.0];
    groupnorm_reference(&input, &mut output, &weight, &bias, 1, 2, 2, 1e-5);
    // All same value: mean=5, var=0, output = bias
    for &v in &output {
        assert!(
            (v - 2.0).abs() < 1e-3,
            "constant input with weight=1 should yield bias, got {v}"
        );
    }
}

#[test]
fn test_reference_zero_mean_output() {
    // With weight=1, bias=0, output should have near-zero mean per group
    let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, channels=2, spatial=2, 1 group
    let mut output = vec![0.0; 4];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    groupnorm_reference(&input, &mut output, &weight, &bias, 1, 2, 2, 1e-5);

    let out_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        out_mean.abs() < 1e-5,
        "normalized group output should have ~zero mean, got {out_mean}"
    );
}

#[test]
fn test_reference_per_group_normalization() {
    // 2 groups, 4 channels, spatial=1: groups = [c0,c1] and [c2,c3]
    // Group 0: [1.0, 3.0], Group 1: [10.0, 20.0]
    let input = vec![1.0, 3.0, 10.0, 20.0]; // batch=1, channels=4, spatial=1
    let mut output = vec![0.0; 4];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    groupnorm_reference(&input, &mut output, &weight, &bias, 2, 4, 1, 0.0);

    // Group 0: mean=2.0, var=1.0, inv_std=1.0
    // c0: (1-2)*1 = -1, c1: (3-2)*1 = 1
    assert!((output[0] - (-1.0)).abs() < 1e-5, "c0: got {}", output[0]);
    assert!((output[1] - 1.0).abs() < 1e-5, "c1: got {}", output[1]);

    // Group 1: mean=15.0, var=25.0, inv_std=1/5=0.2
    // c2: (10-15)*0.2 = -1, c3: (20-15)*0.2 = 1
    assert!((output[2] - (-1.0)).abs() < 1e-5, "c2: got {}", output[2]);
    assert!((output[3] - 1.0).abs() < 1e-5, "c3: got {}", output[3]);
}

#[test]
fn test_reference_weight_scaling() {
    // Group 0: [1.0, 3.0] -> normalized to [-1, 1]
    // With weight=2: output = [-2, 2]
    let input = vec![1.0, 3.0];
    let mut output = vec![0.0; 2];
    let weight = vec![2.0, 2.0];
    let bias = vec![0.0, 0.0];
    groupnorm_reference(&input, &mut output, &weight, &bias, 1, 2, 1, 0.0);

    assert!((output[0] - (-2.0)).abs() < 1e-5, "got {}", output[0]);
    assert!((output[1] - 2.0).abs() < 1e-5, "got {}", output[1]);
}

#[test]
fn test_reference_bias_shift() {
    let input = vec![1.0, 3.0];
    let mut output = vec![0.0; 2];
    let weight = vec![1.0, 1.0];
    let bias = vec![5.0, 5.0];
    groupnorm_reference(&input, &mut output, &weight, &bias, 1, 2, 1, 0.0);

    // normalized = [-1, 1], + bias 5 = [4, 6]
    assert!((output[0] - 4.0).abs() < 1e-5, "got {}", output[0]);
    assert!((output[1] - 6.0).abs() < 1e-5, "got {}", output[1]);
}

// =========================================================================
// Reference computation: edge cases
// =========================================================================

#[test]
fn test_reference_zero_input() {
    let input = vec![0.0f32; 8]; // batch=1, channels=4, spatial=2
    let mut output = vec![0.0; 8];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    groupnorm_reference(&input, &mut output, &weight, &bias, 2, 4, 2, 1e-5);
    for &v in &output {
        assert!(v.abs() < 1e-3, "zero input should yield ~0, got {v}");
    }
}

#[test]
fn test_reference_single_channel_per_group() {
    // groups = channels -> instance norm behavior
    let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, channels=2, spatial=2
    let mut output = vec![0.0; 4];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    groupnorm_reference(&input, &mut output, &weight, &bias, 2, 2, 2, 1e-5);

    // Group/channel 0: [1, 2], mean=1.5, var=0.25, inv_std=2
    // (1-1.5)*2 = -1, (2-1.5)*2 = 1
    assert!((output[0] - (-1.0)).abs() < 1e-4, "got {}", output[0]);
    assert!((output[1] - 1.0).abs() < 1e-4, "got {}", output[1]);

    // Group/channel 1: [3, 4], mean=3.5, var=0.25, inv_std=2
    assert!((output[2] - (-1.0)).abs() < 1e-4, "got {}", output[2]);
    assert!((output[3] - 1.0).abs() < 1e-4, "got {}", output[3]);
}

#[test]
fn test_reference_all_channels_one_group() {
    // 1 group -> all channels normalized together (layer norm-like)
    let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, channels=4, spatial=1
    let mut output = vec![0.0; 4];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    groupnorm_reference(&input, &mut output, &weight, &bias, 1, 4, 1, 1e-5);

    let out_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        out_mean.abs() < 1e-5,
        "should have ~zero mean, got {out_mean}"
    );
}

#[test]
fn test_reference_multi_batch() {
    // batch=2, channels=2, spatial=1, groups=1
    let input = vec![1.0, 3.0, 10.0, 20.0]; // sample0=[1,3], sample1=[10,20]
    let mut output = vec![0.0; 4];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    groupnorm_reference(&input, &mut output, &weight, &bias, 1, 2, 1, 0.0);

    // Sample 0: mean=2, var=1, inv_std=1 -> [-1, 1]
    assert!((output[0] - (-1.0)).abs() < 1e-5);
    assert!((output[1] - 1.0).abs() < 1e-5);

    // Sample 1: mean=15, var=25, inv_std=0.2 -> [-1, 1]
    assert!((output[2] - (-1.0)).abs() < 1e-5);
    assert!((output[3] - 1.0).abs() < 1e-5);
}

// =========================================================================
// Reference: various shapes
// =========================================================================

#[test]
fn test_reference_batch1_groups4_channels16_spatial100() {
    let groups = 4;
    let channels = 16;
    let spatial = 100;
    let batch = 1;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let mut output = vec![0.0; total];
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];

    groupnorm_reference(
        &input,
        &mut output,
        &weight,
        &bias,
        groups,
        channels,
        spatial,
        1e-5,
    );

    // All outputs should be finite
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }

    // Each group should have near-zero mean
    let cpg = channels / groups;
    let group_elems = cpg * spatial;
    for g in 0..groups {
        let base = g * group_elems;
        let group_mean: f32 =
            output[base..base + group_elems].iter().sum::<f32>() / group_elems as f32;
        assert!(group_mean.abs() < 0.01, "group {g} mean = {group_mean}");
    }
}

#[test]
fn test_reference_batch4_groups16_channels64_spatial100() {
    let groups = 16;
    let channels = 64;
    let spatial = 100;
    let batch = 4;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.001).sin()).collect();
    let mut output = vec![0.0; total];
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];

    groupnorm_reference(
        &input,
        &mut output,
        &weight,
        &bias,
        groups,
        channels,
        spatial,
        1e-5,
    );

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_reference_batch4_groups4_channels32_spatial1000() {
    let groups = 4;
    let channels = 32;
    let spatial = 1000;
    let batch = 4;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.0001).collect();
    let mut output = vec![0.0; total];
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];

    groupnorm_reference(
        &input,
        &mut output,
        &weight,
        &bias,
        groups,
        channels,
        spatial,
        1e-5,
    );

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_reference_groups_equal_channels() {
    // Instance norm: each channel is its own group
    let groups = 16;
    let channels = 16;
    let spatial = 10;
    let batch = 1;
    let total = batch * channels * spatial;

    let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let mut output = vec![0.0; total];
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];

    groupnorm_reference(
        &input,
        &mut output,
        &weight,
        &bias,
        groups,
        channels,
        spatial,
        1e-5,
    );

    // Each channel independently normalized
    for c in 0..channels {
        let base = c * spatial;
        let mean: f32 = output[base..base + spatial].iter().sum::<f32>() / spatial as f32;
        assert!(mean.abs() < 0.01, "channel {c} mean = {mean}");
    }
}

// =========================================================================
// Convenience wrappers
// =========================================================================

#[test]
fn test_emit_ptx_groupnorm_default() {
    let ptx = emit_ptx_groupnorm_default("gn_default", 4, 64, 1e-5).unwrap();
    assert!(ptx.contains(".entry gn_default"));
}

#[test]
fn test_generate_groupnorm_ptx() {
    let ptx = generate_groupnorm_ptx(4, 64);
    assert!(ptx.contains(".entry ptx_groupnorm_f32"));
    assert!(ptx.contains("num_groups=4"));
    assert!(ptx.contains("num_channels=64"));
}

// =========================================================================
// Config Clone and Debug
// =========================================================================

#[test]
fn test_config_clone() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, 1e-5);
    let c2 = c.clone();
    assert_eq!(c.num_groups, c2.num_groups);
    assert_eq!(c.num_channels, c2.num_channels);
    assert_eq!(c.kernel_name, c2.kernel_name);
    assert_eq!(c.eps, c2.eps);
}

#[test]
fn test_config_debug() {
    let c = PtxGroupNormConfig::new("gn", 4, 64, 1e-5);
    let debug = format!("{c:?}");
    assert!(debug.contains("PtxGroupNormConfig"));
    assert!(debug.contains("64"));
}

// =========================================================================
// Model-typical dimensions
// =========================================================================

#[test]
fn test_detr_groups32_channels256() {
    let c = PtxGroupNormConfig::new("detr_gn", 32, 256, 1e-5);
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains("num_groups=32"));
    assert!(ptx.contains("num_channels=256"));
}

#[test]
fn test_stable_diffusion_groups32_channels320() {
    let c = PtxGroupNormConfig::new("sd_gn", 32, 320, 1e-6);
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains("num_groups=32"));
    assert!(ptx.contains("num_channels=320"));
}

#[test]
fn test_stable_diffusion_groups32_channels1280() {
    let c = PtxGroupNormConfig::new("sd_gn_1280", 32, 1280, 1e-6);
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains("num_groups=32"));
    assert!(ptx.contains("num_channels=1280"));
}

#[test]
fn test_maskrcnn_groups32_channels256() {
    let c = PtxGroupNormConfig::new("maskrcnn_gn", 32, 256, 1e-5);
    let ptx = emit_ptx_groupnorm(&c).unwrap();
    assert!(ptx.contains("num_groups=32"));
}
