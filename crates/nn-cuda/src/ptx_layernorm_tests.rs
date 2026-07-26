// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX LayerNorm kernel generation.
//!
//! Covers config validation, PTX structural checks, reference computation
//! verification, edge cases, and transformer-typical dimensions.

use super::*;

// =========================================================================
// Config construction and validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxLayerNormConfig::new("layernorm_768", 768, 1e-5);
    assert_eq!(c.normalized_shape, 768);
    assert_eq!(c.kernel_name, "layernorm_768");
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_dim_zero_rejected() {
    let c = PtxLayerNormConfig::new("ln", 0, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxLayerNormConfig::new("", 768, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_nan_eps_rejected() {
    let c = PtxLayerNormConfig::new("ln", 768, f32::NAN);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_negative_eps_rejected() {
    let c = PtxLayerNormConfig::new("ln", 768, -0.001);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_inf_eps_rejected() {
    let c = PtxLayerNormConfig::new("ln", 768, f32::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_eps_valid() {
    let c = PtxLayerNormConfig::new("ln", 768, 0.0);
    assert!(c.validate().is_ok());
}

// =========================================================================
// Block size and warp configuration
// =========================================================================

#[test]
fn test_block_size_small_dim() {
    let c = PtxLayerNormConfig::new("ln", 16, 1e-5);
    assert_eq!(c.block_size(), 32);
    assert_eq!(c.num_warps(), 1);
    assert!(c.is_warp_only());
    assert_eq!(c.shared_memory_bytes(), 0);
}

#[test]
fn test_block_size_warp_boundary() {
    let c = PtxLayerNormConfig::new("ln", 32, 1e-5);
    assert_eq!(c.block_size(), 32);
    assert!(c.is_warp_only());
}

#[test]
fn test_block_size_multi_warp() {
    let c = PtxLayerNormConfig::new("ln", 64, 1e-5);
    assert_eq!(c.block_size(), 64);
    assert_eq!(c.num_warps(), 2);
    assert!(!c.is_warp_only());
    assert_eq!(c.shared_memory_bytes(), 8);
}

#[test]
fn test_block_size_capped_at_256() {
    let c = PtxLayerNormConfig::new("ln", 1024, 1e-5);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
    assert_eq!(c.shared_memory_bytes(), 32);
}

#[test]
fn test_block_size_dim_1() {
    let c = PtxLayerNormConfig::new("ln", 1, 1e-5);
    assert_eq!(c.block_size(), 32);
    assert!(c.is_warp_only());
}

// =========================================================================
// SM target configuration
// =========================================================================

#[test]
fn test_sm_target_custom() {
    let c = PtxLayerNormConfig::new("ln", 768, 1e-5).with_sm_target("sm_70");
    assert_eq!(c.sm_target, "sm_70");
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_sm_90_hopper() {
    let c = PtxLayerNormConfig::new("ln_hopper", 4096, 1e-5).with_sm_target("sm_90");
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

// =========================================================================
// PTX structural validation
// =========================================================================

#[test]
fn test_ptx_contains_version_and_target() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_ptx_contains_entry_point() {
    let c = PtxLayerNormConfig::new("nn_layernorm", 256, 1e-5);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains(".visible .entry nn_layernorm"));
}

#[test]
fn test_ptx_contains_kernel_params() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_gamma"));
    assert!(ptx.contains("param_beta"));
    assert!(ptx.contains("param_row_size"));
    assert!(ptx.contains("param_num_rows"));
}

#[test]
fn test_ptx_has_rsqrt() {
    let ptx = generate_layernorm_ptx(768);
    assert!(
        ptx.contains("rsqrt.approx.f32"),
        "LayerNorm must use rsqrt for 1/sqrt(var + eps)"
    );
}

#[test]
fn test_ptx_has_fma_for_affine() {
    let ptx = generate_layernorm_ptx(768);
    assert!(
        ptx.contains("fma.rn.f32"),
        "LayerNorm must use fma for gamma*norm+beta"
    );
}

#[test]
fn test_ptx_has_mean_subtraction() {
    let ptx = generate_layernorm_ptx(768);
    assert!(
        ptx.contains("sub.f32"),
        "LayerNorm must subtract mean (sub.f32)"
    );
}

#[test]
fn test_ptx_has_warp_shuffle() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_ptx_comment_header() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains("LayerNorm f32"));
}

#[test]
fn test_ptx_register_declarations() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains(".reg .u32  %r<20>"));
    assert!(ptx.contains(".reg .f32  %f<20>"));
    assert!(ptx.contains(".reg .u64  %rd<14>"));
    assert!(ptx.contains(".reg .pred %p<6>"));
}

#[test]
fn test_ptx_reqntid_matches_block_size() {
    let c = PtxLayerNormConfig::new("ln", 768, 1e-5);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains(".reqntid 256"));

    let c2 = PtxLayerNormConfig::new("ln", 16, 1e-5);
    let ptx2 = emit_ptx_layernorm(&c2).unwrap();
    assert!(ptx2.contains(".reqntid 32"));
}

#[test]
fn test_ptx_loads_and_stores_global() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_ptx_has_phase_labels() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains("Phase 1: compute mean"));
    assert!(ptx.contains("Phase 2: compute variance"));
    assert!(ptx.contains("Phase 3: normalize + affine"));
}

#[test]
fn test_ptx_has_loop_labels() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains("LN_MEAN_LOOP"));
    assert!(ptx.contains("LN_MEAN_REDUCE"));
    assert!(ptx.contains("LN_VAR_LOOP"));
    assert!(ptx.contains("LN_VAR_REDUCE"));
    assert!(ptx.contains("LN_NORM_LOOP"));
}

#[test]
fn test_ptx_ends_with_closing_brace() {
    let ptx = generate_layernorm_ptx(768);
    let trimmed = ptx.trim_end();
    assert!(trimmed.ends_with('}'));
}

#[test]
fn test_ptx_not_cuda_cpp() {
    let ptx = generate_layernorm_ptx(768);
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("__shared__"));
}

#[test]
fn test_ptx_reasonable_size() {
    let ptx = generate_layernorm_ptx(768);
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
fn test_warp_only_no_shared_memory() {
    let c = PtxLayerNormConfig::new("ln_16", 16, 1e-5);
    assert!(c.is_warp_only());
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(!ptx.contains("warp_scratch"));
    assert!(!ptx.contains(".shared"));
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_warp_only_no_barrier() {
    let c = PtxLayerNormConfig::new("ln_32", 32, 1e-5);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(!ptx.contains("bar.sync"), "warp-only must not use barrier");
}

#[test]
fn test_multi_warp_uses_shared_memory() {
    let c = PtxLayerNormConfig::new("ln_128", 128, 1e-5);
    assert!(!c.is_warp_only());
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains("warp_scratch"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_multi_warp_cross_warp_labels() {
    let c = PtxLayerNormConfig::new("ln_128", 128, 1e-5);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains("CROSS_MEAN"));
    assert!(ptx.contains("CROSS_VAR"));
}

#[test]
fn test_multi_warp_shared_scratch_size() {
    let c = PtxLayerNormConfig::new("ln_128", 128, 1e-5);
    assert_eq!(c.num_warps(), 4);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains("warp_scratch[4]"));

    let c2 = PtxLayerNormConfig::new("ln_4096", 4096, 1e-5);
    assert_eq!(c2.num_warps(), 8);
    let ptx2 = emit_ptx_layernorm(&c2).unwrap();
    assert!(ptx2.contains("warp_scratch[8]"));
}

// =========================================================================
// Different parameters produce different PTX
// =========================================================================

#[test]
fn test_different_dims_produce_different_ptx() {
    let ptx_32 = generate_layernorm_ptx(32);
    let ptx_128 = generate_layernorm_ptx(128);
    let ptx_768 = generate_layernorm_ptx(768);
    assert_ne!(ptx_32, ptx_128);
    assert_ne!(ptx_128, ptx_768);
}

#[test]
fn test_different_eps_produce_different_ptx() {
    let ptx_1e5 = emit_ptx_layernorm_default("ln", 768, 1e-5).unwrap();
    let ptx_1e6 = emit_ptx_layernorm_default("ln", 768, 1e-6).unwrap();
    assert_ne!(ptx_1e5, ptx_1e6);
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_single_row() {
    let (grid, block) = ptx_layernorm_launch_config(1, 768);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_many_rows() {
    let (grid, block) = ptx_layernorm_launch_config(100_000, 4096);
    assert_eq!(grid, [100_000, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_small_dim() {
    let (grid, block) = ptx_layernorm_launch_config(10, 16);
    assert_eq!(grid, [10, 1, 1]);
    assert_eq!(block, [32, 1, 1]);
}

// =========================================================================
// Reference computation: known values
// =========================================================================

#[test]
fn test_reference_identity_transform() {
    // gamma=1, beta=0 -> pure normalization
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5f32;
    let output = layernorm_reference(&input, &gamma, &beta, eps);

    // mean = 2.5, var = 1.25
    let mean = 2.5f32;
    let var = 1.25f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| (x - mean) * inv_std).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "mismatch: got {o}, expected {e}");
    }
}

#[test]
fn test_reference_zero_mean_output() {
    // With gamma=1, beta=0, output should have near-zero mean
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0; 5];
    let beta = vec![0.0; 5];
    let output = layernorm_reference(&input, &gamma, &beta, 1e-5);

    let out_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        out_mean.abs() < 1e-5,
        "normalized output should have ~zero mean, got {out_mean}"
    );
}

#[test]
fn test_reference_unit_variance_output() {
    // With gamma=1, beta=0, output should have near-unit variance
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0; 5];
    let beta = vec![0.0; 5];
    let output = layernorm_reference(&input, &gamma, &beta, 0.0);

    let out_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    let out_var: f32 = output
        .iter()
        .map(|&x| (x - out_mean) * (x - out_mean))
        .sum::<f32>()
        / output.len() as f32;
    assert!(
        (out_var - 1.0).abs() < 1e-4,
        "normalized output should have ~unit variance, got {out_var}"
    );
}

#[test]
fn test_reference_gamma_scaling() {
    // gamma=2, beta=0 -> output is 2x normalized
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0; 4];
    let beta = vec![0.0; 4];
    let gamma_one = vec![1.0; 4];

    let out_scaled = layernorm_reference(&input, &gamma, &beta, 1e-5);
    let out_normal = layernorm_reference(&input, &gamma_one, &beta, 1e-5);

    for (s, n) in out_scaled.iter().zip(out_normal.iter()) {
        assert!(
            (s - 2.0 * n).abs() < 1e-5,
            "gamma=2 should double: got {s}, expected {}",
            2.0 * n
        );
    }
}

#[test]
fn test_reference_beta_shift() {
    // gamma=1, beta=3 -> output is normalized + 3
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta_zero = vec![0.0; 4];
    let beta_three = vec![3.0; 4];

    let out_base = layernorm_reference(&input, &gamma, &beta_zero, 1e-5);
    let out_shifted = layernorm_reference(&input, &gamma, &beta_three, 1e-5);

    for (b, s) in out_base.iter().zip(out_shifted.iter()) {
        assert!(
            (s - (b + 3.0)).abs() < 1e-5,
            "beta=3 should shift by 3: got {s}, expected {}",
            b + 3.0
        );
    }
}

#[test]
fn test_reference_constant_input() {
    // All same values: var=0, output = gamma*0 + beta = beta (with eps)
    let input = vec![5.0f32; 4];
    let gamma = vec![1.0; 4];
    let beta = vec![2.0; 4];
    let output = layernorm_reference(&input, &gamma, &beta, 1e-5);

    // mean=5, var=0, inv_std = 1/sqrt(eps), (x-mean)=0, so output = beta
    for &v in &output {
        assert!(
            (v - 2.0).abs() < 1e-3,
            "constant input should yield beta, got {v}"
        );
    }
}

// =========================================================================
// Reference computation: edge cases
// =========================================================================

#[test]
fn test_reference_zero_input() {
    let input = vec![0.0f32; 4];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let output = layernorm_reference(&input, &gamma, &beta, 1e-5);
    // All zero, mean=0, var=0, output = 0
    for &v in &output {
        assert!(v.abs() < 1e-3, "zero input should yield ~0, got {v}");
    }
}

#[test]
fn test_reference_single_element() {
    // dim=1: var=0, output = beta
    let input = vec![3.0f32];
    let gamma = vec![1.0];
    let beta = vec![0.5];
    let output = layernorm_reference(&input, &gamma, &beta, 1e-5);
    // mean=3, var=0, (x-mean)=0, output = 0*gamma + beta = 0.5
    assert!(
        (output[0] - 0.5).abs() < 1e-3,
        "single element: expected 0.5, got {}",
        output[0]
    );
}

#[test]
fn test_reference_negative_input() {
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;
    let output = layernorm_reference(&input, &gamma, &beta, eps);

    let mean = -2.5f32;
    let var = 1.25f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| (x - mean) * inv_std).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "mismatch: got {o}, expected {e}");
    }
}

// =========================================================================
// Convenience wrappers
// =========================================================================

#[test]
fn test_emit_ptx_layernorm_default() {
    let ptx = emit_ptx_layernorm_default("ln_default", 768, 1e-5).unwrap();
    assert!(ptx.contains(".entry ln_default"));
}

#[test]
fn test_generate_layernorm_ptx() {
    let ptx = generate_layernorm_ptx(768);
    assert!(ptx.contains(".entry ptx_layernorm_f32"));
    assert!(ptx.contains("normalized_shape=768"));
}

// =========================================================================
// Config Clone and Debug
// =========================================================================

#[test]
fn test_config_clone() {
    let c = PtxLayerNormConfig::new("ln", 768, 1e-5);
    let c2 = c.clone();
    assert_eq!(c.normalized_shape, c2.normalized_shape);
    assert_eq!(c.kernel_name, c2.kernel_name);
    assert_eq!(c.eps, c2.eps);
}

#[test]
fn test_config_debug() {
    let c = PtxLayerNormConfig::new("ln", 768, 1e-5);
    let debug = format!("{c:?}");
    assert!(debug.contains("PtxLayerNormConfig"));
    assert!(debug.contains("768"));
}

// =========================================================================
// Transformer-typical dimensions
// =========================================================================

#[test]
fn test_bert_dim_768() {
    let c = PtxLayerNormConfig::new("bert_ln", 768, 1e-12);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains("normalized_shape=768"));
}

#[test]
fn test_gpt2_dim_1024() {
    let c = PtxLayerNormConfig::new("gpt2_ln", 1024, 1e-5);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains("normalized_shape=1024"));
}

#[test]
fn test_whisper_dim_512() {
    let c = PtxLayerNormConfig::new("whisper_ln", 512, 1e-5);
    let ptx = emit_ptx_layernorm(&c).unwrap();
    assert!(ptx.contains("normalized_shape=512"));
}
