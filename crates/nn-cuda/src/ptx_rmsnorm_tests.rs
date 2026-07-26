// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX RMSNorm kernel generation.
//!
//! Covers config validation, PTX structural checks, reference computation
//! verification, edge cases, and LLM-typical dimensions.

use super::*;

// =========================================================================
// Config construction and validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxRmsNormConfig::new("rmsnorm_4096", 4096, 1e-5);
    assert_eq!(c.hidden_dim, 4096);
    assert_eq!(c.kernel_name, "rmsnorm_4096");
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_dim_zero_rejected() {
    let c = PtxRmsNormConfig::new("rms", 0, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxRmsNormConfig::new("", 768, 1e-5);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_nan_eps_rejected() {
    let c = PtxRmsNormConfig::new("rms", 768, f32::NAN);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_negative_eps_rejected() {
    let c = PtxRmsNormConfig::new("rms", 768, -0.001);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_inf_eps_rejected() {
    let c = PtxRmsNormConfig::new("rms", 768, f32::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_neg_inf_eps_rejected() {
    let c = PtxRmsNormConfig::new("rms", 768, f32::NEG_INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_eps_valid() {
    let c = PtxRmsNormConfig::new("rms", 768, 0.0);
    assert!(c.validate().is_ok());
}

// =========================================================================
// Block size and warp configuration
// =========================================================================

#[test]
fn test_block_size_small_dim() {
    // dim=16 -> rounds up to 32 (one warp)
    let c = PtxRmsNormConfig::new("rms", 16, 1e-5);
    assert_eq!(c.block_size(), 32);
    assert_eq!(c.num_warps(), 1);
    assert!(c.is_warp_only());
    assert_eq!(c.shared_memory_bytes(), 0);
}

#[test]
fn test_block_size_warp_boundary() {
    let c = PtxRmsNormConfig::new("rms", 32, 1e-5);
    assert_eq!(c.block_size(), 32);
    assert!(c.is_warp_only());
}

#[test]
fn test_block_size_multi_warp() {
    let c = PtxRmsNormConfig::new("rms", 64, 1e-5);
    assert_eq!(c.block_size(), 64);
    assert_eq!(c.num_warps(), 2);
    assert!(!c.is_warp_only());
    assert_eq!(c.shared_memory_bytes(), 8); // 2 warps * 4 bytes
}

#[test]
fn test_block_size_capped_at_256() {
    let c = PtxRmsNormConfig::new("rms", 1024, 1e-5);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
    assert_eq!(c.shared_memory_bytes(), 32); // 8 warps * 4 bytes
}

#[test]
fn test_block_size_large_dim_still_capped() {
    let c = PtxRmsNormConfig::new("rms", 8192, 1e-5);
    assert_eq!(c.block_size(), 256);
}

#[test]
fn test_block_size_dim_1() {
    let c = PtxRmsNormConfig::new("rms", 1, 1e-5);
    assert_eq!(c.block_size(), 32);
    assert!(c.is_warp_only());
}

#[test]
fn test_block_size_dim_33_rounds_to_two_warps() {
    let c = PtxRmsNormConfig::new("rms", 33, 1e-5);
    assert_eq!(c.block_size(), 64);
    assert_eq!(c.num_warps(), 2);
}

// =========================================================================
// SM target configuration
// =========================================================================

#[test]
fn test_sm_target_default() {
    let c = PtxRmsNormConfig::new("rms", 768, 1e-5);
    assert_eq!(c.sm_target, "sm_80");
}

#[test]
fn test_sm_target_custom() {
    let c = PtxRmsNormConfig::new("rms", 768, 1e-5).with_sm_target("sm_70");
    assert_eq!(c.sm_target, "sm_70");
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_sm_90_hopper() {
    let c = PtxRmsNormConfig::new("rms_hopper", 4096, 1e-5).with_sm_target("sm_90");
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

// =========================================================================
// PTX structural validation
// =========================================================================

#[test]
fn test_ptx_contains_version_and_target() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_ptx_contains_entry_point() {
    let c = PtxRmsNormConfig::new("nn_rmsnorm", 256, 1e-5);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains(".visible .entry nn_rmsnorm"));
}

#[test]
fn test_ptx_contains_kernel_params() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_weight"));
    assert!(ptx.contains("param_row_size"));
    assert!(ptx.contains("param_num_rows"));
}

#[test]
fn test_ptx_has_no_beta_param() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(
        !ptx.contains("param_beta"),
        "RMSNorm must NOT have a beta parameter"
    );
}

#[test]
fn test_ptx_has_no_gamma_param() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    // RMSNorm uses "weight", not "gamma"
    assert!(
        !ptx.contains("param_gamma"),
        "RMSNorm uses param_weight, not param_gamma"
    );
}

#[test]
fn test_ptx_has_rsqrt() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(
        ptx.contains("rsqrt.approx.f32"),
        "RMSNorm must use rsqrt for 1/sqrt(mean(x^2) + eps)"
    );
}

#[test]
fn test_ptx_has_warp_shuffle() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(
        ptx.contains("shfl.down.sync"),
        "RMSNorm must use warp shuffle for reduction"
    );
}

#[test]
fn test_ptx_has_no_mean_subtraction() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(
        !ptx.contains("sub.f32"),
        "RMSNorm must not subtract mean (no sub.f32)"
    );
}

#[test]
fn test_ptx_has_no_fma() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(
        !ptx.contains("fma.rn.f32"),
        "RMSNorm should not use fma (no beta addition)"
    );
}

#[test]
fn test_ptx_comment_header() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("RMSNorm f32"));
}

#[test]
fn test_ptx_register_declarations() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains(".reg .u32  %r<20>"));
    assert!(ptx.contains(".reg .f32  %f<16>"));
    assert!(ptx.contains(".reg .u64  %rd<12>"));
    assert!(ptx.contains(".reg .pred %p<6>"));
}

#[test]
fn test_ptx_reqntid_matches_block_size() {
    let c = PtxRmsNormConfig::new("rms", 768, 1e-5);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains(".reqntid 256"));

    let c2 = PtxRmsNormConfig::new("rms", 16, 1e-5);
    let ptx2 = emit_ptx_rmsnorm(&c2).unwrap();
    assert!(ptx2.contains(".reqntid 32"));
}

#[test]
fn test_ptx_loads_and_stores_global() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_ptx_uses_ptx_thread_id() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("%tid.x"));
    assert!(ptx.contains("mov.u32       %r2, %tid.x"));
}

#[test]
fn test_ptx_uses_ptx_block_id() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("%ctaid.x"));
    assert!(ptx.contains("mov.u32       %r3, %ctaid.x"));
}

#[test]
fn test_ptx_has_bounds_check() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("setp.ge.u32"));
    assert!(ptx.contains("RMS_EXIT"));
}

#[test]
fn test_ptx_has_phase_labels() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("Phase 1: compute mean(x^2)"));
    assert!(ptx.contains("Phase 2: normalize + scale"));
}

#[test]
fn test_ptx_has_loop_labels() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("RMS_SQ_LOOP"));
    assert!(ptx.contains("RMS_SQ_REDUCE"));
    assert!(ptx.contains("RMS_NORM_LOOP"));
}

#[test]
fn test_ptx_ends_with_closing_brace() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    let trimmed = ptx.trim_end();
    assert!(trimmed.ends_with('}'));
}

#[test]
fn test_ptx_not_cuda_cpp() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("__shared__"));
}

#[test]
fn test_ptx_reasonable_size() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
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
fn test_warp_only_dim_16_no_shared_memory() {
    let c = PtxRmsNormConfig::new("rms_16", 16, 1e-5);
    assert!(c.is_warp_only());
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(!ptx.contains("warp_scratch"));
    assert!(!ptx.contains(".shared"));
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_warp_only_no_barrier() {
    let c = PtxRmsNormConfig::new("rms_32", 32, 1e-5);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(!ptx.contains("bar.sync"), "warp-only must not use barrier");
}

#[test]
fn test_multi_warp_dim_128_uses_shared_memory() {
    let c = PtxRmsNormConfig::new("rms_128", 128, 1e-5);
    assert!(!c.is_warp_only());
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("warp_scratch"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_multi_warp_cross_warp_labels() {
    let c = PtxRmsNormConfig::new("rms_128", 128, 1e-5);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("CROSS_RMS_LOAD"));
    assert!(ptx.contains("CROSS_RMS_DONE"));
    assert!(ptx.contains("BCAST_RMS_LOAD"));
}

#[test]
fn test_warp_only_no_cross_warp_labels() {
    let c = PtxRmsNormConfig::new("rms_32", 32, 1e-5);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(!ptx.contains("CROSS_RMS_LOAD"));
    assert!(!ptx.contains("BCAST_RMS_LOAD"));
}

#[test]
fn test_multi_warp_shared_scratch_size() {
    // 4 warps -> warp_scratch[4]
    let c = PtxRmsNormConfig::new("rms_128", 128, 1e-5);
    assert_eq!(c.num_warps(), 4);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("warp_scratch[4]"));

    // 8 warps -> warp_scratch[8]
    let c2 = PtxRmsNormConfig::new("rms_4096", 4096, 1e-5);
    assert_eq!(c2.num_warps(), 8);
    let ptx2 = emit_ptx_rmsnorm(&c2).unwrap();
    assert!(ptx2.contains("warp_scratch[8]"));
}

// =========================================================================
// Shuffle broadcast
// =========================================================================

#[test]
fn test_ptx_has_shfl_idx_broadcast() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains("shfl.idx.sync"));
}

// =========================================================================
// Different epsilon values produce different PTX
// =========================================================================

#[test]
fn test_different_eps_produce_different_ptx() {
    let ptx_1e5 = emit_ptx_rmsnorm_default("rms", 768, 1e-5).unwrap();
    let ptx_1e6 = emit_ptx_rmsnorm_default("rms", 768, 1e-6).unwrap();
    let ptx_1e8 = emit_ptx_rmsnorm_default("rms", 768, 1e-8).unwrap();
    assert_ne!(ptx_1e5, ptx_1e6);
    assert_ne!(ptx_1e6, ptx_1e8);
}

// =========================================================================
// LLM-typical dimensions
// =========================================================================

#[test]
fn test_llama_dim_4096() {
    let c = PtxRmsNormConfig::new("llama_rms", 4096, 1e-5);
    assert_eq!(c.block_size(), 256);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("hidden_dim=4096"));
    assert!(ptx.contains("warp_scratch[8]"));
}

#[test]
fn test_qwen3_dim_3584() {
    let c = PtxRmsNormConfig::new("qwen_rms", 3584, 1e-6);
    assert_eq!(c.block_size(), 256);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("hidden_dim=3584"));
}

#[test]
fn test_llama_70b_dim_8192() {
    let c = PtxRmsNormConfig::new("llama70b_rms", 8192, 1e-5);
    assert_eq!(c.block_size(), 256);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("hidden_dim=8192"));
}

#[test]
fn test_gemma_dim_2048() {
    let c = PtxRmsNormConfig::new("gemma_rms", 2048, 1e-6);
    let ptx = emit_ptx_rmsnorm(&c).unwrap();
    assert!(ptx.contains("hidden_dim=2048"));
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_single_row() {
    let (grid, block) = ptx_rmsnorm_launch_config(1, 768);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_many_rows() {
    let (grid, block) = ptx_rmsnorm_launch_config(100_000, 4096);
    assert_eq!(grid, [100_000, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_small_dim() {
    let (grid, block) = ptx_rmsnorm_launch_config(10, 16);
    assert_eq!(grid, [10, 1, 1]);
    assert_eq!(block, [32, 1, 1]);
}

#[test]
fn test_launch_config_dim_33() {
    let (_grid, block) = ptx_rmsnorm_launch_config(10, 33);
    assert_eq!(block, [64, 1, 1]);
}

#[test]
fn test_launch_config_dim_1() {
    let (grid, block) = ptx_rmsnorm_launch_config(1, 1);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [32, 1, 1]);
}

// =========================================================================
// Reference computation: known values
// =========================================================================

#[test]
fn test_reference_ones_input_ones_weight() {
    // input = [1, 1, 1, 1], weight = [1, 1, 1, 1], eps = 0
    // mean(x^2) = 1.0, rsqrt(1.0) = 1.0
    // output = [1, 1, 1, 1]
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 4];
    let output = rmsnorm_reference(&input, &weight, 0.0);
    for &v in &output {
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
    }
}

#[test]
fn test_reference_twos_input_ones_weight() {
    // input = [2, 2, 2, 2], weight = [1, 1, 1, 1], eps = 0
    // mean(x^2) = 4.0, rsqrt(4.0) = 0.5
    // output = [2*0.5, ...] = [1, 1, 1, 1]
    let input = vec![2.0f32; 4];
    let weight = vec![1.0f32; 4];
    let output = rmsnorm_reference(&input, &weight, 0.0);
    for &v in &output {
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
    }
}

#[test]
fn test_reference_known_values() {
    // input = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-5
    // mean(x^2) = (1+4+9+16)/4 = 7.5
    // inv_rms = 1/sqrt(7.5 + 1e-5)
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps = 1e-5f32;
    let output = rmsnorm_reference(&input, &weight, eps);

    let mean_sq = 7.5f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| x * inv_rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "mismatch: got {o}, expected {e}");
    }
}

#[test]
fn test_reference_weight_scaling() {
    // input = [1, 2, 3, 4], weight = [2, 2, 2, 2], eps = 0
    // mean(x^2) = 7.5, inv_rms = 1/sqrt(7.5)
    // output = weight * input * inv_rms = 2 * input / sqrt(7.5)
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0; 4];
    let output = rmsnorm_reference(&input, &weight, 0.0);

    let inv_rms = 1.0 / 7.5f32.sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| 2.0 * x * inv_rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "mismatch: got {o}, expected {e}");
    }
}

#[test]
fn test_reference_non_uniform_weight() {
    // input = [1, 2, 3], weight = [0.5, 1.0, 2.0], eps = 1e-5
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![0.5, 1.0, 2.0];
    let eps = 1e-5f32;
    let output = rmsnorm_reference(&input, &weight, eps);

    let mean_sq = (1.0 + 4.0 + 9.0) / 3.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let expected = [0.5 * 1.0 * inv_rms,
        1.0 * 2.0 * inv_rms,
        2.0 * 3.0 * inv_rms];

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "mismatch: got {o}, expected {e}");
    }
}

// =========================================================================
// Reference computation: edge cases
// =========================================================================

#[test]
fn test_reference_zero_input() {
    // input = [0, 0, 0, 0], weight = [1, 1, 1, 1], eps = 1e-5
    // mean(x^2) = 0, inv_rms = 1/sqrt(eps)
    // output = 0 * inv_rms = [0, 0, 0, 0]
    let input = vec![0.0f32; 4];
    let weight = vec![1.0f32; 4];
    let output = rmsnorm_reference(&input, &weight, 1e-5);
    for &v in &output {
        assert!(
            v.abs() < 1e-6,
            "zero input should produce zero output, got {v}"
        );
    }
}

#[test]
fn test_reference_zero_weight() {
    // weight = 0 -> output = 0 regardless of input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.0f32; 4];
    let output = rmsnorm_reference(&input, &weight, 1e-5);
    for &v in &output {
        assert!(
            v.abs() < 1e-6,
            "zero weight should produce zero output, got {v}"
        );
    }
}

#[test]
fn test_reference_negative_input() {
    // RMSNorm works with negative values -- x^2 is always positive
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let weight = vec![1.0f32; 4];
    let output = rmsnorm_reference(&input, &weight, 1e-5);

    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv_rms = 1.0 / (mean_sq + 1e-5f32).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| x * inv_rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "mismatch: got {o}, expected {e}");
    }
    // Output should be negative
    for &v in &output {
        assert!(v < 0.0, "negative input should produce negative output");
    }
}

#[test]
fn test_reference_large_values() {
    // Large input values to check numerical stability
    let input = vec![1000.0, 2000.0, 3000.0, 4000.0];
    let weight = vec![1.0f32; 4];
    let eps = 1e-5f32;
    let output = rmsnorm_reference(&input, &weight, eps);

    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / 4.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| x * inv_rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!(
            (o - e).abs() / e.abs().max(1e-10) < 1e-5,
            "mismatch: got {o}, expected {e}"
        );
    }
}

#[test]
fn test_reference_single_element() {
    // dim = 1: mean(x^2) = x^2, inv_rms = 1/|x| (approximately)
    let input = vec![3.0f32];
    let weight = vec![1.0f32];
    let output = rmsnorm_reference(&input, &weight, 0.0);
    // RMSNorm of a single element with weight=1: x / |x| = sign(x)
    assert!((output[0] - 1.0).abs() < 1e-6);

    let input_neg = vec![-5.0f32];
    let output_neg = rmsnorm_reference(&input_neg, &weight, 0.0);
    assert!((output_neg[0] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_reference_mixed_sign() {
    let input = vec![1.0, -2.0, 3.0, -4.0];
    let weight = vec![1.0f32; 4];
    let eps = 1e-5f32;
    let output = rmsnorm_reference(&input, &weight, eps);

    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| x * inv_rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "mismatch: got {o}, expected {e}");
    }
}

#[test]
fn test_reference_eps_effect() {
    // Larger eps should reduce output magnitude for small inputs
    let input = vec![0.001, 0.002, 0.001, 0.002];
    let weight = vec![1.0f32; 4];

    let out_small_eps = rmsnorm_reference(&input, &weight, 1e-8);
    let out_large_eps = rmsnorm_reference(&input, &weight, 1.0);

    // With large eps, the denominator is larger, so output is smaller
    let mag_small: f32 = out_small_eps.iter().map(|x| x.abs()).sum();
    let mag_large: f32 = out_large_eps.iter().map(|x| x.abs()).sum();
    assert!(
        mag_small > mag_large,
        "larger eps should reduce output magnitude: small={mag_small}, large={mag_large}"
    );
}

// =========================================================================
// Config Clone and Debug
// =========================================================================

#[test]
fn test_config_clone() {
    let c = PtxRmsNormConfig::new("rms", 768, 1e-5);
    let c2 = c.clone();
    assert_eq!(c.hidden_dim, c2.hidden_dim);
    assert_eq!(c.kernel_name, c2.kernel_name);
    assert_eq!(c.eps, c2.eps);
    assert_eq!(c.sm_target, c2.sm_target);
}

#[test]
fn test_config_debug() {
    let c = PtxRmsNormConfig::new("rms", 768, 1e-5);
    let debug = format!("{c:?}");
    assert!(debug.contains("PtxRmsNormConfig"));
    assert!(debug.contains("768"));
}

// =========================================================================
// Different dims produce different PTX
// =========================================================================

#[test]
fn test_different_dims_produce_different_ptx() {
    let ptx_32 = generate_rmsnorm_ptx(32, 1e-5);
    let ptx_128 = generate_rmsnorm_ptx(128, 1e-5);
    let ptx_4096 = generate_rmsnorm_ptx(4096, 1e-5);

    assert_ne!(ptx_32, ptx_128);
    assert_ne!(ptx_128, ptx_4096);
    assert_ne!(ptx_32, ptx_4096);
}

// =========================================================================
// Convenience wrapper
// =========================================================================

#[test]
fn test_emit_ptx_rmsnorm_default() {
    let ptx = emit_ptx_rmsnorm_default("rms_default", 768, 1e-5).unwrap();
    assert!(ptx.contains(".entry rms_default"));
}

#[test]
fn test_generate_rmsnorm_ptx() {
    let ptx = generate_rmsnorm_ptx(768, 1e-5);
    assert!(ptx.contains(".entry ptx_rmsnorm_f32"));
    assert!(ptx.contains("hidden_dim=768"));
}
