// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CUDA C++ attention kernel generation.

use super::*;

// =========================================================================
// Config validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    assert_eq!(c.num_heads, 8);
    assert_eq!(c.head_dim, 64);
    assert_eq!(c.seq_len, 128);
    assert_eq!(c.kv_seq_len, 128);
    assert_eq!(c.num_kv_heads, 8); // default: MHA
    assert_eq!(c.dtype, "float");
    assert!(!c.causal);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_head_dim_zero() {
    let c = PtxAttentionConfig::new("attn", 8, 0, 128, 128);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_num_heads_zero() {
    let c = PtxAttentionConfig::new("attn", 0, 64, 128, 128);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_seq_len_zero() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 0, 128);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_kv_seq_len_zero() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_empty_name() {
    let c = PtxAttentionConfig::new("", 8, 64, 128, 128);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_invalid_dtype() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_dtype("double");
    assert!(c.validate().is_err());
}

#[test]
fn test_config_nan_scale() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_scale(f32::NAN);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_inf_scale() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_scale(f32::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_gqa_not_divisible() {
    // 8 heads, 3 kv heads -> 8 % 3 != 0
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_num_kv_heads(3);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_gqa_valid() {
    // 8 heads, 2 kv heads -> 8 % 2 == 0
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_num_kv_heads(2);
    assert!(c.validate().is_ok());
    assert_eq!(c.heads_per_kv_group(), 4);
    assert!(!c.is_mha());
    assert!(!c.is_mqa());
}

#[test]
fn test_config_mqa() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_num_kv_heads(1);
    assert!(c.validate().is_ok());
    assert_eq!(c.heads_per_kv_group(), 8);
    assert!(!c.is_mha());
    assert!(c.is_mqa());
}

#[test]
fn test_config_mha() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    assert!(c.is_mha());
}

#[test]
fn test_config_scale_default() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    let expected = 1.0 / (64.0f32).sqrt();
    assert!((c.scale - expected).abs() < 1e-6);
}

#[test]
fn test_config_cross_attention() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 256);
    assert!(c.is_cross_attention());
    assert_eq!(c.seq_len, 128);
    assert_eq!(c.kv_seq_len, 256);
}

#[test]
fn test_config_self_attention() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    assert!(!c.is_cross_attention());
}

// =========================================================================
// Block size computation
// =========================================================================

#[test]
fn test_block_size_small_kv_seq() {
    // kv_seq_len=16 -> default block_size = round up to 32 (one warp)
    let c = PtxAttentionConfig::new("attn", 8, 64, 16, 16);
    assert_eq!(c.block_size, 32);
}

#[test]
fn test_block_size_warp_boundary() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 32, 32);
    assert_eq!(c.block_size, 32);
}

#[test]
fn test_block_size_multi_warp() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    assert_eq!(c.block_size, 128);
}

#[test]
fn test_block_size_capped() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 2048, 2048);
    assert_eq!(c.block_size, 256);
}

#[test]
fn test_block_size_custom() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_block_size(64);
    assert_eq!(c.block_size, 64);
}

// =========================================================================
// Shared memory computation
// =========================================================================

#[test]
fn test_shared_memory_bytes() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    // scores: 128 * 4 = 512, reduce_buf: 128 * 4 = 512 -> total = 1024
    assert_eq!(c.shared_memory_bytes(), 1024);
}

#[test]
fn test_shared_memory_bytes_large_seq() {
    let c = PtxAttentionConfig::new("attn", 8, 64, 2048, 2048);
    // scores: 2048 * 4 = 8192, reduce_buf: 256 * 4 = 1024 -> total = 9216
    assert_eq!(c.shared_memory_bytes(), 9216);
}

#[test]
fn test_shared_memory_cross_attention() {
    // kv_seq_len=256 but seq_len=64 -> scores use kv_seq_len
    let c = PtxAttentionConfig::new("attn", 8, 64, 64, 256);
    // scores: 256 * 4 = 1024, reduce_buf: 256 * 4 = 1024 -> total = 2048
    assert_eq!(c.shared_memory_bytes(), 2048);
}

// =========================================================================
// Basic generation (default: causal, MHA)
// =========================================================================

#[test]
fn test_basic_generation_succeeds() {
    let result = emit_ptx_attention_default(8, 64, 128, 128);
    assert!(result.is_ok());
}

#[test]
fn test_default_is_causal() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("Causal mask"),
        "default emit_ptx_attention_default must be causal"
    );
}

#[test]
fn test_contains_cuda_prelude() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("#include <cuda_runtime.h>"),
        "must include CUDA runtime header"
    );
    assert!(
        src.contains("#include <cuda_fp16.h>"),
        "must include fp16 header"
    );
}

#[test]
fn test_contains_global_keyword() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("__global__"),
        "must contain __global__ kernel declaration"
    );
}

#[test]
fn test_contains_kernel_name() {
    let config = PtxAttentionConfig::new("nn_attention", 8, 64, 128, 128);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("__global__ void nn_attention"),
        "must declare the named __global__ kernel"
    );
}

#[test]
fn test_contains_kernel_params() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("const float* __restrict__ Q"),
        "must have Q param"
    );
    assert!(
        src.contains("const float* __restrict__ K"),
        "must have K param"
    );
    assert!(
        src.contains("const float* __restrict__ V"),
        "must have V param"
    );
    assert!(
        src.contains("float* __restrict__ output"),
        "must have output param"
    );
    assert!(src.contains("batch_size"), "must have batch_size param");
}

#[test]
fn test_contains_shared_memory() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("__shared__ float scores["),
        "must declare shared memory for scores"
    );
    assert!(
        src.contains("__shared__ float reduce_buf["),
        "must declare shared memory for reduction"
    );
}

#[test]
fn test_contains_block_indices() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(src.contains("blockIdx.x"), "must use blockIdx.x for batch");
    assert!(src.contains("blockIdx.y"), "must use blockIdx.y for head");
    assert!(
        src.contains("blockIdx.z"),
        "must use blockIdx.z for query pos"
    );
    assert!(src.contains("threadIdx.x"), "must use threadIdx.x for tid");
}

#[test]
fn test_contains_syncthreads() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    let sync_count = src.matches("__syncthreads").count();
    assert!(
        sync_count >= 2,
        "must have at least 2 __syncthreads calls (after scores, after softmax), got {sync_count}"
    );
}

#[test]
fn test_contains_softmax_pattern() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(src.contains("expf("), "must use expf for softmax");
    assert!(
        src.contains("-FLT_MAX"),
        "must use -FLT_MAX for numerical stability"
    );
    assert!(src.contains("inv_sum"), "must compute inverse of sum");
}

#[test]
fn test_contains_dot_product() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("dot +="),
        "must compute dot product between Q and K"
    );
}

#[test]
fn test_contains_shfl_down() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("__shfl_down_sync"),
        "must use warp shuffle for reduction"
    );
}

#[test]
fn test_contains_scale() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(src.contains("scale"), "must contain scale factor reference");
    assert!(
        src.contains("* 0.125"),
        "must apply scale factor (1/sqrt(64) = 0.125)"
    );
}

#[test]
fn test_contains_value_aggregation() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("scores[j] * (float)V["),
        "must aggregate values weighted by attention scores"
    );
}

// =========================================================================
// Causal vs non-causal
// =========================================================================

#[test]
fn test_causal_contains_mask() {
    let config = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_causal(true);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("j > q_pos"),
        "causal attention must check j > q_pos"
    );
    assert!(
        src.contains("Causal mask"),
        "causal attention must have causal mask comment"
    );
}

#[test]
fn test_non_causal_no_mask() {
    let config = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_causal(false);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        !src.contains("j > q_pos"),
        "non-causal attention must not apply causal mask"
    );
}

#[test]
fn test_causal_differs_from_non_causal() {
    let causal_cfg = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_causal(true);
    let non_causal_cfg = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_causal(false);
    let causal = emit_ptx_attention(&causal_cfg).unwrap();
    let non_causal = emit_ptx_attention(&non_causal_cfg).unwrap();
    assert_ne!(causal, non_causal, "causal and non-causal must differ");
}

// =========================================================================
// GQA configuration
// =========================================================================

#[test]
fn test_gqa_kernel_generation() {
    let config = PtxAttentionConfig::new("gqa_attn", 8, 64, 128, 128).with_num_kv_heads(2);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("head_idx / 4u"),
        "GQA must compute kv_head_idx = head_idx / (num_heads / num_kv_heads)"
    );
    assert!(
        src.contains("num_kv_heads=2"),
        "header comment must show num_kv_heads"
    );
}

#[test]
fn test_mha_no_division() {
    let config = PtxAttentionConfig::new("mha_attn", 8, 64, 128, 128);
    let src = emit_ptx_attention(&config).unwrap();
    // MHA: heads_per_group == 1, so no division needed
    assert!(
        src.contains("kv_head_idx = head_idx"),
        "MHA must map head_idx directly to kv_head_idx"
    );
}

#[test]
fn test_mqa_group_size_8() {
    let config = PtxAttentionConfig::new("mqa_attn", 8, 64, 128, 128).with_num_kv_heads(1);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("head_idx / 8u"),
        "MQA with 8 heads must divide by 8"
    );
}

// =========================================================================
// Half precision dtype
// =========================================================================

#[test]
fn test_half_dtype() {
    let config = PtxAttentionConfig::new("half_attn", 8, 64, 128, 128)
        .with_dtype("half")
        .with_causal(false);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("const half* __restrict__ Q"),
        "half dtype must use half* for Q"
    );
    assert!(
        src.contains("half* __restrict__ output"),
        "half dtype must use half* for output"
    );
    assert!(
        src.contains("(half)acc"),
        "half dtype must cast accumulator to half for output"
    );
}

#[test]
fn test_float_dtype() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("const float* __restrict__ Q"),
        "float dtype must use float* for Q"
    );
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_basic() {
    let config = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    let lc = ptx_attention_launch_config(&config, 4);
    assert_eq!(lc.grid.x, 4); // batch_size
    assert_eq!(lc.grid.y, 8); // num_heads
    assert_eq!(lc.grid.z, 128); // seq_len
    assert_eq!(lc.block.x, 128); // block_size
    assert_eq!(lc.block.y, 1);
    assert_eq!(lc.block.z, 1);
}

#[test]
fn test_launch_config_large_seq() {
    let config = PtxAttentionConfig::new("attn", 32, 64, 2048, 2048);
    let lc = ptx_attention_launch_config(&config, 16);
    assert_eq!(lc.grid.x, 16); // batch_size
    assert_eq!(lc.grid.y, 32); // num_heads
    assert_eq!(lc.grid.z, 2048); // seq_len
    assert_eq!(lc.block.x, 256); // capped block_size
}

#[test]
fn test_launch_config_shared_mem() {
    let config = PtxAttentionConfig::new("attn", 8, 64, 128, 128);
    let lc = ptx_attention_launch_config(&config, 1);
    assert_eq!(lc.shared_mem_bytes, config.shared_memory_bytes() as u32);
}

#[test]
fn test_launch_config_small_seq() {
    let config = PtxAttentionConfig::new("attn", 8, 64, 16, 16);
    let lc = ptx_attention_launch_config(&config, 1);
    assert_eq!(lc.block.x, 32); // rounded up to one warp
}

#[test]
fn test_launch_config_grid_dims_reasonable() {
    let config = PtxAttentionConfig::new("attn", 16, 128, 512, 512);
    let lc = ptx_attention_launch_config(&config, 8);
    // Grid dimensions should be batch x heads x seq_len
    assert!(lc.grid.x > 0 && lc.grid.x <= 8);
    assert!(lc.grid.y > 0 && lc.grid.y <= 16);
    assert!(lc.grid.z > 0 && lc.grid.z <= 512);
    // Block size should be reasonable
    assert!(lc.block.x >= 32 && lc.block.x <= 256);
}

// =========================================================================
// Output structural patterns
// =========================================================================

#[test]
fn test_output_contains_expected_cuda_patterns() {
    let config = PtxAttentionConfig::new("attn", 8, 64, 128, 128).with_causal(true);
    let src = emit_ptx_attention(&config).unwrap();
    let expected_patterns = [
        "__global__",       // CUDA kernel declaration
        "__restrict__",     // restrict pointers
        "__shared__",       // shared memory
        "__syncthreads",    // thread synchronization
        "__shfl_down_sync", // warp shuffle
        "blockIdx",         // block index
        "threadIdx",        // thread index
        "expf(",            // exponential function
        "-FLT_MAX",         // float minimum for softmax
        "0xFFFFFFFF",       // warp mask (all lanes active)
        "softmax",          // softmax comment
        "scale",            // scale factor
    ];
    for pattern in &expected_patterns {
        assert!(
            src.contains(pattern),
            "output must contain CUDA pattern: {pattern}"
        );
    }
}

#[test]
fn test_is_cuda_cpp_not_raw_ptx() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    // Must be CUDA C++, not raw PTX assembly
    assert!(
        src.contains("__global__"),
        "must be CUDA C++ (contain __global__)"
    );
    assert!(
        !src.contains(".version 6.5"),
        "must not contain raw PTX version directive"
    );
    assert!(
        !src.contains(".target sm_"),
        "must not contain raw PTX target directive"
    );
}

#[test]
fn test_header_comment_has_config() {
    let config = PtxAttentionConfig::new("attn", 16, 128, 256, 256)
        .with_num_kv_heads(4)
        .with_causal(true);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(src.contains("head_dim=128"), "header must show head_dim");
    assert!(src.contains("num_heads=16"), "header must show num_heads");
    assert!(
        src.contains("num_kv_heads=4"),
        "header must show num_kv_heads"
    );
    assert!(src.contains("seq_len=256"), "header must show seq_len");
    assert!(
        src.contains("kv_seq_len=256"),
        "header must show kv_seq_len"
    );
    assert!(src.contains("causal=true"), "header must show causal=true");
}

// =========================================================================
// Different configurations produce different output
// =========================================================================

#[test]
fn test_different_head_dims_differ() {
    let src_64 = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    let src_128 = emit_ptx_attention_default(8, 128, 128, 128).unwrap();
    assert_ne!(
        src_64, src_128,
        "different head_dim must produce different output"
    );
}

#[test]
fn test_different_seq_lens_differ() {
    let src_128 = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    let src_256 = emit_ptx_attention_default(8, 64, 256, 256).unwrap();
    assert_ne!(
        src_128, src_256,
        "different seq_len must produce different output"
    );
}

#[test]
fn test_head_dim_64_vs_128() {
    let src_64 = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    let src_128 = emit_ptx_attention_default(8, 128, 128, 128).unwrap();
    // head_dim=64 => scale=0.125, head_dim=128 => scale~=0.0884
    assert!(
        src_64.contains("0.125"),
        "head_dim=64 should use scale 0.125"
    );
    assert!(
        !src_128.contains("0.125"),
        "head_dim=128 should use different scale"
    );
}

// =========================================================================
// Cross-attention (kv_seq_len != seq_len)
// =========================================================================

#[test]
fn test_cross_attention_kernel() {
    let config = PtxAttentionConfig::new("cross_attn", 8, 64, 64, 256);
    let src = emit_ptx_attention(&config).unwrap();
    // Scores should iterate over kv_seq_len (256), not seq_len (64)
    assert!(
        src.contains("j < 256u"),
        "score loop must iterate over kv_seq_len=256"
    );
    assert!(
        src.contains("kv_seq_len=256"),
        "header must show kv_seq_len=256"
    );
    assert!(src.contains("seq_len=64"), "header must show seq_len=64");
}

#[test]
fn test_cross_attention_shared_mem_uses_kv_seq() {
    let config = PtxAttentionConfig::new("cross_attn", 8, 64, 64, 256);
    let src = emit_ptx_attention(&config).unwrap();
    // Shared scores array should be sized by kv_seq_len
    assert!(
        src.contains("scores[256]"),
        "shared scores must be sized by kv_seq_len"
    );
}

// =========================================================================
// Builder pattern
// =========================================================================

#[test]
fn test_builder_chain() {
    let config = PtxAttentionConfig::new("chain_attn", 16, 64, 512, 512)
        .with_num_kv_heads(4)
        .with_causal(true)
        .with_dtype("half")
        .with_scale(0.1)
        .with_block_size(128)
        .with_sm_target("sm_90");
    assert_eq!(config.num_kv_heads, 4);
    assert!(config.causal);
    assert_eq!(config.dtype, "half");
    assert!((config.scale - 0.1).abs() < 1e-6);
    assert_eq!(config.block_size, 128);
    assert_eq!(config.sm_target, "sm_90");
    assert!(config.validate().is_ok());
}

// =========================================================================
// Single-warp path (small kv_seq_len)
// =========================================================================

#[test]
fn test_single_warp_uses_shfl_sync() {
    // kv_seq_len=16 -> block_size=32 -> single warp path
    let config = PtxAttentionConfig::new("small_attn", 8, 64, 16, 16);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("__shfl_sync(0xFFFFFFFF"),
        "single-warp path must use __shfl_sync for broadcast"
    );
}

#[test]
fn test_multi_warp_uses_reduce_buf() {
    // kv_seq_len=128 -> block_size=128 -> multi-warp path
    let config = PtxAttentionConfig::new("big_attn", 8, 64, 128, 128);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("reduce_buf[warp_id]"),
        "multi-warp path must use reduce_buf for cross-warp reduction"
    );
}

// =========================================================================
// Default config produces valid output
// =========================================================================

#[test]
fn test_default_config_produces_valid_kernel() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    // Must be a complete kernel with opening and closing braces
    let open_count = src.matches('{').count();
    let close_count = src.matches('}').count();
    assert_eq!(open_count, close_count, "kernel must have balanced braces");
    // Must end with closing brace
    assert!(
        src.trim().ends_with('}'),
        "kernel must end with closing brace"
    );
}

#[test]
fn test_default_config_all_phases_present() {
    let src = emit_ptx_attention_default(8, 64, 128, 128).unwrap();
    assert!(
        src.contains("Phase 1"),
        "must have Phase 1 (score computation)"
    );
    assert!(src.contains("Phase 2a"), "must have Phase 2a (find max)");
    assert!(src.contains("Phase 2b"), "must have Phase 2b (exp + sum)");
    assert!(src.contains("Phase 2c"), "must have Phase 2c (normalize)");
    assert!(
        src.contains("Phase 3"),
        "must have Phase 3 (value aggregation)"
    );
}

// =========================================================================
// ATTENTION_BLOCK_SIZE constant
// =========================================================================

#[test]
fn test_attention_block_size_constant_value() {
    assert_eq!(ATTENTION_BLOCK_SIZE, 256);
}

#[test]
fn test_attention_block_size_matches_max() {
    // For large kv_seq_len, block_size should cap at ATTENTION_BLOCK_SIZE
    let config = PtxAttentionConfig::new("test", 8, 64, 512, 512);
    assert_eq!(config.block_size, ATTENTION_BLOCK_SIZE as usize);
}

// =========================================================================
// generate_sdpa_ptx convenience functions
// =========================================================================

#[test]
fn test_generate_sdpa_ptx_produces_kernel() {
    let src = generate_sdpa_ptx(128, 64);
    assert!(
        src.contains("__global__"),
        "SDPA must produce a __global__ kernel"
    );
    assert!(
        src.contains("sdpa_f32"),
        "SDPA kernel must use sdpa_f32 name"
    );
}

#[test]
fn test_generate_sdpa_ptx_non_causal() {
    let src = generate_sdpa_ptx(64, 32);
    // Non-causal: should NOT contain causal masking logic
    assert!(
        !src.contains("Causal mask"),
        "non-causal SDPA should not have causal mask"
    );
}

#[test]
fn test_generate_sdpa_causal_ptx_produces_kernel() {
    let src = generate_sdpa_causal_ptx(128, 64);
    assert!(
        src.contains("__global__"),
        "causal SDPA must produce a __global__ kernel"
    );
    assert!(
        src.contains("sdpa_causal_f32"),
        "causal SDPA kernel must use sdpa_causal_f32 name"
    );
}

#[test]
fn test_generate_sdpa_causal_ptx_has_causal_mask() {
    let src = generate_sdpa_causal_ptx(64, 32);
    assert!(
        src.contains("Causal mask") || src.contains("causal") || src.contains("-FLT_MAX"),
        "causal SDPA must contain causal masking logic"
    );
}

#[test]
fn test_generate_sdpa_ptx_various_sizes() {
    for (seq, dim) in [(16, 16), (32, 64), (64, 128), (128, 256), (256, 64)] {
        let src = generate_sdpa_ptx(seq, dim);
        assert!(
            !src.is_empty(),
            "generate_sdpa_ptx({seq}, {dim}) should produce non-empty output"
        );
        assert!(
            src.contains("__global__"),
            "generate_sdpa_ptx({seq}, {dim}) must contain kernel entry"
        );
    }
}

#[test]
fn test_generate_sdpa_causal_differs_from_non_causal() {
    let non_causal = generate_sdpa_ptx(64, 64);
    let causal = generate_sdpa_causal_ptx(64, 64);
    assert_ne!(
        non_causal, causal,
        "causal and non-causal SDPA should differ"
    );
}

#[test]
fn test_generate_sdpa_ptx_contains_all_phases() {
    let src = generate_sdpa_ptx(128, 64);
    assert!(
        src.contains("Phase 1"),
        "must have Phase 1 (score computation)"
    );
    assert!(src.contains("Phase 2"), "must have Phase 2 (softmax)");
    assert!(
        src.contains("Phase 3"),
        "must have Phase 3 (value aggregation)"
    );
}

// =========================================================================
// sdpa_reference CPU reference
// =========================================================================

#[test]
fn test_sdpa_reference_identity_keys() {
    // With identity-like K, Q should attend to the matching V row
    let head_dim = 4;
    let seq_len = 3;
    // Q = [[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0]]
    let q = vec![
        100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0,
    ];
    // K = same as Q (identity-like)
    let k = q.clone();
    // V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let out = sdpa_reference(&q, &k, &v, head_dim);
    assert_eq!(out.len(), seq_len * head_dim);

    // Each query should predominantly attend to its matching key
    // Row 0 should be close to V[0] = [1, 0, 0, 0]
    assert!(
        out[0] > 0.5,
        "row 0, dim 0 should be close to 1.0, got {}",
        out[0]
    );
}

#[test]
fn test_sdpa_reference_uniform_query() {
    // Uniform Q: all rows the same -> should give same output
    let head_dim = 2;
    let seq_len = 3;
    let q = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 3x2
    let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

    let out = sdpa_reference(&q, &k, &v, head_dim);
    assert_eq!(out.len(), seq_len * head_dim);

    // All query rows are identical, so all output rows should be the same
    for i in 1..seq_len {
        for d in 0..head_dim {
            let diff = (out[i * head_dim + d] - out[d]).abs();
            assert!(
                diff < 1e-5,
                "uniform query should produce uniform output, row {i} dim {d} diff = {diff}"
            );
        }
    }
}

#[test]
fn test_sdpa_reference_single_element() {
    let q = vec![1.0, 2.0];
    let k = vec![3.0, 4.0];
    let v = vec![5.0, 6.0];

    let out = sdpa_reference(&q, &k, &v, 2);
    // With single element softmax, attention weight is 1.0
    // output = 1.0 * V = V
    assert_eq!(out.len(), 2);
    assert!(
        (out[0] - 5.0).abs() < 1e-6,
        "single element SDPA should return V, got {}",
        out[0]
    );
    assert!(
        (out[1] - 6.0).abs() < 1e-6,
        "single element SDPA should return V, got {}",
        out[1]
    );
}

#[test]
fn test_sdpa_reference_weights_sum_to_one() {
    // Verify softmax weights sum to 1 implicitly via output bounds
    let head_dim = 4;
    let q = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let k = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    // V values between 0 and 10
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 1.0];

    let out = sdpa_reference(&q, &k, &v, head_dim);
    // Output should be convex combination of V rows, so within V's value range
    for &val in &out {
        assert!(
            (0.0..=10.0).contains(&val),
            "output should be within V range [0, 10], got {val}"
        );
    }
}

#[test]
#[should_panic(expected = "head_dim must be > 0")]
fn test_sdpa_reference_zero_head_dim_panics() {
    sdpa_reference(&[1.0], &[1.0], &[1.0], 0);
}

#[test]
#[should_panic(expected = "q length must be a multiple of head_dim")]
fn test_sdpa_reference_mismatched_q_panics() {
    sdpa_reference(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0, 2.0], 2);
}

#[test]
#[should_panic(expected = "k and v must have the same length")]
fn test_sdpa_reference_mismatched_kv_panics() {
    sdpa_reference(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 2);
}
