// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fused multi-head attention PTX generation.

use super::*;

// =========================================================================
// Config validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    assert_eq!(c.num_heads, 8);
    assert_eq!(c.head_dim, 64);
    assert_eq!(c.seq_len, 128);
    assert_eq!(c.kv_seq_len, 128); // default: self-attention
    assert!(!c.causal);
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_zero_heads() {
    let c = PtxMultiHeadAttentionConfig::new(0, 64, 128);
    let err = c.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("num_heads"),
        "error must mention num_heads: {msg}"
    );
}

#[test]
fn test_config_zero_head_dim() {
    let c = PtxMultiHeadAttentionConfig::new(8, 0, 128);
    let err = c.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("head_dim"),
        "error must mention head_dim: {msg}"
    );
}

#[test]
fn test_config_zero_seq_len() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 0);
    let err = c.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("seq_len"), "error must mention seq_len: {msg}");
}

#[test]
fn test_config_zero_kv_seq_len() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128).with_kv_seq_len(0);
    let err = c.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("kv_seq_len"),
        "error must mention kv_seq_len: {msg}"
    );
}

#[test]
fn test_config_invalid_sm_target_no_prefix() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128).with_sm_target("80");
    let err = c.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("sm_target") || msg.contains("sm_"),
        "error must mention sm_target: {msg}"
    );
}

#[test]
fn test_config_invalid_sm_target_bad_number() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128).with_sm_target("sm_abc");
    let err = c.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("numeric") || msg.contains("invalid"),
        "error must mention invalid numeric suffix: {msg}"
    );
}

#[test]
fn test_config_valid_sm_targets() {
    for target in &["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"] {
        let c = PtxMultiHeadAttentionConfig::new(8, 64, 128).with_sm_target(target);
        assert!(c.validate().is_ok(), "sm_target {target} should be valid");
    }
}

// =========================================================================
// Derived properties
// =========================================================================

#[test]
fn test_d_model() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    assert_eq!(c.d_model(), 512);
}

#[test]
fn test_scale() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let expected = 1.0 / (64.0f32).sqrt();
    assert!((c.scale() - expected).abs() < 1e-6);
}

#[test]
fn test_scale_head_dim_128() {
    let c = PtxMultiHeadAttentionConfig::new(8, 128, 64);
    let expected = 1.0 / (128.0f32).sqrt();
    assert!((c.scale() - expected).abs() < 1e-6);
}

#[test]
fn test_block_size_small_seq() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 16);
    assert_eq!(c.block_size(), 32); // rounded up to one warp
}

#[test]
fn test_block_size_large_seq() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 2048);
    assert_eq!(c.block_size(), 256); // capped
}

#[test]
fn test_shared_memory_bytes() {
    let c = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    // q_local: 64*4=256, scores: 128*4=512, reduce_buf: 128*4=512
    assert_eq!(c.shared_memory_bytes(), 256 + 512 + 512);
}

// =========================================================================
// PTX structural validation
// =========================================================================

#[test]
fn test_ptx_generation_succeeds() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let result = generate_multihead_attention_ptx(&config);
    assert!(result.is_ok());
}

#[test]
fn test_ptx_contains_entry_point() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("__global__ void fused_multihead_attention"),
        "must contain __global__ entry point"
    );
}

#[test]
fn test_ptx_contains_params() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("const float* __restrict__ X"),
        "must have X param"
    );
    assert!(
        src.contains("const float* __restrict__ X_kv"),
        "must have X_kv param"
    );
    assert!(
        src.contains("const float* __restrict__ W_Q"),
        "must have W_Q param"
    );
    assert!(
        src.contains("const float* __restrict__ W_K"),
        "must have W_K param"
    );
    assert!(
        src.contains("const float* __restrict__ W_V"),
        "must have W_V param"
    );
    assert!(
        src.contains("const float* __restrict__ W_O"),
        "must have W_O param"
    );
    assert!(
        src.contains("float* __restrict__ output"),
        "must have output param"
    );
    assert!(src.contains("batch_size"), "must have batch_size param");
}

#[test]
fn test_ptx_contains_registers() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("blockIdx.x"), "must use blockIdx.x");
    assert!(src.contains("blockIdx.y"), "must use blockIdx.y");
    assert!(src.contains("blockIdx.z"), "must use blockIdx.z");
    assert!(src.contains("threadIdx.x"), "must use threadIdx.x");
}

#[test]
fn test_ptx_contains_shared_memory() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("__shared__ float q_local[64]"),
        "must have shared q_local[head_dim]"
    );
    assert!(
        src.contains("__shared__ float scores[128]"),
        "must have shared scores[kv_seq_len]"
    );
    assert!(
        src.contains("__shared__ float reduce_buf["),
        "must have shared reduce_buf"
    );
}

#[test]
fn test_ptx_contains_softmax_pattern() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("expf("), "must use expf for softmax");
    assert!(
        src.contains("-FLT_MAX"),
        "must use -FLT_MAX for numerical stability"
    );
    assert!(src.contains("inv_sum"), "must compute inverse of sum");
}

#[test]
fn test_ptx_contains_syncthreads() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    let sync_count = src.matches("__syncthreads").count();
    assert!(
        sync_count >= 3,
        "must have at least 3 __syncthreads (Q proj, scores, softmax), got {sync_count}"
    );
}

#[test]
fn test_ptx_balanced_braces() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    let open = src.matches('{').count();
    let close = src.matches('}').count();
    assert_eq!(open, close, "kernel must have balanced braces");
    assert!(src.trim().ends_with('}'), "must end with closing brace");
}

#[test]
fn test_ptx_contains_cuda_prelude() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("#include <cuda_runtime.h>"),
        "must include CUDA runtime header"
    );
    assert!(
        src.contains("#include <float.h>"),
        "must include float.h for FLT_MAX"
    );
}

#[test]
fn test_ptx_header_comment_has_config() {
    let config = PtxMultiHeadAttentionConfig::new(16, 128, 256);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("num_heads=16"), "header must show num_heads");
    assert!(src.contains("head_dim=128"), "header must show head_dim");
    assert!(src.contains("d_model=2048"), "header must show d_model");
    assert!(src.contains("seq_len=256"), "header must show seq_len");
    assert!(
        src.contains("kv_seq_len=256"),
        "header must show kv_seq_len"
    );
}

#[test]
fn test_ptx_contains_all_phases() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("Phase 1"), "must have Phase 1 (Q projection)");
    assert!(
        src.contains("Phase 2"),
        "must have Phase 2 (score computation)"
    );
    assert!(src.contains("Phase 3a"), "must have Phase 3a (find max)");
    assert!(src.contains("Phase 3b"), "must have Phase 3b (exp + sum)");
    assert!(src.contains("Phase 3c"), "must have Phase 3c (normalize)");
    assert!(
        src.contains("Phase 4"),
        "must have Phase 4 (value + output proj)"
    );
}

#[test]
fn test_ptx_contains_warp_shuffle() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("__shfl_down_sync"),
        "must use warp shuffle for reduction"
    );
}

#[test]
fn test_ptx_contains_atomic_add() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("atomicAdd"),
        "must use atomicAdd for output accumulation across heads"
    );
}

// =========================================================================
// Causal mask
// =========================================================================

#[test]
fn test_causal_ptx_contains_mask() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128).with_causal(true);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("j > q_pos"),
        "causal attention must check j > q_pos"
    );
    assert!(
        src.contains("Causal mask"),
        "causal attention must have mask comment"
    );
    assert!(src.contains("causal=true"), "header must show causal=true");
}

#[test]
fn test_non_causal_ptx_no_mask() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128).with_causal(false);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        !src.contains("j > q_pos"),
        "non-causal must not apply causal mask"
    );
    assert!(
        src.contains("causal=false"),
        "header must show causal=false"
    );
}

#[test]
fn test_causal_differs_from_non_causal() {
    let causal_src = generate_multihead_attention_ptx(
        &PtxMultiHeadAttentionConfig::new(8, 64, 128).with_causal(true),
    )
    .unwrap();
    let non_causal_src = generate_multihead_attention_ptx(
        &PtxMultiHeadAttentionConfig::new(8, 64, 128).with_causal(false),
    )
    .unwrap();
    assert_ne!(
        causal_src, non_causal_src,
        "causal and non-causal must produce different output"
    );
}

// =========================================================================
// Score scaling (1/sqrt(head_dim))
// =========================================================================

#[test]
fn test_scale_factor_head_dim_64() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    // 1/sqrt(64) = 0.125
    assert!(src.contains("0.125"), "head_dim=64 should use scale 0.125");
}

#[test]
fn test_scale_factor_head_dim_128() {
    let config = PtxMultiHeadAttentionConfig::new(8, 128, 64);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    // 1/sqrt(128) ~ 0.08838835
    assert!(
        !src.contains("0.125"),
        "head_dim=128 should NOT use scale 0.125"
    );
    assert!(
        src.contains("0.088"),
        "head_dim=128 should use scale ~0.088"
    );
}

#[test]
fn test_different_head_dims_produce_different_scales() {
    let src_64 =
        generate_multihead_attention_ptx(&PtxMultiHeadAttentionConfig::new(8, 64, 128)).unwrap();
    let src_128 =
        generate_multihead_attention_ptx(&PtxMultiHeadAttentionConfig::new(8, 128, 128)).unwrap();
    assert_ne!(
        src_64, src_128,
        "different head_dim must produce different output"
    );
}

// =========================================================================
// Various seq_len and head_dim combinations
// =========================================================================

#[test]
fn test_small_config() {
    let config = PtxMultiHeadAttentionConfig::new(1, 4, 4);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("__global__"),
        "small config must produce valid kernel"
    );
    assert!(src.contains("q_local[4]"), "must use head_dim=4");
}

#[test]
fn test_large_seq_len() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 2048);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("scores[2048]"),
        "must size scores by seq_len=2048"
    );
    // Block size should be capped at 256
    assert!(
        src.contains("reduce_buf[256]"),
        "block_size must cap at 256"
    );
}

#[test]
fn test_cross_attention_different_kv_seq() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 64).with_kv_seq_len(256);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("kv_seq_len=256"),
        "header must show kv_seq_len=256"
    );
    assert!(src.contains("seq_len=64"), "header must show seq_len=64");
    assert!(
        src.contains("scores[256]"),
        "scores must be sized by kv_seq_len=256"
    );
}

#[test]
fn test_many_heads_small_dim() {
    let config = PtxMultiHeadAttentionConfig::new(32, 16, 64);
    assert_eq!(config.d_model(), 512);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("d_model=512"), "header must show d_model=512");
    assert!(src.contains("q_local[16]"), "q_local sized by head_dim=16");
}

#[test]
fn test_single_head() {
    let config = PtxMultiHeadAttentionConfig::new(1, 64, 32);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("num_heads=1"), "header must show num_heads=1");
}

#[test]
fn test_seq_len_1() {
    // Single-token query (e.g. autoregressive decode step)
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 1).with_kv_seq_len(128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(src.contains("seq_len=1"), "header must show seq_len=1");
    assert!(
        src.contains("kv_seq_len=128"),
        "header must show kv_seq_len=128"
    );
}

// =========================================================================
// Reference computation: single-head, no causal
// =========================================================================

#[test]
fn test_reference_single_head_identity() {
    // 1 batch, 1 head, seq=1, kv_seq=1, head_dim=2
    // Q=[1, 0], K=[1, 0], V=[3, 4]
    // score = dot(Q, K) * scale = 1.0 * 1/sqrt(2) -> softmax -> 1.0
    // output = 1.0 * V = [3, 4]
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0];
    let v = vec![3.0, 4.0];
    let out = attention_reference(&q, &k, &v, 1, 1, 1, 1, 2, false);
    assert_eq!(out.len(), 2);
    assert!((out[0] - 3.0).abs() < 1e-5, "out[0]={}", out[0]);
    assert!((out[1] - 4.0).abs() < 1e-5, "out[1]={}", out[1]);
}

#[test]
fn test_reference_single_head_two_keys() {
    // 1 batch, 1 head, seq=1, kv_seq=2, head_dim=1
    // Q=[1], K=[1, 1], V=[2, 8]
    // scores = [1*scale, 1*scale] -> softmax -> [0.5, 0.5]
    // output = 0.5*2 + 0.5*8 = 5.0
    let q = vec![1.0];
    let k = vec![1.0, 1.0];
    let v = vec![2.0, 8.0];
    let out = attention_reference(&q, &k, &v, 1, 1, 1, 2, 1, false);
    assert_eq!(out.len(), 1);
    assert!((out[0] - 5.0).abs() < 1e-5, "expected ~5.0, got {}", out[0]);
}

#[test]
fn test_reference_multi_head() {
    // 1 batch, 2 heads, seq=1, kv_seq=1, head_dim=1
    // Head 0: Q=[2], K=[1], V=[10] -> score=2*scale, softmax=1, out=10
    // Head 1: Q=[3], K=[1], V=[20] -> score=3*scale, softmax=1, out=20
    let q = vec![2.0, 3.0]; // [1, 2, 1, 1] layout
    let k = vec![1.0, 1.0];
    let v = vec![10.0, 20.0];
    let out = attention_reference(&q, &k, &v, 1, 2, 1, 1, 1, false);
    assert_eq!(out.len(), 2);
    assert!(
        (out[0] - 10.0).abs() < 1e-5,
        "head 0: expected 10, got {}",
        out[0]
    );
    assert!(
        (out[1] - 20.0).abs() < 1e-5,
        "head 1: expected 20, got {}",
        out[1]
    );
}

// =========================================================================
// Reference computation: causal mask
// =========================================================================

#[test]
fn test_reference_causal_first_token() {
    // seq=2, kv_seq=2, head_dim=1
    // Q[0]=[1], Q[1]=[1], K=[1, 1], V=[10, 20]
    // Causal: q_pos=0 can see j=0 only -> output[0]=10
    //         q_pos=1 can see j=0,1 -> softmax([s,s])=[0.5,0.5] -> output[1]=15
    let q = vec![1.0, 1.0]; // [1, 1, 2, 1]
    let k = vec![1.0, 1.0];
    let v = vec![10.0, 20.0];
    let out = attention_reference(&q, &k, &v, 1, 1, 2, 2, 1, true);
    assert_eq!(out.len(), 2);
    // q_pos=0: only key j=0 visible -> weight=1 -> out=10
    assert!(
        (out[0] - 10.0).abs() < 1e-5,
        "q_pos=0 (causal): expected 10, got {}",
        out[0]
    );
    // q_pos=1: both keys visible, equal scores -> out=15
    assert!(
        (out[1] - 15.0).abs() < 1e-5,
        "q_pos=1 (causal): expected 15, got {}",
        out[1]
    );
}

#[test]
fn test_reference_causal_vs_non_causal_differ() {
    let q = vec![1.0, 1.0];
    let k = vec![1.0, 2.0];
    let v = vec![10.0, 20.0];
    let causal = attention_reference(&q, &k, &v, 1, 1, 2, 2, 1, true);
    let non_causal = attention_reference(&q, &k, &v, 1, 1, 2, 2, 1, false);
    // q_pos=0: causal sees only j=0, non-causal sees j=0,1 -> different
    assert!(
        (causal[0] - non_causal[0]).abs() > 1e-3,
        "causal and non-causal should differ for q_pos=0"
    );
}

#[test]
fn test_reference_causal_triangular() {
    // 3 positions: verify triangular attention pattern
    // q_pos 0: sees [0]
    // q_pos 1: sees [0,1]
    // q_pos 2: sees [0,1,2]
    let head_dim = 1;
    let q = vec![1.0, 1.0, 1.0]; // 3 query positions
    let k = vec![1.0, 1.0, 1.0]; // 3 key positions
    let v = vec![1.0, 2.0, 3.0]; // distinct values

    let out = attention_reference(&q, &k, &v, 1, 1, 3, 3, head_dim, true);
    assert_eq!(out.len(), 3);
    // q_pos=0: sees only V[0] -> 1.0
    assert!(
        (out[0] - 1.0).abs() < 1e-5,
        "q_pos=0: expected 1.0, got {}",
        out[0]
    );
    // q_pos=1: sees V[0],V[1] equally -> (1+2)/2 = 1.5
    assert!(
        (out[1] - 1.5).abs() < 1e-5,
        "q_pos=1: expected 1.5, got {}",
        out[1]
    );
    // q_pos=2: sees V[0],V[1],V[2] equally -> (1+2+3)/3 = 2.0
    assert!(
        (out[2] - 2.0).abs() < 1e-5,
        "q_pos=2: expected 2.0, got {}",
        out[2]
    );
}

// =========================================================================
// Reference computation: batched
// =========================================================================

#[test]
fn test_reference_batched() {
    // 2 batches, 1 head, seq=1, kv_seq=1, head_dim=1
    // Batch 0: Q=[1], K=[1], V=[5] -> out=5
    // Batch 1: Q=[1], K=[1], V=[9] -> out=9
    let q = vec![1.0, 1.0];
    let k = vec![1.0, 1.0];
    let v = vec![5.0, 9.0];
    let out = attention_reference(&q, &k, &v, 2, 1, 1, 1, 1, false);
    assert_eq!(out.len(), 2);
    assert!((out[0] - 5.0).abs() < 1e-5);
    assert!((out[1] - 9.0).abs() < 1e-5);
}

// =========================================================================
// Reference: larger dimension
// =========================================================================

#[test]
fn test_reference_head_dim_4() {
    // 1 batch, 1 head, seq=1, kv_seq=2, head_dim=4
    // Q = [1,0,0,0], K = [[1,0,0,0], [0,1,0,0]], V = [[1,2,3,4], [5,6,7,8]]
    // score[0] = dot([1,0,0,0], [1,0,0,0]) * scale = 1 * 0.5
    // score[1] = dot([1,0,0,0], [0,1,0,0]) * scale = 0 * 0.5
    // softmax: exp(0.5)/Z, exp(0)/Z where Z = exp(0.5)+exp(0)
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let out = attention_reference(&q, &k, &v, 1, 1, 1, 2, 4, false);
    assert_eq!(out.len(), 4);

    // Compute expected weights
    let scale = 1.0 / (4.0f32).sqrt();
    let s0 = (1.0 * scale).exp();
    let s1 = (0.0 * scale).exp();
    let z = s0 + s1;
    let w0 = s0 / z;
    let w1 = s1 / z;

    for d in 0..4 {
        let expected = w0 * v[d] + w1 * v[4 + d];
        assert!(
            (out[d] - expected).abs() < 1e-4,
            "dim {d}: expected {expected}, got {}",
            out[d]
        );
    }
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_basic() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let lc = multihead_attention_launch_config(&config, 4);
    assert_eq!(lc.grid.x, 4); // batch_size
    assert_eq!(lc.grid.y, 8); // num_heads
    assert_eq!(lc.grid.z, 128); // seq_len
    assert_eq!(lc.block.x, 128); // block_size
    assert_eq!(lc.block.y, 1);
    assert_eq!(lc.block.z, 1);
}

#[test]
fn test_launch_config_large() {
    let config = PtxMultiHeadAttentionConfig::new(32, 64, 2048);
    let lc = multihead_attention_launch_config(&config, 16);
    assert_eq!(lc.grid.x, 16);
    assert_eq!(lc.grid.y, 32);
    assert_eq!(lc.grid.z, 2048);
    assert_eq!(lc.block.x, 256); // capped at MAX_BLOCK_SIZE
}

#[test]
fn test_launch_config_shared_mem_matches() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let lc = multihead_attention_launch_config(&config, 1);
    assert_eq!(lc.shared_mem_bytes, config.shared_memory_bytes() as u32);
}

#[test]
fn test_launch_config_small_seq() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 16);
    let lc = multihead_attention_launch_config(&config, 1);
    assert_eq!(lc.grid.z, 16);
    assert_eq!(lc.block.x, 32); // one warp
}

// =========================================================================
// Config builder pattern
// =========================================================================

#[test]
fn test_builder_chain() {
    let config = PtxMultiHeadAttentionConfig::new(16, 64, 512)
        .with_kv_seq_len(1024)
        .with_causal(true)
        .with_sm_target("sm_90");
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.seq_len, 512);
    assert_eq!(config.kv_seq_len, 1024);
    assert!(config.causal);
    assert_eq!(config.sm_target, "sm_90");
    assert!(config.validate().is_ok());
}

// =========================================================================
// Different configs produce different output
// =========================================================================

#[test]
fn test_different_seq_lens_differ() {
    let src_128 =
        generate_multihead_attention_ptx(&PtxMultiHeadAttentionConfig::new(8, 64, 128)).unwrap();
    let src_256 =
        generate_multihead_attention_ptx(&PtxMultiHeadAttentionConfig::new(8, 64, 256)).unwrap();
    assert_ne!(src_128, src_256);
}

#[test]
fn test_different_num_heads_differ() {
    let src_8 =
        generate_multihead_attention_ptx(&PtxMultiHeadAttentionConfig::new(8, 64, 128)).unwrap();
    let src_16 =
        generate_multihead_attention_ptx(&PtxMultiHeadAttentionConfig::new(16, 64, 128)).unwrap();
    assert_ne!(src_8, src_16);
}

#[test]
fn test_invalid_config_returns_error() {
    let config = PtxMultiHeadAttentionConfig::new(0, 64, 128);
    assert!(generate_multihead_attention_ptx(&config).is_err());
}
