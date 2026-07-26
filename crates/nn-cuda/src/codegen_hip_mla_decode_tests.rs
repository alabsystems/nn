// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MLA decode kernel emission.

use super::*;

#[test]
fn test_mla_decode_basic_generation() {
    let src = emit_mla_decode_kernel("mla_decode_test", 32, 128, 512, 2048, 1, 0.08838835).unwrap();

    // Verify HIP include.
    assert!(
        src.contains("#include <hip/hip_runtime.h>"),
        "missing HIP include"
    );

    // Verify kernel signature.
    assert!(src.contains("extern \"C\" __global__ void mla_decode_test("));
    assert!(src.contains("const float* __restrict__ Q"));
    assert!(src.contains("const float* __restrict__ C_kv"));
    assert!(src.contains("const float* __restrict__ W_uk"));
    assert!(src.contains("const float* __restrict__ W_uv"));
    assert!(src.contains("float* __restrict__ O"));
}

#[test]
fn test_mla_decode_dimensions_in_source() {
    let src = emit_mla_decode_kernel("k1", 16, 64, 256, 1024, 4, 0.125).unwrap();

    assert!(src.contains("N_HEADS = 16"), "wrong n_heads");
    assert!(src.contains("HEAD_DIM = 64"), "wrong head_dim");
    assert!(src.contains("D_C = 256"), "wrong d_c");
    assert!(src.contains("S_KV = 1024"), "wrong s_kv");
    assert!(src.contains("BATCH_SIZE = 4"), "wrong batch_size");
}

#[test]
fn test_mla_decode_absorbed_key_computation() {
    let src = emit_mla_decode_kernel("k_abs", 8, 64, 128, 512, 1, 0.125).unwrap();

    // Step 1: q_absorbed = Q[h] @ W_uk[h] * scale
    assert!(
        src.contains("q_absorbed[j] = acc * SCALE"),
        "missing absorbed Q computation"
    );
    assert!(
        src.contains("q_ptr[i] * w_uk_ptr[i * D_C + j]"),
        "missing Q @ W_uk matmul"
    );
}

#[test]
fn test_mla_decode_online_softmax() {
    let src = emit_mla_decode_kernel("k_sm", 8, 64, 128, 512, 1, 0.125).unwrap();

    // Online softmax components.
    assert!(src.contains("running_max"), "missing running max comment");
    assert!(src.contains("fmaxf(old_max, score)"), "missing max update");
    assert!(
        src.contains("expf(old_max - new_max)"),
        "missing correction factor"
    );
    assert!(src.contains("expf(score - new_max)"), "missing exp score");
}

#[test]
fn test_mla_decode_v_weighted_accumulation() {
    let src = emit_mla_decode_kernel("k_vw", 8, 64, 128, 512, 1, 0.125).unwrap();

    // v_weighted accumulation with online softmax correction (step 4).
    // Must apply correction factor to existing v_weighted before adding new term.
    assert!(
        src.contains("v_weighted[j] * correction + exp_score * c_s[j]"),
        "missing corrected v_weighted accumulation"
    );
    // Correction factor must be broadcast via shared memory.
    assert!(
        src.contains("softmax_meta[2] = correction"),
        "missing correction broadcast to shared memory"
    );
    assert!(
        src.contains("float correction = softmax_meta[2]"),
        "missing correction read from shared memory"
    );
    // Normalization by softmax denominator.
    assert!(
        src.contains("1.0f / denom"),
        "missing softmax normalization"
    );
}

#[test]
fn test_mla_decode_output_expansion() {
    let src = emit_mla_decode_kernel("k_out", 8, 64, 128, 512, 1, 0.125).unwrap();

    // Step 5: out[h] = W_uv[h] @ v_weighted
    assert!(
        src.contains("w_uv_ptr[i * D_C + j] * v_weighted[j]"),
        "missing output expansion matmul"
    );
    assert!(src.contains("o_ptr[i] = acc"), "missing output store");
}

#[test]
fn test_mla_decode_cooperative_dot_product() {
    let src = emit_mla_decode_kernel("k_dot", 8, 64, 128, 512, 1, 0.125).unwrap();

    // Cooperative dot product with tree reduction.
    assert!(
        src.contains("partial_scores[tid]"),
        "missing partial score buffer"
    );
    assert!(
        src.contains("partial_scores[tid] += partial_scores[tid + stride]"),
        "missing tree reduction"
    );
}

#[test]
fn test_mla_decode_shared_memory_budget() {
    // d_c too large for shared memory (>8192 floats = 32KB per array, two arrays = 64KB).
    let result = emit_mla_decode_kernel("bad", 8, 64, 8193, 512, 1, 0.125);
    assert!(result.is_err(), "d_c=8193 should exceed shared memory");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("shared memory"), "{err}");
}

#[test]
fn test_mla_decode_deepseek_v3_dimensions() {
    // DeepSeek-V3 MLA: n_heads=128, head_dim=128, d_c=512, typical S_kv=4096.
    let src = emit_mla_decode_kernel(
        "mla_dsv3", 128, 128, 512, 4096, 1, 0.08838835, // 1/sqrt(128)
    )
    .unwrap();

    assert!(src.contains("N_HEADS = 128"));
    assert!(src.contains("HEAD_DIM = 128"));
    assert!(src.contains("D_C = 512"));
    assert!(src.contains("S_KV = 4096"));

    // Balanced braces check.
    let opens = src.matches('{').count();
    let closes = src.matches('}').count();
    assert_eq!(
        opens, closes,
        "unbalanced braces: opens={opens}, closes={closes}"
    );
}

#[test]
fn test_mla_decode_launch_config() {
    let cfg = mla_decode_launch_config(4, 32, 512);

    // Grid: B * n_heads = 4 * 32 = 128 blocks.
    assert_eq!(cfg.grid.x, 128);
    assert_eq!(cfg.grid.y, 1);
    assert_eq!(cfg.grid.z, 1);

    // Block: 256 threads.
    assert_eq!(cfg.block.x, 256);

    // Shared memory: (2 * 512 + 3 + 256) * 4 = 5132 bytes.
    assert_eq!(cfg.shared_mem_bytes, (2 * 512 + 3 + 256) * 4);
}

#[test]
fn test_mla_decode_batch_head_indexing() {
    let src = emit_mla_decode_kernel("k_idx", 8, 64, 128, 512, 2, 0.125).unwrap();

    // Verify batch/head decomposition from block index.
    assert!(
        src.contains("batch_idx = bh_idx / N_HEADS"),
        "missing batch decomposition"
    );
    assert!(
        src.contains("head_idx = bh_idx % N_HEADS"),
        "missing head decomposition"
    );
}

#[test]
fn test_mla_decode_scale_preapplied() {
    // Verify scale is applied during q_absorbed computation (not during dot product).
    let src = emit_mla_decode_kernel("k_scale", 8, 64, 128, 512, 1, 0.125).unwrap();
    assert!(
        src.contains("acc * SCALE"),
        "scale must be pre-applied to q_absorbed"
    );
    assert!(src.contains("SCALE = 0.12500000f"), "wrong scale value");
}
