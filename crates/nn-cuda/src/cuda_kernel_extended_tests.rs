// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4186.
//!
//! Extended tests for CUDA PTX kernel generation and configuration covering:
//! 1. Activation kernel PTX generation (softmax, attention, ReLU)
//! 2. Elementwise ops PTX (add, mul, fused multiply-add)
//! 3. Reduction kernel patterns (warp-level and block-level)
//! 4. Thread block configuration validation (multiples of 32)
//! 5. Shared memory allocation for tiled operations
//! 6. Register usage and kernel pressure estimation
//! 7. Grid dimension calculation
//! 8. Embedding lookup PTX generation
//! 9. Transpose PTX with bank conflict avoidance
//! 10. RoPE (Rotary Position Embedding) PTX generation

use crate::{
    add_reference,
    argmax_reference,
    argmin_reference,
    batch_transpose_reference,
    embedding_reference,
    // CUDA C++ emission
    emit_activation_kernels,
    emit_elementwise_kernel,
    // Attention
    emit_ptx_attention,
    // Softmax
    emit_ptx_softmax,
    // Elementwise
    generate_add_ptx,
    generate_argmax_ptx,
    generate_argmin_ptx,
    generate_batch_transpose_ptx,
    generate_div_ptx,
    // Embedding
    generate_embedding_ptx,
    generate_exp_ptx,
    generate_log_ptx,
    generate_log_softmax_ptx,
    generate_max_ptx,
    generate_mean_ptx,
    generate_mul_ptx,
    generate_neg_ptx,
    generate_rope_cached_ptx,
    // RoPE
    generate_rope_ptx,
    generate_scalar_mul_ptx,
    generate_sdpa_causal_ptx,
    generate_sdpa_ptx,
    generate_softmax_ptx,
    generate_sqrt_ptx,
    generate_sub_ptx,
    // Reductions
    generate_sum_ptx,
    // Transpose
    generate_transpose_ptx,
    log_softmax_reference,
    max_reference,
    mean_reference,
    mul_reference,
    // Codegen helpers
    ptx_activation_launch_config,
    ptx_attention_launch_config,
    ptx_batch_transpose_launch_config,
    ptx_elementwise_launch_config,
    ptx_embedding_launch_config,
    ptx_reduce_launch_config,
    ptx_rope_launch_config,
    ptx_softmax_launch_config,
    ptx_transpose_launch_config,
    rope_reference,
    rope_reference_with_base,
    scalar_mul_reference,
    softmax_reference,
    sum_reference,
    transpose_reference,
    PtxAttentionConfig,
    PtxEmbeddingConfig,
    PtxRopeConfig,
    PtxSoftmaxConfig,
    ATTENTION_BLOCK_SIZE,
    ELEMENTWISE_BLOCK_SIZE,
    EMBEDDING_BLOCK_SIZE,
    PTX_VERSION,
    REDUCE_BLOCK_SIZE,
    ROPE_BLOCK_SIZE,
    SOFTMAX_BLOCK_SIZE,
    TRANSPOSE_BLOCK_SIZE,
    WARP_SIZE,
};

// =========================================================================
// Section 1: PTX Activation Kernel Generation (softmax, attention, ReLU)
// =========================================================================

#[test]
fn test_softmax_ptx_contains_warp_shuffle_reduction() {
    let config = PtxSoftmaxConfig::new("softmax_k", 128);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(
        ptx.contains("shfl.down.sync"),
        "Softmax PTX must use warp shuffle for reduction"
    );
    assert!(
        ptx.contains(".entry softmax_k"),
        "Must contain named kernel entry"
    );
    assert!(
        ptx.contains("ex2.approx.f32"),
        "Softmax must use fast exp2 approximation"
    );
}

#[test]
fn test_softmax_ptx_single_warp_no_shared_memory() {
    // dim <= 32 should use pure warp shuffle, no shared memory cross-warp reduction
    let config = PtxSoftmaxConfig::new("sm_warp", 16);
    assert!(config.is_warp_only(), "dim=16 should be warp-only");
    assert_eq!(
        config.shared_memory_bytes(),
        0,
        "Warp-only softmax needs no shared memory"
    );
    assert_eq!(config.num_warps(), 1);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry sm_warp"));
}

#[test]
fn test_softmax_ptx_multi_warp_uses_shared_memory() {
    // dim > 32 needs cross-warp reduction via shared memory
    let config = PtxSoftmaxConfig::new("sm_multi", 128);
    assert!(!config.is_warp_only(), "dim=128 should be multi-warp");
    assert!(config.num_warps() > 1);
    assert!(
        config.shared_memory_bytes() > 0,
        "Multi-warp softmax needs shared memory"
    );
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(
        ptx.contains(".shared"),
        "Multi-warp softmax must declare shared memory"
    );
}

#[test]
fn test_softmax_ptx_numerically_stable_max_subtraction() {
    let ptx = emit_ptx_softmax(&PtxSoftmaxConfig::new("sm_stable", 64)).unwrap();
    // Numerically stable softmax subtracts max before exp
    assert!(
        ptx.contains("max.f32") || ptx.contains("setp.gt.f32"),
        "Softmax must compute row-wise max for numerical stability"
    );
}

#[test]
fn test_log_softmax_ptx_has_log_computation() {
    let config = PtxSoftmaxConfig::new_log("log_sm", 64);
    assert!(config.log_mode);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry log_sm"));
    // Log softmax should use lg2 for log computation
    assert!(
        ptx.contains("lg2.approx.f32"),
        "Log softmax must compute log of sum"
    );
}

#[test]
fn test_attention_ptx_contains_score_computation() {
    let config = PtxAttentionConfig::new("attn_k", 8, 64, 128, 128);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("__global__"),
        "Attention must be a CUDA kernel"
    );
    assert!(src.contains("attn_k"), "Kernel name must be present");
    // Score computation: Q @ K^T * scale
    assert!(
        src.contains("scale") || src.contains("sqrtf"),
        "Attention must include scaling"
    );
}

#[test]
fn test_attention_ptx_causal_mask_neginfinity() {
    let config = PtxAttentionConfig::new("attn_causal", 4, 64, 32, 32).with_causal(true);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(
        src.contains("causal")
            || src.contains("-INFINITY")
            || src.contains("-1e30")
            || src.contains("mask")
            || src.contains("-inf"),
        "Causal attention must apply -inf mask to future positions"
    );
}

#[test]
fn test_attention_ptx_gqa_kv_head_mapping() {
    // GQA: 8 query heads, 2 KV heads
    let config = PtxAttentionConfig::new("attn_gqa", 8, 64, 32, 32).with_num_kv_heads(2);
    let src = emit_ptx_attention(&config).unwrap();
    assert!(src.contains("__global__"));
    // The kernel should reference head group mapping
    assert!(
        src.contains("kv_head") || src.contains("head_idx") || src.contains("num_kv"),
        "GQA kernel must compute KV head index from query head"
    );
}

#[test]
fn test_relu_via_cuda_cpp_emission() {
    let src = emit_activation_kernels();
    assert!(
        src.contains("relu_kernel"),
        "CUDA C++ emission must include ReLU"
    );
    assert!(
        src.contains("x > 0.0f ? x : 0.0f"),
        "ReLU should be max(0, x)"
    );
    // Also verify sigmoid, silu, tanh, gelu are present
    assert!(src.contains("sigmoid_kernel"));
    assert!(src.contains("silu_kernel"));
    assert!(src.contains("tanh_kernel"));
    assert!(src.contains("gelu_kernel"));
}

#[test]
fn test_sdpa_convenience_ptx_contains_entry() {
    let ptx = generate_sdpa_ptx(32, 64);
    assert!(
        ptx.contains("sdpa_f32") || ptx.contains("__global__"),
        "SDPA convenience wrapper must generate a kernel"
    );
}

#[test]
fn test_sdpa_causal_convenience_ptx() {
    let ptx = generate_sdpa_causal_ptx(32, 64);
    assert!(
        ptx.contains("sdpa_causal") || ptx.contains("__global__"),
        "Causal SDPA must generate a kernel"
    );
}

// =========================================================================
// Section 2: PTX Elementwise Ops (add, mul, fused multiply-add)
// =========================================================================

#[test]
fn test_add_ptx_contains_add_instruction() {
    let ptx = generate_add_ptx(1024);
    assert!(
        ptx.contains("add.f32"),
        "Add kernel must contain add.f32 PTX instruction"
    );
    assert!(ptx.contains(".entry"), "Must be a PTX entry point");
    assert!(
        ptx.contains(".param .u64 param_a"),
        "Must have input pointer a"
    );
    assert!(
        ptx.contains(".param .u64 param_b"),
        "Must have input pointer b"
    );
    assert!(
        ptx.contains(".param .u64 param_output"),
        "Must have output pointer"
    );
}

#[test]
fn test_mul_ptx_contains_mul_instruction() {
    let ptx = generate_mul_ptx(512);
    assert!(
        ptx.contains("mul.f32"),
        "Mul kernel must contain mul.f32 PTX instruction"
    );
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_sub_ptx_contains_sub_instruction() {
    let ptx = generate_sub_ptx(256);
    assert!(
        ptx.contains("sub.f32"),
        "Sub kernel must contain sub.f32 PTX instruction"
    );
}

#[test]
fn test_div_ptx_contains_div_instruction() {
    let ptx = generate_div_ptx(256);
    assert!(
        ptx.contains("div.approx.f32") || ptx.contains("div.f32"),
        "Div kernel must contain div PTX instruction"
    );
}

#[test]
fn test_exp_ptx_uses_ex2_with_prescale() {
    let ptx = generate_exp_ptx(256);
    // exp(x) = 2^(x * log2(e)), so we expect ex2 and a log2(e) prescale
    assert!(
        ptx.contains("ex2.approx.f32"),
        "Exp must use fast ex2.approx"
    );
    assert!(ptx.contains("mul.f32"), "Exp must prescale by log2(e)");
}

#[test]
fn test_log_ptx_uses_lg2_with_postscale() {
    let ptx = generate_log_ptx(256);
    // log(x) = lg2(x) * ln(2)
    assert!(
        ptx.contains("lg2.approx.f32"),
        "Log must use fast lg2.approx"
    );
    assert!(ptx.contains("mul.f32"), "Log must postscale by ln(2)");
}

#[test]
fn test_neg_ptx_contains_neg_instruction() {
    let ptx = generate_neg_ptx(256);
    assert!(
        ptx.contains("neg.f32"),
        "Neg kernel must contain neg.f32 instruction"
    );
}

#[test]
fn test_sqrt_ptx_contains_sqrt_instruction() {
    let ptx = generate_sqrt_ptx(256);
    assert!(
        ptx.contains("sqrt.approx.f32") || ptx.contains("sqrt.f32"),
        "Sqrt kernel must contain sqrt PTX instruction"
    );
}

#[test]
fn test_scalar_mul_ptx_has_scalar_parameter() {
    let ptx = generate_scalar_mul_ptx(256);
    assert!(
        ptx.contains(".param .f32 param_scalar"),
        "ScalarMul must accept a scalar parameter"
    );
    assert!(ptx.contains("mul.f32"), "ScalarMul must multiply");
}

#[test]
fn test_fma_pattern_in_exp_ptx() {
    // exp kernel uses fma.rn.f32 for prescale in some implementations
    let ptx = generate_exp_ptx(1024);
    // Either fma or separate mul+ex2 is acceptable
    assert!(
        ptx.contains("fma.rn.f32") || ptx.contains("mul.f32"),
        "Exp kernel should use fma or mul for log2(e) prescale"
    );
}

#[test]
fn test_elementwise_ptx_bounds_check() {
    // All elementwise kernels must have bounds checking (idx < N)
    let ptx = generate_add_ptx(100);
    assert!(
        ptx.contains("setp."),
        "Elementwise must have predicate for bounds check"
    );
    assert!(
        ptx.contains("@%p") || ptx.contains("@!%p"),
        "Elementwise must use predicated branch for bounds check"
    );
}

#[test]
fn test_cuda_cpp_elementwise_kernel_custom_op() {
    let src = emit_elementwise_kernel("fma_kernel", "x * 2.0f + 1.0f", 2048).unwrap();
    assert!(src.contains("__global__ void fma_kernel"));
    assert!(
        src.contains("x * 2.0f + 1.0f"),
        "Custom op expression must appear in kernel"
    );
    assert!(
        src.contains("blockIdx.x"),
        "Must use CUDA grid/block indexing"
    );
}

#[test]
fn test_elementwise_reference_add() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = add_reference(&a, &b);
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_elementwise_reference_mul() {
    let a = vec![2.0, 3.0, 4.0];
    let b = vec![0.5, 2.0, 0.25];
    let result = mul_reference(&a, &b);
    assert_eq!(result, vec![1.0, 6.0, 1.0]);
}

#[test]
fn test_elementwise_reference_scalar_mul() {
    let x = vec![1.0, 2.0, 3.0];
    let result = scalar_mul_reference(&x, 3.0);
    assert_eq!(result, vec![3.0, 6.0, 9.0]);
}

// =========================================================================
// Section 3: PTX Reduction Kernels (warp-level and block-level)
// =========================================================================

#[test]
fn test_sum_reduction_ptx_uses_shared_memory_tree() {
    let ptx = generate_sum_ptx(1024);
    assert!(
        ptx.contains(".shared"),
        "Sum reduction must use shared memory for tree reduction"
    );
    assert!(
        ptx.contains("add.f32"),
        "Sum reduction must contain add instruction"
    );
    assert!(
        ptx.contains("bar.sync"),
        "Sum reduction must synchronize threads via barrier"
    );
    assert!(ptx.contains(".entry ptx_sum_f32"));
}

#[test]
fn test_max_reduction_ptx_uses_max_instruction() {
    let ptx = generate_max_ptx(512);
    assert!(
        ptx.contains(".shared"),
        "Max reduction must use shared memory"
    );
    assert!(
        ptx.contains("max.f32") || ptx.contains("setp.gt.f32"),
        "Max reduction must use max or conditional comparison"
    );
    assert!(ptx.contains("bar.sync"), "Must synchronize threads");
}

#[test]
fn test_mean_reduction_ptx_includes_division() {
    let ptx = generate_mean_ptx(256);
    assert!(
        ptx.contains(".shared"),
        "Mean reduction must use shared memory"
    );
    // Mean = sum / n, so we need both addition and division
    assert!(ptx.contains("add.f32"), "Mean reduction must sum elements");
    assert!(
        ptx.contains("div.approx.f32") || ptx.contains("mul.f32") || ptx.contains("rcp"),
        "Mean reduction must divide by count (or multiply by reciprocal)"
    );
}

#[test]
fn test_argmax_reduction_ptx_tracks_index() {
    let ptx = generate_argmax_ptx(256);
    assert!(ptx.contains(".entry ptx_argmax_f32"));
    assert!(
        ptx.contains("param_out_idx") || ptx.contains("param_output"),
        "Argmax must output an index"
    );
    assert!(
        ptx.contains(".shared"),
        "Argmax must use shared memory for reduction"
    );
}

#[test]
fn test_argmin_reduction_ptx_tracks_index() {
    let ptx = generate_argmin_ptx(256);
    assert!(ptx.contains(".entry ptx_argmin_f32"));
    assert!(
        ptx.contains("param_out_idx") || ptx.contains("param_output"),
        "Argmin must output an index"
    );
}

#[test]
fn test_reduction_reference_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((sum_reference(&data) - 15.0).abs() < 1e-6);
}

#[test]
fn test_reduction_reference_max() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    assert!((max_reference(&data) - 9.0).abs() < 1e-6);
}

#[test]
fn test_reduction_reference_mean() {
    let data = vec![2.0, 4.0, 6.0, 8.0];
    assert!((mean_reference(&data) - 5.0).abs() < 1e-6);
}

#[test]
fn test_reduction_reference_argmax() {
    let data = vec![1.0, 5.0, 3.0, 2.0];
    assert_eq!(argmax_reference(&data), 1);
}

#[test]
fn test_reduction_reference_argmin() {
    let data = vec![5.0, 1.0, 3.0, 2.0];
    assert_eq!(argmin_reference(&data), 1);
}

#[test]
fn test_reduction_ptx_reqntid_directive() {
    // PTX reduction kernels should declare required thread count
    let ptx = generate_sum_ptx(512);
    assert!(
        ptx.contains(".reqntid") || ptx.contains(".maxntid"),
        "Reduction kernel should declare required thread count via .reqntid or .maxntid"
    );
}

// =========================================================================
// Section 4: Thread Block Configuration (multiples of 32)
// =========================================================================

#[test]
fn test_warp_size_is_32() {
    assert_eq!(WARP_SIZE, 32, "NVIDIA warp size must be 32");
}

#[test]
fn test_elementwise_block_size_multiple_of_warp() {
    assert_eq!(
        ELEMENTWISE_BLOCK_SIZE % (WARP_SIZE as u32),
        0,
        "Elementwise block size must be a multiple of warp size"
    );
}

#[test]
fn test_reduce_block_size_multiple_of_warp() {
    assert_eq!(
        REDUCE_BLOCK_SIZE % (WARP_SIZE as u32),
        0,
        "Reduce block size must be a multiple of warp size"
    );
}

#[test]
fn test_softmax_block_size_multiple_of_warp() {
    assert_eq!(
        SOFTMAX_BLOCK_SIZE % (WARP_SIZE as u32),
        0,
        "Softmax block size must be a multiple of warp size"
    );
}

#[test]
fn test_attention_block_size_multiple_of_warp() {
    assert_eq!(
        ATTENTION_BLOCK_SIZE % (WARP_SIZE as u32),
        0,
        "Attention block size must be a multiple of warp size"
    );
}

#[test]
fn test_embedding_block_size_multiple_of_warp() {
    assert_eq!(
        EMBEDDING_BLOCK_SIZE % (WARP_SIZE as u32),
        0,
        "Embedding block size must be a multiple of warp size"
    );
}

#[test]
fn test_rope_block_size_multiple_of_warp() {
    assert_eq!(
        ROPE_BLOCK_SIZE % (WARP_SIZE as u32),
        0,
        "RoPE block size must be a multiple of warp size"
    );
}

#[test]
fn test_transpose_block_size_is_power_of_two() {
    assert!(
        TRANSPOSE_BLOCK_SIZE.is_power_of_two(),
        "Transpose tile size should be a power of two for efficient bank access"
    );
}

#[test]
fn test_softmax_config_block_size_rounds_to_warp() {
    // dim=17 should round up to 32 (one warp)
    let config = PtxSoftmaxConfig::new("t", 17);
    assert_eq!(
        config.block_size() % WARP_SIZE,
        0,
        "Softmax block size must be a warp multiple"
    );
    assert_eq!(config.block_size(), 32);
}

#[test]
fn test_softmax_config_block_size_capped_at_256() {
    let config = PtxSoftmaxConfig::new("t", 1024);
    assert!(
        config.block_size() <= 256,
        "Softmax block size must be capped at 256 threads"
    );
}

#[test]
fn test_attention_default_block_size_rounds_to_warp() {
    let config = PtxAttentionConfig::new("t", 4, 64, 32, 17);
    // kv_seq_len=17 should round up to 32
    assert_eq!(config.block_size % WARP_SIZE, 0);
}

#[test]
fn test_attention_default_block_size_capped() {
    let config = PtxAttentionConfig::new("t", 4, 64, 32, 512);
    assert!(
        config.block_size <= 256,
        "Attention block size must be capped at 256"
    );
}

// =========================================================================
// Section 5: Shared Memory Allocation for Tiled Operations
// =========================================================================

#[test]
fn test_softmax_shared_memory_scales_with_warps() {
    let config_small = PtxSoftmaxConfig::new("s1", 64); // 2 warps
    let config_large = PtxSoftmaxConfig::new("s2", 256); // 8 warps
    assert!(
        config_large.shared_memory_bytes() >= config_small.shared_memory_bytes(),
        "More warps should need more shared memory for cross-warp reduction"
    );
}

#[test]
fn test_softmax_shared_memory_warp_only_is_zero() {
    let config = PtxSoftmaxConfig::new("w", 32);
    assert_eq!(
        config.shared_memory_bytes(),
        0,
        "Single-warp softmax should need zero shared memory"
    );
}

#[test]
fn test_softmax_shared_memory_two_warps() {
    let config = PtxSoftmaxConfig::new("w2", 64);
    // 2 warps * 4 bytes (f32) = 8 bytes
    assert_eq!(config.shared_memory_bytes(), 8);
}

#[test]
fn test_attention_shared_memory_includes_scores_and_reduce() {
    let kv_seq_len = 128;
    let config = PtxAttentionConfig::new("a", 4, 64, 32, kv_seq_len);
    let smem = config.shared_memory_bytes();
    // scores[kv_seq_len] + reduce_buf[block_size], all f32
    let expected_min = kv_seq_len * 4; // at least the score buffer
    assert!(smem >= expected_min,
        "Attention shared memory must include at least score buffer: got {smem}, expected >= {expected_min}");
}

#[test]
fn test_transpose_ptx_shared_memory_padded() {
    let ptx = generate_transpose_ptx(64, 64);
    assert!(ptx.contains(".shared"), "Transpose must use shared memory");
    // Padding: TILE+1 stride to avoid bank conflicts
    let tile = TRANSPOSE_BLOCK_SIZE;
    let padded_stride = tile + 1;
    let total_smem = tile * padded_stride;
    assert!(
        ptx.contains(&format!("tile_smem[{total_smem}]")),
        "Shared memory should be padded to {total_smem} elements (tile * (tile+1))"
    );
}

// =========================================================================
// Section 6: Register Usage Validation
// =========================================================================

#[test]
fn test_softmax_ptx_declares_register_classes() {
    let ptx = emit_ptx_softmax(&PtxSoftmaxConfig::new("rk", 64)).unwrap();
    // PTX must declare register banks
    assert!(
        ptx.contains(".reg .u32"),
        "Must declare 32-bit integer registers"
    );
    assert!(
        ptx.contains(".reg .f32"),
        "Must declare 32-bit float registers"
    );
    assert!(
        ptx.contains(".reg .u64"),
        "Must declare 64-bit registers for pointers"
    );
    assert!(
        ptx.contains(".reg .pred"),
        "Must declare predicate registers"
    );
}

#[test]
fn test_reduction_ptx_declares_registers() {
    let ptx = generate_sum_ptx(256);
    assert!(
        ptx.contains(".reg .u32"),
        "Sum kernel must declare u32 registers"
    );
    assert!(
        ptx.contains(".reg .f32"),
        "Sum kernel must declare f32 registers"
    );
    assert!(
        ptx.contains(".reg .pred"),
        "Sum kernel must declare predicate registers"
    );
}

#[test]
fn test_add_ptx_declares_register_types() {
    let ptx = generate_add_ptx(128);
    assert!(
        ptx.contains(".reg .f32"),
        "Add kernel must declare f32 registers"
    );
    assert!(
        ptx.contains(".reg .u64"),
        "Add kernel must declare u64 registers for pointers"
    );
}

#[test]
fn test_softmax_ptx_register_count_bounded() {
    let ptx = emit_ptx_softmax(&PtxSoftmaxConfig::new("rc", 64)).unwrap();
    // Count register declarations — kernel should not use excessive registers
    // Typical: %r<N> for u32, %f<N> for f32, %rd<N> for u64
    // Check that register count per class is reasonable (< 32 per class)
    for prefix in [".reg .u32  %r<", ".reg .f32  %f<", ".reg .u64  %rd<"] {
        if let Some(pos) = ptx.find(prefix) {
            let after = &ptx[pos + prefix.len()..];
            if let Some(end) = after.find('>') {
                let count_str = &after[..end];
                if let Ok(count) = count_str.parse::<usize>() {
                    assert!(
                        count <= 32,
                        "Register class {prefix} declares {count} registers, should be <= 32"
                    );
                }
            }
        }
    }
}

// =========================================================================
// Section 7: Grid Dimension Calculation
// =========================================================================

#[test]
fn test_elementwise_launch_config_covers_all_elements() {
    let n: u32 = 1000;
    let (grid, block) = ptx_elementwise_launch_config(n);
    let total_threads = u64::from(grid[0]) * u64::from(block[0]);
    assert!(
        total_threads >= u64::from(n),
        "Grid must launch enough threads to cover all {n} elements: got {total_threads}"
    );
}

#[test]
fn test_elementwise_launch_config_exact_multiple() {
    let (grid, block) = ptx_elementwise_launch_config(512);
    assert_eq!(grid, [2, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_elementwise_launch_config_not_exact_multiple() {
    let (grid, block) = ptx_elementwise_launch_config(500);
    // ceil(500/256) = 2
    assert_eq!(grid, [2, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
    let total = grid[0] * block[0];
    assert!(total >= 500, "Must cover all 500 elements");
}

#[test]
fn test_softmax_launch_config_one_block_per_row() {
    let num_rows = 32;
    let dim = 64;
    let (grid, block) = ptx_softmax_launch_config(num_rows, dim);
    assert_eq!(grid[0], num_rows, "Softmax grid.x must equal num_rows");
    assert!(
        block[0] >= dim,
        "Block size must be >= dim (rounded to warp)"
    );
    assert_eq!(block[0] % WARP_SIZE, 0, "Block size must be warp-aligned");
}

#[test]
fn test_reduce_launch_config_single_block() {
    let (grid, block) = ptx_reduce_launch_config();
    assert_eq!(grid, [1, 1, 1], "Reduction should use single block");
    assert_eq!(block, [REDUCE_BLOCK_SIZE as usize, 1, 1]);
}

#[test]
fn test_transpose_launch_config_tiles_cover_matrix() {
    let rows = 100u32;
    let cols = 200u32;
    let (grid, block) = ptx_transpose_launch_config(rows, cols);
    let tile = TRANSPOSE_BLOCK_SIZE;
    // Grid must cover all tiles
    assert_eq!(grid[0], cols.div_ceil(tile));
    assert_eq!(grid[1], rows.div_ceil(tile));
    assert_eq!(block[0], tile);
    assert_eq!(block[1], tile);
}

#[test]
fn test_batch_transpose_launch_config_z_is_batch() {
    let rows = 64u32;
    let cols = 128u32;
    let batch = 4u32;
    // Signature is (batch, rows, cols); Grid.z == batch.
    let (grid, _block) = ptx_batch_transpose_launch_config(batch, rows, cols);
    assert_eq!(grid[2], batch, "Grid.z must equal batch size");
}

#[test]
fn test_attention_launch_config_dimensions() {
    let config = PtxAttentionConfig::new("lc", 8, 64, 32, 32);
    let batch_size = 2;
    let lc = ptx_attention_launch_config(&config, batch_size);
    // Grid: (batch, num_heads, seq_len)
    assert_eq!(lc.grid.x, batch_size as u32);
    assert_eq!(lc.grid.y, 8); // num_heads
    assert_eq!(lc.grid.z, 32); // seq_len
    assert_eq!(lc.block.x, config.block_size as u32);
}

#[test]
fn test_embedding_launch_config_covers_all_elements() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let num_tokens = 128;
    let (grid, block) = ptx_embedding_launch_config(num_tokens, &config);
    let total_elements = (num_tokens * config.embedding_dim) as u64;
    let total_threads = u64::from(grid) * u64::from(block);
    assert!(
        total_threads >= total_elements,
        "Embedding grid must cover all {total_elements} elements"
    );
}

#[test]
fn test_rope_launch_config_covers_all_pairs() {
    let config = PtxRopeConfig::new(128, 64);
    let (grid, block) = ptx_rope_launch_config(128, &config);
    let total_pairs = (128 * 64 / 2) as u64;
    let total_threads = u64::from(grid) * u64::from(block);
    assert!(
        total_threads >= total_pairs,
        "RoPE grid must cover all {total_pairs} dimension pairs"
    );
}

// =========================================================================
// Section 8: Embedding Lookup PTX Generation
// =========================================================================

#[test]
fn test_embedding_ptx_contains_global_kernel() {
    let config = PtxEmbeddingConfig::new(50257, 768);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("__global__"),
        "Embedding must be a __global__ CUDA kernel"
    );
    assert!(
        src.contains("embedding_lookup"),
        "Kernel must be named embedding_lookup"
    );
}

#[test]
fn test_embedding_ptx_has_bounds_check() {
    let config = PtxEmbeddingConfig::new(1000, 256);
    let src = generate_embedding_ptx(&config).unwrap();
    // Must bounds-check token ID against vocab_size
    assert!(
        src.contains("vocab_size") || src.contains("VOCAB"),
        "Embedding must reference vocab_size for bounds checking"
    );
}

#[test]
fn test_embedding_ptx_uses_restrict_pointers() {
    let config = PtxEmbeddingConfig::new(1000, 256);
    let src = generate_embedding_ptx(&config).unwrap();
    assert!(
        src.contains("__restrict__"),
        "Embedding should use __restrict__ for pointer aliasing optimization"
    );
}

#[test]
fn test_embedding_config_validation_zero_vocab_rejected() {
    let config = PtxEmbeddingConfig::new(0, 768);
    assert!(
        config.validate().is_err(),
        "Zero vocab_size should be rejected"
    );
}

#[test]
fn test_embedding_config_validation_zero_dim_rejected() {
    let config = PtxEmbeddingConfig::new(50257, 0);
    assert!(
        config.validate().is_err(),
        "Zero embedding_dim should be rejected"
    );
}

#[test]
fn test_embedding_reference_basic_lookup() {
    // Table: 3 words x 2 dims
    let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices: Vec<u32> = vec![0, 2, 1];
    let result = embedding_reference(&indices, &table, 2);
    assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
}

#[test]
fn test_embedding_reference_out_of_bounds_zeroed() {
    let table = vec![1.0, 2.0, 3.0, 4.0];
    let indices: Vec<u32> = vec![0, 5]; // index 5 is out of bounds for 2-row table
    let result = embedding_reference(&indices, &table, 2);
    // First row: normal lookup
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    // Second row: out of bounds → zero
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 0.0);
}

// =========================================================================
// Section 9: Transpose PTX (shared memory bank conflict avoidance)
// =========================================================================

#[test]
fn test_transpose_ptx_contains_shared_memory_declaration() {
    let ptx = generate_transpose_ptx(64, 128);
    assert!(
        ptx.contains(".shared .align 4 .f32 tile_smem"),
        "Transpose must declare aligned shared memory for tiles"
    );
}

#[test]
fn test_transpose_ptx_has_barrier_sync() {
    let ptx = generate_transpose_ptx(32, 32);
    assert!(
        ptx.contains("bar.sync"),
        "Transpose must synchronize after loading shared memory"
    );
}

#[test]
fn test_transpose_ptx_entry_name() {
    let ptx = generate_transpose_ptx(64, 64);
    assert!(
        ptx.contains(".entry ptx_transpose_f32"),
        "Transpose kernel entry must be ptx_transpose_f32"
    );
}

#[test]
fn test_transpose_ptx_bank_conflict_padding() {
    // Shared memory stride should be TILE+1 to avoid bank conflicts
    let tile = TRANSPOSE_BLOCK_SIZE;
    let ptx = generate_transpose_ptx(64, 64);
    // The stride (tile+1) ensures that column reads hit different banks
    let padded = tile * (tile + 1);
    assert!(
        ptx.contains(&format!("tile_smem[{padded}]")),
        "Shared memory must be padded to {padded} elements for bank conflict avoidance"
    );
}

#[test]
fn test_batch_transpose_ptx_has_batch_parameter() {
    let ptx = generate_batch_transpose_ptx(32, 64, 4);
    assert!(
        ptx.contains("param_batch") || ptx.contains("batch"),
        "Batched transpose must accept batch parameter"
    );
}

#[test]
fn test_transpose_reference_2x3() {
    // Input [2, 3]: [[1, 2, 3], [4, 5, 6]]
    // Output [3, 2]: [[1, 4], [2, 5], [3, 6]]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = transpose_reference(&input, 2, 3);
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_reference_identity_1x1() {
    let input = vec![42.0];
    let result = transpose_reference(&input, 1, 1);
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_batch_transpose_reference() {
    // Batch of 2 matrices, each [2, 2]
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // matrix 0
        5.0, 6.0, 7.0, 8.0, // matrix 1
    ];
    let result = batch_transpose_reference(&input, 2, 2, 2);
    // matrix 0 transposed: [1, 3, 2, 4]
    // matrix 1 transposed: [5, 7, 6, 8]
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
}

// =========================================================================
// Section 10: RoPE (Rotary Position Embedding) PTX Generation
// =========================================================================

#[test]
fn test_rope_ptx_contains_sincos_intrinsics() {
    let config = PtxRopeConfig::new(128, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(
        src.contains("__sinf") || src.contains("sinf") || src.contains("__sincosf"),
        "RoPE must compute sin/cos for rotation"
    );
    assert!(
        src.contains("__cosf") || src.contains("cosf") || src.contains("__sincosf"),
        "RoPE must compute cos for rotation"
    );
}

#[test]
fn test_rope_ptx_has_theta_computation() {
    let config = PtxRopeConfig::new(64, 32);
    let src = generate_rope_ptx(&config).unwrap();
    // theta = pos / base^(2i / head_dim)
    assert!(
        src.contains("theta") || src.contains("10000") || src.contains("base"),
        "RoPE must compute theta from position and base frequency"
    );
}

#[test]
fn test_rope_ptx_is_cuda_kernel() {
    let config = PtxRopeConfig::new(128, 64);
    let src = generate_rope_ptx(&config).unwrap();
    assert!(
        src.contains("__global__"),
        "RoPE must be a __global__ CUDA kernel"
    );
    assert!(src.contains("rope_apply"), "Kernel name must be rope_apply");
}

#[test]
fn test_rope_cached_ptx_reads_precomputed_tables() {
    let config = PtxRopeConfig::new(128, 64);
    let src = generate_rope_cached_ptx(&config).unwrap();
    assert!(src.contains("__global__"), "Cached RoPE must be a kernel");
    // Cached version reads sin/cos from device memory
    assert!(
        src.contains("cos_table") || src.contains("cos_cache") || src.contains("cos_buf"),
        "Cached RoPE must read from precomputed cos table"
    );
    assert!(
        src.contains("sin_table") || src.contains("sin_cache") || src.contains("sin_buf"),
        "Cached RoPE must read from precomputed sin table"
    );
}

#[test]
fn test_rope_config_validation_odd_head_dim_rejected() {
    let config = PtxRopeConfig::new(128, 63); // odd head_dim
    assert!(
        config.validate().is_err(),
        "RoPE must reject odd head_dim (dimension pairs require even)"
    );
}

#[test]
fn test_rope_config_validation_zero_seq_len_rejected() {
    let config = PtxRopeConfig::new(0, 64);
    assert!(
        config.validate().is_err(),
        "Zero seq_len should be rejected"
    );
}

#[test]
fn test_rope_config_validation_zero_head_dim_rejected() {
    let config = PtxRopeConfig::new(128, 0);
    assert!(
        config.validate().is_err(),
        "Zero head_dim should be rejected"
    );
}

#[test]
fn test_rope_config_custom_base() {
    let config = PtxRopeConfig::new(128, 64).with_base(500000.0);
    assert!((config.base - 500000.0).abs() < 1.0);
    assert!(config.validate().is_ok());
}

#[test]
fn test_rope_config_negative_base_rejected() {
    let config = PtxRopeConfig::new(128, 64).with_base(-1.0);
    assert!(
        config.validate().is_err(),
        "Negative base should be rejected"
    );
}

#[test]
fn test_rope_config_nan_base_rejected() {
    let config = PtxRopeConfig::new(128, 64).with_base(f32::NAN);
    assert!(config.validate().is_err(), "NaN base should be rejected");
}

#[test]
fn test_rope_config_inf_base_rejected() {
    let config = PtxRopeConfig::new(128, 64).with_base(f32::INFINITY);
    assert!(
        config.validate().is_err(),
        "Infinite base should be rejected"
    );
}

#[test]
fn test_rope_reference_preserves_norm() {
    // RoPE is a rotation, so it should approximately preserve the L2 norm
    let head_dim = 8;
    let seq_len = 2;
    let x: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
    let result = rope_reference(&x, seq_len, head_dim);
    // Check norm preservation per position
    for pos in 0..seq_len {
        let start = pos * head_dim;
        let end = start + head_dim;
        let input_norm: f32 = x[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        let output_norm: f32 = result[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel_diff = (input_norm - output_norm).abs() / input_norm.max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "RoPE should preserve L2 norm at pos {pos}: input={input_norm}, output={output_norm}"
        );
    }
}

#[test]
fn test_rope_reference_zero_position_is_identity() {
    // At position 0, theta=0 for all dims, so cos(0)=1, sin(0)=0 → identity
    let head_dim = 4;
    let x = vec![1.0, 2.0, 3.0, 4.0]; // 1 position
    let result = rope_reference(&x, 1, head_dim);
    for i in 0..head_dim {
        assert!(
            (result[i] - x[i]).abs() < 1e-5,
            "RoPE at position 0 should be identity: x[{i}]={}, result[{i}]={}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_rope_reference_custom_base() {
    let head_dim = 4;
    let x = vec![1.0, 0.0, 0.0, 1.0]; // 1 position
    let _result_default = rope_reference_with_base(&x, 1, head_dim, 10000.0);
    let _result_custom = rope_reference_with_base(&x, 1, head_dim, 500.0);
    // Different base → different rotation angles
    // At position 0, both should be identity, so use 2 positions
    let x2 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 2 positions
    let r1 = rope_reference_with_base(&x2, 2, head_dim, 10000.0);
    let r2 = rope_reference_with_base(&x2, 2, head_dim, 500.0);
    // Position 1 should differ between bases
    let pos1_start = head_dim;
    let differs = (0..head_dim).any(|i| (r1[pos1_start + i] - r2[pos1_start + i]).abs() > 1e-6);
    assert!(
        differs,
        "Different base frequencies should produce different rotations at position > 0"
    );
}

// =========================================================================
// Bonus: Cross-cutting PTX structural validation
// =========================================================================

#[test]
fn test_all_ptx_kernels_have_version_directive() {
    // Every raw PTX kernel must start with version + target
    let ptx_outputs = [generate_add_ptx(64),
        generate_mul_ptx(64),
        generate_sum_ptx(64),
        generate_transpose_ptx(32, 32)];
    for (i, ptx) in ptx_outputs.iter().enumerate() {
        assert!(
            ptx.contains(&format!(".version {PTX_VERSION}")),
            "PTX kernel {i} missing .version directive"
        );
        assert!(
            ptx.contains(".target"),
            "PTX kernel {i} missing .target directive"
        );
        assert!(
            ptx.contains(".address_size 64"),
            "PTX kernel {i} missing .address_size 64"
        );
    }
}

#[test]
fn test_softmax_and_log_softmax_both_generate() {
    let sm = generate_softmax_ptx(false, 64);
    let lsm = generate_log_softmax_ptx(64);
    assert!(sm.contains(".entry"), "Softmax must generate a PTX entry");
    assert!(
        lsm.contains(".entry"),
        "Log-softmax must generate a PTX entry"
    );
    // They should be different kernels
    assert_ne!(sm, lsm, "Softmax and log-softmax should differ");
}

#[test]
fn test_activation_launch_config_single_element() {
    let (grid, block) = ptx_activation_launch_config(1, 256);
    assert_eq!(grid, [1, 1, 1], "Single element needs exactly 1 block");
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_softmax_reference_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax output must sum to 1.0, got {sum}"
    );
}

#[test]
fn test_softmax_reference_monotonic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = softmax_reference(&input);
    for i in 0..output.len() - 1 {
        assert!(
            output[i] < output[i + 1],
            "Softmax of sorted input should be monotonically increasing"
        );
    }
}

#[test]
fn test_log_softmax_reference_negative() {
    let input = vec![1.0, 2.0, 3.0];
    let output = log_softmax_reference(&input);
    for (i, v) in output.iter().enumerate() {
        assert!(
            *v <= 0.0,
            "Log-softmax values must be <= 0, got {v} at index {i}"
        );
    }
}
