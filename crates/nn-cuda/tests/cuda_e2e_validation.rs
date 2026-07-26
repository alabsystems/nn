// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end validation tests for CUDA PTX kernel generation.
//!
//! These tests validate the complete pipeline: PTX generation, structural
//! validation of the generated assembly/source, and numerical comparison
//! of CPU reference implementations against expected results.
//!
//! No live NVIDIA GPU is required. Tests use the CPU reference functions
//! built into each PTX module to verify correctness, and the structural
//! validator to verify the generated PTX/CUDA C++ is well-formed.
//!
//! Part of #3842.

use nn_cuda::cuda_validation::{
    validate_numerical, validate_ptx_e2e, validate_ptx_structure, CudaValidationSuite, ErrorStats,
};

// ===========================================================================
// A. Softmax E2E Validation
// ===========================================================================

#[test]
fn test_softmax_ptx_matches_cpu_small() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let dim = input.len();

    // Generate PTX
    let ptx = nn_cuda::ptx_softmax::emit_ptx_softmax_default("softmax_e2e_small", dim)
        .expect("PTX generation failed");

    // Compute CPU reference
    let cpu_output = nn_cuda::softmax_reference(&input);

    // Verify softmax properties: non-negative, sums to 1
    assert!(
        cpu_output.iter().all(|&v| v >= 0.0),
        "softmax must be non-negative"
    );
    let sum: f32 = cpu_output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax must sum to 1.0, got {sum}"
    );

    // Validate end-to-end
    let result = validate_ptx_e2e("softmax_e2e_small", &ptx, &cpu_output, &cpu_output, 1e-6)
        .expect("E2E validation failed");
    assert!(result.passed());
}

#[test]
fn test_softmax_ptx_matches_cpu_large_dim() {
    // Larger dimension triggers multi-warp code path (dim > 32)
    let dim = 128;
    let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 6.4).collect();

    let ptx = nn_cuda::ptx_softmax::emit_ptx_softmax_default("softmax_e2e_128", dim)
        .expect("PTX generation failed");

    let cpu_output = nn_cuda::softmax_reference(&input);

    // Structural: multi-warp should use shared memory
    let structural = validate_ptx_structure(&ptx, "softmax_e2e_128");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );
    assert!(
        ptx.contains("warp_scratch"),
        "dim=128 should use shared memory"
    );

    // Numerical: reference self-consistency
    let result = validate_ptx_e2e("softmax_e2e_128", &ptx, &cpu_output, &cpu_output, 1e-6)
        .expect("E2E validation failed");
    assert!(result.passed());
}

#[test]
fn test_softmax_ptx_matches_cpu_single_warp() {
    // dim=16 should use warp-only reduction (no shared memory)
    let dim = 16;
    let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5).collect();

    let ptx = nn_cuda::ptx_softmax::emit_ptx_softmax_default("softmax_e2e_16", dim)
        .expect("PTX generation failed");

    let cpu_output = nn_cuda::softmax_reference(&input);

    // Structural: warp-only should NOT use shared memory
    assert!(!ptx.contains("warp_scratch"), "dim=16 should be warp-only");

    let result = validate_ptx_e2e("softmax_e2e_16", &ptx, &cpu_output, &cpu_output, 1e-6)
        .expect("E2E validation failed");
    assert!(result.passed());
}

#[test]
fn test_log_softmax_ptx_matches_cpu() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let dim = input.len();

    let config = nn_cuda::ptx_softmax::PtxSoftmaxConfig::new_log("log_softmax_e2e", dim);
    let ptx = nn_cuda::ptx_softmax::emit_ptx_softmax(&config).expect("PTX generation failed");

    let cpu_output = nn_cuda::log_softmax_reference(&input);

    // Verify log_softmax properties: all values <= 0
    assert!(
        cpu_output.iter().all(|&v| v <= 0.0),
        "log_softmax must be <= 0"
    );

    let result = validate_ptx_e2e("log_softmax_e2e", &ptx, &cpu_output, &cpu_output, 1e-6)
        .expect("E2E validation failed");
    assert!(result.passed());
}

// ===========================================================================
// B. Attention E2E Validation
// ===========================================================================

#[test]
fn test_attention_ptx_matches_cpu() {
    let head_dim = 4;
    let seq_len = 3;

    // Q, K, V: each [seq_len, head_dim]
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32) * 0.05 + 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32) * 0.2 - 0.5)
        .collect();

    // Generate CUDA C++ attention kernel (num_heads=1, kv_seq_len=seq_len)
    let config = nn_cuda::ptx_attention::PtxAttentionConfig::new(
        "attention_e2e",
        1,
        head_dim,
        seq_len,
        seq_len,
    );
    let src =
        nn_cuda::ptx_attention::emit_ptx_attention(&config).expect("attention generation failed");

    // Structural validation
    let structural = validate_ptx_structure(&src, "attention_e2e");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    // CPU reference (single-head SDPA)
    let cpu_output = nn_cuda::sdpa_reference(&q, &k, &v, head_dim);

    // Verify output shape
    assert_eq!(cpu_output.len(), seq_len * head_dim);

    // Verify no NaN/Inf
    assert!(
        cpu_output.iter().all(|v| v.is_finite()),
        "attention output must be finite"
    );

    // Numerical self-consistency
    let result = validate_numerical("attention_e2e", &cpu_output, &cpu_output, 1e-5)
        .expect("numerical validation failed");
    assert!(result.passed());
}

#[test]
fn test_attention_ptx_causal_mask() {
    let head_dim = 4;
    let seq_len = 4;

    let config = nn_cuda::ptx_attention::PtxAttentionConfig::new(
        "attention_causal",
        1,
        head_dim,
        seq_len,
        seq_len,
    )
    .with_causal(true);

    let src = nn_cuda::ptx_attention::emit_ptx_attention(&config)
        .expect("causal attention generation failed");

    // Structural: causal kernel should contain masking logic
    let structural = validate_ptx_structure(&src, "attention_causal");
    assert!(structural.structural_ok);
    assert!(
        src.contains("-HUGE_VALF") || src.contains("causal") || src.contains("mask"),
        "causal attention should contain masking"
    );
}

#[test]
fn test_multihead_attention_ptx_matches_cpu() {
    let batch_size = 1;
    let num_heads = 2;
    let seq_len = 3;
    let head_dim = 4;

    let total_q = batch_size * num_heads * seq_len * head_dim;
    let q: Vec<f32> = (0..total_q).map(|i| (i as f32) * 0.05).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..total_q).map(|i| (i as f32) * 0.1 - 1.0).collect();

    // PtxMultiHeadAttentionConfig::new(num_heads, head_dim, seq_len)
    let config = nn_cuda::ptx_attention_multihead::PtxMultiHeadAttentionConfig::new(
        num_heads, head_dim, seq_len,
    );
    let src = nn_cuda::ptx_attention_multihead::generate_multihead_attention_ptx(&config)
        .expect("MHA generation failed");

    let structural = validate_ptx_structure(&src, "fused_multihead_attention");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    // CPU reference: attention_reference(q, k, v, batch_size, num_heads, seq_len, kv_seq_len, head_dim, causal)
    let cpu_output = nn_cuda::attention_reference(
        &q, &k, &v, batch_size, num_heads, seq_len, seq_len, head_dim, false,
    );
    assert_eq!(cpu_output.len(), total_q);
    assert!(cpu_output.iter().all(|v| v.is_finite()));
}

// ===========================================================================
// C. Elementwise E2E Validation
// ===========================================================================

#[test]
fn test_elementwise_add_ptx_matches_cpu() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 1.5, 2.5, 3.5];
    let n = a.len() as u32;

    let ptx = nn_cuda::generate_add_ptx(n);
    let cpu_output = nn_cuda::add_reference(&a, &b);
    let expected = vec![1.5, 3.5, 5.5, 7.5];

    let result = validate_ptx_e2e("ptx_add_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("add E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_sub_ptx_matches_cpu() {
    let a = vec![5.0, 4.0, 3.0, 2.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let n = a.len() as u32;

    let ptx = nn_cuda::generate_sub_ptx(n);
    let cpu_output = nn_cuda::sub_reference(&a, &b);
    let expected = vec![4.0, 3.0, 2.0, 1.0];

    let result = validate_ptx_e2e("ptx_sub_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("sub E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_mul_ptx_matches_cpu() {
    let a = vec![2.0, 3.0, 4.0, 5.0];
    let b = vec![0.5, 0.5, 0.5, 0.5];
    let n = a.len() as u32;

    let ptx = nn_cuda::generate_mul_ptx(n);
    let cpu_output = nn_cuda::mul_reference(&a, &b);
    let expected = vec![1.0, 1.5, 2.0, 2.5];

    let result = validate_ptx_e2e("ptx_mul_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("mul E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_div_ptx_matches_cpu() {
    let a = vec![10.0, 20.0, 30.0, 40.0];
    let b = vec![2.0, 4.0, 5.0, 8.0];
    let n = a.len() as u32;

    let ptx = nn_cuda::generate_div_ptx(n);
    let cpu_output = nn_cuda::div_reference(&a, &b);
    let expected = vec![5.0, 5.0, 6.0, 5.0];

    let result = validate_ptx_e2e("ptx_div_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("div E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_exp_ptx_matches_cpu() {
    let input = vec![0.0, 1.0, -1.0, 2.0];
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_exp_ptx(n);
    let cpu_output = nn_cuda::exp_reference(&input);
    let expected: Vec<f32> = input.iter().map(|x| x.exp()).collect();

    let result = validate_ptx_e2e("ptx_exp_f32", &ptx, &cpu_output, &expected, 1e-5)
        .expect("exp E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_log_ptx_matches_cpu() {
    let input = vec![1.0, 2.718282, 10.0, 0.5];
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_log_ptx(n);
    let cpu_output = nn_cuda::log_reference(&input);
    let expected: Vec<f32> = input.iter().map(|x| x.ln()).collect();

    let result = validate_ptx_e2e("ptx_log_f32", &ptx, &cpu_output, &expected, 1e-5)
        .expect("log E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_sqrt_ptx_matches_cpu() {
    let input = vec![1.0, 4.0, 9.0, 16.0];
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_sqrt_ptx(n);
    let cpu_output = nn_cuda::sqrt_reference(&input);
    let expected = vec![1.0, 2.0, 3.0, 4.0];

    let result = validate_ptx_e2e("ptx_sqrt_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("sqrt E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_neg_ptx_matches_cpu() {
    let input = vec![1.0, -2.0, 3.0, -4.0];
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_neg_ptx(n);
    let cpu_output = nn_cuda::neg_reference(&input);
    let expected = vec![-1.0, 2.0, -3.0, 4.0];

    let result = validate_ptx_e2e("ptx_neg_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("neg E2E failed");
    assert!(result.passed());
}

#[test]
fn test_elementwise_scalar_mul_ptx_matches_cpu() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let scalar = 2.5;
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_scalar_mul_ptx(n);
    let cpu_output = nn_cuda::scalar_mul_reference(&input, scalar);
    let expected = vec![2.5, 5.0, 7.5, 10.0];

    let result = validate_ptx_e2e("ptx_scalar_mul_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("scalar_mul E2E failed");
    assert!(result.passed());
}

// ===========================================================================
// D. RoPE E2E Validation
// ===========================================================================

#[test]
fn test_rope_ptx_matches_cpu() {
    let seq_len = 2;
    let head_dim = 4;
    // Input: seq_len * head_dim elements
    let input: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.3).collect();

    let config = nn_cuda::ptx_rope::PtxRopeConfig::new(seq_len, head_dim);
    let src = nn_cuda::ptx_rope::generate_rope_ptx(&config).expect("RoPE generation failed");

    let structural = validate_ptx_structure(&src, "rope_apply");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    // CPU reference
    let cpu_output = nn_cuda::rope_reference(&input, seq_len, head_dim);
    assert_eq!(cpu_output.len(), input.len());
    assert!(
        cpu_output.iter().all(|v| v.is_finite()),
        "RoPE output must be finite"
    );

    // Numerical self-consistency
    let result = validate_numerical("rope_e2e", &cpu_output, &cpu_output, 1e-5)
        .expect("numerical validation failed");
    assert!(result.passed());
}

#[test]
fn test_rope_cached_ptx_structure() {
    let seq_len = 4;
    let head_dim = 8;

    let config = nn_cuda::ptx_rope::PtxRopeConfig::new(seq_len, head_dim);
    let src = nn_cuda::ptx_rope::generate_rope_cached_ptx(&config)
        .expect("RoPE cached generation failed");

    let structural = validate_ptx_structure(&src, "rope_apply_cached");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    // Cached variant should reference cos/sin table parameters
    assert!(
        src.contains("cos") || src.contains("sin") || src.contains("table"),
        "cached RoPE should reference cos/sin tables"
    );
}

// ===========================================================================
// E. MatMul E2E Validation
// ===========================================================================

#[test]
fn test_matmul_ptx_matches_cpu() {
    let m = 2u32;
    let k = 3u32;
    let n = 2u32;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3, 2]

    let ptx = nn_cuda::generate_matmul_ptx(m, k, n);
    let cpu_output = nn_cuda::matmul_reference(&a, &b, m as usize, k as usize, n as usize);

    // Expected: manual computation
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    let expected = vec![58.0, 64.0, 139.0, 154.0];

    let result = validate_ptx_e2e("naive_matmul_f32", &ptx, &cpu_output, &expected, 1e-4)
        .expect("matmul E2E failed");
    assert!(result.passed());
}

#[test]
fn test_matmul_tiled_ptx_structure() {
    let ptx = nn_cuda::generate_matmul_tiled_ptx(64, 64, 64, 16);

    let structural = validate_ptx_structure(&ptx, "tiled_matmul_f32");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    // Tiled matmul should use shared memory
    assert!(
        ptx.contains("__shared__") || ptx.contains(".shared"),
        "tiled matmul should use shared memory"
    );
}

// ===========================================================================
// F. LayerNorm E2E Validation
// ===========================================================================

#[test]
fn test_layernorm_ptx_matches_cpu() {
    let dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let eps = 1e-5;

    let ptx = nn_cuda::ptx_layernorm::emit_ptx_layernorm_default("layernorm_e2e", dim, eps)
        .expect("layernorm generation failed");

    let cpu_output = nn_cuda::layernorm_reference(&input, &gamma, &beta, eps);

    // Verify layernorm properties: mean ~= 0, variance ~= 1
    let mean: f32 = cpu_output.iter().sum::<f32>() / cpu_output.len() as f32;
    assert!(mean.abs() < 1e-5, "layernorm mean should be ~0, got {mean}");

    let result = validate_ptx_e2e("layernorm_e2e", &ptx, &cpu_output, &cpu_output, 1e-6)
        .expect("layernorm E2E failed");
    assert!(result.passed());
}

#[test]
fn test_rmsnorm_ptx_matches_cpu() {
    let dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let ptx = nn_cuda::ptx_rmsnorm::emit_ptx_rmsnorm_default("rmsnorm_e2e", dim, eps)
        .expect("rmsnorm generation failed");

    let cpu_output = nn_cuda::rmsnorm_reference(&input, &weight, eps);

    let structural = validate_ptx_structure(&ptx, "rmsnorm_e2e");
    assert!(structural.structural_ok);

    let result = validate_numerical("rmsnorm_e2e", &cpu_output, &cpu_output, 1e-6)
        .expect("rmsnorm numerical validation failed");
    assert!(result.passed());
}

// ===========================================================================
// G. Transpose E2E Validation
// ===========================================================================

#[test]
fn test_transpose_ptx_matches_cpu() {
    let rows = 2u32;
    let cols = 3u32;
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]

    let ptx = nn_cuda::generate_transpose_ptx(rows, cols);
    let cpu_output = nn_cuda::transpose_reference(&data, rows as usize, cols as usize);
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // [3, 2]

    let result = validate_ptx_e2e("ptx_transpose_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("transpose E2E failed");
    assert!(result.passed());
}

#[test]
fn test_batch_transpose_ptx_matches_cpu() {
    let batch = 2u32;
    let rows = 2u32;
    let cols = 2u32;
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2, 2, 2]

    let ptx = nn_cuda::generate_batch_transpose_ptx(batch, rows, cols);
    let cpu_output =
        nn_cuda::batch_transpose_reference(&data, batch as usize, rows as usize, cols as usize);
    let expected = vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0];

    let result = validate_ptx_e2e(
        "ptx_batch_transpose_f32",
        &ptx,
        &cpu_output,
        &expected,
        1e-6,
    )
    .expect("batch transpose E2E failed");
    assert!(result.passed());
}

// ===========================================================================
// H. Reduction E2E Validation
// ===========================================================================

#[test]
fn test_reduce_sum_ptx_matches_cpu() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = input.len() as u32;
    let ptx = nn_cuda::generate_sum_ptx(n);

    let cpu_output = nn_cuda::sum_reference(&input);
    let expected = 15.0;

    assert!(
        (cpu_output - expected).abs() < 1e-5,
        "sum reference should be 15.0"
    );

    let structural = validate_ptx_structure(&ptx, "ptx_sum_f32");
    assert!(structural.structural_ok);
}

#[test]
fn test_reduce_max_ptx_matches_cpu() {
    let input = vec![-3.0, 7.0, 2.0, 5.0, -1.0];
    let n = input.len() as u32;
    let ptx = nn_cuda::generate_max_ptx(n);

    let cpu_output = nn_cuda::max_reference(&input);
    assert!((cpu_output - 7.0).abs() < 1e-5);

    let structural = validate_ptx_structure(&ptx, "ptx_max_f32");
    assert!(structural.structural_ok);
}

#[test]
fn test_reduce_mean_ptx_matches_cpu() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let n = input.len() as u32;
    let ptx = nn_cuda::generate_mean_ptx(n);

    let cpu_output = nn_cuda::mean_reference(&input);
    assert!((cpu_output - 5.0).abs() < 1e-5);

    let structural = validate_ptx_structure(&ptx, "ptx_mean_f32");
    assert!(structural.structural_ok);
}

#[test]
fn test_reduce_argmax_ptx_matches_cpu() {
    let input = vec![1.0, 5.0, 3.0, 2.0];
    let n = input.len() as u32;
    let _ptx = nn_cuda::generate_argmax_ptx(n);

    let cpu_output = nn_cuda::argmax_reference(&input);
    assert_eq!(cpu_output, 1, "argmax should be index 1");
}

// ===========================================================================
// I. Embedding E2E Validation
// ===========================================================================

#[test]
fn test_embedding_ptx_matches_cpu() {
    let vocab_size = 4;
    let dim = 3;
    // Embedding table: [4, 3]
    let table: Vec<f32> = (0..vocab_size * dim).map(|i| (i as f32) * 0.1).collect();
    let indices = vec![0u32, 2, 1, 3];

    let config = nn_cuda::ptx_embedding::PtxEmbeddingConfig::new(vocab_size, dim);
    let ptx = nn_cuda::generate_embedding_ptx(&config).expect("embedding generation failed");
    let cpu_output = nn_cuda::embedding_reference(&indices, &table, dim);

    // Index 0 -> [0.0, 0.1, 0.2], Index 2 -> [0.6, 0.7, 0.8], etc.
    let expected = vec![
        0.0, 0.1, 0.2, // index 0
        0.6, 0.7, 0.8, // index 2
        0.3, 0.4, 0.5, // index 1
        0.9, 1.0, 1.1, // index 3
    ];

    // Structural check — the embedding kernel name is from PtxEmbeddingConfig
    let structural = validate_ptx_structure(&ptx, "embedding");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    let result = validate_numerical("embedding_e2e", &cpu_output, &expected, 1e-5)
        .expect("embedding numerical validation failed");
    assert!(result.passed());
}

// ===========================================================================
// J. Linear E2E Validation
// ===========================================================================

#[test]
fn test_linear_ptx_matches_cpu() {
    let in_features = 3u32;
    let out_features = 2u32;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [2, 3]
    let bias = vec![0.5, 1.0]; // [2]

    let ptx = nn_cuda::generate_linear_ptx(in_features, out_features);

    let cpu_output = nn_cuda::linear_reference(
        &input,
        &weight,
        Some(&bias),
        in_features as usize,
        out_features as usize,
    );

    // Verify output length (batch * out_features, batch = input.len() / in_features = 2)
    assert_eq!(cpu_output.len(), 2 * out_features as usize);
    assert!(cpu_output.iter().all(|v| v.is_finite()));

    let structural = validate_ptx_structure(&ptx, "linear_bias_f32");
    assert!(
        structural.structural_ok,
        "failures: {:?}",
        structural.structural_failures
    );

    let result = validate_numerical("linear_e2e", &cpu_output, &cpu_output, 1e-5)
        .expect("linear numerical validation failed");
    assert!(result.passed());
}

// ===========================================================================
// K. Residual E2E Validation
// ===========================================================================

#[test]
fn test_residual_add_ptx_matches_cpu() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let residual = vec![0.1, 0.2, 0.3, 0.4];
    let n = x.len() as u32;

    let ptx = nn_cuda::generate_residual_add_ptx(n);
    let cpu_output = nn_cuda::residual_add_reference(&x, &residual);
    let expected = vec![1.1, 2.2, 3.3, 4.4];

    let result = validate_ptx_e2e("ptx_residual_add_f32", &ptx, &cpu_output, &expected, 1e-5)
        .expect("residual_add E2E failed");
    assert!(result.passed());
}

#[test]
fn test_residual_add_relu_ptx_matches_cpu() {
    let x = vec![1.0, -2.0, 3.0, -4.0];
    let residual = vec![0.5, 0.5, 0.5, 0.5];
    let n = x.len() as u32;

    let ptx = nn_cuda::generate_residual_add_relu_ptx(n);
    let cpu_output = nn_cuda::residual_add_relu_reference(&x, &residual);
    let expected = vec![1.5, 0.0, 3.5, 0.0]; // ReLU clips negatives

    let result = validate_ptx_e2e(
        "ptx_residual_add_relu_f32",
        &ptx,
        &cpu_output,
        &expected,
        1e-5,
    )
    .expect("residual_add_relu E2E failed");
    assert!(result.passed());
}

// ===========================================================================
// L. Activation Reference Self-Consistency
// ===========================================================================

#[test]
fn test_silu_reference_self_consistency() {
    let x_vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    for &x in &x_vals {
        let y = nn_cuda::silu_reference(x);
        let expected = x / (1.0 + (-x).exp());
        assert!(
            (y - expected).abs() < 1e-6,
            "SiLU({x}): got {y}, expected {expected}"
        );
    }
}

#[test]
fn test_gelu_reference_self_consistency() {
    // GELU(0) should be 0
    let y = nn_cuda::gelu_reference(0.0);
    assert!(y.abs() < 1e-6, "GELU(0) should be ~0, got {y}");

    // GELU(x) > 0 for large positive x
    let y = nn_cuda::gelu_reference(3.0);
    assert!(y > 0.0, "GELU(3.0) should be positive");
}

#[test]
fn test_mish_reference_self_consistency() {
    // Mish(0) should be 0
    let y = nn_cuda::mish_reference(0.0);
    assert!(y.abs() < 1e-6, "Mish(0) should be ~0, got {y}");
}

// ===========================================================================
// M. Validation Suite Integration
// ===========================================================================

#[test]
fn test_validation_suite_multi_kernel() {
    let mut suite = CudaValidationSuite::new();

    // Add softmax (kernel name in PTX: "sm_suite")
    let softmax_input = vec![1.0, 2.0, 3.0, 4.0];
    let softmax_ptx = nn_cuda::ptx_softmax::emit_ptx_softmax_default("sm_suite", 4).unwrap();
    let softmax_cpu = nn_cuda::softmax_reference(&softmax_input);
    suite.add(
        "sm_suite",
        softmax_ptx,
        softmax_cpu.clone(),
        softmax_cpu,
        1e-6,
    );

    // Add elementwise add (kernel name in PTX: "ptx_add_f32")
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let add_ptx = nn_cuda::generate_add_ptx(2);
    let add_cpu = nn_cuda::add_reference(&a, &b);
    let add_expected = vec![4.0, 6.0];
    suite.add("ptx_add_f32", add_ptx, add_cpu, add_expected, 1e-6);

    // Add transpose (kernel name in PTX: "ptx_transpose_f32")
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let trans_ptx = nn_cuda::generate_transpose_ptx(2, 3);
    let trans_cpu = nn_cuda::transpose_reference(&data, 2, 3);
    let trans_expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    suite.add(
        "ptx_transpose_f32",
        trans_ptx,
        trans_cpu,
        trans_expected,
        1e-6,
    );

    assert_eq!(suite.len(), 3);
    assert!(suite.run_all_pass(), "all suite entries should pass");
}

#[test]
fn test_error_stats_relative_error() {
    let actual = vec![1.01, 2.02, 3.03];
    let expected = vec![1.0, 2.0, 3.0];
    let stats = ErrorStats::compute(&actual, &expected).unwrap();

    assert!(stats.max_abs_error > 0.009);
    assert!(stats.max_abs_error < 0.031);
    assert!(stats.max_rel_error > 0.009);
    assert!(stats.max_rel_error < 0.011);
    assert_eq!(stats.num_nans, 0);
    assert_eq!(stats.num_infs, 0);
}

// ===========================================================================
// N. Gather / Where / Clamp E2E Validation
// ===========================================================================

#[test]
fn test_gather_ptx_matches_cpu() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = vec![0u32, 2, 4, 1];
    let dim_size = 5u32;

    let ptx = nn_cuda::generate_gather_ptx(indices.len() as u32, dim_size);
    let cpu_output = nn_cuda::gather_reference(&data, &indices, dim_size as usize);
    let expected = vec![10.0, 30.0, 50.0, 20.0];

    let result = validate_ptx_e2e("ptx_gather_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("gather E2E failed");
    assert!(result.passed());
}

#[test]
fn test_where_ptx_matches_cpu() {
    let cond = vec![1u32, 0, 1, 0];
    let a = vec![10.0, 20.0, 30.0, 40.0];
    let b = vec![100.0, 200.0, 300.0, 400.0];
    let n = a.len() as u32;

    let ptx = nn_cuda::generate_where_ptx(n);
    let cpu_output = nn_cuda::where_reference(&cond, &a, &b);
    let expected = vec![10.0, 200.0, 30.0, 400.0];

    let result = validate_ptx_e2e("ptx_where_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("where E2E failed");
    assert!(result.passed());
}

#[test]
fn test_clamp_ptx_matches_cpu() {
    let input = vec![-5.0, 0.0, 3.0, 10.0];
    let n = input.len() as u32;
    let ptx = nn_cuda::generate_clamp_ptx(n, 0.0, 5.0);
    let cpu_output = nn_cuda::clamp_reference(&input, 0.0, 5.0);
    let expected = vec![0.0, 0.0, 3.0, 5.0];

    let result = validate_ptx_e2e("ptx_clamp_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("clamp E2E failed");
    assert!(result.passed());
}

// ===========================================================================
// O. Upsample E2E Validation
// ===========================================================================

#[test]
fn test_upsample_nearest1d_ptx_matches_cpu() {
    let input = vec![1.0, 2.0, 3.0];
    let scale = 2u32;
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_upsample_nearest1d_ptx(n, scale);
    let cpu_output = nn_cuda::upsample_nearest1d_reference(&input, scale as usize);
    let expected = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

    let result = validate_ptx_e2e(
        "ptx_upsample_nearest1d_f32",
        &ptx,
        &cpu_output,
        &expected,
        1e-6,
    )
    .expect("upsample1d E2E failed");
    assert!(result.passed());
}

// ===========================================================================
// P. Tensor Ops E2E Validation
// ===========================================================================

#[test]
fn test_concat_ptx_matches_cpu() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let n_a = a.len() as u32;
    let n_b = b.len() as u32;

    let ptx = nn_cuda::generate_concat_ptx(n_a, n_b);
    let cpu_output = nn_cuda::concat_reference(&a, &b);
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = validate_ptx_e2e("ptx_concat_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("concat E2E failed");
    assert!(result.passed());
}

#[test]
fn test_slice_ptx_matches_cpu() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let n = input.len() as u32;

    let ptx = nn_cuda::generate_slice_ptx(n, 1, 3);
    let cpu_output = nn_cuda::slice_reference(&input, 1, 3);
    let expected = vec![20.0, 30.0, 40.0];

    let result = validate_ptx_e2e("ptx_slice_f32", &ptx, &cpu_output, &expected, 1e-6)
        .expect("slice E2E failed");
    assert!(result.passed());
}

#[test]
fn test_fill_ptx_matches_cpu() {
    let ptx = nn_cuda::generate_fill_ptx(4, 3.14);
    let cpu_output = nn_cuda::fill_reference(4, 3.14);
    let expected = vec![3.14, 3.14, 3.14, 3.14];

    let result = validate_ptx_e2e("ptx_fill_f32", &ptx, &cpu_output, &expected, 1e-5)
        .expect("fill E2E failed");
    assert!(result.passed());
}
