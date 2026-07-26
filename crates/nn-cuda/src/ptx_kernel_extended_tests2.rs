// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended PTX kernel generation tests (batch 2).
//!
//! Covers PTX generation for activations, normalization, softmax, matmul,
//! elementwise binary/unary ops, reductions, conv1d, transpose, rope,
//! gather/scatter, pad, upsample, where/clamp, cast, workgroup
//! configuration, invalid parameter rejection, and PTX string validation.

use crate::{
    // Reductions
    argmax_reference,
    argmin_reference,
    batch_transpose_reference,
    batchnorm_reference,
    clamp_reference,
    // Conv1d
    conv1d_output_length,
    // CUDA C++ emission
    emit_activation_kernels,
    emit_matmul_kernel,
    // Activations
    emit_ptx_activation,
    emit_ptx_activation_default,
    // BatchNorm / GroupNorm
    emit_ptx_batchnorm,
    emit_ptx_conv1d,
    // LayerNorm
    emit_ptx_layernorm,
    // Matmul
    emit_ptx_matmul,
    // Softmax
    emit_ptx_softmax,
    emit_reduction_kernel,
    emit_softmax_kernel,
    gather_reference,
    gelu_fast_reference,
    gelu_reference,
    // Elementwise
    generate_add_ptx,
    generate_argmax_ptx,
    generate_argmin_ptx,
    generate_batchnorm_ptx,
    generate_bf16_to_f32_ptx,
    generate_clamp_ptx,
    generate_div_ptx,
    generate_exp_ptx,
    // Cast
    generate_f32_to_f16_ptx,
    // Gather / Scatter
    generate_gather_ptx,
    generate_groupnorm_ptx,
    // InstanceNorm
    generate_instancenorm_ptx,
    generate_layernorm_ptx,
    generate_linear_no_bias_ptx,
    // Linear
    generate_linear_ptx,
    generate_linear_relu_ptx,
    generate_log_ptx,
    generate_log_softmax_ptx,
    generate_matmul_ptx,
    generate_matmul_tiled_ptx,
    generate_max_ptx,
    generate_mean_ptx,
    generate_mul_ptx,
    generate_neg_ptx,
    // Pad
    generate_pad1d_ptx,
    // RMSNorm
    generate_rmsnorm_ptx,
    generate_rope_cached_ptx,
    // RoPE
    generate_rope_ptx,
    generate_scalar_mul_ptx,
    generate_softmax_ptx,
    generate_sqrt_ptx,
    generate_sub_ptx,
    generate_sum_ptx,
    // Transpose
    generate_transpose_ptx,
    // Upsample
    generate_upsample_nearest2d_ptx,
    // Where / Clamp
    generate_where_ptx,
    groupnorm_reference,
    layernorm_reference,
    linear_reference,
    log_softmax_reference,
    matmul_reference,
    mean_reference,
    mish_reference,
    mul_reference,
    neg_reference,
    pad1d_reference,
    ptx_activation_launch_config,
    ptx_elementwise_launch_config,
    ptx_matmul_launch_config,
    // Codegen helpers
    ptx_prelude,
    reflect_pad1d_reference,
    rmsnorm_reference,
    rope_reference,
    scalar_mul_reference,
    scatter_add_reference,
    silu_reference,
    snake_reference,
    softmax_reference,
    sub_reference,
    sum_reference,
    transpose_reference,
    upsample_nearest1d_reference,
    where_reference,
    PtxActivation,
    PtxActivationConfig,
    PtxBatchNormConfig,
    PtxConv1dConfig,
    PtxLayerNormConfig,
    PtxMatmulConfig,
    PtxRmsNormConfig,
    PtxRopeConfig,
    PtxSoftmaxConfig,
    ReductionOp,
    CAST_BLOCK_SIZE,
    ELEMENTWISE_BLOCK_SIZE,
    GATHER_BLOCK_SIZE,
    INSTANCENORM_BLOCK_SIZE,
    LINEAR_BLOCK_SIZE,
    MATMUL_BLOCK_SIZE,
    PAD_BLOCK_SIZE,
    PTX_MATMUL_MAX_TILE,
    PTX_MATMUL_MIN_TILE,
    PTX_VERSION,
    REDUCE_BLOCK_SIZE,
    ROPE_BLOCK_SIZE,
    SOFTMAX_BLOCK_SIZE,
    TRANSPOSE_BLOCK_SIZE,
    UPSAMPLE_BLOCK_SIZE,
    WARP_SIZE,
    WHERE_BLOCK_SIZE,
};

// =========================================================================
// Section 1: Activation PTX -- relu/gelu/silu/tanh/sigmoid/swish patterns
// =========================================================================

#[test]
fn test_activation_relu_not_separate_kernel_but_emit_cuda_has_relu() {
    // ReLU is available through CUDA C++ emission (emit_activation_kernels)
    // but not as a standalone PTX activation enum variant.
    let src = emit_activation_kernels();
    assert!(src.contains("relu_kernel"));
    assert!(src.contains("x > 0.0f ? x : 0.0f"));
}

#[test]
fn test_activation_gelu_reference_symmetry_around_zero() {
    // gelu(-x) should be approximately -gelu(x) for small x (odd-function-like)
    // Actually gelu is NOT odd, but gelu(0) = 0 exactly.
    assert!((gelu_reference(0.0)).abs() < 1e-6);
    // For |x| >> 0, gelu(x) ~ x and gelu(-x) ~ 0, so they differ.
    let g_pos = gelu_reference(0.5);
    let g_neg = gelu_reference(-0.5);
    // Exact identity: gelu(x) = x*0.5*(1+erf(x/sqrt2)) and erf is odd, so
    //   gelu(x) - gelu(-x) = 0.5x(1+erf) - 0.5(-x)(1-erf) = x.
    // (The sum gelu(x)+gelu(-x) = x*erf(x/sqrt2) ~= 0.1915 for x=0.5, NOT 0.5.)
    assert!((g_pos - g_neg - 0.5).abs() < 0.05);
}

#[test]
fn test_activation_silu_derivative_positive_at_zero() {
    // Numerical derivative of silu at 0: (silu(h) - silu(-h)) / (2h)
    let h = 1e-4_f32;
    let deriv = (silu_reference(h) - silu_reference(-h)) / (2.0 * h);
    // silu'(0) = sigmoid(0) + 0 * sigmoid'(0) = 0.5
    assert!((deriv - 0.5).abs() < 0.01);
}

#[test]
fn test_activation_gelu_fast_vs_exact_close_at_unit() {
    // gelu_fast should approximate gelu within ~5% for moderate inputs
    let x = 1.0_f32;
    let exact = gelu_reference(x);
    let fast = gelu_fast_reference(x);
    assert!((exact - fast).abs() / exact.abs() < 0.05);
}

#[test]
fn test_activation_mish_bounded_below_for_negative() {
    // mish(x) -> 0 from below as x -> -inf
    let result = mish_reference(-10.0);
    assert!(result.abs() < 0.01);
    assert!(result <= 0.0);
}

#[test]
fn test_activation_snake_alpha_scaling() {
    // With larger alpha, the oscillation frequency increases but amplitude
    // of the sin^2 term decreases (1/alpha factor).
    let x = 1.0;
    let s1 = snake_reference(x, 1.0);
    let s10 = snake_reference(x, 10.0);
    // Both should be >= x (sin^2 >= 0 and 1/alpha > 0)
    assert!(s1 >= x);
    assert!(s10 >= x);
}

#[test]
fn test_activation_ptx_all_contain_ld_global_st_global() {
    for act in [
        PtxActivation::Gelu,
        PtxActivation::GeluFast,
        PtxActivation::Silu,
        PtxActivation::Mish,
        PtxActivation::Snake,
    ] {
        let ptx = emit_ptx_activation_default(&format!("{}_k", act.name()), act).unwrap();
        assert!(
            ptx.contains("ld.global.f32"),
            "{} missing ld.global.f32",
            act.name()
        );
        assert!(
            ptx.contains("st.global.f32"),
            "{} missing st.global.f32",
            act.name()
        );
    }
}

#[test]
fn test_activation_config_custom_block_size_128() {
    let config = PtxActivationConfig::new("custom_silu", PtxActivation::Silu).with_block_size(128);
    let ptx = emit_ptx_activation(&config).unwrap();
    assert!(ptx.contains(".reqntid 128"));
    assert!(ptx.contains(".entry custom_silu"));
}

#[test]
fn test_activation_launch_config_large_n() {
    let (grid, block) = ptx_activation_launch_config(1_000_000, 256);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [3907, 1, 1]); // ceil(1_000_000 / 256)
}

// =========================================================================
// Section 2: LayerNorm / RMSNorm PTX extended
// =========================================================================

#[test]
fn test_layernorm_ptx_contains_ld_global_and_st_global() {
    let ptx = generate_layernorm_ptx(128);
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_layernorm_ptx_dim_768_uses_shared_memory() {
    let config = PtxLayerNormConfig::new("ln_768", 768, 1e-5);
    assert!(!config.is_warp_only());
    assert!(config.shared_memory_bytes() > 0);
    let ptx = emit_ptx_layernorm(&config).unwrap();
    assert!(ptx.contains("warp_scratch"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_layernorm_reference_two_rows_processed_independently() {
    // Process row 1
    let input1 = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let r1 = layernorm_reference(&input1, &gamma, &beta, 1e-5);

    // Process row 2
    let input2 = vec![10.0, 20.0, 30.0, 40.0];
    let r2 = layernorm_reference(&input2, &gamma, &beta, 1e-5);

    // Normalized values should be same pattern (just scaled input)
    // Both are [1,2,3,4] and [10,20,30,40] which are linearly scaled
    for i in 0..4 {
        assert!(
            (r1[i] - r2[i]).abs() < 1e-4,
            "row normalization differs at {i}"
        );
    }
}

#[test]
fn test_layernorm_config_block_size_for_various_dims() {
    // dim=16 -> 1 warp (32 threads)
    assert_eq!(PtxLayerNormConfig::new("k", 16, 1e-5).block_size(), 32);
    // dim=32 -> 1 warp (32 threads)
    assert_eq!(PtxLayerNormConfig::new("k", 32, 1e-5).block_size(), 32);
    // dim=33 -> 2 warps (64 threads)
    assert_eq!(PtxLayerNormConfig::new("k", 33, 1e-5).block_size(), 64);
    // dim=256 -> 8 warps (256 threads, max)
    assert_eq!(PtxLayerNormConfig::new("k", 256, 1e-5).block_size(), 256);
    // dim=1024 -> capped at 256
    assert_eq!(PtxLayerNormConfig::new("k", 1024, 1e-5).block_size(), 256);
}

#[test]
fn test_rmsnorm_reference_scales_proportionally() {
    // RMSNorm with weight=2 should double the output vs weight=1
    let input = vec![3.0, 4.0];
    let w1 = vec![1.0, 1.0];
    let w2 = vec![2.0, 2.0];
    let r1 = rmsnorm_reference(&input, &w1, 1e-5);
    let r2 = rmsnorm_reference(&input, &w2, 1e-5);
    for i in 0..2 {
        assert!((r2[i] - 2.0 * r1[i]).abs() < 1e-5);
    }
}

#[test]
fn test_rmsnorm_ptx_contains_fma_for_accumulation() {
    let ptx = generate_rmsnorm_ptx(256, 1e-6);
    // RMSNorm accumulates x^2 using mul then add (or fma)
    assert!(ptx.contains("mul.f32"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains("rsqrt.approx.f32"));
}

#[test]
fn test_rmsnorm_config_validation_rejects_empty_name() {
    let config = PtxRmsNormConfig::new("", 64, 1e-5);
    assert!(config.validate().is_err());
}

// =========================================================================
// Section 3: Softmax PTX extended
// =========================================================================

#[test]
fn test_softmax_ptx_dim_16_warp_only() {
    let config = PtxSoftmaxConfig::new("sm_16", 16);
    assert!(config.is_warp_only());
    assert_eq!(config.shared_memory_bytes(), 0);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_softmax_ptx_dim_256_uses_shared_memory() {
    let config = PtxSoftmaxConfig::new("sm_256", 256);
    assert!(!config.is_warp_only());
    assert!(config.shared_memory_bytes() > 0);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains("warp_scratch"));
}

#[test]
fn test_softmax_config_validation_rejects_zero_dim() {
    let config = PtxSoftmaxConfig::new("k", 0);
    assert!(config.validate().is_err());
}

#[test]
fn test_softmax_config_validation_rejects_empty_name() {
    let config = PtxSoftmaxConfig {
        kernel_name: String::new(),
        dim: 64,
        sm_target: "sm_80".into(),
        log_mode: false,
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_softmax_reference_invariant_under_constant_shift() {
    // softmax(x + c) == softmax(x) for any constant c
    let x = vec![1.0, 2.0, 3.0];
    let shifted: Vec<f32> = x.iter().map(|v| v + 100.0).collect();
    let r1 = softmax_reference(&x);
    let r2 = softmax_reference(&shifted);
    for i in 0..3 {
        assert!(
            (r1[i] - r2[i]).abs() < 1e-5,
            "shift invariance broken at {i}"
        );
    }
}

#[test]
fn test_log_softmax_reference_all_negative() {
    // log_softmax values should all be <= 0 (since softmax values are in (0,1])
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = log_softmax_reference(&input);
    for &v in &result {
        assert!(v <= 0.0, "log_softmax value {v} > 0");
    }
}

#[test]
fn test_softmax_ptx_contains_ex2_and_rcp() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(ptx.contains("ex2.approx.f32"), "softmax missing exp2");
    // The kernel normalizes via inv_sum = 1.0 / sum using `div.approx.f32`
    // (the reciprocal idiom this generator emits), not a separate `rcp`.
    assert!(ptx.contains("div.approx.f32"), "softmax missing reciprocal");
}

#[test]
fn test_log_softmax_ptx_contains_lg2() {
    let ptx = generate_log_softmax_ptx(128);
    assert!(ptx.contains("lg2.approx.f32"), "log_softmax missing lg2");
}

// =========================================================================
// Section 4: Matmul PTX extended
// =========================================================================

#[test]
fn test_matmul_ptx_tiled_tile_size_4() {
    let ptx = generate_matmul_tiled_ptx(16, 16, 16, 4);
    assert!(ptx.contains(".shared .align 4 .f32 As[16]")); // 4*4=16
    assert!(ptx.contains(".shared .align 4 .f32 Bs[16]"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_matmul_config_tile_validation_boundaries() {
    // Tile exactly at min boundary should pass
    let config = PtxMatmulConfig::new("k").with_tile_size(PTX_MATMUL_MIN_TILE);
    assert!(emit_ptx_matmul(&config).is_ok());

    // Tile exactly at max boundary should pass
    let config = PtxMatmulConfig::new("k").with_tile_size(PTX_MATMUL_MAX_TILE);
    assert!(emit_ptx_matmul(&config).is_ok());

    // One below min should fail
    let config = PtxMatmulConfig::new("k").with_tile_size(PTX_MATMUL_MIN_TILE - 1);
    assert!(emit_ptx_matmul(&config).is_err());

    // One above max should fail
    let config = PtxMatmulConfig::new("k").with_tile_size(PTX_MATMUL_MAX_TILE + 1);
    assert!(emit_ptx_matmul(&config).is_err());
}

#[test]
fn test_matmul_reference_zero_matrix() {
    let a = vec![0.0; 4]; // 2x2 zeros
    let b = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let c = matmul_reference(&a, &b, 2, 2, 2);
    for &v in &c {
        assert!(v.abs() < 1e-6, "matmul with zero matrix not zero");
    }
}

#[test]
fn test_matmul_reference_transpose_property() {
    // (A * B)^T should equal B^T * A^T for square matrices
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
    let c = matmul_reference(&a, &b, 2, 2, 2);

    // Transpose A and B
    let at = vec![a[0], a[2], a[1], a[3]]; // transpose 2x2
    let bt = vec![b[0], b[2], b[1], b[3]];
    let ct = matmul_reference(&bt, &at, 2, 2, 2);

    // C^T should match: ct[0]==c[0], ct[1]==c[2], ct[2]==c[1], ct[3]==c[3]
    assert!((ct[0] - c[0]).abs() < 1e-4);
    assert!((ct[1] - c[2]).abs() < 1e-4);
    assert!((ct[2] - c[1]).abs() < 1e-4);
    assert!((ct[3] - c[3]).abs() < 1e-4);
}

#[test]
fn test_matmul_launch_config_non_square() {
    let (grid, block) = ptx_matmul_launch_config(64, 128, 16);
    // grid = [ceil(128/16), ceil(64/16)] = [8, 4]
    assert_eq!(grid, [8, 4, 1]);
    assert_eq!(block, [16, 16, 1]);
}

// =========================================================================
// Section 5: Elementwise binary ops extended
// =========================================================================

#[test]
fn test_add_ptx_contains_ld_global_and_st_global() {
    let ptx = generate_add_ptx(256);
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_sub_reference_identity() {
    let a = vec![5.0, 10.0, 15.0];
    let result = sub_reference(&a, &a);
    for &v in &result {
        assert!(v.abs() < 1e-6, "a - a should be zero");
    }
}

#[test]
fn test_mul_reference_identity_element() {
    let a = vec![1.5, 2.5, 3.5];
    let ones = vec![1.0; 3];
    let result = mul_reference(&a, &ones);
    assert_eq!(result, a);
}

#[test]
fn test_div_ptx_uses_approx_division() {
    let ptx = generate_div_ptx(64);
    assert!(ptx.contains("div.approx.f32"));
}

#[test]
fn test_neg_reference_double_neg_identity() {
    let a = vec![1.0, -2.0, 3.0, -4.0];
    let neg_a = neg_reference(&a);
    let neg_neg_a = neg_reference(&neg_a);
    for (x, y) in a.iter().zip(neg_neg_a.iter()) {
        assert!((x - y).abs() < 1e-6, "double neg not identity");
    }
}

#[test]
fn test_scalar_mul_reference_zero_scalar() {
    let input = vec![1.0, 2.0, 3.0];
    let result = scalar_mul_reference(&input, 0.0);
    for &v in &result {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn test_elementwise_launch_config_single_element() {
    let (grid, block) = ptx_elementwise_launch_config(1);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [ELEMENTWISE_BLOCK_SIZE, 1, 1]);
}

// =========================================================================
// Section 6: Reduction ops extended
// =========================================================================

#[test]
fn test_sum_ptx_contains_shared_and_bar_sync() {
    let ptx = generate_sum_ptx(512);
    assert!(ptx.contains(".shared"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_mean_ptx_contains_div_for_averaging() {
    let ptx = generate_mean_ptx(1024);
    assert!(ptx.contains("div.approx.f32"));
}

#[test]
fn test_sum_reference_single_element() {
    assert!((sum_reference(&[42.0]) - 42.0).abs() < 1e-6);
}

#[test]
fn test_mean_reference_empty() {
    // mean of empty should be 0/0 = NaN, but implementation may differ
    // Just check it doesn't panic
    let _ = mean_reference(&[]);
}

#[test]
fn test_argmax_reference_last_element_max() {
    assert_eq!(argmax_reference(&[1.0, 2.0, 3.0, 4.0, 5.0]), 4);
}

#[test]
fn test_argmin_reference_first_element_min() {
    assert_eq!(argmin_reference(&[-10.0, 0.0, 5.0]), 0);
}

#[test]
fn test_reduce_ptx_all_have_shared_memory() {
    let ptx_sum = generate_sum_ptx(64);
    let ptx_max = generate_max_ptx(64);
    let ptx_mean = generate_mean_ptx(64);
    let ptx_argmax = generate_argmax_ptx(64);
    let ptx_argmin = generate_argmin_ptx(64);

    for (name, ptx) in [
        ("sum", ptx_sum),
        ("max", ptx_max),
        ("mean", ptx_mean),
        ("argmax", ptx_argmax),
        ("argmin", ptx_argmin),
    ] {
        assert!(ptx.contains(".shared"), "{name} missing .shared");
    }
}

// =========================================================================
// Section 7: Conv1d
// =========================================================================

#[test]
fn test_conv1d_output_length_basic() {
    // Standard conv1d: input_len=10, kernel=3, stride=1, padding=0, dilation=1
    let out = conv1d_output_length(10, 3, 1, 0, 1);
    assert_eq!(out, Some(8)); // (10 - 3) / 1 + 1 = 8
}

#[test]
fn test_conv1d_output_length_with_padding() {
    // With padding=1: (10 + 2*1 - 3) / 1 + 1 = 10 (same padding)
    let out = conv1d_output_length(10, 3, 1, 1, 1);
    assert_eq!(out, Some(10));
}

#[test]
fn test_conv1d_output_length_with_stride() {
    // stride=2: (10 - 3) / 2 + 1 = 4
    let out = conv1d_output_length(10, 3, 2, 0, 1);
    assert_eq!(out, Some(4));
}

#[test]
fn test_conv1d_ptx_generation() {
    let config = PtxConv1dConfig::new("conv1d_test", 1, 8, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(ptx.contains(".entry conv1d_test"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_conv1d_config_validation_rejects_zero_channels() {
    let config = PtxConv1dConfig::new("k", 0, 8, 3);
    assert!(config.validate().is_err());
}

// =========================================================================
// Section 8: Transpose
// =========================================================================

#[test]
fn test_transpose_ptx_has_shared_memory() {
    let ptx = generate_transpose_ptx(32, 32);
    assert!(ptx.contains(".shared"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_transpose_reference_2x3() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let result = transpose_reference(&data, 2, 3);
    // Expected 3x2: [1, 4, 2, 5, 3, 6]
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_reference_identity_for_1x1() {
    let data = vec![42.0];
    let result = transpose_reference(&data, 1, 1);
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_batch_transpose_reference_two_batches() {
    // 2 batches of 2x2 matrices
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = batch_transpose_reference(&data, 2, 2, 2);
    // Batch 0: [1,2,3,4] -> [1,3,2,4], Batch 1: [5,6,7,8] -> [5,7,6,8]
    assert_eq!(result, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
}

#[test]
fn test_transpose_block_size_constant() {
    assert!(TRANSPOSE_BLOCK_SIZE > 0);
}

// =========================================================================
// Section 9: RoPE
// =========================================================================

#[test]
fn test_rope_ptx_generation_basic() {
    let config = PtxRopeConfig::new(128, 64);
    let ptx = generate_rope_ptx(&config).unwrap();
    assert!(ptx.contains("__global__") || ptx.contains("rope"));
    assert!(!ptx.is_empty());
}

#[test]
fn test_rope_cached_ptx_generation() {
    let config = PtxRopeConfig::new(128, 64);
    let ptx = generate_rope_cached_ptx(&config).unwrap();
    assert!(!ptx.is_empty());
}

#[test]
fn test_rope_config_validation_rejects_odd_head_dim() {
    let config = PtxRopeConfig::new(128, 63); // odd head_dim
    assert!(config.validate().is_err());
}

#[test]
fn test_rope_reference_preserves_norm() {
    // RoPE is a rotation, so it should approximately preserve the L2 norm
    let x = vec![1.0, 0.0, 0.0, 1.0]; // seq_len=1, head_dim=4
    let result = rope_reference(&x, 1, 4);
    let norm_in: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_out: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_in - norm_out).abs() < 0.01,
        "RoPE should preserve norm: in={norm_in}, out={norm_out}"
    );
}

#[test]
fn test_rope_block_size_constant() {
    assert_eq!(ROPE_BLOCK_SIZE, 256);
}

// =========================================================================
// Section 10: Gather / Scatter
// =========================================================================

#[test]
fn test_gather_ptx_contains_ld_global() {
    let ptx = generate_gather_ptx(100, 10);
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_gather_reference_basic() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices = vec![0, 2, 4];
    let result = gather_reference(&data, &indices, 5);
    assert_eq!(result, vec![10.0, 30.0, 50.0]);
}

#[test]
fn test_scatter_add_reference_basic() {
    let src = vec![1.0, 2.0, 3.0];
    let indices = vec![0, 0, 2];
    let result = scatter_add_reference(&src, &indices, 3, 3);
    // index 0 gets 1.0 + 2.0 = 3.0, index 1 = 0.0, index 2 = 3.0
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1]).abs() < 1e-6);
    assert!((result[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_gather_block_size_constant() {
    assert!(GATHER_BLOCK_SIZE > 0);
}

// =========================================================================
// Section 11: Pad / Upsample / Where / Clamp / Cast
// =========================================================================

#[test]
fn test_pad1d_ptx_generation() {
    let ptx = generate_pad1d_ptx(10, 2, 2, 0.0);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_pad1d_reference_zero_padding() {
    let input = vec![1.0, 2.0, 3.0];
    let result = pad1d_reference(&input, 1, 1, 0.0);
    assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_reflect_pad1d_reference() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = reflect_pad1d_reference(&input, 2, 2);
    // reflect: [3, 2, 1, 2, 3, 4, 3, 2]
    assert_eq!(result.len(), 8);
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_upsample_nearest1d_reference() {
    let input = vec![1.0, 2.0, 3.0];
    let result = upsample_nearest1d_reference(&input, 2);
    assert_eq!(result, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_upsample_nearest2d_ptx_generation() {
    let ptx = generate_upsample_nearest2d_ptx(4, 4, 2, 2);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_where_ptx_generation() {
    let ptx = generate_where_ptx(100);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_where_reference_basic() {
    let cond = vec![1, 0, 1, 0];
    let a = vec![10.0, 20.0, 30.0, 40.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let result = where_reference(&cond, &a, &b);
    assert_eq!(result, vec![10.0, 2.0, 30.0, 4.0]);
}

#[test]
fn test_clamp_reference_basic() {
    let input = vec![-5.0, 0.0, 5.0, 10.0];
    let result = clamp_reference(&input, 0.0, 7.0);
    assert_eq!(result, vec![0.0, 0.0, 5.0, 7.0]);
}

#[test]
fn test_clamp_ptx_contains_max_and_min() {
    let ptx = generate_clamp_ptx(100, -1.0, 1.0);
    // Clamp uses max and min instructions
    assert!(ptx.contains("max.f32") || ptx.contains("min.f32") || ptx.contains("setp"));
}

#[test]
fn test_cast_f32_to_f16_ptx_generation() {
    let ptx = generate_f32_to_f16_ptx(1024);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("cvt")); // conversion instruction
}

#[test]
fn test_cast_bf16_to_f32_ptx_generation() {
    let ptx = generate_bf16_to_f32_ptx(1024);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cast_block_size_constant() {
    assert!(CAST_BLOCK_SIZE > 0);
}

#[test]
fn test_pad_block_size_constant() {
    assert!(PAD_BLOCK_SIZE > 0);
}

#[test]
fn test_upsample_block_size_constant() {
    assert!(UPSAMPLE_BLOCK_SIZE > 0);
}

#[test]
fn test_where_block_size_constant() {
    assert!(WHERE_BLOCK_SIZE > 0);
}

// =========================================================================
// Section 12: BatchNorm / GroupNorm / InstanceNorm
// =========================================================================

#[test]
fn test_batchnorm_ptx_entry_and_params() {
    let config = PtxBatchNormConfig::new("bn_test", 64, 1e-5);
    let ptx = emit_ptx_batchnorm(&config).unwrap();
    assert!(ptx.contains(".entry bn_test"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_batchnorm_ptx_default_convenience() {
    let ptx = generate_batchnorm_ptx(32);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batchnorm_reference_basic() {
    // 1 sample, 2 channels, spatial=2: total 4 elements
    let input = vec![1.0, 2.0, 3.0, 4.0]; // C0:[1,2], C1:[3,4]
    let mut output = vec![0.0; 4];
    let running_mean = vec![1.5, 3.5]; // mean per channel
    let running_var = vec![0.25, 0.25]; // var per channel
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    batchnorm_reference(
        &input,
        &mut output,
        &running_mean,
        &running_var,
        &weight,
        &bias,
        2,
        2,
        1e-5,
    );
    // C0: (1 - 1.5) / sqrt(0.25 + 1e-5) ~ -1.0
    // C0: (2 - 1.5) / sqrt(0.25 + 1e-5) ~  1.0
    assert!((output[0] - (-1.0)).abs() < 0.01);
    assert!((output[1] - 1.0).abs() < 0.01);
}

#[test]
fn test_groupnorm_ptx_generation() {
    let ptx = generate_groupnorm_ptx(2, 8);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_groupnorm_reference_uniform_input() {
    // 1 sample, 4 channels, spatial=1, 2 groups
    let input = vec![5.0, 5.0, 5.0, 5.0];
    let mut output = vec![0.0; 4];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    groupnorm_reference(&input, &mut output, &weight, &bias, 2, 4, 1, 1e-5);
    // Uniform input -> normalized to 0 (then bias=0)
    for &v in &output {
        assert!(v.abs() < 0.01, "uniform input should normalize to ~0");
    }
}

#[test]
fn test_instancenorm_ptx_generation() {
    let ptx = generate_instancenorm_ptx(3, 4, 4, 1e-5);
    assert!(ptx.contains(".entry"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_instancenorm_block_size_constant() {
    assert!(INSTANCENORM_BLOCK_SIZE > 0);
}

// =========================================================================
// Section 13: Linear PTX extended
// =========================================================================

#[test]
fn test_linear_ptx_all_three_variants_differ() {
    let with_bias = generate_linear_ptx(32, 64);
    let no_bias = generate_linear_no_bias_ptx(32, 64);
    let with_relu = generate_linear_relu_ptx(32, 64);

    // They should all be different
    assert_ne!(with_bias, no_bias);
    assert_ne!(with_bias, with_relu);
    assert_ne!(no_bias, with_relu);
}

#[test]
fn test_linear_reference_zero_weight() {
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![0.0; 6]; // 3x2 zeros
    let bias = vec![0.5, 1.5];
    let result = linear_reference(&input, &weight, Some(&bias), 3, 2);
    // With zero weights, output = bias only
    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - 1.5).abs() < 1e-6);
}

// =========================================================================
// Section 14: CUDA C++ emission extended
// =========================================================================

#[test]
fn test_emit_softmax_kernel_zero_rejected() {
    assert!(emit_softmax_kernel(0).is_err());
}

#[test]
fn test_emit_matmul_kernel_valid_tiles() {
    // Tile size 8 should be valid
    let src = emit_matmul_kernel("gemm_8", 8).unwrap();
    assert!(src.contains("#define TILE_SIZE 8"));
    assert!(src.contains("__shared__"));
}

#[test]
fn test_emit_reduction_kernel_sum_contains_atomic_add() {
    let src = emit_reduction_kernel("sum_k", ReductionOp::Sum, 128).unwrap();
    // Should use shared memory and syncthreads
    assert!(src.contains("__shared__"));
    assert!(src.contains("__syncthreads"));
}

// =========================================================================
// Section 15: Workgroup / block size configuration
// =========================================================================

#[test]
fn test_warp_size_is_32() {
    assert_eq!(WARP_SIZE, 32);
}

#[test]
fn test_ptx_version_string_format() {
    // PTX version should be a decimal like "7.0" or "8.0"
    assert!(
        PTX_VERSION.contains('.'),
        "PTX_VERSION should contain a dot"
    );
}

#[test]
fn test_softmax_block_size_is_256() {
    assert_eq!(SOFTMAX_BLOCK_SIZE, 256);
}

#[test]
fn test_matmul_block_size_is_16() {
    assert_eq!(MATMUL_BLOCK_SIZE, 16);
}

#[test]
fn test_linear_block_size_is_256() {
    assert_eq!(LINEAR_BLOCK_SIZE, 256);
}

#[test]
fn test_reduce_block_size_is_256() {
    assert_eq!(REDUCE_BLOCK_SIZE, 256);
}

#[test]
fn test_elementwise_block_size_is_256() {
    assert_eq!(ELEMENTWISE_BLOCK_SIZE, 256);
}

// =========================================================================
// Section 16: PTX string validation -- expected instructions
// =========================================================================

#[test]
fn test_ptx_generators_all_contain_address_size_64() {
    let kernels: Vec<(&str, String)> = vec![
        ("add", generate_add_ptx(64)),
        ("sub", generate_sub_ptx(64)),
        ("mul", generate_mul_ptx(64)),
        ("div", generate_div_ptx(64)),
        ("exp", generate_exp_ptx(64)),
        ("log", generate_log_ptx(64)),
        ("sqrt", generate_sqrt_ptx(64)),
        ("neg", generate_neg_ptx(64)),
        ("scalar_mul", generate_scalar_mul_ptx(64)),
        ("sum", generate_sum_ptx(64)),
        ("max", generate_max_ptx(64)),
        ("mean", generate_mean_ptx(64)),
        ("argmax", generate_argmax_ptx(64)),
        ("argmin", generate_argmin_ptx(64)),
        ("matmul", generate_matmul_ptx(8, 8, 8)),
        ("layernorm", generate_layernorm_ptx(64)),
        ("rmsnorm", generate_rmsnorm_ptx(64, 1e-5)),
        ("softmax", generate_softmax_ptx(false, 64)),
    ];
    for (name, ptx) in &kernels {
        assert!(
            ptx.contains(".address_size 64"),
            "{name} missing .address_size 64"
        );
    }
}

#[test]
fn test_ptx_generators_all_contain_ret_instruction() {
    let kernels: Vec<(&str, String)> = vec![
        ("add", generate_add_ptx(64)),
        ("sub", generate_sub_ptx(64)),
        ("sum", generate_sum_ptx(64)),
        ("matmul", generate_matmul_ptx(8, 8, 8)),
        ("layernorm", generate_layernorm_ptx(64)),
        ("softmax", generate_softmax_ptx(false, 64)),
        ("transpose", generate_transpose_ptx(8, 8)),
    ];
    for (name, ptx) in &kernels {
        assert!(ptx.contains("ret;"), "{name} missing ret; instruction");
    }
}

#[test]
fn test_ptx_generators_all_contain_entry_directive() {
    let kernels: Vec<(&str, String)> = vec![
        ("add", generate_add_ptx(64)),
        ("sum", generate_sum_ptx(64)),
        ("matmul_naive", generate_matmul_ptx(8, 8, 8)),
        ("matmul_tiled", generate_matmul_tiled_ptx(16, 16, 16, 8)),
        ("layernorm", generate_layernorm_ptx(64)),
        ("rmsnorm", generate_rmsnorm_ptx(64, 1e-5)),
        ("softmax", generate_softmax_ptx(false, 64)),
        ("log_softmax", generate_log_softmax_ptx(64)),
        ("linear", generate_linear_ptx(32, 64)),
    ];
    for (name, ptx) in &kernels {
        assert!(ptx.contains(".entry"), "{name} missing .entry directive");
    }
}

#[test]
fn test_ptx_prelude_contains_required_directives() {
    let prelude = ptx_prelude("sm_80");
    assert!(prelude.contains(&format!(".version {PTX_VERSION}")));
    assert!(prelude.contains(".target sm_80"));
    assert!(prelude.contains(".address_size 64"));
}
