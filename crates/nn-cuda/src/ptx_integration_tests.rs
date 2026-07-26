// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-kernel integration tests for PTX generation.
//!
//! Verifies structural consistency across all PTX generation functions:
//! non-empty output, `.entry` directives, reference function semantics,
//! block size constraints, and parameter consistency.

// =========================================================================
// 1. All PTX generation functions produce non-empty strings
// =========================================================================

mod non_empty {
    use crate::ptx_activations::{emit_ptx_activation_default, PtxActivation};
    use crate::ptx_attention::{generate_sdpa_causal_ptx, generate_sdpa_ptx};
    use crate::ptx_attention_multihead::{
        generate_multihead_attention_ptx, PtxMultiHeadAttentionConfig,
    };
    use crate::ptx_batchnorm::generate_batchnorm_ptx;
    use crate::ptx_cast::{
        generate_bf16_to_f32_ptx, generate_f16_to_f32_ptx, generate_f32_to_bf16_ptx,
        generate_f32_to_f16_ptx,
    };
    use crate::ptx_conv1d::emit_ptx_conv1d_default;
    use crate::ptx_elementwise::{
        generate_add_ptx, generate_div_ptx, generate_exp_ptx, generate_log_ptx, generate_mul_ptx,
        generate_neg_ptx, generate_scalar_mul_ptx, generate_sqrt_ptx, generate_sub_ptx,
    };
    use crate::ptx_embedding::{generate_embedding_ptx, PtxEmbeddingConfig};
    use crate::ptx_gather::{generate_gather_ptx, generate_scatter_add_ptx};
    use crate::ptx_gemv::{generate_dot_ptx, generate_gemv_ptx, generate_outer_ptx};
    use crate::ptx_groupnorm::generate_groupnorm_ptx;
    use crate::ptx_instancenorm::generate_instancenorm_ptx;
    use crate::ptx_layernorm::generate_layernorm_ptx;
    use crate::ptx_linear::{
        generate_linear_no_bias_ptx, generate_linear_ptx, generate_linear_relu_ptx,
    };
    use crate::ptx_matmul::{
        emit_ptx_matmul_default, generate_matmul_ptx, generate_matmul_tiled_ptx,
    };
    use crate::ptx_pad::{generate_pad1d_ptx, generate_reflect_pad1d_ptx};
    use crate::ptx_quantize::{
        generate_dequantize_int8_to_f32_ptx, generate_quantize_f32_to_int8_ptx,
    };
    use crate::ptx_reduce::{
        generate_argmax_ptx, generate_argmin_ptx, generate_max_ptx, generate_mean_ptx,
        generate_sum_ptx,
    };
    use crate::ptx_residual::{
        generate_residual_add_layernorm_ptx, generate_residual_add_ptx,
        generate_residual_add_relu_ptx,
    };
    use crate::ptx_rmsnorm::generate_rmsnorm_ptx;
    use crate::ptx_rope::{generate_rope_ptx, PtxRopeConfig};
    use crate::ptx_softmax::{generate_log_softmax_ptx, generate_softmax_ptx};
    use crate::ptx_tensor_ops::{
        generate_concat_ptx, generate_fill_ptx, generate_repeat_ptx, generate_slice_ptx,
    };
    use crate::ptx_transpose::{generate_batch_transpose_ptx, generate_transpose_ptx};
    use crate::ptx_upsample::{generate_upsample_nearest1d_ptx, generate_upsample_nearest2d_ptx};
    use crate::ptx_where::{generate_clamp_ptx, generate_where_ptx};

    #[test]
    fn test_activation_ptx_non_empty() {
        for act in [
            PtxActivation::Gelu,
            PtxActivation::GeluFast,
            PtxActivation::Silu,
            PtxActivation::Mish,
            PtxActivation::Snake,
        ] {
            let ptx = emit_ptx_activation_default(&format!("{}_f32", act.name()), act).unwrap();
            assert!(
                !ptx.is_empty(),
                "{act:?} activation PTX must not be empty"
            );
        }
    }

    #[test]
    fn test_matmul_ptx_non_empty() {
        let ptx = emit_ptx_matmul_default("mm").unwrap();
        assert!(!ptx.is_empty());
        let ptx2 = generate_matmul_ptx(64, 32, 64);
        assert!(!ptx2.is_empty());
        let ptx3 = generate_matmul_tiled_ptx(64, 32, 64, 16);
        assert!(!ptx3.is_empty());
    }

    #[test]
    fn test_softmax_ptx_non_empty() {
        assert!(!generate_softmax_ptx(false, 128).is_empty());
        assert!(!generate_softmax_ptx(true, 128).is_empty());
        assert!(!generate_log_softmax_ptx(256).is_empty());
    }

    #[test]
    fn test_layernorm_ptx_non_empty() {
        assert!(!generate_layernorm_ptx(768).is_empty());
    }

    #[test]
    fn test_rmsnorm_ptx_non_empty() {
        assert!(!generate_rmsnorm_ptx(512, 1e-5).is_empty());
    }

    #[test]
    fn test_elementwise_ptx_non_empty() {
        assert!(!generate_add_ptx(1024).is_empty());
        assert!(!generate_sub_ptx(1024).is_empty());
        assert!(!generate_mul_ptx(1024).is_empty());
        assert!(!generate_div_ptx(1024).is_empty());
        assert!(!generate_exp_ptx(1024).is_empty());
        assert!(!generate_log_ptx(1024).is_empty());
        assert!(!generate_sqrt_ptx(1024).is_empty());
        assert!(!generate_neg_ptx(1024).is_empty());
        assert!(!generate_scalar_mul_ptx(1024).is_empty());
    }

    #[test]
    fn test_linear_ptx_non_empty() {
        assert!(!generate_linear_ptx(768, 3072).is_empty());
        assert!(!generate_linear_no_bias_ptx(768, 3072).is_empty());
        assert!(!generate_linear_relu_ptx(768, 3072).is_empty());
    }

    #[test]
    fn test_reduce_ptx_non_empty() {
        assert!(!generate_sum_ptx(256).is_empty());
        assert!(!generate_max_ptx(256).is_empty());
        assert!(!generate_mean_ptx(256).is_empty());
        assert!(!generate_argmax_ptx(256).is_empty());
        assert!(!generate_argmin_ptx(256).is_empty());
    }

    #[test]
    fn test_transpose_ptx_non_empty() {
        assert!(!generate_transpose_ptx(32, 64).is_empty());
        assert!(!generate_batch_transpose_ptx(4, 32, 64).is_empty());
    }

    #[test]
    fn test_conv1d_ptx_non_empty() {
        let ptx = emit_ptx_conv1d_default(3, 16, 3, 1, 1).unwrap();
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_rope_ptx_non_empty() {
        let config = PtxRopeConfig::new(64, 128);
        let ptx = generate_rope_ptx(&config).unwrap();
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_gather_ptx_non_empty() {
        assert!(!generate_gather_ptx(1024, 64).is_empty());
        assert!(!generate_scatter_add_ptx(1024, 64).is_empty());
    }

    #[test]
    fn test_where_ptx_non_empty() {
        assert!(!generate_where_ptx(512).is_empty());
        assert!(!generate_clamp_ptx(512, -1.0, 1.0).is_empty());
    }

    #[test]
    fn test_residual_ptx_non_empty() {
        assert!(!generate_residual_add_ptx(256).is_empty());
        assert!(!generate_residual_add_relu_ptx(256).is_empty());
        assert!(!generate_residual_add_layernorm_ptx(256, 128).is_empty());
    }

    #[test]
    fn test_tensor_ops_ptx_non_empty() {
        assert!(!generate_concat_ptx(128, 128).is_empty());
        assert!(!generate_slice_ptx(256, 64, 128).is_empty());
        assert!(!generate_repeat_ptx(128, 3).is_empty());
        assert!(!generate_fill_ptx(256, 0.0).is_empty());
    }

    #[test]
    fn test_gemv_ptx_non_empty() {
        assert!(!generate_gemv_ptx(128, 64).is_empty());
        assert!(!generate_dot_ptx(256).is_empty());
        assert!(!generate_outer_ptx(32, 64).is_empty());
    }

    #[test]
    fn test_cast_ptx_non_empty() {
        assert!(!generate_f32_to_f16_ptx(256).is_empty());
        assert!(!generate_f16_to_f32_ptx(256).is_empty());
        assert!(!generate_f32_to_bf16_ptx(256).is_empty());
        assert!(!generate_bf16_to_f32_ptx(256).is_empty());
    }

    #[test]
    fn test_pad_ptx_non_empty() {
        assert!(!generate_pad1d_ptx(128, 4, 4, 0.0).is_empty());
        assert!(!generate_reflect_pad1d_ptx(128, 4, 4).is_empty());
    }

    #[test]
    fn test_quantize_ptx_non_empty() {
        assert!(!generate_quantize_f32_to_int8_ptx(256, 0.1, 0).is_empty());
        assert!(!generate_dequantize_int8_to_f32_ptx(256, 0.1, 0).is_empty());
    }

    #[test]
    fn test_upsample_ptx_non_empty() {
        assert!(!generate_upsample_nearest1d_ptx(64, 2).is_empty());
        assert!(!generate_upsample_nearest2d_ptx(32, 32, 2, 2).is_empty());
    }

    #[test]
    fn test_instancenorm_ptx_non_empty() {
        assert!(!generate_instancenorm_ptx(3, 32, 32, 1e-5).is_empty());
    }

    #[test]
    fn test_embedding_ptx_non_empty() {
        let config = PtxEmbeddingConfig::new(10000, 768);
        let ptx = generate_embedding_ptx(&config).unwrap();
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_attention_ptx_non_empty() {
        assert!(!generate_sdpa_ptx(128, 64).is_empty());
        assert!(!generate_sdpa_causal_ptx(128, 64).is_empty());
    }

    #[test]
    fn test_multihead_attention_ptx_non_empty() {
        let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
        let ptx = generate_multihead_attention_ptx(&config).unwrap();
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_batchnorm_ptx_non_empty() {
        assert!(!generate_batchnorm_ptx(64).is_empty());
    }

    #[test]
    fn test_groupnorm_ptx_non_empty() {
        assert!(!generate_groupnorm_ptx(8, 64).is_empty());
    }
}

// =========================================================================
// 2. All PTX kernels contain .entry directive
// =========================================================================

mod entry_directive {
    use crate::ptx_activations::{emit_ptx_activation_default, PtxActivation};
    use crate::ptx_batchnorm::generate_batchnorm_ptx;
    use crate::ptx_cast::generate_f32_to_f16_ptx;
    use crate::ptx_elementwise::{generate_add_ptx, generate_exp_ptx, generate_neg_ptx};
    use crate::ptx_gather::generate_gather_ptx;
    use crate::ptx_gemv::generate_gemv_ptx;
    use crate::ptx_groupnorm::generate_groupnorm_ptx;
    use crate::ptx_instancenorm::generate_instancenorm_ptx;
    use crate::ptx_layernorm::generate_layernorm_ptx;
    use crate::ptx_linear::generate_linear_ptx;
    use crate::ptx_matmul::{emit_ptx_matmul_default, generate_matmul_ptx};
    use crate::ptx_pad::generate_pad1d_ptx;
    use crate::ptx_quantize::generate_quantize_f32_to_int8_ptx;
    use crate::ptx_reduce::{generate_max_ptx, generate_mean_ptx, generate_sum_ptx};
    use crate::ptx_residual::generate_residual_add_ptx;
    use crate::ptx_rmsnorm::generate_rmsnorm_ptx;
    use crate::ptx_softmax::generate_softmax_ptx;
    use crate::ptx_tensor_ops::generate_concat_ptx;
    use crate::ptx_transpose::generate_transpose_ptx;
    use crate::ptx_upsample::generate_upsample_nearest1d_ptx;
    use crate::ptx_where::generate_where_ptx;

    /// Helper: assert that a raw PTX string contains the `.entry` directive.
    fn assert_has_entry(ptx: &str, label: &str) {
        assert!(
            ptx.contains(".entry"),
            "{label}: PTX kernel must contain .entry directive"
        );
    }

    /// Helper: assert that a raw PTX string contains `.version` and `.target`.
    fn assert_has_version_target(ptx: &str, label: &str) {
        assert!(
            ptx.contains(".version"),
            "{label}: PTX must contain .version directive"
        );
        assert!(
            ptx.contains(".target"),
            "{label}: PTX must contain .target directive"
        );
    }

    #[test]
    fn test_activation_kernels_have_entry() {
        for act in [
            PtxActivation::Silu,
            PtxActivation::Gelu,
            PtxActivation::Snake,
        ] {
            let ptx = emit_ptx_activation_default(&format!("{}_f32", act.name()), act).unwrap();
            assert_has_entry(&ptx, act.name());
            assert_has_version_target(&ptx, act.name());
        }
    }

    #[test]
    fn test_matmul_kernels_have_entry() {
        let ptx1 = emit_ptx_matmul_default("mm").unwrap();
        assert_has_entry(&ptx1, "tiled matmul");
        assert_has_version_target(&ptx1, "tiled matmul");

        let ptx2 = generate_matmul_ptx(32, 32, 32);
        assert_has_entry(&ptx2, "naive matmul");
        assert_has_version_target(&ptx2, "naive matmul");
    }

    #[test]
    fn test_softmax_has_entry() {
        let ptx = generate_softmax_ptx(false, 128);
        assert_has_entry(&ptx, "softmax");
        assert_has_version_target(&ptx, "softmax");
    }

    #[test]
    fn test_layernorm_has_entry() {
        let ptx = generate_layernorm_ptx(256);
        assert_has_entry(&ptx, "layernorm");
    }

    #[test]
    fn test_rmsnorm_has_entry() {
        let ptx = generate_rmsnorm_ptx(256, 1e-5);
        assert_has_entry(&ptx, "rmsnorm");
    }

    #[test]
    fn test_elementwise_ops_have_entry() {
        assert_has_entry(&generate_add_ptx(128), "add");
        assert_has_entry(&generate_exp_ptx(128), "exp");
        assert_has_entry(&generate_neg_ptx(128), "neg");
    }

    #[test]
    fn test_linear_has_entry() {
        assert_has_entry(&generate_linear_ptx(64, 128), "linear");
    }

    #[test]
    fn test_reduce_ops_have_entry() {
        assert_has_entry(&generate_sum_ptx(128), "sum");
        assert_has_entry(&generate_max_ptx(128), "max");
        assert_has_entry(&generate_mean_ptx(128), "mean");
    }

    #[test]
    fn test_transpose_has_entry() {
        assert_has_entry(&generate_transpose_ptx(32, 64), "transpose");
    }

    #[test]
    fn test_gather_has_entry() {
        assert_has_entry(&generate_gather_ptx(256, 32), "gather");
    }

    #[test]
    fn test_where_has_entry() {
        assert_has_entry(&generate_where_ptx(128), "where");
    }

    #[test]
    fn test_residual_has_entry() {
        assert_has_entry(&generate_residual_add_ptx(128), "residual_add");
    }

    #[test]
    fn test_concat_has_entry() {
        assert_has_entry(&generate_concat_ptx(64, 64), "concat");
    }

    #[test]
    fn test_gemv_has_entry() {
        assert_has_entry(&generate_gemv_ptx(64, 32), "gemv");
    }

    #[test]
    fn test_cast_has_entry() {
        assert_has_entry(&generate_f32_to_f16_ptx(128), "f32_to_f16");
    }

    #[test]
    fn test_pad_has_entry() {
        assert_has_entry(&generate_pad1d_ptx(64, 2, 2, 0.0), "pad1d");
    }

    #[test]
    fn test_quantize_has_entry() {
        assert_has_entry(&generate_quantize_f32_to_int8_ptx(128, 0.1, 0), "quantize");
    }

    #[test]
    fn test_upsample_has_entry() {
        assert_has_entry(&generate_upsample_nearest1d_ptx(64, 2), "upsample");
    }

    #[test]
    fn test_instancenorm_has_entry() {
        assert_has_entry(&generate_instancenorm_ptx(3, 16, 16, 1e-5), "instancenorm");
    }

    #[test]
    fn test_batchnorm_has_entry() {
        assert_has_entry(&generate_batchnorm_ptx(32), "batchnorm");
    }

    #[test]
    fn test_groupnorm_has_entry() {
        assert_has_entry(&generate_groupnorm_ptx(4, 32), "groupnorm");
    }
}

// =========================================================================
// 3. Reference functions match kernel semantics
// =========================================================================

mod reference_semantics {
    use crate::ptx_activations::{
        gelu_fast_reference, gelu_reference, mish_reference, silu_reference, snake_reference,
    };
    use crate::ptx_elementwise::{
        add_reference, div_reference, exp_reference, log_reference, mul_reference, neg_reference,
        scalar_mul_reference, sqrt_reference, sub_reference,
    };
    use crate::ptx_gather::gather_reference;
    use crate::ptx_gemv::{dot_reference, gemv_reference, outer_reference};
    use crate::ptx_linear::linear_reference;
    use crate::ptx_matmul::matmul_reference;
    use crate::ptx_reduce::{
        argmax_reference, argmin_reference, max_reference, mean_reference, sum_reference,
    };
    use crate::ptx_residual::{residual_add_reference, residual_add_relu_reference};
    use crate::ptx_softmax::{log_softmax_reference, softmax_reference};
    use crate::ptx_tensor_ops::{
        concat_reference, fill_reference, repeat_reference, slice_reference,
    };
    use crate::ptx_transpose::transpose_reference;

    const EPS: f32 = 1e-5;

    // -- Activation references --

    #[test]
    fn test_silu_reference_positive() {
        let result = silu_reference(2.0);
        // silu(2) = 2 * sigmoid(2) = 2 * (1 / (1 + exp(-2)))
        let expected = 2.0 / (1.0 + (-2.0f32).exp());
        assert!(
            (result - expected).abs() < EPS,
            "silu(2) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_silu_reference_zero() {
        let result = silu_reference(0.0);
        assert!((result - 0.0).abs() < EPS, "silu(0) should be 0");
    }

    #[test]
    fn test_gelu_reference_zero() {
        let result = gelu_reference(0.0);
        assert!((result - 0.0).abs() < EPS, "gelu(0) should be ~0");
    }

    #[test]
    fn test_gelu_reference_large_positive() {
        let result = gelu_reference(10.0);
        // For large x, gelu(x) ~ x
        assert!((result - 10.0).abs() < 0.01, "gelu(10) should be ~10");
    }

    #[test]
    fn test_gelu_fast_reference_zero() {
        let result = gelu_fast_reference(0.0);
        assert!((result - 0.0).abs() < EPS, "gelu_fast(0) should be 0");
    }

    #[test]
    fn test_mish_reference_zero() {
        let result = mish_reference(0.0);
        assert!((result - 0.0).abs() < EPS, "mish(0) should be ~0");
    }

    #[test]
    fn test_snake_reference_identity_at_zero() {
        let result = snake_reference(0.0, 1.0);
        // snake(0, alpha) = 0 + (1/alpha) * sin(0)^2 = 0
        assert!((result - 0.0).abs() < EPS, "snake(0, 1) should be 0");
    }

    #[test]
    fn test_snake_reference_identity_component() {
        // For any x, snake(x, alpha) >= x (since sin^2 >= 0)
        let x = 3.0;
        let result = snake_reference(x, 2.0);
        assert!(result >= x, "snake(x) must be >= x");
    }

    // -- Elementwise references --

    #[test]
    fn test_add_reference_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = add_reference(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_reference_basic() {
        let c = sub_reference(&[5.0, 3.0], &[2.0, 1.0]);
        assert_eq!(c, vec![3.0, 2.0]);
    }

    #[test]
    fn test_mul_reference_basic() {
        let c = mul_reference(&[2.0, 3.0], &[4.0, 5.0]);
        assert_eq!(c, vec![8.0, 15.0]);
    }

    #[test]
    fn test_div_reference_basic() {
        let c = div_reference(&[10.0, 6.0], &[2.0, 3.0]);
        assert_eq!(c, vec![5.0, 2.0]);
    }

    #[test]
    fn test_exp_reference_zero() {
        let c = exp_reference(&[0.0]);
        assert!((c[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_log_reference_one() {
        let c = log_reference(&[1.0]);
        assert!((c[0] - 0.0).abs() < EPS);
    }

    #[test]
    fn test_sqrt_reference_four() {
        let c = sqrt_reference(&[4.0]);
        assert!((c[0] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_neg_reference() {
        let c = neg_reference(&[3.0, -2.0, 0.0]);
        assert_eq!(c, vec![-3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_scalar_mul_reference() {
        let c = scalar_mul_reference(&[1.0, 2.0, 3.0], 2.0);
        assert_eq!(c, vec![2.0, 4.0, 6.0]);
    }

    // -- Reduce references --

    #[test]
    fn test_sum_reference() {
        assert!((sum_reference(&[1.0, 2.0, 3.0, 4.0]) - 10.0).abs() < EPS);
    }

    #[test]
    fn test_max_reference() {
        assert!((max_reference(&[1.0, 5.0, 3.0, 2.0]) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_mean_reference() {
        assert!((mean_reference(&[2.0, 4.0, 6.0]) - 4.0).abs() < EPS);
    }

    #[test]
    fn test_argmax_reference() {
        assert_eq!(argmax_reference(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn test_argmin_reference() {
        assert_eq!(argmin_reference(&[3.0, 1.0, 5.0]), 1);
    }

    // -- Matmul reference --

    #[test]
    fn test_matmul_reference_identity() {
        // 2x2 identity * [1,2; 3,4] = [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I_2
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = matmul_reference(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_reference_1x1() {
        let c = matmul_reference(&[3.0], &[4.0], 1, 1, 1);
        assert!((c[0] - 12.0).abs() < EPS);
    }

    // -- Softmax reference --

    #[test]
    fn test_softmax_reference_sums_to_one() {
        let result = softmax_reference(&[1.0, 2.0, 3.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "Softmax must sum to 1, got {sum}");
    }

    #[test]
    fn test_softmax_reference_monotone() {
        let result = softmax_reference(&[1.0, 2.0, 3.0]);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_log_softmax_reference_negative() {
        let result = log_softmax_reference(&[1.0, 2.0, 3.0]);
        for val in &result {
            assert!(*val < 0.0, "log_softmax values must be negative, got {val}");
        }
    }

    // -- Linear reference --

    #[test]
    fn test_linear_reference_identity_weight() {
        // Y = X * I + bias where bias = 0
        // linear_reference(input, weight, bias, in_features, out_features)
        // input: [1, 2] (batch=1, in_features=2)
        // weight: [in_features, out_features] = [2, 2] = identity
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // I_2 as [2, 2]
        let bias = vec![0.0, 0.0];
        let result = linear_reference(&input, &weight, Some(&bias), 2, 2);
        assert!((result[0] - 1.0).abs() < EPS);
        assert!((result[1] - 2.0).abs() < EPS);
    }

    // -- Transpose reference --

    #[test]
    fn test_transpose_reference_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let result = transpose_reference(&data, 2, 3);
        // Transposed: [3, 2] = [1,4, 2,5, 3,6]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // -- Residual references --

    #[test]
    fn test_residual_add_reference() {
        let result = residual_add_reference(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(result, vec![4.0, 6.0]);
    }

    #[test]
    fn test_residual_add_relu_reference() {
        let result = residual_add_relu_reference(&[-5.0, 2.0], &[3.0, 4.0]);
        assert_eq!(result, vec![0.0, 6.0]); // max(0, -5+3)=0, max(0, 2+4)=6
    }

    // -- Gather reference --

    #[test]
    fn test_gather_reference_simple() {
        let data = vec![10.0, 20.0, 30.0, 40.0]; // [4] elements
        let indices = vec![2, 0, 3];
        let result = gather_reference(&data, &indices, 4);
        assert_eq!(result, vec![30.0, 10.0, 40.0]);
    }

    // -- Tensor ops references --

    #[test]
    fn test_concat_reference() {
        let result = concat_reference(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_reference() {
        let result = slice_reference(&[10.0, 20.0, 30.0, 40.0, 50.0], 1, 3);
        assert_eq!(result, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_repeat_reference() {
        // repeat_reference repeats EACH element `repeats` times
        // (src_idx = idx / repeats), so [1,2] with repeats=3 -> [1,1,1,2,2,2].
        let result = repeat_reference(&[1.0, 2.0], 3);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_fill_reference() {
        let result = fill_reference(4, 3.14);
        assert_eq!(result, vec![3.14, 3.14, 3.14, 3.14]);
    }

    // -- GEMV references --

    #[test]
    fn test_gemv_reference_identity() {
        // y = I * x where I is 2x2 identity, x = [3, 5]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 5.0];
        let y = gemv_reference(&a, &x, 2, 2);
        assert!((y[0] - 3.0).abs() < EPS);
        assert!((y[1] - 5.0).abs() < EPS);
    }

    #[test]
    fn test_dot_reference() {
        let result = dot_reference(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((result - 32.0).abs() < EPS); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_outer_reference() {
        let result = outer_reference(&[1.0, 2.0], &[3.0, 4.0]);
        // [1*3, 1*4, 2*3, 2*4] = [3, 4, 6, 8]
        assert_eq!(result, vec![3.0, 4.0, 6.0, 8.0]);
    }
}

// =========================================================================
// 4. Block sizes are reasonable (powers of 2, <= 1024)
// =========================================================================

mod block_sizes {
    use crate::ptx_reduce::REDUCE_BLOCK_SIZE;
    use crate::{
        ATTENTION_BLOCK_SIZE, CAST_BLOCK_SIZE, ELEMENTWISE_BLOCK_SIZE, EMBEDDING_BLOCK_SIZE,
        GATHER_BLOCK_SIZE, GEMV_BLOCK_SIZE, INSTANCENORM_BLOCK_SIZE, LINEAR_BLOCK_SIZE,
        MATMUL_BLOCK_SIZE, PAD_BLOCK_SIZE, QUANTIZE_BLOCK_SIZE, RESIDUAL_BLOCK_SIZE,
        ROPE_BLOCK_SIZE, SOFTMAX_BLOCK_SIZE, TENSOR_OPS_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE,
        UPSAMPLE_BLOCK_SIZE, WHERE_BLOCK_SIZE,
    };

    /// Assert that a block size is a power of two and within the GPU limit.
    fn assert_block_size_valid(name: &str, size: u32) {
        assert!(
            size > 0 && size <= 1024,
            "{name}: block size {size} must be in (0, 1024]"
        );
        assert!(
            size.is_power_of_two(),
            "{name}: block size {size} must be a power of two"
        );
    }

    #[test]
    fn test_all_block_sizes_are_valid() {
        assert_block_size_valid("MATMUL", MATMUL_BLOCK_SIZE);
        assert_block_size_valid("LINEAR", LINEAR_BLOCK_SIZE);
        assert_block_size_valid("SOFTMAX", SOFTMAX_BLOCK_SIZE);
        assert_block_size_valid("ATTENTION", ATTENTION_BLOCK_SIZE);
        assert_block_size_valid("EMBEDDING", EMBEDDING_BLOCK_SIZE);
        assert_block_size_valid("ELEMENTWISE", ELEMENTWISE_BLOCK_SIZE);
        assert_block_size_valid("ROPE", ROPE_BLOCK_SIZE);
        assert_block_size_valid("TRANSPOSE", TRANSPOSE_BLOCK_SIZE);
        assert_block_size_valid("RESIDUAL", RESIDUAL_BLOCK_SIZE);
        assert_block_size_valid("GATHER", GATHER_BLOCK_SIZE);
        assert_block_size_valid("WHERE", WHERE_BLOCK_SIZE);
        assert_block_size_valid("CAST", CAST_BLOCK_SIZE);
        assert_block_size_valid("QUANTIZE", QUANTIZE_BLOCK_SIZE);
        assert_block_size_valid("PAD", PAD_BLOCK_SIZE);
        assert_block_size_valid("UPSAMPLE", UPSAMPLE_BLOCK_SIZE);
        assert_block_size_valid("TENSOR_OPS", TENSOR_OPS_BLOCK_SIZE);
        assert_block_size_valid("GEMV", GEMV_BLOCK_SIZE);
        assert_block_size_valid("INSTANCENORM", INSTANCENORM_BLOCK_SIZE);
        assert_block_size_valid("REDUCE", REDUCE_BLOCK_SIZE);
    }

    #[test]
    fn test_elementwise_block_sizes_are_256() {
        // Standard elementwise kernels should use 256 threads (8 warps)
        assert_eq!(LINEAR_BLOCK_SIZE, 256);
        assert_eq!(ELEMENTWISE_BLOCK_SIZE, 256);
        assert_eq!(RESIDUAL_BLOCK_SIZE, 256);
        assert_eq!(CAST_BLOCK_SIZE, 256);
        assert_eq!(ROPE_BLOCK_SIZE, 256);
    }

    #[test]
    fn test_tile_based_block_sizes() {
        // Matmul and transpose use tile-based layouts (16x16 = 256 threads)
        assert_eq!(MATMUL_BLOCK_SIZE, 16);
        assert_eq!(TRANSPOSE_BLOCK_SIZE, 16);
    }
}

// =========================================================================
// 5. Parameter consistency across related kernels
// =========================================================================

mod parameter_consistency {
    use crate::ptx_activations::{emit_ptx_activation_default, PtxActivation};
    use crate::ptx_cast::{generate_f16_to_f32_ptx, generate_f32_to_f16_ptx};
    use crate::ptx_elementwise::{
        generate_add_ptx, generate_div_ptx, generate_mul_ptx, generate_sub_ptx,
    };
    use crate::ptx_linear::{
        generate_linear_no_bias_ptx, generate_linear_ptx, generate_linear_relu_ptx,
    };
    use crate::ptx_reduce::{generate_max_ptx, generate_mean_ptx, generate_sum_ptx};
    use crate::ptx_softmax::{generate_log_softmax_ptx, generate_softmax_ptx};

    #[test]
    fn test_all_activations_share_param_structure() {
        // All non-Snake activations should have the same param layout:
        // param_input, param_output, param_n
        for act in [
            PtxActivation::Gelu,
            PtxActivation::Silu,
            PtxActivation::Mish,
        ] {
            let ptx = emit_ptx_activation_default(&format!("{}_f32", act.name()), act).unwrap();
            assert!(ptx.contains("param_input"), "{act:?} missing param_input");
            assert!(
                ptx.contains("param_output"),
                "{act:?} missing param_output"
            );
            assert!(ptx.contains("param_n"), "{act:?} missing param_n");
            // Non-Snake activations should NOT have param_alpha
            assert!(
                !ptx.contains("param_alpha"),
                "{act:?} should not have param_alpha"
            );
        }
    }

    #[test]
    fn test_snake_has_alpha_parameter() {
        let ptx = emit_ptx_activation_default("snake_f32", PtxActivation::Snake).unwrap();
        assert!(ptx.contains("param_alpha"), "Snake must have param_alpha");
        assert!(ptx.contains("param_input"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("param_n"));
    }

    #[test]
    fn test_binary_ops_share_param_structure() {
        // All binary ops have param_a, param_b, param_output, param_n
        let binary_ptx = [
            generate_add_ptx(256),
            generate_sub_ptx(256),
            generate_mul_ptx(256),
            generate_div_ptx(256),
        ];
        let names = ["add", "sub", "mul", "div"];
        for (ptx, name) in binary_ptx.iter().zip(names.iter()) {
            assert!(ptx.contains("param_a"), "{name} missing param_a");
            assert!(ptx.contains("param_b"), "{name} missing param_b");
            assert!(ptx.contains("param_output"), "{name} missing param_output");
            assert!(ptx.contains("param_n"), "{name} missing param_n");
        }
    }

    #[test]
    fn test_linear_variants_share_weight_and_input_params() {
        let ptx_bias = generate_linear_ptx(64, 128);
        let ptx_no_bias = generate_linear_no_bias_ptx(64, 128);
        let ptx_relu = generate_linear_relu_ptx(64, 128);

        for (ptx, name) in [
            (&ptx_bias, "linear"),
            (&ptx_no_bias, "linear_no_bias"),
            (&ptx_relu, "linear_relu"),
        ] {
            assert!(ptx.contains("param_input"), "{name} missing param_input");
            assert!(ptx.contains("param_weight"), "{name} missing param_weight");
            assert!(ptx.contains("param_output"), "{name} missing param_output");
        }

        // Bias variants have param_bias, no-bias variant does not
        assert!(
            ptx_bias.contains("param_bias"),
            "linear must have param_bias"
        );
        assert!(
            ptx_relu.contains("param_bias"),
            "linear_relu must have param_bias"
        );
        assert!(
            !ptx_no_bias.contains("param_bias"),
            "linear_no_bias must NOT have param_bias"
        );
    }

    #[test]
    fn test_reduce_ops_share_param_structure() {
        let reduce_ptx = [
            generate_sum_ptx(256),
            generate_max_ptx(256),
            generate_mean_ptx(256),
        ];
        let names = ["sum", "max", "mean"];
        for (ptx, name) in reduce_ptx.iter().zip(names.iter()) {
            assert!(ptx.contains("param_input"), "{name} missing param_input");
            assert!(ptx.contains("param_output"), "{name} missing param_output");
            assert!(ptx.contains("param_n"), "{name} missing param_n");
        }
    }

    #[test]
    fn test_softmax_and_log_softmax_share_structure() {
        let sm = generate_softmax_ptx(false, 128);
        let lsm = generate_softmax_ptx(true, 128);
        let lsm2 = generate_log_softmax_ptx(128);

        for (ptx, name) in [
            (&sm, "softmax"),
            (&lsm, "log_softmax_config"),
            (&lsm2, "log_softmax"),
        ] {
            assert!(ptx.contains("param_input"), "{name} missing param_input");
            assert!(ptx.contains("param_output"), "{name} missing param_output");
        }
    }

    #[test]
    fn test_cast_kernels_share_param_structure() {
        let f2h = generate_f32_to_f16_ptx(256);
        let h2f = generate_f16_to_f32_ptx(256);

        for (ptx, name) in [(&f2h, "f32_to_f16"), (&h2f, "f16_to_f32")] {
            assert!(ptx.contains("param_input"), "{name} missing param_input");
            assert!(ptx.contains("param_output"), "{name} missing param_output");
            assert!(ptx.contains("param_n"), "{name} missing param_n");
        }
    }

    #[test]
    fn test_all_ptx_kernels_use_address_size_64() {
        // All PTX kernels operate on 64-bit pointers
        let ptx_samples = [
            generate_add_ptx(64),
            generate_sum_ptx(64),
            generate_softmax_ptx(false, 64),
        ];
        for (i, ptx) in ptx_samples.iter().enumerate() {
            assert!(
                ptx.contains(".address_size 64"),
                "Sample {i}: PTX kernel must use .address_size 64"
            );
        }
    }

    #[test]
    fn test_all_ptx_kernels_use_sm_70_or_higher() {
        // All generate_* functions use sm_70 by default
        let ptx_samples = [
            generate_add_ptx(64),
            generate_sum_ptx(64),
            generate_linear_ptx(32, 64),
        ];
        for (i, ptx) in ptx_samples.iter().enumerate() {
            assert!(
                ptx.contains(".target sm_70") || ptx.contains(".target sm_80"),
                "Sample {i}: PTX kernel must target sm_70 or sm_80"
            );
        }
    }
}
